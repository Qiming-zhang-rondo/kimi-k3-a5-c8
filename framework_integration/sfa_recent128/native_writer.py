"""Native BF16 recent-ring scatter; A5 compilation/runtime validation PENDING.

Imports are framework-free. Preparation imports Torch/Triton lazily, validates
metadata, and creates a JIT handle without launching. Call ``warmup`` outside
graph capture before ``write`` is captured. Launches use Triton's current stream.
The input is the actual post-RMSNorm latent and post-RoPE key: no recomputation,
quantization, conversion, or C4 reads occur here. Routing contents stay on device.
"""
from functools import lru_cache
from math import prod

from .prolog_provider import RingBinding


class NativeWriterError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise NativeWriterError(message)


def _shape(value, name):
    shape = tuple(getattr(value, "shape", ()))
    _require(shape and all(type(d) is int and d > 0 for d in shape),
             name + " requires concrete positive shape dimensions")
    return shape


def _input_metadata(value, width, name):
    shape = _shape(value, name)
    _require(len(shape) >= 2 and shape[-1] == width,
             name + " must have trailing width " + str(width))
    _require(str(getattr(value, "dtype", None)) == "torch.bfloat16",
             name + " must be original BF16; FP32 conversion is forbidden")
    device = getattr(value, "device", None)
    _require(device is not None and str(device).split(":")[0] != "cpu",
             name + " must be a device tensor")
    stride = tuple(value.stride())
    _require(len(stride) == len(shape) and all(type(s) is int and s >= 0 for s in stride),
             name + " requires nonnegative concrete strides")
    _require(stride[-1] > 0, name + " requires a positive column stride")
    # Flatten the leading token dimensions using metadata, without a view/copy.
    # Singleton dimensions impose no addressing constraint.
    row_stride = next((stride[i] for i in range(len(shape)-2, -1, -1)
                       if shape[i] > 1), stride[0])
    expected = row_stride
    for dim, step in zip(reversed(shape[:-1]), reversed(stride[:-1])):
        if dim > 1:
            _require(step == expected, name + " leading dimensions are not a strided token matrix")
        expected *= dim
    return (shape, stride, str(value.dtype), str(device), prod(shape[:-1]),
            row_stride, stride[-1])


def _ring_metadata(ring, rows, device):
    _require(isinstance(ring, RingBinding), "ring must be the shared RingBinding")
    shape = _shape(ring.recent_latent, "recent_latent")
    _require(len(shape) == 3 and shape[-1] == 512, "recent_latent must be [P,C,512]")
    pools, capacity, _ = shape
    bound = ring.max_query_per_request
    _require(type(bound) is int and bound > 0 and capacity % 128 == 0 and
             capacity >= 128 + bound - 1, "ring C must be a multiple of 128 covering 128+query_bound-1")
    expected = (
        ("recent_latent", (pools, capacity, 512), "bfloat16"),
        ("recent_rope", (pools, capacity, 64), "bfloat16"),
        ("recent_position_tags", (pools, capacity), "int64"),
        ("recent_epoch_tags", (pools, capacity), "int64"),
        ("token_ring_slots", (rows,), "int32"),
        ("token_positions", (rows,), "int64"),
        ("token_request_epochs", (rows,), "int64"),
        ("token_valid", (rows,), "bool"),
    )
    signatures = []
    for name, dimensions, dtype in expected:
        value = getattr(ring, name)
        _require(_shape(value, name) == dimensions, name + " shape mismatch")
        _require(str(getattr(value, "dtype", None)) == "torch." + dtype, name + " dtype mismatch")
        _require(str(getattr(value, "device", None)) == device, name + " device mismatch")
        _require(callable(getattr(value, "is_contiguous", None)) and value.is_contiguous(),
                 name + " must retain contiguous persistent storage")
        signatures.append((name, dimensions, tuple(value.stride()), str(value.dtype), str(value.device)))
    return pools, capacity, tuple(signatures), bound


@lru_cache(maxsize=1)
def _default_kernel_factory():
    import torch  # Device driver initialization is preparation work, never an import side effect.
    import triton
    global tl
    import triton.language as tl

    @triton.jit
    def native_bf16_ring_copy(Latent, Rope, RingLatent, RingRope, PositionTags,
                             EpochTags, Banks, Positions, Epochs, Valid,
                             POOLS: tl.constexpr, CAPACITY: tl.constexpr,
                             LATENT_ROW: tl.constexpr, LATENT_COL: tl.constexpr,
                             ROPE_ROW: tl.constexpr, ROPE_COL: tl.constexpr):
        token = tl.program_id(0)
        valid = tl.load(Valid + token)
        bank = tl.load(Banks + token).to(tl.int64)
        position = tl.load(Positions + token)
        epoch = tl.load(Epochs + token)
        writable = valid & (bank >= 0) & (bank < POOLS) & (position >= 0) & (epoch >= 0)
        # Invalid padding uses a safe address AND masks every payload/tag store.
        row = tl.where(writable, bank * CAPACITY + position % CAPACITY, 0)
        latent_col = tl.arange(0, 512)
        rope_col = tl.arange(0, 64)
        latent = tl.load(Latent + token * LATENT_ROW + latent_col * LATENT_COL,
                         mask=writable, other=0)
        rope = tl.load(Rope + token * ROPE_ROW + rope_col * ROPE_COL,
                       mask=writable, other=0)
        tl.store(RingLatent + row * 512 + latent_col, latent, mask=writable)
        tl.store(RingRope + row * 64 + rope_col, rope, mask=writable)
        tl.store(PositionTags + row, position, mask=writable)
        tl.store(EpochTags + row, epoch, mask=writable)

    return native_bf16_ring_copy


class PreparedNativeWriter:
    """One static token shape and persistent RingBinding, warmed before capture.

    ``kernel_factory`` is a zero-argument injection for host contract tests; its
    result must implement Triton's ``kernel[grid](*args, **constants)`` protocol.
    ``check_is_capture`` optionally supplies the caller's capture-state query.
    Distinct concurrent execution lanes use separate prepared writers/scratch.
    Shared physical rings require disjoint or ordered logical destinations;
    routing must assign unique (bank, position%C) destinations for valid rows.
    """
    def __init__(self, latent, rope, ring, *, kernel_factory=None, check_is_capture=None):
        self._latent_metadata = _input_metadata(latent, 512, "latent")
        self._rope_metadata = _input_metadata(rope, 64, "rope")
        _require(self._latent_metadata[4] == self._rope_metadata[4], "latent/RoPE token count mismatch")
        _require(self._latent_metadata[3] == self._rope_metadata[3], "latent/RoPE device mismatch")
        self.token_count = self._latent_metadata[4]
        self._ring_metadata = _ring_metadata(ring, self.token_count, self._latent_metadata[3])
        self.ring = ring
        self._ring_buffers = tuple(getattr(ring, name) for name, *_ in self._ring_metadata[2])
        self._check_is_capture = check_is_capture
        _require(check_is_capture is None or callable(check_is_capture), "check_is_capture must be callable")
        _require(check_is_capture is None or not check_is_capture(), "Native writer preparation must precede capture")
        self._kernel = (kernel_factory or _default_kernel_factory)()
        self.warmed = False
        self._constants = dict(POOLS=self._ring_metadata[0], CAPACITY=self._ring_metadata[1],
            LATENT_ROW=self._latent_metadata[5], LATENT_COL=self._latent_metadata[6],
            ROPE_ROW=self._rope_metadata[5], ROPE_COL=self._rope_metadata[6])

    def write(self, latent, rope, ring=None):
        """Metadata checks and one launch; warmed calls allocate no device buffers."""
        ring = self.ring if ring is None else ring
        _require(_input_metadata(latent, 512, "latent") == self._latent_metadata,
                 "latent metadata changed; prepare this static shape outside capture")
        _require(_input_metadata(rope, 64, "rope") == self._rope_metadata,
                 "RoPE metadata changed; prepare this static shape outside capture")
        _require(_ring_metadata(ring, self.token_count, self._latent_metadata[3]) == self._ring_metadata,
                 "ring metadata changed; prepare outside capture")
        _require(all(getattr(ring, name) is original for (name, *_), original in
                     zip(self._ring_metadata[2], self._ring_buffers)),
                 "prepared ring/routing buffers must retain their original identities")
        if not self.warmed:
            _require(self._check_is_capture is None or not self._check_is_capture(),
                     "Native writer must be warmed outside capture before its first launch")
        self._kernel[(self.token_count,)](latent, rope, *self._ring_buffers, **self._constants)
        self.warmed = True
        return ring

    warmup = write
    __call__ = write

    def configuration(self):
        return dict(writer="native_bf16_ring_copy", device_validation="PENDING",
                    first_a5_compile="PENDING", first_a5_runtime="PENDING",
                    performance_claim=False, warmed=self.warmed,
                    token_count=self.token_count, ring_capacity=self._ring_metadata[1],
                    original_bf16_copy=True, payload_recompute=False,
                    current_stream=True)


def prepare_native_writer(latent, rope, ring, **kwargs):
    """Prepare metadata/JIT only; explicitly warm before graph capture."""
    return PreparedNativeWriter(latent, rope, ring, **kwargs)
