"""Scoped native BF16 capture and the reviewed DSACP token-row contract.

DSACP shards contiguous token rows over the TP group. Its packed gather is
unchanged: this correctness candidate adds one separate BF16 collective for
TP>1. The gathered columns retain row stride 576 and original BF16 values.
No tensor content is inspected on the host, and no transport/lifecycle support
is implied by these helpers. Prepare buffers and route views before capture.
"""
from contextvars import ContextVar
from dataclasses import dataclass, field

from .prolog_provider import RingBinding


class NativeCPError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise NativeCPError(message)


def _integer(name, value, *, minimum=0):
    _require(type(value) is int and value >= minimum,
             name + " must be an integer >= " + str(minimum))
    return value


_NATIVE_CAPTURE = ContextVar("recent128_native_capture", default=None)


class NativeCaptureScope:
    """One forward's raw tap references, isolated across contexts/nesting.

    The scope must enclose the original exec/gather/store sequence; its only
    device-facing action is retaining the exact tensors passed to the tap.
    Capture success is explicit so a bypassed native producer cannot silently
    reuse an earlier forward's tensors.
    """

    def __init__(self, impl, metadata):
        self.impl = impl
        self.metadata = metadata
        self.latent = self.rope = None
        self.capture_count = 0
        self.writes = 0  # The runtime owns write completion/counting.
        self._token = None
        self._used = False

    def __enter__(self):
        _require(not self._used, "NativeCaptureScope is single-use")
        self._used = True
        self._token = _NATIVE_CAPTURE.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _NATIVE_CAPTURE.reset(self._token)
        self._token = None
        return False

    def require_capture(self):
        _require(self.capture_count == 1,
                 "Native forward requires exactly one original BF16 KV capture")
        return self.latent, self.rope


def capture_native(impl, latent, rope):
    """Record original refs for this implementation; return whether scoped.

    Unscoped taps remain available to the separate accuracy path. A nested
    scope for another implementation never steals its caller's capture.
    """
    scope = _NATIVE_CAPTURE.get()
    if scope is None or scope.impl is not impl:
        return False
    scope.capture_count += 1
    _require(scope.capture_count == 1,
             "Native forward captured BF16 KV more than once")
    scope.latent, scope.rope = latent, rope
    return True


def get_native_scope(impl=None):
    """Return only the innermost matching forward; never search outer scopes."""
    scope = _NATIVE_CAPTURE.get()
    if scope is None or (impl is not None and scope.impl is not impl):
        return None
    return scope


@dataclass(frozen=True)
class DSACPLayout:
    """Static host integers copied from the reviewed DSACPContext fields."""

    global_tokens: int
    num_tokens_pad: int
    local_start: int
    # Actual token validity changes between warmup/capture/replay. It is
    # carried by device token_valid, not a separate allocation/cache shape.
    local_end: int = field(compare=False)
    local_end_with_pad: int
    tp_size: int
    tp_rank: int

    def __post_init__(self):
        for name in ("global_tokens", "num_tokens_pad", "tp_size"):
            _integer(name, getattr(self, name), minimum=1)
        for name in ("local_start", "local_end", "local_end_with_pad", "tp_rank"):
            _integer(name, getattr(self, name))
        _require(self.tp_rank < self.tp_size, "tp_rank exceeds TP group")
        expected_pad = ((self.global_tokens + self.tp_size - 1) // self.tp_size) * self.tp_size
        _require(self.num_tokens_pad == expected_pad,
                 "DSACP num_tokens_pad must equal TP-aligned global token count")
        rows = self.num_tokens_pad // self.tp_size
        start = self.tp_rank * rows
        _require(self.local_start == start and self.local_end_with_pad == start + rows,
                 "DSACP local range must match contiguous TP rank order")
        # In an empty trailing shard local_end may precede local_start: the
        # reviewed builder uses min(local_end_with_pad, num_actual_tokens).
        _require(self.local_end <= min(self.local_end_with_pad, self.global_tokens),
                 "DSACP local_end exceeds its global/input row bound")

    @classmethod
    def from_context(cls, context, *, global_tokens, tp_size, tp_rank):
        _require(context is not None, "DSACPContext is required")
        _require(getattr(context, "num_tokens", None) == global_tokens,
                 "DSACP num_tokens must match global metadata.num_input_tokens")
        return cls(global_tokens=global_tokens,
                   num_tokens_pad=getattr(context, "num_tokens_pad", None),
                   local_start=getattr(context, "local_start", None),
                   local_end=getattr(context, "local_end", None),
                   local_end_with_pad=getattr(context, "local_end_with_pad", None),
                   tp_size=tp_size, tp_rank=tp_rank)

    @property
    def local_rows(self):
        return self.local_end_with_pad - self.local_start

    @property
    def local_input_rows(self):
        return max(0, min(self.local_rows, self.global_tokens - self.local_start))

    @property
    def local_valid_rows(self):
        return max(0, self.local_end - self.local_start)

    @property
    def extra_collectives_per_call(self):
        return int(self.tp_size > 1)

    @property
    def local_payload_bytes(self):
        return self.local_rows * 576 * 2 if self.tp_size > 1 else 0


def _tensor_metadata(tensor, name, *, shape, dtype, device, contiguous=True):
    _require(tuple(getattr(tensor, "shape", ())) == shape,
             name + " shape must be " + str(shape))
    _require(str(getattr(tensor, "dtype", None)) == "torch." + dtype,
             name + " dtype must be torch." + dtype)
    _require(getattr(tensor, "device", None) == device,
             name + " must stay on the prepared device")
    if contiguous:
        check = getattr(tensor, "is_contiguous", None)
        _require(callable(check) and check() is True, name + " must be contiguous")


def _raw_shape(tensor, name, rows, width, device):
    shape = tuple(getattr(tensor, "shape", ()))
    allowed = ((rows, width), (rows, 1, width), (rows, 1, 1, width))
    _require(shape in allowed, name + " must be native [T,1,1,D], [T,1,D] or [T,D]")
    _tensor_metadata(tensor, name, shape=shape, dtype="bfloat16", device=device)
    return shape


@dataclass(frozen=True)
class NativeCPBuffers:
    """Graph-stable BF16 exchange buffers for one exact DSACP bucket/rank."""

    layout: DSACPLayout
    device: object
    local_payload: object
    gathered_payload: object
    latent: object
    rope: object
    _local_latent_views: object
    _local_rope_views: object

    @classmethod
    def prepare(cls, layout, *, device, torch_module):
        _require(isinstance(layout, DSACPLayout), "layout must be DSACPLayout")
        # The degenerate path hands original references straight to the writer.
        if layout.tp_size == 1:
            return cls(layout, device, None, None, None, None, {}, {})
        t = torch_module
        local = t.empty((layout.local_rows, 576), dtype=t.bfloat16, device=device)
        gathered = t.empty((layout.num_tokens_pad, 576), dtype=t.bfloat16, device=device)
        local_latent, local_rope = local[:, :512], local[:, 512:]
        # Raw native singleton dimensions are known in the reviewed source.
        # Cache every accepted shape now; copy_ needs no per-call view creation.
        latent_views = {shape: local_latent.view(shape) for shape in (
            (layout.local_rows, 512), (layout.local_rows, 1, 512),
            (layout.local_rows, 1, 1, 512))}
        rope_views = {shape: local_rope.view(shape) for shape in (
            (layout.local_rows, 64), (layout.local_rows, 1, 64),
            (layout.local_rows, 1, 1, 64))}
        return cls(layout, device, local, gathered,
                   gathered[:layout.global_tokens, :512],
                   gathered[:layout.global_tokens, 512:], latent_views, rope_views)


def gather_native_rows(latent, rope, layout, buffers, *, torch_module, group=None):
    """Gather raw BF16 in the original packed gather's TP token rank order.

    ``group`` is explicitly ``get_tp_group().device_group`` from the real
    caller. Every TP rank must call once in the same collective order; validity
    never controls participation. Global graph padding stays in the result and
    is masked by the binding, while extra TP alignment rows are trimmed.
    """
    _require(isinstance(buffers, NativeCPBuffers) and buffers.layout == layout,
             "Native CP buffers were not prepared for this exact layout")
    latent_shape = _raw_shape(latent, "raw latent", layout.local_rows, 512, buffers.device)
    rope_shape = _raw_shape(rope, "raw rope", layout.local_rows, 64, buffers.device)
    if layout.tp_size == 1:
        return latent, rope
    _require(group is not None, "DSACP BF16 gather requires the explicit TP device group")
    buffers._local_latent_views[latent_shape].copy_(latent)
    buffers._local_rope_views[rope_shape].copy_(rope)
    torch_module.distributed.all_gather_into_tensor(
        buffers.gathered_payload, buffers.local_payload, group=group)
    return buffers.latent, buffers.rope


@dataclass(frozen=True)
class NativeQueryRoutes:
    """Local query routing with global ring ownership and explicit TP padding."""

    layout: DSACPLayout
    source_binding: RingBinding
    binding: RingBinding
    _source_views: tuple
    _target_views: tuple
    _padding_views: tuple

    @classmethod
    def prepare(cls, layout, ring_binding, *, torch_module):
        _require(isinstance(layout, DSACPLayout), "layout must be DSACPLayout")
        _require(isinstance(ring_binding, RingBinding), "source routes require RingBinding")
        names = ("token_ring_slots", "token_positions", "token_request_epochs", "token_valid")
        dtypes = ("int32", "int64", "int64", "bool")
        sources = tuple(getattr(ring_binding, name) for name in names)
        device = getattr(ring_binding.token_positions, "device", None)
        for source, name, dtype in zip(sources, names, dtypes):
            _tensor_metadata(source, name, shape=(layout.global_tokens,), dtype=dtype, device=device)
        if layout.tp_size == 1:
            return cls(layout, ring_binding, ring_binding, (), (), ())
        t = torch_module
        routes = tuple(t.full((layout.local_rows,), fill,
                       dtype=getattr(t, dtype), device=device)
                       for dtype, fill in zip(dtypes, (-1, -1, -1, False)))
        binding = RingBinding(ring_binding.recent_latent, ring_binding.recent_rope,
            ring_binding.recent_position_tags, ring_binding.recent_epoch_tags,
            *routes, ring_binding.max_query_per_request)
        count = layout.local_input_rows
        source_views = tuple(source[layout.local_start:layout.local_start + count] for source in sources)
        target_views = tuple(target[:count] for target in routes)
        padding_views = tuple(target[count:] for target in routes)
        return cls(layout, ring_binding, binding, source_views, target_views, padding_views)

    def update(self):
        """Refresh current routes using only cached device views and in-place ops."""
        for target, source in zip(self._target_views, self._source_views):
            target.copy_(source)
        for padding, fill in zip(self._padding_views, (-1, -1, -1, False)):
            padding.fill_(fill)
        return self.binding
