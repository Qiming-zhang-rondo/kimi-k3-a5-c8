"""Native C8 quantization, then one hot-store plus stable-partition kernel.

Only the already existing raw C8 packing/scatter and two-scan index preparation
are combined. Cold writing, both native attention calls and their merge remain
outside this class. Quantization is the unchanged native call on every call;
neither quantized current tokens nor partition results are precomputed.

The supplied PreparedPartition owns all input/output buffers. Its call method
is never invoked here. Inputs and unique physical hot slots are fixed for this
experiment; values in the input buffers may change in place. Slot validation
copies only the small slot map to CPU at construction, never during call().
CPU tests establish byte/index semantics, not A5 compilation or performance.
"""

from kv_native_store import validate_slots


_KERNEL = None


def _load_kernel():
    global _KERNEL, tl
    if _KERNEL is not None:
        return _KERNEL
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("Hot preparation requires the installed Ascend Triton build") from exc

    @triton.jit
    def _hot_prepare_kernel(INDICES, POSITIONS, OLD, RECENT_GLOBAL, RECENT_RING,
                            OLD_PRESENT, RECENT_PRESENT, INVALID_ROWS,
                            Q8, S8, ROPE, HOT_SLOTS, HOT,
                            S: tl.constexpr, HW: tl.constexpr, WINDOW: tl.constexpr,
                            CAP: tl.constexpr, BLOCK: tl.constexpr, R: tl.constexpr,
                            HOT_WIDTH: tl.constexpr, HOT_CAPACITY: tl.constexpr,
                            STORE_BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        # Store temporaries are independent of, and precede, the scan buffers.
        byte = tl.arange(0, STORE_BLOCK)
        slot = tl.load(HOT_SLOTS + row).to(tl.int64)
        slot_ok = (slot >= 0) & (slot < HOT_CAPACITY)
        q8 = tl.load(Q8 + row*512 + byte, mask=byte < 512, other=0)
        s8 = tl.load(S8 + row*16 + (byte-512-2*R),
                     mask=(byte >= 512+2*R) & (byte < HOT_WIDTH), other=0)
        rope = tl.full((STORE_BLOCK,), 0, tl.uint8)
        if R > 0:
            rope = tl.load(ROPE + row*(2*R) + (byte-512),
                           mask=(byte >= 512) & (byte < 512+2*R), other=0)
        value = tl.where(byte < 512, q8, tl.where(byte < 512+2*R, rope, s8))
        tl.store(HOT + slot*HOT_WIDTH + byte, value, mask=slot_ok & (byte < HOT_WIDTH))

        lane = tl.arange(0, BLOCK)
        key = tl.load(INDICES + row*S + lane, mask=lane < S, other=-1)
        position = tl.load(POSITIONS + row).to(tl.int64)
        visible = (lane < S) & (key >= 0) & (key.to(tl.int64) <= position)
        recent = visible & (key.to(tl.int64) > position-WINDOW)
        old = visible & (recent == 0)
        old_rank = tl.cumsum(old.to(tl.int32), 0)-1
        recent_rank = tl.cumsum(recent.to(tl.int32), 0)-1
        old_count = tl.sum(old.to(tl.int32), 0)
        recent_count = tl.sum(recent.to(tl.int32), 0)
        tl.store(OLD_PRESENT + row, old_count > 0)
        tl.store(RECENT_PRESENT + row, recent_count > 0)
        bad_key = (lane < S) & ((key < -1) | ((key >= 0) & (key.to(tl.int64) > position)))
        invalid = ((position < 0) | (recent_count > HW) | (slot_ok == 0)
                   | (tl.sum(bad_key.to(tl.int32), 0) > 0))
        tl.store(INVALID_ROWS + row, invalid)
        # Prefix, suffix and empty-row dummy stores have disjoint addresses.
        tl.store(OLD + row*S + old_rank, key, mask=old)
        tl.store(OLD + row*S + lane, -1, mask=(lane < S) & (lane >= tl.maximum(old_count, 1)))
        tl.store(OLD + row*S, 0, mask=old_count == 0)
        hot_store = recent & (recent_rank < HW)
        hot_padding = (lane < HW) & (lane >= tl.maximum(recent_count, 1))
        tl.store(RECENT_GLOBAL + row*HW + recent_rank, key, mask=hot_store)
        tl.store(RECENT_GLOBAL + row*HW + lane, -1, mask=hot_padding)
        tl.store(RECENT_GLOBAL + row*HW, 0, mask=recent_count == 0)
        tl.store(RECENT_RING + row*HW + recent_rank, key % CAP, mask=hot_store)
        tl.store(RECENT_RING + row*HW + lane, -1, mask=hot_padding)
        tl.store(RECENT_RING + row*HW, 0, mask=recent_count == 0)

    _KERNEL = _hot_prepare_kernel
    return _KERNEL


class PreparedHotStorePartition:
    """One native C8 quantizer plus one packing/partition launch per call."""

    def __init__(self, base, hot_cache, hot_slots, partition_runner):
        self.base, self.partition = base, partition_runner
        layout, t = partition_runner.layout, base.torch
        if layout.variant != "two_scan" or layout.remap_ring is not True:
            raise ValueError("Hot preparation requires two_scan partition with remap_ring=True")
        if int(base.dim) != 512 or int(base.rope_dim) not in (0, 64):
            raise ValueError("Hot preparation supports D=512 and R in {0,64}")
        self.rope_dim, self.count = int(base.rope_dim), layout.rows
        if (base.current_kv.numel() != self.count*512 or
                base.current_rope.numel() != self.count*self.rope_dim):
            raise ValueError("Current KV/RoPE token count must equal partition rows")
        if base.current_kv.dtype != t.bfloat16 or base.current_rope.dtype != t.bfloat16:
            raise ValueError("Current KV and RoPE must remain original BF16")
        device = str(base.current_kv.device)
        if device.split(":", 1)[0] not in ("npu", "privateuseone"):
            raise ValueError("Hot preparation requires Ascend NPU tensors")
        self.hot_width = 528 + 2*self.rope_dim
        shape = tuple(hot_cache.shape)
        if (len(shape) != 4 or min(shape[:2]) < 1 or shape[2:] != (1, self.hot_width)
                or hot_cache.element_size() != 1
                or hot_cache.dtype not in (t.float8_e4m3fn, t.uint8, t.int8)):
            raise ValueError("Hot cache must be native paged C8/RoPE/FP32 raw-byte storage")
        if hot_slots is None or hot_slots.dtype not in (t.int32, t.int64):
            raise ValueError("Hot slots must use integer storage")
        order = ("old", "recent_global", "recent_ring", "old_present", "recent_present", "invalid_rows")
        inputs = tuple(partition_runner._inputs)
        if len(inputs) != 8 or any(inputs[i+2] is not partition_runner.outputs[name] for i, name in enumerate(order)):
            raise ValueError("Partition buffers do not match the prepared partition contract")
        expected = [((self.count, 1, layout.sparse_capacity), t.int32), ((self.count,), None),
                    ((self.count, 1, layout.sparse_capacity), t.int32),
                    ((self.count, 1, layout.hot_width), t.int32), ((self.count, 1, layout.hot_width), t.int32),
                    ((self.count,), t.bool), ((self.count,), t.bool), ((self.count,), t.bool)]
        for tensor, (wanted_shape, dtype) in zip(inputs, expected):
            if tuple(tensor.shape) != wanted_shape or (dtype is not None and tensor.dtype != dtype):
                raise ValueError("Partition tensor shape/dtype contract mismatch")
        if inputs[1].dtype not in (t.int32, t.int64):
            raise ValueError("Partition positions must use integer storage")
        tensors = (("current_kv", base.current_kv), ("current_rope", base.current_rope),
                   ("hot_cache", hot_cache), ("hot_slots", hot_slots))
        for name, tensor in (*tensors, *(("partition", value) for value in inputs)):
            if str(tensor.device) != device or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous on the same NPU")
            if getattr(tensor, "requires_grad", False):
                raise ValueError("Hot preparation is inference-only")
        capacity = hot_cache.numel()//self.hot_width
        validate_slots(hot_slots.detach().cpu().numpy(), self.count, capacity, "hot_slots")
        self.kv = base.current_kv.view(-1, 1, 512)
        self.rope_bytes, self.hot_bytes = base.current_rope.view(t.uint8), hot_cache.view(t.uint8)
        self.hot_slots, self.outputs, self._inputs = hot_slots, partition_runner.outputs, inputs
        self.constants = dict(S=layout.sparse_capacity, HW=layout.hot_width, WINDOW=layout.window,
                              CAP=layout.ring_capacity, BLOCK=layout.block, R=self.rope_dim,
                              HOT_WIDTH=self.hot_width, HOT_CAPACITY=capacity,
                              STORE_BLOCK=1 << (self.hot_width-1).bit_length())
        self.kernel = _load_kernel()
        self.metadata = dict(implementation="native_c8_then_fused_hot_store_two_scan_partition",
                             quantizer_api_calls=1, prepare_kernel_launches=1, standalone_partition_calls=0,
                             native_quantizer_arguments_unchanged=True, quantization_fused=False,
                             partition_variant="two_scan", partition_outputs_reused=True,
                             partition_implementation=dict(partition_runner.metadata),
                             cold_writer_included=False, attention_launches=0, no_timed_host_sync=True,
                             status="EXPERIMENTAL_REQUIRES_A5_VALIDATION")

    def _bytes(self, value, width, name):
        if value.numel()*value.element_size() != self.count*width:
            raise RuntimeError(f"Unexpected native {name} byte count")
        if not value.is_contiguous() or str(value.device) != str(self.kv.device):
            raise RuntimeError(f"Native {name} must be contiguous on the input NPU")
        return value.view(self.base.torch.uint8)

    def call(self):
        base, t = self.base, self.base.torch
        quant, scale = base.torch_npu.npu_dynamic_block_quant(
            self.kv, dst_type=t.float8_e4m3fn, row_block_size=1, col_block_size=128)
        if quant.dtype != t.float8_e4m3fn or scale.dtype != t.float32:
            raise RuntimeError("Native C8 quantizer must return E4M3FN and FP32 scales")
        qbytes, sbytes = self._bytes(quant, 512, "FP8"), self._bytes(scale, 16, "FP32 scale/128")
        self.kernel[(self.count,)](*self._inputs, qbytes, sbytes, self.rope_bytes,
                                   self.hot_slots, self.hot_bytes, **self.constants)
        return self.outputs
