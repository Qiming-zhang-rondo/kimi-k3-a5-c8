"""Store already-native-quantized C4 bytes inside the BF16 hot preparation.

The caller retains the original native quantizer and packing. This candidate
only removes its separate cold index_copy_: the native packed bytes, original
BF16 hot latent/RoPE and unchanged two-scan partition share one Triton launch.
Cold and hot physical slot tensors must remain unchanged after setup. Runtime
bounds are also masked and reported, but duplicate slot detection is setup-only.
Device compilation, correctness and performance still require A5 validation.
"""
import numpy as np

_KERNEL = None


def _load_kernel():
    global _KERNEL, tl
    if _KERNEL is not None:
        return _KERNEL
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("Cold-store preparation requires the installed Ascend Triton build") from exc

    @triton.jit
    def _cold_bf16_hot_prepare_kernel(INDICES, POSITIONS, OLD, RECENT_GLOBAL, RECENT_RING,
                                 OLD_PRESENT, RECENT_PRESENT, INVALID_ROWS,
                                 KV, ROPE, HOT_SLOTS, HOT_KV, HOT_ROPE,
                                 PACKED, COLD_SLOTS, COLD,
                                 S: tl.constexpr, HW: tl.constexpr, WINDOW: tl.constexpr,
                                 CAP: tl.constexpr, BLOCK: tl.constexpr,
                                 KV_BYTES: tl.constexpr, ROPE_BYTES: tl.constexpr,
                                 HOT_CAPACITY: tl.constexpr, STORE_BLOCK: tl.constexpr,
                                 COLD_WIDTH: tl.constexpr, COLD_CAPACITY: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        byte = tl.arange(0, STORE_BLOCK)
        cold_slot = tl.load(COLD_SLOTS+row).to(tl.int64)
        cold_slot_ok = (cold_slot >= 0) & (cold_slot < COLD_CAPACITY)
        cold_bytes = tl.load(PACKED+row*COLD_WIDTH+byte, mask=byte < COLD_WIDTH, other=0)
        tl.store(COLD+cold_slot*COLD_WIDTH+byte, cold_bytes,
                 mask=cold_slot_ok & (byte < COLD_WIDTH))
        slot = tl.load(HOT_SLOTS+row).to(tl.int64)
        slot_ok = (slot >= 0) & (slot < HOT_CAPACITY)
        kv = tl.load(KV+row*KV_BYTES+byte, mask=byte < KV_BYTES, other=0)
        rope = tl.load(ROPE+row*ROPE_BYTES+byte, mask=byte < ROPE_BYTES, other=0)
        tl.store(HOT_KV+slot*KV_BYTES+byte, kv, mask=slot_ok & (byte < KV_BYTES))
        tl.store(HOT_ROPE+slot*ROPE_BYTES+byte, rope, mask=slot_ok & (byte < ROPE_BYTES))

        lane = tl.arange(0, BLOCK)
        key = tl.load(INDICES+row*S+lane, mask=lane < S, other=-1)
        position = tl.load(POSITIONS+row).to(tl.int64)
        visible = (lane < S) & (key >= 0) & (key.to(tl.int64) <= position)
        recent = visible & (key.to(tl.int64) > position-WINDOW)
        old = visible & (recent == 0)
        old_rank = tl.cumsum(old.to(tl.int32), 0)-1
        recent_rank = tl.cumsum(recent.to(tl.int32), 0)-1
        old_count = tl.sum(old.to(tl.int32), 0)
        recent_count = tl.sum(recent.to(tl.int32), 0)
        tl.store(OLD_PRESENT+row, old_count > 0)
        tl.store(RECENT_PRESENT+row, recent_count > 0)
        bad_key = (lane < S) & ((key < -1) | ((key >= 0) & (key.to(tl.int64) > position)))
        # This reader plan requires a full hot prefix. Changed counts must be
        # visible to the correctness gate, never silently treated as supported.
        invalid = ((position < 0) | (recent_count != HW) | (slot_ok == 0) | (cold_slot_ok == 0)
                   | (tl.sum(bad_key.to(tl.int32), 0) > 0))
        tl.store(INVALID_ROWS+row, invalid)
        tl.store(OLD+row*S+old_rank, key, mask=old)
        tl.store(OLD+row*S+lane, -1, mask=(lane < S) & (lane >= tl.maximum(old_count, 1)))
        tl.store(OLD+row*S, 0, mask=old_count == 0)
        hot_store = recent & (recent_rank < HW)
        hot_padding = (lane < HW) & (lane >= tl.maximum(recent_count, 1))
        tl.store(RECENT_GLOBAL+row*HW+recent_rank, key, mask=hot_store)
        tl.store(RECENT_GLOBAL+row*HW+lane, -1, mask=hot_padding)
        tl.store(RECENT_GLOBAL+row*HW, 0, mask=recent_count == 0)
        tl.store(RECENT_RING+row*HW+recent_rank, key % CAP, mask=hot_store)
        tl.store(RECENT_RING+row*HW+lane, -1, mask=hot_padding)
        tl.store(RECENT_RING+row*HW, 0, mask=recent_count == 0)

    _KERNEL = _cold_bf16_hot_prepare_kernel
    return _KERNEL


class PreparedColdBf16HotStorePartition:
    """Wrap a validated pinned PreparedBf16HotStorePartition, without calling it.

    ``packed_template`` is the actual uint8[Q,400] produced by variant._quantize
    during setup. ``call(packed)`` must receive the current step's result; no
    packed pointer or content from setup is reused. Slot host checks run only
    in construction, outside capture/timing. No quantizer is called here.
    """
    def __init__(self, old_prepare, cold_cache, current_slots, packed_template):
        self.base = old_prepare.base
        self.torch = self.base.torch
        self.count = old_prepare.count
        self.device = str(cold_cache.device)
        if not 1 <= self.count <= 1024 or self.device.split(":", 1)[0] not in ("npu", "privateuseone"):
            raise ValueError("Require Q1..1024 on an Ascend NPU")
        constants = dict(old_prepare.constants)
        required = dict(S=2048, HW=128, WINDOW=128, KV_BYTES=1024, ROPE_BYTES=128, STORE_BLOCK=1024)
        if any(constants.get(name) != value for name, value in required.items()):
            raise ValueError("Require original D512/R64 BF16 hot preparation with TopK2048/W128")
        shape = tuple(cold_cache.shape)
        if (len(shape) != 4 or shape[0] < 1 or shape[1:3] != (128, 1)
                or shape[-1] != 400 or cold_cache.dtype not in (self.torch.int8, self.torch.uint8)
                or cold_cache.element_size() != 1):
            raise ValueError("Cold cache must be paged int8/uint8 [blocks,128,1,400]")
        self.cold_width, self.cold_capacity = shape[-1], cold_cache.numel()//shape[-1]
        if self.cold_capacity not in (8192, 32768):
            raise ValueError("Cold-store experiment requires K8192 or K32768")
        if (tuple(current_slots.shape) != (self.count,) or
                current_slots.dtype not in (self.torch.int32, self.torch.int64)):
            raise ValueError("Require one integer cold physical slot per query")
        self._inputs = tuple(old_prepare._inputs)
        self.outputs = old_prepare.outputs
        names = ("old", "recent_global", "recent_ring", "old_present", "recent_present", "invalid_rows")
        if len(self._inputs) != 8 or any(self._inputs[n+2] is not self.outputs[name] for n, name in enumerate(names)):
            raise ValueError("Prepared partition buffers differ from their outputs")
        self.kv_bytes, self.rope_bytes = old_prepare.kv_bytes, old_prepare.rope_bytes
        self.hot_slots = old_prepare.hot_slots
        self.hot_bytes, self.hot_rope_bytes = old_prepare.hot_bytes, old_prepare.hot_rope_bytes
        for value in (*self._inputs, self.kv_bytes, self.rope_bytes, self.hot_slots,
                      self.hot_bytes, self.hot_rope_bytes, cold_cache, current_slots):
            self._check_tensor(value)
        self._check_packed(packed_template)
        slots = np.asarray(current_slots.detach().cpu().numpy())
        if (slots.shape != (self.count,) or not np.issubdtype(slots.dtype, np.integer) or
                np.any(slots < 0) or np.any(slots >= self.cold_capacity) or
                len(np.unique(slots)) != self.count):
            raise ValueError("Cold physical slots must be unique and in bounds")
        self.cold_slots = current_slots
        self.cold_bytes = cold_cache.view(self.torch.uint8)
        self.constants = dict(constants, COLD_WIDTH=self.cold_width, COLD_CAPACITY=self.cold_capacity)
        self.kernel = _load_kernel()
        self.metadata = dict(implementation="native_packed_cold_store_bf16_hot_two_scan_partition",
            kernel_launches=1, prepare_kernel_launches=1, quantizer_api_calls=0,
            caller_native_quantizer_and_packing_unchanged=True, cold_scatter_kernel_calls=0,
            cold_writer_included=True, cold_packed_bytes_preserved=True,
            bf16_latent_and_rope_byte_preserving=True, partition_outputs_reused=True,
            standalone_partition_calls=0, no_timed_host_sync=True,
            current_packed_pointer_passed_every_call=True, cold_width=self.cold_width,
            cold_capacity=self.cold_capacity, query_rows=self.count,
            slot_uniqueness_validation="setup_only_slots_must_remain_unchanged",
            status="EXPERIMENTAL_REQUIRES_A5_VALIDATION")

    def _check_tensor(self, value):
        if (str(value.device) != self.device or not value.is_contiguous() or
                getattr(value, "requires_grad", False)):
            raise ValueError("Preparation requires contiguous inference tensors on one NPU")

    def _check_packed(self, packed):
        self._check_tensor(packed)
        if (tuple(packed.shape) != (self.count, self.cold_width) or
                packed.dtype != self.torch.uint8 or packed.element_size() != 1):
            raise ValueError("Current native packed bytes must be uint8[Q,400]")

    def call(self, packed):
        self._check_packed(packed)
        self.kernel[(self.count,)](*self._inputs, self.kv_bytes, self.rope_bytes,
            self.hot_slots, self.hot_bytes, self.hot_rope_bytes,
            packed, self.cold_slots, self.cold_bytes, **self.constants)
        return self.outputs
