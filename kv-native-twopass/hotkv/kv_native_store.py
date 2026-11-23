"""Native quantization with a single raw-byte packing/scatter kernel.

This does not implement or approximate either quantizer. MXFP4/E8M0 and
E4M3FN/FP32 are produced by the same Torch-NPU calls and recipe arguments as
the reference writer. Only concatenation and physical-slot stores are fused.
Slot uniqueness/range checks happen once, outside timing and graph capture.
"""

import numpy as np


def validate_slots(slots, count, capacity, name):
    values = np.asarray(slots)
    if values.shape != (count,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must contain one integer physical slot per input token")
    if np.any(values < 0) or np.any(values >= capacity) or len(np.unique(values)) != count:
        raise ValueError(f"{name} must be unique and in cache bounds; no concurrent overwrite")


_KERNEL = None


def _load_kernel():
    global _KERNEL, tl
    if _KERNEL is not None:
        return _KERNEL
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("Fused store requires the Triton build matching this Ascend environment") from exc

    @triton.jit
    def _store_kernel(Q4, S4, Q8, S8, ROPE, COLD_SLOTS, HOT_SLOTS, COLD, HOT,
                      D: tl.constexpr, R: tl.constexpr, MIXED: tl.constexpr,
                      COLD_WIDTH: tl.constexpr, HOT_WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        lane = tl.arange(0, BLOCK)
        cold_slot = tl.load(COLD_SLOTS + row).to(tl.int64)
        q4 = tl.load(Q4 + row*(D//2) + lane, mask=lane < D//2, other=0)
        s4 = tl.load(S4 + row*(D//32) + (lane-D//2-2*R),
                     mask=(lane >= D//2+2*R) & (lane < COLD_WIDTH), other=0)
        rope4 = tl.full((BLOCK,), 0, tl.uint8)
        if R > 0:
            rope4 = tl.load(ROPE + row*(2*R) + (lane-D//2),
                            mask=(lane >= D//2) & (lane < D//2+2*R), other=0)
        cold_byte = tl.where(lane < D//2, q4, tl.where(lane < D//2+2*R, rope4, s4))
        tl.store(COLD + cold_slot*COLD_WIDTH + lane, cold_byte, mask=lane < COLD_WIDTH)
        if MIXED:
            hot_slot = tl.load(HOT_SLOTS + row).to(tl.int64)
            q8 = tl.load(Q8 + row*D + lane, mask=lane < D, other=0)
            s8 = tl.load(S8 + row*(D//32) + (lane-D-2*R),
                         mask=(lane >= D+2*R) & (lane < HOT_WIDTH), other=0)
            rope8 = tl.full((BLOCK,), 0, tl.uint8)
            if R > 0:
                rope8 = tl.load(ROPE + row*(2*R) + (lane-D),
                                mask=(lane >= D) & (lane < D+2*R), other=0)
            hot_byte = tl.where(lane < D, q8, tl.where(lane < D+2*R, rope8, s8))
            tl.store(HOT + hot_slot*HOT_WIDTH + lane, hot_byte, mask=lane < HOT_WIDTH)

    _KERNEL = _store_kernel
    return _KERNEL


class NativeQuantizedStore:
    """Fixed-input experiment writer; all native quantization remains timed."""

    def __init__(self, base, variant, cold_cache, *, hot_cache=None, hot_slots=None):
        self.base, self.variant = base, variant
        self.cold_cache, self.hot_cache = cold_cache, hot_cache
        self.mixed = hot_cache is not None
        self.dim, self.rope_dim = int(base.dim), int(base.rope_dim)
        if self.dim <= 0 or self.dim % 128 or self.rope_dim < 0:
            raise ValueError("Fused store requires positive D divisible by 128 and nonnegative RoPE size")
        self.count = base.current_kv.numel()//self.dim
        if self.count < 1 or base.current_kv.numel() != self.count*self.dim:
            raise ValueError("Fused store requires nonempty whole current tokens")
        t = base.torch
        if base.current_kv.dtype != t.bfloat16 or base.current_rope.dtype != t.bfloat16:
            raise ValueError("Fused store inputs must be original BF16 KV and RoPE")
        if base.current_rope.numel() != self.count*self.rope_dim:
            raise ValueError("Unexpected current RoPE size")
        self.cold_width = self.dim//2+2*self.rope_dim+self.dim//32
        self.hot_width = self.dim+2*self.rope_dim+self.dim//32
        device = str(base.current_kv.device)
        if device.split(":", 1)[0] not in ("npu", "privateuseone"):
            raise ValueError("Fused store requires Ascend NPU tensors")
        for name, tensor in (("current_kv", base.current_kv), ("current_rope", base.current_rope),
                             ("cold_cache", cold_cache), ("cold_slots", base.current_slots)):
            if str(tensor.device) != device or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous on the same NPU")
        if cold_cache.shape[-1] != self.cold_width or cold_cache.element_size() != 1:
            raise ValueError("Cold cache is not native MXFP4/RoPE/E8M0 packed storage")
        if base.current_slots.dtype not in (t.int32, t.int64):
            raise ValueError("Cold slots must use integer storage")
        validate_slots(base.current_slots.detach().cpu().numpy(), self.count,
                       cold_cache.numel()//self.cold_width, "cold_slots")
        if self.mixed:
            if (hot_cache.shape[-1] != self.hot_width or hot_cache.element_size() != 1 or
                    hot_slots is None or hot_slots.dtype not in (t.int32, t.int64)):
                raise ValueError("Hot cache/slots do not match native E4M3FN/RoPE/FP32 storage")
            for name, tensor in (("hot_cache", hot_cache), ("hot_slots", hot_slots)):
                if str(tensor.device) != device or not tensor.is_contiguous():
                    raise ValueError(f"{name} must be contiguous on the same NPU")
            validate_slots(hot_slots.detach().cpu().numpy(), self.count,
                           hot_cache.numel()//self.hot_width, "hot_slots")
        self.kv = base.current_kv.view(-1, 1, self.dim)
        self.rope_bytes = base.current_rope.view(t.uint8)
        self.cold_bytes = cold_cache.view(t.uint8)
        self.hot_bytes = hot_cache.view(t.uint8) if self.mixed else self.cold_bytes
        self.cold_slots = base.current_slots
        self.hot_slots = hot_slots if self.mixed else self.cold_slots
        width = self.hot_width if self.mixed else self.cold_width
        self.constants = dict(D=self.dim, R=self.rope_dim, MIXED=self.mixed,
                              COLD_WIDTH=self.cold_width, HOT_WIDTH=self.hot_width,
                              BLOCK=1 << (width-1).bit_length())
        self.kernel = _load_kernel()
        self.metadata = dict(implementation="native_quantizers_then_fused_raw_byte_store",
                             quantizer_api_calls=2 if self.mixed else 1, pack_scatter_launches=1,
                             native_quantizer_arguments_unchanged=True,
                             quantization_fused=False, no_timed_host_sync=True,
                             input_scope="fixed original BF16 current tokens and physical slots",
                             status="EXPERIMENTAL_REQUIRES_A5_VALIDATION")

    def _bytes(self, tensor, bytes_per_token, name):
        if tensor.numel()*tensor.element_size() != self.count*bytes_per_token:
            raise RuntimeError(f"Unexpected native {name} byte count")
        if not tensor.is_contiguous() or str(tensor.device) != str(self.kv.device):
            raise RuntimeError(f"Native {name} must be contiguous on the input NPU")
        return tensor.view(self.base.torch.uint8)

    def call(self):
        base, t = self.base, self.base.torch
        q4, s4 = base.torch_npu.npu_dynamic_mx_quant(self.kv, **self.variant._quantizer_arguments)
        if q4.element_size() != 1 or s4.element_size() != 1:
            raise RuntimeError("Native MXFP4/E8M0 quantizer must return one-byte storage")
        q4b = self._bytes(q4, self.dim//2, "MXFP4")
        s4b = self._bytes(s4, self.dim//32, "E8M0")
        q8b, s8b = q4b, s4b  # Unused pointers in the split-C4 specialization.
        if self.mixed:
            q8, s8 = base.torch_npu.npu_dynamic_block_quant(
                self.kv, dst_type=t.float8_e4m3fn, row_block_size=1, col_block_size=128)
            if q8.dtype != t.float8_e4m3fn or s8.dtype != t.float32:
                raise RuntimeError("Native C8 quantizer must return E4M3FN and FP32 scales")
            q8b = self._bytes(q8, self.dim, "FP8")
            s8b = self._bytes(s8, self.dim//32, "FP32 scale/128")
        self.kernel[(self.count,)](q4b, s4b, q8b, s8b, self.rope_bytes, self.cold_slots,
                                   self.hot_slots, self.cold_bytes, self.hot_bytes, **self.constants)
        return self.cold_bytes
