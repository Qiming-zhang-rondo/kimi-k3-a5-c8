"""Experimental one-launch merge for two native sparse-attention results.

This module implements ONLY the final elementwise merge. Both attention
segments must already have been computed by the native readers. A state is
``(output[T,H,D] BF16/FP32, maximum[T,H,1] FP32, denominator[T,H,1] FP32)``.
Both outputs and the destination must use the same dtype.
The output is normalized attention; denominator is sum(exp(score-maximum))
in the same natural-exponential domain for both readers. Inputs must use the
same score scale and value basis. Raw reader layouts must be normalized by
the adapter before calling this module.

An empty state is maximum=-inf, denominator=0; its output may contain NaN.
Optional boolean [T] presence masks are authoritative: false marks a native
dummy-row result as empty; true requires valid nonempty statistics, even if
the reader returned the empty sentinel. Invalid NONempty statistics propagate NaN, including
nonfinite maxima/denominators and nonpositive denominators. Both-empty rows
produce zero. No attention, cache decoding, extra rotation or fallback occurs.

Pass a preallocated ``out`` during timing/graph capture. Warm up once before
capture to compile the installed Ascend Triton kernel. Validation examines
host metadata only. A5 compilation, correctness and timing need device tests;
CPU tests do not establish any of them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MergeLayout:
    tokens: int
    heads: int
    features: int


def _tensor_metadata(tensor, name, shape, dtype, device):
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if str(tensor.dtype) != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if str(tensor.device) != device:
        raise ValueError(f"{name} must be on the same NPU as left output")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous; prepare outside the timed call")
    if getattr(tensor, "requires_grad", False):
        raise ValueError("merge_states is inference-only")


def validate_states(left, right, out=None, *, left_present=None, right_present=None):
    """Validate only static metadata; no copies, tensor arithmetic or sync."""
    for name, state in (("left", left), ("right", right)):
        if not isinstance(state, (tuple, list)) or len(state) != 3:
            raise ValueError(f"{name} must be (output, maximum, denominator)")
    shape = tuple(left[0].shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("left output must have shape [T,H,D] with positive dimensions")
    device = str(left[0].device)
    if device.split(":", 1)[0] not in ("npu", "privateuseone"):
        raise ValueError("merge_states requires Ascend NPU tensors")
    output_dtype = str(left[0].dtype)
    if output_dtype not in ("torch.bfloat16", "torch.float32"):
        raise ValueError("native outputs must have dtype torch.bfloat16 or torch.float32")
    stat_shape = shape[:2] + (1,)
    for name, state in (("left", left), ("right", right)):
        _tensor_metadata(state[0], name + " output", shape, output_dtype, device)
        _tensor_metadata(state[1], name + " maximum", stat_shape, "torch.float32", device)
        _tensor_metadata(state[2], name + " denominator", stat_shape, "torch.float32", device)
    if out is not None:
        _tensor_metadata(out, "out", shape, output_dtype, device)
    for name, presence in (("left_present", left_present), ("right_present", right_present)):
        if presence is not None:
            _tensor_metadata(presence, name, (shape[0],), "torch.bool", device)
    return MergeLayout(*shape)


_KERNEL = None
BLOCK = 256


def _load_kernel():
    global _KERNEL, tl
    if _KERNEL is not None:
        return _KERNEL
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("Native-result merge needs the Triton build matching this Ascend environment") from exc

    @triton.jit
    def _merge_kernel(LEFT, LEFT_MAX, LEFT_SUM, RIGHT, RIGHT_MAX, RIGHT_SUM,
                       LEFT_PRESENT, RIGHT_PRESENT, OUT,
                       H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
                       HAS_LEFT_PRESENT: tl.constexpr, HAS_RIGHT_PRESENT: tl.constexpr):
        row = tl.program_id(1)
        feature = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        lm = tl.load(LEFT_MAX + row)
        ls = tl.load(LEFT_SUM + row)
        rm = tl.load(RIGHT_MAX + row)
        rs = tl.load(RIGHT_SUM + row)
        left_empty = (ls == 0.0) & (lm == float("-inf"))
        right_empty = (rs == 0.0) & (rm == float("-inf"))
        if HAS_LEFT_PRESENT:
            left_empty = ~tl.load(LEFT_PRESENT + row // H)
        if HAS_RIGHT_PRESENT:
            right_empty = ~tl.load(RIGHT_PRESENT + row // H)
        both_empty = left_empty & right_empty

        # Presence-masked dummy rows are ignored. Every other malformed state
        # must remain visible instead of being mistaken for an empty segment.
        left_bad = ~left_empty & ((lm != lm) | (lm == float("inf")) | (lm == float("-inf"))
                                  | (ls != ls) | (ls == float("inf")) | (ls <= 0.0))
        right_bad = ~right_empty & ((rm != rm) | (rm == float("inf")) | (rm == float("-inf"))
                                    | (rs != rs) | (rs == float("inf")) | (rs <= 0.0))
        lm = tl.where(left_empty, float("-inf"), lm)
        rm = tl.where(right_empty, float("-inf"), rm)
        common_max = tl.maximum(lm, rm)
        common_max = tl.where(both_empty, 0.0, common_max)
        # Shift maxima BEFORE adding log(sum), retaining relative denominators
        # even when both maxima are large. The second shift bounds exponentials
        # by one and avoids overflow when adding large finite denominators.
        left_log_weight = tl.where(left_empty, float("-inf"),
                                    (lm - common_max) + tl.log(tl.where(left_empty, 1.0, ls)))
        right_log_weight = tl.where(right_empty, float("-inf"),
                                     (rm - common_max) + tl.log(tl.where(right_empty, 1.0, rs)))
        common_log_weight = tl.maximum(left_log_weight, right_log_weight)
        common_log_weight = tl.where(both_empty, 0.0, common_log_weight)
        left_weight = tl.where(left_empty, 0.0, tl.exp(left_log_weight - common_log_weight))
        right_weight = tl.where(right_empty, 0.0, tl.exp(right_log_weight - common_log_weight))
        denominator = tl.where(both_empty, 1.0, left_weight + right_weight)
        left_weight = left_weight / denominator
        right_weight = right_weight / denominator

        offset = row.to(tl.int64) * D + feature
        left_value = tl.load(LEFT + offset, mask=feature < D, other=0).to(tl.float32)
        right_value = tl.load(RIGHT + offset, mask=feature < D, other=0).to(tl.float32)
        # 0 * NaN is still NaN: explicitly neutralize empty outputs first.
        left_value = tl.where(left_empty, 0.0, left_value)
        right_value = tl.where(right_empty, 0.0, right_value)
        result = left_weight * left_value + right_weight * right_value
        result = tl.where(left_bad | right_bad, float("nan"), result)
        # tl.store converts to OUT's element type. Preserve native FP32
        # outputs instead of silently rounding them through BF16 first.
        tl.store(OUT + offset, result, mask=feature < D)

    _KERNEL = _merge_kernel
    return _KERNEL


def merge_states(left, right, out=None, *, left_present=None, right_present=None):
    """Merge two native results in one launch, returning ``out``.

    With ``out`` supplied this performs no allocation, copies or host sync.
    Optional boolean [T] masks handle empty subsets computed using safe dummy
    indices. True mask entries require nonempty valid statistics. The result
    retains the native output dtype (BF16 or FP32); mask tensors must already
    exist on device. The two state outputs
    and statistics must remain alive until this launch finishes on the current
    stream. ``out`` must not partially overlap the input tensor storage.
    """
    layout = validate_states(left, right, out, left_present=left_present, right_present=right_present)
    if out is None:
        import torch
        out = torch.empty_like(left[0])
    kernel = _load_kernel()
    # Unused pointer placeholders keep one fixed ABI; constexpr flags remove
    # their loads entirely when the corresponding presence mask is absent.
    lp = left_present if left_present is not None else left[1]
    rp = right_present if right_present is not None else right[1]
    kernel[((layout.features + BLOCK - 1) // BLOCK, layout.tokens * layout.heads)](
        *left, *right, lp, rp, out, H=layout.heads, D=layout.features, BLOCK=BLOCK,
        HAS_LEFT_PRESENT=left_present is not None, HAS_RIGHT_PRESENT=right_present is not None)
    return out
