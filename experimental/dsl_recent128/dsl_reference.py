"""Independent CPU packing and attention reference for the experimental DSL.

These formats are FP4/E2M1 group16 with BF16 scales and FP8/E4M3FN group32
with BF16 scales. They are deliberately not named MXFP4. All 512 coordinates,
including the 64 RoPE coordinates, are quantized. No NPU or torch dependency.
"""
from __future__ import annotations

import numpy as np


def _finite(x, name):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains nonfinite values")
    return x


def bf16_round(x):
    """Round finite FP32 to BF16 with round-to-nearest, ties-to-even."""
    x = np.array(_finite(x, "BF16 input"), dtype=np.float32, copy=True)
    bits = x.view(np.uint32)
    rounded_bits = np.asarray(
        (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1)))
        & np.uint32(0xFFFF0000), dtype=np.uint32)
    rounded = rounded_bits.view(np.float32)
    if not np.isfinite(rounded).all():
        raise ValueError("BF16 rounding overflowed to nonfinite")
    return rounded


FP4_VALUES = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
_FP8_CODES = np.arange(127, dtype=np.uint8)
_FP8_EXP = (_FP8_CODES.astype(np.int32) >> 3)
_FP8_FRAC = (_FP8_CODES.astype(np.int32) & 7)
FP8_VALUES = np.where(
    _FP8_EXP == 0,
    _FP8_FRAC.astype(np.float32) * np.float32(2.0 ** -9),
    (1.0 + _FP8_FRAC / 8.0) * np.exp2(_FP8_EXP - 7),
).astype(np.float32)


def _encode_nearest(x, table, sign_bit):
    x = _finite(x, "quantization input")
    magnitude = np.abs(x)
    hi = np.searchsorted(table, magnitude, side="left").clip(0, len(table) - 1)
    lo = np.maximum(hi - 1, 0)
    # Encoding parity is significand parity for these monotonically ordered
    # positive formats, including exponent transitions and subnormals.
    low_distance = magnitude - table[lo]
    high_distance = table[hi] - magnitude
    take_lo = ((low_distance < high_distance)
               | ((low_distance == high_distance) & ((lo & 1) == 0)))
    codes = np.where(take_lo, lo, hi).astype(np.uint8)
    return codes | (np.signbit(x).astype(np.uint8) * np.uint8(sign_bit))


def encode_fp4(x):
    """Saturating finite E2M1 encoding, even coordinate in low nibble."""
    return _encode_nearest(x, FP4_VALUES, 8)


def decode_fp4(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    if np.any(codes > 15):
        raise ValueError("FP4 codes must fit in four bits")
    values = FP4_VALUES[codes & 7]
    return np.copysign(values, np.where((codes & 8) != 0, -1.0, 1.0)).astype(np.float32)


def encode_fp8(x):
    """Saturating finite E4M3FN encoding; NaN encodings are never emitted."""
    return _encode_nearest(x, FP8_VALUES, 128)


def decode_fp8(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    if np.any((codes & 127) == 127):
        raise ValueError("FP8 E4M3FN NaN encoding")
    values = FP8_VALUES[codes & 127]
    return np.copysign(values, np.where((codes & 128) != 0, -1.0, 1.0)).astype(np.float32)


def _format(kind):
    if kind == "fp4":
        return 16, 6.0, 256, encode_fp4, decode_fp4
    if kind == "fp8":
        return 32, 448.0, 512, encode_fp8, decode_fp8
    raise ValueError("kind must be 'fp4' or 'fp8'")


def quantize_rows(x, kind):
    """Return packed uint8 rows and the corresponding BF16-decoded values.

    Scale selection follows amax.clamp_min(1e-8)/maximum followed by BF16
    rounding. Encoding uses an independent software nearest-even quantizer.
    The reader's BF16 decoded multiplication is explicitly rounded to BF16.
    """
    x = _finite(x, "KV input")
    if x.ndim != 2 or x.shape[1] != 512:
        raise ValueError("KV input must have shape [K,512]")
    group, maximum, data_bytes, encode, _ = _format(kind)
    grouped = x.reshape(x.shape[0], 512 // group, group)
    amax = np.maximum(np.max(np.abs(grouped), axis=-1), np.float32(1e-8))
    scales = bf16_round(amax / np.float32(maximum))
    codes = encode(grouped / scales[..., None]).reshape(x.shape)
    if kind == "fp4":
        data = codes[:, 0::2] | (codes[:, 1::2] << 4)
    else:
        data = codes
    scale_bits = (scales.view(np.uint32) >> 16).astype("<u2")
    scale_bytes = scale_bits.view(np.uint8).reshape(x.shape[0], -1)
    packed = np.ascontiguousarray(np.concatenate((data, scale_bytes), axis=1))
    expected = data_bytes + (512 // group) * 2
    assert packed.shape == (x.shape[0], expected)
    return packed, unpack_rows(packed, kind)


def unpack_rows(packed, kind):
    """Decode byte rows independently, including little-endian BF16 scales."""
    packed = np.asarray(packed)
    group, _, data_bytes, _, decode = _format(kind)
    row_bytes = data_bytes + (512 // group) * 2
    if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] != row_bytes:
        raise ValueError(f"{kind} packed input must be uint8[K,{row_bytes}]")
    packed = np.ascontiguousarray(packed)
    data = packed[:, :data_bytes]
    if kind == "fp4":
        codes = np.empty((len(packed), 512), dtype=np.uint8)
        codes[:, 0::2] = data & 15
        codes[:, 1::2] = data >> 4
    else:
        codes = data
    scale_bytes = np.ascontiguousarray(packed[:, data_bytes:])
    scales = (scale_bytes.view("<u2").astype(np.uint32) << 16).view(np.float32)
    scales = _finite(scales, "packed scales")
    if np.any(scales <= 0):
        raise ValueError("packed scales must be positive")
    decoded = decode(codes).reshape(len(packed), 512 // group, group)
    return bf16_round(decoded * scales[..., None]).reshape(len(packed), 512)


def _validate_attention(q, kv, indices, scale, sinks):
    q = _finite(q, "query").astype(np.float64)
    kv = _finite(kv, "KV").astype(np.float64)
    raw_indices = np.asarray(indices)
    if not np.issubdtype(raw_indices.dtype, np.integer):
        raise ValueError("indices must be integers")
    indices = raw_indices.astype(np.int64)
    if (q.ndim != 3 or kv.ndim != 2 or q.shape[2] != kv.shape[1]
            or indices.ndim != 2 or len(indices) != len(q)):
        raise ValueError("expected q[T,H,D], kv[K,D], indices[T,N]")
    if not np.isfinite(scale):
        raise ValueError("scale must be finite")
    sinks = np.asarray(sinks, dtype=np.float64)
    if sinks.shape != (q.shape[1],) or np.isnan(sinks).any() or np.isposinf(sinks).any():
        raise ValueError("sinks must be [H], finite or negative infinity")
    for row in indices:
        if np.any(row < -1) or np.any(row >= len(kv)):
            raise ValueError("index out of range")
        valid = row[row >= 0]
        if np.unique(valid).size != valid.size:
            raise ValueError("duplicate sparse index")
        if np.any((row[:-1] == -1) & (row[1:] >= 0)):
            raise ValueError("valid indices must precede -1 padding")
    return q, kv, indices, float(scale), sinks


def _attention_row(query, selected, scale, sinks):
    if len(selected) == 0:
        return np.zeros_like(query), sinks.copy()
    scores = (query @ selected.T) * scale
    maximum = np.maximum(scores.max(axis=-1), sinks)
    probabilities = np.exp(scores - maximum[:, None])
    denominator = probabilities.sum(axis=-1) + np.exp(sinks - maximum)
    output = (probabilities @ selected) / denominator[:, None]
    return output, maximum + np.log(denominator)


def attention_reference(q, kv, indices, scale, sinks):
    """Full FP64 sparse attention; the sink contributes only to denominator."""
    q, kv, indices, scale, sinks = _validate_attention(q, kv, indices, scale, sinks)
    output = np.empty_like(q)
    lse = np.empty(q.shape[:2], dtype=np.float64)
    for i, row in enumerate(indices):
        output[i], lse[i] = _attention_row(q[i], kv[row[row >= 0]], scale, sinks)
    return output, lse


def mixed_reference(q, cold_dequant, hot_dequant, original_indices, hot_start, scale, sinks):
    """Use hot data only for selected positions >= each query's hot_start.

    Both dequantized arrays use the same global [K,D] token address space.
    The original sparse selection is unchanged; no extra window tokens added.
    """
    q, cold, indices, scale, sinks = _validate_attention(
        q, cold_dequant, original_indices, scale, sinks)
    hot = _finite(hot_dequant, "hot KV").astype(np.float64)
    hot_start = np.asarray(hot_start)
    if hot.shape != cold.shape or hot_start.shape != (len(q),):
        raise ValueError("hot KV must match cold shape; hot_start must have shape [T]")
    if not np.issubdtype(hot_start.dtype, np.integer):
        raise ValueError("hot_start must be integers")
    if np.any(hot_start < 0) or np.any(hot_start > len(cold)):
        raise ValueError("hot_start out of range")
    output, lse = np.empty_like(q), np.empty(q.shape[:2], dtype=np.float64)
    for i, row in enumerate(indices):
        valid = row[row >= 0]
        selected = np.where((valid >= hot_start[i])[:, None], hot[valid], cold[valid])
        output[i], lse[i] = _attention_row(q[i], selected, scale, sinks)
    return output, lse


def metrics(actual, reference):
    """Finite-aware global/per-head errors; no passing claim for NaN/Inf."""
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError("comparison shapes differ")
    finite = np.isfinite(actual).all() and np.isfinite(reference).all()
    result = {"finite": bool(finite),
              "actual_nonfinite_count": int((~np.isfinite(actual)).sum()),
              "reference_nonfinite_count": int((~np.isfinite(reference)).sum())}
    if not finite:
        result.update(relative_l2=None, max_abs=None, worst_row_relative_l2=None,
                      mean_row_cosine=None)
        return result
    actual_rows, reference_rows = actual.reshape(-1, actual.shape[-1]), reference.reshape(-1, reference.shape[-1])
    delta = actual - reference
    reference_norm = np.linalg.norm(reference)
    delta_norm = np.linalg.norm(delta)
    result["relative_l2"] = float(delta_norm / max(reference_norm, 1e-30))
    result["max_abs"] = float(np.max(np.abs(delta), initial=0))
    row_delta = np.linalg.norm(actual_rows - reference_rows, axis=-1)
    row_reference = np.linalg.norm(reference_rows, axis=-1)
    row_actual = np.linalg.norm(actual_rows, axis=-1)
    result["worst_row_relative_l2"] = float(np.max(row_delta / np.maximum(row_reference, 1e-30), initial=0))
    both_zero = (row_reference == 0) & (row_actual == 0)
    denominator = row_reference * row_actual
    cosine = np.divide((actual_rows * reference_rows).sum(axis=-1), denominator,
                       out=np.zeros_like(denominator), where=denominator > 0)
    cosine[both_zero] = 1.0
    result["mean_row_cosine"] = float(np.clip(cosine, -1, 1).mean()) if cosine.size else 1.0
    return result
