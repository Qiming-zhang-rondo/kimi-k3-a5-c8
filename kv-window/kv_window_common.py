#!/usr/bin/env python3
"""CPU reference and replay format for a main-MLA KV precision window.

This module never computes or changes QLI TopK. ``indices`` is an explicit,
fixed sparse selection. Reference scores/softmax are evaluated in float64;
they are diagnostic mathematical references, not bit-exact kernel goldens.
"""
from pathlib import Path
import math

import numpy as np


FORMAT_VERSION = 1
INPUT_STAGE = "main_mla_post_norm_pre_quant"
SOURCE_KINDS = ("synthetic", "synthetic_replay", "model_capture")
ARRAY_KEYS = ("q_nope", "q_rope", "kv", "k_rope", "indices", "q_lengths", "k_lengths")


def bf16_round(value):
    """Round finite float32 values to BF16, nearest with ties to even."""
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("BF16 rounding requires finite values")
    bits = array.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)
    if not np.all(np.isfinite(rounded)):
        raise ValueError("BF16 conversion overflow")
    return rounded.copy()


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _bf16_array(value, name):
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise ValueError(f"{name} must store original BF16 values as float32")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    if np.any(array.view(np.uint32) & np.uint32(0xFFFF)):
        raise ValueError(f"{name} is not exactly BF16-representable; capture before C4 quantization")
    return np.ascontiguousarray(array)


def validate_case(data):
    """Validate the complete input before any device call, returning a copy.

    Query rows are trailing causal queries: query r in batch b corresponds to
    logical position ``k_lengths[b] - q_lengths[b] + r``. All valid sparse
    rows must have one or more unique visible indices and trailing -1 padding.
    """
    if not isinstance(data, dict):
        raise ValueError("Case must be a dictionary")
    missing = [name for name in ARRAY_KEYS if name not in data]
    if missing:
        raise ValueError(f"Missing replay arrays: {', '.join(missing)}")
    result = dict(data)
    for name in ("q_nope", "q_rope", "kv", "k_rope"):
        result[name] = _bf16_array(data[name], name)
    q, qr, kv, kr = (result[name] for name in ("q_nope", "q_rope", "kv", "k_rope"))
    if q.ndim != 4 or min(q.shape) < 1 or q.shape[-1] % 64:
        raise ValueError("q_nope must be [B,Q,H,D] with positive dimensions and D a multiple of 64")
    batch, queries, heads, dim = q.shape
    if qr.ndim != 4 or qr.shape[:3] != (batch, queries, heads) or qr.shape[-1] not in (0, 64):
        raise ValueError("q_rope must be [B,Q,H,R], R in {0,64}")
    if kv.ndim != 4 or kv.shape[0] != batch or kv.shape[1] < 1 or kv.shape[2:] != (1, dim):
        raise ValueError("kv must be [B,K,1,D] matching q_nope")
    if kr.shape != kv.shape[:3] + (qr.shape[-1],):
        raise ValueError("k_rope must be [B,K,1,R] matching kv and q_rope")
    for name, capacity in (("q_lengths", queries), ("k_lengths", kv.shape[1])):
        lengths = np.asarray(data[name])
        if lengths.shape != (batch,) or not np.issubdtype(lengths.dtype, np.integer):
            raise ValueError(f"{name} must be an integer [B] array")
        if np.any(lengths < 1) or np.any(lengths > capacity):
            raise ValueError(f"{name} must lie in [1,{capacity}]")
        result[name] = lengths.astype(np.int32)
    if np.any(result["q_lengths"] > result["k_lengths"]):
        raise ValueError("q_lengths must not exceed k_lengths")
    if int(result["q_lengths"].astype(np.int64).sum()) > np.iinfo(np.int32).max:
        raise ValueError("Total query count exceeds int32 metadata")
    indices = np.asarray(data["indices"])
    if indices.ndim != 4 or indices.shape[:3] != (batch, queries, 1) or indices.shape[-1] < 1:
        raise ValueError("indices must be [B,Q,1,S] with S >= 1")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must use an integer dtype")
    if np.any(indices < -1) or np.any(indices > np.iinfo(np.int32).max):
        raise ValueError("indices contain unsupported negative values or exceed int32")
    indices = indices.astype(np.int32)
    for b in range(batch):
        qlen, klen = int(result["q_lengths"][b]), int(result["k_lengths"][b])
        for r in range(queries):
            row = indices[b, r, 0]
            valid = row[row >= 0]
            if r >= qlen:
                if valid.size:
                    raise ValueError(f"batch {b} query {r}: padded query row must contain only -1")
                continue
            if not valid.size:
                raise ValueError(f"batch {b} query {r}: empty sparse selection")
            if np.any(row[:valid.size] < 0) or np.any(row[valid.size:] != -1):
                raise ValueError(f"batch {b} query {r}: -1 padding must be trailing")
            if np.unique(valid).size != valid.size:
                raise ValueError(f"batch {b} query {r}: duplicate sparse index")
            position = klen - qlen + r
            if np.any(valid > position):
                raise ValueError(f"batch {b} query {r}: out-of-range or future index for position {position}")
    result["indices"] = indices
    page = _integer(data.get("page", 128), "page", 16)
    if page > 1024 or page % 16:
        raise ValueError("page must be a multiple of 16 in [16,1024]")
    result["page"] = page
    scale = data.get("scale", 1.0 / math.sqrt(dim + qr.shape[-1]))
    if isinstance(scale, (bool, np.bool_)) or not np.isscalar(scale):
        raise ValueError("scale must be a finite positive scalar")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive scalar")
    result["scale"] = scale
    result["seed"] = _integer(data.get("seed", 20261117), "seed")
    result["format_version"] = _integer(data.get("format_version", FORMAT_VERSION), "format_version", 1)
    if result["format_version"] != FORMAT_VERSION:
        raise ValueError(f"Unsupported format_version {result['format_version']}")
    result["input_stage"] = str(data.get("input_stage", INPUT_STAGE))
    if result["input_stage"] != INPUT_STAGE:
        raise ValueError(f"input_stage must be {INPUT_STAGE}")
    result["source_kind"] = str(data.get("source_kind", "synthetic"))
    if result["source_kind"] not in SOURCE_KINDS:
        raise ValueError(f"source_kind must be one of {SOURCE_KINDS}")
    result["label"] = str(data.get("label", result["source_kind"]))
    result["selection_source"] = str(data.get("selection_source", "provided_fixed_indices"))
    return result


def row_map(data):
    return np.asarray([(b, r) for b, length in enumerate(data["q_lengths"]) for r in range(int(length))], dtype=np.int32)


def flatten_queries(data, key):
    return np.ascontiguousarray(np.concatenate([data[key][b, :int(n)] for b, n in enumerate(data["q_lengths"])], axis=0))


def query_positions(data):
    return np.asarray([int(data["k_lengths"][b]) - int(data["q_lengths"][b]) + r
                       for b, r in row_map(data)], dtype=np.int64)


def paged_array(logical, page, table):
    """Scatter a logical [B,K,...] array through a physical-page permutation."""
    logical, table = np.asarray(logical), np.asarray(table)
    page = _integer(page, "page", 1)
    if logical.ndim < 2:
        raise ValueError("logical must have shape [B,K,...]")
    batch, keys = logical.shape[:2]
    pages = math.ceil(keys / page)
    if (table.shape != (batch, pages) or not np.issubdtype(table.dtype, np.integer)
            or not np.array_equal(np.sort(table.reshape(-1)), np.arange(batch * pages))):
        raise ValueError("table must be a bijection over independent physical pages")
    output = np.zeros((batch * pages, page, *logical.shape[2:]), dtype=logical.dtype)
    for b in range(batch):
        for block in range(pages):
            start, end = block * page, min((block + 1) * page, keys)
            output[int(table[b, block]), :end - start] = logical[b, start:end]
    return output


def dequantize_mxfp4(packed, scale, dim=None):
    """Decode low-nibble-first E2M1 and one raw E8M0 byte per 32 values."""
    packed, scale = np.asarray(packed), np.asarray(scale)
    if packed.dtype != np.uint8 or scale.dtype != np.uint8 or packed.ndim < 1:
        raise ValueError("packed and scale must be raw uint8 arrays")
    inferred = packed.shape[-1] * 2
    dim = inferred if dim is None else _integer(dim, "dim", 64)
    if dim != inferred or dim % 64:
        raise ValueError("MXFP4 dimension must match packed bytes and be a multiple of 64")
    prefix = packed.shape[:-1]
    if scale.shape != prefix + (dim // 32,) and scale.shape != prefix + (dim // 64, 2):
        raise ValueError("E8M0 scale shape must match packed prefix with D/32 bytes")
    if np.any(scale == 255):
        raise ValueError("E8M0 byte 255 encodes NaN")
    lut = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)
    codes = np.stack((packed & 15, packed >> 4), axis=-1).reshape(*prefix, dim)
    factors = np.exp2(scale.astype(np.int16).reshape(*prefix, dim // 32) - 127)
    with np.errstate(over="ignore", invalid="ignore"):
        result = (lut[codes] * np.repeat(factors, 32, axis=-1)).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("MXFP4 dequantization produced nonfinite values")
    return result


def attention_reference(data, kv_c4=None, window=0, hot_kv=None):
    """Float64 sparse MLA reference on original or per-query mixed KV.

    If kv_c4 is supplied, entries within [position-window+1, position] use
    original KV (or hot_kv, if provided) and all other entries use kv_c4.
    Without kv_c4, old entries use the original KV. RoPE stays original BF16.
    Returns [sum(q_lengths),H,D]. No attention sink is included in this format.
    """
    data = validate_case(data)
    window = _integer(window, "window")
    for name, value in (("kv_c4", kv_c4), ("hot_kv", hot_kv)):
        if value is not None:
            value = np.asarray(value)
            if value.shape != data["kv"].shape or not np.issubdtype(value.dtype, np.floating) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite floating point with the original kv shape")
            if name == "kv_c4":
                kv_c4 = value
            else:
                hot_kv = value
    output = []
    for b, r in row_map(data):
        ids = data["indices"][b, r, 0]
        ids = ids[ids >= 0]
        original = data["kv"][b, ids, 0].astype(np.float64)
        values = original
        if kv_c4 is not None or hot_kv is not None:
            values = original.copy() if kv_c4 is None else kv_c4[b, ids, 0].astype(np.float64)
            position = int(data["k_lengths"][b]) - int(data["q_lengths"][b]) + int(r)
            recent = ids >= position - window + 1
            hot = original if hot_kv is None else hot_kv[b, ids, 0].astype(np.float64)
            values[recent] = hot[recent]
        scores = data["q_nope"][b, r].astype(np.float64) @ values.T
        if data["q_rope"].shape[-1]:
            scores += data["q_rope"][b, r].astype(np.float64) @ data["k_rope"][b, ids, 0].astype(np.float64).T
        scores *= data["scale"]
        scores -= np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        output.append(probabilities @ values)
    return np.stack(output, axis=0)


def window_selection_stats(data, window):
    """Count selected recent entries; window does not add new selected tokens."""
    data = validate_case(data)
    window = _integer(window, "window")
    counts, selected_counts = [], []
    for (b, r), position in zip(row_map(data), query_positions(data)):
        ids = data["indices"][b, r, 0]
        counts.append(int(np.sum((ids >= 0) & (ids >= position - window + 1))))
        selected_counts.append(int(np.sum(ids >= 0)))
    return {"window": window, "recent_selected_per_row": counts,
            "selected_per_row": selected_counts,
            "recent_fraction_of_selected_per_row": [hot/total for hot,total in zip(counts,selected_counts)],
            "recent_selected_mean": float(np.mean(counts)), "recent_selected_min": min(counts), "recent_selected_max": max(counts)}


def compare_outputs(actual, reference):
    """Magnitude/direction diagnostics, with no arbitrary accuracy PASS gate."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or actual.ndim < 2:
        raise ValueError("Output/reference shapes must match and include a row dimension")
    if not np.all(np.isfinite(reference)):
        raise ValueError("Reference output contains nonfinite values")
    if not np.all(np.isfinite(actual)):
        return {"finite": False, "nonfinite_count": int(np.sum(~np.isfinite(actual))), "rows": []}
    actual, reference = actual.astype(np.float64), reference.astype(np.float64)
    rows = []
    for index, (a, r) in enumerate(zip(actual, reference)):
        a, r = a.reshape(-1), r.reshape(-1)
        an, rn = float(np.linalg.norm(a)), float(np.linalg.norm(r))
        cosine = float(np.dot(a, r) / (an * rn)) if an > 0 and rn > 0 else (1.0 if an == rn else 0.0)
        rows.append({"row": index, "relative_l2": float(np.linalg.norm(a-r) / max(rn, 1e-30)),
                     "cosine": float(np.clip(cosine, -1, 1)), "max_abs": float(np.max(np.abs(a-r)))})
    rn = float(np.linalg.norm(reference))
    return {"finite": True, "nonfinite_count": 0, "reference_norm_floor": 1e-30,
            "relative_l2": float(np.linalg.norm(actual-reference) / max(rn, 1e-30)),
            "max_abs": float(np.max(np.abs(actual-reference))),
            "mean_row_relative_l2": float(np.mean([r["relative_l2"] for r in rows])),
            "max_row_relative_l2": max(r["relative_l2"] for r in rows),
            "mean_row_cosine": float(np.mean([r["cosine"] for r in rows])),
            "min_row_cosine": min(r["cosine"] for r in rows), "rows": rows}


def synthetic_case(batch=1, queries=1, keys=8192, heads=64, dim=512, rope_dim=64,
                   topk=2048, page=128, seed=20261117):
    """Generate explicit synthetic sparse selections (not QLI-generated TopK)."""
    for name, value in (("batch", batch), ("queries", queries), ("keys", keys), ("heads", heads), ("dim", dim), ("topk", topk)):
        _integer(value, name, 1)
    if queries > keys or dim % 64 or rope_dim not in (0, 64):
        raise ValueError("Require queries <= keys, D multiple of 64 and R in {0,64}")
    rng = np.random.default_rng(seed)
    # Mildly heavy-tailed channels expose quantization loss without claiming
    # to reproduce an actual model activation distribution.
    channel_scale = rng.lognormal(0, .6, size=(dim,)).astype(np.float32)
    kv = bf16_round(rng.standard_normal((batch, keys, 1, dim), dtype=np.float32) * channel_scale)
    q = bf16_round(rng.standard_normal((batch, queries, heads, dim), dtype=np.float32) * .4)
    qr = bf16_round(rng.standard_normal((batch, queries, heads, rope_dim), dtype=np.float32) * .4)
    kr = bf16_round(rng.standard_normal((batch, keys, 1, rope_dim), dtype=np.float32))
    indices = np.full((batch, queries, 1, topk), -1, dtype=np.int32)
    for b in range(batch):
        for r in range(queries):
            visible = keys - queries + r + 1
            count = min(topk, visible)
            recent_count = min(count, 256)
            recent = np.arange(visible-recent_count, visible, dtype=np.int32)
            history = rng.choice(visible-recent_count, size=count-recent_count, replace=False).astype(np.int32)
            indices[b, r, 0, :count] = np.sort(np.concatenate((history, recent)))
    return validate_case(dict(q_nope=q, q_rope=qr, kv=kv, k_rope=kr, indices=indices,
                              q_lengths=np.full(batch, queries, np.int32), k_lengths=np.full(batch, keys, np.int32),
                              scale=1 / math.sqrt(dim+rope_dim), page=page, seed=seed,
                              source_kind="synthetic", selection_source="synthetic_random_plus_recent_256",
                              label=f"synthetic_B{batch}_Q{queries}_K{keys}_H{heads}_D{dim}_R{rope_dim}_S{topk}"))


def save_case(path, data):
    data = validate_case(data)
    payload = {key: data[key] for key in ARRAY_KEYS}
    for key in ("format_version", "input_stage", "source_kind", "label", "selection_source", "scale", "page", "seed"):
        payload[key] = np.asarray(data[key])
    path = Path(path)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    return path


def load_case(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        required = set(ARRAY_KEYS) | {"format_version", "input_stage", "source_kind", "scale", "page"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Replay missing fields: {sorted(missing)}")
        data = {key: archive[key].copy() for key in ARRAY_KEYS}
        for key in ("format_version", "input_stage", "source_kind", "label", "selection_source", "scale", "page", "seed"):
            if key in archive:
                value = archive[key]
                if value.shape != ():
                    raise ValueError(f"Replay metadata {key} must be scalar")
                data[key] = value.item()
    if data["source_kind"] == "synthetic":
        data["source_kind"] = "synthetic_replay"
    return validate_case(data)
