"""CPU index contracts and stable reduction for the KV-split prototype.

These routines validate the partial-state ABI and its mathematics. They do not
emulate native BF16 probability rounding or establish NPU correctness/speed.
The selected TopK columns are split, never sorted or repartitioned by token ID.
"""
from dataclasses import dataclass

import numpy as np


SPLITS = (1, 2, 4)
TOPK = 2048


def _split_count(value):
    if type(value) is not int or value not in SPLITS:
        raise ValueError("KV split count must be 1, 2, or 4")
    return value


def split_topk_columns(indices, splits, *, keys, positions=None):
    """Return contiguous column slices of the unchanged [Q,1,2048] TopK.

    A suffix of -1 padding is allowed, including wholly empty partitions or
    queries. Device support for those shapes must be checked separately.
    """
    splits = _split_count(splits)
    ids = np.asarray(indices)
    if ids.ndim != 3 or ids.shape[0] < 1 or ids.shape[1:] != (1, TOPK):
        raise ValueError("Require original TopK shape [Q,1,2048]")
    if ids.dtype.kind not in "iu" or ids.dtype == np.dtype(bool):
        raise ValueError("TopK must contain integer IDs")
    if type(keys) is not int or not 1 <= keys <= np.iinfo(np.int32).max:
        raise ValueError("Require logical KV length 1..INT32_MAX")
    if np.any(ids < -1) or np.any(ids >= keys):
        raise ValueError("TopK contains an out-of-range ID or invalid sentinel")
    if positions is not None:
        pos = np.asarray(positions)
        if pos.shape != (ids.shape[0],) or pos.dtype.kind not in "iu":
            raise ValueError("Positions must be an integer [Q] array")
        if np.any(pos < 0) or np.any(pos >= keys):
            raise ValueError("Query position lies outside the logical sequence")
        if np.any((ids >= 0) & (ids > pos[:, None, None])):
            raise ValueError("Future TopK IDs are rejected, never silently masked")
    for row in ids[:, 0]:
        missing = row == -1
        if missing.any() and np.any(row[np.argmax(missing):] >= 0):
            raise ValueError("TopK padding must be a trailing suffix")
        valid = row[~missing]
        if len(np.unique(valid)) != len(valid):
            raise ValueError("Duplicate TopK IDs would duplicate attention mass")
    width = TOPK // splits
    return tuple(np.ascontiguousarray(ids[:, :, start:start+width])
                 for start in range(0, TOPK, width))


@dataclass(frozen=True)
class PartialState:
    """Normalized O plus softmax m/s: exp(score-m) sums to s.

    Empty rows use O=0, m=-inf, s=0. All states retain a trailing singleton
    statistics axis, while O's final axis is the value dimension.
    """
    output: np.ndarray
    maximum: np.ndarray
    total: np.ndarray


def attention_state(scores, values, *, valid=None, dtype=np.float32):
    """Stable independent mathematical attention; optionally round to FP32.

    scores [...,K], values [...,K,D] and valid [...,K] may broadcast across
    head/query axes. Invalid slots contribute neither their score nor V.
    """
    if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Partial states must be FP32 or FP64")
    score = np.asarray(scores, dtype=np.float64)
    value = np.asarray(values, dtype=np.float64)
    if score.ndim < 1 or value.ndim < 2 or score.shape[-1] != value.shape[-2]:
        raise ValueError("Expected scores [...,K] and values [...,K,D]")
    if score.shape[-1] < 1 or value.shape[-1] < 1:
        raise ValueError("Score capacity and value dimension must be positive")
    prefix = np.broadcast_shapes(score.shape[:-1], value.shape[:-2])
    count, dim = score.shape[-1], value.shape[-1]
    score = np.broadcast_to(score, prefix+(count,))
    value = np.broadcast_to(value, prefix+(count,dim))
    mask = np.ones(score.shape, dtype=bool) if valid is None else np.asarray(valid)
    if mask.dtype != np.dtype(bool):
        raise ValueError("Valid mask must be boolean")
    mask = np.broadcast_to(mask, score.shape)
    if not np.isfinite(score[mask]).all() or not np.isfinite(value[mask]).all():
        raise ValueError("Active attention inputs must be finite")
    present = mask.any(axis=-1, keepdims=True)
    maximum = np.where(mask, score, -np.inf).max(axis=-1, keepdims=True)
    safe_maximum = np.where(present, maximum, 0.)
    probability = np.exp(np.where(mask, score, -np.inf)-safe_maximum)
    total = probability.sum(axis=-1, keepdims=True)
    numerator = (probability[...,None]*np.where(mask[...,None],value,0.)).sum(axis=-2)
    output = np.divide(numerator,total,out=np.zeros(prefix+(dim,),dtype=np.float64),where=total>0)
    result = PartialState(output.astype(dtype),maximum.astype(dtype),total.astype(dtype))
    _validate_state(result)
    return result


def _validate_state(state):
    if not isinstance(state, PartialState):
        raise ValueError("Expected a PartialState")
    o,m,s = (np.asarray(value) for value in (state.output,state.maximum,state.total))
    if o.ndim < 1 or o.shape[-1] < 1 or m.shape != o.shape[:-1]+(1,) or s.shape != m.shape:
        raise ValueError("Partial O/m/s shape mismatch")
    if any(value.dtype not in (np.dtype(np.float32),np.dtype(np.float64)) for value in (o,m,s)):
        raise ValueError("Partial O/m/s must be FP32 or FP64")
    if not np.isfinite(o).all() or not np.isfinite(s).all() or np.any(s<0):
        raise ValueError("Partial output/total must be finite, with total >= 0")
    present = s > 0
    if np.any(~np.isfinite(m[present])) or np.any(m[~present] != -np.inf):
        raise ValueError("Active max must be finite; empty max must be -inf")
    if np.any(np.where(present,0.,o) != 0):
        raise ValueError("Empty partial rows must have zero output")
    return o,m,s


def merge_partial_states(states, *, dtype=np.float32):
    """Reduce normalized partial outputs using stable m/s weights.

    m=max(m_i), w_i=s_i*exp(m_i-m), O=sum(w_i*O_i)/sum(w_i).
    All-empty rows remain (0,-inf,0), with no exp(-inf-(-inf)) operation.
    """
    states = tuple(states)
    _split_count(len(states))
    if np.dtype(dtype) not in (np.dtype(np.float32),np.dtype(np.float64)):
        raise ValueError("Reduction must use FP32 or FP64")
    checked = [_validate_state(state) for state in states]
    shape = checked[0][0].shape
    if any(item[0].shape != shape for item in checked):
        raise ValueError("All partial states must have identical O/m/s shapes")
    outputs = np.stack([item[0] for item in checked]).astype(dtype)
    maxima = np.stack([item[1] for item in checked]).astype(dtype)
    totals = np.stack([item[2] for item in checked]).astype(dtype)
    present = totals > 0
    maximum = np.where(present,maxima,-np.inf).max(axis=0)
    safe_maximum = np.where(present.any(axis=0),maximum,0.)
    factors = np.exp(np.where(present,maxima-safe_maximum,-np.inf))
    weights = totals*factors
    total = weights.sum(axis=0,dtype=dtype)
    numerator = (outputs*weights).sum(axis=0,dtype=dtype)
    output = np.divide(numerator,total,out=np.zeros(shape,dtype=dtype),where=total>0)
    result = PartialState(output,maximum,total)
    _validate_state(result)
    return result


def split_attention_reference(scores, values, splits, *, valid=None):
    """Produce FP32 partial states and their FP32 reduction for CPU tests."""
    splits = _split_count(splits)
    scores = np.asarray(scores)
    values = np.asarray(values)
    if scores.ndim < 1 or values.ndim < 2 or scores.shape[-1] != TOPK or values.shape[-2] != TOPK:
        raise ValueError("KV-split prototype fixes the original capacity at TopK2048")
    width = TOPK//splits
    mask = None if valid is None else np.asarray(valid)
    partials = tuple(attention_state(scores[...,start:start+width],
        values[...,start:start+width,:],valid=None if mask is None else mask[...,start:start+width])
        for start in range(0,TOPK,width))
    return merge_partial_states(partials),partials
