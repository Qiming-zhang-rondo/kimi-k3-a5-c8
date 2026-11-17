#!/usr/bin/env python3
"""Explicit replay export from the ORIGINAL BF16 main-MLA attention inputs.

Call ``save_paged_mla_case`` immediately before a BF16 sparse-attention call,
with the same query, latent KV, RoPE, block table and sparse indices. The main
latent cache must contain the full original BF16 history for the selected
requests. A C4 cache widened/dequantized to BF16 cannot recover that history
and must not be passed here. Capture from the BF16 baseline, or retain a true
BF16 shadow before quantization in the caller.

The helper installs no hooks and changes no service/model state. Export is a
one-time diagnostic: transferring the original cache to the CPU synchronizes
the device and is unsuitable for timing. Invoke outside ACLGraph capture.

Layouts: q_nope [T,H,D], q_rope [T,H,R], kv_cache [P,page,1,D],
key_rope [P,page,1,R], sparse_indices [T,1,S], block_table [B,max_blocks].
actual_seq_lengths_query contains cumulative endpoints WITHOUT a leading 0;
actual_seq_lengths_kv contains per-request visible KV lengths. Queries are
trailing causal queries. Extra padded query rows are ignored. No attention
sink is represented by this replay format.
"""
from pathlib import Path

import numpy as np

from kv_window_common import INPUT_STAGE, save_case, validate_case


def _host_bf16(tensor, name):
    # Checking before float conversion prevents quantized payloads from being
    # accidentally interpreted as original activation values. Do not import
    # Torch here: capture uses the caller's already initialized environment.
    if str(getattr(tensor, "dtype", None)) != "torch.bfloat16":
        raise ValueError(f"{name} must be an original torch.bfloat16 tensor before KV quantization")
    return tensor.detach().float().cpu().numpy().copy()


def _host_integer(value, name):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy().copy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    return array


def paged_mla_case_arrays(*, q_nope, q_rope, kv_cache, key_rope,
                          sparse_indices, block_table, actual_seq_lengths_query,
                          actual_seq_lengths_kv, scale, label="BF16_main_MLA_capture",
                          attention_sink=None):
    """Reconstruct logical replay arrays from CPU views of a paged BF16 call.

    Floating arrays must be float32 storage of exactly BF16-representable
    original values. This internal conversion helper does not prove capture
    provenance; callers must supply original model activations.
    """
    if attention_sink is not None:
        raise ValueError("Attention sinks are not supported by this replay format")
    q = np.asarray(q_nope)
    qr = np.asarray(q_rope)
    kv = np.asarray(kv_cache)
    kr = None if key_rope is None else np.asarray(key_rope)
    indices = _host_integer(sparse_indices, "sparse_indices")
    table = _host_integer(block_table, "block_table")
    cu = _host_integer(actual_seq_lengths_query, "actual_seq_lengths_query")
    k_lengths = _host_integer(actual_seq_lengths_kv, "actual_seq_lengths_kv")
    if cu.ndim != 1 or cu.size == 0 or np.any(cu <= 0) or np.any(cu > np.iinfo(np.int32).max):
        raise ValueError("actual_seq_lengths_query must be positive cumulative endpoints without a leading 0")
    cu = cu.astype(np.int64)
    q_lengths = np.diff(np.concatenate((np.zeros(1, np.int64), cu)))
    if np.any(q_lengths <= 0):
        raise ValueError("Empty requests and non-increasing cumulative query lengths are unsupported")
    batch = len(cu)
    if k_lengths.shape != (batch,) or np.any(k_lengths <= 0) or np.any(k_lengths > np.iinfo(np.int32).max):
        raise ValueError("actual_seq_lengths_kv must contain one positive int32-range length per request")
    k_lengths = k_lengths.astype(np.int64)
    if np.any(q_lengths > k_lengths):
        raise ValueError("Query lengths must not exceed visible KV lengths")
    if q.ndim != 3 or q.shape[0] < cu[-1] or min(q.shape[1:]) < 1:
        raise ValueError("q_nope must be [T,H,D] with enough valid query rows")
    heads, dim = q.shape[1:]
    if qr.ndim != 3 or qr.shape[:2] != q.shape[:2] or qr.shape[-1] not in (0, 64):
        raise ValueError("q_rope must be [T,H,R] matching q_nope, R in {0,64}")
    rope_dim = qr.shape[-1]
    if kv.ndim != 4 or kv.shape[2:] != (1, dim) or min(kv.shape[:2]) < 1:
        raise ValueError("kv_cache must contain original latent KV with shape [pages,page,1,D]")
    page = kv.shape[1]
    if kr is None:
        if rope_dim:
            raise ValueError("key_rope is required when q_rope has a nonzero dimension")
        kr = np.zeros((*kv.shape[:3], 0), dtype=np.float32)
    if kr.shape != (*kv.shape[:3], rope_dim):
        raise ValueError("key_rope shape must match the paged latent cache and RoPE dimension")
    if table.ndim != 2 or table.shape[0] != batch:
        raise ValueError("block_table must be [B,max_blocks]")
    if indices.ndim != 3 or indices.shape[0] < cu[-1] or indices.shape[1] != 1 or indices.shape[2] < 1:
        raise ValueError("sparse_indices must be [T,1,S] with enough valid query rows")
    max_q, max_k = int(q_lengths.max()), int(k_lengths.max())
    result_q = np.zeros((batch, max_q, heads, dim), dtype=q.dtype)
    result_qr = np.zeros((batch, max_q, heads, rope_dim), dtype=qr.dtype)
    result_kv = np.zeros((batch, max_k, 1, dim), dtype=kv.dtype)
    result_kr = np.zeros((batch, max_k, 1, rope_dim), dtype=kr.dtype)
    result_ids = np.full((batch, max_q, 1, indices.shape[-1]), -1, dtype=indices.dtype)
    start = 0
    for b in range(batch):
        qlen, klen = int(q_lengths[b]), int(k_lengths[b])
        end = int(cu[b])
        result_q[b, :qlen] = q[start:end]
        result_qr[b, :qlen] = qr[start:end]
        result_ids[b, :qlen] = indices[start:end]
        start = end
        required_pages = (klen + page - 1) // page
        if required_pages > table.shape[1]:
            raise ValueError(f"Request {b}: block table does not cover visible KV length")
        pages = table[b, :required_pages]
        if np.any(pages < 0) or np.any(pages >= kv.shape[0]):
            raise ValueError(f"Request {b}: block table references an invalid physical page")
        # Shared physical pages are legitimate for prefix caches. Reading them
        # into independent logical snapshots does not mutate the source cache.
        for block, physical in enumerate(pages):
            lo, hi = block * page, min((block + 1) * page, klen)
            result_kv[b, lo:hi] = kv[int(physical), :hi-lo]
            result_kr[b, lo:hi] = kr[int(physical), :hi-lo]
    return validate_case(dict(q_nope=result_q, q_rope=result_qr, kv=result_kv, k_rope=result_kr,
                              indices=result_ids, q_lengths=q_lengths, k_lengths=k_lengths,
                              scale=scale, page=page, source_kind="model_capture",
                              selection_source="model_supplied_sparse_indices",
                              input_stage=INPUT_STAGE, label=label))


def save_paged_mla_case(path, *, q_nope, q_rope, kv_cache, key_rope,
                        sparse_indices, block_table, actual_seq_lengths_query,
                        actual_seq_lengths_kv, scale, label="BF16_main_MLA_capture",
                        attention_sink=None):
    """Export one original BF16 main-MLA call. Existing files are never replaced.

    Sparse indices must already be logical indices in each request, before any
    DCP-local remap. For DCP, export only after gathering the original full KV
    history and corresponding full block table; local-shard caches cannot be
    silently interpreted as complete histories.
    """
    if attention_sink is not None:
        raise ValueError("Attention sinks are not supported by this replay format")
    if Path(path).exists():
        raise FileExistsError(path)
    # Check all source dtypes before any CPU copy, including optional RoPE.
    for name, tensor in (("q_nope", q_nope), ("q_rope", q_rope), ("kv_cache", kv_cache)):
        if str(getattr(tensor, "dtype", None)) != "torch.bfloat16":
            raise ValueError(f"{name} must be an original torch.bfloat16 tensor before KV quantization")
    if key_rope is not None and str(getattr(key_rope, "dtype", None)) != "torch.bfloat16":
        raise ValueError("key_rope must be an original torch.bfloat16 tensor")
    table = _host_integer(block_table, "block_table")
    k_lengths = _host_integer(actual_seq_lengths_kv, "actual_seq_lengths_kv")
    cu = _host_integer(actual_seq_lengths_query, "actual_seq_lengths_query")
    kv_shape = tuple(getattr(kv_cache, "shape", ()))
    if len(kv_shape) != 4 or min(kv_shape[:2]) < 1:
        raise ValueError("kv_cache must be a nonempty [pages,page,1,D] tensor")
    if table.ndim != 2 or k_lengths.shape != (table.shape[0],) or np.any(k_lengths <= 0):
        raise ValueError("block_table and visible KV lengths must agree on a nonempty batch")
    if cu.shape != (table.shape[0],) or np.any(cu <= 0) or np.any(np.diff(cu.astype(np.int64)) <= 0):
        raise ValueError("Query cumulative endpoints must be positive and increasing without a leading 0")
    page = kv_shape[1]
    used = []
    for b, klen in enumerate(k_lengths):
        required = (int(klen)+page-1)//page
        if required > table.shape[1]:
            raise ValueError(f"Request {b}: block table does not cover visible KV length")
        pages = table[b, :required]
        if np.any(pages < 0) or np.any(pages >= kv_shape[0]):
            raise ValueError(f"Request {b}: invalid physical page")
        used.extend(int(p) for p in pages)
    used = sorted(set(used))
    if not used:
        raise ValueError("Capture has no valid cache pages")
    mapping = {physical: local for local, physical in enumerate(used)}
    local_table = np.full(table.shape, -1, np.int32)
    for b, klen in enumerate(k_lengths):
        required = (int(klen)+page-1)//page
        local_table[b, :required] = [mapping[int(p)] for p in table[b, :required]]
    # Indexing one physical page is a tensor view. Only referenced pages are
    # copied; never materialize the model's whole allocated KV pool on CPU.
    host_kv = np.stack([_host_bf16(kv_cache[p], "kv_cache page") for p in used])
    host_kr = None if key_rope is None else np.stack([_host_bf16(key_rope[p], "key_rope page") for p in used])
    valid_rows = int(cu[-1])
    arrays = paged_mla_case_arrays(
        q_nope=_host_bf16(q_nope[:valid_rows], "q_nope"), q_rope=_host_bf16(q_rope[:valid_rows], "q_rope"),
        kv_cache=host_kv, key_rope=host_kr,
        sparse_indices=_host_integer(sparse_indices, "sparse_indices"),
        block_table=local_table, actual_seq_lengths_query=cu, actual_seq_lengths_kv=k_lengths,
        scale=scale, label=label, attention_sink=attention_sink)
    return save_case(path, arrays)
