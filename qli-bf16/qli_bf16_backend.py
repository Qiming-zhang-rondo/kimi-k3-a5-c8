#!/usr/bin/env python3
"""Native A5 adapters for a four-way BF16 / MXFP4 indexer comparison.

Model-call source (fixed, not current master):
https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/attention/indexer.py
https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/device/device_op.py
BF16 operator contract:
https://github.com/Ascend/op-plugin/blob/93c866aecaf361f79cf78f97d5da480348784709/docs/zh/custom_APIs/torch_npu/torch_npu-npu_lightning_indexer.md

No model/service import, package installation, DSL or main KV changes. BF16 PA
does not expose selected scores; all four calls use return_value=False. Any
scores computed by the caller are diagnostics, not observed kernel outputs.
"""
import hashlib
import importlib
import importlib.metadata
import inspect
import math
from pathlib import Path
import platform
import sys

import numpy as np


SOURCE_REVISION = "c09d3e20b9c45c80b34ae3781440d90904a44023"
PATH_NAMES = ("bf16_original", "bf16_rotated", "bf16_dequantized", "mxfp4_qli")


def hadamard_matrix(size=128):
    """Sylvester ordering, identical to scipy.linalg.hadamard for this size."""
    if size < 1 or size & (size-1):
        raise ValueError("Hadamard size must be a positive power of two")
    result = np.ones((1, 1), dtype=np.float32)
    while len(result) < size:
        result = np.block([[result, result], [result, -result]])
    return result


def dequantize_mxfp4(packed, scale):
    """Decode low nibble first, E2M1 and one E8M0 power per 32 values.

    Input is raw bytes with shape [..., 64] and [..., 2, 2]. The result is
    [..., 128] float32; nonfinite scales/output are rejected explicitly.
    """
    packed, scale = np.asarray(packed), np.asarray(scale)
    if packed.dtype != np.uint8 or scale.dtype != np.uint8:
        raise ValueError("MXFP4 and E8M0 inputs must be uint8 byte views")
    if packed.ndim < 1 or packed.shape[-1] != 64:
        raise ValueError("MXFP4 packed head dimension must be 64 bytes (128 values)")
    if scale.shape != packed.shape[:-1]+(2, 2):
        raise ValueError(f"E8M0 scale shape {scale.shape} does not match {packed.shape[:-1]+(2, 2)}")
    if np.any(scale == 255):
        raise ValueError("E8M0 scale byte 255 is NaN; cannot build a finite BF16 control")
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    codes = np.stack((packed & 15, packed >> 4), axis=-1).reshape(*packed.shape[:-1], 128)
    factor = np.exp2(scale.astype(np.int16).reshape(*packed.shape[:-1], 4)-127).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        decoded = (lut[codes].astype(np.float64)*np.repeat(factor, 32, axis=-1)).astype(np.float32)
    if not np.all(np.isfinite(decoded)):
        raise ValueError("MXFP4 dequantization produced nonfinite values")
    return decoded


def validate_case(data):
    """Validate bounds before importing Torch or submitting any device work."""
    q, k, w = (np.asarray(data[name]) for name in ("q", "k", "w"))
    if q.ndim != 4 or q.shape[-1] != 128 or q.shape[2] not in (8, 16, 24, 32, 64):
        raise ValueError("q must have shape [B,Q,H,128], H in {8,16,24,32,64}")
    batch, queries, heads, _ = q.shape
    if min(batch, queries) < 1:
        raise ValueError("Batch and query dimensions must be positive")
    if k.ndim != 4 or k.shape[0] != batch or k.shape[2:] != (1, 128) or k.shape[1] < 1:
        raise ValueError("k must have shape [B,K,1,128] matching q batch")
    if w.shape != (batch, queries, heads):
        raise ValueError("w must have shape [B,Q,H] matching q")
    for name, value in (("q", q), ("k", k), ("w", w)):
        if not np.issubdtype(value.dtype, np.floating) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain finite floating-point values")
    q_lengths = np.asarray(data.get("q_lengths", np.full(batch, queries, np.int32)))
    k_lengths = np.asarray(data.get("k_lengths", np.full(batch, k.shape[1], np.int32)))
    for name, lengths, capacity in (("q_lengths", q_lengths, queries), ("k_lengths", k_lengths, k.shape[1])):
        if lengths.shape != (batch,) or not np.issubdtype(lengths.dtype, np.integer):
            raise ValueError(f"{name} must be integer [B]")
        if np.any(lengths < 1) or np.any(lengths > capacity):
            raise ValueError(f"{name} must be in [1,{capacity}]")
    if np.any(q_lengths > k_lengths):
        raise ValueError("This replay format requires q_lengths <= k_lengths (trailing causal queries)")
    if int(q_lengths.sum()) > np.iinfo(np.int32).max:
        raise ValueError("Total query length exceeds INT32 metadata")
    page, topk, mask = (data.get(name, default) for name, default in (("page", 128), ("topk", 2048), ("mask", 3)))
    for name, value in (("page", page), ("topk", topk), ("mask", mask)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if page < 16 or page > 1024 or page % 16:
        raise ValueError("page must be a multiple of 16 in [16,1024]")
    if not (1 <= topk <= 2048 or topk in (3072, 4096, 5120, 6144, 7168, 8192)):
        raise ValueError("topk must be 1..2048 or a multiple of 1024 up to 8192 for the BF16 API")
    if mask not in (0, 3):
        raise ValueError("mask must be 0 or 3")
    return q_lengths.astype(np.int32), k_lengths.astype(np.int32)


def paged_array(logical, page, table):
    """Scatter a logical [B,K,...] CPU tensor into independent physical pages."""
    logical, table = np.asarray(logical), np.asarray(table)
    batch, keys = logical.shape[:2]
    pages = math.ceil(keys/page)
    if table.shape != (batch, pages) or not np.array_equal(np.sort(table.reshape(-1)), np.arange(batch*pages)):
        raise ValueError("Test block table must be a bijection over independent physical pages")
    result = np.zeros((batch*pages, page, *logical.shape[2:]), dtype=logical.dtype)
    for b in range(batch):
        for block in range(pages):
            start, end = block*page, min((block+1)*page, keys)
            result[int(table[b, block]), :end-start] = logical[b, start:end]
    return result


def bf16_call(op, q, k, w, cu, lengths, table, topk, mask):
    return op(query=q, key=k, weights=w, actual_seq_lengths_query=cu[1:],
              actual_seq_lengths_key=lengths, block_table=table, layout_query="TND",
              layout_key="PA_BSND", sparse_count=topk, sparse_mode=mask, return_value=False)


def qli_metadata_call(op, heads, batch, topk, mask, cu, lengths):
    return op(num_heads_q=heads, num_heads_k=1, head_dim=128, topk=topk, quant_mode=5,
              cu_seqlens_q=cu, seqused_k=lengths, batch_size=batch,
              max_seqlen_q=-1, max_seqlen_k=-1, layout_q="TND", layout_k="PA_BBND",
              mask_mode=mask, cmp_ratio=1)


def qli_call(op, q, k, w, qs, ks, cu, lengths, table, metadata, topk, mask):
    return op(q, k, w, qs, ks, topk=topk, quant_mode=5, cu_seqlens_q=cu, seqused_k=lengths,
              block_table=table, metadata=metadata, max_seqlen_q=-1, layout_q="TND",
              layout_k="PA_BBND", mask_mode=mask, cmp_ratio=1, return_value=0)


def file_identity(fn):
    result = {"module": getattr(fn, "__module__", None), "repr": str(fn)}
    try:
        path = inspect.getsourcefile(fn)
        if path:
            result.update(file=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    except (TypeError, OSError):
        pass
    return result


def run_npu_case(data, device="npu:0", repeats=3):
    """Execute all four paths, returning actual indices and explicit scoring inputs.

    q/k are post-projection, post-RoPE tensors BEFORE the quantization-path
    Hadamard. k must include the full relevant historical cache, not only the
    current token. Main attention KV and model hidden states are not used here.
    """
    q_lengths, k_lengths = validate_case(data)
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    import torch
    import torch_npu

    torch.npu.set_device(device)
    fp4_dst = getattr(torch_npu, "float4_e2m1fn_x2", None)
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if fp4_dst is None or e8m0 is None:
        raise RuntimeError("Current Torch/Torch-NPU lacks MXFP4/E8M0 dtype support; no package changes were made")
    if not callable(getattr(torch_npu, "npu_lightning_indexer", None)):
        raise RuntimeError("torch_npu.npu_lightning_indexer is unavailable in this Python environment")
    if not callable(getattr(torch_npu, "npu_dynamic_mx_quant", None)):
        raise RuntimeError("torch_npu.npu_dynamic_mx_quant is unavailable in this Python environment")
    native = importlib.import_module("cann_ops_transformer.ops")
    if not all(callable(getattr(native, name, None)) for name in ("quant_lightning_indexer", "quant_lightning_indexer_metadata")):
        raise RuntimeError("Installed cann_ops_transformer.ops lacks native QLI V2 and its metadata entry; no DSL fallback")

    batch, queries, heads, _ = np.asarray(data["q"]).shape
    keys = np.asarray(data["k"]).shape[1]
    page, topk, mask = (int(data.get(name, default)) for name, default in (("page", 128), ("topk", 2048), ("mask", 3)))
    pages = math.ceil(keys/page)
    table = np.random.default_rng(int(data.get("seed", 0))).permutation(batch*pages).astype(np.int32).reshape(batch, pages)
    cu = np.concatenate((np.zeros(1, np.int32), np.cumsum(q_lengths, dtype=np.int32)))
    row_map = np.asarray([(b, r) for b in range(batch) for r in range(int(q_lengths[b]))], np.int32)

    def to_device(array, dtype=None):
        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device)
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    def host_float(tensor):
        return tensor.detach().float().cpu().numpy().copy()

    def host_bytes(tensor):
        if tensor.element_size() != 1:
            raise RuntimeError(f"Expected one-byte packed dtype, got {tensor.dtype}")
        return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().copy()

    def pack_query(array, dtype):
        packed = np.concatenate([array[b, :q_lengths[b]] for b in range(batch)], axis=0)
        tensor = to_device(packed)
        return tensor.view(dtype) if packed.dtype == np.uint8 else tensor.to(dtype=dtype)

    def pack_key(array, dtype):
        tensor = to_device(paged_array(array, page, table))
        return tensor.view(dtype) if array.dtype == np.uint8 else tensor.to(dtype=dtype)

    environment = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                       torch=torch.__version__, torch_npu=torch_npu.__version__,
                       device=torch.npu.get_device_name(torch.npu.current_device()),
                       source_revision=SOURCE_REVISION,
                       implementations={"bf16": file_identity(torch_npu.npu_lightning_indexer),
                                        "qli": file_identity(native.quant_lightning_indexer),
                                        "metadata": file_identity(native.quant_lightning_indexer_metadata),
                                        "quantizer": file_identity(torch_npu.npu_dynamic_mx_quant)},
                       selected_scores="not returned by paged BF16 API; caller recomputation is diagnostic only",
                       main_kv="not accessed", bf16_weights="BF16", qli_weights="same BF16 values widened to FP32")
    for package in ("torch-npu", "cann-ops-transformer"):
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = "not recorded"
    result = dict(outputs={}, repeats={}, inputs={}, row_map=row_map, block_table=table,
                  q_lengths=q_lengths, k_lengths=k_lengths, quantized={}, environment=environment)
    with torch.inference_mode():
        q = to_device(np.asarray(data["q"], np.float32), torch.bfloat16)
        k = to_device(np.asarray(data["k"], np.float32), torch.bfloat16)
        w = to_device(np.asarray(data["w"], np.float32), torch.bfloat16)
        q_ref, k_ref, w_ref = host_float(q), host_float(k), host_float(w)
        if not all(np.all(np.isfinite(x)) for x in (q_ref, k_ref, w_ref)):
            raise ValueError("Input conversion overflowed BF16; cannot compare nonfinite original inputs")
        result["inputs"].update(q_ref=q_ref, k_ref=k_ref, w_ref=w_ref)
        # Match the model: construct H in BF16, divide by sqrt(D) in BF16,
        # then BF16 matmul. This rounding can itself alter TopK, hence a control.
        hadamard = to_device(hadamard_matrix(128), torch.bfloat16)/(128**0.5)
        q_rot, k_rot = q @ hadamard, k @ hadamard
        result["inputs"].update(q_rot=host_float(q_rot), k_rot=host_float(k_rot))
        q_quant, qs = torch_npu.npu_dynamic_mx_quant(q_rot.reshape(-1, 128), dst_type=fp4_dst)
        k_quant, ks = torch_npu.npu_dynamic_mx_quant(k_rot.reshape(-1, 128), dst_type=fp4_dst)
        if q_quant.numel() != batch*queries*heads*64 or k_quant.numel() != batch*keys*64:
            raise RuntimeError("npu_dynamic_mx_quant did not return two packed FP4 values per byte")
        if qs.numel() != batch*queries*heads*4 or ks.numel() != batch*keys*4:
            raise RuntimeError("npu_dynamic_mx_quant returned unexpected scale count (expected 4 E8M0 scales per head)")
        q_bytes = host_bytes(q_quant).reshape(batch, queries, heads, 64)
        k_bytes = host_bytes(k_quant).reshape(batch, keys, 1, 64)
        q_scale_bytes = host_bytes(qs).reshape(batch, queries, heads, 2, 2)
        k_scale_bytes = host_bytes(ks).reshape(batch, keys, 1, 2, 2)
        result["quantized"].update(q_bytes=q_bytes, k_bytes=k_bytes, q_scale_bytes=q_scale_bytes,
                                   k_scale_bytes=k_scale_bytes)
        # Only the diagnostic branch dequantizes; the QLI call receives the
        # untouched packed bytes and E8M0 scale bytes emitted by the NPU.
        q_deq = to_device(dequantize_mxfp4(q_bytes, q_scale_bytes), torch.bfloat16)
        k_deq = to_device(dequantize_mxfp4(k_bytes, k_scale_bytes), torch.bfloat16)
        result["inputs"].update(q_deq=host_float(q_deq), k_deq=host_float(k_deq))
        if not all(np.all(np.isfinite(x)) for x in result["inputs"].values()):
            raise ValueError("Hadamard or dequantized control contains nonfinite BF16 values")
        cu_t, len_t, table_t = to_device(cu), to_device(k_lengths), to_device(table)
        weight_bf16 = pack_query(w_ref, torch.bfloat16)
        weight_fp32 = weight_bf16.float()
        metadata = qli_metadata_call(native.quant_lightning_indexer_metadata, heads, batch, topk, mask, cu_t, len_t)
        torch.npu.synchronize()
        if (metadata.dtype != torch.int32 or metadata.ndim != 1 or metadata.numel() < 1
                or metadata.device != cu_t.device or not metadata.is_contiguous()):
            raise RuntimeError(f"Native QLI metadata has unexpected shape/dtype: {metadata.shape}, {metadata.dtype}")
        environment["metadata_shape"] = list(metadata.shape)
        environment["quantizer_output_dtypes"] = dict(q=str(q_quant.dtype), k=str(k_quant.dtype),
                                                        q_scale=str(qs.dtype), k_scale=str(ks.dtype))
        # Byte views are used for packing/scatter, then restored to the native
        # quantizer output dtype at the QLI boundary. No numerical dtype cast.
        q_packed = pack_query(q_bytes, q_quant.dtype)
        k_packed = pack_key(k_bytes, k_quant.dtype)
        qs_t = pack_query(q_scale_bytes, e8m0)
        ks_t = pack_key(k_scale_bytes, e8m0)
        environment["qli_input_dtypes"] = dict(q=str(q_packed.dtype), k=str(k_packed.dtype),
                                               q_scale=str(qs_t.dtype), k_scale=str(ks_t.dtype))
        for name, q_name, k_name in (("bf16_original", "q_ref", "k_ref"),
                                    ("bf16_rotated", "q_rot", "k_rot"),
                                    ("bf16_dequantized", "q_deq", "k_deq"),
                                    ("mxfp4_qli", None, None)):
            print(f"  Running {name}: {repeats} repeated index selections ...", flush=True)
            if name != "mxfp4_qli":
                q_path = pack_query(result["inputs"][q_name], torch.bfloat16)
                k_path = pack_key(result["inputs"][k_name], torch.bfloat16)
            samples = []
            for _ in range(repeats):
                if name == "mxfp4_qli":
                    raw = qli_call(native.quant_lightning_indexer, q_packed, k_packed, weight_fp32,
                                   qs_t, ks_t, cu_t, len_t, table_t, metadata, topk, mask)
                else:
                    raw = bf16_call(torch_npu.npu_lightning_indexer, q_path, k_path, weight_bf16,
                                    cu_t, len_t, table_t, topk, mask)
                torch.npu.synchronize()
                if not isinstance(raw, (tuple, list)) or len(raw) != 2:
                    raise RuntimeError(f"{name} did not return the documented pair of tensors")
                indices, values = raw
                if indices.dtype != torch.int32 or tuple(indices.shape) != (len(row_map), 1, topk):
                    raise RuntimeError(f"{name} index contract mismatch: {indices.shape}, {indices.dtype}")
                if values.numel() != 0:
                    raise RuntimeError(f"{name} unexpectedly returned scores with return_value=False")
                samples.append(indices.detach().cpu().numpy().copy())
            result["outputs"][name] = samples[0]
            result["repeats"][name] = samples
    return result
