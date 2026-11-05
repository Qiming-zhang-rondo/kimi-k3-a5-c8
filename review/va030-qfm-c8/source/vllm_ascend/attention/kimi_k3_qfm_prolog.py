# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental adapter for pinned CANN DSL Prolog, NOT MlaPrologV4.

Upstream documents CANN 9.2.1 / DSL 0.8.0. CANN 9.2.0 needs an explicit
isolated compile/device probe before this path is enabled in a model server.
Only the current token batch is staged for padded pages; no BF16 history.
"""

import hashlib
import importlib
import importlib.util
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.attention.kimi_k3_qfm import QFM_BLOCK, QFM_WIDTH, scatter_fp8_cache, write_cache

PROLOG_MODULE = "ops.quant_mla_prolog"
PROLOG_SHA256 = "ab0876c2fc09b7cc2b0541063dc7bab16414b2046ea9030bc1fecce4020be1bc"
PROLOG_PA_BSND = 0
PROLOG_PA_NZ = 1


def _validate_staging_indices(indices, tokens, device):
    """Validate caller-owned immutable 0..T indices without reading device data."""
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.int64
        or indices.ndim != 1
        or indices.shape != (tokens,)
        or not indices.is_contiguous()
        or indices.device != device
    ):
        raise ValueError("Prolog staging_indices must be contiguous INT64 [physical_tokens] on the input device")


def load_prolog():
    try:
        spec = importlib.util.find_spec(PROLOG_MODULE)
    except ModuleNotFoundError as exc:
        raise RuntimeError("Load the private Prolog wheel env.sh before starting workers") from exc
    if spec is None or spec.origin is None:
        raise RuntimeError("Pinned Prolog wheel is absent; source its private env.sh first")
    if hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest() != PROLOG_SHA256:
        raise RuntimeError("K3 DSL Prolog source differs from pinned 9e01355; refusing an unknown ABI")
    if not (Path(spec.origin).parent / "_native").is_dir():
        raise RuntimeError("Prolog wheel has no _native directory; source-only/JIT fallback is not selected")
    return importlib.import_module(PROLOG_MODULE).quant_mla_prolog


class QFMProlog:
    def __init__(self, weights, kv_scale):
        self.op = load_prolog()
        # One initialization-time check, never a device-content read in forward
        # or graph capture. Upstream needs a multiplier; QFM needs a descale.
        # Unit scale makes these identical, and is fixed for all cache pages.
        if kv_scale.dtype != torch.float32 or kv_scale.shape != (1,) or kv_scale.detach().cpu().tolist() != [1.0]:
            raise ValueError("QFM DSL Prolog requires the internal unit KV scale")
        self.scale = kv_scale
        self.weights = weights

    def update_weights(self, weights):
        """Preserve captured tensor addresses on compatible weight refreshes."""
        if weights.keys() != self.weights.keys():
            raise ValueError("QFM Prolog weight keys changed; restart workers and recapture")
        for name, value in weights.items():
            old = self.weights[name]
            if isinstance(value, torch.Tensor):
                if old.shape != value.shape or old.dtype != value.dtype or old.device != value.device:
                    raise ValueError(f"QFM Prolog {name} layout changed; restart workers and recapture")
            elif old != value:
                raise ValueError(f"QFM Prolog {name} changed; restart workers and recapture")
        for name, value in weights.items():
            if isinstance(value, torch.Tensor):
                self.weights[name].copy_(value)

    def __call__(self, hidden, slots, cache, *, staging_indices=None, scatter_plan=None):
        tokens = slots.numel()
        if hidden.ndim != 2 or hidden.shape[0] != tokens or hidden.dtype != torch.bfloat16:
            raise ValueError("QFM DSL Prolog needs BF16 [physical_tokens, hidden_dim]")
        if slots.dtype != torch.int64 or slots.ndim != 1 or not slots.is_contiguous():
            raise ValueError("QFM DSL Prolog slots must be contiguous INT64 [physical_tokens]")
        if cache.dtype != torch.float8_e4m3fn or cache.shape[1:] != (1, 18, 128, 32):
            raise ValueError("QFM DSL Prolog requires full576 FP8 PA_NZ cache")
        if cache.stride()[2:] != (4096, 32, 1) or cache.stride(0) < QFM_BLOCK * QFM_WIDTH:
            raise ValueError("Invalid QFM cache page layout")
        if staging_indices is not None:
            _validate_staging_indices(staging_indices, tokens, hidden.device)
        x, descale_x = torch_npu.npu_dynamic_mx_quant(hidden, dst_type=torch.float8_e4m3fn, scale_alg=0)
        descale_x = descale_x.view(torch.float8_e8m0fnu).reshape(tokens, hidden.shape[1] // 32).contiguous()
        direct = cache.stride(0) == QFM_BLOCK * QFM_WIDTH
        if direct:
            # This is a shape-only physical alias, not an NZ-to-ND transpose.
            # Never call contiguous(): that could silently copy the history.
            destination = cache.view(cache.shape[0], QFM_BLOCK, 1, QFM_WIDTH)
            indices, mode = slots, PROLOG_PA_NZ
        else:
            # The upstream writer has no page-stride parameter. Generate only
            # this step's FP8 rows, then scatter bytes to the real strided pages.
            blocks = max(1, (tokens + QFM_BLOCK - 1) // QFM_BLOCK)
            destination = torch.empty((blocks, QFM_BLOCK, 1, QFM_WIDTH), dtype=cache.dtype, device=cache.device)
            indices = staging_indices
            if indices is None:
                indices = torch.arange(tokens, dtype=torch.int64, device=slots.device)
            mode = PROLOG_PA_BSND
        q, q_scale, _, _ = self.op(
            x=x,
            descale_x=descale_x,
            kv_cache=destination,
            cache_index=indices,
            quant_scale_kv=self.scale,
            w_qr_quant_mode=1,
            qkv_quant_mode=2,
            cache_mode=mode,
            rope_mode=0,
            return_qr=False,
            **self.weights,
        )
        if not direct:
            # All scratch rows were written, even when real slots are -1.
            # The final writer alone applies live slots, preserving draft and
            # unrelated main-model pages, page padding and storage offsets.
            scatter_fp8_cache(destination.view(-1, QFM_WIDTH)[:tokens], slots, cache, scatter_plan=scatter_plan)
        return q, q_scale.squeeze(-1).contiguous()

    def floating(self, hidden, slots, cache, *, staging_indices=None, scatter_plan=None):
        """Mode0 current-token BF16 scratch; persistent history stays FP8.

        This is the floating-output path for FIA Prefill. Stage-specific
        mixed batches keep Decode in mode2 and stage Prefill separately.
        """
        tokens = slots.numel()
        if hidden.dtype != torch.bfloat16 or hidden.ndim != 2 or hidden.shape[0] != tokens:
            raise ValueError("Mode0 Prolog needs BF16 [physical_tokens, hidden_dim]")
        if staging_indices is not None:
            _validate_staging_indices(staging_indices, tokens, hidden.device)
        x, scales = torch_npu.npu_dynamic_mx_quant(hidden, dst_type=torch.float8_e4m3fn, scale_alg=0)
        scales = scales.view(torch.float8_e8m0fnu).reshape(tokens, hidden.shape[1] // 32).contiguous()
        blocks = max(1, (tokens + QFM_BLOCK - 1) // QFM_BLOCK)
        current = torch.empty((blocks, QFM_BLOCK, 1, QFM_WIDTH), dtype=hidden.dtype, device=hidden.device)
        indices = staging_indices
        if indices is None:
            indices = torch.arange(tokens, dtype=torch.int64, device=hidden.device)
        q, _, _, _ = self.op(
            x=x,
            descale_x=scales,
            kv_cache=current,
            cache_index=indices,
            quant_scale_kv=None,
            w_qr_quant_mode=1,
            qkv_quant_mode=0,
            cache_mode=PROLOG_PA_BSND,
            rope_mode=0,
            return_qr=False,
            **self.weights,
        )
        kv = current.view(-1, QFM_WIDTH)[:tokens]
        write_cache(kv, slots, cache, self.scale, scatter_plan=scatter_plan)
        return q, kv
