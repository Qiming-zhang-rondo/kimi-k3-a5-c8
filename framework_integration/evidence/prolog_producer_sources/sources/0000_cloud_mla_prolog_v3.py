# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import cloud_ops_turbo_lib
from typing import Optional, Tuple, List


def cloud_mla_prolog_v3(
    token_x: torch.Tensor,
    weight_dq: torch.Tensor,
    weight_uq_qr: torch.Tensor,
    weight_uk: torch.Tensor,
    weight_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    kv_cache: torch.Tensor,
    kr_cache: torch.Tensor,
    rope_sin: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    cache_index: Optional[torch.Tensor] = None,
    dequant_scale_x: Optional[torch.Tensor] = None,
    dequant_scale_w_dq: Optional[torch.Tensor] = None,
    dequant_scale_w_uq_qr: Optional[torch.Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[torch.Tensor] = None,
    quant_scale_ckv: Optional[torch.Tensor] = None,
    quant_scale_ckr: Optional[torch.Tensor] = None,
    smooth_scales_cq: Optional[torch.Tensor] = None,
    actual_seq_len: Optional[torch.Tensor] = None,
    k_nope_clip_alpha: Optional[torch.Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "BSND",
    query_norm_flag: bool = False,
    weight_quant_mode: int = 0,
    kv_cache_quant_mode: int = 0,
    query_quant_mode: int = 0,
    ckvkr_repo_mode: int = 0,
    quant_scale_repo_mode: int = 0,
    tile_size: int = 128,
    qc_qr_scale: float = 1.0,
    kc_scale: float = 1.0,
    token_x_dtype: Optional[int] = None,
    weight_dq_dtype: Optional[int] = None,
    weight_uq_qr_dtype: Optional[int] = None,
    weight_dkv_kr_dtype: Optional[int] = None,
    kv_cache_dtype: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cloud MLA prolog V3 operator (in-place on kv_cache/kr_cache).

    Args:
        token_x: Input tensor of shape [B, T, He] (3D) or [T, He] (2D).
        weight_dq: [He, Hcq] FRACTAL_NZ.
        weight_uq_qr: [Hcq, N*(D+Dr)] FRACTAL_NZ.
        weight_uk: [N, D, Hckv].
        weight_dkv_kr: [He, Hckv+Dr] FRACTAL_NZ.
        rmsnorm_gamma_cq / rmsnorm_gamma_ckv: 1D gamma.
        kv_cache / kr_cache: in-place updated cache tensors.
        rope_sin / rope_cos: optional rope params (both or none).
        cache_index: optional for PA cache modes.
        ... quant modes etc.

    Returns:
        Tuple of (query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm).
    """
    if not isinstance(token_x, torch.Tensor):
        raise ValueError(
            "cloud_mla_prolog_v3(): token_x is required and must be a torch.Tensor, but got {}".format(type(token_x))
        )
    return cloud_ops_turbo_lib.cloud_mla_prolog(
        token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, kv_cache,
        kr_cache, rope_sin, rope_cos, cache_index, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr,
        dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, actual_seq_len, k_nope_clip_alpha,
        rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode, query_norm_flag, weight_quant_mode, kv_cache_quant_mode,
        query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, token_x_dtype,
        weight_dq_dtype, weight_uq_qr_dtype, weight_dkv_kr_dtype, kv_cache_dtype
    )


def cloud_mla_prolog_v3_functional(
    token_x: torch.Tensor,
    weight_dq: torch.Tensor,
    weight_uq_qr: torch.Tensor,
    weight_uk: torch.Tensor,
    weight_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    kv_cache: torch.Tensor,
    kr_cache: torch.Tensor,
    rope_sin: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    cache_index: Optional[torch.Tensor] = None,
    dequant_scale_x: Optional[torch.Tensor] = None,
    dequant_scale_w_dq: Optional[torch.Tensor] = None,
    dequant_scale_w_uq_qr: Optional[torch.Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[torch.Tensor] = None,
    quant_scale_ckv: Optional[torch.Tensor] = None,
    quant_scale_ckr: Optional[torch.Tensor] = None,
    smooth_scales_cq: Optional[torch.Tensor] = None,
    actual_seq_len: Optional[torch.Tensor] = None,
    k_nope_clip_alpha: Optional[torch.Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "BSND",
    query_norm_flag: bool = False,
    weight_quant_mode: int = 0,
    kv_cache_quant_mode: int = 0,
    query_quant_mode: int = 0,
    ckvkr_repo_mode: int = 0,
    quant_scale_repo_mode: int = 0,
    tile_size: int = 128,
    qc_qr_scale: float = 1.0,
    kc_scale: float = 1.0,
    token_x_dtype: Optional[int] = None,
    weight_dq_dtype: Optional[int] = None,
    weight_uq_qr_dtype: Optional[int] = None,
    weight_dkv_kr_dtype: Optional[int] = None,
    kv_cache_dtype: Optional[int] = None,
) -> Tuple[torch.Tensor, ...]:
    """Functional version: clones kv_cache/kr_cache, returns 7 tensors."""
    if not isinstance(token_x, torch.Tensor):
        raise ValueError(
            "cloud_mla_prolog_v3_functional(): token_x is required and must be a torch.Tensor, but got {}".format(
                type(token_x))
        )
    return cloud_ops_turbo_lib.cloud_mla_prolog_functional(
        token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, kv_cache,
        kr_cache, rope_sin, rope_cos, cache_index, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr,
        dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, actual_seq_len, k_nope_clip_alpha,
        rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode, query_norm_flag, weight_quant_mode, kv_cache_quant_mode,
        query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, token_x_dtype,
        weight_dq_dtype, weight_uq_qr_dtype, weight_dkv_kr_dtype, kv_cache_dtype
    )
