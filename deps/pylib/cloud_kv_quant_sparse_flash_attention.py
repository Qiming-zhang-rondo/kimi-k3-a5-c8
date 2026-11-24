#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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


def cloud_kv_quant_sparse_flash_attention(query, key, value, sparse_indices,
                                          key_dequant_scale=None, value_dequant_scale=None,
                                          block_table=None, actual_seq_lengths_query=None,
                                          actual_seq_lengths_kv=None, scale_value=1.0,
                                          key_quant_mode=2, value_quant_mode=2,
                                          sparse_block_size=1, layout_query="BSND",
                                          layout_kv="BSND", sparse_mode=3,
                                          pre_tokens=None, next_tokens=None,
                                          attention_mode=2, quant_scale_repo_mode=1,
                                          tile_size=128, rope_head_dim=64,
                                          return_softmax_lse=False):
    """KV-Quant Sparse Flash Attention (QSFA).

    Extends sparse_flash_attention to support Per-Token-Head-Tile-128 quantized
    KV input (INT8 / FLOAT8_E4M3FN / HIFLOAT8 KV, FP16/BF16 Q).

    Args:
        query: shape (B, S1, N1, D) for BSND or (T1, N1, D) for TND.
            D = head_dim + rope_head_dim (e.g. 576 = 512 + 64).
        key: shape (B, S2, N2, KV_D) for BSND, (T2, N2, KV_D) for TND, or
             (block_num, block_size, N2, KV_D) for PA_BSND.
             KV_D = head_dim + rope_head_dim*2 + dequant_scale_size when
             quant_scale_repo_mode=COMBINE (e.g. 656 = 512 + 64*2 + 4*4).
        value: same shape as key.
        sparse_indices: shape (B, S1, N2, sparse_size) for BSND or
                        (T1, N2, sparse_size) for TND.
        key_dequant_scale: optional dequant scale for key (SEPARATE mode).
        value_dequant_scale: optional dequant scale for value (SEPARATE mode).
        block_table: (B, S2/block_size), required when layout_kv == "PA_BSND".
        actual_seq_lengths_query: (B,), required when layout_query == "TND".
        actual_seq_lengths_kv: (B,), required when layout_kv in ("TND", "PA_BSND").
        scale_value: QK scale.
        key_quant_mode: 0=PER_CHANNEL, 1=PER_TOKEN_HEAD, 2=PER_TILE.
        value_quant_mode: same as key_quant_mode.
        sparse_block_size: sparse granularity.
        layout_query: "BSND" or "TND".
        layout_kv: "BSND", "TND" or "PA_BSND".
        sparse_mode: 0/full, 3/right-down-causal.
        pre_tokens/next_tokens: sparse window bounds.
        attention_mode: attention mode (2 for MLA_ABSORB).
        quant_scale_repo_mode: 0=SEPARATE, 1=COMBINE.
        tile_size: tile size for PER_TILE quantization (default 128).
        rope_head_dim: rope head dimension (default 64).
        return_softmax_lse: whether to return softmax_max/softmax_sum.

    Returns:
        (attention_out, softmax_max, softmax_sum). attention_out last dim is
        D - rope_head_dim (e.g. 512). softmax_max/sum are empty (shape [0])
        when return_softmax_lse is False.
    """
    if pre_tokens is None:
        pre_tokens = torch.iinfo(torch.int64).max
    if next_tokens is None:
        next_tokens = torch.iinfo(torch.int64).max
    return torch.ops.cloud_ops_turbo.cloud_kv_quant_sparse_flash_attention(
        query, key, value, sparse_indices, key_dequant_scale, value_dequant_scale,
        block_table, actual_seq_lengths_query, actual_seq_lengths_kv, scale_value,
        key_quant_mode, value_quant_mode, sparse_block_size, layout_query, layout_kv,
        sparse_mode, pre_tokens, next_tokens, attention_mode, quant_scale_repo_mode,
        tile_size, rope_head_dim, return_softmax_lse)
