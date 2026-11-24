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


def cloud_sparse_flash_attention(query, key, value, sparse_indices, block_table=None,
                                  actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
                                  query_rope=None, key_rope=None, scale_value=1.0,
                                  sparse_block_size=1, layout_query="BSND", layout_kv="BSND",
                                  sparse_mode=3, pre_tokens=None, next_tokens=None,
                                  attention_mode=2, return_softmax_lse=False):
    """Sparse Flash Attention (SFA).

    Args:
        query: shape (B, S1, N1, D) for BSND or (T1, N1, D) for TND.
        key: shape (B, S2, N2, D) for BSND, (T2, N2, D) for TND, or
             (block_num, block_size, N2, D) for PA_BSND.
        value: same shape as key.
        sparse_indices: shape (B, S1, N2, sparse_size) for BSND or (T1, N2, sparse_size) for TND.
        block_table: (B, S2/block_size), required when layout_kv == "PA_BSND".
        actual_seq_lengths_query: (B,), required when layout_query == "TND".
        actual_seq_lengths_kv: (B,), required when layout_kv in ("TND", "PA_BSND").
        query_rope/key_rope: optional rope inputs (MLA).
        scale_value: QK scale.
        sparse_block_size: sparse granularity.
        layout_query: "BSND" or "TND".
        layout_kv: "BSND", "TND" or "PA_BSND".
        sparse_mode: 0/full, 3/right-down-causal.
        pre_tokens/next_tokens: sparse window bounds.
        attention_mode: attention mode (2 for MLA).
        return_softmax_lse: whether to return softmax_max/softmax_sum.

    Returns:
        (attention_out, softmax_max, softmax_sum). softmax_max/sum are empty
        (shape [0]) when return_softmax_lse is False.
    """
    if pre_tokens is None:
        pre_tokens = torch.iinfo(torch.int64).max
    if next_tokens is None:
        next_tokens = torch.iinfo(torch.int64).max
    return torch.ops.cloud_ops_turbo.cloud_sparse_flash_attention(
        query, key, value, sparse_indices, block_table, actual_seq_lengths_query,
        actual_seq_lengths_kv, query_rope, key_rope, scale_value, sparse_block_size,
        layout_query, layout_kv, sparse_mode, pre_tokens, next_tokens, attention_mode,
        return_softmax_lse)
