// Experimental source schema. A matching host tiling/API implementation must be
// built and registered before this new operator can be called. Not a patch to
// the installed SparseFlashAttention ABI.
#pragma once
#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {
REG_OP(SparseFlashAttentionColdMerge)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(value, ge::TensorType::ALL())
    .INPUT(sparse_indices, ge::TensorType::ALL())
    .INPUT(block_table, ge::TensorType::ALL())
    .INPUT(actual_seq_lengths_query, ge::TensorType::ALL())
    .INPUT(actual_seq_lengths_kv, ge::TensorType::ALL())
    .INPUT(query_rope, ge::TensorType::ALL())
    .INPUT(key_rope, ge::TensorType::ALL())
    .INPUT(cold_output, ge::TensorType::ALL())
    .INPUT(cold_max, ge::TensorType::ALL())
    .INPUT(cold_sum, ge::TensorType::ALL())
    .INPUT(cold_present, ge::TensorType::ALL())
    .INPUT(hot_present, ge::TensorType::ALL())
    .OUTPUT(attention_out, ge::TensorType::ALL())
    .OUTPUT(softmax_max, ge::TensorType::ALL())
    .OUTPUT(softmax_sum, ge::TensorType::ALL())
    .REQUIRED_ATTR(scale_value, Float)
    .ATTR(sparse_block_size, Int, 1)
    .ATTR(layout_query, String, "TND")
    .ATTR(layout_kv, String, "PA_BSND")
    .ATTR(sparse_mode, Int, 0)
    .ATTR(pre_tokens, Int, 9223372036854775807)
    .ATTR(next_tokens, Int, 9223372036854775807)
    .ATTR(attention_mode, Int, 2)
    .ATTR(return_softmax_lse, Bool, false)
    .OP_END_FACTORY_REG(SparseFlashAttentionColdMerge);
}

// First-version contract: BF16 query/K/V/RoPE, FP32 cold O/max/sum and final O,
// bool presence; fixed B1/Q4/H64/D512/R64/W128, ring256. Host shape/dtype/value
// validation and original tiling computation must be supplied by the build.
// First fused entry requires return_softmax_lse=false (empty statistics).
// Returning combined statistics requires a separately implemented contract.
