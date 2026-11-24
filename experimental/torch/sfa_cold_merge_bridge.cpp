// Experimental Torch bridge source; not yet SDK-compiled. Includes a real
// original SFA API control and a distinct new fused API. Missing new symbols
// fail explicitly; no fallback or hidden standalone merge is implemented.
#include "aclnn_common.h"  // CANN ACL headers must precede torch_npu headers.
#include <torch/extension.h>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <tuple>
#include "aclnn_sparse_flash_attention_cold_merge.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

namespace {
using State = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

void Check(const at::Tensor &tensor, const at::Tensor &query,
           std::initializer_list<int64_t> shape, at::ScalarType dtype,
           const char *name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device() == query.device(), name, " device mismatch");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(!tensor.requires_grad(), name, " is inference-only");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " dtype mismatch");
    TORCH_CHECK(tensor.sizes().equals(at::IntArrayRef(shape)), name, " fixed-shape mismatch");
}

void CheckHot(const at::Tensor &query, const at::Tensor &key,
              const at::Tensor &value, const at::Tensor &indices,
              const at::Tensor &table, const at::Tensor &queryLengths,
              const at::Tensor &kvLengths, const at::Tensor &queryRope,
              const at::Tensor &keyRope, double scale)
{
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1,
                "An Ascend NPU tensor is required");
    TORCH_CHECK(std::isfinite(scale) && scale > 0, "Scale must be finite and positive");
    Check(query, query, {4, 64, 512}, at::kBFloat16, "query");
    Check(key, query, {2, 128, 1, 512}, at::kBFloat16, "key");
    Check(value, query, {2, 128, 1, 512}, at::kBFloat16, "value");
    Check(indices, query, {4, 1, 128}, at::kInt, "indices");
    Check(table, query, {1, 2}, at::kInt, "block_table");
    Check(queryLengths, query, {1}, at::kInt, "query_lengths");
    Check(kvLengths, query, {1}, at::kInt, "kv_lengths");
    Check(queryRope, query, {4, 64, 64}, at::kBFloat16, "query_rope");
    Check(keyRope, query, {2, 128, 1, 64}, at::kBFloat16, "key_rope");
    // No device value copies/synchronization here. Before timing, the harness
    // must validate query lengths=4, KV length=256, unique in-range compact
    // indices, per-query ring tags/causality and 128 hot selections per row.
}

State Outputs(const at::Tensor &query, bool fused, bool hotStats)
{
    auto output = at::empty({4, 64, 512}, query.options().dtype(fused ? at::kFloat : at::kBFloat16));
    auto maximum = at::empty(hotStats ? at::IntArrayRef({1, 4, 64}) : at::IntArrayRef({0}),
                             query.options().dtype(at::kFloat));
    auto sum = at::empty_like(maximum);
    return {output, maximum, sum};
}

State NativeHot(const at::Tensor &query, const at::Tensor &key,
                const at::Tensor &value, const at::Tensor &indices,
                const at::Tensor &table, const at::Tensor &queryLengths,
                const at::Tensor &kvLengths, const at::Tensor &queryRope,
                const at::Tensor &keyRope, double scale, bool hotStats)
{
    CheckHot(query, key, value, indices, table, queryLengths, kvLengths, queryRope, keyRope, scale);
    auto outputs = Outputs(query, false, hotStats);
    char layoutQuery[] = "TND", layoutKv[] = "PA_BSND";
    const int64_t unlimited = std::numeric_limits<int64_t>::max();
    ACLNN_CMD(aclnnSparseFlashAttention, query, key, value, indices, table,
              queryLengths, kvLengths, queryRope, keyRope,
              scale, int64_t(1), layoutQuery, layoutKv, int64_t(0),
              unlimited, unlimited, int64_t(2), hotStats,
              std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs));
    return outputs;
}

State FusedHot(const at::Tensor &query, const at::Tensor &key,
               const at::Tensor &value, const at::Tensor &indices,
               const at::Tensor &table, const at::Tensor &queryLengths,
               const at::Tensor &kvLengths, const at::Tensor &queryRope,
               const at::Tensor &keyRope, const at::Tensor &coldOutput,
               const at::Tensor &coldMax, const at::Tensor &coldSum,
               const at::Tensor &coldPresent, const at::Tensor &hotPresent,
               double scale, bool hotStats)
{
    TORCH_CHECK(!hotStats, "The first fused entry returns final output only; combined softmax statistics are not implemented");
    CheckHot(query, key, value, indices, table, queryLengths, kvLengths, queryRope, keyRope, scale);
    Check(coldOutput, query, {4, 64, 512}, at::kFloat, "cold_output");
    Check(coldMax, query, {4, 64, 1}, at::kFloat, "cold_max");
    Check(coldSum, query, {4, 64, 1}, at::kFloat, "cold_sum");
    Check(coldPresent, query, {4}, at::kBool, "cold_present");
    Check(hotPresent, query, {4}, at::kBool, "hot_present");
    auto outputs = Outputs(query, true, hotStats);
    char layoutQuery[] = "TND", layoutKv[] = "PA_BSND";
    const int64_t unlimited = std::numeric_limits<int64_t>::max();
    ACLNN_CMD(aclnnSparseFlashAttentionColdMerge, query, key, value, indices, table,
              queryLengths, kvLengths, queryRope, keyRope,
              coldOutput, coldMax, coldSum, coldPresent, hotPresent,
              scale, int64_t(1), layoutQuery, layoutKv, int64_t(0),
              unlimited, unlimited, int64_t(2), hotStats,
              std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs));
    return outputs;
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def("native_hot", &NativeHot, "Original native BF16 SFA control");
    module.def("fused_hot", &FusedHot, "Experimental native hot/cold merge; requires new vendor API");
}
