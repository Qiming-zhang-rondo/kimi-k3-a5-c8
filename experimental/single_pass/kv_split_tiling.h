// Explicit standalone split ABI2. Never pass this POD to old ABI1 or ACLNN.
#pragma once
#include "single_pass_tiling.h"

namespace recent128_single_pass {
inline constexpr std::uint32_t splitAbiVersion = 2;
inline constexpr std::uint32_t splitTopK = 2048;
inline constexpr std::uint32_t maximumSplitTasks = 16;

#if !defined(__NPU_ARCH__) || __NPU_ARCH__ == 0
inline SplitTiling MakeSplitTiling(std::uint32_t queries, std::uint32_t keys,
                                  std::uint32_t splits, float scale, bool returnStats)
{
    if (queries == 0 || (splits != 1 && splits != 2 && splits != 4) ||
        queries > maximumSplitTasks / splits || keys < splitTopK || keys % pageSize != 0) {
        throw std::invalid_argument("KV-split requires S1/2/4, Q*S<=16 and page-aligned K>=2048");
    }
    const auto original = MakeTiling(queries, keys, queries, scale, returnStats, splitTopK);
    SplitTiling result{};
    result.baseParams = original.baseParams;
    result.singleCoreParams = original.singleCoreParams;
    result.singleCoreParams.usedCoreNum = queries * splits;
    auto &p = result.splitParams;
    p.abiVersion = splitAbiVersion;
    p.splitCount = splits;
    const std::uint64_t tasks = static_cast<std::uint64_t>(queries) * splits;
    p.partialOutputOffsetBytes = 0;
    p.partialMaximumOffsetBytes = tasks * heads * latentDim * sizeof(float);
    p.partialSumOffsetBytes = p.partialMaximumOffsetBytes + tasks * heads * sizeof(float);
    p.workspaceBytes = p.partialSumOffsetBytes + tasks * heads * sizeof(float);
    if (p.partialMaximumOffsetBytes % 32 || p.partialSumOffsetBytes % 32 || p.workspaceBytes % 32) {
        throw std::invalid_argument("KV-split workspace is not 32-byte aligned");
    }
    return result;
}
#endif
}  // namespace recent128_single_pass
