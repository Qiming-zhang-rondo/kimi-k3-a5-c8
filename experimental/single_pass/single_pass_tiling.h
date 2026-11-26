// New standalone prototype ABI1, serialized by THIS host and consumed by THIS
// rebuilt kernel. This is NOT the installed private C4 tiling layout and must
// never be passed to an installed ACLNN/TBE binary. Field names below come from
// native_c4_source's arch35 QSFAVectorService::InitCubeVecSharedParams.
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace recent128_single_pass {
inline constexpr std::uint32_t abiVersion = 1;
inline constexpr std::uint32_t pageSize = 128;
inline constexpr std::uint32_t heads = 64;
inline constexpr std::uint32_t queryDim = 576;
inline constexpr std::uint32_t latentDim = 512;
inline constexpr std::uint32_t ropeDim = 64;
inline constexpr std::uint32_t packedDim = 400;

struct alignas(8) Base {
    std::uint32_t batchSize;
    std::uint32_t seqSize;
    std::uint32_t qSeqSize;
    std::int64_t blockSize;
    std::uint32_t maxBlockNumPerBatch;
    float scaleValue;
    std::uint32_t nNumOfQInOneGroup;
    std::uint32_t sparseMode;
    std::uint32_t outputLayout;
    std::uint32_t dSizeVInput;
    std::uint32_t sparseBlockCount;
    std::uint32_t isActualLenDimsNull;
    std::uint32_t isActualLenDimsKVNull;
    std::uint32_t returnSoftmaxLse;
};
struct alignas(8) SingleCore {
    std::uint32_t usedCoreNum;
};
struct alignas(8) Tiling {
    Base baseParams;
    SingleCore singleCoreParams;
};
// Additive ABI2 for the isolated KV-split entry. Existing Tiling stays 72B.
// Offsets are byte offsets into caller-owned FP32 partial workspace.
struct alignas(8) SplitParams {
    std::uint32_t abiVersion;
    std::uint32_t splitCount;
    std::uint64_t partialOutputOffsetBytes;
    std::uint64_t partialMaximumOffsetBytes;
    std::uint64_t partialSumOffsetBytes;
    std::uint64_t workspaceBytes;
};
struct alignas(8) SplitTiling {
    Base baseParams;
    SingleCore singleCoreParams;
    SplitParams splitParams;
};
static_assert(sizeof(SplitParams) == 40 && sizeof(SplitTiling) == 112);
static_assert(offsetof(SplitTiling, splitParams) == 72);
static_assert(std::is_standard_layout<SplitTiling>::value && std::is_trivially_copyable<SplitTiling>::value);
static_assert(std::is_standard_layout<Tiling>::value && std::is_trivially_copyable<Tiling>::value);
static_assert(sizeof(Base) == 64 && alignof(Base) == 8);
static_assert(sizeof(SingleCore) == 8 && sizeof(Tiling) == 72 && alignof(Tiling) == 8);
static_assert(offsetof(Tiling, baseParams) == 0 && offsetof(Tiling, singleCoreParams) == 64);
static_assert(offsetof(Base, blockSize) == 16 && offsetof(Base, dSizeVInput) == 44);

// The actual C4 InitMMResBuf reserves GM v0 buffers ONLY for IS_SPLIT_G=true.
// This prototype fixes IS_SPLIT_G=false, so no GM scratch is consumed. Keep a
// nonnull 1-byte allocation for the direct pointer ABI. Do NOT import the BF16
// reader's different 442368-bytes/core scratch formula.
inline constexpr std::size_t userWorkspaceBytes = 0;
inline constexpr std::size_t userWorkspaceAllocationBytes = 1;
} // namespace recent128_single_pass

// The pinned real kernel refers to this type name. Its fields now belong to
// the explicitly new ABI above; both host and kernel MUST rebuild together.
#if defined(QSFA_RECENT_KV_SPLIT) && QSFA_RECENT_KV_SPLIT
using KvQuantSparseFlashAttentionTilingDataMla = recent128_single_pass::SplitTiling;
#else
using KvQuantSparseFlashAttentionTilingDataMla = recent128_single_pass::Tiling;
#endif

#if !defined(__NPU_ARCH__) || __NPU_ARCH__ == 0
#include <cmath>
#include <stdexcept>
namespace recent128_single_pass {
inline Tiling MakeTiling(std::uint32_t queries, std::uint32_t keys,
                         std::uint32_t aicCount, float scale, bool returnStats,
                         std::uint32_t topK = 2048)
{
    if (queries == 0 || queries > 1024 || keys < queries || keys > INT32_MAX ||
        aicCount == 0 || aicCount > queries || topK == 0 || topK > 2048 ||
        !std::isfinite(scale) || scale <= 0.0F) {
        throw std::invalid_argument("Invalid new standalone single-pass ABI1 geometry or scale");
    }
    Tiling result{};
    auto &b = result.baseParams;
    b.batchSize = 1;
    b.seqSize = keys;
    b.qSeqSize = queries;
    b.blockSize = pageSize;
    b.maxBlockNumPerBatch = (keys + pageSize - 1) / pageSize;
    b.scaleValue = scale;
    b.nNumOfQInOneGroup = heads;
    b.sparseMode = 0; // Caller rejects future IDs before launch.
    b.outputLayout = 1; // TND; template KV layout is PA_BSND=2.
    b.dSizeVInput = packedDim; // Selects the REAL MXFP4 branch, not FP8 C8.
    b.sparseBlockCount = topK;
    b.isActualLenDimsNull = 0;
    b.isActualLenDimsKVNull = 0;
    b.returnSoftmaxLse = returnStats ? 1 : 0;
    result.singleCoreParams.usedCoreNum = aicCount;
    return result;
}
} // namespace recent128_single_pass
#endif
