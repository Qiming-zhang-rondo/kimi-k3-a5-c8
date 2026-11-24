// Fixed host-side contract for the archived arch35 BF16 SFA compile probe.
// This header has no SDK dependency and never allocates, launches, or touches a
// device. Its POD is the pinned SGL definition, NOT an installed binary's ABI.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <type_traits>

#include "sparse_flash_attention_tiling_data.h"

namespace sfa_fixed {

inline constexpr std::uint32_t batch = 1;
inline constexpr std::uint32_t queries = 4;
inline constexpr std::uint32_t heads = 64;
inline constexpr std::uint32_t kvHeads = 1;
inline constexpr std::uint32_t latentDim = 512;
inline constexpr std::uint32_t ropeDim = 64;
inline constexpr std::uint32_t physicalKv = 256;
inline constexpr std::uint32_t pageSize = 128;
inline constexpr std::uint32_t physicalBlocks = 2;
inline constexpr std::uint32_t sparseCompact = 128;
inline constexpr std::uint32_t launchAic = 1;
inline constexpr std::uint32_t launchAiv = 2;
inline constexpr float softmaxScale = 1.0F / 24.0F; // 1 / sqrt(512 + 64).
inline constexpr std::size_t bf16Bytes = 2;
inline constexpr std::size_t fp32Bytes = 4;
inline constexpr std::size_t int32Bytes = 4;
inline constexpr std::size_t queryRows = batch * queries * heads;
inline constexpr std::size_t indexElements = batch * queries * kvHeads * sparseCompact;

// original/arch35/sparse_flash_attention_kernel_mla.h::InitMMResBuf:
// v0ResSize = s2BaseSize(128) * 576 * sizeof(Q_T=BF16), three GM buffers,
// indexed by aicIdx (IS_SPLIT_G=0). Both AIVs share their AIC's reservation.
// This is USER scratch only. The fixed direct wrapper forwards it unchanged;
// it must not pass through GetUserWorkspace or include a runtime system prefix.
inline constexpr std::size_t userWorkspaceBytes =
    std::size_t{128} * (latentDim + ropeDim) * bf16Bytes * 3 * launchAic;
static_assert(userWorkspaceBytes == 442368);
static_assert(physicalBlocks * pageSize == physicalKv);
static_assert(heads / kvHeads == 64 && heads % kvHeads == 0);
static_assert(launchAiv == 2 * launchAic);
static_assert(sizeof(float) == fp32Bytes && sizeof(std::int32_t) == int32Bytes);

using Tiling = SparseFlashAttentionTilingDataMla;
using Base = SparseFlashAttentionBaseParamsMla;
static_assert(std::is_standard_layout<Tiling>::value && std::is_trivially_copyable<Tiling>::value);
static_assert(sizeof(Tiling) == 144 && alignof(Tiling) == 8);
static_assert(sizeof(Base) == 104 && alignof(Base) == 8);
#define SFA_FIXED_OFFSET(type, field, offset) \
    static_assert(offsetof(type, field) == offset, "Pinned SFA POD field offset changed")
SFA_FIXED_OFFSET(Base, batchSize, 0);
SFA_FIXED_OFFSET(Base, seqSize, 4);
SFA_FIXED_OFFSET(Base, qSeqSize, 8);
SFA_FIXED_OFFSET(Base, blockSize, 16);
SFA_FIXED_OFFSET(Base, maxBlockNumPerBatch, 24);
SFA_FIXED_OFFSET(Base, scaleValue, 28);
SFA_FIXED_OFFSET(Base, nNumOfQInOneGroup, 32);
SFA_FIXED_OFFSET(Base, actualLenDimsQ, 36);
SFA_FIXED_OFFSET(Base, actualLenDimsKV, 40);
SFA_FIXED_OFFSET(Base, outputLayout, 44);
SFA_FIXED_OFFSET(Base, sparseMode, 48);
SFA_FIXED_OFFSET(Base, preTokens, 56);
SFA_FIXED_OFFSET(Base, nextTokens, 64);
SFA_FIXED_OFFSET(Base, attentionMode, 72);
SFA_FIXED_OFFSET(Base, returnSoftmaxLse, 76);
SFA_FIXED_OFFSET(Base, sparseBlockSize, 80);
SFA_FIXED_OFFSET(Base, sparseBlockCount, 88);
SFA_FIXED_OFFSET(Base, isActualLenDimsNull, 92);
SFA_FIXED_OFFSET(Base, isActualLenDimsKVNull, 96);
SFA_FIXED_OFFSET(Base, dispatchKey, 100);
SFA_FIXED_OFFSET(Tiling, baseParams, 0);
SFA_FIXED_OFFSET(Tiling, splitKVParams, 104);
SFA_FIXED_OFFSET(Tiling, singleCoreParams, 120);
SFA_FIXED_OFFSET(Tiling, singleCoreTensorSize, 128);
SFA_FIXED_OFFSET(Tiling, innerSplitParams, 136);
SFA_FIXED_OFFSET(SparseFlashAttentionSplitKVParamsMla, s2, 0);
SFA_FIXED_OFFSET(SparseFlashAttentionSplitKVParamsMla, accumOutSize, 4);
SFA_FIXED_OFFSET(SparseFlashAttentionSplitKVParamsMla, logSumExpSize, 8);
SFA_FIXED_OFFSET(SparseFlashAttentionSingleCoreParamsMla, usedCoreNum, 0);
SFA_FIXED_OFFSET(SparseFlashAttentionSingleCoreTensorSizeMla, mmResUbSize, 0);
SFA_FIXED_OFFSET(SparseFlashAttentionSingleCoreTensorSizeMla, bmm2ResUbSize, 4);
SFA_FIXED_OFFSET(SparseFlashAttentionInnerSplitParams, mBaseSize, 0);
SFA_FIXED_OFFSET(SparseFlashAttentionInnerSplitParams, s2BaseSize, 4);
#undef SFA_FIXED_OFFSET

inline Tiling MakeTiling(float scale = softmaxScale)
{
    Tiling t{};
    auto& b = t.baseParams;
    // Actual consumers: original arch35 vector::InitCubeVecSharedParams,
    // kernel::ComputeConstexpr and kvcache::{GetSingleCoreParam,ComputeS2LoopInfo}.
    // Semantic cross-check: pinned vllm host::FillTilingBaseParamsMla (354-372).
    b.batchSize = batch;
    b.seqSize = physicalKv;
    b.qSeqSize = queries;
    b.blockSize = pageSize;
    b.maxBlockNumPerBatch = physicalBlocks;
    b.scaleValue = scale;
    b.nNumOfQInOneGroup = heads / kvHeads; // gSize, NOT n2Size (which is 1).
    b.actualLenDimsQ = 1; // Length ARRAY size, not its value 4.
    b.actualLenDimsKV = 1; // Length ARRAY size, not its value 256.
    b.outputLayout = 1; // original SFA_LAYOUT::TND; KV template is PA_BSND=2.
    b.sparseMode = 0; // Every compact index is valid; no -1 padded tail.
    b.attentionMode = 2; // Pinned host validates mode 2; selected kernel is MLA.
    b.returnSoftmaxLse = 1; // Original ABI returns separate FP32 max and sum.
    b.sparseBlockSize = 1; // Kernel fixes sparse block size to one token.
    b.sparseBlockCount = sparseCompact;
    b.isActualLenDimsNull = 0;
    b.isActualLenDimsKVNull = 0;
    t.singleCoreParams.usedCoreNum = launchAic;
    // preTokens/nextTokens are not read from tiling on this sparseMode=0 path.
    // dispatchKey is not read by the fixed non-template entry. splitKVParams,
    // singleCoreTensorSize and innerSplitParams have no consumer in this
    // archived arch35 implementation (isFd=false, IS_SPLIT_G=false); leave zero.
    // Do not copy other host variants' buffer formulas into unused POD fields.
    return t;
}

inline bool ValidateTiling(const Tiling& t)
{
    const auto& b = t.baseParams;
    return std::isfinite(b.scaleValue) && b.scaleValue > 0.0F &&
        b.batchSize == batch && b.seqSize == physicalKv && b.qSeqSize == queries &&
        b.blockSize == pageSize && b.maxBlockNumPerBatch == physicalBlocks &&
        b.nNumOfQInOneGroup == heads / kvHeads && b.actualLenDimsQ == 1 &&
        b.actualLenDimsKV == 1 && b.outputLayout == 1 && b.sparseMode == 0 &&
        b.preTokens == 0 && b.nextTokens == 0 && b.attentionMode == 2 &&
        b.returnSoftmaxLse == 1 && b.sparseBlockSize == 1 &&
        b.sparseBlockCount == sparseCompact && b.isActualLenDimsNull == 0 &&
        b.isActualLenDimsKVNull == 0 && b.dispatchKey == 0 &&
        t.singleCoreParams.usedCoreNum == launchAic &&
        t.splitKVParams.s2 == 0 && t.splitKVParams.accumOutSize == 0 &&
        t.splitKVParams.logSumExpSize == 0 &&
        t.singleCoreTensorSize.mmResUbSize == 0 &&
        t.singleCoreTensorSize.bmm2ResUbSize == 0 &&
        t.innerSplitParams.mBaseSize == 0 && t.innerSplitParams.s2BaseSize == 0;
}

struct BufferBytes {
    std::size_t query, key, queryRope, keyRope, output, softmaxMax, softmaxSum;
    std::size_t indices, blockTable, actualQuery, actualKv, tiling, userWorkspace;
};

inline constexpr BufferBytes GetBufferBytes()
{
    // Original BF16 output only; an FP32 fused candidate requires a distinct
    // output allocation. Key is the MLA latent V too: pass key as value alias.
    return {queryRows * latentDim * bf16Bytes,
        std::size_t{physicalKv} * kvHeads * latentDim * bf16Bytes,
        queryRows * ropeDim * bf16Bytes,
        std::size_t{physicalKv} * kvHeads * ropeDim * bf16Bytes,
        queryRows * latentDim * bf16Bytes, queryRows * fp32Bytes,
        queryRows * fp32Bytes, indexElements * int32Bytes,
        std::size_t{batch} * physicalBlocks * int32Bytes,
        batch * int32Bytes, batch * int32Bytes, sizeof(Tiling), userWorkspaceBytes};
}

inline bool ValidateBufferBytes(const BufferBytes& allocated)
{
    const auto need = GetBufferBytes();
    return allocated.query >= need.query && allocated.key >= need.key &&
        allocated.queryRope >= need.queryRope && allocated.keyRope >= need.keyRope &&
        allocated.output >= need.output && allocated.softmaxMax >= need.softmaxMax &&
        allocated.softmaxSum >= need.softmaxSum && allocated.indices >= need.indices &&
        allocated.blockTable >= need.blockTable && allocated.actualQuery >= need.actualQuery &&
        allocated.actualKv >= need.actualKv && allocated.tiling >= need.tiling &&
        allocated.userWorkspace >= need.userWorkspace;
}

inline bool ValidateIndicesAndLengths(
    const std::int32_t* indices, std::size_t indicesCount,
    const std::int32_t* blockTable, std::size_t blockCount,
    const std::int32_t* actualQ, std::size_t actualQCount,
    const std::int32_t* actualKV, std::size_t actualKVCount)
{
    // Counts are host ELEMENT counts. Validate host mirrors before any device
    // launch. No synchronization/device read is performed by this function.
    if (!indices || !blockTable || !actualQ || !actualKV ||
        indicesCount != indexElements || blockCount != physicalBlocks ||
        actualQCount != batch || actualKVCount != batch ||
        actualQ[0] != static_cast<std::int32_t>(queries) ||
        actualKV[0] != static_cast<std::int32_t>(physicalKv)) return false;
    // This fixed probe uses an identity page table; no arbitrary cache mapping.
    for (std::size_t p = 0; p < physicalBlocks; ++p) {
        if (blockTable[p] != static_cast<std::int32_t>(p)) return false;
    }
    for (std::size_t q = 0; q < batch * queries; ++q) {
        std::array<bool, physicalKv> seen{};
        for (std::size_t j = 0; j < sparseCompact; ++j) {
            const auto index = indices[q * sparseCompact + j];
            if (index < 0 || index >= static_cast<std::int32_t>(physicalKv)) return false;
            if (seen[static_cast<std::size_t>(index)]) return false;
            seen[static_cast<std::size_t>(index)] = true;
            const auto block = static_cast<std::size_t>(index) / pageSize;
            const auto slot = static_cast<std::size_t>(blockTable[block]) * pageSize +
                static_cast<std::size_t>(index) % pageSize;
            if (slot >= physicalKv) return false;
        }
    }
    return true;
}

inline void WritePreflight(std::ostream& out)
{
    const auto b = GetBufferBytes();
    out << "HOST_PREFLIGHT_ONLY B=1 Q=4 H=64 Nkv=1 D=512 rope=64"
        << " KV=256 page=128 compact=128 launch_AIC=1 launch_AIV=2\n"
        << "bytes query=" << b.query << " key_and_value_alias=" << b.key
        << " query_rope=" << b.queryRope << " key_rope=" << b.keyRope
        << " bf16_output=" << b.output << " max=" << b.softmaxMax
        << " sum=" << b.softmaxSum << " indices=" << b.indices
        << " block_table=" << b.blockTable << " actual_q=" << b.actualQuery
        << " actual_kv=" << b.actualKv << " tiling=" << b.tiling
        << " user_workspace=" << b.userWorkspace << '\n'
        << "Raw-user-scratch direct ABI only; installed-package ABI is not reused.\n";
}

} // namespace sfa_fixed
