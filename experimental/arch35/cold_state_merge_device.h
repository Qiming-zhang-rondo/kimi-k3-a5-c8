// Experimental AscendC epilogue: NOT yet compiled or measured with the A5 SDK.
// Contract: TND, N2=1, H=64, D=512, contiguous FP32 cold statistics.
// Default cold/output storage remains FP32. SFA_MERGE_BF16_IO=1 is a separate
// compiled variant with BF16 cold storage and final BF16 output; math stays FP32.
// hotOutput contains the FINAL normalized FP32 hot result; this epilogue preserves
// the original native BF16 CAST_ROUND before merging, then leaves FP32 in UB.
// A call covers head rows of ONE token: flatOffset/H == (flatOffset+rows-1)/H.
// Caller validates shapes, bounds, disjoint storage and owns the selected final store.
// This bounded native-state implementation is not a generic arbitrary-sum merge:
// present hot/cold sums must be approximately [1,128] / [1,2048], respectively.
#ifndef SFA_EXPERIMENTAL_COLD_STATE_MERGE_DEVICE_H
#define SFA_EXPERIMENTAL_COLD_STATE_MERGE_DEVICE_H

#include "kernel_operator.h"
#ifndef SFA_MERGE_BF16_IO
#define SFA_MERGE_BF16_IO 0
#endif
#if SFA_MERGE_BF16_IO != 0 && SFA_MERGE_BF16_IO != 1
#error "SFA_MERGE_BF16_IO must be 0 or 1"
#endif
namespace AscendC {
#if SFA_MERGE_BF16_IO
// The two IO variants coexist in one shared library. Give every differing
// inline/helper symbol a distinct name rather than relying on weak coalescing.
namespace Bf16ColdMerge {
#endif

#if defined(__DAV_C310_CUBE__)
// The shared entry includes the vector-service template on both cores. AIC
// selects its existing dummy service, but nondependent helper names must still
// be declared while parsing that template. Do not instantiate any VF on AIC.
// Deliberately no definitions: accidental AIC use must fail, never fall back.
struct ColdMergeDeviceScratch {
    __aicore__ void Init(TPipe *pipe);
};
__aicore__ void MergeColdStateDevice(
    LocalTensor<float> &hotOutput, LocalTensor<float> &hotSum, LocalTensor<float> &hotMax,
    const __gm__ uint8_t *coldOutput, const __gm__ uint8_t *coldMax,
    const __gm__ uint8_t *coldSum, const __gm__ uint8_t *coldPresent,
    const __gm__ uint8_t *hotPresent, int64_t flatOffset, int64_t numHeads,
    int64_t rows, int64_t dim, int64_t ubStride, ColdMergeDeviceScratch &scratch);
#else

#if SFA_MERGE_BF16_IO
using ColdOutputType = bfloat16_t;
#else
using ColdOutputType = float;
#endif

struct ColdMergeDeviceScratch {
    static constexpr uint32_t ROWS = 4;
    static constexpr uint32_t DIM = 512;
    TBuf<> coldBuffer;
    TBuf<> statsBuffer;
    TEventID mte2ToVector;
    TEventID vectorToMte2;

    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(coldBuffer, ROWS * DIM * sizeof(ColdOutputType));
        pipe->InitBuffer(statsBuffer, 64); // Two separately padded 32-byte rows.
        mte2ToVector = pipe->AllocEventID<HardEvent::MTE2_V>();
        vectorToMte2 = pipe->AllocEventID<HardEvent::V_MTE2>();
    }
};

namespace ColdMergeDeviceDetail {
constexpr MicroAPI::CastTrait HOT_TO_BF16 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
constexpr MicroAPI::CastTrait BF16_TO_FLOAT = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <bool COLD_PRESENT, bool HOT_PRESENT>
__simd_vf__ inline void MergeRowsVF(
    __ubuf__ float *hotOutput, __ubuf__ float *hotSum, __ubuf__ float *hotMax,
    __ubuf__ ColdOutputType *coldOutput, __ubuf__ float *coldSum, __ubuf__ float *coldMax,
    const uint16_t rows, const uint16_t hotStride, const float nanValue)
{
    using namespace MicroAPI;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    MaskReg validMask, check;
    RegTensor<float> zero, one, nan, valid;
    RegTensor<float> hs, hm, cs, cm, common, hw, cw, denominator;
    RegTensor<float> hot, hotRaw, cold, result, coldTerm;
    RegTensor<bfloat16_t> hotBf16;
#if SFA_MERGE_BF16_IO
    RegTensor<bfloat16_t> coldBf16;
#endif
    Duplicate(zero, 0.0F);
    Duplicate(one, 1.0F);
    Duplicate(nan, nanValue);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t row = 0; row < rows; ++row) {
        Duplicate(valid, 1.0F);
        if constexpr (HOT_PRESENT) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(hs, hotSum + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(hm, hotMax + row);
            Compares<float, CMPMODE::GE>(check, hm, -3.402823466e38F, all);
            Select(valid, valid, zero, check);
            Compares<float, CMPMODE::LE>(check, hm, 3.402823466e38F, all);
            Select(valid, valid, zero, check);
            Compares<float, CMPMODE::GE>(check, hs, 0.9999F, all);
            Select(valid, valid, zero, check);
            Compares<float, CMPMODE::LE>(check, hs, 128.02F, all);
            Select(valid, valid, zero, check);
        }
        if constexpr (COLD_PRESENT) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(cs, coldSum + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(cm, coldMax + row);
            Compares<float, CMPMODE::GE>(check, cm, -3.402823466e38F, all);
            Select(valid, valid, zero, check);
            Compares<float, CMPMODE::LE>(check, cm, 3.402823466e38F, all);
            Select(valid, valid, zero, check);
            Compares<float, CMPMODE::GE>(check, cs, 0.9999F, all);
            Select(valid, valid, zero, check);
            Compares<float, CMPMODE::LE>(check, cs, 2048.25F, all);
            Select(valid, valid, zero, check);
        }
        if constexpr (COLD_PRESENT && HOT_PRESENT) {
            Max(common, hm, cm, all);
            ExpSub(hw, hm, common, all);
            ExpSub(cw, cm, common, all);
            Mul(hw, hw, hs, all);
            Mul(cw, cw, cs, all);
            Add(denominator, hw, cw, all);
            // Normalize BEFORE multiplying outputs. Valid bounded sums keep the
            // denominator finite and >=0.9999 even for extreme finite maxima.
            Div(hw, hw, denominator, all);
            Div(cw, cw, denominator, all);
        }
        Compares<float, CMPMODE::EQ>(validMask, valid, 1.0F, all);
        for (uint16_t column = 0; column < 512; column += 64) {
            if constexpr (HOT_PRESENT) {
                LoadAlign(hotRaw, hotOutput + row * hotStride + column);
                Cast<bfloat16_t, float, HOT_TO_BF16>(hotBf16, hotRaw, all);
                Cast<float, bfloat16_t, BF16_TO_FLOAT>(hot, hotBf16, all);
                // Do not let a saturating conversion hide an invalid input.
                // Preserve NaN/Inf even when this side's numerical weight is 0.
                Compares<float, CMPMODE::GE>(check, hotRaw, -3.402823466e38F, all);
                Select(hot, hot, hotRaw, check);
                Compares<float, CMPMODE::LE>(check, hotRaw, 3.402823466e38F, all);
                Select(hot, hot, hotRaw, check);
            }
            if constexpr (COLD_PRESENT) {
#if SFA_MERGE_BF16_IO
                // Native dependency vf_flashupdate_new.h uses this exact B16
                // unpack-load form. ZERO-layout casts consume its even lanes.
                LoadAlign<uint16_t, LoadDist::DIST_UNPACK_B16>(
                    (RegTensor<uint16_t> &)coldBf16,
                    (__ubuf__ uint16_t *)(coldOutput + row * 512 + column));
                Cast<float, bfloat16_t, BF16_TO_FLOAT>(cold, coldBf16, all);
#else
                LoadAlign(cold, coldOutput + row * 512 + column);
#endif
            }
            if constexpr (COLD_PRESENT && HOT_PRESENT) {
                Mul(result, hot, hw, all);
                Mul(coldTerm, cold, cw, all);
                Add(result, result, coldTerm, all);
            } else if constexpr (HOT_PRESENT) {
                // Arithmetic preserves NaN/Inf in a present output.
                Mul(result, hot, one, all);
            } else if constexpr (COLD_PRESENT) {
                Mul(result, cold, one, all);
            } else {
                Duplicate(result, 0.0F);
            }
            Select(result, result, nan, validMask);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(
                hotOutput + row * hotStride + column, result, all);
        }
    }
}

template <bool COLD_PRESENT, bool HOT_PRESENT>
__aicore__ inline void MergeMode(
    LocalTensor<float> &hotOutput, LocalTensor<float> &hotSum, LocalTensor<float> &hotMax,
    const __gm__ uint8_t *coldOutputAddress, const __gm__ uint8_t *coldMaxAddress,
    const __gm__ uint8_t *coldSumAddress, int64_t flatOffset, int64_t rows,
    int64_t ubStride, ColdMergeDeviceScratch &scratch)
{
    GlobalTensor<ColdOutputType> coldOutputGm;
    GlobalTensor<float> coldMaxGm, coldSumGm;
    if constexpr (COLD_PRESENT) {
        coldOutputGm.SetGlobalBuffer((__gm__ ColdOutputType *)coldOutputAddress);
        coldMaxGm.SetGlobalBuffer((__gm__ float *)coldMaxAddress);
        coldSumGm.SetGlobalBuffer((__gm__ float *)coldSumAddress);
    }
    LocalTensor<ColdOutputType> cold = scratch.coldBuffer.Get<ColdOutputType>();
    LocalTensor<float> stats = scratch.statsBuffer.Get<float>();
    union { uint32_t bits; float value; } nanBits;
    nanBits.bits = 0x7FC00000U;
    for (int64_t start = 0; start < rows; start += ColdMergeDeviceScratch::ROWS) {
        uint16_t count = static_cast<uint16_t>(
            rows - start < ColdMergeDeviceScratch::ROWS ? rows - start : ColdMergeDeviceScratch::ROWS);
        if constexpr (COLD_PRESENT) {
            // Fence the preceding VF before reusing its cold input scratch.
            // Every set has a matching wait: no pending event on empty paths.
            SetFlag<HardEvent::V_MTE2>(scratch.vectorToMte2);
            WaitFlag<HardEvent::V_MTE2>(scratch.vectorToMte2);
            DataCopyExtParams copy = {};
            copy.blockCount = 1;
            copy.blockLen = count * 512 * sizeof(ColdOutputType);
            DataCopyPadExtParams<ColdOutputType> coldPad = {};
            DataCopyPad(cold, coldOutputGm[(flatOffset + start) * 512], copy, coldPad);
            copy.blockLen = count * sizeof(float);
            DataCopyPadExtParams<float> pad = {};
            // count is only 1..4 (4..16 bytes). Plain DataCopy rounds down to
            // a 32-byte transfer, so it would silently copy no max/sum data.
            // Keep byte-length DataCopyPad for these bounded GM->UB tails.
            pad.isPad = true;
            pad.rightPadding = 8 - count;
            pad.paddingValue = 0.0F;
            DataCopyPad(stats, coldMaxGm[flatOffset + start], copy, pad);
            DataCopyPad(stats[8], coldSumGm[flatOffset + start], copy, pad);
            SetFlag<HardEvent::MTE2_V>(scratch.mte2ToVector);
            WaitFlag<HardEvent::MTE2_V>(scratch.mte2ToVector);
        }
        PipeBarrier<PIPE_V>();
        // No hot state address is dereferenced by the !HOT_PRESENT VF variant.
        __ubuf__ float *hotSumAddress = nullptr;
        __ubuf__ float *hotMaxAddress = nullptr;
        if constexpr (HOT_PRESENT) {
            hotSumAddress = (__ubuf__ float *)hotSum.GetPhyAddr() + start;
            hotMaxAddress = (__ubuf__ float *)hotMax.GetPhyAddr() + start;
        }
        MergeRowsVF<COLD_PRESENT, HOT_PRESENT>(
            (__ubuf__ float *)hotOutput.GetPhyAddr() + start * ubStride,
            hotSumAddress, hotMaxAddress,
            (__ubuf__ ColdOutputType *)cold.GetPhyAddr(),
            (__ubuf__ float *)stats.GetPhyAddr() + 8,
            (__ubuf__ float *)stats.GetPhyAddr(), count, static_cast<uint16_t>(ubStride), nanBits.value);
    }
    // Caller must retain the existing V->MTE3 dependency before the FP32 store.
    PipeBarrier<PIPE_V>();
}
} // namespace ColdMergeDeviceDetail

__aicore__ inline void MergeColdStateDevice(
    LocalTensor<float> &hotOutput, LocalTensor<float> &hotSum, LocalTensor<float> &hotMax,
    const __gm__ uint8_t *coldOutput, const __gm__ uint8_t *coldMax,
    const __gm__ uint8_t *coldSum, const __gm__ uint8_t *coldPresent,
    const __gm__ uint8_t *hotPresent, int64_t flatOffset, int64_t numHeads,
    int64_t rows, int64_t dim, int64_t ubStride, ColdMergeDeviceScratch &scratch)
{
    // Geometry is validated by the experimental caller, before device dispatch:
    // H=64,D=512,0<=rows<=64, ubStride>=512, one-token head tile, valid GM ranges.
    // This helper does not claim to repair malformed geometry or absent hot work
    // skipped upstream: the caller must dispatch such empty-token work itself.
    (void)dim;
    if (rows == 0) { return; }
    GlobalTensor<uint8_t> coldPresentGm, hotPresentGm;
    coldPresentGm.SetGlobalBuffer((__gm__ uint8_t *)coldPresent);
    hotPresentGm.SetGlobalBuffer((__gm__ uint8_t *)hotPresent);
    const int64_t token = flatOffset / numHeads;
    const bool cp = coldPresentGm.GetValue(token) != 0;
    const bool hp = hotPresentGm.GetValue(token) != 0;
    if (cp && hp) {
        ColdMergeDeviceDetail::MergeMode<true, true>(hotOutput, hotSum, hotMax,
            coldOutput, coldMax, coldSum, flatOffset, rows, ubStride, scratch);
    } else if (cp) {
        ColdMergeDeviceDetail::MergeMode<true, false>(hotOutput, hotSum, hotMax,
            coldOutput, coldMax, coldSum, flatOffset, rows, ubStride, scratch);
    } else if (hp) {
        ColdMergeDeviceDetail::MergeMode<false, true>(hotOutput, hotSum, hotMax,
            coldOutput, coldMax, coldSum, flatOffset, rows, ubStride, scratch);
    } else {
        ColdMergeDeviceDetail::MergeMode<false, false>(hotOutput, hotSum, hotMax,
            coldOutput, coldMax, coldSum, flatOffset, rows, ubStride, scratch);
    }
}
#endif // __DAV_C310_CUBE__: AIV implementation / AIC declarations only.
#if SFA_MERGE_BF16_IO
} // namespace Bf16ColdMerge
#endif
} // namespace AscendC
#endif
