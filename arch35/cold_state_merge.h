// Scalar contract reference for merging a normalized hot state with a cold state.
//
// REVIEW STATUS: host C++ tests exercise this header directly. This is NOT a
// compiled/integrated AscendC operator or a performance implementation. The
// scalar math calls still require verification/replacement in the device
// toolchain. Native ABI, tiling, vector loads and final FP32 store are pending.
//
// Restricted layout: contiguous TND output [T,H,D], N2=1. A local M row advances
// the flattened (token,head) index, NOT the token index. The caller derives
// flatRowOffset from the real attention output offset and includes AIV/G splits.
//
// Hot and cold outputs are normalized. Statistics are FP32 in the same natural
// exponential domain: logZ = maximum + log(sum), with finite maximum and finite
// positive sum on every present side. Nonpresent sides are never inspected.
// T is the accumulator/final output element type; COLD_T independently describes
// cold GM bytes. HOT_OUTPUT_T describes the original hot reader output rounding.
// The caller must verify its conversion matches the original native Cast round
// mode; replacing the final GM store with FP32 requires separate integration.
// This experiment calls the helper with T=float and an independent cold type.

#pragma once

#include <cmath>
#include <cstdint>
#include "sparse_flash_attention_common_arch35.h"

namespace AscendC {

__aicore__ inline bool ValidColdMergeStatistics(float maximum, float sum)
{
    return std::isfinite(maximum) && std::isfinite(sum) && sum > 0.0f;
}

template <typename T, typename COLD_T, typename HOT_OUTPUT_T = T>
__aicore__ inline void MergeColdStateTile(
    LocalTensor<T> &hotOutputUb,
    LocalTensor<float> &hotSumUb,
    LocalTensor<float> &hotMaxUb,
    __gm__ const uint8_t *coldOutputGm,
    __gm__ const uint8_t *coldMaxGm,
    __gm__ const uint8_t *coldSumGm,
    __gm__ const uint8_t *coldPresentGm,
    __gm__ const uint8_t *hotPresentGm,
    int64_t flatRowOffset,
    int64_t numHeads,
    int64_t mRealSize,
    int64_t outputDim,
    int64_t ubRowStride)
{
    // Layout/range/alignment and non-overlap validation belongs in the native
    // host/tiling adapter. Supported dimensions require 0 < D <= UB stride.
    const auto *coldOutput = reinterpret_cast<__gm__ const COLD_T *>(coldOutputGm);
    const auto *coldMax = reinterpret_cast<__gm__ const float *>(coldMaxGm);
    const auto *coldSum = reinterpret_cast<__gm__ const float *>(coldSumGm);
    for (int64_t m = 0; m < mRealSize; ++m) {
        const int64_t statIdx = flatRowOffset + m;
        const int64_t tokenIdx = statIdx / numHeads;
        const bool coldPresent = coldPresentGm[tokenIdx] != 0;
        const bool hotPresent = hotPresentGm[tokenIdx] != 0;

        // Zero UB padding without reading unrelated cold GM elements.
        for (int64_t d = outputDim; d < ubRowStride; ++d) {
            hotOutputUb.SetValue(m * ubRowStride + d, static_cast<T>(0));
        }
        if (!coldPresent && !hotPresent) {
            for (int64_t d = 0; d < outputDim; ++d) {
                hotOutputUb.SetValue(m * ubRowStride + d, static_cast<T>(0));
            }
            continue;
        }

        // Validate the effective side(s) even in the one-sided cases. Invalid
        // present state is an observable error, not an empty segment.
        float hm = 0.0f, hs = 1.0f, cm = 0.0f, cs = 1.0f;
        bool valid = true;
        if (hotPresent) {
            hm = hotMaxUb.GetValue(m);
            hs = hotSumUb.GetValue(m);
            valid = ValidColdMergeStatistics(hm, hs);
        }
        if (coldPresent) {
            cm = coldMax[statIdx];
            cs = coldSum[statIdx];
            valid = valid && ValidColdMergeStatistics(cm, cs);
        }
        if (!valid) {
            const float invalid = std::nanf("");
            for (int64_t d = 0; d < outputDim; ++d) {
                hotOutputUb.SetValue(m * ubRowStride + d, static_cast<T>(invalid));
            }
            continue;
        }

        float hotWeight = hotPresent ? 1.0f : 0.0f;
        float coldWeight = coldPresent ? 1.0f : 0.0f;
        if (hotPresent && coldPresent) {
            // Shift before log(sum), then normalize log weights. This also
            // handles finite huge sums without overflowing hs + cs.
            const float common = hm > cm ? hm : cm;
            const float hotLog = (hm - common) + std::log(hs);
            const float coldLog = (cm - common) + std::log(cs);
            const float shift = hotLog > coldLog ? hotLog : coldLog;
            hotWeight = std::exp(hotLog - shift);
            coldWeight = std::exp(coldLog - shift);
            const float denominator = hotWeight + coldWeight;
            hotWeight /= denominator;
            coldWeight /= denominator;
        }
        for (int64_t d = 0; d < outputDim; ++d) {
            float value;
            if (hotPresent && coldPresent) {
                const float hot = static_cast<float>(static_cast<HOT_OUTPUT_T>(
                    hotOutputUb.GetValue(m * ubRowStride + d)));
                const float cold = static_cast<float>(coldOutput[statIdx * outputDim + d]);
                value = hot * hotWeight + cold * coldWeight;
            } else if (hotPresent) {
                value = static_cast<float>(static_cast<HOT_OUTPUT_T>(
                    hotOutputUb.GetValue(m * ubRowStride + d)));
            } else {
                value = static_cast<float>(coldOutput[statIdx * outputDim + d]);
            }
            hotOutputUb.SetValue(m * ubRowStride + d, static_cast<T>(value));
        }
    }
}

} // namespace AscendC

