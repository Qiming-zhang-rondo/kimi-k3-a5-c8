// ============================================================================
// cold_state_merge.h — LSE-based cold state merge for BF16 hot-end attention
//
// Inserts into SFAVectorService::ProcessVec2 between the final normalization
// (FlashUpdateLastNew / LastDivNew) and the output copy-out (CopyOutAttentionOut).
//
// Math contract (O_h and O_c are ALREADY normalized by their respective sums):
//   m = max(m_h, m_c)
//   w_h = exp(m_h - m)   (stable, never direct exp(m))
//   w_c = exp(m_c - m)
//   Z   = s_h * w_h + s_c * w_c
//   O   = (O_h * s_h * w_h + O_c * s_c * w_c) / Z
//
// Edge cases:
//   hot_present && !cold_present  -> O = O_h (already in vec2ResUb, no-op)
//   !hot_present && cold_present  -> O = O_c
//   !hot_present && !cold_present -> O = 0
//   Non-finite m/s on a valid side -> propagate as error (do NOT mask).
//
// Presence tensors are device-side bool (1 byte per element, treated as logical
// boolean; never bitwise-NOT a uint8).
//
// STATUS: first version, scalar per-row implementation. Vectorized (MicroAPI)
// optimization is the next step after correctness verification.
// ============================================================================

#pragma once

#include "sparse_flash_attention_common_arch35.h"

namespace AscendC {

// ---------------------------------------------------------------------------
// Stable merge weight: exp(own - global). Never overflows because
// own <= global, so the exponent is always <= 0.
// ---------------------------------------------------------------------------
__aicore__ inline float StableMergeWeight(float ownMax, float globalMax)
{
    return exp(ownMax - globalMax);
}

// ---------------------------------------------------------------------------
// Tile-level merge: merges cold state into the hot output tile.
// Called once per output tile in ProcessVec2, after normalization, before
// ZeroInvalidAttnOut and CopyOutAttentionOut.
//
// The hot output in vec2ResUb is the NORMALIZED output O_h = A_h / s_h.
// The merge therefore re-multiplies by s_h to reconstruct the numerator,
// combines with the cold numerator, and re-normalizes.
//
// GM layout of cold tensors:
//   coldOutput: [T, H, D] contiguous, dtype = OUTPUT_T (bf16)
//   coldMax:    [T, H]   contiguous, float32
//   coldSum:    [T, H]   contiguous, float32
//   coldPresent: [T]     contiguous, bool (1 byte)
//   hotPresent:  [T]     contiguous, bool (1 byte)
//
// This tile covers rows [tileRowOffset, tileRowOffset + mRealSize) for
// head `headIdx`.
// ---------------------------------------------------------------------------
template <typename T, typename OUTPUT_T>
__aicore__ inline void MergeColdStateTile(
    LocalTensor<T> &vec2ResUb,           // [mRealSize, dAlign] normalized hot output (in/out)
    LocalTensor<float> &sumUb,           // [mRealSize] hot sum (per row)
    LocalTensor<float> &maxUb,           // [mRealSize] hot max (per row)
    __gm__ uint8_t *coldOutputGm,        // [T, H, D] cold output
    __gm__ uint8_t *coldMaxGm,           // [T, H] float32 cold max
    __gm__ uint8_t *coldSumGm,           // [T, H] float32 cold sum
    __gm__ uint8_t *coldPresentGm,       // [T] bool
    __gm__ uint8_t *hotPresentGm,        // [T] bool
    int64_t tileRowOffset,               // global row index of first row in this tile
    int64_t headIdx,                     // current head index
    int64_t numHeads,                    // total heads
    int64_t mRealSize,                   // rows in this tile
    int64_t dAlign)                     // aligned D (e.g., 576)
{
    // Pre-compute GM strides
    int64_t statIdxBase = tileRowOffset * numHeads + headIdx;
    int64_t statStride = numHeads;  // stride between consecutive rows for same head
    int64_t coldOutRowStride = dAlign * sizeof(OUTPUT_T);  // bytes per (T,H) row

    for (int64_t m = 0; m < mRealSize; ++m) {
        int64_t globalRow = tileRowOffset + m;
        int64_t statIdx = globalRow * numHeads + headIdx;

        // Read presence flags (logical bool, not bitwise)
        bool coldPresent = coldPresentGm[globalRow] != 0;
        bool hotPresent = hotPresentGm[globalRow] != 0;

        // Case 1: hot only — vec2ResUb already has the correct value
        if (hotPresent && !coldPresent) {
            continue;
        }

        // Case 2: both empty — zero the row
        if (!hotPresent && !coldPresent) {
            for (int64_t d = 0; d < dAlign; ++d) {
                vec2ResUb.SetValue(m * dAlign + d, static_cast<T>(0));
            }
            continue;
        }

        // Load cold statistics
        float coldSum = *reinterpret_cast<__gm__ float*>(
            coldSumGm + statIdx * sizeof(float));
        float coldMax = *reinterpret_cast<__gm__ float*>(
            coldMaxGm + statIdx * sizeof(float));

        // Case 3: cold only — copy cold output
        if (!hotPresent && coldPresent) {
            __gm__ OUTPUT_T *coldOutRow = reinterpret_cast<__gm__ OUTPUT_T*>(
                coldOutputGm + statIdx * coldOutRowStride);
            for (int64_t d = 0; d < dAlign; ++d) {
                vec2ResUb.SetValue(m * dAlign + d,
                    static_cast<T>(coldOutRow[d]));
            }
            continue;
        }

        // Case 4: both valid — full LSE merge
        float hotSum = sumUb.GetValue(m);
        float hotMax = maxUb.GetValue(m);

        // Combined max and stable weights
        float combinedMax = hotMax > coldMax ? hotMax : coldMax;
        float wH = StableMergeWeight(hotMax, combinedMax);  // exp(m_h - m) <= 1
        float wC = StableMergeWeight(coldMax, combinedMax); // exp(m_c - m) <= 1

        // Combined normalizer
        float z = hotSum * wH + coldSum * wC;
        if (z <= 0.0f) {
            // Degenerate: zero total mass
            for (int64_t d = 0; d < dAlign; ++d) {
                vec2ResUb.SetValue(m * dAlign + d, static_cast<T>(0));
            }
            continue;
        }

        // Pre-computed merge coefficients
        float zh = hotSum * wH / z;
        float zc = coldSum * wC / z;

        // Load cold output row and merge with hot output row
        __gm__ OUTPUT_T *coldOutRow = reinterpret_cast<__gm__ OUTPUT_T*>(
            coldOutputGm + statIdx * coldOutRowStride);

        // Scalar merge (vectorized TODO after correctness gate)
        for (int64_t d = 0; d < dAlign; ++d) {
            float oh = static_cast<float>(vec2ResUb.GetValue(m * dAlign + d));
            float oc = static_cast<float>(coldOutRow[d]);
            vec2ResUb.SetValue(m * dAlign + d,
                static_cast<T>(oh * zh + oc * zc));
        }
    }
}

} // namespace AscendC
