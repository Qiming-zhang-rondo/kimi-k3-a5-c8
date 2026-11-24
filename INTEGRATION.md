# Integration Patch: Cold-State Merge into ProcessVec2

## File: `arch35/sparse_flash_attention_service_vector_mla.h`

### Insertion point: `ProcessVec2` (between L743 and L744)

**Context (existing code, L738-751):**
```cpp
    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0);
        }
        // ═══════════════════════════════════════════════════════════════
        // [BEGIN] MXLLM-COLD-MERGE-001: merge cold-end state here
        // ═══════════════════════════════════════════════════════════════
        if (constInfo.hasColdState) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            MergeColdStateTile<T, OUTPUT_T>(
                vec2ResUb, sumUb, maxUb,
                this->coldOutputGm, this->coldMaxGm, this->coldSumGm,
                this->coldPresentGm, this->hotPresentGm,
                runInfo.coldMergeRowOffset, runInfo.headIdx,
                constInfo.numHeads,
                runInfo.vec2MRealSize, dTemplateAlign64);
        }
        // ═══════════════════════════════════════════════════════════════
        // [END] MXLLM-COLD-MERGE-001
        // ═══════════════════════════════════════════════════════════════

        // 全无效行attention_out写0, 避免陈旧L1数据经BMM2污染输出
        LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
        ZeroInvalidAttnOut(vec2ResUb, maxUb, negativeFloatScalar * constInfo.softmaxScale,
            runInfo.vec2MRealSize, dTemplateAlign64, dTemplateAlign64);

        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
```

### New members to add to `SFAVectorService` class (near L170):
```cpp
    // [MXLLM-COLD-MERGE-001] Cold-state GM tensors for LSE merge
    GlobalTensor<OUTPUT_T> coldOutputGm;     // [T, H, D]
    GlobalTensor<float> coldMaxGm;           // [T, H]
    GlobalTensor<float> coldSumGm;            // [T, H]
    GlobalTensor<uint8_t> coldPresentGm;     // [T]
    GlobalTensor<uint8_t> hotPresentGm;      // [T]
```

### New `ConstInfo` fields (in `sparse_flash_attention_common.h`):
```cpp
    // [MXLLM-COLD-MERGE-001]
    bool hasColdState = false;   // if true, cold tensors are valid
    int64_t numHeads = 0;        // for stride calculation
```

### New `RunInfo` fields:
```cpp
    // [MXLLM-COLD-MERGE-001]
    int64_t coldMergeRowOffset = 0;  // global row offset for this tile
    int64_t headIdx = 0;             // current head index
```

### New `InitGlobalBuffer` overload (near L844):
```cpp
    // [MXLLM-COLD-MERGE-001] Initialize cold-state GM tensors
    __aicore__ inline void InitColdStateBuffers(
        __gm__ uint8_t *coldOutput, __gm__ uint8_t *coldMax,
        __gm__ uint8_t *coldSum, __gm__ uint8_t *coldPresent,
        __gm__ uint8_t *hotPresent)
    {
        this->coldOutputGm.SetGlobalBuffer(reinterpret_cast<__gm__ OUTPUT_T*>(coldOutput));
        this->coldMaxGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(coldMax));
        this->coldSumGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(coldSum));
        this->coldPresentGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(coldPresent));
        this->hotPresentGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(hotPresent));
    }
```

## File: `sparse_flash_attention.cpp` — New kernel parameters

The kernel entry needs 5 additional GM pointers:
```cpp
// Existing parameters (unchanged):
//   query, key, value, sparseIndices, blockTable, queryRope, keyRope,
//   attentionOut, softmaxMax, softmaxSum, workspace, tiling

// NEW parameters (append after softmaxSum):
    __gm__ uint8_t *coldOutput,      // [T, H, D] OUTPUT_T
    __gm__ uint8_t *coldMax,         // [T, H] float32
    __gm__ uint8_t *coldSum,         // [T, H] float32
    __gm__ uint8_t *coldPresent,     // [T] bool
    __gm__ uint8_t *hotPresent,      // [T] bool
```

These are passed through the kernel template to `SFAVectorService::InitColdStateBuffers`.

## File: `sparse_flash_attention_common.h` — ConstInfo extension

Add to `ConstInfo`:
```cpp
    bool hasColdState = false;
    int64_t numHeads = 0;
```

## Build instructions

### Prerequisites
- CANN 9.2.1 toolkit (ascend-toolkit)
- AscendC compiler (`ccec`)
- Target: Ascend950DT (arch35)

### Build steps (standard CANN custom operator build)
```bash
# 1. Set environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. The operator builds as part of the custom_cot package.
#    Modified files replace the originals in:
#    custom_cot/op_impl/ai_core/tbe/custom_cot_impl/ascendc/sparse_flash_attention/

# 3. Rebuild the operator .so using the CANN build system:
#    (This step requires the full CANN operator build pipeline, which is
#     NOT available in the experiment container. The files in this repo
#     are drop-in replacements for the source tree.)

# Alternative: standalone kernel direct-invoke (for correctness testing only):
#    Use msopgen to create a test project and compile cold_state_merge.h
#    as a standalone kernel.
```

## Verification plan

1. **Precision gate**: merged output vs BF16 math reference (same formula as `kv_native_merge.py`)
2. **Empty-side gate**: cold-only, hot-only, both-empty produce correct results
3. **Stale-buffer gate**: change cold state between runs, verify no stale reads
4. **Performance**: 10-round AB/BA paired timing vs two-pass champion (126.17 µs reference)
5. **Profiler**: confirm external Triton merge kernel disappears

## Numerical contract

The hot output in `vec2ResUb` at the merge point is the NORMALIZED output `O_h = A_h / s_h`.
The merge reconstitutes the numerator and combines:
```
O = (O_h * s_h * w_h + O_c * s_c * w_c) / (s_h * w_h + s_c * w_c)
```
This is algebraically identical to merging unnormalized accumulators:
```
O = (A_h * w_h + A_c * w_c) / (s_h * w_h + s_c * w_c)
```
where `A_h = O_h * s_h` and `A_c = O_c * s_c`.

## Limitations (first version)

1. **Scalar per-row merge** — correct but slow. Vectorized MicroAPI version is the optimization step.
2. **GM scalar loads** — coldMax/coldSum loaded per element via GM. Should be DataCopy'd to UB as a batch.
3. **No dual-buffer overlap** — cold output loaded one row at a time. Should tile the load.
4. **Requires host-side integration** — the ACLNN API layer and tiling logic need new parameter plumbing.
