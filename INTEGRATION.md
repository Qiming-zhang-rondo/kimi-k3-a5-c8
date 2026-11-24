# Verified integration contract and remaining blockers

This is an implementation guide, not an applied patch. No source file in
`original/` currently includes or calls `MergeColdStateTile`.

## 1. Where the source actually normalizes and stores output

The supplied arch35 kernel documents no S2 core splitting
(`original/arch35/sparse_flash_attention_kernel_mla.h:211`). Each task loops
over its S2 tiles, and the last `ProcessVec2` iteration performs
`FlashUpdateLastNew` or `LastDivNew` before `CopyOutAttentionOut`
(`original/arch35/sparse_flash_attention_service_vector_mla.h:705-750`).

For this supplied path, the merge belongs after hot normalization and after
cleaning invalid HOT output, before the final combined output store.
It must not run once per intermediate S2 tile.

The old proposed placement before `ZeroInvalidAttnOut` is unsafe: hot-only
invalid cleanup may erase a valid cold-only combined output. Statistics-on
can modify the hot sentinel in `CopyFALseToGm`; correctness must not depend on
that flag. A fully empty hot task may also be skipped by
`kernel_mla.h:581-583`, so implementing cold-only requires handling that path,
not just adding a call in the normal epilogue.

## 2. Correct local-row to GM mapping

The current restricted experiment uses TND, one KV head (N2=1) and contiguous
cold `[T,H,D]` output with `[T,H,1]` statistics and `[T]` presence masks.

The local M axis combines query and head. It does not enumerate tokens at a
fixed head. `kvcache.h:139-151` computes M from S1 times G, and the two AIV
subblocks divide those head rows. Q4/H64 normally means 32 head rows of ONE
query in each AIV half.

For the verified restricted TND/N2=1 mapping:

```text
flatRowOffset = runInfo.attentionOutOffset / constInfo.dSizeV
flat = flatRowOffset + m
query = flat / numHeads
head = flat % numHeads
cold output element = flat * outputDim + d
cold max/sum element = flat
presence element = query
hot local output element = m * ubRowStride + d
hot local max/sum element = m
```

The actual attention offset already includes batch/query, G-split and AIV
half offsets (`kvcache.h:179-186`). Verify this mapping in the selected tiling
branch. Reject unsupported layouts/N2/strides in the host adapter rather than
silently generalizing it.

Output D is 512 in the supplied vector service. 576 is the input latent plus
RoPE dimension, not the output width. GM logical output D and UB stride are
separate parameters in the corrected helper.

## 3. Helper interface and dtype

The corrected helper receives raw read-only GM byte pointers and explicit
flattened row/stride metadata:

```cpp
template <typename T, typename COLD_T, typename HOT_OUTPUT_T = T>
void MergeColdStateTile(
    LocalTensor<T>& hotOutputUb,
    LocalTensor<float>& hotSumUb,
    LocalTensor<float>& hotMaxUb,
    __gm__ const uint8_t* coldOutput,
    __gm__ const uint8_t* coldMax,
    __gm__ const uint8_t* coldSum,
    __gm__ const uint8_t* coldPresent,
    __gm__ const uint8_t* hotPresent,
    int64_t flatRowOffset, int64_t numHeads,
    int64_t mRealSize, int64_t outputDim, int64_t ubRowStride);
```

This is a scalar contract helper; its std math and scalar loads require device
implementation/compilation review. A host C++ compile is not an AscendC compile.
A high-performance version should stage GM loads and use vector operations.

`COLD_T` describes the actual cold GM bytes, independently of hot dtype.
The existing C4 reader may return FP32; interpreting those bytes as BF16 is
incorrect. The current native hot path has FP32 local accumulators and BF16
output. `HOT_OUTPUT_T` makes its original output rounding explicit; device
conversion must be checked against the original `Cast(...CAST_ROUND...)`.

The supplied final `Bmm2DataCopyOut` casts to its original `OUTPUT_T` and
sizes strides using that type. A fused FP32 output requires new final store
and output-schema/dispatch support. Passing FP32 cold bytes into the existing
BF16 output template does not provide that support. Changing only the output
macro can also select the wrong input branch in `sparse_flash_attention.cpp`.

Existing `GlobalTensor` members are not raw byte pointers. Use a verified
physical-address API or pass/retain the raw kernel parameters; do not copy the
old guide's type-incompatible call literally.

## 4. Mathematical and error contract

Both outputs are normalized; sums are unnormalized softmax denominators in
the same exponent domain. Verify the real hot max/sum domain from its VF
implementation rather than assuming it from names.

For each present side, maximum must be finite and sum finite and positive.
Any invalid PRESENT state yields a visible nonfinite output, including
one-sided cases. An absent state is ignored even if its buffers contain NaN.
Both absent produces zero. Nonfinite values on a present output remain
visible. Cold inputs are read-only and must not overlap final output.

Use stable normalized log weights; the scalar helper first shifts maxima,
then adds log(sum), then shifts those log weights. Do not clear invalid
positive-presence states to zero.

Original two-pass merge rounds hot output before combining it with cold
output. First validate that exact contract; a higher precision hot-accumulator
variant needs separate accuracy evidence.

## 5. Native integration still required

The upload has no applied modifications to kernel parameters, Init, vector
service call sites or the host API. Implement an experimental native entry
with the original query/KV inputs plus five cold/presence pointers and a
correctly typed final output. Complete schema, ACLNN, Torch, tiling and all
selected AIC/AIV dispatch paths.

Do not add fields to the wrong structure: the supplied
`original/sparse_flash_attention_common.h` defines older fields, while arch35
uses `RunInfo`/`ConstInfo` members from dependencies such as `util_regbase.h`.
Those actual definitions and shared VF/buffer headers were not uploaded.

Before native code can be built, obtain the matching dependencies actually
included by the source, including:

- `util_regbase.h` and the actual shared RunInfo/ConstInfo definitions;
- `vf_mul_sel_softmaxflashv2_cast_nz_sfa.h`, `vf_flashupdate_new.h`;
- shared buffer, buffer-manager, policy and tiling headers;
- host op definition/schema and tiling sources, ACLNN and Torch binding;
- real build configuration, commands and current package/toolchain version.

No executable rebuild command or complete build tree was supplied. Do not
guess a CANN upgrade or claim the files are drop-in replacements. Start from
the same package/toolchain combination that passed the existing BF16 A/B.

## 6. Validation and acceptance

Host regression tests compile and invoke this helper with tensor stubs and
check flattened mapping, independent dtype/stride, presence, invalid stats,
normalization and finite extreme sums. They are not native runtime tests.
The two Torch scripts check formulas only; assertion failures now exit
nonzero.

Once the native entry builds, check actual B1/Q4/K8192/H64/D512/R64/TopK2048/W128:

1. Original byte-writer and partition controls.
2. Native hot output/LSE against independent math; zero-query counts 128 hot
   and 1920 cold.
3. Fused output against both current two-pass output and matching math.
4. Cold-only, hot-only, both-empty, invalid present statistics, stats-on/off,
   changed input buffers, and no cold-input mutation.
5. Same-process idle-NPU AB/BA, warmup20, iterations100, 10 paired rounds,
   comparing both the existing BF16 pipeline and native C4 W0.
6. Profiler proves the external merge disappears without a hidden replacement
   launch, and native memory/cross-core dependencies remain correct.

Keep full pipeline timing, raw per-round event/wall pairs and actual output
dtype. Do not add standalone stage medians to estimate pipeline savings.

