# Experimental native hot-read / cold-state merge wiring

This is an uncompiled source candidate. No AscendC build, device correctness,
UB allocation, or performance result is claimed by the source wiring or host
logic tests. `original/` is unchanged.

## Implemented source path

`experimental/sparse_flash_attention_cold_merge.cpp` registers the private
`SparseFlashAttentionColdMerge` template identity and dispatches the global
`sparse_flash_attention_cold_merge` entry. Both AIC and AIV macro paths pass the
five extra GM pointers to the experimental kernel's `Init`. That method installs
them in the vector service; the AIC dummy has a no-op matching method. No field
has been added to `SparseFlashAttentionTilingDataMla`, `CVSharedParams`,
`RunInfo`, or `ConstInfo`.

In `ProcessVec2`, native hot normalization and `ZeroInvalidAttnOut` run first.
`MergeColdStateDevice` then combines the normalized hot output with the supplied
cold state, before final `DataCopyPad`. The helper performs the BF16 rounding
of the hot output needed to match the existing native BF16 reader contract.
The final output is FP32: the old in-place BF16 cast has been removed. Q/K/V
remain BF16 even though the output is FP32; native's half-type fallback is not
used.

The device helper is `experimental/arch35/cold_state_merge_device.h`, not the
host-reference `arch35/cold_state_merge.h`. Its bounded scratch is allocated
from the same vector `TPipe` in `InitLocalBuffer`. The added scratch and event
allocations must pass the real compiler's resource check; arithmetic estimates
are not proof of UB fit.

## GM argument ABI

The kernel entry receives, in order:

1. `query, key, value, sparseIndices, blocktable`
2. `actualSeqLengthsQuery, actualSeqLengthsKV, queryRope, keyRope`
3. `coldOutput, coldMax, coldSum, coldPresent, hotPresent`
4. `attentionOut, softmaxMax, softmaxSum, workspace, tiling`

The internal `op.Init` has a different pre-existing ordering and appends cold
state after `softmaxSum`, before workspace. These are deliberate distinct ABIs;
a launcher must use the global-entry order above.

| Tensor | Physical type and contiguous shape |
| --- | --- |
| Cold normalized output | FP32 `[T, 64, 512]` |
| Cold maximum and exponential sum | FP32 `[T, 64, 1]` (flattened row storage) |
| Cold/hot presence | one-byte bool or uint8 `[T]`; one value per query, not per head |
| Fused output | FP32 `[T, 64, 512]` |
| Hot max/sum UB | FP32 contiguous flattened rows; `FaVectorApi::REDUCE_SIZE == 1` is checked |

For each vector tile, `flatRowOffset = attentionOutOffset / dSizeV`; the helper
uses `flatRowOffset + m` for cold state and divides by 64 for presence. The
fixed test's tile must stay within one query's head group. Do not reinterpret
`m` as a query index or use padded UB stride to address compact cold GM rows.

The compilation adapter must define actual schema-derived dtype macros:
`ORIG_DTYPE_QUERY`, `KEY`, `VALUE`, `ATTENTION_OUT`, `COLD_OUTPUT`, `COLD_MAX`,
`COLD_SUM`, `COLD_PRESENT`, and `HOT_PRESENT` (all prefixed `ORIG_DTYPE_`). Their
compile-time checks deliberately reject an incompatible binding instead of
silently reinterpreting bytes.

## Fixed first-device-run contract

The integration runner/host layer must enforce these conditions before launch:

- A5/arch35; BF16 Q, hot latent K/V, and separate BF16 RoPE. FP32 cold state.
- The agreed single case: B=1, Q=4, H=64, N2=1, D=512, RoPE=64,
  original KV length 8192, selected TopK=2048, recent window=128.
- TND query and PA_BSND hot KV, sparse mode 0, attention mode 2, no sinks;
  retain the native softmax scale unchanged on both sides.
- Every query has exactly 128 valid hot indices, compact physical capacity
  128, with no `-1` padding, and hot presence true. Cold presence, when true,
  refers to the cold reader's actual selected subset and matching max/sum.
- `return_softmax_lse=false`. The retained native Vec1 export, if enabled,
  exports hot-only statistics, not merged statistics. It is not a supported
  output of this candidate.
- Reject empty-hot queries and any native scheduling path that skips Vec2.
  The helper can test empty-state algebra, but that does not establish that
  native attention schedules a merge for an empty hot side.

The private template key keeps the native parameter order and bit widths.
It registers only FLASH_DECODE=0, PAGE_ATTENTION=1, TND/PA_BSND; native C/V
modes and IS_SPLIT_G values retain their encoding. Use real native tiling for
the same hot call. Do not fabricate tiling data or infer a layout from the
unrelated SparseFlashMla binding. The candidate does not support a generalized
variable-count serving workload yet.

## Include and build isolation

Compile the experimental entry, not `original/sparse_flash_attention.cpp`.
The experimental kernel includes its experimental vector sibling first;
shared cube/cache/common headers remain original. Required project include
roots, before SDK paths, are:

- `experimental/` and `experimental/arch35/`
- `original/` and `original/arch35/`
- `deps/sparse_flash_attention/arch35/`
- `deps/ascendc_common/`
- the matching generated tiling headers and locally collected source closure

Do not place `tests/cpp_stubs` on a real device build's include path. Do not
link this candidate under the original installed operator identity or replace
its working vendor package. Host schema, native tiling integration, ACLNN or
direct launch, Torch binding, and actual AscendC build must agree with this ABI
before the candidate is runnable.
