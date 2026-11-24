# Pinned public SFA tiling sources

These are exact source extracts from the projects that maintain them, retrieved
2026-11-24. They are kept outside `original/`, `deps/`, and `experimental/`.
Nothing here has been injected into the installed/private Cloud operator ABI.
The recorded commits, original paths, URLs, Git blob hashes, byte sizes, and
SHA256 checksums are in `SOURCE_MANIFEST.json`. All 10 saved files match their
upstream Git blob hashes exactly, including original copyright notices.

## Sources

- vLLM Ascend commit `f47787297410372e8898cb268fcfb9e80681ff4e`:
  [host tiling definition](https://github.com/vllm-project/vllm-ascend/blob/f47787297410372e8898cb268fcfb9e80681ff4e/csrc/attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.h),
  [actual host tiling implementation](https://github.com/vllm-project/vllm-ascend/blob/f47787297410372e8898cb268fcfb9e80681ff4e/csrc/attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp),
  [arch35 kernel](https://github.com/vllm-project/vllm-ascend/blob/f47787297410372e8898cb268fcfb9e80681ff4e/csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_kernel_mla.h),
  host CMake and repository LICENSE.
- SGL Kernel NPU commit `597250fcc1b31f2607bf817d42bd0699ea60b8eb`:
  [host macro definition](https://github.com/sgl-project/sgl-kernel-npu/blob/597250fcc1b31f2607bf817d42bd0699ea60b8eb/csrc/sparse_flash_attention/op_host/tiling/sparse_flash_attention_tiling.h),
  [raw device POD](https://github.com/sgl-project/sgl-kernel-npu/blob/597250fcc1b31f2607bf817d42bd0699ea60b8eb/csrc/sparse_flash_attention/op_kernel/sparse_flash_attention_tiling_data.h),
  [arch35 kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/597250fcc1b31f2607bf817d42bd0699ea60b8eb/csrc/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_kernel_mla.h),
  repository LICENSE, and that repository's complete CANN Open Software
  License Agreement Version 2.0 text at its original
  `python/sgl_kernel_npu/sgl_kernel_npu/implementation/cannbot-dsl/LICENSE.cannbot-dsl`
  path. The retained per-file CANN notices are not replaced by repository-level
  MIT/Apache license files.

This is a small source evidence bundle, not a complete standalone build tree.

## Exact comparison performed

`FIELD_COMPARISON.json` records ordered `(type, field)` tuples and hashes. All
six SGL raw POD definitions match the six host macro definitions from the same
project and same commit. This is source comparison, not a claim that the private
Cloud package uses this layout.

The unmodified SGL device POD compiles on the host with `/usr/bin/c++ -std=c++17`.
It reports total struct size 144 bytes, `singleCoreParams` offset 120, and
`baseParams.dispatchKey` offset 100. Its own static assertions also pass.
These are host layout checks only; they do not replace the SDK-generated header
or establish installed-kernel ABI compatibility.

The uploaded private arch35 vector source reads 13 named base fields and
`singleCoreParams.usedCoreNum`; the complete list is in the comparison JSON.
The similarly named public definitions are **not interchangeable evidence**:

| Source | Last base field | Split-KV suffix |
| --- | --- | --- |
| vLLM pinned above | `ropeHeadDim` | three uint32 fields |
| SGL pinned above | `dispatchKey` | three uint32 fields |
| hicann/ops-transformer `e32b1329d9e4e22bc4b478ada2b70951b08efbf8` (inspected, not bundled) | `keyStride0` | adds uint64 `stagingOffset` |

The first two preserve the same source field prefix while assigning different
meaning to the last base field. The third changes subsequent member offsets.
Therefore neither a matching type name nor successful compilation proves that
an installed package's serialized tiling bytes match any of them.

The pinned vLLM and SGL main arch35 headers are about 96.8% and 96.7% identical
to the uploaded original after whitespace normalization (SequenceMatcher, not
semantic equivalence). Both have real differences: a sparse-index GM member,
valid-prefix binary-search handling before task scheduling, and AIV `n2Size`
initialization. They are related implementations, not identical package sources.

## Concrete usable next paths

1. **Compile-only recovery:** an explicitly selected, logged include of the
   pinned SGL raw POD can supply real source definitions to an isolated compiler
   experiment. Label its output as compiled against the SGL schema. Do not run
   that kernel with private package tiling bytes or call that experiment an
   unmodified native-package build. This avoids inventing a struct merely to
   progress past an unknown type error.
2. **Private-package execution:** repair the SDK/platform initialization probe,
   obtain the real generated header or matching package host definition, and
   preserve its source/hash evidence. Compile the baseline and candidate against
   that exact definition. The public files here are comparison references.
3. **Independent public-source build:** use the pinned vLLM operator source,
   real host tiling implementation, and its build system as a matched pair; first
   establish its native baseline on the fixed test, then apply the candidate
   epilogue in that isolated build. Obtain the rest of its source closure at the
   same commit. Do not combine vLLM's host serializer with SGL's dispatcher or
   silently link either against private `f6e7b882` tiling.

The SDK `set_soc/get_platform_info` crash happens before kernel compilation;
these headers do not claim to fix that initialization fault. They demonstrate
that an actual public tiling definition exists and that source recovery need not
be blocked on guessing the private binary layout.
