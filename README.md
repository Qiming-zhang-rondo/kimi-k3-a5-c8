# SFA cold-state merge: reviewed source contract

**内网代理执行入口：[START_HERE.md](START_HERE.md)。**
按其中的完整任务自主完成依赖、实现、构建和设备验证，将结果直接提交此分支。

This branch contains an original packaged kernel snapshot, a corrected scalar
merge contract helper, and integration notes. It is **not yet a buildable or
callable fused attention operator**.

## Current status

- Original `original/` snapshot preserved.
- Corrected helper uses flattened token/head row mapping and independent cold
  element type; it distinguishes cold GM D from the local UB row stride.
- Present invalid max/sum produces a visible nonfinite result, including
  one-sided cases. Empty sides are ignored.
- Shifted log weights avoid overflow when two positive finite sums are large.
- A local C++ compiler directly builds and executes the actual helper header
  with minimal host tensor stubs. This checks C++ scalar logic, not AscendC.
- Native kernel call-site wiring, host schema/tiling, ACLNN/Torch binding,
  device compilation and A5 performance are **not implemented or validated**.

The uploaded Python checks only run Torch formulas. They now fail with a
nonzero exit code on an assertion failure and identify their scope explicitly.

## Files

```
arch35/cold_state_merge.h                 corrected scalar contract helper
original/                               unmodified packaged source snapshot
tests/test_uploaded_merge_regressions.py actual-header host C++ regression tests
tests/cpp_stubs/                         minimal host-only LocalTensor stub
tests/verify_merge.py                    Torch formula checks only
tests/verify_merge_t3_fixed.py           Torch formula checks only
INTEGRATION.md                          verified mapping and remaining work
```

Run the host-only regression tests (Python standard library + a C++ compiler):

```bash
python3 -m unittest discover -s tests -p 'test_uploaded_merge_regressions.py' -v
```

No torch, AscendC SDK or NPU is used by these host tests. Passing them does not
establish device intrinsic support, rounding equivalence, UB/synchronization,
native ABI integration or hardware speed.

## Actual measured baseline, supplied by the A5 experiment

B1/Q4/K8192/H64/D512/RoPE64/TopK2048/W128, fixed synthetic selection with
128 hot hits per query:

| Path | Full pipeline us | Paired W0 ratio | Relative L2 vs BF16 |
|---|---:|---:|---:|
| Native C4 W0 | 87.53 | — | 0.10410 |
| BF16 hot + existing Triton merge | 126.17 | 1.438 | 0.099006 |
| C8 hot + existing Triton merge | 128.57 | 1.466 | 0.099081 |
| New AscendC fused operator | **not measured** | — | — |

These are prior baseline measurements, not results from this branch.
The current target is to preserve the native BF16 hot reader and merge cold
state at its final epilogue. The standalone merge's measured stage duration
cannot simply be subtracted from full-pipeline time. No 87us target is promised.

## Provenance

Reviewed uploaded branch head: `7e1f007fab89a77f8e6ef6728a3cf91e36c5dba4`.
The upload does not establish which exact deployed Cloud/CANN binary the
snapshot matches. Confirm that from the working container before rebuilding.

