# VA 0.30 QFM framework experiment candidate

Base: c09d3e20b9c45c80b34ae3781440d90904a44023.
Native ops-transformer: 8b6924e68ed27181141cde84fa5a942c0c060aac, built for CANN 9.2.0.

This branch preserves the older standalone experiments. This patch does NOT
reuse the packed656 QSFA layout, and must not be stacked onto old C8 patches.

## Install
Download install-va030-qfm.py from this directory at the pinned commit and run:
`python /mnt/install-va030-qfm.py`.
It auto-detects the editable vllm-ascend installation. Override only when necessary:
`--source /absolute/path/to/vllm-ascend`.
`--check` performs read-only validation.
The installer verifies the exact base and affected file hashes, backs up
existing files, checks the patch, applies it, and verifies resulting hashes.
It does not install dependencies, compile operators, restart services, or run
NPU tests. Unknown/partial changes are refused without reset or force.

Run on every worker container using the same pinned native runtime and Python
wrapper. Restart workers after installation. Source native vendor environment
files and include its Python target directory in PYTHONPATH in the actual
service launch environment, not just an installer subprocess.

The installer prints the absolute command for the required lightweight
`check_kimi_k3_qfm_writer.py --card 0` smoke. It validates the newly introduced
ScatterPaCache tile writer in eager and graph replay, including offset,
padding and invalid slots. The previously tested QFM reader did not validate
this framework writer.

## Experiment configuration
Use MRV2, --kv-cache-dtype auto, --block-size 128.
Merge into existing additional-config:
```json
{"enable_kimi_k3_qfm_c8":true,"enable_mlapo":false,"enable_kv_nz":false,"qfm_kv_scale":0.03125}
```
**0.03125 is an explicit SYNTHETIC scale for wiring experiments, not calibrated
model accuracy acceptance.** Full576 static FP32 dequant scale is required;
a scalar or exact per-layer name mapping is accepted. Legacy Group128 scales
must not be substituted. Do not change the existing projection weights or
quantization settings just to switch cache readers.

Only main-model MLA uses full576 FP8 PA_NZ, with a single persistent cache.
Draft MLA and KDA caches remain unchanged. APC and chunked prefill read FP8
history directly; QFM is called once per layer, not per 16-query virtual row.
Captured metadata has stable addresses and runtime device length updates.
DCP uses local per-query prefix lengths, head gather and VA's LSE combine.

## Validation boundary
Local CPU contracts and neighboring existing regressions: 118 passed.
Ruff and patch installation/idempotence checked.
This is NOT full-model, distributed HCCL, or model accuracy acceptance.

First compare identical weights/settings with QFM off/on in eager, DCP=1,
DSpark off, short and 128K input. Then enable DSpark, graph replay and DCP
separately before the combined case. Include partial APC hits, mixed requests,
rejection followed by cache reuse, and long-context performance.
Full576 FP8 and native P quantization both require model-level accuracy checks.

MLAPO fused full576 writing is not admitted in this candidate. PCP, KV transfer,
KV layer parallelism, sparse offload and non-DSpark speculation are explicitly
rejected. TP/DSpark/graphs/DCP are wired, not certified on the full model.
Rollback: restart with the new flag off; do not reinterpret existing FP8 pages.

See the developer guide included in the patch for storage and execution details.
