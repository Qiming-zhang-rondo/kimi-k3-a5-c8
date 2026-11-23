C8 versus BF16 recent-128: one-shape, fixed-input A5 experiment
===========================================================

Purpose
Compare C8 and BF16 hot KV precision and full write/partition/read/merge cost.
Use the already working Cloud operator/container environment and an idle NPU.
No package installation, CANN upgrade, vendor sourcing, or model modification.

One workload
B=1, Q=4, K=8192, H=64, D=512, RoPE=64, TopK=2048, W=128,
page=128, seed=20261120, baseline C4 recipe.
The synthetic selection contains the latest 256 visible tokens. Consequently
every row has 128 hot hits. This is an intentional fixed-workload comparison,
not a model activation capture or evidence about arbitrary QLI selections.
The physical ring can be wider than W to preserve all four query windows.

Compared paths
- Full BF16 and full C8 controls; unchanged native C4 W0 baseline.
- Split-C4 control, original C8 two-pass, current C8 hot_store_partition.
- BF16 hot ring with ordinary stores and with fused hot-store/partition.
Cold C4 quantization/reader, TopK, query basis and RoPE are unchanged.
BF16 hot values come directly from original BF16 KV, never C4 dequantization.
The BF16 reader receives compact indices to avoid the known padded-subset
problem in the previously deployed package. Reader call counts are reported.

Measurement
Compare BF16 ordinary writer against C8 ordinary writer, and BF16 fused writer
against C8 fused writer, with alternating AB/BA pairs. Full pipeline includes
current-token writes, device-side partition, both native attention reads and
merge. Setup/capture/warmup/reference computation are outside measurement.
Standalone stage times are diagnostic and must not be added to predict pipeline
latency. Pair min/max are observed ranges, not confidence intervals.

Correctness
Timing requires the mathematical references and pipeline/prepared comparison,
partition and writer checks, and zero-query subset counts to pass. Errors retain
their traceback and raw comparisons. No output sanitization or silent fallback.
MEASURED means these controls and measurements completed, not zero overhead or
production acceptance. CPU tests do not establish A5 compilation or runtime.

Scope
Fixed input and count plan, not continuous decoding, variable-hit graph support,
serving concurrency, speculative rollback, cache migration, or PD transfer.
BF16 may improve storage precision while using more memory/read bandwidth;
it also removes C8 quantization. Neither latency nor end-to-end model accuracy
is assumed better before the A5 measurements.
