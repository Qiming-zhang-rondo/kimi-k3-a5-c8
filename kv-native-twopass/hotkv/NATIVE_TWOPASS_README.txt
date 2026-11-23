Native C4 / C8 sparse-attention experiment v4
=================================================

Status: executable single-layer A5 microbenchmark, not model integration.
CPU/mock tests do not prove NPU compilation, graph support, accuracy or speed.
The author has not run this payload on A5. Use the current working container's
native operators and environment. The launcher installs/upgrades nothing and
does not source another vendor package. Native reader and quantizer adapters
are byte-identical to the previous payload. No custom cold attention is used.

Native cold + fused hot reader/merge sweep (v4)
----------------------------------------------
--reader-fusion-sweep uses bounded indices and the original native writers.
For each recipe/window it evaluates the original native two-pass champion,
an optimized descending-sort control, and twelve C8 hot-reader/merge choices:
  --hot-block-keys 16,32,64 --hot-block-heads 16,32
  --hot-decodes software,native_cast
These sizes are candidates; compiler resource limits and speed need A5 tests.
No single tile or this finite search is asserted to be a theoretical optimum.
The native cold reader, W0, and both quantizers are unchanged.

The argsort_desc control uses exactly representable FP32 ORDINAL sort keys,
descending=True, and trims the hot order tensor before gathering IDs. Token
IDs stay INT32 and stable selection order is preserved. This incorporates the
reported ideas; it is not a copy or measured parity with a colleague's source.

The hot_merge candidate performs one native C4 cold read, then one custom
kernel that reads only C8 hot tokens, computes their attention, and merges
with the native cold output/max/sum. It does not decode cold C4 or replace its
fast reader. It removes the separate hot native API call and standalone merge
by combining their work, not by omitting required tokens or writes.
Native API-call count is not a claim about total device kernel count.

The new reader uses the installed Ascend Triton API
  triton.language.extra.cann.extension.insert_slice
to assemble decoded BF16 tiles. This API, compilation and UB use must be
verified in the current working A5 environment. The launcher installs nothing.
An unavailable API or failing tile is reported, never silently replaced.
The native_cast candidate bitcasts original uint8 bytes to E4M3FN
(tl.float8e4nv), converts to FP32, multiplies the unchanged FP32 scale, and
rounds to BF16. The software decoder is a separately measured control, not a
silent fallback. Current official 950 cast support does not establish support
in the deployed compiler. Both decoders are checked against all 256 byte
encodings in CPU tests; actual device correctness remains gated by references.
Both readers validate the trailing -1 index prefix on device and stop the
hot QK/PV loop at its actual selected count. Padding holes/invalid addresses
produce NaN for controls to reject. Tail trimming is data-dependent device
work with fixed buffers; no host synchronization or variable-shape allocation
is introduced. Presence-false dummy rows bypass the hot loop.

Before timing, the fused reader must match the native two-pass output within
the existing pipeline tolerance (0.1% global and worst-row L2), the native hot
output within the existing control tolerance (1%), and forced cold-only and
both-empty controls. An independent zero-query specialization must give
exp(hot_max)*hot_sum = selected-hot count. Q/cache inputs are not mutated by
these diagnostics. The original full matching-reference and graph-replay
checks also remain. Hot normalized-output BF16 rounding is retained when the
actual installed native hot reader returns BF16; FP32 outputs remain FP32.

The native split-C4 controls are reused only for identical partition/merge/
writer configurations. A failed split blocks the associated mixed candidates.
Fused candidates are paired with the same-recipe original mixed champion as
well as W0. With the default one shape/two recipes, there are 34 variant rows
(2 W0 + 4 split controls + 28 mixed) plus 2 BF16/C8 baseline rows = 36 entries.
Optional compile failures do not stop other configurations; device execution
faults still stop the run. Native W0 is never redefined as a slower baseline.
paired_summary.txt lists each mixed candidate's exact W0 pair and champion
pair, including all displayed medians and ratio range, and explicit counts.
Champion self-comparisons remain NA. Reported paired ratios are medians of
per-round ratios, not the quotient of the two displayed medians. Use this
file directly instead of reconstructing or relabeling results by hand.

Suggested first A5 run:
  python test_kv_native_twopass_a5.py --reader-fusion-sweep \
    --batch 1 --keys 8192 --queries 4 --topk 2048 --windows 0,128 \
    --cold-recipes baseline,alg2_max7 --hot-block-keys 16,32,64 \
    --hot-block-heads 16,32 --hot-decodes software,native_cast --timing graph \
    --warmup 20 --iterations 100 --repeats 5 --diagnose --stages \
    --output /tmp/native-hot-reader-v4

The measured full pipeline still includes current-token C4/C8 quantization,
both writes, dynamic partition, cold read and fused hot-read/merge. Prepared
read excludes partition/writes and is explicitly a separate metric. All
reference/history preparation, compilation and checks remain outside events.
Framework step-level reuse, PrologV3 integration, colleague 27-op count and
end-to-end TPOT are not measured by this fixed-input experiment.

Optimization sweep (v3)
-----------------------
--optimization-sweep overrides the capacity/merge/partition/writer matrix with
seven bounded configurations, evaluated separately for split_c4 and mixed:
  champion: triton_scan + triton (log256) + native writer
  partition_only: one encoded integer prefix scan (triton_scan_packed)
  merge_log512_only: original log-domain merge with a larger 512-wide block
  merge_direct512_only: max/sum weights with a guarded log-domain fallback
  writer_only: original native quantizers + one fused packing/scatter kernel
  stacked_log512: packed scan + log512 merge + fused_store
  stacked_direct512: packed scan + direct512 merge + fused_store
The current reported 132 us mixed configuration is the champion reference;
its timing is remeasured, not copied from the old report. Every candidate is
paired against BOTH same-mode/same-recipe champion and original native W0.

fused_store is NOT fused quantization: it retains the exact native C4 recipe
arguments and the native C8 row=1/column=128 quantizer. It eliminates separate
concatenation/scatter operations by directly storing both packed records in one
Triton launch. Before timing, complete caches are poisoned and compared byte
for byte against native writers, including RoPE, scale bytes and untouched
slots. Caches are restored even on failure. Duplicate/out-of-bounds write slots
are rejected at setup. W0 remains unchanged and uses no custom store.

The packed scan preserves arbitrary selection order and empty-subset handling;
its int32 encoding has a checked capacity bound. direct512 uses direct
max/sum weighting only for valid sums in [1,65536], otherwise uses the same
stable log formulation. No approximate exp or relaxed numerical tolerance is
introduced. Any candidate compilation or correctness failure is reported,
never silently replaced. MEASURED is still not a performance acceptance.

The v3 optimization-sweep mode does NOT replace the two attention calls with an unverified
mixed_quant_sparse_flash_mla invocation. Existing dual-KV API names alone do
not establish support for these MXFP4-400B/FP8-656B layouts and BF16 RoPE.
inspect_kv_fusion_capability.py collects installed source/header/version
evidence without importing or executing operators. It does not enable a path.
One native mixed reader needs a matching kernel contract/build before A/B.

Suggested all-candidate run:
  python test_kv_native_twopass_a5.py --optimization-sweep \
    --batch 1 --keys 8192 --queries 4 --topk 2048 --windows 0,128 \
    --cold-recipes baseline,alg2_max7 --timing graph \
    --warmup 20 --iterations 100 --repeats 5 --diagnose --stages \
    --output /tmp/native-two-pass-v3
New candidates require A5 validation. No 86 us or no-regression result is
promised. Start in the same already-working container; do not swap CANN/Cloud
packages. short_summary.txt shows measured mixed-only winners for this run;
full sample distributions and rejected candidates remain in results.json.

Measured paths
--------------
1. Native BF16 and native C8 controls.
2. Native C4 W0, for baseline and scale_alg=2,dst_type_max=7 recipes. W0 uses
   the original stats-off reader, writer, and no ring or split allocation.
3. split_c4: partition original TopK into old/recent; both subsets use native
   C4 reads from the same private C4 cache, followed by a statistics merge.
   This must match both the mathematical C4 reference and native C4 W0.
4. mixed: native C4 old subset + native C8 recent ring + the same merge.
   Timing is gated by the corresponding split_c4 control and its own reference.

Original index-partition implementations (v2)
-----------------------------------
--partitions argsort_int,argsort_float,triton_scan
  argsort_int is the unchanged integer-sort baseline. It is always evaluated
  first and must be included. The placement reported by the actual deployed
  operator/profiler is authoritative; this script does not assume an engine.
  argsort_float converts only the unique ordinal+mask sort keys to FP32. It
  keeps token IDs INT32 and requires 2*S-1 <= 2**24 for exact sort keys. This
  tests a dtype-dispatch hypothesis; it does not promise an AiCore dispatch.
  triton_scan uses one index-only NPU kernel with integer prefix scans. Output
  buffers are preallocated. Prefix scatter, -1 suffix fill and empty-row dummy
  writes target disjoint addresses. There is no sorting, nonzero, variable-size
  output or host synchronization in the timed partition. It also emits invalid
  row flags for malformed positions/indices or overflow; uniqueness is checked
  when input cases are validated. A5 compilation and latency are unverified.

Every variant first compares all device partition outputs exactly against the
CPU oracle, including global/ring indices and empty-row masks. Integer copies
preserve IDs above 2**24. These untimed checks and invalid-row flags gate reads
and measurements. End-to-end output controls still run separately.

Candidates retain the same-recipe native W0 comparisons. They ALSO run paired
full-pipeline comparisons against argsort_int for the SAME recipe, window,
index capacity, merge and split_c4/mixed mode. Never compare a split_c4 control
against a mixed candidate and attribute the difference to a single component.
brief.txt explicitly displays role and partition algorithm to avoid this mixup.

Two other independent knobs remain available:
--index-capacities full,bounded
  full: both index tensors retain original TopK capacity S.
  bounded: hot index capacity is min(W,S); cold stays S. Unique visible input
  indices are validated before execution, so at most W selected keys are hot.
  This does not enlarge TopK or shorten the actual selected recent set. The
  ring itself retains W+Qmax-1 positions rounded to a page; its storage capacity
  is distinct from index capacity. Dynamic split/compact work remains timed.
--merges torch,triton
  torch: ordinary device tensor operations implement the stable merge.
  triton: only the small elementwise merge/presence masking is one custom kernel,
  with preallocated output. Both attention calls still use the native reader.
  No silent fallback: a Triton compile/error is recorded and other configurations
  continue unless a device execution fault makes further device work unsafe.
When triton_scan is selected there is also one custom INDEX kernel; neither
custom kernel replaces or decodes native attention's KV data.

Correctness
-----------
All paths consume the same original BF16 Q/KV/RoPE and fixed TopK. References
dequantize the actual native-generated C4/C8 bytes. A recent token is selected
only if already in TopK and query_position-W < token <= query_position.
Hot data comes directly from original BF16, never from already quantized C4.
No extra Hadamard is applied; real captures must use the model's actual basis.

The C4 layout is E2M1 + E8M0 scale/32 + BF16 RoPE; C8 is E4M3FN + FP32
scale/128 + BF16 RoPE. D512/R64 use 400/656 bytes per token respectively.
Sparse padding is trailing -1. Empty subsets use a valid dummy native read;
presence masks exclude that dummy from the final output and statistics.
The merge combines already-normalized outputs using max/sum (equivalently LSE).
It is algebraically equivalent in real arithmetic for the SAME mixed KV, not
bit-exact equivalent to unquantized BF16 or a single FP32 implementation.
Attention sinks are not used in this experiment; production must avoid double
counting a sink if splitting a reader that has one.

--diagnose additionally executes zero-query reads and checks each nonempty
subset's exp(max)*sum against its valid count. A wrong padding/statistics
contract prevents timing. Empty dummy rows are explicitly excluded.
Prepared, dynamic-split, and current-token-write+dynamic-split outputs must
agree. Default matching-reference tolerance is 1% global AND worst-query L2;
pipeline consistency is 0.1%. These are experiment controls, not task accuracy.

Timing and reporting
--------------------
Prepared read: prebuilt subset indices + two native readers + merge.
Full pipeline: original current-token C4 quantization/write, C8 quantization/
ring write for mixed, dynamic split/compact, two native readers and merge.
split_c4 has no hot-cache write. W0 includes its original C4 write and one read.
All candidates are paired against SAME-recipe W0 in alternating AB/BA rounds.
Compilation, initial history preparation, warmup and graph capture are untimed.
Graph replay is checked against eager output before and after measurement.
Each captured graph contains ONE call; iterations=100 means 100 graph replays
outside capture, not a graph containing 100 repeated calls. Fixed-shape
argsort/gather/where and scan paths do not themselves prove a serving graph
break. This experiment does not measure whole-model capture or recapture.

--stages adds standalone partition, reader, merge and writer timings. These
are diagnostic observations; do not add them or subtract them from full-path
medians. Cache reuse and dispatch interactions make such arithmetic misleading.
One ACLNN API call is not asserted to correspond to one device kernel.

MEASURED means numerical controls passed and requested measurements completed.
OBSERVED_SLOWDOWN means the paired median event ratio exceeds 1; this does not
meet the no-regression requirement. NO_SLOWDOWN_OBSERVED is only a statement
about these samples, not a serving-performance guarantee. Results never mark
production validation as complete.

brief.txt is printed at completion. results.json/summary.csv preserve paired
W0 times, all timing samples, failure tracebacks, controls and stage timings.
Each case retains inputs.npz and outputs.npz. Device execution error 507015
stops further device work. Optional configurations are not silently omitted.

Suggested partition-isolation A5 run (in the downloaded payload directory):
  python test_kv_native_twopass_a5.py --keys 8192 --queries 4 --topk 2048 \
    --windows 0,128 --cold-recipes baseline,alg2_max7 \
    --index-capacities bounded --merges triton \
    --partitions argsort_int,argsort_float,triton_scan \
    --timing graph --warmup 20 --iterations 100 --repeats 5 \
    --diagnose --stages --output /tmp/native-two-pass-first

Expand --keys 8192,32768 --queries 1,4 --windows 0,128,256,512 after the first
run. For original model activations use --input /path/to/capture.npz instead.
The same capture schema as the previous KV experiments is supported.
--batches 1,8,32 optionally sweeps batch sizes (otherwise --batch is used).
Use actual scheduled batch/query/KV lengths when aligning with serving traces:
request concurrency is not necessarily device batch, Q4 is four queries per
sequence rather than four requests, and a character count is not token length.
Multiplying per-layer latency differences by model depth is only an assumption;
it does not attribute TPOT differences to Prolog, launch count or graph breaks.

Scope limits
------------
This is an independent reproduction of two-native-reader mathematics, not a
copy or verified execution of an internal framework patch. It does not measure
QLI, model forward, request scheduling, prefix cache, PD transfer, MTP rejection
or readiness transitions. Fixed rings are prepared for the supplied query batch;
repeated timing rewrites those same tokens and does not simulate advancing decode.
The extra ring/metadata allocation is reported; preparation caches and reference
buffers are not production memory requirements. Actual framework integration
still needs ownership/generation checks and write/transfer-before-ready ordering.

CPU verification from repository's kv-native-twopass directory:
  python -m unittest discover -s tests -v
Launcher tests require the repository-root run_kv_native_twopass.py as well.
