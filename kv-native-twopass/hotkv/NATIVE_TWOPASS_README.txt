Native C4 / C8 two-pass sparse-attention experiment
=================================================

Status: executable single-layer A5 microbenchmark, not model integration.
CPU/mock tests do not prove NPU compilation, graph support, accuracy or speed.
The author has not run this payload on A5. Use the current working container's
native operators and environment. The launcher installs/upgrades nothing and
does not source another vendor package. Native reader and quantizer adapters
are byte-identical to the previous payload. No custom cold attention is used.

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

Two independent optimizations are compared in the default matrix:
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

Suggested first A5 run (in the downloaded payload directory):
  python test_kv_native_twopass_a5.py --keys 8192 --queries 4 --topk 2048 \
    --windows 0,128 --cold-recipes baseline,alg2_max7 \
    --index-capacities full,bounded --merges torch,triton \
    --timing graph --warmup 20 --iterations 100 --repeats 5 \
    --diagnose --stages --output /tmp/native-two-pass-first

Expand --keys 8192,32768 --queries 1,4 --windows 0,128,256,512 after the first
run. For original model activations use --input /path/to/capture.npz instead.
The same capture schema as the previous KV experiments is supported.

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
