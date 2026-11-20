GLM-5.3 main-KV recent-window experiment — executable prototype
============================================================

Status
------
This v3 delivery contains an actual Ascend Triton attention kernel, a native
quantization/write adapter, an A5 experiment runner, and a separately tested
request-lifecycle metadata core. It has NOT been compiled or run on A5 by the
author. It is NOT installed into GLM-5.3, mxllm, vLLM, or a PD connector.
Production correctness, task accuracy, and performance are NOT_VALIDATED.

The supplied v2 run passed its numerical controls but showed a severe latency
regression versus native C4. V3 restructures that custom reader. Both native
backend files remain byte-identical to the working v1/v2 payload. V3 A5
compilation, numerical results and performance remain unverified.

The acceptance requirement is no performance regression. One attention launch
is a design choice, not evidence that this kernel is fast. In particular, the
experimental writer performs both native quantizers and cache scatters; these
costs are included in write+read timing. Do not deploy a slower configuration.

What runs
---------
kv_hot_fused.py reads existing packed C4 history and a C8 recent-token ring
inside ONE kernel, with one online softmax and no separate merge kernel.
V3 retains Q outside the key loop and decodes each KV tile once, reusing the
BF16 result for QK and PV. With RoPE, each key tile now has three dot operations
instead of seventeen. Integer/FP32 decoder intermediates remain 64 features
wide; a full-width BF16 tile is assembled with a dynamic loop. The FP32 output
accumulator is updated once per key tile. Cold-only tiles skip C8 decoding;
hot-only tiles skip C4 latent decoding. Default key tiles are 16 and compiler
multibuffering is disabled. These are structural changes, not measured speedups.
Retaining Q/KV can increase peak compiler storage; A5 compilation must verify it.
The portable BF16 assembly uses [BK,8,64]; device alignment and compiler copies
can inflate its storage beyond its logical byte count. V3 is not asserted to
fit UB until the installed A5 compiler has accepted the actual specialization.
The original TopK is unchanged. For query position p, an already-selected
token uses C8 only if p-W < token <= p. Missing required ring tags produce
NaN, never a silent C4 fallback. Padding and future tokens are masked.

Supported: one shared KV head, latent dimension 512, RoPE dimension 0 or 64,
BF16 queries, TND queries, paged packed C4 history. The C4 layout is low-nibble
first E2M1 + BF16 RoPE + E8M0 scale per 32 values. The C8 layout is E4M3FN +
BF16 RoPE + FP32 scale per 128 values. Hot and cold copies must originate from
the same original KV; the reader uses the cold history's unchanged RoPE bytes.

The harness creates C4/C8 from original BF16 activations, not by converting
already-quantized C4 to C8. It compares native BF16, native C8, native C4 W=0,
and fused C4+C8 W>0. Cold recipes are baseline and scale_alg=2,dst_type_max=7.
Each hot variant is timed against native W=0 using the SAME cold recipe.
W=0 bypasses the custom reader and allocates no hot ring. Existing reference
setup still prepares native controls outside measurement.

--fused-cold-control adds a separate read-only diagnostic using the custom
reader with hot selection, tags and C8 decoding disabled at compile time. It
uses the positive window's cache layout but computes pure C4 (semantic W=0).
This isolates custom-reader overhead from the window path. It never replaces
the native C4 baseline, runs no writer, and has no write+read timing. Its
matching-reference control must pass before its read latency is measured.

No extra Hadamard is applied. Captured activations must already be in the
model's actual latent basis, including any transformation folded into weights.

Files
-----
kv_hot_fused.py             Actual single-launch mixed-precision reader.
kv_hot_backend.py           Native quantize/scatter + reader adapter.
test_kv_hot_fused_a5.py      Controls, diagnostics, paired performance runs.
kv_hot_state.py             CPU request/ring/speculation metadata state core.
kv_c4_scale_backend.py      Existing native C4 recipe adapter (unchanged).
kv_window_backend.py        Existing Cloud reader and native writers (unchanged).
kv_window_common.py         Existing BF16 input/reference/metrics (unchanged).
kv_window_capture.py        Existing original-activation capture helper.

Environment
-----------
Use the new container environment in which BF16/C8/C4 controls passed, with
the container's matching native operator package and Python wrapper. Do not
assume that a package version label proves CANN compatibility. No package or
CANN upgrade is performed. This prototype requires an Ascend-compatible
Triton build in that same Python environment. Imports alone do not prove kernel
compiler support. A compilation failure is an ERROR with full traceback; there
is no slower substitute presented as the fused kernel.

The wheel may be present at /tmp/cloudops/pylib. In that case:

  export PYTHONPATH="/tmp/cloudops/pylib${PYTHONPATH:+:$PYTHONPATH}"

The launcher does NOT source /tmp/cloudops/cann_pkg's vendor script. Retain
the container operator environment that passed BF16. If an old vendor was
previously sourced, skipping source does not remove its paths: clear only that
vendor's entries from ASCEND_CUSTOM_OPP_PATH and LD_LIBRARY_PATH, then use a
new Python process. Keep the matching container libraries and other vendors.
Use a fresh TORCH_EXTENSIONS_DIR when switching native packages.

The launcher inherits your working proxy and Python. It adds cloudops/bin when
ninja is otherwise absent, and preserves an existing TORCH_EXTENSIONS_DIR or
uses an isolated cache in the new run directory. It downloads fixed-commit files, verifies SHA256, logs
everything, and creates a fresh output directory. It never edits model files.

Commands after downloading the payload
-------------------------------------
CPU reference/ring self-check (does not test the NPU kernel):

  python test_kv_hot_fused_a5.py --self-test

Environment check (does not compile or execute attention):

  python test_kv_hot_fused_a5.py --preflight

First A5 compilation, accuracy and timing experiment:

  python test_kv_hot_fused_a5.py --keys 8192 --queries 4 \
    --windows 0,128 --cold-recipes baseline,alg2_max7 \
    --block-keys 16 --fused-cold-control \
    --timing eager,graph --output /tmp/kv-hot-first-run

After the first run passes, expand the existing experiment:

  python test_kv_hot_fused_a5.py --keys 8192,32768 --queries 1,4 \
    --windows 0,128,256,512 --cold-recipes baseline,alg2_max7 \
    --block-keys 16 \
    --timing eager,graph --output /tmp/kv-hot-matrix

Use --block-keys 16 (default), 32, 64, or 128 to evaluate kernel tile choices.
Larger tiles can exceed compiler/device resources; no tile is asserted to
compile on your installed Ascend Triton until the actual A5 run succeeds.
Only --block-heads 16 is currently supported. All compilation is outside timed
events; graph capture and warmup are also outside timed events. Default timing
uses warmup=20, iterations=100, repeats=5. Cold and hot are measured in AB/BA
order to expose drift; repeated-call latency is not serving latency.

For real activations, use --input /path/to/capture.npz instead of synthetic
shape arguments. The capture helper and validate_case define its schema:
original BF16-valued Q/KV/RoPE, actual lengths, and the same fixed TopK.
Never use dequantized C4 as the purported BF16 source. Synthetic inputs cannot
establish GLM task-score recovery or the effect of checkpoint-folded rotation.

Interpreting results
--------------------
brief.txt is printed at completion and provides a short pasteable table of
read, write+read, paired native W0 times, paired ratios and relative L2.
Diagnostic rows have no pipeline timing; missing measurements are NA.
results.json, summary.csv, run.log, and per-case inputs.npz/outputs.npz retain
errors, references, original indices, query positions, and ring logical tags.
CONTROL_FAILED/ERROR rows are not timed. Matching references use the actual
native-quantized bytes after dequantization; a reduction in quantization error
cannot excuse a reader implementation mismatch. The default matching-reference
limit is 1% for both total and worst-query relative L2. Pipeline-vs-prepared
limit is 0.1%. These are experiment controls, not a model acceptance threshold.
Errors distinguish NATIVE_OPERATOR_CONTRACT, TRITON_UB_OVERFLOW and device
execution faults. A 507015 device execution fault stops all remaining work in
that process, preserving the original traceback instead of timing later calls.

MEASURED means controls passed and requested measurements completed.
VALIDATED_ONLY means controls passed with --timing none; no speed was measured.
OBSERVED_SLOWDOWN means the median paired event-time ratio exceeds 1.
NO_SLOWDOWN_OBSERVED only describes these samples. Neither result certifies
production performance; raw event and wall ratios are retained. Read-only
timing excludes writes; write+read includes current-token C4 and C8 quantizers,
both scatters, tag updates, and the fused read. Initial history construction,
full model forward, QLI, PD traffic, scheduling and MTP commits are not timed.

Lifecycle core and remaining production work
-------------------------------------------
RecentKVState in kv_hot_state.py has a separate committed ring and speculative
scratch. Stage/publish and commit/complete are explicit two-phase operations;
the caller must order real C4/hot writes and copies before publication. Prefix
acceptance, suffix rejection, ring wrap, stale generations and empty acceptance
are handled. Existing history must explicitly seed or restore genuine hot
payload; metadata cannot recreate lost precision. Export/import plans describe
PD payload transfer but do not perform it. No device memory is allocated by
this class, and a publish method is not a device synchronization operation.

The benchmark's fixed-batch modulo ring retains W+Q-1 logical positions
(page-rounded). It is intentionally NOT wired to that speculative state core:
the production reader still needs committed/scratch addressing, scheduler and
prefix-cache hooks, genuine high-precision prefill handoff, PD payload transport,
and MTP rejection integration. Arbitrary rollback of committed history requires
a matching restored tail. A fused production dual-writer is also still needed
if native dual-write costs fail the performance requirement. Do not describe
this prototype as an end-to-end serving implementation.

CPU tests
---------
From the repository's kv-hot-fused directory:

  python -m unittest discover -s tests -v

The download-launcher test requires the full repository checkout; it resolves
the run_kv_hot_fused.py file at the repository root automatically.
CPU/mocked tests validate formats, state transitions, gating and reporting.
They do not validate NPU compilation, execution, graph support or speed.
