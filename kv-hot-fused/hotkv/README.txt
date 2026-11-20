GLM-5.3 main-KV recent-window experiment — executable prototype
============================================================

Status
------
This delivery contains an actual Ascend Triton attention kernel, a native
quantization/write adapter, an A5 experiment runner, and a separately tested
request-lifecycle metadata core. It has NOT been compiled or run on A5 by the
author. It is NOT installed into GLM-5.3, mxllm, vLLM, or a PD connector.
Production correctness, task accuracy, and performance are NOT_VALIDATED.

The acceptance requirement is no performance regression. One attention launch
is a design choice, not evidence that this kernel is fast. In particular, the
experimental writer performs both native quantizers and cache scatters; these
costs are included in write+read timing. Do not deploy a slower configuration.

What runs
---------
kv_hot_fused.py reads existing packed C4 history and a C8 recent-token ring
inside ONE kernel, with one online softmax and no separate merge kernel.
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

No extra Hadamard is applied. Captured activations must already be in the
model's actual latent basis, including any transformation folded into weights.

Files
-----
kv_hot_fused.py             Actual single-launch mixed-precision reader.
kv_hot_backend.py           Native quantize/scatter + reader adapter.
test_kv_hot_fused_a5.py      Controls, diagnostics, paired performance runs.
kv_hot_state.py             CPU request/ring/speculation metadata state core.
kv_c4_scale_backend.py      Existing native C4 recipe adapter (unchanged).
kv_window_backend.py        Existing native Cloud reader/writer (unchanged).
kv_window_common.py         Existing BF16 input/reference/metrics (unchanged).
kv_window_capture.py        Existing original-activation capture helper.

Environment
-----------
Use the already-validated A5 / Cloud operator environment. No package or CANN
upgrade is performed. This prototype additionally requires an Ascend-compatible
Triton build in that same Python environment. Imports alone do not prove kernel
compiler support. A compilation failure is an ERROR with full traceback; there
is no slower substitute presented as the fused kernel.

For the previously validated /tmp/cloudops deployment:

  source /tmp/cloudops/cann_pkg/vendors/custom_cot/bin/set_env.bash
  export PYTHONPATH="/tmp/cloudops/pylib${PYTHONPATH:+:$PYTHONPATH}"

The launcher inherits your working proxy and Python. It adds cloudops/bin when
ninja is otherwise absent, and preserves an existing TORCH_EXTENSIONS_DIR or
uses cloudops/torch_ext. It downloads fixed-commit files, verifies SHA256, logs
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
    --timing eager,graph --output /tmp/kv-hot-first-run

After the first run passes, expand the existing experiment:

  python test_kv_hot_fused_a5.py --keys 8192,32768 --queries 1,4 \
    --windows 0,128,256,512 --cold-recipes baseline,alg2_max7 \
    --timing eager,graph --output /tmp/kv-hot-matrix

Use --block-keys 32 (default), 64, or 128 to evaluate kernel tile choices.
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
results.json, summary.csv, run.log, and per-case inputs.npz/outputs.npz retain
errors, references, original indices, query positions, and ring logical tags.
CONTROL_FAILED/ERROR rows are not timed. Matching references use the actual
native-quantized bytes after dequantization; a reduction in quantization error
cannot excuse a reader implementation mismatch. The default matching-reference
limit is 1% for both total and worst-query relative L2. Pipeline-vs-prepared
limit is 0.1%. These are experiment controls, not a model acceptance threshold.

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
