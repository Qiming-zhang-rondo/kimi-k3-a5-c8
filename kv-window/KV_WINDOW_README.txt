Main MLA KV recent-precision experiment for Ascend A5
===================================================

V2 diagnostic update
  This release fixes error reporting/evidence preservation. It DOES NOT claim
  to fix the reported BF16 mixed-window numerical failure on A5.
  NaN in an observed kernel output used as a comparison baseline is now named
  explicitly; it is not reported as a generic mathematical-reference NaN.
  Partial outputs are saved before later comparisons can fail. summary.csv
  includes the control status, so failed paths are not accepted quality results.

  Targeted untimed investigation, using an already validated operator/toolkit
  combination (do not recreate a combination that has caused device faults):
    python test_kv_window_a5.py --keys 32768 --queries 4 \
      --windows 128,512 --hot-dtypes bf16 --timing none --diagnose

  Per-window diagnostics compare:
    * full BF16 output with softmax-statistics return on/off;
    * the same hot subset read from global and ring caches, stats on/off;
    * individual outputs and LSE against a named FP64 subset reference;
    * two BF16 global partitions merged vs one unsplit BF16 read;
    * selected ring latent/RoPE values against the original BF16 values.
  The raw output/max/sum/LSE arrays, including nonfinites, stay in outputs.npz.
  These extra calls are opt-in and outside all performance measurements.

  Synthetic selection caveat: the generator deliberately includes the most
  recent 256 tokens, then samples history. Extending W=256 to512 therefore
  adds only the selected historical entries that fall in that band; saturation
  here is not evidence of the optimal window on model-generated TopK.

  Compatibility caveat: a user reported device faults using an operator build
  labelled 9.1.0 inside a CANN9.2.1 container. We have not reproduced that
  report or established a version-wide incompatibility. Keep the exact build,
  toolkit, driver and firmware identities with any fault report; this program
  neither installs a replacement wheel nor changes the environment.

Goal
  Keep fixed sparse indices and compare full BF16, full C4 (MXFP4), full C8
  (FP8 E4M3, block128), and historical C4 + recent BF16/C8. The recent window
  changes the precision of selected tokens; it never adds tokens to TopK.
  This is an experimental single-attention program, not a production model
  patch, a GLM benchmark, or proof of an end-to-end score/throughput change.

Environment
  Use the SAME Python environment as the working model. Requires numpy,
  torch, torch_npu, and the installed cloud_ops_turbo operators. The program
  does not install packages, upgrade CANN, or change model/service files.
  The first real operator call on A5 is still required to validate that the
  installed operator wheel implements the inspected argument contract.

Run from this directory
  python test_kv_window_a5.py --self-test
  python test_kv_window_a5.py --keys 8192,32768 --queries 1,4 \
    --windows 0,128,256,512 --hot-dtypes bf16,c8 --timing eager,graph

  Defaults use B=1,H=64,D=512,RoPE=64,page=128,TopK=2048. Generated inputs
  are synthetic (mildly heavy-tailed); selected indices are random history
  plus recent tokens, NOT the output of QLI. Synthetic results cannot explain
  a GLM task-score difference. --input path.npz replays real captured inputs.
  For a fast first run use --keys 8192 --queries 1 --windows 0,128,256,512.

What is checked
  - Original, matching C4-dequantized and matching C8-dequantized references
    use FP64 mathematical sparse MLA. They are not bit-exact BF16 goldens.
  - Split-C4 vs unsplit-C4 validates partitioning and softmax merge without
    changing precision. Window=0 is a no-window control. Include a window
    >= key length in a small test to check the all-hot endpoint.
  - Pipeline (writes + device partition + reads + merge) vs prepared output
    detects inconsistencies introduced by cache updates and index generation.
  - Invalid indices, duplicates, future tokens, padding holes, nonfinite
    values and non-original-BF16 replay data are rejected before NPU calls.
  - There is no attention sink in this replay/operator contract.

Precision interpretation
  vs_bf16_operator.relative_l2 is the principal single-attention discrepancy.
  relative_l2_reduction_vs_c4 > 0 means improvement on those inputs;
  < 0 means worse. No monotonic improvement is promised. Report also contains
  per-row errors, cosine similarity, maximum absolute difference and recent
  selected-token counts. A recent BF16 window must contain pre-quantization
  values: casting C4-dequantized data to BF16 cannot recover lost precision.
  kernel_vs_matching_reference and split_c4_vs_unsplit_c4 are correctness
  guards, not a quality criterion for quantization. Default relative-L2 guards
  are 0.03 and 0.01. Failed controls suppress performance claims for that path.

Performance interpretation
  prepared_read measures attention reads and (for a window) output merge
  after precomputing index partitions. It excludes index partition and writes.
  write_partition_read measures C4 current-token quantization/scatter, hot
  window write, device index partition/remapping, two reads and merge. Baseline
  write_read includes its corresponding current-token quantization/scatter.
  Writes repeat the same trailing token positions/values. This is steady-state
  microbenchmark work, not an autoregressive rollout or request scheduler.
  Input generation, initial full-cache quantization, CPU reference, initial
  ring population, first-use compilation, and output copies are outside timing.
  Model projections, RMSNorm, RoPE, QLI, distributed collectives, scheduler,
  full-model ACLGraph interactions, prefix caching and MTP lifecycle are not
  measured. Projecting a latency ratio into end-to-end tokens/s is invalid.
  Eager reports event and synchronized wall timing. Optional graph timing
  captures the selected microbenchmark; graph failures are reported explicitly,
  never silently replaced by eager numbers. A two-read prototype need not
  represent the performance of a future fused mixed-cache kernel.
  Ring capacity accounts for W+max_query_length-1 and page alignment. Extra
  memory is a high-precision ring IN ADDITION to full C4, not merely the
  difference between C4 and the high-precision byte widths. Reference/test
  buffers and physical ring residency are reported separately.

Real model input capture
  kv_window_capture.save_paged_mla_case() exports one existing BF16 main-MLA
  operator input snapshot. It must be called on an eager BF16-cache run after
  projection/RMSNorm/RoPE, with the actual fixed QLI sparse indices and scale.
  It rejects a C4/C8 packed cache; collecting a C4 run requires an independent
  original-BF16 shadow history, which this program does not manufacture.
  Example at a BF16 sparse-attention call site (outside graph capture):

    from kv_window_capture import save_paged_mla_case
    save_paged_mla_case(
        '/mnt/main_mla_layer20.npz', q_nope=ql_nope, q_rope=q_pe,
        kv_cache=kv_cache[0], key_rope=kv_cache[1],
        sparse_indices=topk_indices, block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_key, scale=self.scale,
        label='layer20_real_input')

    python test_kv_window_a5.py --input /mnt/main_mla_layer20.npz \
      --windows 0,128,256,512 --hot-dtypes bf16,c8 --timing eager,graph

  The helper is opt-in: importing it does not patch or start a model.
  Select a bounded representative snapshot; copying a long full cache can be
  expensive and must never be included in measured latency. Data stays local.
  A single captured step checks local attention error. An end-to-end score
  experiment needs integration and full autoregressive evaluation separately.

Output
  results.json: environment/operator identities, controls, accuracy and timings.
  case_XXX/summary.csv: relative L2, read/pipeline microseconds, latency ratios
  versus full C4/C8, and extra ring bytes. Ratio >1 means slower; <1 faster.
  case_XXX/outputs.npz: actual BF16/C4/C8/mixed/split outputs and reference.
  --save-inputs additionally saves replayable original inputs (can be large).
  Each invocation creates a fresh result directory. Existing data is not
  overwritten. Errors include full traceback; no environment repair is attempted.

Public basis for the experiment
  GLM-5.3-Flash's public tail cache is an incomplete Indexer pooling buffer;
  we found no public implementation of recent BF16/C8 plus historical FP4
  main KV in the inspected model paths. This experiment is not a reproduction
  of a claimed GLM-5.3-Flash feature.
  https://github.com/vllm-project/vllm/blob/61c3895fc937b4ffc2a59abe188959de1e459c8a/vllm/models/glm5next/common/attention.py
  https://github.com/vllm-project/vllm/blob/61c3895fc937b4ffc2a59abe188959de1e459c8a/vllm/models/glm5next/nvidia/specialized_decode/dsa_attention.py
  DeepSeek V4.1 Flash retains FP8 SWA KV but uses a different FP4 format with
  QAT for global KV. It does not establish the benefit of this GLM PTQ change.
  https://arxiv.org/html/2609.19969v1
  Two attention partitions must be combined using their softmax statistics.
  https://docs.flashinfer.ai/tutorials/recursive_attention.html
