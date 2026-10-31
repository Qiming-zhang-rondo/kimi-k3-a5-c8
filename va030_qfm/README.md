# Unit-scale correction (r2)

The hand-entered `qfm_kv_scale=0.03125` experiment is withdrawn.
The required native KV descale is now internal `[1.0]`, matching the pinned
CANN K3 recipe. Remove the old config key and restart all workers; do not
reuse old in-memory KV/APC pages across the change.

`install-va030-qfm.py` accepts the clean pinned base or the exact candidate
published in commit 2873327403cec8d3404d2ff38ef7babe23978b63. It verifies
all affected file hashes, backs up the original files, and applies either the
full patch or the incremental correction. Unknown changes are refused.
It does not compile/reinstall native operators or run model tests.

Local regression checks: 120 passed. Installation, upgrade and idempotence
are checked locally. Native/model accuracy is still to be validated.

# Kimi K3 full576 FP8 QFM candidate

Base: VA 0.30.0 commit `c09d3e20b9c45c80b34ae3781440d90904a44023`.
Native provider: ops-transformer `8b6924e68ed27181141cde84fa5a942c0c060aac`,
compiled separately against the validated CANN 9.2.0 image.

This is a framework experiment candidate, not a model-accepted release.
The native reader was tested independently; the new framework tile writer
must pass `check_kimi_k3_qfm_writer.py` on the target image before model tests.
CPU tests are not a substitute for that native smoke or full-model acceptance.

## Configuration

Keep the existing model's weight quantization and DSpark draft configuration.
Use ModelRunner V2, `--kv-cache-dtype auto`, and `--block-size 128`.
Merge these keys into the existing additional-config object:

```json
{
  "enable_kimi_k3_qfm_c8": true,
  "enable_mlapo": false,
  "enable_kv_nz": false
}
```

KV is unscaled FP8: internally the required FP32 descale tensor is `[1.0]`.
This matches `prepare_prolog_weights()` in the
[pinned official K3 recipe](https://gitcode.com/cann/cann-recipes-infer/blob/4447f2f0972cb9287e069e3994b1d46ffe244ebc/models/kimi_k3/models/modeling_kimi_k3.py#L2580).
It is an identity scale, not a calibrated value from the checkpoint and not
a model-accuracy guarantee. Writing still clips to E4M3's finite range before
casting. Q retains its independent per-token/head dynamic scales.

**Remove `qfm_kv_scale` from the launch configuration.** The previous synthetic
`0.03125` example is withdrawn; the deleted config field now fails validation
if accidentally retained. Restart all workers and discard old in-memory KV
pages: pages written with the old scale must never be read with the new one.
Do not change KV scale dynamically between chunks or APC hits. The native
reader accepts one scalar for the entire cache, not per-token/group scales.

Only main-model MLA is quantized. Draft model and KDA state caches retain
their existing representation. kv_b_proj must remain unquantized. Existing
MXFP8 projection weight modules are executed through their ordinary path.
MLAPO is explicitly rejected: this patch does not claim fused full576 writes.

## Storage and execution

- One FP8 PA_NZ cache: `[blocks, 1, 18, 128, 32]`, 576 bytes per token,
  plus one FP32 scale per layer. No BF16 history cache is allocated.
- MRV2 physical page stride and storage offset are preserved. Hybrid KDA
  page padding can still limit the actual allocator capacity savings.
- ScatterPaCache sees contiguous 32-byte aliases and remapped tile slots;
  invalid slots remain -1. No invalid write is redirected to a live slot.
- Prefill, chunked prefill, partial APC hits and verification read the same
  FP8 pages. Q is absorbed to latent512+raw64, dynamically quantized per
  token/head, and passed to QFM.
- Ordinary eager batches organize queries into virtual rows of at most 16
  query tokens, but invoke attention ONCE, not once per virtual row.
- Graph decode uses one query per virtual row and stable cached metadata
  buffers. Every iteration refreshes device lengths, slots and page tables;
  DSpark rejection-adjusted lengths are not replaced by stale CPU mirrors.
- DCP gathers distinct query heads, uses interleave-aware local prefix lengths
  for every query, neutralizes empty shard output/LSE, and reuses VA's
  all-to-all/LSE combine. All ranks execute the same collective structure.
  This is wired code, not a claim of distributed model validation.
- Metadata generation is shared across the attention group, not repeated per
  MLA layer. Eager buffer shapes are not retained indefinitely.

## Explicit scope

Candidate integration covers TP head counts admitted by QFM, APC/chunked
prefill, DSpark main-model verification, MRV2 graph replay and DCP. Each still
requires end-to-end validation. PCP, KV transfer, KV layer parallelism,
sparse offload and other speculative methods are rejected, not silently
executed on the old cache layout. No V1 model-runner changes are included.

Roll back by restarting without `enable_kimi_k3_qfm_c8`; caches cannot be
reinterpreted across the switch. Do not load the old packed656 patch on top.
Save the startup log confirming `QFM C8 ... cache=FP8-full576` and profiling
showing quant_flash_mla. Confirm short and 128K requests before enabling
feature combinations. Compare with the same weights and server settings,
changing only the QFM feature, not weight quantization or concurrency.

Native P quantization produces a measurable FP32-reference error. Preserve
the raw FP32 accuracy metrics alongside implementation-consistency metrics;
do not label this candidate a precision pass based solely on short outputs.
