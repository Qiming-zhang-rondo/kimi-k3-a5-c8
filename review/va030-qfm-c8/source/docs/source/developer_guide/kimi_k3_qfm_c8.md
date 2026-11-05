# Kimi K3 full576 FP8 QFM candidate

## Local experimental FIA Prefill variant (not the published bb79bb2 installer)

Set `enable_kimi_k3_qfm_fia_prefill=true` together with
`enable_kimi_k3_qfm_c8=true`. Keep `enable_mlapo=false` and
`enable_kv_nz=false`. Disabling the new flag retains the QFM-only reader.

Scheduler-marked Prefill requests use BF16 FIA-v2 in the absorbed MLA
latent512/raw64 representation. Existing FP8 history (including APC hits)
is gathered through the request's kernel-page table and converted to temporary
BF16. Current KV stays floating for Attention and is also written to FP8 cache.
Only the past prefix is gathered; current KV must not be counted twice.
History and causal current outputs are merged with natural-log LSE weights.
There is one persistent FP8 cache, not a second BF16 cache pool.

Decode and main-model speculative verification keep QFM, including in mixed
batches. Draft routing is unchanged. For DCP, each rank reads its local history;
only rank zero contributes current KV, then the existing head exchange/LSE
merge combines all contributions. Real multi-card validation is still required.

This candidate supports eager Prefill only. Decode-only capture/replay remains
on QFM. Prefill graph capture is explicitly rejected, not silently downgraded.
Metadata building reads corrected sequence lengths to CPU once for a Prefill
step; it does not synchronize once per layer. That synchronization is an
explicit candidate performance cost. History conversion is request-at-a-time,
with memory proportional to that request's local history (including FP32
conversion and gather temporaries). It is not an allocation-free path.

Run `check_kimi_k3_qfm_fia_prefill.py` directly on one A5 before model use.
It checks paged FP8 history, nonzero storage offsets, padding, causal current
KV, and simulated DCP partitions against sampled FP32 reference rows. It does
not validate distributed HCCL or full-model accuracy. Compare actual TTFT,
TPOT and peak memory; changing the API to FIA alone is not a speedup claim.
The new option does not make toolkit MlaPrologV4 compatible with QFM storage.

Base: VA 0.30.0 commit `c09d3e20b9c45c80b34ae3781440d90904a44023`.
Native provider: ops-transformer `8b6924e68ed27181141cde84fa5a942c0c060aac`,
compiled separately against the validated CANN 9.2.0 image.

This is a framework experiment candidate, not a model-accepted release.
The native reader was tested independently; the new framework tile writer
must pass `check_kimi_k3_qfm_writer.py` on the target image before model tests.
CPU tests are not a substitute for that native smoke or full-model acceptance.

## Configuration

Keep the existing model's weight quantization and DSpark draft configuration.
Use ModelRunner V2 and `--kv-cache-dtype auto`. Explicitly set `--block-size`
to 128 or a positive multiple (for example 256, 512 or 1024). This is the KV
manager block size; the QFM kernel page size remains 128. The final hybrid
planner may promote the manager size after model loading; the allocator
uses that final size rather than the earlier command-line value.
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

DSpark exclusion uses the loader's live `dspark_head` construction tag, not
only `model_config.runner_type`. MRV2 deliberately retains the target config
while loading its draft, and Kimi sets the draft RoPE flag after creating the
MLA implementation. Relying on those two temporary values selected QFM for
draft layers and sent a 5D FP8 cache to their 4D BF16 writer. The fix excludes
that scope before QFM format checks and preserves the original draft backend.

## Storage and execution

- One FP8 PA_NZ cache: `[blocks, 1, 18, 128, 32]`, 576 bytes per token,
  plus one FP32 scale per layer. No BF16 history cache is allocated.
- MRV2 physical page stride and storage offset are preserved. Hybrid KDA
  page padding can still limit the actual allocator capacity savings.
- MRV2 already expands each manager block into `manager_block_size / 128`
  kernel block-table entries and computes matching slots. The QFM view now
  exposes that many kernel pages per manager page; metadata does NOT expand
  the table a second time. For manager=512, manager ID 2 maps to kernel IDs
  8, 9, 10 and 11. Slots and page IDs remain consistent for APC, chunked
  prefill, DSpark and DCP. No token-index matrix or extra KV copy is added.
- With physical manager page size P and ratio R, each kernel page uses
  stride P/R. Padding is distributed between subpages, not appended only
  after the last one. P must cover the payload and be divisible by R*32;
  shared Attention/KDA page sizes are rounded to this alignment before
  memory budgeting and allocation, with the same size applied to both.
  incompatible physical layouts fail before creating an overlapping view.
  Startup logs report `QFM cache geometry: manager_block=... kernel_block=128`.
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
requires end-to-end validation. PCP, KV layer parallelism,
sparse offload and other speculative methods are rejected, not silently
executed on the old cache layout. No V1 model-runner changes are included.

Roll back by restarting without `enable_kimi_k3_qfm_c8`; caches cannot be
reinterpreted across the switch. Do not load the old packed656 patch on top.
Restart all workers after installing the manager-block mapping fix; do not
reuse old in-memory/APC cache pages. This update does not include the separate
experimental DSL Prolog; keep `enable_mlapo=false`.
Save the startup log confirming `QFM C8 ... cache=FP8-full576` and profiling
showing quant_flash_mla. Confirm short and 128K requests before enabling
feature combinations. Compare with the same weights and server settings,
changing only the QFM feature, not weight quantization or concurrency.

Native P quantization produces a measurable FP32-reference error. Preserve
the raw FP32 accuracy metrics alongside implementation-consistency metrics;
do not label this candidate a precision pass based solely on short outputs.
# Local FIA/QFM + DSL Prolog integration candidate

The local candidate adds `kimi_k3_c8_prefill_backend="fia"|"qfm"`
(default `qfm`). The previous experimental `enable_kimi_k3_qfm_fia_prefill`
alias remains accepted. Keep `enable_kimi_k3_qfm_c8=true`,
`enable_kv_nz=false`, and cache dtype `auto` for either route.

With `enable_mlapo=true`, selected target MLA layers use the pinned DSL
Prolog, **not toolkit MlaPrologV4**. QFM-only batches use FP8-output mode2.
Batches containing FIA Prefill use BF16-output mode0 for the current batch,
write the same persistent FP8 cache, and quantize any Decode queries at the
QFM reader boundary. This preserves the floating current KV for FIA without
a second permanent BF16 cache. FIA history is gathered/dequantized into a
temporary BF16 tensor per request; peak memory and performance require
long-context measurement. Prefill graph capture is not admitted by this
candidate; Decode graph metadata remains on the existing path.

The DSL runtime must be explicitly loaded. This delivery imports the pinned
wheel entry `ops.quant_mla_prolog` and checks its source hash and `_native`
directory; it does not fall back to the old standalone source module. Source
the private wheel environment with `CANNBOTDSL_NATIVE_BINARY_MODE=require`
before starting workers to forbid JIT for uncovered configurations. Actual
cold-process binary coverage is not yet validated. Synthetic probes for the two paths
passed before integration; they do not validate this integrated candidate's
model execution, real DCP/HCCL, DSpark acceptance, or performance.

PCP and KV layer parallelism remain unsupported. Preserve
the existing target/draft construction-tag isolation and cache format checks.

## Mooncake V2 PD candidate

### Stage-specific Prolog

Set `enable_mlapo=true` and `enable_kimi_k3_mlapo_prefill=false` to keep
DSL Prolog for Decode while using ordinary W8A8 MXFP8 projections, RMSNorm
and explicit FP8 Cache writing for Prefill. The new flag defaults to true;
`enable_mlapo=false` still disables DSL Prolog for both stages. Attention's
`kimi_k3_c8_prefill_backend` (qfm/fia) is independent of this writer policy.

Routing uses scheduler CPU stage flags, including short final Prefill chunks,
not query length. Pure Decode and DSpark verification retain the existing
FP8-output mode2 path and Decode graph behavior. Mixed batches split adjacent
stage ranges: Prefill uses ordinary projections, Decode uses mode0 DSL
floating outputs, then Q is quantized at the shared QFM reader. This preserves
the floating-Q contract for FIA Prefill; mixed-batch preparation adds launches
and temporary tensors and needs separate performance measurement. No permanent
Cache format or scale changes. CPU tests cover stage routing, padding, slots,
gapped Cache writes and Decode path selection; NPU/model validation is pending.

Both P and D must install this revision and enable QFM C8. Use the built-in
`MooncakeConnectorV2` (or its `MooncakePullConnector` alias), not MultiConnector,
AscendStore or a custom connector module. Keep community K3 MRV2 hybrid cache,
KDA state, DSpark and PP/DP scheduling configuration; this patch does not
introduce another transfer engine or a permanent BF16 cache.

The handshake identifies full576 E4M3 PA_NZ with unit KV descale. Missing or
different format identifiers fail before transfer. Each descriptor carries
one 128-token payload, its actual byte stride/base pointer, and the number of
kernel pages per manager block. Only payload bytes are transferred; P/D page
padding may differ. Old packed656, BF16 and unpatched endpoints are not peers.

For PD with DCP, `cp_kv_cache_interleave_size` must equal that endpoint's
manager `block_size`, matching the community whole-block ownership mapper.
Both must be positive multiples of 128. This is an explicit limitation of
this candidate, not a claim that all existing C8 DCP settings support PD.
Non-PD DCP behavior is unchanged. KDA and draft cache formats are unchanged.

CPU tests cover the wire contract and real community block-ID mapping with
nonzero offsets, distinct P/D padding, manager blocks 128/256/768 and DCP
1/2/4/8. They do NOT validate network registration, asynchronous completion,
HCCL, graph replay, APC handoff or DSpark acceptance end to end. Start with
P/D TP8 and DCP1, compare output against mixed deployment, then test APC
partial/full hits, chunked prefill, DSpark, graph decode and DCP separately.
