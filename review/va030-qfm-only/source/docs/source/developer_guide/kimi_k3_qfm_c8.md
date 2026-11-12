# Kimi K3 A5 C8: QFM + DSL candidate

Framework base: VA 0.30 commit `c09d3e20b9c45c80b34ae3781440d90904a44023`.
This candidate contains one main-model C8 path for both Prefill and Decode:
DSL Prolog -> full576 FP8 KV Cache -> QFM -> Value up-projection.
It does not include the experimental FIA/QFA readers, BF16 historical
mirrors, stage-specific frontends, or a non-fused C8 frontend.

## Configuration

Merge these settings into both P and D processes:

```json
{
  "enable_kimi_k3_qfm_c8": true,
  "enable_mlapo": true,
  "enable_kv_nz": false
}
```

Keep `--kv-cache-dtype auto` and `--block-size 128` or an admitted positive
multiple. The scheduler may promote its manager block size; kernel pages
remain 128 tokens. The path requires A5, ModelRunner V2 and K3's NoPE
latent512/raw64 MLA. Target projections use the existing ModelSlim dynamic
W8A8 MXFP8 Group32 scheme; `kv_b_proj` remains BF16. Draft/KDA formats are
unchanged. The default feature flag is off.

Remove old experiment options from launch configurations:
`kimi_k3_c8_prefill_backend`, `kimi_k3_c8_fia_form`,
`enable_kimi_k3_mlapo_prefill`, and `enable_kimi_k3_qfm_fia_prefill`.
Do not combine this route with packed656/SFA C8, a separate FA KV quantizer,
or the obsolete `qfm_kv_scale` option. Disabling MLAPO while enabling this
C8 flag is rejected instead of selecting another implementation.

## Storage and attention

The sole persistent main-model MLA cache contains latent512 and raw64,
both in E4M3 FP8 PA_NZ: `[blocks,1,18,128,32]`, 576 effective bytes/token.
The KV descale is the fixed internal value 1.0; there are no Group128 scales
in each cache row. Prolog returns FP8 Q and matching per-token/head scales.
Projection weights and activations retain their independent Group32 MXFP8
scales. This storage format is not the packed656 FP8-latent/BF16-raw format.

Prolog writes directly to dense pages. For physically padded/strided hybrid
pages, it emits only current FP8 rows into a temporary staging buffer and
the tile Scatter writer updates the actual pages. Storage offset, negative
slots and live physical strides are preserved. No historical cache is copied
to BF16. APC uses the physical FP8 pages; block copy-on-write preserves FP8
bytes and manager/kernel page mapping, including a two-kernel-page pool.

Prefill divides Query ranges into virtual sequences of at most 16 rows and
submits them in one batched QFM call. This is not one Attention launch per
virtual sequence and does not create Q-by-K token-index lists. Decode/Verify
and graph/DCP descriptors use their established per-query visibility rules.
Each batch updates device lengths, tables and Query scales; captured buffers
keep stable addresses. DCP gathers Q bytes with matching scales, accounts
for empty shards and merges natural-log LSE outputs before V up-projection.

## PD and dependencies

Built-in MooncakeConnectorV2/MooncakePullConnector transfer only FP8 MLA
payloads: 73728 bytes per 128-token kernel page. P/D format, manager/kernel
mapping and physical strides are checked. KDA and draft transfer protocols
remain unchanged. PD with DCP requires the existing whole-manager-block
interleave contract. PCP, KV layer parallelism and sparse KV offload are not
adapted by this candidate.

The framework patch does not install or replace any CANN/Python packages.
Keep the QFM, metadata and DSL runtime environments that passed the target
container tests. QFM/metadata must be registered in `cann_ops_transformer`;
the separate Stage1 provider wheel and its distribution metadata must be
visible as well as the native vendor packages. Prolog uses the pinned
`ops.quant_mla_prolog` interface from CANNBot source revision `9e01355`.
Its weight layouts, quantization modes and packaged native artifacts are
validated. Missing providers or incompatible dependencies are explicit errors.

CANN version labels alone do not establish interface compatibility. This
candidate does not claim that an old 9.2.0 binary archive works on every
9.2.1 image. Load dependency environments in the actual service launch shell
before starting workers; a source operation in another child shell does not
activate them for the service.

## Candidate validation

The user reported a reduced-layer model at concurrency 32, APC disabled,
PD disaggregation, P/D QFM+DSL: TTFT approximately matched the BF16-cache
baseline and TPOT changed from 4.2 ms to 3.95 ms (about 6% lower). That result is
for the previously delivered runtime at that test point. This pruned
candidate needs a fresh launch and accuracy/performance regression on A5.
It is not a claim of full-model or all-feature-combination acceptance.

CPU tests cover bytes/layout, offsets/padding, virtual sequences, multi-query
verification visibility, graph metadata refresh, scale/head correspondence,
PD contracts, projection weight preparation and APC block copying. CPU tests
do not execute native kernels, HCCL, real network transfers or model accuracy.
The native tile-writer probe is available at
`tests/e2e/nightly/single_node/ops/singlecard_ops/check_kimi_k3_qfm_writer.py`.

After migration, restart all workers and discard old in-memory KV/APC pages.
The installer backs up known affected files, removes known experimental
files and restores unrelated experiment-only changes to the pinned base.
Unknown/partial edits are refused rather than overwritten or reset.
