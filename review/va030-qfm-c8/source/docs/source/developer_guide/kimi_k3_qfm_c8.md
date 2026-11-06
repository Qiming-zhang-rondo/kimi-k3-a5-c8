# Kimi K3 A5 C8 candidate (VA 0.30)

Pinned framework: `c09d3e20b9c45c80b34ae3781440d90904a44023`.
The framework and standalone probes are candidates; CPU regression is not
NPU performance, distributed correctness or model accuracy acceptance.

## Storage and common settings

There is one persistent main-model MLA cache: full576 E4M3 PA_NZ,
`[blocks,1,18,128,32]`, 576 payload bytes/token. Both latent512 and raw64
are FP8, with an internal scalar descale of 1.0. This is different from
packed656 Group128 C8. Do not set the obsolete `qfm_kv_scale` option.
QFM receives separately scaled FP8 Q; W8A8 MXFP8 projection weights retain
their Group32 scales. `kv_b_proj` remains BF16. Draft and KDA caches retain
the upstream formats.

Use A5, MRV2, `--kv-cache-dtype auto`, `enable_kv_nz=false`, and
`--block-size 128` or a positive multiple. The hybrid manager may promote
its block size. Kernel pages remain 128 tokens. The allocator/view/metadata
preserve the actual storage offset and padded page stride; do not derive
an address from the earlier CLI size or divide by a fixed factor.

Both readers retain the existing FP8 tile Scatter writer, shared batch
addresses, negative-slot handling and the manager-to-kernel spec-copy fix.
A stale Scatter plan is checked and rebuilt from the live physical stride;
its warning must not be suppressed to hide a mapping error. Restart workers
after installation; discard all old in-memory KV/APC pages.

## Compare two expanded-FIA Prefill frontends

The existing QFM-only route and absorbed FIA baseline remain available.
`kimi_k3_c8_prefill_backend="fia"` plus `kimi_k3_c8_fia_form="expanded"`
selects the new eager Prefill candidate: unabsorbed Q/K192 (128+raw64),
V128. The default FIA form remains `latent` (QK576/V512), so installing
this patch alone does not silently change an existing benchmark.

Merge these keys into your existing additional configuration:

```json
{
  "enable_kimi_k3_qfm_c8": true,
  "enable_kv_nz": false,
  "enable_mlapo": true,
  "kimi_k3_c8_prefill_backend": "fia",
  "kimi_k3_c8_fia_form": "expanded",
  "enable_kimi_k3_mlapo_prefill": false
}
```

With the last flag **false**, Prefill uses the existing MXFP8 projection
modules and RMSNorm operators. With it **true**, Prefill uses the package
`cann_ops_transformer.mla_prolog` NoPE/V4 wrapper, modes 3/0/0 (MXFP8
projection weights, BF16 Query, BF16 current scratch). This is the native
CANN package implementation, not the DSL frontend used by Decode.
`enable_mlapo=false` globally disables both stage frontends.

The native wrapper normally emits UK-absorbed Q512. The expanded adapter
uses a private UK matrix `[I128 | 0]`, retains the first128 output columns,
and appends raw64. The model's real UK/UV weights are not changed. Native
still computes 128-to-512 and emits Q512 before slicing, so fusion alone
is not a speedup claim. Private NZ weights/scales and the identity matrix
are prepared at loading, not rebuilt per layer invocation. Reloads copy
compatible buffers in place.

The native package contract was checked against ops-transformer v9.2.0,
commit `dff221330eecd2132233b1e73ed1072cd8925140`,
`attention/mla_prolog_v3/torch_extension/mla_prolog.py` and its C++ wrapper
for `MlaPrologV4WeightNz`. Required geometry is hidden7168 (or another
admitted native size), q-rank1536/2048, NoPE128/raw64 and BF16 norm weights.
A container must actually provide the matching wrapper and native runtime;
checking source documentation does not establish local availability. The
standalone probe verifies the installed implementation before model use.
No new DSL or custom Prolog compilation is added by this patch.

### Historical and current KV

Historical FP8 pages, including APC-hit prefixes, are gathered from the live
page table, converted directly to BF16, and up-projected into temporary
per-head K/V. The historical KV axis is bounded by a workspace budget derived
from VA's existing MLA workspace sizing, including head expansion costs.
Every FIA call retains the complete Query chunk. There is no Q-by-K index
matrix and no limit of eight Query rows. History partial outputs are combined
in FP32 using natural-log LSE weights.

Current rows use the original BF16 latent from this step, up-project it for
causal FIA, and also write the persistent FP8 cache. They are not counted a
second time in history. Native Prolog uses two bounded, current-chunk BF16
scratch buffers; ordinary preprocessing also has temporary BF16 tensors.
Neither path allocates a second permanent BF16 history cache. Temporary
workspace, current inputs, private weights and native kernel workspace still
contribute to peak memory.

Expanded output already has V128 and bypasses latent V-up projection.
QFM Decode output remains latent512 and still applies the original V-up.
Do not compare only the FIA kernel: include projections, cache writing,
history conversion/expansion, FIA and output merging in the timing interval.

## Decode, mixed batches and feature boundaries

Pure Decode and main-model DSpark verification retain the existing DSL
Prolog mode2 plus QFM, including Decode graph capture/replay. Source the
same pinned DSL/QFM runtime environment used by the accepted configuration.
The Prefill flag never sends Decode through native BF16-output mode0.
Mixed batches preserve Decode FP8 Q codes/scales without re-quantization;
CPU scheduler stage flags select the Prefill ranges, including short final
chunks. Pure Decode does not construct the new Prefill ranges or load
historical expanded K/V.

For DCP, head weights are gathered once on every rank before the first
Prefill Query gather, in the same rank/head order, and invalidated on reload.
Each rank reads only local history. Current BF16 KV contributes on rank zero
only. Empty shards produce zero output and -infinity LSE. Prefill V128 and
Decode latent512 share the existing 512-wide opaque merge container; unused
Prefill columns are zero. After the common DCP merge, only Decode rows get
V-up projection. CPU tests simulate these contracts; actual HCCL/TP/DCP
execution and performance remain unverified for the new expanded route.

APC hashes still use token IDs and prefix ancestry. The persistent FP8 layout,
unit scale, page tables and prefix lengths do not change between these two
frontends. Partial hits read FP8 history and current chunks stay BF16 for
Attention. This wiring requires model validation with partial and full hits.

FIA Prefill remains eager; Prefill graph capture is explicitly rejected.
FULL_DECODE_ONLY still uses the existing QFM graph route. PCP, KV layer
parallelism, sparse offload and speculative methods other than DSpark are
not adapted and retain explicit guards.

The existing PD candidate supports built-in MooncakeConnectorV2 or its
MooncakePullConnector alias on both endpoints. Its full576/unit-scale format
handshake, physical payload/stride descriptors and mapping are unchanged.
PD with DCP requires `cp_kv_cache_interleave_size == manager block_size`.
Custom/pool connectors are not admitted. Network completion, KDA/draft
handoff and end-to-end PD acceptance are separate from CPU wire tests.

## Validation and performance comparison

1. Run the standalone expanded-FIA probe in the target CANN environment.
   It compares ordinary and native frontends with real K3 projection geometry,
   independent encoded-weight FP32 references, paged FP8 history, causal
   current KV, nonzero offset, page gaps and unchanged cache guards.
2. Use its optional benchmark only after correctness passes. Exclude first
   import/JIT/warmup; compare complete pipeline NPU and synchronized wall
   times, peak memory, and the same Q/history/head sizes. This is single-card
   eager testing, not a TP, DCP, graph or DSpark acceptance result.
3. Install the framework candidate; restart every worker. Confirm startup
   `fia_form=expanded prefill_frontend=ordinary` or `native-prolog`.
   Keep weights, DSpark settings, APC state, chunk size, concurrency and prompt
   lengths fixed. Compare the two Prefill flags, then the existing latent and
   BF16-cache baselines. Measure TTFT, TPOT, acceptance lengths and peak memory.
4. Recheck mixed-stage batches, APC partial/full hits, long chunks, Decode graph
   replay and actual DCP/PD separately. Source-level isolation does not prove
   zero TPOT impact in a scheduler with concurrent Prefill and Decode.

The guarded installer accepts a clean pinned checkout or a recognized complete
previous delivery, keeps backups, and refuses unknown or mixed modifications.
It does not install native packages, compile operators, launch services or
run NPU tests automatically.
