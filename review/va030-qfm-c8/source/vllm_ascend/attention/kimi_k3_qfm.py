# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in K3 full576 FP8 cache and optional paged BF16 Prefill history.

KV uses unscaled FP8, matching the pinned CANN K3 recipe's unit descale.
Caller-owned cache pages and their padding/offset are preserved. Metadata is
shared across layers in an attention group; only captured capacities are kept.
"""

import importlib
import math
import warnings
from dataclasses import dataclass, field

import torch
import torch_npu

QFM_WIDTH = 576
QFM_VALUE = 512
QFM_BLOCK = 128
QFM_TILE = 32
QFM_MAX_Q = 16
QFM_MASK = 2048
QFM_FP8_MAX = 448.0
# cann-recipes-infer 4447f2f0972cb9287e069e3994b1d46ffe244ebc:
# KimiMLAAttention.prepare_prolog_weights registers mla_kv_scale=[1.0].
# This is an identity conversion, not a calibrated model accuracy guarantee.
QFM_KV_DESCALE = 1.0
QFM_HEADS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def select_qfm(impl, config) -> bool:
    """Do not turn on the draft model or unrelated attention layers."""
    if not config.enable_kimi_k3_qfm_c8 or impl.is_draft_model:
        return False
    # MRV2 DSpark retains the target model_config while constructing its
    # draft. Kimi also sets draft RoPE only AFTER the MLA impl is created.
    # Neither runner_type nor use_mla_rope identifies that construction
    # reliably. Read the loader's live tag (never import its value by copy)
    # before validating/choosing a target-only QFM cache layout.
    if importlib.import_module("vllm.compilation.backends").model_tag == "dspark_head":
        return False
    if config.enable_sparse_sfa_c8:
        raise ValueError("Disable enable_sparse_sfa_c8: packed656 and full576 QFM are different cache formats")
    vcfg = impl.vllm_config
    if impl.use_mla_rope or impl.kv_lora_rank != QFM_VALUE or impl.qk_rope_head_dim != 64:
        raise ValueError("K3 QFM requires No-RoPE MLA latent512 + raw64")
    if not vcfg.use_v2_model_runner or not impl.support_fp8_attention:
        raise ValueError("K3 QFM requires A5 and ModelRunner V2")
    if impl.num_kv_heads != 1 or impl.num_heads not in QFM_HEADS:
        raise ValueError(f"QFM unsupported local query/KV heads: {impl.num_heads}/{impl.num_kv_heads}")
    kernel_pages_per_manager(vcfg.cache_config.block_size)
    if vcfg.cache_config.cache_dtype != "auto" or impl.fa_quant_layer:
        raise ValueError("QFM owns full576 FP8 storage: use --kv-cache-dtype auto and no legacy FA quant scheme")
    if impl.enable_kv_nz:
        raise ValueError("QFM owns its PA_NZ layout; use enable_kv_nz=false")
    if impl.pcp_enabled or impl.is_pcp_decode_sharded:
        raise ValueError("QFM PCP is not adapted in this candidate")
    if config.kvpp_config.size != 1:
        raise ValueError("QFM KV layer parallelism is not adapted in this candidate")
    if vcfg.kv_transfer_config is not None:
        transfer = vcfg.kv_transfer_config
        if (
            transfer.kv_connector not in ("MooncakeConnectorV2", "MooncakePullConnector")
            or getattr(transfer, "kv_connector_module_path", None) is not None
        ):
            raise ValueError("QFM PD requires the built-in MooncakeConnectorV2 (not a custom/pool connector)")
        # Community Mooncake maps DCP ownership at manager-block granularity.
        # Sub-block token striping cannot be copied as whole PA_NZ pages.
        if (
            vcfg.parallel_config.decode_context_parallel_size > 1
            and vcfg.parallel_config.cp_kv_cache_interleave_size != vcfg.cache_config.block_size
        ):
            raise ValueError("QFM PD with DCP requires cp_kv_cache_interleave_size == block_size")
    if config.sparse_kv_offload_config.enabled:
        raise ValueError("QFM does not use sparse KV offload")
    if vcfg.speculative_config is not None and vcfg.speculative_config.method != "dspark":
        raise ValueError("QFM speculative path is currently scoped to DSpark")
    size = vcfg.parallel_config.decode_context_parallel_size
    if impl.num_heads * size not in QFM_HEADS:
        raise ValueError("QFM does not support this DCP-gathered query head count")
    backend = getattr(config, "kimi_k3_c8_prefill_backend", "qfm")
    if backend not in ("fia", "qfm", "qfa", "fia_paged"):
        raise ValueError("kimi_k3_c8_prefill_backend must be fia, qfa, qfm or fia_paged")
    impl.qfm_fia_paged = backend == "fia_paged"
    if impl.qfm_fia_paged and vcfg.kv_transfer_config is not None:
        if vcfg.kv_transfer_config.kv_role != "kv_producer":
            raise ValueError("fia_paged is a Prefill producer backend; use qfm on KV consumers")
    impl.qfm_fia_prefill = backend in ("fia", "qfa", "fia_paged") or getattr(
        config, "enable_kimi_k3_qfm_fia_prefill", False
    )
    impl.qfm_mlapo_prefill = getattr(config, "enable_kimi_k3_mlapo_prefill", True)
    form = getattr(config, "kimi_k3_c8_fia_form", "latent")
    if form not in ("latent", "expanded"):
        raise ValueError("kimi_k3_c8_fia_form must be latent or expanded")
    if backend == "qfa" and form != "expanded":
        raise ValueError("QFA Prefill requires kimi_k3_c8_fia_form=expanded (temporary Q/K/V padded to 256)")
    if impl.qfm_fia_paged and form != "latent":
        raise ValueError("fia_paged uses latent FIA: set kimi_k3_c8_fia_form=latent")
    impl.qfm_qfa_prefill = backend == "qfa"
    impl.qfm_fia_expanded = impl.qfm_fia_prefill and form == "expanded"
    if impl.qfm_fia_expanded and (impl.qk_nope_head_dim != 128 or impl.v_head_dim != 128):
        raise ValueError("Expanded K3 FIA requires QK192 (128+64) and V128")
    return True


def kernel_pages_per_manager(manager_block_size):
    """The KV manager owns large blocks; QFM consumes fixed 128-token pages."""
    if (
        isinstance(manager_block_size, bool)
        or not isinstance(manager_block_size, int)
        or manager_block_size <= 0
        or manager_block_size % QFM_BLOCK
    ):
        raise ValueError(
            f"QFM requires --block-size to be a positive multiple of {QFM_BLOCK}; "
            f"resolved manager block size is {manager_block_size}"
        )
    return manager_block_size // QFM_BLOCK


def align_qfm_page_bytes(page_bytes, manager_block_sizes):
    """Align the shared physical page before cache budgeting/allocation."""
    alignment = math.lcm(*(kernel_pages_per_manager(size) * QFM_TILE for size in manager_block_sizes))
    return (page_bytes + alignment - 1) // alignment * alignment


def cache_view_for_manager(raw, manager_block_size, page_bytes, kernel_block_size=QFM_BLOCK):
    """Split final allocator pages into uniformly strided QFM kernel pages.

    MRV2 BlockTables already expands manager ID b to b*ratio+[0..ratio-1]
    and computes slots with kernel size 128. Do not expand either again in
    QFM metadata. Page padding is distributed between the kernel views, so
    ratio*kernel_stride is exactly the allocator's physical manager stride.
    """
    ratio = kernel_pages_per_manager(manager_block_size)
    if kernel_block_size != QFM_BLOCK:
        raise ValueError(f"QFM needs kernel block size {QFM_BLOCK}, got {kernel_block_size}")
    if page_bytes < manager_block_size * QFM_WIDTH or page_bytes % (ratio * QFM_TILE):
        raise ValueError(
            f"QFM physical manager page {page_bytes} bytes must cover {manager_block_size * QFM_WIDTH} "
            f"payload bytes and divide into {ratio} kernel pages with {QFM_TILE}-byte aligned strides"
        )
    if raw.numel() == 0 or raw.numel() % page_bytes:
        raise ValueError("QFM raw buffer must contain whole physical manager pages")
    return cache_view(raw, raw.numel() // page_bytes * ratio, page_bytes // ratio)


def cache_view(raw, blocks, page_bytes):
    """One physical page per block, with offset inherited from the allocator."""
    if raw.dtype not in (torch.uint8, torch.int8) or raw.ndim != 1:
        raise ValueError("QFM allocator requires a single byte buffer")
    if page_bytes < QFM_BLOCK * QFM_WIDTH or page_bytes % QFM_TILE:
        raise ValueError("QFM physical page must hold full576 FP8 and align to 32 bytes")
    needed = (blocks - 1) * page_bytes + QFM_BLOCK * QFM_WIDTH
    if blocks <= 0 or needed > raw.numel() or raw.storage_offset() % QFM_TILE:
        raise ValueError("QFM cache allocation is too small or unaligned")
    return torch.as_strided(
        raw.view(torch.float8_e4m3fn),
        (blocks, 1, QFM_WIDTH // QFM_TILE, QFM_BLOCK, QFM_TILE),
        (page_bytes, QFM_BLOCK * QFM_WIDTH, QFM_BLOCK * QFM_TILE, QFM_TILE, 1),
    )


@dataclass
class QFMScatterPlan:
    slots: torch.Tensor
    offsets: torch.Tensor
    stride: int
    fallback_reported: bool = False

    def slice(self, begin, end):
        return QFMScatterPlan(self.slots[begin:end], self.offsets[begin:end], self.stride)


def build_scatter_offsets(slots, stride, tile=None):
    """Pure slot arithmetic, shared by every MLA layer in a metadata group."""
    ids = slots.to(torch.int64)
    if tile is None:
        tile = torch.arange(QFM_WIDTH // QFM_TILE, device=slots.device)
    offsets = ids[:, None].div(QFM_BLOCK, rounding_mode="floor") * (stride // QFM_TILE)
    offsets = offsets + tile[None, :] * QFM_BLOCK + ids[:, None].remainder(QFM_BLOCK)
    return torch.where(ids[:, None] >= 0, offsets, -1)


def write_cache(rows, slots, cache, scale, scatter_plan=None):
    values = (rows.float() / scale).clamp(-QFM_FP8_MAX, QFM_FP8_MAX).to(torch.float8_e4m3fn)
    scatter_fp8_cache(values, slots, cache, scatter_plan=scatter_plan)


def scatter_fp8_cache(values, slots, cache, scatter_plan=None):
    """Scatter 32-byte NZ tiles; native ScatterPaCache skips slot -1.

    An offset-preserving contiguous alias avoids the old DSpark stride bug.
    Invalid tokens do not alias a live token or depend on duplicate-write order.
    """
    if values.dtype != torch.float8_e4m3fn or values.shape != (slots.numel(), QFM_WIDTH):
        raise ValueError("QFM scatter expects already quantized full576 rows")
    if cache.dtype != torch.float8_e4m3fn or cache.shape[1:] != (1, 18, 128, 32):
        raise ValueError("QFM requires full576 FP8 PA_NZ cache, not legacy packed656")
    stride = cache.stride(0)
    if cache.stride()[2:] != (4096, 32, 1) or stride % QFM_TILE:
        raise ValueError("Invalid QFM PA_NZ strides")
    span = (cache.shape[0] - 1) * stride + QFM_BLOCK * QFM_WIDTH
    alias = torch.as_strided(cache, (span // QFM_TILE, 1, 1, QFM_TILE), (32, 32, 32, 1))
    if scatter_plan is None:
        offsets = build_scatter_offsets(slots, stride)
    else:
        offsets = scatter_plan.offsets
        if (
            scatter_plan.stride != stride
            or scatter_plan.slots.data_ptr() != slots.data_ptr()
            or scatter_plan.slots.shape != slots.shape
            or offsets.shape != (slots.numel(), QFM_WIDTH // QFM_TILE)
            or offsets.dtype != torch.int64
            or offsets.device != cache.device
            or not offsets.is_contiguous()
        ):
            # A shared plan is an optimization, not the address authority.
            # Recompute from the LIVE slots and physical view. Keep this work
            # inside the captured graph too: caching the rebuilt addresses in
            # Python would leave them stale when graph replay updates slots.
            if not scatter_plan.fallback_reported:
                mismatches = {
                    "stride": scatter_plan.stride != stride,
                    "slots_pointer": scatter_plan.slots.data_ptr() != slots.data_ptr(),
                    "slots_shape": scatter_plan.slots.shape != slots.shape,
                    "offsets_shape": offsets.shape != (slots.numel(), QFM_WIDTH // QFM_TILE),
                    "offsets_dtype": offsets.dtype != torch.int64,
                    "offsets_device": offsets.device != cache.device,
                    "offsets_contiguous": not offsets.is_contiguous(),
                }
                warnings.warn(
                    "QFM Scatter plan fallback: "
                    f"mismatch={[name for name, bad in mismatches.items() if bad]}; "
                    f"plan_stride={scatter_plan.stride}, cache_stride={stride}; "
                    f"plan_slots={tuple(scatter_plan.slots.shape)}, live_slots={tuple(slots.shape)}; "
                    f"offsets={tuple(offsets.shape)}/{offsets.dtype}/{offsets.device}, cache_device={cache.device}. "
                    "Rebuilding offsets from live slots; no cache layout or quantization change.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                scatter_plan.fallback_reported = True
            offsets = build_scatter_offsets(slots, stride)
    torch_npu.npu_scatter_pa_cache(values.reshape(-1, 1, QFM_TILE).contiguous(), offsets.flatten(), key_cache=alias)


def load_ops():
    # Load only after selecting the worker device. Never catch kernel failures.
    importlib.import_module("cann_ops_transformer.ops")
    return torch.ops.cann_ops_transformer.quant_flash_mla, torch.ops.cann_ops_transformer.quant_flash_mla_metadata


@dataclass
class QFMBatch:
    cu: torch.Tensor
    used: torch.Tensor
    lengths: torch.Tensor
    table: torch.Tensor
    slots: torch.Tensor
    live: torch.Tensor
    visible: torch.Tensor
    schedule: torch.Tensor
    mask: torch.Tensor | None
    mask_mode: int
    max_q: int
    max_k: int
    has_prefill: bool
    # (query begin, query end, local history length, original request page table).
    fia_requests: tuple = ()
    qfm_needed: bool = True
    # CPU scheduler stage ranges: (begin, end, is_prefill), including padding.
    preprocess_segments: tuple = ()
    invalid: torch.Tensor | None = None
    invisible: torch.Tensor | None = None
    fia_mask: torch.Tensor | None = None
    scatter_plan: QFMScatterPlan | None = None
    prolog_kwargs: dict = field(default_factory=dict)
    # Rebuilt with the live request page tables each batch; shared by layers.
    fia_history_plans: tuple = ()
    # Optional QFA schedules depend on lengths/layouts, never layer Q/K/V.
    qfa_metadata: dict = field(default_factory=dict)
    qfa_mask: torch.Tensor | None = None


class QFMMetadataBuilder:
    def __init__(self, impl, device, kv_cache_spec=None):
        self.device = device
        self.dcp_size = impl.vllm_config.parallel_config.decode_context_parallel_size
        self.dcp_rank = getattr(impl, "dcp_rank", 0)
        self.interleave = impl.vllm_config.parallel_config.cp_kv_cache_interleave_size
        self.heads = impl.num_heads * self.dcp_size
        self.max_k = max(1, impl.vllm_config.model_config.max_model_len)
        _, self.metadata_op = load_ops()
        self.mask = torch.ones((QFM_MASK, QFM_MASK), dtype=torch.int8, device=device).triu_(1)
        self.buffers = {}
        self.fia_prefill = getattr(impl, "qfm_fia_prefill", False)
        self.mlapo_prefill = getattr(impl, "qfm_mlapo_prefill", True)
        self.expanded_fia = getattr(impl, "qfm_fia_expanded", False)
        self.qfa_prefill = getattr(impl, "qfm_qfa_prefill", False)
        self.paged_fia = getattr(impl, "qfm_fia_paged", False)
        self.fia_mask = self.mask.to(torch.bool) if self.fia_prefill else None
        self.cache_stride = None
        self.has_prolog = getattr(impl, "enable_mlapo", False)
        self.scatter_possible = self.staging_required = False
        self.scatter_tiles = None
        if kv_cache_spec is not None:
            ratio = kernel_pages_per_manager(kv_cache_spec.block_size)
            physical_page = kv_cache_spec.page_size_padded or kv_cache_spec.page_size_bytes
            if physical_page % ratio:
                raise ValueError("QFM physical page does not split into equal kernel strides")
            self.cache_stride = physical_page // ratio
            gapped = self.cache_stride != QFM_BLOCK * QFM_WIDTH
            self.scatter_possible = gapped or not self.has_prolog or not self.mlapo_prefill or self.fia_prefill
            self.staging_required = self.has_prolog and (gapped or self.fia_prefill)
            if self.scatter_possible:
                self.scatter_tiles = torch.arange(QFM_WIDTH // QFM_TILE, device=device)

    def metadata(self, lengths, cu, used, max_q, mode):
        return self.metadata_op(
            lengths,
            self.heads,
            1,
            1,
            cu_seqlens_q=cu,
            seqused_q=used,
            max_seqlen_q=max_q,
            max_seqlen_kv=self.max_k,
            head_dim_qk=QFM_WIDTH,
            head_dim_v=QFM_VALUE,
            mask_mode=mode,
            layout_q="TND",
        )

    def build(self, common, retain_for_graph=False):
        tokens = max(common.num_actual_tokens, common.num_input_tokens)
        # These are already KERNEL-page columns from MRV2 BlockTables, even
        # when the final cache manager uses 256/512/... token blocks.
        columns = common.block_table_tensor.shape[1]
        key = tokens, columns, common.causal
        batch = self.buffers.get(key)
        flags = common.is_prefilling
        if flags is not None and flags.device.type != "cpu":
            raise ValueError("QFM expects scheduler CPU stage flags, not a device-to-host sync")
        fia_only = self.fia_prefill and flags is not None and bool(flags[: common.num_reqs].all())
        if fia_only and (retain_for_graph or batch is not None):
            raise ValueError("FIA Prefill is eager only; Decode-only graphs remain supported")
        # All decode graph capacities use one physical query per virtual row.
        # Eager prefill batches use <=16 rows per sequence, NOT small op calls.
        per_token = not fia_only and (retain_for_graph or batch is not None or self.dcp_size > 1)
        if fia_only:
            # No QFM call consumes these descriptors. Keep one inert row;
            # never expand Q-by-page tables or <=16-token virtual sequences.
            cu_cpu = [0, tokens]
            rows = 1
        elif per_token:
            rows = tokens
        else:
            bounds = common.query_start_loc_cpu[: common.num_reqs + 1].clamp_max(common.num_actual_tokens).tolist()
            cu_cpu = [0]
            for begin, end in zip(bounds, bounds[1:]):
                while begin < end:
                    begin = min(begin + QFM_MAX_Q, end)
                    cu_cpu.append(begin)
            while cu_cpu[-1] < tokens:
                cu_cpu.append(min(cu_cpu[-1] + QFM_MAX_Q, tokens))
            rows = len(cu_cpu) - 1
        if not 0 < rows < 65536:
            raise ValueError("QFM virtual batch must contain 1..65535 rows")
        mode = 0 if per_token or not common.causal else 3
        ints = dict(dtype=torch.int32, device=self.device)
        if batch is None:
            cu = torch.arange(rows + 1, **ints) if per_token else torch.tensor(cu_cpu, **ints)
            used, lengths = torch.zeros(rows, **ints), torch.zeros(rows, **ints)
            # Populate once after device lengths are ready. Do not invoke the
            # scheduling kernel twice or require a separate Meta registration.
            schedule = torch.empty(0, **ints)
            batch = QFMBatch(
                cu,
                used,
                lengths,
                torch.zeros((0, 0) if fia_only else (rows, columns), **ints),
                torch.full((tokens,), -1, dtype=torch.int64, device=self.device),
                torch.zeros(tokens, dtype=torch.bool, device=self.device),
                torch.zeros(tokens, dtype=torch.bool, device=self.device),
                schedule,
                self.mask if mode == 3 else None,
                mode,
                1 if per_token else QFM_MAX_Q,
                self.max_k,
                False,
                invalid=torch.empty(tokens, dtype=torch.bool, device=self.device),
                invisible=torch.empty(tokens, dtype=torch.bool, device=self.device),
                fia_mask=self.fia_mask,
                qfa_mask=self.mask if self.qfa_prefill else None,
            )
            if self.scatter_possible:
                batch.scatter_plan = QFMScatterPlan(
                    batch.slots,
                    torch.empty((tokens, QFM_WIDTH // QFM_TILE), dtype=torch.int64, device=self.device),
                    self.cache_stride,
                )
                batch.prolog_kwargs["scatter_plan"] = batch.scatter_plan
            if self.staging_required:
                batch.prolog_kwargs["staging_indices"] = torch.arange(tokens, dtype=torch.int64, device=self.device)
            if retain_for_graph:
                self.buffers[key] = batch
        # Only device lengths: DSpark rejection correction may not be mirrored
        # to CPU. Updates are on the caller's stream before graph replay.
        boundaries = common.query_start_loc[: common.num_reqs + 1].contiguous()
        token_ids = batch.cu[:-1] if per_token else torch.arange(tokens, device=self.device, dtype=torch.int32)
        owners = torch.searchsorted(boundaries[1:], token_ids, right=True).clamp_max(common.num_reqs - 1).long()
        starts, ends = boundaries[owners], boundaries[owners + 1]
        seq = common.seq_lens[owners]
        live = (token_ids < common.num_actual_tokens) & (token_ids < ends) & (seq > 0)
        batch.live.copy_(live)
        batch.invalid.copy_(~live)
        count = min(tokens, common.slot_mapping.numel())
        if count < tokens:
            batch.slots[count:].fill_(-1)
        batch.slots[:count].copy_(common.slot_mapping[:count])
        batch.slots.masked_fill_(batch.invalid, -1)
        lengths_per_token = seq - (ends - starts) + token_ids - starts + 1 if common.causal else seq
        if self.dcp_size > 1:
            cycle = self.interleave * self.dcp_size
            lengths_per_token = lengths_per_token.div(cycle, rounding_mode="floor") * self.interleave + (
                lengths_per_token.remainder(cycle) - self.dcp_rank * self.interleave
            ).clamp(0, self.interleave)
        lengths_per_token = torch.where(live, lengths_per_token.clamp_min(0), 0)
        batch.visible.copy_(live & (lengths_per_token > 0))
        batch.invisible.copy_(~batch.visible)
        if fia_only:
            row_owner = None
        elif per_token:
            # Each virtual row contains exactly this token. The general
            # first/last-token gathers reduce to identity, even for padding
            # and DCP ranks with zero local history.
            row_owner = owners
            batch.used.copy_(live)
            batch.lengths.copy_(lengths_per_token)
        else:
            first = batch.cu[:-1].long()
            row_owner = owners[first]
            row_used = (ends[first] - batch.cu[:-1]).clamp_min(0)
            row_used = torch.minimum(row_used, batch.cu[1:] - batch.cu[:-1])
            row_used = torch.where(live[first], row_used, 0)
            last = (batch.cu[:-1] + row_used - 1).clamp_min(0).long()
            batch.used.copy_(row_used)
            batch.lengths.copy_(torch.where(row_used > 0, lengths_per_token[last], 0))
        if not fia_only:
            batch.table.copy_(common.block_table_tensor[row_owner])
        batch.has_prefill = bool(flags[: common.num_reqs].any()) if flags is not None else common.max_query_len > 16
        batch.preprocess_segments = ()
        if not self.mlapo_prefill or self.expanded_fia or self.paged_fia:
            if flags is None:
                raise ValueError("Stage-specific MLAPO requires scheduler Prefill flags")
        if (not self.mlapo_prefill or self.expanded_fia or self.paged_fia) and batch.has_prefill:
            bounds = common.query_start_loc_cpu[: common.num_reqs + 1].clamp_max(common.num_actual_tokens).tolist()
            segments = []
            for begin, end, prefill in zip(bounds, bounds[1:], flags[: common.num_reqs].tolist()):
                if begin == end:
                    continue
                if segments and segments[-1][2] == prefill:
                    segments[-1] = (segments[-1][0], end, prefill)
                else:
                    segments.append((begin, end, prefill))
            if segments:
                segments[-1] = (segments[-1][0], tokens, segments[-1][2])
            batch.preprocess_segments = tuple(segments)
        batch.fia_requests = ()
        batch.fia_history_plans = ()
        batch.qfm_needed = True
        if self.fia_prefill:
            if flags is None:
                raise ValueError("FIA Prefill routing requires explicit scheduler stage flags")
            selected = flags[: common.num_reqs].bool()
            if bool(selected.any()):
                if retain_for_graph or batch is self.buffers.get(key):
                    raise ValueError("FIA Prefill is eager only; Decode-only graphs remain supported")
                if not common.causal:
                    raise ValueError("FIA Prefill currently requires causal attention")
                # One metadata-time host synchronization, never once per layer.
                # Device lengths include DSpark corrections that CPU mirrors may miss.
                seq_cpu = common.seq_lens[: common.num_reqs].detach().cpu().tolist()
                bounds = common.query_start_loc_cpu[: common.num_reqs + 1].tolist()
                requests = []
                for owner in selected.nonzero().flatten().tolist():
                    begin, end = bounds[owner : owner + 2]
                    if begin == end:
                        continue
                    history = seq_cpu[owner] - (end - begin)
                    if history < 0 or end > common.num_actual_tokens:
                        raise ValueError("Inconsistent FIA Prefill query/history lengths")
                    cycle = self.interleave * self.dcp_size
                    local_history = (history // cycle) * self.interleave + min(
                        max(history % cycle - self.dcp_rank * self.interleave, 0), self.interleave
                    )
                    requests.append((begin, end, local_history, common.block_table_tensor[owner]))
                batch.fia_requests = tuple(requests)
                if not self.paged_fia:
                    batch.fia_history_plans = tuple(
                        build_fia_history_plan(table, length) for _, _, length, table in requests
                    )
                if not fia_only:
                    selected_device = selected.to(self.device)
                    skip = selected_device[row_owner]
                    batch.used.masked_fill_(skip, 0)
                    batch.lengths.masked_fill_(skip, 0)
                batch.qfm_needed = not bool(selected.all())
        if batch.scatter_plan is not None and (
            not self.has_prolog or self.cache_stride != QFM_BLOCK * QFM_WIDTH or batch.has_prefill
        ):
            batch.scatter_plan.offsets.copy_(build_scatter_offsets(batch.slots, self.cache_stride, self.scatter_tiles))
        if batch.qfm_needed:
            schedule = self.metadata(batch.lengths, batch.cu, batch.used, batch.max_q, mode)
            if batch.schedule.numel() == 0:
                batch.schedule = schedule
            else:
                batch.schedule.copy_(schedule)
        return batch


@dataclass
class FIAHistoryPlan:
    pages: torch.Tensor
    positions: torch.Tensor
    tiles: torch.Tensor


def build_fia_history_plan(table, length):
    """Prepare O(history) gather coordinates once per request and batch.

    Cache values remain layer-specific. Never retain this plan across batch
    updates: APC reuse, preemption and DSpark can change the live page table.
    """
    position = torch.arange(length, device=table.device)
    pages = table[position.div(QFM_BLOCK, rounding_mode="floor")].long()
    return FIAHistoryPlan(
        pages,
        position.remainder(QFM_BLOCK),
        torch.arange(QFM_WIDTH // QFM_TILE, device=table.device),
    )


def read_fia_history(cache, table, length, scale, *, plan=None, unit_scale=False):
    """Gather only live history, preserving NZ strides and storage offset.

    Index FP8 as bytes (advanced FP8 indexing is not supported everywhere).
    The result is temporary BF16 latent512+raw64, not a second cache pool.
    """
    if plan is None:
        plan = build_fia_history_plan(table, length)
    packed = cache.view(torch.uint8)[plan.pages[:, None], 0, plan.tiles[None, :], plan.positions[:, None], :]
    values = packed.view(torch.float8_e4m3fn).reshape(length, 1, QFM_WIDTH)
    if unit_scale:
        # Production QFM fixes KV descale to 1 at weight initialization.
        # No device scalar read, multiply or FP32 intermediate is needed.
        return values.to(torch.bfloat16)
    return (values.float() * scale).to(torch.bfloat16)


def read_fia_history_parts(cache, plan):
    """Gather/cast into FIA's contiguous components without full576 BF16.

    Production history has unit descale. Each result is already contiguous
    in token-major order, avoiding another whole-history split/copy in FIA.
    """
    source = cache.view(torch.uint8)
    length = plan.pages.numel()
    latent_tiles = QFM_VALUE // QFM_TILE
    latent = source[plan.pages[:, None], 0, plan.tiles[None, :latent_tiles], plan.positions[:, None], :]
    latent = latent.view(torch.float8_e4m3fn).reshape(length, 1, QFM_VALUE).to(torch.bfloat16)
    raw = source[plan.pages[:, None], 0, plan.tiles[None, latent_tiles:], plan.positions[:, None], :]
    raw = raw.view(torch.float8_e4m3fn).reshape(length, 1, QFM_WIDTH - QFM_VALUE).to(torch.bfloat16)
    return latent, raw


def fia_partial(q, kv, scale, mask=None, *, prepared_q=None, prepared_kv=None):
    """Use the same latent512/raw64 TND FIA-v2 contract as MLA CP."""
    if kv.shape[0] == 0:
        return (
            torch.zeros((*q.shape[:2], QFM_VALUE), dtype=torch.bfloat16, device=q.device),
            torch.full((*q.shape[:2], 1), -float("inf"), device=q.device),
        )
    if prepared_kv is None:
        prepared_kv = (kv[..., :QFM_VALUE].contiguous(), kv[..., QFM_VALUE:].contiguous())
    latent = prepared_kv[0]
    if prepared_q is None:
        prepared_q = (
            q[..., :QFM_VALUE].to(torch.bfloat16).contiguous(),
            q[..., QFM_VALUE:].to(torch.bfloat16).contiguous(),
        )
    output, lse = torch_npu.npu_fused_infer_attention_score_v2(
        prepared_q[0],
        latent,
        latent,
        query_rope=prepared_q[1],
        key_rope=prepared_kv[1],
        num_query_heads=q.shape[1],
        num_key_value_heads=1,
        input_layout="TND",
        atten_mask=mask,
        sparse_mode=3 if mask is not None else 0,
        softmax_scale=scale,
        actual_seq_qlen=[q.shape[0]],
        actual_seq_kvlen=[kv.shape[0]],
        return_softmax_lse=True,
    )
    # FIA-v2 TND returns token-major output and token-major LSE.
    return output.reshape(q.shape[0], q.shape[1], QFM_VALUE), lse.reshape(q.shape[0], q.shape[1], 1)


def merge_fia_parts(left, right):
    """Combine independently normalized Attention outputs in natural-log space."""
    a, la = left
    b, lb = right
    total = torch.logaddexp(la, lb)
    # Empty+empty has -inf LSE: replace NaN weights with zero.
    wa = torch.nan_to_num(torch.exp(la - total), nan=0.0)
    wb = torch.nan_to_num(torch.exp(lb - total), nan=0.0)
    return (a.float() * wa + b.float() * wb).to(a.dtype), total


def fill_fia_prefill(impl, q, kv, cache, batch, latent, lse):
    """History is local FP8; current floating KV contributes on DCP rank zero.

    All ranks run the same final DCP merge. No per-query FIA loops, no dense
    Q-by-history index lists. Process one request at a time to bound workspace.
    """
    mask = getattr(batch, "fia_mask", None)
    if mask is None:
        mask = torch.ones((QFM_MASK, QFM_MASK), dtype=torch.bool, device=q.device).triu_(1)
    plans = getattr(batch, "fia_history_plans", ())
    for index, (begin, end, history_length, table) in enumerate(batch.fia_requests):
        query = q[begin:end]
        if history_length == 0 and impl.dcp_enabled and impl.dcp_rank != 0:
            # This rank contributes neither history nor current. Preserve the
            # final collective, but skip Q packing and temporary neutral rows.
            latent[begin:end].zero_()
            lse[begin:end].fill_(-float("inf"))
            continue
        prepared_q = (
            query[..., :QFM_VALUE].to(torch.bfloat16).contiguous(),
            query[..., QFM_VALUE:].to(torch.bfloat16).contiguous(),
        )
        result = None
        if history_length:
            plan = plans[index] if plans else build_fia_history_plan(table, history_length)
            history = read_fia_history_parts(cache, plan)
            result = fia_partial(query, history[0], impl.scale, prepared_q=prepared_q, prepared_kv=history)
            del history
        if not impl.dcp_enabled or impl.dcp_rank == 0:
            current = kv[begin:end].reshape(end - begin, 1, QFM_WIDTH).to(torch.bfloat16)
            current_result = fia_partial(query, current, impl.scale, mask, prepared_q=prepared_q)
            result = current_result if result is None else merge_fia_parts(result, current_result)
        latent[begin:end].copy_(result[0])
        lse[begin:end].copy_(result[1])


def ordinary_qfm_inputs(impl, hidden, tokens):
    """Existing projection modules for the non-fused writer."""
    if impl.fused_qkv_a_proj is not None:
        projected = impl.fused_qkv_a_proj(hidden)[0]
        q, kv = projected.split([impl.q_lora_rank, QFM_WIDTH], dim=-1)
        q = impl.q_a_layernorm(q)
    else:
        q, kv = hidden, impl.kv_a_proj_with_mqa(hidden)[0]
        if getattr(impl, "q_a_proj", None) is not None:
            q = impl.q_a_layernorm(impl.q_a_proj(hidden)[0])
    if q.shape[0] < tokens or kv.shape[0] < tokens:
        raise ValueError("QFM projected inputs do not cover physical graph capacity")
    q, kv = q[:tokens], kv[:tokens]
    q_nope, q_raw = impl._q_proj_and_k_up_proj(q)
    q = torch.cat((q_nope, q_raw), dim=-1)
    latent, raw = kv.split([QFM_VALUE, QFM_WIDTH - QFM_VALUE], dim=-1)
    kv = torch.cat((impl.kv_a_layernorm(latent.contiguous()), raw), dim=-1)
    return q, kv


def ordinary_fia_inputs(impl, hidden, tokens):
    """Baseline Prefill projections: unabsorbed BF16 Q192 and latent KV576."""
    if impl.fused_qkv_a_proj is not None:
        projected = impl.fused_qkv_a_proj(hidden)[0]
        q, kv = projected.split([impl.q_lora_rank, QFM_WIDTH], dim=-1)
        q = impl.q_a_layernorm(q)
    else:
        q, kv = hidden, impl.kv_a_proj_with_mqa(hidden)[0]
        if getattr(impl, "q_a_proj", None) is not None:
            q = impl.q_a_layernorm(impl.q_a_proj(hidden)[0])
    if q.shape[0] < tokens or kv.shape[0] < tokens:
        raise ValueError("FIA projected inputs do not cover the physical token capacity")
    q = impl.q_proj(q[:tokens])[0].view(tokens, impl.num_heads, impl.qk_head_dim)
    latent, raw = kv[:tokens].split([QFM_VALUE, QFM_WIDTH - QFM_VALUE], dim=-1)
    kv = torch.cat((impl.kv_a_layernorm(latent.contiguous()), raw), dim=-1)
    return q, kv


def stage_expanded_fia_inputs(impl, hidden, cache, batch, prolog):
    """Route eager expanded Prefill separately; preserve Decode mode2 bytes."""
    tokens = batch.slots.numel()
    segments = batch.preprocess_segments
    if not segments:
        raise ValueError("Expanded FIA requires scheduler stage ranges")
    native = getattr(impl, "fia_native_prolog", None) if getattr(impl, "qfm_mlapo_prefill", True) else None

    def prefill_inputs(chunk):
        return native(chunk) if native is not None else ordinary_fia_inputs(impl, chunk, chunk.shape[0])

    if len(segments) == 1 and segments[0] == (0, tokens, True):
        q, kv = prefill_inputs(hidden[:tokens].contiguous())
        write_cache(kv, batch.slots, cache, impl.qfm_scale, scatter_plan=batch.scatter_plan)
        return q, kv, None, None
    q_bytes = torch.zeros((tokens, impl.num_heads, QFM_WIDTH), dtype=torch.uint8, device=hidden.device)
    q_scale = torch.ones((tokens, impl.num_heads), dtype=torch.float32, device=hidden.device)
    kv = torch.zeros((tokens, QFM_WIDTH), dtype=torch.bfloat16, device=hidden.device)
    fia_q = torch.zeros((tokens, impl.num_heads, impl.qk_head_dim), dtype=torch.bfloat16, device=hidden.device)
    for begin, end, prefill in segments:
        chunk = hidden[begin:end].contiguous()
        slots = batch.slots[begin:end]
        scatter_plan = batch.scatter_plan.slice(begin, end) if batch.scatter_plan is not None else None
        if prefill:
            chunk_q, chunk_kv = prefill_inputs(chunk)
            write_cache(chunk_kv, slots, cache, impl.qfm_scale, scatter_plan=scatter_plan)
            fia_q[begin:end].copy_(chunk_q)
            kv[begin:end].copy_(chunk_kv)
        else:
            if prolog is None:
                chunk_q, chunk_kv = ordinary_qfm_inputs(impl, chunk, end - begin)
                write_cache(chunk_kv, slots, cache, impl.qfm_scale, scatter_plan=scatter_plan)
                chunk_q, chunk_scale = quantize_query(chunk_q)
            else:
                kwargs = dict(batch.prolog_kwargs)
                if scatter_plan is not None:
                    kwargs["scatter_plan"] = scatter_plan
                if "staging_indices" in kwargs:
                    kwargs["staging_indices"] = kwargs["staging_indices"][: end - begin]
                chunk_q, chunk_scale = prolog(chunk, slots, cache, **kwargs)
            q_bytes[begin:end].copy_(chunk_q.view(torch.uint8))
            q_scale[begin:end].copy_(chunk_scale)
    return q_bytes.view(torch.float8_e4m3fn), kv, q_scale, fia_q


def forward_qfm(impl, hidden, cache, metadata, output):
    batch = metadata.qfm
    if batch is None:
        raise RuntimeError("QFM selected but QFM metadata is missing")
    tokens = batch.slots.numel()
    prolog = getattr(impl, "qfm_prolog", None)
    q_scale = None
    kv = None
    fia_q = None
    expanded = bool(batch.fia_requests) and getattr(impl, "qfm_fia_expanded", False)
    if expanded:
        # Collectives must run on every DCP rank before gathering Query heads,
        # including ranks with no history. Private weights are cached thereafter.
        impl.fia_expanded.prepare_for_dcp(impl)
        q, kv, q_scale, fia_q = stage_expanded_fia_inputs(impl, hidden, cache, batch, prolog)
    elif prolog is not None and not getattr(impl, "qfm_mlapo_prefill", True) and batch.has_prefill:
        q, kv, q_scale, fia_q = stage_specific_qfm_inputs(impl, hidden, cache, batch, prolog)
    elif prolog is None:
        q, kv = ordinary_qfm_inputs(impl, hidden, tokens)
        write_cache(kv, batch.slots, cache, impl.qfm_scale, scatter_plan=batch.scatter_plan)
    elif batch.fia_requests:
        q, kv = prolog.floating(hidden[:tokens].contiguous(), batch.slots, cache, **batch.prolog_kwargs)
    else:
        q, q_scale = prolog(hidden[:tokens].contiguous(), batch.slots, cache, **batch.prolog_kwargs)
    if getattr(impl, "qfm_fia_paged", False):
        # Mirror EVERY local writer, including Decode/Verify. APC and
        # copy-on-write then retain the BF16 history for physical KV pages.
        impl.fia_paged.update(cache, batch, kv)
    if impl.dcp_enabled:
        # HCCL byte gather preserves FP8 codes on implementations without
        # float8 collective dtype support. Gather the matching head scales.
        if q_scale is not None:
            q = impl._dcp_all_gather(q.view(torch.int8), dim=1).view(torch.float8_e4m3fn)
            q_scale = impl._dcp_all_gather(q_scale, dim=1)
        else:
            q = impl._dcp_all_gather(q, dim=1)
        if fia_q is not None:
            # FIA needs the original floating Prefill Q, not a cast of Q8
            # without descales. All ranks take this branch from CPU stage flags.
            fia_q = impl._dcp_all_gather(fia_q, dim=1)
    if batch.qfm_needed:
        latent, lse = run_qfm_reader(impl, q, cache, batch, q_scale=q_scale)
    else:
        latent = torch.zeros((*q.shape[:2], QFM_VALUE), dtype=torch.bfloat16, device=q.device)
        lse = torch.full((*q.shape[:2], 1), -float("inf"), device=q.device)
    if batch.fia_requests:
        if getattr(impl, "qfm_fia_paged", False):
            impl.fia_paged.fill(impl, q, kv, cache, batch, latent, lse)
        elif expanded:
            from vllm_ascend.attention.kimi_k3_fia_expanded import fill_expanded_fia

            fill_expanded_fia(impl, q if fia_q is None else fia_q, kv, cache, batch, latent, lse)
        else:
            fill_fia_prefill(impl, q if fia_q is None else fia_q, kv, cache, batch, latent, lse)
    if impl.dcp_enabled:
        latent = impl._merge_dcp_attention_output(latent, lse)
    if expanded:
        # The common DCP merge treats every row as an opaque vector. P rows
        # already contain V128; only D rows still require the latent V-up step.
        projected = torch.empty((tokens, impl.num_heads, impl.v_head_dim), dtype=latent.dtype, device=latent.device)
        for begin, end, prefill in batch.preprocess_segments:
            if prefill:
                projected[begin:end].copy_(latent[begin:end, :, : impl.v_head_dim])
            else:
                value = impl._v_up_proj_batch_major(latent[begin:end])
                projected[begin:end].copy_(value.view(end - begin, impl.num_heads, impl.v_head_dim))
        projected = projected.view(tokens, -1)
    else:
        projected = impl._v_up_proj_batch_major(latent)
    invalid = batch.invalid if batch.invalid is not None else ~batch.live
    projected = projected.masked_fill(invalid[:, None], 0)
    if impl.use_output_gate:
        projected = projected * torch.sigmoid(impl.g_proj(hidden.contiguous())[0][:tokens])
    result = impl.o_proj(projected, is_prefill=batch.has_prefill)[0]
    output.copy_(result)
    return output


def stage_specific_qfm_inputs(impl, hidden, cache, batch, prolog):
    """Keep mode2 Decode output/cache writes even in mixed-stage batches.

    Only ordinary Prefill queries are quantized outside Prolog. Copy Decode
    FP8 codes and descales unchanged. FIA Prefill optionally retains its own
    floating Q/current KV; these step-local tensors never enter graph state.
    """
    tokens = batch.slots.numel()
    segments = batch.preprocess_segments
    if not segments:
        raise ValueError("Missing scheduler stage ranges for Prefill MLAPO routing")
    if len(segments) == 1 and segments[0] == (0, tokens, True):
        q, kv = ordinary_qfm_inputs(impl, hidden, tokens)
        write_cache(kv, batch.slots, cache, impl.qfm_scale, scatter_plan=batch.scatter_plan)
        return q, kv, None, None
    # Byte views avoid relying on FP8 cat/indexing support on the installed PTA.
    q_bytes = torch.empty((tokens, impl.num_heads, QFM_WIDTH), dtype=torch.uint8, device=hidden.device)
    q_scale = torch.empty((tokens, impl.num_heads), dtype=torch.float32, device=hidden.device)
    kv = fia_q = None
    if batch.fia_requests:
        kv = torch.zeros((tokens, QFM_WIDTH), dtype=torch.bfloat16, device=hidden.device)
        fia_q = torch.zeros((tokens, impl.num_heads, QFM_WIDTH), dtype=torch.bfloat16, device=hidden.device)
    for begin, end, prefill in segments:
        chunk = hidden[begin:end].contiguous()
        slots = batch.slots[begin:end]
        scatter_plan = batch.scatter_plan.slice(begin, end) if batch.scatter_plan is not None else None
        if prefill:
            chunk_q, chunk_kv = ordinary_qfm_inputs(impl, chunk, end - begin)
            write_cache(chunk_kv, slots, cache, impl.qfm_scale, scatter_plan=scatter_plan)
            if fia_q is not None:
                fia_q[begin:end].copy_(chunk_q)
                kv[begin:end].copy_(chunk_kv)
                # These rows have used_q=0 in QFM and are overwritten by
                # FIA using the original BF16 Q. Keep legal placeholders for
                # the unchanged whole-batch QFM/DCP descriptor and gathers.
                q_bytes[begin:end].zero_()
                q_scale[begin:end].fill_(1.0)
                continue
            chunk_q, chunk_scale = quantize_query(chunk_q)
        else:
            kwargs = dict(batch.prolog_kwargs)
            if scatter_plan is not None:
                kwargs["scatter_plan"] = scatter_plan
            if "staging_indices" in kwargs:
                kwargs["staging_indices"] = kwargs["staging_indices"][: end - begin]
            chunk_q, chunk_scale = prolog(chunk, slots, cache, **kwargs)
        q_bytes[begin:end].copy_(chunk_q.view(torch.uint8))
        q_scale[begin:end].copy_(chunk_scale)
    return q_bytes.view(torch.float8_e4m3fn), kv, q_scale, fia_q


def quantize_query(q):
    floating = q.float()
    scale = floating.abs().amax(-1).clamp_min(1e-12) / QFM_FP8_MAX
    values = (floating / scale[..., None]).clamp(-QFM_FP8_MAX, QFM_FP8_MAX).to(torch.float8_e4m3fn)
    return values, scale


def run_qfm_reader(impl, q, cache, batch, q_scale=None):
    if q_scale is None:
        q8, q_scale = quantize_query(q)
    else:
        q8 = q
    attention, _ = impl.qfm_ops
    latent, lse = attention(
        q8,
        cache,
        q_scale,
        impl.qfm_scale,
        batch.table,
        batch.lengths,
        1,
        cu_seqlens_q=batch.cu,
        seqused_q=batch.used,
        metadata=batch.schedule,
        head_dim_v=QFM_VALUE,
        attn_mask=batch.mask,
        softmax_scale=impl.scale,
        mask_mode=batch.mask_mode,
        max_seqlen_q=batch.max_q,
        max_seqlen_kv=batch.max_k,
        layout_q="TND",
        layout_kv="PA_NZ",
        layout_out="TND",
        return_softmax_lse=impl.dcp_enabled,
    )
    invisible = batch.invisible if batch.invisible is not None else ~batch.visible
    latent = latent.masked_fill(invisible[:, None, None], 0)
    if impl.dcp_enabled:
        lse = lse.T.masked_fill(invisible[:, None], -float("inf")).unsqueeze(-1)
    elif batch.fia_requests:
        lse = torch.empty((*q.shape[:2], 1), device=q.device)
    return latent, lse
