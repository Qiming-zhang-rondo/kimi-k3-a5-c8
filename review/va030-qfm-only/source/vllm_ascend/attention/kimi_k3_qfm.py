# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in K3 full576 FP8 cache with shared DSL Prolog/QFM for P and D.

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
    if not getattr(impl, "enable_mlapo", False):
        raise ValueError("K3 QFM requires enable_mlapo=true for both Prefill and Decode")
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
    """Resolve the existing native runtime after device selection."""
    try:
        provider = importlib.import_module("cann_ops_transformer.ops")
    except (ImportError, OSError) as exc:
        raise RuntimeError("K3 QFM requires its installed CANN QFM runtime; source its env.sh before workers") from exc
    try:
        return torch.ops.cann_ops_transformer.quant_flash_mla, torch.ops.cann_ops_transformer.quant_flash_mla_metadata
    except AttributeError as exc:
        source = getattr(provider, "__file__", "<unknown provider>")
        raise RuntimeError(
            "K3 QFM/metadata operators are not registered; loaded provider="
            + str(source)
            + ". Source the pinned QFM runtime env.sh before starting new workers."
        ) from exc


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
    invalid: torch.Tensor | None = None
    invisible: torch.Tensor | None = None
    scatter_plan: QFMScatterPlan | None = None
    prolog_kwargs: dict = field(default_factory=dict)


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
        self.cache_stride = None
        self.scatter_possible = self.staging_required = False
        self.scatter_tiles = None
        if kv_cache_spec is not None:
            ratio = kernel_pages_per_manager(kv_cache_spec.block_size)
            physical_page = kv_cache_spec.page_size_padded or kv_cache_spec.page_size_bytes
            if physical_page % ratio:
                raise ValueError("QFM physical page does not split into equal kernel strides")
            self.cache_stride = physical_page // ratio
            gapped = self.cache_stride != QFM_BLOCK * QFM_WIDTH
            self.scatter_possible = self.staging_required = gapped
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
        # All decode graph capacities use one physical query per virtual row.
        # Eager prefill batches use <=16 rows per sequence, NOT small op calls.
        per_token = retain_for_graph or batch is not None or self.dcp_size > 1
        if per_token:
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
                torch.zeros((rows, columns), **ints),
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
        if per_token:
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
        batch.table.copy_(common.block_table_tensor[row_owner])
        batch.has_prefill = bool(flags[: common.num_reqs].any()) if flags is not None else common.max_query_len > 16
        if batch.scatter_plan is not None:
            batch.scatter_plan.offsets.copy_(build_scatter_offsets(batch.slots, self.cache_stride, self.scatter_tiles))
        schedule = self.metadata(batch.lengths, batch.cu, batch.used, batch.max_q, mode)
        if batch.schedule.numel() == 0:
            batch.schedule = schedule
        else:
            batch.schedule.copy_(schedule)
        return batch


def forward_qfm(impl, hidden, cache, metadata, output):
    """Run the same fused FP8 writer and latent reader in both phases."""
    batch = metadata.qfm
    if batch is None:
        raise RuntimeError("QFM selected but QFM metadata is missing")
    tokens = batch.slots.numel()
    prolog = getattr(impl, "qfm_prolog", None)
    if prolog is None:
        raise RuntimeError("K3 QFM requires initialized DSL Prolog; enable_mlapo=true")
    q, q_scale = prolog(hidden[:tokens].contiguous(), batch.slots, cache, **batch.prolog_kwargs)
    if impl.dcp_enabled:
        # HCCL byte gather preserves FP8 codes; scales follow the same heads.
        q = impl._dcp_all_gather(q.view(torch.int8), dim=1).view(torch.float8_e4m3fn)
        q_scale = impl._dcp_all_gather(q_scale, dim=1)
    latent, lse = run_qfm_reader(impl, q, cache, batch, q_scale=q_scale)
    if impl.dcp_enabled:
        latent = impl._merge_dcp_attention_output(latent, lse)
    projected = impl._v_up_proj_batch_major(latent)
    invalid = batch.invalid if batch.invalid is not None else ~batch.live
    projected = projected.masked_fill(invalid[:, None], 0)
    if impl.use_output_gate:
        projected = projected * torch.sigmoid(impl.g_proj(hidden.contiguous())[0][:tokens])
    result = impl.o_proj(projected, is_prefill=batch.has_prefill)[0]
    output.copy_(result)
    return output


def run_qfm_reader(impl, q, cache, batch, *, q_scale):
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
    return latent, lse
