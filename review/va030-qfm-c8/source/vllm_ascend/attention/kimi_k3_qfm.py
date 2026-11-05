# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in K3 full576 FP8 cache. No persistent BF16 history or Q-by-K indices.

KV uses unscaled FP8, matching the pinned CANN K3 recipe's unit descale.
Caller-owned cache pages and their padding/offset are preserved. Metadata is
shared across layers in an attention group; only captured capacities are kept.
"""

import importlib
import math
from dataclasses import dataclass

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
    if backend not in ("fia", "qfm"):
        raise ValueError("kimi_k3_c8_prefill_backend must be fia or qfm")
    impl.qfm_fia_prefill = backend == "fia" or getattr(config, "enable_kimi_k3_qfm_fia_prefill", False)
    impl.qfm_mlapo_prefill = getattr(config, "enable_kimi_k3_mlapo_prefill", True)
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


def write_cache(rows, slots, cache, scale):
    values = (rows.float() / scale).clamp(-QFM_FP8_MAX, QFM_FP8_MAX).to(torch.float8_e4m3fn)
    scatter_fp8_cache(values, slots, cache)


def scatter_fp8_cache(values, slots, cache):
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
    ids = slots.to(torch.int64)
    tile = torch.arange(QFM_WIDTH // QFM_TILE, device=values.device)
    offsets = ids[:, None].div(QFM_BLOCK, rounding_mode="floor") * (stride // QFM_TILE)
    offsets = offsets + tile[None, :] * QFM_BLOCK + ids[:, None].remainder(QFM_BLOCK)
    offsets = torch.where(ids[:, None] >= 0, offsets, -1)
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


class QFMMetadataBuilder:
    def __init__(self, impl, device):
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
        # All decode graph capacities use one physical query per virtual row.
        # Eager prefill batches use <=16 rows per sequence, NOT small op calls.
        per_token = retain_for_graph or batch is not None or self.dcp_size > 1
        if per_token:
            cu_cpu = list(range(tokens + 1))
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
            cu = torch.tensor(cu_cpu, **ints)
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
            )
            if retain_for_graph:
                self.buffers[key] = batch
        # Only device lengths: DSpark rejection correction may not be mirrored
        # to CPU. Updates are on the caller's stream before graph replay.
        boundaries = common.query_start_loc[: common.num_reqs + 1].contiguous()
        token_ids = torch.arange(tokens, device=self.device, dtype=torch.int32)
        owners = torch.searchsorted(boundaries[1:], token_ids, right=True).clamp_max(common.num_reqs - 1).long()
        starts, ends = boundaries[owners], boundaries[owners + 1]
        seq = common.seq_lens[owners]
        live = (token_ids < common.num_actual_tokens) & (token_ids < ends) & (seq > 0)
        batch.live.copy_(live)
        batch.slots.fill_(-1)
        count = min(tokens, common.slot_mapping.numel())
        batch.slots[:count].copy_(common.slot_mapping[:count])
        batch.slots.masked_fill_(~live, -1)
        lengths_per_token = seq - (ends - starts) + token_ids - starts + 1 if common.causal else seq
        if self.dcp_size > 1:
            cycle = self.interleave * self.dcp_size
            lengths_per_token = lengths_per_token.div(cycle, rounding_mode="floor") * self.interleave + (
                lengths_per_token.remainder(cycle) - self.dcp_rank * self.interleave
            ).clamp(0, self.interleave)
        lengths_per_token = torch.where(live, lengths_per_token.clamp_min(0), 0)
        batch.visible.copy_(live & (lengths_per_token > 0))
        first = batch.cu[:-1].long()
        row_owner = owners[first]
        row_used = (ends[first] - batch.cu[:-1]).clamp_min(0)
        row_used = torch.minimum(row_used, batch.cu[1:] - batch.cu[:-1])
        row_used = torch.where(live[first], row_used, 0)
        last = (batch.cu[:-1] + row_used - 1).clamp_min(0).long()
        batch.used.copy_(row_used)
        batch.lengths.copy_(torch.where(row_used > 0, lengths_per_token[last], 0))
        batch.table.copy_(common.block_table_tensor[row_owner])
        flags = common.is_prefilling
        if flags is not None and flags.device.type != "cpu":
            raise ValueError("QFM expects scheduler CPU stage flags, not a device-to-host sync")
        batch.has_prefill = bool(flags[: common.num_reqs].any()) if flags is not None else common.max_query_len > 16
        batch.preprocess_segments = ()
        if not self.mlapo_prefill:
            if flags is None:
                raise ValueError("Stage-specific MLAPO requires scheduler Prefill flags")
        if not self.mlapo_prefill and batch.has_prefill:
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
                selected_device = selected.to(self.device)
                skip = selected_device[row_owner]
                batch.used.masked_fill_(skip, 0)
                batch.lengths.masked_fill_(skip, 0)
                batch.qfm_needed = not bool(selected.all())
        schedule = self.metadata(batch.lengths, batch.cu, batch.used, batch.max_q, mode)
        if batch.schedule.numel() == 0:
            batch.schedule = schedule
        else:
            batch.schedule.copy_(schedule)
        return batch


def read_fia_history(cache, table, length, scale):
    """Gather only live history, preserving NZ strides and storage offset.

    Index FP8 as bytes (advanced FP8 indexing is not supported everywhere).
    The result is temporary BF16 latent512+raw64, not a second cache pool.
    """
    position = torch.arange(length, device=cache.device)
    pages = table[position.div(QFM_BLOCK, rounding_mode="floor")].long()
    tiles = torch.arange(QFM_WIDTH // QFM_TILE, device=cache.device)
    packed = cache.view(torch.uint8)[pages[:, None], 0, tiles[None, :], position.remainder(QFM_BLOCK)[:, None], :]
    return (packed.view(torch.float8_e4m3fn).reshape(length, 1, QFM_WIDTH).float() * scale).to(torch.bfloat16)


def fia_partial(q, kv, scale, mask=None):
    """Use the same latent512/raw64 TND FIA-v2 contract as MLA CP."""
    if kv.shape[0] == 0:
        return (
            torch.zeros((*q.shape[:2], QFM_VALUE), dtype=torch.bfloat16, device=q.device),
            torch.full((*q.shape[:2], 1), -float("inf"), device=q.device),
        )
    latent = kv[..., :QFM_VALUE].contiguous()
    output, lse = torch_npu.npu_fused_infer_attention_score_v2(
        q[..., :QFM_VALUE].to(torch.bfloat16).contiguous(),
        latent,
        latent,
        query_rope=q[..., QFM_VALUE:].to(torch.bfloat16).contiguous(),
        key_rope=kv[..., QFM_VALUE:].contiguous(),
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
    mask = torch.ones((QFM_MASK, QFM_MASK), dtype=torch.bool, device=q.device).triu_(1)
    for begin, end, history_length, table in batch.fia_requests:
        query = q[begin:end]
        history = read_fia_history(cache, table, history_length, impl.qfm_scale)
        result = fia_partial(query, history, impl.scale)
        del history
        if not impl.dcp_enabled or impl.dcp_rank == 0:
            current = kv[begin:end].reshape(end - begin, 1, QFM_WIDTH).to(torch.bfloat16)
            result = merge_fia_parts(result, fia_partial(query, current, impl.scale, mask))
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
    if q.shape[0] < tokens or kv.shape[0] < tokens:
        raise ValueError("QFM projected inputs do not cover physical graph capacity")
    q, kv = q[:tokens], kv[:tokens]
    q_nope, q_raw = impl._q_proj_and_k_up_proj(q)
    q = torch.cat((q_nope, q_raw), dim=-1)
    latent, raw = kv.split([QFM_VALUE, QFM_WIDTH - QFM_VALUE], dim=-1)
    kv = torch.cat((impl.kv_a_layernorm(latent.contiguous()), raw), dim=-1)
    return q, kv


def forward_qfm(impl, hidden, cache, metadata, output):
    batch = metadata.qfm
    if batch is None:
        raise RuntimeError("QFM selected but QFM metadata is missing")
    tokens = batch.slots.numel()
    prolog = getattr(impl, "qfm_prolog", None)
    q_scale = None
    kv = None
    fia_q = None
    if prolog is not None and not getattr(impl, "qfm_mlapo_prefill", True) and batch.has_prefill:
        q, kv, q_scale, fia_q = stage_specific_qfm_inputs(impl, hidden, cache, batch, prolog)
    elif prolog is None:
        q, kv = ordinary_qfm_inputs(impl, hidden, tokens)
        write_cache(kv, batch.slots, cache, impl.qfm_scale)
    elif batch.fia_requests:
        q, kv = prolog.floating(hidden[:tokens].contiguous(), batch.slots, cache)
    else:
        q, q_scale = prolog(hidden[:tokens].contiguous(), batch.slots, cache)
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
        latent = torch.zeros((*q.shape[:2], QFM_VALUE), dtype=q.dtype, device=q.device)
        lse = torch.full((*q.shape[:2], 1), -float("inf"), device=q.device)
    if batch.fia_requests:
        fill_fia_prefill(impl, q if fia_q is None else fia_q, kv, cache, batch, latent, lse)
    if impl.dcp_enabled:
        latent = impl._merge_dcp_attention_output(latent, lse)
    projected = impl._v_up_proj_batch_major(latent)
    projected = projected.masked_fill(~batch.live[:, None], 0)
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
        write_cache(kv, batch.slots, cache, impl.qfm_scale)
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
        if prefill:
            chunk_q, chunk_kv = ordinary_qfm_inputs(impl, chunk, end - begin)
            write_cache(chunk_kv, slots, cache, impl.qfm_scale)
            if fia_q is not None:
                fia_q[begin:end].copy_(chunk_q)
                kv[begin:end].copy_(chunk_kv)
            chunk_q, chunk_scale = quantize_query(chunk_q)
        else:
            chunk_q, chunk_scale = prolog(chunk, slots, cache)
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
    latent = latent.masked_fill(~batch.visible[:, None, None], 0)
    if impl.dcp_enabled:
        lse = lse.T.masked_fill(~batch.visible[:, None], -float("inf")).unsqueeze(-1)
    elif batch.fia_requests:
        lse = torch.empty((*q.shape[:2], 1), device=q.device)
    return latent, lse
