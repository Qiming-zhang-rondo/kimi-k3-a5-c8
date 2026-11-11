#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
PAYLOAD = {'qfm': {'source': '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"""Opt-in K3 full576 FP8 cache and optional paged BF16 Prefill history.\n\nKV uses unscaled FP8, matching the pinned CANN K3 recipe\'s unit descale.\nCaller-owned cache pages and their padding/offset are preserved. Metadata is\nshared across layers in an attention group; only captured capacities are kept.\n"""\n\nimport importlib\nimport math\nimport warnings\nfrom dataclasses import dataclass, field\n\nimport torch\nimport torch_npu\n\nQFM_WIDTH = 576\nQFM_VALUE = 512\nQFM_BLOCK = 128\nQFM_TILE = 32\nQFM_MAX_Q = 16\nQFM_MASK = 2048\nQFM_FP8_MAX = 448.0\n# cann-recipes-infer 4447f2f0972cb9287e069e3994b1d46ffe244ebc:\n# KimiMLAAttention.prepare_prolog_weights registers mla_kv_scale=[1.0].\n# This is an identity conversion, not a calibrated model accuracy guarantee.\nQFM_KV_DESCALE = 1.0\nQFM_HEADS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)\n\n\ndef select_qfm(impl, config) -> bool:\n    """Do not turn on the draft model or unrelated attention layers."""\n    if not config.enable_kimi_k3_qfm_c8 or impl.is_draft_model:\n        return False\n    # MRV2 DSpark retains the target model_config while constructing its\n    # draft. Kimi also sets draft RoPE only AFTER the MLA impl is created.\n    # Neither runner_type nor use_mla_rope identifies that construction\n    # reliably. Read the loader\'s live tag (never import its value by copy)\n    # before validating/choosing a target-only QFM cache layout.\n    if importlib.import_module("vllm.compilation.backends").model_tag == "dspark_head":\n        return False\n    if config.enable_sparse_sfa_c8:\n        raise ValueError("Disable enable_sparse_sfa_c8: packed656 and full576 QFM are different cache formats")\n    vcfg = impl.vllm_config\n    if impl.use_mla_rope or impl.kv_lora_rank != QFM_VALUE or impl.qk_rope_head_dim != 64:\n        raise ValueError("K3 QFM requires No-RoPE MLA latent512 + raw64")\n    if not vcfg.use_v2_model_runner or not impl.support_fp8_attention:\n        raise ValueError("K3 QFM requires A5 and ModelRunner V2")\n    if impl.num_kv_heads != 1 or impl.num_heads not in QFM_HEADS:\n        raise ValueError(f"QFM unsupported local query/KV heads: {impl.num_heads}/{impl.num_kv_heads}")\n    kernel_pages_per_manager(vcfg.cache_config.block_size)\n    if vcfg.cache_config.cache_dtype != "auto" or impl.fa_quant_layer:\n        raise ValueError("QFM owns full576 FP8 storage: use --kv-cache-dtype auto and no legacy FA quant scheme")\n    if impl.enable_kv_nz:\n        raise ValueError("QFM owns its PA_NZ layout; use enable_kv_nz=false")\n    if impl.pcp_enabled or impl.is_pcp_decode_sharded:\n        raise ValueError("QFM PCP is not adapted in this candidate")\n    if config.kvpp_config.size != 1:\n        raise ValueError("QFM KV layer parallelism is not adapted in this candidate")\n    if vcfg.kv_transfer_config is not None:\n        transfer = vcfg.kv_transfer_config\n        if (\n            transfer.kv_connector not in ("MooncakeConnectorV2", "MooncakePullConnector")\n            or getattr(transfer, "kv_connector_module_path", None) is not None\n        ):\n            raise ValueError("QFM PD requires the built-in MooncakeConnectorV2 (not a custom/pool connector)")\n        # Community Mooncake maps DCP ownership at manager-block granularity.\n        # Sub-block token striping cannot be copied as whole PA_NZ pages.\n        if (\n            vcfg.parallel_config.decode_context_parallel_size > 1\n            and vcfg.parallel_config.cp_kv_cache_interleave_size != vcfg.cache_config.block_size\n        ):\n            raise ValueError("QFM PD with DCP requires cp_kv_cache_interleave_size == block_size")\n    if config.sparse_kv_offload_config.enabled:\n        raise ValueError("QFM does not use sparse KV offload")\n    if vcfg.speculative_config is not None and vcfg.speculative_config.method != "dspark":\n        raise ValueError("QFM speculative path is currently scoped to DSpark")\n    size = vcfg.parallel_config.decode_context_parallel_size\n    if impl.num_heads * size not in QFM_HEADS:\n        raise ValueError("QFM does not support this DCP-gathered query head count")\n    backend = getattr(config, "kimi_k3_c8_prefill_backend", "qfm")\n    if backend not in ("fia", "qfm", "qfa", "fia_paged"):\n        raise ValueError("kimi_k3_c8_prefill_backend must be fia, qfa, qfm or fia_paged")\n    impl.qfm_fia_paged = backend == "fia_paged"\n    if impl.qfm_fia_paged and vcfg.kv_transfer_config is not None:\n        if vcfg.kv_transfer_config.kv_role != "kv_producer":\n            raise ValueError("fia_paged is a Prefill producer backend; use qfm on KV consumers")\n    impl.qfm_fia_prefill = backend in ("fia", "qfa", "fia_paged") or getattr(\n        config, "enable_kimi_k3_qfm_fia_prefill", False\n    )\n    impl.qfm_mlapo_prefill = getattr(config, "enable_kimi_k3_mlapo_prefill", True)\n    form = getattr(config, "kimi_k3_c8_fia_form", "latent")\n    if form not in ("latent", "expanded"):\n        raise ValueError("kimi_k3_c8_fia_form must be latent or expanded")\n    if backend == "qfa" and form != "expanded":\n        raise ValueError("QFA Prefill requires kimi_k3_c8_fia_form=expanded (temporary Q/K/V padded to 256)")\n    if impl.qfm_fia_paged and form != "latent":\n        raise ValueError("fia_paged uses latent FIA: set kimi_k3_c8_fia_form=latent")\n    impl.qfm_qfa_prefill = backend == "qfa"\n    impl.qfm_fia_expanded = impl.qfm_fia_prefill and form == "expanded"\n    if impl.qfm_fia_expanded and (impl.qk_nope_head_dim != 128 or impl.v_head_dim != 128):\n        raise ValueError("Expanded K3 FIA requires QK192 (128+64) and V128")\n    return True\n\n\ndef kernel_pages_per_manager(manager_block_size):\n    """The KV manager owns large blocks; QFM consumes fixed 128-token pages."""\n    if (\n        isinstance(manager_block_size, bool)\n        or not isinstance(manager_block_size, int)\n        or manager_block_size <= 0\n        or manager_block_size % QFM_BLOCK\n    ):\n        raise ValueError(\n            f"QFM requires --block-size to be a positive multiple of {QFM_BLOCK}; "\n            f"resolved manager block size is {manager_block_size}"\n        )\n    return manager_block_size // QFM_BLOCK\n\n\ndef align_qfm_page_bytes(page_bytes, manager_block_sizes):\n    """Align the shared physical page before cache budgeting/allocation."""\n    alignment = math.lcm(*(kernel_pages_per_manager(size) * QFM_TILE for size in manager_block_sizes))\n    return (page_bytes + alignment - 1) // alignment * alignment\n\n\ndef cache_view_for_manager(raw, manager_block_size, page_bytes, kernel_block_size=QFM_BLOCK):\n    """Split final allocator pages into uniformly strided QFM kernel pages.\n\n    MRV2 BlockTables already expands manager ID b to b*ratio+[0..ratio-1]\n    and computes slots with kernel size 128. Do not expand either again in\n    QFM metadata. Page padding is distributed between the kernel views, so\n    ratio*kernel_stride is exactly the allocator\'s physical manager stride.\n    """\n    ratio = kernel_pages_per_manager(manager_block_size)\n    if kernel_block_size != QFM_BLOCK:\n        raise ValueError(f"QFM needs kernel block size {QFM_BLOCK}, got {kernel_block_size}")\n    if page_bytes < manager_block_size * QFM_WIDTH or page_bytes % (ratio * QFM_TILE):\n        raise ValueError(\n            f"QFM physical manager page {page_bytes} bytes must cover {manager_block_size * QFM_WIDTH} "\n            f"payload bytes and divide into {ratio} kernel pages with {QFM_TILE}-byte aligned strides"\n        )\n    if raw.numel() == 0 or raw.numel() % page_bytes:\n        raise ValueError("QFM raw buffer must contain whole physical manager pages")\n    return cache_view(raw, raw.numel() // page_bytes * ratio, page_bytes // ratio)\n\n\ndef cache_view(raw, blocks, page_bytes):\n    """One physical page per block, with offset inherited from the allocator."""\n    if raw.dtype not in (torch.uint8, torch.int8) or raw.ndim != 1:\n        raise ValueError("QFM allocator requires a single byte buffer")\n    if page_bytes < QFM_BLOCK * QFM_WIDTH or page_bytes % QFM_TILE:\n        raise ValueError("QFM physical page must hold full576 FP8 and align to 32 bytes")\n    needed = (blocks - 1) * page_bytes + QFM_BLOCK * QFM_WIDTH\n    if blocks <= 0 or needed > raw.numel() or raw.storage_offset() % QFM_TILE:\n        raise ValueError("QFM cache allocation is too small or unaligned")\n    return torch.as_strided(\n        raw.view(torch.float8_e4m3fn),\n        (blocks, 1, QFM_WIDTH // QFM_TILE, QFM_BLOCK, QFM_TILE),\n        (page_bytes, QFM_BLOCK * QFM_WIDTH, QFM_BLOCK * QFM_TILE, QFM_TILE, 1),\n    )\n\n\n@dataclass\nclass QFMScatterPlan:\n    slots: torch.Tensor\n    offsets: torch.Tensor\n    stride: int\n    fallback_reported: bool = False\n\n    def slice(self, begin, end):\n        return QFMScatterPlan(self.slots[begin:end], self.offsets[begin:end], self.stride)\n\n\ndef build_scatter_offsets(slots, stride, tile=None):\n    """Pure slot arithmetic, shared by every MLA layer in a metadata group."""\n    ids = slots.to(torch.int64)\n    if tile is None:\n        tile = torch.arange(QFM_WIDTH // QFM_TILE, device=slots.device)\n    offsets = ids[:, None].div(QFM_BLOCK, rounding_mode="floor") * (stride // QFM_TILE)\n    offsets = offsets + tile[None, :] * QFM_BLOCK + ids[:, None].remainder(QFM_BLOCK)\n    return torch.where(ids[:, None] >= 0, offsets, -1)\n\n\ndef write_cache(rows, slots, cache, scale, scatter_plan=None):\n    values = (rows.float() / scale).clamp(-QFM_FP8_MAX, QFM_FP8_MAX).to(torch.float8_e4m3fn)\n    scatter_fp8_cache(values, slots, cache, scatter_plan=scatter_plan)\n\n\ndef scatter_fp8_cache(values, slots, cache, scatter_plan=None):\n    """Scatter 32-byte NZ tiles; native ScatterPaCache skips slot -1.\n\n    An offset-preserving contiguous alias avoids the old DSpark stride bug.\n    Invalid tokens do not alias a live token or depend on duplicate-write order.\n    """\n    if values.dtype != torch.float8_e4m3fn or values.shape != (slots.numel(), QFM_WIDTH):\n        raise ValueError("QFM scatter expects already quantized full576 rows")\n    if cache.dtype != torch.float8_e4m3fn or cache.shape[1:] != (1, 18, 128, 32):\n        raise ValueError("QFM requires full576 FP8 PA_NZ cache, not legacy packed656")\n    stride = cache.stride(0)\n    if cache.stride()[2:] != (4096, 32, 1) or stride % QFM_TILE:\n        raise ValueError("Invalid QFM PA_NZ strides")\n    span = (cache.shape[0] - 1) * stride + QFM_BLOCK * QFM_WIDTH\n    alias = torch.as_strided(cache, (span // QFM_TILE, 1, 1, QFM_TILE), (32, 32, 32, 1))\n    if scatter_plan is None:\n        offsets = build_scatter_offsets(slots, stride)\n    else:\n        offsets = scatter_plan.offsets\n        if (\n            scatter_plan.stride != stride\n            or scatter_plan.slots.data_ptr() != slots.data_ptr()\n            or scatter_plan.slots.shape != slots.shape\n            or offsets.shape != (slots.numel(), QFM_WIDTH // QFM_TILE)\n            or offsets.dtype != torch.int64\n            or offsets.device != cache.device\n            or not offsets.is_contiguous()\n        ):\n            # A shared plan is an optimization, not the address authority.\n            # Recompute from the LIVE slots and physical view. Keep this work\n            # inside the captured graph too: caching the rebuilt addresses in\n            # Python would leave them stale when graph replay updates slots.\n            if not scatter_plan.fallback_reported:\n                mismatches = {\n                    "stride": scatter_plan.stride != stride,\n                    "slots_pointer": scatter_plan.slots.data_ptr() != slots.data_ptr(),\n                    "slots_shape": scatter_plan.slots.shape != slots.shape,\n                    "offsets_shape": offsets.shape != (slots.numel(), QFM_WIDTH // QFM_TILE),\n                    "offsets_dtype": offsets.dtype != torch.int64,\n                    "offsets_device": offsets.device != cache.device,\n                    "offsets_contiguous": not offsets.is_contiguous(),\n                }\n                warnings.warn(\n                    "QFM Scatter plan fallback: "\n                    f"mismatch={[name for name, bad in mismatches.items() if bad]}; "\n                    f"plan_stride={scatter_plan.stride}, cache_stride={stride}; "\n                    f"plan_slots={tuple(scatter_plan.slots.shape)}, live_slots={tuple(slots.shape)}; "\n                    f"offsets={tuple(offsets.shape)}/{offsets.dtype}/{offsets.device}, cache_device={cache.device}. "\n                    "Rebuilding offsets from live slots; no cache layout or quantization change.",\n                    RuntimeWarning,\n                    stacklevel=2,\n                )\n                scatter_plan.fallback_reported = True\n            offsets = build_scatter_offsets(slots, stride)\n    torch_npu.npu_scatter_pa_cache(values.reshape(-1, 1, QFM_TILE).contiguous(), offsets.flatten(), key_cache=alias)\n\n\ndef load_ops():\n    # Load only after selecting the worker device. Never catch kernel failures.\n    importlib.import_module("cann_ops_transformer.ops")\n    return torch.ops.cann_ops_transformer.quant_flash_mla, torch.ops.cann_ops_transformer.quant_flash_mla_metadata\n\n\n@dataclass\nclass QFMBatch:\n    cu: torch.Tensor\n    used: torch.Tensor\n    lengths: torch.Tensor\n    table: torch.Tensor\n    slots: torch.Tensor\n    live: torch.Tensor\n    visible: torch.Tensor\n    schedule: torch.Tensor\n    mask: torch.Tensor | None\n    mask_mode: int\n    max_q: int\n    max_k: int\n    has_prefill: bool\n    # (query begin, query end, local history length, original request page table).\n    fia_requests: tuple = ()\n    qfm_needed: bool = True\n    # CPU scheduler stage ranges: (begin, end, is_prefill), including padding.\n    preprocess_segments: tuple = ()\n    invalid: torch.Tensor | None = None\n    invisible: torch.Tensor | None = None\n    fia_mask: torch.Tensor | None = None\n    scatter_plan: QFMScatterPlan | None = None\n    prolog_kwargs: dict = field(default_factory=dict)\n    # Rebuilt with the live request page tables each batch; shared by layers.\n    fia_history_plans: tuple = ()\n    # Optional QFA schedules depend on lengths/layouts, never layer Q/K/V.\n    qfa_metadata: dict = field(default_factory=dict)\n    qfa_mask: torch.Tensor | None = None\n\n\nclass QFMMetadataBuilder:\n    def __init__(self, impl, device, kv_cache_spec=None):\n        self.device = device\n        self.dcp_size = impl.vllm_config.parallel_config.decode_context_parallel_size\n        self.dcp_rank = getattr(impl, "dcp_rank", 0)\n        self.interleave = impl.vllm_config.parallel_config.cp_kv_cache_interleave_size\n        self.heads = impl.num_heads * self.dcp_size\n        self.max_k = max(1, impl.vllm_config.model_config.max_model_len)\n        _, self.metadata_op = load_ops()\n        self.mask = torch.ones((QFM_MASK, QFM_MASK), dtype=torch.int8, device=device).triu_(1)\n        self.buffers = {}\n        self.fia_prefill = getattr(impl, "qfm_fia_prefill", False)\n        self.mlapo_prefill = getattr(impl, "qfm_mlapo_prefill", True)\n        self.expanded_fia = getattr(impl, "qfm_fia_expanded", False)\n        self.qfa_prefill = getattr(impl, "qfm_qfa_prefill", False)\n        self.paged_fia = getattr(impl, "qfm_fia_paged", False)\n        self.fia_mask = self.mask.to(torch.bool) if self.fia_prefill else None\n        self.cache_stride = None\n        self.has_prolog = getattr(impl, "enable_mlapo", False)\n        self.scatter_possible = self.staging_required = False\n        self.scatter_tiles = None\n        if kv_cache_spec is not None:\n            ratio = kernel_pages_per_manager(kv_cache_spec.block_size)\n            physical_page = kv_cache_spec.page_size_padded or kv_cache_spec.page_size_bytes\n            if physical_page % ratio:\n                raise ValueError("QFM physical page does not split into equal kernel strides")\n            self.cache_stride = physical_page // ratio\n            gapped = self.cache_stride != QFM_BLOCK * QFM_WIDTH\n            self.scatter_possible = gapped or not self.has_prolog or not self.mlapo_prefill or self.fia_prefill\n            self.staging_required = self.has_prolog and (gapped or self.fia_prefill)\n            if self.scatter_possible:\n                self.scatter_tiles = torch.arange(QFM_WIDTH // QFM_TILE, device=device)\n\n    def metadata(self, lengths, cu, used, max_q, mode):\n        return self.metadata_op(\n            lengths,\n            self.heads,\n            1,\n            1,\n            cu_seqlens_q=cu,\n            seqused_q=used,\n            max_seqlen_q=max_q,\n            max_seqlen_kv=self.max_k,\n            head_dim_qk=QFM_WIDTH,\n            head_dim_v=QFM_VALUE,\n            mask_mode=mode,\n            layout_q="TND",\n        )\n\n    def build(self, common, retain_for_graph=False):\n        tokens = max(common.num_actual_tokens, common.num_input_tokens)\n        # These are already KERNEL-page columns from MRV2 BlockTables, even\n        # when the final cache manager uses 256/512/... token blocks.\n        columns = common.block_table_tensor.shape[1]\n        key = tokens, columns, common.causal\n        batch = self.buffers.get(key)\n        flags = common.is_prefilling\n        if flags is not None and flags.device.type != "cpu":\n            raise ValueError("QFM expects scheduler CPU stage flags, not a device-to-host sync")\n        fia_only = self.fia_prefill and flags is not None and bool(flags[: common.num_reqs].all())\n        if fia_only and (retain_for_graph or batch is not None):\n            raise ValueError("FIA Prefill is eager only; Decode-only graphs remain supported")\n        # All decode graph capacities use one physical query per virtual row.\n        # Eager prefill batches use <=16 rows per sequence, NOT small op calls.\n        per_token = not fia_only and (retain_for_graph or batch is not None or self.dcp_size > 1)\n        if fia_only:\n            # No QFM call consumes these descriptors. Keep one inert row;\n            # never expand Q-by-page tables or <=16-token virtual sequences.\n            cu_cpu = [0, tokens]\n            rows = 1\n        elif per_token:\n            rows = tokens\n        else:\n            bounds = common.query_start_loc_cpu[: common.num_reqs + 1].clamp_max(common.num_actual_tokens).tolist()\n            cu_cpu = [0]\n            for begin, end in zip(bounds, bounds[1:]):\n                while begin < end:\n                    begin = min(begin + QFM_MAX_Q, end)\n                    cu_cpu.append(begin)\n            while cu_cpu[-1] < tokens:\n                cu_cpu.append(min(cu_cpu[-1] + QFM_MAX_Q, tokens))\n            rows = len(cu_cpu) - 1\n        if not 0 < rows < 65536:\n            raise ValueError("QFM virtual batch must contain 1..65535 rows")\n        mode = 0 if per_token or not common.causal else 3\n        ints = dict(dtype=torch.int32, device=self.device)\n        if batch is None:\n            cu = torch.arange(rows + 1, **ints) if per_token else torch.tensor(cu_cpu, **ints)\n            used, lengths = torch.zeros(rows, **ints), torch.zeros(rows, **ints)\n            # Populate once after device lengths are ready. Do not invoke the\n            # scheduling kernel twice or require a separate Meta registration.\n            schedule = torch.empty(0, **ints)\n            batch = QFMBatch(\n                cu,\n                used,\n                lengths,\n                torch.zeros((0, 0) if fia_only else (rows, columns), **ints),\n                torch.full((tokens,), -1, dtype=torch.int64, device=self.device),\n                torch.zeros(tokens, dtype=torch.bool, device=self.device),\n                torch.zeros(tokens, dtype=torch.bool, device=self.device),\n                schedule,\n                self.mask if mode == 3 else None,\n                mode,\n                1 if per_token else QFM_MAX_Q,\n                self.max_k,\n                False,\n                invalid=torch.empty(tokens, dtype=torch.bool, device=self.device),\n                invisible=torch.empty(tokens, dtype=torch.bool, device=self.device),\n                fia_mask=self.fia_mask,\n                qfa_mask=self.mask if self.qfa_prefill else None,\n            )\n            if self.scatter_possible:\n                batch.scatter_plan = QFMScatterPlan(\n                    batch.slots,\n                    torch.empty((tokens, QFM_WIDTH // QFM_TILE), dtype=torch.int64, device=self.device),\n                    self.cache_stride,\n                )\n                batch.prolog_kwargs["scatter_plan"] = batch.scatter_plan\n            if self.staging_required:\n                batch.prolog_kwargs["staging_indices"] = torch.arange(tokens, dtype=torch.int64, device=self.device)\n            if retain_for_graph:\n                self.buffers[key] = batch\n        # Only device lengths: DSpark rejection correction may not be mirrored\n        # to CPU. Updates are on the caller\'s stream before graph replay.\n        boundaries = common.query_start_loc[: common.num_reqs + 1].contiguous()\n        token_ids = batch.cu[:-1] if per_token else torch.arange(tokens, device=self.device, dtype=torch.int32)\n        owners = torch.searchsorted(boundaries[1:], token_ids, right=True).clamp_max(common.num_reqs - 1).long()\n        starts, ends = boundaries[owners], boundaries[owners + 1]\n        seq = common.seq_lens[owners]\n        live = (token_ids < common.num_actual_tokens) & (token_ids < ends) & (seq > 0)\n        batch.live.copy_(live)\n        batch.invalid.copy_(~live)\n        count = min(tokens, common.slot_mapping.numel())\n        if count < tokens:\n            batch.slots[count:].fill_(-1)\n        batch.slots[:count].copy_(common.slot_mapping[:count])\n        batch.slots.masked_fill_(batch.invalid, -1)\n        lengths_per_token = seq - (ends - starts) + token_ids - starts + 1 if common.causal else seq\n        if self.dcp_size > 1:\n            cycle = self.interleave * self.dcp_size\n            lengths_per_token = lengths_per_token.div(cycle, rounding_mode="floor") * self.interleave + (\n                lengths_per_token.remainder(cycle) - self.dcp_rank * self.interleave\n            ).clamp(0, self.interleave)\n        lengths_per_token = torch.where(live, lengths_per_token.clamp_min(0), 0)\n        batch.visible.copy_(live & (lengths_per_token > 0))\n        batch.invisible.copy_(~batch.visible)\n        if fia_only:\n            row_owner = None\n        elif per_token:\n            # Each virtual row contains exactly this token. The general\n            # first/last-token gathers reduce to identity, even for padding\n            # and DCP ranks with zero local history.\n            row_owner = owners\n            batch.used.copy_(live)\n            batch.lengths.copy_(lengths_per_token)\n        else:\n            first = batch.cu[:-1].long()\n            row_owner = owners[first]\n            row_used = (ends[first] - batch.cu[:-1]).clamp_min(0)\n            row_used = torch.minimum(row_used, batch.cu[1:] - batch.cu[:-1])\n            row_used = torch.where(live[first], row_used, 0)\n            last = (batch.cu[:-1] + row_used - 1).clamp_min(0).long()\n            batch.used.copy_(row_used)\n            batch.lengths.copy_(torch.where(row_used > 0, lengths_per_token[last], 0))\n        if not fia_only:\n            batch.table.copy_(common.block_table_tensor[row_owner])\n        batch.has_prefill = bool(flags[: common.num_reqs].any()) if flags is not None else common.max_query_len > 16\n        batch.preprocess_segments = ()\n        if not self.mlapo_prefill or self.expanded_fia or self.paged_fia:\n            if flags is None:\n                raise ValueError("Stage-specific MLAPO requires scheduler Prefill flags")\n        if (not self.mlapo_prefill or self.expanded_fia or self.paged_fia) and batch.has_prefill:\n            bounds = common.query_start_loc_cpu[: common.num_reqs + 1].clamp_max(common.num_actual_tokens).tolist()\n            segments = []\n            for begin, end, prefill in zip(bounds, bounds[1:], flags[: common.num_reqs].tolist()):\n                if begin == end:\n                    continue\n                if segments and segments[-1][2] == prefill:\n                    segments[-1] = (segments[-1][0], end, prefill)\n                else:\n                    segments.append((begin, end, prefill))\n            if segments:\n                segments[-1] = (segments[-1][0], tokens, segments[-1][2])\n            batch.preprocess_segments = tuple(segments)\n        batch.fia_requests = ()\n        batch.fia_history_plans = ()\n        batch.qfm_needed = True\n        if self.fia_prefill:\n            if flags is None:\n                raise ValueError("FIA Prefill routing requires explicit scheduler stage flags")\n            selected = flags[: common.num_reqs].bool()\n            if bool(selected.any()):\n                if retain_for_graph or batch is self.buffers.get(key):\n                    raise ValueError("FIA Prefill is eager only; Decode-only graphs remain supported")\n                if not common.causal:\n                    raise ValueError("FIA Prefill currently requires causal attention")\n                # One metadata-time host synchronization, never once per layer.\n                # Device lengths include DSpark corrections that CPU mirrors may miss.\n                seq_cpu = common.seq_lens[: common.num_reqs].detach().cpu().tolist()\n                bounds = common.query_start_loc_cpu[: common.num_reqs + 1].tolist()\n                requests = []\n                for owner in selected.nonzero().flatten().tolist():\n                    begin, end = bounds[owner : owner + 2]\n                    if begin == end:\n                        continue\n                    history = seq_cpu[owner] - (end - begin)\n                    if history < 0 or end > common.num_actual_tokens:\n                        raise ValueError("Inconsistent FIA Prefill query/history lengths")\n                    cycle = self.interleave * self.dcp_size\n                    local_history = (history // cycle) * self.interleave + min(\n                        max(history % cycle - self.dcp_rank * self.interleave, 0), self.interleave\n                    )\n                    requests.append((begin, end, local_history, common.block_table_tensor[owner]))\n                batch.fia_requests = tuple(requests)\n                if not self.paged_fia:\n                    batch.fia_history_plans = tuple(\n                        build_fia_history_plan(table, length) for _, _, length, table in requests\n                    )\n                if not fia_only:\n                    selected_device = selected.to(self.device)\n                    skip = selected_device[row_owner]\n                    batch.used.masked_fill_(skip, 0)\n                    batch.lengths.masked_fill_(skip, 0)\n                batch.qfm_needed = not bool(selected.all())\n        if batch.scatter_plan is not None and (\n            not self.has_prolog or self.cache_stride != QFM_BLOCK * QFM_WIDTH or batch.has_prefill\n        ):\n            batch.scatter_plan.offsets.copy_(build_scatter_offsets(batch.slots, self.cache_stride, self.scatter_tiles))\n        if batch.qfm_needed:\n            schedule = self.metadata(batch.lengths, batch.cu, batch.used, batch.max_q, mode)\n            if batch.schedule.numel() == 0:\n                batch.schedule = schedule\n            else:\n                batch.schedule.copy_(schedule)\n        return batch\n\n\n@dataclass\nclass FIAHistoryPlan:\n    pages: torch.Tensor\n    positions: torch.Tensor\n    tiles: torch.Tensor\n\n\ndef build_fia_history_plan(table, length):\n    """Prepare O(history) gather coordinates once per request and batch.\n\n    Cache values remain layer-specific. Never retain this plan across batch\n    updates: APC reuse, preemption and DSpark can change the live page table.\n    """\n    position = torch.arange(length, device=table.device)\n    pages = table[position.div(QFM_BLOCK, rounding_mode="floor")].long()\n    return FIAHistoryPlan(\n        pages,\n        position.remainder(QFM_BLOCK),\n        torch.arange(QFM_WIDTH // QFM_TILE, device=table.device),\n    )\n\n\ndef read_fia_history(cache, table, length, scale, *, plan=None, unit_scale=False):\n    """Gather only live history, preserving NZ strides and storage offset.\n\n    Index FP8 as bytes (advanced FP8 indexing is not supported everywhere).\n    The result is temporary BF16 latent512+raw64, not a second cache pool.\n    """\n    if plan is None:\n        plan = build_fia_history_plan(table, length)\n    packed = cache.view(torch.uint8)[plan.pages[:, None], 0, plan.tiles[None, :], plan.positions[:, None], :]\n    values = packed.view(torch.float8_e4m3fn).reshape(length, 1, QFM_WIDTH)\n    if unit_scale:\n        # Production QFM fixes KV descale to 1 at weight initialization.\n        # No device scalar read, multiply or FP32 intermediate is needed.\n        return values.to(torch.bfloat16)\n    return (values.float() * scale).to(torch.bfloat16)\n\n\ndef read_fia_history_parts(cache, plan):\n    """Gather/cast into FIA\'s contiguous components without full576 BF16.\n\n    Production history has unit descale. Each result is already contiguous\n    in token-major order, avoiding another whole-history split/copy in FIA.\n    """\n    source = cache.view(torch.uint8)\n    length = plan.pages.numel()\n    latent_tiles = QFM_VALUE // QFM_TILE\n    latent = source[plan.pages[:, None], 0, plan.tiles[None, :latent_tiles], plan.positions[:, None], :]\n    latent = latent.view(torch.float8_e4m3fn).reshape(length, 1, QFM_VALUE).to(torch.bfloat16)\n    raw = source[plan.pages[:, None], 0, plan.tiles[None, latent_tiles:], plan.positions[:, None], :]\n    raw = raw.view(torch.float8_e4m3fn).reshape(length, 1, QFM_WIDTH - QFM_VALUE).to(torch.bfloat16)\n    return latent, raw\n\n\ndef fia_partial(q, kv, scale, mask=None, *, prepared_q=None, prepared_kv=None):\n    """Use the same latent512/raw64 TND FIA-v2 contract as MLA CP."""\n    if kv.shape[0] == 0:\n        return (\n            torch.zeros((*q.shape[:2], QFM_VALUE), dtype=torch.bfloat16, device=q.device),\n            torch.full((*q.shape[:2], 1), -float("inf"), device=q.device),\n        )\n    if prepared_kv is None:\n        prepared_kv = (kv[..., :QFM_VALUE].contiguous(), kv[..., QFM_VALUE:].contiguous())\n    latent = prepared_kv[0]\n    if prepared_q is None:\n        prepared_q = (\n            q[..., :QFM_VALUE].to(torch.bfloat16).contiguous(),\n            q[..., QFM_VALUE:].to(torch.bfloat16).contiguous(),\n        )\n    output, lse = torch_npu.npu_fused_infer_attention_score_v2(\n        prepared_q[0],\n        latent,\n        latent,\n        query_rope=prepared_q[1],\n        key_rope=prepared_kv[1],\n        num_query_heads=q.shape[1],\n        num_key_value_heads=1,\n        input_layout="TND",\n        atten_mask=mask,\n        sparse_mode=3 if mask is not None else 0,\n        softmax_scale=scale,\n        actual_seq_qlen=[q.shape[0]],\n        actual_seq_kvlen=[kv.shape[0]],\n        return_softmax_lse=True,\n    )\n    # FIA-v2 TND returns token-major output and token-major LSE.\n    return output.reshape(q.shape[0], q.shape[1], QFM_VALUE), lse.reshape(q.shape[0], q.shape[1], 1)\n\n\ndef merge_fia_parts(left, right):\n    """Combine independently normalized Attention outputs in natural-log space."""\n    a, la = left\n    b, lb = right\n    total = torch.logaddexp(la, lb)\n    # Empty+empty has -inf LSE: replace NaN weights with zero.\n    wa = torch.nan_to_num(torch.exp(la - total), nan=0.0)\n    wb = torch.nan_to_num(torch.exp(lb - total), nan=0.0)\n    return (a.float() * wa + b.float() * wb).to(a.dtype), total\n\n\ndef fill_fia_prefill(impl, q, kv, cache, batch, latent, lse):\n    """History is local FP8; current floating KV contributes on DCP rank zero.\n\n    All ranks run the same final DCP merge. No per-query FIA loops, no dense\n    Q-by-history index lists. Process one request at a time to bound workspace.\n    """\n    mask = getattr(batch, "fia_mask", None)\n    if mask is None:\n        mask = torch.ones((QFM_MASK, QFM_MASK), dtype=torch.bool, device=q.device).triu_(1)\n    plans = getattr(batch, "fia_history_plans", ())\n    for index, (begin, end, history_length, table) in enumerate(batch.fia_requests):\n        query = q[begin:end]\n        if history_length == 0 and impl.dcp_enabled and impl.dcp_rank != 0:\n            # This rank contributes neither history nor current. Preserve the\n            # final collective, but skip Q packing and temporary neutral rows.\n            latent[begin:end].zero_()\n            lse[begin:end].fill_(-float("inf"))\n            continue\n        prepared_q = (\n            query[..., :QFM_VALUE].to(torch.bfloat16).contiguous(),\n            query[..., QFM_VALUE:].to(torch.bfloat16).contiguous(),\n        )\n        result = None\n        if history_length:\n            plan = plans[index] if plans else build_fia_history_plan(table, history_length)\n            history = read_fia_history_parts(cache, plan)\n            result = fia_partial(query, history[0], impl.scale, prepared_q=prepared_q, prepared_kv=history)\n            del history\n        if not impl.dcp_enabled or impl.dcp_rank == 0:\n            current = kv[begin:end].reshape(end - begin, 1, QFM_WIDTH).to(torch.bfloat16)\n            current_result = fia_partial(query, current, impl.scale, mask, prepared_q=prepared_q)\n            result = current_result if result is None else merge_fia_parts(result, current_result)\n        latent[begin:end].copy_(result[0])\n        lse[begin:end].copy_(result[1])\n\n\ndef ordinary_qfm_inputs(impl, hidden, tokens):\n    """Existing projection modules for the non-fused writer."""\n    if impl.fused_qkv_a_proj is not None:\n        projected = impl.fused_qkv_a_proj(hidden)[0]\n        q, kv = projected.split([impl.q_lora_rank, QFM_WIDTH], dim=-1)\n        q = impl.q_a_layernorm(q)\n    else:\n        q, kv = hidden, impl.kv_a_proj_with_mqa(hidden)[0]\n        if getattr(impl, "q_a_proj", None) is not None:\n            q = impl.q_a_layernorm(impl.q_a_proj(hidden)[0])\n    if q.shape[0] < tokens or kv.shape[0] < tokens:\n        raise ValueError("QFM projected inputs do not cover physical graph capacity")\n    q, kv = q[:tokens], kv[:tokens]\n    q_nope, q_raw = impl._q_proj_and_k_up_proj(q)\n    q = torch.cat((q_nope, q_raw), dim=-1)\n    latent, raw = kv.split([QFM_VALUE, QFM_WIDTH - QFM_VALUE], dim=-1)\n    kv = torch.cat((impl.kv_a_layernorm(latent.contiguous()), raw), dim=-1)\n    return q, kv\n\n\ndef ordinary_fia_inputs(impl, hidden, tokens):\n    """Baseline Prefill projections: unabsorbed BF16 Q192 and latent KV576."""\n    if impl.fused_qkv_a_proj is not None:\n        projected = impl.fused_qkv_a_proj(hidden)[0]\n        q, kv = projected.split([impl.q_lora_rank, QFM_WIDTH], dim=-1)\n        q = impl.q_a_layernorm(q)\n    else:\n        q, kv = hidden, impl.kv_a_proj_with_mqa(hidden)[0]\n        if getattr(impl, "q_a_proj", None) is not None:\n            q = impl.q_a_layernorm(impl.q_a_proj(hidden)[0])\n    if q.shape[0] < tokens or kv.shape[0] < tokens:\n        raise ValueError("FIA projected inputs do not cover the physical token capacity")\n    q = impl.q_proj(q[:tokens])[0].view(tokens, impl.num_heads, impl.qk_head_dim)\n    latent, raw = kv[:tokens].split([QFM_VALUE, QFM_WIDTH - QFM_VALUE], dim=-1)\n    kv = torch.cat((impl.kv_a_layernorm(latent.contiguous()), raw), dim=-1)\n    return q, kv\n\n\ndef stage_expanded_fia_inputs(impl, hidden, cache, batch, prolog):\n    """Route eager expanded Prefill separately; preserve Decode mode2 bytes."""\n    tokens = batch.slots.numel()\n    segments = batch.preprocess_segments\n    if not segments:\n        raise ValueError("Expanded FIA requires scheduler stage ranges")\n    native = getattr(impl, "fia_native_prolog", None) if getattr(impl, "qfm_mlapo_prefill", True) else None\n\n    def prefill_inputs(chunk):\n        return native(chunk) if native is not None else ordinary_fia_inputs(impl, chunk, chunk.shape[0])\n\n    if len(segments) == 1 and segments[0] == (0, tokens, True):\n        q, kv = prefill_inputs(hidden[:tokens].contiguous())\n        write_cache(kv, batch.slots, cache, impl.qfm_scale, scatter_plan=batch.scatter_plan)\n        return q, kv, None, None\n    q_bytes = torch.zeros((tokens, impl.num_heads, QFM_WIDTH), dtype=torch.uint8, device=hidden.device)\n    q_scale = torch.ones((tokens, impl.num_heads), dtype=torch.float32, device=hidden.device)\n    kv = torch.zeros((tokens, QFM_WIDTH), dtype=torch.bfloat16, device=hidden.device)\n    fia_q = torch.zeros((tokens, impl.num_heads, impl.qk_head_dim), dtype=torch.bfloat16, device=hidden.device)\n    for begin, end, prefill in segments:\n        chunk = hidden[begin:end].contiguous()\n        slots = batch.slots[begin:end]\n        scatter_plan = batch.scatter_plan.slice(begin, end) if batch.scatter_plan is not None else None\n        if prefill:\n            chunk_q, chunk_kv = prefill_inputs(chunk)\n            write_cache(chunk_kv, slots, cache, impl.qfm_scale, scatter_plan=scatter_plan)\n            fia_q[begin:end].copy_(chunk_q)\n            kv[begin:end].copy_(chunk_kv)\n        else:\n            if prolog is None:\n                chunk_q, chunk_kv = ordinary_qfm_inputs(impl, chunk, end - begin)\n                write_cache(chunk_kv, slots, cache, impl.qfm_scale, scatter_plan=scatter_plan)\n                chunk_q, chunk_scale = quantize_query(chunk_q)\n            else:\n                kwargs = dict(batch.prolog_kwargs)\n                if scatter_plan is not None:\n                    kwargs["scatter_plan"] = scatter_plan\n                if "staging_indices" in kwargs:\n                    kwargs["staging_indices"] = kwargs["staging_indices"][: end - begin]\n                chunk_q, chunk_scale = prolog(chunk, slots, cache, **kwargs)\n            q_bytes[begin:end].copy_(chunk_q.view(torch.uint8))\n            q_scale[begin:end].copy_(chunk_scale)\n    return q_bytes.view(torch.float8_e4m3fn), kv, q_scale, fia_q\n\n\ndef forward_qfm(impl, hidden, cache, metadata, output):\n    batch = metadata.qfm\n    if batch is None:\n        raise RuntimeError("QFM selected but QFM metadata is missing")\n    tokens = batch.slots.numel()\n    prolog = getattr(impl, "qfm_prolog", None)\n    q_scale = None\n    kv = None\n    fia_q = None\n    expanded = bool(batch.fia_requests) and getattr(impl, "qfm_fia_expanded", False)\n    if expanded:\n        # Collectives must run on every DCP rank before gathering Query heads,\n        # including ranks with no history. Private weights are cached thereafter.\n        impl.fia_expanded.prepare_for_dcp(impl)\n        q, kv, q_scale, fia_q = stage_expanded_fia_inputs(impl, hidden, cache, batch, prolog)\n    elif prolog is not None and not getattr(impl, "qfm_mlapo_prefill", True) and batch.has_prefill:\n        q, kv, q_scale, fia_q = stage_specific_qfm_inputs(impl, hidden, cache, batch, prolog)\n    elif prolog is None:\n        q, kv = ordinary_qfm_inputs(impl, hidden, tokens)\n        write_cache(kv, batch.slots, cache, impl.qfm_scale, scatter_plan=batch.scatter_plan)\n    elif batch.fia_requests:\n        q, kv = prolog.floating(hidden[:tokens].contiguous(), batch.slots, cache, **batch.prolog_kwargs)\n    else:\n        q, q_scale = prolog(hidden[:tokens].contiguous(), batch.slots, cache, **batch.prolog_kwargs)\n    if getattr(impl, "qfm_fia_paged", False):\n        # Mirror EVERY local writer, including Decode/Verify. APC and\n        # copy-on-write then retain the BF16 history for physical KV pages.\n        impl.fia_paged.update(cache, batch, kv)\n    if impl.dcp_enabled:\n        # HCCL byte gather preserves FP8 codes on implementations without\n        # float8 collective dtype support. Gather the matching head scales.\n        if q_scale is not None:\n            q = impl._dcp_all_gather(q.view(torch.int8), dim=1).view(torch.float8_e4m3fn)\n            q_scale = impl._dcp_all_gather(q_scale, dim=1)\n        else:\n            q = impl._dcp_all_gather(q, dim=1)\n        if fia_q is not None:\n            # FIA needs the original floating Prefill Q, not a cast of Q8\n            # without descales. All ranks take this branch from CPU stage flags.\n            fia_q = impl._dcp_all_gather(fia_q, dim=1)\n    if batch.qfm_needed:\n        latent, lse = run_qfm_reader(impl, q, cache, batch, q_scale=q_scale)\n    else:\n        latent = torch.zeros((*q.shape[:2], QFM_VALUE), dtype=torch.bfloat16, device=q.device)\n        lse = torch.full((*q.shape[:2], 1), -float("inf"), device=q.device)\n    if batch.fia_requests:\n        if getattr(impl, "qfm_fia_paged", False):\n            impl.fia_paged.fill(impl, q, kv, cache, batch, latent, lse)\n        elif expanded:\n            from vllm_ascend.attention.kimi_k3_fia_expanded import fill_expanded_fia\n\n            fill_expanded_fia(impl, q if fia_q is None else fia_q, kv, cache, batch, latent, lse)\n        else:\n            fill_fia_prefill(impl, q if fia_q is None else fia_q, kv, cache, batch, latent, lse)\n    if impl.dcp_enabled:\n        latent = impl._merge_dcp_attention_output(latent, lse)\n    if expanded:\n        # The common DCP merge treats every row as an opaque vector. P rows\n        # already contain V128; only D rows still require the latent V-up step.\n        projected = torch.empty((tokens, impl.num_heads, impl.v_head_dim), dtype=latent.dtype, device=latent.device)\n        for begin, end, prefill in batch.preprocess_segments:\n            if prefill:\n                projected[begin:end].copy_(latent[begin:end, :, : impl.v_head_dim])\n            else:\n                value = impl._v_up_proj_batch_major(latent[begin:end])\n                projected[begin:end].copy_(value.view(end - begin, impl.num_heads, impl.v_head_dim))\n        projected = projected.view(tokens, -1)\n    else:\n        projected = impl._v_up_proj_batch_major(latent)\n    invalid = batch.invalid if batch.invalid is not None else ~batch.live\n    projected = projected.masked_fill(invalid[:, None], 0)\n    if impl.use_output_gate:\n        projected = projected * torch.sigmoid(impl.g_proj(hidden.contiguous())[0][:tokens])\n    result = impl.o_proj(projected, is_prefill=batch.has_prefill)[0]\n    output.copy_(result)\n    return output\n\n\ndef stage_specific_qfm_inputs(impl, hidden, cache, batch, prolog):\n    """Keep mode2 Decode output/cache writes even in mixed-stage batches.\n\n    Only ordinary Prefill queries are quantized outside Prolog. Copy Decode\n    FP8 codes and descales unchanged. FIA Prefill optionally retains its own\n    floating Q/current KV; these step-local tensors never enter graph state.\n    """\n    tokens = batch.slots.numel()\n    segments = batch.preprocess_segments\n    if not segments:\n        raise ValueError("Missing scheduler stage ranges for Prefill MLAPO routing")\n    if len(segments) == 1 and segments[0] == (0, tokens, True):\n        q, kv = ordinary_qfm_inputs(impl, hidden, tokens)\n        write_cache(kv, batch.slots, cache, impl.qfm_scale, scatter_plan=batch.scatter_plan)\n        return q, kv, None, None\n    # Byte views avoid relying on FP8 cat/indexing support on the installed PTA.\n    q_bytes = torch.empty((tokens, impl.num_heads, QFM_WIDTH), dtype=torch.uint8, device=hidden.device)\n    q_scale = torch.empty((tokens, impl.num_heads), dtype=torch.float32, device=hidden.device)\n    kv = fia_q = None\n    if batch.fia_requests:\n        kv = torch.zeros((tokens, QFM_WIDTH), dtype=torch.bfloat16, device=hidden.device)\n        fia_q = torch.zeros((tokens, impl.num_heads, QFM_WIDTH), dtype=torch.bfloat16, device=hidden.device)\n    for begin, end, prefill in segments:\n        chunk = hidden[begin:end].contiguous()\n        slots = batch.slots[begin:end]\n        scatter_plan = batch.scatter_plan.slice(begin, end) if batch.scatter_plan is not None else None\n        if prefill:\n            chunk_q, chunk_kv = ordinary_qfm_inputs(impl, chunk, end - begin)\n            write_cache(chunk_kv, slots, cache, impl.qfm_scale, scatter_plan=scatter_plan)\n            if fia_q is not None:\n                fia_q[begin:end].copy_(chunk_q)\n                kv[begin:end].copy_(chunk_kv)\n                # These rows have used_q=0 in QFM and are overwritten by\n                # FIA using the original BF16 Q. Keep legal placeholders for\n                # the unchanged whole-batch QFM/DCP descriptor and gathers.\n                q_bytes[begin:end].zero_()\n                q_scale[begin:end].fill_(1.0)\n                continue\n            chunk_q, chunk_scale = quantize_query(chunk_q)\n        else:\n            kwargs = dict(batch.prolog_kwargs)\n            if scatter_plan is not None:\n                kwargs["scatter_plan"] = scatter_plan\n            if "staging_indices" in kwargs:\n                kwargs["staging_indices"] = kwargs["staging_indices"][: end - begin]\n            chunk_q, chunk_scale = prolog(chunk, slots, cache, **kwargs)\n        q_bytes[begin:end].copy_(chunk_q.view(torch.uint8))\n        q_scale[begin:end].copy_(chunk_scale)\n    return q_bytes.view(torch.float8_e4m3fn), kv, q_scale, fia_q\n\n\ndef quantize_query(q):\n    floating = q.float()\n    scale = floating.abs().amax(-1).clamp_min(1e-12) / QFM_FP8_MAX\n    values = (floating / scale[..., None]).clamp(-QFM_FP8_MAX, QFM_FP8_MAX).to(torch.float8_e4m3fn)\n    return values, scale\n\n\ndef run_qfm_reader(impl, q, cache, batch, q_scale=None):\n    if q_scale is None:\n        q8, q_scale = quantize_query(q)\n    else:\n        q8 = q\n    attention, _ = impl.qfm_ops\n    latent, lse = attention(\n        q8,\n        cache,\n        q_scale,\n        impl.qfm_scale,\n        batch.table,\n        batch.lengths,\n        1,\n        cu_seqlens_q=batch.cu,\n        seqused_q=batch.used,\n        metadata=batch.schedule,\n        head_dim_v=QFM_VALUE,\n        attn_mask=batch.mask,\n        softmax_scale=impl.scale,\n        mask_mode=batch.mask_mode,\n        max_seqlen_q=batch.max_q,\n        max_seqlen_kv=batch.max_k,\n        layout_q="TND",\n        layout_kv="PA_NZ",\n        layout_out="TND",\n        return_softmax_lse=impl.dcp_enabled,\n    )\n    invisible = batch.invisible if batch.invisible is not None else ~batch.visible\n    latent = latent.masked_fill(invisible[:, None, None], 0)\n    if impl.dcp_enabled:\n        lse = lse.T.masked_fill(invisible[:, None], -float("inf")).unsqueeze(-1)\n    elif batch.fia_requests:\n        lse = torch.empty((*q.shape[:2], 1), device=q.device)\n    return latent, lse\n', 'sha256': 'df0cedd0e516ed17e59c716902c50bd580a894e40711cad7dcae6fab3f982229'}, 'paged': {'source': '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"""CANN-style paged BF16 Prefill over an accounted mirror of FP8 KV.\n\nThe FP8 main pool retains the existing hybrid layout. Independently allocated\nBF16 latent512/auxiliary64 pools are reserved in the worker\'s memory budget.\nOnly FP8 is published for PD. All pools follow the same physical page IDs,\nincluding APC and block copies. Native FIA receives dense BF16 NZ inputs.\n"""\n\nimport torch\nimport torch_npu\n\nBLOCK = 128\nLATENT = 512\nRAW = 64\nWIDTH = LATENT + RAW\nBF16_TILE = 16\nFP8_TILE = 32\nFP8_PAGE_BYTES = BLOCK * WIDTH\nBF16_LATENT_PAGE_BYTES = BLOCK * LATENT * 2\nBF16_RAW_PAGE_BYTES = BLOCK * RAW * 2\nTOTAL_PAGE_BYTES = FP8_PAGE_BYTES + BF16_LATENT_PAGE_BYTES + BF16_RAW_PAGE_BYTES\n\n\ndef sidecar_views(cache):\n    """Return explicitly allocated BF16 pools without reinterpreting shared KV."""\n    if (\n        cache.dtype != torch.float8_e4m3fn\n        or cache.ndim != 5\n        or cache.shape[1:] != (1, WIDTH // FP8_TILE, BLOCK, FP8_TILE)\n        or cache.stride()[2:] != (BLOCK * FP8_TILE, FP8_TILE, 1)\n        or cache.stride(0) < FP8_PAGE_BYTES\n        or not getattr(cache, "_qfm_bf16_prefill", False)\n    ):\n        raise ValueError("Paged FIA requires an explicitly reserved FP8 cache with BF16 history pools")\n    blocks = cache.shape[0]\n    views = getattr(cache, "_qfm_bf16_views", None)\n    if blocks < 1 or not isinstance(views, tuple) or len(views) != 2:\n        raise ValueError("Paged FIA needs the two separately budgeted BF16 history pools")\n    ranges = [(cache.data_ptr(), cache.data_ptr() + (blocks - 1) * cache.stride(0) + FP8_PAGE_BYTES)]\n    for part, dim in zip(views, (LATENT, RAW)):\n        if (\n            not isinstance(part, torch.Tensor)\n            or part.dtype != torch.bfloat16\n            or part.device != cache.device\n            or part.shape != (blocks, 1, dim // BF16_TILE, BLOCK, BF16_TILE)\n            or not part.is_contiguous()\n            or part.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()\n        ):\n            raise ValueError("Paged FIA requires independent, dense BF16 tile16 history pools")\n        begin, end = part.data_ptr(), part.data_ptr() + part.numel() * part.element_size()\n        if any(begin < stop and start < end for start, stop in ranges):\n            raise ValueError("Paged FIA history must not alias FP8/KDA backing or the other BF16 pool")\n        ranges.append((begin, end))\n    return views\n\n\ndef _scatter(rows, slots, destination):\n    """Use the same native -1-skipping tile writer as the FP8 path."""\n    tiles = destination.shape[2]\n    ids = slots.reshape(-1).to(torch.int64)\n    tile = torch.arange(tiles, device=slots.device, dtype=torch.int64)\n    offsets = ids[:, None].div(BLOCK, rounding_mode="floor") * (tiles * BLOCK)\n    offsets = offsets + tile[None, :] * BLOCK + ids[:, None].remainder(BLOCK)\n    offsets = torch.where(ids[:, None] >= 0, offsets, -1)\n    torch_npu.npu_scatter_pa_cache(\n        rows.to(torch.bfloat16).reshape(-1, 1, BF16_TILE).contiguous(),\n        offsets.reshape(-1),\n        key_cache=destination.view(-1, 1, 1, BF16_TILE),\n    )\n\n\ndef write_bf16(cache, rows, slots, *, views=None):\n    """Write this step\'s floating KV; never quantize the BF16 Prefill copy."""\n    if rows.shape != (slots.numel(), WIDTH):\n        raise ValueError("Paged FIA BF16 rows must match live full576 slots")\n    latent, raw = sidecar_views(cache) if views is None else views\n    _scatter(rows[:, :LATENT].contiguous(), slots, latent)\n    _scatter(rows[:, LATENT:].contiguous(), slots, raw)\n\n\ndef mirror_fp8_slots(cache, slots, *, views=None):\n    """Mirror only newly written Decode/Verify slots, never the whole history.\n\n    Fused Decode writes FP8 in place without returning floating KV. Mirroring\n    those small writes also keeps later APC and resumed Prefill pages valid.\n    Padding reads a safe slot and is discarded by the -1-skipping writer.\n    """\n    ids = slots.reshape(-1).to(torch.int64).clamp_min(0)\n    tiles = torch.arange(WIDTH // FP8_TILE, dtype=torch.int64, device=cache.device)\n    encoded = cache.view(torch.uint8)[\n        ids[:, None].div(BLOCK, rounding_mode="floor"),\n        0,\n        tiles[None, :],\n        ids[:, None].remainder(BLOCK),\n        :,\n    ]\n    rows = encoded.view(torch.float8_e4m3fn).reshape(-1, WIDTH).to(torch.bfloat16)\n    write_bf16(cache, rows, slots, views=views)\n\n\nclass PagedFIA:\n    """Keep BF16 history beside the FP8 pool, with no request-address cache."""\n\n    def __init__(self):\n        self.identity = None\n        self.views = None\n\n    def bind(self, cache):\n        pools = getattr(cache, "_qfm_bf16_views", ())\n        identity = (cache.data_ptr(), cache.shape[0], cache.device, tuple(part.data_ptr() for part in pools))\n        if self.identity != identity:\n            self.views = sidecar_views(cache)\n            self.identity = identity\n        return self.views\n\n    def update(self, cache, batch, current_kv):\n        views = self.bind(cache)\n        if not batch.has_prefill:\n            mirror_fp8_slots(cache, batch.slots, views=views)\n            return\n        if current_kv is None or not batch.preprocess_segments:\n            raise ValueError("Paged FIA needs floating Prefill KV and explicit scheduler stage ranges")\n        for begin, end, prefill in batch.preprocess_segments:\n            slots = batch.slots[begin:end]\n            if prefill:\n                write_bf16(cache, current_kv[begin:end].reshape(-1, WIDTH), slots, views=views)\n            else:\n                mirror_fp8_slots(cache, slots, views=views)\n\n    def _partial(self, query, table, length, scale, mask):\n        latent, raw = self.views\n        output, lse = torch_npu.npu_fused_infer_attention_score_v2(\n            query[..., :LATENT].to(torch.bfloat16).contiguous(),\n            latent,\n            latent,\n            query_rope=query[..., LATENT:].to(torch.bfloat16).contiguous(),\n            key_rope=raw,\n            num_query_heads=query.shape[1],\n            num_key_value_heads=1,\n            input_layout="TND",\n            atten_mask=mask,\n            sparse_mode=3 if mask is not None else 0,\n            softmax_scale=scale,\n            actual_seq_qlen=[query.shape[0]],\n            actual_seq_kvlen=[length],\n            block_table=table.reshape(1, -1).to(torch.int32).contiguous(),\n            block_size=BLOCK,\n            return_softmax_lse=True,\n        )\n        return output.reshape(*query.shape[:2], LATENT), lse.reshape(*query.shape[:2], 1)\n\n    def fill(self, impl, query, current_kv, cache, batch, output, lse):\n        # Imported lazily because the common dispatcher imports this adapter.\n        from vllm_ascend.attention.kimi_k3_qfm import fia_partial, merge_fia_parts\n\n        self.bind(cache)\n        for begin, end, history, table in batch.fia_requests:\n            q = query[begin:end]\n            if not impl.dcp_enabled:\n                # The current chunk is already in the BF16 paged cache. One\n                # causal FIA reads history+current, matching the CANN recipe.\n                result = self._partial(q, table, history + end - begin, impl.scale, batch.fia_mask)\n            else:\n                # DCP locally stripes history. Preserve the established split:\n                # every rank contributes its history, current only on rank0.\n                result = self._partial(q, table, history, impl.scale, None) if history else None\n                if impl.dcp_rank == 0:\n                    current = current_kv[begin:end].reshape(end - begin, 1, WIDTH).to(torch.bfloat16)\n                    contribution = fia_partial(q, current, impl.scale, batch.fia_mask)\n                    result = contribution if result is None else merge_fia_parts(result, contribution)\n                if result is None:\n                    output[begin:end].zero_()\n                    lse[begin:end].fill_(-float("inf"))\n                    continue\n            output[begin:end].copy_(result[0])\n            lse[begin:end].copy_(result[1])\n', 'sha256': '8b18b293b9b3882401dd9acb85cca30f2db63e462e7c1052863e5eb957566086'}}
# SPDX-License-Identifier: Apache-2.0
"""Synthetic native paged BF16 FIA probe; no model or distributed acceptance.

The standalone bundle embeds the two exact framework modules. No installed VA,
QFM provider or DSL Prolog is imported. --plan does not import torch/torch_npu.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def memory_estimate(args):
    q = args.queries if args.benchmark else 33
    history = args.history if args.benchmark else 257
    heads = args.heads if args.benchmark else 48
    pages = max(4, math.ceil((history + q) / 128) + 1)
    components = {
        "accounted_fp8_and_bf16_pools_with_guards": pages * 128 * 576 * 3 + 1024,
        "query_and_two_outputs_lse": q * heads * (576 * 2 + 2 * (512 * 2 + 4)),
        "synthetic_kv_input_and_gather_cast_temporaries": (history + q) * 576 * 8,
        "causal_mask": 2048 * 2048,
        "merge_temporaries": q * heads * 512 * 4 * 4,
        "runtime_workspace_placeholder": 512 * 1024**2,
    }
    return {
        "estimated_peak_mib": sum(components.values()) / 1024**2,
        "components_bytes": components,
        "caveat": (
            "Approximate synthetic tensors plus a workspace placeholder; native workspace, "
            "allocator retention and other card users are unknown. Not a reservation or hard cap."
        ),
    }


def load_modules(directory):
    payload = globals().get("PAYLOAD")
    if payload is None:
        root = Path(__file__).resolve().parents[6] / "vllm_ascend/attention"
        sources = {"qfm": root / "kimi_k3_qfm.py", "paged": root / "kimi_k3_fia_paged.py"}
    else:
        sources = {}
        for name, entry in payload.items():
            data = entry["source"].encode()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Embedded source checksum failed: {name}")
            path = directory / (name + ".py")
            path.write_bytes(data)
            sources[name] = path
    modules = {}
    for name, path in sources.items():
        module_name = "vllm_ascend.attention.kimi_k3_qfm" if name == "qfm" else "k3_fia_paged_probe_adapter"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["qfm"], modules["paged"]


def require_close(torch, name, actual, expected, *, atol, rtol):
    actual, expected = actual.float(), expected.float()
    diff = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    bad = int((diff > atol + rtol * expected.abs()).sum())
    relative_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp_min(1e-12))
    result = {
        # Absolute tolerance alone can accept zero output for very long
        # contexts. Require the global numerical signal to match as well.
        "pass": finite and bad == 0 and relative_l2 <= 0.02,
        "finite": finite,
        "max_abs": float(diff.max()),
        "relative_l2": relative_l2,
        "relative_l2_limit": 0.02,
        "bad_elements": bad,
        "elements": actual.numel(),
        "atol": atol,
        "rtol": rtol,
    }
    emit("check", name=name, **result)
    if not result["pass"]:
        raise AssertionError(f"{name}: {result}")
    return result


def reference(torch, query, keys, history, rows):
    scores = torch.einsum("thd,kd->thk", query[rows].float(), keys.float()) / math.sqrt(192)
    hidden = torch.arange(keys.shape[0])[None, :] > (history + torch.tensor(rows))[:, None]
    scores.masked_fill_(hidden[:, None, :], -float("inf"))
    return torch.einsum("thk,kd->thd", scores.softmax(-1), keys[:, :512].float()), scores.logsumexp(-1)


def expected_write(torch, expected, rows, slots, offset, *, stride=73984, fp8=True, bf16=True):
    """Independent flat-byte layout oracle, without production view helpers."""
    valid = slots >= 0
    rows, slots = rows[valid], slots[valid]
    pages, tokens = slots // 128, slots % 128
    lanes = torch.arange(32)
    if fp8:
        encoded = rows.float().clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        for tile in range(18):
            address = offset + pages * stride + tile * 4096 + tokens * 32
            expected[0][address[:, None] + lanes] = encoded[:, tile * 32 : (tile + 1) * 32]
    if bf16:
        for component, dim, row_start in ((1, 512, 0), (2, 64, 512)):
            encoded = rows[:, row_start : row_start + dim].to(torch.bfloat16).contiguous().view(torch.uint8)
            for tile in range(dim // 16):
                address = offset + pages * 128 * dim * 2 + tile * 4096 + tokens * 32
                expected[component][address[:, None] + lanes] = encoded[:, tile * 32 : (tile + 1) * 32]


def make_case(torch, qfm, paged, heads, history, chunks, *, seed, quantized_history=False):
    offset, tail, stride = 256, 512, 73728 + 256
    total = history + sum(chunks)
    blocks = max(4, math.ceil(total / 128) + 1)
    backings = tuple(
        torch.full((offset + blocks * page_bytes + tail,), 0x5A, dtype=torch.uint8, device="npu")
        for page_bytes in (stride, 131072, 16384)
    )
    expected = [backing.cpu() for backing in backings]
    cache = qfm.cache_view(backings[0][offset : offset + blocks * stride], blocks, stride)
    cache._qfm_bf16_prefill = True
    cache._qfm_bf16_views = tuple(
        backing[offset : offset + blocks * dim * 128 * 2].view(torch.bfloat16).view(blocks, 1, dim // 16, 128, 16)
        for backing, dim in zip(backings[1:], (512, 64))
    )
    latent, raw = paged.sidecar_views(cache)
    assert latent.is_contiguous() and raw.is_contiguous()
    table_cpu = torch.arange(blocks - 1, -1, -1, dtype=torch.int32)
    table = table_cpu.npu()
    positions = torch.arange(total)
    slots_cpu = table_cpu[positions // 128].long() * 128 + positions % 128
    slots = slots_cpu.npu()
    torch.manual_seed(seed)
    keys_cpu = (torch.randn(total, 576) * 0.5).to(torch.bfloat16)
    if quantized_history and history:
        keys_cpu[:history] = keys_cpu[:history].to(torch.float8_e4m3fn).to(torch.bfloat16)
    keys = keys_cpu.npu()
    adapter = paged.PagedFIA()
    mask = torch.ones((2048, 2048), dtype=torch.bool, device="npu").triu_(1)
    scale = torch.ones(1, device="npu")

    def write(begin, end, *, padded=False):
        rows, live = keys[begin:end], slots[begin:end]
        rows_cpu, live_cpu = keys_cpu[begin:end], slots_cpu[begin:end]
        if padded:
            rows = torch.cat((rows, torch.full((1, 576), 31.0, dtype=torch.bfloat16, device="npu")))
            live = torch.cat((live, torch.full((1,), -1, dtype=torch.int64, device="npu")))
            rows_cpu = torch.cat((rows_cpu, torch.full((1, 576), 31.0, dtype=torch.bfloat16)))
            live_cpu = torch.cat((live_cpu, torch.tensor([-1])))
        qfm.write_cache(rows, live, cache, scale)
        before_fp8 = cache.view(torch.uint8).clone()
        batch = SimpleNamespace(has_prefill=True, slots=live, preprocess_segments=((0, len(live), True),))
        adapter.update(cache, batch, rows)
        torch.npu.synchronize()
        if not torch.equal(cache.view(torch.uint8), before_fp8):
            raise AssertionError("BF16 writer modified FP8 payload")
        expected_write(torch, expected, rows_cpu, live_cpu, offset, stride=stride)
        if not all(torch.equal(backing.cpu(), wanted) for backing, wanted in zip(backings, expected)):
            raise AssertionError("Native writer bytes, invalid-slot handling or guards differ from independent oracle")

    if history:
        write(0, history, padded=True)
    return SimpleNamespace(
        backings=backings,
        expected=expected,
        cache=cache,
        table=table,
        table_cpu=table_cpu,
        slots=slots,
        slots_cpu=slots_cpu,
        keys=keys,
        keys_cpu=keys_cpu,
        adapter=adapter,
        write=write,
        mask=mask,
        scale=scale,
        heads=heads,
        blocks=blocks,
        offset=offset,
        stride=stride,
    )


def run_case(torch, qfm, paged, heads, history, chunks, seed):
    emit("case_begin", heads=heads, history=history, chunks=chunks)
    case = make_case(torch, qfm, paged, heads, history, chunks, seed=seed)
    impl = SimpleNamespace(dcp_enabled=False, dcp_rank=0, scale=1 / math.sqrt(192))
    start = history
    for index, count in enumerate(chunks):
        case.write(start, start + count, padded=True)
        query_cpu = (torch.randn(count, heads, 576) * 0.5).to(torch.bfloat16)
        query = query_cpu.npu()
        out = torch.empty(count, heads, 512, dtype=torch.bfloat16, device="npu")
        lse = torch.empty(count, heads, 1, dtype=torch.float32, device="npu")
        batch = SimpleNamespace(fia_requests=((0, count, start, case.table),), fia_mask=case.mask)
        before = [backing.clone() for backing in case.backings]
        case.adapter.fill(impl, query, case.keys[start : start + count], case.cache, batch, out, lse)
        torch.npu.synchronize()
        if not all(torch.equal(old, backing) for old, backing in zip(before, case.backings)):
            raise AssertionError("Paged FIA reader modified cache/guards")
        rows = list(range(count))
        golden, golden_lse = reference(torch, query_cpu, case.keys_cpu[: start + count], start, rows)
        require_close(torch, "paged_fia_vs_fp32", out.cpu(), golden, atol=0.02, rtol=0.02)
        require_close(torch, "paged_lse_vs_fp32", lse.cpu()[..., 0], golden_lse, atol=0.005, rtol=0.001)
        emit("chunk_complete", index=index, queries=count, history=start, pass_=True)
        start += count
    # Reuse a complete 128-token prefix under a different physical suffix table.
    if history >= 128:
        apc_history, count = 128, 9
        changed = case.table.clone()
        # Replacing this suffix with a different physical page proves the read
        # takes the supplied live table; the shared prefix remains untouched.
        changed[1] = 0
        positions = torch.arange(apc_history, apc_history + count, device="npu")
        slots = changed[positions // 128].long() * 128 + positions % 128
        current_cpu = (torch.randn(count, 576) * 0.5).to(torch.bfloat16)
        current = current_cpu.npu()
        qfm.write_cache(current, slots, case.cache, case.scale)
        case.adapter.update(
            case.cache, SimpleNamespace(has_prefill=True, slots=slots, preprocess_segments=((0, count, True),)), current
        )
        q_cpu = (torch.randn(count, heads, 576) * 0.5).to(torch.bfloat16)
        output = torch.empty(count, heads, 512, dtype=torch.bfloat16, device="npu")
        lse = torch.empty(count, heads, 1, dtype=torch.float32, device="npu")
        case.adapter.fill(
            impl,
            q_cpu.npu(),
            current,
            case.cache,
            SimpleNamespace(fia_requests=((0, count, apc_history, changed),), fia_mask=case.mask),
            output,
            lse,
        )
        golden, golden_lse = reference(
            torch, q_cpu, torch.cat((case.keys_cpu[:128], current_cpu)), apc_history, list(range(count))
        )
        require_close(torch, "simulated_apc_live_page_table", output.cpu(), golden, atol=0.02, rtol=0.02)
        require_close(torch, "simulated_apc_lse", lse.cpu()[..., 0], golden_lse, atol=0.005, rtol=0.001)
    # Decode/Verify writes expose only FP8 values to the mirror. Check exact
    # encoded rounding and padding without claiming speculative acceptance.
    mirror_rows = case.keys[-2:]
    mirror_slots = torch.cat((case.slots[-2:], torch.tensor([-1], dtype=torch.int64, device="npu")))
    mirror_input = torch.cat((mirror_rows, torch.zeros(1, 576, dtype=torch.bfloat16, device="npu")))
    qfm.write_cache(mirror_input, mirror_slots, case.cache, case.scale)
    expected = [backing.cpu() for backing in case.backings]
    rounded = mirror_input.cpu().to(torch.float8_e4m3fn).to(torch.bfloat16)
    expected_write(torch, expected, rounded, mirror_slots.cpu(), case.offset, stride=case.stride, fp8=False)
    case.adapter.update(case.cache, SimpleNamespace(has_prefill=False, slots=mirror_slots), None)
    torch.npu.synchronize()
    if not all(torch.equal(backing.cpu(), wanted) for backing, wanted in zip(case.backings, expected)):
        raise AssertionError("Decode/Verify mirror wrote incorrect BF16 bytes or guards")
    emit(
        "case_complete",
        heads=heads,
        history=history,
        chunks=chunks,
        passed=True,
        writer_bytes_and_guards=True,
        reader_cache_unchanged=True,
        decode_slots_mirror=True,
        simulated_apc=history >= 128,
    )


def time_chain(torch, function, warmup, iterations):
    for _ in range(warmup):
        function()
    torch.npu.synchronize()
    live = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    wall = time.perf_counter()
    begin.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.npu.synchronize()
    return {
        "mean_npu_ms": begin.elapsed_time(end) / iterations,
        "mean_wall_ms": (time.perf_counter() - wall) * 1000 / iterations,
        "peak_allocated_mib": torch.npu.max_memory_allocated() / 1024**2,
        "incremental_peak_mib": (torch.npu.max_memory_allocated() - live) / 1024**2,
    }


def run_benchmark(torch, qfm, paged, args):
    case = make_case(torch, qfm, paged, args.heads, args.history, [args.queries], seed=971, quantized_history=True)
    case.write(args.history, args.history + args.queries)
    q_cpu = (torch.randn(args.queries, args.heads, 576) * 0.5).to(torch.bfloat16)
    q = q_cpu.npu()
    current = case.keys[args.history :]
    impl = SimpleNamespace(dcp_enabled=False, dcp_rank=0, scale=1 / math.sqrt(192), qfm_scale=case.scale)
    out = torch.empty(args.queries, args.heads, 512, dtype=torch.bfloat16, device="npu")
    lse = torch.empty(args.queries, args.heads, 1, dtype=torch.float32, device="npu")
    old_out, old_lse = torch.empty_like(out), torch.empty_like(lse)
    batch = SimpleNamespace(
        fia_requests=((0, args.queries, args.history, case.table),),
        fia_mask=case.mask,
        fia_history_plans=(qfm.build_fia_history_plan(case.table, args.history),),
    )
    write_batch = SimpleNamespace(
        has_prefill=True, slots=case.slots[args.history :], preprocess_segments=((0, args.queries, True),)
    )

    def old():
        qfm.fill_fia_prefill(impl, q, current, case.cache, batch, old_out, old_lse)

    def new():
        case.adapter.update(case.cache, write_batch, current)
        case.adapter.fill(impl, q, current, case.cache, batch, out, lse)

    old()
    new()
    rows = sorted({0, args.queries // 2, args.queries - 1})
    golden, golden_lse = reference(torch, q_cpu, case.keys_cpu, args.history, rows)
    require_close(torch, "benchmark_paged_vs_fp32", out.cpu()[rows], golden, atol=0.02, rtol=0.02)
    require_close(torch, "benchmark_gather_vs_fp32", old_out.cpu()[rows], golden, atol=0.02, rtol=0.02)
    require_close(torch, "benchmark_lse", lse.cpu()[rows, :, 0], golden_lse, atol=0.005, rtol=0.001)
    old_result = time_chain(torch, old, args.warmup, args.iterations)
    new_result = time_chain(torch, new, args.warmup, args.iterations)
    emit(
        "benchmark",
        queries=args.queries,
        history=args.history,
        heads=args.heads,
        gather_cast_history_current_fia_merge=old_result,
        bf16_current_write_and_single_paged_fia=new_result,
        old_over_new=old_result["mean_npu_ms"] / max(new_result["mean_npu_ms"], 1e-12),
        history_values="Both paths use the same FP8-rounded BF16 history; current stays BF16.",
        scope=(
            "Prefill history preparation/current BF16 mirror/Attention only; common FP8 writer, "
            "projections, V-up, gate, scheduler, HCCL and PD excluded. No TTFT claim."
        ),
    )


def worker(args):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    os.environ.pop("ASCEND_DEVICE_ID", None)
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    qfm, paged = load_modules(Path(args.directory))
    for index, (heads, history, chunks) in enumerate(
        ((12, 0, [1, 9, 33]), (12, 127, [9, 33]), (24, 129, [33, 9]), (48, 257, [9]))
    ):
        run_case(torch, qfm, paged, heads, history, chunks, seed=901 + index)
    if args.benchmark:
        run_benchmark(torch, qfm, paged, args)
    emit(
        "suite_complete",
        all_passed=True,
        cases=4,
        benchmark=args.benchmark,
        peak_allocated_mib=torch.npu.max_memory_allocated() / 1024**2,
        scope=(
            "Synthetic native BF16 writer + paged FIA; simulated APC page reuse and Decode slots only. "
            "NOT model, real APC scheduler, DSpark acceptance, DCP/HCCL, graph or PD acceptance."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--queries", type=int, default=8192)
    parser.add_argument("--history", type=int, default=32768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--output-root", default="/mnt")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--directory", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(args.queries, args.heads, args.iterations, args.timeout) < 1
        or min(args.history, args.warmup, args.card) < 0
    ):
        parser.error("queries/heads/iterations/timeout must be positive; history/warmup/card nonnegative")
    if args.plan:
        emit(
            "plan",
            physical_card=args.card,
            memory=memory_estimate(args),
            benchmark=args.benchmark,
            scope="CPU-only plan; no torch/NPU import or native validation",
        )
        return 0
    if args.worker:
        worker(args)
        return 0
    directory = Path(tempfile.mkdtemp(prefix="fia-paged-probe-", dir=args.output_root))
    logfile = directory / "native.log"
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--worker",
        "--directory",
        str(directory),
    ]
    emit("begin", directory=str(directory), log=str(logfile), memory=memory_estimate(args))
    timed_out = False
    with logfile.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait()
    lines = logfile.read_text(errors="replace").splitlines()
    complete = False
    for line in lines:
        if line.startswith('{"event":'):
            print(line, flush=True)
            record = json.loads(line)
            complete |= record.get("event") == "suite_complete" and record.get("all_passed") is True
    passed = returncode == 0 and not timed_out and complete
    if not passed:
        print("\n".join(lines[-60:]), flush=True)
    summary = {
        "all_passed": passed,
        "returncode": returncode,
        "timeout": timed_out,
        "log": str(logfile),
        "directory": str(directory),
        "benchmark": args.benchmark,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit("summary", **summary)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
