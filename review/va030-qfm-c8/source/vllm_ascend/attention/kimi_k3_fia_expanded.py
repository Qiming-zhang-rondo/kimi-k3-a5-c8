# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expanded BF16 FIA Prefill over the single persistent full576 FP8 cache.

Only live history is temporarily converted and projected. Its KV axis is
bounded by the existing MLA workspace budget; each FIA call retains every Q
in the request. Decode continues to consume the latent cache through QFM.
"""

import torch
import torch_npu


class ExpandedFIA:
    """Private, reloadable BF16 KV-up-projection weights for eager Prefill."""

    def __init__(self, impl):
        # Import lazily: attention utils imports the normal MLA backends.
        from vllm_ascend.attention.utils import ascend_chunked_prefill_workspace_size

        self.latent_dim = impl.kv_lora_rank
        self.nope_dim = impl.qk_nope_head_dim
        self.raw_dim = impl.qk_rope_head_dim
        self.value_dim = impl.v_head_dim
        if self.latent_dim != 512 or self.raw_dim != 64 or not 0 < self.value_dim <= self.latent_dim:
            raise ValueError("Expanded FIA requires latent512/raw64 and value dimension in [1,512]")
        tokens = ascend_chunked_prefill_workspace_size(impl.vllm_config)
        self.workspace_bytes = tokens * (self.latent_dim + self.raw_dim) * 2
        self.local_head_weight = None
        self.w_kv = None
        self.global_head_weight = None
        self.global_w_kv = None
        self._default_max_context_tokens = None
        self.max_context_tokens = None
        self.refresh_weights(impl)

    @staticmethod
    def _store(old, new):
        if old is not None and old.shape == new.shape and old.dtype == new.dtype and old.device == new.device:
            old.copy_(new)
            return old
        return new

    @staticmethod
    def _flatten_weight(weight):
        flat = weight.transpose(0, 1).contiguous().view(weight.shape[1], -1)
        return torch_npu.npu_format_cast(flat, 29)

    def _update_context_limit(self, heads):
        # Gathered latent/raw, projected KV, concatenated K and contiguous V
        # coexist during a native call. Native workspace is managed by FIA.
        bytes_per_token = 2 * (
            self.latent_dim + self.raw_dim + heads * (2 * self.nope_dim + 2 * self.value_dim + self.raw_dim)
        )
        limit = max(1, self.workspace_bytes // bytes_per_token)
        if self.max_context_tokens is None or self.max_context_tokens == self._default_max_context_tokens:
            self.max_context_tokens = limit
        self._default_max_context_tokens = limit

    @torch.no_grad()
    def refresh_weights(self, impl):
        """Copy private weights; invalidate DCP gathering after a model reload."""
        key = impl.W_UK_T
        value = impl.W_UV
        if key.shape != (impl.num_heads, self.nope_dim, self.latent_dim) or value.shape != (
            impl.num_heads,
            self.latent_dim,
            self.value_dim,
        ):
            raise ValueError("Expanded FIA KV projection weights do not match the MLA geometry")
        local = torch.cat((key.transpose(1, 2), value), dim=-1).to(torch.bfloat16).contiguous()
        self.local_head_weight = self._store(self.local_head_weight, local)
        self.w_kv = self._store(self.w_kv, self._flatten_weight(local))
        # Prefill is eager. Its global weights must be gathered again on all
        # DCP ranks after reload, rather than using stale remote head weights.
        self.global_head_weight = None
        self.global_w_kv = None
        self._update_context_limit(impl.num_heads)

    @torch.no_grad()
    def prepare_for_dcp(self, impl):
        """Call on every rank before Query gathering, including empty shards.

        Gathering the head-major weights in rank order matches the existing
        head-axis Query gather. Cache the result across all Prefill chunks.
        """
        if impl.dcp_enabled and self.global_w_kv is None:
            self.global_head_weight = impl._dcp_all_gather(self.local_head_weight, dim=0).contiguous()
            self.global_w_kv = self._flatten_weight(self.global_head_weight)
            self._update_context_limit(self.global_head_weight.shape[0])

    def expand(self, latent, raw, *, dcp=False):
        """Project BF16 latent into per-head K/V, retaining raw64 unchanged."""
        weight = self.global_w_kv if dcp else self.w_kv
        if weight is None:
            raise RuntimeError("Call ExpandedFIA.prepare_for_dcp on every rank before Query gathering")
        heads = weight.shape[1] // (self.nope_dim + self.value_dim)
        projected = torch.matmul(latent.reshape(-1, self.latent_dim), weight)
        projected = projected.view(-1, heads, self.nope_dim + self.value_dim)
        key, value = projected.split((self.nope_dim, self.value_dim), dim=-1)
        raw = raw.reshape(-1, 1, self.raw_dim).expand(-1, heads, -1)
        return torch.cat((key, raw), dim=-1), value.contiguous()


def _partial(query, key, value, scale, mask=None):
    """Baseline expanded MLA TND FIA-v1 contract; return FP32 merge inputs."""
    output, lse = torch_npu.npu_fused_infer_attention_score(
        query,
        key,
        value,
        num_heads=query.shape[1],
        num_key_value_heads=key.shape[1],
        input_layout="TND",
        atten_mask=mask,
        sparse_mode=3 if mask is not None else 0,
        scale=scale,
        antiquant_mode=0,
        antiquant_scale=None,
        block_table=None,
        block_size=0,
        softmax_lse_flag=True,
        actual_seq_lengths=[query.shape[0]],
        actual_seq_lengths_kv=[key.shape[0]],
    )
    if lse.ndim == 2:
        lse = lse.transpose(0, 1).unsqueeze(-1)
    return output.reshape(query.shape[0], query.shape[1], value.shape[-1]).float(), lse.reshape(
        query.shape[0], query.shape[1], 1
    ).float()


def _merge(left, right):
    if left is None:
        return right
    output, lse = left
    next_output, next_lse = right
    total = torch.logaddexp(lse, next_lse)
    weight = torch.nan_to_num(torch.exp(lse - total), nan=0.0)
    next_weight = torch.nan_to_num(torch.exp(next_lse - total), nan=0.0)
    return output * weight + next_output * next_weight, total


def fill_expanded_fia(impl, q, current_kv, cache, batch, latent_container, lse):
    """Fill Prefill rows of the mixed P-expanded/D-latent DCP merge buffer.

    Prefill stores V128 in the first V columns, zeros the remaining latent512
    container columns and never applies V up-projection. Decode rows remain
    untouched. The caller interprets the rows after the common DCP merge.
    """
    # qfm also imports MLA implementation helpers; keep this dependency lazy.
    from vllm_ascend.attention.kimi_k3_qfm import build_fia_history_plan, read_fia_history_parts

    expanded = impl.fia_expanded
    expanded.prepare_for_dcp(impl)
    limit = expanded.max_context_tokens
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("Expanded FIA max_context_tokens must be a positive integer")
    heads = (expanded.global_head_weight if impl.dcp_enabled else expanded.local_head_weight).shape[0]
    if q.shape[1:] != (heads, expanded.nope_dim + expanded.raw_dim):
        raise ValueError("Expanded FIA Query must contain unabsorbed Q and all DCP heads")
    if latent_container.shape[1:] != (heads, expanded.latent_dim):
        raise ValueError("Expanded FIA requires the mixed head-major latent512 output container")
    plans = getattr(batch, "fia_history_plans", ())
    mask = getattr(batch, "fia_mask", None)
    current_rank = not impl.dcp_enabled or impl.dcp_rank == 0
    if current_rank and mask is None:
        # Match the compressed right-down causal mask used by the BF16 MLA
        # baseline; its 2048 format also handles longer current Q chunks.
        mask = torch.ones((2048, 2048), dtype=torch.bool, device=q.device).triu_(1)
    for index, (begin, end, history_length, table) in enumerate(batch.fia_requests):
        latent_container[begin:end].zero_()
        lse[begin:end].fill_(-float("inf"))
        if history_length == 0 and not current_rank:
            continue
        query = q[begin:end].to(torch.bfloat16).contiguous()
        result = None
        if history_length:
            plan = plans[index] if plans else build_fia_history_plan(table, history_length)
            if plan.pages.numel() != history_length or plan.positions.numel() != history_length:
                raise ValueError("Expanded FIA history plan does not cover the live local history")
            for start in range(0, history_length, limit):
                stop = min(start + limit, history_length)
                chunk_plan = type(plan)(plan.pages[start:stop], plan.positions[start:stop], plan.tiles)
                latent, raw = read_fia_history_parts(cache, chunk_plan)
                key, value = expanded.expand(latent, raw, dcp=impl.dcp_enabled)
                result = _merge(result, _partial(query, key, value, impl.scale))
                del latent, raw, key, value
        if current_rank:
            current = current_kv[begin:end].reshape(end - begin, expanded.latent_dim + expanded.raw_dim)
            current = current.to(torch.bfloat16)
            key, value = expanded.expand(
                current[:, : expanded.latent_dim], current[:, expanded.latent_dim :], dcp=impl.dcp_enabled
            )
            result = _merge(result, _partial(query, key, value, impl.scale, mask))
        if result is not None:
            latent_container[begin:end, :, : expanded.value_dim].copy_(result[0].to(latent_container.dtype))
            lse[begin:end].copy_(result[1])
