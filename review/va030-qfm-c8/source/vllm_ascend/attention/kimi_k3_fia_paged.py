# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CANN-style paged BF16 Prefill over an accounted mirror of FP8 KV.

The FP8 main pool retains the existing hybrid layout. Independently allocated
BF16 latent512/auxiliary64 pools are reserved in the worker's memory budget.
Only FP8 is published for PD. All pools follow the same physical page IDs,
including APC and block copies. Native FIA receives dense BF16 NZ inputs.
"""

import torch
import torch_npu

BLOCK = 128
LATENT = 512
RAW = 64
WIDTH = LATENT + RAW
BF16_TILE = 16
FP8_TILE = 32
FP8_PAGE_BYTES = BLOCK * WIDTH
BF16_LATENT_PAGE_BYTES = BLOCK * LATENT * 2
BF16_RAW_PAGE_BYTES = BLOCK * RAW * 2
TOTAL_PAGE_BYTES = FP8_PAGE_BYTES + BF16_LATENT_PAGE_BYTES + BF16_RAW_PAGE_BYTES


def sidecar_views(cache):
    """Return explicitly allocated BF16 pools without reinterpreting shared KV."""
    if (
        cache.dtype != torch.float8_e4m3fn
        or cache.ndim != 5
        or cache.shape[1:] != (1, WIDTH // FP8_TILE, BLOCK, FP8_TILE)
        or cache.stride()[2:] != (BLOCK * FP8_TILE, FP8_TILE, 1)
        or cache.stride(0) < FP8_PAGE_BYTES
        or not getattr(cache, "_qfm_bf16_prefill", False)
    ):
        raise ValueError("Paged FIA requires an explicitly reserved FP8 cache with BF16 history pools")
    blocks = cache.shape[0]
    views = getattr(cache, "_qfm_bf16_views", None)
    if blocks < 1 or not isinstance(views, tuple) or len(views) != 2:
        raise ValueError("Paged FIA needs the two separately budgeted BF16 history pools")
    ranges = [(cache.data_ptr(), cache.data_ptr() + (blocks - 1) * cache.stride(0) + FP8_PAGE_BYTES)]
    for part, dim in zip(views, (LATENT, RAW)):
        if (
            not isinstance(part, torch.Tensor)
            or part.dtype != torch.bfloat16
            or part.device != cache.device
            or part.shape != (blocks, 1, dim // BF16_TILE, BLOCK, BF16_TILE)
            or not part.is_contiguous()
            or part.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
        ):
            raise ValueError("Paged FIA requires independent, dense BF16 tile16 history pools")
        begin, end = part.data_ptr(), part.data_ptr() + part.numel() * part.element_size()
        if any(begin < stop and start < end for start, stop in ranges):
            raise ValueError("Paged FIA history must not alias FP8/KDA backing or the other BF16 pool")
        ranges.append((begin, end))
    return views


def _scatter(rows, slots, destination):
    """Use the same native -1-skipping tile writer as the FP8 path."""
    tiles = destination.shape[2]
    ids = slots.reshape(-1).to(torch.int64)
    tile = torch.arange(tiles, device=slots.device, dtype=torch.int64)
    offsets = ids[:, None].div(BLOCK, rounding_mode="floor") * (tiles * BLOCK)
    offsets = offsets + tile[None, :] * BLOCK + ids[:, None].remainder(BLOCK)
    offsets = torch.where(ids[:, None] >= 0, offsets, -1)
    torch_npu.npu_scatter_pa_cache(
        rows.to(torch.bfloat16).reshape(-1, 1, BF16_TILE).contiguous(),
        offsets.reshape(-1),
        key_cache=destination.view(-1, 1, 1, BF16_TILE),
    )


def write_bf16(cache, rows, slots, *, views=None):
    """Write this step's floating KV; never quantize the BF16 Prefill copy."""
    if rows.shape != (slots.numel(), WIDTH):
        raise ValueError("Paged FIA BF16 rows must match live full576 slots")
    latent, raw = sidecar_views(cache) if views is None else views
    _scatter(rows[:, :LATENT].contiguous(), slots, latent)
    _scatter(rows[:, LATENT:].contiguous(), slots, raw)


def mirror_fp8_slots(cache, slots, *, views=None):
    """Mirror only newly written Decode/Verify slots, never the whole history.

    Fused Decode writes FP8 in place without returning floating KV. Mirroring
    those small writes also keeps later APC and resumed Prefill pages valid.
    Padding reads a safe slot and is discarded by the -1-skipping writer.
    """
    ids = slots.reshape(-1).to(torch.int64).clamp_min(0)
    tiles = torch.arange(WIDTH // FP8_TILE, dtype=torch.int64, device=cache.device)
    encoded = cache.view(torch.uint8)[
        ids[:, None].div(BLOCK, rounding_mode="floor"),
        0,
        tiles[None, :],
        ids[:, None].remainder(BLOCK),
        :,
    ]
    rows = encoded.view(torch.float8_e4m3fn).reshape(-1, WIDTH).to(torch.bfloat16)
    write_bf16(cache, rows, slots, views=views)


class PagedFIA:
    """Keep BF16 history beside the FP8 pool, with no request-address cache."""

    def __init__(self):
        self.identity = None
        self.views = None

    def bind(self, cache):
        pools = getattr(cache, "_qfm_bf16_views", ())
        identity = (cache.data_ptr(), cache.shape[0], cache.device, tuple(part.data_ptr() for part in pools))
        if self.identity != identity:
            self.views = sidecar_views(cache)
            self.identity = identity
        return self.views

    def update(self, cache, batch, current_kv):
        views = self.bind(cache)
        if not batch.has_prefill:
            mirror_fp8_slots(cache, batch.slots, views=views)
            return
        if current_kv is None or not batch.preprocess_segments:
            raise ValueError("Paged FIA needs floating Prefill KV and explicit scheduler stage ranges")
        for begin, end, prefill in batch.preprocess_segments:
            slots = batch.slots[begin:end]
            if prefill:
                write_bf16(cache, current_kv[begin:end].reshape(-1, WIDTH), slots, views=views)
            else:
                mirror_fp8_slots(cache, slots, views=views)

    def _partial(self, query, table, length, scale, mask):
        latent, raw = self.views
        output, lse = torch_npu.npu_fused_infer_attention_score_v2(
            query[..., :LATENT].to(torch.bfloat16).contiguous(),
            latent,
            latent,
            query_rope=query[..., LATENT:].to(torch.bfloat16).contiguous(),
            key_rope=raw,
            num_query_heads=query.shape[1],
            num_key_value_heads=1,
            input_layout="TND",
            atten_mask=mask,
            sparse_mode=3 if mask is not None else 0,
            softmax_scale=scale,
            actual_seq_qlen=[query.shape[0]],
            actual_seq_kvlen=[length],
            block_table=table.reshape(1, -1).to(torch.int32).contiguous(),
            block_size=BLOCK,
            return_softmax_lse=True,
        )
        return output.reshape(*query.shape[:2], LATENT), lse.reshape(*query.shape[:2], 1)

    def fill(self, impl, query, current_kv, cache, batch, output, lse):
        # Imported lazily because the common dispatcher imports this adapter.
        from vllm_ascend.attention.kimi_k3_qfm import fia_partial, merge_fia_parts

        self.bind(cache)
        for begin, end, history, table in batch.fia_requests:
            q = query[begin:end]
            if not impl.dcp_enabled:
                # The current chunk is already in the BF16 paged cache. One
                # causal FIA reads history+current, matching the CANN recipe.
                result = self._partial(q, table, history + end - begin, impl.scale, batch.fia_mask)
            else:
                # DCP locally stripes history. Preserve the established split:
                # every rank contributes its history, current only on rank0.
                result = self._partial(q, table, history, impl.scale, None) if history else None
                if impl.dcp_rank == 0:
                    current = current_kv[begin:end].reshape(end - begin, 1, WIDTH).to(torch.bfloat16)
                    contribution = fia_partial(q, current, impl.scale, batch.fia_mask)
                    result = contribution if result is None else merge_fia_parts(result, contribution)
                if result is None:
                    output[begin:end].zero_()
                    lse[begin:end].fill_(-float("inf"))
                    continue
            output[begin:end].copy_(result[0])
            lse[begin:end].copy_(result[1])
