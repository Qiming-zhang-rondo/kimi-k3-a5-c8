# SPDX-License-Identifier: Apache-2.0
"""FIA preparation regressions; native execution and performance require NPU."""

from types import SimpleNamespace

import pytest
import torch

from tests.ut.attention import test_kimi_k3_qfm_cpu as base

qfm = base.qfm


@pytest.mark.parametrize("gap,offset", [(0, 0), (256, 256), (1024, 512)])
def test_shared_history_plan_preserves_permuted_pages_and_guards(qfm, monkeypatch, gap, offset):
    stride = 128 * 576 + gap
    backing = torch.full((offset + 4 * stride + 256,), 0x5A, dtype=torch.uint8)
    cache = qfm.cache_view(backing[offset : offset + 4 * stride], 4, stride)
    table = torch.tensor([3, 1, 2, 0])
    length = 257
    positions = torch.arange(length)
    slots = table[positions // 128] * 128 + positions % 128
    rows = torch.randn(length, 576)
    scale = torch.ones(1)
    qfm.write_cache(rows, slots, cache, scale)
    before = backing.clone()
    expected = rows.to(torch.float8_e4m3fn).to(torch.bfloat16)
    plan = qfm.build_fia_history_plan(table, length)

    def forbidden(*args, **kwargs):
        raise AssertionError("A layer reader must reuse batch-prepared history addresses")

    monkeypatch.setattr(torch, "arange", forbidden)
    monkeypatch.setattr(qfm, "build_fia_history_plan", forbidden)
    for _ in range(3):
        actual = qfm.read_fia_history(cache, table, length, scale, plan=plan, unit_scale=True)
        assert torch.equal(actual[:, 0], expected)
        assert torch.equal(backing, before)


def test_unit_scale_history_uses_direct_fp8_bf16_conversion(qfm, monkeypatch):
    table = torch.tensor([0])
    cache = qfm.cache_view(torch.zeros(128 * 576, dtype=torch.uint8), 1, 128 * 576)
    # Include extremes, signed zero and the smallest finite FP8 code.
    # Direct FP8->BF16 avoids an FP32 tensor and the identity multiply;
    # encoded history must remain numerically identical.
    row = torch.tensor([-448.0, -0.0, 0.0, 1 / 512, 0.1, 448.0]).repeat(96).view(1, 576)
    scale = torch.ones(1)
    qfm.write_cache(row, torch.tensor([0]), cache, scale)
    expected = (row.to(torch.float8_e4m3fn).float() * scale).to(torch.bfloat16)
    plan = qfm.build_fia_history_plan(table, 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unit-scale history must not materialize FP32 or multiply by one")

    monkeypatch.setattr(torch.Tensor, "float", forbidden)
    monkeypatch.setattr(torch.Tensor, "__mul__", forbidden)
    actual = qfm.read_fia_history(cache, table, 1, scale, plan=plan, unit_scale=True)
    assert torch.equal(actual[:, 0].view(torch.int16), expected.view(torch.int16))


def test_nonunit_history_retains_scaled_reference(qfm):
    table = torch.tensor([0])
    cache = qfm.cache_view(torch.zeros(128 * 576, dtype=torch.uint8), 1, 128 * 576)
    rows = torch.randn(9, 576)
    scale = torch.tensor([0.5])
    qfm.write_cache(rows, torch.arange(9), cache, scale)
    plan = qfm.build_fia_history_plan(table, 9)
    actual = qfm.read_fia_history(cache, table, 9, scale, plan=plan)
    expected = ((rows / scale).to(torch.float8_e4m3fn).float() * scale).to(torch.bfloat16)
    assert torch.equal(actual[:, 0], expected)


def test_history_and_current_share_prepared_query(qfm, monkeypatch):
    queries, heads, length = 9, 2, 5
    q = torch.randn(queries, heads, 576).to(torch.bfloat16)
    current = torch.randn(queries, 576).to(torch.bfloat16)
    history = torch.randn(length, 1, 576).to(torch.bfloat16)
    table = torch.tensor([0])
    plan = qfm.build_fia_history_plan(table, length)
    calls = []

    history_parts = history[..., :512].contiguous(), history[..., 512:].contiguous()

    def read(cache, live_plan):
        assert live_plan is history_plan
        return history_parts

    def native(query, key, value, **kwargs):
        calls.append((query, kwargs["query_rope"], kwargs["sparse_mode"]))
        return torch.zeros(queries, heads, 512, dtype=torch.bfloat16), torch.zeros(queries, heads, 1)

    history_plan = plan
    monkeypatch.setattr(qfm, "read_fia_history_parts", read)
    monkeypatch.setattr(qfm.torch_npu, "npu_fused_infer_attention_score_v2", native, raising=False)
    batch = SimpleNamespace(fia_requests=((0, queries, length, table),), fia_history_plans=(plan,))
    impl = SimpleNamespace(dcp_enabled=False, dcp_rank=0, qfm_scale=torch.ones(1), scale=1 / 24)
    output = torch.empty(queries, heads, 512, dtype=torch.bfloat16)
    lse = torch.empty(queries, heads, 1)
    qfm.fill_fia_prefill(impl, q, current, None, batch, output, lse)
    assert [call[2] for call in calls] == [0, 3]
    assert calls[0][0].data_ptr() == calls[1][0].data_ptr()
    assert calls[0][1].data_ptr() == calls[1][1].data_ptr()
    assert torch.equal(calls[0][0], q[..., :512])
    assert torch.equal(calls[0][1], q[..., 512:])


@pytest.mark.parametrize("dcp,rank", [(False, 0), (True, 0), (True, 1)])
def test_empty_history_skips_read_and_merge_preserves_current(qfm, monkeypatch, dcp, rank):
    queries, heads = 9, 2
    q = torch.randn(queries, heads, 576).to(torch.bfloat16)
    # This value is intentionally changed by FP8 round-trip conversion.
    current = torch.full((queries, 576), 0.10009765625, dtype=torch.bfloat16)
    assert not torch.equal(current, current.to(torch.float8_e4m3fn).to(torch.bfloat16))
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("Empty history must not be read or merged")

    def native(query, key, value, **kwargs):
        calls.append(kwargs["sparse_mode"])
        assert key.dtype == torch.bfloat16 and value.dtype == torch.bfloat16
        assert torch.equal(key[:, 0], current[:, :512])
        assert torch.equal(kwargs["key_rope"][:, 0], current[:, 512:])
        return torch.full((queries, heads, 512), 3.0, dtype=torch.bfloat16), torch.full((queries, heads, 1), 2.0)

    monkeypatch.setattr(qfm, "read_fia_history", forbidden)
    monkeypatch.setattr(qfm, "read_fia_history_parts", forbidden)
    monkeypatch.setattr(qfm, "merge_fia_parts", forbidden)
    monkeypatch.setattr(qfm.torch_npu, "npu_fused_infer_attention_score_v2", native, raising=False)
    batch = SimpleNamespace(fia_requests=((0, queries, 0, torch.tensor([0])),), fia_history_plans=(None,))
    impl = SimpleNamespace(dcp_enabled=dcp, dcp_rank=rank, qfm_scale=torch.ones(1), scale=1 / 24)
    output = torch.empty(queries, heads, 512, dtype=torch.bfloat16)
    lse = torch.empty(queries, heads, 1)
    qfm.fill_fia_prefill(impl, q, current, None, batch, output, lse)
    if not dcp or rank == 0:
        assert calls == [3]
        assert (output == 3).all() and (lse == 2).all()
    else:
        assert not calls
        assert (output == 0).all() and torch.isneginf(lse).all()


def test_fia_builder_prepares_plans_and_skips_unused_qfm_schedule(qfm, monkeypatch):
    builder, calls = base.builder(qfm, monkeypatch)
    builder.fia_prefill = True
    common = base.common([33], [290])
    result = builder.build(common)
    assert not result.qfm_needed and not calls
    assert len(result.fia_history_plans) == len(result.fia_requests) == 1
    expected = qfm.build_fia_history_plan(common.block_table_tensor[0], 257)
    actual = result.fia_history_plans[0]
    assert torch.equal(actual.pages, expected.pages)
    assert torch.equal(actual.positions, expected.positions)
    # A subsequent APC/page-table change must build current addresses, not
    # retain a previous eager batch's physical page IDs.
    common.block_table_tensor[0, :3] = torch.tensor([4, 7, 2])
    refreshed = builder.build(common)
    assert torch.equal(refreshed.fia_history_plans[0].pages, torch.tensor([4] * 128 + [7] * 128 + [2]))
    assert not calls
    decode = base.common([9], [290])
    result = builder.build(decode, retain_for_graph=True)
    assert result.qfm_needed and len(calls) == 1
    mixed = base.common([9, 33], [290, 513])
    result = builder.build(mixed)
    assert result.qfm_needed and len(calls) == 2


def test_mixed_fia_prefill_skips_q8_quantization_and_preserves_decode(qfm, monkeypatch):
    builder, _ = base.builder(qfm, monkeypatch)
    builder.fia_prefill = True
    builder.mlapo_prefill = False
    common = base.common([4, 9, 3], [127, 257, 513], capacity=20)
    common.is_prefilling = torch.tensor([False, True, False])
    batch = builder.build(common)
    assert batch.preprocess_segments == ((0, 4, False), (4, 13, True), (13, 20, False))
    assert not batch.used[1:2].any()
    hidden = torch.arange(20, dtype=torch.bfloat16)[:, None].expand(20, 4).contiguous()
    cache = qfm.cache_view(torch.zeros(128 * 576, dtype=torch.uint8), 1, 128 * 576)
    impl = SimpleNamespace(num_heads=2, qfm_scale=torch.ones(1))
    expected_q = torch.full((9, 2, 576), 0.10009765625, dtype=torch.bfloat16)
    expected_kv = torch.full((9, 576), 0.2001953125, dtype=torch.bfloat16)
    decode_results = []

    def ordinary(live_impl, chunk, tokens):
        assert live_impl is impl and tokens == 9
        assert torch.equal(chunk[:, 0], torch.arange(4, 13, dtype=torch.bfloat16))
        return expected_q, expected_kv

    def forbidden(*args, **kwargs):
        raise AssertionError("FIA-only Prefill Q must not be quantized for inactive QFM rows")

    def prolog(chunk, slots, live_cache, **kwargs):
        assert live_cache is cache
        begin = int(chunk[0, 0])
        assert begin in (0, 13)
        q = torch.full((len(chunk), 2, 576), 2.0 if begin == 0 else 3.0).to(torch.float8_e4m3fn)
        scales = torch.full((len(chunk), 2), 0.25 if begin == 0 else 0.125)
        decode_results.append((begin, slots.clone(), q.view(torch.uint8).clone(), scales.clone()))
        return q, scales

    monkeypatch.setattr(qfm, "ordinary_qfm_inputs", ordinary)
    monkeypatch.setattr(qfm, "quantize_query", forbidden)
    q, kv, scales, fia_q = qfm.stage_specific_qfm_inputs(impl, hidden, cache, batch, prolog)
    assert len(decode_results) == 2
    for begin, slots, codes, expected_scales in decode_results:
        end = begin + len(slots)
        assert torch.equal(q[begin:end].view(torch.uint8), codes)
        assert torch.equal(scales[begin:end], expected_scales)
    assert (decode_results[-1][1][3:] == -1).all()  # Preserve graph-capacity padding slots.
    assert (q[4:13].view(torch.uint8) == 0).all()
    assert (scales[4:13] == 1).all()
    assert torch.equal(fia_q[4:13], expected_q)
    assert torch.equal(kv[4:13], expected_kv)
    assert not torch.equal(expected_q, expected_q.to(torch.float8_e4m3fn).to(torch.bfloat16))


@pytest.mark.parametrize("dcp_size,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("global_history", [0, 1, 127, 129])
@pytest.mark.parametrize("queries", [1, 9, 33])
def test_optimized_fia_matches_independent_legacy_fill(qfm, monkeypatch, dcp_size, rank, global_history, queries):
    """Numerical old/new differential, not constant-output native mocking.

    The legacy side retains the original gather, FP32 unit scaling, repeated
    Q splitting, and unconditional history/current merge. Neither it nor the
    CPU native stub calls the optimized read/partial/merge/fill helpers.
    """
    torch.manual_seed(2026 + global_history + queries)
    heads, width, value_dim = 2, 576, 512
    q = torch.randn(queries, heads, width).to(torch.bfloat16)
    rows = torch.randn(global_history + queries, width).to(torch.bfloat16)
    current = rows[global_history:]
    local_indices = [position for position in range(len(rows)) if (position // 16) % dcp_size == rank]
    local_history = sum(position < global_history for position in local_indices)
    local_rows = rows[torch.tensor(local_indices, dtype=torch.long)]
    table = torch.tensor([3, 1, 4, 2])
    stride, offset = 128 * width + 256, 256
    backing = torch.full((offset + 5 * stride + 256,), 0x5A, dtype=torch.uint8)
    cache = qfm.cache_view(backing[offset : offset + 5 * stride], 5, stride)
    positions = torch.arange(len(local_rows))
    slots = table[positions // 128] * 128 + positions % 128
    scale = torch.ones(1)
    qfm.write_cache(local_rows, slots, cache, scale)
    before = backing.clone()
    mask = torch.ones(64, 64, dtype=torch.bool).triu_(1)

    def native(query, key, value, **kwargs):
        assert kwargs["input_layout"] == "TND"
        assert kwargs["actual_seq_qlen"] == [queries]
        assert kwargs["actual_seq_kvlen"] == [len(key)]
        floating_q = torch.cat((query, kwargs["query_rope"]), -1).float()
        floating_k = torch.cat((key, kwargs["key_rope"]), -1)[:, 0].float()
        scores = torch.einsum("thd,kd->thk", floating_q, floating_k) * kwargs["softmax_scale"]
        if kwargs["sparse_mode"] == 3:
            assert len(key) == queries  # Current contributes its own causal chunk.
            scores.masked_fill_(
                torch.arange(len(key))[None, None, :] > torch.arange(queries)[:, None, None], -torch.inf
            )
        else:
            assert kwargs["sparse_mode"] == 0  # All local history is visible.
        output = torch.einsum("thk,kd->thd", scores.softmax(-1), value[:, 0].float()).to(torch.bfloat16)
        return output, scores.logsumexp(-1).unsqueeze(-1)

    def legacy_read(length):
        position = torch.arange(length, device=cache.device)
        pages = table[position.div(128, rounding_mode="floor")].long()
        tiles = torch.arange(width // 32, device=cache.device)
        packed = cache.view(torch.uint8)[pages[:, None], 0, tiles[None, :], position.remainder(128)[:, None], :]
        return (packed.view(torch.float8_e4m3fn).reshape(length, 1, width).float() * scale).to(torch.bfloat16)

    def legacy_partial(kv, causal=False):
        if kv.shape[0] == 0:
            return (
                torch.zeros(queries, heads, value_dim, dtype=torch.bfloat16),
                torch.full((queries, heads, 1), -torch.inf),
            )
        latent = kv[..., :value_dim].contiguous()
        return native(
            q[..., :value_dim].to(torch.bfloat16).contiguous(),
            latent,
            latent,
            query_rope=q[..., value_dim:].to(torch.bfloat16).contiguous(),
            key_rope=kv[..., value_dim:].contiguous(),
            input_layout="TND",
            atten_mask=mask if causal else None,
            sparse_mode=3 if causal else 0,
            softmax_scale=1 / 24,
            actual_seq_qlen=[queries],
            actual_seq_kvlen=[len(kv)],
        )

    def legacy_merge(left, right):
        output_a, lse_a = left
        output_b, lse_b = right
        combined_lse = torch.logaddexp(lse_a, lse_b)
        weight_a = torch.nan_to_num(torch.exp(lse_a - combined_lse), nan=0.0)
        weight_b = torch.nan_to_num(torch.exp(lse_b - combined_lse), nan=0.0)
        return (output_a.float() * weight_a + output_b.float() * weight_b).to(output_a.dtype), combined_lse

    # Original fill always computes the history partial, including empty KV.
    legacy = legacy_partial(legacy_read(local_history))
    if dcp_size == 1 or rank == 0:
        legacy = legacy_merge(legacy, legacy_partial(current.view(queries, 1, width), causal=True))
    assert torch.equal(backing, before)
    monkeypatch.setattr(qfm.torch_npu, "npu_fused_infer_attention_score_v2", native, raising=False)
    batch = SimpleNamespace(
        fia_requests=((0, queries, local_history, table),),
        fia_history_plans=(qfm.build_fia_history_plan(table, local_history),),
        fia_mask=mask,
    )
    impl = SimpleNamespace(dcp_enabled=dcp_size > 1, dcp_rank=rank, qfm_scale=scale, scale=1 / 24)
    output = torch.empty(queries, heads, value_dim, dtype=torch.bfloat16)
    lse = torch.empty(queries, heads, 1)
    qfm.fill_fia_prefill(impl, q, current, cache, batch, output, lse)
    torch.testing.assert_close(output, legacy[0], rtol=0, atol=0)
    torch.testing.assert_close(lse, legacy[1], rtol=0, atol=0)
    assert torch.equal(backing, before)


@pytest.mark.parametrize("gap,offset", [(0, 0), (256, 256)])
@pytest.mark.parametrize("length", [0, 1, 129, 257])
def test_history_parts_are_exact_contiguous_and_preserve_guards(qfm, monkeypatch, gap, offset, length):
    stride = 128 * 576 + gap
    backing = torch.full((offset + 4 * stride + 256,), 0x5A, dtype=torch.uint8)
    cache = qfm.cache_view(backing[offset : offset + 4 * stride], 4, stride)
    table = torch.tensor([3, 1, 2, 0])
    position = torch.arange(length)
    slots = table[position // 128] * 128 + position % 128
    rows = torch.randn(length, 576)
    qfm.write_cache(rows, slots, cache, torch.ones(1))
    before = backing.clone()
    expected = rows.to(torch.float8_e4m3fn).to(torch.bfloat16)
    plan = qfm.build_fia_history_plan(table, length)

    def forbidden(*args, **kwargs):
        raise AssertionError("Parts reader must gather into its final layout without FP32/full-history repacking")

    monkeypatch.setattr(qfm, "read_fia_history", forbidden)
    monkeypatch.setattr(torch.Tensor, "float", forbidden)
    monkeypatch.setattr(torch.Tensor, "contiguous", forbidden)
    latent, raw = qfm.read_fia_history_parts(cache, plan)
    assert latent.shape == (length, 1, 512) and raw.shape == (length, 1, 64)
    assert latent.dtype == raw.dtype == torch.bfloat16
    assert latent.is_contiguous() and raw.is_contiguous()
    assert torch.equal(latent[:, 0], expected[:, :512])
    assert torch.equal(raw[:, 0], expected[:, 512:])
    assert torch.equal(backing, before)


@pytest.mark.parametrize("dcp,rank", [(1, 0), (2, 0), (2, 1)])
def test_large_pure_fia_omits_virtual_qfm_table_but_mixed_keeps_it(qfm, monkeypatch, dcp, rank):
    builder, calls = base.builder(qfm, monkeypatch, dcp=dcp, rank=rank)
    builder.fia_prefill = True
    common = base.common([8192], [131072])
    result = builder.build(common)
    assert not result.qfm_needed and not calls
    assert result.table.shape == (0, 0)
    assert result.used.numel() == result.lengths.numel() == 1
    assert len(result.fia_requests) == len(result.fia_history_plans) == 1
    history = 131072 - 8192
    local_history = sum((position // 16) % dcp == rank for position in range(history))
    assert result.fia_requests[0][2] == local_history
    assert result.fia_history_plans[0].pages.numel() == local_history
    mixed = base.common([4, 33, 9], [127, 513, 99], capacity=48)
    active = builder.build(mixed)
    assert active.qfm_needed and len(calls) == 1
    assert active.table.shape[1] == mixed.block_table_tensor.shape[1]
    if dcp > 1:
        assert active.table.shape[0] == 48
        owners = torch.tensor([0] * 4 + [1] * 33 + [2] * 11)
        assert torch.equal(active.table, mixed.block_table_tensor[owners])
