# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch/numerical contracts; native FIA and model acceptance need A5."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.attention import test_kimi_k3_qfm_blocks_cpu as blocks
from tests.ut.attention import test_kimi_k3_qfm_cpu as base

qfm = base.qfm


@pytest.fixture
def paged(qfm, monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/kimi_k3_fia_paged.py"
    spec = importlib.util.spec_from_file_location("paged_fia_cpu_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.kimi_k3_qfm", qfm)
    return module


def cache_for(qfm, paged, count=4):
    storage = torch.zeros(256 + count * paged.TOTAL_PAGE_BYTES + 256, dtype=torch.uint8)
    cache = qfm.cache_view(storage[256 : 256 + count * paged.FP8_PAGE_BYTES], count, paged.FP8_PAGE_BYTES)
    cache._qfm_bf16_prefill = True
    cache._qfm_bf16_views = (
        torch.zeros(count, 1, 32, 128, 16, dtype=torch.bfloat16),
        torch.zeros(count, 1, 4, 128, 16, dtype=torch.bfloat16),
    )
    return cache


def slots_for(table, begin, end):
    positions = torch.arange(begin, end)
    return table[positions // 128] * 128 + positions % 128


def native_reference(q, k, v, **kwargs):
    assert kwargs["input_layout"] == "TND"
    assert k.is_contiguous() and v.is_contiguous() and kwargs["key_rope"].is_contiguous()
    length = kwargs["actual_seq_kvlen"][0]
    assert length > 0
    if kwargs.get("block_table") is not None:
        table = kwargs["block_table"][0]
        positions = torch.arange(length)
        pages = table[positions // 128].long()
        key = k[pages, 0, :, positions % 128, :].reshape(length, 512).float()
        raw = kwargs["key_rope"][pages, 0, :, positions % 128, :].reshape(length, 64).float()
    else:
        key = k[:, 0].float()
        raw = kwargs["key_rope"][:, 0].float()
    scores = torch.einsum("qhd,kd->qhk", q.float(), key)
    scores += torch.einsum("qhd,kd->qhk", kwargs["query_rope"].float(), raw)
    scores *= kwargs["softmax_scale"]
    if kwargs["sparse_mode"] == 3:
        future = torch.arange(length)[None, :] > (length - q.shape[0] + torch.arange(q.shape[0]))[:, None]
        scores.masked_fill_(future[:, None, :], -float("inf"))
    output = torch.einsum("qhk,kd->qhd", scores.softmax(-1), key)
    return output.to(torch.bfloat16), scores.logsumexp(-1).unsqueeze(-1)


@pytest.mark.parametrize("history,queries,heads", [(0, 1, 12), (127, 9, 12), (129, 33, 24), (257, 9, 48)])
def test_one_paged_call_reads_bf16_history_and_current_without_gather(paged, qfm, monkeypatch, history, queries, heads):
    torch.manual_seed(128 + history)
    cache = cache_for(qfm, paged)
    table = torch.tensor([3, 1, 0, 2], dtype=torch.int32)
    all_kv = torch.randn(history + queries, 576).to(torch.bfloat16)
    all_slots = slots_for(table, 0, history + queries)
    qfm.write_cache(all_kv, all_slots, cache, torch.ones(1))
    adapter = paged.PagedFIA()
    paged.write_bf16(cache, all_kv[:history], all_slots[:history])
    batch = SimpleNamespace(
        has_prefill=True,
        slots=all_slots[history:],
        preprocess_segments=((0, queries, True),),
        fia_requests=((0, queries, history, table),),
        fia_mask=torch.ones(2048, 2048, dtype=torch.bool).triu(1),
    )
    adapter.update(cache, batch, all_kv[history:])
    before = cache.view(torch.uint8).clone()
    q = torch.randn(queries, heads, 576).to(torch.bfloat16) / 8
    calls = []

    def native(*args, **kwargs):
        calls.append(kwargs)
        return native_reference(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Paged FIA must not gather/cast historical FP8 data")

    monkeypatch.setattr(paged.torch_npu, "npu_fused_infer_attention_score_v2", native, raising=False)
    monkeypatch.setattr(qfm, "read_fia_history_parts", forbidden)
    out = torch.empty(queries, heads, 512, dtype=torch.bfloat16)
    lse = torch.empty(queries, heads, 1)
    impl = SimpleNamespace(dcp_enabled=False, dcp_rank=0, scale=192**-0.5)
    adapter.fill(impl, q, all_kv[history:], cache, batch, out, lse)
    assert len(calls) == 1
    assert calls[0]["actual_seq_kvlen"] == [history + queries]
    assert calls[0]["sparse_mode"] == 3 and calls[0]["block_size"] == 128
    scores = torch.einsum("qhd,kd->qhk", q.float(), all_kv.float()) * impl.scale
    future = torch.arange(history + queries)[None, :] > (history + torch.arange(queries))[:, None]
    scores.masked_fill_(future[:, None, :], -float("inf"))
    expected = torch.einsum("qhk,kd->qhd", scores.softmax(-1), all_kv[:, :512].float()).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=0.002, rtol=0.01)
    torch.testing.assert_close(lse, scores.logsumexp(-1).unsqueeze(-1), atol=1e-5, rtol=1e-5)
    assert torch.equal(cache.view(torch.uint8), before)


def test_mixed_batch_mirrors_decode_codes_and_preserves_prefill_floats(paged, qfm):
    cache = cache_for(qfm, paged)
    rows = torch.full((10, 576), 0.10009765625, dtype=torch.bfloat16)
    slots = torch.tensor([5, 6, 7, 128, 129, 130, 131, 132, -1, -1])
    qfm.write_cache(rows, slots, cache, torch.ones(1))
    batch = SimpleNamespace(has_prefill=True, slots=slots, preprocess_segments=((0, 3, False), (3, 10, True)))
    adapter = paged.PagedFIA()
    adapter.update(cache, batch, rows)
    latent, raw = adapter.views
    for index, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        actual = torch.cat(
            (latent[slot // 128, 0, :, slot % 128].reshape(-1), raw[slot // 128, 0, :, slot % 128].reshape(-1))
        )
        expected = rows[index] if index >= 3 else rows[index].to(torch.float8_e4m3fn).to(torch.bfloat16)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("rank,history,expected", [(0, 0, 1), (1, 0, 0), (0, 129, 2), (1, 129, 1)])
def test_dcp_retains_history_current_ownership(paged, qfm, monkeypatch, rank, history, expected):
    cache = cache_for(qfm, paged)
    table = torch.tensor([3, 1, 0, 2])
    past = torch.randn(history, 576).to(torch.bfloat16)
    paged.write_bf16(cache, past, slots_for(table, 0, history))
    q = torch.randn(9, 12, 576).to(torch.bfloat16)
    current = torch.randn(9, 576).to(torch.bfloat16)
    batch = SimpleNamespace(fia_requests=((0, 9, history, table),), fia_mask=torch.ones(2048, 2048).bool())
    adapter, calls = paged.PagedFIA(), []

    def native(*args, **kwargs):
        calls.append(kwargs)
        return native_reference(*args, **kwargs)

    monkeypatch.setattr(paged.torch_npu, "npu_fused_infer_attention_score_v2", native, raising=False)
    output = torch.empty(9, 12, 512, dtype=torch.bfloat16)
    lse = torch.empty(9, 12, 1)
    adapter.fill(
        SimpleNamespace(dcp_enabled=True, dcp_rank=rank, scale=192**-0.5), q, current, cache, batch, output, lse
    )
    assert len(calls) == expected
    if expected == 0:
        assert (output == 0).all() and torch.isneginf(lse).all()
    if history:
        assert calls[0]["actual_seq_kvlen"] == [history] and calls[0]["sparse_mode"] == 0
    if rank == 0:
        assert calls[-1].get("block_table") is None and calls[-1]["sparse_mode"] == 3


@pytest.mark.parametrize("role", [None, "kv_producer", "kv_consumer", "kv_both"])
def test_paged_backend_is_explicit_and_excludes_imported_unmirrored_kv(qfm, role):
    impl, config = blocks.selection_inputs(128)
    impl.qk_nope_head_dim = impl.v_head_dim = 128
    config.kimi_k3_c8_prefill_backend = "fia_paged"
    if role is not None:
        impl.vllm_config.kv_transfer_config = SimpleNamespace(
            kv_connector="MooncakeConnectorV2", kv_connector_module_path=None, kv_role=role
        )
    if role in ("kv_consumer", "kv_both"):
        with pytest.raises(ValueError, match="Prefill producer"):
            qfm.select_qfm(impl, config)
    else:
        assert qfm.select_qfm(impl, config)
        assert impl.qfm_fia_paged and impl.qfm_fia_prefill and not impl.qfm_fia_expanded


def test_paged_metadata_skips_full_history_indices_and_preserves_physical_fp8_stride(qfm, monkeypatch):
    impl = base.impl_config()
    impl.qfm_fia_paged = impl.qfm_fia_prefill = True
    impl.enable_mlapo = False
    spec = SimpleNamespace(block_size=768, page_size_padded=2_000_064, page_size_bytes=2_000_064)
    monkeypatch.setattr(qfm, "load_ops", lambda: (None, lambda *a, **kw: torch.empty(1)))
    builder = qfm.QFMMetadataBuilder(impl, torch.device("cpu"), spec)
    assert builder.cache_stride == spec.page_size_bytes // 6

    def forbidden(*args):
        raise AssertionError("Native paged FIA needs no O(history) framework index list")

    monkeypatch.setattr(qfm, "build_fia_history_plan", forbidden)
    common = base.common([33], [131072])
    result = builder.build(common)
    assert not result.fia_history_plans and not result.qfm_needed
    assert result.preprocess_segments == ((0, 33, True),)
    assert result.scatter_plan.stride == spec.page_size_bytes // 6


def test_dispatch_writes_both_caches_before_paged_attention(paged, qfm, monkeypatch):
    cache = cache_for(qfm, paged)
    table = torch.tensor([3, 1, 0, 2], dtype=torch.int32)
    past = torch.randn(129, 576).to(torch.bfloat16)
    paged.write_bf16(cache, past, slots_for(table, 0, 129))
    qfm.write_cache(past, slots_for(table, 0, 129), cache, torch.ones(1))
    q = torch.randn(9, 12, 576).to(torch.bfloat16) / 8
    current = torch.full((9, 576), 0.10009765625, dtype=torch.bfloat16)
    live_slots = slots_for(table, 129, 138)
    adapter = paged.PagedFIA()
    batch = SimpleNamespace(
        slots=live_slots,
        preprocess_segments=((0, 9, True),),
        fia_requests=((0, 9, 129, table),),
        has_prefill=True,
        qfm_needed=False,
        invalid=torch.zeros(9, dtype=torch.bool),
        fia_mask=torch.ones(2048, 2048, dtype=torch.bool).triu(1),
        scatter_plan=None,
    )
    impl = SimpleNamespace(
        qfm_fia_paged=True,
        qfm_fia_expanded=False,
        qfm_mlapo_prefill=False,
        fia_paged=adapter,
        qfm_scale=torch.ones(1),
        dcp_enabled=False,
        dcp_rank=0,
        scale=192**-0.5,
        use_output_gate=False,
        _v_up_proj_batch_major=lambda value: value.reshape(9, -1),
        o_proj=lambda value, **kwargs: (value,),
    )
    calls = []

    def native(*args, **kwargs):
        calls.append(kwargs)
        latent, raw = adapter.views
        for index, slot in enumerate(live_slots.tolist()):
            assert torch.equal(latent[slot // 128, 0, :, slot % 128].reshape(-1), current[index, :512])
            assert torch.equal(raw[slot // 128, 0, :, slot % 128].reshape(-1), current[index, 512:])
        return native_reference(*args, **kwargs)

    monkeypatch.setattr(qfm, "ordinary_qfm_inputs", lambda *args: (q, current))
    monkeypatch.setattr(paged.torch_npu, "npu_fused_infer_attention_score_v2", native, raising=False)
    output = torch.empty(9, 12 * 512, dtype=torch.bfloat16)
    result = qfm.forward_qfm(impl, torch.zeros(9, 4), cache, SimpleNamespace(qfm=batch), output)
    assert result is output and torch.isfinite(output).all() and len(calls) == 1


def test_ordinary_latent_frontend_applies_explicit_q_a_before_q_b(qfm):
    hidden = torch.randn(9, 4).to(torch.bfloat16)
    low_q = torch.randn(9, 3).to(torch.bfloat16)
    kv = torch.randn(9, 576).to(torch.bfloat16)
    observed = []

    def project(value):
        observed.append(value)
        assert value.shape[-1] == 3
        return torch.zeros(9, 12, 512), torch.zeros(9, 12, 64)

    impl = SimpleNamespace(
        fused_qkv_a_proj=None,
        q_a_proj=lambda value: (low_q,),
        q_a_layernorm=lambda value: value + 0.5,
        kv_a_proj_with_mqa=lambda value: (kv,),
        kv_a_layernorm=lambda value: value,
        _q_proj_and_k_up_proj=project,
    )
    query, current = qfm.ordinary_qfm_inputs(impl, hidden, 9)
    assert query.shape == (9, 12, 576) and torch.equal(current, kv)
    assert torch.equal(observed[0], low_q + 0.5)
