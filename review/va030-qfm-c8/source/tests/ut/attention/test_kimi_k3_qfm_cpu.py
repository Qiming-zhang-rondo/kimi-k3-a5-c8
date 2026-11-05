# SPDX-License-Identifier: Apache-2.0
"""CPU contracts only. No claim of NPU or distributed acceptance."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def qfm(monkeypatch):
    def scatter(values, indices, *, key_cache):
        live = indices >= 0
        key_cache.view(torch.uint8)[indices[live].long(), 0, 0] = values.view(torch.uint8)[live, 0]

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_scatter_pa_cache=scatter))
    monkeypatch.setitem(sys.modules, "vllm.compilation.backends", SimpleNamespace(model_tag="backbone"))
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/kimi_k3_qfm.py"
    spec = importlib.util.spec_from_file_location("qfm_cpu_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("gap,offset", [(0, 0), (256, 256), (1024, 512)])
def test_nz_writer_preserves_offset_padding_and_skips_invalid(qfm, gap, offset):
    stride = 128 * 576 + gap
    backing = torch.full((offset + 3 * stride + 256,), 0x5A, dtype=torch.uint8)
    raw = backing[offset : offset + 3 * stride]
    cache = qfm.cache_view(raw, 3, stride)
    assert cache.storage_offset() == offset
    slots = torch.tensor([128, -1, 0, 383, 127])
    rows = torch.randn(5, 576)
    scale = torch.tensor([0.03125])
    expected = backing.clone()
    quant = (rows / scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    for i, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        page, token = divmod(slot, 128)
        for tile in range(18):
            start = offset + page * stride + tile * 128 * 32 + token * 32
            expected[start : start + 32] = quant[i, tile * 32 : (tile + 1) * 32]
    qfm.write_cache(rows, slots, cache, scale)
    assert torch.equal(backing, expected)
    before = backing.clone()
    qfm.write_cache(rows, torch.full_like(slots, -1), cache, scale)
    assert torch.equal(backing, before)


def impl_config(dcp=1, rank=0):
    return SimpleNamespace(
        num_heads=12,
        dcp_rank=rank,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(decode_context_parallel_size=dcp, cp_kv_cache_interleave_size=16),
            model_config=SimpleNamespace(max_model_len=131072),
        ),
    )


def common(queries, lengths, capacity=None):
    boundaries = [0]
    for length in queries:
        boundaries.append(boundaries[-1] + length)
    actual = boundaries[-1]
    offsets = torch.tensor(boundaries, dtype=torch.int32)
    return SimpleNamespace(
        num_actual_tokens=actual,
        num_input_tokens=capacity or actual,
        num_reqs=len(queries),
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
        block_table_tensor=torch.arange(len(queries) * 1024, dtype=torch.int32).view(len(queries), 1024),
        slot_mapping=torch.arange(actual, dtype=torch.int64),
        causal=True,
        is_prefilling=torch.tensor([q > 16 for q in queries]),
        max_query_len=max(queries),
    )


def builder(qfm, monkeypatch, dcp=1, rank=0):
    calls = []

    def metadata(lengths, *args, **kwargs):
        calls.append((lengths.clone(), kwargs))
        return lengths.clone()

    monkeypatch.setattr(qfm, "load_ops", lambda: (None, metadata))
    return qfm.QFMMetadataBuilder(impl_config(dcp, rank), torch.device("cpu")), calls


def test_chunked_prefill_groups_one_native_batch_and_no_dense_indices(qfm, monkeypatch):
    b, calls = builder(qfm, monkeypatch)
    c = common([8192], [131072])
    result = b.build(c)
    assert result.used.numel() == 512
    assert result.used.unique().tolist() == [16]
    assert result.lengths[0] == 122896 and result.lengths[-1] == 131072
    assert result.table.shape == (512, 1024)
    assert result.mask_mode == 3
    assert not b.buffers  # no persistent eager-shape accumulation
    assert torch.equal(result.schedule, result.lengths)


def test_fia_stage_routing_keeps_decode_and_speculation(qfm, monkeypatch):
    b, _ = builder(qfm, monkeypatch)
    b.fia_prefill = True
    c = common([1, 33, 9], [129, 513, 99])
    c.is_prefilling = torch.tensor([False, True, False])
    result = b.build(c)
    assert [(a, z, h) for a, z, h, _ in result.fia_requests] == [(1, 34, 480)]
    assert result.used.tolist() == [1, 0, 0, 0, 9]
    assert result.qfm_needed
    assert torch.equal(result.fia_requests[0][3], c.block_table_tensor[1])


@pytest.mark.parametrize("rank", [0, 1])
def test_fia_dcp_history_excludes_current_and_uses_device_lengths(qfm, monkeypatch, rank):
    b, _ = builder(qfm, monkeypatch, 2, rank)
    b.fia_prefill = True
    c = common([33], [513])
    c.seq_lens_cpu = torch.tensor([9999])
    result = b.build(c)
    assert result.fia_requests[0][2] == sum((p // 16) % 2 == rank for p in range(480))
    assert not result.qfm_needed
    assert not result.used.any()


def test_fia_decode_graph_unchanged_and_prefill_capture_rejected(qfm, monkeypatch):
    b, _ = builder(qfm, monkeypatch)
    b.fia_prefill = True
    c = common([9], [129])
    result = b.build(c, retain_for_graph=True)
    assert not result.fia_requests and result.qfm_needed
    c.is_prefilling.fill_(True)
    with pytest.raises(ValueError, match="eager only"):
        b.build(c, retain_for_graph=True)


@pytest.mark.parametrize("gap,offset", [(0, 0), (256, 256)])
def test_fia_history_reads_permuted_pages_and_preserves_guards(qfm, gap, offset):
    stride = 128 * 576 + gap
    backing = torch.full((offset + 4 * stride + 256,), 0x5A, dtype=torch.uint8)
    cache = qfm.cache_view(backing[offset : offset + 4 * stride], 4, stride)
    table = torch.tensor([3, 1, 2, 0])
    length = 257
    positions = torch.arange(length)
    slots = table[positions // 128] * 128 + positions % 128
    values = torch.randn(length, 576)
    scale = torch.tensor([0.5])
    qfm.write_cache(values, slots, cache, scale)
    before = backing.clone()
    actual = qfm.read_fia_history(cache, table, length, scale)
    expected = ((values / scale).to(torch.float8_e4m3fn).float() * scale).to(torch.bfloat16)
    assert torch.equal(actual[:, 0], expected)
    assert torch.equal(backing, before)


def test_fia_split_numerics_and_dcp_current_counted_once(qfm, monkeypatch):
    def fia(query, key, value, **kw):
        q = torch.cat((query, kw["query_rope"]), dim=-1).float()
        k = torch.cat((key, kw["key_rope"]), dim=-1)[:, 0].float()
        scores = torch.einsum("thd,kd->thk", q, k) * kw["softmax_scale"]
        if kw["sparse_mode"] == 3:
            mask = torch.arange(k.shape[0])[None, :] > torch.arange(q.shape[0])[:, None]
            scores.masked_fill_(mask[:, None], -float("inf"))
        out = torch.einsum("thk,kd->thd", scores.softmax(-1), value[:, 0].float())
        return out.to(torch.bfloat16), scores.logsumexp(-1).unsqueeze(-1)

    monkeypatch.setattr(qfm.torch_npu, "npu_fused_infer_attention_score_v2", fia, raising=False)
    torch.manual_seed(7)
    q = torch.randn(9, 2, 576).to(torch.bfloat16)
    history = torch.randn(17, 1, 576).to(torch.bfloat16)
    current = torch.randn(9, 1, 576).to(torch.bfloat16)
    mask = torch.ones(2048, 2048, dtype=torch.bool).triu_(1)
    # Independent full causal reference.
    keys = torch.cat((history, current))[:, 0].float()
    scores = torch.einsum("thd,kd->thk", q.float(), keys) / 24
    invalid = torch.arange(26)[None, :] > (17 + torch.arange(9))[:, None]
    scores.masked_fill_(invalid[:, None], -float("inf"))
    expected = torch.einsum("thk,kd->thd", scores.softmax(-1), keys[:, :512])
    left = qfm.merge_fia_parts(
        qfm.fia_partial(q, history[::2].contiguous(), 1 / 24),
        qfm.fia_partial(q, current, 1 / 24, mask),
    )
    right = qfm.fia_partial(q, history[1::2].contiguous(), 1 / 24)
    actual, lse = qfm.merge_fia_parts(left, right)
    torch.testing.assert_close(actual.float(), expected, atol=0.015, rtol=0.015)
    torch.testing.assert_close(lse[..., 0], scores.logsumexp(-1), atol=1e-5, rtol=1e-5)
    empty = qfm.fia_partial(q, history[:0], 1 / 24)
    out, lse = qfm.merge_fia_parts(empty, empty)
    assert (out == 0).all() and torch.isneginf(lse).all()


def test_apc_mixed_requests_and_graph_padding(qfm, monkeypatch):
    b, _ = builder(qfm, monkeypatch)
    result = b.build(common([1, 17, 9], [129, 257, 511], 48))
    assert result.used.tolist() == [1, 16, 1, 9, 0, 0]
    assert result.lengths.tolist() == [129, 256, 257, 511, 0, 0]
    assert (result.slots[27:] == -1).all()
    assert result.live.sum() == 27


def test_graph_buffers_refresh_same_addresses_after_empty(qfm, monkeypatch):
    b, _ = builder(qfm, monkeypatch)
    first = b.build(common([128], [256], 128), retain_for_graph=True)
    addresses = [getattr(first, field).data_ptr() for field in ("cu", "used", "lengths", "slots", "table", "schedule")]
    for queries, lengths in [([4], [127]), ([9], [129]), ([0], [0]), ([1, 8], [1, 129])]:
        c = common(queries, lengths, 128)
        result = b.build(c)
        assert result is first
        assert addresses == [
            getattr(result, field).data_ptr() for field in ("cu", "used", "lengths", "slots", "table", "schedule")
        ]
        assert result.used.sum() == sum(queries)
        assert torch.equal(result.schedule, result.lengths)
        assert (result.slots[sum(queries) :] == -1).all()


@pytest.mark.parametrize("rank", [0, 1])
def test_dcp_per_query_prefix_not_naive_local_plus_one(qfm, monkeypatch, rank):
    b, _ = builder(qfm, monkeypatch, 2, rank)
    result = b.build(common([17], [129]))
    expected = [sum((p // 16) % 2 == rank for p in range(length)) for length in range(113, 130)]
    assert result.lengths.tolist() == expected
    assert result.mask_mode == 0 and result.max_q == 1
    assert result.table.shape[0] == 17


def test_official_unit_scale_is_internal_not_a_user_knob(qfm):
    assert qfm.QFM_KV_DESCALE == 1.0
    root = Path(__file__).resolve().parents[3]
    config = ast.parse((root / "vllm_ascend/ascend_config.py").read_text())
    fields = [
        node.target.id
        for node in ast.walk(config)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    assert "qfm_kv_scale" not in fields
    impl = (root / "vllm_ascend/attention/mla_v1.py").read_text()
    assert "value = QFM_KV_DESCALE" in impl
    assert "resolve_kv_scale" not in impl
    assert not qfm.select_qfm(SimpleNamespace(is_draft_model=True), SimpleNamespace(enable_kimi_k3_qfm_c8=True))


@pytest.mark.parametrize("value", [-500.0, -448.0, -16.0, -0.00001, 0.0, 0.00001, 16.0, 448.0, 500.0])
def test_unit_scale_writer_matches_direct_fp8_and_retains_prior_pages(qfm, value):
    cache = qfm.cache_view(torch.zeros(2 * 128 * 576, dtype=torch.uint8), 2, 128 * 576)
    scale = torch.tensor([qfm.QFM_KV_DESCALE])
    first = torch.full((1, 576), 32.0)
    qfm.write_cache(first, torch.tensor([0]), cache, scale)
    previous = cache[0].view(torch.uint8).clone()
    rows = torch.full((1, 576), value)
    qfm.write_cache(rows, torch.tensor([128]), cache, scale)
    assert torch.equal(cache[0].view(torch.uint8), previous)
    actual = cache[1, 0, :, 0].reshape(1, 576).float() * scale
    expected = rows.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_forward_writes_before_reader_and_masks_padding(qfm, monkeypatch):
    b, _ = builder(qfm, monkeypatch)
    batch = b.build(common([4], [4], capacity=8))
    cache = qfm.cache_view(torch.zeros(128 * 576, dtype=torch.uint8), 1, 128 * 576)
    hidden = torch.randn(8, 576)
    scale = torch.tensor([qfm.QFM_KV_DESCALE])
    observed = []

    def attention(q, kv, q_scale, kv_scale, table, lengths, *args, **kwargs):
        assert kv_scale.shape == (1,) and kv_scale.dtype == torch.float32
        assert kv_scale.item() == 1.0
        # The reader must see the current write, not stale pages or a copy.
        stored = kv[0, 0].permute(1, 0, 2).reshape(128, 576).float()
        expected = (hidden[:4] / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        torch.testing.assert_close(stored[:4], expected, rtol=0, atol=0)
        assert torch.count_nonzero(stored[4:]) == 0
        assert lengths.tolist() == [4, 0]
        assert kwargs["layout_q"] == "TND" and kwargs["layout_kv"] == "PA_NZ"
        observed.append(q.shape)
        return torch.ones(8, 12, 512), None

    impl = SimpleNamespace(
        fused_qkv_a_proj=None,
        kv_a_proj_with_mqa=lambda x: (x,),
        _q_proj_and_k_up_proj=lambda x: (x[:, None, :512].expand(-1, 12, -1), x[:, None, 512:].expand(-1, 12, -1)),
        kv_a_layernorm=lambda x: x,
        qfm_scale=scale,
        dcp_enabled=False,
        qfm_ops=(attention, None),
        scale=0.1,
        _v_up_proj_batch_major=lambda x: x.flatten(1),
        use_output_gate=True,
        g_proj=lambda x: (torch.zeros(8, 12 * 512),),
        o_proj=lambda x, **kwargs: (x,),
    )
    output = torch.empty(8, 12 * 512)
    assert qfm.forward_qfm(impl, hidden, cache, SimpleNamespace(qfm=batch), output) is output
    assert observed == [torch.Size([8, 12, 576])]
    torch.testing.assert_close(output[:4], torch.full_like(output[:4], 0.5))
    assert torch.count_nonzero(output[4:]) == 0
