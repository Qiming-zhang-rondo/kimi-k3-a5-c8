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
        enable_mlapo=True,
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


def _per_token_reference(c, dcp_size, rank):
    """Independent request/token ownership reference, including empty shards."""
    bounds = c.query_start_loc_cpu.tolist()
    lengths = c.seq_lens.tolist()
    slots = c.slot_mapping.tolist()
    owners, live, visible, used, local_lengths, mapped_slots = [], [], [], [], [], []
    capacity = max(c.num_actual_tokens, c.num_input_tokens)
    for token in range(capacity):
        owner = next((i for i, end in enumerate(bounds[1:]) if token < end), c.num_reqs - 1)
        active = token < c.num_actual_tokens and token < bounds[owner + 1] and lengths[owner] > 0
        prefix = lengths[owner]
        if c.causal:
            prefix -= bounds[owner + 1] - token - 1
        prefix = max(prefix, 0)
        local = sum((position // 16) % dcp_size == rank for position in range(prefix))
        local = local if active else 0
        owners.append(owner)
        live.append(active)
        visible.append(active and local > 0)
        used.append(int(active))
        local_lengths.append(local)
        mapped_slots.append(slots[token] if active and token < len(slots) else -1)
    return dict(owners=owners, live=live, visible=visible, used=used, lengths=local_lengths, slots=mapped_slots)


@pytest.mark.parametrize("dcp,rank,retained", [(1, 0, True), (2, 0, False), (2, 1, False), (4, 3, True)])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "queries,lengths,capacity",
    [
        ([1, 8], [127, 129], 16),
        ([0, 4, 0, 2], [0, 129, 0, 17], 16),
        ([1, 1], [1, 0], 8),
        ([0], [0], 8),
        ([17], [129], 32),
    ],
)
def test_per_token_metadata_matches_request_reference(
    qfm, monkeypatch, dcp, rank, retained, causal, queries, lengths, capacity
):
    b, _ = builder(qfm, monkeypatch, dcp=dcp, rank=rank)
    c = common(queries, lengths, capacity)
    c.causal = causal
    # Distinct physical slots prevent confusing token IDs with cache addresses.
    c.slot_mapping = torch.arange(c.num_actual_tokens, dtype=torch.int64).flip(0) + 128
    expected = _per_token_reference(c, dcp, rank)
    batch = b.build(c, retain_for_graph=retained)
    assert batch.cu.tolist() == list(range(capacity + 1))
    for name in ("live", "visible", "used", "lengths", "slots"):
        assert getattr(batch, name).tolist() == expected[name], name
    assert torch.equal(batch.table, c.block_table_tensor[expected["owners"]])
    assert batch.mask_mode == 0 and batch.max_q == 1


def test_per_token_refresh_clears_only_unprovided_slots_without_stale_values(qfm, monkeypatch):
    b, _ = builder(qfm, monkeypatch)
    first_input = common([8], [128], 8)
    first = b.build(first_input, retain_for_graph=True)
    pointers = {
        name: getattr(first, name).data_ptr() for name in ("cu", "used", "lengths", "slots", "table", "schedule")
    }
    for queries, lengths, supplied_slots in [
        ([4, 2], [129, 17], [100, 101, 102]),
        ([1], [1], [300] * 8),
        ([0], [0], []),
        ([3, 5], [130, 200], list(reversed(range(8)))),
    ]:
        c = common(queries, lengths, 8)
        c.slot_mapping = torch.tensor(supplied_slots, dtype=torch.int64)
        expected = _per_token_reference(c, 1, 0)
        result = b.build(c)
        assert result is first
        assert pointers == {name: getattr(result, name).data_ptr() for name in pointers}
        for name in ("live", "visible", "used", "lengths", "slots"):
            assert getattr(result, name).tolist() == expected[name], name
        assert torch.equal(result.table, c.block_table_tensor[expected["owners"]])


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

    def prolog(h, slots, destination, **kwargs):
        qfm.write_cache(h, slots, destination, scale)
        return h[:, None, :].expand(-1, 12, -1).float().to(torch.float8_e4m3fn), torch.ones(h.shape[0], 12)

    impl = SimpleNamespace(
        qfm_prolog=prolog,
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


def test_missing_qfm_registration_reports_loaded_provider(qfm, monkeypatch):
    monkeypatch.setattr(
        qfm.importlib, "import_module", lambda _: SimpleNamespace(__file__="/private/qfm/python/ops/__init__.py")
    )
    monkeypatch.setattr(torch, "ops", SimpleNamespace(cann_ops_transformer=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="not registered; loaded provider=/private/qfm/python/ops/__init__.py"):
        qfm.load_ops()


def test_forward_has_no_unfused_fallback(qfm):
    batch = SimpleNamespace(slots=torch.arange(1))
    with pytest.raises(RuntimeError, match="initialized DSL Prolog"):
        qfm.forward_qfm(
            SimpleNamespace(qfm_prolog=None),
            torch.zeros(1, 1024),
            None,
            SimpleNamespace(qfm=batch),
            torch.empty(1),
        )
