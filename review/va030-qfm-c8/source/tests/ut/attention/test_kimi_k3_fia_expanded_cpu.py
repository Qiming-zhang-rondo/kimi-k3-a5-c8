# SPDX-License-Identifier: Apache-2.0
"""CPU numerical contracts; native FIA and distributed performance need NPU."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.attention import test_kimi_k3_qfm_cpu as base

qfm = base.qfm


@pytest.fixture
def expanded(qfm, monkeypatch):
    casts, calls = [], []

    def format_cast(weight, fmt):
        assert fmt == 29 and weight.dtype == torch.bfloat16
        casts.append(weight.clone())
        return weight.clone()

    def native(query, key, value, **kwargs):
        assert kwargs["input_layout"] == "TND"
        assert kwargs["num_heads"] == query.shape[1] == kwargs["num_key_value_heads"]
        assert kwargs["actual_seq_lengths"] == [query.shape[0]]
        assert kwargs["actual_seq_lengths_kv"] == [key.shape[0]]
        assert kwargs["softmax_lse_flag"] and kwargs["block_table"] is None
        assert kwargs["block_size"] == 0 and kwargs["antiquant_scale"] is None
        assert query.dtype == key.dtype == value.dtype == torch.bfloat16
        assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
        scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * kwargs["scale"]
        if kwargs["sparse_mode"] == 3:
            assert kwargs["atten_mask"].dtype == torch.bool
            assert query.shape[0] == key.shape[0]
            scores.masked_fill_(torch.ones(query.shape[0], key.shape[0], dtype=torch.bool).triu(1), -float("inf"))
        else:
            assert kwargs["sparse_mode"] == 0 and kwargs["atten_mask"] is None
        calls.append((query.shape, key.shape, kwargs["sparse_mode"]))
        output = torch.einsum("hqk,khv->qhv", scores.softmax(-1), value.float()).to(torch.bfloat16)
        return output, scores.logsumexp(-1)

    monkeypatch.setattr(qfm.torch_npu, "npu_format_cast", format_cast, raising=False)
    monkeypatch.setattr(qfm.torch_npu, "npu_fused_infer_attention_score", native, raising=False)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.kimi_k3_qfm", qfm)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.attention.utils",
        SimpleNamespace(ascend_chunked_prefill_workspace_size=lambda config: config.workspace_tokens),
    )
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/kimi_k3_fia_expanded.py"
    spec = importlib.util.spec_from_file_location("expanded_fia_cpu_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, casts=casts, calls=calls)


def make_impl(expanded, heads=2, nope=16, value=12, dcp=False, rank=0):
    impl = SimpleNamespace(
        num_heads=heads,
        kv_lora_rank=512,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=64,
        v_head_dim=value,
        W_UK_T=(torch.randn(heads, nope, 512) * 0.02).to(torch.bfloat16),
        W_UV=(torch.randn(heads, 512, value) * 0.02).to(torch.bfloat16),
        vllm_config=SimpleNamespace(workspace_tokens=4096),
        dcp_enabled=dcp,
        dcp_rank=rank,
        scale=(nope + 64) ** -0.5,
    )
    impl.fia_expanded = expanded.module.ExpandedFIA(impl)
    return impl


def history_cache(qfm, length, *, gap=256, offset=256):
    stride = 128 * 576 + gap
    pages = max(1, (length + 127) // 128)
    table = torch.arange(pages - 1, -1, -1)
    backing = torch.full((offset + pages * stride + 256,), 0x5A, dtype=torch.uint8)
    cache = qfm.cache_view(backing[offset : offset + pages * stride], pages, stride)
    rows = torch.randn(length, 576).to(torch.bfloat16)
    positions = torch.arange(length)
    slots = table[positions // 128] * 128 + positions % 128
    qfm.write_cache(rows, slots, cache, torch.ones(1))
    return cache, backing, table, rows.to(torch.float8_e4m3fn).to(torch.bfloat16)


def reference(query, history, current, head_weight, scale):
    """Independent expanded whole-history FP32 softmax with BF16 projections."""
    heads, latent, width = head_weight.shape
    nope = query.shape[-1] - 64
    value = width - nope
    flat = head_weight.permute(1, 0, 2).reshape(latent, heads * width)
    rows = torch.cat((history, current)) if current is not None else history
    if rows.shape[0] == 0:
        return torch.zeros(query.shape[0], heads, value), torch.full((*query.shape[:2], 1), -float("inf"))
    projected = (rows[:, :latent] @ flat).reshape(len(rows), heads, width)
    key = torch.cat((projected[:, :, :nope], rows[:, None, latent:].expand(-1, heads, -1)), -1)
    scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if current is not None:
        q_positions = torch.arange(query.shape[0])[:, None]
        kv_positions = torch.arange(len(rows))[None, :]
        scores.masked_fill_(kv_positions > len(history) + q_positions, -float("inf"))
    result = torch.einsum("hqk,khv->qhv", scores.softmax(-1), projected[:, :, nope:].float())
    return result, scores.logsumexp(-1).transpose(0, 1).unsqueeze(-1)


@pytest.mark.parametrize(
    "history_length,chunk,heads,nope,value",
    [(0, 3, 2, 16, 12), (1, 1, 3, 24, 8), (127, 31, 2, 16, 12), (129, 32, 3, 24, 8), (129, 257, 12, 128, 128)],
)
def test_full_query_bounded_history_matches_independent_reference(
    qfm, expanded, history_length, chunk, heads, nope, value
):
    torch.manual_seed(20261106 + history_length)
    impl = make_impl(expanded, heads, nope, value)
    impl.fia_expanded.max_context_tokens = chunk
    q = torch.randn(9, heads, nope + 64).to(torch.bfloat16)
    current = torch.randn(9, 576).to(torch.bfloat16)
    cache, backing, table, history = history_cache(qfm, history_length)
    before = backing.clone()
    batch = SimpleNamespace(
        fia_requests=((0, 9, history_length, table),),
        fia_history_plans=(qfm.build_fia_history_plan(table, history_length),),
        fia_mask=torch.ones(32, 32, dtype=torch.bool).triu(1),
    )
    output = torch.full((9, heads, 512), 999.0, dtype=torch.bfloat16)
    lse = torch.empty(9, heads, 1)
    expanded.module.fill_expanded_fia(impl, q, current, cache, batch, output, lse)
    expected, expected_lse = reference(q, history, current, impl.fia_expanded.local_head_weight, impl.scale)
    torch.testing.assert_close(output[..., :value].float(), expected, atol=0.008, rtol=0.015)
    torch.testing.assert_close(lse, expected_lse, atol=2e-6, rtol=2e-6)
    assert (output[..., value:] == 0).all() and torch.equal(backing, before)
    assert len(expanded.casts) == 1  # No per-context-chunk weight conversion.
    assert all(call[0][0] == 9 for call in expanded.calls)
    assert all(call[1][0] <= chunk for call in expanded.calls if call[2] == 0)
    assert expanded.calls[-1][2] == 3


def test_mixed_container_preserves_decode_and_multiple_prefill_requests(qfm, expanded):
    torch.manual_seed(2)
    impl = make_impl(expanded)
    impl.fia_expanded.max_context_tokens = 7
    q = torch.randn(13, 2, 80).to(torch.bfloat16)
    current = torch.randn(13, 576).to(torch.bfloat16)
    cache, _, table, history = history_cache(qfm, 129, gap=1024, offset=512)
    batch = SimpleNamespace(
        fia_requests=((2, 6, 127, table), (9, 13, 129, table)),
        fia_history_plans=(qfm.build_fia_history_plan(table, 127), qfm.build_fia_history_plan(table, 129)),
    )
    output = torch.full((13, 2, 512), 31.0, dtype=torch.bfloat16)
    lse = torch.full((13, 2, 1), 29.0)
    expanded.module.fill_expanded_fia(impl, q, current, cache, batch, output, lse)
    for begin, end, length, _ in batch.fia_requests:
        expected, expected_lse = reference(
            q[begin:end], history[:length], current[begin:end], impl.fia_expanded.local_head_weight, impl.scale
        )
        torch.testing.assert_close(output[begin:end, :, :12].float(), expected, atol=0.008, rtol=0.015)
        torch.testing.assert_close(lse[begin:end], expected_lse, atol=2e-6, rtol=2e-6)
        assert (output[begin:end, :, 12:] == 0).all()
    assert (output[[0, 1, 6, 7, 8]] == 31).all()
    assert (lse[[0, 1, 6, 7, 8]] == 29).all()


@pytest.mark.parametrize("rank,history_length", [(0, 0), (1, 0), (1, 129)])
def test_dcp_head_gather_order_and_current_rank_ownership(qfm, expanded, rank, history_length):
    torch.manual_seed(12)
    impl = make_impl(expanded, dcp=True, rank=rank)
    local = impl.fia_expanded.local_head_weight
    remote = local.flip(0) * 1.5
    global_weight = torch.cat((local, remote))
    events = []

    def gather(weight, dim):
        events.append("weights")
        assert dim == 0 and torch.equal(weight, local)
        return global_weight

    impl._dcp_all_gather = gather
    # Framework must call prepare before Query head gathering. Empty ranks
    # still participate in the same first-prefill weight collective.
    impl.fia_expanded.prepare_for_dcp(impl)
    events.append("queries")
    assert events == ["weights", "queries"]
    impl.fia_expanded.max_context_tokens = 17
    q = torch.randn(4, 4, 80).to(torch.bfloat16)
    current = torch.randn(4, 576).to(torch.bfloat16)
    cache, _, table, history = history_cache(qfm, history_length)
    batch = SimpleNamespace(fia_requests=((0, 4, history_length, table),))
    output = torch.empty(4, 4, 512, dtype=torch.bfloat16)
    lse = torch.empty(4, 4, 1)
    for _ in range(2):
        expanded.module.fill_expanded_fia(impl, q, current, cache, batch, output, lse)
    assert events == ["weights", "queries"]  # Cached across calls and KV chunks.
    expected, expected_lse = reference(q, history, current if rank == 0 else None, global_weight, impl.scale)
    torch.testing.assert_close(output[..., :12].float(), expected, atol=0.008, rtol=0.015)
    torch.testing.assert_close(lse, expected_lse, atol=2e-6, rtol=2e-6)
    assert (output[..., 12:] == 0).all()
    assert sum(call[2] == 3 for call in expanded.calls) == (2 if rank == 0 else 0)
    if rank == 1 and history_length == 0:
        assert not expanded.calls


def test_reload_preserves_private_addresses_but_invalidates_remote_weights(expanded):
    impl = make_impl(expanded, dcp=True)
    key_before, value_before = impl.W_UK_T.clone(), impl.W_UV.clone()
    state = impl.fia_expanded
    local_pointer, flat_pointer = state.local_head_weight.data_ptr(), state.w_kv.data_ptr()
    calls = []

    def gather(weight, dim):
        calls.append(dim)
        return torch.cat((weight, weight + 0.125))

    impl._dcp_all_gather = gather
    state.prepare_for_dcp(impl)
    assert state.global_w_kv is not None
    assert torch.equal(impl.W_UK_T, key_before) and torch.equal(impl.W_UV, value_before)
    impl.W_UK_T.add_(0.0625)
    state.refresh_weights(impl)
    assert state.global_w_kv is None and state.global_head_weight is None
    assert state.local_head_weight.data_ptr() == local_pointer and state.w_kv.data_ptr() == flat_pointer
    state.prepare_for_dcp(impl)
    assert calls == [0, 0]
    torch.testing.assert_close(state.local_head_weight[:, :, :16], impl.W_UK_T.transpose(1, 2))
    assert torch.equal(impl.W_UV, value_before)


def test_many_chunk_merge_stays_fp32(qfm, expanded, monkeypatch):
    impl = make_impl(expanded)
    impl.fia_expanded.max_context_tokens = 1
    q = torch.randn(4, 2, 80).to(torch.bfloat16)
    current = torch.randn(4, 576).to(torch.bfloat16)
    cache, _, table, _ = history_cache(qfm, 33)
    original = expanded.module._merge
    states = []

    def checked(left, right):
        assert right[0].dtype == right[1].dtype == torch.float32
        if left is not None:
            assert left[0].dtype == left[1].dtype == torch.float32
        result = original(left, right)
        states.append((right[0].double(), right[1].double()))
        return result

    monkeypatch.setattr(expanded.module, "_merge", checked)
    output = torch.empty(4, 2, 512, dtype=torch.bfloat16)
    lse = torch.empty(4, 2, 1)
    expanded.module.fill_expanded_fia(
        impl, q, current, cache, SimpleNamespace(fia_requests=((0, 4, 33, table),)), output, lse
    )
    all_lse = torch.stack([part[1] for part in states])
    total_lse = all_lse.logsumexp(0)
    expected = sum(part[0] * (part[1] - total_lse).exp() for part in states)
    torch.testing.assert_close(output[..., :12], expected.to(torch.bfloat16), atol=0, rtol=0)
    torch.testing.assert_close(lse.double(), total_lse, atol=1e-6, rtol=1e-6)
    assert len(states) == 34


def test_workspace_budget_accounts_for_expanded_global_heads(expanded):
    impl = make_impl(expanded, heads=12, nope=128, value=128, dcp=True)
    state = impl.fia_expanded
    local_limit = state.max_context_tokens
    assert local_limit < impl.vllm_config.workspace_tokens
    impl._dcp_all_gather = lambda weight, dim: torch.cat((weight, weight))
    state.prepare_for_dcp(impl)
    assert state.max_context_tokens < local_limit
    state.max_context_tokens = 257
    state.refresh_weights(impl)
    state.prepare_for_dcp(impl)
    assert state.max_context_tokens == 257  # Explicit experiment override retained.
