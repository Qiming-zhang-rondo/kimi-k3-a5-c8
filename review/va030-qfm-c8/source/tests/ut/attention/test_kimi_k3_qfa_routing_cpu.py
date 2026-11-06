# SPDX-License-Identifier: Apache-2.0
"""CPU expanded-QFA dispatch; native/HCCL/model acceptance requires A5."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.ut.attention import test_kimi_k3_qfa_expanded_cpu as qfa_base
from tests.ut.attention import test_kimi_k3_qfm_cpu as qfm_base

qfa = qfa_base.qfa
qfm = qfm_base.qfm


@pytest.fixture
def routing(qfm, qfa, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.kimi_k3_qfm", qfm)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.kimi_k3_qfa_expanded", qfa.module)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.attention.utils",
        SimpleNamespace(ascend_chunked_prefill_workspace_size=lambda config: config.workspace_tokens),
    )
    monkeypatch.setattr(qfa.module.torch_npu, "npu_format_cast", lambda weight, fmt: weight.clone(), raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("The QFA candidate must not execute the BF16 FIA fallback")

    monkeypatch.setattr(qfa.module.torch_npu, "npu_fused_infer_attention_score", forbidden, raising=False)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/kimi_k3_fia_expanded.py"
    spec = importlib.util.spec_from_file_location("qfa_routing_expanded_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, qfa=qfa, qfm=qfm)


def layer(routing, *, enabled=True, dcp=False, rank=0):
    impl = SimpleNamespace(
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        num_heads=12,
        qfm_qfa_prefill=enabled,
        dcp_enabled=dcp,
        dcp_rank=rank,
        scale=192**-0.5,
        W_UK_T=(torch.randn(12, 128, 512) / 32).to(torch.bfloat16),
        W_UV=(torch.randn(12, 512, 128) / 32).to(torch.bfloat16),
        vllm_config=SimpleNamespace(workspace_tokens=4096),
    )
    impl.fia_expanded = routing.module.ExpandedFIA(impl)
    return impl


def inputs(routing, history_length=65, heads=12):
    qfm = routing.qfm
    stride, offset = 128 * 576 + 256, 256
    backing = torch.full((offset + 2 * stride + 256,), 0x5A, dtype=torch.uint8)
    cache = qfm.cache_view(backing[offset : offset + 2 * stride], 2, stride)
    history = torch.randn(history_length, 576).to(torch.bfloat16)
    table = torch.tensor([1, 0])
    positions = torch.arange(history_length)
    slots = table[positions // 128] * 128 + positions % 128
    qfm.write_cache(history, slots, cache, torch.ones(1))
    query = torch.randn(14, heads, 192).to(torch.bfloat16)
    current = torch.randn(14, 576).to(torch.bfloat16)
    qfa_mask = torch.ones((2048, 2048), dtype=torch.int8).triu(1)
    batch = SimpleNamespace(
        fia_requests=((2, 11, history_length, table),),
        fia_history_plans=(qfm.build_fia_history_plan(table, history_length),),
        fia_mask=qfa_mask.bool(),
        qfa_mask=qfa_mask,
        qfa_metadata={},
    )
    return query, current, cache, backing, batch


def test_fill_qfa_quantizes_full_q_once_per_request_and_reuses_layer_independent_plans(routing, monkeypatch):
    torch.manual_seed(65)
    impl = layer(routing)
    assert isinstance(impl.fia_expanded.qfa, routing.qfa.module.ExpandedQFA)
    impl.fia_expanded.max_context_tokens = 32
    query, current, cache, backing, batch = inputs(routing)
    before = backing.clone()
    prepared, masks, kv_inputs = [], [], []
    adapter = impl.fia_expanded.qfa
    old_prepare, old_quant, old_attention = adapter.prepare_query, adapter.quantize_kv, adapter.attention

    def prepare(query):
        result = old_prepare(query)
        prepared.append(result)
        return result

    def quant(key, value):
        kv_inputs.append((key.clone(), value.clone()))
        return old_quant(key, value)

    def attention(*args, **kwargs):
        masks.append(kwargs["attn_mask"])
        assert kwargs["softmax_scale"] == impl.scale
        return old_attention(*args, **kwargs)

    monkeypatch.setattr(adapter, "prepare_query", prepare)
    monkeypatch.setattr(adapter, "quantize_kv", quant)
    monkeypatch.setattr(adapter, "attention", attention)
    output = torch.full((14, 12, 512), 19.0, dtype=torch.bfloat16)
    lse = torch.full((14, 12, 1), 23.0)
    routing.module.fill_expanded_fia(impl, query, current, cache, batch, output, lse)
    assert len(prepared) == 1 and prepared[0].values.shape == (9, 12, 256)
    assert [key.shape[0] for key, _ in kv_inputs] == [32, 32, 1, 9]
    assert len(routing.qfa.quant_calls) == 9  # One full Q, then K/V for four KV parts.
    assert masks[:3] == [None, None, None]
    assert masks[-1].dtype == torch.int8 and masks[-1].data_ptr() == batch.qfa_mask.data_ptr()
    assert len(routing.qfa.native.metadata_calls) == len(batch.qfa_metadata) == 3
    assert (output[2:11, :, 128:] == 0).all()
    untouched = torch.tensor([0, 1, 11, 12, 13])
    assert (output[untouched] == 19).all() and (lse[untouched] == 23).all()
    assert torch.equal(backing, before)

    # Independently quantize each actual BF16 expanded KV block on its proper
    # axis, then merge the BF16 native partials in FP64 for comparison.
    expected_query, _ = qfa_base.reference_quantized(F.pad(query[2:11], (0, 64)), -1)
    partials = []
    for index, (key, value) in enumerate(kv_inputs):
        expected_key, _ = qfa_base.reference_quantized(F.pad(key, (0, 64)), -1)
        expected_value, _ = qfa_base.reference_quantized(F.pad(value, (0, 128)), 0)
        scores = torch.einsum("qhd,khd->hqk", expected_query, expected_key) * impl.scale
        if index == 3:
            scores.masked_fill_(torch.ones(9, 9, dtype=torch.bool).triu(1), -float("inf"))
        result = torch.einsum("hqk,khv->qhv", scores.softmax(-1), expected_value)[..., :128]
        partials.append((result.to(torch.bfloat16).double(), scores.logsumexp(-1).T[..., None].double()))
    expected_lse = torch.stack([part[1] for part in partials]).logsumexp(0)
    expected = sum(result * (part_lse - expected_lse).exp() for result, part_lse in partials)
    torch.testing.assert_close(
        output[2:11, :, :128].float(), expected.to(torch.bfloat16).float(), atol=0.002, rtol=0.008
    )
    torch.testing.assert_close(lse[2:11].double(), expected_lse, atol=1e-6, rtol=1e-6)

    # A different MLA layer reuses lengths/layout schedules, but quantizes
    # its own Q/K/V. Changing the layer weights must change the output.
    other = layer(routing)
    other.fia_expanded.max_context_tokens = 32
    other_output = torch.empty_like(output)
    other_lse = torch.empty_like(lse)
    routing.module.fill_expanded_fia(other, query, current, cache, batch, other_output, other_lse)
    assert len(routing.qfa.native.metadata_calls) == 3
    assert len(routing.qfa.quant_calls) == 18
    assert not torch.equal(other_output[2:11, :, :128], output[2:11, :, :128])


def test_empty_dcp_rank_prepares_weights_without_query_quantization_or_qfa_call(routing):
    impl = layer(routing, dcp=True, rank=1)
    events = []

    def gather(weight, dim):
        events.append((weight.shape, dim))
        return torch.cat((weight, weight), dim=dim)

    impl._dcp_all_gather = gather
    query, current, cache, _, batch = inputs(routing, history_length=0, heads=24)
    output = torch.full((14, 24, 512), 19.0, dtype=torch.bfloat16)
    lse = torch.full((14, 24, 1), 23.0)
    routing.module.fill_expanded_fia(impl, query, current, cache, batch, output, lse)
    assert events == [(torch.Size([12, 512, 256]), 0)]
    assert not routing.qfa.quant_calls and not routing.qfa.native.attention_calls
    assert not batch.qfa_metadata
    assert (output[2:11] == 0).all() and torch.isneginf(lse[2:11]).all()
    assert (output[:2] == 19).all() and (lse[:2] == 23).all()


def test_qfa_workspace_limit_accounts_for_quantization_temporaries(routing):
    normal = layer(routing, enabled=False)
    quantized = layer(routing)
    assert normal.fia_expanded.qfa is None
    assert quantized.fia_expanded.qfa is not None
    assert quantized.fia_expanded.max_context_tokens < normal.fia_expanded.max_context_tokens
