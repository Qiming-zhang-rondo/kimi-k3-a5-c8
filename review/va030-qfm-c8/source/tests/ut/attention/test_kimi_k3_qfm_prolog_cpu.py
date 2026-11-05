# SPDX-License-Identifier: Apache-2.0
"""Adapter contracts only; mocks are NOT native Prolog/HCCL acceptance."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.attention.test_kimi_k3_qfm_cpu import builder, common
from tests.ut.attention.test_kimi_k3_qfm_cpu import qfm as _qfm_fixture

qfm = _qfm_fixture


def test_floating_prolog_only_stages_current_and_preserves_history(prolog, qfm, monkeypatch):
    hidden = torch.randn(9, 1024, dtype=torch.bfloat16)
    slots = torch.tensor([128, -1, 0, 383, 127, 7, 129, -1, -1])
    stride, offset = 128 * 576 + 256, 256
    backing = torch.full((offset + 3 * stride + 256,), 0x5A, dtype=torch.uint8)
    cache = qfm.cache_view(backing[offset : offset + 3 * stride], 3, stride)
    expected = backing.clone()
    qfm.write_cache(
        hidden[:, :576], slots, qfm.cache_view(expected[offset : offset + 3 * stride], 3, stride), torch.ones(1)
    )

    def native(**kw):
        assert kw["qkv_quant_mode"] == 0 and kw["quant_scale_kv"] is None
        assert kw["cache_mode"] == 0 and kw["kv_cache"].numel() == 128 * 576
        assert kw["cache_index"].tolist() == list(range(9))
        kw["kv_cache"].view(-1, 576)[:9].copy_(hidden[:, :576])
        return torch.zeros(9, 12, 576, dtype=torch.bfloat16), None, None, None

    monkeypatch.setattr(prolog, "load_prolog", lambda: native)
    q, kv = prolog.QFMProlog({}, torch.ones(1)).floating(hidden, slots, cache)
    assert q.dtype == torch.bfloat16
    torch.testing.assert_close(kv, hidden[:, :576], rtol=0, atol=0)
    assert torch.equal(backing, expected)


@pytest.fixture
def prolog(qfm, monkeypatch):
    def quant(hidden, **kwargs):
        assert kwargs == dict(dst_type=torch.float8_e4m3fn, scale_alg=0)
        x = hidden.float().clamp(-448, 448).to(torch.float8_e4m3fn)
        scale = torch.full((hidden.shape[0], hidden.shape[1] // 32), 127, dtype=torch.uint8)
        return x, scale

    sys.modules["torch_npu"].npu_dynamic_mx_quant = quant
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.kimi_k3_qfm", qfm)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/kimi_k3_qfm_prolog.py"
    spec = importlib.util.spec_from_file_location("qfm_prolog_cpu_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("gap,offset", [(0, 0), (0, 256), (256, 256), (1024, 512)])
@pytest.mark.parametrize("heads", [12, 24, 48])
def test_fused_writer_direct_or_staged_preserves_pages(prolog, qfm, monkeypatch, gap, offset, heads):
    stride = 128 * 576 + gap
    backing = torch.full((offset + 3 * stride + 256,), 0x5A, dtype=torch.uint8)
    cache = qfm.cache_view(backing[offset : offset + 3 * stride], 3, stride)
    hidden = torch.randn(9, 1024, dtype=torch.bfloat16)
    slots = torch.tensor([128, -1, 0, 383, 127, 7, 129, -1, -1])
    rows = hidden[:, :576].float().to(torch.float8_e4m3fn)
    seen = []

    def native(**kwargs):
        destination, indices = kwargs["kv_cache"], kwargs["cache_index"]
        mode = kwargs["cache_mode"]
        assert kwargs["rope_mode"] == 0 and kwargs["qkv_quant_mode"] == 2
        assert kwargs["w_qr_quant_mode"] == 1 and not kwargs["return_qr"]
        assert kwargs["descale_x"].dtype == torch.float8_e8m0fnu
        if gap == 0:
            assert mode == 1 and destination.data_ptr() == cache.data_ptr()
            assert destination.storage_offset() == offset
            qfm.scatter_fp8_cache(rows, indices, destination.view(3, 1, 18, 128, 32))
        else:
            assert mode == 0 and destination.data_ptr() != cache.data_ptr()
            assert destination.numel() == 128 * 576  # current batch, NOT whole history
            assert indices.tolist() == list(range(9))
            destination.view(torch.uint8).view(-1, 576)[:9].copy_(rows.view(torch.uint8))
        seen.append(mode)
        return torch.zeros(9, heads, 576).to(torch.float8_e4m3fn), torch.ones(9, heads, 1), None, None

    monkeypatch.setattr(prolog, "load_prolog", lambda: native)
    fused = prolog.QFMProlog({}, torch.ones(1))
    expected_backing = backing.clone()
    expected_cache = qfm.cache_view(expected_backing[offset : offset + 3 * stride], 3, stride)
    qfm.scatter_fp8_cache(rows, slots, expected_cache)
    q, scales = fused(hidden, slots, cache)
    assert q.shape == (9, heads, 576) and scales.shape == (9, heads)
    assert torch.equal(backing, expected_backing)
    before = backing.clone()
    fused(hidden, torch.full_like(slots, -1), cache)
    assert torch.equal(before, backing)
    assert seen == ([0, 0] if gap else [1, 1])


def test_weight_refresh_retains_addresses_and_validates_first(prolog, monkeypatch):
    monkeypatch.setattr(prolog, "load_prolog", lambda: None)
    fused = prolog.QFMProlog({"wqa": torch.zeros(2, 3), "norm_eps": 1e-5}, torch.ones(1))
    address = fused.weights["wqa"].data_ptr()
    fused.update_weights({"wqa": torch.ones(2, 3), "norm_eps": 1e-5})
    assert fused.weights["wqa"].data_ptr() == address
    assert fused.weights["wqa"].sum() == 6
    with pytest.raises(ValueError, match="norm_eps changed"):
        fused.update_weights({"wqa": torch.zeros(2, 3), "norm_eps": 1e-4})
    assert fused.weights["wqa"].sum() == 6
    with pytest.raises(ValueError, match="internal unit"):
        prolog.QFMProlog({}, torch.tensor([0.03125]))


def test_fused_dcp_gathers_query_bytes_and_scales_together(qfm, monkeypatch):
    b, _ = builder(qfm, monkeypatch, dcp=2, rank=1)
    batch = b.build(common([4], [129], capacity=8), retain_for_graph=True)
    q8 = torch.randn(8, 12, 576).to(torch.float8_e4m3fn)
    qs = torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12) + 1
    gather_calls = []

    def gather(x, dim):
        assert dim == 1
        gather_calls.append(x.dtype)
        return torch.cat((x, x), dim=dim)

    def attention(q, cache, scales, *args, **kwargs):
        assert q.shape == (8, 24, 576) and scales.shape == (8, 24)
        assert torch.equal(q[:, :12].view(torch.uint8), q8.view(torch.uint8))
        assert torch.equal(scales[:, 12:], qs)
        return torch.ones(8, 24, 512), torch.zeros(24, 8)

    def merge(x, lse):
        assert torch.isneginf(lse[4:]).all()
        return x[:, :12]

    impl = SimpleNamespace(
        qfm_prolog=lambda h, s, c: (q8, qs),
        dcp_enabled=True,
        _dcp_all_gather=gather,
        _merge_dcp_attention_output=merge,
        qfm_ops=(attention, None),
        qfm_scale=torch.ones(1),
        scale=0.1,
        _v_up_proj_batch_major=lambda x: x.flatten(1),
        use_output_gate=False,
        o_proj=lambda x, **kw: (x,),
    )
    output = torch.empty(8, 12 * 512)
    qfm.forward_qfm(impl, torch.randn(8, 1024), None, SimpleNamespace(qfm=batch), output)
    assert gather_calls == [torch.int8, torch.float32]
    assert output[4:].count_nonzero() == 0
