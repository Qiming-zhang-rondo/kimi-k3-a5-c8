# SPDX-License-Identifier: Apache-2.0
"""CPU MXFP8/QFA contracts only; not native performance or model accuracy."""

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def quantize_last_dim(value, *, dst_type, scale_alg):
    """CPU test double with genuine Group32 power-of-two scales and FP8 codes."""
    assert dst_type == torch.float8_e4m3fn and scale_alg == 0
    groups = value.float().reshape(*value.shape[:-1], -1, 32)
    maximum = groups.abs().amax(-1)
    exponent = torch.where(maximum == 0, 0, torch.ceil(torch.log2(maximum.clamp_min(1e-30) / 448)))
    scale = torch.exp2(exponent)
    codes = (groups / scale[..., None]).to(torch.float8_e4m3fn).reshape(value.shape)
    scales = (exponent + 127).to(torch.uint8).reshape(*value.shape[:-1], value.shape[-1] // 64, 2)
    return codes, scales.view(torch.float8_e8m0fnu)


def decode_qk(codes, scales):
    scales = torch.exp2(scales.view(torch.uint8).float() - 127).flatten(-2).repeat_interleave(32, -1)
    return codes.float() * scales


def decode_value(codes, scales):
    token = torch.arange(codes.shape[0])
    exponent = scales.view(torch.uint8)[token // 64, :, :, token // 32 % 2].float() - 127
    return codes.float() * torch.exp2(exponent)


def reference_quantized(value, axis):
    """Independent axis-based reference, including exact encoded scale bytes."""
    floating = value.float()
    restored = torch.empty_like(floating)
    scales = []
    for begin in range(0, value.shape[axis], 32):
        selection = [slice(None)] * value.ndim
        selection[axis] = slice(begin, begin + 32)
        part = floating[tuple(selection)]
        maximum = part.abs().amax(axis, keepdim=True)
        exponent = torch.where(maximum == 0, 0, torch.ceil(torch.log2(maximum.clamp_min(1e-30) / 448)))
        scale = exponent.exp2()
        restored[tuple(selection)] = (part / scale).to(torch.float8_e4m3fn).float() * scale
        scales.append((exponent.squeeze(axis) + 127).to(torch.uint8))
    return restored, torch.stack(scales, -1)


class NativeDouble:
    def __init__(self):
        self.metadata_calls = []
        self.attention_calls = []

    def quant_flash_attn_metadata(
        self,
        num_heads_q,
        num_heads_kv,
        head_dim,
        quant_mode,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        seqused_q=None,
        seqused_kv=None,
        batch_size=None,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        head_dim_v=None,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout_q="BSND",
        layout_q_descale="BSND",
        layout_kv="BSND",
        layout_out="BSND",
        is_grad_enabled=False,
    ):
        assert num_heads_q == num_heads_kv and 0 < num_heads_q <= 256
        assert head_dim == 256 and quant_mode == 1 and head_dim_v is None
        assert batch_size is None and not is_grad_enabled
        assert layout_q == layout_q_descale == layout_kv == layout_out == "TND"
        assert seqused_q is seqused_kv is None and win_left == win_right == -1
        assert cu_seqlens_q.tolist() == [0, max_seqlen_q]
        assert cu_seqlens_kv.tolist() == [0, max_seqlen_kv]
        result = torch.tensor([[len(self.metadata_calls)], [mask_mode]], dtype=torch.int32)
        self.metadata_calls.append(dict(heads=num_heads_q, q=max_seqlen_q, k=max_seqlen_kv, mode=mask_mode))
        return result

    def quant_flash_attn(
        self,
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        quant_mode,
        block_table=None,
        p_scale=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        seqused_q=None,
        seqused_kv=None,
        sinks=None,
        attn_mask=None,
        metadata=None,
        softmax_scale=1.0,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        layout_q="BSND",
        layout_q_descale="BSND",
        layout_kv="BSND",
        layout_out="BSND",
        return_softmax_lse=False,
    ):
        assert quant_mode == 1 and return_softmax_lse
        assert block_table is p_scale is sinks is None
        assert layout_q == layout_q_descale == layout_kv == layout_out == "TND"
        assert q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn
        assert q_descale.dtype == k_descale.dtype == v_descale.dtype == torch.float8_e8m0fnu
        assert all(value.is_contiguous() for value in (q, k, v, q_descale, k_descale, v_descale))
        assert q.shape[-1] == k.shape[-1] == v.shape[-1] == 256
        assert cu_seqlens_q.tolist() == [0, len(q)] and cu_seqlens_kv.tolist() == [0, len(k)]
        plan = self.metadata_calls[metadata[0, 0].item()]
        assert plan == dict(heads=q.shape[1], q=len(q), k=len(k), mode=mask_mode)
        query, key, value = decode_qk(q, q_descale), decode_qk(k, k_descale), decode_value(v, v_descale)
        assert (query[..., 192:] == 0).all() and (key[..., 192:] == 0).all() and (value[..., 128:] == 0).all()
        scores = torch.einsum("qhd,khd->hqk", query, key) * softmax_scale
        if mask_mode == 3:
            assert attn_mask.dtype == torch.int8 and attn_mask.shape == (2048, 2048)
            positions = torch.arange(len(q))[:, None] + len(k) - len(q)
            scores.masked_fill_(torch.arange(len(k))[None, :] > positions, -float("inf"))
        else:
            assert mask_mode == 0 and attn_mask is None
        result = torch.einsum("hqk,khv->qhv", scores.softmax(-1), value).to(torch.bfloat16)
        self.attention_calls.append(
            (q.clone(), k.clone(), v.clone(), q_descale.clone(), k_descale.clone(), v_descale.clone())
        )
        return result, scores.logsumexp(-1)


@pytest.fixture
def qfa(monkeypatch):
    native = NativeDouble()
    quant_calls = []

    def quant(value, *, dst_type, scale_alg):
        quant_calls.append(value.shape)
        return quantize_last_dim(value, dst_type=dst_type, scale_alg=scale_alg)

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_dynamic_mx_quant=quant))
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", native)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/kimi_k3_qfa_expanded.py"
    spec = importlib.util.spec_from_file_location("expanded_qfa_cpu_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, adapter=module.ExpandedQFA(), native=native, quant_calls=quant_calls)


@pytest.mark.parametrize("tokens", [1, 31, 32, 33, 63, 64, 65, 127, 129])
def test_token_axis_v_scales_and_non64_boundaries(qfa, tokens):
    torch.manual_seed(tokens)
    heads = 2
    key = torch.randn(tokens, heads, 192).to(torch.bfloat16)
    # Distinct time/head/channel ranges expose accidental D-axis V scales or
    # a transpose that pairs the wrong 32-token group with its scale.
    magnitudes = torch.exp2((torch.arange(tokens) // 32)[:, None, None] + torch.arange(heads)[None, :, None])
    value = (torch.randn(tokens, heads, 128) * magnitudes).to(torch.bfloat16)
    key8, value8, key_scale, value_scale = qfa.adapter.quantize_kv(key, value)
    padded_key = F.pad(key, (0, 64))
    expected_key, expected_kscale = reference_quantized(padded_key, -1)
    padded_tokens = (tokens + 63) // 64 * 64
    padded_value = F.pad(value, (0, 128, 0, 0, 0, padded_tokens - tokens))
    expected_value, expected_vscale = reference_quantized(padded_value, 0)
    torch.testing.assert_close(decode_qk(key8, key_scale), expected_key, atol=0, rtol=0)
    torch.testing.assert_close(decode_value(value8, value_scale), expected_value[:tokens], atol=0, rtol=0)
    assert torch.equal(key_scale.view(torch.uint8).flatten(-2), expected_kscale)
    assert torch.equal(value_scale.view(torch.uint8), expected_vscale.reshape(heads, 256, -1, 2).permute(2, 0, 1, 3))
    assert key8.shape == value8.shape == (tokens, heads, 256)
    assert value_scale.shape == (padded_tokens // 64, heads, 256, 2)


@pytest.mark.parametrize(
    "queries,keys,heads,causal", [(1, 1, 12, True), (9, 9, 12, True), (33, 129, 12, False), (17, 65, 2, False)]
)
def test_native_contract_matches_independent_axis_quantized_attention(qfa, queries, keys, heads, causal):
    torch.manual_seed(queries + keys)
    query = torch.randn(queries, heads, 192).to(torch.bfloat16)
    key = torch.randn(keys, heads, 192).to(torch.bfloat16)
    value = torch.randn(keys, heads, 128).to(torch.bfloat16)
    prepared = qfa.adapter.prepare_query(query)
    mask = torch.ones(2048, 2048, dtype=torch.bool).triu(1) if causal else None
    scale = 192**-0.5
    output, lse = qfa.adapter.partial(prepared, key, value, scale, mask)
    expected_q, _ = reference_quantized(F.pad(query, (0, 64)), -1)
    expected_k, _ = reference_quantized(F.pad(key, (0, 64)), -1)
    expected_v, _ = reference_quantized(F.pad(value, (0, 128)), 0)
    scores = torch.einsum("qhd,khd->hqk", expected_q, expected_k) * scale
    if causal:
        scores.masked_fill_(
            torch.arange(keys)[None, :] > torch.arange(queries)[:, None] + keys - queries, -float("inf")
        )
    expected = torch.einsum("hqk,khv->qhv", scores.softmax(-1), expected_v)[..., :128].to(torch.bfloat16).float()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(lse, scores.logsumexp(-1).transpose(0, 1).unsqueeze(-1), atol=0, rtol=0)
    assert output.dtype == lse.dtype == torch.float32
    assert prepared.descales.shape == (queries, heads, 4, 2)
    assert len(qfa.quant_calls) == 3


def test_prepared_query_reused_and_shared_metadata_contains_no_qkv(qfa):
    query = torch.randn(9, 12, 192).to(torch.bfloat16)
    key = torch.randn(65, 12, 192).to(torch.bfloat16)
    value = torch.randn(65, 12, 128).to(torch.bfloat16)
    prepared = qfa.adapter.prepare_query(query)
    cache = {}
    first_pointer = prepared.values.data_ptr()
    for _ in range(2):
        qfa.adapter.partial(prepared, key, value, 192**-0.5, metadata_cache=cache)
    second = qfa.adapter.prepare_query(query + 1)
    qfa.adapter.partial(second, key, value, 192**-0.5, metadata_cache=cache)
    assert len(qfa.native.metadata_calls) == 1
    assert prepared.values.data_ptr() == first_pointer
    assert len(qfa.quant_calls) == 8  # Two Q preparations + three independent K/V pairs.
    for schedule, attrs in cache.values():
        assert schedule.dtype == torch.int32
        tensors = [item for item in attrs.values() if isinstance(item, torch.Tensor)]
        assert all(item.dtype == torch.int32 and item.shape == (2,) for item in tensors)
    assert not second.plans and not prepared.plans


@pytest.mark.parametrize("bad", ["missing", "variadic_metadata", "wrong_attention", "master_optional_tail"])
def test_unknown_native_schema_fails_before_quantization(qfa, monkeypatch, bad):
    if bad == "missing":
        monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", SimpleNamespace())
    elif bad == "variadic_metadata":
        monkeypatch.setattr(qfa.native, "quant_flash_attn_metadata", lambda *args, **kwargs: None)
    elif bad == "wrong_attention":
        monkeypatch.setattr(qfa.native, "quant_flash_attn", lambda q, k, v: None)
    else:
        original = qfa.native.quant_flash_attn

        def master_wrapper(*args, **kwargs):
            raise AssertionError("Unknown optional tail ABI must never be invoked")

        master_wrapper.__signature__ = inspect.signature(original).replace(
            parameters=[
                *inspect.signature(original).parameters.values(),
                inspect.Parameter("v_tail", inspect.Parameter.KEYWORD_ONLY, default=None),
                inspect.Parameter("block_table_tail", inspect.Parameter.KEYWORD_ONLY, default=None),
                inspect.Parameter("seqused_v_tail", inspect.Parameter.KEYWORD_ONLY, default=None),
            ]
        )
        monkeypatch.setattr(qfa.native, "quant_flash_attn", master_wrapper)
    with pytest.raises(RuntimeError, match="CANN 9.2"):
        qfa.module.ExpandedQFA()
    assert not qfa.quant_calls


def test_empty_inputs_are_not_sent_to_native_and_bad_output_has_no_fallback(qfa, monkeypatch):
    with pytest.raises(ValueError, match="nonempty"):
        qfa.adapter.prepare_query(torch.empty(0, 12, 192, dtype=torch.bfloat16))
    prepared = qfa.adapter.prepare_query(torch.ones(1, 12, 192, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="nonempty"):
        qfa.adapter.partial(
            prepared, torch.empty(0, 12, 192, dtype=torch.bfloat16), torch.empty(0, 12, 128, dtype=torch.bfloat16), 0.1
        )
    monkeypatch.setattr(qfa.adapter, "attention", lambda *args, **kwargs: (torch.empty(1, 12, 128), torch.empty(12, 1)))
    with pytest.raises(RuntimeError, match="output/LSE"):
        qfa.adapter.partial(
            prepared, torch.ones(1, 12, 192, dtype=torch.bfloat16), torch.ones(1, 12, 128, dtype=torch.bfloat16), 0.1
        )
