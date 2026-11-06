# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU layout/composition checks; the native Prolog is emulated, not validated."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def adapter(monkeypatch):
    class MXFP8:
        group_size = 32
        dynamic_mx_quant_scale_alg = 0

    conversions, calls, quant_calls = [], [], []

    def format_cast(tensor, fmt):
        conversions.append((tensor.shape, fmt))
        return tensor.clone()

    def dynamic_quant(tensor, **kwargs):
        quant_calls.append((tensor.shape, kwargs))
        scales = torch.full((*tensor.shape[:-1], tensor.shape[-1] // 64, 2), 127, dtype=torch.uint8)
        return tensor.to(torch.float8_e4m3fn), scales

    def native(**kwargs):
        calls.append(kwargs)
        tokens = kwargs["token_x"].shape[0]
        heads = kwargs["weight_uk"].shape[0]
        small = torch.arange(tokens * heads * 128, dtype=torch.float32).reshape(tokens, heads, 128).remainder(31)
        absorbed = torch.einsum("thd,hdl->thl", small, kwargs["weight_uk"].float()).bfloat16()
        raw = torch.full((tokens, heads, 64), 13, dtype=torch.bfloat16)
        kwargs["kv_cache"].view(-1, 512)[:tokens].copy_(torch.arange(tokens)[:, None].expand(-1, 512))
        kwargs["kr_cache"].view(-1, 64)[:tokens].fill_(7)
        return absorbed, raw, None, None, None

    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_format_cast=format_cast, npu_dynamic_mx_quant=dynamic_quant)
    )
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", SimpleNamespace(mla_prolog=native))
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8",
        SimpleNamespace(AscendW8A8MXFP8DynamicLinearMethod=MXFP8),
    )
    for module_name, filename in (
        ("vllm_ascend.attention.kimi_k3_qfm_prolog_weights", "kimi_k3_qfm_prolog_weights.py"),
        ("native_fia_prolog_under_test", "kimi_k3_fia_native_prolog.py"),
    ):
        spec = importlib.util.spec_from_file_location(module_name, ROOT / "vllm_ascend/attention" / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
    return module, MXFP8, conversions, calls, quant_calls


def projection(scheme, inputs, outputs, device):
    return SimpleNamespace(
        input_size=inputs,
        quant_method=SimpleNamespace(quant_method=scheme()),
        _mxfp8_transformed=True,
        weight=torch.full((inputs, outputs), 50, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn),
        weight_scale=torch.full((inputs // 64, outputs, 2), 127, dtype=torch.uint8, device=device),
        bias=None,
    )


def make_impl(scheme, *, heads=2, hidden=1024, rank=1536, device="cpu", fused=True):
    impl = SimpleNamespace(
        use_mla_rope=False,
        q_lora_rank=rank,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        num_heads=heads,
        q_a_layernorm=SimpleNamespace(
            weight=torch.ones(rank, dtype=torch.bfloat16, device=device), variance_epsilon=1e-6
        ),
        kv_a_layernorm=SimpleNamespace(
            weight=torch.ones(512, dtype=torch.bfloat16, device=device), variance_epsilon=1e-6
        ),
        q_proj=projection(scheme, rank, heads * 192, device),
        W_UK_T=torch.full((heads, 128, 512), 5, dtype=torch.bfloat16, device=device),
        fused_qkv_a_proj=None,
    )
    if fused:
        impl.fused_qkv_a_proj = projection(scheme, hidden, rank + 576, device)
    else:
        impl.q_a_proj = projection(scheme, hidden, rank, device)
        impl.kv_a_proj_with_mqa = projection(scheme, hidden, 576, device)
    return impl


@pytest.mark.parametrize("fused", [True, False])
def test_native_private_nz_and_scale_layouts_preserve_source(adapter, fused):
    module, scheme, conversions, _, _ = adapter
    impl = make_impl(scheme, fused=fused)
    layers = [impl.fused_qkv_a_proj] if fused else [impl.q_a_proj, impl.kv_a_proj_with_mqa]
    layers.append(impl.q_proj)
    for index, layer in enumerate(layers):
        layer.weight.view(torch.uint8)[0, 0] = index + 12
        layer.weight.view(torch.uint8)[-1, -1] = index + 45
        layer.weight_scale[0, 0] = torch.tensor([100, 101])
    before = [(layer.weight.view(torch.uint8).clone(), layer.weight_scale.clone()) for layer in layers]
    result = module.prepare_native_fia_prolog_weights(impl)
    if fused:
        wa, scales = before[0]
        expected_weights = [wa[:, :1536], wa[:, 1536:], before[-1][0]]
        scale_outputs = scales.transpose(0, 1).flatten(1)
        expected_scales = [scale_outputs[:1536], scale_outputs[1536:], before[-1][1].transpose(0, 1).flatten(1)]
    else:
        expected_weights = [item[0] for item in before]
        expected_scales = [item[1].transpose(0, 1).flatten(1) for item in before]
    for name, expected in zip(("weight_dq", "weight_dkv_kr", "weight_uq_qr"), expected_weights):
        assert torch.equal(result[name].view(torch.uint8), expected)
        result[name].view(torch.uint8).zero_()
    for name, expected in zip(
        ("dequant_scale_w_dq", "dequant_scale_w_dkv_kr", "dequant_scale_w_uq_qr"), expected_scales
    ):
        assert result[name].dtype == torch.float8_e8m0fnu
        assert result[name].ndim == 2 and result[name].is_contiguous()
        assert torch.equal(result[name].view(torch.uint8), expected)
        result[name].view(torch.uint8).zero_()
    assert [fmt for _, fmt in conversions].count(29) == 3
    for layer, (weight, scale) in zip(layers, before):
        assert torch.equal(layer.weight.view(torch.uint8), weight)
        assert torch.equal(layer.weight_scale, scale)
    assert torch.all(impl.W_UK_T == 5)
    assert result["weight_uk"].shape == (2, 128, 512)
    assert torch.equal(result["weight_uk"][:, :, :128], torch.eye(128).bfloat16().expand(2, -1, -1))
    assert torch.count_nonzero(result["weight_uk"][:, :, 128:]) == 0


@pytest.mark.parametrize("heads", [1, 6, 12, 24, 48, 96, 128])
def test_native_preserves_non_power_of_two_heads(adapter, heads):
    module, scheme, _, _, _ = adapter
    result = module.prepare_native_fia_prolog_weights(make_impl(scheme, heads=heads, hidden=7168, device="meta"))
    assert result["weight_uk"].shape == (heads, 128, 512)
    assert result["weight_uq_qr"].shape == (1536, heads * 192)
    assert result["dequant_scale_w_dq"].shape == (1536, 224)
    assert result["dequant_scale_w_uq_qr"].shape == (heads * 192, 48)


def test_forward_uses_nope_bf16_current_only_and_reuses_prepared_state(adapter):
    module, scheme, conversions, calls, quant_calls = adapter
    impl = make_impl(scheme)
    impl.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=256))
    prolog = module.NativeFIAProlog(impl)
    pointers = tuple(value.data_ptr() for value in (prolog._current_latent, prolog._current_raw, prolog._indices))
    prepared = len(conversions)
    for tokens in (9, 129, 1):
        # A non-contiguous caller input is normalized on-device; no scalar read.
        hidden = torch.zeros((tokens, 2048), dtype=torch.bfloat16)[:, ::2]
        q, kv = prolog(hidden)
        assert q.shape == (tokens, 2, 192) and q.dtype == torch.bfloat16
        expected = torch.arange(tokens * 2 * 128).reshape(tokens, 2, 128).remainder(31).bfloat16()
        assert torch.equal(q[:, :, :128], expected)
        assert torch.all(q[:, :, 128:] == 13)
        assert kv.shape == (tokens, 576) and kv.dtype == torch.bfloat16
        assert torch.equal(kv[:, :512], torch.arange(tokens).bfloat16()[:, None].expand(-1, 512))
        assert torch.all(kv[:, 512:] == 7)
        call = calls[-1]
        assert call["rope_sin"] is call["rope_cos"] is None
        assert (call["weight_quant_mode"], call["query_quant_mode"], call["kv_cache_quant_mode"]) == (3, 0, 0)
        assert call["cache_mode"] == "PA_BSND"
        assert call["rmsnorm_epsilon_cq"] == call["rmsnorm_epsilon_ckv"] == 1e-6
        assert torch.equal(call["cache_index"], torch.arange(tokens))
        assert call["dequant_scale_x"].dtype == torch.float8_e8m0fnu
        assert call["dequant_scale_x"].shape == (tokens, 32)
    assert len(conversions) == prepared
    assert (
        tuple(value.data_ptr() for value in (prolog._current_latent, prolog._current_raw, prolog._indices)) == pointers
    )
    assert len(quant_calls) == len(calls) == 3
    assert torch.all(impl.W_UK_T == 5)


def test_reload_copies_compatible_private_tensors_in_place(adapter):
    module, scheme, _, _, _ = adapter
    impl = make_impl(scheme)
    prolog = module.NativeFIAProlog(impl)
    pointers = {name: value.data_ptr() for name, value in prolog.weights.items() if isinstance(value, torch.Tensor)}
    impl.fused_qkv_a_proj.weight.view(torch.uint8)[0, 0] = 23
    prolog.update_weights(impl)
    assert prolog.weights["weight_dq"].view(torch.uint8)[0, 0] == 23
    assert {name: prolog.weights[name].data_ptr() for name in pointers} == pointers
    impl.q_a_layernorm.variance_epsilon = impl.kv_a_layernorm.variance_epsilon = 1e-5
    with pytest.raises(ValueError, match="recapture"):
        prolog.update_weights(impl)


def test_long_prefill_has_one_native_call_without_decode_token_cap(adapter, monkeypatch):
    module, scheme, _, _, _ = adapter
    prolog = module.NativeFIAProlog(make_impl(scheme, heads=12, device="meta"))
    calls = []

    def quant(tensor, **kwargs):
        return tensor.to(torch.float8_e4m3fn), tensor.new_empty((8192, 16, 2), dtype=torch.uint8)

    def native(**kwargs):
        calls.append(kwargs)
        q = kwargs["token_x"].new_empty((8192, 12, 512), dtype=torch.bfloat16)
        return q, q.new_empty((8192, 12, 64)), None, None, None

    monkeypatch.setattr(module.torch_npu, "npu_dynamic_mx_quant", quant)
    prolog.op = native
    q, kv = prolog(torch.empty((8192, 1024), dtype=torch.bfloat16, device="meta"))
    assert q.shape == (8192, 12, 192) and kv.shape == (8192, 576)
    assert len(calls) == 1
    assert calls[0]["cache_index"].shape == (8192,)
    assert calls[0]["kv_cache"].shape == (64, 128, 1, 512)


@pytest.mark.parametrize("field,value,match", [("q_lora_rank", 640, "1536/2048"), ("qk_nope_head_dim", 192, "128")])
def test_rejects_dsl_only_geometry(adapter, field, value, match):
    module, scheme, _, _, _ = adapter
    impl = make_impl(scheme, device="meta")
    setattr(impl, field, value)
    with pytest.raises(ValueError, match=match):
        module.prepare_native_fia_prolog_weights(impl)


def test_rejects_unsupported_hidden_and_float_norm(adapter):
    module, scheme, _, _, _ = adapter
    with pytest.raises(ValueError, match="hidden size"):
        module.prepare_native_fia_prolog_weights(make_impl(scheme, hidden=1280, device="meta"))
    impl = make_impl(scheme, device="meta")
    impl.kv_a_layernorm.weight = impl.kv_a_layernorm.weight.float()
    with pytest.raises(ValueError, match="BF16 RMSNorm"):
        module.prepare_native_fia_prolog_weights(impl)


def test_empty_and_invalid_inputs_do_not_launch_native(adapter):
    module, scheme, _, calls, _ = adapter
    prolog = module.NativeFIAProlog(make_impl(scheme))
    q, kv = prolog(torch.empty((0, 1024), dtype=torch.bfloat16))
    assert q.shape == (0, 2, 192) and kv.shape == (0, 576)
    for hidden in (torch.empty((1, 1024)), torch.empty((1, 1025), dtype=torch.bfloat16)):
        with pytest.raises(ValueError, match="BF16"):
            prolog(hidden)
    assert calls == []
