# SPDX-License-Identifier: Apache-2.0
"""Post-load weight/layout contracts only; no native Prolog accuracy claim."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def adapter(monkeypatch):
    class MXFP8:
        group_size = 32
        dynamic_mx_quant_scale_alg = 0

    conversions = []

    def format_cast(tensor, fmt):
        conversions.append((tensor.shape, fmt))
        return tensor

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_format_cast=format_cast))
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8",
        SimpleNamespace(AscendW8A8MXFP8DynamicLinearMethod=MXFP8),
    )
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/kimi_k3_qfm_prolog_weights.py"
    spec = importlib.util.spec_from_file_location("qfm_prolog_weights_cpu_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, MXFP8, conversions


def projection(scheme, inputs, outputs, device):
    return SimpleNamespace(
        input_size=inputs,
        quant_method=SimpleNamespace(quant_method=scheme()),
        _mxfp8_transformed=True,
        weight=torch.full((inputs, outputs), 0x32, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn),
        weight_scale=torch.full((inputs // 64, outputs, 2), 120, dtype=torch.uint8, device=device),
        bias=None,
    )


def make_impl(scheme, *, hidden=1024, q_rank=640, heads=6, nope=128, fused=True, device="meta"):
    impl = SimpleNamespace(
        use_mla_rope=False,
        q_lora_rank=q_rank,
        kv_lora_rank=512,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=64,
        num_heads=heads,
        q_a_layernorm=SimpleNamespace(
            weight=torch.ones(q_rank, dtype=torch.bfloat16, device=device), variance_epsilon=1e-6
        ),
        kv_a_layernorm=SimpleNamespace(
            weight=torch.ones(512, dtype=torch.bfloat16, device=device), variance_epsilon=1e-6
        ),
        q_proj=projection(scheme, q_rank, heads * (nope + 64), device),
        W_UK_T=torch.ones((heads, nope, 512), dtype=torch.bfloat16, device=device),
        fused_qkv_a_proj=None,
    )
    if fused:
        impl.fused_qkv_a_proj = projection(scheme, hidden, q_rank + 576, device)
    else:
        impl.q_a_proj = projection(scheme, hidden, q_rank, device)
        impl.kv_a_proj_with_mqa = projection(scheme, hidden, 576, device)
    return impl


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
def test_reconstructs_exact_weight_and_scale_bytes_without_mutation(adapter, fused, norm_dtype):
    module, scheme, conversions = adapter
    impl = make_impl(scheme, fused=fused, device="cpu")
    impl.q_a_layernorm.weight = impl.q_a_layernorm.weight.to(norm_dtype)
    impl.kv_a_layernorm.weight = impl.kv_a_layernorm.weight.to(norm_dtype)
    layers = [impl.fused_qkv_a_proj] if fused else [impl.q_a_proj, impl.kv_a_proj_with_mqa]
    layers.append(impl.q_proj)
    for index, layer in enumerate(layers):
        layer.weight.view(torch.uint8)[0, 0] = 11 + index
        layer.weight.view(torch.uint8)[-1, -1] = 41 + index
        layer.weight_scale[0, 0, :] = torch.tensor([100 + index, 110 + index], dtype=torch.uint8)
        layer.weight_scale[-1, -1, :] = torch.tensor([130 + index, 140 + index], dtype=torch.uint8)
    originals = [(layer.weight.view(torch.uint8).clone(), layer.weight_scale.clone()) for layer in layers]
    result = module.prepare_prolog_weights(impl)
    if fused:
        expected_wa = originals[0][0].T
        expected_sa = originals[0][1].transpose(0, 1)
        targets = [("wqa", expected_wa[:640]), ("wkva", expected_wa[640:])]
        scales = [("descale_wqa", expected_sa[:640]), ("descale_wkva", expected_sa[640:])]
    else:
        targets = [("wqa", originals[0][0].T), ("wkva", originals[1][0].T)]
        scales = [("descale_wqa", originals[0][1].transpose(0, 1)), ("descale_wkva", originals[1][1].transpose(0, 1))]
    targets.append(("wqb", originals[-1][0].T))
    scales.append(("descale_wqb", originals[-1][1].transpose(0, 1)))
    for key, expected in targets + scales:
        assert result[key].is_contiguous()
        assert torch.equal(result[key].view(torch.uint8), expected)
        if key.startswith("descale"):
            assert result[key].dtype == torch.float8_e8m0fnu
        result[key].view(torch.uint8).zero_()
    for key in ("norm_weight_qa", "norm_weight_kva"):
        assert result[key].dtype == torch.float32
        assert result[key].device == impl.q_a_layernorm.weight.device
        assert torch.all(result[key] == 1)
        result[key].zero_()
    result["wkb"].zero_()
    for layer, (weight, scale) in zip(layers, originals):
        assert torch.equal(layer.weight.view(torch.uint8), weight)
        assert torch.equal(layer.weight_scale, scale)
    assert torch.all(impl.W_UK_T == 1)
    assert torch.all(impl.q_a_layernorm.weight == 1)
    assert torch.all(impl.kv_a_layernorm.weight == 1)
    assert all(fmt == 2 for _, fmt in conversions)
    assert result["norm_eps"] == 1e-6


@pytest.mark.parametrize("heads", [1, 6, 12, 24, 48, 96, 128])
def test_real_k3_geometry_retains_local_heads(adapter, heads):
    module, scheme, _ = adapter
    impl = make_impl(scheme, hidden=7168, q_rank=1536, heads=heads)
    result = module.prepare_prolog_weights(impl)
    assert result["wqa"].shape == (1536, 7168)
    assert result["wkva"].shape == (576, 7168)
    assert result["wqb"].shape == (heads * 192, 1536)
    assert result["wkb"].shape == (heads, 128, 512)
    assert result["descale_wqa"].shape == (1536, 112, 2)
    assert result["descale_wqb"].shape == (heads * 192, 24, 2)


@pytest.mark.parametrize("q_rank,nope", [(640, 128), (1024, 192), (1536, 128), (2048, 192)])
def test_new_dsl_supported_small_and_large_shapes(adapter, q_rank, nope):
    module, scheme, _ = adapter
    result = module.prepare_prolog_weights(make_impl(scheme, q_rank=q_rank, nope=nope))
    assert result["wqb"].shape == (6 * (nope + 64), q_rank)


@pytest.mark.parametrize(
    "case,match",
    [
        ("rope", "NoRoPE"),
        ("qrank", "q rank"),
        ("raw", "latent512"),
        ("heads", "local heads"),
        ("epsilon", "RMSNorm epsilon"),
        ("epsilon_nan", "RMSNorm epsilon"),
        ("norm_dtype", "RMSNorm weight"),
        ("hidden", "hidden size"),
        ("scheme", "W8A8_MXFP8"),
        ("group", "Group32"),
        ("scale_alg", "scale_alg=0"),
        ("unloaded", "after MXFP8"),
        ("bias", "bias-free"),
        ("grouped_gmm", "grouped-GMM"),
        ("chunked", "chunked"),
        ("padding", "TP-padded"),
        ("weight_shape", "processed E4M3"),
        ("scale_shape", "paired E8M0"),
        ("scale_dtype", "paired E8M0"),
        ("wkb_heads", "without padded/global heads"),
    ],
)
def test_rejects_unknown_or_incompatible_projection_contracts(adapter, case, match):
    module, scheme, _ = adapter
    impl = make_impl(scheme)
    layer = impl.fused_qkv_a_proj
    if case == "rope":
        impl.use_mla_rope = True
    elif case == "qrank":
        impl.q_lora_rank = 128
    elif case == "raw":
        impl.qk_rope_head_dim = 128
    elif case == "heads":
        impl.num_heads = 129
    elif case == "epsilon":
        impl.kv_a_layernorm.variance_epsilon = 1e-5
    elif case == "epsilon_nan":
        impl.q_a_layernorm.variance_epsilon = float("nan")
    elif case == "norm_dtype":
        impl.q_a_layernorm.weight = impl.q_a_layernorm.weight.half()
    elif case == "hidden":
        layer.input_size = 1280 + 1
    elif case == "scheme":
        layer.quant_method = SimpleNamespace(group_size=32)
    elif case == "group":
        layer.quant_method.quant_method.group_size = 128
    elif case == "scale_alg":
        layer.quant_method.quant_method.dynamic_mx_quant_scale_alg = 1
    elif case == "unloaded":
        layer._mxfp8_transformed = False
    elif case == "bias":
        layer.bias = torch.zeros(1216, device="meta")
    elif case == "grouped_gmm":
        layer._mxfp8_gmm_nz_layout = True
    elif case == "chunked":
        impl.q_proj._chunk_size = 128
    elif case == "padding":
        layer.mxfp8_tp_padding = (1, 31)
    elif case == "weight_shape":
        layer.weight = layer.weight.T
    elif case == "scale_shape":
        layer.weight_scale = layer.weight_scale.flatten(1)
    elif case == "scale_dtype":
        layer.weight_scale = layer.weight_scale.float()
    elif case == "wkb_heads":
        impl.W_UK_T = impl.W_UK_T[:1]
    with pytest.raises(ValueError, match=match):
        module.prepare_prolog_weights(impl)
