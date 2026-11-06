# SPDX-License-Identifier: Apache-2.0
"""CPU routing/byte contracts; no native, HCCL or model performance claim."""

import sys
from types import SimpleNamespace

import pytest
import torch

from tests.ut.attention import test_kimi_k3_qfm_blocks_cpu as blocks
from tests.ut.attention import test_kimi_k3_qfm_cpu as base

qfm = base.qfm


@pytest.mark.parametrize(
    "backend,form,expected",
    [("fia", None, False), ("fia", "latent", False), ("fia", "expanded", True), ("qfm", "expanded", False)],
)
def test_expanded_form_is_explicit_fia_only_opt_in(qfm, backend, form, expected):
    impl, config = blocks.selection_inputs(128)
    impl.qk_nope_head_dim = impl.v_head_dim = 128
    config.kimi_k3_c8_prefill_backend = backend
    if form is not None:
        config.kimi_k3_c8_fia_form = form
    assert qfm.select_qfm(impl, config)
    assert impl.qfm_fia_expanded is expected
    assert impl.qfm_fia_prefill is (backend == "fia")


def test_expanded_form_rejects_unknown_value_and_wrong_model_dimensions(qfm):
    impl, config = blocks.selection_inputs(128)
    impl.qk_nope_head_dim = impl.v_head_dim = 128
    config.kimi_k3_c8_prefill_backend = "fia"
    config.kimi_k3_c8_fia_form = "automatic"
    with pytest.raises(ValueError, match="latent or expanded"):
        qfm.select_qfm(impl, config)
    config.kimi_k3_c8_fia_form = "expanded"
    impl.v_head_dim = 64
    with pytest.raises(ValueError, match="V128"):
        qfm.select_qfm(impl, config)


@pytest.mark.parametrize("fused", [False, True])
def test_ordinary_fia_uses_unabsorbed_q_projection_and_normalized_latent(qfm, fused):
    torch.manual_seed(61)
    tokens, capacity, heads = 5, 8, 2
    hidden = torch.randn(capacity, 4).to(torch.bfloat16)
    kv = torch.randn(capacity, 576).to(torch.bfloat16)
    low_q = torch.randn(capacity, 3).to(torch.bfloat16)
    projected_q = torch.randn(tokens, heads * 192).to(torch.bfloat16)
    seen = []

    def q_proj(value):
        seen.append(value.clone())
        return (projected_q,)

    def forbidden(*args):
        raise AssertionError("Expanded Prefill must not absorb Q into latent512")

    impl = SimpleNamespace(
        fused_qkv_a_proj=(lambda x: (torch.cat((low_q, kv), dim=-1),)) if fused else None,
        q_lora_rank=3,
        q_a_layernorm=lambda value: value * 1.5,
        kv_a_proj_with_mqa=lambda value: (kv,),
        q_proj=q_proj,
        _q_proj_and_k_up_proj=forbidden,
        kv_a_layernorm=lambda value: value + 0.25,
        num_heads=heads,
        qk_head_dim=192,
    )
    query, current = qfm.ordinary_fia_inputs(impl, hidden, tokens)
    torch.testing.assert_close(seen[0], (low_q * 1.5)[:tokens] if fused else hidden[:tokens], rtol=0, atol=0)
    torch.testing.assert_close(query, projected_q.view(tokens, heads, 192), rtol=0, atol=0)
    torch.testing.assert_close(current[:, :512], kv[:tokens, :512] + 0.25, rtol=0, atol=0)
    torch.testing.assert_close(current[:, 512:], kv[:tokens, 512:], rtol=0, atol=0)
    assert current.dtype == query.dtype == torch.bfloat16


def test_ordinary_fia_explicit_q_a_projects_hidden7168_before_q_b1536(qfm):
    torch.manual_seed(7168)
    tokens, capacity, heads = 3, 5, 2
    hidden = torch.randn(capacity, 7168).to(torch.bfloat16)
    q_weight = (torch.randn(1536, heads * 192) / 64).to(torch.bfloat16)
    events, q_b_inputs = [], []

    def kv_a(value):
        events.append("kv_a")
        assert value.shape[-1] == 7168
        return (value[:, :576] + 0.125,)

    def q_a(value):
        events.append("q_a")
        assert value.shape[-1] == 7168
        return (value[:, :1536] * 0.5,)

    def q_norm(value):
        events.append("q_norm")
        assert value.shape[-1] == 1536
        floating = value.float()
        return (floating * torch.rsqrt(floating.square().mean(-1, keepdim=True) + 1e-5)).to(torch.bfloat16)

    def q_b(value):
        events.append("q_b")
        assert value.shape == (tokens, 1536), "The hidden7168 tensor cannot be fed directly into q_b_proj"
        q_b_inputs.append(value.clone())
        return (value @ q_weight,)

    impl = SimpleNamespace(
        fused_qkv_a_proj=None,
        kv_a_proj_with_mqa=kv_a,
        q_a_proj=q_a,
        q_a_layernorm=q_norm,
        q_proj=q_b,
        kv_a_layernorm=lambda value: value * 2,
        num_heads=heads,
        qk_head_dim=192,
    )
    query, current = qfm.ordinary_fia_inputs(impl, hidden, tokens)
    expected_low = (hidden[:tokens, :1536] * 0.5).float()
    expected_low = (expected_low * (expected_low.square().mean(-1, keepdim=True) + 1e-5).rsqrt()).to(torch.bfloat16)
    torch.testing.assert_close(q_b_inputs[0], expected_low, atol=0, rtol=0)
    torch.testing.assert_close(query, (expected_low @ q_weight).view(tokens, heads, 192), atol=0, rtol=0)
    expected_kv = hidden[:tokens, :576] + 0.125
    torch.testing.assert_close(current[:, :512], expected_kv[:, :512] * 2, atol=0, rtol=0)
    torch.testing.assert_close(current[:, 512:], expected_kv[:, 512:], atol=0, rtol=0)
    assert events == ["kv_a", "q_a", "q_norm", "q_b"]


def mixed_batch(qfm, monkeypatch):
    builder, _ = base.builder(qfm, monkeypatch)
    builder.fia_prefill = True
    builder.expanded_fia = True
    common = base.common([2, 9, 3], [129, 257, 513], capacity=16)
    common.is_prefilling = torch.tensor([False, True, False])
    common.slot_mapping = torch.tensor([128, 127, 2, 257, 254, 7, 129, 256, 3, 130, 6, 131, 255, 5])
    return builder.build(common)


def guarded_cache(qfm):
    offset, stride, pages = 256, 128 * 576 + 1024, 3
    backing = torch.full((offset + pages * stride + 4096,), 0x5A, dtype=torch.uint8)
    # The suffix also represents separately owned draft-state storage.
    backing[offset + pages * stride :].fill_(0x31)
    cache = qfm.cache_view(backing[offset : offset + pages * stride], pages, stride)
    return cache, backing, offset, stride


def synthetic_rows(chunk, width):
    return (chunk[:, :1] / 16 + torch.arange(width)[None, :] % 23 / 32).to(torch.bfloat16)


def install_stage_inputs(qfm, monkeypatch, *, native_prefill=False, fused_decode=True):
    observed = SimpleNamespace(decode=[], prefill=[], native=[])
    impl = SimpleNamespace(num_heads=2, qk_head_dim=192, qfm_scale=torch.ones(1), qfm_mlapo_prefill=True)

    def prefill(live_impl, chunk, tokens):
        assert live_impl is impl and tokens == len(chunk)
        observed.prefill.append(chunk[:, 0].clone())
        return synthetic_rows(chunk, 384).view(-1, 2, 192), synthetic_rows(chunk, 576)

    def native(chunk):
        observed.native.append(chunk[:, 0].clone())
        return synthetic_rows(chunk, 384).view(-1, 2, 192), synthetic_rows(chunk, 576)

    def ordinary_decode(live_impl, chunk, tokens):
        assert live_impl is impl and tokens == len(chunk)
        return synthetic_rows(chunk, 1152).view(-1, 2, 576), synthetic_rows(chunk, 576)

    def prolog(chunk, slots, cache, **kwargs):
        query = synthetic_rows(chunk, 1152).view(-1, 2, 576).to(torch.float8_e4m3fn)
        scales = (chunk[:, :1] / 128 + 0.125).expand(-1, 2).contiguous()
        qfm.write_cache(synthetic_rows(chunk, 576), slots, cache, impl.qfm_scale, kwargs.get("scatter_plan"))
        observed.decode.append((chunk[:, 0].clone(), query.view(torch.uint8).clone(), scales.clone()))
        return query, scales

    monkeypatch.setattr(qfm, "ordinary_fia_inputs", prefill)
    monkeypatch.setattr(qfm, "ordinary_qfm_inputs", ordinary_decode)
    if native_prefill:
        impl.fia_native_prolog = native
    return impl, prolog if fused_decode else None, observed


@pytest.mark.parametrize("native_prefill,fused_decode", [(False, False), (False, True), (True, False), (True, True)])
def test_stage_mixed_preserves_decode_bytes_scales_and_single_fp8_cache(qfm, monkeypatch, native_prefill, fused_decode):
    batch = mixed_batch(qfm, monkeypatch)
    cache, backing, offset, stride = guarded_cache(qfm)
    expected = backing.clone()
    hidden = torch.arange(16, dtype=torch.bfloat16)[:, None].expand(-1, 4).contiguous()
    impl, prolog, observed = install_stage_inputs(
        qfm, monkeypatch, native_prefill=native_prefill, fused_decode=fused_decode
    )
    query, kv, scales, fia_q = qfm.stage_expanded_fia_inputs(impl, hidden, cache, batch, prolog)
    assert query.dtype == torch.float8_e4m3fn and query.shape == (16, 2, 576)
    assert fia_q.dtype == kv.dtype == torch.bfloat16
    assert fia_q.shape == (16, 2, 192) and kv.shape == (16, 576)
    assert torch.equal(fia_q[2:11], synthetic_rows(hidden[2:11], 384).view(9, 2, 192))
    assert torch.equal(kv[2:11], synthetic_rows(hidden[2:11], 576))
    assert (query.view(torch.uint8)[2:11] == 0).all() and (scales[2:11] == 1).all()
    decode_rows = torch.tensor([0, 1, 11, 12, 13, 14, 15])
    assert (kv[decode_rows] == 0).all() and (fia_q[decode_rows] == 0).all()
    assert len(observed.native if native_prefill else observed.prefill) == 1
    assert not (observed.prefill if native_prefill else observed.native)
    for begin, end in [(0, 2), (11, 16)]:
        if fused_decode:
            record = observed.decode[0 if begin == 0 else 1]
            assert torch.equal(query[begin:end].view(torch.uint8), record[1])
            assert torch.equal(scales[begin:end], record[2])
        else:
            q8, expected_scale = qfm.quantize_query(synthetic_rows(hidden[begin:end], 1152).view(-1, 2, 576))
            assert torch.equal(query[begin:end].view(torch.uint8), q8.view(torch.uint8))
            assert torch.equal(scales[begin:end], expected_scale)
    encoded = synthetic_rows(hidden, 576).to(torch.float8_e4m3fn).view(torch.uint8)
    for token, slot in enumerate(batch.slots.tolist()):
        if slot < 0:
            continue
        page, position = divmod(slot, 128)
        for tile in range(18):
            start = offset + page * stride + tile * 128 * 32 + position * 32
            expected[start : start + 32] = encoded[token, tile * 32 : (tile + 1) * 32]
    assert torch.equal(backing, expected)


@pytest.mark.parametrize("native_available", [False, True])
def test_stage_pure_prefill_retains_bf16_query_and_original_current(qfm, monkeypatch, native_available):
    builder, _ = base.builder(qfm, monkeypatch)
    builder.fia_prefill = builder.expanded_fia = True
    common = base.common([9], [129])
    common.is_prefilling = torch.tensor([True])
    batch = builder.build(common)
    impl, prolog, observed = install_stage_inputs(qfm, monkeypatch, native_prefill=native_available)
    impl.qfm_mlapo_prefill = False
    hidden = torch.arange(9, dtype=torch.bfloat16)[:, None].expand(-1, 4).contiguous()
    cache, _, _, _ = guarded_cache(qfm)
    query, current, scales, fia_q = qfm.stage_expanded_fia_inputs(impl, hidden, cache, batch, prolog)
    assert scales is None and fia_q is None and not observed.decode
    assert not observed.native and len(observed.prefill) == 1
    assert query.shape == (9, 2, 192) and query.dtype == torch.bfloat16
    assert torch.equal(current, synthetic_rows(hidden, 576))
    assert not torch.equal(current, current.to(torch.float8_e4m3fn).to(torch.bfloat16))


def test_forward_dcp_weight_preparation_precedes_query_gathers_and_only_decode_runs_v_up(qfm, monkeypatch):
    batch = mixed_batch(qfm, monkeypatch)
    impl, prolog, _ = install_stage_inputs(qfm, monkeypatch)
    cache, _, _, _ = guarded_cache(qfm)
    hidden = torch.arange(16, dtype=torch.bfloat16)[:, None].expand(-1, 4).contiguous()
    events, v_up_rows = [], []
    impl.qfm_fia_expanded = True
    impl.qfm_prolog = prolog
    impl.v_head_dim = 128
    impl.dcp_enabled = True
    impl.fia_expanded = SimpleNamespace(prepare_for_dcp=lambda live: events.append("weights"))
    impl.use_output_gate = False
    impl.o_proj = lambda x, **kwargs: (x,)

    def gather(tensor, dim):
        assert events[0] == "weights" and dim == 1
        events.append(f"gather:{tensor.dtype}")
        return torch.cat((tensor, tensor), dim=dim)

    def reader(live, query, cache, live_batch, q_scale=None):
        events.append("reader")
        assert query.shape == (16, 4, 576) and q_scale.shape == (16, 4)
        value = torch.arange(16, dtype=torch.bfloat16)[:, None, None] + 1
        return value.expand(16, 4, 512).clone(), torch.zeros(16, 4, 1)

    def fill(live, query, current, cache, live_batch, output, lse):
        events.append("fill")
        assert query.shape == (16, 4, 192) and current.shape == (16, 576)
        output[2:11].zero_()
        output[2:11, :, :128] = 100 + torch.arange(2, 11)[:, None, None]

    def merge(output, lse):
        events.append("merge")
        assert output.shape == (16, 4, 512)
        return output[:, :2].contiguous()

    def v_up(value):
        assert value.shape[1:] == (2, 512)
        assert (value < 100).all(), "Expanded Prefill output must never be V-up-projected a second time"
        v_up_rows.append(value[:, 0, 0].clone())
        return (value[..., :128] * 2 + 1).flatten(1)

    impl._dcp_all_gather = gather
    impl._merge_dcp_attention_output = merge
    impl._v_up_proj_batch_major = v_up
    monkeypatch.setattr(qfm, "run_qfm_reader", reader)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.attention.kimi_k3_fia_expanded",
        SimpleNamespace(fill_expanded_fia=fill),
    )
    output = torch.empty(16, 256, dtype=torch.bfloat16)
    qfm.forward_qfm(impl, hidden, cache, SimpleNamespace(qfm=batch), output)
    assert events == [
        "weights",
        "gather:torch.int8",
        "gather:torch.float32",
        "gather:torch.bfloat16",
        "reader",
        "fill",
        "merge",
    ]
    assert [rows.tolist() for rows in v_up_rows] == [[1, 2], [12, 13, 14, 15, 16]]
    assert torch.equal(output[2:11], (100 + torch.arange(2, 11))[:, None].expand(9, 256).to(torch.bfloat16))
    assert (output[:2] == torch.tensor([3, 5])[:, None]).all()
    assert (output[11:14] == torch.tensor([25, 27, 29])[:, None]).all()
    assert (output[14:] == 0).all()


@pytest.mark.parametrize("queries,expanded_flag", [(9, True), (33, False)])
def test_pure_decode_and_qfm_prefill_never_enter_expanded_route(qfm, monkeypatch, queries, expanded_flag):
    builder, _ = base.builder(qfm, monkeypatch)
    batch = builder.build(base.common([queries], [129]))
    assert not batch.fia_requests
    cache, _, _, _ = guarded_cache(qfm)
    hidden = torch.randn(queries, 4).to(torch.bfloat16)
    query = torch.ones(queries, 2, 576).to(torch.float8_e4m3fn)
    scale = torch.ones(queries, 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("QFM or pure Decode must not prepare or execute expanded FIA")

    monkeypatch.setattr(qfm, "stage_expanded_fia_inputs", forbidden)
    monkeypatch.setattr(qfm, "fill_fia_prefill", forbidden)
    monkeypatch.setattr(
        qfm,
        "run_qfm_reader",
        lambda *args, **kwargs: (torch.ones(queries, 2, 512), torch.zeros(queries, 2, 1)),
    )
    impl = SimpleNamespace(
        qfm_fia_expanded=expanded_flag,
        qfm_prolog=lambda *args, **kwargs: (query, scale),
        fia_expanded=SimpleNamespace(prepare_for_dcp=forbidden),
        dcp_enabled=False,
        _v_up_proj_batch_major=lambda value: value[..., :128].flatten(1),
        use_output_gate=False,
        o_proj=lambda value, **kwargs: (value,),
    )
    output = torch.empty(queries, 256)
    qfm.forward_qfm(impl, hidden, cache, SimpleNamespace(qfm=batch), output)
    assert (output == 1).all()
