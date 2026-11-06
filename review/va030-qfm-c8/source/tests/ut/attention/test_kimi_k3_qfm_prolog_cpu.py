# SPDX-License-Identifier: Apache-2.0
"""Adapter contracts only; mocks are NOT native Prolog/HCCL acceptance."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.attention.test_kimi_k3_qfm_blocks_cpu import prepared_metadata
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


@pytest.mark.parametrize("mode", [0, 2])
@pytest.mark.parametrize("gap", [0, 256])
def test_precomputed_prolog_indices_and_scatter_plan_avoid_arange(prolog, qfm, monkeypatch, mode, gap):
    tokens, offset, stride = 9, 256, 73728 + gap
    backing = torch.full((offset + 3 * stride + 256,), 0x5A, dtype=torch.uint8)
    cache = qfm.cache_view(backing[offset : offset + 3 * stride], 3, stride)
    slots = torch.tensor([128, -1, 0, 383, 127, 7, 129, -1, -1])
    staging = torch.arange(tokens, dtype=torch.int64)
    plan = qfm.QFMScatterPlan(slots, qfm.build_scatter_offsets(slots, stride), stride)
    saved_offsets = plan.offsets.clone()
    hidden = torch.randn(tokens, 1024, dtype=torch.bfloat16)
    rows = hidden[:, :576].float().clamp(-448, 448).to(torch.float8_e4m3fn)
    expected = backing.clone()
    qfm.write_cache(
        hidden[:, :576], slots, qfm.cache_view(expected[offset : offset + 3 * stride], 3, stride), torch.ones(1)
    )

    def native(**kwargs):
        indices, destination = kwargs["cache_index"], kwargs["kv_cache"]
        assert kwargs["qkv_quant_mode"] == mode
        if mode == 2 and gap == 0:
            assert kwargs["cache_mode"] == 1
            # Direct PA_NZ writes must keep the actual scheduler slots, not
            # replace them with the immutable sequential staging indices.
            assert indices is slots
            qfm.scatter_fp8_cache(rows, indices, cache, scatter_plan=plan)
        else:
            assert kwargs["cache_mode"] == 0 and indices is staging
            if mode == 0:
                destination.view(-1, 576)[:tokens].copy_(hidden[:, :576])
            else:
                destination.view(torch.uint8).view(-1, 576)[:tokens].copy_(rows.view(torch.uint8))
        return (
            torch.zeros(tokens, 12, 576, dtype=torch.bfloat16 if mode == 0 else torch.float8_e4m3fn),
            None if mode == 0 else torch.ones(tokens, 12, 1),
            None,
            None,
        )

    monkeypatch.setattr(prolog, "load_prolog", lambda: native)
    fused = prolog.QFMProlog({}, torch.ones(1))

    def forbidden_arange(*args, **kwargs):
        raise AssertionError("Prepared Prolog paths must reuse indices and Scatter offsets")

    monkeypatch.setattr(torch, "arange", forbidden_arange)
    if mode == 0:
        fused.floating(hidden, slots, cache, staging_indices=staging, scatter_plan=plan)
    else:
        fused(hidden, slots, cache, staging_indices=staging, scatter_plan=plan)
    assert torch.equal(backing, expected)
    assert torch.equal(plan.offsets, saved_offsets)


@pytest.mark.parametrize("mode", [0, 2])
@pytest.mark.parametrize("case", ["dtype", "length", "rank", "contiguous", "device"])
def test_prolog_rejects_invalid_precomputed_indices_before_quantization(prolog, qfm, monkeypatch, mode, case):
    hidden = torch.zeros(9, 1024, dtype=torch.bfloat16)
    slots = torch.arange(9, dtype=torch.int64)
    cache = qfm.cache_view(torch.zeros(73728, dtype=torch.uint8), 1, 73728)
    staging = torch.arange(9, dtype=torch.int64)
    if case == "dtype":
        staging = staging.int()
    elif case == "length":
        staging = staging[:8]
    elif case == "rank":
        staging = staging.view(3, 3)
    elif case == "contiguous":
        staging = torch.arange(18, dtype=torch.int64)[::2]
    elif case == "device":
        staging = torch.empty(9, dtype=torch.int64, device="meta")
    monkeypatch.setattr(prolog, "load_prolog", lambda: None)
    fused = prolog.QFMProlog({}, torch.ones(1))

    def forbidden_quant(*args, **kwargs):
        raise AssertionError("Invalid metadata must fail before any device operation")

    monkeypatch.setattr(sys.modules["torch_npu"], "npu_dynamic_mx_quant", forbidden_quant)
    with pytest.raises(ValueError, match="staging_indices"):
        if mode == 0:
            fused.floating(hidden, slots, cache, staging_indices=staging)
        else:
            fused(hidden, slots, cache, staging_indices=staging)


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


@pytest.mark.parametrize("dcp", [1, 2])
def test_scheduler_stage_segments_keep_short_prefill_and_padding(qfm, monkeypatch, dcp):
    b, _ = builder(qfm, monkeypatch, dcp=dcp)
    b.mlapo_prefill = False
    c = common([4, 1, 9, 2], [132, 129, 200, 300], capacity=20)
    c.is_prefilling = torch.tensor([False, True, True, False])
    batch = b.build(c)
    assert batch.preprocess_segments == ((0, 4, False), (4, 14, True), (14, 20, False))
    assert batch.has_prefill
    c.is_prefilling = None
    with pytest.raises(ValueError, match="stage|Stage"):
        b.build(c)


@pytest.mark.parametrize("queries,flags", [([1], [True]), ([4, 9, 1], [False, True, False])])
@pytest.mark.parametrize("fia_prefill", [False, True])
def test_stage_routing_preserves_scatter_slots_and_decode_mode2(qfm, monkeypatch, queries, flags, fia_prefill):
    b, _ = builder(qfm, monkeypatch)
    b.mlapo_prefill = False
    b.fia_prefill = fia_prefill
    tokens = sum(queries)
    c = common(queries, [128 + q for q in queries], capacity=tokens + 2)
    c.is_prefilling = torch.tensor(flags)
    # Distinct, non-monotonic cache slots and invalid padding detect slicing errors.
    c.slot_mapping = torch.tensor(list(reversed(range(tokens))) + [-1, -1])
    batch = b.build(c)
    hidden = torch.randn(tokens + 2, 1024, dtype=torch.bfloat16)
    raw = torch.full((256 + 2 * (73728 + 256),), 91, dtype=torch.uint8)
    cache = qfm.cache_view(raw[256:], 2, 73728 + 256)
    expected = raw.clone()
    qfm.write_cache(hidden[:, :576], batch.slots, qfm.cache_view(expected[256:], 2, 73728 + 256), torch.ones(1))
    ordinary_calls, dsl_calls = [], []

    def ordinary(impl, chunk, count):
        ordinary_calls.append(count)
        return chunk[:, :576, None].transpose(1, 2).repeat(1, 12, 1), chunk[:, :576]

    class NativeProlog:
        def __call__(self, chunk, slots, destination):
            count = chunk.shape[0]
            qfm.write_cache(chunk[:, :576], slots, destination, torch.ones(1))
            codes = (torch.arange(count * 12 * 576).reshape(count, 12, 576) % 31 - 15).to(torch.float8_e4m3fn)
            scales = torch.arange(count * 12, dtype=torch.float32).reshape(count, 12) + 0.25
            dsl_calls.append((count, codes.clone(), scales.clone()))
            return codes, scales

        def floating(self, *args):
            raise AssertionError("Mixed-batch Decode must keep native FP8 mode2 outputs")

    monkeypatch.setattr(qfm, "ordinary_qfm_inputs", ordinary)
    impl = SimpleNamespace(qfm_scale=torch.ones(1), num_heads=12)
    q, kv, q_scale, fia_q = qfm.stage_specific_qfm_inputs(impl, hidden, cache, batch, NativeProlog())
    assert torch.equal(raw, expected)
    assert ordinary_calls == [end - begin for begin, end, p in batch.preprocess_segments if p]
    assert [entry[0] for entry in dsl_calls] == [end - begin for begin, end, p in batch.preprocess_segments if not p]
    expected_q = hidden[:, None, :576].repeat(1, 12, 1)
    if all(flags):
        # Pure Prefill keeps the existing ordinary BF16 route, including its
        # current floating KV for FIA; it never calls the DSL Prolog.
        torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(kv, hidden[:, :576], rtol=0, atol=0)
        assert q_scale is None and fia_q is None
    else:
        assert q.dtype == torch.float8_e4m3fn
        assert q_scale.shape == (tokens + 2, 12)
        decode_index = 0
        for begin, end, prefill in batch.preprocess_segments:
            if prefill and fia_prefill:
                expected_codes = torch.zeros(end - begin, 12, 576).to(torch.float8_e4m3fn)
                expected_scale = torch.ones(end - begin, 12)
            elif prefill:
                expected_codes, expected_scale = qfm.quantize_query(expected_q[begin:end])
            else:
                _, expected_codes, expected_scale = dsl_calls[decode_index]
                decode_index += 1
            assert torch.equal(q[begin:end].view(torch.uint8), expected_codes.view(torch.uint8))
            assert torch.equal(q_scale[begin:end], expected_scale)
            if fia_prefill:
                if prefill:
                    torch.testing.assert_close(fia_q[begin:end], expected_q[begin:end], rtol=0, atol=0)
                    torch.testing.assert_close(kv[begin:end], hidden[begin:end, :576], rtol=0, atol=0)
                else:
                    assert fia_q[begin:end].count_nonzero() == 0
                    assert kv[begin:end].count_nonzero() == 0
        if not fia_prefill:
            assert kv is None and fia_q is None


def test_mixed_stage_shared_plan_keeps_slot_slice_and_zero_based_staging(prolog, qfm, monkeypatch):
    b = prepared_metadata(qfm, monkeypatch, manager=256, gap=256, mlapo=True, prefill_mlapo=False)
    c = common([4, 9, 3], [132, 137, 131], capacity=20)
    c.is_prefilling = torch.tensor([False, True, False])
    c.slot_mapping = torch.tensor(
        [128, -1, 0, 383, 127, 7, 129, 255, 4, 256, 257, 300, -1, 380, 2, 200, -1, -1, -1, -1]
    )
    batch = b.build(c)
    assert batch.preprocess_segments == ((0, 4, False), (4, 13, True), (13, 20, False))
    assert batch.scatter_plan is not None
    staging = batch.prolog_kwargs["staging_indices"]
    hidden = torch.randn(20, 1024, dtype=torch.bfloat16)
    offset, stride = 256, 73728 + 256
    raw = torch.full((offset + 4 * stride + 256,), 0x6B, dtype=torch.uint8)
    cache = qfm.cache_view(raw[offset : offset + 4 * stride], 4, stride)
    expected = raw.clone()
    qfm.write_cache(
        hidden[:, :576],
        batch.slots,
        qfm.cache_view(expected[offset : offset + 4 * stride], 4, stride),
        torch.ones(1),
    )
    native_calls, observed_ranges = [], []
    decode_ranges = [(0, 4), (13, 20)]

    def native(**kw):
        count = kw["x"].shape[0]
        indices, destination = kw["cache_index"], kw["kv_cache"]
        assert kw["qkv_quant_mode"] == 2 and kw["cache_mode"] == 0
        # Every fresh temporary Cache starts at zero, including Decode at
        # global row 13. A slice [begin:end] here would corrupt/skip its rows.
        assert indices.data_ptr() == staging.data_ptr()
        assert indices.storage_offset() == staging.storage_offset()
        assert indices.tolist() == list(range(count))
        assert destination.data_ptr() != cache.data_ptr()
        rows = kw["x"].view(torch.uint8)[:, :576].contiguous()
        destination.view(torch.uint8).view(-1, 576)[:count].copy_(rows)
        native_calls.append(count)
        return rows[:, None, :].repeat(1, 12, 1).view(torch.float8_e4m3fn), torch.full((count, 12, 1), 0.25), None, None

    monkeypatch.setattr(prolog, "load_prolog", lambda: native)
    fused = prolog.QFMProlog({}, torch.ones(1))

    class ObservedProlog:
        def __call__(self, chunk, slots, destination, *, staging_indices, scatter_plan):
            begin, end = decode_ranges[len(observed_ranges)]
            assert slots.data_ptr() == batch.slots[begin:end].data_ptr()
            assert torch.equal(slots, batch.slots[begin:end])
            assert scatter_plan.slots.data_ptr() == slots.data_ptr()
            assert scatter_plan.offsets.data_ptr() == batch.scatter_plan.offsets[begin:end].data_ptr()
            assert torch.equal(scatter_plan.offsets, batch.scatter_plan.offsets[begin:end])
            assert scatter_plan.stride == stride
            assert staging_indices.shape == (end - begin,)
            assert staging_indices.data_ptr() == staging.data_ptr()
            observed_ranges.append((begin, end))
            return fused(chunk, slots, destination, staging_indices=staging_indices, scatter_plan=scatter_plan)

        def floating(self, *args, **kwargs):
            raise AssertionError("Decode must not downgrade to mode0")

    def ordinary(impl, chunk, count):
        assert count == 9
        return chunk[:, None, :576].repeat(1, 12, 1), chunk[:, :576]

    monkeypatch.setattr(qfm, "ordinary_qfm_inputs", ordinary)
    impl = SimpleNamespace(qfm_scale=torch.ones(1), num_heads=12)
    q, kv, scales, fia_q = qfm.stage_specific_qfm_inputs(impl, hidden, cache, batch, ObservedProlog())
    assert observed_ranges == decode_ranges and native_calls == [4, 7]
    assert q.dtype == torch.float8_e4m3fn and scales.shape == (20, 12)
    assert (scales[:4] == 0.25).all() and (scales[13:] == 0.25).all()
    assert kv is None and fia_q is None
    # Full backing equality includes offset guards, every per-page padding
    # interval, invalid slots, unselected history and the trailing guard.
    assert torch.equal(raw, expected)


@pytest.mark.parametrize("prefill", [False, True])
def test_forward_decode_keeps_mode2_prefill_uses_ordinary(qfm, monkeypatch, prefill):
    b, _ = builder(qfm, monkeypatch)
    b.mlapo_prefill = False
    c = common([4], [132], capacity=8)
    c.is_prefilling = torch.tensor([prefill])
    batch = b.build(c, retain_for_graph=not prefill)
    calls = []

    def prolog(hidden, slots, cache):
        calls.append("decode_mode2")
        return torch.zeros(8, 12, 576).to(torch.float8_e4m3fn), torch.ones(8, 12)

    def staged(*args):
        calls.append("ordinary_prefill")
        return torch.zeros(8, 12, 576, dtype=torch.bfloat16), torch.zeros(8, 576, dtype=torch.bfloat16), None, None

    def reader(impl, q, cache, batch, q_scale):
        assert (q_scale is None) == prefill
        return torch.ones(8, 12, 512), None

    monkeypatch.setattr(qfm, "stage_specific_qfm_inputs", staged)
    monkeypatch.setattr(qfm, "run_qfm_reader", reader)
    impl = SimpleNamespace(
        qfm_prolog=prolog,
        qfm_mlapo_prefill=False,
        dcp_enabled=False,
        _v_up_proj_batch_major=lambda x: x.flatten(1),
        use_output_gate=False,
        o_proj=lambda x, **kw: (x,),
    )
    qfm.forward_qfm(impl, torch.randn(8, 1024), None, SimpleNamespace(qfm=batch), torch.empty(8, 12 * 512))
    assert calls == ["ordinary_prefill" if prefill else "decode_mode2"]


@pytest.mark.parametrize("dcp,retain_for_graph", [(1, True), (2, False), (2, True)])
def test_pure_decode_skips_host_stage_boundaries(qfm, monkeypatch, dcp, retain_for_graph):
    b, _ = builder(qfm, monkeypatch, dcp=dcp)
    b.mlapo_prefill = False
    c = common([8], [1024], capacity=16)
    c.is_prefilling.fill_(False)

    class ForbiddenHostBoundaries:
        def __getitem__(self, index):
            raise AssertionError("Pure Decode must not read or reconstruct CPU stage boundaries")

    c.query_start_loc_cpu = ForbiddenHostBoundaries()
    batch = b.build(c, retain_for_graph=retain_for_graph)
    assert not batch.has_prefill
    assert batch.preprocess_segments == ()


@pytest.mark.parametrize("same_capacity", [False, True])
def test_decode_capture_then_prefill_then_decode_keeps_mode2(qfm, monkeypatch, same_capacity):
    b, _ = builder(qfm, monkeypatch)
    b.mlapo_prefill = False
    c = common([4], [132], capacity=8)
    c.is_prefilling.fill_(False)
    captured = b.build(c, retain_for_graph=True)
    original_pointers = {name: getattr(captured, name).data_ptr() for name in ("table", "slots", "lengths", "used")}
    p = common([4 if same_capacity else 33], [1024], capacity=8 if same_capacity else 40)
    p.is_prefilling.fill_(True)
    prefill = b.build(p)
    assert prefill.has_prefill
    c.seq_lens.fill_(260)
    c.slot_mapping.add_(128)
    decoded = b.build(c)
    assert decoded is captured
    assert not decoded.has_prefill and not decoded.preprocess_segments
    assert original_pointers == {name: getattr(decoded, name).data_ptr() for name in original_pointers}
    assert torch.equal(decoded.slots[:4], c.slot_mapping)
    assert decoded.lengths[:4].tolist() == [257, 258, 259, 260]
    mode2 = []

    class Prolog:
        def __call__(self, hidden, slots, cache):
            mode2.append(slots.clone())
            return torch.zeros(8, 12, 576).to(torch.float8_e4m3fn), torch.full((8, 12), 0.25)

        def floating(self, *args):
            raise AssertionError("Decode graph path switched to mode0 after Prefill")

    def reader(impl, q, cache, batch, q_scale):
        assert q.dtype == torch.float8_e4m3fn
        assert torch.equal(q_scale, torch.full((8, 12), 0.25))
        return torch.zeros(8, 12, 512), None

    monkeypatch.setattr(qfm, "run_qfm_reader", reader)
    impl = SimpleNamespace(
        qfm_prolog=Prolog(),
        qfm_mlapo_prefill=False,
        dcp_enabled=False,
        _v_up_proj_batch_major=lambda x: x.flatten(1),
        use_output_gate=False,
        o_proj=lambda x, **kw: (x,),
    )
    qfm.forward_qfm(impl, torch.zeros(8, 1024), None, SimpleNamespace(qfm=decoded), torch.empty(8, 12 * 512))
    assert len(mode2) == 1


@pytest.mark.parametrize("fia_prefill", [False, True])
def test_mixed_dcp_keeps_fp8_q_scales_and_separate_fia_q(qfm, monkeypatch, fia_prefill):
    b, _ = builder(qfm, monkeypatch, dcp=2, rank=1)
    b.mlapo_prefill = False
    b.fia_prefill = fia_prefill
    c = common([4, 9], [132, 1024], capacity=16)
    c.is_prefilling = torch.tensor([False, True])
    batch = b.build(c)
    q8 = (torch.arange(16 * 12 * 576).reshape(16, 12, 576) % 31 - 15).to(torch.float8_e4m3fn)
    scales = torch.arange(16 * 12, dtype=torch.float32).reshape(16, 12) + 0.25
    floating_q = torch.zeros(16, 12, 576, dtype=torch.bfloat16)
    floating_q[4:13].fill_(3.5)
    kv = torch.zeros(16, 576, dtype=torch.bfloat16)
    kv[4:13].fill_(2.5)
    gathers, fia_calls = [], []

    def staged(*args):
        return q8, kv if fia_prefill else None, scales, floating_q if fia_prefill else None

    def gather(x, dim):
        assert dim == 1
        gathers.append(x.dtype)
        return torch.cat((x, x), dim=dim)

    def attention(q, cache, q_scale, *args, **kwargs):
        assert q.dtype == torch.float8_e4m3fn
        assert torch.equal(q[:, :12].view(torch.uint8), q8.view(torch.uint8))
        assert torch.equal(q[:, 12:].view(torch.uint8), q8.view(torch.uint8))
        assert torch.equal(q_scale[:, :12], scales)
        assert torch.equal(q_scale[:, 12:], scales)
        return torch.ones(16, 24, 512), torch.zeros(24, 16)

    def fill(impl, q, current_kv, cache, batch, latent, lse):
        assert fia_prefill
        assert q.dtype == torch.bfloat16
        torch.testing.assert_close(q, torch.cat((floating_q, floating_q), dim=1), rtol=0, atol=0)
        torch.testing.assert_close(current_kv, kv, rtol=0, atol=0)
        fia_calls.append(True)

    monkeypatch.setattr(qfm, "stage_specific_qfm_inputs", staged)
    monkeypatch.setattr(qfm, "fill_fia_prefill", fill)
    impl = SimpleNamespace(
        qfm_prolog=object(),
        qfm_mlapo_prefill=False,
        dcp_enabled=True,
        _dcp_all_gather=gather,
        _merge_dcp_attention_output=lambda x, lse: x[:, :12],
        qfm_ops=(attention, None),
        qfm_scale=torch.ones(1),
        scale=0.1,
        _v_up_proj_batch_major=lambda x: x.flatten(1),
        use_output_gate=False,
        o_proj=lambda x, **kw: (x,),
    )
    output = torch.empty(16, 12 * 512)
    qfm.forward_qfm(impl, torch.zeros(16, 1024), None, SimpleNamespace(qfm=batch), output)
    assert gathers.count(torch.int8) == 1
    assert gathers.count(torch.float32) == 1
    assert gathers.count(torch.bfloat16) == int(fia_prefill)
    assert len(fia_calls) == int(fia_prefill)
    assert output[13:].count_nonzero() == 0
