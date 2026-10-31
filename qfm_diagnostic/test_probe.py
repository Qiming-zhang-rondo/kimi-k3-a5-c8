"""CPU contract tests for the standalone probe; these do not validate NPU ops."""

from types import SimpleNamespace
from contextlib import contextmanager, nullcontext

import pytest
import qfm_probe as probe
import torch
from planner import plan_qfm_batch


def test_p_diagnostic_is_unnormalized_exp_with_unrounded_denominator(cpu_runtime):
    q = torch.zeros(1, 1, 576)
    q[0, 0, 0] = 1
    k = torch.zeros(2, 576)
    k[1, 0] = -1.1
    k[:, 1] = torch.tensor([2.0, -3.0])
    result = probe.p_fp8_single_tile_reference(q, [k], [1], [2], 1.0, [0])
    p = torch.tensor([1.0, torch.exp(torch.tensor(-1.1)).item()])
    rounded = (p * 448).to(torch.float8_e4m3fn).float() / 448
    expected = ((rounded @ k[:, :512]) / p.sum()).bfloat16().float()
    torch.testing.assert_close(result[0, 0], expected, rtol=0, atol=0)
    assert probe.p_fp8_single_tile_reference(q, [k], [1], [257], 1.0, [0]) is None


def test_failed_output_still_checks_lse_cache_and_emits_diagnostics(cpu_runtime, monkeypatch):
    reader, _, _ = probe.make_case(CpuNative(), [1], [1], 12)
    original = reader.run
    def wrong_output():
        output, lse = original()
        return output + 1, lse
    monkeypatch.setattr(reader, "run", wrong_output)
    events = []
    monkeypatch.setattr(probe, "emit", lambda event, **fields: events.append((event, fields)))
    with pytest.raises(AssertionError, match="original gates unchanged"):
        probe.check_reader(reader, [1], [1])
    names = {fields.get("name") for event, fields in events if event == "check"}
    assert {"natural_log_lse", "reader_cache_unchanged"} <= names
    assert any(event == "p_fp8_diagnostic" and not fields["acceptance_gate"] for event, fields in events)


class CpuTorch:
    """Redirect explicit NPU allocations while retaining real torch arithmetic."""

    npu = SimpleNamespace(synchronize=lambda: None)
    factories = frozenset({"arange", "tensor", "ones", "zeros", "empty", "full"})

    def __getattr__(self, name):
        value = getattr(torch, name)
        if name not in self.factories:
            return value

        def allocate(*args, **kwargs):
            if kwargs.get("device") == "npu":
                kwargs["device"] = "cpu"
            return value(*args, **kwargs)

        return allocate


@pytest.fixture
def cpu_runtime(monkeypatch):
    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if args and args[0] == "npu":
            args = ("cpu", *args[1:])
        if kwargs.get("device") == "npu":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    calls = []

    def scatter(target, indices, updates):
        assert indices.dtype == torch.int64
        assert target.dtype == updates.dtype == torch.uint8
        calls.append(indices.clone())
        target[indices.flatten()] = updates

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(probe, "torch", CpuTorch(), raising=False)
    monkeypatch.setattr(
        probe,
        "torch_npu",
        SimpleNamespace(npu_scatter_nd_update_=scatter),
        raising=False,
    )
    monkeypatch.setattr(probe, "emit", lambda *args, **kwargs: None)
    return calls


class CpuNative:
    """Independent logical attention evaluator for testing probe plumbing."""

    def __init__(self):
        self.calls = []

    def quant_flash_mla_metadata(self, lengths, heads, kv_heads, quant_mode, **kwargs):
        assert heads in probe.HEADS and kv_heads == quant_mode == 1
        assert kwargs["max_seqlen_q"] <= 16
        assert kwargs["max_seqlen_kv"] > 0
        return (
            lengths.clone(),
            {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
            },
        )

    def quant_flash_mla(self, q, cache, qs, ks, tables, lengths, mode, **kwargs):
        assert mode == 1 and kwargs["layout_kv"] == "PA_NZ"
        assert q.dtype == cache.dtype == torch.float8_e4m3fn
        assert qs.dtype == ks.dtype == torch.float32
        assert qs.shape == q.shape[:2] and ks.shape == (1,)
        assert cache.shape[1:] == (1, 18, 128, 32)
        assert torch.equal(kwargs["metadata"][0], lengths)
        cu, used = kwargs["cu_seqlens_q"], kwargs["seqused_q"]
        assert int(cu[-1]) == q.shape[0]
        mask = kwargs["attn_mask"]
        if kwargs["mask_mode"] == 3:
            assert mask.dtype == torch.int8
            assert torch.equal(mask, torch.ones_like(mask).triu(1))
        else:
            assert kwargs["mask_mode"] == 0 and mask is None
        # Keep inactive rows conspicuous: the probe must compare only model rows.
        output = torch.full((*q.shape[:2], 512), float("nan"), dtype=torch.bfloat16)
        lse = torch.full((q.shape[1], q.shape[0]), float("inf"))
        decoded = (
            cache.float().permute(0, 1, 3, 2, 4).reshape(cache.shape[0], 128, 576) * ks
        )
        dq = q.float() * qs[..., None]
        for batch in range(len(lengths)):
            count, extent = int(used[batch]), int(cu[batch + 1] - cu[batch])
            assert 0 <= count <= extent <= 16
            length = int(lengths[batch])
            if not count:
                continue
            keys = decoded[tables[batch].long()].reshape(-1, 576)[:length]
            for row in range(count):
                token = int(cu[batch]) + row
                visible = (
                    length if kwargs["mask_mode"] == 0 else length - count + row + 1
                )
                if visible == 0:
                    output[token].zero_()
                    continue
                score = dq[token] @ keys[:visible].T * kwargs["softmax_scale"]
                output[token] = score.softmax(-1) @ keys[:visible, :512]
                lse[:, token] = score.logsumexp(-1)
        self.calls.append((cu.clone(), used.clone(), lengths.clone()))
        return output, lse


@pytest.mark.parametrize("gap,offset", [(0, 0), (256, 256), (512, 96)])
def test_writer_preserves_offsets_page_gaps_and_permuted_pages(
    cpu_runtime, gap, offset
):
    generator = torch.Generator().manual_seed(73)
    rows = [
        torch.randn(length, 576, generator=generator).bfloat16()
        for length in (0, 1, 129, 260)
    ]
    state = probe.nz_storage(rows, gap=gap, offset=offset)
    assert len(cpu_runtime) == 1
    assert len(torch.unique(cpu_runtime[0])) == sum(map(len, rows)) * 18
    decoded = state["cache"].float().permute(0, 1, 3, 2, 4).reshape(-1, 128, 576)
    for owner, original in enumerate(rows):
        gathered = decoded[state["table_cpu"][owner].long()].reshape(-1, 576)[
            : len(original)
        ]
        expected = (
            (original.float() / 0.03125)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .float()
        )
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
    assert torch.all(state["storage"][:offset] == 0x5A)
    stride = 128 * 576 + gap
    for page in range(state["cache"].shape[0]):
        begin = offset + page * stride + 128 * 576
        assert torch.all(state["storage"][begin : begin + gap] == 0x5A)
    assert torch.all(state["storage"][-256:] == 0x5A)


def test_writer_all_empty_local_shard_is_noop(cpu_runtime):
    state = probe.nz_storage([torch.empty(0, 576, dtype=torch.bfloat16)], gap=256)
    assert cpu_runtime == []
    assert state["cache"].shape == (1, 1, 18, 128, 32)
    assert torch.all(state["storage"] == 0x5A)


def test_grouped_mixed_requests_and_padding_match_logical_reference(cpu_runtime):
    native = CpuNative()
    reader, _, _ = probe.make_case(
        native, [1, 17, 9], [129, 257, 511], 6, gap=256, padded=48
    )
    result = probe.check_reader(reader, [1, 17, 9], [129, 257, 511])
    assert result["output"]["pass"] and result["lse"]["pass"]
    cu, used, lengths = native.calls[-1]
    assert cu.tolist() == [0, 1, 17, 18, 27, 43, 48]
    assert used.tolist() == [1, 16, 1, 9, 0, 0]
    assert lengths.tolist() == [129, 256, 257, 511, 0, 0]


def test_dcp_prefixes_and_empty_shard_merge_against_global_reference(cpu_runtime):
    native = CpuNative()
    assert probe.dcp_suite(native)["cases"] == 3
    assert len(native.calls) == 6
    assert native.calls[0][2].tolist() == [1]
    assert native.calls[1][2].tolist() == [0]


def test_query_quantization_preserves_zero_and_uses_one_scale_per_head(cpu_runtime):
    q = torch.zeros(3, 6, 576, dtype=torch.bfloat16)
    q[1, 2] = torch.linspace(-2, 2, 576).bfloat16()
    quantized, scales = probe.quantize_query(q)
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.shape == (3, 6) and scales.is_contiguous()
    assert torch.isfinite(scales).all() and (scales > 0).all()
    assert quantized[0].float().count_nonzero() == 0
    assert scales[1, 2] == 2 / 448


def test_nonfinite_results_cannot_pass_even_with_large_tolerances(cpu_runtime):
    assert not probe.compare(torch.tensor([float("nan")]), torch.zeros(1))["pass"]
    assert not probe.compare(
        torch.tensor([float("inf")]), torch.tensor([float("inf")])
    )["pass"]


def test_large_query_plan_has_512_short_sequences_without_dense_indices():
    plan = plan_qfm_batch([8192], [131072])
    assert len(plan.owners) == 512
    assert max(plan.seqused_q) == 16
    assert plan.cache_seqlens[0] == 131072 - 8192 + 16
    assert plan.cache_seqlens[-1] == 131072


def test_graph_budget_failure_does_not_truncate_replays(cpu_runtime, monkeypatch):
    events = []
    monkeypatch.setattr(probe, "emit", lambda event, **fields: events.append((event, fields)))
    active = [None]
    original_run = probe.Reader.run

    def tracked_run(reader):
        result = original_run(reader)
        if active[0] is not None:
            active[0].reader, active[0].buffers = reader, result
        return result

    class Graph:
        def replay(self):
            result = original_run(self.reader)
            for buffer, value in zip(self.buffers, result):
                buffer.copy_(value)

    @contextmanager
    def capture(graph):
        active[0] = graph
        try:
            yield
        finally:
            active[0] = None

    stream = SimpleNamespace(wait_stream=lambda other: None)
    monkeypatch.setattr(probe.torch, "npu", SimpleNamespace(
        synchronize=lambda: None, Stream=lambda: stream, current_stream=lambda: stream,
        stream=lambda value: nullcontext(), NPUGraph=Graph, graph=capture))
    monkeypatch.setattr(probe.Reader, "run", tracked_run)
    original_reference = probe.reference

    def shifted_reference(*args, **kwargs):
        output, lse = original_reference(*args, **kwargs)
        return output + 1, lse

    monkeypatch.setattr(probe, "reference", shifted_reference)
    with pytest.raises(AssertionError, match="All 6 graph replays completed"):
        probe.graph_suite(CpuNative())
    replays = [fields for event, fields in events if event == "graph_replay"]
    assert [item["used"] for item in replays] == [4, 9, 1, 128, 0, 9]
    summary = next(fields for event, fields in events if event == "graph_contract_summary")
    assert summary["replay_contract_passed"]
    assert not summary["fp32_accuracy_budget_passed"]
    assert len(summary["fp32_budget_failures"]) == 5
