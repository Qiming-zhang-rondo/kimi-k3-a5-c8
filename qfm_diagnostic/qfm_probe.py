"""Standalone A5 QFM contract probe. No vLLM imports, model weights or patches.

All BF16 history here is a TEST oracle/baseline, not a proposed resident cache.
The tested production representation is full576 E4M3 + one static FP32 scale.
"""

import argparse
import importlib
import importlib.metadata
import json
import math
import os
import statistics
import time
from pathlib import Path

from planner import plan_qfm_batch

BLOCK = 128
WIDTH = 576
VALUE = 512
HEADS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
DEFAULT_SCALING = 1 / math.sqrt(192)


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def inspect_runtime():
    global torch, torch_npu
    import torch
    import torch_npu

    versions = {}
    for name in ("torch", "torch-npu", "cann-ops-transformer", "cannbot-dsl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    providers = {}
    for name in ("cann_ops_transformer", "cann_ops_transformer.ops"):
        try:
            module = importlib.import_module(name)
            providers[name] = {"path": getattr(module, "__file__", None)}
        except Exception as error:  # noqa: BLE001 - inventory reports loader/build errors without hiding them.
            providers[name] = {"error": f"{type(error).__name__}: {error}"}
    schemas = [
        str(schema)
        for schema in torch._C._jit_get_all_schemas()
        if any(name in str(schema) for name in ("mla_prolog", "quant_flash_mla"))
    ]
    cann_files = {}
    roots = [Path("/usr/local/Ascend/ascend-toolkit/latest")]
    if os.environ.get("ASCEND_HOME_PATH"):
        roots.insert(0, Path(os.environ["ASCEND_HOME_PATH"]))
    for root in roots:
        for path in (root / "version.cfg", root / "version.info"):
            if path.is_file():
                lines = [
                    line
                    for line in path.read_text(errors="replace").splitlines()
                    if "version" in line.lower()
                ]
                cann_files[str(path)] = lines[:12]
    report = {
        "versions": versions,
        "cann_version_files": cann_files,
        "torch_npu_prolog_v4_python_available": hasattr(torch_npu, "npu_mla_prolog_v4"),
        "operator_providers": providers,
        "mla_and_qfm_registered_schemas": schemas,
        "prolog_v4_qfm_cache_contract_validated": False,
        "note": "V4 may be exposed as cann_ops_transformer::mla_prolog (V4WeightNz), not a torch_npu V4 symbol. "
        "Registration alone does not prove full576 per-tensor FP8 PA_NZ writer compatibility.",
    }
    emit("environment", **report)
    return report


def load_runtime(card=0):
    inspect_runtime()
    torch.npu.set_device(card)
    try:
        importlib.import_module("cann_ops_transformer.ops")
        native = torch.ops.cann_ops_transformer
        for name in ("quant_flash_mla", "quant_flash_mla_metadata"):
            getattr(native, name)
    except (ImportError, AttributeError, OSError) as error:
        raise RuntimeError(
            "Missing/incompatible QFM+metadata extension. Run build_qfm.sh in the A5 container; "
            "do not upgrade CANN or count this as a skipped/pass case."
        ) from error
    emit(
        "native_interfaces",
        device=torch.npu.get_device_name(card),
        qfm_schema=str(native.quant_flash_mla.default._schema),
        metadata_schema=str(native.quant_flash_mla_metadata.default._schema),
    )
    return native


def quantize_query(q):
    q32 = q.float()
    scale = q32.abs().amax(-1).clamp_min(1e-12) / 448.0
    fp8 = (q32 / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return fp8, scale.contiguous()


def compare(actual, expected, *, rtol=0.03, atol=0.025, relative_l2=0.02):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    error = (actual - expected).abs()
    rel = float(error.norm() / expected.norm().clamp_min(1e-12))
    bad = int((error > atol + rtol * expected.abs()).sum())
    return {
        "pass": finite and bad == 0 and rel <= relative_l2,
        "finite": finite,
        "relative_l2": rel,
        "max_abs": float(error.max()) if error.numel() else 0.0,
        "bad_elements": bad,
        "elements": actual.numel(),
    }


def require_check(name, actual, expected, **kwargs):
    result = compare(actual, expected, **kwargs)
    emit("check", name=name, **result)
    if not result["pass"]:
        raise AssertionError(f"{name}: {result}")
    return result


def output_diagnostics(actual, expected, samples):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    error = (actual - expected).abs()
    margin = error - (0.025 + 0.03 * expected.abs())
    count = min(8, margin.numel())
    indices = margin.flatten().topk(count).indices.tolist()
    records = []
    for flat in indices:
        row, rem = divmod(flat, actual.shape[1] * actual.shape[2])
        head, dim = divmod(rem, actual.shape[2])
        records.append(dict(token=samples[row], head=head, dim=dim,
                            actual=float(actual[row, head, dim]),
                            expected=float(expected[row, head, dim]),
                            abs_error=float(error[row, head, dim]),
                            excess_over_tolerance=float(margin[row, head, dim])))
    emit("output_error_details", worst_tolerance_margins=records)


def p_fp8_single_tile_reference(q, kv_list, query_lengths, cache_lengths, scaling, samples):
    """Diagnostic only: <=256 keys, exp(score-max)*448 rounded to E4M3.

    Denominator stays FP32 BEFORE rounding, as in QfmlaSoftmax256.
    Not a complete emulator of split-K/online softmax or hardware rounding.
    Never replaces the original accuracy acceptance gate.
    """
    if max(cache_lengths) > 256:
        return None
    outputs = []
    starts = [0]
    for length in query_lengths:
        starts.append(starts[-1] + length)
    for token in samples:
        owner = next(i for i in range(len(query_lengths)) if starts[i] <= token < starts[i + 1])
        visible = cache_lengths[owner] - query_lengths[owner] + token - starts[owner] + 1
        k = kv_list[owner][:visible].float()
        logits = q[token].float() @ k.T * scaling
        p = (logits - logits.amax(-1, keepdim=True)).exp()
        p_rounded = (p * 448.0).to(torch.float8_e4m3fn).float() / 448.0
        outputs.append(((p_rounded @ k[:, :VALUE]) / p.sum(-1, keepdim=True)).bfloat16().float())
    return torch.stack(outputs)


def nz_storage(kv_list, *, scale=0.03125, gap=0, offset=256, seed=417):
    """Independent CPU oracle plus native tile scatter, including guard bytes.

    Pages are deliberately permuted. Every unwritten byte is sentinel-filled.
    Padded page strides and nonzero storage offsets are tested explicitly.
    """
    if gap % 32 or offset % 32:
        raise ValueError("Page padding/offset must be aligned to a 32-byte tile")
    generator = torch.Generator().manual_seed(seed)
    counts = [max(1, math.ceil(len(kv) / BLOCK)) for kv in kv_list]
    pages = sum(counts)
    permutation = torch.randperm(pages, generator=generator)
    table = torch.zeros((len(kv_list), max(counts)), dtype=torch.int32)
    stride = BLOCK * WIDTH + gap
    expected = torch.full((offset + pages * stride + 256,), 0x5A, dtype=torch.uint8)
    initial = expected.clone()
    kv_quantized = []
    slots = []
    base = 0
    for owner, (kv, count) in enumerate(zip(kv_list, counts)):
        table[owner, :count] = permutation[base : base + count]
        quantized = (kv.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        kv_quantized.append(quantized.float() * scale)
        for page in range(count):
            physical = int(table[owner, page])
            begin, end = page * BLOCK, min(len(kv), (page + 1) * BLOCK)
            if begin == end:
                continue
            # Oracle uses 2D tile transpose, NOT the writer's flat index formula.
            page_view = expected[
                offset + physical * stride : offset + physical * stride + BLOCK * WIDTH
            ].view(18, BLOCK, 32)
            page_view[:, : end - begin].copy_(
                quantized[begin:end]
                .view(torch.uint8)
                .reshape(end - begin, 18, 32)
                .transpose(0, 1)
            )
            slots.extend(physical * BLOCK + i for i in range(end - begin))
        base += count
    storage = initial.to("npu")
    cache = torch.as_strided(
        storage,
        (pages, 1, 18, BLOCK, 32),
        (stride, BLOCK * WIDTH, BLOCK * 32, 32, 1),
        offset,
    )
    cache = cache.view(torch.float8_e4m3fn)
    rows = torch.cat(kv_list).to("npu")
    slot_tensor = torch.tensor(slots, dtype=torch.int64, device="npu")
    descale = torch.tensor([scale], dtype=torch.float32, device="npu")

    def scatter(input_rows, input_slots):
        # One scatter for all current rows. No Q-by-K token index tensor.
        if input_rows.shape[0] == 0:
            return
        data = (input_rows.float() / descale).clamp(-448, 448).to(torch.float8_e4m3fn)
        indices = (
            input_slots.div(BLOCK, rounding_mode="floor")[:, None] * (stride // 32)
            + torch.arange(18, device="npu")[None, :] * BLOCK
            + input_slots.remainder(BLOCK)[:, None]
        )
        alias = storage[offset : offset + pages * stride].view(-1, 32)
        torch_npu.npu_scatter_nd_update_(
            alias, indices.reshape(-1, 1), data.view(torch.uint8).reshape(-1, 32)
        )

    def write():
        scatter(rows, slot_tensor)

    write()
    torch.npu.synchronize()
    if not torch.equal(storage.cpu(), expected):
        raise AssertionError("FP8 scatter mismatch or guard/page-padding corruption")
    emit(
        "check",
        name="native_writer_bytes_and_guards",
        **{"pass": True},
        storage_offset=offset,
        page_padding_bytes=gap,
        pages=pages,
    )
    return {
        "cache": cache,
        "storage": storage,
        "expected": expected,
        "table_cpu": table,
        "table": table.to("npu"),
        "scale": descale,
        "kv_quantized": kv_quantized,
        "write": write,
        "scatter": scatter,
        "stride": stride,
        "offset": offset,
        "scalar_scale": scale,
    }


def logical_cache_cpu(state, table_row, length):
    """Independent physical NZ decode for page-table mutation tests only."""
    pages = []
    for physical in table_row[: math.ceil(length / BLOCK)].tolist():
        start = state["offset"] + physical * state["stride"]
        raw = state["expected"][start : start + BLOCK * WIDTH]
        pages.append(
            raw.view(18, BLOCK, 32).transpose(0, 1).contiguous().view(BLOCK, WIDTH)
        )
    return (
        torch.cat(pages)[:length].view(torch.float8_e4m3fn).float()
        * state["scalar_scale"]
    )


class Reader:
    def __init__(self, native, q, state, plan, *, scaling=DEFAULT_SCALING):
        self.native, self.q_bf16, self.state, self.plan = native, q, state, plan
        self.return_lse = True
        self.q, self.qscale = quantize_query(q)
        self.scaling = scaling
        self.cu = torch.tensor(plan.cu_seqlens_q, dtype=torch.int32, device="npu")
        self.used = torch.tensor(plan.seqused_q, dtype=torch.int32, device="npu")
        self.lengths = torch.tensor(plan.cache_seqlens, dtype=torch.int32, device="npu")
        self.owners = torch.tensor(
            [max(0, x) for x in plan.owners], dtype=torch.int64, device="npu"
        )
        self.table = state["table"].index_select(0, self.owners).contiguous()
        self.max_q = max(
            b - a for a, b in zip(plan.cu_seqlens_q, plan.cu_seqlens_q[1:])
        )
        self.max_k = self.table.shape[1] * BLOCK
        # Maintained operator test inputs use an upper-triangular blocked mask.
        # This is validated against an independent right-aligned causal oracle.
        self.mask = (
            torch.ones((2048, 2048), dtype=torch.int8, device="npu").triu(1)
            if plan.mask_mode == 3
            else None
        )

    def metadata(self):
        return self.native.quant_flash_mla_metadata(
            self.lengths,
            self.q.shape[1],
            1,
            1,
            cu_seqlens_q=self.cu,
            seqused_q=self.used,
            max_seqlen_q=self.max_q,
            max_seqlen_kv=self.max_k,
            head_dim_qk=WIDTH,
            head_dim_v=VALUE,
            mask_mode=self.plan.mask_mode,
            layout_q="TND",
        )

    def attention(self, metadata):
        return self.native.quant_flash_mla(
            self.q,
            self.state["cache"],
            self.qscale,
            self.state["scale"],
            self.table,
            self.lengths,
            1,
            cu_seqlens_q=self.cu,
            seqused_q=self.used,
            metadata=metadata,
            head_dim_v=VALUE,
            softmax_scale=self.scaling,
            attn_mask=self.mask,
            mask_mode=self.plan.mask_mode,
            max_seqlen_q=self.max_q,
            max_seqlen_kv=self.max_k,
            layout_q="TND",
            layout_kv="PA_NZ",
            layout_out="TND",
            return_softmax_lse=self.return_lse,
        )

    def run(self):
        return self.attention(self.metadata())

    def prepared_read(self):
        self.q, self.qscale = quantize_query(self.q_bf16)
        self.table = self.state["table"].index_select(0, self.owners).contiguous()
        return self.run()


def selected_rows(total, limit=0):
    if not limit or total <= limit:
        return list(range(total))
    # Both sides of segment boundaries, endpoints, and evenly spaced samples.
    important = {0, 1, 14, 15, 16, 17, total - 1, total - 2}
    important.update(round(i * (total - 1) / max(1, limit - 1)) for i in range(limit))
    return sorted(x for x in important if 0 <= x < total)


def reference(q, kv_list, query_lengths, cache_lengths, scaling, samples=None):
    """CPU FP32 oracle on logical requests, not virtual-segment metadata."""
    samples = list(range(q.shape[0])) if samples is None else samples
    output, lses = [], []
    starts = [0]
    for count in query_lengths:
        starts.append(starts[-1] + count)
    for token in samples:
        owner = next(
            i for i in range(len(query_lengths)) if starts[i] <= token < starts[i + 1]
        )
        visible = (
            cache_lengths[owner] - query_lengths[owner] + token - starts[owner] + 1
        )
        k = kv_list[owner][:visible].float()
        logits = q[token].float() @ k.T * scaling
        lses.append(logits.logsumexp(-1))
        output.append(logits.softmax(-1) @ k[:, :VALUE])
    return torch.stack(output), torch.stack(lses)


def make_case(native, query_lengths, cache_lengths, heads, *, gap=0, padded=None):
    generator = torch.Generator().manual_seed(502 + heads)
    kv = [
        torch.randn(length, WIDTH, generator=generator).to(torch.bfloat16)
        for length in cache_lengths
    ]
    tokens = sum(query_lengths)
    q_cpu = torch.randn(padded or tokens, heads, WIDTH, generator=generator).to(
        torch.bfloat16
    )
    state = nz_storage(kv, gap=gap)
    plan = plan_qfm_batch(query_lengths, cache_lengths, padded_tokens=padded)
    reader = Reader(native, q_cpu.to("npu"), state, plan)
    return reader, q_cpu, kv


def check_reader(reader, query_lengths, cache_lengths, *, sample_limit=0):
    emit(
        "stage",
        name="qfm_attention",
        status="begin",
        query_tokens=sum(query_lengths),
        virtual_sequences=len(reader.plan.owners),
        native_attention_calls=1,
        mask_mode=reader.plan.mask_mode,
    )
    output, lse = reader.run()
    torch.npu.synchronize()
    if tuple(output.shape) != (reader.q.shape[0], reader.q.shape[1], VALUE):
        raise AssertionError(f"Unexpected QFM output layout: {output.shape}")
    if tuple(lse.shape) != (reader.q.shape[1], reader.q.shape[0]):
        raise AssertionError(f"Unexpected QFM LSE layout: {lse.shape}")
    samples = selected_rows(sum(query_lengths), sample_limit)
    # Dequantization is an independent TEST reference only, not the reader path.
    q_ref = reader.q.cpu().float() * reader.qscale.cpu()[..., None]
    expected, expected_lse = reference(
        q_ref,
        reader.state["kv_quantized"],
        query_lengths,
        cache_lengths,
        reader.scaling,
        samples,
    )
    actual = output.cpu()[samples]
    checks = {
        "output": compare(actual, expected),
        "lse": compare(
            lse.cpu().T[samples],
            expected_lse,
            relative_l2=0.003,
            atol=0.02,
            rtol=0.003,
        ),
    }
    emit("check", name="qfm_vs_quantized_fp32_reference", **checks["output"])
    emit("check", name="natural_log_lse", **checks["lse"])
    cache_unchanged = torch.equal(reader.state["storage"].cpu(), reader.state["expected"])
    emit("check", name="reader_cache_unchanged", **{"pass": cache_unchanged})
    output_diagnostics(actual, expected, samples)
    p_expected = p_fp8_single_tile_reference(q_ref, reader.state["kv_quantized"],
                                           query_lengths, cache_lengths, reader.scaling, samples)
    if p_expected is not None:
        emit("p_fp8_diagnostic", acceptance_gate=False,
             scope="single-tile numerical hypothesis, not full native kernel emulator",
             native_vs_p_fp8=compare(actual, p_expected),
             p_fp8_vs_fp32=compare(p_expected, expected))
    emit(
        "reference_scope",
        sampled_rows=len(samples),
        total_rows=sum(query_lengths),
        exhaustive=len(samples) == sum(query_lengths),
        cache_unchanged=cache_unchanged,
    )
    if not cache_unchanged or not all(check["pass"] for check in checks.values()):
        raise AssertionError(f"Reader accuracy failed (original gates unchanged): {checks}; cache_unchanged={cache_unchanged}")
    return checks


def small_suite(native):
    cases = [
        ([1], [1], 12, 0, None),
        ([4], [127], 24, 0, None),
        ([9], [129], 48, 0, None),
        ([9], [127], 48, 0, None),
        ([9], [128], 48, 0, None),
        ([15], [129], 6, 0, None),
        ([16], [128], 12, 0, None),
        ([17], [129], 12, 0, None),
        ([33], [1024], 12, 256, None),
        ([1, 17, 9], [129, 257, 511], 12, 256, None),
        ([17], [129], 12, 0, 48),
    ]
    failures = []
    for q, k, heads, gap, padded in cases:
        emit(
            "case_begin",
            queries=q,
            kv_lengths=k,
            heads=heads,
            page_gap=gap,
            padded=padded,
        )
        reader, _, _ = make_case(native, q, k, heads, gap=gap, padded=padded)
        try:
            check_reader(reader, q, k)
        except AssertionError as error:
            failures.append(dict(queries=q, kv_lengths=k, heads=heads, error=str(error)))
            emit("case_complete", **{"pass": False}, queries=q, kv_lengths=k, heads=heads, error=str(error))
        else:
            emit("case_complete", **{"pass": True}, queries=q, kv_lengths=k, heads=heads)
        del reader
    if failures:
        emit("small_accuracy_summary", cases=len(cases), failed=len(failures), failures=failures)
        raise AssertionError(f"{len(failures)} accuracy cases failed; diagnostics collected, thresholds unchanged")
    return {
        "cases": len(cases),
        "scope": "single-card reader/writer; not TP collectives/model",
    }


def graph_suite(native):
    """Fixed physical Q capacity; refresh device lengths, used rows and pages."""
    capacity = 128
    reader, _, _ = make_case(native, [capacity], [256], 12)
    # Per-query rows make graph capacity unchanged across speculative lengths.
    plan = plan_qfm_batch([capacity], [256], max_query_rows=1)
    reader = Reader(native, reader.q_bf16, reader.state, plan)
    reader.plan = type(plan)(**{**plan.__dict__, "mask_mode": 0})
    reader.mask = None
    side = torch.npu.Stream()
    side.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(side):
        for _ in range(3):
            reader.run()
    torch.npu.current_stream().wait_stream(side)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured_out, captured_lse = reader.run()
    fp32_budget_failures = []
    completed_replays = 0
    for step, (used, length) in enumerate(
        ((4, 127), (9, 129), (1, 1), (128, 256), (0, 0), (9, 129))
    ):
        # Graph inputs retain storage addresses. Fresh metadata is inside graph.
        lengths = [length - used + i + 1 if i < used else 0 for i in range(capacity)]
        reader.lengths.copy_(torch.tensor(lengths, dtype=torch.int32, device="npu"))
        reader.used.copy_(
            torch.tensor(
                [int(i < used) for i in range(capacity)],
                dtype=torch.int32,
                device="npu",
            )
        )
        table_cpu = (
            reader.state["table_cpu"].flip(1) if step % 2 else reader.state["table_cpu"]
        )
        reader.table.copy_(table_cpu.to("npu").expand_as(reader.table))
        changed_q, changed_scale = quantize_query(reader.q_bf16 * (1.0 + step * 0.125))
        reader.q.copy_(changed_q)
        reader.qscale.copy_(changed_scale)
        graph.replay()
        torch.npu.synchronize()
        replay_out, replay_lse = captured_out.clone(), captured_lse.clone()
        eager_out, eager_lse = reader.run()
        torch.npu.synchronize()
        if used:
            require_check("graph_vs_eager", replay_out[:used], eager_out[:used])
            require_check("graph_lse_vs_eager", replay_lse.T[:used], eager_lse.T[:used])
            q_ref = reader.q.cpu().float() * reader.qscale.cpu()[..., None]
            logical_kv = logical_cache_cpu(reader.state, table_cpu[0], length)
            expected, expected_lse = reference(
                q_ref[:used], [logical_kv], [used], [length], reader.scaling
            )
            budget = compare(replay_out[:used], expected)
            emit("graph_fp32_accuracy_budget", step=step, used=used,
                 kv_length=length, **budget)
            if not budget["finite"]:
                raise AssertionError("Nonfinite graph output/reference")
            if not budget["pass"]:
                fp32_budget_failures.append(dict(step=step, used=used, kv_length=length, **budget))
            output_diagnostics(replay_out[:used], expected, list(range(used)))
            p_expected = p_fp8_single_tile_reference(
                q_ref[:used], [logical_kv], [used], [length], reader.scaling, list(range(used))
            )
            if p_expected is not None:
                emit("graph_p_fp8_diagnostic", step=step, acceptance_gate=False,
                     native_vs_p_fp8=compare(replay_out[:used], p_expected),
                     p_fp8_vs_fp32=compare(p_expected, expected))
            require_check(
                "graph_lse",
                replay_lse.T[:used],
                expected_lse,
                relative_l2=0.003,
                atol=0.02,
                rtol=0.003,
            )
        # Inactive physical rows are not model outputs; native LSE may be +inf.
        active = reader.used > 0
        neutral_out = replay_out.masked_fill(~active[:, None, None], 0)
        neutral_lse = replay_lse.T.masked_fill(~active[:, None], -float("inf"))
        if used < capacity and (
            bool(neutral_out[used:].count_nonzero())
            or not bool(torch.isneginf(neutral_lse[used:]).all())
        ):
            raise AssertionError("Inactive graph rows were not neutralized")
        if not torch.equal(reader.state["storage"].cpu(), reader.state["expected"]):
            raise AssertionError(f"Graph replay {step} modified cache")
        completed_replays += 1
        emit(
            "graph_replay",
            **{"pass": True},
            scope="graph replay consistency only; FP32 accuracy budget reported separately",
            step=step,
            used=used,
            kv_length=length,
            page_table_reversed=bool(step % 2),
            invalid_rows_neutralized=True,
            inactive_raw_finite=bool(torch.isfinite(replay_out[used:]).all()),
        )
    emit("graph_contract_summary", completed_replays=completed_replays,
         replay_contract_passed=True,
         fp32_accuracy_budget_passed=not fp32_budget_failures,
         fp32_budget_failures=fp32_budget_failures,
         scope="one-card synthetic replay checks only; not VA/HCCL or model accuracy acceptance")
    if fp32_budget_failures:
        raise AssertionError(
            f"All {completed_replays} graph replays completed and consistency checks passed; "
            f"{len(fp32_budget_failures)} FP32 accuracy budgets failed (original thresholds unchanged)"
        )
    return {
        "replays": 6,
        "scope": "one-card QFM metadata capture/replay; not VA/HCCL graphs",
    }


def dcp_suite(native):
    """Two logical shards on ONE card; validates math, never claims HCCL."""
    for q_len, k_len in ((1, 1), (9, 129), (17, 257)):
        generator = torch.Generator().manual_seed(901)
        kv = torch.randn(k_len, WIDTH, generator=generator).to(torch.bfloat16)
        q = torch.randn(q_len, 12, WIDTH, generator=generator).to(
            "npu", dtype=torch.bfloat16
        )
        outputs, lses = [], []
        for rank in (0, 1):
            positions = torch.arange(k_len)
            local = kv[(positions // 16) % 2 == rank]
            state = nz_storage([local])
            plan = plan_qfm_batch(
                [q_len], [k_len], dcp_size=2, dcp_rank=rank, interleave=16
            )
            reader = Reader(native, q, state, plan)
            out, lse = reader.run()
            torch.npu.synchronize()
            valid = reader.lengths > 0
            outputs.append(out.float().masked_fill(~valid[:, None, None], 0))
            lses.append(lse.T.masked_fill(~valid[:, None], -float("inf")))
        stacked_lse = torch.stack(lses)
        weights = stacked_lse.softmax(0)
        merged = (torch.stack(outputs) * weights[..., None]).sum(0)
        q8, qs = quantize_query(q)
        q_ref = q8.cpu().float() * qs.cpu()[..., None]
        kv_ref = (kv.float() / 0.03125).clamp(-448, 448).to(
            torch.float8_e4m3fn
        ).float() * 0.03125
        expected, _ = reference(q_ref, [kv_ref], [q_len], [k_len], reader.scaling)
        require_check("simulated_dcp_lse_merge", merged, expected)
        emit(
            "case_complete",
            **{"pass": True},
            queries=q_len,
            kv_length=k_len,
            scope="simulated two shards on one NPU, no collectives",
        )
    return {
        "cases": 3,
        "scope": "local-prefix/empty-shard/LSE math ONLY; not distributed DCP",
    }


def measure(fn, iterations):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    device_ms, wall_ms = [], []
    for _ in range(iterations):
        start, end = (
            torch.npu.Event(enable_timing=True),
            torch.npu.Event(enable_timing=True),
        )
        before = time.perf_counter()
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - before) * 1000)
        device_ms.append(start.elapsed_time(end))
        del result
    return {
        "device_median_ms": statistics.median(device_ms),
        "wall_median_ms": statistics.median(wall_ms),
        "device_samples_ms": device_ms,
        "iterations": iterations,
    }


def fia_baseline(q_cpu, kv_cpu, scaling):
    """BF16 MLA FIA baseline: one full-length call, NO FP8 dequant timed.

    This allocates separate TEST inputs, never an extra production cache.
    Explicit power-of-two head padding is reported, not concealed in ratios.
    """
    real_heads = q_cpu.shape[1]
    native_heads = 1 << (real_heads - 1).bit_length()
    q = torch.zeros(
        q_cpu.shape[0], native_heads, WIDTH, dtype=torch.bfloat16, device="npu"
    )
    q[:, :real_heads].copy_(q_cpu.to("npu"))
    qn, qr = q[..., :VALUE].contiguous(), q[..., VALUE:].contiguous()
    kn = kv_cpu[:, :VALUE].unsqueeze(1).contiguous().to("npu")
    kr = kv_cpu[:, VALUE:].unsqueeze(1).contiguous().to("npu")
    mask = torch.ones((2048, 2048), dtype=torch.bool, device="npu").triu(1)

    def run():
        result, _ = torch_npu.npu_fused_infer_attention_score(
            qn,
            kn,
            kn,
            query_rope=qr,
            key_rope=kr,
            num_heads=native_heads,
            num_key_value_heads=1,
            input_layout="TND",
            actual_seq_lengths=[q.shape[0]],
            actual_seq_lengths_kv=[kn.shape[0]],
            atten_mask=mask,
            sparse_mode=3,
            scale=scaling,
            antiquant_mode=0,
            antiquant_scale=None,
            block_table=None,
            block_size=0,
            softmax_lse_flag=False,
        )
        return result[:, :real_heads]

    return run, native_heads


def performance_suite(native, args):
    q_len, k_len = args.queries, args.kv_length
    reader, q_cpu, kv = make_case(native, [q_len], [k_len], args.heads)
    check_reader(reader, [q_len], [k_len], sample_limit=8)
    reader.return_lse = False  # Fair no-LSE ordinary-reader timing for both paths.
    metadata = reader.metadata()
    torch.npu.synchronize()
    times = {
        "qfm_kernel_only": measure(lambda: reader.attention(metadata), args.iterations),
        "qfm_metadata_only": measure(reader.metadata, args.iterations),
        "qfm_quantize_q_tables_metadata_attention": measure(
            reader.prepared_read, args.iterations
        ),
    }
    positions = torch.arange(k_len - q_len, k_len, device="npu")
    slots = (
        reader.state["table"][0].long()[positions // BLOCK] * BLOCK + positions % BLOCK
    )
    current_kv = kv[0][-q_len:].to("npu")
    times["current_kv_quantize_indices_scatter"] = measure(
        lambda: reader.state["scatter"](current_kv, slots), args.iterations
    )
    baseline, baseline_heads = fia_baseline(q_cpu, kv[0], reader.scaling)
    base_out = baseline()
    torch.npu.synchronize()
    samples = selected_rows(q_len, 8)
    expected_bf16, _ = reference(
        q_cpu.float(), [kv[0].float()], [q_len], [k_len], reader.scaling, samples
    )
    require_check("bf16_fia_reference", base_out.cpu()[samples], expected_bf16)
    times["bf16_fia_kernel_only"] = measure(baseline, args.iterations)
    quantized, _ = reader.run()
    torch.npu.synchronize()
    # Report numerical change separately, do NOT claim model-quality acceptance.
    quantization_delta = compare(
        quantized.cpu()[samples],
        base_out.cpu()[samples],
        relative_l2=1.0,
        atol=1.0,
        rtol=1.0,
    )
    quantization_delta.pop("pass")
    report = {
        "queries": q_len,
        "kv_length": k_len,
        "heads": args.heads,
        "fia_padded_heads": baseline_heads,
        "qfm_virtual_sequences": len(reader.plan.owners),
        "qfm_native_attention_calls": 1,
        "timing_returns_lse": False,
        "qfm_table_bytes": reader.table.numel() * 4,
        "qfm_schedule_bytes": metadata.numel() * 4,
        "old_dense_token_index_bytes": q_len * k_len * 4,
        "qfm_cache_payload_bytes_per_token": WIDTH,
        "times": times,
        "sampled_fp8_vs_bf16_delta": quantization_delta,
        "scope": "synthetic single-layer MLA reader; no end-to-end performance claim",
    }
    emit("performance", **report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument(
        "--suite",
        choices=("inspect", "preflight", "small", "graph", "dcp", "performance"),
        default="small",
    )
    parser.add_argument("--queries", type=int, default=8192)
    parser.add_argument("--kv-length", type=int, default=131072)
    parser.add_argument("--heads", type=int, choices=HEADS, default=12)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args(argv)
    if not 1 <= args.queries <= args.kv_length or args.iterations < 1 or args.card < 0:
        parser.error("Need 1 <= queries <= kv-length, iterations >= 1, card >= 0")
    if args.suite == "inspect":
        result = inspect_runtime()
        emit(
            "suite_complete",
            suite="inspect",
            **{"pass": True},
            result=result,
            scope="environment inventory only; no NPU computation validated",
        )
        return 0
    native = load_runtime(args.card)
    torch.npu.reset_peak_memory_stats()
    if args.suite == "preflight":
        result = {"scope": "imports/schema only; no native computation validated"}
    elif args.suite == "small":
        result = small_suite(native)
    elif args.suite == "graph":
        result = graph_suite(native)
    elif args.suite == "dcp":
        result = dcp_suite(native)
    else:
        result = performance_suite(native, args)
    torch.npu.synchronize()
    emit(
        "suite_complete",
        suite=args.suite,
        **{"pass": True},
        result=result,
        peak_allocated_mib=torch.npu.max_memory_allocated() / 2**20,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
