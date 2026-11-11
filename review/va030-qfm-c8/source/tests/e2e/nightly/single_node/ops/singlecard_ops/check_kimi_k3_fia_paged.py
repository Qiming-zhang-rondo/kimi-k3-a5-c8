# SPDX-License-Identifier: Apache-2.0
"""Synthetic native paged BF16 FIA probe; no model or distributed acceptance.

The standalone bundle embeds the two exact framework modules. No installed VA,
QFM provider or DSL Prolog is imported. --plan does not import torch/torch_npu.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def memory_estimate(args):
    q = args.queries if args.benchmark else 33
    history = args.history if args.benchmark else 257
    heads = args.heads if args.benchmark else 48
    pages = max(4, math.ceil((history + q) / 128) + 1)
    components = {
        "accounted_fp8_and_bf16_pools_with_guards": pages * 128 * 576 * 3 + 1024,
        "query_and_two_outputs_lse": q * heads * (576 * 2 + 2 * (512 * 2 + 4)),
        "synthetic_kv_input_and_gather_cast_temporaries": (history + q) * 576 * 8,
        "causal_mask": 2048 * 2048,
        "merge_temporaries": q * heads * 512 * 4 * 4,
        "runtime_workspace_placeholder": 512 * 1024**2,
    }
    return {
        "estimated_peak_mib": sum(components.values()) / 1024**2,
        "components_bytes": components,
        "caveat": (
            "Approximate synthetic tensors plus a workspace placeholder; native workspace, "
            "allocator retention and other card users are unknown. Not a reservation or hard cap."
        ),
    }


def load_modules(directory):
    payload = globals().get("PAYLOAD")
    if payload is None:
        root = Path(__file__).resolve().parents[6] / "vllm_ascend/attention"
        sources = {"qfm": root / "kimi_k3_qfm.py", "paged": root / "kimi_k3_fia_paged.py"}
    else:
        sources = {}
        for name, entry in payload.items():
            data = entry["source"].encode()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Embedded source checksum failed: {name}")
            path = directory / (name + ".py")
            path.write_bytes(data)
            sources[name] = path
    modules = {}
    for name, path in sources.items():
        module_name = "vllm_ascend.attention.kimi_k3_qfm" if name == "qfm" else "k3_fia_paged_probe_adapter"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["qfm"], modules["paged"]


def require_close(torch, name, actual, expected, *, atol, rtol):
    actual, expected = actual.float(), expected.float()
    diff = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    bad = int((diff > atol + rtol * expected.abs()).sum())
    relative_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp_min(1e-12))
    result = {
        # Absolute tolerance alone can accept zero output for very long
        # contexts. Require the global numerical signal to match as well.
        "pass": finite and bad == 0 and relative_l2 <= 0.02,
        "finite": finite,
        "max_abs": float(diff.max()),
        "relative_l2": relative_l2,
        "relative_l2_limit": 0.02,
        "bad_elements": bad,
        "elements": actual.numel(),
        "atol": atol,
        "rtol": rtol,
    }
    emit("check", name=name, **result)
    if not result["pass"]:
        raise AssertionError(f"{name}: {result}")
    return result


def reference(torch, query, keys, history, rows):
    scores = torch.einsum("thd,kd->thk", query[rows].float(), keys.float()) / math.sqrt(192)
    hidden = torch.arange(keys.shape[0])[None, :] > (history + torch.tensor(rows))[:, None]
    scores.masked_fill_(hidden[:, None, :], -float("inf"))
    return torch.einsum("thk,kd->thd", scores.softmax(-1), keys[:, :512].float()), scores.logsumexp(-1)


def expected_write(torch, expected, rows, slots, offset, *, stride=73984, fp8=True, bf16=True):
    """Independent flat-byte layout oracle, without production view helpers."""
    valid = slots >= 0
    rows, slots = rows[valid], slots[valid]
    pages, tokens = slots // 128, slots % 128
    lanes = torch.arange(32)
    if fp8:
        encoded = rows.float().clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        for tile in range(18):
            address = offset + pages * stride + tile * 4096 + tokens * 32
            expected[0][address[:, None] + lanes] = encoded[:, tile * 32 : (tile + 1) * 32]
    if bf16:
        for component, dim, row_start in ((1, 512, 0), (2, 64, 512)):
            encoded = rows[:, row_start : row_start + dim].to(torch.bfloat16).contiguous().view(torch.uint8)
            for tile in range(dim // 16):
                address = offset + pages * 128 * dim * 2 + tile * 4096 + tokens * 32
                expected[component][address[:, None] + lanes] = encoded[:, tile * 32 : (tile + 1) * 32]


def make_case(torch, qfm, paged, heads, history, chunks, *, seed, quantized_history=False):
    offset, tail, stride = 256, 512, 73728 + 256
    total = history + sum(chunks)
    blocks = max(4, math.ceil(total / 128) + 1)
    backings = tuple(
        torch.full((offset + blocks * page_bytes + tail,), 0x5A, dtype=torch.uint8, device="npu")
        for page_bytes in (stride, 131072, 16384)
    )
    expected = [backing.cpu() for backing in backings]
    cache = qfm.cache_view(backings[0][offset : offset + blocks * stride], blocks, stride)
    cache._qfm_bf16_prefill = True
    cache._qfm_bf16_views = tuple(
        backing[offset : offset + blocks * dim * 128 * 2].view(torch.bfloat16).view(blocks, 1, dim // 16, 128, 16)
        for backing, dim in zip(backings[1:], (512, 64))
    )
    latent, raw = paged.sidecar_views(cache)
    assert latent.is_contiguous() and raw.is_contiguous()
    table_cpu = torch.arange(blocks - 1, -1, -1, dtype=torch.int32)
    table = table_cpu.npu()
    positions = torch.arange(total)
    slots_cpu = table_cpu[positions // 128].long() * 128 + positions % 128
    slots = slots_cpu.npu()
    torch.manual_seed(seed)
    keys_cpu = (torch.randn(total, 576) * 0.5).to(torch.bfloat16)
    if quantized_history and history:
        keys_cpu[:history] = keys_cpu[:history].to(torch.float8_e4m3fn).to(torch.bfloat16)
    keys = keys_cpu.npu()
    adapter = paged.PagedFIA()
    mask = torch.ones((2048, 2048), dtype=torch.bool, device="npu").triu_(1)
    scale = torch.ones(1, device="npu")

    def write(begin, end, *, padded=False):
        rows, live = keys[begin:end], slots[begin:end]
        rows_cpu, live_cpu = keys_cpu[begin:end], slots_cpu[begin:end]
        if padded:
            rows = torch.cat((rows, torch.full((1, 576), 31.0, dtype=torch.bfloat16, device="npu")))
            live = torch.cat((live, torch.full((1,), -1, dtype=torch.int64, device="npu")))
            rows_cpu = torch.cat((rows_cpu, torch.full((1, 576), 31.0, dtype=torch.bfloat16)))
            live_cpu = torch.cat((live_cpu, torch.tensor([-1])))
        qfm.write_cache(rows, live, cache, scale)
        before_fp8 = cache.view(torch.uint8).clone()
        batch = SimpleNamespace(has_prefill=True, slots=live, preprocess_segments=((0, len(live), True),))
        adapter.update(cache, batch, rows)
        torch.npu.synchronize()
        if not torch.equal(cache.view(torch.uint8), before_fp8):
            raise AssertionError("BF16 writer modified FP8 payload")
        expected_write(torch, expected, rows_cpu, live_cpu, offset, stride=stride)
        if not all(torch.equal(backing.cpu(), wanted) for backing, wanted in zip(backings, expected)):
            raise AssertionError("Native writer bytes, invalid-slot handling or guards differ from independent oracle")

    if history:
        write(0, history, padded=True)
    return SimpleNamespace(
        backings=backings,
        expected=expected,
        cache=cache,
        table=table,
        table_cpu=table_cpu,
        slots=slots,
        slots_cpu=slots_cpu,
        keys=keys,
        keys_cpu=keys_cpu,
        adapter=adapter,
        write=write,
        mask=mask,
        scale=scale,
        heads=heads,
        blocks=blocks,
        offset=offset,
        stride=stride,
    )


def run_case(torch, qfm, paged, heads, history, chunks, seed):
    emit("case_begin", heads=heads, history=history, chunks=chunks)
    case = make_case(torch, qfm, paged, heads, history, chunks, seed=seed)
    impl = SimpleNamespace(dcp_enabled=False, dcp_rank=0, scale=1 / math.sqrt(192))
    start = history
    for index, count in enumerate(chunks):
        case.write(start, start + count, padded=True)
        query_cpu = (torch.randn(count, heads, 576) * 0.5).to(torch.bfloat16)
        query = query_cpu.npu()
        out = torch.empty(count, heads, 512, dtype=torch.bfloat16, device="npu")
        lse = torch.empty(count, heads, 1, dtype=torch.float32, device="npu")
        batch = SimpleNamespace(fia_requests=((0, count, start, case.table),), fia_mask=case.mask)
        before = [backing.clone() for backing in case.backings]
        case.adapter.fill(impl, query, case.keys[start : start + count], case.cache, batch, out, lse)
        torch.npu.synchronize()
        if not all(torch.equal(old, backing) for old, backing in zip(before, case.backings)):
            raise AssertionError("Paged FIA reader modified cache/guards")
        rows = list(range(count))
        golden, golden_lse = reference(torch, query_cpu, case.keys_cpu[: start + count], start, rows)
        require_close(torch, "paged_fia_vs_fp32", out.cpu(), golden, atol=0.02, rtol=0.02)
        require_close(torch, "paged_lse_vs_fp32", lse.cpu()[..., 0], golden_lse, atol=0.005, rtol=0.001)
        emit("chunk_complete", index=index, queries=count, history=start, pass_=True)
        start += count
    # Reuse a complete 128-token prefix under a different physical suffix table.
    if history >= 128:
        apc_history, count = 128, 9
        changed = case.table.clone()
        # Replacing this suffix with a different physical page proves the read
        # takes the supplied live table; the shared prefix remains untouched.
        changed[1] = 0
        positions = torch.arange(apc_history, apc_history + count, device="npu")
        slots = changed[positions // 128].long() * 128 + positions % 128
        current_cpu = (torch.randn(count, 576) * 0.5).to(torch.bfloat16)
        current = current_cpu.npu()
        qfm.write_cache(current, slots, case.cache, case.scale)
        case.adapter.update(
            case.cache, SimpleNamespace(has_prefill=True, slots=slots, preprocess_segments=((0, count, True),)), current
        )
        q_cpu = (torch.randn(count, heads, 576) * 0.5).to(torch.bfloat16)
        output = torch.empty(count, heads, 512, dtype=torch.bfloat16, device="npu")
        lse = torch.empty(count, heads, 1, dtype=torch.float32, device="npu")
        case.adapter.fill(
            impl,
            q_cpu.npu(),
            current,
            case.cache,
            SimpleNamespace(fia_requests=((0, count, apc_history, changed),), fia_mask=case.mask),
            output,
            lse,
        )
        golden, golden_lse = reference(
            torch, q_cpu, torch.cat((case.keys_cpu[:128], current_cpu)), apc_history, list(range(count))
        )
        require_close(torch, "simulated_apc_live_page_table", output.cpu(), golden, atol=0.02, rtol=0.02)
        require_close(torch, "simulated_apc_lse", lse.cpu()[..., 0], golden_lse, atol=0.005, rtol=0.001)
    # Decode/Verify writes expose only FP8 values to the mirror. Check exact
    # encoded rounding and padding without claiming speculative acceptance.
    mirror_rows = case.keys[-2:]
    mirror_slots = torch.cat((case.slots[-2:], torch.tensor([-1], dtype=torch.int64, device="npu")))
    mirror_input = torch.cat((mirror_rows, torch.zeros(1, 576, dtype=torch.bfloat16, device="npu")))
    qfm.write_cache(mirror_input, mirror_slots, case.cache, case.scale)
    expected = [backing.cpu() for backing in case.backings]
    rounded = mirror_input.cpu().to(torch.float8_e4m3fn).to(torch.bfloat16)
    expected_write(torch, expected, rounded, mirror_slots.cpu(), case.offset, stride=case.stride, fp8=False)
    case.adapter.update(case.cache, SimpleNamespace(has_prefill=False, slots=mirror_slots), None)
    torch.npu.synchronize()
    if not all(torch.equal(backing.cpu(), wanted) for backing, wanted in zip(case.backings, expected)):
        raise AssertionError("Decode/Verify mirror wrote incorrect BF16 bytes or guards")
    emit(
        "case_complete",
        heads=heads,
        history=history,
        chunks=chunks,
        passed=True,
        writer_bytes_and_guards=True,
        reader_cache_unchanged=True,
        decode_slots_mirror=True,
        simulated_apc=history >= 128,
    )


def time_chain(torch, function, warmup, iterations):
    for _ in range(warmup):
        function()
    torch.npu.synchronize()
    live = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    wall = time.perf_counter()
    begin.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.npu.synchronize()
    return {
        "mean_npu_ms": begin.elapsed_time(end) / iterations,
        "mean_wall_ms": (time.perf_counter() - wall) * 1000 / iterations,
        "peak_allocated_mib": torch.npu.max_memory_allocated() / 1024**2,
        "incremental_peak_mib": (torch.npu.max_memory_allocated() - live) / 1024**2,
    }


def run_benchmark(torch, qfm, paged, args):
    case = make_case(torch, qfm, paged, args.heads, args.history, [args.queries], seed=971, quantized_history=True)
    case.write(args.history, args.history + args.queries)
    q_cpu = (torch.randn(args.queries, args.heads, 576) * 0.5).to(torch.bfloat16)
    q = q_cpu.npu()
    current = case.keys[args.history :]
    impl = SimpleNamespace(dcp_enabled=False, dcp_rank=0, scale=1 / math.sqrt(192), qfm_scale=case.scale)
    out = torch.empty(args.queries, args.heads, 512, dtype=torch.bfloat16, device="npu")
    lse = torch.empty(args.queries, args.heads, 1, dtype=torch.float32, device="npu")
    old_out, old_lse = torch.empty_like(out), torch.empty_like(lse)
    batch = SimpleNamespace(
        fia_requests=((0, args.queries, args.history, case.table),),
        fia_mask=case.mask,
        fia_history_plans=(qfm.build_fia_history_plan(case.table, args.history),),
    )
    write_batch = SimpleNamespace(
        has_prefill=True, slots=case.slots[args.history :], preprocess_segments=((0, args.queries, True),)
    )

    def old():
        qfm.fill_fia_prefill(impl, q, current, case.cache, batch, old_out, old_lse)

    def new():
        case.adapter.update(case.cache, write_batch, current)
        case.adapter.fill(impl, q, current, case.cache, batch, out, lse)

    old()
    new()
    rows = sorted({0, args.queries // 2, args.queries - 1})
    golden, golden_lse = reference(torch, q_cpu, case.keys_cpu, args.history, rows)
    require_close(torch, "benchmark_paged_vs_fp32", out.cpu()[rows], golden, atol=0.02, rtol=0.02)
    require_close(torch, "benchmark_gather_vs_fp32", old_out.cpu()[rows], golden, atol=0.02, rtol=0.02)
    require_close(torch, "benchmark_lse", lse.cpu()[rows, :, 0], golden_lse, atol=0.005, rtol=0.001)
    old_result = time_chain(torch, old, args.warmup, args.iterations)
    new_result = time_chain(torch, new, args.warmup, args.iterations)
    emit(
        "benchmark",
        queries=args.queries,
        history=args.history,
        heads=args.heads,
        gather_cast_history_current_fia_merge=old_result,
        bf16_current_write_and_single_paged_fia=new_result,
        old_over_new=old_result["mean_npu_ms"] / max(new_result["mean_npu_ms"], 1e-12),
        history_values="Both paths use the same FP8-rounded BF16 history; current stays BF16.",
        scope=(
            "Prefill history preparation/current BF16 mirror/Attention only; common FP8 writer, "
            "projections, V-up, gate, scheduler, HCCL and PD excluded. No TTFT claim."
        ),
    )


def worker(args):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    os.environ.pop("ASCEND_DEVICE_ID", None)
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    qfm, paged = load_modules(Path(args.directory))
    for index, (heads, history, chunks) in enumerate(
        ((12, 0, [1, 9, 33]), (12, 127, [9, 33]), (24, 129, [33, 9]), (48, 257, [9]))
    ):
        run_case(torch, qfm, paged, heads, history, chunks, seed=901 + index)
    if args.benchmark:
        run_benchmark(torch, qfm, paged, args)
    emit(
        "suite_complete",
        all_passed=True,
        cases=4,
        benchmark=args.benchmark,
        peak_allocated_mib=torch.npu.max_memory_allocated() / 1024**2,
        scope=(
            "Synthetic native BF16 writer + paged FIA; simulated APC page reuse and Decode slots only. "
            "NOT model, real APC scheduler, DSpark acceptance, DCP/HCCL, graph or PD acceptance."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--queries", type=int, default=8192)
    parser.add_argument("--history", type=int, default=32768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--output-root", default="/mnt")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--directory", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(args.queries, args.heads, args.iterations, args.timeout) < 1
        or min(args.history, args.warmup, args.card) < 0
    ):
        parser.error("queries/heads/iterations/timeout must be positive; history/warmup/card nonnegative")
    if args.plan:
        emit(
            "plan",
            physical_card=args.card,
            memory=memory_estimate(args),
            benchmark=args.benchmark,
            scope="CPU-only plan; no torch/NPU import or native validation",
        )
        return 0
    if args.worker:
        worker(args)
        return 0
    directory = Path(tempfile.mkdtemp(prefix="fia-paged-probe-", dir=args.output_root))
    logfile = directory / "native.log"
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--worker",
        "--directory",
        str(directory),
    ]
    emit("begin", directory=str(directory), log=str(logfile), memory=memory_estimate(args))
    timed_out = False
    with logfile.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait()
    lines = logfile.read_text(errors="replace").splitlines()
    complete = False
    for line in lines:
        if line.startswith('{"event":'):
            print(line, flush=True)
            record = json.loads(line)
            complete |= record.get("event") == "suite_complete" and record.get("all_passed") is True
    passed = returncode == 0 and not timed_out and complete
    if not passed:
        print("\n".join(lines[-60:]), flush=True)
    summary = {
        "all_passed": passed,
        "returncode": returncode,
        "timeout": timed_out,
        "log": str(logfile),
        "directory": str(directory),
        "benchmark": args.benchmark,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit("summary", **summary)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
