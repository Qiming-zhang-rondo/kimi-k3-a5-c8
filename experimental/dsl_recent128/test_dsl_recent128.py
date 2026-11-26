#!/usr/bin/env python3
"""One-shape FP4/BF16-scale + FP8/BF16-scale recent-window experiment.

Prepared attention only. This does not change the model, existing C4 cache,
toolkit, vendor environment, or native operator installation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

import numpy as np

from dsl_reference import (attention_reference, bf16_round, metrics,
                           mixed_reference, quantize_rows)

VERSION = "dsl-fp4-fp8-recent128-v1"
UPSTREAM_OPS = "477e8d2cf7673cbe2a6ab1b8f3cd6f635ee0e637"
UPSTREAM_DSL = "c36d3a2e8a60710cf49d1d09b0b126241dcb6530"
SCOPE = "one prepared attention call; excludes cache writes, partition and metadata"


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def module_identity(module):
    path = Path(module.__file__).resolve()
    return dict(module=module.__name__, path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def versions():
    names = ("numpy", "torch", "torch-npu", "cannbot-dsl", "cannbot-arena-net",
             "cannbot-arena-net-ops", "cann-ops-transformer", "cann-ops-transformer-dsl")
    result = dict(python=sys.version, platform=platform.platform(),
                  ascend_home_path=os.environ.get("ASCEND_HOME_PATH"))
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def load_dsl(root=None):
    if root:
        root = Path(root).expanduser().resolve()
        if not (root / "ops").is_dir():
            raise ValueError("--dsl-root must contain the installed ops/ package")
        sys.path.insert(0, str(root))
    # Actually import both modules; find_spec does not initialize a package.
    attention_module = importlib.import_module("ops.mixed_quant_sparse_flash_mla")
    metadata_module = importlib.import_module("ops.mixed_quant_sparse_flash_mla_metadata")
    if root:
        for module in (attention_module, metadata_module):
            if not Path(module.__file__).resolve().is_relative_to(root / "ops"):
                raise RuntimeError("An already imported ops package shadows --dsl-root: " + str(module.__file__))
    attention = attention_module.mixed_quant_sparse_flash_mla
    metadata = metadata_module.mixed_quant_sparse_flash_mla_metadata
    identity = dict(attention=module_identity(attention_module),
                    metadata=module_identity(metadata_module))
    for name, fn in (("attention", attention), ("metadata", metadata)):
        identity[name]["signature"] = str(inspect.signature(fn))
    return attention, metadata, identity


def make_inputs(args):
    rng = np.random.default_rng(args.seed)
    q = bf16_round(rng.standard_normal((args.queries, 64, 512)).astype(np.float32))
    kv = bf16_round(rng.standard_normal((args.keys, 512)).astype(np.float32))
    positions = np.arange(args.keys - args.queries, args.keys, dtype=np.int32)
    indices = np.empty((args.queries, args.topk), dtype=np.int32)
    for t, pos in enumerate(positions):
        if args.selection == "uniform":
            selected = rng.choice(int(pos) + 1, args.topk, replace=False)
        else:
            # Explicit stress case, not a model-derived TopK: include every
            # recent token to exercise all 128 hot slots.
            hot = np.arange(max(0, int(pos) - args.window + 1), int(pos) + 1)
            cold = rng.choice(int(hot[0]), args.topk - len(hot), replace=False)
            selected = np.concatenate((cold, hot))
        indices[t] = np.sort(selected)
    return q, kv, positions, indices


def partition(indices, positions, window, ring_capacity, cold_capacity, hot_capacity):
    """CPU preparation; prove disjoint union and ring slot provenance."""
    indices = np.asarray(indices)
    positions = np.asarray(positions)
    if (indices.ndim != 2 or positions.shape != (len(indices),)
            or not np.issubdtype(indices.dtype, np.integer)
            or not np.issubdtype(positions.dtype, np.integer)):
        raise ValueError("indices/positions must be integer [T,K]/[T]")
    if window < 0 or ring_capacity < max(1, window):
        raise ValueError("invalid window or ring capacity")
    hot = np.full((len(indices), 1, hot_capacity), -1, np.int32)
    cold = np.full((len(indices), 1, cold_capacity), -1, np.int32)
    hot_lengths = np.zeros((len(indices), 1), np.int32)
    cold_lengths = np.zeros_like(hot_lengths)
    for t, (row, pos) in enumerate(zip(indices, positions)):
        if np.any(row < 0) or np.any(row > pos) or len(np.unique(row)) != len(row):
            raise ValueError("selection must contain unique visible original indices")
        mask = (row >= int(pos) - window + 1) if window else np.zeros(len(row), bool)
        h, c = row[mask], row[~mask]
        if len(h) > hot_capacity or len(c) > cold_capacity:
            raise ValueError("partition exceeds native index capacity")
        if len(np.unique(h % ring_capacity)) != len(h):
            raise ValueError("ring aliases two active hot tokens")
        hot[t, 0, :len(h)] = h % ring_capacity
        cold[t, 0, :len(c)] = c
        hot_lengths[t, 0], cold_lengths[t, 0] = len(h), len(c)
        if not np.array_equal(np.sort(np.concatenate((h, c))), np.sort(row)):
            raise AssertionError("partition changed the original TopK")
    return dict(ori_sparse_indices=hot, cmp_sparse_indices=cold,
                ori_topk_length=hot_lengths, cmp_topk_length=cold_lengths)


def prepare_host(args):
    q, kv, positions, indices = make_inputs(args)
    cold_bytes, cold_decoded = quantize_rows(kv, "fp4")
    hot_first = max(0, int(positions[0]) - args.window + 1)
    hot_bytes, hot_decoded = quantize_rows(kv[hot_first:], "fp8")
    capacity = math.ceil((args.window + args.queries - 1) / args.page) * args.page
    ring = np.zeros((capacity, 544), np.uint8)
    tags = np.full(capacity, -1, np.int32)
    hot_global = kv.copy()
    hot_global[hot_first:] = hot_decoded
    logical = np.arange(hot_first, args.keys)
    ring[logical % capacity], tags[logical % capacity] = hot_bytes, logical
    arms = {}
    for name, window in (("fp4_w0", 0), (f"fp4_fp8_w{args.window}", args.window)):
        data = partition(indices, positions, window, capacity, args.topk, args.window)
        for t, pos in enumerate(positions):
            selected = indices[t][indices[t] >= int(pos) - window + 1] if window else []
            if len(selected) and not np.array_equal(tags[np.asarray(selected) % capacity], selected):
                raise AssertionError("ring tags do not match original hot tokens")
        arms[name] = data
    return dict(q=q, kv=kv, positions=positions, indices=indices,
                cold_bytes=cold_bytes, cold_decoded=cold_decoded,
                hot_ring_bytes=ring, hot_global=hot_global, tags=tags,
                ring_capacity=capacity, arms=arms)


def gate(actual, expected, tolerance):
    result = metrics(actual, expected)
    result["passed"] = bool(result["finite"] and result["relative_l2"] <= tolerance
                            and result["worst_row_relative_l2"] <= tolerance)
    result["tolerance_global_and_worst_row"] = tolerance
    return result


class Call:
    def __init__(self, torch, attention, metadata, args, host, partition_data, q=None):
        self.torch = torch
        self.attention = attention
        dev = args.device
        def device(array, dtype=None):
            tensor = torch.from_numpy(np.ascontiguousarray(array)).to(dev)
            return tensor.to(dtype) if dtype is not None else tensor
        self.q = device(host["q"] if q is None else q, torch.bfloat16)
        cold = host["cold_bytes"]
        cold_storage = np.zeros((math.ceil(args.keys / args.page) * args.page, 320), np.uint8)
        cold_storage[:len(cold)] = cold
        self.kw = dict(
            ori_kv=device(host["hot_ring_bytes"].reshape(-1, args.page, 1, 544)),
            cmp_kv=device(cold_storage.reshape(-1, args.page, 1, 320)),
            ori_block_table=device(np.arange(host["ring_capacity"] // args.page,
                                            dtype=np.int32)[None, :]),
            cmp_block_table=device(np.arange(len(cold_storage) // args.page,
                                            dtype=np.int32)[None, :]),
            cu_seqlens_q=device(np.array([0, args.queries], np.int32)),
            sinks=device(np.full(64, args.sink, np.float32)),
            quant_mode=1, softmax_scale=1.0 / math.sqrt(512),
            layout_q="TND", layout_kv="PA_BBND", return_softmax_lse=False)
        self.kw.update({k: device(v) for k, v in partition_data.items()})
        self.kw["metadata"] = metadata(
            self.kw["ori_topk_length"], self.kw["cmp_topk_length"],
            cu_seqlens_q=self.kw["cu_seqlens_q"],
            num_heads_q=64, num_heads_kv=1, head_dim=512, quant_mode=1,
            layout_q="TND", layout_kv="PA_BBND", has_ori_kv=True, has_cmp_kv=True)
        self.out = torch.empty_like(self.q)
        self.lse = torch.empty((0,), dtype=torch.float32, device=dev)
        self.kw.update(out=self.out, lse=self.lse)
        self.graph = None

    def __call__(self):
        returned = self.attention(self.q, **self.kw)
        if returned is not None:
            raise RuntimeError("Expected direct DSL write-to-out API returning None")

    def array(self):
        return self.out.detach().float().cpu().numpy().copy()


def benchmark(torch, call, iterations):
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    started = time.perf_counter_ns()
    begin.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    elapsed = (time.perf_counter_ns() - started) / 1000 / iterations
    return dict(event_us=float(begin.elapsed_time(end) * 1000 / iterations),
                wall_us=float(elapsed))


def prepare_replay(torch, call, expected, args):
    for _ in range(args.warmup):
        call()
    torch.npu.synchronize()
    if args.timing == "eager":
        return call, dict(kind="eager", calls_per_graph=None)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call()
    torch.npu.synchronize()
    call.out.fill_(float("nan"))
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    captured = gate(call.array(), expected, args.tolerance)
    if not captured["passed"]:
        raise RuntimeError("Graph replay failed poisoned-output mathematical control: " + str(captured))
    call.graph = graph
    return graph.replay, dict(kind="graph", calls_per_graph=1,
                             iterations="replays outside capture", poison_control=captured)


def paired_timing(torch, baseline, candidate, args):
    rounds = []
    for r in range(args.repeats):
        order = ("baseline", "candidate") if r % 2 == 0 else ("candidate", "baseline")
        row = dict(round=r, order=list(order))
        for name in order:
            row[name] = benchmark(torch, baseline if name == "baseline" else candidate,
                                  args.iterations)
        row["event_us_ratio"] = row["candidate"]["event_us"] / row["baseline"]["event_us"]
        rounds.append(row)
    ratios = [r["event_us_ratio"] for r in rounds]
    return dict(scope=SCOPE, rounds=rounds,
                baseline_median=statistics.median(r["baseline"]["event_us"] for r in rounds),
                candidate_median=statistics.median(r["candidate"]["event_us"] for r in rounds),
                median_paired_ratio=statistics.median(ratios),
                min_paired_ratio=min(ratios), max_paired_ratio=max(ratios),
                candidate_faster_rounds=sum(r < 1 for r in ratios))


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--dsl-root", default=None)
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--queries", type=int, default=4)
    p.add_argument("--keys", type=int, default=8192)
    p.add_argument("--topk", type=int, default=512)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--page", type=int, default=128)
    p.add_argument("--selection", choices=("uniform", "covered_hot"), default="uniform")
    p.add_argument("--seed", type=int, default=20261126)
    p.add_argument("--sink", type=float, default=0.0)
    p.add_argument("--timing", choices=("none", "eager", "graph"), default="graph")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--tolerance", type=float, default=0.01)
    p.add_argument("--output", default="dsl-recent128-results")
    args = p.parse_args(argv)
    if (not args.device.startswith("npu:") or args.queries < 1 or args.keys < args.queries
            or not 1 <= args.window <= args.keys - args.queries + 1
            or not 1 <= args.topk <= args.keys - args.queries + 1
            or not 1 <= args.page <= 1024 or args.warmup < 1 or args.iterations < 1
            or args.repeats < 1 or not math.isfinite(args.sink)
            or not 0 < args.tolerance <= 0.02):
        p.error("invalid NPU, shape, loop count, finite sink, or tolerance")
    if args.selection == "covered_hot" and args.topk < args.window:
        p.error("covered_hot requires topk >= window")
    return args


def main(argv=None):
    args = arguments(argv)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(version=VERSION, args=vars(args), environment=versions(), status="RUNNING",
                  stage="host_preparation", scope=SCOPE,
                  reference_sources=dict(ops=UPSTREAM_OPS, dsl=UPSTREAM_DSL),
                  contract=dict(q="BF16 [T,64,512]=448 latent+64 RoPE",
                                cold="FP4 E2M1/group16/BF16 scale; 320B",
                                hot="FP8 E4M3FN/group32/BF16 scale; 544B",
                                all_coordinates_quantized=True, original_topk_unchanged=True,
                                sink_semantics="sink contributes exp(sink) to denominator; zero adds 1",
                                timing_comparable_to_yesterdays_D576=False))
    code = 1
    try:
        if args.self_test:
            host = prepare_host(args)
            sinks = np.full(64, args.sink, np.float32)
            ref, _ = attention_reference(host["q"], host["kv"], host["indices"],
                                         1 / math.sqrt(512), sinks)
            report.update(status="HOST_VALIDATED_ONLY", reference_finite=bool(np.isfinite(ref).all()),
                          hot_counts=host["arms"][f"fp4_fp8_w{args.window}"]["ori_topk_length"].ravel().tolist())
            code = 0
            return code
        report["stage"] = "dependency_import"
        # Never let a missing native profile silently compile in the timed run.
        os.environ["CANNBOTDSL_NATIVE_BINARY_MODE"] = "require"
        os.environ["CANNBOTDSL_NATIVE_BINARY_REPORT"] = str(output / "native_binary_report.json")
        report["native_binary_policy"] = dict(mode="require", report_path=os.environ["CANNBOTDSL_NATIVE_BINARY_REPORT"],
                                               jit_fallback_allowed=False)
        import torch
        import torch_npu  # noqa: F401
        attention, metadata, report["implementation"] = load_dsl(args.dsl_root)
        report["environment"] = versions()
        if args.probe_only:
            report.update(status="IMPORTED_ONLY", stage="dependency_import")
            code = 0
            return code
        report["stage"] = "host_preparation"
        host = prepare_host(args)
        sinks = np.full(64, args.sink, np.float32)
        scale = 1.0 / math.sqrt(512)
        bf16_ref, _ = attention_reference(host["q"], host["kv"], host["indices"], scale, sinks)
        cold_ref, _ = attention_reference(host["q"], host["cold_decoded"], host["indices"], scale, sinks)
        hot_ref, _ = mixed_reference(host["q"], host["cold_decoded"], host["hot_global"],
                                     host["indices"], host["positions"] - args.window + 1, scale, sinks)
        names = list(host["arms"])
        refs = {names[0]: cold_ref, names[1]: hot_ref}
        report["shape"] = dict(batch=1, queries=args.queries, keys=args.keys, heads=64,
                               dimension=512, topk=args.topk, window=args.window,
                               ring_capacity=host["ring_capacity"],
                               selection="synthetic_" + args.selection,
                               hot_counts=host["arms"][names[1]]["ori_topk_length"].ravel().tolist())
        report["stage"] = "npu_controls"
        torch.npu.set_device(args.device)
        report["environment"]["device"] = torch.npu.get_device_name(args.device)
        calls, report["arms"] = {}, {}
        for name in names:
            print(f"Validating {name} ...", flush=True)
            call = Call(torch, attention, metadata, args, host, host["arms"][name])
            calls[name] = call
            call.out.fill_(float("nan"))
            call()
            torch.npu.synchronize()
            actual = call.array()
            check = gate(actual, refs[name], args.tolerance)
            report["arms"][name] = dict(status="VALIDATED_ONLY" if check["passed"] else "CONTROL_FAILED",
                                        kernel_vs_matching_reference=check,
                                        output_vs_original_bf16=metrics(actual, bf16_ref),
                                        quantization_only_vs_original_bf16=metrics(refs[name], bf16_ref),
                                        metadata_shape=list(call.kw["metadata"].shape))
            if not check["passed"]:
                np.savez_compressed(output / (name + "_failure.npz"), actual=actual,
                                    expected=refs[name], indices=host["indices"])
                raise AssertionError(name + " failed full mathematical reference")
        # Single-side controls catch padding/normalization mistakes independent
        # of random hot coverage. Keep both KV tensors/structures present.
        report["controls"] = {}
        control_ids = np.stack([np.arange(int(pos) - args.window + 1, int(pos) + 1,
                                         dtype=np.int32) for pos in host["positions"]])
        for label, ids, window in (("zero_query_mixed", host["indices"], args.window),
                                   ("hot_only_with_zero_cold_length", control_ids, args.window)):
            cq = np.zeros_like(host["q"]) if label.startswith("zero") else host["q"]
            cp = partition(ids, host["positions"], window, host["ring_capacity"],
                           args.topk, args.window)
            expected, _ = mixed_reference(cq, host["cold_decoded"], host["hot_global"], ids,
                                            host["positions"] - window + 1, scale, sinks)
            control = Call(torch, attention, metadata, args, host, cp, q=cq)
            control.out.fill_(float("nan"))
            control()
            torch.npu.synchronize()
            checked = gate(control.array(), expected, args.tolerance)
            report["controls"][label] = checked
            if not checked["passed"]:
                raise AssertionError(label + " failed mathematical reference")
        if args.timing == "none":
            report.update(status="VALIDATED_ONLY", stage="complete")
            code = 0
            return code
        report["stage"] = "paired_prepared_attention"
        replay = {}
        for name in names:
            replay[name], capture = prepare_replay(torch, calls[name], refs[name], args)
            report["arms"][name]["capture"] = capture
        report["paired"] = paired_timing(torch, replay[names[0]], replay[names[1]], args)
        # Catch corruption/races introduced by capture/replay, outside timing.
        for name in names:
            checked = gate(calls[name].array(), refs[name], args.tolerance)
            report["arms"][name]["post_timing_control"] = checked
            if not checked["passed"]:
                raise AssertionError(name + " post-timing control failed")
            report["arms"][name]["status"] = "MEASURED"
        report.update(status="MEASURED", stage="complete")
        code = 0
        return code
    except Exception as exc:
        stage = report.get("stage")
        message = str(exc)
        fault = any(x in message.lower() for x in ("507015", "507035", "device fault", "aicore exception"))
        report.update(status="DEVICE_FAULT_STOPPED" if fault else
                      "DEPENDENCY_BLOCKED" if stage == "dependency_import" or "Native binary required" in message else
                      "CONTROL_FAILED" if isinstance(exc, AssertionError) else "ERROR",
                      error=message, traceback=traceback.format_exc())
        if fault:
            (output / "DEVICE_FAULT_STOPPED").write_text(message, encoding="utf-8")
        print(report["traceback"], file=sys.stderr, flush=True)
        if stage == "dependency_import" or "Native binary required" in message:
            print("Use EXECUTION_TASK.md to locate/build the matching arena-net wheel in an isolated path. "
                  "cannbot-dsl runtime alone does not contain this operator. No CANN upgrade/install was attempted.",
                  file=sys.stderr, flush=True)
        return code
    finally:
        native_path = output / "native_binary_report.json"
        if native_path.is_file():
            try:
                report["native_binary_report"] = json.loads(native_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                report["native_binary_report_error"] = str(exc)
        save_json(output / "results.json", report)
        lines = [f"status: {report['status']}", f"scope: {SCOPE}",
                 f"D512 / topk={args.topk} / recent={args.window} / selection={args.selection}"]
        if "shape" in report:
            lines.append("hot_selected_per_query: " + str(report["shape"]["hot_counts"]))
        for name, arm in report.get("arms", {}).items():
            def number(value):
                return f"{value:.7f}" if value is not None else "NONFINITE"
            lines.append(f"{name}: L2_vs_BF16={number(arm['output_vs_original_bf16']['relative_l2'])}; "
                         f"kernel_L2={number(arm['kernel_vs_matching_reference']['relative_l2'])}")
        if "paired" in report:
            paired = report["paired"]
            lines.append(f"W0={paired['baseline_median']:.3f}us; W{args.window}={paired['candidate_median']:.3f}us; "
                         f"paired_ratio={paired['median_paired_ratio']:.4f} "
                         f"[{paired['min_paired_ratio']:.4f},{paired['max_paired_ratio']:.4f}]")
        (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
        print(f"Results: {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
