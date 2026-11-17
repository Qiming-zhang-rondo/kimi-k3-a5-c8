#!/usr/bin/env python3
"""Measure main MLA KV precision windows with installed A5 cloud operators.

This is a single-attention experiment, not a GLM evaluation or serving benchmark.
TopK is fixed. Synthetic selections are explicitly labelled; --input replays
original model tensors saved before main-KV quantization. No packages are changed.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from kv_window_common import (attention_reference, compare_outputs, load_case,
                              save_case, synthetic_case, validate_case,
                              window_selection_stats)


VERSION = "main-kv-window-a5-v1"


def integers(value, minimum=0):
    result = [int(x) for x in value.split(",")]
    if not result or any(x < minimum for x in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(f"Expected distinct integers >= {minimum}")
    return result


def json_write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def check_metric(metric, limit):
    return bool(metric.get("finite") and metric["relative_l2"] <= limit
                and metric["max_row_relative_l2"] <= limit)


def time_call(fn, args, bench):
    result = {}
    for mode in args.timing.split(","):
        try:
            result[mode] = bench(fn, warmup=args.warmup, iterations=args.iterations,
                                 repeats=args.repeats, graph=mode == "graph")
        except Exception:
            result[mode] = {"status": "ERROR", "traceback": traceback.format_exc()}
    return result


def event_median(item, field, mode):
    return item.get(field, {}).get(mode, {}).get("event_us", {}).get("median")


def summarize(report, directory):
    """Expose accuracy/latency tradeoffs without requiring a JSON deep dive."""
    rows = []
    for name, item in report["baselines"].items():
        for mode in ("eager", "graph"):
            read = event_median(item, "timing_prepared_read", mode)
            pipeline = event_median(item, "timing_write_read", mode)
            rows.append(dict(path=name, timing=mode, relative_l2=item.get("vs_bf16_operator", {}).get("relative_l2"),
                             read_us=read, write_partition_read_us=pipeline,
                             pipeline_ratio_vs_c4=None, pipeline_ratio_vs_c8=None, extra_ring_bytes=0))
    for item in report["windows"]:
        item["pipeline_ratios"] = {}
        for mode in ("eager", "graph"):
            read = event_median(item, "timing_prepared_read", mode)
            pipeline = event_median(item, "timing_write_partition_read", mode)
            ratios = {}
            for kind in ("c4", "c8"):
                base = event_median(report["baselines"].get(kind, {}), "timing_write_read", mode)
                ratios[kind] = pipeline/base if pipeline is not None and base is not None and base > 0 else None
            item["pipeline_ratios"][mode] = ratios
            rows.append(dict(path=f"c4+{item['high_precision']}_w{item['window']}", timing=mode,
                             relative_l2=item.get("vs_bf16_operator", {}).get("relative_l2"), read_us=read,
                             write_partition_read_us=pipeline, pipeline_ratio_vs_c4=ratios["c4"],
                             pipeline_ratio_vs_c8=ratios["c8"],
                             extra_ring_bytes=item.get("memory", {}).get("additional_ring_bytes")))
    if rows:
        with (directory/"summary.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"  Summary: {directory}/summary.csv", flush=True)


def run_case(data, args, directory, backend_class=None, bench=None):
    if backend_class is None:
        from kv_window_backend import NPUWindowCase, benchmark
        backend_class, bench = NPUWindowCase, benchmark
    data = validate_case(data)
    directory.mkdir()
    report = dict(label=data["label"], source_kind=data["source_kind"],
                  selection_source=data["selection_source"], scale=data["scale"],
                  shapes={k: list(data[k].shape) for k in ("q_nope", "q_rope", "kv", "indices")},
                  q_lengths=data["q_lengths"].tolist(), k_lengths=data["k_lengths"].tolist(),
                  baselines={}, windows=[], status="RUNNING")
    if args.save_inputs:
        save_case(directory/"input.npz", data)
    arrays = {}
    try:
        npu = backend_class(data, device=args.device)
        report["environment"] = npu.environment
        original_ref = attention_reference(data)
        arrays["bf16_mathematical_reference"] = original_ref
        decoded = {kind: npu.decoded_kv(kind) for kind in ("c4", "c8")}
        references = {"bf16": original_ref, **{k: attention_reference(data, kv_c4=v) for k, v in decoded.items()}}
        outputs = {kind: npu.host(npu.call(kind)) for kind in ("bf16", "c4", "c8")}
        valid = True
        for kind in ("bf16", "c4", "c8"):
            matched = compare_outputs(outputs[kind], references[kind])
            updated = npu.host(npu.run(kind, write=True))
            item = dict(kernel_vs_matching_reference=matched,
                        write_vs_prepared=compare_outputs(updated, outputs[kind]),
                        vs_bf16_operator=compare_outputs(outputs[kind], outputs["bf16"]),
                        vs_original_mathematical_reference=compare_outputs(outputs[kind], original_ref))
            item["control_passed"] = (check_metric(matched, args.kernel_rel_l2_max)
                                      and check_metric(item["write_vs_prepared"], args.split_rel_l2_max))
            valid = valid and item["control_passed"]
            report["baselines"][kind] = item
            arrays[kind] = outputs[kind]
        # Keep failures attributable to a window; do not advertise speed for a
        # path whose matching reference or same-C4 split control has failed.
        for hot in args.hot_dtypes.split(","):
            for window in args.windows:
                name = f"{hot}_w{window}"
                print(f"  {name}: accuracy", flush=True)
                item = dict(window=window, high_precision=hot)
                try:
                    runner = npu.prepare(window, high_precision=hot)
                    mixed = npu.host(runner.run("mixed"))
                    split = npu.host(runner.run("split_c4"))
                    dynamic = npu.host(runner.run("mixed", dynamic=True, write=True))
                    reference = attention_reference(data, kv_c4=decoded["c4"], window=window,
                                                    hot_kv=decoded["c8"] if hot == "c8" else None)
                    item.update(kernel_vs_matching_reference=compare_outputs(mixed, reference),
                                split_c4_vs_unsplit_c4=compare_outputs(split, outputs["c4"]),
                                pipeline_vs_prepared=compare_outputs(dynamic, mixed),
                                vs_bf16_operator=compare_outputs(mixed, outputs["bf16"]),
                                vs_original_mathematical_reference=compare_outputs(mixed, original_ref),
                                selection=window_selection_stats(data, window),
                                partition=runner.partition_summary, memory=runner.memory)
                    item["control_passed"] = (
                        check_metric(item["kernel_vs_matching_reference"], args.kernel_rel_l2_max)
                        and check_metric(item["split_c4_vs_unsplit_c4"], args.split_rel_l2_max)
                        and check_metric(item["pipeline_vs_prepared"], args.split_rel_l2_max))
                    base_error = report["baselines"]["c4"]["vs_bf16_operator"].get("relative_l2")
                    new_error = item["vs_bf16_operator"].get("relative_l2")
                    item["relative_l2_reduction_vs_c4"] = (1-new_error/base_error
                        if base_error is not None and new_error is not None and base_error > 0 else None)
                    arrays[name], arrays[name+"_split_c4"], arrays[name+"_pipeline"] = mixed, split, dynamic
                    if args.timing != "none" and valid and item["control_passed"]:
                        item["timing_prepared_read"] = time_call(lambda: runner.run("mixed"), args, bench)
                        item["timing_write_partition_read"] = time_call(
                            lambda: runner.run("mixed", dynamic=True, write=True), args, bench)
                    elif args.timing != "none":
                        item["timing_skipped"] = "Matching-reference/control failure"
                    item["status"] = "MEASURED" if item["control_passed"] else "CONTROL_FAILED"
                except Exception:
                    item.update(status="ERROR", control_passed=False, traceback=traceback.format_exc())
                report["windows"].append(item)
                json_write(directory/"results.json", report)
                print(f"  {name}: {item['status']}; relative L2 vs BF16 = "
                      f"{item.get('vs_bf16_operator', {}).get('relative_l2', 'n/a')}", flush=True)
        if args.timing != "none" and valid:
            for kind, item in report["baselines"].items():
                item["timing_prepared_read"] = time_call(lambda k=kind: npu.call(k), args, bench)
                item["timing_write_read"] = time_call(lambda k=kind: npu.run(k, write=True), args, bench)
        timings = [v for b in report["baselines"].values() for k, v in b.items() if k.startswith("timing_")]
        timings += [v for w in report["windows"] for k, v in w.items() if k.startswith("timing_") and isinstance(v, dict)]
        timing_failed = any(isinstance(t, dict) and t.get("status") == "ERROR"
                            for group in timings for t in group.values())
        report["status"] = ("MEASURED" if valid and not timing_failed
                            and all(w["control_passed"] for w in report["windows"]) else "ERROR_OR_CONTROL_FAILED")
        report["operator_result_shapes"] = npu.raw_result_shapes
    except Exception:
        report.update(status="ERROR", traceback=traceback.format_exc())
    finally:
        if arrays:
            with (directory/"outputs.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
        summarize(report, directory)
        json_write(directory/"results.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", default=[], help="Original BF16 main-MLA snapshot NPZ; repeatable")
    p.add_argument("--keys", type=lambda s: integers(s, 1), default=[8192])
    p.add_argument("--queries", type=lambda s: integers(s, 1), default=[1])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--page", type=int, default=128)
    p.add_argument("--windows", type=integers, default=[0, 128, 256, 512])
    p.add_argument("--hot-dtypes", default="bf16,c8")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--timing", default="eager", help="eager, graph, eager,graph, or none")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--kernel-rel-l2-max", type=float, default=0.03,
                   help="Matching mathematical-reference guard, NOT a C4 quality target")
    p.add_argument("--split-rel-l2-max", type=float, default=0.01)
    p.add_argument("--save-inputs", action="store_true")
    p.add_argument("--self-test", action="store_true", help="CPU format/reference checks; no NPU claim")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    if (args.timing not in ("eager", "graph", "eager,graph", "none")
            or args.hot_dtypes not in ("bf16", "c8", "bf16,c8")):
        p.error("Unsupported --timing or --hot-dtypes")
    if min(args.batch, args.heads, args.topk, args.iterations, args.repeats) < 1 or args.warmup < 0:
        p.error("Counts must be positive (warmup may be zero)")
    if any(not math.isfinite(x) or x <= 0 for x in (args.kernel_rel_l2_max, args.split_rel_l2_max)):
        p.error("Control thresholds must be finite and positive")
    if args.self_test:
        data = synthetic_case(keys=65, queries=2, heads=2, dim=64, topk=32)
        original = attention_reference(data)
        changed = data["kv"]*.75
        np.testing.assert_array_equal(original, attention_reference(data, kv_c4=changed, window=65))
        assert compare_outputs(original, original)["relative_l2"] == 0
        assert window_selection_stats(data, 0)["recent_selected_max"] == 0
        print("PASS: CPU replay/reference checks. NPU operators/performance have NOT been tested.")
        return 0
    root = Path(args.output or ("kv-window-"+time.strftime("%Y%m%d_%H%M%S"))).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = dict(version=VERSION, args=vars(args), status="RUNNING", cases=[],
                  scope="single-attention main KV; fixed indices; no QLI execution or full-model accuracy/throughput",
                  timing_scope="steady-state repeated positions; excludes model projections/RMSNorm/RoPE/QLI, scheduler and distributed collectives",
                  quality_note="High-precision windows do not guarantee monotonic output error or task-score improvement",
                  scripts={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in Path(__file__).parent.glob("*kv_window*.py")})
    try:
        if args.input:
            cases = (load_case(path) for path in args.input)
        else:
            cases = (synthetic_case(batch=args.batch, keys=k, queries=q, heads=args.heads,
                                    topk=args.topk, page=args.page, seed=args.seed)
                     for k in args.keys for q in args.queries)
        for i, data in enumerate(cases):
            print(f"[{i}] {data['label']} ({data['source_kind']})", flush=True)
            item = run_case(data, args, root/f"case_{i:03d}")
            report["cases"].append(item)
            json_write(root/"results.json", report)
        report["status"] = "MEASURED" if report["cases"] and all(c["status"] == "MEASURED" for c in report["cases"]) else "ERROR_OR_CONTROL_FAILED"
    except Exception:
        report.update(status="ERROR", traceback=traceback.format_exc())
    finally:
        json_write(root/"results.json", report)
    print(f"{report['status']}: {root}/results.json", flush=True)
    return 0 if report["status"] == "MEASURED" else 1


if __name__ == "__main__":
    sys.exit(main())
