#!/usr/bin/env python3
"""Native two-pass sparse attention: correctness-gated, paired A5 timings.

The fixed-input single-layer experiment is not PD/MTP integration. Native C4
W0 remains the baseline; optional Triton implements only the small merge.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

from kv_window_common import (ARRAY_KEYS, attention_reference, load_case, save_case,
                              synthetic_case, validate_case, window_selection_stats)
from test_kv_hot_fused_a5 import (DeviceExecutionAborted, SUCCESS_STATUSES, acceptable,
    compare, error_record, evaluate, identity, integer_list, paired_benchmark,
    recipe_list, standalone_benchmark, success_status, timing_failed)

VERSION = "kv-native-twopass-a5-v1"


def choices_list(value, allowed):
    values = value.split(",")
    if not values or len(values) != len(set(values)) or any(v not in allowed for v in values):
        raise argparse.ArgumentTypeError("Expected distinct values from " + ",".join(allowed))
    return values


def run_case(data, args, directory, *, backend_factory=None, benchmark=None, progress=None):
    data = validate_case(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_case(directory / "inputs.npz", data)
    arrays = {"indices": data["indices"]}
    report = dict(label=data["label"], status="RUNNING", input_sha256=identity(data),
                  source_kind=data["source_kind"], selection_source=data["selection_source"],
                  shape={key: list(data[key].shape) for key in ARRAY_KEYS}, baselines={}, variants=[],
                  production_validation="NOT_VALIDATED", directory=str(directory))

    def flush():
        (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        np.savez_compressed(directory / "outputs.npz", **arrays)
        if progress is not None:
            progress(report)

    def finish(entry):
        print(f"  {entry['name']}: {entry['status']} L2={entry.get('vs_bf16_math', {}).get('relative_l2')}", flush=True)
        flush()
        if entry.get("failure_kind") == "DEVICE_EXECUTION_FAULT":
            report["stopped_after_device_fault"] = entry["name"]
            raise DeviceExecutionAborted(entry["name"])

    try:
        if backend_factory is None:
            from kv_native_twopass import NativeTwoPassBackend
            backend_factory = NativeTwoPassBackend
        if benchmark is None:
            from kv_window_backend import benchmark
        backend = backend_factory(data, device=args.device)
        report["environment"] = backend.environment
        print(f"[{data['label']}] full BF16 mathematical reference", flush=True)
        bf16_math = attention_reference(data)
        arrays["bf16_math"] = bf16_math
        bf16_output = None
        controls_ok = True
        for kind in ("bf16", "c8"):
            entry = {"name": kind, "status": "RUNNING", "role": "BASELINE"}
            report["baselines"][kind] = entry
            try:
                matching = bf16_math if kind == "bf16" else attention_reference(data, kv_c4=backend.baseline_decoded(kind))
                read = lambda kind=kind: backend.baseline(kind)
                pipeline = lambda kind=kind: backend.baseline_run(kind, write=True)
                metrics, output = evaluate(read, pipeline, matching, bf16_math, bf16_output,
                                           args, backend, arrays, kind)
                entry.update(metrics)
                if kind == "bf16":
                    bf16_output = output
                if entry["control_passed"]:
                    entry["timing_prepared_read"] = standalone_benchmark(read, args, benchmark)
                    entry["timing_write_read"] = standalone_benchmark(pipeline, args, benchmark)
                    entry["status"] = "TIMING_ERROR" if timing_failed(entry) else success_status(args)
                else:
                    entry["status"] = "CONTROL_FAILED"
            except Exception:
                entry.update(error_record())
            controls_ok &= entry.get("control_passed", False)
            finish(entry)

        for recipe in args.cold_recipes:
            name = f"{recipe}_w0"
            entry = {"name": name, "recipe": recipe, "window": 0, "mode": "native",
                     "role": "NATIVE_W0", "status": "RUNNING"}
            report["variants"].append(entry)
            native, native_output, cold = None, None, None
            try:
                native = backend.prepare(recipe, 0)
                entry["implementation"] = native.metadata
                cold = native.decoded_kv()
                cold_reference = attention_reference(data, kv_c4=cold)
                metrics, native_output = evaluate(native.call, lambda: native.run(write=True),
                    cold_reference, bf16_math, bf16_output, args, backend, arrays, name)
                entry.update(metrics)
                entry["control_passed"] &= controls_ok
                if entry["control_passed"]:
                    entry["timing_prepared_read"] = standalone_benchmark(native.call, args, benchmark)
                    entry["timing_write_read"] = standalone_benchmark(lambda: native.run(write=True), args, benchmark)
                    entry["status"] = "TIMING_ERROR" if timing_failed(entry) else success_status(args)
                else:
                    entry["status"] = "CONTROL_FAILED"
            except Exception:
                entry.update(error_record())
            finish(entry)
            native_entry = entry
            references = {}
            for window in sorted(w for w in args.windows if w > 0):
                for capacity in args.index_capacities:
                    for merge in args.merges:
                        split_passed = False
                        for mode in ("split_c4", "mixed"):
                            name = f"{recipe}_{mode}_w{window}_{capacity}_{merge}"
                            entry = dict(name=name, recipe=recipe, window=window, mode=mode,
                                         index_capacity=capacity, merge=merge, status="RUNNING",
                                         role="SPLIT_CONTROL" if mode == "split_c4" else "HOT_WINDOW",
                                         timing_reference=f"{recipe}_w0", production_performance="NOT_VALIDATED",
                                         selection=window_selection_stats(data, window))
                            report["variants"].append(entry)
                            try:
                                dependency = controls_ok and native_entry.get("control_passed", False)
                                if mode == "mixed":
                                    dependency &= split_passed
                                if not dependency:
                                    entry.update(status="CONTROL_FAILED", control_passed=False,
                                        control_failure_reason="BF16/C8, matching native W0, or same-configuration split C4 control failed")
                                else:
                                    runner = backend.prepare(recipe, window, mode=mode,
                                                             index_capacity=capacity, merge=merge)
                                    entry["implementation"] = runner.metadata
                                    key = (window, mode)
                                    if key not in references:
                                        references[key] = (cold_reference if mode == "split_c4" else
                                            attention_reference(data, kv_c4=cold, window=window,
                                                                hot_kv=runner.decoded_hot_kv()))
                                    read = runner.call
                                    pipeline = lambda runner=runner: runner.run(write=True, dynamic=True)
                                    metrics, output = evaluate(read, pipeline, references[key], bf16_math,
                                        bf16_output, args, backend, arrays, name)
                                    entry.update(metrics)
                                    dynamic = backend.host(runner.run(write=False, dynamic=True))
                                    arrays[name + "_dynamic"] = dynamic
                                    entry["dynamic_vs_prepared"] = compare(dynamic, output)
                                    entry["control_passed"] &= acceptable(entry["dynamic_vs_prepared"], args.pipeline_l2)
                                    if mode == "split_c4":
                                        entry["split_vs_native_w0"] = compare(output, native_output)
                                        entry["control_passed"] &= acceptable(entry["split_vs_native_w0"], args.control_l2)
                                    if args.diagnose:
                                        entry["zero_query_statistics"] = runner.zero_query_diagnostics()
                                        entry["control_passed"] &= bool(entry["zero_query_statistics"]["passed"])
                                    if entry["control_passed"]:
                                        entry["timing_prepared_read"] = paired_benchmark(native.call, read, args, benchmark)
                                        entry["timing_write_read"] = paired_benchmark(
                                            lambda: native.run(write=True), pipeline, args, benchmark)
                                        entry["status"] = "TIMING_ERROR" if timing_failed(entry) else success_status(args)
                                        if args.stages and args.timing != "none":
                                            entry["stage_timings"] = {stage: standalone_benchmark(call, args, benchmark)
                                                for stage, call in runner.component_calls().items()}
                                            if any(v.get("status") == "ERROR" for timings in entry["stage_timings"].values() for v in timings.values()):
                                                entry["status"] = "TIMING_ERROR"
                                        original = native_entry["vs_bf16_math"]["relative_l2"]
                                        entry["relative_l2_reduction_vs_same_recipe_w0"] = (
                                            1-entry["vs_bf16_math"]["relative_l2"]/original if original else None)
                                    else:
                                        entry["status"] = "CONTROL_FAILED"
                            except Exception:
                                entry.update(error_record())
                            if mode == "split_c4":
                                split_passed = entry.get("control_passed", False) and entry["status"] in SUCCESS_STATUSES
                            finish(entry)
        entries = list(report["baselines"].values()) + report["variants"]
        report["status"] = success_status(args) if all(e["status"] in SUCCESS_STATUSES for e in entries) else "FAILED"
        report["host_input_unchanged"] = identity(data) == report["input_sha256"]
        if not report["host_input_unchanged"]:
            report["status"] = "FAILED"
    except DeviceExecutionAborted:
        report["status"] = "DEVICE_ERROR"
    except Exception:
        report.update(error_record())
    finally:
        flush()
    return report


def write_reports(report, directory):
    directory = Path(directory)
    (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    modes = report["args"]["timing"].split(",")
    display = "graph" if "graph" in modes else modes[0]
    lines = [f"{VERSION} status={report['status']} timing={display}",
             "All times us; paired ratios use same-recipe native W0. MEASURED is not a performance pass.",
             "Full pipeline includes dynamic partition, current-token writes, native reads and merge."]
    rows = []
    def num(x, digits=2):
        return "NA" if x is None else f"{x:.{digits}f}"
    for case in report["cases"]:
        lines.append(f"case={case['label']}")
        lines.append("name | status | prepared read | full pipeline | paired W0 pipeline | ratio [min,max] | observation | L2 | matching L2")
        for entry in list(case["baselines"].values()) + case["variants"]:
            for timing_mode in modes:
                read = entry.get("timing_prepared_read", {}).get(timing_mode, {}).get("event_us", {})
                item = entry.get("timing_write_read", {}).get(timing_mode, {})
                pipe = item.get("event_us", {})
                row = dict(case=case["label"], name=entry["name"], status=entry["status"], role=entry["role"],
                    timing=timing_mode, read_us=read.get("candidate_median", read.get("median")),
                    pipeline_us=pipe.get("candidate_median", pipe.get("median")),
                    paired_w0_pipeline_us=pipe.get("baseline_median"), paired_ratio=pipe.get("median_paired_ratio"),
                    min_ratio=pipe.get("min_paired_ratio"), max_ratio=pipe.get("max_paired_ratio"),
                    observation=item.get("observation", "NOT_PAIRED"),
                    l2=entry.get("vs_bf16_math", {}).get("relative_l2"),
                    matching_l2=entry.get("kernel_vs_matching_reference", {}).get("relative_l2"),
                    production_performance="NOT_VALIDATED")
                rows.append(row)
                if timing_mode == display:
                    ratio = "NA" if row["paired_ratio"] is None else f"{num(row['paired_ratio'],3)} [{num(row['min_ratio'],3)},{num(row['max_ratio'],3)}]"
                    lines.append(" | ".join([entry["name"], entry["status"], num(row["read_us"]),
                        num(row["pipeline_us"]), num(row["paired_w0_pipeline_us"]), ratio, row["observation"],
                        num(row["l2"],6), num(row["matching_l2"],6)]))
            if entry.get("traceback"):
                lines.append("  error: " + entry["traceback"].strip().splitlines()[-1][:300])
            if entry.get("stage_timings"):
                times = [f"{name}={num(value.get(display, {}).get('event_us', {}).get('median'))}"
                         for name, value in entry["stage_timings"].items()]
                lines.append("  standalone stages (do not add/subtract): " + ", ".join(times))
    if rows:
        with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    text = "\n".join(lines) + "\n"
    (directory / "brief.txt").write_text(text, encoding="utf-8")
    return text


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--input", type=Path)
    p.add_argument("--keys", type=integer_list, default=[8192])
    p.add_argument("--queries", type=integer_list, default=[4])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--rope-dim", type=int, choices=(0,64), default=64)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--page", type=int, default=128)
    p.add_argument("--windows", type=lambda s: integer_list(s,0), default=[0,128])
    p.add_argument("--cold-recipes", type=recipe_list, default=["baseline", "alg2_max7"])
    p.add_argument("--index-capacities", type=lambda s: choices_list(s,("full","bounded")), default=["full","bounded"])
    p.add_argument("--merges", type=lambda s: choices_list(s,("torch","triton")), default=["torch","triton"])
    p.add_argument("--device", default="npu:0")
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--timing", choices=("none","eager","graph","eager,graph"), default="graph")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--control-l2", type=float, default=.01)
    p.add_argument("--pipeline-l2", type=float, default=.001)
    p.add_argument("--diagnose", action="store_true", help="Gate timing on zero-query per-subset softmax counts")
    p.add_argument("--stages", action="store_true", help="Extra standalone component timings; not additive")
    p.add_argument("--output", type=Path)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.self_test:
        from test_kv_hot_fused_a5 import self_test
        from kv_native_twopass import partition_numpy
        self_test()
        parts = partition_numpy(np.array([[[8,0,9,-1]], [[0,1,-1,-1]]], np.int32),
                                np.array([9,9]), 2, 4, "bounded")
        assert parts["old_counts"].tolist() == [1,2]
        assert parts["recent_counts"].tolist() == [2,0]
        assert parts["recent_ring"][0,0].tolist() == [0,1]
        print("Native two-pass CPU setup self-test only; no A5 execution.")
        return 0
    if 0 not in args.windows:
        p.error("--windows must include 0 for native C4 baseline")
    if min(args.batch,args.heads,args.topk,args.iterations,args.repeats) < 1 or args.warmup < 0:
        p.error("Positive batch/heads/TopK/iterations/repeats and nonnegative warmup required")
    if not all(np.isfinite(x) and x > 0 for x in (args.control_l2,args.pipeline_l2)):
        p.error("Correctness thresholds must be finite and positive")
    directory = (args.output or Path("kv_native_twopass_" + time.strftime("%Y%m%d_%H%M%S"))).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(version=VERSION, args={k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  status="RUNNING", cases=[], production_validation="NOT_VALIDATED")
    def persist(current=None):
        if current is not None and (not report["cases"] or report["cases"][-1] is not current):
            report["cases"].append(current)
        write_reports(report, directory)
    try:
        cases = [load_case(args.input)] if args.input else (
            synthetic_case(batch=args.batch,queries=q,keys=k,heads=args.heads,dim=512,
                           rope_dim=args.rope_dim,topk=args.topk,page=args.page,seed=args.seed)
            for k in args.keys for q in args.queries)
        for index,data in enumerate(cases):
            current = run_case(data,args,directory/f"case_{index:03d}",progress=persist)
            if current["status"] == "DEVICE_ERROR":
                report["stopped_after_device_fault"] = current["label"]
                break
        report["status"] = ("DEVICE_ERROR" if report.get("stopped_after_device_fault") else
            success_status(args) if report["cases"] and all(c["status"] in SUCCESS_STATUSES for c in report["cases"]) else "FAILED")
    except Exception:
        report.update(error_record())
    finally:
        persist()
    print(write_reports(report,directory),flush=True)
    print(f"Results: {directory / 'results.json'}\nShort summary: {directory / 'brief.txt'}",flush=True)
    return 0 if report["status"] in SUCCESS_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())
