#!/usr/bin/env python3
"""Real A5 fused C4+C8 attention experiment; fixed TopK, no model mutation.

MEASURED means correctness controls passed and requested measurements completed.
It never means a production accuracy or performance requirement has passed.
"""
import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np

from kv_window_common import (ARRAY_KEYS, attention_reference, compare_outputs,
                              load_case, query_positions, save_case, synthetic_case,
                              validate_case, window_selection_stats)

VERSION = "kv-hot-fused-a5-v2"
RECIPES = ("baseline", "alg2_max7")


class DeviceExecutionAborted(RuntimeError):
    """A device execution fault makes later work in this process unreliable."""


def failure_kind(message):
    if any(x in message for x in ("507015", "ACL_ERROR_RT_AICORE_EXCEPTION",
                                  "The address for scalar to access GM is invalid")):
        return "DEVICE_EXECUTION_FAULT"
    if "ub overflow" in message.lower():
        return "TRITON_UB_OVERFLOW"
    if "EZ1009" in message or "operator prototype OpDef" in message:
        return "NATIVE_OPERATOR_CONTRACT"
    return "OTHER"
SUCCESS_STATUSES = ("MEASURED", "VALIDATED_ONLY")


def success_status(args):
    return "VALIDATED_ONLY" if args.timing == "none" else "MEASURED"


def integer_list(value, minimum=1):
    try:
        values = [int(x) for x in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not values or min(values) < minimum or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"Expected distinct integers >= {minimum}")
    return values


def recipe_list(value):
    values = value.split(",")
    if not values or len(set(values)) != len(values) or any(x not in RECIPES for x in values):
        raise argparse.ArgumentTypeError("Expected baseline,alg2_max7 or one of these recipes")
    return values


def identity(data):
    digest = hashlib.sha256()
    for key in ARRAY_KEYS:
        value = np.ascontiguousarray(data[key])
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    digest.update(str((data["scale"], data["page"])).encode())
    return digest.hexdigest()


def compare(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        raise ValueError(f"Output shape {actual.shape} != reference shape {reference.shape}")
    invalid = ~np.isfinite(actual)
    if not np.isfinite(reference).all():
        raise ValueError("Reference contains nonfinite values")
    metric = compare_outputs(actual, reference)
    metric["first_nonfinite_indices"] = np.argwhere(invalid)[:16].tolist()
    metric["nan_count"] = int(np.isnan(actual).sum())
    return metric


def acceptable(metric, limit):
    return bool(metric.get("finite") and metric["relative_l2"] <= limit
                and metric["max_row_relative_l2"] <= limit)


def error_record():
    message = traceback.format_exc()
    print(message, file=sys.stderr, flush=True)
    return {"status": "ERROR", "failure_kind": failure_kind(message), "traceback": message}


def paired_benchmark(baseline, candidate, args, benchmark):
    """Alternate AB / BA; capture/warmup outside device-event measurements.

    Event times are amortized repeated calls, not serving latency confidence
    intervals. Preserve every pair, the order, and both event and wall ratios.
    """
    result = {}
    modes = [] if args.timing == "none" else args.timing.split(",")
    for mode in modes:
        item = {"status": "MEASURED", "rounds": [], "production_performance": "NOT_VALIDATED",
                "scope": "alternating AB/BA repeated-call timing; each graph capture/warmup outside timed events"}
        result[mode] = item
        for round_index in range(args.repeats):
            order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
            pair = {"round": round_index, "order": list(order)}
            item["rounds"].append(pair)
            try:
                for name in order:
                    pair[name] = benchmark(baseline if name == "baseline" else candidate,
                                           warmup=args.warmup, iterations=args.iterations,
                                           repeats=1, graph=mode == "graph")
                for clock in ("event_us", "wall_us"):
                    original, updated = pair["baseline"][clock]["median"], pair["candidate"][clock]["median"]
                    if not (np.isfinite(original) and np.isfinite(updated) and original > 0 and updated > 0):
                        raise ValueError("Timing returned a nonpositive or nonfinite latency")
                    pair[clock + "_ratio"] = updated / original
            except Exception:
                pair.update(error_record())
                item["status"] = "ERROR"
                if pair["failure_kind"] == "DEVICE_EXECUTION_FAULT":
                    raise
                break
        complete = [r for r in item["rounds"] if "event_us_ratio" in r and "wall_us_ratio" in r]
        if complete:
            for clock in ("event_us", "wall_us"):
                ratios = [r[clock + "_ratio"] for r in complete]
                item[clock] = {"baseline_median": statistics.median(r["baseline"][clock]["median"] for r in complete),
                               "candidate_median": statistics.median(r["candidate"][clock]["median"] for r in complete),
                               "paired_ratios": ratios, "median_paired_ratio": statistics.median(ratios),
                               "min_paired_ratio": min(ratios), "max_paired_ratio": max(ratios)}
        if item["status"] == "MEASURED":
            ratio = item["event_us"]["median_paired_ratio"]
            item["observation"] = "OBSERVED_SLOWDOWN" if ratio > 1.0 else "NO_SLOWDOWN_OBSERVED"
        else:
            item["observation"] = "NOT_MEASURED"
    return result


def standalone_benchmark(call, args, benchmark):
    result = {}
    for mode in ([] if args.timing == "none" else args.timing.split(",")):
        try:
            result[mode] = benchmark(call, warmup=args.warmup, iterations=args.iterations,
                                     repeats=args.repeats, graph=mode == "graph")
        except Exception:
            result[mode] = error_record()
            if result[mode]["failure_kind"] == "DEVICE_EXECUTION_FAULT":
                raise
    return result


def timing_failed(entry):
    return any(value.get("status") == "ERROR" for field in ("timing_prepared_read", "timing_write_read")
               for value in entry.get(field, {}).values())


def evaluate(call, pipeline, matching, bf16_math, bf16_output, args, backend, arrays, name):
    output = backend.host(call())
    arrays[name + "_prepared"] = output
    arrays[name + "_matching_reference"] = matching
    pipeline_output = backend.host(pipeline())
    arrays[name + "_write_read"] = pipeline_output
    checks = {"kernel_vs_matching_reference": compare(output, matching),
              "pipeline_vs_prepared": compare(pipeline_output, output),
              "pipeline_vs_matching_reference": compare(pipeline_output, matching),
              "vs_bf16_math": compare(output, bf16_math)}
    if bf16_output is not None:
        checks["vs_bf16_operator"] = compare(output, bf16_output)
    checks["control_passed"] = (acceptable(checks["kernel_vs_matching_reference"], args.control_l2)
                                 and acceptable(checks["pipeline_vs_prepared"], args.pipeline_l2)
                                 and acceptable(checks["pipeline_vs_matching_reference"], args.control_l2))
    return checks, output


def run_case(data, args, directory, *, backend_factory=None, benchmark=None, progress=None):
    """Injectable boundary keeps CPU tests independent of torch and Triton."""
    data = validate_case(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_case(directory / "inputs.npz", data)
    arrays = {"indices": data["indices"], "query_positions": query_positions(data)}
    report = {"label": data["label"], "input_sha256": identity(data), "status": "RUNNING",
              "source_kind": data["source_kind"], "selection_source": data["selection_source"],
              "shape": {key: list(data[key].shape) for key in ARRAY_KEYS}, "baselines": {}, "variants": [],
              "production_validation": "NOT_VALIDATED", "directory": str(directory)}

    def flush():
        (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        np.savez_compressed(directory / "outputs.npz", **arrays)
        if progress is not None:
            progress(report)

    try:
        if backend_factory is None:
            from kv_hot_backend import HotWindowBackend
            backend_factory = HotWindowBackend
        if benchmark is None:
            from kv_window_backend import benchmark as benchmark
        backend = backend_factory(data, device=args.device, block_keys=args.block_keys, block_heads=args.block_heads)
        report["environment"] = backend.environment
        print(f"[{data['label']}] computing original BF16 mathematical reference", flush=True)
        bf16_math = attention_reference(data)
        arrays["bf16_mathematical_reference"] = bf16_math
        bf16_output = None
        controls_ok = True
        for kind in ("bf16", "c8"):
            entry = {"kind": kind, "status": "RUNNING"}
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
            print(f"  {kind}: {entry['status']}", flush=True)
            flush()
            if entry.get("failure_kind") == "DEVICE_EXECUTION_FAULT":
                report["stopped_after_device_fault"] = kind
                raise DeviceExecutionAborted(kind)
        for recipe in args.cold_recipes:
            native_runner, native_entry = None, None
            cold_decoded = None
            for window in sorted(args.windows):
                name = f"{recipe}_w{window}"
                entry = {"name": name, "recipe": recipe, "window": window, "status": "RUNNING",
                         "production_performance": "NOT_VALIDATED", "selection": window_selection_stats(data, window)}
                report["variants"].append(entry)
                try:
                    runner = backend.prepare(recipe, window)
                    entry["implementation"] = runner.metadata
                    if hasattr(runner, "plan"):
                        arrays[name + "_hot_positions"] = runner.plan["tags"]
                    if cold_decoded is None:
                        cold_decoded = runner.decoded_kv()
                    matching = attention_reference(data, kv_c4=cold_decoded, window=window,
                                                   hot_kv=runner.decoded_hot_kv() if window else None)
                    read = runner.call
                    pipeline = lambda runner=runner: runner.run(write=True)
                    metrics, output = evaluate(read, pipeline, matching, bf16_math, bf16_output,
                                               args, backend, arrays, name)
                    entry.update(metrics)
                    entry["control_passed"] &= controls_ok
                    if window == 0:
                        native_runner, native_entry = runner, entry
                    elif native_entry is None or not native_entry.get("control_passed"):
                        entry["control_passed"] = False
                        entry["control_failure_reason"] = "matching native C4 W0 controls unavailable or failed"
                    if not controls_ok:
                        entry["control_failure_reason"] = "native BF16/C8 reference controls failed"
                    if entry["control_passed"]:
                        if window:
                            entry["timing_prepared_read"] = paired_benchmark(native_runner.call, read, args, benchmark)
                            entry["timing_write_read"] = paired_benchmark(lambda: native_runner.run(write=True), pipeline, args, benchmark)
                            original = native_entry["vs_bf16_math"]["relative_l2"]
                            entry["relative_l2_reduction_vs_same_recipe_w0"] = (1-entry["vs_bf16_math"]["relative_l2"]/original) if original else None
                            entry["timing_reference"] = f"{recipe}_w0 (same C4 quantization recipe)"
                        else:
                            entry["timing_prepared_read"] = standalone_benchmark(read, args, benchmark)
                            entry["timing_write_read"] = standalone_benchmark(pipeline, args, benchmark)
                        entry["status"] = "TIMING_ERROR" if timing_failed(entry) else success_status(args)
                    else:
                        entry["status"] = "CONTROL_FAILED"
                except Exception:
                    entry.update(error_record())
                metric = entry.get("vs_bf16_math", {}).get("relative_l2")
                print(f"  {name}: {entry['status']} L2_vs_bf16_math={metric}", flush=True)
                flush()
                if entry.get("failure_kind") == "DEVICE_EXECUTION_FAULT":
                    report["stopped_after_device_fault"] = name
                    raise DeviceExecutionAborted(name)
        entries = list(report["baselines"].values()) + report["variants"]
        report["status"] = success_status(args) if all(e["status"] in SUCCESS_STATUSES for e in entries) else "FAILED"
        report["host_input_unchanged"] = identity(data) == report["input_sha256"]
        if not report["host_input_unchanged"]:
            report["status"] = "FAILED"
    except DeviceExecutionAborted:
        report["status"] = "DEVICE_ERROR"
        print("Device execution fault: stopped this case and all remaining cases; see the original entry traceback.",
              file=sys.stderr, flush=True)
    except Exception:
        report.update(error_record())
    finally:
        flush()
    return report


def write_summary(report, directory):
    rows = []
    for case in report["cases"]:
        entries = list(case["baselines"].items()) + [(e["name"], e) for e in case["variants"]]
        for name, entry in entries:
            for mode in ([] if report["args"]["timing"] == "none" else report["args"]["timing"].split(",")) or ["none"]:
                row = {"case": case["label"], "name": name, "status": entry["status"], "mode": mode,
                       "relative_l2_vs_bf16_math": entry.get("vs_bf16_math", {}).get("relative_l2"),
                       "relative_l2_vs_bf16_operator": entry.get("vs_bf16_operator", {}).get("relative_l2"),
                       "worst_row_l2_vs_bf16_math": entry.get("vs_bf16_math", {}).get("max_row_relative_l2"),
                       "matching_reference_l2": entry.get("kernel_vs_matching_reference", {}).get("relative_l2"),
                       "production_performance": "NOT_VALIDATED"}
                for field, short in (("timing_prepared_read", "read"), ("timing_write_read", "write_read")):
                    timing = entry.get(field, {}).get(mode, {})
                    event = timing.get("event_us", {})
                    row[short + "_us"] = event.get("candidate_median", event.get("median"))
                    row[short + "_paired_ratio_vs_same_recipe_w0"] = event.get("median_paired_ratio")
                    row[short + "_observation"] = timing.get("observation")
                rows.append(row)
    if rows:
        with (Path(directory) / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def preflight(device="npu:0"):
    result = {"device": device, "production_validation": "NOT_VALIDATED", "modules": {}}
    for name in ("torch", "torch_npu", "triton", "cann_ops_transformer.ops", "cloud_ops_turbo"):
        try:
            mod = importlib.import_module(name)
            result["modules"][name] = {"file": getattr(mod, "__file__", None), "version": getattr(mod, "__version__", None)}
        except Exception:
            result["modules"][name] = error_record()
    result["status"] = "AVAILABLE" if all(v.get("status") != "ERROR" for v in result["modules"].values()) else "ERROR"
    if result["status"] == "AVAILABLE":
        try:
            import torch
            torch.npu.set_device(device)
            result["device_name"] = torch.npu.get_device_name()
            import triton
            target = triton.runtime.driver.active.get_current_target()
            result["triton_target"] = {"repr": str(target), "backend": str(getattr(target, "backend", "unknown")),
                                      "arch": str(getattr(target, "arch", "unknown"))}
            if result["triton_target"]["backend"].lower() in ("cuda", "hip"):
                raise RuntimeError("The active Triton target is not Ascend; this experiment requires Triton Ascend")
            result["note"] = "Imports/device lookup only; kernel compilation and execution have not been tested"
        except Exception:
            result.update(error_record())
    print(json.dumps(result, indent=2), flush=True)
    return result["status"] == "AVAILABLE"


def self_test():
    from kv_hot_backend import fixed_ring_plan
    data = synthetic_case(batch=2, queries=4, keys=257, heads=2, dim=64, rope_dim=0, topk=129)
    for window in (1, 128, 256, 512):
        plan = fixed_ring_plan(data, window)
        assert plan["capacity"] >= window + 3
        positions = query_positions(data)
        assert np.array_equal(plan["tags"].reshape(-1)[plan["current_slots"]], positions)
        reference = attention_reference(data)
        same = attention_reference(data, kv_c4=data["kv"], window=window, hot_kv=data["kv"])
        assert np.array_equal(reference, same)
    print("PASS: CPU fixed-query ring coverage, original-input mixed-reference equivalence. NPU code NOT tested.", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--input", type=Path, help="Captured BF16 post-transform/pre-quantization NPZ with fixed TopK")
    p.add_argument("--keys", type=integer_list, default=[2048])
    p.add_argument("--queries", type=integer_list, default=[4])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--rope-dim", type=int, choices=(0, 64), default=64)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--page", type=int, default=128)
    p.add_argument("--windows", type=lambda x: integer_list(x, 0), default=[0, 128, 256, 512])
    p.add_argument("--cold-recipes", type=recipe_list, default=list(RECIPES))
    p.add_argument("--device", default="npu:0")
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--timing", choices=("none", "eager", "graph", "eager,graph"), default="eager,graph")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--repeats", type=int, default=5, help="AB/BA rounds for each hot candidate")
    p.add_argument("--control-l2", type=float, default=.01, help="Global AND worst-row relative L2 vs matching mathematical reference")
    p.add_argument("--pipeline-l2", type=float, default=.001)
    p.add_argument("--block-keys", type=int, default=16)
    p.add_argument("--block-heads", type=int, default=16)
    p.add_argument("--output", type=Path)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.preflight:
        return 0 if preflight(args.device) else 1
    if 0 not in args.windows:
        p.error("--windows must include 0 for the native same-recipe baseline")
    if min(args.batch, args.heads, args.topk, args.iterations, args.repeats) < 1 or args.warmup < 0:
        p.error("Batch, heads, TopK, iterations and repeats must be positive; warmup >= 0")
    if not all(np.isfinite(x) and x > 0 for x in (args.control_l2, args.pipeline_l2)):
        p.error("Correctness thresholds must be finite and positive")
    directory = (args.output or Path("kv_hot_fused_" + time.strftime("%Y%m%d_%H%M%S"))).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = {"version": VERSION, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "status": "RUNNING", "production_validation": "NOT_VALIDATED", "cases": [],
              "interpretation": "MEASURED only certifies requested experiment completed; performance observations are not proof of no serving regression"}

    def persist(current=None):
        # Current case is already in the list while its partial file is updated.
        if current is not None and (not report["cases"] or report["cases"][-1] is not current):
            report["cases"].append(current)
        (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        write_summary(report, directory)

    try:
        if args.input:
            cases = [load_case(args.input)]
        else:
            cases = (synthetic_case(batch=args.batch, queries=q, keys=k, heads=args.heads,
                                    dim=512, rope_dim=args.rope_dim, topk=args.topk,
                                    page=args.page, seed=args.seed) for k in args.keys for q in args.queries)
        for index, data in enumerate(cases):
            current = run_case(data, args, directory / f"case_{index:03d}", progress=persist)
            if current["status"] == "DEVICE_ERROR":
                report["stopped_after_device_fault"] = current["label"]
                break
        report["status"] = ("DEVICE_ERROR" if report.get("stopped_after_device_fault") else
                            success_status(args) if report["cases"] and all(c["status"] in SUCCESS_STATUSES for c in report["cases"]) else "FAILED")
    except Exception:
        report.update(error_record())
    finally:
        persist()
    print(f"{report['status']}: {directory / 'results.json'}; production validation NOT_VALIDATED", flush=True)
    return 0 if report["status"] in SUCCESS_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())
