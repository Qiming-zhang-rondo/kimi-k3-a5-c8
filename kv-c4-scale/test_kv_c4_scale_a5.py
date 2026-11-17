#!/usr/bin/env python3
"""Compare main-KV C4 quantization scales using fixed inputs and fixed TopK.

Single-attention accuracy and latency experiment; no model evaluation, QLI,
Hadamard, hot window, installation, or changes to the reader/cache format.
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

from kv_window_common import (ARRAY_KEYS, attention_reference, compare_outputs,
                              load_case, save_case, synthetic_case, validate_case)

VERSION = "main-kv-c4-scale-a5-v1"
RECIPES = ("baseline", "alg2_max6", "alg2_max7", "alg2_max7_25")
TIMING_FIELDS = ("timing_prepared_read", "timing_write_read")


def integers(value):
    try:
        result = [int(x) for x in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers") from exc
    if not result or min(result) < 1 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Expected distinct positive integers")
    return result


def recipe_names(value):
    result = value.split(",")
    if not result or len(set(result)) != len(result) or any(x not in RECIPES for x in result):
        raise argparse.ArgumentTypeError("Expected distinct recipes from " + ",".join(RECIPES))
    return result


def json_write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def input_identity(data):
    digest = hashlib.sha256()
    for key in ARRAY_KEYS:
        array = np.ascontiguousarray(data[key])
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        if array.size:
            digest.update(memoryview(array).cast("B"))
    digest.update(str((data["scale"], data["page"])).encode())
    return digest.hexdigest()


def check_metric(metric, limit):
    return bool(metric.get("finite") and metric["relative_l2"] <= limit
                and metric["max_row_relative_l2"] <= limit)


def compare_named(actual, reference, actual_name, reference_name):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {actual_name} {actual.shape}; {reference_name} {reference.shape}")

    def finite_summary(value):
        invalid = ~np.isfinite(value)
        return dict(finite=not bool(invalid.any()), nonfinite_count=int(invalid.sum()),
                    nan_count=int(np.isnan(value).sum()), positive_inf_count=int(np.isposinf(value).sum()),
                    negative_inf_count=int(np.isneginf(value).sum()),
                    first_nonfinite_indices=np.argwhere(invalid)[:8].tolist())

    a, r = finite_summary(actual), finite_summary(reference)
    labels = dict(actual_name=actual_name, reference_name=reference_name,
                  actual_finiteness=a, reference_finiteness=r)
    if not a["finite"] or not r["finite"]:
        return dict(finite=False, nonfinite_count=a["nonfinite_count"], rows=[],
                    invalid_reason="nonfinite_comparison_input", **labels)
    return dict(compare_outputs(actual, reference), **labels)


def improvement(candidate, baseline):
    """Positive reduction means lower measured error; zero baseline is undefined."""
    result = {}
    for metric in ("relative_l2", "max_row_relative_l2"):
        original, updated = baseline.get(metric), candidate.get(metric)
        result[metric + "_reduction_vs_c4"] = (
            1 - updated / original if original is not None and original > 0
            and updated is not None else None)
    return result


def time_call(fn, args, bench):
    result = {}
    for mode in args.timing.split(","):
        try:
            result[mode] = bench(fn, warmup=args.warmup, iterations=args.iterations,
                                 repeats=args.repeats, graph=mode == "graph")
        except Exception:
            result[mode] = dict(status="ERROR", traceback=traceback.format_exc())
    return result


def timing_failed(item):
    return any(value.get("status") == "ERROR" for field in TIMING_FIELDS
               for value in item.get(field, {}).values())


def event_median(item, field, mode):
    return item.get(field, {}).get(mode, {}).get("event_us", {}).get("median")


def quality_eligibility(item, baseline, baseline_valid):
    """Require global improvement without a worse worst-row error."""
    reasons = []
    if not baseline_valid:
        reasons.append("baseline_controls_failed")
    if not item.get("control_passed"):
        reasons.append("recipe_controls_failed")
    if timing_failed(item) or timing_failed(baseline):
        reasons.append("requested_timing_failed")
    metric = item.get("vs_bf16_operator", {})
    base_metric = baseline.get("vs_bf16_operator", {})
    if not metric.get("finite") or not base_metric.get("finite"):
        reasons.append("quality_comparison_unavailable_or_nonfinite")
    else:
        if metric["relative_l2"] >= base_metric["relative_l2"]:
            reasons.append("global_relative_l2_not_lower_than_original_c4")
        if metric["max_row_relative_l2"] > base_metric["max_row_relative_l2"]:
            reasons.append("max_row_relative_l2_worse_than_original_c4")
    item["eligible_for_best_on_these_inputs"] = not reasons
    item["eligibility_reasons"] = reasons


def summarize(report, directory, timing):
    rows = []
    base = report["baselines"].get("c4", {})
    entries = [("reference", name, item) for name, item in report["baselines"].items()]
    entries += [("recipe", item["recipe"], item) for item in report["recipes"]]
    modes = timing.split(",") if timing != "none" else ["none"]
    for group, name, item in entries:
        item["latency_ratios_vs_original_c4"] = {}
        for mode in modes:
            read, write = (event_median(item, f, mode) for f in TIMING_FIELDS)
            base_read, base_write = (event_median(base, f, mode) for f in TIMING_FIELDS)
            ratios = {"prepared_read": read/base_read if read is not None and base_read and base_read > 0 else None,
                      "write_read": write/base_write if write is not None and base_write and base_write > 0 else None}
            item["latency_ratios_vs_original_c4"][mode] = ratios
            metric = item.get("vs_bf16_operator", {})
            rows.append(dict(group=group, recipe=name, status=item.get("status"),
                             controls_passed=item.get("control_passed", False),
                             eligible_for_best=item.get("eligible_for_best_on_these_inputs", False),
                             relative_l2=metric.get("relative_l2"), max_row_relative_l2=metric.get("max_row_relative_l2"),
                             relative_l2_reduction_vs_c4=item.get("relative_l2_reduction_vs_c4"),
                             max_row_relative_l2_reduction_vs_c4=item.get("max_row_relative_l2_reduction_vs_c4"),
                             timing=mode, prepared_read_us=read, write_read_us=write,
                             prepared_read_ratio_vs_c4=ratios["prepared_read"], write_read_ratio_vs_c4=ratios["write_read"],
                             prepared_read_timing_status=item.get("timing_prepared_read", {}).get(mode, {}).get("status", "MEASURED" if read is not None else "NOT_MEASURED"),
                             write_read_timing_status=item.get("timing_write_read", {}).get(mode, {}).get("status", "MEASURED" if write is not None else "NOT_MEASURED"),
                             attention_calls_per_read=item.get("attention_calls_per_read")))
    fields = list(rows[0]) if rows else ["group", "recipe", "status"]
    with (directory/"summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def show_item(name, item):
    metric = item.get("vs_bf16_operator", {})
    val = lambda key: f"{metric[key]:.6g}" if key in metric else "n/a"
    reduction = item.get("relative_l2_reduction_vs_c4")
    suffix = f"; L2 reduction vs original C4={reduction:+.2%}" if reduction is not None else ""
    print(f"  {name}: {item['status']}; L2={val('relative_l2')}; worst row={val('max_row_relative_l2')}{suffix}", flush=True)
    if item.get("traceback"):
        print(f"    failed stage: {item.get('stage', 'initialization')}\n{item['traceback']}", flush=True)
    elif item["status"] == "CONTROL_FAILED":
        for key in ("kernel_vs_matching_reference", "write_vs_prepared", "baseline_vs_original_c4"):
            control = item.get(key)
            if control is not None:
                print(f"    {key}: finite={control.get('finite')}; L2={control.get('relative_l2')}; "
                      f"worst row={control.get('max_row_relative_l2')}", flush=True)
        if item.get("timing_skipped"):
            print(f"    timing skipped: {item['timing_skipped']}", flush=True)


def show_timings(report, timing):
    if timing == "none":
        return
    entries = list(report["baselines"].items()) + [(x["recipe"], x) for x in report["recipes"]]
    for name, item in entries:
        for mode in timing.split(","):
            read, write = (event_median(item, field, mode) for field in TIMING_FIELDS)
            ratios = item.get("latency_ratios_vs_original_c4", {}).get(mode, {})
            fmt = lambda x: f"{x:.3f}" if x is not None else "n/a"
            print(f"  {name} {mode}: read={fmt(read)} us ({fmt(ratios.get('prepared_read'))}x C4); "
                  f"write+read={fmt(write)} us ({fmt(ratios.get('write_read'))}x C4)", flush=True)
            for field in TIMING_FIELDS:
                result = item.get(field, {}).get(mode, {})
                if result.get("status") == "ERROR":
                    detail = result.get("traceback", "").strip().splitlines()
                    print(f"    {field} ERROR: {detail[-1] if detail else 'see results.json traceback'}", flush=True)


def run_case(data, args, directory, backend_class=None, bench=None):
    if backend_class is None:
        from kv_c4_scale_backend import C4ScaleBackend
        backend_class = C4ScaleBackend
    if bench is None and args.timing != "none":
        from kv_window_backend import benchmark
        bench = benchmark
    data = validate_case(data)
    directory = Path(directory)
    directory.mkdir()
    fingerprint = input_identity(data)
    report = dict(label=data["label"], source_kind=data["source_kind"],
                  selection_source=data["selection_source"], scale=data["scale"],
                  input_sha256=fingerprint, fixed_topk=True, window=0, hadamard=False,
                  shapes={k: list(data[k].shape) for k in ("q_nope", "q_rope", "kv", "indices")},
                  q_lengths=data["q_lengths"].tolist(), k_lengths=data["k_lengths"].tolist(),
                  baselines={}, recipes=[], best_on_these_inputs=None, status="RUNNING")
    arrays, outputs = {}, {}
    try:
        if args.save_inputs:
            save_case(directory/"input.npz", data)
        backend = backend_class(data, device=args.device)
        report["environment"] = backend.environment
        original_ref = attention_reference(data)
        arrays["bf16_mathematical_reference"] = original_ref
        for kind in ("bf16", "c4", "c8"):
            item = dict(status="RUNNING", control_passed=False, stage="prepared_read",
                        input_sha256=fingerprint, attention_calls_per_read=1)
            report["baselines"][kind] = item
            if kind == "bf16":
                item["attention_calls_per_read"] = backend.environment.get("bf16_full_read_plan", {}).get("operator_calls")
            try:
                outputs[kind] = backend.host(backend.baseline(kind))
                arrays[kind] = outputs[kind]
                item["stage"] = "write_read"
                updated = backend.host(backend.baseline_run(kind, write=True))
                arrays[kind + "_write_read"] = updated
                item["stage"] = "matching_reference"
                reference = original_ref if kind == "bf16" else attention_reference(data, kv_c4=backend.baseline_decoded(kind))
                arrays[kind + "_mathematical_reference"] = reference
                item["kernel_vs_matching_reference"] = compare_named(outputs[kind], reference, kind + "_operator", kind + "_mathematical_reference")
                item["write_vs_prepared"] = compare_named(updated, outputs[kind], kind + "_write_read", kind + "_prepared_operator")
                item["vs_original_mathematical_reference"] = compare_named(outputs[kind], original_ref, kind + "_operator", "original_mathematical_reference")
                if "bf16" in outputs:
                    item["vs_bf16_operator"] = compare_named(outputs[kind], outputs["bf16"], kind + "_operator", "bf16_operator")
                item["same_input_and_topk"] = input_identity(data) == fingerprint
                item["control_passed"] = (check_metric(item["kernel_vs_matching_reference"], args.kernel_rel_l2_max)
                                          and check_metric(item["write_vs_prepared"], args.write_rel_l2_max)
                                          and item["same_input_and_topk"])
                item.update(stage="completed", status="MEASURED" if item["control_passed"] else "CONTROL_FAILED")
            except Exception:
                item.update(status="ERROR", traceback=traceback.format_exc())
            json_write(directory/"results.json", report)
            show_item(kind, item)
        baseline_valid = all(item.get("control_passed") for item in report["baselines"].values())
        report["baseline_controls_passed"] = baseline_valid
        base = report["baselines"]["c4"]
        for kind, item in report["baselines"].items():
            item.update(improvement(item.get("vs_bf16_operator", {}), base.get("vs_bf16_operator", {})))
            if args.timing != "none" and baseline_valid:
                item["timing_prepared_read"] = time_call(lambda k=kind: backend.baseline(k), args, bench)
                item["timing_write_read"] = time_call(lambda k=kind: backend.baseline_run(k, write=True), args, bench)
                if timing_failed(item):
                    item["status"] = "TIMING_ERROR"
            elif args.timing != "none":
                item["timing_skipped"] = "baseline_controls_failed"
        for name in args.recipes:
            item = dict(recipe=name, status="RUNNING", control_passed=False,
                        stage="prepare", input_sha256=fingerprint, attention_calls_per_read=1)
            report["recipes"].append(item)
            try:
                runner = backend.prepare(name)
                item["metadata"] = runner.metadata
                item["stage"] = "prepared_read"
                actual = backend.host(runner.call())
                arrays["recipe_" + name] = actual
                item["stage"] = "write_read"
                updated = backend.host(runner.run(write=True))
                arrays["recipe_" + name + "_write_read"] = updated
                item["stage"] = "matching_reference"
                reference = attention_reference(data, kv_c4=runner.decoded_kv())
                arrays["recipe_" + name + "_mathematical_reference"] = reference
                item["kernel_vs_matching_reference"] = compare_named(actual, reference, name + "_operator", name + "_mathematical_reference")
                item["write_vs_prepared"] = compare_named(updated, actual, name + "_write_read", name + "_prepared_operator")
                item["vs_original_mathematical_reference"] = compare_named(actual, original_ref, name + "_operator", "original_mathematical_reference")
                if "bf16" in outputs:
                    item["vs_bf16_operator"] = compare_named(actual, outputs["bf16"], name + "_operator", "bf16_operator")
                item["same_input_and_topk"] = input_identity(data) == fingerprint
                item["control_passed"] = (check_metric(item["kernel_vs_matching_reference"], args.kernel_rel_l2_max)
                                          and check_metric(item["write_vs_prepared"], args.write_rel_l2_max)
                                          and item["same_input_and_topk"])
                if name == "baseline":
                    item["baseline_vs_original_c4"] = compare_named(actual, outputs["c4"], "baseline_recipe_operator", "original_c4_operator")
                    item["control_passed"] = item["control_passed"] and check_metric(item["baseline_vs_original_c4"], args.write_rel_l2_max)
                item.update(improvement(item.get("vs_bf16_operator", {}), base.get("vs_bf16_operator", {})))
                item["status"] = "MEASURED" if item["control_passed"] and baseline_valid else "CONTROL_FAILED"
                if args.timing != "none" and baseline_valid and item["control_passed"]:
                    item["stage"] = "timing"
                    item["timing_prepared_read"] = time_call(runner.call, args, bench)
                    item["timing_write_read"] = time_call(lambda: runner.run(write=True), args, bench)
                    if timing_failed(item):
                        item["status"] = "TIMING_ERROR"
                elif args.timing != "none":
                    item["timing_skipped"] = "baseline_or_recipe_controls_failed"
                item["stage"] = "completed"
            except Exception:
                item.update(status="ERROR", control_passed=False, traceback=traceback.format_exc())
            quality_eligibility(item, base, baseline_valid)
            json_write(directory/"results.json", report)
            show_item(name, item)
        eligible = [x for x in report["recipes"] if x["recipe"] != "baseline"
                    and x["eligible_for_best_on_these_inputs"]]
        if eligible:
            best = min(eligible, key=lambda x: x["vs_bf16_operator"]["relative_l2"])
            report["best_on_these_inputs"] = dict(recipe=best["recipe"],
                relative_l2=best["vs_bf16_operator"]["relative_l2"],
                max_row_relative_l2=best["vs_bf16_operator"]["max_row_relative_l2"],
                relative_l2_reduction_vs_c4=best["relative_l2_reduction_vs_c4"],
                scope="This input only; no claim about GLM task scores or other inputs")
        report["status"] = ("MEASURED" if all(x["status"] == "MEASURED" for x in
                            list(report["baselines"].values()) + report["recipes"])
                            else "ERROR_OR_CONTROL_FAILED")
    except Exception:
        report.update(status="ERROR", traceback=traceback.format_exc())
        print(f"  Case ERROR:\n{report['traceback']}", flush=True)
    finally:
        with (directory/"outputs.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        summarize(report, directory, args.timing)
        json_write(directory/"results.json", report)
    show_timings(report, args.timing)
    winner = report.get("best_on_these_inputs")
    print("  best_on_these_inputs: " + (winner["recipe"] if winner else "none; see eligibility_reasons")
          + f"; results: {directory}/results.json", flush=True)
    return report


def self_test():
    data = synthetic_case(keys=65, queries=2, heads=2, dim=64, topk=32)
    original = attention_reference(data)
    assert check_metric(compare_named(original, original, "actual", "reference"), .03)
    a = dict(relative_l2=.05, max_row_relative_l2=.08)
    b = dict(relative_l2=.1, max_row_relative_l2=.16)
    assert improvement(a, b)["relative_l2_reduction_vs_c4"] == .5
    changed = original.copy()
    changed[0] = np.nan
    assert not compare_named(changed, original, "actual", "reference")["finite"]
    assert input_identity(data) == input_identity(validate_case(data))
    print("PASS: CPU input/reference/metric checks. No NPU accuracy or performance claim.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", default=[], help="Original BF16 main-KV snapshot NPZ; repeatable")
    p.add_argument("--keys", type=integers, default=[32768])
    p.add_argument("--queries", type=integers, default=[4])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--page", type=int, default=128)
    p.add_argument("--recipes", type=recipe_names, default=list(RECIPES))
    p.add_argument("--device", default="npu:0")
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--timing", choices=("none", "eager", "graph", "eager,graph"), default="eager,graph")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--kernel-rel-l2-max", type=float, default=.03,
                   help="Global AND worst-row matching-reference guard; not a quantization quality target")
    p.add_argument("--write-rel-l2-max", type=float, default=.01)
    p.add_argument("--save-inputs", action="store_true")
    p.add_argument("--self-test", action="store_true", help="CPU only; no NPU initialization")
    p.add_argument("--output", default=None, help="New directory, must not already exist")
    args = p.parse_args(argv)
    if min(args.batch, args.heads, args.topk, args.iterations, args.repeats) < 1 or args.warmup < 0:
        p.error("Counts must be positive; warmup may be zero")
    if args.seed < 0 or args.page < 16 or args.page > 1024 or args.page % 16:
        p.error("Seed must be nonnegative; page must be a multiple of 16 in [16,1024]")
    if any(not math.isfinite(x) or x <= 0 for x in (args.kernel_rel_l2_max, args.write_rel_l2_max)):
        p.error("Control thresholds must be finite and positive")
    if args.self_test:
        self_test()
        return 0
    root = Path(args.output or ("kv-c4-scale-" + time.strftime("%Y%m%d_%H%M%S"))).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    names = ("test_kv_c4_scale_a5.py", "kv_c4_scale_backend.py", "kv_window_common.py", "kv_window_backend.py")
    report = dict(version=VERSION, args=vars(args), status="RUNNING", cases=[],
                  scope="Pure main-KV C4 scale A/B; same original input and TopK for every recipe; no QLI execution or GLM evaluation",
                  timing_scope="Prepared single reader and current-token quantize/scatter/read; excludes initial full-cache construction, model projections, QLI, scheduler, and collectives",
                  quality_note="Synthetic inputs are not model activations. Lower attention-output L2 does not establish a GLM score improvement.",
                  selection_rule="Per case: passing controls and requested timing, lower global L2 and no worse max-row L2 than original C4. No cross-case or model-score winner inferred.",
                  scripts={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                           for name in names if (Path(__file__).parent/name).is_file()})
    try:
        if args.input:
            cases = (load_case(path) for path in args.input)
        else:
            cases = (synthetic_case(batch=args.batch, keys=k, queries=q, heads=args.heads,
                                    topk=args.topk, page=args.page, seed=args.seed)
                     for k in args.keys for q in args.queries)
        for i, data in enumerate(cases):
            print(f"[{i}] {data['label']} ({data['source_kind']}); fixed TopK; pure main-KV C4 scale comparison", flush=True)
            report["cases"].append(run_case(data, args, root/f"case_{i:03d}"))
            json_write(root/"results.json", report)
        report["status"] = ("MEASURED" if report["cases"] and all(c["status"] == "MEASURED" for c in report["cases"])
                            else "ERROR_OR_CONTROL_FAILED")
    except Exception:
        report.update(status="ERROR", traceback=traceback.format_exc())
        print(f"ERROR:\n{report['traceback']}", flush=True)
    finally:
        json_write(root/"results.json", report)
    print(f"{report['status']}: {root}/results.json", flush=True)
    return 0 if report["status"] == "MEASURED" else 1


if __name__ == "__main__":
    sys.exit(main())
