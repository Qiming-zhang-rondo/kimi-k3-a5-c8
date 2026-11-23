#!/usr/bin/env python3
"""One fixed A5 experiment: BF16 versus C8 recent KV, with native C4 history."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from kv_window_common import (ARRAY_KEYS, attention_reference, load_case, save_case,
                              synthetic_case, validate_case, window_selection_stats)
from test_kv_hot_fused_a5 import (DeviceExecutionAborted, SUCCESS_STATUSES, acceptable,
    compare, error_record, evaluate, identity, paired_benchmark, standalone_benchmark,
    success_status)

VERSION = "kv-bf16-hot-ab-a5-v1"
NAMES = ("full_bf16", "full_c8", "c4_w0", "split_c4", "c8_native",
         "c8_hot_store_partition", "bf16_native", "bf16_hot_store_partition")


class ControlFailed(RuntimeError):
    pass


def validate_fixed_case(data):
    data = validate_case(data)
    expected = {"q_nope": (1,4,64,512), "q_rope": (1,4,64,64),
                "kv": (1,8192,1,512), "k_rope": (1,8192,1,64), "indices": (1,4,1,2048)}
    if any(tuple(data[key].shape) != shape for key, shape in expected.items()):
        raise ValueError("This experiment requires B1 Q4 K8192 H64 D512 R64 TopK2048")
    if data["page"] != 128 or data["q_lengths"].tolist() != [4] or data["k_lengths"].tolist() != [8192]:
        raise ValueError("This experiment requires page128 and full Q4/K8192 lengths")
    selection = window_selection_stats(data, 128)
    if selection["recent_selected_min"] != 128 or selection["recent_selected_max"] != 128:
        raise ValueError("All 128 recent tokens must be selected in each query; no padded BF16 hot subset")
    return data


def all_entries(case):
    return list(case["baselines"].values()) + case["variants"]


def timing_failed(entry):
    blocks = [value for key, value in entry.items() if key.startswith("timing_") and isinstance(value, dict)]
    blocks += list(entry.get("stage_timings", {}).values())
    return any(item.get("status") == "ERROR" for block in blocks for item in block.values() if isinstance(item, dict))


def run_case(data, args, directory, *, backend_factory=None, bf16_runner_factory=None,
             benchmark=None, progress=None):
    data = validate_fixed_case(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_case(directory / "inputs.npz", data)
    rows = {name: dict(name=name, status="PENDING", recipe="baseline", window=0 if name in NAMES[:3] else 128,
                      role="BASELINE" if name.startswith("full_") else "NATIVE_W0" if name == "c4_w0" else
                           "SPLIT_CONTROL" if name == "split_c4" else "HOT_WINDOW") for name in NAMES}
    report = dict(label=data["label"], status="RUNNING", input_sha256=identity(data),
        source_kind=data["source_kind"], selection_source=data["selection_source"],
        shape={key: list(data[key].shape) for key in ARRAY_KEYS},
        baselines={name: rows[name] for name in NAMES[:2]}, variants=[rows[name] for name in NAMES[2:]],
        selection=window_selection_stats(data, 128), directory=str(directory), production_validation="NOT_VALIDATED",
        benchmark_scope={"fixed_shape": "B1 Q4 K8192 H64 D512 R64 TopK2048 W128",
            "forced_hot_hits": "each row selects all 128 recent tokens; default synthetic TopK includes recent256",
            "selection_bias": "default synthetic includes recent256; replay source/selection recorded separately; fixed all128-hot-hit case is not a serving distribution",
            "bf16_hot_calls": "one native compact BF16 hot call for this equal-count128 shape only",
            "graph_capture": "one invocation per graph; repeated replay outside capture; setup excluded",
            "full_pipeline": "cold write + hot write/preparation + dynamic partition + both native reads + merge",
            "stage_timings": "independent captures; do not add/subtract to predict full-pipeline latency",
            "production_performance": "NOT_VALIDATED"})
    arrays = {"indices": data["indices"]}
    runners, outputs, calls = {}, {}, {}

    def flush():
        (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        np.savez_compressed(directory / "outputs.npz", **arrays)
        if progress:
            progress(report)

    def finish(entry):
        print(f"{entry['name']}: {entry['status']} matching_L2={entry.get('kernel_vs_matching_reference', {}).get('relative_l2')}", flush=True)
        flush()
        if entry.get("failure_kind") == "DEVICE_EXECUTION_FAULT":
            report["stopped_after_device_fault"] = entry["name"]
            raise DeviceExecutionAborted(entry["name"])

    def require(entry, dependencies):
        failed = [name for name in dependencies if not rows[name].get("control_passed")]
        if failed:
            raise ControlFailed("Required accuracy control failed: " + ", ".join(failed))

    def diagnostics(entry, runner, bf16=False):
        for field, method in (("partition_consistency", "partition_diagnostics"),
                              ("writer_consistency", "writer_diagnostics"),
                              ("zero_query_statistics", "zero_query_diagnostics")):
            entry[field] = getattr(runner, method)()
            if not entry[field].get("passed"):
                raise ControlFailed(field + " failed")
        if bf16:
            entry["reader_consistency"] = runner.reader_diagnostics(control_l2=args.control_l2, pipeline_l2=args.pipeline_l2)
            if not entry["reader_consistency"].get("passed"):
                raise ControlFailed("Independent compact BF16 hot reader/mixed merge control failed")

    try:
        if backend_factory is None:
            from kv_native_twopass import NativeTwoPassBackend
            backend_factory = NativeTwoPassBackend
        if bf16_runner_factory is None:
            from kv_bf16_hot_ab import Bf16HotRunner
            bf16_runner_factory = Bf16HotRunner
        if benchmark is None:
            from kv_window_backend import benchmark
        backend = backend_factory(data, device=args.device)
        report["environment"] = backend.environment
        bf16_math = attention_reference(data)
        arrays["bf16_math"] = bf16_math
        cold = None
        for name in NAMES:
            entry = rows[name]
            entry["status"] = "VALIDATING"
            try:
                if name in NAMES[:2]:
                    kind = name.removeprefix("full_")
                    matching = bf16_math if kind == "bf16" else attention_reference(data, kv_c4=backend.baseline_decoded(kind))
                    read = lambda kind=kind: backend.baseline(kind)
                    pipeline = lambda kind=kind: backend.baseline_run(kind, write=True)
                    entry.update(mode="native", read_statistics=False)
                else:
                    require(entry, NAMES[:2])
                    if name == "c4_w0":
                        runner = backend.prepare("baseline", 0)
                        cold = runner.decoded_kv()
                        matching = attention_reference(data, kv_c4=cold)
                        entry.update(mode="native", read_statistics=False)
                    else:
                        require(entry, ["c4_w0"] if name == "split_c4" else ["c4_w0", "split_c4"])
                        split, bf16 = name == "split_c4", name.startswith("bf16_")
                        writer = "hot_store_partition" if name.endswith("hot_store_partition") else "native"
                        if bf16:
                            runner = bf16_runner_factory(backend, recipe="baseline", window=128, writer=writer)
                        else:
                            runner = backend.prepare("baseline", 128, mode="split_c4" if split else "mixed",
                                index_capacity="bounded", merge="triton", partition="triton_scan", writer=writer, reader="native")
                        entry.update(mode="split_c4" if split else "mixed", hot_dtype="c4" if split else "bf16" if bf16 else "c8",
                                     writer=writer, reader="native", partition="triton_scan", merge="triton", read_statistics=True)
                        entry["implementation"] = runner.metadata
                        diagnostics(entry, runner, bf16=bf16)
                        matching = attention_reference(data, kv_c4=cold) if split else attention_reference(
                            data, kv_c4=cold, window=128, hot_kv=runner.decoded_hot_kv())
                    runners[name] = runner
                    entry["implementation"] = runner.metadata
                    read, pipeline = runner.call, lambda runner=runner: runner.run(write=True, dynamic=True)
                calls[name] = (read, pipeline)
                metrics, output = evaluate(read, pipeline, matching, bf16_math, outputs.get("full_bf16"),
                                           args, backend, arrays, name)
                entry.update(metrics)
                outputs[name] = output
                if name not in NAMES[:3]:
                    dynamic = backend.host(runner.run(write=False, dynamic=True))
                    arrays[name + "_dynamic"] = dynamic
                    entry["dynamic_vs_prepared"] = compare(dynamic, output)
                    entry["control_passed"] &= acceptable(entry["dynamic_vs_prepared"], args.pipeline_l2)
                if name == "split_c4":
                    entry["split_vs_native_w0"] = compare(output, outputs["c4_w0"])
                    entry["control_passed"] &= acceptable(entry["split_vs_native_w0"], args.control_l2)
                entry["status"] = "VALIDATED_ONLY" if entry["control_passed"] else "CONTROL_FAILED"
            except ControlFailed as exc:
                entry.update(status="CONTROL_FAILED", control_passed=False, control_failure_reason=str(exc))
            except Exception:
                entry.update(error_record(), control_passed=False)
            finish(entry)

        # All accuracy checks finish before any timing. Each optional failure
        # remains visible and does not stop independent validated comparisons.
        for name in NAMES:
            entry = rows[name]
            if not entry.get("control_passed"):
                continue
            try:
                if name.startswith("bf16_"):
                    reference_name = name.replace("bf16_", "c8_", 1)
                    entry["same_writer_c8_reference"] = reference_name
                    if not rows[reference_name].get("control_passed") or rows[reference_name]["status"] not in SUCCESS_STATUSES:
                        entry.update(status="REFERENCE_UNAVAILABLE", comparison_failure_reason="Same-writer C8 reference failed")
                        finish(entry)
                        continue
                read, pipeline = calls[name]
                if name in NAMES[:3]:
                    entry["timing_prepared_read"] = standalone_benchmark(read, args, benchmark)
                    entry["timing_write_read"] = standalone_benchmark(pipeline, args, benchmark)
                else:
                    entry["timing_reference"] = "c4_w0"
                    entry["timing_prepared_read"] = paired_benchmark(calls["c4_w0"][0], read, args, benchmark)
                    entry["timing_dynamic_read"] = paired_benchmark(calls["c4_w0"][0],
                        lambda runner=runners[name]: runner.run(write=False, dynamic=True), args, benchmark)
                    entry["timing_write_read"] = paired_benchmark(calls["c4_w0"][1], pipeline, args, benchmark)
                    if name.startswith("bf16_"):
                        entry["timing_vs_same_writer_c8_read"] = paired_benchmark(calls[reference_name][0], read, args, benchmark)
                        entry["timing_vs_same_writer_c8"] = paired_benchmark(calls[reference_name][1], pipeline, args, benchmark)
                    if args.stages and args.timing != "none":
                        entry["stage_timings"] = {stage: standalone_benchmark(call, args, benchmark)
                                                  for stage, call in runners[name].component_calls().items()}
                entry["status"] = "TIMING_ERROR" if timing_failed(entry) else success_status(args)
            except Exception:
                entry.update(error_record())
            finish(entry)
        report["host_input_unchanged"] = identity(data) == report["input_sha256"]
        report["status"] = success_status(args) if report["host_input_unchanged"] and all(
            entry["status"] in SUCCESS_STATUSES for entry in rows.values()) else "FAILED"
    except DeviceExecutionAborted:
        report["status"] = "DEVICE_ERROR"
        for entry in rows.values():
            if entry["status"] in ("PENDING", "VALIDATING"):
                entry["status"] = "NOT_RUN_DEVICE_FAULT"
    except Exception:
        report.update(error_record())
    finally:
        flush()
    return report


def write_reports(report, directory):
    directory = Path(directory)
    (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    mode = "graph" if "graph" in report["args"]["timing"] else report["args"]["timing"]
    lines = [f"{VERSION} status={report['status']} timing={mode}",
             "Fixed B1 Q4 K8192 TopK2048 W128; all128 hot tokens selected. Default synthetic includes recent256; replay provenance shown per case. No serving claim.",
             "W0 and same-writer C8 columns are separate recorded pairs; median/min/max are samples, not confidence bounds. Stages are not additive.",
             "name | status | L2vsBF16 | matchingL2 | pipeline_us | paired_W0_us | W0_ratio | C8_reference | C8_pair_base_us | C8_pair_BF16_us | BF16/C8_ratio[min,max]"]
    def number(value):
        return "NA" if value is None else f"{value:.6g}"
    rows = []
    for case in report["cases"]:
        lines.append(f"case={case['label']} entries={len(all_entries(case))} source={case.get('source_kind', 'unrecorded')} selection={case.get('selection_source', 'unrecorded')}")
        for entry in all_entries(case):
            eligible = entry.get("status") == "MEASURED" and entry.get("control_passed")
            def measured(field):
                item = entry.get(field, {}).get(mode, {})
                return item.get("event_us", {}) if eligible and item.get("status", "MEASURED") == "MEASURED" else {}
            pipe, paired = measured("timing_write_read"), measured("timing_vs_same_writer_c8")
            read, dynamic, paired_read = (measured(field) for field in (
                "timing_prepared_read", "timing_dynamic_read", "timing_vs_same_writer_c8_read"))
            row = dict(case=case["label"], name=entry["name"], status=entry["status"], role=entry["role"],
                hot_dtype=entry.get("hot_dtype"), writer=entry.get("writer"), read_statistics=entry.get("read_statistics"),
                l2_vs_bf16=entry.get("vs_bf16_math", {}).get("relative_l2"),
                matching_l2=entry.get("kernel_vs_matching_reference", {}).get("relative_l2"),
                prepared_read_us=read.get("candidate_median", read.get("median")),
                paired_w0_read_us=read.get("baseline_median"), dynamic_read_us=dynamic.get("candidate_median"),
                pipeline_us=pipe.get("candidate_median", pipe.get("median")), paired_w0_us=pipe.get("baseline_median"),
                w0_ratio=pipe.get("median_paired_ratio"), c8_reference=entry.get("same_writer_c8_reference"),
                c8_pair_baseline_us=paired.get("baseline_median"), c8_pair_candidate_us=paired.get("candidate_median"),
                bf16_vs_c8_ratio=paired.get("median_paired_ratio"), bf16_vs_c8_min_ratio=paired.get("min_paired_ratio"),
                bf16_vs_c8_max_ratio=paired.get("max_paired_ratio"),
                c8_read_pair_baseline_us=paired_read.get("baseline_median"),
                c8_read_pair_bf16_us=paired_read.get("candidate_median"),
                bf16_vs_c8_read_ratio=paired_read.get("median_paired_ratio"))
            rows.append(row)
            lines.append(" | ".join([entry["name"], entry["status"], number(row["l2_vs_bf16"]), number(row["matching_l2"]),
                number(row["pipeline_us"]), number(row["paired_w0_us"]), number(row["w0_ratio"]), row["c8_reference"] or "NA",
                number(row["c8_pair_baseline_us"]), number(row["c8_pair_candidate_us"]),
                f"{number(row['bf16_vs_c8_ratio'])}[{number(row['bf16_vs_c8_min_ratio'])},{number(row['bf16_vs_c8_max_ratio'])}]"]))
    text = "\n".join(lines) + "\n"
    for filename in ("paired_summary.txt", "summary.txt"):
        (directory / filename).write_text(text, encoding="utf-8")
    if rows:
        with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return text


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--input", type=Path, help="Optional replay; must match the fixed shape and all128 hot hits")
    for name, value in (("batch", 1), ("queries", 4), ("keys", 8192), ("heads", 64),
                        ("topk", 2048), ("window", 128), ("page", 128), ("rope-dim", 64)):
        p.add_argument("--"+name, type=int, choices=(value,), default=value)
    p.add_argument("--recipe", choices=("baseline",), default="baseline")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--timing", choices=("none", "eager", "graph", "eager,graph"), default="graph")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--repeats", type=int, default=10, help="Alternating AB/BA paired rounds, not serving samples")
    p.add_argument("--stages", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--diagnose", action="store_true", default=True, help="Subset zero-query gates are always enabled")
    p.add_argument("--control-l2", type=float, default=.01)
    p.add_argument("--pipeline-l2", type=float, default=.001)
    p.add_argument("--output", type=Path)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.self_test:
        data = validate_fixed_case(synthetic_case(batch=1, queries=4, keys=8192, heads=64, dim=512,
                                                  rope_dim=64, topk=2048, page=128, seed=args.seed))
        assert window_selection_stats(data, 128)["recent_selected_min"] == 128
        print("Fixed BF16/C8 hot-window CPU setup self-test only; no A5 execution.")
        return 0
    if args.warmup < 0 or min(args.iterations, args.repeats) < 1:
        p.error("Nonnegative warmup and positive iterations/repeats required")
    if not all(np.isfinite(value) and value > 0 for value in (args.control_l2, args.pipeline_l2)):
        p.error("Correctness thresholds must be finite and positive")
    directory = (args.output or Path("kv_bf16_hot_ab_"+time.strftime("%Y%m%d_%H%M%S"))).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(version=VERSION, args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  status="RUNNING", cases=[], production_validation="NOT_VALIDATED")
    def persist(case=None):
        if case is not None and not report["cases"]:
            report["cases"].append(case)
        write_reports(report, directory)
    try:
        data = load_case(args.input) if args.input else synthetic_case(batch=1, queries=4, keys=8192,
            heads=64, dim=512, rope_dim=64, topk=2048, page=128, seed=args.seed)
        case = run_case(data, args, directory / "case_000", progress=persist)
        report["status"] = case["status"]
    except Exception:
        report.update(error_record())
    finally:
        persist()
    print(write_reports(report, directory), flush=True)
    print(f"Results: {directory / 'results.json'}\nPaired summary: {directory / 'paired_summary.txt'}", flush=True)
    return 0 if report["status"] in SUCCESS_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())
