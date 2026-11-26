#!/usr/bin/env python3
"""Collect existing DSL run evidence without importing torch or running devices.

The runner's complete JSON remains at the top level. Collection checks only
that recorded controls and timing are present and consistent; it cannot prove
that an NPU executed the run or authenticate the recorded measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


RUNNER_VERSION = "dsl-fp4-fp8-recent128-v1"
COLLECTOR_VERSION = "dsl-recent128-evidence-v1"
PRESENT = "CONTROLS_AND_TIMING_PRESENT"


def _number(value, *, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not positive or value > 0))


def _finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON number")
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)


def _reject_constant(value):
    raise ValueError("nonstandard JSON constant: " + value)


def _read(path, issues, *, required=True):
    path = Path(path)
    if not path.is_file():
        if required:
            issues.append(f"Missing file: {path}")
        return None, dict(path=str(path), present=False)
    record = dict(path=str(path), present=True)
    try:
        data = path.read_bytes()
        record.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        value = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
        _finite_tree(value)
        return value, record
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        record["error"] = str(exc)
        issues.append(f"Invalid JSON {path}: {exc}")
        return None, record


def _check_gate(value, name, issues):
    if not isinstance(value, dict):
        issues.append("Missing control: " + name)
        return
    if value.get("passed") is not True or value.get("finite") is not True:
        issues.append("Control not recorded as finite/pass: " + name)
    for field in ("relative_l2", "worst_row_relative_l2"):
        if not _number(value.get(field)):
            issues.append(f"Missing finite {name}.{field}")


def _check_metrics(value, name, issues):
    if not isinstance(value, dict) or value.get("finite") is not True:
        issues.append("Missing finite metrics: " + name)
    elif not _number(value.get("relative_l2")):
        issues.append("Missing finite relative_l2: " + name)


def _check_timing(paired, args, issues):
    computed = {}
    if not isinstance(paired, dict):
        issues.append("Missing paired timing")
        return computed
    baseline, candidate = paired.get("baseline_median"), paired.get("candidate_median")
    if _number(baseline, positive=True) and _number(candidate, positive=True):
        computed["ratio_of_medians"] = candidate / baseline
    else:
        issues.append("Missing positive paired baseline/candidate medians")
    reported_ratio = paired.get("median_paired_ratio")
    if _number(reported_ratio, positive=True):
        computed["reported_median_paired_ratio"] = reported_ratio
    else:
        issues.append("Missing positive median_paired_ratio")
    rounds = paired.get("rounds")
    if not isinstance(rounds, list):
        issues.append("Missing original paired rounds")
        return computed
    computed["round_count"] = len(rounds)
    if len(rounds) < 10:
        issues.append("Fewer than 10 original paired rounds")
    expected_count = args.get("repeats")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool):
        issues.append("Missing integer args.repeats")
    elif len(rounds) != expected_count:
        issues.append("Round count differs from args.repeats")
    event_ratios, wall_ratios, bases, candidates = [], [], [], []
    for index, row in enumerate(rounds):
        prefix = f"paired.rounds[{index}]"
        if not isinstance(row, dict):
            issues.append(prefix + " is not an object")
            continue
        expected_order = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
        if row.get("round") != index or row.get("order") != expected_order:
            issues.append(prefix + " lacks original alternating AB/BA order")
        valid = True
        for side in ("baseline", "candidate"):
            block = row.get(side)
            for clock in ("event_us", "wall_us"):
                if not isinstance(block, dict) or not _number(block.get(clock), positive=True):
                    issues.append(f"Missing positive {prefix}.{side}.{clock}")
                    valid = False
        if not valid:
            continue
        event_ratio = row["candidate"]["event_us"] / row["baseline"]["event_us"]
        wall_ratios.append(row["candidate"]["wall_us"] / row["baseline"]["wall_us"])
        event_ratios.append(event_ratio)
        bases.append(row["baseline"]["event_us"])
        candidates.append(row["candidate"]["event_us"])
        recorded = row.get("event_us_ratio")
        if not _number(recorded, positive=True) or not math.isclose(recorded, event_ratio, rel_tol=1e-8):
            issues.append(prefix + ".event_us_ratio missing or inconsistent with raw event times")
    if event_ratios:
        computed.update(recomputed_median_paired_ratio=statistics.median(event_ratios),
                        recomputed_median_wall_ratio=statistics.median(wall_ratios),
                        candidate_faster_rounds=sum(value < 1 for value in event_ratios))
    if len(event_ratios) == len(rounds) and rounds:
        for field, expected in (("baseline_median", statistics.median(bases)),
                                ("candidate_median", statistics.median(candidates)),
                                ("median_paired_ratio", statistics.median(event_ratios))):
            recorded = paired.get(field)
            if not _number(recorded, positive=True) or not math.isclose(recorded, expected, rel_tol=1e-8):
                issues.append("paired." + field + " is inconsistent with original rounds")
    computed["ratio_note"] = "Ratio of medians and median of paired ratios are different statistics."
    return computed


def collect(source):
    """Return evidence JSON; malformed/incomplete input is an INCOMPLETE record."""
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        source = source / ("results.json" if (source / "results.json").is_file()
                           else "results/results.json")
    issues = []
    raw, source_record = _read(source, issues)
    if not isinstance(raw, dict):
        if raw is not None:
            issues.append("Runner results must be a JSON object")
        raw = {}
    result = dict(raw)  # Preserve all original fields, including unknown fields.
    computed = {}
    if raw.get("version") != RUNNER_VERSION:
        issues.append("Unsupported/missing runner version; expected " + RUNNER_VERSION)
    if raw.get("status") != "MEASURED":
        issues.append("Runner status is not MEASURED")
    for field in ("shape", "args", "environment", "implementation", "native_binary_policy"):
        if not isinstance(raw.get(field), dict) or not raw[field]:
            issues.append("Missing runner object: " + field)
    shape = raw.get("shape") if isinstance(raw.get("shape"), dict) else {}
    args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
    if not raw.get("scope"):
        issues.append("Missing timing scope")
    policy = raw.get("native_binary_policy")
    if not isinstance(policy, dict) or policy.get("mode") != "require":
        issues.append("Native binary policy is not recorded as require")
    arms = raw.get("arms") if isinstance(raw.get("arms"), dict) else {}
    window = args.get("window", shape.get("window"))
    names = ("fp4_w0", f"fp4_fp8_w{window}")
    for name in names:
        arm = arms.get(name)
        if not isinstance(arm, dict):
            issues.append("Missing arm: " + name)
            continue
        if arm.get("status") != "MEASURED":
            issues.append("Arm status is not MEASURED: " + name)
        for key in ("kernel_vs_matching_reference", "post_timing_control"):
            _check_gate(arm.get(key), name + "." + key, issues)
        for key in ("quantization_only_vs_original_bf16", "output_vs_original_bf16"):
            _check_metrics(arm.get(key), name + "." + key, issues)
        capture = arm.get("capture")
        if not isinstance(capture, dict) or capture.get("kind") != args.get("timing"):
            issues.append("Missing/mismatched capture kind: " + name)
        elif args.get("timing") == "graph":
            if capture.get("calls_per_graph") != 1:
                issues.append("Graph capture is not one call: " + name)
            _check_gate(capture.get("poison_control"), name + ".capture.poison_control", issues)
    controls = raw.get("controls") if isinstance(raw.get("controls"), dict) else {}
    for name in ("zero_query_mixed", "hot_only_with_zero_cold_length"):
        _check_gate(controls.get(name), "controls." + name, issues)
    computed["timing"] = _check_timing(raw.get("paired"), args, issues)
    topk, counts = shape.get("topk"), shape.get("hot_counts")
    if (isinstance(topk, int) and not isinstance(topk, bool) and topk > 0
            and isinstance(counts, list) and counts
            and all(isinstance(n, int) and not isinstance(n, bool) and 0 <= n <= topk for n in counts)):
        computed["hot_selection"] = dict(counts=counts, mean_count=statistics.mean(counts),
            fraction_per_query=[n / topk for n in counts], mean_fraction=statistics.mean(counts) / topk)
    else:
        issues.append("Missing/invalid original shape.topk/hot_counts")
    if len(arms) == 2 and all(isinstance(arms.get(name), dict) for name in names):
        baseline_metrics = arms[names[0]].get("output_vs_original_bf16")
        candidate_metrics = arms[names[1]].get("output_vs_original_bf16")
        a = baseline_metrics.get("relative_l2") if isinstance(baseline_metrics, dict) else None
        b = candidate_metrics.get("relative_l2") if isinstance(candidate_metrics, dict) else None
        if _number(a, positive=True) and _number(b):
            computed["l2_vs_bf16"] = dict(w0=a, window=b, absolute_drop=a-b, relative_drop=(a-b)/a)
    launcher_path = next((parent / "launcher.json" for parent in (source.parent, source.parent.parent)
                          if (parent / "launcher.json").is_file()), source.parent.parent / "launcher.json")
    launcher, launcher_record = _read(launcher_path, issues)
    if not isinstance(launcher, dict) or not re.fullmatch(r"[0-9a-fA-F]{40}", str(launcher.get("payload_commit", ""))):
        issues.append("Missing valid launcher payload_commit")
    native, native_record = _read(source.parent / "native_binary_report.json", issues, required=False)
    embedded_native = raw.get("native_binary_report")
    if native is None and embedded_native is None:
        issues.append("Missing native binary report evidence")
    try:
        _finite_tree(computed)
    except ValueError:
        issues.append("Derived summary overflowed; original finite fields are preserved")
        computed = dict(error="Nonfinite derived arithmetic; inspect original measurements")
    result.update(collector_version=COLLECTOR_VERSION,
                  evidence_status="INCOMPLETE" if issues else PRESENT,
                  evidence_issues=issues, computed=computed,
                  collection=dict(source=source_record, launcher_source=launcher_record,
                                  launcher=launcher, native_report_source=native_record,
                                  native_report_file=native,
                                  native_report_interpretation="Preserved verbatim; native-hit schema is not interpreted.",
                                  assurance="File-presence and recorded-control consistency only; not independent NPU execution verification."))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Run/results directory or the runner's results.json")
    parser.add_argument("--output", help="Write a new evidence JSON; existing files are never overwritten")
    args = parser.parse_args(argv)
    result = collect(args.source)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        try:
            target = Path(args.output).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as handle:
                handle.write(text)
        except OSError as exc:
            print(f"Cannot write new evidence file: {exc}", file=sys.stderr)
            return 2
    else:
        print(text, end="")
    return 0 if result["evidence_status"] == PRESENT else 2


if __name__ == "__main__":
    sys.exit(main())
