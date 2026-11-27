"""Three-arm precision and paired-latency reports; never invent measured data."""
import csv
import io
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

from attention_compare_metrics import compare_outputs

ARMS = ("installed_bf16", "installed_c4", "w128_split4")
REFERENCE_CONTRACTS = {
    "matching_reference": "Independent whole-selected-set FP64 attention using this arm's actual cache values; implementation error",
    "vs_original_bf16_math": "Independent whole-selected-set FP64 attention using original BF16 Q/KV/RoPE; total output error",
    "vs_native_bf16_output": "Same-input installed BF16 operator output; device-baseline output difference (BF16 self-comparison is zero)",
}


def _finite_number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _fraction(numerator, denominator):
    if not all(_finite_number(value) for value in (numerator, denominator)) or denominator == 0:
        return None
    result = numerator / denominator
    return float(result) if math.isfinite(result) else None


def build_accuracy_report(outputs, matching_references, bf16_math):
    """Preserve three distinct reference contracts for every supplied arm.

    Recovery fractions are descriptive and deliberately not clipped to [0,1].
    Nonfinite metrics or an exactly zero denominator yield None.
    """
    if not isinstance(outputs, Mapping) or not isinstance(matching_references, Mapping):
        raise ValueError("outputs and matching_references must be arm mappings")
    missing = set(ARMS) - set(outputs)
    if missing:
        raise ValueError("Required output arms missing: " + ", ".join(sorted(missing)))
    missing = set(outputs) - set(matching_references)
    if missing:
        raise ValueError("Matching references missing: " + ", ".join(sorted(missing)))
    arms = {}
    for name, actual in outputs.items():
        arms[name] = {
            "matching_reference": compare_outputs(actual, matching_references[name]),
            "vs_original_bf16_math": compare_outputs(actual, bf16_math),
            "vs_native_bf16_output": compare_outputs(actual, outputs["installed_bf16"]),
        }
    error = {name: arms[name]["vs_original_bf16_math"]["relative_l2"] for name in ARMS}
    c4, mixed, bf16 = (error[name] for name in ("installed_c4", "w128_split4", "installed_bf16"))
    valid = all(_finite_number(value) for value in (c4, mixed, bf16))
    recovered = c4 - mixed if valid else None
    gap = c4 - bf16 if valid else None
    improvement = dict(
        reference="vs_original_bf16_math.relative_l2",
        c4_relative_l2=c4, mixed_relative_l2=mixed, native_bf16_relative_l2=bf16,
        recovered_relative_l2=recovered, recoverable_relative_l2_gap=gap,
        relative_l2_error_reduction=_fraction(recovered, c4),
        relative_l2_error_reduction_formula="(c4_l2 - mixed_l2) / c4_l2",
        relative_l2_gap_recovery=_fraction(recovered, gap),
        relative_l2_gap_recovery_formula="(c4_l2 - mixed_l2) / (c4_l2 - native_bf16_l2)",
        clipping="none; negative values and values above one are retained",
    )
    return dict(reference_contracts=dict(REFERENCE_CONTRACTS), arms=arms, improvement=improvement)


def _timing_mode(report):
    args = report.get("args", {})
    requested = args.get("timing", "graph") if isinstance(args, Mapping) else "graph"
    modes = str(requested).split(",")
    return "graph" if "graph" in modes else modes[0]


def _pairwise_timings(report, mode):
    pairs = []
    for key, section in report.get("verification_timings", {}).items():
        if not isinstance(section, Mapping):
            continue
        baseline, candidate = section.get("baseline"), section.get("candidate")
        if not isinstance(baseline, str) or not isinstance(candidate, str):
            continue  # Do not infer arm identities from an ambiguous key string.
        timing = section.get("timing", {}).get(mode)
        if not isinstance(timing, Mapping):
            continue
        pair = dict(name=key, baseline=baseline, candidate=candidate, mode=mode,
            status=timing.get("status", "UNKNOWN"),
            ratio_definition="candidate / baseline; supplied median of per-round ratios",
            raw_round_count=len(timing.get("rounds", [])), event_us={}, wall_us={})
        for clock in ("event_us", "wall_us"):
            raw = timing.get(clock, {})
            if isinstance(raw, Mapping):
                pair[clock] = {field: raw.get(field) for field in (
                    "baseline_median", "candidate_median", "median_paired_ratio",
                    "min_paired_ratio", "max_paired_ratio")}
        pairs.append(pair)
    return pairs


def _choose_latency(arm, pairs):
    choices = [pair for pair in pairs if arm in (pair["baseline"], pair["candidate"])]
    if not choices:
        return None
    def preference(pair):
        if arm == "installed_c4":
            return (pair["baseline"] != arm or pair["candidate"] != "w128_split4", pair["name"])
        return (pair["baseline"] != "installed_c4" or pair["candidate"] != arm, pair["name"])
    pair = sorted(choices, key=preference)[0]
    side = "baseline" if pair["baseline"] == arm else "candidate"
    event, wall = pair["event_us"], pair["wall_us"]
    return dict(event_us=event.get(side + "_median"), wall_us=wall.get(side + "_median"),
        pair=pair["name"], pair_role=side, baseline=pair["baseline"], candidate=pair["candidate"],
        median_paired_ratio=event.get("median_paired_ratio"),
        min_paired_ratio=event.get("min_paired_ratio"), max_paired_ratio=event.get("max_paired_ratio"),
        raw_round_count=pair["raw_round_count"], status=pair["status"],
        ratio_definition=pair["ratio_definition"])


def _formatted(value):
    return "NOT_AVAILABLE" if value is None else format(value, ".9g") if _finite_number(value) else str(value)


def write_summary(report, output_dir):
    """Write new summary.json/csv/txt files without overwriting prior evidence.

    Post-timing precision takes priority because it reads real timed buffers.
    Latencies and paired ratios retain their own explicitly named comparison;
    independently measured medians are never divided or pooled across pairs.
    Full raw rounds remain in results.json, with their count recorded here.
    """
    source = "post_timing_precision_comparison" if "post_timing_precision_comparison" in report else "precision_comparison"
    precision = report.get(source, {})
    mode = _timing_mode(report)
    pairs = _pairwise_timings(report, mode)
    source_arms = precision.get("arms", {}) if isinstance(precision, Mapping) else {}
    arms = {name: dict(precision=source_arms.get(name), latency=_choose_latency(name, pairs)) for name in ARMS}
    summary = dict(version="attention-compare-summary-v1", status=report.get("status", "UNKNOWN"),
        performance_scope=report.get("performance_scope", "NOT_AVAILABLE"), timing_mode=mode,
        precision_source=source, reference_contracts=precision.get("reference_contracts", {}) if isinstance(precision, Mapping) else {},
        improvement=precision.get("improvement", {}) if isinstance(precision, Mapping) else {},
        arms=arms, pairwise_timings=pairs,
        latency_contract="Each arm latency comes from its named pair/side; paired ratio always means that pair's candidate / baseline",
        results_file="results.json (full per-round timings and controls)")
    columns = ["status", "arm", "precision_source", "matching_reference_l2", "vs_original_bf16_math_l2",
        "vs_native_bf16_output_l2", "cosine_mean_vs_original_bf16_math", "cosine_min_vs_original_bf16_math",
        "cosine_p05_vs_original_bf16_math", "nonfinite_count", "event_us", "wall_us", "latency_pair", "pair_role",
        "pair_baseline", "pair_candidate", "median_paired_ratio", "min_paired_ratio", "max_paired_ratio", "raw_round_count"]
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=columns)
    writer.writeheader()
    lines = ["Status: " + str(summary["status"]), "Precision source: " + source,
        "Scope: " + str(summary["performance_scope"]), "Timing mode: " + mode,
        "L2/cosine below use original BF16-input FP64 math; matching-cache and native-BF16 comparisons are separate in JSON/CSV.",
        "Paired ratio is candidate / baseline for the explicitly named pair; min/max are observed extrema."]
    for name, entry in arms.items():
        metrics = entry["precision"] or {}
        original = metrics.get("vs_original_bf16_math", {})
        latency = entry["latency"] or {}
        row = dict(status=summary["status"], arm=name, precision_source=source,
            matching_reference_l2=metrics.get("matching_reference", {}).get("relative_l2"),
            vs_original_bf16_math_l2=original.get("relative_l2"),
            vs_native_bf16_output_l2=metrics.get("vs_native_bf16_output", {}).get("relative_l2"),
            cosine_mean_vs_original_bf16_math=original.get("cosine_mean"),
            cosine_min_vs_original_bf16_math=original.get("cosine_min"),
            cosine_p05_vs_original_bf16_math=original.get("cosine_p05"),
            nonfinite_count=original.get("nonfinite_count"), event_us=latency.get("event_us"),
            wall_us=latency.get("wall_us"), latency_pair=latency.get("pair"), pair_role=latency.get("pair_role"),
            pair_baseline=latency.get("baseline"), pair_candidate=latency.get("candidate"),
            median_paired_ratio=latency.get("median_paired_ratio"), min_paired_ratio=latency.get("min_paired_ratio"),
            max_paired_ratio=latency.get("max_paired_ratio"), raw_round_count=latency.get("raw_round_count"))
        writer.writerow(row)
        lines.append(name + ": L2=" + _formatted(row["vs_original_bf16_math_l2"]) +
            ", cosine_mean=" + _formatted(row["cosine_mean_vs_original_bf16_math"]) +
            ", event_us=" + _formatted(row["event_us"]) + ", wall_us=" + _formatted(row["wall_us"]))
        if latency:
            lines.append("  Pair: " + latency["candidate"] + " / " + latency["baseline"] +
                "; this arm=" + latency["pair_role"] + "; ratio=" + _formatted(latency["median_paired_ratio"]) +
                " [" + _formatted(latency["min_paired_ratio"]) + ", " + _formatted(latency["max_paired_ratio"]) +
                "]; raw rounds=" + str(latency["raw_round_count"]))
    improvement = summary["improvement"]
    lines += ["Relative L2 error reduction: " + _formatted(improvement.get("relative_l2_error_reduction")),
        "Relative L2 gap recovery: " + _formatted(improvement.get("relative_l2_gap_recovery")),
        "Recovery fractions are un-clipped; zero denominators or invalid metrics produce null/NOT_AVAILABLE."]
    texts = {"summary.json": json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
             "summary.csv": table.getvalue(), "summary.txt": "\n".join(lines) + "\n"}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name in texts:
        if (directory / name).exists():
            raise FileExistsError("Summary evidence already exists: " + str(directory / name))
    for name, text in texts.items():
        with (directory / name).open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
    return summary
