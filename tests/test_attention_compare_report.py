"""Three reference contracts and paired evidence must survive summarization."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "experimental" / "single_pass"
sys.path.insert(0, str(HERE))
try:
    import attention_compare_report as reports
finally:
    sys.path.pop(0)


def precision(bf16=1.01, c4=1.2, mixed=1.1):
    original = np.ones((2, 2, 3), dtype=np.float32)
    outputs = {name: original * value for name, value in (
        ("installed_bf16", bf16), ("installed_c4", c4), ("w128_split4", mixed))}
    return reports.build_accuracy_report(outputs, {k: v.copy() for k, v in outputs.items()}, original)


def pair(baseline, candidate, candidate_median=50., ratio=.51):
    values = [(10., 9.), (100., 50.), (100., 51.)]
    rounds = [dict(round=i, order=["baseline", "candidate"] if i % 2 == 0 else ["candidate", "baseline"],
        baseline={"event_us": {"median": a}}, candidate={"event_us": {"median": b}},
        event_us_ratio=b/a) for i, (a, b) in enumerate(values)]
    return dict(baseline=baseline, candidate=candidate,
        timing={"graph": dict(status="MEASURED", rounds=rounds,
            event_us=dict(baseline_median=100., candidate_median=candidate_median,
                median_paired_ratio=ratio, min_paired_ratio=.5, max_paired_ratio=.9),
            wall_us=dict(baseline_median=102., candidate_median=candidate_median+2.,
                median_paired_ratio=.52, min_paired_ratio=.51, max_paired_ratio=.92))})


class AttentionCompareReportTest(unittest.TestCase):
    def test_three_references_are_distinct_and_query_head_cosines_preserved(self):
        data = precision()
        value = data["arms"]["w128_split4"]
        self.assertEqual(value["matching_reference"]["relative_l2"], 0.)
        self.assertAlmostEqual(value["vs_original_bf16_math"]["relative_l2"], .1, places=6)
        self.assertAlmostEqual(value["vs_native_bf16_output"]["relative_l2"], .09/1.01, places=6)
        self.assertEqual(value["vs_original_bf16_math"]["row_count"], 4)
        self.assertEqual(data["arms"]["installed_bf16"]["vs_native_bf16_output"]["relative_l2"], 0.)
        self.assertEqual(set(data["reference_contracts"]), set(value))
        improvement = data["improvement"]
        self.assertAlmostEqual(improvement["relative_l2_error_reduction"], .5, places=6)
        self.assertAlmostEqual(improvement["relative_l2_gap_recovery"], .1/.19, places=6)

    def test_recovery_not_clipped_and_zero_denominators_are_null(self):
        worse = precision(c4=1.2, mixed=1.3)
        self.assertLess(worse["improvement"]["relative_l2_error_reduction"], 0.)
        better_than_bf16 = precision(bf16=1.01, c4=1.2, mixed=1.)
        self.assertGreater(better_than_bf16["improvement"]["relative_l2_gap_recovery"], 1.)
        zero = precision(bf16=1., c4=1., mixed=1.)
        self.assertIsNone(zero["improvement"]["relative_l2_error_reduction"])
        self.assertIsNone(zero["improvement"]["relative_l2_gap_recovery"])
        no_gap = precision(bf16=1.2, c4=1.2, mixed=1.1)
        self.assertIsNone(no_gap["improvement"]["relative_l2_gap_recovery"])

    def test_missing_matching_reference_fails_and_nonfinite_is_not_repaired(self):
        original = np.ones((1, 2, 3))
        outputs = {name: original.copy() for name in reports.ARMS}
        with self.assertRaisesRegex(ValueError, "Matching references missing"):
            reports.build_accuracy_report(outputs, {"installed_bf16": original}, original)
        outputs["w128_split4"][0, 0, 0] = np.nan
        data = reports.build_accuracy_report(outputs, {k: original for k in outputs}, original)
        value = data["arms"]["w128_split4"]["vs_original_bf16_math"]
        self.assertFalse(value["metrics_valid"])
        self.assertEqual(value["nonfinite_count"], 1)
        self.assertIsNone(data["improvement"]["relative_l2_error_reduction"])

    def test_summary_prefers_post_timing_and_retains_actual_pair_ratio(self):
        report = dict(status="MEASURED", args={"timing": "graph"}, performance_scope="prepared read only",
            precision_comparison=precision(mixed=1.1), post_timing_precision_comparison=precision(mixed=1.15),
            verification_timings={
                "w128_split4_vs_installed_c4": pair("installed_c4", "w128_split4"),
                "installed_bf16_vs_installed_c4": pair("installed_c4", "installed_bf16", 40., .4)})
        with tempfile.TemporaryDirectory() as directory:
            summary = reports.write_summary(report, directory)
            self.assertEqual(summary["precision_source"], "post_timing_precision_comparison")
            candidate = summary["arms"]["w128_split4"]
            self.assertAlmostEqual(candidate["precision"]["vs_original_bf16_math"]["relative_l2"], .15, places=6)
            timing = candidate["latency"]
            self.assertEqual(timing["median_paired_ratio"], .51)
            self.assertNotEqual(timing["median_paired_ratio"], timing["event_us"] / 100.)
            self.assertEqual(timing["raw_round_count"], 3)
            self.assertEqual(timing["baseline"], "installed_c4")
            self.assertEqual(timing["candidate"], "w128_split4")
            self.assertEqual(timing["pair_role"], "candidate")
            self.assertEqual(summary["arms"]["installed_c4"]["latency"]["pair_role"], "baseline")
            self.assertEqual(json.loads((Path(directory)/"summary.json").read_text()), summary)
            with (Path(directory)/"summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["arm"] for row in rows], list(reports.ARMS))
            self.assertEqual(rows[2]["median_paired_ratio"], "0.51")
            text = (Path(directory)/"summary.txt").read_text()
            self.assertIn("w128_split4 / installed_c4", text)
            self.assertIn("observed extrema", text)
            self.assertNotIn("confidence interval", text)

    def test_failed_or_partial_run_keeps_failure_and_has_no_fabricated_latency(self):
        report = dict(status="CONTROL_FAILED", precision_comparison=precision())
        with tempfile.TemporaryDirectory() as directory:
            summary = reports.write_summary(report, directory)
            self.assertEqual(summary["status"], "CONTROL_FAILED")
            self.assertFalse(summary["pairwise_timings"])
            self.assertTrue(all(entry["latency"] is None for entry in summary["arms"].values()))
            self.assertNotIn("MEASURED", (Path(directory)/"summary.txt").read_text())
            with self.assertRaises(FileExistsError):
                reports.write_summary(report, directory)


if __name__ == "__main__": unittest.main()
