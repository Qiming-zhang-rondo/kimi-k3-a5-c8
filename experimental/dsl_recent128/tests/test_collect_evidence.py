"""Evidence-only collection fixtures; no torch/NPU imports."""
import copy
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collect_evidence as collector


def fixture():
    check = dict(passed=True, finite=True, relative_l2=0.002, worst_row_relative_l2=0.003)
    arm = dict(status="MEASURED", kernel_vs_matching_reference=check,
               output_vs_original_bf16=dict(finite=True, relative_l2=0.11),
               quantization_only_vs_original_bf16=dict(finite=True, relative_l2=0.11),
               post_timing_control=check,
               capture=dict(kind="graph", calls_per_graph=1, poison_control=check))
    rounds = []
    for i in range(10):
        base, candidate = 20 + i, 22 + i
        rounds.append(dict(round=i, order=["baseline", "candidate"] if i % 2 == 0 else ["candidate", "baseline"],
                           baseline=dict(event_us=base, wall_us=base+1),
                           candidate=dict(event_us=candidate, wall_us=candidate+1),
                           event_us_ratio=candidate/base))
    return dict(version=collector.RUNNER_VERSION, status="MEASURED", scope="prepared read",
                args=dict(repeats=10, timing="graph", window=128),
                shape=dict(topk=512, hot_counts=[4, 8, 9, 11]),
                environment=dict(device="Ascend950"), implementation=dict(source="fixture"),
                native_binary_policy=dict(mode="require"),
                native_binary_report=dict(events=[dict(event="installed_hit")]),
                arms={"fp4_w0": copy.deepcopy(arm), "fp4_fp8_w128": copy.deepcopy(arm)},
                controls={"zero_query_mixed": check, "hot_only_with_zero_cold_length": check},
                paired=dict(rounds=rounds, baseline_median=statistics.median([r["baseline"]["event_us"] for r in rounds]),
                            candidate_median=statistics.median([r["candidate"]["event_us"] for r in rounds]),
                            median_paired_ratio=statistics.median([r["event_us_ratio"] for r in rounds])),
                unknown_original_field=dict(preserve="verbatim"))


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.results = self.run / "results"
        self.results.mkdir()
        (self.run / "launcher.json").write_text(json.dumps(dict(payload_commit="a" * 40)))

    def write(self, value):
        path = self.results / "results.json"
        path.write_text(json.dumps(value))
        return path

    def test_full_raw_fields_and_rounds_survive_and_statistics_stay_distinct(self):
        raw = fixture()
        self.write(raw)
        got = collector.collect(self.run)
        self.assertEqual(got["evidence_status"], collector.PRESENT)
        self.assertEqual(got["evidence_issues"], [])
        for key, value in raw.items():
            self.assertEqual(got[key], value)
        self.assertEqual(got["collection"]["launcher"]["payload_commit"], "a" * 40)
        self.assertAlmostEqual(got["computed"]["hot_selection"]["mean_fraction"], 0.015625)
        self.assertNotEqual(got["computed"]["timing"]["ratio_of_medians"],
                            got["computed"]["timing"]["reported_median_paired_ratio"])

    def test_incomplete_summary_and_failed_control_are_not_upgraded(self):
        for mutate in (lambda x: x.pop("controls"),
                       lambda x: x["paired"].pop("rounds"),
                       lambda x: x["controls"]["zero_query_mixed"].update(passed=False),
                       lambda x: x.update(version="hand-written-summary")):
            raw = fixture()
            mutate(raw)
            got = collector.collect(self.write(raw))
            self.assertEqual(got["evidence_status"], "INCOMPLETE")
            self.assertTrue(got["evidence_issues"])

    def test_invalid_and_nonfinite_json_fail_gracefully(self):
        for text in ('{status: MEASURED}', '{"v": NaN}', '{"v": 1e1000}', '[]'):
            path = self.results / "results.json"
            path.write_text(text)
            got = collector.collect(path)
            self.assertEqual(got["evidence_status"], "INCOMPLETE")
            json.dumps(got, allow_nan=False)

    def test_malformed_nested_types_are_preserved_without_crashing(self):
        for mutate in (lambda x: x.update(native_binary_policy=None),
                       lambda x: x["arms"]["fp4_w0"].update(output_vs_original_bf16=[]),
                       lambda x: x["paired"].update(rounds=[None] * 10)):
            raw = fixture()
            mutate(raw)
            got = collector.collect(self.write(raw))
            self.assertEqual(got["evidence_status"], "INCOMPLETE")
            json.dumps(got, allow_nan=False)

    def test_missing_input_and_output_overwrite_are_safe(self):
        self.assertEqual(collector.collect(self.results / "missing.json")["evidence_status"], "INCOMPLETE")
        path = self.write(fixture())
        original = path.read_bytes()
        self.assertEqual(collector.main([str(path), "--output", str(path)]), 2)
        self.assertEqual(path.read_bytes(), original)
        target = self.run / "new-evidence-dir" / "evidence.json"
        self.assertEqual(collector.main([str(path), "--output", str(target)]), 0)
        self.assertEqual(json.loads(target.read_text())["evidence_status"], collector.PRESENT)


if __name__ == "__main__":
    unittest.main()
