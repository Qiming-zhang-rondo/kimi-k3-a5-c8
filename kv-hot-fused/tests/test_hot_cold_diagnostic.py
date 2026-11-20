"""CPU harness regressions; no A5 compilation or performance claim."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

from test_hot_backend_harness import FakeBackend, tiny_case, harness
from kv_window_common import attention_reference


class TracedBackend(FakeBackend):
    """Keep normal references while making every timed path identifiable."""
    cold_failure = None

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.calls = []
        self.prepared_controls = []

    def baseline(self, kind):
        self.calls.append((kind, "read"))
        return super().baseline(kind)

    def baseline_run(self, kind, write=False):
        value = super().baseline_run(kind, write=write)
        self.calls.append((kind, "pipeline" if write else "read"))
        return value

    def prepare(self, recipe, window):
        runner = super().prepare(recipe, window)
        name = f"{recipe}_w{window}"

        def read():
            self.calls.append((name, "read"))
            return runner.call()

        def run(write=False):
            self.calls.append((name, "pipeline" if write else "read"))
            return runner.run(write=write)

        return types.SimpleNamespace(metadata=runner.metadata, call=read, run=run,
                                     decoded_kv=runner.decoded_kv,
                                     decoded_hot_kv=runner.decoded_hot_kv)

    def prepare_cold_control(self, recipe, window):
        self.prepared_controls.append((recipe, window))
        if self.cold_failure == "prepare":
            raise RuntimeError("CUSTOM_COLD_COMPILATION_FAILED")
        cold = self.data["kv"] * (.95 if recipe == "baseline" else .97)
        output = attention_reference(self.data, kv_c4=cold)
        if self.cold_failure == "wrong_output":
            output = output + 100

        def read():
            self.calls.append((f"{recipe}_fused_cold_control", "read"))
            return output

        def forbidden(*args, **kwargs):
            raise AssertionError("A cold diagnostic must never write or use hot KV")

        return types.SimpleNamespace(
            metadata={"window": window, "effective_window": 0,
                      "diagnostic_cold_only": True},
            call=read, run=forbidden, decoded_kv=lambda: cold,
            decoded_hot_kv=forbidden)


class ColdDiagnosticTests(unittest.TestCase):
    def args(self, enabled=True, windows="16,0,8"):
        flags = ["--timing", "graph", "--windows", windows,
                 "--repeats", "2", "--warmup", "0", "--iterations", "1"]
        return harness.parser().parse_args(flags + (["--fused-cold-control"] if enabled else []))

    def run_traced(self, directory, args, backend_class=TracedBackend):
        instances, timed = [], []

        def factory(data, **kwargs):
            backend = backend_class(data, **kwargs)
            instances.append(backend)
            return backend

        def benchmark(call, **kwargs):
            call()
            timed.append(instances[0].calls[-1])
            return {"event_us": {"median": 10.}, "wall_us": {"median": 12.}}

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            report = harness.run_case(tiny_case(), args, directory,
                                      backend_factory=factory, benchmark=benchmark)
        return report, instances[0], timed

    def test_one_read_only_control_per_recipe_uses_cold_reference_and_native_pairs(self):
        args = self.args()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "case"
            report, backend, timed = self.run_traced(directory, args)
            self.assertEqual(report["status"], "MEASURED")
            self.assertEqual(backend.prepared_controls, [("baseline", 8), ("alg2_max7", 8)])
            self.assertEqual(len(report["variants"]), 6)
            self.assertEqual(len(report["diagnostics"]), 2)
            with np.load(directory / "outputs.npz") as arrays:
                for entry in report["diagnostics"]:
                    recipe, name = entry["recipe"], entry["name"]
                    self.assertEqual(entry["window"], 0)
                    self.assertEqual(entry["layout_window"], 8)
                    self.assertEqual(entry["role"], "READ_ONLY_DIAGNOSTIC")
                    self.assertFalse(entry["production_comparison_eligible"])
                    self.assertEqual(entry["production_performance"], "NOT_VALIDATED")
                    self.assertTrue(entry["control_passed"])
                    self.assertEqual(entry["kernel_vs_matching_reference"]["relative_l2"], 0)
                    self.assertNotIn("timing_write_read", entry)
                    self.assertIn(f"{recipe}_w0", entry["timing_reference"])
                    reference = arrays[name + "_matching_reference"]
                    np.testing.assert_array_equal(reference, arrays[f"{recipe}_w0_matching_reference"])
                    self.assertFalse(np.array_equal(reference, arrays[f"{recipe}_w8_matching_reference"]))
                    self.assertNotIn(name + "_write_read", arrays)
                    indices = [i for i, call in enumerate(timed) if call[0] == name]
                    self.assertEqual(len(indices), args.repeats)
                    # AB followed by BA: both partners are native W0 reads.
                    self.assertEqual(timed[indices[0]-1], (f"{recipe}_w0", "read"))
                    self.assertEqual(timed[indices[1]+1], (f"{recipe}_w0", "read"))
                    self.assertTrue(all(timed[i][1] == "read" for i in indices))
                    native = next(e for e in report["variants"] if e["name"] == f"{recipe}_w0")
                    self.assertIn("median", native["timing_write_read"]["graph"]["event_us"])
                    self.assertNotIn("baseline_median", native["timing_write_read"]["graph"]["event_us"])
                    self.assertIn((f"{recipe}_w0", "pipeline"), timed)

    def test_diagnostic_failure_is_not_timed_and_fails_case(self):
        for failure in ("wrong_output", "prepare"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                broken = type("BrokenControl", (TracedBackend,), {"cold_failure": failure})
                report, backend, timed = self.run_traced(Path(tmp)/"case", self.args(), broken)
                self.assertEqual(report["status"], "FAILED")
                self.assertEqual(len(backend.prepared_controls), 2)
                self.assertFalse(any("fused_cold_control" in name for name, _ in timed))
                self.assertTrue(all(e["status"] == "MEASURED" for e in report["variants"]))
                for entry in report["diagnostics"]:
                    self.assertEqual(entry["status"], "CONTROL_FAILED" if failure == "wrong_output" else "ERROR")
                    self.assertNotIn("timing_prepared_read", entry)
                    self.assertNotIn("timing_write_read", entry)
                    if failure == "prepare":
                        self.assertIn("CUSTOM_COLD_COMPILATION_FAILED", entry["traceback"])

    def test_failed_native_controls_prevent_diagnostic_preparation_and_timing(self):
        class WrongNativeW0(TracedBackend):
            corrupt_window = 0

        class WrongBF16(TracedBackend):
            def baseline(self, kind):
                return super().baseline(kind) + (100 if kind == "bf16" else 0)

        for broken in (WrongNativeW0, WrongBF16):
            with self.subTest(backend=broken.__name__), tempfile.TemporaryDirectory() as tmp:
                report, backend, timed = self.run_traced(Path(tmp)/"case", self.args(), broken)
                self.assertEqual(report["status"], "FAILED")
                self.assertEqual(backend.prepared_controls, [])
                self.assertFalse(any("fused_cold_control" in name for name, _ in timed))
                for entry in report["diagnostics"]:
                    self.assertEqual(entry["status"], "CONTROL_FAILED")
                    self.assertFalse(entry["control_passed"])
                    self.assertIn("native", entry["control_failure_reason"])
                    self.assertNotIn("timing_prepared_read", entry)

    def test_no_flag_adds_no_diagnostic_and_changes_no_native_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, backend, timed = self.run_traced(Path(tmp)/"case", self.args(enabled=False))
        self.assertEqual(report["status"], "MEASURED")
        self.assertEqual(report["diagnostics"], [])
        self.assertEqual(backend.prepared_controls, [])
        self.assertEqual(len(report["variants"]), 6)
        self.assertFalse(any("fused_cold_control" in name for name, _ in timed))
        self.assertEqual(set(report["baselines"]), {"bf16", "c8"})

    def test_zero_window_alone_adds_no_diagnostic_even_with_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, backend, timed = self.run_traced(Path(tmp)/"case", self.args(windows="0"))
        self.assertEqual(report["status"], "MEASURED")
        self.assertEqual(report["diagnostics"], [])
        self.assertEqual(backend.prepared_controls, [])
        self.assertEqual(len(report["variants"]), 2)


class DiagnosticReportTests(unittest.TestCase):
    def test_brief_and_csv_preserve_paired_baseline_and_read_only_scope(self):
        standalone = {"graph": {"event_us": {"median": 100.}}}
        read_pair = {"graph": {"event_us": {"baseline_median": 11., "candidate_median": 22.,
                                             "median_paired_ratio": 2., "min_paired_ratio": 1.9,
                                             "max_paired_ratio": 2.1},
                                "observation": "OBSERVED_SLOWDOWN"}}
        pipeline_pair = {"graph": {"event_us": {"baseline_median": 7., "candidate_median": 21.,
                                                 "median_paired_ratio": 3., "min_paired_ratio": 2.9,
                                                 "max_paired_ratio": 3.1},
                                    "observation": "OBSERVED_SLOWDOWN"}}
        metric = {"relative_l2": .08, "max_row_relative_l2": .09}
        native = {"name": "baseline_w0", "status": "MEASURED", "vs_bf16_math": metric,
                  "timing_prepared_read": standalone, "timing_write_read": standalone}
        hot = {"name": "baseline_w8", "status": "MEASURED", "vs_bf16_math": metric,
               "timing_prepared_read": read_pair, "timing_write_read": pipeline_pair,
               "kernel_vs_matching_reference": {"relative_l2": .002}}
        diagnostic = {"name": "baseline_fused_cold_control", "status": "MEASURED",
                      "role": "READ_ONLY_DIAGNOSTIC", "vs_bf16_math": metric,
                      "timing_prepared_read": read_pair}
        report = {"status": "FAILED", "args": {"timing": "graph"}, "cases": [
            {"label": "report_contract", "baselines": {
                "bf16": {"status": "MEASURED", "timing_prepared_read": standalone},
                "c8": {"status": "CONTROL_FAILED"}},
             "variants": [native, hot], "diagnostics": [diagnostic]}]}
        with tempfile.TemporaryDirectory() as tmp:
            brief = harness.write_brief(report, tmp)
            harness.write_summary(report, tmp)
            self.assertEqual(brief, (Path(tmp)/"brief.txt").read_text())
            with (Path(tmp)/"summary.csv").open(newline="") as stream:
                rows = {row["name"]: row for row in csv.DictReader(stream)}
        self.assertIn("bf16=MEASURED c8=CONTROL_FAILED", brief)
        self.assertIn("baseline_w8 | MEASURED | 22.00 | 2.000 | 21.00 | 7.00 | 3.000 [2.900,3.100]", brief)
        self.assertIn("baseline_fused_cold_control | MEASURED | 22.00 | 2.000 | NA | NA | NA", brief)
        self.assertIn("MEASURED means experiment completed, not that performance passed.", brief)
        self.assertEqual(set(rows), {"bf16", "c8", "baseline_w0", "baseline_w8", "baseline_fused_cold_control"})
        self.assertEqual(rows["c8"]["status"], "CONTROL_FAILED")
        self.assertEqual(rows["baseline_w0"]["write_read_us"], "100.0")
        self.assertEqual(rows["baseline_w8"]["write_read_us"], "21.0")
        self.assertEqual(rows["baseline_w8"]["write_read_paired_w0_us"], "7.0")
        self.assertEqual(rows["baseline_w8"]["read_paired_w0_us"], "11.0")
        self.assertEqual(rows["baseline_w8"]["write_read_paired_ratio_vs_same_recipe_w0"], "3.0")
        self.assertEqual(rows["baseline_w8"]["write_read_min_paired_ratio"], "2.9")
        self.assertEqual(rows["baseline_w8"]["write_read_max_paired_ratio"], "3.1")
        self.assertEqual(rows["baseline_w8"]["matching_reference_l2"], "0.002")
        control = rows["baseline_fused_cold_control"]
        self.assertEqual(control["role"], "READ_ONLY_DIAGNOSTIC")
        self.assertEqual(control["write_read_us"], "")
        self.assertEqual(control["write_read_paired_w0_us"], "")
        self.assertEqual(control["write_read_paired_ratio_vs_same_recipe_w0"], "")
        self.assertTrue(all(row["production_performance"] == "NOT_VALIDATED" for row in rows.values()))


if __name__ == "__main__":
    unittest.main()
