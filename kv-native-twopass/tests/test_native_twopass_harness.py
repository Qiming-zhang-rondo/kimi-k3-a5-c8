"""CPU gates/report tests; these do not establish A5 correctness or speed."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from test_hot_backend_harness import FakeBackend, tiny_case
from kv_window_common import attention_reference
import test_kv_native_twopass_a5 as harness


class NativeHarnessBackend(FakeBackend):
    fault = None

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.prepared, self.calls = [], []

    def baseline(self, kind):
        self.calls.append((kind, "read"))
        return super().baseline(kind)

    def baseline_run(self, kind, write=False):
        output = super().baseline_run(kind, write=write)
        self.calls.append((kind, "pipeline" if write else "read"))
        return output

    def prepare(self, recipe, window, mode="mixed", index_capacity="full", merge="torch"):
        config = (recipe, window, mode, index_capacity, merge)
        self.prepared.append(config)
        name = f"{recipe}_w0" if window == 0 else f"{recipe}_{mode}_w{window}_{index_capacity}_{merge}"
        selected = recipe == "baseline" and window > 0 and index_capacity == "full" and merge == "torch"
        if self.fault == "triton" and window and merge == "triton":
            raise RuntimeError("OPTIONAL_MERGE_COMPILATION_FAILED")
        if self.fault == "device" and selected and mode == "mixed":
            raise RuntimeError("507015: The address for scalar to access GM is invalid")
        cold = self.data["kv"] * (.95 if recipe == "baseline" else .97)
        hot = self.baseline_decoded("c8")
        output = attention_reference(self.data, kv_c4=cold,
            hot_kv=hot if window and mode == "mixed" else None,
            window=window if mode == "mixed" else 0)
        if self.fault == "split_relative" and recipe == "baseline":
            if not window:
                output = output*1.006
            elif selected and mode == "split_c4":
                output = output*.994
        if self.fault == "mixed_output" and selected and mode == "mixed":
            output = output+100

        def read():
            self.calls.append((name, "read"))
            return output

        def run(write=False, dynamic=True):
            self.calls.append((name, "pipeline" if write else "dynamic" if dynamic else "read"))
            if self.fault == "dynamic" and selected and mode == "mixed" and dynamic and not write:
                return output+100
            return output

        def diagnose():
            self.calls.append((name, "diagnose"))
            return {"passed": not (self.fault == "diagnose" and selected and mode == "split_c4"),
                    "scope": "CPU test double, not actual native stats"}

        def stage_call(stage):
            self.calls.append((name, "stage_"+stage))
            return output

        def components():
            return {stage: lambda stage=stage: stage_call(stage)
                    for stage in ("partition", "cold_read", "hot_read", "merge", "cold_write") +
                    (("hot_write",) if mode == "mixed" else ())}

        return types.SimpleNamespace(
            metadata={"window": window, "mode": mode, "index_capacity": index_capacity,
                      "merge": merge, "zero_window_native_bypass": not window},
            decoded_kv=lambda: cold, decoded_hot_kv=lambda: hot,
            call=read, run=run, zero_query_diagnostics=diagnose, component_calls=components)


class NativeHarnessTests(unittest.TestCase):
    def args(self, extra=()):
        return harness.parser().parse_args(["--windows", "0,8", "--timing", "graph",
            "--warmup", "0", "--iterations", "1", "--repeats", "2", "--diagnose"] + list(extra))

    def run_fake(self, directory, args=None, fault=None):
        instances, timed = [], []
        backend_class = type("ConfiguredBackend", (NativeHarnessBackend,), {"fault": fault})
        def factory(data, **kwargs):
            backend = backend_class(data, **kwargs)
            instances.append(backend)
            return backend
        def benchmark(call, **kwargs):
            call()
            timed.append(instances[0].calls[-1])
            return {"event_us": {"median": 10.}, "wall_us": {"median": 12.}}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            report = harness.run_case(tiny_case(), args or self.args(), directory,
                                      backend_factory=factory, benchmark=benchmark)
        return report, instances[0], timed

    def test_full_matrix_preserves_references_controls_dynamic_pipeline_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"case"
            report, backend, timed = self.run_fake(directory, self.args(["--stages"]))
            self.assertEqual(report["status"], "MEASURED")
            self.assertTrue(report["host_input_unchanged"])
            self.assertEqual(report["production_validation"], "NOT_VALIDATED")
            self.assertEqual(set(report["baselines"]), {"bf16", "c8"})
            self.assertEqual(len(report["variants"]), 18)
            with np.load(directory/"outputs.npz") as arrays:
                for entry in report["variants"]:
                    name = entry["name"]
                    self.assertEqual(entry["status"], "MEASURED")
                    self.assertTrue(entry["control_passed"])
                    self.assertEqual(entry["kernel_vs_matching_reference"]["relative_l2"], 0.)
                    self.assertEqual(entry["pipeline_vs_prepared"]["relative_l2"], 0.)
                    if not entry["window"]:
                        self.assertTrue(entry["implementation"]["zero_window_native_bypass"])
                        continue
                    self.assertEqual(entry["dynamic_vs_prepared"]["relative_l2"], 0.)
                    self.assertTrue(entry["zero_query_statistics"]["passed"])
                    self.assertEqual(entry["timing_reference"], entry["recipe"]+"_w0")
                    self.assertEqual(entry["production_performance"], "NOT_VALIDATED")
                    self.assertEqual(entry["implementation"]["index_capacity"], entry["index_capacity"])
                    self.assertIn("partition", entry["stage_timings"])
                    self.assertIn((name, "dynamic"), backend.calls)
                    self.assertIn((name, "pipeline"), timed)
                    self.assertIn((name, "stage_merge"), timed)
                    reference = arrays[name+"_matching_reference"]
                    native_reference = arrays[entry["recipe"]+"_w0_matching_reference"]
                    if entry["mode"] == "split_c4":
                        self.assertEqual(entry["split_vs_native_w0"]["relative_l2"], 0.)
                        np.testing.assert_array_equal(reference, native_reference)
                    else:
                        self.assertFalse(np.array_equal(reference, native_reference))
                    pipeline_timed = [i for i, item in enumerate(timed) if item == (name, "pipeline")]
                    self.assertEqual(len(pipeline_timed), 2)
                    self.assertEqual(timed[pipeline_timed[0]-1], (entry["recipe"]+"_w0", "pipeline"))
                    self.assertEqual(timed[pipeline_timed[1]+1], (entry["recipe"]+"_w0", "pipeline"))
            self.assertEqual(json.loads((directory/"results.json").read_text())["status"], "MEASURED")

    def test_split_vs_native_gate_skips_matching_mixed_but_other_configurations_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, backend, timed = self.run_fake(Path(tmp)/"case", fault="split_relative")
        self.assertEqual(report["status"], "FAILED")
        entries = {entry["name"]: entry for entry in report["variants"]}
        name = "baseline_split_c4_w8_full_torch"
        split = entries[name]
        self.assertLess(split["kernel_vs_matching_reference"]["relative_l2"], .01)
        self.assertGreater(split["split_vs_native_w0"]["relative_l2"], .01)
        self.assertEqual(split["status"], "CONTROL_FAILED")
        self.assertFalse(any(item[0] == name for item in timed))
        self.assertEqual(entries["baseline_mixed_w8_full_torch"]["status"], "CONTROL_FAILED")
        self.assertNotIn(("baseline", 8, "mixed", "full", "torch"), backend.prepared)
        self.assertEqual(entries["baseline_mixed_w8_bounded_torch"]["status"], "MEASURED")
        self.assertEqual(entries["alg2_max7_mixed_w8_full_torch"]["status"], "MEASURED")

    def test_corrupted_mixed_output_or_dynamic_path_is_never_timed(self):
        for fault in ("mixed_output", "dynamic"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                report, backend, timed = self.run_fake(Path(tmp)/"case", fault=fault)
                self.assertEqual(report["status"], "FAILED")
                entries = {entry["name"]: entry for entry in report["variants"]}
                name = "baseline_mixed_w8_full_torch"
                self.assertEqual(entries[name]["status"], "CONTROL_FAILED")
                self.assertNotIn("timing_prepared_read", entries[name])
                self.assertNotIn("timing_write_read", entries[name])
                self.assertFalse(any(item[0] == name for item in timed))
                self.assertEqual(entries["baseline_split_c4_w8_full_torch"]["status"], "MEASURED")
                self.assertEqual(entries["baseline_mixed_w8_bounded_torch"]["status"], "MEASURED")
                if fault == "dynamic":
                    self.assertEqual(entries[name]["kernel_vs_matching_reference"]["relative_l2"], 0.)
                    self.assertEqual(entries[name]["pipeline_vs_matching_reference"]["relative_l2"], 0.)
                    self.assertGreater(entries[name]["dynamic_vs_prepared"]["relative_l2"], .001)

    def test_failed_zero_query_diagnostic_gates_timing_and_matching_mixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, backend, timed = self.run_fake(Path(tmp)/"case", fault="diagnose")
        entries = {entry["name"]: entry for entry in report["variants"]}
        name = "baseline_split_c4_w8_full_torch"
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(entries[name]["kernel_vs_matching_reference"]["relative_l2"], 0.)
        self.assertFalse(entries[name]["zero_query_statistics"]["passed"])
        self.assertEqual(entries[name]["status"], "CONTROL_FAILED")
        self.assertFalse(any(item[0] == name for item in timed))
        self.assertNotIn(("baseline", 8, "mixed", "full", "torch"), backend.prepared)
        self.assertEqual(entries["baseline_mixed_w8_bounded_torch"]["status"], "MEASURED")

    def test_optional_triton_failure_does_not_replace_or_block_later_torch_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, backend, timed = self.run_fake(Path(tmp)/"case",
                self.args(["--merges", "triton,torch"]), fault="triton")
        self.assertEqual(report["status"], "FAILED")
        for entry in report["variants"]:
            if not entry["window"]:
                continue
            if entry["merge"] == "torch":
                self.assertEqual(entry["status"], "MEASURED")
            elif entry["mode"] == "split_c4":
                self.assertEqual(entry["status"], "ERROR")
                self.assertIn("OPTIONAL_MERGE_COMPILATION_FAILED", entry["traceback"])
                self.assertFalse(any(item[0] == entry["name"] for item in timed))
            else:
                self.assertEqual(entry["status"], "CONTROL_FAILED")

    def test_device_fault_stops_remaining_operators_and_main_stops_remaining_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, backend, timed = self.run_fake(Path(tmp)/"case", fault="device")
            self.assertEqual(report["status"], "DEVICE_ERROR")
            self.assertEqual(report["stopped_after_device_fault"], "baseline_mixed_w8_full_torch")
            self.assertEqual(backend.prepared[-1], ("baseline", 8, "mixed", "full", "torch"))
            self.assertFalse(any(config[0] == "alg2_max7" for config in backend.prepared))
            self.assertEqual(report["variants"][-1]["failure_kind"], "DEVICE_EXECUTION_FAULT")

            def failed_case(data, args, directory, progress=None):
                progress(report)
                return report

            root = Path(tmp)/"main"
            with mock.patch.object(harness, "synthetic_case", return_value=tiny_case()), \
                 mock.patch.object(harness, "run_case", side_effect=failed_case) as run_case, \
                 contextlib.redirect_stdout(io.StringIO()):
                code = harness.main(["--keys", "40,80", "--output", str(root)])
            self.assertEqual(code, 1)
            self.assertEqual(run_case.call_count, 1)
            saved = json.loads((root/"results.json").read_text())
            self.assertEqual(saved["status"], "DEVICE_ERROR")
            self.assertEqual(len(saved["cases"]), 1)


class NativeReportTests(unittest.TestCase):
    def test_brief_and_csv_use_actual_paired_full_pipeline_baseline_and_preserve_controls(self):
        standalone = {"graph": {"event_us": {"median": 100.}}}
        paired = {"graph": {"event_us": {"baseline_median": 7., "candidate_median": 21.,
            "median_paired_ratio": 3., "min_paired_ratio": 2.9, "max_paired_ratio": 3.1},
            "observation": "OBSERVED_SLOWDOWN"}}
        metric = {"relative_l2": .08}
        native = {"name": "baseline_w0", "role": "NATIVE_W0", "status": "MEASURED",
                  "timing_prepared_read": standalone, "timing_write_read": standalone}
        candidate = {"name": "baseline_mixed_w8_full_torch", "role": "HOT_WINDOW", "status": "MEASURED",
                     "timing_prepared_read": standalone, "timing_write_read": paired,
                     "vs_bf16_math": metric, "kernel_vs_matching_reference": {"relative_l2": .002},
                     "stage_timings": {"merge": standalone}}
        control = {"name": "baseline_split_c4_w8_full_triton", "role": "SPLIT_CONTROL",
                   "status": "CONTROL_FAILED", "control_passed": False}
        report = {"status": "FAILED", "args": {"timing": "graph"}, "cases": [{"label": "summary",
            "baselines": {"bf16": {"name": "bf16", "role": "BASELINE", "status": "MEASURED"},
                          "c8": {"name": "c8", "role": "BASELINE", "status": "CONTROL_FAILED"}},
            "variants": [native, candidate, control]}]}
        with tempfile.TemporaryDirectory() as tmp:
            brief = harness.write_reports(report, tmp)
            with (Path(tmp)/"summary.csv").open(newline="") as stream:
                rows = {row["name"]: row for row in csv.DictReader(stream)}
            self.assertEqual((Path(tmp)/"brief.txt").read_text(), brief)
            self.assertEqual(json.loads((Path(tmp)/"results.json").read_text())["status"], "FAILED")
        self.assertIn("MEASURED is not a performance pass", brief)
        self.assertIn("standalone stages (do not add/subtract)", brief)
        self.assertIn("baseline_mixed_w8_full_torch | MEASURED | 100.00 | 21.00 | 7.00 | 3.000 [2.900,3.100]", brief)
        self.assertIn("c8 | CONTROL_FAILED", brief)
        self.assertEqual(set(rows), {"bf16", "c8", native["name"], candidate["name"], control["name"]})
        self.assertEqual(rows[native["name"]]["pipeline_us"], "100.0")
        self.assertEqual(rows[native["name"]]["paired_w0_pipeline_us"], "")
        self.assertEqual(rows[candidate["name"]]["paired_w0_pipeline_us"], "7.0")
        self.assertEqual(rows[candidate["name"]]["pipeline_us"], "21.0")
        self.assertEqual(rows[candidate["name"]]["paired_ratio"], "3.0")
        self.assertEqual(rows[candidate["name"]]["observation"], "OBSERVED_SLOWDOWN")
        self.assertEqual(rows[control["name"]]["pipeline_us"], "")
        self.assertEqual(rows[control["name"]]["status"], "CONTROL_FAILED")
        self.assertTrue(all(row["production_performance"] == "NOT_VALIDATED" for row in rows.values()))


if __name__ == "__main__":
    unittest.main()
