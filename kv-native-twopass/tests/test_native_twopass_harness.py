"""CPU gates/report tests; these do not establish A5 correctness or speed."""
import contextlib
import copy
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

    def prepare(self, recipe, window, mode="mixed", index_capacity="full", merge="torch",
                partition="argsort_int", writer="native"):
        config = (recipe, window, mode, index_capacity, merge, partition)
        if writer != "native":
            config += (writer,)
        self.prepared.append(config)
        name = f"{recipe}_w0" if window == 0 else f"{recipe}_{mode}_w{window}_{index_capacity}_{merge}"
        if window and partition != "argsort_int":
            name += "_" + partition
        if window and writer != "native":
            name += "_" + writer
        selected = recipe == "baseline" and window > 0 and index_capacity == "full" and merge == "torch"
        if self.fault == "triton" and window and merge == "triton":
            raise RuntimeError("OPTIONAL_MERGE_COMPILATION_FAILED")
        if self.fault == "device" and selected and mode == "mixed":
            raise RuntimeError("507015: The address for scalar to access GM is invalid")
        if self.fault == "new_merge" and merge == "triton_direct512":
            raise RuntimeError("CANDIDATE_MERGE_COMPILATION_FAILED")
        if (self.fault == "champion" and window and recipe == "baseline" and
                partition == "triton_scan" and merge == "triton" and writer == "native"):
            raise RuntimeError("CHAMPION_FAILED_FOR_TEST")
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

        def partition_diagnostics():
            self.calls.append((name, "partition_diagnostics"))
            return {"passed": not (self.fault == "partition" and selected
                                    and partition == "argsort_float" and mode == "split_c4")}

        def writer_diagnostics():
            self.calls.append((name, "writer_diagnostics"))
            bad = (self.fault == "writer" and recipe == "baseline" and mode == "split_c4" and
                   writer == "fused_store" and partition == "triton_scan" and merge == "triton")
            return {"passed": not bad, "scope": "exact-cache CPU test double", "changed_bytes": int(bad)}

        def stage_call(stage):
            self.calls.append((name, "stage_"+stage))
            return output

        def components():
            return {stage: lambda stage=stage: stage_call(stage)
                    for stage in ("partition", "cold_read", "hot_read", "merge", "cold_write") +
                    (("hot_write",) if mode == "mixed" else ())}

        runner = types.SimpleNamespace(
            metadata={"window": window, "mode": mode, "index_capacity": index_capacity,
                      "merge": merge, "partition": partition, "writer": writer, "zero_window_native_bypass": not window},
            decoded_kv=lambda: cold, decoded_hot_kv=lambda: hot,
            call=read, run=run, zero_query_diagnostics=diagnose, component_calls=components,
            partition_diagnostics=partition_diagnostics, writer_diagnostics=writer_diagnostics)
        if self.fault == "writer_missing" and writer != "native":
            del runner.writer_diagnostics
        return runner


class NativeHarnessTests(unittest.TestCase):
    def args(self, extra=()):
        return harness.parser().parse_args(["--windows", "0,8", "--timing", "graph",
            "--warmup", "0", "--iterations", "1", "--repeats", "2", "--diagnose",
            "--partitions", "argsort_int"] + list(extra))

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
        self.assertNotIn(("baseline", 8, "mixed", "full", "torch", "argsort_int"), backend.prepared)
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
        self.assertNotIn(("baseline", 8, "mixed", "full", "torch", "argsort_int"), backend.prepared)
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

    def test_partition_mismatch_gates_its_split_and_mixed_without_blocking_other_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, backend, timed = self.run_fake(Path(tmp)/"case", self.args([
                "--partitions", "argsort_int,argsort_float,triton_scan",
                "--index-capacities", "full", "--merges", "torch"]), fault="partition")
        entries = {entry["name"]: entry for entry in report["variants"]}
        split_name = "baseline_split_c4_w8_full_torch_argsort_float"
        mixed_name = "baseline_mixed_w8_full_torch_argsort_float"
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(entries[split_name]["status"], "CONTROL_FAILED")
        self.assertFalse(entries[split_name]["partition_consistency"]["passed"])
        self.assertNotIn("kernel_vs_matching_reference", entries[split_name])
        self.assertNotIn((split_name, "read"), backend.calls)
        self.assertEqual(entries[mixed_name]["status"], "CONTROL_FAILED")
        self.assertNotIn(("baseline", 8, "mixed", "full", "torch", "argsort_float"), backend.prepared)
        self.assertFalse(any(name in (split_name, mixed_name) for name, _ in timed))
        for name in ("baseline_mixed_w8_full_torch",
                     "baseline_mixed_w8_full_torch_triton_scan",
                     "alg2_max7_mixed_w8_full_torch_argsort_float"):
            self.assertEqual(entries[name]["status"], "MEASURED")

    def test_partition_timings_pair_same_recipe_mode_capacity_and_merge(self):
        # Request the integer baseline last; execution must put it first and
        # never compare a mixed candidate against the cheaper split-C4 control.
        with tempfile.TemporaryDirectory() as tmp:
            report, backend, timed = self.run_fake(Path(tmp)/"case", self.args([
                "--partitions", "triton_scan,argsort_float,argsort_int",
                "--index-capacities", "full", "--merges", "torch"]))
        self.assertEqual(report["status"], "MEASURED")
        entries = {entry["name"]: entry for entry in report["variants"]}
        for entry in report["variants"]:
            if not entry["window"] or entry["partition"] == "argsort_int":
                continue
            reference_name = f"{entry['recipe']}_{entry['mode']}_w8_full_torch"
            reference = entries[reference_name]
            self.assertEqual(entry["partition_timing_reference"], reference_name)
            for field in ("recipe", "window", "mode", "index_capacity", "merge"):
                self.assertEqual(entry[field], reference[field])
            self.assertEqual(reference["partition"], "argsort_int")
            result = entry["timing_vs_argsort_int"]["graph"]
            self.assertEqual(result["event_us"]["median_paired_ratio"], 1.)
            self.assertEqual([r["order"] for r in result["rounds"]],
                             [["baseline", "candidate"], ["candidate", "baseline"]])
            calls = [i for i, call in enumerate(timed) if call == (entry["name"], "pipeline")]
            self.assertEqual(len(calls), 4)  # Two W0 pairs, then two same-path pairs.
            self.assertEqual(timed[calls[2]-1], (reference_name, "pipeline"))
            self.assertEqual(timed[calls[3]+1], (reference_name, "pipeline"))
        for recipe in ("baseline", "alg2_max7"):
            positive = [config for config in backend.prepared if config[0] == recipe and config[1]]
            self.assertEqual(positive[0][-1], "argsort_int")

    def test_batch_and_query_sweeps_remain_distinct_reported_shapes(self):
        observed = []
        def validated_case(data, args, directory, progress=None):
            observed.append((data["q_nope"].shape[0], data["q_nope"].shape[1]))
            report = dict(label=data["label"], status="VALIDATED_ONLY", baselines={}, variants=[],
                          shape={name: list(data[name].shape) for name in harness.ARRAY_KEYS})
            progress(report)
            return report

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"sweep"
            with mock.patch.object(harness, "run_case", side_effect=validated_case), \
                 contextlib.redirect_stdout(io.StringIO()):
                status = harness.main(["--batch", "7", "--batches", "1,3", "--queries", "1,4",
                    "--keys", "40", "--heads", "2", "--topk", "16", "--rope-dim", "0",
                    "--partitions", "argsort_int", "--timing", "none", "--output", str(directory)])
            saved = json.loads((directory/"results.json").read_text())
        self.assertEqual(status, 0)
        self.assertEqual(observed, [(1,1), (1,4), (3,1), (3,4)])
        self.assertEqual([tuple(case["shape"]["q_nope"][:2]) for case in saved["cases"]], observed)
        self.assertEqual(len({case["label"] for case in saved["cases"]}), 4)
        self.assertEqual(saved["args"]["batches"], [1,3])

    def test_optimization_sweep_is_precise_and_pairs_same_mode_champion_without_replacing_w0(self):
        args = self.args(["--optimization-sweep"])
        configs = harness.optimization_configurations()
        self.assertEqual(len(configs), 7)
        self.assertEqual(configs[0], {"merge": "triton", "partition": "triton_scan", "writer": "native",
                                    "optimization": "champion", "changed_knobs": [], "champion": True})
        self.assertEqual([len(config["changed_knobs"]) for config in configs], [0,1,1,1,1,3,3])
        with tempfile.TemporaryDirectory() as tmp:
            report, backend, timed = self.run_fake(Path(tmp)/"case", args)
        self.assertEqual(report["status"], "MEASURED")
        self.assertEqual(len(report["variants"]), 30)
        self.assertEqual(len(report["configuration_plan"]), 7)
        self.assertTrue(all(config["index_capacity"] == "bounded" for config in report["configuration_plan"]))
        entries = {entry["name"]: entry for entry in report["variants"]}
        for entry in report["variants"]:
            self.assertEqual(entry["status"], "MEASURED")
            if not entry["window"]:
                self.assertEqual(entry["role"], "NATIVE_W0")
                self.assertTrue(entry["implementation"]["zero_window_native_bypass"])
                self.assertNotIn("timing_vs_champion", entry)
                continue
            self.assertTrue(entry["writer_consistency"]["passed"])
            self.assertEqual(entry["index_capacity"], "bounded")
            self.assertNotIn("timing_vs_argsort_int", entry)
            if entry["comparison_role"] == "CHAMPION":
                self.assertNotIn("timing_vs_champion", entry)
                continue
            champion = entries[entry["champion_timing_reference"]]
            self.assertEqual(champion["comparison_role"], "CHAMPION")
            for field in ("recipe", "window", "mode", "index_capacity"):
                self.assertEqual(entry[field], champion[field])
            self.assertEqual(entry["timing_reference"], entry["recipe"]+"_w0")
            self.assertEqual(entry["timing_vs_champion"]["graph"]["event_us"]["median_paired_ratio"], 1.)
            calls = [i for i, call in enumerate(timed) if call == (entry["name"], "pipeline")]
            self.assertEqual(len(calls), 4)
            self.assertEqual(timed[calls[2]-1], (champion["name"], "pipeline"))
            self.assertEqual(timed[calls[3]+1], (champion["name"], "pipeline"))
            self.assertEqual(entry["production_performance"], "NOT_VALIDATED")

    def test_writer_byte_failure_or_missing_diagnostic_is_never_timed(self):
        for fault in ("writer", "writer_missing"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                report, backend, timed = self.run_fake(Path(tmp)/"case", self.args(["--optimization-sweep"]), fault=fault)
            self.assertEqual(report["status"], "FAILED")
            for entry in report["variants"]:
                if not entry["window"] or entry["writer"] != "fused_store":
                    self.assertEqual(entry["status"], "MEASURED")
                    continue
                should_fail = fault == "writer_missing" or (entry["recipe"] == "baseline" and entry["optimization"] == "writer_only")
                if should_fail:
                    self.assertEqual(entry["status"], "CONTROL_FAILED")
                    self.assertNotIn("kernel_vs_matching_reference", entry)
                    self.assertFalse(any(name == entry["name"] for name, _ in timed))
                    self.assertNotIn((entry["name"], "read"), backend.calls)
                else:
                    self.assertEqual(entry["status"], "MEASURED")

    def test_optional_candidate_or_champion_failure_does_not_stop_independent_configs(self):
        for fault in ("new_merge", "champion"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                report, backend, timed = self.run_fake(Path(tmp)/"case", self.args(["--optimization-sweep"]), fault=fault)
            self.assertEqual(report["status"], "FAILED")
            for entry in report["variants"]:
                if not entry["window"]:
                    self.assertEqual(entry["status"], "MEASURED")
                    continue
                failed = entry["merge"] == "triton_direct512" if fault == "new_merge" else (
                    entry["recipe"] == "baseline" and entry["optimization"] == "champion")
                if failed:
                    self.assertEqual(entry["status"], "ERROR" if entry["mode"] == "split_c4" else "CONTROL_FAILED")
                    self.assertFalse(any(name == entry["name"] for name, _ in timed))
                else:
                    self.assertEqual(entry["status"], "MEASURED")
                    if fault == "champion" and entry["recipe"] == "baseline":
                        self.assertNotIn("timing_vs_champion", entry)
                        self.assertIn("UNAVAILABLE", entry["champion_timing_comparison"])

    def test_device_fault_stops_remaining_operators_and_main_stops_remaining_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, backend, timed = self.run_fake(Path(tmp)/"case", fault="device")
            self.assertEqual(report["status"], "DEVICE_ERROR")
            self.assertEqual(report["stopped_after_device_fault"], "baseline_mixed_w8_full_torch")
            self.assertEqual(backend.prepared[-1], ("baseline", 8, "mixed", "full", "torch", "argsort_int"))
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
    def test_short_summary_selects_only_measured_mixed_by_same_mode_paired_ratio(self):
        def entry(name, recipe, mode, ratio, candidate_us, baseline_us, status="MEASURED", control=True):
            return dict(name=name, recipe=recipe, mode=mode, role="HOT_WINDOW" if mode == "mixed" else "SPLIT_CONTROL",
                status=status, control_passed=control, optimization=name, window=128, index_capacity="bounded",
                partition="triton_scan", merge="triton", writer="native", vs_bf16_math={"relative_l2": .08},
                timing_write_read={"graph": {"status": "MEASURED", "event_us": {
                    "median_paired_ratio": ratio, "candidate_median": candidate_us, "baseline_median": baseline_us}}})
        better_absolute = entry("lower_us_but_worse_ratio", "baseline", "mixed", 1.5, 15., 10.)
        best = entry("best_ratio", "baseline", "mixed", 1.2, 24., 20.)
        best["timing_vs_champion"] = {"graph": {"status": "MEASURED", "event_us": {"median_paired_ratio": .8}}}
        failed = entry("failed_fast", "baseline", "mixed", .1, 1., 10., status="CONTROL_FAILED", control=False)
        split = entry("split_fast", "baseline", "split_c4", .2, 2., 10.)
        other_mode = entry("eager_only", "baseline", "mixed", .3, 3., 10.)
        other_mode["timing_write_read"] = {"eager": other_mode["timing_write_read"]["graph"]}
        alg = entry("alg_measured", "alg2_max7", "mixed", 1.1, 22., 20.)
        variants = [better_absolute, best, failed, split, other_mode, alg]
        report = {"status": "FAILED", "args": {"timing": "eager,graph"}, "cases": [
            {"label": "one_shape", "baselines": {}, "variants": variants}]}
        before = copy.deepcopy(report)
        with tempfile.TemporaryDirectory() as tmp:
            harness.write_reports(report, tmp)
            text = (Path(tmp)/"short_summary.txt").read_text()
        self.assertEqual(report, before)
        self.assertLessEqual(len(text.splitlines()), 15)
        self.assertIn("empirical selection, no serving-performance claim", text)
        self.assertIn("recipe=baseline best=best_ratio", text)
        self.assertIn("pipeline_us=24.000 paired_W0_us=20.000 ratio=1.200 champion_ratio=0.800", text)
        self.assertIn("recipe=alg2_max7 best=alg_measured", text)
        self.assertNotIn("best=failed_fast", text)
        self.assertNotIn("best=split_fast", text)
        self.assertNotIn("best=eager_only", text)
        self.assertIn("failed_configurations=1", text)

    def test_short_summary_refuses_partial_timing_or_missing_control(self):
        variants = [dict(name="partial", recipe="baseline", mode="mixed", status="MEASURED", control_passed=True,
                        timing_write_read={"graph": {"status": "ERROR", "event_us": {"median_paired_ratio": .5,
                            "candidate_median": 5., "baseline_median": 10.}}}),
                    dict(name="uncontrolled", recipe="baseline", mode="mixed", status="MEASURED", control_passed=False,
                         timing_write_read={"graph": {"status": "MEASURED", "event_us": {"median_paired_ratio": .4,
                            "candidate_median": 4., "baseline_median": 10.}}})]
        report = {"status": "FAILED", "cases": [{"label": "none", "baselines": {}, "variants": variants}]}
        text = harness.short_summary(report, "graph")
        self.assertIn("best_on_this_run=NONE", text)

    def test_brief_and_csv_use_actual_paired_full_pipeline_baseline_and_preserve_controls(self):
        standalone = {"graph": {"event_us": {"median": 100.}}}
        paired = {"graph": {"event_us": {"baseline_median": 7., "candidate_median": 21.,
            "median_paired_ratio": 3., "min_paired_ratio": 2.9, "max_paired_ratio": 3.1},
            "observation": "OBSERVED_SLOWDOWN"}}
        metric = {"relative_l2": .08}
        native = {"name": "baseline_w0", "role": "NATIVE_W0", "status": "MEASURED",
                  "timing_prepared_read": standalone, "timing_write_read": standalone}
        candidate = {"name": "baseline_mixed_w8_full_torch_argsort_float", "role": "HOT_WINDOW", "status": "MEASURED",
                     "partition": "argsort_float", "partition_timing_reference": "baseline_mixed_w8_full_torch",
                     "timing_vs_argsort_int": {"graph": {"event_us": {"baseline_median": 30.,
                         "candidate_median": 20., "median_paired_ratio": 2/3}}},
                     "timing_prepared_read": standalone, "timing_write_read": paired,
                     "vs_bf16_math": metric, "kernel_vs_matching_reference": {"relative_l2": .002},
                     "stage_timings": {"merge": standalone},
                     "merge": "triton_log512", "writer": "fused_store", "optimization": "stacked_log512",
                     "comparison_role": "CANDIDATE", "changed_knobs": ["partition","merge","writer"],
                     "champion_timing_reference": "same_mode_champion",
                     "timing_vs_champion": {"graph": {"event_us": {"baseline_median": 50.,
                         "candidate_median": 40., "median_paired_ratio": .8,
                         "min_paired_ratio": .79, "max_paired_ratio": .81},
                         "observation": "NO_SLOWDOWN_OBSERVED"}}}
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
        self.assertIn("name | role | partition | status", brief)
        self.assertIn(candidate["name"] + " | HOT_WINDOW | argsort_float | MEASURED | 100.00 | 21.00 | 7.00 | 3.000 [2.900,3.100] | 0.667", brief)
        self.assertIn("c8 | BASELINE | none | CONTROL_FAILED", brief)
        self.assertEqual(set(rows), {"bf16", "c8", native["name"], candidate["name"], control["name"]})
        self.assertEqual(rows[native["name"]]["pipeline_us"], "100.0")
        self.assertEqual(rows[native["name"]]["paired_w0_pipeline_us"], "")
        self.assertEqual(rows[candidate["name"]]["paired_w0_pipeline_us"], "7.0")
        self.assertEqual(rows[candidate["name"]]["pipeline_us"], "21.0")
        self.assertEqual(rows[candidate["name"]]["paired_ratio"], "3.0")
        self.assertEqual(rows[candidate["name"]]["observation"], "OBSERVED_SLOWDOWN")
        self.assertEqual(rows[candidate["name"]]["partition"], "argsort_float")
        self.assertEqual(rows[candidate["name"]]["paired_argsort_int_pipeline_us"], "30.0")
        self.assertEqual(rows[candidate["name"]]["paired_argsort_int_candidate_us"], "20.0")
        self.assertEqual(float(rows[candidate["name"]]["ratio_vs_same_path_argsort_int"]), 2/3)
        self.assertEqual(rows[candidate["name"]]["partition_timing_reference"], "baseline_mixed_w8_full_torch")
        self.assertEqual(rows[candidate["name"]]["paired_champion_pipeline_us"], "50.0")
        self.assertEqual(rows[candidate["name"]]["paired_champion_candidate_us"], "40.0")
        self.assertEqual(rows[candidate["name"]]["ratio_vs_champion"], "0.8")
        self.assertEqual(rows[candidate["name"]]["champion_timing_reference"], "same_mode_champion")
        self.assertEqual(rows[candidate["name"]]["writer"], "fused_store")
        self.assertEqual(rows[candidate["name"]]["changed_knobs"], "partition,merge,writer")
        self.assertEqual(rows[control["name"]]["pipeline_us"], "")
        self.assertEqual(rows[control["name"]]["status"], "CONTROL_FAILED")
        self.assertTrue(all(row["production_performance"] == "NOT_VALIDATED" for row in rows.values()))


if __name__ == "__main__":
    unittest.main()
