"""CPU orchestration tests; no claim of BF16/C8 A5 correctness or latency."""
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
from kv_window_common import attention_reference, synthetic_case, validate_case
import test_kv_bf16_hot_ab_a5 as harness


class FakeABBackend(FakeBackend):
    def __init__(self, data, fault=None, **kwargs):
        super().__init__(data, **kwargs)
        self.fault, self.calls, self.prepared = fault, [], []

    def baseline(self, kind):
        self.calls.append(("full_"+kind, "read"))
        return super().baseline(kind)

    def baseline_run(self, kind, write=False):
        output = super().baseline(kind)
        self.calls.append(("full_"+kind, "pipeline"))
        return output

    def prepare(self, recipe, window, mode="mixed", writer="native", **options):
        name = "c4_w0" if not window else "split_c4" if mode == "split_c4" else "c8_"+writer
        self.prepared.append((name, recipe, window, mode, writer, options))
        return self.runner(name, window, mode, writer, "c8")

    def bf16_runner(self, backend, recipe, window, writer):
        assert backend is self and recipe == "baseline" and window == 128
        name = "bf16_"+writer
        self.prepared.append((name, recipe, window, "mixed", writer, {}))
        return self.runner(name, window, "mixed", writer, "bf16")

    def runner(self, name, window, mode, writer, dtype):
        if self.fault == "c8_compile" and name == "c8_hot_store_partition":
            raise RuntimeError("OPTIONAL_C8_FUSED_COMPILE_FAILURE")
        if self.fault == "device" and name == "bf16_native":
            raise RuntimeError("507015: The address for scalar to access GM is invalid")
        cold, hot = self.data["kv"]*.95, self.baseline_decoded(dtype)
        output = attention_reference(self.data, kv_c4=cold, hot_kv=hot,
                                     window=window if mode != "split_c4" else 0)
        if self.fault == "split" and name == "split_c4":
            output = output+100
        def read():
            self.calls.append((name, "read"))
            return output
        def run(write=False, dynamic=True):
            self.calls.append((name, "pipeline" if write else "dynamic"))
            if self.fault == "dynamic" and name == "bf16_native" and dynamic and not write:
                return output+100
            return output
        def diagnostic(field):
            self.calls.append((name, field))
            return {"passed": not (self.fault == field and name == "bf16_native"),
                    "raw_evidence": {"field": field, "threshold_unchanged": True}}
        def reader_diagnostics(**kwargs):
            self.calls.append((name, "reader"))
            return {"passed": not (self.fault == "reader" and name == "bf16_native"),
                    "arguments": kwargs, "independent_hot_math": True}
        def stage(stage_name):
            self.calls.append((name, "stage_"+stage_name))
            return output
        stage_names = ("cold_read", "hot_read", "merge", "cold_write") + (
            ("hot_prepare",) if writer == "hot_store_partition" else ("partition", "hot_write"))
        return types.SimpleNamespace(metadata={"reader": "native", "hot_dtype": dtype, "writer": writer,
                "cold_writer_included": True, "compact_hot_call_count": 1 if dtype == "bf16" else None},
            decoded_kv=lambda: cold, decoded_hot_kv=lambda: hot, call=read, run=run,
            partition_diagnostics=lambda: diagnostic("partition"), writer_diagnostics=lambda: diagnostic("writer"),
            zero_query_diagnostics=lambda: diagnostic("zero_query"), reader_diagnostics=reader_diagnostics,
            component_calls=lambda: {key: lambda key=key: stage(key) for key in stage_names})


class Bf16HotHarnessTests(unittest.TestCase):
    def run_fake(self, directory, fault=None, timing="graph"):
        data, instances, timed = tiny_case(), [], []
        args = harness.parser().parse_args(["--timing", timing, "--warmup", "0", "--iterations", "1", "--repeats", "2"])
        def factory(data, **kwargs):
            backend = FakeABBackend(data, fault=fault, **kwargs)
            instances.append(backend)
            return backend
        def bf16_factory(backend, **kwargs):
            return backend.bf16_runner(backend, **kwargs)
        def benchmark(call, **kwargs):
            call()
            name, kind = instances[0].calls[-1]
            timed.append((name, kind, kwargs))
            latency = {"c4_w0": 5., "c8_native": 10., "bf16_native": 12.,
                       "c8_hot_store_partition": 9., "bf16_hot_store_partition": 8.}.get(name, 10.)
            return {"event_us": {"median": latency, "samples": [latency]},
                    "wall_us": {"median": latency+1, "samples": [latency+1]}}
        with mock.patch.object(harness, "validate_fixed_case", side_effect=validate_case), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            case = harness.run_case(data, args, directory, backend_factory=factory,
                                    bf16_runner_factory=bf16_factory, benchmark=benchmark)
        return case, instances[0], timed, args

    def test_all_eight_rows_have_independent_references_and_same_writer_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, backend, timed, args = self.run_fake(Path(tmp)/"case")
            with np.load(Path(tmp)/"case"/"outputs.npz") as saved:
                self.assertIn("bf16_native_matching_reference", saved.files)
                self.assertIn("c8_native_matching_reference", saved.files)
                self.assertFalse(np.array_equal(saved["bf16_native_matching_reference"], saved["c8_native_matching_reference"]))
        entries = {entry["name"]: entry for entry in harness.all_entries(case)}
        self.assertEqual(list(entries), list(harness.NAMES))
        self.assertEqual(case["status"], "MEASURED")
        self.assertTrue(case["host_input_unchanged"])
        self.assertTrue(all(entry["control_passed"] for entry in entries.values()))
        for name, reference, ratio in (("bf16_native", "c8_native", 1.2),
                                       ("bf16_hot_store_partition", "c8_hot_store_partition", 8/9)):
            entry = entries[name]
            self.assertEqual(entry["same_writer_c8_reference"], reference)
            self.assertEqual(entry["timing_reference"], "c4_w0")
            pair = entry["timing_vs_same_writer_c8"]["graph"]
            self.assertEqual(pair["event_us"]["median_paired_ratio"], ratio)
            self.assertEqual([row["order"] for row in pair["rounds"]], [["baseline", "candidate"], ["candidate", "baseline"]])
            self.assertIn("cold_write", entry["stage_timings"])
            self.assertTrue(entry["reader_consistency"]["independent_hot_math"])
            self.assertEqual(entry["reader_consistency"]["arguments"], {"control_l2": .01, "pipeline_l2": .001})
        self.assertNotIn("reader_consistency", entries["c8_native"])
        self.assertEqual(len([item for item in backend.prepared if item[0] == "split_c4"]), 1)
        self.assertTrue(all(options["graph"] for _, _, options in timed))
        self.assertEqual(entries["split_c4"]["split_vs_native_w0"]["relative_l2"], 0.)

    def test_independent_hot_partition_writer_zero_and_dynamic_failures_never_get_timed(self):
        for fault in ("reader", "partition", "writer", "zero_query", "dynamic"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                case, backend, timed, _ = self.run_fake(Path(tmp)/"case", fault=fault)
            entries = {entry["name"]: entry for entry in harness.all_entries(case)}
            self.assertEqual(case["status"], "FAILED")
            self.assertEqual(entries["bf16_native"]["status"], "CONTROL_FAILED")
            self.assertNotIn("timing_write_read", entries["bf16_native"])
            self.assertFalse(any(name == "bf16_native" for name, _, _ in timed))
            self.assertEqual(entries["bf16_hot_store_partition"]["status"], "MEASURED")
            self.assertIn("implementation", entries["bf16_native"])

    def test_failed_split_blocks_every_mixed_candidate_and_failed_c8_reference_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, backend, timed, _ = self.run_fake(Path(tmp)/"split", fault="split")
        entries = {entry["name"]: entry for entry in harness.all_entries(case)}
        for name in harness.NAMES[4:]:
            self.assertEqual(entries[name]["status"], "CONTROL_FAILED")
            self.assertFalse(any(call[0] == name for call in backend.prepared))
            self.assertFalse(any(t[0] == name for t in timed))
        with tempfile.TemporaryDirectory() as tmp:
            case, backend, timed, _ = self.run_fake(Path(tmp)/"compile", fault="c8_compile")
        entries = {entry["name"]: entry for entry in harness.all_entries(case)}
        self.assertEqual(entries["c8_hot_store_partition"]["status"], "ERROR")
        self.assertEqual(entries["bf16_hot_store_partition"]["status"], "REFERENCE_UNAVAILABLE")
        self.assertFalse(any(name == "bf16_hot_store_partition" for name, _, _ in timed))
        self.assertEqual(entries["bf16_native"]["status"], "MEASURED")

    def test_device_fault_stops_all_remaining_operators_and_all_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, backend, timed, _ = self.run_fake(Path(tmp)/"case", fault="device")
        entries = {entry["name"]: entry for entry in harness.all_entries(case)}
        self.assertEqual(case["status"], "DEVICE_ERROR")
        self.assertEqual(case["stopped_after_device_fault"], "bf16_native")
        self.assertEqual(entries["bf16_native"]["failure_kind"], "DEVICE_EXECUTION_FAULT")
        self.assertEqual(entries["bf16_hot_store_partition"]["status"], "NOT_RUN_DEVICE_FAULT")
        self.assertFalse(any(item[0] == "bf16_hot_store_partition" for item in backend.prepared))
        self.assertEqual(timed, [])

    def test_validation_only_keeps_checks_and_emits_no_timing_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, _, timed, _ = self.run_fake(Path(tmp)/"case", timing="none")
        self.assertEqual(case["status"], "VALIDATED_ONLY")
        self.assertEqual(timed, [])
        for entry in harness.all_entries(case):
            self.assertEqual(entry["status"], "VALIDATED_ONLY")
            self.assertEqual(entry["timing_write_read"], {})

    def test_report_preserves_raw_pairs_and_writes_their_own_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, _, _, args = self.run_fake(Path(tmp)/"case")
            entry = next(entry for entry in case["variants"] if entry["name"] == "bf16_native")
            pair = entry["timing_vs_same_writer_c8"]["graph"]["event_us"]
            pair.update(baseline_median=100., candidate_median=115., median_paired_ratio=1.2)
            report = {"version": harness.VERSION, "status": case["status"], "args": {"timing": "graph"}, "cases": [case]}
            before = copy.deepcopy(report)
            text = harness.write_reports(report, tmp)
            with (Path(tmp)/"summary.csv").open() as stream:
                rows = {row["name"]: row for row in csv.DictReader(stream)}
            self.assertEqual(json.loads((Path(tmp)/"results.json").read_text()), report)
        self.assertEqual(report, before)
        self.assertEqual(rows["bf16_native"]["pipeline_us"], "12.0")
        self.assertEqual(rows["bf16_native"]["paired_w0_us"], "5.0")
        self.assertEqual(rows["bf16_native"]["c8_pair_baseline_us"], "100.0")
        self.assertEqual(rows["bf16_native"]["c8_pair_candidate_us"], "115.0")
        self.assertEqual(rows["bf16_native"]["bf16_vs_c8_ratio"], "1.2")
        self.assertIn("not confidence bounds", text)
        self.assertIn("Stages are not additive", text)
        self.assertIn("entries=8", text)

    def test_fixed_shape_all_hot_hits_and_cli_defaults(self):
        args = harness.parser().parse_args([])
        self.assertEqual(args.repeats, 10)
        self.assertTrue(args.stages)
        self.assertTrue(args.diagnose)
        with self.assertRaises(ValueError):
            harness.validate_fixed_case(tiny_case())
        data = synthetic_case(batch=1, queries=4, keys=8192, heads=64, dim=512, rope_dim=64, topk=2048, page=128)
        harness.validate_fixed_case(data)
        data["indices"][0, 0, 0, -1] = -1
        with self.assertRaisesRegex(ValueError, "128 recent"):
            harness.validate_fixed_case(data)
        for argv in (["--queries", "1"], ["--window", "512"], ["--recipe", "alg2_max7"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                harness.parser().parse_args(argv)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(harness.main(["--self-test"]), 0)
        self.assertIn("no A5 execution", out.getvalue())


if __name__ == "__main__":
    unittest.main()
