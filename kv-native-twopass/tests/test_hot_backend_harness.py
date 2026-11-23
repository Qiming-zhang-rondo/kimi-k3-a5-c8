"""CPU contract tests; these do not claim that the A5 kernel compiled or ran."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hotkv"))

from kv_hot_backend import HotWindowBackend, fixed_ring_plan
from kv_window_common import attention_reference, query_positions, synthetic_case, validate_case
import test_kv_hot_fused_a5 as harness


def tiny_case():
    return synthetic_case(batch=1, queries=2, keys=40, heads=2, dim=64, rope_dim=0, topk=16)


class ArrayTensor:
    """Minimal CPU byte-storage double for the adapter's timed scatter contract."""
    def __init__(self, value):
        self.array = np.asarray(value)
        self.shape = self.array.shape

    def clone(self):
        return ArrayTensor(self.array.copy())

    def numel(self):
        return self.array.size

    def view(self, *args):
        if len(args) == 1 and isinstance(args[0], type):
            return ArrayTensor(self.array.view(args[0]))
        return ArrayTensor(self.array.reshape(*args))

    def index_copy_(self, axis, slots, values):
        assert axis == 0
        self.array[slots.array] = values.array
        return self


class RingPlanTests(unittest.TestCase):
    def test_ring_wrap_and_q_gt_one_keep_every_required_token(self):
        data = synthetic_case(batch=2, queries=4, keys=257, heads=2, dim=64, rope_dim=0, topk=32)
        data["q_lengths"][1] = 2
        data["k_lengths"][1] = 250
        data["indices"][1] = -1
        for r in range(2):
            position = 248 + r
            data["indices"][1, r, 0] = np.arange(position-31, position+1)
        data = validate_case(data)
        for window in (1, 128, 256, 512):
            plan = fixed_ring_plan(data, window)
            self.assertEqual(plan["capacity"] % data["page"], 0)
            self.assertGreaterEqual(plan["capacity"], window+3)
            for b in range(2):
                first = int(data["k_lengths"][b])-int(data["q_lengths"][b])
                for p in range(first, int(data["k_lengths"][b])):
                    needed = np.arange(max(0, p-window+1), p+1)
                    np.testing.assert_array_equal(plan["tags"][b, needed % plan["capacity"]], needed)
            np.testing.assert_array_equal(plan["tags"].reshape(-1)[plan["current_slots"]], query_positions(data))

    def test_invalid_windows_rejected(self):
        for invalid in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                fixed_ring_plan(tiny_case(), invalid)

    def test_zero_window_bypasses_all_hot_allocation_and_imports(self):
        calls = []
        output = object()
        variant = types.SimpleNamespace(metadata={"recipe": "baseline"},
                                        call=lambda: calls.append("call") or output,
                                        run=lambda write=False: calls.append(("run", write)) or output)
        base = types.SimpleNamespace(data=tiny_case())  # no torch/cache/tensor attributes at all
        scales = types.SimpleNamespace(base=base, environment={}, prepare=lambda name: variant)
        def forbidden(*args, **kwargs):
            raise AssertionError("W0 must never instantiate a fused reader")
        backend = HotWindowBackend(base.data, scale_backend=scales, reader_factory=forbidden)
        runner = backend.prepare("baseline", 0)
        self.assertIs(runner.call(), output)
        self.assertIs(runner.run(write=True), output)
        self.assertEqual(calls, ["call", ("run", True)])
        self.assertFalse(hasattr(runner, "hot_cache"))
        self.assertTrue(runner.metadata["zero_window_native_bypass"])

    def test_hot_writer_quantizes_original_bf16_twice_and_isolates_each_cache(self):
        data = synthetic_case(batch=2, queries=2, keys=40, heads=2, dim=512, rope_dim=64, topk=16)
        counts = {"cold": 0, "hot": 0, "attention": 0}
        original_kv, original_rope = object(), object()
        cache = ArrayTensor(np.zeros((2, 128, 1, 400), np.uint8))
        def cold_quant(kv, rope):
            self.assertIs(kv, original_kv)
            self.assertIs(rope, original_rope)
            counts["cold"] += 1
            return ArrayTensor(np.full((4, 400), 11, np.uint8)), None, None
        def hot_quant(kv, rope, kind):
            self.assertIs(kv, original_kv)
            self.assertIs(rope, original_rope)
            self.assertEqual(kind, "c8")
            counts["hot"] += 1
            return ArrayTensor(np.full((4, 656), 22, np.uint8)), None, None
        base = types.SimpleNamespace(data=data, torch=types.SimpleNamespace(uint8=np.uint8, int64=np.int64, int32=np.int32),
                                     batch=2, page=128, scale=data["scale"], c8_logical=np.ones((2, 40, 1, 656), np.uint8),
                                     tensor=lambda a, dtype=None: ArrayTensor(np.asarray(a, dtype=dtype).copy()),
                                     q_nope=None, q_rope=None, indices=None, table=None, lengths=None,
                                     current_slots=ArrayTensor(np.array([38, 39, 166, 167], np.int64)),
                                     current_kv=original_kv, current_rope=original_rope, quantize=hot_quant)
        variant = types.SimpleNamespace(metadata={}, cache=cache, _quantize=cold_quant)
        scales = types.SimpleNamespace(base=base, environment={}, prepare=lambda name: variant)
        def call():
            counts["attention"] += 1
            return "single attention output"
        backend = HotWindowBackend(data, scale_backend=scales,
                                   reader_factory=lambda *a, **kw: types.SimpleNamespace(metadata={}, call=call))
        first, second = backend.prepare("baseline", 8), backend.prepare("baseline", 16)
        second_before = second.cold_cache.array.copy()
        self.assertEqual(first.run(write=True), "single attention output")
        self.assertEqual(counts, {"cold": 1, "hot": 1, "attention": 1})
        self.assertFalse(np.any(cache.array))
        np.testing.assert_array_equal(second.cold_cache.array, second_before)
        cold_slots = first.cold_cache.array.reshape(-1, 400)[base.current_slots.array]
        self.assertTrue(np.all(cold_slots == 11))
        hot_slots = first.hot_cache.array.reshape(-1, 656)[first.plan["current_slots"]]
        self.assertTrue(np.all(hot_slots == 22))
        np.testing.assert_array_equal(first.hot_positions.array.reshape(-1)[first.plan["current_slots"]], query_positions(data))

    def test_fused_cold_control_is_explicit_read_only_and_keeps_native_w0(self):
        data = synthetic_case(batch=1, queries=2, keys=40, heads=2, dim=512, rope_dim=64, topk=16)
        factory_calls, native_calls, writer_calls = [], [], []
        cache = ArrayTensor(np.zeros((1, 128, 1, 400), np.uint8))

        def quantize(*args):
            writer_calls.append(args)
            raise AssertionError("A read-only diagnostic must not quantize or scatter")

        base = types.SimpleNamespace(
            data=data, torch=types.SimpleNamespace(uint8=np.uint8, int64=np.int64, int32=np.int32),
            batch=1, page=128, scale=data["scale"], c8_logical=np.ones((1, 40, 1, 656), np.uint8),
            tensor=lambda a, dtype=None: ArrayTensor(np.asarray(a, dtype=dtype).copy()),
            q_nope=None, q_rope=None, indices=None, table=None, lengths=None, quantize=quantize)
        variant = types.SimpleNamespace(metadata={}, cache=cache, _quantize=quantize,
                                        decoded_kv=lambda: data["kv"],
                                        call=lambda: native_calls.append("native_read") or "native",
                                        run=lambda write=False: native_calls.append(("native_run", write)) or "native")
        scales = types.SimpleNamespace(base=base, environment={}, prepare=lambda name: variant)

        def factory(*args, **kwargs):
            factory_calls.append(kwargs)
            return types.SimpleNamespace(metadata={"diagnostic_cold_only": kwargs["diagnostic_cold_only"]},
                                         call=lambda: "custom_cold" if kwargs["diagnostic_cold_only"] else "custom_hot")

        backend = HotWindowBackend(data, scale_backend=scales, reader_factory=factory)
        control = backend.prepare_cold_control("baseline", 16)
        self.assertIs(control, backend.prepare_cold_control("baseline", 16))
        self.assertEqual(control.call(), "custom_cold")
        self.assertEqual(control.run(write=False), "custom_cold")
        with self.assertRaisesRegex(ValueError, "read-only"):
            control.run(write=True)
        self.assertEqual(writer_calls, [])
        self.assertTrue(factory_calls[0]["diagnostic_cold_only"])
        self.assertEqual(control.metadata["effective_window"], 0)
        self.assertIsNone(control.metadata["hot_precision"])
        self.assertEqual(control.metadata["role"], "READ_ONLY_DIAGNOSTIC")
        self.assertFalse(control.metadata["production_comparison_eligible"])
        self.assertIsNot(control.cold_cache, cache)

        native = backend.prepare("baseline", 0)
        hot = backend.prepare("baseline", 16)
        self.assertEqual(native.call(), "native")
        self.assertEqual(native.run(write=True), "native")
        self.assertTrue(native.metadata["zero_window_native_bypass"])
        self.assertFalse(native.diagnostic_cold_only)
        self.assertEqual(hot.call(), "custom_hot")
        self.assertFalse(factory_calls[1]["diagnostic_cold_only"])
        self.assertIsNot(hot.cold_cache, control.cold_cache)
        self.assertEqual(native_calls, ["native_read", ("native_run", True)])

    def test_fused_cold_control_rejects_native_zero_window_and_bad_recipe(self):
        base = types.SimpleNamespace(data=tiny_case())
        scales = types.SimpleNamespace(base=base, environment={})
        backend = HotWindowBackend(base.data, scale_backend=scales)
        for window in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                backend.prepare_cold_control("baseline", window)
        with self.assertRaisesRegex(ValueError, "Unknown cold C4 recipe"):
            backend.prepare_cold_control("unknown", 16)


class FakeBackend:
    fail_window = None
    corrupt_window = None

    def __init__(self, data, **kwargs):
        self.data, self.environment = data, {"device": "CPU_TEST_DOUBLE"}
        self.outputs = {}

    def host(self, value):
        return np.array(value, copy=True)

    def baseline_decoded(self, kind):
        return self.data["kv"] * (1.0 if kind == "bf16" else .999)

    def baseline(self, kind):
        return attention_reference(self.data, kv_c4=self.baseline_decoded(kind))

    def baseline_run(self, kind, write=False):
        return self.baseline(kind)

    def prepare(self, recipe, window):
        if window == self.fail_window:
            raise RuntimeError("ACTUAL_KERNEL_COMPILATION_FAILURE_FOR_TEST")
        cold = self.data["kv"] * (.95 if recipe == "baseline" else .97)
        hot = self.baseline_decoded("c8")
        output = attention_reference(self.data, kv_c4=cold, hot_kv=hot, window=window)
        if window == self.corrupt_window:
            output = output + 100
        return types.SimpleNamespace(metadata={"attention_calls_per_read": 1},
                                     decoded_kv=lambda: cold, decoded_hot_kv=lambda: hot,
                                     call=lambda: output, run=lambda write=False: output)


class HarnessTests(unittest.TestCase):
    def args(self):
        return harness.parser().parse_args(["--timing", "none", "--windows", "0,8,16"])

    def test_same_inputs_and_matching_references_and_partial_outputs(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            data = tiny_case()
            before = harness.identity(data)
            report = harness.run_case(data, self.args(), Path(tmp)/"case", backend_factory=FakeBackend)
            self.assertEqual(report["status"], "VALIDATED_ONLY")
            self.assertEqual(before, harness.identity(data))
            self.assertTrue(report["host_input_unchanged"])
            self.assertEqual(report["production_validation"], "NOT_VALIDATED")
            self.assertEqual(len(report["variants"]), 6)
            for entry in report["variants"]:
                self.assertEqual(entry["kernel_vs_matching_reference"]["relative_l2"], 0)
                self.assertTrue(entry["control_passed"])
            with np.load(Path(tmp)/"case"/"outputs.npz") as outputs:
                self.assertIn("indices", outputs)
                self.assertIn("alg2_max7_w16_prepared", outputs)
            self.assertEqual(json.loads((Path(tmp)/"case"/"results.json").read_text())["status"], "VALIDATED_ONLY")

    def test_real_failure_preserved_and_remaining_cases_continue(self):
        class BrokenBackend(FakeBackend):
            fail_window = 8
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            report = harness.run_case(tiny_case(), self.args(), Path(tmp)/"case", backend_factory=BrokenBackend)
            self.assertEqual(report["status"], "FAILED")
            failed = [x for x in report["variants"] if x["window"] == 8]
            self.assertTrue(all(x["status"] == "ERROR" for x in failed))
            self.assertTrue(all("ACTUAL_KERNEL_COMPILATION_FAILURE_FOR_TEST" in x["traceback"] for x in failed))
            self.assertIn("ACTUAL_KERNEL_COMPILATION_FAILURE_FOR_TEST", errors.getvalue())
            self.assertTrue(all(x["status"] == "VALIDATED_ONLY" for x in report["variants"] if x["window"] != 8))

    def test_device_fault_stops_before_any_other_operator(self):
        calls = []
        class FaultyBackend(FakeBackend):
            def baseline(self, kind):
                calls.append(kind)
                raise RuntimeError("error code is 507015; The address for scalar to access GM is invalid")
            def prepare(self, *args):
                raise AssertionError("No more operators after a device fault")
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            report = harness.run_case(tiny_case(), self.args(), Path(tmp)/"case", backend_factory=FaultyBackend)
            self.assertEqual(report["status"], "DEVICE_ERROR")
            self.assertEqual(report["stopped_after_device_fault"], "bf16")
            self.assertEqual(report["baselines"]["bf16"]["failure_kind"], "DEVICE_EXECUTION_FAULT")
            self.assertIn("507015", report["baselines"]["bf16"]["traceback"])
            self.assertEqual(report["variants"], [])
            self.assertEqual(calls, ["bf16"])
            saved = json.loads((Path(tmp)/"case"/"results.json").read_text())
            self.assertEqual(saved["status"], "DEVICE_ERROR")

    def test_distinct_failures(self):
        self.assertEqual(harness.failure_kind("EZ1009: operator prototype OpDef"), "NATIVE_OPERATOR_CONTRACT")
        self.assertEqual(harness.failure_kind("ub overflow, requires 10799872 bits"), "TRITON_UB_OVERFLOW")
        self.assertEqual(harness.failure_kind("GRAPH_CAPTURE_FAILURE"), "OTHER")

    def test_wrong_output_never_timed(self):
        class WrongBackend(FakeBackend):
            corrupt_window = 8
        timed = []
        def bench(fn, **kwargs):
            value = fn()
            self.assertLess(float(np.abs(value).max()), 100)
            timed.append(1)
            return {"event_us": {"median": 10.}, "wall_us": {"median": 12.}}
        args = self.args()
        args.timing, args.repeats = "eager", 2
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            report = harness.run_case(tiny_case(), args, Path(tmp)/"case", backend_factory=WrongBackend, benchmark=bench)
        self.assertTrue(timed)
        for entry in report["variants"]:
            if entry["window"] == 8:
                self.assertEqual(entry["status"], "CONTROL_FAILED")
                self.assertNotIn("timing_prepared_read", entry)

    def test_pair_order_ratios_and_no_false_production_gate(self):
        args = self.args()
        args.timing, args.repeats = "eager", 3
        observed = []
        def bench(fn, **kwargs):
            name = fn()
            observed.append(name)
            us = 10. if name == "A" else 11.
            self.assertEqual(kwargs["repeats"], 1)
            return {"event_us": {"median": us}, "wall_us": {"median": us * 2}}
        result = harness.paired_benchmark(lambda: "A", lambda: "B", args, bench)["eager"]
        self.assertEqual(observed, ["A", "B", "B", "A", "A", "B"])
        self.assertEqual(result["event_us"]["paired_ratios"], [1.1, 1.1, 1.1])
        self.assertEqual(result["observation"], "OBSERVED_SLOWDOWN")
        self.assertEqual(result["production_performance"], "NOT_VALIDATED")

    def test_partial_timing_keeps_failure_including_completed_pairs(self):
        args = self.args()
        args.timing, args.repeats = "graph", 3
        count = [0]
        def bench(fn, **kwargs):
            count[0] += 1
            if count[0] == 4:
                raise RuntimeError("GRAPH_CAPTURE_FAILURE")
            return {"event_us": {"median": 10.}, "wall_us": {"median": 10.}}
        with contextlib.redirect_stderr(io.StringIO()):
            result = harness.paired_benchmark(lambda: None, lambda: None, args, bench)["graph"]
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["observation"], "NOT_MEASURED")
        self.assertEqual(len(result["rounds"]), 2)
        self.assertEqual(result["event_us"]["paired_ratios"], [1.0])
        self.assertIn("GRAPH_CAPTURE_FAILURE", result["rounds"][1]["traceback"])


if __name__ == "__main__":
    unittest.main()
