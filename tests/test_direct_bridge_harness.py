"""CPU control-flow tests; these do not compile or execute an Ascend kernel."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "direct_bridge"))
import pinned_payload
import run_direct_ab as harness


class Plan:
    def __init__(self, owner, fused):
        self.owner, self.fused = owner, fused

    def native_hot(self, *args):
        self.owner.calls.append(("hot", args))
        return ("direct_bf16", "hot_max", "hot_sum")

    def fused_hot(self, *args):
        self.owner.calls.append(("fused", args))
        return ("fused_fp32", "empty", "empty")


class Bridge:
    def __init__(self):
        self.calls = []

    def prepare(self, *args):
        self.calls.append(("prepare", args))
        return Plan(self, args[2])


class ParentRunner:
    """Model only the inherited writer/partition boundary, with changing cold pointers."""
    def __init__(self, backend, recipe, window, writer):
        self.base = types.SimpleNamespace(q_nope="query", q_rope="qrope", cu="qlength", scale=.125)
        self.hot_cache, self.hot_rope = "ring", "rope"
        self.ring_table, self.ring_lengths = "table", "kvlength"
        self.metadata, self.events, self.n = {}, [], 0
        self.merge_output = types.SimpleNamespace(dtype=backend or "torch.float32")

    def _cold_read(self, parts):
        self.events.append("cold_read")
        self.n += 1
        return (f"cold_out_{self.n}", f"cold_max_{self.n}", f"cold_sum_{self.n}")

    def _merge(self, left, right, parts):
        self.events.append(("merge", left, right))
        return "triton_merged"

    def run(self, write=False, dynamic=True):
        if write:
            self.events.extend(["cold_write", "hot_prepare"])
        parts = dict(recent_ring="dynamic_indices", old_present="cold_mask", recent_present="hot_mask")
        return self._read(parts)


class FakeTensor:
    def __init__(self, value, dtype="torch.float32"):
        self.value, self.shape = value, (1,)
        self.dtype = dtype

    def numel(self):
        return 1

    def is_floating_point(self):
        return True

    def fill_(self, value):
        self.value = value

    def is_contiguous(self):
        return True

    def stride(self):
        return (1,)


class CopyScalar(float):
    def copy(self):
        return CopyScalar(self)


class DirectBridgeHarnessTests(unittest.TestCase):
    def test_manifest_checks_actual_bytes_and_rejects_missing_or_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"module.py"
            expected = hashlib.sha256(b"content\n").hexdigest()
            with self.assertRaisesRegex(ValueError, "Missing"):
                pinned_payload.verify_payload(directory, {path.name: expected})
            path.write_bytes(b"content\n")
            self.assertEqual(pinned_payload.verify_payload(directory, {path.name: expected}), {path.name: expected})
            path.write_bytes(b"content")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                pinned_payload.verify_payload(directory, {path.name: expected})

    def test_fused_uses_every_new_cold_state_and_runtime_partition(self):
        bridge = Bridge()
        runner = harness.make_direct_runner_class(ParentRunner)(None, bridge, fused=True)
        self.assertEqual(bridge.calls[0], ("prepare", ("query", .125, True, False, False)))
        self.assertEqual(runner.run(write=True), "fused_fp32")
        self.assertEqual(runner.run(write=True), "fused_fp32")
        calls = [args for kind,args in bridge.calls if kind == "fused"]
        self.assertEqual(calls[0][:9], ("query", "ring", "ring", "dynamic_indices", "table", "qlength", "kvlength", "qrope", "rope"))
        self.assertEqual(calls[0][9:], ("cold_out_1", "cold_max_1", "cold_sum_1", "cold_mask", "hot_mask"))
        self.assertEqual(calls[1][9], "cold_out_2")
        self.assertEqual(runner.events, ["cold_write", "hot_prepare", "cold_read"]*2)
        self.assertEqual(runner.direct_state_buffers()[0], "fused_fp32")

    def test_twopass_keeps_original_merge_and_requests_native_statistics(self):
        bridge = Bridge()
        runner = harness.make_direct_runner_class(ParentRunner)(None, bridge, fused=False)
        self.assertEqual(bridge.calls[0], ("prepare", ("query", .125, False, True, False)))
        self.assertEqual(runner.run(write=True), "triton_merged")
        self.assertEqual(runner.events[-1], ("merge", ("cold_out_1", "cold_max_1", "cold_sum_1"),
                                            ("direct_bf16", "hot_max", "hot_sum")))

    def test_native_bf16_cold_selects_bf16_io_without_python_conversion(self):
        bridge = Bridge()
        runner = harness.make_direct_runner_class(ParentRunner)("torch.bfloat16", bridge, fused=True)
        self.assertEqual(bridge.calls[0], ("prepare", ("query", .125, True, False, True)))
        self.assertEqual(runner.metadata["expected_pipeline_output_dtype"], "torch.bfloat16")
        self.assertEqual(runner.metadata["cold_state_cast_kernel_calls"], 0)
        runner.run(write=True)
        self.assertEqual(bridge.calls[-1][1][9], "cold_out_1")
        native_bridge = Bridge()
        harness.make_direct_runner_class(ParentRunner)("torch.bfloat16", native_bridge, fused=False)
        self.assertFalse(native_bridge.calls[0][1][-1], "Native BF16 hot reader uses its original specialization")

    def test_dtype_validation_preserves_cold_but_never_narrows_stats(self):
        for dtype in ("torch.bfloat16", "torch.float32"):
            state = (FakeTensor(1., dtype), FakeTensor(2.), FakeTensor(3.))
            harness.validate_cold_state(state, dtype)
            wrong_stats = (state[0], FakeTensor(2., "torch.bfloat16"), state[2])
            with self.assertRaisesRegex(harness.ControlFailed, "maximum and sum must remain FP32"):
                harness.validate_cold_state(wrong_stats, dtype)
        with self.assertRaisesRegex(harness.ControlFailed, "dtype changed"):
            harness.validate_cold_state((FakeTensor(1.), FakeTensor(2.), FakeTensor(3.)), "torch.bfloat16")
        with self.assertRaisesRegex(harness.ControlFailed, "actual dtype"):
            harness.require_same_dtype(FakeTensor(1.), FakeTensor(1., "torch.bfloat16"), "strict output")

    def test_raw_state_recording_retains_dtype_bytes_before_rejection(self):
        import numpy as np
        state = (FakeTensor(1., "torch.bfloat16"), FakeTensor(float("nan")), FakeTensor(0.))
        arrays = {}
        description = harness.state_diagnostics(state, lambda t: np.asarray([t.value], np.float32),
            lambda t: np.asarray([0x80, 0x3f] if t.dtype == "torch.bfloat16" else [0,0,0,0], np.uint8), arrays, "cold")
        self.assertEqual(description["out"]["dtype"], "torch.bfloat16")
        self.assertEqual(description["max"]["nonfinite_count"], 1)
        self.assertIsNone(description["max"]["min"])
        self.assertEqual(arrays["cold_out_raw_bytes"].tolist(), [0x80, 0x3f])
        with self.assertRaises(harness.ControlFailed):
            harness.validate_cold_state(state, "torch.float32")
        self.assertIn("cold_sum", arrays)

    def test_destination_rounding_does_not_hide_unrounded_error_or_change_baseline(self):
        import numpy as np
        reference = np.asarray([1.003], np.float64)
        bf16_baseline = np.asarray([1.2], np.float64)
        output = FakeTensor(1., "torch.bfloat16")
        arrays = {}
        def compare(actual, expected):
            return dict(relative_l2=float(np.linalg.norm(np.asarray(actual)-expected)/np.linalg.norm(expected)))
        def round_reference(values, dtype):
            self.assertEqual(dtype, "torch.bfloat16")
            # These fixture values are below the midpoint between 1 and
            # 1+1/128, so BF16 round-to-nearest gives exactly 1.
            np.testing.assert_array_equal(values, reference)
            return np.asarray([1.], np.float32)
        def evaluate(call, pipeline, matching, full_bf16, _, args, backend, records, name):
            self.assertIs(call(), output)
            self.assertIs(full_bf16, bf16_baseline, "Quantization-error baseline must remain unrounded")
            values = np.asarray([pipeline().value])
            records[name+"_write_read"] = values
            return dict(kernel_vs_matching_reference=compare(values, matching),
                        vs_bf16_math=compare(values, full_bf16)), values
        metrics, _ = harness.evaluate_output_contract(output, lambda: output, reference, bf16_baseline,
            None, None, arrays, "bf16", evaluate, round_reference, compare)
        self.assertEqual(metrics["kernel_vs_matching_reference"]["relative_l2"], 0.)
        self.assertGreater(metrics["kernel_vs_unrounded_matching_reference"]["relative_l2"], .001)
        self.assertGreater(metrics["destination_rounding_vs_unrounded_matching_reference"]["relative_l2"], .001)
        self.assertAlmostEqual(metrics["vs_bf16_math"]["relative_l2"], 1/6)

    def test_no_timing_after_failed_controls_and_five_required_pairs_when_valid(self):
        calls = {name: object() for name in harness.ARMS}
        seen = []
        def paired(left, right, args, benchmark):
            seen.append((left, right))
            return {"graph": {"status": "MEASURED"}}
        with self.assertRaises(harness.ControlFailed):
            harness.timed_pairs({name: name != "direct_fused" for name in calls}, calls, None, None, paired)
        self.assertFalse(seen)
        result = harness.timed_pairs(dict.fromkeys(calls, True), calls, None, None, paired)
        self.assertEqual(len(result), 5)
        self.assertEqual(seen, [(calls[a], calls[b]) for a,b in harness.PAIRS])
        self.assertIn((calls["installed_native_twopass"], calls["direct_fused"]), seen)
        seen.clear()
        with self.assertRaises(harness.ControlFailed):
            harness.timed_pairs({name: name != "installed_native_twopass" for name in calls}, calls, None, None, paired)
        self.assertFalse(seen, "An unavailable installed baseline must block every performance comparison")

    def test_partial_measurement_is_not_accepted(self):
        with self.assertRaisesRegex(harness.ControlFailed, "Partial|partial"):
            harness.timed_pairs(dict.fromkeys(harness.ARMS, True), dict.fromkeys(harness.ARMS), None, None,
                lambda *_: {"graph": {"status": "ERROR", "rounds": [{"event_us_ratio": .5}]}})

    def test_graph_poison_rejects_omitted_external_launch(self):
        import math
        def exercise(captured):
            hot, output = FakeTensor(3.), FakeTensor(3.)
            class Graph:
                def __enter__(self): return self
                def __exit__(self, *_): return False
                def replay(self):
                    if captured: hot.value = 3.
                    output.value = hot.value
            torch = types.SimpleNamespace(npu=types.SimpleNamespace(
                synchronize=lambda: None, NPUGraph=Graph, graph=lambda x: x))
            def call():
                hot.value = output.value = 3.
                return output
            services = dict(compare=lambda actual, ref: dict(finite=math.isfinite(actual), relative_l2=0., max_row_relative_l2=0.),
                            acceptable=lambda metric, limit: metric["finite"])
            return harness.graph_replay_control(call, lambda: (hot,), torch,
                                                lambda x: CopyScalar(x.value), services)
        with self.assertRaisesRegex(harness.ControlFailed, "Graph replay"):
            exercise(False)
        result = exercise(True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["poisoned_buffers"], [[1], [1]])

    def test_fixed_thresholds_are_not_cli_relaxable(self):
        args = harness.parser().parse_args(["--payload-dir", "/payload", "--bridge-library", "/build/libsfa_torch_direct.so",
                                           "--output", "/results", "--device", "npu:0"])
        self.assertEqual(args.repeats, 10)
        self.assertEqual(args.timing, "graph")
        self.assertEqual(args.writer, "hot_store_partition")
        self.assertNotIn("control_l2", vars(args))


if __name__ == "__main__":
    unittest.main()
