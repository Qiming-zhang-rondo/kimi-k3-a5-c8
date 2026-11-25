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
    def __init__(self, owner, fused, aic_count, packed_output=False, physical_kv=256):
        self.owner, self.fused = owner, fused
        self.aic_count = aic_count
        self.packed_output,self.physical_kv=packed_output,physical_kv

    def configuration(self):
        return dict(abi_version=4,merge_tile_rows=16,aic_count=self.aic_count,
                    packed_output=self.packed_output,physical_kv=self.physical_kv)

    def native_hot(self, *args):
        self.owner.calls.append(("hot", args))
        return ("direct_bf16", "hot_max", "hot_sum")

    def fused_hot(self, *args):
        self.owner.calls.append(("fused", args))
        return ("fused_packed" if self.packed_output else "fused_fp32", "empty", "empty")


class Bridge:
    def __init__(self):
        self.calls = []

    def prepare(self, *args):
        self.calls.append(("prepare", args))
        return Plan(self,args[2],args[5],args[6],args[7])


class ParentRunner:
    """Model only the inherited writer/partition boundary, with changing cold pointers."""
    def __init__(self, backend, recipe, window, writer):
        queries=int(backend.base.queries) if hasattr(backend,"base") else 4
        self.base = types.SimpleNamespace(q_nope="query",q_rope="qrope",cu="qlength",scale=.125,queries=queries)
        self.capacity=((128+queries-2)//128+1)*128
        self.hot_cache, self.hot_rope = "ring", "rope"
        self.ring_table, self.ring_lengths = "table", "kvlength"
        self.metadata, self.events, self.n = {}, [], 0
        dtype=backend if isinstance(backend,str) else getattr(backend,"dtype","torch.float32")
        self.merge_output = types.SimpleNamespace(dtype=dtype)

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
    @staticmethod
    def rounding_fixture():
        import numpy as np
        shape, stats = (4,64,512), (4,64,1)
        cold = (np.ones(shape,np.float32), np.zeros(stats,np.float32), np.full(stats,1920,np.float32))
        hot = (np.full(shape,1.0625,np.float32), np.zeros(stats,np.float32), np.full(stats,128,np.float32))
        actual, native = np.ones(shape,np.float32), np.ones(shape,np.float32)
        native.reshape(-1)[:4096] = 1+1/128
        kwargs = dict(cold_dtype="torch.bfloat16",hot_dtype="torch.bfloat16",output_dtype="torch.bfloat16")
        return cold,hot,actual,native,kwargs

    def test_zero_query_exact_rne_accepts_exact_oracle_and_only_adjacent_ties(self):
        cold,hot,actual,native,kwargs = self.rounding_fixture()
        result = harness.zero_query_exact_rne_gate(cold,hot,actual,native,**kwargs)
        self.assertTrue(result["passed"])
        self.assertEqual(result["actual_oracle_bit_mismatch_count"],0)
        self.assertEqual(result["native_mismatches_at_exact_midpoint"],4096)
        self.assertEqual(result["strict_relative_l2_limit_unchanged"],.001)

    def test_zero_query_exact_rne_rejects_nontie_even_when_only_one_ulp(self):
        cold,hot,actual,native,kwargs = self.rounding_fixture()
        hot[0].fill(1.)  # Exact merge is 1, not a midpoint.
        result = harness.zero_query_exact_rne_gate(cold,hot,actual,native,**kwargs)
        self.assertFalse(result["passed"])
        self.assertEqual(result["native_mismatches_not_at_exact_midpoint"],4096)

    def test_zero_query_exact_rne_rejects_wrong_tie_result_stats_and_dtype(self):
        cold,hot,actual,native,kwargs = self.rounding_fixture()
        actual.reshape(-1)[0] = 1+1/128
        self.assertFalse(harness.zero_query_exact_rne_gate(cold,hot,actual,native,**kwargs)["passed"])
        actual.fill(1.)
        cold[2].reshape(-1)[0] = 1919
        self.assertFalse(harness.zero_query_exact_rne_gate(cold,hot,actual,native,**kwargs)["passed"])
        cold[2].fill(1920)
        kwargs["output_dtype"] = "torch.float32"
        self.assertFalse(harness.zero_query_exact_rne_gate(cold,hot,actual,native,**kwargs)["passed"])

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
        self.assertEqual(bridge.calls[0],("prepare",("query",.125,True,False,False,1,False,256)))
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
        self.assertEqual(bridge.calls[0],("prepare",("query",.125,False,True,False,1,False,256)))
        self.assertEqual(runner.run(write=True), "triton_merged")
        self.assertEqual(runner.events[-1], ("merge", ("cold_out_1", "cold_max_1", "cold_sum_1"),
                                            ("direct_bf16", "hot_max", "hot_sum")))

    def test_native_bf16_cold_selects_bf16_io_without_python_conversion(self):
        bridge = Bridge()
        runner = harness.make_direct_runner_class(ParentRunner)("torch.bfloat16", bridge, fused=True)
        self.assertEqual(bridge.calls[0],("prepare",("query",.125,True,False,True,1,False,256)))
        self.assertEqual(runner.metadata["expected_pipeline_output_dtype"], "torch.bfloat16")
        self.assertEqual(runner.metadata["cold_state_cast_kernel_calls"], 0)
        runner.run(write=True)
        self.assertEqual(bridge.calls[-1][1][9], "cold_out_1")
        native_bridge = Bridge()
        harness.make_direct_runner_class(ParentRunner)("torch.bfloat16", native_bridge, fused=False)
        self.assertFalse(native_bridge.calls[0][1][4], "Native BF16 hot reader uses its original specialization")

    def test_task_count_is_explicit_for_both_direct_arms_and_invalid_count_never_prepares(self):
        cls = harness.make_direct_runner_class(ParentRunner)
        for fused in (False, True):
            for count in (1, 2, 4):
                bridge = Bridge()
                runner = cls("torch.bfloat16", bridge, fused=fused, aic_count=count)
                self.assertEqual(bridge.calls[0][1][5], count)
                self.assertEqual(runner.metadata["direct_configuration"]["aic_count"], count)
                self.assertEqual(runner.metadata["direct_aiv_count"], 2*count)
                self.assertEqual(runner.metadata["direct_configuration"]["merge_tile_rows"], 16)
        bridge = Bridge()
        with self.assertRaisesRegex(ValueError, "1/2/4/8/16"):
            cls(None, bridge, fused=True, aic_count=3)
        self.assertEqual(bridge.calls, [])

    def test_dynamic_queries_pass_actual_ring_capacity_and_packed_choice(self):
        cls=harness.make_direct_runner_class(ParentRunner)
        for queries,capacity in ((1,128),(32,256),(128,256),(256,384),(1024,1152)):
            backend=types.SimpleNamespace(base=types.SimpleNamespace(queries=queries),dtype="torch.bfloat16")
            count=min(queries,16)
            bridge=Bridge()
            runner=cls(backend,bridge,fused=True,aic_count=count,packed_output=True)
            self.assertEqual(bridge.calls[0][1][-3:],(count,True,capacity))
            self.assertEqual(runner.run(write=True),"fused_packed")
        for queries,count in ((1,2),(3,2),(1025,1),(32,3),(32,True)):
            with self.assertRaises(ValueError): harness.validate_task_count(queries,count)

    def test_summary_artifacts_discard_full_states_but_keep_transient_full_accuracy_input(self):
        import numpy as np
        store=harness.ArtifactStore("summary")
        value=np.arange(32,dtype=np.float32)
        store["reader_cold_out"]=value
        self.assertNotIn("reader_cold_out",store)
        self.assertEqual(store.array_metadata["reader_cold_out"]["shape"],[32])
        store["arm_write_read"]=value
        self.assertIs(store["arm_write_read"],value)
        with tempfile.TemporaryDirectory() as directory:
            store.save(directory)
            self.assertFalse((Path(directory)/"outputs.npz").exists())
        store.release_transient()
        self.assertFalse(store)
        full=harness.ArtifactStore("full")
        full["all_elements"]=value
        with tempfile.TemporaryDirectory() as directory:
            full.save(directory)
            with np.load(Path(directory)/"outputs.npz",allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["all_elements"],value)

    def test_optional_packed_arm_requires_all_controls_and_adds_three_pairs(self):
        names=(*harness.ARMS,"direct_fused_packed")
        calls={name:object() for name in names}
        seen=[]
        def paired(left,right,*_):
            seen.append((left,right))
            return {"graph":{"status":"MEASURED"}}
        with self.assertRaises(harness.ControlFailed):
            harness.timed_pairs({name:name!="direct_fused_packed" for name in names},calls,None,None,paired)
        self.assertFalse(seen)
        results=harness.timed_pairs(dict.fromkeys(names,True),calls,None,None,paired)
        self.assertEqual(len(results),8)
        self.assertIn("direct_fused_packed_vs_direct_fused",results)
        self.assertIn("direct_fused_packed_vs_installed_native_twopass",results)

    def test_cold_store_candidate_is_opt_in_and_paired_against_current_fusion(self):
        names=(*harness.ARMS,"direct_fused_cold_store")
        calls={name:object() for name in names};seen=[]
        def paired(left,right,*_):
            seen.append((left,right));return {"graph":{"status":"MEASURED"}}
        with self.assertRaises(harness.ControlFailed):
            harness.timed_pairs({name:name!="direct_fused_cold_store" for name in names},calls,None,None,paired)
        self.assertEqual(seen,[])
        result=harness.timed_pairs(dict.fromkeys(names,True),calls,None,None,paired)
        self.assertEqual(len(result),8)
        self.assertIn((calls["direct_fused"],calls["direct_fused_cold_store"]),seen)
        self.assertIn("direct_fused_cold_store_vs_installed_native_twopass",result)

    def test_zero_query_exact_gate_accepts_variable_q_without_sampling(self):
        import numpy as np
        q=32
        out,stats=(q,64,512),(q,64,1)
        cold=(np.ones(out,np.float32),np.zeros(stats,np.float32),np.full(stats,1920,np.float32))
        hot=(np.full(out,1.0625,np.float32),np.zeros(stats,np.float32),np.full(stats,128,np.float32))
        actual=np.ones(out,np.float32);native=actual.copy();native[-1,-1,-1]=1+1/128
        kw=dict(cold_dtype="torch.bfloat16",hot_dtype="torch.bfloat16",output_dtype="torch.bfloat16")
        self.assertTrue(harness.zero_query_exact_rne_gate(cold,hot,actual,native,**kw)["passed"])
        actual[-1,-1,-1]=1+1/128
        result=harness.zero_query_exact_rne_gate(cold,hot,actual,native,**kw)
        self.assertFalse(result["passed"])
        self.assertEqual(result["actual_oracle_bit_mismatch_count"],1)

    def test_prepared_reader_stages_do_not_hide_cold_read_inside_hot_only_timer(self):
        class StageParent(ParentRunner):
            def partition(self):
                return dict(recent_ring="prepared_indices", old_present="cp", recent_present="hp")

            def _hot_read(self, parts):
                return ("installed_hot", "max", "sum")

        cls = harness.make_direct_runner_class(StageParent)
        installed = StageParent(None, "baseline", 128, "hot_store_partition")
        native_bridge, fused_bridge = Bridge(), Bridge()
        native = cls(None, native_bridge, fused=False, aic_count=4)
        fused = cls(None, fused_bridge, fused=True, aic_count=4)
        stages = harness.prepared_reader_stages(dict(installed_native_twopass=installed,
            direct_native_twopass=native, direct_fused=fused))
        self.assertEqual(fused.n, 1, "real cold state is prepared once outside stage timing")
        self.assertEqual(stages["direct_fused_hot_read_merge"](), "fused_fp32")
        self.assertEqual(stages["direct_fused_hot_read_merge"](), "fused_fp32")
        self.assertEqual(fused.n, 1, "hot-only stage must not time or rerun the cold reader")
        self.assertEqual(fused_bridge.calls[-1][1][9], "cold_out_1")
        self.assertEqual(stages["direct_bf16_hot_read"](), "direct_bf16")
        self.assertEqual(stages["installed_bf16_hot_read"](), "installed_hot")
        self.assertEqual(stages["installed_external_merge"](), "triton_merged")
        self.assertEqual(stages["native_c4_cold_read"](), "cold_out_2")

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
        self.assertEqual(args.direct_aic_count, 1)
        self.assertFalse(args.cold_store_candidate)
        self.assertNotIn("control_l2", vars(args))


if __name__ == "__main__":
    unittest.main()
