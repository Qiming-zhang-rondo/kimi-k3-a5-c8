"""Validate runner dispatch identity without importing Torch or loading an NPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("single_pass_routed_plumbing",
    ROOT/"experimental/single_pass/run_single_pass.py")
RUNNER=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class FakePlan:
    def __init__(self,name):
        self.name=name
        self.calls=[]

    def run(self,*args):
        self.calls.append(args)
        return self.name,None,None


class RoutedPlumbingTest(unittest.TestCase):
    def test_default_keeps_original_and_overlay_only(self):
        modes=RUNNER.plan_modes()
        self.assertEqual(modes,dict(original_c4_direct=(0,True,False,False,False),
            single_pass_w0=(0,False,False,False,False),single_pass_w128=(128,False,False,False,False)))
        args=RUNNER.parser().parse_args(["--payload-dir","payload","--library","lib.so",
            "--output","out","--device","npu:0"])
        self.assertFalse(args.routed_ab)
        self.assertFalse(args.fast_pair_ab)
        self.assertFalse(args.early_hot_ab)
        self.assertFalse(args.stats_ab)

    def test_stats_matrix_uses_independent_plans_with_identical_inputs_and_geometry(self):
        class Bridge:
            def __init__(self):
                self.prepared=[]
            def prepare(self,*args):
                plan=FakePlan("stats_plan_"+str(len(self.prepared)))
                self.prepared.append((args,plan))
                return plan
        bridge=Bridge()
        query=object()
        inputs=(query,)+tuple(object() for _ in range(9))
        plans,calls,specs=RUNNER.prepare_stats_ab(bridge,query,8192,256,4,.125,inputs)
        self.assertEqual(len({id(plan) for plan in plans.values()}),4)
        self.assertEqual([args[-1] for args,plan in bridge.prepared],[False,False,True,True])
        for args,plan in bridge.prepared:
            self.assertEqual(args[:-1],(query,8192,256,4,.125))
        for name in reversed(list(calls)):
            self.assertEqual(calls[name](),plans[name].name)
            self.assertEqual(plans[name].calls,[inputs+specs[name][1]])
        pairs=RUNNER.stats_ab_pairs()
        self.assertEqual(len(pairs),4)
        for _,baseline,candidate in pairs[:2]:
            self.assertEqual(specs[baseline][0],specs[candidate][0])
            self.assertTrue(specs[baseline][1][1])
            self.assertTrue(specs[candidate][1][3])
        for _,baseline,candidate in pairs[2:]:
            self.assertFalse(specs[baseline][0])
            self.assertTrue(specs[candidate][0])
            self.assertEqual(specs[baseline][1],specs[candidate][1])

    def test_collector_retains_complete_stats_section_including_failed_pairs(self):
        spec=importlib.util.spec_from_file_location("stats_ab_collector",
            ROOT/"experimental/single_pass/collect_run_evidence.py")
        collector=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collector)
        section=dict(status="ERROR",controls={"original_c4_stats_on":{"passed":True}},
            graph_controls={"original_c4_stats_on":{"passed":False,
                "returned_stats_finite_positive_sum_after_replay":False}},
            timings={"fast_pair_vs_original_stats_off":{"graph":{"status":"ERROR",
                "rounds":[{"order":["candidate","baseline"],"error":"preserve raw failure"}]}}})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"results.json"
            path.write_text(json.dumps(dict(version="qsfa-recent128-single-pass-a5-v1.5-stats-ab",
                status="ERROR",stats_ab=section)))
            self.assertEqual(collector.collect(path)["stats_ab"],section)

    def test_ab_calls_bind_distinct_plans_modes_and_shared_inputs(self):
        modes=RUNNER.plan_modes(True)
        plans={name:FakePlan(name) for name in modes}
        inputs=tuple(object() for _ in range(10))
        calls=RUNNER.plan_calls(plans,inputs,modes)
        # Reverse order detects accidental late-bound closures selecting only
        # the last (routed) plan. Inputs must stay identical between both arms.
        for name in reversed(list(calls)):
            self.assertEqual(calls[name](),name)
            self.assertEqual(plans[name].calls,[inputs+modes[name]])
        self.assertEqual(modes["single_pass_w128"],(128,False,False,False,False))
        self.assertEqual(modes["single_pass_w128_routed"],(128,False,True,False,False))
        self.assertEqual(RUNNER.plan_modes(),{k:v for k,v in modes.items() if not v[2]})

    def test_fast_pair_adds_distinct_entry_without_replacing_previous_routed(self):
        modes=RUNNER.plan_modes(fast_pair_ab=True)
        self.assertEqual({k:v for k,v in modes.items() if not v[3]},RUNNER.plan_modes(True))
        self.assertEqual(modes["single_pass_w128_fast_pair"],(128,False,False,True,False))
        plans={name:FakePlan(name) for name in modes}
        inputs=tuple(object() for _ in range(10))
        calls=RUNNER.plan_calls(plans,inputs,modes)
        for name in reversed(list(calls)):
            self.assertEqual(calls[name](),name)
            self.assertEqual(plans[name].calls,[inputs+modes[name]])
        for mode in modes.values():
            self.assertLessEqual(sum(mode[1:]),1)

    def test_early_hot_adds_distinct_entry_and_keeps_all_previous_controls(self):
        modes=RUNNER.plan_modes(early_hot_ab=True)
        self.assertEqual({k:v for k,v in modes.items() if not v[4]},
                         RUNNER.plan_modes(fast_pair_ab=True))
        self.assertEqual(modes["single_pass_w128_early_hot"],(128,False,False,False,True))
        plans={name:FakePlan(name) for name in modes}
        inputs=tuple(object() for _ in range(10))
        for name,call in reversed(list(RUNNER.plan_calls(plans,inputs,modes).items())):
            self.assertEqual(call(),name)
            self.assertEqual(plans[name].calls,[inputs+modes[name]])
        for mode in modes.values():
            self.assertLessEqual(sum(mode[1:]),1)

    def test_correctness_patterns_preserve_multiset_and_exercise_tile_routes(self):
        positions=np.array([8190,8191],dtype=np.int64)
        generator=np.random.default_rng(19)
        ids=np.stack([generator.permutation(np.concatenate((np.arange(1920),
            np.arange(position-127,position+1)))) for position in positions]).astype(np.int32)
        original=ids.copy()
        patterns=RUNNER.routed_index_patterns(ids,positions)
        self.assertEqual(set(patterns),{"hot_then_cold","alternating_prefix"})
        for pattern,actual in patterns.items():
            self.assertEqual(actual.dtype,ids.dtype)
            self.assertTrue(actual.flags.c_contiguous)
            np.testing.assert_array_equal(np.sort(actual,axis=1),np.sort(ids,axis=1))
            hot=actual>=positions[:,None]-127
            if pattern=="hot_then_cold":
                self.assertTrue(hot[:,:128].all())
                self.assertFalse(hot[:,128:].any())
                self.assertTrue(np.isin(hot.reshape(2,128,16).sum(axis=2),(0,16)).all())
            else:
                self.assertTrue(hot[:,:256:2].all())
                self.assertFalse(hot[:,1:256:2].any())
                self.assertFalse(hot[:,256:].any())
                np.testing.assert_array_equal(hot[:,:256].reshape(2,16,16).sum(axis=2),8)
        np.testing.assert_array_equal(ids,original)

    def test_correctness_patterns_reject_future_ids(self):
        with self.assertRaises(ValueError):
            RUNNER.routed_index_patterns(np.array([[2]],dtype=np.int32),np.array([1]))


if __name__=="__main__":
    unittest.main()
