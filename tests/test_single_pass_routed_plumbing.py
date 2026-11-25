"""Validate runner dispatch identity without importing Torch or loading an NPU."""
import importlib.util
from pathlib import Path
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
        self.assertEqual(modes,dict(original_c4_direct=(0,True,False),
            single_pass_w0=(0,False,False),single_pass_w128=(128,False,False)))
        args=RUNNER.parser().parse_args(["--payload-dir","payload","--library","lib.so",
            "--output","out","--device","npu:0"])
        self.assertFalse(args.routed_ab)

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
        self.assertEqual(modes["single_pass_w128"],(128,False,False))
        self.assertEqual(modes["single_pass_w128_routed"],(128,False,True))
        self.assertEqual(RUNNER.plan_modes(),{k:v for k,v in modes.items() if not v[2]})

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
