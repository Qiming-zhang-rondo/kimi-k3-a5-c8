"""Host-only gates for the new KV-split runner; never imports Torch/NPU."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experimental/single_pass"))
import run_kv_split as runner


class KvSplitRunnerTest(unittest.TestCase):
    def test_default_matrix_keeps_both_windows_and_degenerate_split_one(self):
        args=runner.parser().parse_args(["--payload-dir","/tmp/p","--library","/tmp/l.so",
            "--device","npu:3","--output","/tmp/o"])
        self.assertEqual(args.timing,"none")
        specs=runner.candidate_specs(args.windows,args.splits)
        self.assertEqual(len(specs),6)
        self.assertEqual(specs["w128_split4"],dict(window=128,splits=4))
        self.assertIn("w0_split1",specs)
        for windows,splits in (([0,0],[1]),([0],[2,2]),([64],[1]),([0],[3])):
            with self.assertRaises(ValueError): runner.candidate_specs(windows,splits)

    def test_timing_never_starts_before_every_math_partial_and_graph_gate_passes(self):
        calls={name:lambda:None for name in ("w0_v4","w0_split1")}
        specs=runner.candidate_specs([0],[1])
        args=SimpleNamespace(timing="graph",repeats=2)
        trace=[]
        def pair(*unused):
            trace.append("timed")
            return {"graph":dict(status="MEASURED",rounds=[{},{}])}
        controls={"main":{"passed":True},"zero_padding":{"passed":False}}
        with self.assertRaises(runner.ControlFailed):
            runner.measure_candidates(calls,specs,controls,{"main":{"passed":True}},args,None,pair)
        self.assertEqual(trace,[])
        controls["zero_padding"]["passed"]=True
        with self.assertRaises(runner.ControlFailed):
            runner.measure_candidates(calls,specs,controls,{"main":{"passed":False}},args,None,pair)
        self.assertEqual(trace,[])
        result=runner.measure_candidates(calls,specs,controls,{"main":{"passed":True}},args,None,pair)
        self.assertEqual(trace,["timed"])
        self.assertEqual(result["w0_split1_vs_v4"]["graph"]["status"],"MEASURED")

    def test_incomplete_pair_rounds_cannot_be_marked_measured(self):
        args=SimpleNamespace(timing="eager",repeats=3)
        calls={name:lambda:None for name in ("w0_v4","w0_split1")}
        with self.assertRaises(runner.ControlFailed):
            runner.measure_candidates(calls,runner.candidate_specs([0],[1]),{"math":{"passed":True}},
                {},args,None,lambda *args:{"eager":dict(status="MEASURED",rounds=[{}])})

    def test_actual_partial_normalized_outputs_are_merged_then_bf16_rounded(self):
        output=np.stack((np.full((2,64,512),2.,np.float32),np.full((2,64,512),4.,np.float32)),axis=1)
        maxima=np.zeros((2,2,64),np.float32)
        sums=np.stack((np.full((2,64),1.,np.float32),np.full((2,64),3.,np.float32)),axis=1)
        def compare(actual,expected):
            return dict(error=float(np.max(np.abs(np.asarray(actual)-expected))))
        metric,state=runner.host_partial_control((output,maxima,sums),np.full((2,64,512),3.5,np.float32),
            compare,lambda m,tol:m["error"]<=tol)
        self.assertTrue(metric["passed"])
        np.testing.assert_array_equal(state.output,3.5)
        np.testing.assert_array_equal(state.total,4.)
        with self.assertRaises(ValueError):
            runner.host_partial_control((output.astype(np.float16),maxima,sums),state.output,compare,lambda *args:True)
        sums[:,1]=0
        maxima[:,1]=-np.inf
        output[:,1]=0
        metric,state=runner.host_partial_control((output,maxima,sums),output[:,0],compare,lambda m,tol:m["error"]<=tol)
        self.assertTrue(metric["passed"])
        self.assertEqual(metric["empty_partial_rows"],128)

    def test_partial_accessor_uses_query_then_split_axis_even_when_they_differ(self):
        output=np.zeros((3,2,64,512),np.float32)
        for query in range(3):
            output[query,0]=query*4
            output[query,1]=query*4+2
        maxima=np.zeros((3,2,64),np.float32)
        sums=np.ones_like(maxima)
        expected=np.broadcast_to((np.arange(3)*4+1)[:,None,None],(3,64,512)).astype(np.float32)
        compare=lambda actual,reference:dict(error=float(np.max(np.abs(actual-reference))))
        metric,state=runner.host_partial_control((output,maxima,sums),expected,
            compare,lambda m,tol:m["error"]<=tol)
        self.assertTrue(metric["passed"])
        np.testing.assert_array_equal(state.output,expected)

    def test_native_zero_max_empty_partial_does_not_dominate_negative_live_max(self):
        output=np.zeros((2,2,64,512),np.float32)
        output[:,0]=3
        maxima=np.zeros((2,2,64),np.float32)
        maxima[:,0]=-1000
        sums=np.zeros_like(maxima)
        sums[:,0]=257
        compare=lambda actual,reference:dict(error=float(np.max(np.abs(actual-reference))))
        for empty_max in (0.,-np.inf,np.float32(-np.finfo(np.float32).max)):
            maxima[:,1]=empty_max
            metric,state=runner.host_partial_control((output,maxima,sums),output[:,0],
                compare,lambda m,tol:m["error"]<=tol)
            self.assertTrue(metric["passed"])
            np.testing.assert_array_equal(state.output,3)
            np.testing.assert_array_equal(state.maximum,-1000)
            np.testing.assert_array_equal(state.total,257)
        for bad_empty_max in (np.nan,np.inf,1.,-42.,np.nextafter(np.float32(-np.finfo(np.float32).max),np.float32(0))):
            maxima[:,1]=bad_empty_max
            with self.assertRaises(ValueError):
                runner.host_partial_control((output,maxima,sums),output[:,0],compare,lambda *args:True)

    def test_zero_query_empty_tail_accepts_both_native_zero_and_minus_inf_maxima(self):
        output=np.ones((3,4,64,512),np.float32)
        output[:,1:]=0
        maxima=np.zeros((3,4,64),np.float32)
        sums=np.zeros_like(maxima)
        sums[:,0]=257
        for empty_max in (0.,-np.inf,np.float32(-np.finfo(np.float32).max)):
            maxima[:,1:]=empty_max
            gate=runner.zero_query_partial_control((output,maxima,sums),257)
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["expected_partial_counts"],[257.,0.,0.,0.])
        maxima[:,0]=1
        self.assertFalse(runner.zero_query_partial_control((output,maxima,sums),257)["passed"])
        maxima[:,0]=0
        output[:,1]=1
        self.assertFalse(runner.zero_query_partial_control((output,maxima,sums),257)["passed"])

    def test_all_empty_native_zero_max_partials_canonicalize_to_zero_mass(self):
        output=np.zeros((2,4,64,512),np.float32)
        maxima=np.zeros((2,4,64),np.float32)
        sums=np.zeros_like(maxima)
        compare=lambda actual,reference:dict(error=float(np.max(np.abs(actual-reference))))
        metric,state=runner.host_partial_control((output,maxima,sums),output[:,0],
            compare,lambda m,tol:m["error"]<=tol)
        self.assertTrue(metric["passed"])
        np.testing.assert_array_equal(state.output,0)
        np.testing.assert_array_equal(state.total,0)
        self.assertTrue(np.isneginf(state.maximum).all())
        self.assertTrue(runner.zero_query_partial_control((output,maxima,sums),0)["passed"])

    def test_empty_finite_sentinel_is_exact_and_never_waives_wrong_output_or_sum(self):
        finite_limit=np.float32(-np.finfo(np.float32).max)
        values=np.array([0.,-np.inf,finite_limit,-42.,np.nan,np.inf,
            np.nextafter(finite_limit,np.float32(0))],np.float32)
        np.testing.assert_array_equal(runner.supported_empty_maximum(values),
            [True,True,True,False,False,False,False])
        self.assertEqual(int(np.array([finite_limit]).view(np.uint32)[0]),0xff7fffff)
        output=np.zeros((1,2,64,512),np.float32)
        maxima=np.full((1,2,64),finite_limit,np.float32)
        sums=np.zeros_like(maxima)
        compare=lambda actual,reference:dict(error=float(np.max(np.abs(actual-reference))))
        output[:,1]=1
        with self.assertRaises(ValueError):
            runner.host_partial_control((output,maxima,sums),output[:,0],compare,lambda *args:True)
        output[:]=0
        for invalid in (-1.,np.nan,np.inf):
            sums[:,1]=invalid
            with self.assertRaises(ValueError):
                runner.host_partial_control((output,maxima,sums),output[:,0],compare,lambda *args:True)
        sums[:]=0
        sums[:,0]=258
        maxima[:,0]=0
        self.assertFalse(runner.zero_query_partial_control((output,maxima,sums),257)["passed"])

    def test_diagnostics_preserve_raw_bits_even_when_following_gate_rejects(self):
        output=np.zeros((1,2,64,512),np.float32)
        maxima=np.zeros((1,2,64),np.float32)
        maxima[:,1]=-42
        sums=np.zeros_like(maxima)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"partial"/"failure.npz"
            record=runner.partial_diagnostics((output,maxima,sums),path)
            self.assertTrue(path.is_file())
            self.assertEqual(record["empty_maximum_uint32_histogram"],
                {"0x00000000":64,"0xc2280000":64})
            self.assertEqual(record["known_empty_maximum_rows"],64)
            with self.assertRaises(ValueError):
                runner.host_partial_control((output,maxima,sums),output[:,0],
                    lambda *args:{},lambda *args:True)
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["partial_maximum"].view(np.uint32),maxima.view(np.uint32))
                np.testing.assert_array_equal(saved["partial_output"],output)
                np.testing.assert_array_equal(saved["partial_sum"],sums)


if __name__=="__main__":
    unittest.main()
