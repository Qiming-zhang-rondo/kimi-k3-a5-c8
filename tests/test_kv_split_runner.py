"""Host-only gates for the new KV-split runner; never imports Torch/NPU."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experimental/single_pass"))
import run_kv_split as runner


class KvSplitRunnerTest(unittest.TestCase):
    def test_failure_exports_are_cpu_only_and_do_not_change_original_evidence(self):
        with tempfile.TemporaryDirectory(prefix="split-failure-export-") as temp:
            path=Path(temp)/"results.json"
            report=dict(status="CONTROL_FAILED",version="qsfa-recent128-kv-split-a5-v1.1-native-empty-sentinel",
                args=dict(queries=4,splits=[1,2,4]),controls=dict(mathematical={
                    "w0_split1":dict(passed=False,relative_l2=1.16),
                    "w128_split1":dict(passed=False,relative_l2=1.14)}))
            path.write_text(json.dumps(report))
            before=path.read_bytes()
            torch_before={name for name in sys.modules if name=="torch" or name.startswith("torch.")}
            outputs=runner.write_failure_diagnostics(path)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(torch_before,{name for name in sys.modules if name=="torch" or name.startswith("torch.")})
            self.assertEqual(len(outputs),2)
            for output,case in zip(outputs,runner.FAILURE_CASES):
                self.assertLessEqual(output.stat().st_size,50001)
                data=json.loads(output.read_text())
                self.assertTrue(data["read_only"])
                self.assertFalse(data["npu_executed"])
                self.assertEqual(data["source_results_sha256"],hashlib.sha256(before).hexdigest())
                self.assertEqual(data["experiment_status"],"CONTROL_FAILED")
                self.assertFalse(data["cases"][case]["mathematical"]["passed"])
                self.assertEqual(data["full_outputs"]["status"],"NOT_AVAILABLE")

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


class KvSplitVerificationTest(unittest.TestCase):
    def arguments(self,**changes):
        values=dict(verify_comparison=True,windows=[0,128],splits=[1,2,4],return_stats=False,
            timing="graph",repeats=20)
        values.update(changes)
        return SimpleNamespace(**values)

    def calls(self):
        names={name for pair in runner.VERIFICATION_PAIRS for name in pair}
        result={}
        for name in names:
            def call(name=name):
                raise AssertionError("The host pair fixture must not invoke an operator: "+name)
            call.arm=name
            result[name]=call
        return result

    def timing(self):
        rounds=[]
        for index in range(20):
            baseline_event=80.+index*.17
            candidate_event=63.+index*.11
            rounds.append(dict(round=index,
                order=["baseline","candidate"] if index%2==0 else ["candidate","baseline"],
                baseline=dict(event_us=dict(median=baseline_event,samples=[baseline_event-.01,baseline_event+.01]),
                    wall_us=dict(median=baseline_event+2)),
                candidate=dict(event_us=dict(median=candidate_event,samples=[candidate_event-.01,candidate_event+.01]),
                    wall_us=dict(median=candidate_event+3)),
                event_us_ratio=candidate_event/baseline_event,
                wall_us_ratio=(candidate_event+3)/(baseline_event+2)))
        return dict(graph=dict(status="MEASURED",rounds=rounds,unmodified_raw_marker="retain-this-field"))

    def test_verification_options_reject_incomplete_or_mismatched_contract_before_device(self):
        runner.validate_verification_options(self.arguments())
        runner.validate_verification_options(self.arguments(timing="eager"))
        for changes in (dict(windows=[128]),dict(splits=[1,2]),dict(repeats=19),
                        dict(return_stats=True),dict(timing="none")):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                runner.validate_verification_options(self.arguments(**changes))

    def test_exact_six_pairs_keep_twenty_event_wall_rounds_and_raw_order(self):
        expected=(("installed_c4","w0_v4"),("installed_c4","w0_split4"),
            ("installed_c4","w128_split4"),("w0_v4","w0_split4"),
            ("w128_v4","w128_split4"),("w0_split4","w128_split4"))
        self.assertEqual(runner.VERIFICATION_PAIRS,expected)
        calls=self.calls();trace=[];raw=[]
        def paired(baseline,candidate,args,benchmark):
            trace.append((baseline.arm,candidate.arm))
            item=self.timing();raw.append(copy.deepcopy(item))
            return item
        results={};persisted=[]
        actual=runner.measure_verification(calls,{"math":{"passed":True}},{"poison":{"passed":True}},
            self.arguments(),None,paired,lambda:persisted.append(len(results)),results=results)
        self.assertIs(actual,results)
        self.assertEqual(trace,list(expected))
        self.assertEqual(persisted,[1,2,3,4,5,6])
        for index,(baseline,candidate) in enumerate(expected):
            item=results[candidate+"_vs_"+baseline]
            self.assertEqual((item["baseline"],item["candidate"]),(baseline,candidate))
            self.assertEqual(item["timing"],raw[index])
            self.assertEqual(len(item["timing"]["graph"]["rounds"]),20)

    def test_no_verification_pair_runs_when_any_required_gate_or_arm_is_missing(self):
        calls=self.calls();trace=[]
        def paired(*unused):
            trace.append(True)
            return self.timing()
        cases=((calls,{"math":{"passed":False}},{"poison":{"passed":True}},self.arguments()),
               (calls,{"math":{"passed":True}},{"poison":{"passed":False}},self.arguments()),
               (calls,{"math":{"passed":True}},{"poison":{"passed":True}},self.arguments(repeats=19)),
               ({k:v for k,v in calls.items() if k!="installed_c4"},
                {"math":{"passed":True}},{"poison":{"passed":True}},self.arguments()))
        for selected,controls,graphs,args in cases:
            with self.assertRaises(runner.ControlFailed):
                runner.measure_verification(selected,controls,graphs,args,None,paired)
        self.assertEqual(trace,[])

    def test_dropped_round_clock_ratio_or_abba_order_cannot_claim_verification(self):
        def damage(payload,kind):
            rounds=payload["graph"]["rounds"]
            if kind=="missing_round": rounds.pop()
            elif kind=="missing_wall": del rounds[0]["candidate"]["wall_us"]
            elif kind=="missing_event": del rounds[2]["baseline"]["event_us"]
            elif kind=="missing_ratio": del rounds[3]["wall_us_ratio"]
            elif kind=="wrong_ratio": rounds[4]["event_us_ratio"]*=.5
            elif kind=="wrong_order": rounds[1]["order"]=["baseline","candidate"]
            elif kind=="wrong_round_index": rounds[2]["round"]=3
            elif kind=="nonfinite": rounds[3]["candidate"]["event_us"]["median"]=float("nan")
            elif kind=="nonpositive": rounds[3]["baseline"]["wall_us"]["median"]=0
            elif kind=="wrong_mode": payload["eager"]=payload.pop("graph")
        for kind in ("missing_round","missing_wall","missing_event","missing_ratio","wrong_ratio",
                     "wrong_order","wrong_round_index","nonfinite","nonpositive","wrong_mode"):
            payload=self.timing();damage(payload,kind)
            with self.subTest(kind=kind),self.assertRaises(runner.ControlFailed):
                runner.measure_verification(self.calls(),{"math":{"passed":True}},
                    {"poison":{"passed":True}},self.arguments(),None,lambda *unused:payload)

    def test_posttiming_reads_wrong_saved_output_without_rerunning_to_repair_it(self):
        class Prepared:
            def __init__(self,value):
                self.value=np.full((1,64,512),value,np.float32)
                self.reads=self.runs=0
            def saved(self):
                self.reads+=1
                return self.value
            def run(self):
                self.runs+=1
                self.value.fill(1)
                raise AssertionError("posttiming validation must not repair output by rerunning")
        broken=Prepared(0);correct=Prepared(2)
        references={0:np.ones((1,64,512),np.float64),128:np.full((1,64,512),2,np.float64)}
        compare=lambda actual,reference:dict(error=float(np.max(np.abs(np.asarray(actual)-reference))))
        checks,snapshots=runner.check_saved_verification_outputs(
            {"w0_v4":broken.saved,"installed_c4":broken.saved,"w128_split4":correct.saved},
            {},{"w0_v4":0,"installed_c4":0,"w128_split4":128},references,
            lambda value:np.asarray(value).copy(),compare,lambda metric,limit:metric["error"]<=limit)
        self.assertFalse(runner.all_passed(checks))
        self.assertTrue(snapshots)
        self.assertEqual((broken.reads,correct.reads),(2,1))
        self.assertEqual((broken.runs,correct.runs),(0,0))
        np.testing.assert_array_equal(broken.value,0)

    def test_posttiming_correct_final_does_not_hide_corrupted_saved_partial_state(self):
        final=np.full((1,64,512),2,np.float32)
        partial_o=np.full((1,4,64,512),9,np.float32)
        maximum=np.zeros((1,4,64),np.float32)
        total=np.ones_like(maximum)
        trace=[]
        def get_partial():
            trace.append("read_partial")
            return partial_o,maximum,total
        compare=lambda actual,reference:dict(error=float(np.max(np.abs(np.asarray(actual)-reference))))
        checks,_=runner.check_saved_verification_outputs({"w128_split4":lambda:final},
            {"w128_split4":get_partial},{"w128_split4":128},{128:final.copy()},
            lambda value:np.asarray(value).copy(),compare,lambda metric,limit:metric["error"]<=limit)
        self.assertFalse(runner.all_passed(checks))
        self.assertEqual(trace,["read_partial"])


if __name__=="__main__":
    unittest.main()
