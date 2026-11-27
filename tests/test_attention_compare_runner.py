"""Host regressions for same-input C4/BF16/split4 comparisons; no device imports."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

SOURCE=Path(__file__).resolve().parents[1]/"experimental"/"single_pass"
sys.path.insert(0,str(SOURCE))
import run_attention_compare as entry
import run_kv_split as runner
from attention_compare_metrics import compare_outputs


class AttentionCompareRunnerTests(unittest.TestCase):
    def arguments(self,*extra,output="/tmp/threeway-output"):
        return entry.parser().parse_args(["--payload-dir","/tmp/payload","--library","/tmp/reader.so",
            "--device","npu:7","--output",output,*extra])

    def timing(self):
        rounds=[]
        for index in range(20):
            baseline,candidate=80.+index*.17,62.+index*.11
            rounds.append(dict(round=index,
                order=["baseline","candidate"] if index%2==0 else ["candidate","baseline"],
                baseline=dict(event_us=dict(median=baseline,samples=[baseline]),wall_us=dict(median=baseline+2)),
                candidate=dict(event_us=dict(median=candidate,samples=[candidate]),wall_us=dict(median=candidate+3)),
                event_us_ratio=candidate/baseline,wall_us_ratio=(candidate+3)/(baseline+2)))
        return dict(graph=dict(status="MEASURED",rounds=rounds,raw_marker="do not round or reconstruct"))

    def calls(self):
        result={}
        for name in ("installed_c4","installed_bf16","w128_split4","w0_split4","w128_v4"):
            def call(name=name):
                raise AssertionError("Host validation must not invoke an operator: "+name)
            call.arm=name
            result[name]=call
        return result

    def test_defaults_request_graph_twenty_rounds_split4_and_no_returned_stats(self):
        args=self.arguments()
        self.assertTrue(args.compare_bf16)
        self.assertTrue(args.verify_comparison)
        self.assertEqual((args.windows,args.splits),([0,128],[4]))
        self.assertEqual((args.timing,args.repeats,args.queries),("graph",20,4))
        self.assertFalse(args.return_stats)
        runner.validate_verification_options(args)
        # An explicit accuracy-only run remains possible and cannot claim timing.
        runner.validate_verification_options(self.arguments("--timing","none"))

    def test_invalid_overrides_are_rejected_before_output_or_execute(self):
        invalid=(("--splits","1","4"),("--splits","2"),("--windows","128"),
            ("--repeats","19"),("--return-stats",),("--queries","8"))
        with tempfile.TemporaryDirectory() as directory:
            for index,options in enumerate(invalid):
                destination=Path(directory)/str(index)
                with self.subTest(options=options),mock.patch.object(runner,"execute") as execute:
                    with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                        entry.main(["--payload-dir","/tmp/payload","--library","/tmp/reader.so",
                            "--device","npu:7","--output",str(destination),*options])
                    self.assertEqual(error.exception.code,2)
                    execute.assert_not_called()
                    self.assertFalse(destination.exists())
        args=self.arguments()
        args.verify_comparison=False
        with self.assertRaises(ValueError):runner.validate_verification_options(args)

    def test_exact_three_direct_pairs_preserve_twenty_raw_event_wall_rounds(self):
        expected=(("installed_c4","installed_bf16"),("installed_c4","w128_split4"),
                  ("installed_bf16","w128_split4"))
        self.assertEqual(runner.FULL_PRECISION_PAIRS,expected)
        trace,original,persisted=[],[],[]
        def pair(baseline,candidate,args,benchmark):
            trace.append((baseline.arm,candidate.arm))
            value=self.timing();original.append(copy.deepcopy(value))
            return value
        result={}
        actual=runner.measure_verification(self.calls(),{"math":{"passed":True}},
            {"poison":{"passed":True}},self.arguments(),None,pair,
            lambda:persisted.append(len(result)),results=result,pairs=runner.FULL_PRECISION_PAIRS)
        self.assertIs(actual,result)
        self.assertEqual(trace,list(expected))
        self.assertEqual(persisted,[1,2,3])
        self.assertEqual(len(result),3)
        for index,(baseline,candidate) in enumerate(expected):
            value=result[candidate+"_vs_"+baseline]
            self.assertEqual((value["baseline"],value["candidate"]),(baseline,candidate))
            self.assertEqual(value["timing"],original[index])

    def test_threeway_pairs_keep_original_raw_pair_validation_and_accuracy_gates(self):
        for kind in ("round","clock","ratio","order","mode","nonfinite"):
            raw=self.timing()
            rounds=raw["graph"]["rounds"]
            if kind=="round":rounds.pop()
            elif kind=="clock":del rounds[2]["candidate"]["wall_us"]
            elif kind=="ratio":rounds[2]["event_us_ratio"]*=.5
            elif kind=="order":rounds[1]["order"]=["baseline","candidate"]
            elif kind=="mode":raw["eager"]=raw.pop("graph")
            elif kind=="nonfinite":rounds[0]["baseline"]["event_us"]["median"]=float("nan")
            with self.subTest(kind=kind),self.assertRaises(runner.ControlFailed):
                runner.measure_verification(self.calls(),{"math":{"passed":True}},
                    {"poison":{"passed":True}},self.arguments(),None,lambda *unused:raw,
                    pairs=runner.FULL_PRECISION_PAIRS)
        for controls,graphs in (({"passed":False},{"passed":True}),
                                ({"passed":True},{"passed":False})):
            pair=mock.Mock(side_effect=AssertionError("timing before controls"))
            with self.assertRaises(runner.ControlFailed):
                runner.measure_verification(self.calls(),controls,graphs,self.arguments(),None,pair,
                    pairs=runner.FULL_PRECISION_PAIRS)
            pair.assert_not_called()

    def test_posttiming_bf16_uses_its_own_reference_and_never_repairs_saved_output(self):
        refs={0:np.ones((1,64,512),np.float64),128:np.full((1,64,512),2,np.float64),
              "full_bf16":np.full((1,64,512),7,np.float64)}
        keys={"installed_c4":0,"w128_split4":128,"installed_bf16":"full_bf16"}
        values={name:refs[key].astype(np.float32).copy() for name,key in keys.items()}
        trace=[]
        def read(name):
            trace.append(name)
            return values[name]
        readers={name:lambda name=name:read(name) for name in keys}
        acceptable=lambda metric,limit:metric["finite"] and metric["relative_l2"]<=limit and metric["max_row_relative_l2"]<=limit
        checks,snapshots=runner.check_saved_verification_outputs(readers,{},keys,refs,
            lambda value:np.asarray(value).copy(),compare_outputs,acceptable)
        self.assertTrue(runner.all_passed(checks))
        self.assertEqual(trace,list(keys))
        np.testing.assert_array_equal(snapshots["installed_bf16"],7)
        # A BF16 buffer equal to C4 is still wrong against the BF16 oracle.
        values["installed_bf16"].fill(1)
        checks,snapshots=runner.check_saved_verification_outputs(readers,{},keys,refs,
            lambda value:np.asarray(value).copy(),compare_outputs,acceptable)
        self.assertFalse(checks["installed_bf16"]["mathematical"]["passed"])
        self.assertTrue(checks["installed_c4"]["mathematical"]["passed"])
        np.testing.assert_array_equal(values["installed_bf16"],1)
        np.testing.assert_array_equal(snapshots["installed_bf16"],1)

    def test_failed_main_persists_real_failure_and_lossless_export_without_device_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)/"failed"
            before={name for name in sys.modules if name=="torch" or name.startswith("torch.")}
            with mock.patch.object(runner,"execute",side_effect=runner.ControlFailed("actual failed control")),\
                    contextlib.redirect_stdout(io.StringIO()):
                code=entry.main(["--payload-dir","/tmp/payload","--library","/tmp/reader.so",
                    "--device","npu:7","--output",str(destination)])
            self.assertEqual(code,1)
            raw=(destination/"results.json").read_bytes()
            report=json.loads(raw)
            self.assertEqual(report["status"],"CONTROL_FAILED")
            self.assertIn("actual failed control",report["traceback"])
            self.assertEqual(json.loads((destination/"summary.json").read_text())["status"],"CONTROL_FAILED")
            from export_verification_evidence import restore_evidence
            self.assertEqual(restore_evidence(destination/"verification_evidence"/"manifest.json"),raw)
            self.assertEqual(before,{name for name in sys.modules if name=="torch" or name.startswith("torch.")})

    def test_entry_import_and_parser_never_import_torch_or_npu(self):
        script="""import importlib.abc,sys
class RejectDeviceImports(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','cloud_ops_turbo'):
   raise AssertionError('unexpected device import: '+fullname)
sys.meta_path.insert(0,RejectDeviceImports())
sys.path.insert(0,sys.argv[1])
import run_attention_compare
args=run_attention_compare.parser().parse_args(['--payload-dir','/tmp/p','--library','/tmp/l','--device','npu:7','--output','/tmp/o'])
assert args.compare_bf16 and args.timing=='graph' and args.repeats==20
assert not any(n=='torch' or n.startswith('torch.') or n=='torch_npu' for n in sys.modules)
print('HOST_ONLY')
"""
        result=subprocess.run([sys.executable,"-I","-c",script,str(SOURCE)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),"HOST_ONLY")


if __name__=="__main__":unittest.main()
