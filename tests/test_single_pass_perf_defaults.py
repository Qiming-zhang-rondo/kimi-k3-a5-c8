"""Verify safe launch defaults and lossless extraction without loading Torch."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]/"experimental/single_pass"


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class PerfDefaultsTest(unittest.TestCase):
    def test_aic_defaults_and_explicit_comparison(self):
        m=load("single_pass_launch_defaults","run_single_pass.py")
        self.assertEqual([m.select_aic_count(q) for q in (1,3,4,6,32,1024)],[1,1,4,2,16,16])
        self.assertEqual(m.select_aic_count(4,1),1)
        for q,aic in ((0,None),(4,3),(4,8),(4,True)):
            with self.assertRaises(ValueError):m.select_aic_count(q,aic)

    def test_extract_retains_failed_rounds_original_control_and_explicit_aic(self):
        m=load("single_pass_raw_evidence","collect_run_evidence.py")
        rounds=[dict(round=0,order=["candidate","baseline"],status="ERROR",error="actual failure")]
        data=dict(version="qsfa-recent128-single-pass-a5-v1",status="ERROR",
            args=dict(queries=4,aic_count=1,timing="graph"),
            controls=dict(single_pass_w128=dict(passed=False,finite=True,relative_l2=.02,max_row_relative_l2=.03)),
            timings=dict(original_c4_direct=dict(graph=dict(status="ERROR",rounds=rounds))),
            all_accuracy_controls_passed=False,
            routed_pattern_controls=dict(alternating_prefix=dict(passed=False,
                checks=dict(routed_vs_overlay=dict(relative_l2=.02,passed=False)))))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"results.json";path.write_text(json.dumps(data))
            result=m.collect(path)
        self.assertEqual(result["args"]["aic_count"],1)
        self.assertEqual(result["run_status"],"ERROR")
        self.assertFalse(result["controls"]["single_pass_w128"]["passed"])
        self.assertEqual(result["timings"]["original_c4_direct"]["graph"]["rounds"],rounds)
        self.assertIsNone(result["launch_geometry"])
        self.assertEqual(result["routed_pattern_controls"],data["routed_pattern_controls"])
        self.assertEqual(m.portable(dict(bad=float("nan"),high=float("inf"))),dict(bad="nan",high="inf"))


if __name__=="__main__":unittest.main()
