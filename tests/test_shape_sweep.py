"""CPU-only sweep planning, exact paired summaries and device-fault stop tests."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("shape_sweep",ROOT/"direct_bridge/run_shape_sweep.py")
sweep=importlib.util.module_from_spec(spec);spec.loader.exec_module(sweep)


class ShapeSweepTests(unittest.TestCase):
    def test_large_q_and_row_batch_are_independent_and_large_coverage_comes_first(self):
        plan=sweep.cases([4,32,1024],[8192],[16,32,64],[4,8,16])
        self.assertEqual([(r["queries"],r["aic_count"]) for r in plan[:3]],[(4,4),(32,16),(1024,16)])
        measured=[r for r in plan if r["status"]=="PENDING"]
        self.assertEqual({r["queries"] for r in measured},{4,32,1024})
        self.assertEqual({r["merge_rows"] for r in measured},{16,32})
        self.assertTrue(all(r["status"]=="DEGENERATE_ALIAS_OF32" for r in plan if r["merge_rows"]==64))
        self.assertTrue(all(r["aic_count"]<=r["queries"] for r in plan))

    def test_dry_run_never_reads_sdk_builds_or_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"not-created"
            with contextlib.redirect_stdout(io.StringIO()),patch.object(sweep,"execute",side_effect=AssertionError("no execution")):
                result=sweep.main(["--work-dir",str(target),"--device","npu:3","--dry-run"])
            self.assertEqual(result,0);self.assertFalse(target.exists())

    def test_cache_inherits_only_real_build_settings_and_preserves_space_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,"CMakeCache.txt").write_text("SFA_SDK_ROOT:PATH=/toolkit with spaces\nASC_DIR:PATH=/toolkit with spaces/asc\nSECRET:STRING=do-not-copy\nCMAKE_ASC_FLAGS:STRING=-O2 -g\n")
            options=sweep.cache_options(tmp)
            self.assertIn("-DSFA_SDK_ROOT=/toolkit with spaces",options)
            self.assertIn("-DCMAKE_ASC_FLAGS=-O2 -g",options)
            self.assertNotIn("SECRET"," ".join(options))

    def test_summary_uses_actual_per_round_ratios_not_ratio_of_medians(self):
        rounds=[dict(baseline={"event_us":{"median":100}},candidate={"event_us":{"median":200}},event_us_ratio=2),
                dict(baseline={"event_us":{"median":200}},candidate={"event_us":{"median":100}},event_us_ratio=.5)]
        result=sweep.summarize(dict(status="MEASURED",timings={"pair":{"graph":{"rounds":rounds}}}))
        self.assertEqual(result["pairs"]["pair"]["ratio"],1.25)

    def test_device_fault_stops_all_remaining_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/"validated";old.mkdir()
            (old/"CMakeCache.txt").write_text("SFA_SDK_ROOT:PATH=/sdk\n")
            children=[]
            def execute(command,log):
                if "--output" in command:
                    children.append(command)
                    output=Path(command[command.index("--output")+1]);output.mkdir()
                    (output/"results.json").write_text(json.dumps({"status":"DEVICE_ERROR"}))
                    return 1
                return 0
            with patch.object(sweep,"execute",side_effect=execute),contextlib.redirect_stdout(io.StringIO()):
                code=sweep.main(["--validated-build",str(old),"--work-dir",str(root/"new"),
                    "--fault-marker",str(root/"fault.json"),"--device","npu:3","--queries","4,32", "--keys","8192","--merge-rows","16"])
            self.assertEqual(code,1);self.assertEqual(len(children),1)
            report=json.loads((root/"new/matrix.json").read_text())
            self.assertEqual(report["status"],"STOPPED_DEVICE_FAULT")


if __name__=="__main__":
    unittest.main()
