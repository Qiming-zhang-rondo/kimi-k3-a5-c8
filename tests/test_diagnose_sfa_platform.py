import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/diagnose_sfa_platform.py"
SPEC = importlib.util.spec_from_file_location("diagnose_sfa_platform", SCRIPT)
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


FAKE_PROBE = r'''
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--work-dir'); p.add_argument('--soc-version')
a,_=p.parse_known_args()
if 'Crash' in a.soc_version:
    print('Simulated get_platform_info failure; no fatal signal',flush=True)
    raise SystemExit(86)
out=Path(a.work_dir)/'original_sfa_test'
out.mkdir(parents=True)
artifacts=[]
if 'Good' in a.soc_version or 'Nonzero' in a.soc_version:
    obj=out/'kernel.o'; obj.write_bytes(b'test-only fake compiler object')
    artifacts=[{'path':str(obj),'bytes':obj.stat().st_size,'sha256':hashlib.sha256(obj.read_bytes()).hexdigest()}]
report={'status':'PASS','compile_requested':True,'compile_calls':[{'test_only':True}],
        'compiler_artifacts':artifacts,'soc':{'actual':a.soc_version}}
(out/'probe_report.json').write_text(json.dumps(report))
if 'Nonzero' in a.soc_version: raise SystemExit(4)
'''


class PlatformDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sfa platform ")
        self.root = Path(self.temp.name).resolve()
        self.sdk = self.root / "sdk"
        self.config = self.sdk / "compiler/data/platform_config"
        self.config.mkdir(parents=True)
        self.fake = self.root / "fake probe.py"
        self.fake.write_text(FAKE_PROBE)

    def tearDown(self):
        self.temp.cleanup()

    def discover(self, roots=None, **kwargs):
        return diag.discover(roots or [self.sdk], environ={}, known_roots=(), **kwargs)

    def run_diagnosis(self, **kwargs):
        return diag.diagnose(self.root / "evidence", [self.sdk], environ={}, known_roots=(),
                             probe_script=self.fake, quiet=True, **kwargs)

    def test_actual_filename_section_and_json_names_preserve_evidence(self):
        path = self.config / "Ascend950PR_9599.ini"
        path.write_text('[SocVersion]\nsocVersion=Ascend950PR_9599\nShort_SoC_version=Ascend950\n')
        (self.config / "other.json").write_text('{"socVersion":"Ascend950DT_9582"}')
        (self.config / "junk.ini").write_text('[other]\ncomment=Ascend950Invented\nsocVersion=Ascend950;touch bad\n')
        (self.config / "platform950.ini").write_text('[other]\nversion=1\n')
        report = self.discover()
        names = [item["name"] for item in report["candidates"]]
        self.assertEqual(names, ["Ascend950PR_9599", "Ascend950DT_9582"])
        self.assertTrue(all(item["configured"] for item in report["candidates"]))
        record = next(item for item in report["config_files"] if item["path"] == str(path))
        self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertTrue(any(item["source"] == "filename" for item in record["soc_names"]))

    def test_colliding_configs_and_sdk_root_precedence_are_not_overwritten(self):
        first = self.config / "Ascend950Good.ini"
        first.write_text('[x]\nvalue=first\n')
        other = self.root / "other sdk"
        second = other / "data/platform_config/Ascend950Good.ini"
        second.parent.mkdir(parents=True)
        second.write_text('[x]\nvalue=second\n')
        report = diag.discover([self.sdk], environ={"ASCEND_HOME_PATH": str(other)}, known_roots=())
        self.assertEqual([x["source"] for x in report["roots"]], ["explicit", "ASCEND_HOME_PATH"])
        self.assertEqual(len(report["candidates"]), 1)
        self.assertEqual(len(report["candidates"][0]["evidence"]), 2)
        self.assertEqual(len(report["config_collisions"]), 1)
        self.assertEqual(len(report["config_collisions"][0]["sha256_values"]), 2)

    def test_inspect_only_never_runs_process_and_no_config_does_not_guess(self):
        def forbidden(*a, **k):
            self.fail("Unexpected compile subprocess")
        report = self.run_diagnosis(stage_runner=forbidden)
        self.assertEqual(report["status"], "CONFIG_NOT_FOUND")
        self.assertEqual(report["attempts"], [])
        report = self.run_diagnosis(compile_requested=True, stage_runner=forbidden)
        self.assertEqual(report["status"], "CONFIG_NOT_FOUND")
        (self.config / "Ascend950Good.ini").write_text('[x]\na=1\n')
        report = self.run_diagnosis(stage_runner=forbidden)
        self.assertEqual(report["status"], "CONFIG_DISCOVERED")
        self.assertEqual(report["attempts"], [])

    def test_explicit_unknown_is_not_mislabeled_configured(self):
        report = self.run_diagnosis(soc_version="Ascend950Explicit")
        self.assertEqual(report["status"], "CONFIG_NOT_FOUND")
        candidate = report["planned_candidates"][0]
        self.assertFalse(candidate["configured"])
        self.assertTrue(candidate["explicit"])
        self.assertEqual(candidate["evidence"], [])
        with self.assertRaises(ValueError):
            self.run_diagnosis(soc_version="Ascend950; echo injected")

    def test_simulated_child_segfault_is_preserved_then_next_candidate_can_pass(self):
        (self.config / "Ascend950Crash.ini").write_text('[x]\na=1\n')
        (self.config / "Ascend950Good.ini").write_text('[x]\na=1\n')
        real_runner = diag.load_stage_runner()
        def injected_signal(stage, work_dir, command, **kwargs):
            if "Ascend950Crash" not in command:
                return real_runner(stage, work_dir, command, **kwargs)
            # Test-only process metadata injection. Never launch a crashing
            # Python on a shared desktop just to test signal classification.
            root = Path(work_dir)
            root.mkdir(parents=True)
            log = root / "simulated_signal.log"
            log.write_text("SIMULATED SDK get_platform_info SIGSEGV\n")
            return {"status": "ERROR", "returncode": -signal.SIGSEGV,
                    "pid": None, "log": str(log), "simulation": True}
        report = self.run_diagnosis(compile_requested=True, stage_runner=injected_signal)
        self.assertEqual(report["status"], "COMPILE_PASS")
        self.assertEqual(len(report["attempts"]), 2)
        first, second = report["attempts"]
        self.assertEqual(first["process"]["returncode"], -signal.SIGSEGV)
        self.assertTrue(first["process"]["simulation"])
        self.assertEqual(first["signal"]["name"], "SIGSEGV")
        self.assertIn("get_platform_info", first["log_signatures"])
        self.assertFalse(first["probe"]["passed"])
        self.assertTrue(second["compile_passed"])
        self.assertEqual(report["selected"]["name"], "Ascend950Good")
        self.assertNotEqual(first["process"]["pid"], second["process"]["pid"])
        self.assertEqual(json.loads(Path(report["report"]).read_text()), report)

    def test_nested_pass_needs_real_artifact_and_zero_child_status(self):
        (self.config / "Ascend950FakePass.ini").write_text('[x]\na=1\n')
        (self.config / "Ascend950Nonzero.ini").write_text('[x]\na=1\n')
        report = self.run_diagnosis(compile_requested=True)
        self.assertEqual(report["status"], "COMPILE_FAILED")
        first, second = report["attempts"]
        self.assertEqual(first["process"]["returncode"], 0)
        self.assertFalse(first["probe"]["passed"])
        self.assertTrue(second["probe"]["passed"])
        self.assertEqual(second["process"]["returncode"], 4)
        self.assertFalse(second["compile_passed"])
        self.assertIsNone(report["selected"])

    def test_attempt_and_search_bounds(self):
        for index in range(5):
            (self.config / ("Ascend950FakePass%d.ini" % index)).write_text('[x]\na=1\n')
        deep = self.sdk / "a/b/c/d/e/f/g/platform_config"
        deep.mkdir(parents=True)
        (deep / "Ascend950TooDeep.ini").write_text('[x]\na=1\n')
        report = self.run_diagnosis(compile_requested=True, max_attempts=2)
        self.assertEqual(len(report["attempts"]), 2)
        self.assertEqual(len(report["candidates_not_attempted_due_to_limit"]), 3)
        self.assertNotIn("Ascend950TooDeep", [x["name"] for x in report["discovery"]["candidates"]])
        with self.assertRaises(ValueError):
            self.run_diagnosis(max_attempts=5)

    def test_architecture_data_layout_and_short_names_are_not_guessed_targets(self):
        config = self.sdk / "aarch64-linux/data/platform_config/Ascend950DT_9582.ini"
        config.parent.mkdir(parents=True)
        config.write_text('[SocVersion]\nshort_soc_version=Ascend950\nSocVersion=Ascend950DT_9582\n')
        nested = self.config / "nested.json"
        nested.write_text('{"socVersion":{"short":"Ascend950"},"short_soc_version":"Ascend950DT"}')
        report = self.discover()
        self.assertEqual([item["name"] for item in report["candidates"]], ["Ascend950DT_9582"])


if __name__ == "__main__":
    unittest.main()
