import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "run_logged_stage.py"
SPEC = importlib.util.spec_from_file_location("run_logged_stage", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class LoggedStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stage evidence ")
        self.root = Path(self.temp.name).resolve()
        self.console = io.BytesIO()

    def tearDown(self):
        self.temp.cleanup()

    def run_command(self, code, stage="baseline_build", extra=()):
        return runner.run_stage(stage, self.root, [sys.executable, "-c", code, *extra],
                                cwd=self.root, console=self.console)

    def test_capture_exit_and_hash_not_printed_pass(self):
        report = self.run_command("import sys; print('PASS'); print('stderr evidence', file=sys.stderr); sys.exit(7)")
        log = Path(report["log"]).read_bytes()
        self.assertEqual(report["status"], "ERROR")
        self.assertEqual(report["returncode"], 7)
        self.assertIn(b"PASS", log)
        self.assertIn(b"stderr evidence", log)
        self.assertEqual(log, self.console.getvalue())
        self.assertEqual(report["log_sha256"], hashlib.sha256(log).hexdigest())
        self.assertGreater(report["pid"], 0)
        self.assertTrue(report["started_utc"].endswith("Z"))
        self.assertTrue(report["ended_utc"].endswith("Z"))
        self.assertEqual(json.loads((Path(report["run_dir"]) / "status.json").read_text()), report)

    def test_arguments_spaces_metacharacters_and_cwd_are_literal(self):
        value = "two words; $(touch SHOULD_NOT_EXIST)"
        report = self.run_command("import json,os,sys; print(json.dumps([os.getcwd(), sys.argv[1:]]))", extra=(value,))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(json.loads(Path(report["log"]).read_text()), [str(self.root), [value]])
        self.assertEqual(report["argv"][-1], value)
        self.assertFalse((self.root / "SHOULD_NOT_EXIST").exists())

    def test_repeated_stage_keeps_previous_run(self):
        first = self.run_command("print('first')")
        second = self.run_command("print('second')")
        self.assertNotEqual(first["run_dir"], second["run_dir"])
        self.assertEqual(Path(first["log"]).read_text(), "first\n")
        self.assertEqual(len(list(self.root.glob("*/status.json"))), 2)
        self.assertEqual({p.parent.name.split("_")[0] for p in self.root.glob("*/status.json")}, {"baseline"})

    def test_fault_blocks_future_device_stages_but_not_build(self):
        first = self.run_command("print('ACL_ERROR_RT_AICORE_EXCEPTION 507015')", stage="accuracy")
        self.assertEqual(first["returncode"], 0)
        self.assertEqual(first["status"], "ERROR")
        self.assertTrue(first["device_fault"])
        marker = (self.root / "DEVICE_FAULT.json").read_bytes()
        for stage in ("accuracy", "performance"):
            blocked = self.run_command("from pathlib import Path; Path('executed').touch()", stage=stage)
            self.assertTrue(blocked["blocked"])
            self.assertIsNone(blocked["pid"])
            self.assertIsNone(blocked["returncode"])
            self.assertEqual(blocked["status"], "ERROR")
        self.assertFalse((self.root / "executed").exists())
        build = self.run_command("print('build remains possible')", stage="candidate_build")
        self.assertEqual(build["status"], "PASS")
        self.assertEqual((self.root / "DEVICE_FAULT.json").read_bytes(), marker)

    def test_fault_detection_across_stream_chunks(self):
        report = self.run_command("import os,time; os.write(1,b'GM in'); time.sleep(.05); os.write(1,b'valid address\\n')")
        self.assertEqual(report["status"], "ERROR")
        self.assertEqual(report["device_fault_signatures"], ["GM invalid"])

    def test_crash_negative_returncode_and_spawn_failure(self):
        report = self.run_command("import os,signal; os.kill(os.getpid(),signal.SIGTERM)")
        self.assertEqual(report["returncode"], -signal.SIGTERM)
        self.assertEqual(report["status"], "ERROR")
        absent = runner.run_stage("baseline_build", self.root, [str(self.root / "missing command")], console=self.console)
        self.assertEqual(absent["status"], "ERROR")
        self.assertIsNone(absent["returncode"])
        self.assertIn("FileNotFoundError", absent["error"])

    def test_cli_requires_explicit_separator_and_preserves_failure(self):
        command = [sys.executable, str(SCRIPT), "--stage", "accuracy", "--work-dir", str(self.root)]
        failure = subprocess.run(command + ["--", sys.executable, "-c", "print('PASS'); raise SystemExit(3)"], capture_output=True)
        self.assertEqual(failure.returncode, 1)
        self.assertEqual(len(list(self.root.glob("*/status.json"))), 1)
        for args in (command, command + ["--"]):
            self.assertEqual(subprocess.run(args, capture_output=True).returncode, 2)
        self.assertEqual(len(list(self.root.glob("*/status.json"))), 1)

    def test_help_does_not_require_a_stage_or_launch_a_command(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--stage", result.stdout)
        self.assertIn("--work-dir", result.stdout)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
