"""Exercise the real fusion launcher's host paths; never access an SDK or NPU.

For the ACL-mock tests, remove exactly one known ASC launch delimiter and call
the entry's empty host-pass body. This tests admission/fault/control gates,
not AscendC lowering, scheduling, attention correctness or performance.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
STANDALONE = ROOT / "standalone"
helper_spec = importlib.util.spec_from_file_location(
    "sfa_direct_launcher_mock_helpers", ROOT / "tests/test_sfa_direct_launcher.py")
helpers = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helpers)


class FusionLauncherTests(unittest.TestCase):
    def test_real_host_preflight_and_explicit_device_refusal(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("Host C++ compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="sfa-fusion-preflight-") as temp:
            binary = Path(temp) / "preflight"
            command = [compiler, "-std=c++17", "-O2", "-x", "c++",
                       "-DSFA_FUSION_HOST_PREFLIGHT_ONLY=1", "-DSFA_PROBE_TEMPLATE_MODE=0",
                       "-I", str(ROOT / "standalone"),
                       str(ROOT / "standalone/sfa_fusion_launch.asc"), "-o", str(binary)]
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            preflight = subprocess.run([str(binary)], cwd=temp, capture_output=True, text=True, timeout=10)
            self.assertEqual(preflight.returncode, 0, preflight.stderr)
            self.assertIn("FP32_output_bytes=524288", preflight.stdout)
            self.assertIn("returnSoftmaxLse=false", preflight.stdout)
            self.assertFalse((Path(temp) / "sfa_fusion_device_results.json").exists())
            refused = subprocess.run([str(binary), "--run", "--device", "0"],
                                     cwd=temp, capture_output=True, text=True, timeout=10)
            self.assertEqual(refused.returncode, 2)
            self.assertIn("Host-only build cannot run", refused.stderr)
            self.assertFalse((Path(temp) / "sfa_fusion_device_results.json").exists())


class FusionLauncherHostControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("Host C++ compiler unavailable; no device test is claimed")
        cls.build = tempfile.TemporaryDirectory(prefix="sfa-fusion-launcher-host-")
        cls.addClassCleanup(cls.build.cleanup)
        directory = Path(cls.build.name)
        source = (STANDALONE / "sfa_fusion_launch.asc").read_text()
        transformed, count = re.subn(
            r"<<<\s*sfa_fixed::launchAic\s*,\s*nullptr\s*,\s*stream\s*>>>", "", source)
        if count != 1 or "<<<" in transformed:
            raise AssertionError("Expected exactly one known ASC launch; never silently rewrite new launches")
        (directory / "sfa_fusion_launch_host.cpp").write_text(transformed)
        (directory / "acl").mkdir()
        sync_failure = "Probe::Call(); ++Probe::syncs; Probe::fault=true; return 507015;"
        if helpers.ACL_STUB.count(sync_failure) != 1:
            raise AssertionError("ACL mock synchronization contract changed")
        acl_stub = helpers.ACL_STUB.replace(
            sync_failure,
            'Probe::Call(); ++Probe::syncs; if (std::getenv("SFA_HOST_MOCK_SYNC_SUCCESS")) return 0; '
            'Probe::fault=true; return 507015;')
        (directory / "acl/acl.h").write_text(acl_stub)
        (directory / "kernel_operator.h").write_text(helpers.KERNEL_STUB)
        (directory / "driver.cpp").write_text(helpers.DRIVER.replace(
            '#include "sfa_direct_launch_host.cpp"', '#include "sfa_fusion_launch_host.cpp"'))
        cls.binary = directory / "fusion_launcher_host"
        result = subprocess.run(
            [compiler, "-std=c++17", "-O0", "-D__NPU_ARCH__=0", "-DSFA_PROBE_TEMPLATE_MODE=0",
             "-I", str(directory), "-I", str(STANDALONE), str(directory / "driver.cpp"),
             "-o", str(cls.binary)], capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def run_case(self, directory, args, sync_success=False):
        environment = dict(os.environ)
        environment.pop("SFA_HOST_MOCK_SYNC_SUCCESS", None)
        if sync_success:
            environment["SFA_HOST_MOCK_SYNC_SUCCESS"] = "1"
        result = subprocess.run([str(self.binary), *args], cwd=directory, env=environment,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [line.removeprefix("HOST_RESULT ") for line in result.stdout.splitlines()
                   if line.startswith("HOST_RESULT ")]
        self.assertEqual(len(records), 1, result.stdout + result.stderr)
        return json.loads(records[0])

    def test_default_preflight_never_calls_acl_or_writes_device_report(self):
        with tempfile.TemporaryDirectory() as work:
            result = self.run_case(work, [])
            self.assertEqual(result["exit"], 0)
            self.assertEqual(result["calls"], 0)
            self.assertEqual(result["kernels"], 0)
            self.assertFalse((Path(work) / "DEVICE_FAULT.json").exists())
            self.assertFalse((Path(work) / "sfa_fusion_device_results.json").exists())

    def test_run_without_explicit_device_refuses_before_acl(self):
        with tempfile.TemporaryDirectory() as work:
            result = self.run_case(work, ["--run"])
            self.assertEqual(result["exit"], 2)
            self.assertEqual(result["calls"], 0)
            self.assertEqual(result["kernels"], 0)
            self.assertFalse((Path(work) / "sfa_fusion_device_results.json").exists())

    def test_existing_fault_marker_refuses_before_acl_and_is_preserved(self):
        with tempfile.TemporaryDirectory() as work:
            marker = Path(work) / "DEVICE_FAULT.json"
            marker.write_text('{"previous":true}')
            result = self.run_case(work, ["--run", "--device", "0"])
            self.assertEqual(result["exit"], 2)
            self.assertEqual(result["calls"], 0)
            self.assertEqual(result["kernels"], 0)
            self.assertEqual(marker.read_text(), '{"previous":true}')
            self.assertFalse((Path(work) / "sfa_fusion_device_results.json").exists())

    def test_sync_507015_stops_download_cleanup_reset_and_second_kernel(self):
        with tempfile.TemporaryDirectory() as work:
            result = self.run_case(work, ["--run", "--device", "0"])
            self.assertEqual(result["exit"], 1)
            self.assertGreater(result["calls"], 0)
            self.assertEqual(result["syncs"], 1)
            self.assertEqual(result["kernels"], 1)
            for field in ("after_fault", "downloads", "frees", "destroys", "finalizes", "resets"):
                self.assertEqual(result[field], 0, field)
            marker = json.loads((Path(work) / "DEVICE_FAULT.json").read_text())
            self.assertEqual(marker["acl_error"], 507015)
            self.assertEqual(marker["operation"], "aclrtSynchronizeStream")
            self.assertTrue(marker["device_tests_stopped"])
            report = json.loads((Path(work) / "sfa_fusion_device_results.json").read_text())
            self.assertEqual(report["status"], "EXECUTION_ERROR")
            self.assertEqual(report["acl_error"], 507015)
            self.assertNotIn("timing", report)

    def test_successful_sync_with_noop_kernel_fails_nan_control_and_stops(self):
        with tempfile.TemporaryDirectory() as work:
            result = self.run_case(work, ["--run", "--device", "0"], sync_success=True)
            self.assertEqual(result["exit"], 1)
            self.assertEqual(result["syncs"], 1)
            self.assertEqual(result["kernels"], 1)
            self.assertEqual(result["downloads"], 1)
            self.assertEqual(result["resets"], 0)
            self.assertFalse((Path(work) / "DEVICE_FAULT.json").exists())
            report = json.loads((Path(work) / "sfa_fusion_device_results.json").read_text())
            self.assertEqual(report["status"], "CONTROL_FAILED")
            self.assertEqual(report["output_dtype"], "float32")
            self.assertFalse(report["c4_pipeline_validated"])
            self.assertFalse(report["native_twopass_compared"])
            self.assertFalse(report["real_hot_empty_dispatch_validated"])
            self.assertEqual(len(report["cases"]), 1)
            self.assertFalse(report["cases"][0]["passed"])
            self.assertGreater(report["cases"][0]["nonfinite_count"], 0)
            for field in ("timing", "performance", "pipeline_us", "speedup"):
                self.assertNotIn(field, report)


if __name__ == "__main__":
    unittest.main()
