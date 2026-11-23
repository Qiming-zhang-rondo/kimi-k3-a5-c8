import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_ROOT = ROOT if (ROOT / "run_kv_bf16_hot_ab.py").is_file() else ROOT.parent
sys.path.insert(0, str(LAUNCHER_ROOT))
import run_kv_bf16_hot_ab as launcher


class Bf16HotLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        names = ("NATIVE_TWOPASS_README.txt", "kv_c4_scale_backend.py", "kv_hot_backend.py",
                 "kv_window_backend.py", "kv_window_capture.py", "kv_window_common.py",
                 "test_kv_hot_fused_a5.py", "kv_native_twopass.py", "kv_native_merge.py",
                 "test_kv_native_twopass_a5.py", "kv_native_partition.py", "kv_native_store.py",
                 "inspect_kv_fusion_capability.py", "kv_native_hot_merge.py", "kv_native_hot_prepare.py", "kv_bf16_hot_ab.py",
                 "test_kv_bf16_hot_ab_a5.py", "BF16_HOT_AB_README.txt")
        self.payload = {n: (ROOT / "hotkv" / n).read_bytes() for n in names}
        self.hashes = {n: hashlib.sha256(v).hexdigest() for n,v in self.payload.items()}

    def urlopen(self, url, **kwargs):
        self.assertIn("/" + "a"*40 + "/kv-native-twopass/hotkv/", url)
        return io.BytesIO(self.payload[url.rsplit("/",1)[-1]])

    def test_published_manifest_matches_runtime_files(self):
        self.assertEqual(launcher.FILES, self.hashes)

    def test_verified_download_selftest_and_isolated_output(self):
        with mock.patch.object(launcher,"PAYLOAD_COMMIT","a"*40), \
             mock.patch.object(launcher,"FILES",self.hashes), \
             mock.patch.object(launcher.urllib.request,"urlopen",self.urlopen), \
             contextlib.redirect_stdout(io.StringIO()):
            result = launcher.main(["--work-dir",str(self.root),"--self-test",
                                    "--cloudops-root",str(self.root / "absent")])
        self.assertEqual(result,0)
        runs = list(self.root.glob("kv-bf16-hot-ab-*"))
        self.assertEqual(len(runs),1)
        report = json.loads((runs[0] / "launcher.json").read_text())
        self.assertEqual(report["status"],"COMPLETED")
        self.assertEqual(len(report["files"]),18)
        self.assertIn("no A5 execution",(runs[0] / "run.log").read_text())

    def test_corrupt_payload_never_executes(self):
        with mock.patch.object(launcher,"PAYLOAD_COMMIT","a"*40), \
             mock.patch.object(launcher,"FILES",self.hashes), \
             mock.patch.object(launcher.urllib.request,"urlopen",return_value=io.BytesIO(b"wrong")), \
             mock.patch.object(launcher.subprocess,"Popen") as process, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(launcher.main(["--work-dir",str(self.root)]),1)
        process.assert_not_called()

    def test_partial_run_does_not_overwrite_existing_payload(self):
        target = self.root / "NATIVE_TWOPASS_README.txt"
        target.write_bytes(b"keep")
        with mock.patch.object(launcher,"PAYLOAD_COMMIT","a"*40), \
             mock.patch.object(launcher,"FILES",self.hashes), \
             mock.patch.object(launcher.urllib.request,"urlopen",self.urlopen):
            with self.assertRaises(FileExistsError):
                launcher.download_payload(self.root)
        self.assertEqual(target.read_bytes(),b"keep")

    def test_no_output_escape(self):
        for option in (["--output",str(self.root)], ["--output="+str(self.root)]):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launcher.main(option)

    def test_inherits_environment_without_sourcing_old_vendor_package(self):
        original = {"PATH":str(self.root / "empty"), "https_proxy":"http://proxy.invalid:8000",
                    "ASCEND_CUSTOM_OPP_PATH":"/working/current/vendor", "LD_LIBRARY_PATH":"/working/current/lib"}
        with mock.patch.dict(launcher.os.environ,original,clear=True):
            env, info = launcher.test_environment(self.root / "run",self.root / "old-cloudops")
            self.assertEqual(dict(launcher.os.environ),original)
        for key,value in original.items():
            self.assertEqual(env[key],value)
        self.assertEqual(info["cache_source"],"isolated_run")
        self.assertEqual(Path(env["TORCH_EXTENSIONS_DIR"]),self.root / "run" / "torch_extensions")

    def test_keeps_requested_cache_and_uses_available_ninja(self):
        cloud = self.root / "cloud"
        (cloud / "bin").mkdir(parents=True)
        ninja = cloud / "bin" / "ninja"
        ninja.write_text("#!/bin/sh\nexit 0\n")
        ninja.chmod(0o755)
        with mock.patch.dict(launcher.os.environ,{"PATH":str(self.root / "empty")},clear=True):
            env,info = launcher.test_environment(self.root / "run",cloud,self.root / "cache")
        self.assertEqual(info["ninja_source"],"cloudops_bin")
        self.assertEqual(info["cache_source"],"launcher_argument")
        self.assertEqual(env["PATH"].split(launcher.os.pathsep)[0],str(ninja.parent.resolve()))


if __name__ == "__main__":
    unittest.main()
