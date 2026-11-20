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
LAUNCHER_ROOT = ROOT if (ROOT / "run_kv_hot_fused.py").is_file() else ROOT.parent
sys.path.insert(0, str(LAUNCHER_ROOT))
import run_kv_hot_fused as launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        names = (
            "test_kv_hot_fused_a5.py", "kv_hot_backend.py", "kv_hot_fused.py",
            "kv_hot_state.py", "kv_c4_scale_backend.py", "kv_window_backend.py",
            "kv_window_common.py", "kv_window_capture.py", "README.txt",
        )
        self.payload = {name: (ROOT/"hotkv"/name).read_bytes() for name in names}
        self.hashes = {n: hashlib.sha256(v).hexdigest() for n, v in self.payload.items()}

    def open_payload(self, url, **kwargs):
        self.assertIn("/" + "a"*40 + "/kv-hot-fused/hotkv/", url)
        return io.BytesIO(self.payload[url.rsplit("/", 1)[-1]])

    def test_verified_download_cpu_subprocess_and_isolated_output(self):
        with mock.patch.object(launcher, "PAYLOAD_COMMIT", "a"*40), \
             mock.patch.object(launcher, "FILES", self.hashes), \
             mock.patch.object(launcher.urllib.request, "urlopen", self.open_payload), \
             contextlib.redirect_stdout(io.StringIO()):
            code = launcher.main(["--work-dir", str(self.root), "--self-test",
                                  "--cloudops-root", str(self.root/"not-installed")])
        self.assertEqual(code, 0)
        runs = list(self.root.glob("kv-hot-fused-*"))
        self.assertEqual(len(runs), 1)
        report = json.loads((runs[0]/"launcher.json").read_text())
        self.assertEqual(report["status"], "COMPLETED")
        self.assertTrue((runs[0]/"run.log").stat().st_size > 0)
        self.assertEqual(len(report["files"]), 9)

    def test_corruption_prevents_execution(self):
        with mock.patch.object(launcher, "PAYLOAD_COMMIT", "a"*40), \
             mock.patch.object(launcher, "FILES", self.hashes), \
             mock.patch.object(launcher.urllib.request, "urlopen", return_value=io.BytesIO(b"corrupt")), \
             mock.patch.object(launcher.subprocess, "Popen") as process, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(launcher.main(["--work-dir", str(self.root)]), 1)
            process.assert_not_called()

    def test_existing_file_not_overwritten(self):
        target = self.root/"test_kv_hot_fused_a5.py"
        target.write_bytes(b"keep")
        with mock.patch.object(launcher, "PAYLOAD_COMMIT", "a"*40), \
             mock.patch.object(launcher, "FILES", self.hashes), \
             mock.patch.object(launcher.urllib.request, "urlopen", self.open_payload):
            with self.assertRaises(FileExistsError):
                launcher.download_payload(self.root)
        self.assertEqual(target.read_bytes(), b"keep")

    def test_forwarded_output_cannot_escape_isolation(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.main(["--output", str(self.root/"elsewhere")])

    def test_environment_preserves_proxy_and_uses_explicit_cache(self):
        original = {"PATH": str(self.root/"empty"), "https_proxy": "http://proxy.invalid:8000"}
        with mock.patch.dict(launcher.os.environ, original, clear=True):
            env, info = launcher.test_environment(self.root, self.root/"missing", self.root/"cache")
            self.assertEqual(dict(launcher.os.environ), original)
        self.assertEqual(env["https_proxy"], original["https_proxy"])
        self.assertEqual(Path(info["torch_extensions_dir"]), (self.root/"cache").resolve())
        self.assertNotIn("proxy", str(info))


if __name__ == "__main__":
    unittest.main()
