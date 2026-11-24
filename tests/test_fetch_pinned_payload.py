"""Exercise payload integrity/atomic installation without executing its code."""
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

DIRECT = Path(__file__).resolve().parents[1] / "direct_bridge"
sys.path.insert(0, str(DIRECT))
spec = importlib.util.spec_from_file_location("sfa_fetch_payload_test", DIRECT / "fetch_pinned_payload.py")
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)
import pinned_payload


class FetchPayloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sfa-download-host-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / "hotkv"
        self.body = b"# downloaded data, never executed\n"
        self.manifest = {"fixture.py": hashlib.sha256(self.body).hexdigest()}
        for target in (fetcher, pinned_payload):
            p = patch.object(target, "FILES", self.manifest)
            p.start()
            self.addCleanup(p.stop)

    def test_verified_bytes_installed_then_reused_without_network(self):
        with patch.object(fetcher, "urlopen", return_value=io.BytesIO(self.body)) as request:
            self.assertEqual(fetcher.fetch(self.output), self.output)
        self.assertEqual((self.output / "fixture.py").read_bytes(), self.body)
        self.assertIn(pinned_payload.PAYLOAD_COMMIT, request.call_args.args[0])
        with patch.object(fetcher, "urlopen", side_effect=AssertionError("must not refetch")):
            self.assertEqual(fetcher.fetch(self.output), self.output)

    def test_hash_failure_leaves_no_partial_payload_or_owned_staging(self):
        with patch.object(fetcher, "urlopen", return_value=io.BytesIO(b"tampered")):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                fetcher.fetch(self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_modified_payload_is_never_overwritten(self):
        self.output.mkdir()
        path = self.output / "fixture.py"
        path.write_bytes(b"user modified")
        with patch.object(fetcher, "urlopen", side_effect=AssertionError("must not overwrite")):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                fetcher.fetch(self.output)
        self.assertEqual(path.read_bytes(), b"user modified")


if __name__ == "__main__":
    unittest.main()
