import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "hotkv" / "inspect_kv_fusion_capability.py"
SPEC = importlib.util.spec_from_file_location("fusion_inspector", SCRIPT)
inspector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspector)


class TestSourceInventory(unittest.TestCase):
    def test_never_imports_or_executes_operator_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "executed"
            (root / "__init__.py").write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
                "raise RuntimeError('never import this package')\n"
                "def mixed_quant_sparse_flash_mla(q, *, quant_mode=4, ori_kv=None): pass\n")
            report = inspector.inspect_paths([root])
            self.assertFalse(marker.exists())
            self.assertFalse(report["operator_packages_imported"])
            self.assertFalse(report["device_operations_executed"])
            self.assertEqual(report["capability"], "NOT_VERIFIED")
            self.assertEqual(report["source_matches"][0]["definitions"][0]["parameters"],
                             ["q", "quant_mode", "ori_kv"])

    def test_versions_allowlisted_and_content_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "version.info"
            path.write_text("Version=9.1.0\ncommitid=f6e7b882\nprivate_unrelated=do_not_emit\n")
            record = inspector.inspect_paths([path])["versions"][0]
            self.assertEqual(record["fields"], {"Version": "9.1.0", "commitid": "f6e7b882"})
            self.assertEqual(len(record["sha256"]), 64)
            self.assertNotIn("do_not_emit", json.dumps(record))

    def test_layout_evidence_is_bounded_and_never_grants_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed_quant_sparse_flash_mla.h"
            path.write_text("\n".join(f"// quant_mode=4, 400/656 supported claim {i}" for i in range(100)))
            report = inspector.inspect_paths([path], max_snippets=6)
            self.assertEqual(report["capability"], "NOT_VERIFIED")
            record = report["source_matches"][0]
            self.assertEqual(len(record["snippets"]), 6)
            self.assertTrue(record["snippets_truncated"])

    def test_unrelated_quant_mode_and_symlink_are_not_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "moe.py").write_text("quant_mode=4\n")
            (root / "mixed_quant_sparse_flash_mla.py").symlink_to(root / "moe.py")
            report = inspector.inspect_paths([root])
            self.assertEqual(report["source_matches"], [])

    def test_limits_and_missing_roots_report_incompleteness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(3):
                (root / f"mixed_quant_sparse_flash_mla_{i}.py").write_text("quant_mode=4\n")
            report = inspector.inspect_paths([root], max_files=1)
            self.assertTrue(report["truncated"])
            self.assertEqual(len(report["source_matches"]), 1)
            self.assertTrue(inspector.inspect_paths([root / "missing"])["warnings"])
            with self.assertRaises(ValueError):
                inspector.inspect_paths([root], max_files=0)

    def test_cli_needs_only_standard_library_and_outputs_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed_quant_sparse_flash_mla.h"
            path.write_text("// quant_mode 1\n")
            proc = subprocess.run([sys.executable, "-S", str(SCRIPT), "--root", str(path)],
                                  check=True, capture_output=True, text=True)
            report = json.loads(proc.stdout)
            self.assertEqual(report["capability"], "NOT_VERIFIED")
            self.assertEqual(report["required_contract"]["q_total"], 576)


if __name__ == "__main__":
    unittest.main()
