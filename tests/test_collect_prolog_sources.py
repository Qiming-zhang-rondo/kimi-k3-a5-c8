"""Read-only source handoff fixtures; never import the collected framework."""
import base64
import builtins
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "tools/collect_prolog_sources.py"
spec = importlib.util.spec_from_file_location("collect_prolog_sources_test", SOURCE)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class PrologSourceCollectionTests(unittest.TestCase):
    def test_exact_small_and_large_utf8_sources_restore_with_matching_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            root, output = folder / "source", folder / "evidence"
            root.mkdir()
            content = {
                "cloud_mla_prolog_v3.py": b"# original CRLF\r\nraise AssertionError('never execute source')\r\n",
                "mla_prolog_v3_kernel.cpp": "// 完整内核，保持原始字节\r\n".encode("utf-8") + b"".join(
                    b"// " + hashlib.sha256(str(i).encode()).hexdigest().encode() + b"\r\n"
                    for i in range(2048)),
            }
            for name, data in content.items():
                (root / name).write_bytes(data)
            generic = root / "mla_prolog_v3" / "op_kernel" / "arch35" / "common.h"
            generic.parent.mkdir(parents=True)
            generic.write_bytes(b"// exact generic dependency\nstruct KernelArgs {};\n")
            content[generic.name] = generic.read_bytes()
            for name, data in {"CMakeLists.txt": b"add_library(prolog kernel.cpp)\n",
                               "device.cmake": b"set(TARGET_ARCH ascend950)\n",
                               "build.sh": b"#!/bin/sh\ncmake -S . -B build\n"}.items():
                (generic.parent / name).write_bytes(data)
                content[name] = data
            (generic.parent / "unrelated.txt").write_bytes(b"do not gather every text file")
            (generic.parent / "unrelated.json").write_bytes(b"{}")
            report = collector.collect_sources([root], output)
            self.assertEqual(report["status"], "SOURCES_COLLECTED")
            self.assertTrue(report["source_found"])
            self.assertEqual(report["source_count"], 6)
            self.assertEqual(json.loads((output / "manifest.json").read_text()), report)
            for item in report["sources"]:
                expected = content[Path(item["source_path"]).name]
                storage = item["storage"]
                if storage["encoding"] == "raw_utf8":
                    restored = (output / storage["path"]).read_bytes()
                    self.assertLessEqual(len(restored), collector.PART_BYTES)
                else:
                    self.assertGreater(len(storage["parts"]), 1)
                    encoded = b""
                    for part in storage["parts"]:
                        data = (output / part["path"]).read_bytes()
                        self.assertLessEqual(len(data), collector.PART_BYTES)
                        self.assertEqual(hashlib.sha256(data).hexdigest(), part["sha256"])
                        encoded += data
                    restored = gzip.decompress(base64.b64decode(encoded))
                self.assertEqual(restored, expected)
                self.assertEqual(item["sha256"], hashlib.sha256(expected).hexdigest())
                self.assertEqual(item["bytes"], len(expected))
                self.assertEqual(Path(item["source_path"]).read_bytes(), expected)
            with self.assertRaises(FileExistsError):
                collector.collect_sources([root], output)

    def test_binary_only_is_not_source_and_symlinks_hidden_and_excluded_trees_are_not_read(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            root, output = folder / "source", folder / "evidence"
            root.mkdir()
            binary = b"\x7fELF\x00mla_prolog\xff"
            (root / "libmla_prolog_v3.so.1").write_bytes(binary)
            outside = folder / "outside_mla_prolog.cpp"
            outside.write_bytes(b"must not be copied")
            (root / "mla_prolog_link.cpp").symlink_to(outside)
            for name in (".git", ".hidden", "node_modules", "proc", "sys", "dev"):
                hidden = root / name
                hidden.mkdir()
                (hidden / "mla_prolog_secret.h").write_bytes(b"excluded")
            original_import = builtins.__import__
            def no_framework(name, *args, **kwargs):
                if name.split(".")[0] in ("torch", "torch_npu", "numpy", "vllm", "vllm_ascend"):
                    raise AssertionError("collector imported framework: " + name)
                return original_import(name, *args, **kwargs)
            with patch("builtins.__import__", new=no_framework):
                report = collector.collect_sources([root], output)
            self.assertFalse(report["source_found"])
            self.assertEqual(report["status"], "NO_SOURCES_FOUND")
            self.assertEqual(report["source_count"], 0)
            self.assertEqual(report["binary_count"], 1)
            self.assertEqual(report["binaries"][0]["sha256"], hashlib.sha256(binary).hexdigest())
            self.assertFalse(report["binaries"][0]["copied"])
            self.assertEqual(list(output.iterdir()), [output / "manifest.json"])
            self.assertEqual(report["skipped_reasons"]["symlink"], 1)

    def test_budgets_report_incomplete_and_output_inside_root_is_not_recursively_collected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mla_prolog_big.asc").write_bytes(b"x" * 100)
            report = collector.collect_sources([root], root / "evidence", max_source_bytes=32)
            self.assertFalse(report["scan_complete_within_scope"])
            self.assertFalse(report["source_found"])
            self.assertEqual(report["skipped_reasons"]["source_bytes_budget"], 1)
            self.assertIn("excluded_directory", report["skipped_reasons"])
            (root / "mla_prolog_other.hpp").write_bytes(b"schema")
            report = collector.collect_sources([root], root / "walk_budget", max_walk_files=1)
            self.assertFalse(report["scan_complete_within_scope"])
            self.assertIn("walk_file_budget", report["skipped_reasons"])
            self.assertLessEqual(report["source_count"], 1)


if __name__ == "__main__":
    unittest.main()
