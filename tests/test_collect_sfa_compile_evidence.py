import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("collect_sfa_compile_evidence", Path(__file__).resolve().parents[1] / "tools/collect_sfa_compile_evidence.py")
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class CompileEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.build = self.root / "build"
        self.sdk = self.root / "sdk"
        self.build.mkdir()
        self.sdk.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_raw_copy_context_no_mutation_or_command_execution(self):
        command = "'/path with spaces/bisheng' -D__NPU_ARCH__=3510 --npu-arch=dav-3510 -c source.cpp; touch NEVER"
        (self.build / "compile_commands.json").write_text(json.dumps([{"directory": str(self.build), "command": command, "file": "source.cpp"}]))
        (self.build / "CMakeCache.txt").write_text("CMAKE_ASC_COMPILER:FILEPATH=/sdk/asc\nCMAKE_ASC_FLAGS:STRING=-O2\n")
        cmake = self.build / "CMakeFiles/3.30/CMakeASCCompiler.cmake"
        cmake.parent.mkdir(parents=True)
        cmake.write_text('set(CMAKE_ASC_COMPILER "/sdk/asc")\n')
        for name in collector.HEADERS:
            path = self.sdk / "include" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"#if __NPU_ARCH__ == 3510\r\nnamespace AscendC { namespace MicroAPI {} }\r\n#endif\r\n")
        before = {str(p): p.read_bytes() for parent in (self.build, self.sdk) for p in parent.rglob("*") if p.is_file()}
        report = collector.collect(self.build, self.root / "output", self.sdk)
        self.assertTrue(report["complete"])
        self.assertEqual(report["compile_commands"][0]["first_command"], "/path with spaces/bisheng")
        self.assertEqual(report["cmake_compiler_flags"]["CMAKE_ASC_FLAGS"], "-O2")
        self.assertFalse((self.build / "NEVER").exists())
        for item in report["files"]:
            self.assertEqual(Path(item["copy"]).read_bytes(), before[item["source"]])
            self.assertEqual(item["sha256"], hashlib.sha256(before[item["source"]]).hexdigest())
            if item["kind"] == "sdk":
                self.assertEqual(item["context_matches"][0]["line"], 1)
        self.assertTrue(all(Path(path).read_bytes() == data for path, data in before.items()))

    def test_escape_missing_and_existing_output_fail_closed(self):
        outside = self.root / "outside.h"
        outside.write_text("DO NOT COPY")
        (self.sdk / "kernel_macros.h").symlink_to(outside)
        report = collector.collect(self.build, self.root / "output", self.sdk)
        self.assertFalse(report["complete"])
        self.assertEqual(report["files"], [])
        self.assertIn("kernel_macros.h", report["missing_headers"])
        self.assertTrue(any("Rejected symlink escape" in item for item in report["notes"]))
        with self.assertRaises(FileExistsError):
            collector.collect(self.build, self.root / "output", self.sdk)
        with self.assertRaises(ValueError):
            collector.collect(self.build, self.sdk / "output", self.sdk)
        with self.assertRaises(ValueError):
            collector.collect(Path("/"), self.root / "forbidden")

    def test_no_sdk_root_is_unknown_and_invalid_command_is_only_recorded(self):
        # Valid JSON containing an invalid shell-like quoting sequence: never execute it.
        (self.build / "compile_commands.json").write_text(json.dumps([{"command": "clang 'unterminated"}]))
        report = collector.collect(self.build, self.root / "output")
        self.assertIn("unknown", report["sdk_status"])
        self.assertIn("parse_error", report["compile_commands"][0])
        self.assertFalse(report["complete"])


if __name__ == "__main__":
    unittest.main()
