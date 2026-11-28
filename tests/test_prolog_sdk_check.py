"""Host-only SDK evidence: occurrences are never invented compiler values."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT=Path(__file__).resolve().parents[1]/"tools/check_prolog_sdk.py"
spec=importlib.util.spec_from_file_location("prolog_sdk_check",SCRIPT)
checker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class PrologSdkCheckTest(unittest.TestCase):
    def fixture(self,root,*,define_version=True):
        root=root.resolve()
        sdk=root/"sdk";source=root/"staged";include=sdk/"aarch64-linux/asc/include"
        include.mkdir(parents=True);source.mkdir()
        version=include/"asc_devkit_version.h"
        version.write_text("#define ASC_DEVKIT_VERSION_NUM (90100 + 21)\n" if define_version else "// absent on purpose\n")
        (include/"kernel_type.h").write_text("enum KernelType {\n ASCENDC_TPL_MIX_AIC_1_1 = 4,\n KERNEL_TYPE_AIV_ONLY,\n};\n")
        (source/"prolog.asc").write_text("#if ASC_DEVKIT_VERSION_NUM >= 90000\n// compile conditional from staged source\n#endif\nint type = ASCENDC_TPL_MIX_AIC_1_1;\n")
        compiler=sdk/"aarch64-linux/ccec_compiler/bin/bisheng"
        compiler.parent.mkdir(parents=True)
        compiler.write_text("#!/bin/sh\nif [ \"$#\" = 1 ] && [ \"$1\" = --version ]; then echo 'fixture SDK bisheng'; exit 0; fi\necho 'unexpected compiler execution' >&2; exit 97\n")
        compiler.chmod(0o755)
        return sdk,source,compiler,version

    def test_real_header_locations_and_expressions_do_not_claim_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk,source,compiler,version=self.fixture(Path(directory))
            report=checker.inspect_sdk(sdk,source)
            self.assertEqual(report["status"],"SDK_SYMBOL_EVIDENCE_CAPTURED")
            self.assertEqual(report["compatibility"],"NOT_VERIFIED")
            self.assertFalse(report["preprocessing_executed"])
            self.assertFalse(report["npu_executed"])
            self.assertEqual(report["compiler"]["status"],"READ")
            self.assertEqual(Path(report["compiler"]["path"]),compiler)
            macro=report["symbols"]["ASC_DEVKIT_VERSION_NUM"]
            self.assertEqual(macro["header_evidence"][0]["expression"],"(90100 + 21)")
            self.assertEqual(macro["header_evidence"][0]["file"],str(version))
            self.assertTrue(macro["source_uses"][0]["conditional"])
            enum=report["symbols"]["ASCENDC_TPL_MIX_AIC_1_1"]
            self.assertEqual(enum["header_evidence"][0]["expression"],"4")
            for symbol in (macro,enum):
                self.assertFalse(symbol["resolved"])
                self.assertIsNone(symbol["value"])
                self.assertEqual(symbol["status"],"FOUND_IN_HEADER_NOT_RESOLVED")

    def test_missing_version_is_explicit_and_macro_probe_does_not_include_kernel_operator(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sdk,source,_,_=self.fixture(root,define_version=False)
            output=root/"evidence"
            status=checker.main(["--sdk-root",str(sdk),"--source-root",str(source),"--output",str(output)])
            self.assertEqual(status,2)
            report=json.loads((output/"report.json").read_text())
            self.assertEqual(report["required_symbols_unlocated_in_headers"],["ASC_DEVKIT_VERSION_NUM"])
            self.assertFalse(report["symbols"]["ASC_DEVKIT_VERSION_NUM"]["resolved"])
            self.assertIn('#error "Matched SDK did not define ASC_DEVKIT_VERSION_NUM',(output/"probe.asc").read_text())
            self.assertNotIn('#include "kernel_operator.h"',(output/"macro_probe.cpp").read_text())
            command=json.loads((output/"preprocess_command.json").read_text())
            self.assertEqual(command["status"],"GENERATED_NOT_EXECUTED")
            self.assertEqual(Path(command["command"][-1]),(output/"macro_probe.cpp").resolve())
            self.assertNotIn("-DASC_DEVKIT_VERSION_NUM"," ".join(command["command"]))

    def test_matched_compiler_only_and_external_header_symlinks_are_not_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sdk,source,_,_=self.fixture(root)
            outside=root/"external.h";outside.write_text("#define KERNEL_TYPE_MIX_AIC_1_2 999\n")
            (sdk/"aarch64-linux/asc/include/external.h").symlink_to(outside)
            report=checker.inspect_sdk(sdk,source)
            self.assertEqual(report["status"],"INCOMPLETE_HEADER_OR_SOURCE_SCAN")
            self.assertEqual(report["symbols"]["KERNEL_TYPE_MIX_AIC_1_2"]["header_evidence"],[])
            self.assertEqual(report["sdk_scan"]["skipped"][0]["reason"],"external_file_symlink")
            external_compiler=root/"bisheng";external_compiler.write_text("#!/bin/sh\nexit 88\n")
            external_compiler.chmod(0o755)
            with self.assertRaisesRegex(ValueError,"within the explicitly selected SDK"):
                checker.inspect_sdk(sdk,source,external_compiler)

    def test_stdlib_cli_works_isolated_and_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sdk,source,_,version=self.fixture(root);output=root/"report"
            command=[sys.executable,"-I","-S",str(SCRIPT),"--sdk-root",str(sdk),"--source-root",str(source),"--output",str(output)]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            original=(output/"report.json").read_bytes()
            macro=(output/"macro_probe.cpp").read_text()
            self.assertIn('#include '+json.dumps(str(version)),macro)
            self.assertNotIn('#include "kernel_operator.h"',macro)
            again=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(again.returncode,0)
            self.assertIn("new local output directory",again.stderr)
            self.assertEqual((output/"report.json").read_bytes(),original)
            # The tool must remain usable on a diagnostic host without ML packages.
            source_text=SCRIPT.read_text()
            self.assertNotIn("import torch",source_text)
            self.assertNotIn("import numpy",source_text)


if __name__=="__main__":unittest.main()
