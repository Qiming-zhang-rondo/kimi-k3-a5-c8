import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


BUNDLE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("probe_original_sfa", BUNDLE / "tools/probe_original_sfa.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def fake_sdk(compiler, target="Ascend950"):
    modules = {}

    def module(name, **attrs):
        result = types.ModuleType(name)
        result.__dict__.update(attrs)
        modules[name] = result
        return result

    module("asc_op_compile_base")
    module("asc_op_compile_base.common")
    platform = module("asc_op_compile_base.common.platform",
                      get_soc_spec=lambda key: target if key == "SOC_VERSION" else "ascend950")
    constants = ["REQUIRED_INPUT", "OPTION_INPUT", "REQUIRED_OUTPUT", "OPTION_ATTR_FLOAT",
                 "OPTION_ATTR_INT", "OPTION_ATTR_STR", "OPTION_ATTR_BOOL", "KERNEL_NAME"]
    para = types.SimpleNamespace(**{name: name for name in constants})
    para.check_op_params = lambda *args, **kwargs: lambda function: function
    module("asc_op_compile_base.common.utils", para_check=para)
    module("asc_op_compile_base.common.register", register_operator=lambda *a, **k: lambda function: function)
    module("asc_op_compile_base.common.buildcfg", get_default_build_config=lambda: {},
           get_current_build_config=lambda key: 0)
    module("asc_op_compile_base.asc_op_compiler", compile_op=compiler,
           replay_op=lambda *a: None, check_op_cap=lambda *a: b"{}", generalize_op_params=lambda *a: None,
           get_code_channel=lambda *a: 0, OpInfo=lambda **kwargs: types.SimpleNamespace(**kwargs))
    return modules, platform


class OriginalSfaProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sfa probe ")
        self.root = Path(self.temp.name).resolve()
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def compile_ok(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        output = Path("kernel_meta")
        output.mkdir()
        (output / "sfa_original_probe.o").write_bytes(b"test compiler object")
        (output / "sfa_original_probe.json").write_text('{"kernelName":"sfa_original_probe"}')

    def run_compile(self, compiler=None, target="Ascend950", **kwargs):
        modules, platform = fake_sdk(compiler or self.compile_ok, target)
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            return probe.probe(self.root, BUNDLE, compile_requested=True, **kwargs)

    def test_prepare_preserves_uploaded_bytes_and_imports_no_sdk(self):
        def forbidden(path):
            self.fail("prepare-only imported SDK")
        report = probe.probe(self.root, BUNDLE, adapter_loader=forbidden)
        self.assertEqual(report["status"], "PREPARED")
        stage = Path(report["stage_dir"])
        self.assertEqual(report["compile_calls"], [])
        for item in report["sources"]:
            self.assertEqual(Path(item["source"]).read_bytes(), Path(item["path"]).read_bytes())
        self.assertTrue((stage / "ascendc/sparse_flash_attention/arch35/common/op_kernel/memcopy/gm_layout.h").is_file())
        self.assertTrue((stage / "ascendc/common/op_kernel/math_util.h").is_file())
        self.assertFalse(any(stage.rglob("kernel_tiling.h")))
        self.assertEqual(report["compiler_artifacts"], [])

    def test_real_uploaded_adapter_passes_exact_source_options_and_descriptors(self):
        before = Path.cwd()
        report = self.run_compile()
        self.assertEqual(report["status"], "PASS", report.get("error"))
        self.assertEqual(Path.cwd(), before)
        self.assertEqual(len(self.calls), 1)
        args, kwargs = self.calls[0]
        evidence = report["compile_calls"][0]
        self.assertEqual(str(Path(args[0]).resolve()), report["expected_source"])
        self.assertEqual(args[1], "sparse_flash_attention")
        self.assertEqual(args[3], evidence["options"])
        self.assertEqual(args[5:], ("{}",))
        self.assertEqual(kwargs, {})
        self.assertEqual(args[2].inputs[0]["shape"], [4, 64, 512])
        self.assertEqual(args[2].inputs[3]["shape"], [4, 1, 128])
        self.assertEqual(args[2].outputs[1]["shape"], [1, 4, 64])
        self.assertEqual(len(args[2].inputs), 9)
        self.assertIn("-DORIG_DTYPE_QUERY=DT_BF16", args[3])
        self.assertIn("-I" + str(Path(report["stage_dir"]) / "dynamic/../ascendc/common"), args[3])
        attrs = {x["name"]: x["value"] for x in args[2].attrs}
        self.assertEqual((attrs["layout_query"], attrs["layout_kv"], attrs["sparse_mode"]), ("TND", "PA_BSND", 0))
        self.assertEqual(attrs["scale_value"], 1 / 24)
        self.assertTrue(any(x["path"].endswith(".o") for x in report["compiler_artifacts"]))
        self.assertEqual(json.loads(Path(report["report"]).read_text()), report)

    def test_wrong_source_refused_before_compiler(self):
        def wrong_loader(path):
            adapter = probe.load_adapter(path)
            adapter.get_kernel_source = lambda *args: str(BUNDLE / "original/sparse_flash_attention.cpp")
            return adapter
        report = self.run_compile(adapter_loader=wrong_loader)
        self.assertEqual(report["status"], "ERROR")
        self.assertIn("Refusing non-staged", report["error"])
        self.assertEqual(self.calls, [])

    def test_no_object_or_prebuild_return_does_not_pass(self):
        def json_only(*args):
            Path("compiler.json").write_text("{}")
        report = self.run_compile(json_only)
        self.assertEqual(report["status"], "ERROR")
        self.assertIn("nonempty .o", report["error"])
        def prebuild_loader(path):
            adapter = probe.load_adapter(path)
            adapter.get_current_build_config = lambda key: key == "enable_op_prebuild"
            return adapter
        skipped = self.run_compile(adapter_loader=prebuild_loader)
        self.assertEqual(skipped["status"], "ERROR")
        self.assertIn("without invoking compile_op", skipped["error"])

    def test_partial_outputs_and_generated_tiling_survive_compiler_error(self):
        def failure(*args):
            output = Path("kernel_tiling")
            output.mkdir()
            (output / "kernel_tiling.h").write_text("struct SparseFlashAttentionTilingDataMla { int field; };\n")
            Path("partial.json").write_text("{}")
            raise RuntimeError("first compiler failure\nmore diagnostics")
        report = self.run_compile(failure)
        self.assertEqual(report["status"], "ERROR")
        self.assertEqual(report["first_error"], "RuntimeError: first compiler failure")
        self.assertIn("more diagnostics", report["traceback"])
        self.assertEqual(len(report["tiling_headers"]), 1)
        self.assertTrue(report["tiling_headers"][0]["generated_in_probe"])
        self.assertTrue(report["tiling_headers"][0]["has_type_definition_candidate"])
        self.assertTrue(any(x["path"].endswith("partial.json") for x in report["compiler_artifacts"]))

    def test_sdk_import_failure_and_wrong_target_are_errors(self):
        def missing(path):
            raise ModuleNotFoundError("SDK unavailable")
        report = probe.probe(self.root, BUNDLE, compile_requested=True, adapter_loader=missing)
        self.assertEqual(report["status"], "ERROR")
        self.assertIn("SDK unavailable", report["error"])
        report = self.run_compile(target="Ascend910")
        self.assertEqual(report["status"], "ERROR")
        self.assertEqual(self.calls, [])

    def test_soc_setter_requires_real_callable_and_signature(self):
        modules, platform = fake_sdk(self.compile_ok)
        set_calls = []
        def setter(soc_version):
            set_calls.append(soc_version)
        platform.set_current_compile_soc_info = setter
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            report = probe.probe(self.root, BUNDLE, True, "Ascend950")
        self.assertEqual(report["status"], "PASS", report.get("error"))
        self.assertEqual(set_calls, ["Ascend950"])
        self.assertEqual(report["soc"]["setter"]["signature"], "(soc_version)")
        modules, _ = fake_sdk(self.compile_ok)
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            missing = probe.probe(self.root, BUNDLE, True, "Ascend950")
        self.assertEqual(missing["status"], "ERROR")
        self.assertIn("No verified SDK", missing["error"])

    def test_repeated_probe_is_fresh_and_never_overwrites_old_evidence(self):
        first = probe.probe(self.root, BUNDLE)
        original = Path(first["report"]).read_bytes()
        second = probe.probe(self.root, BUNDLE)
        self.assertNotEqual(first["stage_dir"], second["stage_dir"])
        self.assertEqual(Path(first["report"]).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
