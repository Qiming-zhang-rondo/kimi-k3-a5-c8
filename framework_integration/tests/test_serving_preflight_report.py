"""Aggregate preflight is host-only and must not stop at the first failure."""
import builtins
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "serving_preflight.py"
SPEC = importlib.util.spec_from_file_location("serving_preflight_report", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ServingPreflightReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.lib = self.root / "public-source", self.root / "lib"
        self.lib.mkdir()
        for relative in MODULE.REQUIRED_SOURCE_FILES:
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("// synthetic public fixture\n")
        (self.lib / MODULE.LIBRARY_NAME).write_bytes(b"host-fixture-not-a-real-library")
        self.engine = dict(max_num_seqs=8, max_num_batched_tokens=32, enforce_eager=False,
            enable_prefix_caching=True, speculative_config={"num_speculative_tokens": 3},
            kv_transfer_config={"role": "consumer"}, prefill_context_parallel_size=1,
            decode_context_parallel_size=1, compilation_config={"cudagraph_capture_sizes": [8, 16, 32]},
            additional_config={"sfa_recent128": dict(window=128, execution="serving", preprocess="preserve",
                reader="single_pass", single_pass_kernel="batched_v4", max_query_chunk=4,
                source_root=str(self.source), library_dir=str(self.lib), diagnostics_path=str(self.root / "diagnostics"))})

    @staticmethod
    def by_name(report):
        return {check["name"]: check for check in report["checks"]}

    def test_default_never_imports_torch_vendor_or_changes_engine(self):
        before = json.dumps(self.engine, sort_keys=True)
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.startswith(("torch", "cloud_ops_turbo", "vllm", "ascend_vllm")):
                raise AssertionError("Default preflight attempted a framework/device import")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", guarded), patch.object(MODULE, "_probe_provider") as provider, \
                patch.object(MODULE, "_probe_library") as library:
            report = MODULE.build_report(self.engine, environ={})
        provider.assert_not_called()
        library.assert_not_called()
        self.assertEqual(json.dumps(self.engine, sort_keys=True), before)
        self.assertEqual(report["status"], "HOST_CONFIG_CHECKED_RUNTIME_PENDING")
        self.assertFalse(report["all_serving_requirements_validated"])
        self.assertEqual(report["device_launches"], 0)
        checks = self.by_name(report)
        for feature in ("Graph", "MTP", "APC", "PD"):
            self.assertEqual(checks["feature." + feature]["status"], "UNKNOWN")
        self.assertEqual(checks["capacity.ring"]["actual"]["ring_capacity"], 256)

    def test_multiple_independent_errors_and_missing_files_are_all_reported(self):
        config = self.engine["additional_config"]["sfa_recent128"]
        config.update(window=127, execution="accuracy_eager", reader="native", preprocess="native_accuracy",
                      single_pass_kernel="kv_split4", max_query_chunk=-2, invented=True)
        self.engine.update(max_num_seqs=False, max_num_batched_tokens=-1)
        (self.source / MODULE.REQUIRED_SOURCE_FILES[0]).unlink()
        (self.lib / MODULE.LIBRARY_NAME).unlink()
        report = MODULE.build_report(self.engine, environ={})
        checks = self.by_name(report)
        for name in ("config.window", "config.execution", "config.reader", "config.preprocess",
                     "config.single_pass_kernel", "config.keys", "capacity.max_query_chunk",
                     "capacity.max_num_seqs", "capacity.max_num_batched_tokens", "reader.library_file",
                     "source." + MODULE.REQUIRED_SOURCE_FILES[0]):
            self.assertEqual(checks[name]["status"], "BLOCKED", name)
        self.assertGreaterEqual(report["counts"]["blocked"], 11)
        self.assertIn("service.validation", checks)

    def test_host_provider_and_library_exceptions_do_not_hide_each_other(self):
        with patch.object(MODULE, "_probe_provider", side_effect=ImportError("provider fixture missing")), \
             patch.object(MODULE, "_probe_original_prolog_exports", side_effect=RuntimeError("original exports missing")), \
             patch.object(MODULE, "_probe_library", side_effect=OSError("reader fixture cannot load")), \
             patch.object(MODULE.shutil, "which", return_value=None):
            report = MODULE.build_report(self.engine, check_host=True, environ={})
        checks = self.by_name(report)
        self.assertIn("provider fixture missing", checks["host.provider_function_abi"]["detail"])
        self.assertIn("original exports missing", checks["host.original_prolog_exports"]["detail"])
        self.assertIn("reader fixture cannot load", checks["host.reader_abi"]["detail"])
        self.assertEqual(checks["host.cxx_compiler"]["status"], "BLOCKED")
        self.assertEqual(checks["host.ninja"]["status"], "BLOCKED")
        self.assertEqual(checks["producer.first_call"]["status"], "PENDING")

    def test_cli_writes_aggregate_reports_and_refuses_existing_output(self):
        engine = self.root / "engine.json"
        engine.write_text(json.dumps(self.engine))
        output = self.root / "report"
        self.assertEqual(MODULE.main(["--engine-json", str(engine), "--output", str(output)]), 0)
        report = json.loads((output / "preflight.json").read_text())
        self.assertGreater(report["counts"]["unknown"], 0)
        self.assertIn("[UNKNOWN] feature.PD", (output / "preflight.txt").read_text())
        original = (output / "preflight.json").read_bytes()
        with self.assertRaises(SystemExit):
            MODULE.main(["--engine-json", str(engine), "--output", str(output)])
        self.assertEqual((output / "preflight.json").read_bytes(), original)

    def test_explicit_dsa_cp_route_quant_mode_and_sdk_mismatch_are_visible_together(self):
        self.engine["additional_config"]["enable_dsa_cp"] = True
        self.engine["attention_config"] = {"mla_kv_quant_mode": "c8"}
        sdk = self.root / "recorded-sdk"
        sdk.mkdir()
        (self.lib / "CMakeCache.txt").write_text("SFA_SDK_ROOT:PATH=" + str(sdk) + "\n")
        report = MODULE.build_report(self.engine, environ={"ASCEND_HOME_PATH": str(self.root / "other-sdk")})
        checks = self.by_name(report)
        for name in ("route.enable_dsa_cp", "route.mla_kv_quant_mode.attention_config", "sdk.identity"):
            self.assertEqual(checks[name]["status"], "BLOCKED")
        self.assertTrue(self.engine["additional_config"]["enable_dsa_cp"])

    def test_integer_window_and_implemented_cp_restrictions_match_runtime(self):
        self.engine["additional_config"]["sfa_recent128"]["window"] = 128.0
        self.engine["prefill_context_parallel_size"] = 2
        self.engine["decode_context_parallel_size"] = 4
        checks = self.by_name(MODULE.build_report(self.engine, environ={}))
        self.assertEqual(checks["config.window"]["status"], "BLOCKED")
        self.assertEqual(checks["feature.prefill_context_parallel_size"]["status"], "BLOCKED")
        self.assertEqual(checks["feature.decode_context_parallel_size"]["status"], "BLOCKED")

    def test_new_provider_alone_does_not_satisfy_original_python_selector_exports(self):
        vendor=SimpleNamespace(cloud_mla_prolog_v3_recent128=lambda **kwargs:None)
        with patch.object(MODULE.importlib,"import_module",return_value=vendor):
            with self.assertRaisesRegex(RuntimeError,"cloud_mla_prolog_v3_functional"):
                MODULE._probe_original_prolog_exports()
            vendor.cloud_mla_prolog_v3=lambda **kwargs:None
            vendor.cloud_mla_prolog_v3_functional=lambda **kwargs:None
            self.assertIn("Both original",MODULE._probe_original_prolog_exports())


if __name__ == "__main__":
    unittest.main()
