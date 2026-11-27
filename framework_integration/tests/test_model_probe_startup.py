"""Run the real startup control flow against host-only framework stand-ins."""
import builtins
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "run_model_probe.py"


class StartupFixtureFailure(RuntimeError):
    pass


class ModelProbeStartupTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("model_probe_startup_fixture", SOURCE)
        self.probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.probe)
        temporary = tempfile.TemporaryDirectory(prefix="recent128-startup-test-")
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.root = Path("/host-only-mxllm-checkout")
        self.events = []
        self.fail_at = None
        self.model = object()
        self.sampling_params = type("FakeSamplingParams", (), {})
        self.options = {"model": "/host-only-checkpoint", "additional_config": {
            "sfa_recent128": {"window": 128}, "vendor": {"flag": True}}}

        def at_stage(name, stage):
            # Persist each stage before entering its potentially failing call.
            snapshot = json.loads((self.output / "startup.json").read_text())
            self.assertEqual(snapshot["status"], "STARTING")
            self.assertEqual(snapshot["stage"], stage)
            self.events.append(name)
            if self.fail_at == name:
                self._trigger_failure(name)

        def bootstrap(root):
            at_stage("bootstrap", "MXLLM_GLOBAL_BOOTSTRAP")
            self.assertEqual(root, self.root)
            return {"status": "BOOTSTRAPPED", "root": str(root)}

        def bridge():
            at_stage("bridge", "RECENT128_CONFIG_BRIDGE")
            return {"status": "INSTALLED", "config_key": "sfa_recent128"}

        def llm(**options):
            at_stage("LLM", "LLM_CONSTRUCTION")
            self.assertEqual(options, self.options)
            self.assertIs(options["additional_config"], self.options["additional_config"])
            return self.model

        bootstrap_module = types.ModuleType("mxllm_bootstrap")
        bootstrap_module.bootstrap_mxllm = bootstrap
        overlay = types.ModuleType("sfa_recent128")
        overlay.__path__ = []
        bridge_module = types.ModuleType("sfa_recent128.config_bridge")
        bridge_module.install = bridge
        vllm = types.ModuleType("vllm")
        vllm.LLM, vllm.SamplingParams = llm, self.sampling_params
        modules = patch.dict(sys.modules, {
            "mxllm_bootstrap": bootstrap_module,
            "sfa_recent128": overlay,
            "sfa_recent128.config_bridge": bridge_module,
            "vllm": vllm,
        })
        modules.start()
        self.addCleanup(modules.stop)
        original_import = builtins.__import__

        def no_device_imports(name, *args, **kwargs):
            if name == "torch" or name.startswith(("torch.", "torch_npu")):
                raise AssertionError("Startup host tests must never import Torch/NPU")
            return original_import(name, *args, **kwargs)

        guard = patch("builtins.__import__", new=no_device_imports)
        guard.start()
        self.addCleanup(guard.stop)

    @staticmethod
    def _trigger_failure(stage):
        raise StartupFixtureFailure("original failure at " + stage)

    def initialize(self):
        return self.probe.initialize_model(self.options, mxllm_root=self.root,
            output=self.output, arm="single_pass", single_pass_kernel="kv_split4")

    def test_success_orders_bootstrap_bridge_then_llm_and_persists_initialized_json(self):
        model, sampling_params, report = self.initialize()
        self.assertEqual(self.events, ["bootstrap", "bridge", "LLM"])
        self.assertIs(model, self.model)
        self.assertIs(sampling_params, self.sampling_params)
        self.assertEqual(report, json.loads((self.output / "startup.json").read_text()))
        self.assertEqual(report["status"], "MODEL_INITIALIZED")
        self.assertEqual(report["stage"], "MODEL_INITIALIZED")
        self.assertEqual(report["arm"], "single_pass")
        self.assertEqual(report["single_pass_kernel"], "kv_split4")
        self.assertEqual(report["bootstrap"]["status"], "BOOTSTRAPPED")
        self.assertEqual(report["config_bridge"]["status"], "INSTALLED")
        self.assertEqual(report["model_accuracy"], "NOT_MEASURED")
        self.assertEqual(report["serving_performance"], "NOT_MEASURED")
        self.assertNotIn("traceback", report)
        self.assertFalse((self.output / "probe.json").exists())
        self.assertFalse((self.output / "startup_traceback.txt").exists())

    def assert_failure(self, name, stage, events, present, absent):
        self.fail_at = name
        with self.assertRaisesRegex(StartupFixtureFailure, "original failure at " + name):
            self.initialize()
        self.assertEqual(self.events, events)
        startup = json.loads((self.output / "startup.json").read_text())
        failure = json.loads((self.output / "probe.json").read_text())
        detail = (self.output / "startup_traceback.txt").read_text()
        self.assertEqual(startup, failure)
        self.assertEqual(startup["status"], "STARTUP_FAILED")
        self.assertEqual(startup["stage"], stage)
        self.assertEqual(startup["arm"], "single_pass")
        self.assertEqual(startup["single_pass_kernel"], "kv_split4")
        self.assertEqual(startup["model_accuracy"], "NOT_MEASURED")
        self.assertEqual(startup["serving_performance"], "NOT_MEASURED")
        self.assertEqual(startup["traceback"], detail)
        self.assertTrue(detail.startswith("Traceback (most recent call last):"))
        self.assertIn("initialize_model", detail)
        self.assertIn("_trigger_failure", detail)
        self.assertIn("StartupFixtureFailure: original failure at " + name, detail)
        for key in present:
            self.assertIn(key, startup)
        for key in absent:
            self.assertNotIn(key, startup)

    def test_bootstrap_failure_retains_stage_and_original_complete_traceback(self):
        self.assert_failure("bootstrap", "MXLLM_GLOBAL_BOOTSTRAP", ["bootstrap"],
                            present=(), absent=("bootstrap", "config_bridge"))

    def test_bridge_failure_retains_bootstrap_and_does_not_construct_llm(self):
        self.assert_failure("bridge", "RECENT128_CONFIG_BRIDGE", ["bootstrap", "bridge"],
                            present=("bootstrap",), absent=("config_bridge",))

    def test_llm_failure_retains_both_completed_startup_steps(self):
        self.assert_failure("LLM", "LLM_CONSTRUCTION", ["bootstrap", "bridge", "LLM"],
                            present=("bootstrap", "config_bridge"), absent=())


if __name__ == "__main__":
    unittest.main()
