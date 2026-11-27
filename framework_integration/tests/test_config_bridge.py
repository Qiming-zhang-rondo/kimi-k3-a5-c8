"""Exercise the configuration bridge with a strict factory, without Torch/NPU."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "sfa_recent128/config_bridge.py"


class ConfigBridgeTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("recent128_config_bridge_test", SOURCE)
        self.bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bridge)
        self.seen = []
        self.result = object()
        self.failure = None

        def strict_factory(config, *args, **kwargs):
            additional = config.additional_config
            self.seen.append((additional, args, kwargs))
            if additional is not None:
                unknown = set(additional) - {"vendor", "dspark_config"}
                if unknown:
                    raise ValueError("Unexpected vendor config: " + ",".join(sorted(unknown)))
            if self.failure is not None:
                raise self.failure
            return self.result

        self.original = strict_factory
        va = types.ModuleType("vllm_ascend")
        va.__path__ = []
        factory_module = types.ModuleType("vllm_ascend.ascend_config")
        factory_module.init_ascend_config = self.original
        self.factory_module = factory_module
        self.modules = {"vllm_ascend": va, "vllm_ascend.ascend_config": factory_module}
        for name in ("vllm_ascend.platform", "vllm_ascend.worker.worker",
                     "ascend_vllm.patch.platform.fixture", "vllm.engine.fixture"):
            module = types.ModuleType(name)
            module.init_ascend_config = self.original
            module.other_alias = self.original
            # A module-level __getattr__ must never be invoked by alias scanning.
            module.__getattr__ = lambda name: (_ for _ in ()).throw(AssertionError(name))
            self.modules[name] = module
        unrelated = types.ModuleType("user_application")
        unrelated.init_ascend_config = self.original
        self.modules[unrelated.__name__] = unrelated
        deceptive = types.ModuleType("vllm_user_application")
        deceptive.init_ascend_config = self.original
        self.modules[deceptive.__name__] = deceptive
        self.module_patch = patch.dict(sys.modules, self.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_success_hides_only_owned_key_and_restores_original_object(self):
        nested = {"enabled": True}
        overlay = {"window": 128}
        additional = {"sfa_recent128": overlay, "vendor": nested, "dspark_config": {"mode": 1}}
        config = types.SimpleNamespace(additional_config=additional)
        self.bridge.install()
        factory = self.factory_module.init_ascend_config
        self.assertIs(factory(config, "argument", enabled=True), self.result)
        filtered, args, kwargs = self.seen[0]
        self.assertEqual(filtered, {"vendor": nested, "dspark_config": {"mode": 1}})
        self.assertIs(filtered["vendor"], nested)
        self.assertEqual((args, kwargs), (("argument",), {"enabled": True}))
        self.assertIs(config.additional_config, additional)
        self.assertIs(config.additional_config["sfa_recent128"], overlay)

    def test_failure_also_restores_original_object(self):
        additional = {"sfa_recent128": {"window": 128}, "vendor": {}}
        config = types.SimpleNamespace(additional_config=additional)
        self.failure = RuntimeError("factory failed")
        self.bridge.install()
        with self.assertRaisesRegex(RuntimeError, "factory failed"):
            self.factory_module.init_ascend_config(config)
        self.assertIs(config.additional_config, additional)
        self.assertNotIn("sfa_recent128", self.seen[0][0])

    def test_other_unknown_keys_remain_rejected(self):
        additional = {"sfa_recent128": {}, "misspelled_vendor_option": True}
        config = types.SimpleNamespace(additional_config=additional)
        self.bridge.install()
        with self.assertRaisesRegex(ValueError, "misspelled_vendor_option"):
            self.factory_module.init_ascend_config(config)
        self.assertIs(config.additional_config, additional)

    def test_no_owned_key_delegates_without_copying_or_assigning(self):
        for additional in (None, {}, {"vendor": {"flag": True}}):
            with self.subTest(additional=additional):
                class ReadOnlyConfig:
                    @property
                    def additional_config(self):
                        return additional

                self.bridge.install()
                self.assertIs(self.factory_module.init_ascend_config(ReadOnlyConfig()), self.result)
                self.assertIs(self.seen[-1][0], additional)

    def test_aliases_rebound_only_in_framework_modules_and_install_is_idempotent(self):
        installed = self.bridge.install()
        factory = self.factory_module.init_ascend_config
        self.assertEqual(installed["status"], "INSTALLED")
        again = self.bridge.install()
        self.assertEqual(again["status"], "ALREADY_INSTALLED")
        self.assertEqual(again["aliases_rebound"], 0)
        json.dumps(installed, allow_nan=False)
        self.assertIs(self.factory_module.init_ascend_config, factory)
        for name, module in self.modules.items():
            if name in ("user_application", "vllm_user_application"):
                self.assertIs(module.init_ascend_config, self.original)
            elif "init_ascend_config" in vars(module):
                self.assertIs(vars(module)["init_ascend_config"], factory)
                if "other_alias" in vars(module):
                    self.assertIs(vars(module)["other_alias"], factory)
        # A stale alias introduced after the first installation is repaired too.
        self.modules["vllm.engine.fixture"].late_alias = self.original
        self.assertEqual(self.bridge.install()["aliases_rebound"], 1)
        self.assertIs(self.modules["vllm.engine.fixture"].late_alias, factory)

    def test_missing_factory_is_an_explicit_error(self):
        self.factory_module.init_ascend_config = None
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            self.bridge.install()


if __name__ == "__main__":
    unittest.main()
