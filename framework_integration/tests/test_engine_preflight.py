"""Invalid engine contracts fail before the vendor factory or weight loading."""
import builtins
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sfa_recent128.preflight import validate_engine_config
from sfa_recent128.contract import ContractError


def engine(serving=False):
    overlay=dict(window=128,diagnostics_path="/isolated/evidence")
    if serving:
        overlay.update(execution="serving",reader="single_pass",single_pass_kernel="batched_v4",
            source_root="/operator",library_dir="/isolated/library",max_query_chunk=32)
    return SimpleNamespace(additional_config={"sfa_recent128":overlay,"vendor":{"unchanged":True}},
        model_config=SimpleNamespace(enforce_eager=not serving),
        scheduler_config=SimpleNamespace(max_num_seqs=8 if serving else 1),
        cache_config=SimpleNamespace(enable_prefix_caching=serving),
        kv_transfer_config=object() if serving else None,
        speculative_config=object() if serving else None,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1,decode_context_parallel_size=1))


class EnginePreflightTest(unittest.TestCase):
    def test_accuracy_static_errors_are_actionable_and_do_not_change_engine(self):
        changes=(("scheduler_config","max_num_seqs",8,"max_num_seqs=1"),
            ("model_config","enforce_eager",False,"enforce_eager=True"),
            ("cache_config","enable_prefix_caching",True,"prefix caching"),
            (None,"kv_transfer_config",object(),"PD/KV transfer"),
            (None,"speculative_config",object(),"MTP/speculation"),
            ("parallel_config","decode_context_parallel_size",2,"decode_context_parallel_size=1"))
        for section,key,value,message in changes:
            with self.subTest(key=key):
                vc=engine();additional=vc.additional_config
                setattr(getattr(vc,section) if section else vc,key,value)
                with self.assertRaisesRegex(ContractError,message):validate_engine_config(vc)
                self.assertIs(vc.additional_config,additional)
                self.assertEqual(getattr(getattr(vc,section) if section else vc,key),value)

    def test_serving_accepts_production_features_statically_without_importing_device(self):
        vc=engine(True);additional=vc.additional_config
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in ("torch","torch_npu","cloud_ops_turbo"):
                raise AssertionError("Device/private import during parent preflight")
            return original(name,*args,**kwargs)
        with patch("builtins.__import__",new=guarded):report=validate_engine_config(vc)
        self.assertEqual(report["status"],"STATIC_CONFIG_VALIDATED")
        self.assertEqual(report["execution"],"serving")
        self.assertEqual(report["device_support"],"NOT_VERIFIED")
        self.assertEqual(report["preprocessing_type"],"NOT_INFERRED_BEFORE_WEIGHT_PREPARATION")
        self.assertIs(vc.additional_config,additional)
        self.assertEqual(vc.scheduler_config.max_num_seqs,8)

    def test_minimal_factory_objects_skip_but_explicit_off_native_override_is_checked(self):
        minimal=SimpleNamespace(additional_config={"sfa_recent128":{"window":128}})
        self.assertEqual(validate_engine_config(minimal)["status"],"SKIPPED_INCOMPLETE_ENGINE_CONFIG")
        self.assertEqual(validate_engine_config(SimpleNamespace())["status"],"SKIPPED_FEATURE_ABSENT")
        vc=engine()
        vc.additional_config["sfa_recent128"]={"window":0,"preprocess":"native_accuracy"}
        vc.scheduler_config.max_num_seqs=8
        with self.assertRaisesRegex(ContractError,"native_accuracy.*before loading weights"):
            validate_engine_config(vc)

    def test_bridge_rejects_before_calling_vendor_factory_and_preserves_original_config(self):
        calls=[]
        def original(vc):calls.append(vc.additional_config);return "vendor-result"
        va=ModuleType("vllm_ascend");va.__path__=[]
        module=ModuleType("vllm_ascend.ascend_config");module.init_ascend_config=original
        spec=importlib.util.spec_from_file_location("sfa_recent128.preflight_bridge_fixture",ROOT/"sfa_recent128/config_bridge.py")
        bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
        with patch.dict(sys.modules,{"vllm_ascend":va,"vllm_ascend.ascend_config":module}):
            bridge.install()
            invalid=engine();additional=invalid.additional_config;invalid.scheduler_config.max_num_seqs=8
            with self.assertRaisesRegex(ContractError,"before loading weights"):
                module.init_ascend_config(invalid)
            self.assertEqual(calls,[])
            self.assertIs(invalid.additional_config,additional)
            valid=engine(True);original_additional=valid.additional_config
            self.assertEqual(module.init_ascend_config(valid),"vendor-result")
            self.assertEqual(calls,[{"vendor":{"unchanged":True}}])
            self.assertIs(valid.additional_config,original_additional)


if __name__=="__main__":unittest.main()
