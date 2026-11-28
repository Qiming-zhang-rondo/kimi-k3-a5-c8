"""Serving/accuracy hook dispatch without importing Torch, workers or NPU libs."""
import builtins
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sfa_recent128.contract  # Keep the CPU contract's NumPy import stable.


class ServingHookTest(unittest.TestCase):
    def setup_hooks(self, *, prolog_available=True):
        events=[];imports=[]
        testcase=self
        class Framework:
            def __init__(impl,config):
                impl.vllm_config=SimpleNamespace(additional_config={"sfa_recent128":config},
                    model_config=SimpleNamespace(enforce_eager=False),
                    scheduler_config=SimpleNamespace(max_num_seqs=8))
                impl.preprocess_type="PROLOG_V3"
                impl.result=[object() for _ in range(5)]
                impl.weight_result=object();impl.exec_result=object();impl.read_result=object()
                impl.failure=None
            def _fused_preprocess_type(impl):
                events.append("original_selection");return "PROLOG_V3"
            def process_weights_after_loading(impl,*args,**kwargs):
                events.append(("weights",args,kwargs));impl.weights_done=True
                return impl.weight_result
            def exec_kv(impl,*args):
                events.append(("exec",args))
                testcase.hooks.capture_native_kv(impl,"latent","rope")
                return impl.exec_result
            def _execute_sparse_flash_attention_process(impl,*args,**kwargs):
                events.append(("read",args,kwargs));return impl.read_result
        if prolog_available:
            def original_prolog(impl,hidden,*args,**kwargs):
                events.append(("prolog",hidden,args,kwargs))
                if impl.failure:raise impl.failure
                return impl.result
            Framework._sfa_preprocess_prolog_v3=original_prolog

        class ServingRuntime:
            def __init__(runtime,impl,config):
                assert impl.weights_done
                assert impl._recent128_prolog_hook_available
                events.append("serving_created")
                runtime.impl=impl;runtime.config=config;runtime.read_result=object()
            @contextmanager
            def prolog_scope(runtime,metadata,rows):
                events.append(("scope_enter",metadata,rows))
                try:yield
                finally:events.append(("scope_exit",metadata,rows))
            def read(runtime,*args):
                events.append(("serving_read",args));return runtime.read_result
            def capture(runtime,*args):raise AssertionError("Serving must not use native capture")
        class AccuracyRuntime:
            def __init__(runtime,impl,config):
                events.append("accuracy_created");runtime.captures=[]
            def capture(runtime,*args):runtime.captures.append(args)
            def read(runtime,*args):return object()
        top=ModuleType("vllm_ascend");top.__path__=[]
        attention=ModuleType("vllm_ascend.attention");attention.__path__=[]
        sfa=ModuleType("vllm_ascend.attention.sfa_v1");sfa.AscendSFAImpl=Framework
        serving=ModuleType("sfa_recent128.serving_runtime");serving.ServingRuntime=ServingRuntime
        serving.preflight_serving=lambda config:events.append(("serving_preflight",config)) or {"status":"HOST_SYMBOLS_CHECKED"}
        runner=ModuleType("sfa_recent128.serving_runner");runner.install_runner_hooks=lambda:events.append("runner_installed")
        accuracy=ModuleType("sfa_recent128.runtime");accuracy.Recent128Runtime=AccuracyRuntime
        modules={"vllm_ascend":top,"vllm_ascend.attention":attention,"vllm_ascend.attention.sfa_v1":sfa,
            "sfa_recent128.serving_runtime":serving,"sfa_recent128.serving_runner":runner,"sfa_recent128.runtime":accuracy}
        active=patch.dict(sys.modules,modules);active.start();self.addCleanup(active.stop)
        original_import=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name=="torch" or name.startswith(("torch.","torch_npu")):
                raise AssertionError("Device framework imported by host hook tests")
            if name in ("serving_runtime","serving_runner","runtime"):imports.append(name)
            return original_import(name,*args,**kwargs)
        guard=patch("builtins.__import__",new=guarded);guard.start();self.addCleanup(guard.stop)
        spec=importlib.util.spec_from_file_location("sfa_recent128.hooks",ROOT/"sfa_recent128/hooks.py")
        self.hooks=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.hooks)
        self.hooks.parse_recent128_config=lambda additional:additional["sfa_recent128"]
        self.hooks.require_explicit_config=lambda additional:None
        self.hooks.install()
        return Framework,events,imports

    def test_serving_keeps_prolog_and_scopes_exact_inputs_and_output_after_weights(self):
        Framework,events,imports=self.setup_hooks()
        config=SimpleNamespace(enabled=True,execution="serving",preprocess="preserve")
        impl=Framework(config)
        self.assertEqual(imports,["serving_runtime"])
        self.assertEqual(events,[("serving_preflight",config)])
        self.assertEqual(impl._fused_preprocess_type(),"PROLOG_V3")
        self.assertIs(impl.process_weights_after_loading("weights",flag=True),impl.weight_result)
        self.assertLess(events.index("runner_installed"),events.index("serving_created"))
        self.assertLess(next(i for i,x in enumerate(events) if isinstance(x,tuple) and x[0]=="weights"),events.index("serving_created"))
        hidden=SimpleNamespace(shape=(32,7168));args=tuple(object() for _ in range(4));metadata=object()
        actual=impl._sfa_preprocess_prolog_v3(hidden,*args,attn_metadata=metadata)
        self.assertIs(actual,impl.result)
        self.assertIn(("scope_enter",metadata,32),events)
        self.assertIn(("prolog",hidden,args,{"attn_metadata":metadata}),events)
        self.assertIn(("scope_exit",metadata,32),events)
        read_args=tuple(object() for _ in range(7));table=object()
        self.assertIs(impl._execute_sparse_flash_attention_process(*read_args,block_table=table),impl._recent128_runtime.read_result)
        self.assertIn(("serving_read",read_args+(table,)),events)
        with self.assertRaisesRegex(RuntimeError,"native exec_kv fallback"):
            impl.exec_kv(*(object() for _ in range(6)))
        self.hooks.capture_native_kv(impl,object(),object())  # Serving native tap is a no-op.
        impl.failure=RuntimeError("producer failed")
        with self.assertRaisesRegex(RuntimeError,"producer failed"):
            impl._sfa_preprocess_prolog_v3(hidden,*args,attn_metadata=metadata)
        self.assertEqual(events[-1],("scope_exit",metadata,32))

    def test_off_and_accuracy_do_not_import_serving_or_change_original_prolog(self):
        Framework,events,imports=self.setup_hooks()
        for enabled in (False,True):
            impl=Framework(SimpleNamespace(enabled=enabled,execution="accuracy_eager",preprocess="preserve"))
            impl.process_weights_after_loading()
            hidden=SimpleNamespace(shape=(4,7168))
            self.assertIs(impl._sfa_preprocess_prolog_v3(hidden,attn_metadata="metadata"),impl.result)
            self.assertFalse(any(isinstance(x,tuple) and x[0]=="scope_enter" for x in events))
            impl.exec_kv(*(object() for _ in range(5)),"metadata")
            if enabled:self.assertEqual(impl._recent128_runtime.captures,[("latent","rope","metadata")])
        self.assertNotIn("serving_runtime",imports);self.assertNotIn("serving_runner",imports)
        self.assertNotIn("runner_installed",events)

    def test_missing_prolog_entry_keeps_legacy_fakes_installable_but_rejects_serving(self):
        Framework,events,imports=self.setup_hooks(prolog_available=False)
        off=Framework(SimpleNamespace(enabled=False,execution="accuracy_eager",preprocess="preserve"))
        off.process_weights_after_loading()
        self.assertFalse(off._recent128_prolog_hook_available)
        with self.assertRaisesRegex(RuntimeError,"requires the real _sfa_preprocess_prolog_v3.*before loading weights"):
            Framework(SimpleNamespace(enabled=True,execution="serving",preprocess="preserve"))
        self.assertNotIn("serving_runtime",imports);self.assertNotIn("serving_runner",imports)
        self.assertEqual(sum(isinstance(x,tuple) and x[0]=="weights" for x in events),1)


if __name__=="__main__":unittest.main()
