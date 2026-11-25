"""Real hook control flow with framework/runtime fakes; never import Torch/NPU."""
import builtins
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Keep this CPU-only dependency cached outside each isolated fake-module scope.
# Otherwise patch.dict cleanup would unload NumPy after every test.
import sfa_recent128.contract  # noqa: E402,F401


class Recent128HookTests(unittest.TestCase):
    def setUp(self):
        self.events=[]
        self.created=[]
        self.runtime_imports=[]
        events,created=self.events,self.created

        class FrameworkImpl:
            def __init__(impl,additional_config=None,marker=None):
                events.append(("init",marker))
                impl.vllm_config=types.SimpleNamespace(additional_config=additional_config,
                    model_config=types.SimpleNamespace(enforce_eager=True),
                    scheduler_config=types.SimpleNamespace(max_num_seqs=1))
                impl.preprocess_type="PROLOG_V3"
                impl.tap_latent,impl.tap_rope=object(),object()
                impl.exec_error=None
                impl.exec_result,impl.read_result,impl.weight_result=object(),object(),object()

            def _fused_preprocess_type(impl,*args,**kwargs):
                events.append(("original_preprocess",args,kwargs))
                return "PROLOG_V3"

            def process_weights_after_loading(impl,*args,**kwargs):
                selected=impl._fused_preprocess_type()
                events.append(("weight_conversion",selected,args,kwargs))
                impl.preprocess_type="NATIVE" if selected is None else selected
                impl.weights_finished=True
                return impl.weight_result

            def exec_kv(impl,*args):
                events.append(("original_exec",args))
                self.hooks.capture_native_kv(impl,impl.tap_latent,impl.tap_rope)
                if impl.exec_error:
                    raise impl.exec_error
                return impl.exec_result

            def _execute_sparse_flash_attention_process(impl,*args,**kwargs):
                events.append(("original_read",args,kwargs))
                return impl.read_result

        class Runtime:
            def __init__(runtime,impl,config):
                events.append(("runtime_created",getattr(impl,"weights_finished",False),impl.preprocess_type))
                created.append(runtime)
                runtime.impl,runtime.config=impl,config
                runtime.captures,runtime.reads=[],[]
                runtime.capture_error=None
                runtime.read_result=object()

            def capture(runtime,latent,rope,metadata):
                runtime.captures.append((latent,rope,metadata))
                if runtime.capture_error:
                    raise runtime.capture_error

            def read(runtime,*args):
                runtime.reads.append(args)
                return runtime.read_result

        top=types.ModuleType("vllm_ascend");top.__path__=[]
        attention=types.ModuleType("vllm_ascend.attention");attention.__path__=[]
        sfa=types.ModuleType("vllm_ascend.attention.sfa_v1");sfa.AscendSFAImpl=FrameworkImpl
        runtime=types.ModuleType("sfa_recent128.runtime");runtime.Recent128Runtime=Runtime
        self.modules=patch.dict(sys.modules,{"vllm_ascend":top,"vllm_ascend.attention":attention,
            "vllm_ascend.attention.sfa_v1":sfa,"sfa_recent128.runtime":runtime})
        self.modules.start();self.addCleanup(self.modules.stop)
        original_import=builtins.__import__

        def guarded_import(name,*args,**kwargs):
            if name=="torch" or name.startswith(("torch.","torch_npu")):
                raise AssertionError("Hook tests must never import real Torch/NPU")
            if name=="runtime" or name=="sfa_recent128.runtime":
                self.runtime_imports.append(name)
            return original_import(name,*args,**kwargs)

        self.import_guard=patch("builtins.__import__",new=guarded_import)
        self.import_guard.start();self.addCleanup(self.import_guard.stop)
        spec=importlib.util.spec_from_file_location("sfa_recent128.hooks",ROOT/"sfa_recent128/hooks.py")
        self.hooks=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.hooks)
        self.impl_class=FrameworkImpl

    def construct(self,window=0,preprocess="preserve"):
        self.hooks.install()
        config=dict(window=window,preprocess=preprocess)
        if window:
            config["diagnostics_path"]="/tmp/framework-hook-fixture-no-runtime-files"
        return self.impl_class(additional_config={"sfa_recent128":config},marker="original_marker")

    def test_off_does_not_import_or_create_runtime_and_delegates_original_calls(self):
        impl=self.construct()
        self.assertIsNone(impl._recent128_runtime)
        self.assertIs(impl.process_weights_after_loading("weights",flag=True),impl.weight_result)
        self.assertEqual(self.created,[])
        self.assertEqual(self.runtime_imports,[])
        arguments=tuple(object() for _ in range(6))
        self.assertIs(impl.exec_kv(*arguments),impl.exec_result)
        read_args=tuple(object() for _ in range(7));table=object()
        self.assertIs(impl._execute_sparse_flash_attention_process(*read_args,block_table=table),impl.read_result)
        self.assertIn(("original_exec",arguments),self.events)
        self.assertIn(("original_read",read_args,{"block_table":table}),self.events)
        self.hooks.capture_native_kv(object(),object(),object())  # OFF/raw tap is a no-op.
        self.assertEqual(self.runtime_imports,[])
        self.assertEqual(impl.preprocess_type,"PROLOG_V3")

    def test_on_runtime_created_only_after_original_weight_conversion(self):
        impl=self.construct(128)
        self.assertEqual(self.created,[])
        self.assertIsNone(impl._recent128_runtime)
        result=impl.process_weights_after_loading("actual_weights",copy=True)
        self.assertIs(result,impl.weight_result)
        self.assertEqual(len(self.created),1)
        self.assertIs(impl._recent128_runtime,self.created[0])
        self.assertIs(self.created[0].impl,impl)
        self.assertTrue(self.created[0].config.enabled)
        self.assertIn(("runtime_created",True,"PROLOG_V3"),self.events)
        conversion=next(i for i,event in enumerate(self.events) if event[0]=="weight_conversion")
        creation=next(i for i,event in enumerate(self.events) if event[0]=="runtime_created")
        self.assertLess(conversion,creation)

    def test_on_raw_tap_uses_this_exec_metadata_and_always_clears_it(self):
        impl=self.construct(128);impl.process_weights_after_loading()
        runtime=impl._recent128_runtime
        prefix=tuple(object() for _ in range(5))
        first,second=object(),object()
        self.assertIs(impl.exec_kv(*prefix,first),impl.exec_result)
        self.assertIsNone(impl._recent128_write_metadata)
        self.assertIs(impl.exec_kv(*prefix,second),impl.exec_result)
        self.assertEqual(runtime.captures,[(impl.tap_latent,impl.tap_rope,first),
                                           (impl.tap_latent,impl.tap_rope,second)])
        with self.assertRaisesRegex(RuntimeError,"outside native exec_kv"):
            self.hooks.capture_native_kv(impl,object(),object())
        for location in ("exec","capture"):
            with self.subTest(location=location):
                impl.exec_error=RuntimeError("original_exec_failure") if location=="exec" else None
                runtime.capture_error=RuntimeError("raw_capture_failure") if location=="capture" else None
                with self.assertRaisesRegex(RuntimeError,"failure"):
                    impl.exec_kv(*prefix,object())
                self.assertIsNone(impl._recent128_write_metadata)

    def test_on_read_passes_exact_current_objects_to_runtime(self):
        impl=self.construct(128);impl.process_weights_after_loading()
        runtime=impl._recent128_runtime
        arguments=tuple(object() for _ in range(7));table=object()
        self.assertIs(impl._execute_sparse_flash_attention_process(*arguments,block_table=table),runtime.read_result)
        self.assertEqual(runtime.reads,[arguments+(table,)])
        self.assertFalse(any(event[0]=="original_read" for event in self.events))

    def test_explicit_native_accuracy_overrides_before_conversion_for_off_and_on(self):
        for window in (0,128):
            with self.subTest(window=window):
                impl=self.construct(window,"native_accuracy")
                before=len(self.events)
                impl.process_weights_after_loading()
                recent=self.events[before:]
                conversion=next(event for event in recent if event[0]=="weight_conversion")
                self.assertIsNone(conversion[1])
                self.assertFalse(any(event[0]=="original_preprocess" for event in recent))
                self.assertEqual(impl.preprocess_type,"NATIVE")

    def test_install_is_idempotent_and_preserve_keeps_original_selection(self):
        self.hooks.install()
        names=("__init__","process_weights_after_loading","exec_kv", "_execute_sparse_flash_attention_process",
               "_fused_preprocess_type")
        wrappers={name:getattr(self.impl_class,name) for name in names}
        self.hooks.install()
        self.assertEqual(wrappers,{name:getattr(self.impl_class,name) for name in names})
        impl=self.construct(0,"preserve");impl.process_weights_after_loading()
        self.assertEqual(sum(event[0]=="weight_conversion" for event in self.events),1)
        self.assertEqual(sum(event[0]=="original_preprocess" for event in self.events),1)
        self.assertEqual(impl.preprocess_type,"PROLOG_V3")

    def test_native_accuracy_rejects_non_eager_or_multi_request_before_weight_conversion(self):
        for window in (0,128):
            for eager,max_seqs in ((False,1),(True,2)):
                with self.subTest(window=window,enforce_eager=eager,max_num_seqs=max_seqs):
                    impl=self.construct(window,"native_accuracy")
                    impl.vllm_config.model_config.enforce_eager=eager
                    impl.vllm_config.scheduler_config.max_num_seqs=max_seqs
                    before=len(self.events)
                    with self.assertRaisesRegex(RuntimeError,"single-request eager experiment"):
                        impl.process_weights_after_loading()
                    recent=self.events[before:]
                    self.assertFalse(any(event[0] in ("weight_conversion","runtime_created") for event in recent))
                    self.assertIsNone(impl._recent128_runtime)


if __name__=="__main__": unittest.main()
