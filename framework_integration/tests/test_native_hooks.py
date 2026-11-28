"""Actual Native tap/store interception preserves the original C4 sequence."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfa_recent128.native_cp import capture_native, get_native_scope
from sfa_recent128.native_hooks import install_native_hooks


def capture_native_kv(impl, latent, rope):
    return capture_native(impl,latent,rope)


def helper(impl,latent,rope):
    capture_native_kv(impl,latent,rope)


class NativeHookTests(unittest.TestCase):
    def setUp(self):
        self.events=[];self.latent=object();self.rope=object();self.output=object();self.stored=object()
        test=self
        class AscendSFAImpl:
            def forward(self,layer_name,hidden_states,kv_cache,attn_metadata):
                test.events.append("tap")
                helper(self,test.latent,test.rope)
                result=self._store_parallel_kv()
                test.assertIs(result,test.stored)
                test.events.append("attention")
                return test.output
            def _store_parallel_kv(self):
                test.events.append("original_store")
                return test.stored
        self.base=AscendSFAImpl
        self.impl=self.base()
        def write(scope):
            test.assertIs(scope.impl,self.impl)
            test.assertEqual(scope.require_capture(),(test.latent,test.rope))
            test.events.append("native_write")
            scope.writes+=1
        self.runtime=SimpleNamespace(producer_kind="native",native_write=write)
        self.impl._recent128_runtime=self.runtime
        self.patch=SimpleNamespace(_mxfp4_custom_kv_rmsnorm_rope=helper)

    def install(self):
        return install_native_hooks(self.impl,patch_module=self.patch,base_class=self.base)

    def test_preserves_original_return_and_store_before_one_sidewrite(self):
        self.install()
        actual=self.impl.forward("layer",object(),object(),object())
        self.assertIs(actual,self.output)
        self.assertEqual(self.events,["tap","original_store","native_write","attention"])
        self.assertIsNone(get_native_scope())

    def test_off_and_missing_metadata_leave_original_behavior(self):
        self.install();self.runtime.producer_kind="prolog"
        self.assertIs(self.impl.forward("layer",None,None,object()),self.output)
        self.assertEqual(self.events,["tap","original_store","attention"])

    def test_missing_actual_tap_is_not_silently_accepted(self):
        self.patch._mxfp4_custom_kv_rmsnorm_rope=lambda *args:None
        with self.assertRaisesRegex(RuntimeError,"actual MXFP4 pre-quant"):
            self.install()
        self.assertFalse(self.base.__dict__.get("_recent128_native_forward_hook",False))

    def test_store_outside_forward_and_nested_hook_are_explicit(self):
        self.install();self.install()
        with self.assertRaisesRegex(RuntimeError,"outside its forward"):
            self.impl._store_parallel_kv()
        self.events.clear()
        self.impl.forward("layer",None,None,object())
        self.assertEqual(self.events.count("native_write"),1)

    def test_dsacp_store_override_delegation_does_not_double_write(self):
        self.install()  # Base store may already be wrapped by another layer.
        test=self
        class AscendSFADSACPImpl(self.base):
            def _store_parallel_kv(self):return super()._store_parallel_kv()
        AscendSFADSACPImpl.__module__="vllm_ascend.attention.context_parallel.sfa_cp"
        self.impl=AscendSFADSACPImpl();self.impl._recent128_runtime=self.runtime
        self.install()
        self.assertIs(self.impl.forward("layer",None,None,object()),test.output)
        self.assertEqual(self.events.count("native_write"),1)


if __name__=="__main__":unittest.main()
