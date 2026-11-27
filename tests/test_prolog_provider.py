"""Vendor handshake tests use metadata-only tensors that forbid host reads."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/"framework_integration/sfa_recent128/prolog_provider.py"
spec = importlib.util.spec_from_file_location("testable_prolog_provider", SOURCE)
provider = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = provider
spec.loader.exec_module(provider)


class TensorMetadata:
    def __init__(self, shape, dtype="bfloat16", device="npu:0", contiguous=True):
        self.shape = shape
        self.dtype = "torch." + dtype
        self.device = device
        self.contiguous = contiguous

    def is_contiguous(self): return self.contiguous
    def cpu(self): raise AssertionError("Host copy forbidden")
    def item(self): raise AssertionError("Device scalar extraction forbidden")
    def tolist(self): raise AssertionError("Host values forbidden")
    def numpy(self): raise AssertionError("Host copy forbidden")
    def __array__(self): raise AssertionError("Tensor conversion forbidden")


def binding(tokens=32, pools=8, capacity=256, bound=4):
    def tensor(shape, dtype="bfloat16"): return TensorMetadata(shape, dtype)
    return provider.RingBinding(
        recent_latent=tensor((pools,capacity,512)), recent_rope=tensor((pools,capacity,64)),
        recent_position_tags=tensor((pools,capacity),"int64"), recent_epoch_tags=tensor((pools,capacity),"int64"),
        token_ring_slots=tensor((tokens,),"int32"), token_positions=tensor((tokens,),"int64"),
        token_request_epochs=tensor((tokens,),"int64"), token_valid=tensor((tokens,),"bool"),
        max_query_per_request=bound)


class PrologProviderTest(unittest.TestCase):
    def test_off_is_exact_pass_through_without_provider_or_framework_import(self):
        values = (object(), object())
        original_kwargs = {"some_original_key": object()}
        returned = object()
        calls = []
        def original(*args, **kwargs):
            calls.append((args,kwargs))
            return returned
        class ForbiddenModule:
            def __getattr__(self, name): raise AssertionError("Provider lookup while OFF")
        with patch.object(provider.importlib, "import_module", side_effect=AssertionError("Import while OFF")):
            result = provider.call_prolog_with_ring(original,values,original_kwargs,module=ForbiddenModule())
        self.assertIs(result,returned)
        self.assertEqual(calls,[(values,original_kwargs)])

    def test_on_preserves_original_args_keywords_buffers_and_five_query_outputs(self):
        for shape in ((32,7168),(8,4,7168)):
            with self.subTest(shape=shape):
                args=(TensorMetadata(shape),)+(object(),)*8
                original_kwargs={"kv_cache_quant_mode":4,"query_norm_flag":True,"scale":object()}
                before=dict(original_kwargs)
                ring=binding()
                returned=[object() for _ in range(5)]
                captured={}
                def candidate(*actual_args,**actual_kwargs):
                    captured.update(args=actual_args,kwargs=actual_kwargs)
                    return returned
                candidate.recent128_abi_version=1
                module=SimpleNamespace(cloud_mla_prolog_v3_recent128=candidate)
                def forbidden_original(*args,**kwargs):raise AssertionError("ON fell back to original")
                result=provider.call_prolog_with_ring(forbidden_original,args,original_kwargs,ring,module)
                self.assertIs(result,returned)
                self.assertEqual(original_kwargs,before)
                self.assertTrue(all(a is b for a,b in zip(captured["args"],args)))
                self.assertEqual(set(captured["kwargs"]),set(before)|set(provider.PROVIDER_KEYWORDS))
                for name in provider.RING_KEYWORDS:
                    self.assertIs(captured["kwargs"][name],getattr(ring,name))
                self.assertEqual(captured["kwargs"]["recent_window"],128)

    def test_missing_or_wrong_function_abi_never_falls_back(self):
        args=(TensorMetadata((32,7168)),)+(object(),)*8
        ring=binding()
        for marker in (None,True,2):
            def candidate(*args,**kwargs):raise AssertionError("Bad ABI must not execute")
            if marker is not None:candidate.recent128_abi_version=marker
            module=SimpleNamespace(recent128_abi_version=1,cloud_mla_prolog_v3_recent128=candidate)
            with self.subTest(marker=marker),self.assertRaises(provider.PrologProviderError):
                provider.call_prolog_with_ring(None,args,{},ring,module)
        with self.assertRaisesRegex(provider.PrologProviderError,"lacks explicit new provider"):
            provider.call_prolog_with_ring(None,args,{},ring,SimpleNamespace())

    def test_geometry_dtype_device_padding_metadata_and_collisions_rejected(self):
        token=TensorMetadata((32,7168))
        args=(token,)+(object(),)*8
        ring=binding()
        invalid=[binding(capacity=128,bound=4),binding(capacity=129),binding(pools=0),binding(tokens=31),
            replace(ring,max_query_per_request=True),
            replace(ring,recent_rope=TensorMetadata((8,256,64),"float32")),
            replace(ring,recent_latent=TensorMetadata((8,256,512),device="npu:1")),
            replace(ring,token_valid=TensorMetadata((32,),"bool",contiguous=False)),
            replace(ring,recent_position_tags=TensorMetadata((8,256),"int32"))]
        for invalid_ring in invalid:
            with self.subTest(invalid_ring=invalid_ring),self.assertRaises(provider.PrologProviderError):
                provider.validate_ring_binding(token,invalid_ring)
        for key in provider.PROVIDER_KEYWORDS:
            with self.subTest(collision=key),self.assertRaisesRegex(provider.PrologProviderError,"collision"):
                provider.call_prolog_with_ring(None,args,{key:None},ring,SimpleNamespace())
        with self.assertRaisesRegex(provider.PrologProviderError,"at least nine"):
            provider.call_prolog_with_ring(None,args[:8],{},ring,SimpleNamespace())

    def test_return_shape_and_lazy_vendor_lookup(self):
        args=(TensorMetadata((32,7168)),)+(object(),)*8
        def candidate(*args,**kwargs):return (1,2,3,4)
        candidate.recent128_abi_version=1
        module=SimpleNamespace(cloud_mla_prolog_v3_recent128=candidate)
        with patch.object(provider.importlib,"import_module",return_value=module) as lookup:
            with self.assertRaisesRegex(provider.PrologProviderError,"exactly five"):
                provider.call_prolog_with_ring(None,args,{},binding())
        lookup.assert_called_once_with("cloud_ops_turbo")


if __name__ == "__main__": unittest.main()
