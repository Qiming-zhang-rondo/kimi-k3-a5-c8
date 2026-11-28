"""Targeted Prolog routing is tested with metadata-only, framework-free fakes."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"framework_integration"))
from sfa_recent128 import serving_prolog as routing
from sfa_recent128.prolog_provider import RingBinding, PrologProviderError


class Tensor:
    def __init__(self, shape, dtype="bfloat16"):
        self.shape=shape;self.dtype="torch."+dtype;self.device="npu:0";self.allocations=[]
    def is_contiguous(self):return True
    def numel(self):
        value=1
        for dimension in self.shape:value*=dimension
        return value
    def new_empty(self,shape):
        result=Tensor(tuple(shape));self.allocations.append(result);return result
    def cpu(self):raise AssertionError("host read forbidden")
    def item(self):raise AssertionError("host scalar forbidden")
    def tolist(self):raise AssertionError("host read forbidden")


def ring():
    return RingBinding(Tensor((2,256,512)),Tensor((2,256,64)),Tensor((2,256),"int64"),Tensor((2,256),"int64"),
        Tensor((4,),"int32"),Tensor((4,),"int64"),Tensor((4,),"int64"),Tensor((4,),"bool"),4)


def fixtures():
    original_calls=[];producer_calls=[];output=[object() for _ in range(5)]
    names=("token_x","weight_dq","weight_uq_qr","weight_uk","weight_dkv_kr","rmsnorm_gamma_cq","rmsnorm_gamma_ckv","kv_cache","kr_cache")
    def original(kwargs):
        original_calls.append(kwargs)
        # Represents the original functional path committing its own caches.
        kwargs["kv_cache"].committed=True
        return output
    def candidate(*args,**kwargs):
        producer_calls.append((args,kwargs));return output
    candidate.recent128_abi_version=1
    patch=SimpleNamespace(_call_mxfp4_cloud_prolog=original,_CLOUD_PROLOG_POSITIONAL_ARGS=names,
        _MXFP4_KV_CACHE_QUANT_MODE=4,_CKVKR_REPO_MODE=1,_QUANT_SCALE_REPO_MODE=1,_MXFP4_PROLOG_TILE_SIZE=128,
        _expected_kr_cache_shape=lambda kv,rope:(0,kv.shape[1],kv.shape[2],rope.shape[-1]))
    vendor=SimpleNamespace(cloud_mla_prolog_v3_recent128=candidate)
    kwargs={name:Tensor((4,7168)) if name=="token_x" else object() for name in names}
    kwargs.update(kv_cache=Tensor((32,128,1,400)),kr_cache=Tensor((0,128,1,64)),rope_cos=Tensor((4,64)),
        k_nope_clip_alpha=object(),kv_cache_quant_mode=3,ckvkr_repo_mode=0,quant_scale_repo_mode=0,tile_size=64)
    return patch,vendor,kwargs,original_calls,producer_calls,output


class ServingPrologTest(unittest.TestCase):
    def test_off_preserves_original_mapping_object_and_functional_commit(self):
        patch,vendor,kwargs,old,new,output=fixtures()
        routing.install_prolog_route(patch,vendor)
        self.assertIs(patch._call_mxfp4_cloud_prolog(kwargs),output)
        self.assertIs(old[0],kwargs)
        self.assertTrue(kwargs["kv_cache"].committed)
        self.assertFalse(new)
        self.assertIn("k_nope_clip_alpha",kwargs)

    def test_on_normalizes_once_preserving_cache_input_outputs_and_original_kwargs(self):
        patch,vendor,kwargs,old,new,output=fixtures()
        before=dict(kwargs);binding=ring()
        report=routing.install_prolog_route(patch,vendor)
        self.assertTrue(report["provider_available"])
        with routing.prolog_scope(object(),binding) as state:
            actual=patch._call_mxfp4_cloud_prolog(kwargs)
        self.assertIs(actual,output);self.assertEqual(state.provider_calls,1)
        self.assertFalse(old);self.assertEqual(len(new),1);self.assertEqual(kwargs,before)
        args,extra=new[0]
        self.assertIs(args[0],kwargs["token_x"]);self.assertIs(args[7],kwargs["kv_cache"]);self.assertIs(args[8],kwargs["kr_cache"])
        self.assertEqual((extra["kv_cache_quant_mode"],extra["ckvkr_repo_mode"],extra["quant_scale_repo_mode"],extra["tile_size"]),(4,1,1,128))
        self.assertNotIn("k_nope_clip_alpha",extra)
        self.assertIs(extra["recent_latent"],binding.recent_latent)
        self.assertEqual(extra["recent_window"],128)
        self.assertFalse(kwargs["kr_cache"].allocations)

    def test_wrong_empty_dummy_shape_only_uses_original_pr199_normalization(self):
        patch,vendor,kwargs,old,new,output=fixtures()
        dummy=kwargs["kr_cache"]=Tensor((0,))
        routing.install_prolog_route(patch,vendor)
        with routing.prolog_scope(object(),ring()):patch._call_mxfp4_cloud_prolog(kwargs)
        self.assertEqual(len(dummy.allocations),1)
        self.assertEqual(new[0][0][8].shape,(0,128,1,64))
        self.assertIs(kwargs["kr_cache"],dummy)

    def test_nested_off_exception_restoration_and_context_isolation(self):
        patch,vendor,kwargs,old,new,output=fixtures();routing.install_prolog_route(patch,vendor)
        with routing.prolog_scope(object(),ring()) as outer:
            with routing.prolog_scope(object(),None):patch._call_mxfp4_cloud_prolog(kwargs)
            try:
                with routing.prolog_scope(object(),ring()):raise ValueError("test")
            except ValueError:pass
            patch._call_mxfp4_cloud_prolog(kwargs)
        self.assertEqual(outer.provider_calls,1)
        self.assertEqual(len(old),1);self.assertEqual(len(new),1)
        patch._call_mxfp4_cloud_prolog(kwargs)
        self.assertEqual(len(old),2)
        # A fresh Context has no inherited active request/layer.
        from contextvars import Context
        Context().run(patch._call_mxfp4_cloud_prolog,kwargs)
        self.assertEqual(len(old),3)

    def test_startup_missing_provider_is_actionable_and_install_is_idempotent(self):
        patch,vendor,kwargs,old,new,output=fixtures()
        with self.assertRaisesRegex(PrologProviderError,"Build/register.*private Prolog producer"):
            routing.install_prolog_route(patch,SimpleNamespace())
        self.assertFalse(getattr(patch._call_mxfp4_cloud_prolog,routing._MARKER,False))
        first=routing.install_prolog_route(patch,vendor)
        wrapped=patch._call_mxfp4_cloud_prolog
        second=routing.install_prolog_route(patch,vendor)
        self.assertEqual(first["status"],"INSTALLED");self.assertEqual(second["status"],"ALREADY_INSTALLED")
        self.assertIs(wrapped,patch._call_mxfp4_cloud_prolog)
        with routing.prolog_scope(object(),ring()):patch._call_mxfp4_cloud_prolog(kwargs)
        self.assertEqual(len(new),1)
        self.assertFalse(old)


if __name__=="__main__":unittest.main()
