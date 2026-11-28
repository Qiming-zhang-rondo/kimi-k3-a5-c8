"""Attach a scoped Native producer to the existing exec/gather/store chain.

The original C4 calls and results are preserved. A separate BF16 producer is
invoked after the original store, before attention. No route is forced.
"""
from functools import wraps
import importlib

from .native_cp import NativeCaptureScope, get_native_scope


def install_native_hooks(impl, *, patch_module=None, base_class=None):
    if base_class is None:
        base_class = next((cls for cls in type(impl).__mro__ if
            cls.__name__ == "AscendSFAImpl" and cls.__module__ == "vllm_ascend.attention.sfa_v1"), None)
    if base_class is None:
        raise RuntimeError("Native recent128 requires the reviewed SFA base implementation")
    if type(impl) is not base_class and (type(impl).__name__ != "AscendSFADSACPImpl" or
            type(impl).__module__ != "vllm_ascend.attention.context_parallel.sfa_cp"):
        raise RuntimeError("Native recent128 currently adapts base SFA and plain DSACP; actual PCP/DCP reader overrides need separate routing")
    module = patch_module if patch_module is not None else importlib.import_module(
        "ascend_vllm.patch.worker.patch_sfa_v1_mxfp4")
    helper = getattr(module, "_mxfp4_custom_kv_rmsnorm_rope", None)
    code = getattr(helper, "__code__", None)
    if code is None or "capture_native_kv" not in code.co_names:
        raise RuntimeError("Install the actual MXFP4 pre-quant raw-KV tap before enabling the Native producer")
    if patch_module is None:
        # Parse/create the JIT handle now, without launching. A missing Native
        # writer dependency must fail before accepting inference requests.
        from .native_writer import _default_kernel_factory
        _default_kernel_factory()
    forward = getattr(base_class, "forward", None)
    if not callable(forward):
        raise RuntimeError("Reviewed Native SFA forward entry is unavailable")
    if not base_class.__dict__.get("_recent128_native_forward_hook", False):
        original_forward = forward

        @wraps(original_forward)
        def native_forward(self, *args, **kwargs):
            runtime = getattr(self, "_recent128_runtime", None)
            if runtime is None or getattr(runtime, "producer_kind", None) != "native":
                return original_forward(self, *args, **kwargs)
            metadata = kwargs.get("attn_metadata", args[3] if len(args) > 3 else None)
            if metadata is None:
                return original_forward(self, *args, **kwargs)
            with NativeCaptureScope(self, metadata) as scope:
                result = original_forward(self, *args, **kwargs)
                scope.require_capture()
                if scope.writes != 1:
                    raise RuntimeError("Native forward did not complete exactly one high-precision ring write")
                return result

        base_class.forward = native_forward
        base_class._recent128_native_forward_hook = True
    owner = next((cls for cls in type(impl).__mro__ if "_store_parallel_kv" in cls.__dict__), None)
    if owner is None:
        raise RuntimeError("Native recent128 requires the original parallel KV store entry")
    if not owner.__dict__.get("_recent128_native_store_hook", False):
        original_store = owner.__dict__["_store_parallel_kv"]

        @wraps(original_store)
        def native_store(self, *args, **kwargs):
            result = original_store(self, *args, **kwargs)
            runtime = getattr(self, "_recent128_runtime", None)
            if runtime is not None and getattr(runtime, "producer_kind", None) == "native":
                scope = get_native_scope(self)
                if scope is None:
                    raise RuntimeError("Native KV store ran outside its forward capture scope")
                if scope.writes == 0:
                    runtime.native_write(scope)
            return result

        owner._store_parallel_kv = native_store
        owner._recent128_native_store_hook = True
    impl._recent128_native_hooks_ready = True
    return dict(producer="native_bf16_candidate", raw_tap_verified=True,
        original_quantizer_and_store_preserved=True, cp_exchange="extra_BF16_collective_for_DSACP_TP_gt_1",
        device_validation="PENDING", performance_claim=False)
