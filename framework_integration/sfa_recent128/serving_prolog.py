"""Route only MXLLM's selected MXFP4 Prolog call to the explicit ring producer.

Install during the existing worker plugin initialization. A layer's normal
``_sfa_preprocess_prolog_v3`` call is wrapped in ``prolog_scope`` by the caller.
No process-wide torch operator is replaced and no private kernel is copied.
Without an active binding the ORIGINAL MXLLM function owns all eager/functional
selection and cache-commit behavior unchanged.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import importlib

from .prolog_provider import (PROVIDER_ABI, PROVIDER_NAME, PrologProviderError,
                              RingBinding, call_prolog_with_ring)

PATCH_MODULE = "ascend_vllm.patch.worker.patch_sfa_v1_mxfp4"
_MARKER = "_sfa_recent128_serving_prolog_route"
_ORIGINAL = "_sfa_recent128_original_prolog_call"
_PROVIDER_STATE = "_sfa_recent128_provider_state"
_ACTIVE = ContextVar("sfa_recent128_active_prolog", default=None)
_REVIEWED_CLOUD_PROLOG_POSITIONAL_ARGS = (
    "token_x", "weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr",
    "rmsnorm_gamma_cq", "rmsnorm_gamma_ckv", "kv_cache", "kr_cache")


@dataclass
class PrologRouteScope:
    impl: object
    binding: object
    provider_calls: int = 0


@contextmanager
def prolog_scope(impl, binding):
    """Bind one layer invocation, restoring outer state even after exceptions.

    ``binding=None`` explicitly masks an enclosing ON scope. ContextVars keep
    independent threads/async tasks separate. The returned state counts actual
    successful provider invocations; creating a scope is not producer evidence.
    """
    if impl is None:
        raise PrologProviderError("Prolog scope requires its actual SFA implementation")
    if binding is not None and not isinstance(binding, RingBinding):
        raise PrologProviderError("Prolog scope binding must be RingBinding or None")
    state = PrologRouteScope(impl, binding)
    token = _ACTIVE.set(state)
    try:
        yield state
    finally:
        _ACTIVE.reset(token)


def _provider_capability(module, required):
    if module is None and required:
        try:
            module = importlib.import_module("cloud_ops_turbo")
        except ImportError as error:
            raise PrologProviderError(
                "cloud_ops_turbo is unavailable. Build/register the explicit private Prolog "
                "producer variant cloud_mla_prolog_v3_recent128 with recent128_abi_version=1 "
                "in the isolated worker environment first"
            ) from error
    function = getattr(module, PROVIDER_NAME, None) if module is not None else None
    marker = getattr(function, "recent128_abi_version", None)
    available = callable(function) and type(marker) is int and marker == PROVIDER_ABI
    if required and not available:
        raise PrologProviderError(
            "MXFP4 Prolog ring producer is not installed: require cloud_ops_turbo."
            + PROVIDER_NAME + ".recent128_abi_version=1. Build/register the explicit "
            "private Prolog producer variant in the isolated vendor first; the "
            "standalone ring-writer smoke is not this producer"
        )
    return module, available


def _source_contract(module):
    names = getattr(module, "_CLOUD_PROLOG_POSITIONAL_ARGS", None)
    if (not isinstance(names, (tuple, list))
            or tuple(names) != _REVIEWED_CLOUD_PROLOG_POSITIONAL_ARGS):
        raise PrologProviderError("MXLLM Prolog positional contract differs from the reviewed nine-argument adapter")
    normalized = {}
    for key, source, expected in (
            ("kv_cache_quant_mode", "_MXFP4_KV_CACHE_QUANT_MODE", 4),
            ("ckvkr_repo_mode", "_CKVKR_REPO_MODE", 1),
            ("quant_scale_repo_mode", "_QUANT_SCALE_REPO_MODE", 1),
            ("tile_size", "_MXFP4_PROLOG_TILE_SIZE", 128)):
        actual = getattr(module, source, None)
        if type(actual) is not int or actual != expected:
            raise PrologProviderError("MXLLM Prolog source constant mismatch: " + source)
        normalized[key] = actual
    if not callable(getattr(module, "_expected_kr_cache_shape", None)):
        raise PrologProviderError("MXLLM Prolog adapter lacks _expected_kr_cache_shape")
    return tuple(names), normalized


def _normalized_arguments(module, prolog_kwargs, names, mode_kwargs):
    if not isinstance(prolog_kwargs, Mapping):
        raise PrologProviderError("MXLLM Prolog call must provide its original keyword mapping")
    kwargs = dict(prolog_kwargs)
    missing = [name for name in names if name not in kwargs]
    if missing:
        raise PrologProviderError("MXLLM Prolog call arguments missing: " + ", ".join(missing))
    args = [kwargs.pop(name) for name in names]
    kv_index, kr_index = names.index("kv_cache"), names.index("kr_cache")
    kv, kr = args[kv_index], args[kr_index]
    if kr is not None and kr.numel() == 0:
        # This is the existing PR199 empty-cache normalization, not a new KV
        # allocation/copy. Correctly shaped static dummies are reused as-is.
        expected = module._expected_kr_cache_shape(kv, kwargs["rope_cos"])
        if tuple(kr.shape) != tuple(expected):
            args[kr_index] = kr.new_empty(expected)
    kwargs.pop("k_nope_clip_alpha", None)
    kwargs.update(mode_kwargs)
    return args, kwargs


def install_prolog_route(patch_module=None, provider_module=None, *, require_provider=True):
    """Install a targeted, idempotent route without running a device operation.

    Default startup validation requires the vendor's explicit ABI1 function.
    ``require_provider=False`` is only useful for OFF/import tests; an ON call
    still fails if the real provider is absent and never falls back.
    """
    module = patch_module if patch_module is not None else importlib.import_module(PATCH_MODULE)
    current = getattr(module, "_call_mxfp4_cloud_prolog", None)
    if not callable(current):
        raise PrologProviderError("MXLLM lacks its _call_mxfp4_cloud_prolog routing entry")
    names, mode_kwargs = _source_contract(module)
    installed = getattr(current, _MARKER, False) is True
    if installed:
        provider_state = getattr(current, _PROVIDER_STATE)
        bound_module = provider_state["module"]
        if provider_module is not None and bound_module is not None and provider_module is not bound_module:
            raise PrologProviderError("Prolog route already targets a different provider module; use a fresh worker")
        selected = provider_module if provider_module is not None else bound_module
        selected, available = _provider_capability(selected, require_provider)
        provider_state["module"] = selected
    else:
        selected, available = _provider_capability(provider_module, require_provider)
        provider_state = {"module": selected}
        original = current

        @wraps(original)
        def route(prolog_kwargs):
            active = _ACTIVE.get()
            if active is None or active.binding is None:
                # Do not normalize, call the new provider, or change the
                # original functional fallback/whole-cache commit on OFF.
                return original(prolog_kwargs)
            args, kwargs = _normalized_arguments(module, prolog_kwargs, names, mode_kwargs)
            result = call_prolog_with_ring(None, args, kwargs, ring=active.binding,
                                          module=provider_state["module"], positional_names=names)
            active.provider_calls += 1
            return result

        setattr(route, _MARKER, True)
        setattr(route, _ORIGINAL, original)
        setattr(route, _PROVIDER_STATE, provider_state)
        module._call_mxfp4_cloud_prolog = route
    return dict(status="ALREADY_INSTALLED" if installed else "INSTALLED", target=PATCH_MODULE + "._call_mxfp4_cloud_prolog",
        provider=PROVIDER_NAME, provider_abi=PROVIDER_ABI, provider_available=available,
        off_behavior="original MXLLM operator selection and functional cache commit unchanged",
        on_behavior="one explicit in-place ring producer; five original query outputs",
        scope="routing/capability only; graph, producer numerics and serving performance require device evidence")
