"""Read existing worker dispatch state on failure, without selecting a route.

No framework import, cache clear, tensor read, device operation or setting
change. Python module source fingerprints stay in the worker diagnostics.
"""
import hashlib
from pathlib import Path
import sys


def _scalar(value):
    return value if type(value) in (str, int, float, bool, type(None)) else type(value).__name__


def _function_identity(function):
    value = getattr(function, "__func__", function)
    chain, seen = [], set()
    for _ in range(8):
        if value is None or id(value) in seen:
            break
        seen.add(id(value))
        code = getattr(value, "__code__", None)
        chain.append(dict(module=getattr(value, "__module__", None),
            name=getattr(value, "__qualname__", type(value).__name__),
            file=getattr(code, "co_filename", None), line=getattr(code, "co_firstlineno", None)))
        value = getattr(value, "__wrapped__", None)
    return chain


def collect_dispatch_state(impl, *, modules=None):
    modules = sys.modules if modules is None else modules
    vc = getattr(impl, "vllm_config", None)
    raw = getattr(vc, "additional_config", None)
    parallel = getattr(vc, "parallel_config", None)
    result = dict(raw_additional_enable_dsa_cp=_scalar(raw.get("enable_dsa_cp")) if isinstance(raw, dict) else None,
        tp_size=_scalar(getattr(impl, "tp_size", None)),
        pcp_size=_scalar(getattr(parallel, "prefill_context_parallel_size", None)),
        dcp_size=_scalar(getattr(parallel, "decode_context_parallel_size", None)),
        class_mro=[base.__module__ + "." + base.__qualname__ for base in type(impl).__mro__],
        bound_methods={name: _function_identity(getattr(impl, name, None)) for name in
            ("_fused_preprocess_type", "_get_fused_type_unsupported_reasons", "exec_kv",
             "_prepare_kv_for_parallel", "_store_parallel_kv", "_execute_sparse_flash_attention_process")},
        modules={}, probes={}, scope="existing_worker_metadata_only_no_route_changes_or_cache_clear")
    names = ("vllm_ascend.attention.sfa_v1", "vllm_ascend.attention.context_parallel.sfa_cp",
             "vllm_ascend.utils", "vllm_ascend.ascend_config",
             "ascend_vllm.patch.worker.patch_sfa_v1_mxfp4")
    for name in names:
        module = modules.get(name)
        filename = getattr(module, "__file__", None)
        identity = dict(loaded=module is not None, file=filename)
        if isinstance(filename, str):
            try:
                path = Path(filename)
                if path.is_file() and path.stat().st_size <= 2 * 1024 * 1024:
                    identity["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as error:
                identity["fingerprint_error"] = str(error)
        result["modules"][name] = identity
    cp = modules.get("vllm_ascend.attention.context_parallel.sfa_cp")
    flag = getattr(cp, "enable_dsa_cp", None)
    if not callable(flag):
        flag = getattr(modules.get("vllm_ascend.utils"), "enable_dsa_cp", None)
    if callable(flag):
        try:
            result["probes"]["effective_enable_dsa_cp"] = _scalar(flag())
            cache_info = getattr(flag, "cache_info", None)
            if callable(cache_info):
                info = cache_info()
                result["probes"]["flag_cache_info"] = dict(info._asdict())
        except Exception as error:
            result["probes"]["effective_flag_error"] = type(error).__name__ + ": " + str(error)
    config_module = modules.get("vllm_ascend.ascend_config")
    get_config = getattr(config_module, "get_ascend_config", None)
    if callable(get_config):
        try:
            result["probes"]["validated_ascend_enable_dsa_cp"] = _scalar(getattr(get_config(), "enable_dsa_cp", None))
        except Exception as error:
            result["probes"]["validated_config_error"] = type(error).__name__ + ": " + str(error)
    return result
