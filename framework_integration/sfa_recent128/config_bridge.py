"""Keep the overlay's configuration out of AscendConfig's strict schema.

Install during platform patching, before any engine/worker constructs its
AscendConfig. Like MXLLM's dspark bridge, the original additional_config object
remains on VllmConfig after construction so spawned workers retain our key.
Only this overlay's key is hidden while calling the original factory.
"""
from collections.abc import Mapping
from functools import wraps
import importlib
import sys
from types import ModuleType

from .launch_guard import require_explicit_config
from .preflight import validate_engine_config


CONFIG_KEY = "sfa_recent128"
_MARKER = "_sfa_recent128_config_bridge"
_ORIGINAL = "_sfa_recent128_original_factory"
_MODULE_ROOTS = ("vllm_ascend", "ascend_vllm", "vllm")


def _rebind_loaded_aliases(original, replacement):
    """Replace already-imported aliases without evaluating module properties."""
    replaced = 0
    for name, module in tuple(sys.modules.items()):
        if not isinstance(module, ModuleType) or not any(
                name == root or name.startswith(root + ".") for root in _MODULE_ROOTS):
            continue
        namespace = vars(module)
        for attribute, value in tuple(namespace.items()):
            if value is original:
                namespace[attribute] = replacement
                replaced += 1
    return replaced


def install():
    """Install once in this process, including aliases imported beforehand.

    This must run in both the parent and spawned worker platform initialization.
    It does not add a new field to VllmConfig or relax vendor validation: any
    other unrecognized additional_config key still reaches the strict factory.
    """
    module = importlib.import_module("vllm_ascend.ascend_config")
    original = vars(module).get("init_ascend_config")
    if not callable(original):
        raise RuntimeError("vllm_ascend.ascend_config.init_ascend_config is unavailable")
    if getattr(original, _MARKER, False):
        count = _rebind_loaded_aliases(getattr(original, _ORIGINAL), original)
        return dict(status="ALREADY_INSTALLED", config_key=CONFIG_KEY, aliases_rebound=count)

    @wraps(original)
    def init_ascend_config(vllm_config, *args, **kwargs):
        additional = getattr(vllm_config, "additional_config", None)
        require_explicit_config(additional)
        validate_engine_config(vllm_config)
        if not isinstance(additional, Mapping) or CONFIG_KEY not in additional:
            return original(vllm_config, *args, **kwargs)
        vllm_config.additional_config = {
            key: value for key, value in additional.items() if key != CONFIG_KEY
        }
        try:
            return original(vllm_config, *args, **kwargs)
        finally:
            vllm_config.additional_config = additional

    setattr(init_ascend_config, _MARKER, True)
    setattr(init_ascend_config, _ORIGINAL, original)
    module.init_ascend_config = init_ascend_config
    count = _rebind_loaded_aliases(original, init_ascend_config)
    return dict(status="INSTALLED", config_key=CONFIG_KEY, aliases_rebound=count)
