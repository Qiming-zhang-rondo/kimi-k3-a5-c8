"""Initialize MXLLM global patches before building any vLLM config.

Importing this helper is stdlib-only. Framework imports happen only inside
``bootstrap_mxllm``; worker patches retain their normal plugin/worker timing.
"""
from collections.abc import Mapping
import importlib
import importlib.util
from pathlib import Path
import sys


class MxllmBootstrapError(RuntimeError):
    """The requested checkout did not establish the required framework contract."""


def _checked_origin(module, package_root):
    origin = getattr(module, "__file__", None)
    if not origin:
        origin = getattr(getattr(module, "__spec__", None), "origin", None) or getattr(module, "origin", None)
    if not origin:
        raise MxllmBootstrapError("ascend_vllm has no verifiable source origin")
    actual = Path(origin).resolve()
    if not actual.is_relative_to(package_root):
        raise MxllmBootstrapError(
            f"ascend_vllm source mismatch: requested {package_root}, imported {actual}; "
            "use a fresh Python process with the intended checkout"
        )
    return str(actual)


def bootstrap_mxllm(mxllm_root):
    """Apply the checkout's formal global bootstrap and verify its schema.

    No environment defaults or missing config fields are synthesized. A missing
    field/registry entry identifies an unresolved source/stack mismatch before
    constructing LLM or importing MXLLM worker patches directly.
    """
    root = Path(mxllm_root).expanduser().resolve()
    package_root = root / "ascend_vllm"
    if not (package_root / "__init__.py").is_file():
        raise MxllmBootstrapError(f"MXLLM checkout lacks ascend_vllm/__init__.py: {root}")
    package_root = package_root.resolve()
    loaded = sys.modules.get("ascend_vllm")
    if loaded is not None:
        _checked_origin(loaded, package_root)
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    try:
        spec = importlib.util.find_spec("ascend_vllm")
        if spec is None:
            raise MxllmBootstrapError(f"ascend_vllm cannot be found under {root}")
        _checked_origin(spec, package_root)
        mxllm = importlib.import_module("ascend_vllm")
        origin = _checked_origin(mxllm, package_root)
        ensure = getattr(mxllm, "_ensure_global_patch", None)
        if callable(ensure):
            entry = "ascend_vllm._ensure_global_patch"
            ensure()
        else:
            utils = importlib.import_module("ascend_vllm.utils")
            adapt = getattr(utils, "adapt_patch", None)
            if not callable(adapt):
                raise MxllmBootstrapError(
                    "MXLLM exposes neither _ensure_global_patch() nor "
                    "utils.adapt_patch(is_global_patch=True)"
                )
            entry = "ascend_vllm.utils.adapt_patch"
            adapt(is_global_patch=True)
        config = importlib.import_module("vllm.config")
        attention = getattr(config, "AttentionConfig", None)
        dataclass_fields = getattr(attention, "__dataclass_fields__", {})
        model_fields = getattr(attention, "model_fields", {})
        field = "mla_kv_quant_mode"
        if isinstance(dataclass_fields, Mapping) and field in dataclass_fields:
            declaration = "dataclass"
        elif isinstance(model_fields, Mapping) and field in model_fields:
            declaration = "pydantic"
        else:
            raise MxllmBootstrapError(
                "MXLLM global patch did not declare AttentionConfig.mla_kv_quant_mode; "
                "verify this checkout's platform patch and the installed vLLM revision"
            )
        envs = importlib.import_module("vllm_ascend.envs")
        registry = getattr(envs, "env_variables", None)
        name = "ASCEND_VLLM_KV_CONTIG"
        if not isinstance(registry, Mapping) or not callable(registry.get(name)):
            raise MxllmBootstrapError(
                "MXLLM global bootstrap left vllm_ascend.envs.ASCEND_VLLM_KV_CONTIG "
                "unregistered; verify the matching MXLLM/vllm-ascend sources. "
                "Setting an environment default cannot repair a missing registry entry"
            )
        return dict(status="MXLLM_BOOTSTRAPPED", mxllm_root=root_text,
                    ascend_vllm_origin=origin, global_patch_entry=entry,
                    attention_config_field=field, attention_config_declaration=declaration,
                    kv_contig_registry="vllm_ascend.envs.env_variables",
                    worker_patch_import="DEFERRED_TO_FRAMEWORK_PLUGIN")
    except MxllmBootstrapError:
        raise
    except Exception as error:
        raise MxllmBootstrapError(
            f"MXLLM global bootstrap failed before LLM construction ({type(error).__name__}: {error})"
        ) from error
