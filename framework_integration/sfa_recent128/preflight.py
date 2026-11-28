"""Reject static recent128 engine errors before vendor config/model loading.

Only configuration objects are inspected. No Torch/private operator import,
weight access, device initialization, or preprocessing-type guess is made.
"""
from collections.abc import Mapping

from .contract import ContractError, parse_recent128_config

_MISSING = object()


def validate_engine_config(vllm_config):
    """Validate available engine settings without changing any option.

    Minimal factory/plugin probes may carry only additional_config; leave those
    to the normal layer/config validation. Real engine configs have both model
    and scheduler sections, so their errors are rejected before weight loading.
    Production serving acceptance is not evidence of provider/device support.
    """
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, Mapping) or "sfa_recent128" not in additional:
        return dict(status="SKIPPED_FEATURE_ABSENT", scope="static_config_only")
    model = getattr(vllm_config, "model_config", None)
    scheduler = getattr(vllm_config, "scheduler_config", None)
    if model is None or scheduler is None:
        return dict(status="SKIPPED_INCOMPLETE_ENGINE_CONFIG", scope="static_config_only")
    config = parse_recent128_config(additional)
    checks = []
    selected_native = config.preprocess == "native_accuracy"
    accuracy_enabled = config.enabled and config.execution == "accuracy_eager"
    if selected_native or accuracy_enabled:
        label = "native_accuracy preprocessing" if selected_native else "recent128 execution=accuracy_eager"
        value = getattr(scheduler, "max_num_seqs", _MISSING)
        if value is not _MISSING:
            if type(value) is not int or value != 1:
                raise ContractError(label + " requires scheduler_config.max_num_seqs=1 before loading weights; "
                    "production batching uses execution=serving with its explicit producer/reader")
            checks.append("max_num_seqs=1")
        value = getattr(model, "enforce_eager", _MISSING)
        if value is not _MISSING:
            if value is not True:
                raise ContractError(label + " requires model_config.enforce_eager=True before loading weights; "
                    "Graph execution uses the serving implementation")
            checks.append("enforce_eager=True")
    if accuracy_enabled:
        cache = getattr(vllm_config, "cache_config", None)
        prefix = getattr(cache, "enable_prefix_caching", _MISSING)
        if prefix is not _MISSING:
            if prefix:
                raise ContractError("recent128 execution=accuracy_eager excludes prefix caching before loading weights")
            checks.append("prefix_caching=False")
        for name, label in (("kv_transfer_config", "PD/KV transfer"), ("speculative_config", "MTP/speculation")):
            value = getattr(vllm_config, name, _MISSING)
            if value is not _MISSING:
                if value is not None:
                    raise ContractError("recent128 execution=accuracy_eager excludes " + label + " before loading weights")
                checks.append(name + "=None")
        parallel = getattr(vllm_config, "parallel_config", None)
        for name in ("prefill_context_parallel_size", "decode_context_parallel_size"):
            value = getattr(parallel, name, _MISSING)
            if value is not _MISSING:
                if type(value) is not int or value != 1:
                    raise ContractError("recent128 execution=accuracy_eager requires " + name + "=1 before loading weights")
                checks.append(name + "=1")
    return dict(status="STATIC_CONFIG_VALIDATED", execution=config.execution, enabled=config.enabled,
        preprocess=config.preprocess, checked=checks, scope="static_config_only",
        preprocessing_type="NOT_INFERRED_BEFORE_WEIGHT_PREPARATION", device_support="NOT_VERIFIED")
