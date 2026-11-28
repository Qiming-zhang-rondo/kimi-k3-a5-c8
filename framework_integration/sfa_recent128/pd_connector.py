"""Committed BF16-tail boundaries for the reviewed VA Mooncake V1 child.

This module adds control metadata only. It preserves the original connector,
metadata classes, C4 transfers, asynchronous-save timestamps and store siblings.
The worker transport consumes ``recent128_sends`` before finished-bank release,
and ``recent128_end`` before declaring the native receive complete.
"""
from contextvars import ContextVar
from functools import wraps
import importlib

from .contract import parse_recent128_config

LEGACY_MODULE = "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector"
_BUILDING = ContextVar("recent128_legacy_connector_build", default=False)
_MARKER = "_recent128_pd_connector"


def is_serving_config(vllm_config):
    config = parse_recent128_config(getattr(vllm_config, "additional_config", None))
    return config.enabled and config.execution == "serving"


def _has_legacy_type(value):
    return value is not None and any(
        cls.__module__ == LEGACY_MODULE
        and cls.__name__ in ("MooncakeConnector", "MooncakeConnectorWorker", "MooncakeConnectorScheduler")
        for cls in type(value).__mro__)


def is_legacy_connector(connector):
    """Use loaded implementation identity; registry/display names are insufficient."""
    return any(_has_legacy_type(value) for value in (
        connector, getattr(connector, "connector_worker", None),
        getattr(connector, "connector_scheduler", None)))


def iter_connector_leaves(connector, metadata=None):
    """Pair live Multi children with their exact positional metadata children.

    AscendMultiConnector inherits ``_connectors`` and ``metadata.metadata``
    from upstream MultiConnector. ``sub_connectors`` is accepted for an
    already-resolved wrapper that exposes that spelling. Never search a config
    name or assume that the first child is Mooncake.
    """
    def visit(current, current_meta, ancestors):
        if id(current) in ancestors:
            raise RuntimeError("KV connector child graph contains a cycle")
        children = getattr(current, "_connectors", None)
        if children is None:
            children = getattr(current, "sub_connectors", None)
        if children is None:
            yield current, current_meta
            return
        children = tuple(children)
        if current_meta is None:
            child_metadata = (None,) * len(children)
        else:
            child_metadata = getattr(current_meta, "metadata", None)
            if child_metadata is None or len(child_metadata) != len(children):
                raise RuntimeError("Multi connector metadata does not match its actual child count")
        ancestors = ancestors | {id(current)}
        for child, child_meta in zip(children, child_metadata):
            yield from visit(child, child_meta, ancestors)

    yield from visit(connector, metadata, set())


def iter_legacy_connectors(connector, metadata=None):
    for child, child_meta in iter_connector_leaves(connector, metadata):
        if is_legacy_connector(child):
            yield child, child_meta


def _end(value):
    if type(value) is not int or value < 0:
        raise RuntimeError("recent128_end must be a nonnegative committed integer boundary")
    return value


def install_connector_hooks(scheduler_class=None, metadata_class=None):
    """Install once, composing with the currently loaded legacy implementation.

    Tests may supply the reviewed interface classes without importing an NPU
    stack. Real installation resolves the exact public legacy module lazily.
    OFF instances delegate without adding parameters or metadata attributes.
    """
    if scheduler_class is None or metadata_class is None:
        module = importlib.import_module(LEGACY_MODULE)
        scheduler_class = scheduler_class or module.MooncakeConnectorScheduler
        metadata_class = metadata_class or module.MooncakeConnectorMetadata

    original_add = metadata_class.add_new_req
    if not getattr(original_add, _MARKER, False):
        @wraps(original_add)
        def add_new_req(self, request_id, local_block_ids, num_external_tokens,
                        kv_transfer_params, local_full_block_ids=None):
            result = original_add(self, request_id, local_block_ids,
                num_external_tokens, kv_transfer_params, local_full_block_ids)
            if _BUILDING.get() and "recent128_end" in kv_transfer_params:
                self.requests[request_id].recent128_end = _end(kv_transfer_params["recent128_end"])
            return result

        setattr(add_new_req, _MARKER, True)
        metadata_class.add_new_req = add_new_req

    original_finished = scheduler_class.request_finished
    if not getattr(original_finished, _MARKER, False):
        @wraps(original_finished)
        def request_finished(self, request, block_ids):
            result = original_finished(self, request, block_ids)
            if not is_serving_config(self.vllm_config):
                return result
            delay_free, params = result
            # The native method only emits these flags after a completed
            # producer prefill. The prompt may already have been truncated
            # by native on_new_request; use this actual computed prompt.
            if params is not None and params.get("do_remote_prefill") and not params.get("do_remote_decode"):
                committed_end = _end(len(request.prompt_token_ids))
                params["recent128_end"] = committed_end
                if delay_free:
                    pending = getattr(self, "_recent128_sends_pending", None)
                    if pending is None:
                        pending = self._recent128_sends_pending = {}
                    pending[request.request_id] = committed_end
            return result

        setattr(request_finished, _MARKER, True)
        scheduler_class.request_finished = request_finished

    original_build = scheduler_class.build_connector_meta
    if not getattr(original_build, _MARKER, False):
        @wraps(original_build)
        def build_connector_meta(self, scheduler_output):
            if not is_serving_config(self.vllm_config):
                return original_build(self, scheduler_output)
            token = _BUILDING.set(True)
            try:
                metadata = original_build(self, scheduler_output)
            finally:
                _BUILDING.reset(token)
            pending = getattr(self, "_recent128_sends_pending", {})
            sends = {}
            for request_id in metadata.requests_to_send:
                if request_id not in pending:
                    raise RuntimeError("Native Mooncake send lacks its recent128 committed boundary")
                sends[request_id] = pending.pop(request_id)
            metadata.recent128_sends = sends
            return metadata

        setattr(build_connector_meta, _MARKER, True)
        scheduler_class.build_connector_meta = build_connector_meta

    return dict(status="INSTALLED", scheduler=scheduler_class.__name__,
                metadata=metadata_class.__name__, implementation=LEGACY_MODULE)
