"""PD connector identity, not source export or transfer validation.

The live probe runs through an existing worker RPC and never constructs a
second connector, loads weights, reads tensor contents or changes settings.
"""
import hashlib
import inspect
import marshal
import os
from pathlib import Path
import sys

_METHODS = ("register_kv_caches", "build_connector_worker_meta", "update_connector_output",
    "start_load_kv", "get_finished", "get_transfer_results", "request_finished", "_handle_request", "_handle_requests",
    "_reformat_pending_kv_caches", "process_pulling_result", "run_busy_loop", "_run_busy_loop",
    "register_memory", "unregister_memory", "batch_transfer_sync_read", "batch_transfer_sync_write")


def describe(value):
    cls = value if inspect.isclass(value) else type(value)
    result = dict(module=cls.__module__, qualname=cls.__qualname__, methods={})
    bridge = getattr(value, "_recent128_tail_bridge", None)
    report = getattr(bridge, "diagnostics", None)
    if callable(report):
        result["recent128_pd"] = report()
    module = sys.modules.get(cls.__module__)
    filename = getattr(module, "__file__", None)
    result["module_file"] = filename
    if isinstance(filename, str):
        try:
            path = Path(filename)
            if path.is_file() and path.stat().st_size <= 2 * 1024 * 1024:
                result["module_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            result["fingerprint_error"] = str(error)
    for name in _METHODS:
        function = getattr(value, name, None)
        if not callable(function):
            continue
        try:
            signature = [dict(name=p.name, kind=str(p.kind), required=p.default is p.empty)
                         for p in inspect.signature(function).parameters.values()]
        except (TypeError, ValueError):
            signature = "UNAVAILABLE"
        result["methods"][name] = signature
    return result


def inspect_runtime_state(*, modules=None):
    modules = sys.modules if modules is None else modules
    runtime = modules.get("sfa_recent128.serving_runtime")
    states = getattr(runtime, "_STATES", {}) if runtime is not None else {}
    layers = []
    for state in tuple(states.values()):
        for layer in tuple(getattr(state, "runtime_layers", ())):
            entry = dict(layer=getattr(layer, "layer_key", None), device=str(getattr(state, "device", None)),
                producer_kind=getattr(layer, "producer_kind", None),
                producer_calls=getattr(layer, "producer_calls", None), read_calls=getattr(layer, "read_calls", None),
                diagnostic_file=str(getattr(layer, "report", None)))
            function = getattr(getattr(layer, "read", None), "__func__", None)
            code = getattr(function, "__code__", None)
            if code is not None:
                entry["loaded_read_code_sha256"] = hashlib.sha256(marshal.dumps(code)).hexdigest()
            controller = getattr(layer, "native_controller", None)
            plans = tuple(getattr(controller, "_plans", {}).values()) if controller is not None else ()
            entry["native_plan_count"] = len(plans)
            entry["warmed_native_plan_count"] = sum(bool(getattr(getattr(p, "writer", None), "warmed", False)) for p in plans)
            sources = []
            for plan in plans:
                kernel = getattr(getattr(plan, "writer", None), "_kernel", None)
                try:
                    source = getattr(kernel, "src", None)
                except Exception as error:
                    entry.setdefault("kernel_fingerprint_errors", []).append(type(error).__name__)
                    continue
                if isinstance(source, str):sources.append(hashlib.sha256(source.encode()).hexdigest())
            entry["loaded_kernel_source_sha256"] = sorted(set(sources))
            layers.append(entry)
    return dict(worker_pid=os.getpid(), status="EXISTING_WORKER_STATE" if layers else "NO_EXISTING_WORKER_STATE",
        layers=layers, counts_semantics="Python eager/warmup/capture entries, not GPU kernel execution or Graph replay counters",
        file_refresh="Files are written at initialization and explicit runtime flush/fault RPC, not after every inference")


def inspect_live_pd(*, modules=None):
    modules = sys.modules if modules is None else modules
    report = dict(version="recent128-pd-identity-v1", status="NO_LIVE_CONNECTOR",
        scope="existing_worker_identity_only", device_operations=0, connector=None, children={},
        runtime=inspect_runtime_state(modules=modules))
    for name in ("vllm.distributed.kv_transfer", "vllm.distributed.kv_transfer.kv_transfer_state"):
        getter = getattr(modules.get(name), "get_kv_transfer_group", None)
        if not callable(getter):
            continue
        try:
            connector = getter()
        except Exception as error:
            report["getter_error_type"] = type(error).__name__
            continue
        if connector is None:
            continue
        report.update(status="LIVE_CONNECTOR_IDENTIFIED_NOT_TRANSFER_VALIDATED", connector=describe(connector), getter_module=name)
        seen = {id(connector): "connector"}

        def visit(parent, prefix="", depth=0):
            if depth > 32:
                report.setdefault("traversal_errors", {})[prefix] = "MAX_DEPTH"
                return
            # MultiConnector owns children here; sub_connectors returns a copy.
            children = getattr(parent, "_connectors", None)
            if not isinstance(children, (list, tuple)):
                children = getattr(parent, "sub_connectors", None)
            links = []
            if isinstance(children, (list, tuple)):
                links.extend((f"_connectors[{index}]", child) for index, child in enumerate(children))
            for key in ("connector_worker", "_connector_worker", "worker", "_worker",
                        "engine", "kv_send_thread", "kv_recv_thread", "_recving_thread", "_sending_thread"):
                child = getattr(parent, key, None)
                if child is not None:
                    links.append((key, child))
            for key, child in links:
                path = prefix + "." + key if prefix else key
                if id(child) in seen:
                    report["children"][path] = dict(alias_of=seen[id(child)])
                    continue
                seen[id(child)] = path
                report["children"][path] = describe(child)
                visit(child, path, depth + 1)

        visit(connector)
        return report
    return report


def probe_worker_pd(worker):
    """Pass this callable to the service's existing collective_rpc."""
    return inspect_live_pd()


def verify_worker_recent128(worker):
    """Explicit post-request sync/flush via the actual worker's existing RPC.

    Never call in forward or measured timing. No imports create a substitute
    runtime; an empty standalone process remains NOT_EXERCISED.
    """
    report = inspect_live_pd()
    module = sys.modules.get("sfa_recent128.serving_runtime")
    validate = getattr(module, "validate_serving_faults", None)
    report["validation"] = validate() if callable(validate) else dict(status="NOT_EXERCISED")
    report["runtime"] = inspect_runtime_state()
    report["device_operations"] = "EXPLICIT_POST_REQUEST_FAULT_SYNC_ONLY"
    return report
