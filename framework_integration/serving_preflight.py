#!/usr/bin/env python3
"""Aggregate serving checks before model loading; never launch a device kernel.

Default mode reads JSON and local paths only, without Torch/vendor imports.
--check-host additionally imports the provider, queries reader host ABI symbols
and locates build tools. It does not install routes, compile a bridge, allocate
NPU tensors, load model weights, or change any engine setting.
Existing provider module/extension initialization may execute during its import;
this command makes no explicit Torch/NPU API call or provider invocation.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import ctypes
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

REQUIRED_SOURCE_FILES = (
    "experimental/single_pass/batched_serving_bridge.cpp",
    "experimental/single_pass/batched_serving_api.h",
    "experimental/single_pass/single_pass_api.h",
    "experimental/single_pass/single_pass_tiling.h",
    "experimental/torch/load_sfa_direct.py",
)
CONFIG_KEYS = {"window", "execution", "preprocess", "max_query_chunk", "reader",
               "library_dir", "source_root", "diagnostics_path", "single_pass_kernel"}
LIBRARY_NAME = "libqsfa_recent128_direct.so"


def _positive(value):
    return type(value) is int and value > 0


def _probe_provider():
    module = importlib.import_module("cloud_ops_turbo")
    function = getattr(module, "cloud_mla_prolog_v3_recent128", None)
    if not callable(function):
        raise RuntimeError("cloud_ops_turbo lacks callable cloud_mla_prolog_v3_recent128")
    marker = getattr(function, "recent128_abi_version", None)
    if type(marker) is not int or marker != 1:
        raise RuntimeError("The exported FUNCTION must have integer recent128_abi_version=1")
    return ("Callable provider with function ABI1, module=" + str(getattr(module, "__file__", None)) +
            "; first-call TBE compilation and side effects remain untested")


def _probe_original_prolog_exports():
    module = importlib.import_module("cloud_ops_turbo")
    names = ("cloud_mla_prolog_v3", "cloud_mla_prolog_v3_functional")
    missing = [name for name in names if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError("Reviewed PR199 Python selection also requires the ORIGINAL Cloud exports: " + ", ".join(missing))
    return "Both original Cloud Prolog exports are callable; Python layer selection still depends on the actual weights/role"


def _probe_library(path):
    handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    version = handle.QsfaBatchedServingAbiVersion
    size = handle.QsfaBatchedServingBuffersBytes
    getattr(handle, "QsfaBatchedServingLaunch")  # Presence only; NEVER called.
    version.argtypes = size.argtypes = []
    version.restype = size.restype = ctypes.c_uint32
    if version() != 1 or size() != 160:
        raise RuntimeError("Reader requires additive serving ABI1/160B")
    return "Reader host exports and ABI1/160B present; no device launch or bridge compilation"


def build_report(engine, *, check_host=False, environ=None):
    environ = os.environ if environ is None else environ
    checks = []

    def add(name, status, detail, actual=None):
        item = dict(name=name, status=status, detail=detail)
        if actual is not None:
            item["actual"] = actual
        checks.append(item)

    def require(name, condition, detail, actual=None):
        add(name, "PASS" if condition else "BLOCKED", detail, actual)

    def run(name, function):
        try:
            detail = function()
        except Exception as error:
            add(name, "BLOCKED", type(error).__name__ + ": " + str(error))
        else:
            add(name, "PASS", detail)

    if not isinstance(engine, Mapping):
        add("engine.mapping", "BLOCKED", "Engine JSON must be an object")
        engine = {}
    options = engine.get("engine_options", engine)
    if not isinstance(options, Mapping):
        add("engine.options", "BLOCKED", "engine_options must be an object when supplied")
        options = {}
    else:
        add("engine.options", "PASS", "Read actual engine options; no values were changed")
    additional = options.get("additional_config")
    require("activation.explicit_config", isinstance(additional, Mapping) and "sfa_recent128" in additional,
            "Serving activation requires additional_config.sfa_recent128")
    legacy_env = [key for key in ("SFA_RECENT128_CONFIG", "SFA_RECENT128_LIB_DIR") if key in environ]
    require("activation.environment", not legacy_env or
            (isinstance(additional, Mapping) and "sfa_recent128" in additional),
            "Legacy environment variables alone do not activate this overlay", legacy_env)
    config = additional.get("sfa_recent128") if isinstance(additional, Mapping) else None
    require("config.mapping", isinstance(config, Mapping), "sfa_recent128 must be an object")
    config = config if isinstance(config, Mapping) else {}
    dsa_cp = additional.get("enable_dsa_cp") if isinstance(additional, Mapping) else None
    add("route.enable_dsa_cp", "BLOCKED" if dsa_cp is True else ("PASS" if dsa_cp is False else "UNKNOWN"),
        "Reviewed VA enable_dsa_cp=True selects a path that disables fused preprocessing; serving requires PROLOG_V3. No setting was changed; the actual selected implementation still needs runtime evidence",
        dsa_cp)
    attention = options.get("attention_config")
    attention = attention if isinstance(attention, Mapping) else {}
    modes = [("engine", options["mla_kv_quant_mode"])] if "mla_kv_quant_mode" in options else []
    if "mla_kv_quant_mode" in attention:
        modes.append(("attention_config", attention["mla_kv_quant_mode"]))
    if not modes:
        add("route.mla_kv_quant_mode", "UNKNOWN", "No explicit known MLA KV quantization key; verify the actual resolved MXFP4 route")
    for location, mode in modes:
        require("route.mla_kv_quant_mode." + location,
                isinstance(mode, str) and mode.strip().lower() == "mxfp4",
                "Serving requires the original MLA KV mode mxfp4; checked only the explicit known key", mode)
    require("config.keys", not (set(config) - CONFIG_KEYS), "Only implemented configuration fields are accepted",
            sorted(set(config) - CONFIG_KEYS))
    for field, expected in (("window", 128), ("execution", "serving"), ("preprocess", "preserve"),
                            ("reader", "single_pass"), ("single_pass_kernel", "batched_v4")):
        # Config defaults match contract.py; this command checks the serving arm.
        default = {"window": 0, "execution": "accuracy_eager", "preprocess": "preserve",
                   "reader": "native", "single_pass_kernel": "overlay"}[field]
        actual = config.get(field, default)
        require("config." + field, actual == expected and
                (type(actual) is int if field == "window" else isinstance(actual, str)),
                "Serving requires " + field + "=" + str(expected), actual)
    max_q = config.get("max_query_chunk", 1024)
    require("capacity.max_query_chunk", _positive(max_q) and max_q <= 65536,
            "Serving max_query_chunk must be an integer in [1,65536]", max_q)
    scheduler = options.get("scheduler_config", {})
    scheduler = scheduler if isinstance(scheduler, Mapping) else {}
    p = options.get("max_num_seqs", scheduler.get("max_num_seqs"))
    tokens = options.get("max_num_batched_tokens", scheduler.get("max_num_batched_tokens"))
    for name, value in (("max_num_seqs", p), ("max_num_batched_tokens", tokens)):
        add("capacity." + name, "UNKNOWN" if value is None else ("PASS" if _positive(value) else "BLOCKED"),
            "Positive resolved scheduler value required; absent engine defaults are not guessed", value)
    capacity = ((128 + max_q - 1 + 127) // 128) * 128 if _positive(max_q) and max_q <= 65536 else None
    if capacity is not None:
        add("capacity.ring", "PASS", "C is 128-aligned and retains W+maxQ-1; per-layer bytes include payload and tags",
            dict(ring_capacity=capacity, request_pools=p,
                 bytes_per_layer=p * capacity * 1168 if _positive(p) else None))
    add("capacity.actual_query_bound", "PASS" if _positive(tokens) and _positive(max_q) and tokens <= max_q else "PENDING",
        "Every actual per-request query chunk must be <=max_query_chunk; aggregate batch tokens are not a per-request count")

    compilation = options.get("compilation_config", {})
    compilation = compilation if isinstance(compilation, Mapping) else {}
    captures = compilation.get("cudagraph_capture_sizes", options.get("cudagraph_capture_sizes"))
    if captures is not None:
        valid_captures = isinstance(captures, (list, tuple)) and all(_positive(x) for x in captures)
        require("graph.capture_sizes", valid_captures, "Capture sizes must be positive integer token buckets", captures)
        if valid_captures and _positive(tokens):
            require("graph.token_capacity", all(x <= tokens for x in captures),
                    "Every capture token bucket must fit the preallocated max_num_batched_tokens")
    else:
        add("graph.capture_sizes", "UNKNOWN", "Resolved Graph capture buckets must be checked before warmup")

    roots = {}
    for field in ("source_root", "library_dir", "diagnostics_path"):
        value = config.get(field)
        valid = isinstance(value, str) and bool(value.strip())
        require("paths." + field, valid, "Serving requires a nonempty " + field, value)
        if valid:
            roots[field] = Path(value).expanduser()
    source = roots.get("source_root")
    for relative in REQUIRED_SOURCE_FILES:
        add("source." + relative, "UNKNOWN" if source is None else
            ("PASS" if (source / relative).is_file() else "BLOCKED"),
            "Required public reader/bridge source file")
    library = roots.get("library_dir")
    library = library / LIBRARY_NAME if library is not None else None
    try:
        library_status = "UNKNOWN" if library is None else (
            "PASS" if library.is_file() and library.stat().st_size > 0 else "BLOCKED")
        library_detail = "A nonempty " + LIBRARY_NAME + " is required; legacy B1 exports are insufficient"
    except OSError as error:
        library_status, library_detail = "BLOCKED", str(error)
    add("reader.library_file", library_status, library_detail,
        str(library) if library is not None else None)
    sdk_record, asc_compiler = None, None
    if library is not None:
        for parent in (library.parent, library.parent.parent):
            cache = parent / "CMakeCache.txt"
            if not cache.is_file():
                continue
            try:
                if cache.stat().st_size > 2 * 1024 * 1024:
                    raise ValueError("CMakeCache exceeds bounded metadata read")
                for line in cache.read_text(errors="replace").splitlines():
                    if line.startswith("SFA_SDK_ROOT:PATH="):
                        sdk_record = line.split("=", 1)[1]
                    if line.startswith("CMAKE_ASC_COMPILER:FILEPATH="):
                        asc_compiler = line.split("=", 1)[1]
            except (OSError, ValueError) as error:
                add("sdk.cache_read", "UNKNOWN", str(error))
            break
    active_sdk = environ.get("ASCEND_HOME_PATH")
    if sdk_record and active_sdk:
        try:
            require("sdk.identity", Path(sdk_record).expanduser().resolve() == Path(active_sdk).expanduser().resolve(),
                    "ASCEND_HOME_PATH and the reader's recorded SDK must resolve to the same installation")
        except (OSError, RuntimeError) as error:
            add("sdk.identity", "BLOCKED", str(error))
    else:
        add("sdk.identity", "UNKNOWN", "Both active and recorded SDK identities are needed to prove a match")
    selected_sdk = active_sdk or sdk_record
    if selected_sdk:
        sdk = Path(selected_sdk).expanduser()
        require("sdk.directory", sdk.is_dir(), "Selected/recorded SDK directory must exist", str(sdk))
        require("sdk.acl_header", any((sdk / suffix).is_file() for suffix in (
            "aarch64-linux/include/acl/acl.h", "include/acl/acl.h")), "Matched ACL header must exist")
    else:
        add("sdk.directory", "UNKNOWN", "No active SDK path or reader CMakeCache SDK record; no installation is guessed")
    if asc_compiler:
        require("sdk.asc_compiler", Path(asc_compiler).is_file(), "Recorded ASC compiler must still exist", asc_compiler)
    diagnostics = roots.get("diagnostics_path")
    if diagnostics is not None:
        parent = diagnostics
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        require("paths.diagnostics_writable", (not diagnostics.exists() or diagnostics.is_dir()) and
                parent.is_dir() and os.access(parent, os.W_OK | os.X_OK),
                "Diagnostics needs a directory or a writable ancestor; this check creates nothing")

    # Read only checkpoint JSON when a local model path is actually supplied.
    if isinstance(options.get("model"), str) and Path(options["model"]).expanduser().is_dir():
        try:
            from model_preflight import inspect_model_format, COMPATIBLE
        except ImportError:
            # Supports package-style host tests as well as direct CLI execution.
            spec = importlib.util.spec_from_file_location("recent128_json_format", Path(__file__).with_name("model_preflight.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            inspect_model_format, COMPATIBLE = module.inspect_model_format, module.COMPATIBLE
        try:
            model = inspect_model_format(dict(options))
            state = "PASS" if model["status"] == COMPATIBLE else (
                "UNKNOWN" if model["status"] == "UNKNOWN_FORMAT" else "BLOCKED")
            add("model.json_format", state, model.get("reason", model["status"]), model)
        except Exception as error:
            add("model.json_format", "UNKNOWN", str(error))
    else:
        add("model.json_format", "UNKNOWN", "No available local checkpoint directory; no model download/import is attempted")

    parallel = options.get("parallel_config", {})
    parallel = parallel if isinstance(parallel, Mapping) else {}
    for field in ("prefill_context_parallel_size", "decode_context_parallel_size"):
        value = options.get(field, parallel.get(field))
        if value is not None:
            require("parallel." + field, _positive(value), "Context-parallel size must be a positive integer", value)
        cp_status = "PASS" if type(value) is int and value == 1 else (
            "BLOCKED" if type(value) is int and value > 1 else "UNKNOWN")
        add("feature." + field, cp_status,
            "Explicit size1 needs no CP exchange; CP>1 is rejected by the current serving preflight because matched sidecar gather/shard completion is not implemented; absent selectors remain unresolved", value)
    feature_settings = {
        "B8": p,
        "Graph": options.get("enforce_eager"),
        "MTP": options.get("speculative_config"),
        "APC": options.get("enable_prefix_caching"),
        "PD": options.get("kv_transfer_config"),
    }
    for name, value in feature_settings.items():
        disabled = ((name == "Graph" and value is True) or
                    (name == "APC" and value is False) or
                    (name in ("MTP", "PD") and name_setting_present(options, name) and value is None))
        add("feature." + name, "PASS" if disabled else "UNKNOWN",
            "Explicitly disabled in the actual engine" if disabled else {
                "B8": "Stable request-bank/epoch lifecycle and real batch execution must be verified; host acceptance is insufficient",
                "Graph": "Every device/stream/shape bucket and routing lane needs preparation before capture and dynamic replay validation",
                "MTP": "Corrected GPU positions, target/draft tail availability and accept/reject transactions cannot be proved from config",
                "APC": "Requires paired prefix-boundary C4+original-BF16 snapshots, bounded eviction and miss/recompute wiring; callback API is not proof",
                "PD": "Requires actual BF16 send/receive, all-layer readiness, bank/epoch rebinding and completion ordering; C4 arrival alone is insufficient",
            }[name], value)
    spec = options.get("speculative_config")
    if isinstance(spec, Mapping) and "num_speculative_tokens" in spec:
        draft = spec["num_speculative_tokens"]
        require("capacity.speculative_query_bound", type(draft) is int and draft >= 0 and
                _positive(max_q) and draft + 1 <= max_q,
                "Target speculative query width (draft+1) must fit max_query_chunk", draft)

    for name, detail in (
        ("model.runtime_format", "Verify actual PROLOG_V3/MXFP4, D512/R64/KV-head1, BF16 Q, contiguous page128/packed400, INT32 TopK2048 and positive scale"),
        ("runtime.hooks", "Verify actual installed overlay/source hashes and runner/builder/Prolog/read signatures before model execution"),
        ("runtime.budgets", "Resolve all target/draft layers and TP ranks, full sidecar/reader scratch memory and producer UB/event budgets"),
        ("producer.first_call", "Actual TBE compilation, exactly one in-Prolog side-write, OFF/ON query+C4 packed/scale byte controls and BF16/KR references remain required"),
        ("device.matrix", "Run the required B1/B8, padding/reorder/epoch, wrap/rollback, Graph and enabled lifecycle cases; drain sticky reader faults"),
        ("service.validation", "Model task accuracy, TTFT/TPOT, concurrency and memory costs require real paired measurements"),
    ):
        add(name, "PENDING", detail)
    if check_host:
        run("host.provider_function_abi", _probe_provider)
        run("host.original_prolog_exports", _probe_original_prolog_exports)
        if library is None:
            add("host.reader_abi", "UNKNOWN", "library_dir is absent; independent provider/tool checks still ran")
        else:
            run("host.reader_abi", lambda: _probe_library(library))
        require("host.cxx_compiler", any(shutil.which(name) for name in ("c++", "g++", "clang++")),
                "A host C++ compiler is needed for the matched Torch bridge")
        require("host.ninja", shutil.which("ninja") is not None, "Ninja must be available for the Torch bridge")
        add("host.python", "PASS", "Current Python executable; Torch/NPU ABI compatibility remains a runtime prerequisite", sys.executable)
    else:
        for name in ("provider_function_abi", "original_prolog_exports", "reader_abi", "cxx_compiler", "ninja"):
            add("host." + name, "PENDING", "Not requested; use --check-host for independent host capability checks")
    add("host.bridge_compilation", "PENDING", "No Torch bridge is compiled/loaded by this command; matched SDK/Torch ABI must be verified before weights")
    counts = {status.lower(): sum(check["status"] == status for check in checks)
              for status in ("PASS", "BLOCKED", "UNKNOWN", "PENDING")}
    return dict(version="recent128-serving-preflight-v1",
        status="BLOCKED_BEFORE_MODEL_LOAD" if counts["blocked"] else "HOST_CONFIG_CHECKED_RUNTIME_PENDING",
        mode="config_and_host" if check_host else "config_only", checks=checks, counts=counts,
        host_import_caveat="--check-host provider import may execute existing module/extension initialization; this command does not invoke the provider or explicitly call Torch/NPU APIs",
        engine_settings_changed=False, model_weights_loaded=False, device_launches=0,
        runtime_validated=False, all_serving_requirements_validated=False)


def name_setting_present(options, feature):
    return {"MTP": "speculative_config", "PD": "kv_transfer_config"}[feature] in options


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New local output directory")
    parser.add_argument("--check-host", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("Use a NEW output directory; existing paths are never overwritten")
    try:
        engine = json.loads(args.engine_json.read_text())
        report = build_report(engine, check_host=args.check_host)
    except (OSError, ValueError) as error:
        report = dict(status="BLOCKED_BEFORE_MODEL_LOAD", checks=[dict(name="engine.json", status="BLOCKED", detail=str(error))],
            counts={"pass": 0, "blocked": 1, "unknown": 0, "pending": 0}, runtime_validated=False,
            all_serving_requirements_validated=False, engine_settings_changed=False, model_weights_loaded=False, device_launches=0)
    args.output.mkdir(parents=True)
    (args.output / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [report["status"], "No settings changed, model weights loaded, or device kernels launched.",
             "Counts: " + json.dumps(report["counts"], sort_keys=True)]
    lines += ["[" + item["status"] + "] " + item["name"] + ": " + item["detail"] for item in report["checks"]]
    (args.output / "preflight.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(status=report["status"], counts=report["counts"], report=str(args.output / "preflight.json"))))
    return 2 if report["counts"]["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
