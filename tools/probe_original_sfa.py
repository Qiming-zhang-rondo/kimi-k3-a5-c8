#!/usr/bin/env python3
"""Stage the uploaded, unmodified SFA sources; optionally compile descriptors only.

This does not execute an NPU kernel, install an operator, or supply missing tiling
definitions. A successful probe is a kernel compilation gate, not an operator test.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import traceback


TYPE = "SparseFlashAttentionTilingDataMla"
OP_FIELDS = ("kernel_name", "op_type", "inputs", "outputs", "attrs", "impl_mode",
             "origin_inputs", "origin_outputs", "param_type_dynamic", "mc2_ctx",
             "param_type_list", "init_value_list", "output_shape_depend_on_compute")


def file_record(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def json_value(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return repr(value)


def fixed_arguments():
    def tensor(shape, dtype):
        return {"shape": shape, "ori_shape": list(shape), "dtype": dtype,
                "format": "ND", "ori_format": "ND"}
    return {
        "query_in__": tensor([4, 64, 512], "bfloat16"),
        "key_in__": tensor([2, 128, 1, 512], "bfloat16"),
        "value_in__": tensor([2, 128, 1, 512], "bfloat16"),
        "sparse_indices_in__": tensor([4, 1, 128], "int32"),
        "block_table_in__": tensor([1, 2], "int32"),
        "actual_seq_lengths_query_in__": tensor([1], "int32"),
        "actual_seq_lengths_kv_in__": tensor([1], "int32"),
        "query_rope_in__": tensor([4, 64, 64], "bfloat16"),
        "key_rope_in__": tensor([2, 128, 1, 64], "bfloat16"),
        "attention_out_out_": tensor([4, 64, 512], "bfloat16"),
        "softmax_max_out_": tensor([1, 4, 64], "float32"),
        "softmax_sum_out_": tensor([1, 4, 64], "float32"),
        "scale_value": 1.0 / math.sqrt(576), "sparse_block_size": 1,
        "layout_query": "TND", "layout_kv": "PA_BSND", "sparse_mode": 0,
        "attention_mode": 2, "return_softmax_lse": True,
        "kernel_name": "sfa_original_probe",
    }


def stage_sources(bundle, stage):
    mappings = (
        (bundle / "original", stage / "ascendc/sparse_flash_attention"),
        (bundle / "deps/sparse_flash_attention/arch35", stage / "ascendc/sparse_flash_attention/arch35"),
        (bundle / "deps/ascendc_common", stage / "ascendc/common"),
    )
    manifest = []
    for source_root, destination_root in mappings:
        if not source_root.is_dir():
            raise FileNotFoundError(str(source_root))
        for source in sorted(source_root.rglob("*")):
            if not source.is_file():
                continue
            destination = destination_root / source.relative_to(source_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise ValueError("Conflicting source overlay: " + str(destination))
            shutil.copyfile(source, destination)
            manifest.append({"source": str(source), **file_record(destination)})
    adapter_source = bundle / "deps/sparse_flash_attention_dynamic/sparse_flash_attention.py"
    adapter = stage / "dynamic/sparse_flash_attention.py"
    adapter.parent.mkdir()
    shutil.copyfile(adapter_source, adapter)
    manifest.append({"source": str(adapter_source), **file_record(adapter)})
    return manifest


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("_original_sfa_probe_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_soc(adapter, requested, importer=importlib.import_module, observer=None):
    evidence = {"requested": requested}
    def observe(phase):
        if observer is not None:
            observer(phase, json_value(evidence))
    if requested:
        checked = []
        for name in ("asc_op_compile_base.common.platform", "tbe.common.platform"):
            evidence["module_being_imported"] = name
            observe("before_setter_module_import")
            try:
                module = importer(name)
            except ImportError as error:
                checked.append({"module": name, "error": str(error)})
                continue
            setter = getattr(module, "set_current_compile_soc_info", None)
            if not callable(setter):
                checked.append({"module": name, "api_present": False})
                continue
            # Call only an actually present API with a signature accepting this argument.
            signature = inspect.signature(setter)
            signature.bind(requested)
            evidence["setter"] = {"module": name, "file": getattr(module, "__file__", None),
                                  "signature": str(signature)}
            observe("before_setter")
            result = setter(requested)
            evidence["setter"]["return_value"] = json_value(result)
            observe("after_setter")
            if result is False or (isinstance(result, str) and result.lower() in ("error", "failed", "failure")):
                raise RuntimeError("SDK rejected compile target: " + repr(evidence))
            break
        else:
            raise RuntimeError("No verified SDK set_current_compile_soc_info API: " + json.dumps(checked))
    observe("before_get_soc_version")
    evidence["actual"] = adapter.get_soc_spec("SOC_VERSION")
    observe("before_get_short_soc_version")
    evidence["short"] = adapter.get_soc_spec("SHORT_SOC_VERSION")
    observe("after_soc_queries")
    if "950" not in str(evidence["actual"]).lower():
        raise RuntimeError("Expected an Ascend950 target, got " + repr(evidence))
    if requested and str(evidence["actual"]).lower() != str(requested).lower():
        raise RuntimeError("SDK target differs from the requested exact SoC: " + repr(evidence))
    return evidence


def tiling_evidence(stage, prepared):
    found = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.suffix not in (".h", ".hpp", ".cpp", ".inc"):
            continue
        text = path.read_text(errors="replace")
        clean = re.sub(r'/\*[\s\S]*?\*/|//[^\n]*', "", text)
        definitions = bool(re.search(r'\b(?:struct|class)\s+' + TYPE + r'\b[^;{}]*\{', clean)
                           or re.search(r'\bBEGIN_TILING_DATA_DEF\s*\(\s*' + TYPE + r'\s*\)', clean)
                           or re.search(r'\btypedef\s+struct\b[\s\S]*?}\s*' + TYPE + r'\s*;', clean))
        if path.name == "kernel_tiling.h" or definitions:
            found.append({**file_record(path), "generated_in_probe": str(path) not in prepared,
                          "has_type_definition_candidate": definitions})
    return found


def probe(work_dir, bundle_root=None, compile_requested=False, soc_version=None,
          adapter_loader=load_adapter, importer=importlib.import_module):
    bundle = Path(bundle_root or Path(__file__).resolve().parents[1]).resolve()
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="original_sfa_", dir=root))
    report_path = stage / "probe_report.json"
    report = {"status": "ERROR", "compile_requested": compile_requested,
              "stage_dir": str(stage), "report": str(report_path), "bundle_root": str(bundle),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "unmodified kernel compile only; no NPU execution or operator installation",
              "command": list(sys.argv), "inputs": fixed_arguments(), "compile_calls": []}
    prepared = set()
    previous_cwd = Path.cwd()
    def checkpoint(phase, evidence=None):
        report["last_phase"] = phase
        if evidence is not None:
            report.setdefault("sdk_progress", []).append({"phase": phase, "evidence": evidence})
        write_json(report_path, report)
    checkpoint("before_source_staging")
    try:
        report["sources"] = stage_sources(bundle, stage)
        prepared = {item["path"] for item in report["sources"]}
        source = stage / "ascendc/sparse_flash_attention/sparse_flash_attention.cpp"
        if not source.is_file():
            raise FileNotFoundError(str(source))
        report["expected_source"] = str(source)
        if not compile_requested:
            report["status"] = "PREPARED"
        else:
            build = stage / "build"
            build.mkdir()
            os.chdir(build)
            report["compile_cwd"] = str(build)
            checkpoint("before_sdk_adapter_import")
            adapter = adapter_loader(stage / "dynamic/sparse_flash_attention.py")
            checkpoint("after_sdk_adapter_import")
            report["soc"] = select_soc(adapter, soc_version, importer,
                                       observer=lambda phase, evidence: checkpoint("soc." + phase, evidence))
            original_compile = adapter.compile_op
            original_select = adapter.get_kernel_source

            def require_source(selected):
                if Path(selected).resolve() != source.resolve():
                    raise RuntimeError("Refusing non-staged kernel source: " + str(selected))
                report["selected_source"] = str(Path(selected).resolve())
                return selected

            def checked_select(*args, **kwargs):
                return require_source(original_select(*args, **kwargs))

            def recorded_compile(src, origin_name, op_info, options, code_channel, *args, **kwargs):
                require_source(src)
                report["compile_calls"].append({
                    "source": str(src), "origin_func_name": origin_name,
                    "op_info": {key: json_value(getattr(op_info, key)) for key in OP_FIELDS if hasattr(op_info, key)},
                    "options": json_value(options), "code_channel": json_value(code_channel),
                    "remaining_args": json_value(args), "kwargs": json_value(kwargs)})
                write_json(report_path, report)
                return original_compile(src, origin_name, op_info, options, code_channel, *args, **kwargs)

            adapter.get_kernel_source = checked_select
            adapter.compile_op = recorded_compile
            require_source(adapter.get_kernel_source("sparse_flash_attention.cpp", "sparse_flash_attention", "sparse_flash_attention"))
            # The uploaded adapter chooses all compiler options; do not invent tiling or flags.
            checkpoint("before_adapter_compile")
            adapter.sparse_flash_attention(**fixed_arguments())
            checkpoint("after_adapter_compile")
            if not report["compile_calls"]:
                raise RuntimeError("Adapter returned without invoking compile_op (prebuild is not compilation)")
            artifacts = [file_record(path) for path in sorted(stage.rglob("*"))
                         if path.is_file() and str(path) not in prepared and path.suffix in (".o", ".json")
                         and path != report_path]
            report["compiler_artifacts"] = artifacts
            if not any(item["path"].endswith(".o") and item["bytes"] > 0 for item in artifacts):
                raise RuntimeError("compile_op returned without a new nonempty .o kernel artifact inside the isolated stage")
            report["status"] = "PASS"
    except (Exception, SystemExit) as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        report["first_error"] = report["error"].splitlines()[0]
        report["traceback"] = traceback.format_exc()
    finally:
        os.chdir(previous_cwd)
        report["compiler_artifacts"] = [file_record(path) for path in sorted(stage.rglob("*"))
                                        if path.is_file() and str(path) not in prepared
                                        and path.suffix in (".o", ".json") and path != report_path]
        report["tiling_headers"] = tiling_evidence(stage, prepared)
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, help="Evidence parent; each probe creates a fresh child")
    parser.add_argument("--bundle-root")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Default: no SDK imports")
    mode.add_argument("--compile", action="store_true", dest="compile_requested")
    parser.add_argument("--soc-version", help="Set via a verified installed SDK API; otherwise use current SDK target")
    args = parser.parse_args(argv)
    report = probe(args.work_dir, args.bundle_root, args.compile_requested, args.soc_version)
    print(report["status"] + ": " + report["report"])
    if "error" in report:
        print(report["error"], file=sys.stderr)
    return 0 if report["status"] in ("PREPARED", "PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
