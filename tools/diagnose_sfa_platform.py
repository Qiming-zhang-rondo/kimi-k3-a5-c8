#!/usr/bin/env python3
"""Find installed 950 platform configuration and isolate each compile attempt.

Default: inspect files only, without importing an SDK or touching an NPU.
--compile launches the original SFA probe in fresh processes. No package,
environment, service, device, or vendor installation is changed by this tool.
"""
import argparse
import configparser
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import signal
import sys
import tempfile


DEFAULT_ROOTS = ("/usr/local/Ascend/ascend-toolkit/latest", "/usr/local/Ascend/cann/latest",
                 "/usr/local/Ascend/ascend-toolkit", "/usr/local/Ascend/cann")
ROOT_VARIABLES = ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME")
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")
CONFIG_EXTENSIONS = {".ini", ".cfg", ".conf", ".config", ".json"}
SIGNATURES = ("get_platform_info", "SIGSEGV", "Segmentation fault",
              "SparseFlashAttentionTilingDataMla", "fatal error", "507015")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def within(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def evidence(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def valid_soc(value):
    return isinstance(value, str) and bool(NAME.fullmatch(value)) and value.lower().startswith("ascend950")


def names_from_config(path, text):
    result = []
    def add(value, source):
        if valid_soc(value) and {"name": value, "source": source} not in result:
            result.append({"name": value, "source": source})
    add(path.stem, "filename")
    if path.suffix.lower() == ".json":
        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    normalized = re.sub(r"[_\s-]", "", key).lower()
                    if normalized == "socversion":
                        if isinstance(item, list):
                            for name in item:
                                add(name, "json:" + key)
                        else:
                            add(item, "json:" + key)
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        try:
            walk(json.loads(text))
        except (ValueError, TypeError):
            pass
    else:
        config = configparser.ConfigParser(interpolation=None, strict=False, inline_comment_prefixes=("#", ";"))
        try:
            config.read_string(text if re.search(r"^\s*\[", text, re.M) else "[DEFAULT]\n" + text)
            for section in ["DEFAULT", *config.sections()]:
                for key, value in config[section].items():
                    # SHORT_SOC_VERSION identifies a family and must never be
                    # promoted to a supported complete compile-target name.
                    if re.sub(r"[_\s-]", "", key).lower() == "socversion":
                        add(value.strip().strip('"\''), "ini:" + section + ":" + key)
        except configparser.Error:
            pass
    return result


def discover(sdk_roots=(), environ=None, known_roots=DEFAULT_ROOTS):
    environ = os.environ if environ is None else environ
    roots = [(value, "explicit") for value in sdk_roots]
    roots += [(environ[key], key) for key in ROOT_VARIABLES if environ.get(key) and Path(environ[key]).is_dir()]
    roots += [(value, "known_path") for value in known_roots if Path(value).is_dir()]
    result = {"roots": [], "config_files": [], "candidates": [], "config_collisions": [],
              "limits": {"search_depth": 6, "directories_per_root": 512, "configs_total": 512,
                         "config_file_bytes": 2 * 1024 * 1024}, "scan_notes": [], "version_files": []}
    seen_roots, seen_configs, name_entries = set(), set(), {}
    for value, origin in roots:
        root = Path(value).expanduser().resolve()
        if root in seen_roots:
            continue
        seen_roots.add(root)
        result["roots"].append({"path": str(root), "source": origin, "exists": root.is_dir()})
        if not root.is_dir():
            continue
        for relative in ("version.info", "version.cfg", "compiler/version.info"):
            version = root / relative
            try:
                if version.is_file() and within(version, root) and version.stat().st_size <= 2 * 1024 * 1024:
                    result["version_files"].append(evidence(version))
            except OSError as error:
                result["scan_notes"].append(str(error))
        config_dirs = []
        for relative in ("compiler/data/platform_config", "data/platform_config",
                         "aarch64-linux/data/platform_config", "x86_64-linux/data/platform_config",
                         "aarch64-linux/compiler/data/platform_config", "x86_64-linux/compiler/data/platform_config"):
            path = root / relative
            if path.is_dir():
                config_dirs.append(path)
        # Only directory names are inspected outside platform_config; no SDK import.
        walked = 0
        for current, dirs, _files in os.walk(root, followlinks=False, onerror=lambda error: result["scan_notes"].append(str(error))):
            walked += 1
            current = Path(current)
            depth = len(current.relative_to(root).parts)
            dirs[:] = sorted(d for d in dirs if not (current / d).is_symlink())
            if current.name == "platform_config":
                config_dirs.append(current)
                dirs[:] = []
            if depth >= 6:
                dirs[:] = []
            if walked >= 512:
                result["scan_notes"].append("Directory limit reached: " + str(root))
                break
        for directory in dict.fromkeys(config_dirs):
            for current, dirs, files in os.walk(directory, followlinks=False):
                current = Path(current)
                dirs[:] = sorted(d for d in dirs if not (current / d).is_symlink())
                if len(current.relative_to(directory).parts) >= 2:
                    dirs[:] = []
                for filename in sorted(files):
                    path = current / filename
                    if path.suffix.lower() not in CONFIG_EXTENSIONS or path.resolve() in seen_configs:
                        continue
                    if len(seen_configs) >= 512:
                        result["scan_notes"].append("Configuration file limit reached")
                        break
                    try:
                        if not within(path, root) or path.stat().st_size > 2 * 1024 * 1024:
                            result["scan_notes"].append("Skipped external/oversize configuration: " + str(path))
                            continue
                        seen_configs.add(path.resolve())
                        item = {**evidence(path), "sdk_root": str(root)}
                        item["soc_names"] = names_from_config(path, path.read_text(errors="replace"))
                    except OSError as error:
                        result["scan_notes"].append(str(error))
                        continue
                    result["config_files"].append(item)
                    for name in item["soc_names"]:
                        entry = name_entries.setdefault(name["name"], {"name": name["name"], "configured": True, "evidence": []})
                        entry["evidence"].append({"file": item["path"], "sha256": item["sha256"], "source": name["source"], "sdk_root": str(root)})
    result["candidates"] = list(name_entries.values())
    for entry in result["candidates"]:
        hashes = sorted({item["sha256"] for item in entry["evidence"]})
        if len(hashes) > 1:
            result["config_collisions"].append({"name": entry["name"], "sha256_values": hashes, "evidence": entry["evidence"]})
    return result


def load_stage_runner():
    path = Path(__file__).with_name("run_logged_stage.py")
    spec = importlib.util.spec_from_file_location("_sfa_logged_stage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_stage


def validate_probe(probe_root):
    reports = sorted(probe_root.glob("original_sfa_*/probe_report.json"))
    if len(reports) != 1:
        return {"passed": False, "reason": "Expected exactly one probe report", "reports": [str(x) for x in reports]}
    path = reports[0]
    try:
        report = json.loads(path.read_text())
        good_objects = []
        for item in report.get("compiler_artifacts", []):
            obj = Path(item.get("path", ""))
            if obj.suffix == ".o" and within(obj, path.parent) and obj.is_file() and obj.stat().st_size > 0:
                actual = evidence(obj)
                if actual["sha256"] == item.get("sha256") and actual["bytes"] == item.get("bytes"):
                    good_objects.append(actual)
        passed = report.get("status") == "PASS" and report.get("compile_requested") is True and bool(report.get("compile_calls")) and bool(good_objects)
        return {"passed": passed, "report": str(path), "probe_status": report.get("status"),
                "objects_verified": good_objects, "soc": report.get("soc"),
                "last_phase": report.get("last_phase"), "sdk_progress": report.get("sdk_progress", []),
                "first_error": report.get("first_error"), "error": report.get("error"),
                "reason": None if passed else "Probe PASS, actual compile call, and matching nonempty object are all required"}
    except (OSError, ValueError, TypeError) as error:
        return {"passed": False, "report": str(path), "reason": str(error)}


def diagnose(work_dir, sdk_roots=(), compile_requested=False, soc_version=None, max_attempts=4,
             bundle_root=None, *, environ=None, known_roots=DEFAULT_ROOTS, stage_runner=None, probe_script=None, quiet=False):
    if not 1 <= max_attempts <= 4:
        raise ValueError("max_attempts must be 1..4")
    if soc_version is not None and not valid_soc(soc_version):
        raise ValueError("Explicit SoC must be a single safe 950 target name")
    root = Path(work_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="sfa_platform_", dir=root))
    bundle = Path(bundle_root or Path(__file__).resolve().parents[1]).resolve()
    report = {"report": str(output / "diagnosis.json"), "output_dir": str(output),
              "scope": "Platform files and original kernel compilation only; no NPU execution or full operator validation",
              "environment_modified": False, "compile_requested": compile_requested, "attempts": [],
              "explicit_soc_version": soc_version, "selected": None}
    report["discovery"] = discover(sdk_roots, environ, known_roots)
    candidates = [dict(item) for item in report["discovery"]["candidates"]]
    if soc_version:
        explicit = next((item for item in candidates if item["name"] == soc_version), None)
        if explicit:
            candidates.remove(explicit)
        else:
            explicit = {"name": soc_version, "configured": False, "evidence": []}
        explicit["explicit"] = True
        candidates.insert(0, explicit)
    report["planned_candidates"] = candidates[:max_attempts]
    report["candidates_not_attempted_due_to_limit"] = [x["name"] for x in candidates[max_attempts:]]
    report["status"] = "CONFIG_DISCOVERED" if report["discovery"]["candidates"] else "CONFIG_NOT_FOUND"
    save(Path(report["report"]), report)
    if not compile_requested:
        return report
    if not candidates:
        report["next_action"] = "Internal execution agent should locate this installed SDK's platform_config package; no target name was invented"
        save(Path(report["report"]), report)
        return report
    run_stage = stage_runner or load_stage_runner()
    script = Path(probe_script or bundle / "tools/probe_original_sfa.py").resolve()
    report["status"] = "COMPILE_FAILED"
    for index, candidate in enumerate(candidates[:max_attempts]):
        attempt = output / ("attempt_%02d" % index)
        attempt.mkdir()
        probe_root = attempt / "probes"
        argv = [sys.executable, "-X", "faulthandler", str(script), "--bundle-root", str(bundle), "--work-dir", str(probe_root),
                "--compile", "--soc-version", candidate["name"]]
        result = run_stage("baseline_build", attempt / "logged", argv, cwd=attempt,
                           console=io.BytesIO() if quiet else None)
        entry = {"target": candidate, "process": result, "probe": validate_probe(probe_root)}
        code = result.get("returncode")
        if isinstance(code, int) and code < 0:
            try:
                signame = signal.Signals(-code).name
            except ValueError:
                signame = "unknown"
            entry["signal"] = {"number": -code, "name": signame}
        log = Path(result["log"]).read_text(errors="replace")
        entry["log_signatures"] = [text for text in SIGNATURES if text.lower() in log.lower()]
        entry["compile_passed"] = result.get("status") == "PASS" and code == 0 and entry["probe"]["passed"]
        report["attempts"].append(entry)
        if entry["compile_passed"]:
            report["status"] = "COMPILE_PASS"
            report["selected"] = candidate
        save(Path(report["report"]), report)
        if entry["compile_passed"]:
            break
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", action="append", default=[])
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--bundle-root")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect-only", action="store_true", help="Default: file inspection only")
    mode.add_argument("--compile", action="store_true", dest="compile_requested")
    parser.add_argument("--soc-version", help="Optional exact comparison target; unconfigured names are marked unverified")
    parser.add_argument("--max-attempts", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--quiet", action="store_true", help="Keep child output in stage.log instead of streaming")
    args = parser.parse_args(argv)
    if args.soc_version and not valid_soc(args.soc_version):
        parser.error("--soc-version must be a single safe 950 target name")
    report = diagnose(args.work_dir, args.sdk_root, args.compile_requested, args.soc_version,
                      args.max_attempts, args.bundle_root, quiet=args.quiet)
    print(report["status"] + ": " + report["report"])
    return 0 if report["status"] in ("CONFIG_DISCOVERED", "COMPILE_PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
