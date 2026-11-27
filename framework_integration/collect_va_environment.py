#!/usr/bin/env python3
"""Read-only VA/CANN environment inventory; never import a framework or use NPU.

Version strings alone cannot establish binary/API compatibility. This captures
the selected Python paths, installed metadata, bounded version files and source
revisions so that the actual PR/image combination can be identified first.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


VERSION = "va-environment-readonly-v1"
FILE_LIMIT = 64 * 1024
ASCEND_ENVIRONMENT = (
    "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH",
)
MODULE_DISTRIBUTIONS = {
    "ascend_vllm": ("mxllm", "ascend-vllm"),
    "vllm_ascend": ("vllm-ascend",),
    "vllm": ("vllm",),
    "torch": ("torch",),
    "torch_npu": ("torch-npu",),
    "cloud_ops_turbo": ("cloud-ops-turbo",),
    "cann_ops_transformer": ("cann-ops-transformer", "cann-ops-transformer-dsl"),
    "cann_ops_transformer_dsl": ("cann-ops-transformer-dsl",),
    "cannbotdsl": ("cannbot-dsl",),
    "ops": ("cannbot-arena-net", "cannbot-arena-net-ops"),
    "triton": ("triton", "triton-ascend"),
}
DEFAULT_ASCEND_ROOTS = (
    "/usr/local/Ascend/cann", "/usr/local/Ascend/cann/latest",
    "/usr/local/Ascend/ascend-toolkit", "/usr/local/Ascend/ascend-toolkit/latest",
    "/usr/local/Ascend/latest",
)
DEFAULT_VERSION_FILES = (
    "/etc/Ascend/ascend_toolkit_install.info",
    "/usr/local/Ascend/ascend_toolkit_install.info",
    "/usr/local/Ascend/driver/version.info",
    "/tmp/cloudops/cann_pkg/vendors/custom_cot/version.info",
)
VERSION_NAMES = ("version.info", "version.cfg", "ascend_toolkit_install.info")


def _decode(value, limit=FILE_LIMIT):
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[:limit]


def _command(arguments, *, timeout):
    """Only fixed read-only commands enter this helper; no shell interpolation."""
    try:
        result = subprocess.run(arguments, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout,
                                check=False)
        return dict(returncode=result.returncode, stdout=_decode(result.stdout),
                    stderr=_decode(result.stderr, 8192), timed_out=False,
                    stdout_truncated=len(result.stdout)>FILE_LIMIT,
                    stderr_truncated=len(result.stderr)>8192)
    except subprocess.TimeoutExpired as error:
        return dict(returncode=None, stdout=_decode(error.stdout),
                    stderr=_decode(error.stderr, 8192), timed_out=True)
    except OSError as error:
        return dict(returncode=None, stdout="", stderr=str(error)[:8192], timed_out=False)


def _path_record(value, source):
    path = Path(value).expanduser().absolute()
    record = dict(path=str(path), source=source, exists=path.exists(),
                  is_symlink=path.is_symlink())
    try:
        record["real_path"] = str(path.resolve(strict=False))
    except (OSError, RuntimeError) as error:
        record["real_path"] = str(path)
        record["resolve_error"] = str(error)[:2048]
    if record["is_symlink"]:
        try:
            record["symlink_target"] = os.readlink(path)
        except OSError as error:
            record["symlink_error"] = str(error)[:2048]
    return record


def _git_root(path):
    """Inspect only ancestor .git markers, never recurse through the filesystem."""
    path = Path(path)
    if not path.is_dir():
        path = path.parent
    try:
        path = path.resolve(strict=False)
    except (OSError, RuntimeError):
        return None
    for candidate in (path, *path.parents):
        if (candidate / ".git").is_dir() or (candidate / ".git").is_file():
            return str(candidate)
    return None


def _git_identity(root):
    record = dict(root=root, head=None, branch=None, dirty=None, detached=None)
    git = shutil.which("git")
    if git is None:
        record["error"] = "git command unavailable"
        return record
    # No remotes/config URLs are collected: credential-bearing userinfo or
    # query strings therefore cannot leak through a remote URL field.
    git_read = [git, "--no-optional-locks", "-C", root]
    head = _command([*git_read, "rev-parse", "HEAD"], timeout=5)
    branch = _command([*git_read, "symbolic-ref", "--quiet", "--short", "HEAD"], timeout=5)
    status = _command([*git_read, "status", "--porcelain=v1", "--untracked-files=normal"], timeout=5)
    if head["returncode"] == 0:
        record["head"] = head["stdout"].strip()
    if branch["returncode"] == 0:
        record["branch"] = branch["stdout"].strip() or None
        record["detached"] = False
    elif branch["returncode"] == 1 and record["head"]:
        record["detached"] = True
    if status["returncode"] == 0:
        record["dirty"] = bool(status["stdout"])
    errors = {}
    for name, result in (("head", head), ("branch", branch), ("status", status)):
        if result["timed_out"] or (result["returncode"] != 0 and not
                                   (name == "branch" and record["detached"])):
            errors[name] = dict(returncode=result["returncode"], timed_out=result["timed_out"],
                                error=result["stderr"][:2048])
    if errors:
        record["errors"] = errors
    return record


def _version_kind(path, source):
    parts = Path(path).parts
    if "custom_cot" in parts:
        return "custom_cot"
    if "driver" in parts:
        return "driver"
    if source == "extra":
        return "extra"
    return "source" if source.startswith("source") else "toolkit"


def _read_version(path, source):
    result = _path_record(path, source)
    result.update(kind=_version_kind(path, source), content=None, truncated=False,
                  bytes_captured=0, size_bytes=None)
    path = Path(result["path"])
    if not path.is_file():
        result["readable_file"] = False
        return result
    try:
        result["size_bytes"] = path.stat().st_size
        with path.open("rb") as stream:
            captured = stream.read(FILE_LIMIT + 1)
        result["truncated"] = len(captured)>FILE_LIMIT
        captured = captured[:FILE_LIMIT]
        result.update(readable_file=True, bytes_captured=len(captured),
                      content=captured.decode("utf-8", errors="replace"),
                      sha256_captured_bytes=hashlib.sha256(captured).hexdigest())
    except OSError as error:
        result.update(readable_file=False, read_error=str(error)[:2048])
    return result


def collect_environment(source_roots=(), extra_version_files=()):
    """Capture evidence without asserting CANN/vendor/framework compatibility."""
    report = dict(version=VERSION, status="ENVIRONMENT_CAPTURED", compatibility="NOT_VERIFIED",
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        scope="metadata/find_spec/read-only files and git; no framework import or NPU initialization",
        python=dict(executable=sys.executable, version=sys.version),
        platform=dict(system=platform.system(), machine=platform.machine(), release=platform.release()),
        ascend_environment={name:os.environ[name] for name in ASCEND_ENVIRONMENT if name in os.environ},
        package_versions={}, modules={}, source_roots=[], git_repositories=[], ascend_paths=[],
        version_files=[], version_file_candidates_checked=[],
        compatibility_requires=["导师已验证的PR编号及实际commit", "实际镜像名称及不可变digest",
                                "该镜像内的VA/CANN/vendor/wheel与候选库配套验证"])
    for distribution in sorted({name for values in MODULE_DISTRIBUTIONS.values() for name in values}):
        try:
            report["package_versions"][distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            report["package_versions"][distribution] = None
        except (OSError, ValueError) as error:
            report["package_versions"][distribution] = None
            report.setdefault("metadata_errors", {})[distribution] = str(error)[:2048]

    git_roots = set()
    for name in MODULE_DISTRIBUTIONS:
        record = dict(found=False, origin=None, real_origin=None, search_locations=[], git_roots=[])
        try:
            # These are all TOP-LEVEL names. Unlike find_spec('package.child'),
            # this does not import a framework parent to find a child module.
            spec = importlib.util.find_spec(name)
            if spec is not None:
                record["found"] = True
                record["origin"] = spec.origin
                candidates = []
                if spec.origin and spec.origin not in ("built-in", "frozen"):
                    origin = _path_record(spec.origin, "module:" + name)
                    record["real_origin"] = origin["real_path"]
                    candidates.append(origin["real_path"])
                for location in spec.submodule_search_locations or ():
                    location = _path_record(location, "module:" + name)
                    record["search_locations"].append(location["real_path"])
                    candidates.append(location["real_path"])
                record["git_roots"] = sorted({root for value in candidates
                                              if (root := _git_root(value)) is not None})
                git_roots.update(record["git_roots"])
        except (ImportError, AttributeError, ValueError, OSError) as error:
            record["error"] = str(error)[:2048]
        report["modules"][name] = record

    roots = []
    for index, value in enumerate(source_roots):
        record = _path_record(value, "source:" + str(index))
        root = _git_root(record["real_path"])
        record["git_roots"] = [root] if root else []
        if root:
            git_roots.add(root)
        report["source_roots"].append(record)
        roots.append((record["path"], record["source"]))
    for name, value in report["ascend_environment"].items():
        for index, component in enumerate(value.split(os.pathsep)):
            if component:
                roots.append((component, "environment:" + name + ":" + str(index)))
    roots.extend((value, "known_install_root") for value in DEFAULT_ASCEND_ROOTS)

    seen_paths = set()
    version_candidates = {}
    for raw, source in roots:
        root = Path(raw).expanduser().absolute()
        # Only these known direct locations are checked. No glob/rglob, no
        # traversal through all CANN versions or arbitrary mount points.
        targets = [root]
        if (root / "latest").is_symlink() or (root / "latest").is_dir():
            targets.append(root / "latest")
        if root.name in ("aarch64-linux", "x86_64-linux", "opp"):
            targets.append(root.parent)
        for target in targets:
            key = str(target)
            if key not in seen_paths:
                seen_paths.add(key)
                report["ascend_paths"].append(_path_record(target, source))
            for filename in VERSION_NAMES:
                version_candidates.setdefault(str(target / filename), source)
                version_candidates.setdefault(str(target / "aarch64-linux" / filename), source)
                version_candidates.setdefault(str(target / "x86_64-linux" / filename), source)
            for prefix in (target / "vendors", target / "opp" / "vendors"):
                version_candidates.setdefault(str(prefix / "custom_cot" / "version.info"), source)
    version_candidates.update({value:"known_version_file" for value in DEFAULT_VERSION_FILES})
    for value in extra_version_files:
        version_candidates[str(Path(value).expanduser().absolute())] = "extra"
    explicit = {str(Path(value).expanduser().absolute()) for value in extra_version_files}
    seen_versions = set()
    for value, source in version_candidates.items():
        report["version_file_candidates_checked"].append(value)
        path = Path(value)
        if not path.is_file() and value not in explicit:
            continue
        record = _read_version(path, source)
        key = record["real_path"]
        if key not in seen_versions:
            seen_versions.add(key)
            report["version_files"].append(record)
    report["git_repositories"] = [_git_identity(root) for root in sorted(git_roots)]
    smi = shutil.which("npu-smi")
    report["npu_smi"] = dict(available=bool(smi), path=smi, status="NOT_AVAILABLE" if not smi else "STARTED")
    if smi:
        report["npu_smi"].update(_command([smi, "info"], timeout=10))
        record=report["npu_smi"]
        record["status"]="TIMEOUT" if record["timed_out"] else (
            "CAPTURED" if record["returncode"]==0 else "ERROR")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; never overwrite")
    parser.add_argument("--source-root", action="append", default=[], help="Existing source checkout; repeatable")
    parser.add_argument("--extra-version-file", action="append", default=[], help="Explicit version file; repeatable")
    args = parser.parse_args(argv)
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error("--output must be a new file; existing output will not be overwritten")
    report = collect_environment(args.source_root, args.extra_version_file)
    # Exclusive creation also closes the check/write race. Parent directories
    # must already exist; the only persistent mutation is this requested file.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=report["status"], compatibility=report["compatibility"],
                          output=str(output)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
