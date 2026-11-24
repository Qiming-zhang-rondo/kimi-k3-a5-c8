#!/usr/bin/env python3
"""Copy bounded build/header evidence without executing commands or importing SDKs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex


HEADERS = {"kernel_macros.h", "kernel_operator.h", "kernel_basic_intf.h", "kernel_reg_compute_utils.h"}
HITS = re.compile(r"MicroAPI|__NPU_ARCH__|__CCE_AICORE__|__DAV_C310|ASCENDC_BEGIN_NAMESPACE|namespace\s+AscendC")


def inside(path, root):
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def bounded_files(root, notes, max_depth=8, max_dirs=1024, max_files=20000):
    directories = count = 0
    for current, dirs, files in os.walk(root, followlinks=False, onerror=lambda error: notes.append(str(error))):
        directories += 1
        current = Path(current)
        dirs[:] = sorted(name for name in dirs if not (current / name).is_symlink())
        if len(current.relative_to(root).parts) >= max_depth:
            dirs[:] = []
        for name in sorted(files):
            count += 1
            if count > max_files:
                notes.append("File search limit reached: " + str(root))
                return
            yield current / name
        if directories >= max_dirs:
            notes.append("Directory search limit reached: " + str(root))
            return


def collect(build_dir, output, sdk_root=None):
    build = Path(build_dir).expanduser().resolve()
    sdk = Path(sdk_root).expanduser().resolve() if sdk_root else None
    destination = Path(output).expanduser().resolve()
    for source in (build, sdk):
        if source is not None:
            if source == Path(source.anchor):
                raise ValueError("Filesystem root is not an allowed source directory")
            if inside(destination, source):
                raise ValueError("Output must be outside source directories")
    destination.mkdir(parents=True, exist_ok=False)
    report = {"build_dir": str(build), "sdk_root": str(sdk) if sdk else None,
              "scope": "Read-only file evidence; no compiler, SDK, or NPU execution; no environment inspection",
              "files": [], "notes": [], "compile_commands": [], "cmake_compiler_flags": {},
              "missing_headers": sorted(HEADERS), "sdk_status": "unknown: --sdk-root not supplied" if sdk is None else "inspected"}

    def copy_file(path, root, kind):
        if not path.is_file():
            report["notes"].append("Missing: " + str(path))
            return None
        if not inside(path, root):
            report["notes"].append("Rejected symlink escape: " + str(path))
            return None
        try:
            if path.stat().st_size > 16 * 1024 * 1024:
                report["notes"].append("Skipped file larger than 16 MiB: " + str(path))
                return None
            data = path.read_bytes()
        except OSError as error:
            report["notes"].append(str(error))
            return None
        copied = destination / kind / path.relative_to(root)
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(data)
        item = {"source": str(path), "resolved_source": str(path.resolve()), "copy": str(copied),
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "kind": kind}
        if kind == "sdk":
            lines = data.decode("utf-8", errors="replace").splitlines()
            hits = [index for index, line in enumerate(lines) if HITS.search(line)]
            item["context_matches"] = [{"line": index + 1, "context": [
                {"line": other + 1, "text": lines[other]} for other in range(max(0, index - 2), min(len(lines), index + 3))
            ]} for index in hits[:128]]
            item["total_match_count"] = len(hits)
        report["files"].append(item)
        return data.decode("utf-8", errors="replace")

    commands = copy_file(build / "compile_commands.json", build, "build")
    if commands is not None:
        try:
            entries = json.loads(commands)
            if not isinstance(entries, list):
                raise ValueError("compile_commands.json must contain a list")
            for entry in entries:
                record = {"file": entry.get("file"), "directory": entry.get("directory")}
                try:
                    argv = entry.get("arguments")
                    if argv is None:
                        argv = shlex.split(entry["command"], posix=True)
                    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
                        raise ValueError("arguments is not a string list")
                    record.update(argv=argv, first_command=argv[0] if argv else None,
                                  compiler_resolution="unknown; argv is evidence, no executable was resolved",
                                  flag_tokens=[value for value in argv if value.startswith("-")])
                except (KeyError, ValueError, TypeError) as error:
                    record["parse_error"] = str(error)
                report["compile_commands"].append(record)
        except (ValueError, TypeError, AttributeError) as error:
            report["notes"].append("compile_commands parse error: " + str(error))
    cache = copy_file(build / "CMakeCache.txt", build, "build")
    if cache is not None:
        for line in cache.splitlines():
            match = re.match(r"(CMAKE_[A-Za-z0-9_]*(?:COMPILER|FLAGS)[A-Za-z0-9_]*):[^=]*=(.*)", line)
            if match:
                report["cmake_compiler_flags"][match[1]] = match[2]
    cmake = build / "CMakeFiles"
    if cmake.is_dir() and inside(cmake, build):
        for path in bounded_files(cmake, report["notes"], max_depth=5, max_dirs=128, max_files=5000):
            if path.suffix.lower() == ".cmake" and "compiler" in path.name.lower():
                copy_file(path, build, "build")
    if sdk is not None:
        for path in bounded_files(sdk, report["notes"]):
            if path.name in HEADERS:
                if copy_file(path, sdk, "sdk") is not None and path.name in report["missing_headers"]:
                    report["missing_headers"].remove(path.name)
    report["status"] = "EVIDENCE_COLLECTED"
    report["compiler_cmake_found"] = any(item["kind"] == "build" and Path(item["source"]).suffix.lower() == ".cmake"
                                         for item in report["files"])
    if not report["compiler_cmake_found"]:
        report["notes"].append("No compiler*.cmake collected from CMakeFiles")
    report["complete"] = commands is not None and cache is not None and report["compiler_cmake_found"] and sdk is not None and not report["missing_headers"]
    (destination / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output", required=True, help="New directory, outside source roots")
    parser.add_argument("--sdk-root", help="Only this explicit SDK root is searched for the four headers")
    args = parser.parse_args(argv)
    report = collect(args.build_dir, args.output, args.sdk_root)
    print("EVIDENCE_COLLECTED: " + str(Path(args.output).resolve() / "report.json"))
    print("Evidence complete: " + str(report["complete"]) + "; no compilation was performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
