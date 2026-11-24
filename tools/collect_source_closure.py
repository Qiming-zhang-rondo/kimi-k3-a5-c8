#!/usr/bin/env python3
"""Collect project-source evidence from explicit narrow directories only.

This is a lexical include collector, not a C/C++ preprocessor or build test.
All literal conditional branches are recorded. Unresolved SDK/system includes
remain unresolved; naming heuristics never substitute a file or a type body.
Source files are read only. A new output directory is required on every run.
"""

import argparse
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import re
import sys


SEEDS = (
    "vf_basic_block_aligned128_no_update_sfa.h",
    "vf_basic_block_aligned128_update_sfa.h",
    "vf_basic_block_unaligned64_update_sfa.h",
    "vf_basic_block_unaligned64_no_update_sfa.h",
    "vf_basic_block_unaligned128_no_update_sfa.h",
    "vf_basic_block_unaligned128_update_sfa.h",
    "gm_layout.h", "parser.h", "fa_gm_tensor.h", "fa_l1_tensor.h", "gm_coord.h",
)
TYPE = "SparseFlashAttentionTilingDataMla"
EXTENSIONS = {".h", ".hh", ".hpp", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".cuh", ".inc", ".inl", ".ipp", ".tpp", ".def"}
LEXEMES = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"\n]+)[>"]', re.M)
SDK_PREFIXES = ("acl/", "aclnn/", "torch/", "torch_npu/", "ATen/", "c10/", "graph/", "register/",
                "kernel_tiling/", "lib/", "runtime/", "tiling/", "exe_graph/", "ascendc/")
SDK_NAMES = {"kernel_operator.h", "kernel_operator_list_tensor_intf.h", "securec.h", "tiling_api.h"}


def stripped(text, strings=False):
    def replace(match):
        value = match.group()
        if value.startswith(("//", "/*")) or strings:
            return "".join("\n" if c == "\n" else " " for c in value)
        return value
    return LEXEMES.sub(replace, text)


def type_definitions(text):
    clean = stripped(text, strings=True)
    patterns = (
        ("struct_or_class_body", rf'\b(?:struct|class)\s+{TYPE}\b[^;{{}}]*\{{'),
        ("tiling_definition_macro_body", rf'\bBEGIN_TILING_DATA_DEF\s*\(\s*{TYPE}\s*\)[\s\S]*?\bEND_TILING_DATA_DEF\b'),
        ("typedef_struct_body", rf'\btypedef\s+struct(?:\s+\w+)?\s*\{{[\s\S]*?\}}\s*{TYPE}\s*;'),
    )
    return [dict(pattern=kind, line=clean.count("\n", 0, match.start())+1)
            for kind, pattern in patterns for match in re.finditer(pattern, clean)]


def within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def classify_include(name, angled):
    if angled:
        return "system_or_sdk_candidate"
    if name in SDK_NAMES or name.startswith(SDK_PREFIXES):
        return "sdk_candidate_by_name"
    return "project_candidate_by_name"


def collect(source_roots, output, bundle_root=None, *, required_seeds=SEEDS):
    bundle_root = Path(bundle_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    roots = [("bundle", bundle_root)] + [("source", Path(p).expanduser().resolve()) for p in source_roots]
    if output.exists():
        raise ValueError("Output already exists; choose a fresh directory (nothing was overwritten)")
    if any(within(output, path) for _, path in roots):
        raise ValueError("Output must be outside every input root")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(version=1, type_name=TYPE, root_mappings=[], records=[], required_seeds=[],
                  definition_matches=[], definition_resolution=None, includes=[], resolved=[], unresolved=[],
                  errors=[], skipped_symlinks=[], build_validated=False,
                  limitations=["Lexical collection only: conditional include branches are all recorded, not evaluated.",
                               "Include search order is unknown; differing candidate bytes remain ambiguous.",
                               "SDK/system classification is a naming heuristic, never a successful resolution.",
                               "Type recognition accepts explicit class/struct bodies, complete BEGIN/END tiling macros, and typedef struct bodies.",
                               "This bundle contains source evidence; successful collection does not prove buildability or version compatibility."])
    inventory, basename_index, snapshots = {}, defaultdict(list), {}
    initial = []
    broad = {Path(p).resolve() for p in ("/", "/usr", "/usr/local", "/opt", "/mnt", "/tmp", "/home", "/Users")}
    broad.add(Path.home().resolve())

    def error(operation, path, failure):
        report["errors"].append(dict(operation=operation, path=str(path), error=str(failure)))

    for number, (kind, root) in enumerate(roots):
        root_id = f"root{number}"
        entry = dict(root_id=root_id, kind=kind, path=str(root), scanned_files=0)
        report["root_mappings"].append(entry)
        if not root.is_dir() or root in broad:
            error("root", root, "Source root must be an existing narrow package/project directory")
            continue
        # Do not collect host-only test stubs or scripts as device dependencies.
        starts = [root/name for name in ("original", "deps", "arch35")] if kind == "bundle" else [root]
        for start in starts:
            if not start.is_dir():
                continue
            for directory, dirs, files in os.walk(start, followlinks=False,
                                                   onerror=lambda e: error("walk", root, e)):
                dirs[:] = sorted(d for d in dirs if d not in (".git", ".hg", ".svn", "__pycache__")
                                 and not (Path(directory)/d).is_symlink())
                for name in sorted(files):
                    path = Path(directory)/name
                    if path.suffix.lower() not in EXTENSIONS:
                        continue
                    if not within(path.resolve(), root):
                        report["skipped_symlinks"].append(str(path))
                        continue
                    relative = path.relative_to(root).as_posix()
                    identity = root_id+":"+relative
                    if identity in inventory:
                        continue
                    record = dict(id=identity, root_id=root_id, relative_path=relative, source_path=str(path))
                    inventory[identity] = record
                    basename_index[name].append(identity)
                    entry["scanned_files"] += 1
                    if kind == "bundle":
                        initial.append(identity)

    def snapshot(identity):
        if identity not in snapshots:
            record = inventory[identity]
            try:
                content = Path(record["source_path"]).read_bytes()
            except OSError as failure:
                error("read", record["source_path"], failure)
                snapshots[identity] = None
            else:
                snapshots[identity] = (content, hashlib.sha256(content).hexdigest())
        return snapshots[identity]

    def resolve_candidates(ids):
        candidates = []
        for identity in sorted(set(ids)):
            value = snapshot(identity)
            if value is not None:
                record = inventory[identity]
                candidates.append(dict(id=identity, source_path=record["source_path"], sha256=value[1]))
        hashes = {candidate["sha256"] for candidate in candidates}
        status = ("missing" if not candidates else "ambiguous" if len(hashes) > 1 else
                  "resolved_identical_candidates" if len(candidates) > 1 else "resolved")
        return dict(status=status, candidates=candidates)

    pending = deque(initial)
    reasons = defaultdict(set)
    for identity in initial:
        reasons[identity].add("provided_bundle_source")
    for name in required_seeds:
        resolution = resolve_candidates(basename_index.get(name, []))
        report["required_seeds"].append(dict(name=name, **resolution))
        for item in resolution["candidates"]:
            pending.append(item["id"])
            reasons[item["id"]].add("required_seed:"+name)

    definition_ids = []
    # Definitions may be in generated build directories, not just op_kernel.
    for identity, record in inventory.items():
        value = snapshot(identity)
        if value is None:
            continue
        definitions = type_definitions(value[0].decode("utf-8", errors="replace"))
        if definitions:
            definition_ids.append(identity)
            report["definition_matches"].append(dict(id=identity, source_path=record["source_path"],
                                                     sha256=value[1], matches=definitions))
            pending.append(identity)
            reasons[identity].add("actual_tiling_type_definition")
    report["definition_resolution"] = resolve_candidates(definition_ids)

    visited = set()
    while pending:
        identity = pending.popleft()
        if identity in visited:
            continue
        visited.add(identity)
        record, value = inventory[identity], snapshot(identity)
        if value is None:
            continue
        content, digest = value
        destination = Path("collected")/record["root_id"]/record["relative_path"]
        try:
            (output/destination).parent.mkdir(parents=True, exist_ok=True)
            with (output/destination).open("xb") as stream:
                stream.write(content)
        except OSError as failure:
            error("copy", output/destination, failure)
            continue
        report["records"].append(dict(record, collected_path=destination.as_posix(), sha256=digest,
                                      bytes=len(content)))
        text = stripped(content.decode("utf-8", errors="replace"))
        for match in INCLUDE.finditer(text):
            angled, name = match.group(1) == "<", match.group(2).strip()
            candidates = []
            local = (Path(record["source_path"]).parent/name).resolve()
            # A qualified include cannot be satisfied by an unrelated basename
            # (e.g. arch22/foo.h must not silently resolve to arch35/foo.h).
            for candidate in basename_index.get(Path(name).name, []):
                target = inventory[candidate]
                target_path = Path(target["source_path"])
                if (target_path.resolve() == local or "/" not in name or
                        target_path.as_posix().endswith("/"+name.lstrip("./"))):
                    candidates.append(candidate)
            resolution = resolve_candidates(candidates)
            edge = dict(from_id=identity, line=text.count("\n", 0, match.start())+1,
                        include=name, delimiter="angle" if angled else "quote",
                        classification=classify_include(name, angled), **resolution)
            report["includes"].append(edge)
            report["resolved" if resolution["status"].startswith("resolved") else "unresolved"].append(edge)
            for item in resolution["candidates"]:
                pending.append(item["id"])
                reasons[item["id"]].add("include_from:"+identity)

    for record in report["records"]:
        record["reasons"] = sorted(reasons[record["id"]])
    required_ok = all(item["status"].startswith("resolved") for item in report["required_seeds"])
    required_ok &= report["definition_resolution"]["status"].startswith("resolved")
    ambiguous = [edge for edge in report["includes"] if edge["status"] == "ambiguous"]
    report["required_sources_resolved"] = bool(required_ok)
    report["lexical_includes_all_resolved"] = not report["unresolved"]
    report["exit_code"] = 1 if not required_ok or report["errors"] or ambiguous else 0
    report["status"] = ("PARTIAL_OR_AMBIGUOUS" if report["exit_code"] else
                        "REQUIRED_FOUND_WITH_OPEN_INCLUDES" if report["unresolved"] else "COLLECTED_NOT_COMPILED")
    report["summary"] = dict(collected_files=len(report["records"]), actual_definition_files=len(definition_ids),
                              missing_required=[item["name"] for item in report["required_seeds"] if item["status"] == "missing"],
                              ambiguous_required=[item["name"] for item in report["required_seeds"] if item["status"] == "ambiguous"],
                              unresolved_includes=len(report["unresolved"]), ambiguous_includes=len(ambiguous),
                              errors=len(report["errors"]))
    (output/"source_closure.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, action="append", required=True,
                        help="Repeat for explicit narrow package/project directories; no automatic system search")
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory outside input roots")
    args = parser.parse_args(argv)
    try:
        report = collect(args.source_root, args.output, args.bundle_root)
    except (ValueError, OSError) as failure:
        parser.error(str(failure))
    print(json.dumps(dict(status=report["status"], manifest=str(args.output.resolve()/"source_closure.json"),
                          **report["summary"]), ensure_ascii=False))
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
