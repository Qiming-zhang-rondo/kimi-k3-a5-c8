#!/usr/bin/env python3
"""Read installed source contracts without importing Torch or operator packages.

This is a bounded source inventory, not an execution test or permission to use a
mixed reader. A Python signature or exported symbol does not establish that a
deployed device kernel understands the benchmark's 400/656-byte records.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re


OPERATOR = re.compile(r"mixed_quant_sparse_flash_mla|MixedQuantSparseFlashMla|"
                      r"cloud_kv_quant_sparse_flash_attention|KvQuantSparseFlashAttention")
DETAIL = re.compile(r"quant_?mode|mxfp4|e8m0|group.?size|tile.?size|"
                    r"\b(?:400|656|544|320|608|584|258)\b", re.I)
MODE = re.compile(r"quant_?mode", re.I)
TEXT_SUFFIXES = {".py", ".h", ".hpp", ".cpp", ".cc", ".cxx", ".md", ".json"}
PACKAGES = {"cloud-ops-turbo": "cloud_ops_turbo",
            "cann-ops-transformer": "cann_ops_transformer",
            "cann-ops-transformer-dsl": "cann_ops_transformer_dsl"}
VERSION_KEYS = {"Version", "version_dir", "cloud_turbo_ops_version", "timestamp", "commitid"}
PRUNE = {".git", "__pycache__", ".cache", "node_modules"}


def discover_roots():
    """Distribution metadata locates files without executing package __init__.py."""
    roots, distributions = [], []
    for dist_name, package_name in PACKAGES.items():
        try:
            dist = importlib.metadata.distribution(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        distributions.append({"name": dist_name, "version": dist.version})
        roots.append(Path(dist.locate_file(package_name)))
    roots.extend((Path("/tmp/cloudops/pylib/cloud_ops_turbo"),
                  Path("/tmp/cloudops/cann_pkg/vendors/custom_cot")))
    for value in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep):
        if value:
            roots.append(Path(value))
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        if os.environ.get(name):
            home = Path(os.environ[name])
            roots.extend((home / "include", home / "opp/vendors", home / "version.info"))
    return [p for p in roots if p.exists()], distributions


def _source_record(path, raw, max_snippets):
    text = raw.decode("utf-8", errors="replace")
    if not OPERATOR.search(str(path)) and not OPERATOR.search(text):
        return None
    lines = text.splitlines()
    specific_path = bool(OPERATOR.search(str(path)))
    selected = set()
    for i, line in enumerate(lines):
        if OPERATOR.search(line) or (specific_path and DETAIL.search(line)):
            selected.update(range(max(0, i - 1), min(len(lines), i + 2)))
    snippets = [{"line": i + 1, "text": lines[i][:240]} for i in sorted(selected)[:max_snippets]]
    # Keep late validation checks visible even if imports exhaust the snippets.
    mode_lines = [i for i, line in enumerate(lines) if specific_path and MODE.search(line)]
    mode_indices = mode_lines if len(mode_lines) <= 8 else mode_lines[:4] + mode_lines[-4:]
    mode_evidence = [{"line": i + 1, "text": lines[i][:240]} for i in mode_indices]
    definitions = []
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and OPERATOR.search(node.name):
                    args = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
                    definitions.append({"name": node.name, "line": node.lineno, "parameters": args})
        except SyntaxError:
            pass
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "definitions": definitions,
            "snippets": snippets, "snippets_truncated": len(selected) > max_snippets,
            "quant_mode_text": mode_evidence, "quant_mode_text_truncated": len(mode_lines) > 8,
            "evidence_kind": "source_text_only"}


def inspect_paths(roots, *, max_files=5000, max_bytes=1_048_576, max_matches=30, max_snippets=18):
    if min(max_files, max_bytes, max_matches, max_snippets) <= 0:
        raise ValueError("Inventory limits must be positive")
    report = {"capability": "NOT_VERIFIED", "operator_packages_imported": False,
              "device_operations_executed": False,
              "required_contract": {"q_nope": 512, "q_rope": 64, "q_total": 576,
                                    "cold": "400 bytes: FP4 256 + BF16 RoPE 128 + E8M0/group32 16",
                                    "hot": "656 bytes: FP8 512 + BF16 RoPE 128 + FP32/group128 16"},
              "interpretation": "Source mentions/signatures do not prove binary support; no backend is enabled.",
              "documentation": {"canonical_upstream": "https://gitcode.com/cann/ops-transformer/tree/master/attention/mixed_quant_sparse_flash_mla",
                                "source_mirror": "https://github.com/hicann/ops-transformer/tree/master/attention/mixed_quant_sparse_flash_mla",
                                "note": "Current upstream/mirror documentation is not the installed binary's contract."},
              "roots": [], "versions": [], "source_matches": [], "binary_paths": [],
              "files_examined": 0, "truncated": False, "warnings": []}
    seen = set()
    for root in roots:
        root = Path(root).expanduser().absolute()
        report["roots"].append(str(root))
        if not root.exists():
            report["warnings"].append(f"Missing root: {root}")
            continue
        candidates = ((root.parent, [], [root.name]),) if root.is_file() else os.walk(root, followlinks=False)
        for directory, dirs, names in candidates:
            dirs[:] = sorted(d for d in dirs if d not in PRUNE)
            if len(Path(directory).relative_to(root if root.is_dir() else root.parent).parts) >= 12:
                if dirs:
                    report["truncated"] = True
                dirs[:] = []
            for name in sorted(names):
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                report["files_examined"] += 1
                if report["files_examined"] > max_files:
                    report["truncated"] = True
                    return report
                if ".so" in path.name and OPERATOR.search(str(path)):
                    if len(report["binary_paths"]) < max_matches:
                        report["binary_paths"].append(str(path))
                    else:
                        report["truncated"] = True
                if path.suffix not in TEXT_SUFFIXES and path.name != "version.info":
                    continue
                try:
                    if path.stat().st_size > max_bytes:
                        if OPERATOR.search(str(path)):
                            report["warnings"].append(f"Skipped oversized source: {path}")
                            report["truncated"] = True
                        continue
                    raw = path.read_bytes()
                except OSError as exc:
                    report["warnings"].append(f"Unreadable file: {path}: {type(exc).__name__}")
                    continue
                if path.name == "version.info":
                    values = {}
                    for line in raw.decode("utf-8", errors="replace").splitlines():
                        key, sep, value = line.partition("=")
                        if sep and key.strip() in VERSION_KEYS:
                            values[key.strip()] = value.strip()[:160]
                    report["versions"].append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                                               "fields": values})
                match = _source_record(path, raw, max_snippets)
                if match:
                    if len(report["source_matches"]) >= max_matches:
                        report["truncated"] = True
                        return report
                    report["source_matches"].append(match)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path,
                        help="Inspect this directory/file; repeatable. Explicit roots disable auto-discovery.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; otherwise prints JSON")
    parser.add_argument("--max-files", type=int, default=5000)
    parser.add_argument("--max-matches", type=int, default=30)
    args = parser.parse_args(argv)
    roots, distributions = (args.root, []) if args.root else discover_roots()
    report = inspect_paths(roots, max_files=args.max_files, max_matches=args.max_matches)
    report["distributions"] = distributions
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        print(f"NOT_VERIFIED: source inventory written to {args.output}")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
