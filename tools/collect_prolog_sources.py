#!/usr/bin/env python3
"""Collect exact MLA Prolog source bytes without importing or running them.

Run against actual source directories, not a service process. Binary artifacts
are inventoried and hashed only; they are never copied. No framework, NPU,
environment dump, package installation or service mutation is involved.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat


VERSION = "mla-prolog-source-collection-v1"
PART_BYTES = 32 * 1024
SOURCE_SUFFIXES = {".py", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".asc", ".cce", ".c", ".cmake", ".sh"}
BINARY_SUFFIXES = {".so", ".a", ".o", ".bin", ".dylib", ".dll", ".pyc", ".run"}
SKIP_DIRECTORIES = {"node_modules", "proc", "sys", "dev"}
SKIP_DETAIL_LIMIT = 256


class ReadBudgetExceeded(ValueError):
    pass


def _open_regular(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    stream = os.fdopen(descriptor, "rb")
    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
        stream.close()
        raise OSError("not a regular file")
    return stream


def _source_bytes(path, remaining):
    with _open_regular(path) as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > remaining:
            raise ReadBudgetExceeded("source_bytes")
        content = stream.read(remaining + 1)
        after = os.fstat(stream.fileno())
    if len(content) > remaining:
        raise ReadBudgetExceeded("source_bytes")
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError("source changed during read")
    return content


def _binary_hash(path, remaining):
    digest, count = hashlib.sha256(), 0
    with _open_regular(path) as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > remaining:
            raise ReadBudgetExceeded("binary_hash_bytes")
        while True:
            chunk = stream.read(min(65536, remaining - count + 1))
            if not chunk:
                break
            count += len(chunk)
            if count > remaining:
                raise ReadBudgetExceeded("binary_hash_bytes")
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError("binary changed during hash")
    return digest.hexdigest(), count


def _store_source(output, number, name, content):
    relative = Path("sources") / (f"{number:04d}_" + name)
    try:
        content.decode("utf-8")
        utf8 = True
    except UnicodeDecodeError:
        utf8 = False
    if utf8 and len(content) <= PART_BYTES:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return dict(encoding="raw_utf8", path=relative.as_posix(), part_bytes_limit=PART_BYTES)
    packed = gzip.compress(content, mtime=0)
    encoded = base64.b64encode(packed)
    folder = Path("parts") / f"{number:04d}"
    (output / folder).mkdir(parents=True)
    parts = []
    for offset in range(0, len(encoded), PART_BYTES):
        relative = folder / f"{len(parts):04d}.b64"
        block = encoded[offset:offset + PART_BYTES]
        (output / relative).write_bytes(block)
        parts.append(dict(path=relative.as_posix(), bytes=len(block),
                          sha256=hashlib.sha256(block).hexdigest()))
    return dict(encoding="gzip+base64", original_utf8=utf8, parts=parts,
        compressed_bytes=len(packed), compressed_sha256=hashlib.sha256(packed).hexdigest(),
        part_bytes_limit=PART_BYTES,
        restore="concatenate parts in listed order, base64-decode, then gzip-decompress")


def collect_sources(roots, output, *, max_walk_files=20000,
                    max_source_bytes=64 * 1024 * 1024, max_matches=256,
                    max_binary_hash_bytes=256 * 1024 * 1024):
    """Create a new bounded evidence directory; never overwrite existing data."""
    for name, value in (("max_walk_files", max_walk_files), ("max_source_bytes", max_source_bytes),
                        ("max_matches", max_matches), ("max_binary_hash_bytes", max_binary_hash_bytes)):
        if type(value) is not int or value <= 0:
            raise ValueError(name + " must be a positive integer")
    roots = [Path(root).expanduser().absolute() for root in roots]
    if not roots:
        raise ValueError("at least one actual source root is required")
    output = Path(output).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(version=VERSION, captured_at_utc=datetime.now(timezone.utc).isoformat(),
        scope="read-only named-source scan; no framework imports, NPU execution or service changes",
        filename_filter="case-insensitive filename or component directory containing mla_prolog",
        roots=[str(root) for root in roots], source_suffixes=sorted(SOURCE_SUFFIXES),
        exclusions="hidden entries, node_modules/proc/sys/dev directories, all symlinks and output tree",
        budgets=dict(max_walk_files=max_walk_files, max_walk_entries=max_walk_files * 4,
            max_source_bytes=max_source_bytes, max_matches=max_matches,
            max_binary_hash_bytes=max_binary_hash_bytes, part_bytes_limit=PART_BYTES),
        sources=[], binaries=[], skipped=[], skipped_reasons={}, scan_complete_within_scope=True,
        walk_files=0, walk_entries=0, source_bytes=0, binary_hash_bytes=0)
    skipped = Counter()
    kinds = Counter()
    seen = set()
    matches = 0
    stop = False

    def skip(path, reason, *, incomplete=False):
        skipped[reason] += 1
        if len(report["skipped"]) < SKIP_DETAIL_LIMIT:
            report["skipped"].append(dict(path=str(path), reason=reason))
        if incomplete:
            report["scan_complete_within_scope"] = False

    for root_index, root in enumerate(roots):
        if stop:
            skip(root, "unscanned_after_budget", incomplete=True)
            continue
        if root.is_symlink():
            skip(root, "root_symlink", incomplete=True)
            continue
        if not root.is_dir():
            skip(root, "missing_or_non_directory_root", incomplete=True)
            continue
        stack = [root]
        while stack and not stop:
            directory = stack.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        path = Path(entry.path)
                        report["walk_entries"] += 1
                        if report["walk_entries"] > max_walk_files * 4:
                            skip(path, "walk_entry_budget", incomplete=True)
                            stop = True
                            break
                        if entry.is_symlink():
                            skip(path, "symlink")
                            continue
                        if entry.name.startswith("."):
                            skip(path, "hidden_entry")
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name in SKIP_DIRECTORIES or path == output:
                                skip(path, "excluded_directory")
                            else:
                                stack.append(path)
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            skip(path, "non_regular_file")
                            continue
                        report["walk_files"] += 1
                        if report["walk_files"] > max_walk_files:
                            skip(path, "walk_file_budget", incomplete=True)
                            stop = True
                            break
                        relative_path = path.relative_to(root)
                        component_match = "mla_prolog" in root.name.lower() or any(
                            "mla_prolog" in component.lower() for component in relative_path.parts[:-1])
                        if "mla_prolog" not in entry.name.lower() and not component_match:
                            continue
                        identity = str(path)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        matches += 1
                        if matches > max_matches:
                            skip(path, "match_budget", incomplete=True)
                            stop = True
                            break
                        suffix = path.suffix.lower()
                        base = dict(source_path=str(path), root_index=root_index,
                                    relative_path=str(relative_path), suffix=suffix)
                        try:
                            if suffix in SOURCE_SUFFIXES or (component_match and entry.name == "CMakeLists.txt"):
                                content = _source_bytes(path, max_source_bytes - report["source_bytes"])
                                item = dict(base, kind="source", source_kind=suffix.lstrip("."),
                                    bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
                                item["storage"] = _store_source(output, len(report["sources"]), entry.name, content)
                                report["sources"].append(item)
                                report["source_bytes"] += len(content)
                                kinds[item["source_kind"]] += 1
                            elif suffix in BINARY_SUFFIXES or ".so." in entry.name.lower():
                                item = dict(base, kind="binary", copied=False, bytes=entry.stat(follow_symlinks=False).st_size,
                                            sha256=None, hash_status="NOT_HASHED")
                                report["binaries"].append(item)
                                digest, count = _binary_hash(path, max_binary_hash_bytes - report["binary_hash_bytes"])
                                item.update(sha256=digest, bytes=count, hash_status="HASHED")
                                report["binary_hash_bytes"] += count
                            else:
                                skip(path, "unsupported_extension")
                        except ReadBudgetExceeded as error:
                            skip(path, str(error) + "_budget", incomplete=True)
                        except OSError as error:
                            skip(path, "read_or_store_error: " + str(error), incomplete=True)
            except OSError as error:
                skip(directory, "directory_error: " + str(error), incomplete=True)

    report.update(source_found=bool(report["sources"]), source_count=len(report["sources"]),
        binary_count=len(report["binaries"]), source_kinds=dict(kinds), skipped_reasons=dict(skipped),
        skipped_details_truncated=sum(skipped.values()) > len(report["skipped"]))
    report["status"] = ("SOURCES_COLLECTED" if report["scan_complete_within_scope"] else "PARTIAL_SOURCES_COLLECTED") \
        if report["source_found"] else "NO_SOURCES_FOUND"
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True,
                        help="Actual source directory; repeat for multiple roots; symlinks are not followed")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--max-walk-files", type=int, default=20000)
    parser.add_argument("--max-source-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-matches", type=int, default=256)
    parser.add_argument("--max-binary-hash-bytes", type=int, default=256 * 1024 * 1024)
    args = parser.parse_args()
    try:
        report = collect_sources(args.root, args.output, max_walk_files=args.max_walk_files,
            max_source_bytes=args.max_source_bytes, max_matches=args.max_matches,
            max_binary_hash_bytes=args.max_binary_hash_bytes)
    except (ValueError, OSError) as error:
        parser.exit(2, "Collection refused: " + str(error) + "\n")
    print(json.dumps(dict(status=report["status"], source_count=report["source_count"],
        binary_count=report["binary_count"], scan_complete_within_scope=report["scan_complete_within_scope"],
        manifest=str(args.output / "manifest.json"))))
    return 0 if report["source_found"] and report["scan_complete_within_scope"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
