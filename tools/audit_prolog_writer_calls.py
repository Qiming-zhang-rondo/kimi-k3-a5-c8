#!/usr/bin/env python3
"""Audit LOCAL private Prolog call sites without copying or uploading source.

This is a lexical static gate, not a C++ parser, branch reachability check,
producer verification or device test. Comments/strings/definitions do not count.
Reports contain member receivers and locations only; arguments are never emitted.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re

try:
    from tools.prepare_prolog_direct import code_mask
except ModuleNotFoundError:
    from prepare_prolog_direct import code_mask

PUBLIC_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = frozenset((".h", ".hpp", ".hxx", ".inc", ".cpp", ".cc", ".cxx", ".asc"))
EXCLUDED_DIRS = frozenset((".git", ".svn", "__pycache__", "build", "build_out", "build_outputs",
    "CMakeFiles", "cmake-build-debug", "cmake-build-release", "sdk", "SDK", "third_party", "deps",
    "node_modules", "venv", ".venv", "ascend-toolkit", "ccec_compiler", "toolkit", "hcc"))
METHODS = ("Init", "ResolveRow", "WriteLatentSlice", "WriteRopeSlice", "PublishTags", "Release", "SetRingWriter")
MANDATORY = ("ResolveRow", "WriteLatentSlice", "WriteRopeSlice", "PublishTags")
MAX_SOURCE_BYTES = 256*1024*1024
MAX_FILES = 10000
IDENTIFIER = re.compile(r"[A-Za-z_]\w*\Z")
# The receiver is deliberately bounded to identifier/member/subscript chains.
# Ordinary method declarations/definitions use ::, not a . or -> member call.
RECEIVER = r"[A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*|\s*\[[^\[\];{}()]*\])*"
MEMBER_CALL = re.compile(r"(?P<receiver>"+RECEIVER+r")\s*(?:\.|->)\s*(?P<method>"+
                         "|".join(METHODS)+r")\s*\(")


def _without_directives(masked):
    result = []
    continued = False
    for line in masked.splitlines(keepends=True):
        directive = continued or line.lstrip().startswith("#")
        continued = directive and line.rstrip("\r\n").rstrip().endswith("\\")
        result.append("".join(c if c in "\r\n" else " " for c in line) if directive else line)
    return "".join(result)


def find_calls(text, filename):
    """Return locations and setter-inferred aliases, never source arguments."""
    masked = _without_directives(code_mask(text))
    calls, inferred = [], set()
    for match in MEMBER_CALL.finditer(masked):
        receiver = re.sub(r"\s+", "", match.group("receiver"))
        terminal = re.search(r"([A-Za-z_]\w*)(?:\[[^]]*\])?$",receiver)
        name = terminal.group(1) if terminal else receiver
        method = match.group("method")
        calls.append(dict(file=str(filename), line=masked.count("\n",0,match.start("method"))+1,
                          receiver=receiver, receiver_name=name, method=method))
        if method=="SetRingWriter":
            argument = re.match(r"\s*&\s*(?:this\s*->\s*)?([A-Za-z_]\w*)\s*(?:,|\))",masked[match.end():])
            if argument:inferred.add(argument.group(1))
    return calls,inferred


def summarize(calls, *, writer_names=(), inferred_names=(), files_scanned=0, source_bytes=0,
              skipped_files=0, preprocessed=False):
    explicit = tuple(writer_names)
    if any(not IDENTIFIER.fullmatch(name) for name in explicit):
        raise ValueError("--writer-name must be a simple C++ identifier; repeat for aliases")
    selected = set(explicit)
    inferred = set(inferred_names)
    # Only explicit aliases restrict mandatory unique methods. Inference cannot
    # establish aliases crossing SetRingWriter's argument and its private field.
    applicable = [call for call in calls if not selected or call["receiver_name"] in selected]
    methods = {method:[call for call in applicable if call["method"]==method] for method in MANDATORY}
    missing = [method for method in MANDATORY if not methods[method]]
    known_lifecycle = selected or inferred
    lifecycle = {}
    for method in ("Init","Release"):
        matches = [call for call in calls if call["method"]==method and call["receiver_name"] in known_lifecycle]
        lifecycle[method] = dict(status="TARGET_RECEIVER_CALLS_FOUND_REQUIRES_MANUAL_REVIEW" if matches else
            "REQUIRES_MANUAL_TARGET_RECEIVER", calls=matches)
    return dict(status="BLOCKED_MISSING_CALLS" if missing else "STATIC_CALLS_PRESENT_NOT_RUNTIME_PROOF",
        scope="local_lexical_call_audit_not_runtime_or_device_proof", source_mode="explicit_preprocessed" if preprocessed else "source_tree",
        files_scanned=files_scanned, source_bytes=source_bytes, skipped_files=skipped_files,
        writer_name_filter=sorted(selected), setter_inferred_writer_names=sorted(inferred),
        missing_mandatory_calls=missing, mandatory_calls=methods, lifecycle=lifecycle,
        setter_calls=[call for call in calls if call["method"]=="SetRingWriter"],
        manual_checks=["Calls may be in inactive branches or uninstantiated templates; this tool does not prove execution.",
            "Confirm the selected producer branch and pointer aliases; unrelated receivers are not one validated writer.",
            "Confirm exact prequant latent/post-RoPE sources, token/column offsets, ownership, synchronization and full ABI.",
            "Compile the actual variant and validate payload, tags, unchanged C4/query outputs and device replay."],
        uploads_performed=False, source_copied=False, device_launched=False)


def _source_files(root):
    for directory,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=sorted(name for name in dirs if name not in EXCLUDED_DIRS and
                       not (Path(directory)/name).is_symlink())
        for name in sorted(files):
            path=Path(directory)/name
            if path.suffix.lower() in SOURCE_SUFFIXES and path.name!="prolog_ring_write_device.h":
                yield path


def audit(source_root, *, writer_names=(), preprocessed=None):
    raw=Path(source_root).expanduser()
    if raw.is_symlink() or not raw.is_dir():raise ValueError("source-root must be a real bounded operator directory")
    root=raw.resolve()
    paths=_source_files(root)
    if preprocessed is not None:
        pp=Path(preprocessed).expanduser()
        if not pp.is_absolute():pp=root/pp
        if pp.is_symlink() or not pp.is_file():raise ValueError("--preprocessed must be an explicit regular local file")
        pp=pp.resolve()
        if not pp.is_relative_to(root):raise ValueError("--preprocessed must stay within source-root")
        paths=(pp,)
    calls,inferred=[],set()
    count,total,skipped=0,0,0
    for path in paths:
        if path.is_symlink() or not path.is_file():
            skipped+=1
            continue
        resolved=path.resolve()
        if not resolved.is_relative_to(root):raise ValueError("Source path escapes operator root")
        size=path.stat().st_size
        total+=size;count+=1
        if total>MAX_SOURCE_BYTES or count>MAX_FILES:
            raise ValueError("Operator source exceeds 256 MiB/10000 files; select a smaller operator subtree")
        text=path.read_text(encoding="utf-8",errors="strict")
        found,names=find_calls(text,path.relative_to(root).as_posix())
        calls.extend(found);inferred.update(names)
    if not count:raise ValueError("No local operator C++/ASC source files found")
    return summarize(calls,writer_names=writer_names,inferred_names=inferred,files_scanned=count,
                     source_bytes=total,skipped_files=skipped,preprocessed=preprocessed is not None)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path,help="NEW local output directory outside this public repository and source-root")
    parser.add_argument("--writer-name",action="append",default=[],help="Simple writer receiver identifier; repeat for pointer aliases")
    parser.add_argument("--preprocessed",type=Path,help="Optional actual compiler-produced file within source-root; overrides tree scan")
    args=parser.parse_args(argv)
    output=args.output.expanduser()
    try:
        if output.exists() or output.is_symlink():raise ValueError("Use a NEW output directory")
        root=args.source_root.expanduser().resolve()
        output=output.resolve()
        if output.is_relative_to(PUBLIC_ROOT) or output.is_relative_to(root):
            raise ValueError("Private audit output must stay outside the public repository and source-root")
        result=audit(args.source_root,writer_names=args.writer_name,preprocessed=args.preprocessed)
        output.mkdir(parents=True)
        (output/"writer_calls.json").write_text(json.dumps(result,indent=2)+"\n")
        print(result["status"])
        print(output/"writer_calls.json")
        return 1 if result["missing_mandatory_calls"] else 0
    except (ValueError,OSError,UnicodeError) as error:
        parser.exit(2,"Local audit error: "+str(error)+"\n")


if __name__=="__main__":raise SystemExit(main())
