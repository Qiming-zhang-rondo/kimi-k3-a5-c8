#!/usr/bin/env python3
"""Inspect matched SDK symbol evidence before an isolated Prolog ASC build.

No NPU, Torch, vendor writes, guessed definitions or compiler target execution.
Only the selected SDK's bisheng --version runs. Generated preprocessing commands
are NOT executed and must be checked against the real ASC compile command.
Header declarations alone never establish a resolved compiler/enum value.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import traceback

SYMBOLS = ("ASC_DEVKIT_VERSION_NUM", "ASCENDC_TPL_MIX_AIC_1_1", "KERNEL_TYPE_MIX_AIC_1_1",
           "KERNEL_TYPE_MIX_AIC_1_2", "KERNEL_TYPE_AIV_ONLY", "KERNEL_TYPE_AIC_ONLY")
SUFFIXES = {".h", ".hpp", ".inc", ".def", ".cpp", ".cxx", ".cc", ".c", ".cce", ".asc", ".cmake"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
MAX_FILES = 30000
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024


def _inside(path, root):
    return path == root or root in path.parents


def _scan(roots, boundary, *, headers_only=False):
    files=[];seen=set();skipped=[];total=0;complete=True
    for initial in roots:
        if not initial.is_dir():continue
        resolved=initial.resolve()
        if not _inside(resolved,boundary):
            skipped.append(dict(path=str(initial),reason="outside_selected_root"));complete=False;continue
        for current,dirs,names in os.walk(resolved,followlinks=False):
            base=Path(current)
            dirs[:]=sorted(name for name in dirs if name not in SKIP_DIRS and not name.startswith("."))
            retained=[]
            for name in dirs:
                candidate=base/name
                if candidate.is_symlink():
                    skipped.append(dict(path=str(candidate),reason="directory_symlink_not_followed"));complete=False
                else:retained.append(name)
            dirs[:]=retained
            for name in sorted(names):
                path=base/name
                if path.suffix.lower() not in SUFFIXES and name!="CMakeLists.txt":continue
                if headers_only and path.suffix.lower() not in {".h",".hpp",".inc",".def"}:continue
                actual=path.resolve()
                if not _inside(actual,boundary):
                    skipped.append(dict(path=str(path),reason="external_file_symlink"));complete=False;continue
                if actual in seen:continue
                seen.add(actual)
                if len(files)>=MAX_FILES:
                    return files,dict(complete=False,reason="file_budget",files=len(files),bytes=total,skipped=skipped)
                size=actual.stat().st_size
                if size>MAX_FILE_BYTES:
                    skipped.append(dict(path=str(actual),reason="file_byte_budget",bytes=size));complete=False;continue
                if total+size>MAX_TOTAL_BYTES:
                    return files,dict(complete=False,reason="total_byte_budget",files=len(files),bytes=total,skipped=skipped)
                data=actual.read_bytes();total+=len(data)
                files.append((actual,data.decode("utf-8",errors="replace"),hashlib.sha256(data).hexdigest()))
    return files,dict(complete=complete,files=len(files),bytes=total,skipped=skipped)


def _select_compiler(sdk, explicit):
    if explicit is not None:
        candidates=[Path(explicit).expanduser()]
    else:
        candidates=[sdk/relative for relative in (
            "aarch64-linux/ccec_compiler/bin/bisheng", "aarch64-linux/asc/bin/bisheng",
            "compiler/ccec_compiler/bin/bisheng", "ccec_compiler/bin/bisheng", "asc/bin/bisheng")]
    for candidate in candidates:
        if candidate.is_file():
            actual=candidate.resolve()
            if not _inside(actual,sdk) or actual.name!="bisheng":
                raise ValueError("Compiler must resolve to bisheng within the explicitly selected SDK")
            if not os.access(actual,os.X_OK):raise ValueError("Selected SDK bisheng is not executable")
            return actual
    if explicit is not None:raise ValueError("Explicit compiler was not found in the selected SDK")
    return None


def inspect_sdk(sdk_root,source_root,compiler=None):
    sdk=Path(sdk_root).expanduser().resolve();source=Path(source_root).expanduser().resolve()
    if not sdk.is_dir() or not source.is_dir():raise ValueError("--sdk-root and --source-root must name existing directories")
    source_files,source_scan=_scan([source],source)
    include_candidates=[sdk/relative for relative in (
        "aarch64-linux/asc/include", "aarch64-linux/asc/impl", "aarch64-linux/include", "include",
        "aarch64-linux/ccec_compiler/include", "compiler/ccec_compiler/include", "asc/include")]
    include_roots=[path.resolve() for path in include_candidates if path.is_dir() and _inside(path.resolve(),sdk)]
    sdk_files,sdk_scan=_scan(include_roots,sdk,headers_only=True)
    evidence={name:dict(source_uses=[],header_evidence=[],resolved=False,value=None) for name in SYMBOLS}
    for path,text,digest in source_files:
        for number,line in enumerate(text.splitlines(),1):
            if line.lstrip().startswith(("//","*")):continue
            for name in SYMBOLS:
                if re.search(r"\b"+name+r"\b",line):
                    evidence[name]["source_uses"].append(dict(file=str(path),line=number,text=line.strip()[:1000],
                        sha256=digest,conditional=bool(re.match(r"\s*#\s*(?:if|elif|ifdef|ifndef)\b",line))))
    for path,text,digest in sdk_files:
        for number,line in enumerate(text.splitlines(),1):
            for name in SYMBOLS:
                define=re.match(r"\s*#\s*(define|undef)\s+"+name+r"\b(.*)",line)
                declaration=re.search(r"\b"+name+r"\s*=\s*([^,;}]+)",line)
                enum_declaration=re.match(r"\s*"+name+r"\s*[,}]",line)
                if define:
                    kind=define.group(1);expression=define.group(2).strip()
                elif declaration:
                    kind="declaration";expression=declaration.group(1).strip()
                elif enum_declaration:
                    kind="implicit_enum_declaration";expression=None
                else:continue
                evidence[name]["header_evidence"].append(dict(file=str(path),line=number,kind=kind,
                    expression=expression,sha256=digest,compiler_selection="NOT_PREPROCESSED"))
    required=[];missing=[]
    for name,item in evidence.items():
        positive=[entry for entry in item["header_evidence"] if entry["kind"]!="undef"]
        item["status"]="FOUND_IN_HEADER_NOT_RESOLVED" if positive else "NOT_FOUND_IN_SCANNED_HEADERS"
        if item["source_uses"]:
            required.append(name)
            if not positive:missing.append(name)
    selected=_select_compiler(sdk,compiler)
    version=dict(status="NOT_AVAILABLE",path=None)
    if selected:
        version["path"]=str(selected)
        try:
            result=subprocess.run([str(selected),"--version"],capture_output=True,text=True,timeout=10)
            version.update(status="READ" if result.returncode==0 else "ERROR",exit_code=result.returncode,
                stdout=result.stdout[:16384],stderr=result.stderr[:16384])
        except (OSError,subprocess.TimeoutExpired) as error:
            version.update(status="ERROR",error=str(error))
    version_headers=sorted(set(entry["file"] for entry in evidence["ASC_DEVKIT_VERSION_NUM"]["header_evidence"]
        if entry["kind"]=="define"))
    report=dict(version="prolog-sdk-symbol-check-v1",status="SDK_SYMBOL_EVIDENCE_CAPTURED",
        compatibility="NOT_VERIFIED",sdk_root=str(sdk),source_root=str(source),
        scope="bounded read-only source/header evidence plus matched bisheng --version",
        compiler=version,source_scan=source_scan,sdk_scan=sdk_scan,include_roots=[str(path) for path in include_roots],
        symbols=evidence,required_symbols=required,required_symbols_unlocated_in_headers=missing,
        version_header_candidates=version_headers,preprocessing_executed=False,npu_executed=False,
        caution="Header occurrence does not resolve an active preprocessor definition or enum value; builtins/compile flags remain unverified")
    if missing:report["status"]="REQUIRED_SYMBOLS_NEED_COMPILER_PROBE"
    if not source_scan["complete"] or not sdk_scan["complete"]:report["status"]="INCOMPLETE_HEADER_OR_SOURCE_SCAN"
    return report


def write_probe(report,directory):
    output=Path(directory)
    lines=["// Evidence probe only. Compile with the MATCHED real ASC target flags; do not run.",
        "#include \"kernel_operator.h\""]
    candidates=report["version_header_candidates"]
    if len(candidates)==1:
        lines.append("#include "+json.dumps(candidates[0]))
    lines.append("#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && (defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__))")
    lines.append("using namespace AscendC;")
    if "ASC_DEVKIT_VERSION_NUM" in report["required_symbols"]:
        lines.extend(["#ifndef ASC_DEVKIT_VERSION_NUM",'#error "Matched SDK did not define ASC_DEVKIT_VERSION_NUM; do not substitute an assumed value"',"#endif"])
    for symbol in report["required_symbols"]:
        lines.append('extern "C" __aicore__ uint64_t QsfaSdkProbe_'+symbol+'() { return static_cast<uint64_t>('+symbol+'); }')
    lines.append("#endif\n")
    (output/"probe.asc").write_text("\n".join(lines))
    macro_lines=["// Macro-only evidence. No kernel_operator.h or host API include is added."]
    for candidate in candidates:
        macro_lines.append("#include "+json.dumps(candidate))
    if not candidates:
        macro_lines.append("// No actual version header was located; compiler builtins/flags remain unverified.")
    (output/"macro_probe.cpp").write_text("\n".join(macro_lines)+"\n")
    compiler=report["compiler"]["path"]
    command=None
    if compiler:
        command=[compiler,"--npu-arch=dav-3510","-E","-dM"]
        for include in report["include_roots"]:command.extend(["-I",include])
        command.append(str((output/"macro_probe.cpp").resolve()))
    probe=dict(status="GENERATED_NOT_EXECUTED",command=command,
        target_flag_source="same dav-3510 ASC target used by this repository, not a host-default compile",
        caution="Before execution, compare with the actual matched CMake compile_commands.json; preprocessing does not resolve enum values",
        macro_probe="macro_probe.cpp includes only actual located version headers; probe.asc is a separate compile-only device symbol probe",
        device_probe="The ASC compile-only functions refer to actual SDK symbols; no guessed numeric definitions and no NPU execution")
    (output/"preprocess_command.json").write_text(json.dumps(probe,indent=2)+"\n")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root",type=Path,required=True)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--compiler",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    output=args.output.expanduser().resolve();sdk=args.sdk_root.expanduser().resolve()
    if output.exists():parser.error("Use a new local output directory")
    if _inside(output,sdk):parser.error("Do not write diagnostics into the selected SDK")
    output.mkdir(parents=True)
    try:
        report=inspect_sdk(args.sdk_root,args.source_root,args.compiler)
        write_probe(report,output)
        (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        print(str(output/"report.json"))
        return 0 if report["status"]=="SDK_SYMBOL_EVIDENCE_CAPTURED" else 2
    except Exception as error:
        report=dict(status="ERROR",compatibility="NOT_VERIFIED",error=str(error),traceback=traceback.format_exc(),npu_executed=False)
        (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        print(str(output/"report.json"));return 1


if __name__=="__main__":raise SystemExit(main())
