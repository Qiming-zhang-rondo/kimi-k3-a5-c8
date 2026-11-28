#!/usr/bin/env python3
"""Prepare a LOCAL ONLY private Prolog clone for independent ASC compilation.

No build, device launch, SDK write, network operation or upload occurs. Only
the new output tree is modified. Original template registration remains intact
when QSFA_ASC_DIRECT_BUILD is not 1; no tiling layout or launch ABI is invented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

PUBLIC_ROOT = Path(__file__).resolve().parents[1]
HELPER = PUBLIC_ROOT / "experimental/prolog_inplace_build/direct_tiling_load.h"
MAX_SOURCE_BYTES = 256 * 1024 * 1024
HOST_GUARD = "#if !defined(QSFA_ASC_DIRECT_BUILD) || QSFA_ASC_DIRECT_BUILD != 1\n"
TYPE_NAME = re.compile(r"(?:::)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*")
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def code_mask(source):
    """Mask comments/quoted literals without moving any source offsets."""
    result = list(source)
    i, size = 0, len(source)
    while i < size:
        end = None
        if source.startswith("//", i):
            end = source.find("\n", i + 2)
            while end >= 0 and source[:end].rstrip("\r").endswith("\\"):
                end = source.find("\n", end + 1)
            end = size if end < 0 else end
        elif source.startswith("/*", i):
            close = source.find("*/", i + 2)
            if close < 0:
                raise ValueError("Unterminated C++ block comment")
            end = close + 2
        elif source.startswith('R"', i):
            opening = source.find("(", i + 2, i + 19)
            if opening < 0:
                raise ValueError("Unsupported or unterminated C++ raw string")
            delimiter = source[i + 2:opening]
            if any(character.isspace() or character in "()\\" for character in delimiter):
                raise ValueError("Invalid C++ raw string delimiter")
            close = source.find(")" + delimiter + '"', opening + 1)
            if close < 0:
                raise ValueError("Unterminated C++ raw string")
            end = close + len(delimiter) + 2
        elif source[i] in "\"'":
            quote, end = source[i], i + 1
            while end < size:
                if source[end] == "\\":
                    end += 2
                elif source[end] == quote:
                    end += 1
                    break
                else:
                    end += 1
            else:
                raise ValueError("Unterminated C++ quoted literal")
        if end is None:
            i += 1
        else:
            for index in range(i, min(end, size)):
                if source[index] not in "\r\n":
                    result[index] = " "
            i = end
    return "".join(result)


def macro_calls(source, names, *, semicolon):
    masked = code_mask(source)
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\s*\(")
    calls = []
    for match in pattern.finditer(masked):
        # Only complete standalone macro statements are changed, never macro
        # definitions, embedded expressions or a nested occurrence.
        if calls and match.start() < calls[-1][1]:
            continue
        line_start = source.rfind("\n", 0, match.start()) + 1
        if masked[line_start:match.start()].strip():
            raise ValueError("Unsupported non-standalone " + match.group(1))
        opening = masked.find("(", match.start(), match.end())
        depth, closing = 1, opening + 1
        while closing < len(masked) and depth:
            depth += (masked[closing] == "(") - (masked[closing] == ")")
            closing += 1
        if depth:
            raise ValueError("Unbalanced " + match.group(1))
        end = closing
        while end < len(masked) and masked[end].isspace():
            end += 1
        if end < len(masked) and masked[end] == ";":
            end += 1
        elif semicolon:
            raise ValueError("Expected a complete semicolon-terminated " + match.group(1))
        else:
            end = closing
        calls.append((match.start(), end, masked[opening + 1:closing - 1], match.group(1)))
    return calls


def _apply(source, replacements):
    for start, end, replacement in sorted(replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def patch_template_header(source):
    masked = code_mask(source)
    include = re.compile(r'(?m)^[ \t]*#[ \t]*include[ \t]*[<"]'
        r'ascendc/host_api/tiling/template_argument\.h[>"][^\r\n]*(?:\r?\n|$)')
    replacements = []
    for match in include.finditer(source):
        if not masked[match.start():match.end()].lstrip().startswith("#"):
            continue
        original = source[match.start():match.end()]
        replacements.append((match.start(), match.end(), HOST_GUARD + original +
                             ("" if original.endswith("\n") else "\n") + "#endif\n"))
    if len(replacements) != 1:
        raise ValueError("Expected exactly one active template_argument.h include")
    calls = macro_calls(source, ("ASCENDC_TPL_ARGS_DECL", "ASCENDC_TPL_SEL"), semicolon=False)
    if not any(call[3] == "ASCENDC_TPL_ARGS_DECL" for call in calls) or not any(
            call[3] == "ASCENDC_TPL_SEL" for call in calls):
        raise ValueError("Template header must contain real ARGS_DECL and SEL blocks")
    for start, end, _, _ in calls:
        replacements.append((start, end, HOST_GUARD + source[start:end] + "\n#endif\n"))
    return _apply(source, replacements), len(calls)


def patch_entry(source, *, device_constants=False):
    calls = macro_calls(source, ("GET_TILING_DATA_WITH_STRUCT",), semicolon=True)
    if not calls:
        raise ValueError("Entry lacks GET_TILING_DATA_WITH_STRUCT statements")
    replacements = []
    for start, end, arguments, _ in calls:
        parts = [part.strip() for part in arguments.split(",")]
        if len(parts) != 3:
            raise ValueError("Expected exactly three tiling macro arguments")
        type_name = re.sub(r"\s*::\s*", "::", parts[0])
        if not TYPE_NAME.fullmatch(type_name) or not all(IDENTIFIER.fullmatch(part) for part in parts[1:]):
            raise ValueError("Tiling macro requires a real qualified type, identifier variable and identifier GM argument")
        variable, raw = parts[1:]
        original = source[start:end]
        replacement = ("#if defined(QSFA_ASC_DIRECT_BUILD) && QSFA_ASC_DIRECT_BUILD == 1\n"
            + type_name + " " + variable + "{};\n"
            + "#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && "
              "(defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__))\n"
            + "qsfa_direct::LoadTilingWords(" + variable + ", " + raw + ");\n"
            + "#endif\n#else\n" + original + "\n#endif\n")
        replacements.append((start, end, replacement))
    prefix = ('#if defined(QSFA_ASC_DIRECT_BUILD) && QSFA_ASC_DIRECT_BUILD == 1\n'
              '#include "direct_tiling_load.h"\n')
    if device_constants:
        prefix += '#include "qsfa_actual_device_constants.h"\n'
    return prefix + '#endif\n' + _apply(source, replacements), len(calls)


def _relative_regular(root, value):
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or any(part in (".", "..") for part in relative.parts):
        raise ValueError("Entry/header must be a relative path within source-root")
    path = root / relative
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Patched source paths and their parents must not be symlinks")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("Patched file must be a regular file inside source-root")
    return relative, path


def _source_budget(root):
    size = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Refusing nonregular source tree entries")
            size += info.st_size
            if size >= MAX_SOURCE_BYTES:
                raise ValueError("Private source tree must be smaller than 256 MiB; choose its operator subtree")
    return size


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source_root, entry, template_key_header, output, *, helper_path=None,
            device_constants_header=None):
    raw_root, raw_output = Path(source_root).expanduser(), Path(output).expanduser()
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ValueError("source-root must be a real directory")
    if raw_output.exists() or raw_output.is_symlink():
        raise ValueError("Use a NEW output directory; existing paths are never overwritten")
    root, output = raw_root.resolve(), raw_output.resolve()
    if output.is_relative_to(root) or output.is_relative_to(PUBLIC_ROOT):
        raise ValueError("Private output must be outside source-root AND the public repository")
    entry_rel, entry_path = _relative_regular(root, entry)
    key_rel, key_path = _relative_regular(root, template_key_header)
    source_bytes = _source_budget(root)
    if entry_rel == key_rel or entry_rel.suffix not in (".asc", ".cpp", ".cc", ".cxx"):
        raise ValueError("Provide a separate ASC/C++ entry and template-key header")
    new_entry_rel = entry_rel.with_suffix(".asc")
    if new_entry_rel != entry_rel and (root / new_entry_rel).exists():
        raise ValueError("The staged .asc entry name already belongs to another source file")
    helper = Path(helper_path) if helper_path is not None else HELPER
    if helper.is_symlink() or not helper.is_file():
        raise ValueError("The public direct_tiling_load.h helper is unavailable")
    helper_bytes = helper.read_bytes()
    collision = root / entry_rel.parent / helper.name
    if collision.exists() or collision.is_symlink():
        if collision.is_symlink() or not collision.is_file() or collision.read_bytes() != helper_bytes:
            raise ValueError("Private source already owns a different helper filename")
    constants_bytes = None
    if device_constants_header is not None:
        constants = Path(device_constants_header).expanduser()
        if constants.is_symlink() or not constants.is_file():
            raise ValueError("device-constants-header must be a regular local file")
        constants_bytes = constants.read_bytes()
        constants_source = constants_bytes.decode("utf-8")
        if ("template_argument.h" in constants_source or
                re.search(r"\bASCENDC_TPL_(?:ARGS_DECL|SEL)\s*\(", code_mask(constants_source)) or
                re.search(r"(?m)^\s*#\s*define\s+ASCENDC_TPL_\w+(?:\([^\n]*?\))?\s*(?://[^\n]*)?$", constants_source)):
            raise ValueError("Constants header must not reintroduce host registration or empty template macros")
        constants_target = root / entry_rel.parent / "qsfa_actual_device_constants.h"
        if constants_target.exists() or constants_target.is_symlink():
            raise ValueError("Private source already owns the device-constants staging name")
    entry_bytes, key_bytes = entry_path.read_bytes(), key_path.read_bytes()
    patched_entry, load_count = patch_entry(entry_bytes.decode("utf-8"),
                                           device_constants=constants_bytes is not None)
    patched_key, registration_count = patch_template_header(key_bytes.decode("utf-8"))
    # The excluded host registration header may also define kernel-type enum
    # constants. Inventory usages; never guess their values or swallow them.
    key_source = key_bytes.decode("utf-8")
    key_code = code_mask(key_source)
    key_spans = macro_calls(key_source, ("ASCENDC_TPL_ARGS_DECL", "ASCENDC_TPL_SEL"), semicolon=False)
    key_outside = _apply(key_code, [(start, end, " " * (end - start))
                                  for start, end, _, _ in key_spans])
    remaining_symbols = sorted(set(re.findall(r"\bASCENDC_TPL_[A-Za-z0-9_]+\b",
                                  code_mask(entry_bytes.decode("utf-8")) + "\n" + key_outside)))
    clone = output / "private_source"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        shutil.copytree(root, clone, symlinks=True)
        # No original file, linked SDK, vendor tree or linked dependency is
        # opened for writing. Only the verified regular clone files change.
        (clone / key_rel).write_text(patched_key, encoding="utf-8", newline="")
        (clone / new_entry_rel).write_text(patched_entry, encoding="utf-8", newline="")
        (clone / entry_rel.parent / helper.name).write_bytes(helper_bytes)
        if constants_bytes is not None:
            (clone / entry_rel.parent / "qsfa_actual_device_constants.h").write_bytes(constants_bytes)
        manifest = dict(status="LOCAL_STAGE_PREPARED_NOT_COMPILED", disclosure="DO_NOT_UPLOAD",
            warning="PRIVATE CLONE AND MANIFEST: KEEP IN INTERNAL ENVIRONMENT",
            source_root=str(root), cloned_source=str(clone), source_bytes=source_bytes,
            original_entry=str(entry_rel), staged_entry=str(new_entry_rel),
            template_key_header=str(key_rel), tiling_load_statements=load_count,
            guarded_template_registration_blocks=registration_count,
            required_build_define="QSFA_ASC_DIRECT_BUILD=1",
            remaining_template_symbol_tokens_to_verify=remaining_symbols,
            device_constants_header_supplied=constants_bytes is not None,
            device_constants_sha256=_sha(constants_bytes) if constants_bytes is not None else None,
            constants_instruction="Resolve every remaining template token from real headers. If a symbol came from the excluded host header, extract its exact declaration locally into --device-constants-header; never guess numeric values or define empty template macros.",
            original_sha256={str(entry_rel): _sha(entry_bytes), str(key_rel): _sha(key_bytes)},
            staged_sha256={str(new_entry_rel): _sha(patched_entry.encode()),
                          str(key_rel): _sha(patched_key.encode()), helper.name: _sha(helper_bytes)},
            launch_signature_changed=False, private_abi_invented=False,
            sdk_modified=False, source_modified=False, device_executed=False,
            remaining_gates=["real device POD matches host tiling bytes/offsets",
                "original tiling-key selects exact compiled specialization",
                "matched workspace and AIC/AIV ownership",
                "actual ASC baseline compile before any device launch"])
        (output / "DO_NOT_UPLOAD_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--template-key-header", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-constants-header", type=Path,
                        help="Optional locally extracted REAL device constants; no guessed enum values")
    args = parser.parse_args(argv)
    try:
        prepare(args.source_root, args.entry, args.template_key_header, args.output,
                device_constants_header=args.device_constants_header)
    except (ValueError, OSError, UnicodeError) as error:
        parser.exit(2, "Local staging refused: " + str(error) + "\n")
    print("LOCAL_ONLY stage prepared. Do not upload the clone or DO_NOT_UPLOAD_manifest.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
