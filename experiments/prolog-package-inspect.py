"""Inspect installed Prolog binary files using only Python's standard library."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

KEY = "b2324aedf7041b9b1f196f7c5c4094e6eaea068214c387c1580e1d7effd6c945"
SOURCE_SHA = "ab0876c2fc09b7cc2b0541063dc7bab16414b2046ea9030bc1fecce4020be1bc"


def emit(event, **values):
    print(json.dumps(dict(event=event, **values), ensure_ascii=False), flush=True)


def brief(value, depth=0):
    if depth > 3:
        return (
            {"type": type(value).__name__, "length": len(value)}
            if isinstance(value, (dict, list))
            else str(value)[:160]
        )
    if isinstance(value, dict):
        items = list(value.items())
        return {str(k): brief(v, depth + 1) for k, v in items[:12]}
    if isinstance(value, list):
        return {"length": len(value), "first": [brief(v, depth + 1) for v in value[:1]]}
    return value if not isinstance(value, str) else value[:300]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prolog-python", type=Path, required=True)
    parser.add_argument("--key", default=KEY)
    args = parser.parse_args()
    root = args.prolog_python.resolve()
    source = root / "ops/quant_mla_prolog.py"
    native = root / "ops/_native"
    if not source.is_file() or not native.is_dir():
        parser.error(
            "Expected ops/quant_mla_prolog.py and ops/_native under --prolog-python"
        )
    emit(
        "package",
        path=str(root),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        pinned_source_match=hashlib.sha256(source.read_bytes()).hexdigest()
        == SOURCE_SHA,
        distributions=[
            {"name": d.metadata["Name"], "version": d.version}
            for d in importlib.metadata.distributions(path=[str(root)])
        ],
    )
    init = root / "ops/__init__.py"
    emit(
        "registration_source",
        text=init.read_text()[:3000] if init.exists() else "MISSING",
    )
    files = sorted(p for p in native.rglob("*") if p.is_file())
    json_files = [p for p in files if p.suffix == ".json"]
    matches, skipped, schemas = [], [], []
    for p in files:
        if args.key in p.name:
            matches.append({"path": str(p.relative_to(native)), "match": "filename"})
    for p in json_files:
        if p.stat().st_size > 64 * 1024 * 1024:
            skipped.append(str(p.relative_to(native)))
            continue
        text = p.read_text(errors="replace")
        if args.key in text:
            pos = text.index(args.key)
            matches.append(
                {
                    "path": str(p.relative_to(native)),
                    "match": "json",
                    "context": text[max(0, pos - 500) : pos + 1500],
                }
            )
        try:
            data = json.loads(text)
        except ValueError:
            emit("invalid_json", path=str(p.relative_to(native)))
            continue
        keys = tuple(data.keys()) if isinstance(data, dict) else ("list",)
        if keys not in schemas and len(schemas) < 4:
            schemas.append(keys)
            emit(
                "manifest_example", path=str(p.relative_to(native)), content=brief(data)
            )
    emit(
        "summary",
        native_files=len(files),
        shared_libraries=sum(p.suffix == ".so" for p in files),
        json_files=len(json_files),
        target_key_found=bool(matches),
        matches=matches[:4],
        skipped=skipped,
        scope="CPU filesystem inspection only. Presence does not prove runtime registration or ABI compatibility.",
    )


if __name__ == "__main__":
    main()
