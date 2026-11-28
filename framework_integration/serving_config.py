#!/usr/bin/env python3
"""Merge only our serving options into an existing additional-config JSON.

No launcher/container/environment is changed. Existing vendor keys retain
their values; attention-config (including indexer MXFP4) is separate.
"""
import argparse
import json
from pathlib import Path


def serving_config(original, *, source_root=None, library_dir=None, max_query_chunk=None):
    if not isinstance(original, dict):
        raise ValueError("additional-config must be a JSON object")
    result = dict(original)
    own = dict(result.get("sfa_recent128", {}))
    own.update(window=128, execution="serving", preprocess="preserve",
               reader="single_pass", single_pass_kernel="batched_v4")
    for name, value in (("source_root", source_root), ("library_dir", library_dir),
                        ("max_query_chunk", max_query_chunk)):
        if value is not None:
            own[name] = value
    for name in ("source_root", "library_dir", "diagnostics_path", "max_query_chunk"):
        if not own.get(name):
            raise ValueError("Existing recent128 config must provide " + name)
    if type(own["max_query_chunk"]) is not int or not 1 <= own["max_query_chunk"] <= 65536:
        raise ValueError("max_query_chunk must be the real per-request scheduler bound in 1..65536")
    result["sfa_recent128"] = own
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--library-dir")
    parser.add_argument("--max-query-chunk", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file; the input is never overwritten")
    try:
        value = serving_config(json.loads(args.input.read_text()), source_root=args.source_root,
            library_dir=args.library_dir, max_query_chunk=args.max_query_chunk)
        with args.output.open("x") as out:
            out.write(json.dumps(value, indent=2) + "\n")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
