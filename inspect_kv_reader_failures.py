#!/usr/bin/env python3
"""Read existing KV reader results; never import operators or rerun experiments.

Examples:
  python inspect_kv_reader_failures.py /tmp/kv-native-twopass
  python inspect_kv_reader_failures.py /tmp/run/results/results.json --decode software

The default selection is baseline / mixed / hot_merge / k16 / h16 /
native_cast. JSON is written to stdout; selection notices and unmatched
candidate tables go to stderr. All reported diagnostic metrics are unmodified.
"""
import argparse
import json
from pathlib import Path
import sys


HOT_READERS = {"hot_merge", "native_c4_plus_fused_c8_hot_merge"}
ENTRY_FIELDS = (
    "name", "status", "role", "recipe", "window", "mode", "reader",
    "requested_reader", "hot_block_keys", "hot_block_heads", "hot_decode",
    "index_capacity", "partition", "merge", "merge_selector", "writer",
    "control_passed", "control_failure_reason", "failure_kind", "error",
    "traceback", "reader_consistency", "partition_consistency", "writer_consistency",
)
DIAGNOSTIC_FIELDS = ("reader_consistency", "partition_consistency", "writer_consistency")


def select_results(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        candidates, rule = [path], "explicit_file"
    elif path.is_dir():
        candidates = [p for p in path.rglob("results.json") if p.is_file()]
        rule = "newest_results_json_by_mtime_ns; lexical_absolute_path_breaks_ties"
    else:
        raise FileNotFoundError(f"Input does not exist: {path}")
    if not candidates:
        raise FileNotFoundError(f"No results.json found under {path}")
    selected = max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))
    return selected, {"requested_path": str(path), "source_path": str(selected),
                      "results_found": len(candidates), "selection_rule": rule,
                      "source_mtime_ns": selected.stat().st_mtime_ns}


def read_object(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def launcher_identity(source):
    """Use only the result's directory or its parent, never a different run."""
    searched = [source.parent / "launcher.json", source.parent.parent / "launcher.json"]
    existing = next((p for p in searched if p.is_file()), None)
    identity = {"path": str(existing) if existing else None,
                "searched_paths": [str(p) for p in searched], "payload_commit": None}
    if existing is not None:
        try:
            document = read_object(existing)
            if "payload_commit" in document:
                identity["payload_commit"] = document["payload_commit"]
            else:
                identity["missing_fields"] = ["payload_commit"]
        except (OSError, ValueError) as exc:
            identity["read_error"] = str(exc)
    return identity


def entries(document):
    cases = document.get("cases")
    if not isinstance(cases, list):
        cases = [document] if isinstance(document.get("variants"), list) else []
    for case_index, case in enumerate(cases):
        if not isinstance(case, dict):
            continue
        variants = case.get("variants", [])
        if not isinstance(variants, list):
            continue
        for entry in variants:
            if isinstance(entry, dict):
                yield case_index, case.get("label"), entry


def selection_values(entry):
    implementation = entry.get("implementation")
    implementation = implementation if isinstance(implementation, dict) else {}
    evidence = {}
    for field in ("recipe", "mode", "hot_block_keys", "hot_block_heads", "hot_decode"):
        if entry.get(field) is not None:
            evidence[field] = {"source": field, "value": entry[field]}
        elif implementation.get(field) is not None:
            evidence[field] = {"source": "implementation." + field, "value": implementation[field]}
        else:
            evidence[field] = {"source": None, "value": None}
    # Actual reader wins over requested reader when both are present. Do not
    # mistake a native fallback/control for an executed fused reader.
    for source, value in (("reader", entry.get("reader")),
                          ("requested_reader", entry.get("requested_reader")),
                          ("implementation.reader", implementation.get("reader"))):
        if value is not None:
            evidence["reader"] = {"source": source, "value": value}
            break
    else:
        evidence["reader"] = {"source": None, "value": None}
    return evidence


def matches(evidence, args):
    values = {key: item["value"] for key, item in evidence.items()}
    return (values["recipe"] == args.recipe and values["mode"] == "mixed"
            and values["reader"] in HOT_READERS
            and values["hot_block_keys"] == args.block_keys
            and values["hot_block_heads"] == args.block_heads
            and values["hot_decode"] == args.decode)


def candidate_table(records, limit=20):
    lines = ["Candidates (fields only; names are not used for matching):",
             "case | name | status | recipe | mode | reader | keys | heads | decode"]
    for case_index, label, entry in records[:limit]:
        values = {key: value["value"] for key, value in selection_values(entry).items()}
        row = [label if label is not None else case_index, entry.get("name"), entry.get("status"),
               values["recipe"], values["mode"], values["reader"],
               values["hot_block_keys"], values["hot_block_heads"], values["hot_decode"]]
        lines.append(" | ".join("<missing>" if value is None else str(value).replace("\n", " ")[:100]
                                for value in row))
    if len(records) > limit:
        lines.append(f"... {len(records)-limit} further entries omitted")
    if not records:
        lines.append("(No variant entries)")
    return "\n".join(lines)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="results.json, one run directory, or a root containing runs")
    p.add_argument("--recipe", choices=("baseline", "alg2_max7"), default="baseline")
    p.add_argument("--block-keys", type=int, choices=(16, 32, 64), default=16)
    p.add_argument("--block-heads", type=int, choices=(16, 32), default=16)
    p.add_argument("--decode", choices=("native_cast", "software"), default="native_cast")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        source, selection = select_results(args.path)
        print(f"Selected {source} ({selection['results_found']} result files found; "
              f"{selection['selection_rule']})", file=sys.stderr)
        document = read_object(source)
        records, selected = list(entries(document)), []
        for case_index, label, entry in records:
            evidence = selection_values(entry)
            if matches(evidence, args):
                selected.append({"case_index": case_index, "case_label": label,
                    "selection_evidence": evidence,
                    "entry": {field: entry[field] for field in ENTRY_FIELDS if field in entry},
                    "missing_diagnostic_fields": [field for field in DIAGNOSTIC_FIELDS if field not in entry]})
        if not selected:
            print(f"No matching entry: recipe={args.recipe} mode=mixed reader=hot_merge "
                  f"keys={args.block_keys} heads={args.block_heads} decode={args.decode}. "
                  "No older result file was substituted.", file=sys.stderr)
            print(candidate_table(records), file=sys.stderr)
            for field in ("error", "traceback"):
                if field in document:
                    print(f"Report {field} (unmodified):", file=sys.stderr)
                    print(document[field], file=sys.stderr)
            return 2
        output = {"selection": selection,
                  "version": document.get("version"), "args": document.get("args"),
                  "missing_report_fields": [field for field in ("version", "args") if field not in document],
                  "launcher": launcher_identity(source),
                  "filter": {"recipe": args.recipe, "mode": "mixed", "reader": "hot_merge",
                             "hot_block_keys": args.block_keys, "hot_block_heads": args.block_heads,
                             "hot_decode": args.decode},
                  "selected_count": len(selected), "entries": selected}
        for field in ("error", "traceback"):
            if field in document:
                output[field] = document[field]
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError) as exc:
        print(f"Read failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
