#!/usr/bin/env python3
"""Losslessly package existing verification JSON into small GitHub text files.

Standard library only. Never imports Torch or runs a device. The original JSON
bytes, including every round and failed gate, are restored and checked by SHA.
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

PART_CHARS=32768


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def export_evidence(results_path,output_dir):
    results_path,output_dir=Path(results_path),Path(output_dir)
    raw=results_path.read_bytes()
    report=json.loads(raw)
    if not isinstance(report,dict) or not isinstance(report.get("args"),dict) or \
            report["args"].get("verify_comparison") is not True:
        raise ValueError("Use the original --verify-comparison runner results.json, not a handwritten summary")
    if report.get("status")=="MEASURED" and not all(key in report for key in
            ("verification_timings","controls","graph_controls","post_timing_controls","bridge_build")):
        raise ValueError("MEASURED source lacks original timing/control/build sections; do not package the summary")
    # Parse before creating files; preserve even ERROR/CONTROL_FAILED reports.
    encoded=base64.b64encode(gzip.compress(raw,mtime=0)).decode("ascii")
    output_dir.mkdir(parents=True,exist_ok=False)
    parts=[]
    for index,start in enumerate(range(0,len(encoded),PART_CHARS)):
        name=f"source.{index:04d}.b64.txt"
        payload=(encoded[start:start+PART_CHARS]+"\n").encode("ascii")
        (output_dir/name).write_bytes(payload)
        parts.append(dict(path=name,sha256=sha256(payload),bytes=len(payload)))
    manifest=dict(version="kv-split-verification-evidence-v1",encoding="gzip+base64",
        source_results=str(results_path),source_sha256=sha256(raw),source_bytes=len(raw),
        source_status=report.get("status"),source_version=report.get("version"),
        read_only=True,npu_executed=False,parts=parts)
    summary=dict(source_sha256=manifest["source_sha256"],status=report.get("status"),
        all_accuracy_controls_passed=report.get("all_accuracy_controls_passed"),
        note="Display only; original controls and all unrounded rounds are in the lossless source archive",
        verification_pairs={})
    for name,pair in report.get("verification_timings",{}).items():
        summary["verification_pairs"][name]=dict(baseline=pair.get("baseline"),candidate=pair.get("candidate"),
            timing={mode:dict(status=item.get("status"),round_count=len(item.get("rounds",[])),
                event_us=item.get("event_us"),wall_us=item.get("wall_us"))
                for mode,item in pair.get("timing",{}).items()})
    try:
        summary_text=json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)
    except ValueError:
        summary_text=json.dumps(dict(source_sha256=manifest["source_sha256"],status=report.get("status"),
            note="Nonfinite reported summary values are preserved verbatim in the complete source archive"))
    summary_bytes=(summary_text+"\n").encode("utf-8")
    (output_dir/"verification_summary.json").write_bytes(summary_bytes)
    manifest["summary"]=dict(path="verification_summary.json",sha256=sha256(summary_bytes),bytes=len(summary_bytes))
    (output_dir/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # Check the actual files, not just the in-memory encoding, before success.
    if restore_evidence(output_dir/"manifest.json")!=raw:
        raise ValueError("Exported source does not match the original results bytes")
    return manifest


def restore_evidence(manifest_path):
    manifest_path=Path(manifest_path)
    manifest=json.loads(manifest_path.read_bytes())
    if manifest.get("encoding")!="gzip+base64" or not manifest.get("parts"):
        raise ValueError("Unsupported or empty evidence manifest")
    pieces=[]
    for index,part in enumerate(manifest["parts"]):
        if part["path"]!=f"source.{index:04d}.b64.txt":
            raise ValueError("Evidence parts must have consecutive local names")
        payload=(manifest_path.parent/part["path"]).read_bytes()
        if len(payload)!=part["bytes"] or sha256(payload)!=part["sha256"]:
            raise ValueError("Evidence part byte count/SHA256 mismatch")
        pieces.append(payload.strip())
    raw=gzip.decompress(base64.b64decode(b"".join(pieces),validate=True))
    if len(raw)!=manifest["source_bytes"] or sha256(raw)!=manifest["source_sha256"]:
        raise ValueError("Restored source byte count/SHA256 mismatch")
    return raw


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path,help="Existing full runner results.json, not a handwritten summary")
    parser.add_argument("output",type=Path,help="New output directory; existing directories are never overwritten")
    args=parser.parse_args(argv)
    manifest=export_evidence(args.results,args.output)
    print(json.dumps(dict(status="EXPORTED_EXISTING_BYTES",manifest=str(args.output/"manifest.json"),
        source_status=manifest["source_status"],source_sha256=manifest["source_sha256"],
        source_bytes=manifest["source_bytes"],parts=len(manifest["parts"]),npu_executed=False)))


if __name__=="__main__":
    main()
