#!/usr/bin/env python3
"""Read existing results only; retain launch geometry and raw paired evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path

METRICS=("passed","finite","relative_l2","max_row_relative_l2","max_abs",
         "nonfinite_count","nan_count","tolerance","acceptance_gate",
         "returned_stats_finite_positive_sum")


def portable(value):
    if isinstance(value,float) and not math.isfinite(value):
        return str(value)
    if isinstance(value,dict):
        return {key:portable(item) for key,item in value.items()}
    if isinstance(value,list):
        return [portable(item) for item in value]
    return value


def collect(path):
    path=Path(path)
    if path.is_dir(): path=path/"results.json"
    data=json.loads(path.read_text())
    if not str(data.get("version","")).startswith("qsfa-recent128-single-pass-a5-"):
        raise ValueError("Not a single-pass experiment result")
    controls={name:{key:value[key] for key in METRICS if key in value}
              for name,value in data.get("controls",{}).items()}
    pairs={}
    for name,modes in data.get("timings",{}).items():
        pairs[name]={}
        for mode,item in modes.items():
            pairs[name][mode]={key:item[key] for key in ("status","event_us","wall_us","rounds") if key in item}
    return dict(version="single-pass-raw-evidence-v1",source_results=str(path.resolve()),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),run_version=data.get("version"),
        run_status=data.get("status"),args=data.get("args"),launch_geometry=data.get("launch_geometry"),
        preflight=data.get("preflight"),bridge_build=data.get("bridge_build"),
        bridge_configuration=data.get("bridge_configuration"),
        all_accuracy_controls_passed=data.get("all_accuracy_controls_passed"),controls=controls,
        cross_checks={name:{key:value[key] for key in METRICS if key in value}
                     for name,value in data.get("cross_checks",{}).items()},
        routed_pattern_controls=data.get("routed_pattern_controls"),
        early_hot_refresh_controls=data.get("early_hot_refresh_controls"),
        stats_ab=data.get("stats_ab"),
        load_tile_ab=data.get("load_tile_ab"),
        graph_controls={name:{key:value[key] for key in METRICS if key in value}
                        for name,value in data.get("graph_controls",{}).items()},
        timings=pairs,traceback=data.get("traceback"),performance_scope=data.get("performance_scope"),
        verdict="EXTRACTED_ONLY; source run status and failures retained without reinterpretation")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error("Use a new output path; do not replace existing evidence")
    result=collect(args.results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(portable(result),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status="EXTRACTED",run_status=result["run_status"],output=str(args.output))))


if __name__=="__main__": main()
