#!/usr/bin/env python3
"""Read existing result JSONs into a small review artifact; no Torch or NPU."""
import argparse
import json
from pathlib import Path

METRIC_KEYS=("passed","finite","nonfinite_count","relative_l2","max_row_relative_l2",
             "max_abs","relative_l2_limit","worst_row_l2_limit")


def metric(value):
    return {key:value[key] for key in METRIC_KEYS if key in value}


def failed_checks(value, prefix="controls"):
    failures={}
    if not isinstance(value,dict): return failures
    if value.get("passed") is False or value.get("control_passed") is False:
        small={key:item for key,item in value.items() if isinstance(item,(str,int,float,bool,type(None)))}
        for key in ("expected_shape","actual_shape","first_bad_positions"):
            if key in value: small[key]=value[key]
        failures[prefix]=small
    for key,item in value.items():
        if isinstance(item,dict): failures.update(failed_checks(item,prefix+"."+key))
    return failures


def check_status(value):
    # Presence matters: {} means neither field was supplied; explicit null is
    # preserved as null, and must never be promoted to a successful check.
    return {key:value[key] for key in ("passed","control_passed") if key in value}


def graph_detail(value):
    result=check_status(value)
    if "fields" in value:
        fields=value["fields"]
        if fields is None:
            result["fields"]=None
        elif isinstance(fields,dict):
            allowed=("passed","control_passed","mismatch_count","shape","dtype",
                     "expected_shape","actual_shape","expected_dtype","actual_dtype")
            result["fields"]={name:{key:field[key] for key in allowed if key in field}
                              for name,field in fields.items() if isinstance(field,dict)}
    return result


def extract(path):
    report=json.loads(Path(path).read_text())
    result=dict(results=str(path),version=report.get("version"),status=report.get("status"),
        input_sha256=report.get("input_sha256"),
        args={key:report.get("args",{}).get(key) for key in
              ("queries","keys","direct_aic_count","bridge_library","timing")},
        bridge_configuration=report.get("bridge_configuration"),
        bridge_build={key:report.get("bridge_build",{}).get(key) for key in
                      ("library","library_sha256","sdk_root","bridge_source_sha256")},
        controls={},timings={})
    for name,value in report.get("controls",{}).items():
        item=check_status(value)
        item["check_statuses"]={key:check_status(check) for key,check in value.items()
                                if isinstance(check,dict)}
        if "reader" in value:
            reader=value["reader"]
            item["reader"]={"passed":reader.get("passed")}
            for key,check in reader.items():
                if not isinstance(check,dict): continue
                if key.endswith("same_native_states_merge") or key.endswith("direct_hot_vs_installed"):
                    item["reader"][key]=metric(check)
                elif "rne" in key or "merge_acceptance" in key or key.endswith("counts"):
                    item["reader"][key]={k:v for k,v in check.items() if isinstance(v,(str,int,float,bool,type(None)))}
        if "kernel_vs_matching_reference" in value:
            item["kernel_vs_matching_reference"]=metric(value["kernel_vs_matching_reference"])
        result["controls"][name]=item
    result["graph_capture_controls"]={key:value.get("passed") for key,value in report.get("graph_capture_controls",{}).items()}
    result["graph_capture_detail"]={key:graph_detail(value)
        for key,value in report.get("graph_capture_controls",{}).items() if isinstance(value,dict)}
    result["failed_check_paths"]=failed_checks(report.get("controls",{}))
    result["failed_check_paths"].update(failed_checks(report.get("graph_capture_controls",{}),"graph_capture_controls"))
    for name,value in report.get("timings",{}).items():
        graph=value.get("graph",{})
        result["timings"][name]={key:value for key,value in graph.items() if key!="rounds"}
        result["timings"][name]["rounds"]=[dict(round=row.get("round"),order=row.get("order"),
            baseline_event_us=row.get("baseline",{}).get("event_us",{}).get("median"),
            candidate_event_us=row.get("candidate",{}).get("event_us",{}).get("median"),
            event_us_ratio=row.get("event_us_ratio")) for row in graph.get("rounds",[])]
    if "traceback" in report: result["traceback"]=report["traceback"]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if output in {path.resolve() for path in args.results}:
        raise ValueError("Summary must not replace original results")
    values=[extract(path) for path in args.results]
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(source_results_unchanged=True,device_launched=False,cases=values),
                                indent=2,allow_nan=False)+"\n")
    print(f"Saved {len(values)} cases: {output}")


if __name__=="__main__":main()
