#!/usr/bin/env python3
"""Collect small framework evidence without mistaking generation for accuracy."""
import argparse
import json
import math
from pathlib import Path


def collect(directory, *, require_fusion=True):
    root = Path(directory)
    probe_path = root/"probe.json"
    if not probe_path.is_file() or json.loads(probe_path.read_text()).get("status") != "GENERATION_FINISHED":
        return dict(status="FAILED", reason="model generation did not finish", model_accuracy="NOT_MEASURED")
    probe=json.loads(probe_path.read_text())
    requested_kernel=probe.get("single_pass_kernel") if probe.get("arm")=="single_pass" else None
    paths = sorted((root/"layers").glob("recent128-*.json"))
    if not paths:
        return dict(status="NOT_EXERCISED", reason="no framework layer evidence", model_accuracy="NOT_MEASURED")
    layers = [json.loads(path.read_text()) for path in paths]
    problems = []
    for path, layer in zip(paths, layers):
        if layer.get("status") != "EXERCISED" or layer.get("failure"):
            problems.append(dict(file=path.name, reason=layer.get("failure") or layer.get("status")))
        if layer.get("performance_claim") or layer.get("native_quantizer_changed") or layer.get("topk_changed"):
            problems.append(dict(file=path.name, reason="experiment contract changed"))
        counts = layer.get("counters", {})
        if require_fusion and (not layer.get("fusion_exercised") or counts.get("fused_rows", 0) <= 0
                               or counts.get("fused_kernel_calls", 0) <= 0):
            problems.append(dict(file=path.name, reason="fusion not exercised in this layer/rank"))
        single_pass=layer.get("reader")=="single_pass"
        if requested_kernel in ("fast_pair","load_tile4"):
            call=layer.get("single_pass_last_call") or {}
            identity=layer.get("library_identity") or {}
            digest=identity.get("library_sha256", "")
            variants=identity.get("configuration",{}).get("load_tile_variants",{})
            geometry=variants.get("4",variants.get(4,{}))
            expected_entry={"fast_pair":"QsfaSinglePassLaunchFastPair",
                            "load_tile4":"QsfaSinglePassLaunchLoadTile"}[requested_kernel]
            if (not single_pass or layer.get("single_pass_kernel")!=requested_kernel
                    or call.get("kernel_variant")!=requested_kernel
                    or call.get("entry")!=expected_entry
                    or (requested_kernel=="load_tile4" and (call.get("load_variant")!=4 or
                        call.get("device_entry")!="QsfaSinglePassLaunchLoadTile32" or
                        geometry!={"rows":32,"input_stride_bytes":416,"input_depth":2,"output_depth":1}))
                    or call.get("external_merge_calls")!=0
                    or call.get("topk_partitioned") is not False
                    or call.get("return_stats") is not False
                    or call.get("candidate_attention_calls")!=call.get("head_groups")
                    or not isinstance(call.get("head_groups"),int) or call["head_groups"]<1
                    or counts.get("single_pass_expected_attention_calls")!=counts.get("candidate_attention_calls")
                    or counts.get("single_pass_"+requested_kernel+"_calls")!=counts.get("candidate_attention_calls")
                    or not isinstance(digest,str) or len(digest)!=64
                    or any(c not in "0123456789abcdef" for c in digest)
                    or identity.get("configuration",{}).get("abi_version")!=1):
                problems.append(dict(file=path.name,reason="requested "+requested_kernel+" actual entry/library/call coverage missing"))
        if single_pass and (layer.get("control_scope")!="full_FP64_whole_selected_set"
                            or layer.get("control_limit")!=.01
                            or counts.get("query_rows",0)<=0
                            or counts.get("full_math_checked_rows")!=counts.get("query_rows")
                            or counts.get("candidate_attention_calls",0)<=0
                            or counts.get("validation_attention_calls")!=0):
            problems.append(dict(file=path.name,reason="single-pass full mathematical control/coverage missing"))
        limit=.01 if single_pass else .001
        for name in ("relative_l2", "worst_row_relative_l2"):
            metric = layer.get("control_max", {}).get(name)
            if not isinstance(metric, (int, float)) or not math.isfinite(metric) or not 0 <= metric <= limit:
                problems.append(dict(file=path.name, reason="invalid or failing native control " + name))
    keys = set().union(*(layer.get("counters", {}) for layer in layers))
    totals = {key:sum(layer.get("counters", {}).get(key, 0) for layer in layers) for key in sorted(keys)}
    return dict(status="WIRING_VALIDATED" if not problems else "FAILED", layers=len(layers),
        fusion_required=require_fusion, counters=totals, problems=problems,
        model_accuracy="NOT_MEASURED", serving_performance="NOT_MEASURED",
        next_gate="Run the unchanged existing GLM accuracy task in OFF/ON arms; report scores and samples")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--native-only", action="store_true")
    args=p.parse_args()
    report=collect(args.directory, require_fusion=not args.native_only)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(0 if report["status"]=="WIRING_VALIDATED" else 1)


if __name__=="__main__": main()
