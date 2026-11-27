#!/usr/bin/env python3
"""Collect small framework evidence without mistaking generation for accuracy."""
import argparse
import json
import math
from pathlib import Path


def kv_split_evidence_valid(layer):
    """Require actual split coverage even when the last call was prefill.

    Runtime checks every dispatch before counting it. This collector checks
    both the most recent actual entry and cumulative split/reduction counts;
    a requested variant or a generation-only fallback cannot satisfy them.
    """
    counts=layer.get("counters",{})
    call=layer.get("single_pass_last_call") or {}
    identity=layer.get("library_identity") or {}
    configuration=identity.get("configuration") or {}
    digest=identity.get("library_sha256","")
    variants=configuration.get("load_tile_variants") or {}
    geometry=variants.get("4",variants.get(4,{}))
    expected_geometry={"rows":32,"input_stride_bytes":416,"input_depth":2,"output_depth":1}
    if (layer.get("reader")!="single_pass" or layer.get("single_pass_kernel")!="kv_split4"
            or not isinstance(digest,str) or len(digest)!=64
            or any(value not in "0123456789abcdef" for value in digest)
            or configuration.get("abi_version")!=1
            or configuration.get("buffers_bytes")!=120
            or configuration.get("kv_split_reader_available") is not True
            or configuration.get("kv_split_abi_version")!=2
            or configuration.get("kv_split_tiling_bytes")!=112
            or configuration.get("kv_split_reader_entry")!="QsfaSinglePassLaunchKvSplit"
            or configuration.get("load_tile_reader_available") is not True
            or geometry!=expected_geometry):
        return False

    query=call.get("queries")
    heads=call.get("heads")
    groups=call.get("head_groups")
    if (type(query) is not int or not 1<=query<=1024 or type(heads) is not int or heads<1
            or type(groups) is not int or groups!=(heads+63)//64):
        return False
    split=query<=4
    entry="QsfaSinglePassLaunchKvSplit" if split else "QsfaSinglePassLaunchLoadTile"
    device_entry="qsfa_recent128_kv_split_attention" if split else "QsfaSinglePassLaunchLoadTile32"
    aic=query*4 if split else max(value for value in (1,2,4,8,16) if value<=query and query%value==0)
    numeric=("split_count","aic_count","candidate_attention_calls","device_reduction_calls")
    if (any(type(call.get(name)) is not int for name in numeric)
            or call.get("status")!="EXERCISED" or call.get("kernel_variant")!="kv_split4"
            or call.get("executed_kernel_variant")!=("kv_split4" if split else "load_tile4")
            or call.get("dispatch_reason")!=("q_le_4_kv_split4" if split else "q_gt_4_prefill_load_tile4")
            or call.get("entry")!=entry or call.get("device_entry")!=device_entry
            or call.get("split_count")!=(4 if split else 1) or call.get("aic_count")!=aic
            or (not split and call.get("load_variant")!=4)
            or call.get("return_stats") is not False
            or call.get("external_merge_calls")!=0
            or call.get("topk_partitioned") is not split
            or call.get("original_topk_unchanged") is not True
            or call.get("candidate_attention_calls")!=groups
            or call.get("device_reduction_calls")!=(groups if split else 0)
            or call.get("validation_attention_calls")!=0):
        return False

    names=("candidate_attention_calls","single_pass_expected_attention_calls",
           "single_pass_kv_split4_calls","kv_split_attention_calls","kv_split_reduction_calls",
           "kv_split_rows","query_rows","fused_rows","fused_kernel_calls")
    if any(type(counts.get(name)) is not int or counts[name]<=0 for name in names):
        return False
    fallback=counts.get("kv_split_prefill_fallback_calls",0)
    if type(fallback) is not int or fallback<0:
        return False
    candidate=counts["candidate_attention_calls"]
    actual_split=counts["kv_split_attention_calls"]
    if (candidate!=counts["single_pass_expected_attention_calls"]
            or candidate!=counts["single_pass_kv_split4_calls"]
            or candidate!=counts["fused_kernel_calls"]
            or candidate!=actual_split+fallback
            or counts["kv_split_reduction_calls"]!=actual_split
            or counts["fused_rows"]!=counts["query_rows"]
            or not 0<counts["kv_split_rows"]<=counts["query_rows"]
            or counts["kv_split_rows"]>4*actual_split
            or (fallback==0 and counts["kv_split_rows"]!=counts["query_rows"])
            or (fallback>0 and counts["kv_split_rows"]>=counts["query_rows"])
            or (split and actual_split<groups) or (not split and fallback<groups)):
        return False
    return True


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
        if (requested_kernel=="kv_split4" or layer.get("single_pass_kernel")=="kv_split4") and (
                requested_kernel!="kv_split4" or not kv_split_evidence_valid(layer)):
            problems.append(dict(file=path.name,
                reason="requested kv_split4 actual dispatch/library/split-reduction coverage missing"))
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
