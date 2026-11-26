#!/usr/bin/env python3
"""Read existing KV-split evidence and emit one compact JSON; never run NPU.

Usage: python collect_kv_split_failure.py /path/to/results.json
The tool opens JSON/NPZ files read-only, imports neither Torch nor an operator,
and never writes to the experiment. Missing arrays remain NOT_AVAILABLE.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def portable(value,*,computed=False):
    if isinstance(value,dict):
        return {str(k):portable(v,computed=computed) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [portable(v,computed=computed) for v in value]
    if isinstance(value,np.generic):
        value=value.item()
    if isinstance(value,float):
        if not math.isfinite(value):
            return str(value)
        return float(f"{value:.7g}") if computed else value
    return value


def distribution(array):
    array=np.asarray(array)
    finite=np.isfinite(array)
    values=array[finite].astype(np.float64)
    result=dict(shape=list(array.shape),dtype=str(array.dtype),count=int(array.size),
        finite=int(finite.sum()),nan=int(np.isnan(array).sum()),
        positive_inf=int(np.isposinf(array).sum()),negative_inf=int(np.isneginf(array).sum()))
    if values.size:
        result.update(min=float(values.min()),max=float(values.max()),mean=float(values.mean()),
            p50=float(np.median(values)),p95=float(np.percentile(values,95)))
    return portable(result,computed=True)


def bit_histogram(array,limit=8):
    array=np.asarray(array)
    if array.dtype!=np.float32:
        return dict(status="UNSUPPORTED_DTYPE",dtype=str(array.dtype))
    patterns,counts=np.unique(np.ascontiguousarray(array).view(np.uint32),return_counts=True)
    order=np.lexsort((patterns,-counts.astype(np.int64)))[:limit]
    return dict(unique_patterns=len(patterns),truncated=len(patterns)>limit,
        most_frequent={f"0x{int(patterns[i]):08x}":int(counts[i]) for i in order})


def output_metrics(actual,reference):
    a,r=np.asarray(actual),np.asarray(reference)
    if a.shape!=r.shape or a.ndim!=3 or a.shape[-2:]!=(64,512):
        return dict(status="SHAPE_MISMATCH",actual_shape=list(a.shape),reference_shape=list(r.shape))
    a,r=a.astype(np.float64),r.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(r).all():
        return dict(status="NONFINITE",actual_nonfinite=int((~np.isfinite(a)).sum()),
            reference_nonfinite=int((~np.isfinite(r)).sum()))
    delta=a-r
    def rows(axes):
        an=np.sqrt(np.sum(a*a,axis=axes)); rn=np.sqrt(np.sum(r*r,axis=axes))
        error=np.sqrt(np.sum(delta*delta,axis=axes))
        norm_product=an*rn
        cosine=np.divide(np.sum(a*r,axis=axes),norm_product,
            out=np.where((an==0)&(rn==0),1.,0.),where=norm_product>0)
        return dict(actual_l2=an.tolist(),reference_l2=rn.tolist(),
            relative_l2=(error/np.maximum(rn,1e-30)).tolist(),cosine=np.clip(cosine,-1,1).tolist())
    pair_an=np.linalg.norm(a,axis=-1); pair_rn=np.linalg.norm(r,axis=-1)
    pair_error=np.linalg.norm(delta,axis=-1)/np.maximum(pair_rn,1e-30)
    pair_product=pair_an*pair_rn
    pair_cosine=np.divide(np.sum(a*r,axis=-1),pair_product,
        out=np.where((pair_an==0)&(pair_rn==0),1.,0.),where=pair_product>0)
    worst=np.argsort(pair_error.ravel())[-8:][::-1]
    result=dict(status="COMPUTED_FROM_SAVED_ARRAYS",relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(r),1e-30)),
        max_abs=float(np.abs(delta).max(initial=0)),per_query=rows((1,2)),
        per_head_across_queries=rows((0,2)),worst_query_heads=[dict(query=int(i//64),head=int(i%64),
            relative_l2=float(pair_error.ravel()[i]),actual_l2=float(pair_an.ravel()[i]),
            reference_l2=float(pair_rn.ravel()[i]),cosine=float(np.clip(pair_cosine.ravel()[i],-1,1))) for i in worst])
    return portable(result,computed=True)


def bf16_rne(value):
    value=np.ascontiguousarray(value,dtype=np.float32)
    bits=value.view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)


def cpu_merge(partials):
    """Diagnostic FP32 math, excluding zero-mass max/O exactly as the reducer."""
    o,m,s=partials
    if o.ndim!=4 or o.shape[1] not in (1,2,4) or o.shape[2:]!=(64,512) or m.shape!=o.shape[:-1] or s.shape!=m.shape:
        raise ValueError("Unexpected saved partial O/max/sum shapes")
    if any(x.dtype!=np.float32 for x in (o,m,s)):
        raise ValueError("Saved partial O/max/sum must actually be FP32")
    active=s>0
    if not np.isfinite(s).all() or np.any(s<0) or not np.isfinite(m[active]).all() or not np.isfinite(o[active]).all():
        raise ValueError("Nonfinite/negative sums or nonfinite active partials")
    maximum=np.where(active,m,-np.inf).max(axis=1,keepdims=True)
    safe_max=np.where(active.any(axis=1,keepdims=True),maximum,0.)
    weights=s*np.exp(np.where(active,m-safe_max,-np.inf))
    denominator=weights.sum(axis=1)[...,None]
    numerator=(np.where(active[...,None],o,0.)*weights[...,None]).sum(axis=1)
    result=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
    return bf16_rne(result)


def partial_profile(path):
    with np.load(path,allow_pickle=False) as saved:
        required=("partial_output","partial_maximum","partial_sum")
        if any(key not in saved for key in required):
            raise ValueError("NPZ lacks actual partial_output/partial_maximum/partial_sum")
        o,m,s=(saved[key] for key in required)
    result=dict(status="READ",path=str(path),sha256=sha256(path),
        output=distribution(o),maximum=distribution(m),total=distribution(s),
        maximum_bits=bit_histogram(m),total_bits=bit_histogram(s))
    if m.shape==s.shape:
        empty=s==0
        result.update(empty_rows=int(empty.sum()),empty_maximum_bits=bit_histogram(m[empty]),
            active_maximum=distribution(m[s>0]))
        if o.shape[:-1]==s.shape and o.ndim==4:
            result["empty_output_nonzero_elements"]=int(np.count_nonzero(o[empty]))
            result["split_output_norms"]=[]
            for split in range(o.shape[1]):
                values=o[:,split].astype(np.float64)
                norms=np.linalg.norm(values,axis=-1)
                result["split_output_norms"].append(dict(split=split,
                    output_l2=float(np.linalg.norm(values)),query_head_norms=distribution(norms),
                    active_rows=int((s[:,split]>0).sum())))
    return portable(result,computed=True),(o,m,s)


def _missing():
    return dict(status="NOT_AVAILABLE",reason="Field was not produced in the source results.json")


def reported_cases(source):
    controls=source.get("controls",{})
    mathematical=controls.get("mathematical",{})
    partial=controls.get("partial_reduce",{})
    graph=source.get("graph_controls",{})
    names=set(mathematical)|set(partial)|set(graph)
    result={name:dict(mathematical=copy.deepcopy(mathematical.get(name,_missing())),
        npu_vs_cpu_partial_merge=copy.deepcopy(partial.get(name,_missing())),
        graph=copy.deepcopy(graph.get(name,_missing()))) for name in sorted(names)}
    for label,cases in controls.get("zero_query",{}).items():
        for name,item in cases.items():
            # cpu_partial_merge is already inside the original zero_query
            # object. Retain it there verbatim rather than duplicating it.
            result[label+"_"+name]=dict(zero_query=copy.deepcopy(item))
    return result


def collect(results_path,*,case_filter=None):
    path=Path(results_path).expanduser().resolve()
    raw=path.read_bytes()
    source=json.loads(raw)
    report=dict(version="kv-split-readonly-failure-collector-v1",source_results=str(path),
        source_results_sha256=hashlib.sha256(raw).hexdigest(),source_results_bytes=len(raw),
        experiment_status=source.get("status","NOT_AVAILABLE"),
        experiment_version=source.get("version","NOT_AVAILABLE"),
        args=copy.deepcopy(source.get("args",{})),bridge_build=copy.deepcopy(source.get("bridge_build",{})),
        bridge_configuration=copy.deepcopy(source.get("bridge_configuration",{})),
        environment=copy.deepcopy(source.get("environment",{})),
        device_core_limits=copy.deepcopy(source.get("device_core_limits",{})),
        reported_fields="Copied without rounding or reinterpreting PASS/FAIL; absent means NOT_AVAILABLE",
        computed_precision="Derived distributions/norms use 7 significant digits; original reported fields remain unrounded",
        read_only=True,npu_executed=False,cases=reported_cases(source),partial_files={})
    if source.get("traceback"):
        report["traceback_tail"]=source["traceback"].splitlines()[-8:]
    for name,item in report["cases"].items():
        if name.startswith(("w0_split1","w128_split1")):
            missing=dict(status="NOT_AVAILABLE",reason="Exact main partial O and/or corresponding saved output/reference array absent")
            item["split1_identity_checks"]=dict(saved_final_vs_bf16_partial0=copy.deepcopy(missing),
                partial0_vs_matching_math=copy.deepcopy(missing),
                contract="S1 final must equal BF16(partial O); no max/sum weighting is needed")
    entries=source.get("partial_diagnostics",{})
    partial_paths={label:Path(item["path"]) for label,item in entries.items()
        if isinstance(item,dict) and isinstance(item.get("path"),str)}
    for partial_path in sorted((path.parent/"partial_diagnostics").glob("*.npz")):
        partial_paths.setdefault(partial_path.stem,partial_path)
    outputs_path=path.parent/"outputs.npz"
    saved_outputs=None
    if outputs_path.is_file():
        try:
            saved_outputs=np.load(outputs_path,allow_pickle=False)
            report["full_outputs"]=dict(status="AVAILABLE",path=str(outputs_path),sha256=sha256(outputs_path),
                array_names=list(saved_outputs.files))
        except Exception as error:
            report["full_outputs"]=dict(status="ERROR",error=str(error))
    else:
        report["full_outputs"]=dict(status="NOT_AVAILABLE",reason="outputs.npz does not exist; actual NPU outputs are not fabricated")
    try:
        if saved_outputs is not None:
            for name,case in report["cases"].items():
                if case_filter and case_filter not in name:
                    continue
                window=name.split("_",1)[0]
                reference_key=window+"_matching_math"
                if name in saved_outputs and reference_key in saved_outputs:
                    case["saved_output_vs_matching_math"]=output_metrics(saved_outputs[name],saved_outputs[reference_key])
        for label,original_path in sorted(partial_paths.items()):
            if case_filter and case_filter not in label:
                continue
            selected=original_path if original_path.is_file() else path.parent/"partial_diagnostics"/original_path.name
            if not selected.is_file():
                report["partial_files"][label]=dict(status="NOT_AVAILABLE",recorded_path=str(original_path))
                continue
            try:
                profile,partials=partial_profile(selected)
                if label.startswith("main_") and partials[0].ndim==4 and partials[0].shape[1]==1:
                    name=label[5:]
                    case=report["cases"].setdefault(name,{})
                    missing=dict(status="NOT_AVAILABLE",reason="Exact main partial O and/or corresponding saved output/reference array absent")
                    identity=case.setdefault("split1_identity_checks",dict(
                        saved_final_vs_bf16_partial0=copy.deepcopy(missing),
                        partial0_vs_matching_math=copy.deepcopy(missing),
                        contract="S1 final must equal BF16(partial O); no max/sum weighting is needed"))
                    if saved_outputs is not None and name in saved_outputs:
                        identity["saved_final_vs_bf16_partial0"]=output_metrics(saved_outputs[name],bf16_rne(partials[0][:,0]))
                    window=name.split("_",1)[0]
                    reference_key=window+"_matching_math"
                    if saved_outputs is not None and reference_key in saved_outputs:
                        identity["partial0_vs_matching_math"]=output_metrics(partials[0][:,0],saved_outputs[reference_key])
                if label.startswith("main_") and saved_outputs is not None and label[5:] in saved_outputs:
                    try:
                        merged=cpu_merge(partials)
                        profile["recomputed_cpu_merge_vs_saved_npu"]=output_metrics(saved_outputs[label[5:]],merged)
                        profile["merge_reference"]="FP32 m/s weighted normalized-O merge, ignores zero-mass states, then BF16 RNE"
                    except Exception as error:
                        profile["recomputed_cpu_merge_vs_saved_npu"]=dict(status="ERROR",error=str(error))
                else:
                    profile["recomputed_cpu_merge_vs_saved_npu"]=dict(status="NOT_AVAILABLE",
                        reason="No actual saved NPU output for this exact partial invocation; consult verbatim reported merge field")
                report["partial_files"][label]=profile
            except Exception as error:
                report["partial_files"][label]=dict(status="ERROR",path=str(selected),error=str(error))
    finally:
        if saved_outputs is not None:
            saved_outputs.close()
    if case_filter:
        report["case_filter"]=case_filter
        report["cases"]={name:item for name,item in report["cases"].items() if case_filter in name}
    return portable(report)


def encode_bounded(report,max_bytes=50000):
    if type(max_bytes) is not int or max_bytes<4096:
        raise ValueError("max_bytes must be an integer >=4096")
    report=copy.deepcopy(report)
    report["output_budget"]=dict(max_bytes=max_bytes,omitted_computed_fields=[])
    def encoded():
        return json.dumps(report,ensure_ascii=False,allow_nan=False,separators=(",",":"))
    def derived_metrics(value,prefix=""):
        if not isinstance(value,dict):
            return
        if value.get("status")=="COMPUTED_FROM_SAVED_ARRAYS":
            yield prefix,value
        for key,item in value.items():
            if isinstance(item,dict):
                yield from derived_metrics(item,prefix+"/"+key)
    result=encoded()
    if len(result.encode())<=max_bytes:
        return result
    # Preserve each split's actual O norm and m/s distributions, but drop
    # duplicate O distribution/row-norm detail. The saved-array math result
    # still retains per-query errors and its worst individual query/head rows.
    for item in report.get("partial_files",{}).values():
        if item.get("status")!="READ":
            continue
        output=item.get("output",{})
        item["output_shape"]=output.get("shape")
        item["output_dtype"]=output.get("dtype")
        item.pop("output",None)
        for split in item.get("split_output_norms",[]):
            split.pop("query_head_norms",None)
        for key in ("maximum_bits","total_bits","empty_maximum_bits"):
            histogram=item.get(key,{})
            frequent=histogram.get("most_frequent",{})
            if len(frequent)>3:
                histogram["most_frequent"]=dict(list(frequent.items())[:3])
                histogram["truncated"]=True
                histogram["retained_patterns"]=3
        metric=item.get("recomputed_cpu_merge_vs_saved_npu",{})
        metric.pop("worst_query_heads",None)
    report["output_budget"]["partial_detail_compacted"]=True
    result=encoded()
    if len(result.encode())<=max_bytes:
        return result
    # Retain all reported gates. Compact only our derived distributions and
    # split norms; raw arrays remain available at the recorded hashed NPZ.
    for item in report.get("partial_files",{}).values():
        if item.get("status")!="READ":
            continue
        for key in ("output","maximum","total","active_maximum"):
            if key not in item:
                continue
            values=item.get(key,{})
            item[key]={k:v for k,v in values.items() if k in
                ("shape","dtype","finite","nan","positive_inf","negative_inf","min","max","mean")}
        for split in item.get("split_output_norms",[]):
            if "query_head_norms" not in split:
                continue
            values=split.get("query_head_norms",{})
            split["query_head_norms"]={k:v for k,v in values.items() if k in ("min","max","mean","nan")}
    report["output_budget"]["derived_distributions_compacted"]=True
    result=encoded()
    if len(result.encode())<=max_bytes:
        return result
    # Finally remove derived full-head vectors, retaining worst query/heads.
    for name,metric in derived_metrics(report):
        if "per_head_across_queries" in metric:
            metric.pop("per_head_across_queries")
            report["output_budget"]["omitted_computed_fields"].append(name+"/per_head_across_queries")
    result=encoded()
    if len(result.encode())<=max_bytes:
        return result
    # Keep original scalar math verdicts and ALL original CPU-merge fields
    # unchanged. Only under the byte budget, omit repeated math prose/rows.
    for item in report.get("cases",{}).values():
        metric=item.get("mathematical")
        if metric is None:
            metric=item.get("zero_query",{}).get("mathematical")
        if isinstance(metric,dict):
            for key in ("rows","reference","reference_contract"):
                metric.pop(key,None)
    report["output_budget"]["reported_math_fields_omitted"]=["rows","reference","reference_contract"]
    report["output_budget"]["original_cpu_merge_fields_unchanged"]=True
    result=encoded()
    if len(result.encode())<=max_bytes:
        return result
    for _,metric in derived_metrics(report):
        if "worst_query_heads" in metric:
            metric["worst_query_heads"]=metric["worst_query_heads"][:2]
    for label,item in report.get("partial_files",{}).items():
        if not label.startswith("main_"):
            item.pop("active_maximum",None)
        metric=item.get("recomputed_cpu_merge_vs_saved_npu",{})
        if isinstance(metric,dict) and metric.get("status")=="COMPUTED_FROM_SAVED_ARRAYS":
            item["recomputed_cpu_merge_vs_saved_npu"]={k:v for k,v in metric.items()
                if k in ("status","relative_l2","max_abs")}
    report["output_budget"]["worst_query_heads_limit"]=2
    report["output_budget"]["derived_cpu_merge_detail_compacted"]=True
    result=encoded()
    if len(result.encode())<=max_bytes:
        return result
    # The source report's CPU-merge comparison remains verbatim in cases.
    # Prefer retaining the new independent S1 identity checks over this
    # redundant, additionally recomputed comparison when nearing the limit.
    for item in report.get("partial_files",{}).values():
        item.pop("merge_reference",None)
        metric=item.get("recomputed_cpu_merge_vs_saved_npu",{})
        if metric.get("status")=="COMPUTED_FROM_SAVED_ARRAYS":
            item.pop("recomputed_cpu_merge_vs_saved_npu",None)
    report["output_budget"]["redundant_recomputed_merge_omitted"]=True
    omitted=report["output_budget"].get("omitted_computed_fields",[])
    if len(omitted)>1:
        report["output_budget"]["omitted_computed_fields"]=[
            f"{len(omitted)} derived per_head_across_queries arrays omitted; --case retains full head detail"]
    base_directory=Path(report["source_results"]).parent
    for item in report.get("partial_files",{}).values():
        if item.get("status")=="READ" and isinstance(item.get("path"),str):
            try:
                item["path"]=str(Path(item["path"]).relative_to(base_directory))
            except ValueError:
                pass
    report["output_budget"]["local_partial_paths_relative_to_source_results_directory"]=True
    result=encoded()
    if len(result.encode())<=max_bytes:
        return result
    # A genuine large run should be collected one case at a time. Never return
    # a silently truncated/mislabelled set of original gates as a complete run.
    fallback=dict(version=report["version"],status="OUTPUT_BUDGET_EXCEEDED",read_only=True,npu_executed=False,
        source_results=report["source_results"],source_results_sha256=report["source_results_sha256"],
        experiment_status=report["experiment_status"],max_bytes=max_bytes,
        required_bytes_after_compaction=len(result.encode()),
        available_cases=list(report.get("cases",{}))[:64],available_case_count=len(report.get("cases",{})),
        action="Run again with --case w0_split1 (or another listed name); no NPU rerun or NPZ upload required")
    return json.dumps(fallback,ensure_ascii=False,allow_nan=False,separators=(",",":"))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    parser.add_argument("--case",dest="case_filter")
    parser.add_argument("--max-bytes",type=int,default=50000)
    args=parser.parse_args(argv)
    try:
        output=encode_bounded(collect(args.results,case_filter=args.case_filter),args.max_bytes)
    except Exception as error:
        print(json.dumps(dict(status="ERROR",read_only=True,npu_executed=False,error=str(error)),ensure_ascii=False))
        return 1
    print(output)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
