#!/usr/bin/env python3
"""Validate/timestamp the experimental KV-split reader against same-library V4.

Each timed candidate includes its partial-attention launch AND stable reduction.
No CPU mathematical work, quantization or cache write is timed. This is an A5
prototype harness; host tests do not establish device correctness or speed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"direct_bridge"))

from kv_split_math import PartialState,merge_partial_states,split_topk_columns
from run_single_pass import select_aic_count

FAULT_CODES=("507015","507035","507053","507036","507011",
             "ACL_ERROR_RT_AICORE_EXCEPTION","GM invalid")
FAILURE_CASES=("w0_split1","w128_split1")
VERIFICATION_PAIRS=(("installed_c4","w0_v4"),("installed_c4","w0_split4"),
    ("installed_c4","w128_split4"),("w0_v4","w0_split4"),
    ("w128_v4","w128_split4"),("w0_split4","w128_split4"))
FULL_PRECISION_PAIRS=(("installed_c4","installed_bf16"),
    ("installed_c4","w128_split4"),("installed_bf16","w128_split4"))


class ControlFailed(RuntimeError):
    pass


def require(condition,message):
    if not condition:
        raise ControlFailed(message)


def candidate_specs(windows,splits):
    if not windows or len(set(windows))!=len(windows) or any(type(w) is not int or w not in (0,128) for w in windows):
        raise ValueError("Select unique windows from 0 and 128")
    if not splits or len(set(splits))!=len(splits) or any(type(s) is not int or s not in (1,2,4) for s in splits):
        raise ValueError("Select unique split counts from 1, 2, 4")
    return {f"w{window}_split{count}":dict(window=window,splits=count)
            for window in windows for count in splits}


def all_passed(value):
    """A nested gate tree must contain explicit passing leaves, never empties."""
    if not isinstance(value,dict) or not value:
        return False
    if "passed" in value:
        return value["passed"] is True
    return all(all_passed(child) for child in value.values())


def supported_empty_maximum(maximum):
    """Known native empty sentinels, not an arbitrary finite-value waiver."""
    import numpy as np
    maximum=np.asarray(maximum)
    negative_finite_limit=np.float32(-np.finfo(np.float32).max)
    return (maximum==0)|np.isneginf(maximum)|(maximum==negative_finite_limit)


def failure_diagnostic_paths(results_path):
    return [Path(results_path).parent/("failure_small_"+case+".json") for case in FAILURE_CASES]


def write_failure_diagnostics(results_path):
    """CPU-only post-failure export; preserve the source JSON and failed status.

    Each case is bounded to 50KB. No tensor reads or device calls are made;
    only states already saved by the runner are consulted.
    """
    paths=failure_diagnostic_paths(results_path)
    for case,path in zip(FAILURE_CASES,paths):
        try:
            from collect_kv_split_failure import collect,encode_bounded
            text=encode_bounded(collect(results_path,case_filter=case))+"\n"
        except Exception as error:
            text=json.dumps(dict(status="CPU_DIAGNOSTIC_EXPORT_ERROR",case=case,
                source_results=str(results_path),error=str(error),npu_executed=False),
                ensure_ascii=False)+"\n"
        try:
            with path.open("x",encoding="utf-8") as handle:
                handle.write(text)
        except OSError as error:
            print(f"Diagnostic export unavailable: {path}: {error}",file=sys.stderr)
    return paths


def partial_diagnostics(partials,path):
    """Persist unmodified device states before any validator can reject them."""
    import numpy as np
    if not isinstance(partials,(tuple,list)) or len(partials)!=3:
        raise ValueError("Diagnostic accessor needs actual partial O/max/sum")
    o,m,s=(np.asarray(value) for value in partials)
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,partial_output=o,partial_maximum=m,partial_sum=s)
    result=dict(status="SAVED_BEFORE_VALIDATION",path=str(path),
        shapes=dict(output=list(o.shape),maximum=list(m.shape),total=list(s.shape)),
        dtypes=dict(output=str(o.dtype),maximum=str(m.dtype),total=str(s.dtype)))
    if m.shape==s.shape and m.dtype==np.float32:
        empty=s==0
        values=np.ascontiguousarray(m[empty]).view(np.uint32)
        patterns,counts=np.unique(values,return_counts=True)
        result.update(empty_partial_rows=int(np.sum(empty)),
            empty_maximum_uint32_histogram={f"0x{int(bits):08x}":int(count)
                for bits,count in zip(patterns[:16],counts[:16])},
            empty_maximum_unique_patterns=len(patterns),histogram_truncated=len(patterns)>16,
            known_empty_maximum_rows=int(np.sum(supported_empty_maximum(m[empty]))),
            nonfinite_sums=int(np.sum(~np.isfinite(s))))
        if o.shape[:-1]==m.shape:
            result["empty_output_nonzero_elements"]=int(np.count_nonzero(o[empty]))
    return result


def measure_candidates(calls,specs,controls,graph_controls,args,benchmark,paired_benchmark,persist=lambda:None,results=None):
    require(all_passed(controls),"All accuracy/partial/empty controls must pass before ANY timing")
    if args.timing=="graph":
        require(all_passed(graph_controls),"Every graph replay/partial poison control must pass before ANY timing")
    result={} if results is None else results
    for name,spec in specs.items():
        baseline=f"w{spec['window']}_v4"
        timing=paired_benchmark(calls[baseline],calls[name],args,benchmark)
        require(bool(timing) and all(item.get("status")=="MEASURED" and
                    len(item.get("rounds",[]))==args.repeats for item in timing.values()),
                "Partial or failed paired timing cannot be marked MEASURED")
        result[name+"_vs_v4"]=timing
        persist()
    return result


def validate_verification_options(args):
    if getattr(args,"compare_bf16",False) and (not args.verify_comparison or args.splits!=[4]):
        raise ValueError("--compare-bf16 requires --verify-comparison and --splits 4; use run_attention_compare.py")
    if not getattr(args,"verify_comparison",False):
        return
    if set(args.windows)!={0,128} or 4 not in args.splits:
        raise ValueError("--verify-comparison requires both --windows 0 128 and --splits including 4")
    timing_modes=("none","graph","eager") if getattr(args,"compare_bf16",False) else ("graph","eager")
    if args.return_stats or args.repeats<20 or args.timing not in timing_modes:
        raise ValueError("Verification requires stats off, --repeats >=20 and --timing graph or eager")


def measure_verification(calls,controls,graph_controls,args,benchmark,paired_benchmark,
                         persist=lambda:None,results=None,pairs=VERIFICATION_PAIRS):
    """Direct pairs, including the actual installed C4 operator as an anchor."""
    require(all_passed(controls),"Verification requires all original accuracy controls")
    if args.timing=="graph":
        require(all_passed(graph_controls),"Verification requires all graph/poison controls")
    require(args.repeats>=20,"Verification needs at least 20 complete alternating rounds")
    require(all(name in calls for pair in pairs for name in pair),
        "Verification needs installed C4, old W0/V4 and both split4 windows")
    result={} if results is None else results
    for baseline,candidate in pairs:
        timing=paired_benchmark(calls[baseline],calls[candidate],args,benchmark)
        require(set(timing)=={args.timing},"Verification timing mode missing or substituted")
        for item in timing.values():
            rounds=item.get("rounds",[])
            require(item.get("status")=="MEASURED" and len(rounds)==args.repeats,
                "Incomplete verification pair cannot be MEASURED")
            for index,pair in enumerate(rounds):
                expected=["baseline","candidate"] if index%2==0 else ["candidate","baseline"]
                require(pair.get("round")==index and pair.get("order")==expected,
                    "Verification rounds must alternate AB/BA without dropping rounds")
                for clock in ("event_us","wall_us"):
                    original=pair.get("baseline",{}).get(clock,{}).get("median")
                    updated=pair.get("candidate",{}).get(clock,{}).get("median")
                    require(all(isinstance(x,(int,float)) and not isinstance(x,bool) and
                        math.isfinite(x) and x>0 for x in (original,updated)),
                        "Verification needs positive finite raw event AND wall times")
                    ratio=pair.get(clock+"_ratio")
                    require(isinstance(ratio,(int,float)) and not isinstance(ratio,bool) and
                        math.isclose(ratio,updated/original,rel_tol=1e-9),
                        "Verification ratio must match its own raw pair")
        result[candidate+"_vs_"+baseline]=dict(baseline=baseline,candidate=candidate,
            ratio_definition="candidate / baseline per round; median of ratios, not ratio of medians",
            timing=timing)
        persist()
    return result


def check_saved_verification_outputs(output_readers,partial_readers,windows,references,host,compare,acceptable,
                                     partial_host=None):
    """Read the last timed buffers; never rerun an operator to refresh them."""
    checks,snapshots={},{}
    partial_host=host if partial_host is None else partial_host
    for name,read in output_readers.items():
        actual=host(read())
        snapshots[name]=actual
        metric=compare(actual,references[windows[name]])
        metric.update(passed=bool(acceptable(metric,.01)),tolerance=.01)
        item=dict(mathematical=metric)
        if name in partial_readers:
            partials=tuple(partial_host(x) for x in partial_readers[name]())
            item["partial_reduce"],_=host_partial_control(partials,actual,compare,acceptable)
            for suffix,value in zip(("partial_o","partial_max","partial_sum"),partials):
                snapshots[name+"_"+suffix]=value
        checks[name]=item
    return checks,snapshots


def host_partial_control(partials,final,compare,acceptable):
    """Check actual device-produced FP32 normalized O/m/s and reduction."""
    import numpy as np
    from host_contract import bf16_round
    if not isinstance(partials,(tuple,list)) or len(partials)!=3:
        raise ValueError("Partial accessor must return (FP32 O, FP32 max, FP32 sum)")
    o,m,s=(np.asarray(value) for value in partials)
    if o.ndim!=4 or o.shape[0]<1 or o.shape[1] not in (1,2,4) or o.shape[2:]!=(64,512):
        raise ValueError("Partial output must be [Q,splits,64,512]")
    if m.shape!=o.shape[:-1] or s.shape!=m.shape or any(x.dtype!=np.float32 for x in (o,m,s)):
        raise ValueError("Partial max/sum must be FP32 [Q,splits,64]")
    # Native ZeroInvalidLseVF also uses exact FP32 -FLT_MAX (0xff7fffff),
    # despite one upstream comment calling it -inf. The independent math
    # helper uses canonical -inf/sum=0. All known empty representations have
    # zero mass; do not admit arbitrary finite values or poison as sentinels.
    empty=s==0
    if np.any(empty & ~supported_empty_maximum(m)):
        raise ValueError("Empty device partial max must be 0, -inf, or exact FP32 -FLT_MAX (0xff7fffff)")
    canonical_max=np.where(empty,np.float32(-np.inf),m)
    states=tuple(PartialState(o[:,i],canonical_max[:,i,:,None],s[:,i,:,None]) for i in range(o.shape[1]))
    merged=merge_partial_states(states)
    expected=bf16_round(merged.output)
    metric=compare(final,expected)
    metric.update(passed=bool(acceptable(metric,.001)),tolerance=.001,
        reference="CPU FP32 stable merge of actual device FP32 normalized O/max/sum, then BF16 RNE",
        normalized_partials=True,empty_partial_rows=int(np.sum(empty)),
        empty_maximum_contract="0, -inf, or exact FP32 -FLT_MAX; ignored only when sum=0 and O=0",
        native_zero_max_empty_rows=int(np.sum(empty&(m==0))),
        native_finite_limit_empty_rows=int(np.sum(empty&(m==np.float32(-np.finfo(np.float32).max)))))
    return metric,merged


def zero_query_partial_control(partials,prefix):
    """Count check without imposing mathematical -inf on native empty max."""
    import numpy as np
    o,m,s=(np.asarray(value) for value in partials)
    if o.ndim!=4 or o.shape[0]<1 or o.shape[1] not in (1,2,4) or o.shape[2:]!=(64,512):
        raise ValueError("Expected partial O [Q,splits,64,512]")
    if m.shape!=o.shape[:-1] or s.shape!=m.shape or any(x.dtype!=np.float32 for x in (o,m,s)):
        raise ValueError("Expected FP32 O and compact FP32 max/sum")
    if type(prefix) is not int or not 0<=prefix<=2048:
        raise ValueError("Expected original TopK valid prefix 0..2048")
    width=2048//o.shape[1]
    expected_counts=np.array([max(0,min(width,prefix-index*width))
        for index in range(o.shape[1])],dtype=np.float32)
    count_ok=bool(np.array_equal(s,np.broadcast_to(expected_counts[None,:,None],s.shape)))
    active=s>0
    max_ok=bool(np.all(m[active]==0))
    empty_ok=bool(np.all(o[~active]==0) and np.all(s[~active]==0) and
        np.all(supported_empty_maximum(m[~active])))
    return dict(passed=count_ok and max_ok and empty_ok,expected_partial_counts=expected_counts.tolist(),
        partial_counts_exact=count_ok,zero_query_active_maxima_exact=max_ok,
        empty_partial_output_and_sum_zero=empty_ok,
        empty_maximum_contract="0, -inf, or exact FP32 -FLT_MAX; zero sum excludes it from max/weights")


def execute(args,report,persist):
    import numpy as np
    from pinned_payload import activate_payload
    report["payload"]=activate_payload(args.payload_dir)
    from kv_native_twopass import NativeTwoPassBackend
    from variable_bf16_runner import VariableBf16HotRunner,validate_variable_case
    from kv_window_common import save_case,synthetic_case,attention_reference
    from test_kv_hot_fused_a5 import compare,acceptable,paired_benchmark
    from kv_window_backend import benchmark
    from run_direct_ab import graph_replay_control
    from load_single_pass import load_single_pass
    from host_contract import SinglePassContract,dense_attention_reference

    full_comparison=getattr(args,"compare_bf16",False)
    if full_comparison:
        # In this experiment a row is one query/head's D512 output vector.
        # Quantization quality is reported separately from these strict gates.
        from attention_compare_metrics import compare_outputs as compare
        from attention_compare_report import build_accuracy_report

    specs=candidate_specs(args.windows,args.splits)
    require(args.queries*max(args.splits)<=16,"KV-split prototype requires Q*split_count <= 16")
    data=validate_variable_case(synthetic_case(batch=1,queries=args.queries,keys=args.keys,
        heads=64,dim=512,rope_dim=64,topk=2048,page=128,seed=args.seed),
        queries=args.queries,keys=args.keys)
    save_case(args.output/"inputs.npz",data)
    backend=NativeTwoPassBackend(data,device=args.device)
    ring=VariableBf16HotRunner(backend,writer="native")
    base,torch=backend.base,backend.base.torch
    properties=torch.npu.get_device_properties(args.device)
    available_aic=int(getattr(properties,"cube_core_num",0))
    require(available_aic>0,"Cannot verify the device's available cube-core count")
    require(args.queries*max(args.splits)<=available_aic,
        "Candidate Q*splits exceeds this device's available AIC cores")
    host=lambda tensor:tensor.detach().float().cpu().numpy().copy()
    partial_host=lambda tensor:tensor.detach().cpu().numpy().copy() # Preserve actual partial dtype/bits.
    report["environment"]=backend.environment
    report["device_core_limits"]=dict(cube_core_num=available_aic,
        largest_attention_launch=args.queries*max(args.splits),
        reduction_aiv_blocks=16,actual_active_cores="kernel task mapping; profiler not measured")
    ids=base.indices.detach().cpu().numpy().copy()
    positions_cpu=np.asarray(ring.positions_cpu,dtype=np.int64)
    tags_cpu=np.asarray(ring.plan["tags"][0],dtype=np.int64)
    for splits in args.splits:
        split_topk_columns(ids,splits,keys=args.keys,positions=positions_cpu)
    require(np.all(ids>=0),"Timed original case must retain the full unique TopK2048")
    hot=ids[:,0]>=positions_cpu[:,None]-127
    require(bool(np.all(hot.sum(axis=1)==128)),"Controlled experiment must select 128 hot tokens per query")
    selected=ids[:,0][hot]
    require(np.array_equal(tags_cpu[selected%ring.capacity],selected),"Original BF16 ring tags missing")
    require(np.array_equal(base.cu.detach().cpu().numpy(),[args.queries]),"Actual query length mismatch")
    require(np.array_equal(base.lengths.detach().cpu().numpy(),[args.keys]),"Actual KV length mismatch")
    table_cpu=base.table.detach().cpu().numpy().copy()
    require(np.array_equal(np.sort(table_cpu[0]),np.arange(args.keys//128)),"Expected a real physical-page permutation")
    require(tuple(ring.cold_cache.shape)==(args.keys//128,128,1,400) and
        ring.cold_cache.dtype==torch.int8,"Cold cache must retain original packed400 bytes")
    report["preflight"]=dict(passed=True,queries=args.queries,keys=args.keys,heads=64,topk=2048,
        original_topk_unchanged=True,selected_cold_per_query=1920,selected_hot_per_query=128,
        native_quantizer_unchanged=True,ring_capacity=int(ring.capacity),
        split_axis="original TopK columns; no sorting, membership changes or cold/hot partition")

    bridge=load_single_pass(args.library,sdk_root=args.sdk_root,
        build_directory=args.build_directory,verbose=args.verbose)
    report["bridge_build"]=getattr(bridge,"_single_pass_build_identity",{})
    report["bridge_configuration"]=dict(bridge.configuration())
    require(hasattr(bridge,"prepare_kv_split"),"Rebuild the matching host bridge with prepare_kv_split")
    require(report["bridge_configuration"].get("load_tile_reader_available") is True,
        "Same-library measured V4 load-tile baseline is required")
    require(report["bridge_configuration"].get("kv_split_reader_available") is True,
        "Library lacks the real KV-split attention/reduction exports; never substitute Python merge")
    tags=base.tensor(tags_cpu,torch.int64)
    positions=base.tensor(positions_cpu,torch.int64)
    inputs=(base.query,ring.cold_cache,base.indices,base.table,base.cu,base.lengths,
        ring.hot_cache,ring.hot_rope,tags,positions)
    baselines={window:bridge.prepare(base.query,args.keys,ring.capacity,args.aic_count,
        base.scale,args.return_stats) for window in args.windows}
    plans={name:bridge.prepare_kv_split(base.query,args.keys,ring.capacity,spec["splits"],
        base.scale,args.return_stats) for name,spec in specs.items()}

    def baseline_call(window,current_inputs=inputs):
        # W0 intentionally preserves the original native cold dispatch. The
        # measured V4 load variant is W128-only in the unchanged old ABI.
        return baselines[window].run(*current_inputs,window,False,False,False,False,
            4 if window else 0)[0]

    calls={f"w{window}_v4":lambda window=window:baseline_call(window) for window in args.windows}
    calls.update({name:lambda name=name:plans[name].run(*inputs,specs[name]["window"])[0] for name in specs})
    installed_outputs={}
    if args.verify_comparison:
        def installed_c4_call():
            # Retain the actual captured/timed output. A post-timing eager
            # refresh could hide a failure in graph replay.
            value=base._read("c4",base.indices,cache=ring.cold_cache,statistics=False)[0]
            installed_outputs["installed_c4"]=value
            return value
        calls["installed_c4"]=installed_c4_call
    if full_comparison:
        require(args.verify_comparison,"Full BF16 comparison needs the actual installed C4 anchor")
        require(base.full_bf16_plan.uniform_count==2048 and
            base.full_bf16_plan.summary["operator_calls"]==1,
            "Full BF16 must use a single native call with all 2048 indices valid")
        def installed_bf16_call():
            value=base._read("bf16",base.indices,statistics=False)[0]
            installed_outputs["installed_bf16"]=value
            return value
        calls["installed_bf16"]=installed_bf16_call
    report["entries"]={f"w{window}_v4":dict(window=window,
        reader="same-library measured V4 load_variant4" if window else "same-library unchanged native W0 cold entry",
        attention_aic_tasks=args.aic_count,attention_launches=1,reduction_launches=0,return_stats=args.return_stats)
        for window in args.windows}
    report["entries"].update({name:dict(spec,attention_aic_tasks=args.queries*spec["splits"],
        attention_launches=1,reduction_launches=1,timed_reduction_included=True,
        normalized_partial_dtype="FP32",return_stats=args.return_stats) for name,spec in specs.items()})
    if args.verify_comparison:
        report["entries"]["installed_c4"]=dict(window=0,reader="actual installed cloud_kv_quant_sparse_flash_attention",
            same_query_cache_indices_table_lengths_scale=True,return_stats=False,
            attention_aic_tasks="selected by installed operator; not forced",
            python_entry="base._read('c4',base.indices,cache=ring.cold_cache,statistics=False)[0]")
        report["comparison_contract"]=dict(pairs=[dict(baseline=a,candidate=b) for a,b in VERIFICATION_PAIRS],
            attention_and_reduction_included=True,writes_included=False,original_topk_unchanged=True,
            installed_c4_baseline_included=True,return_stats=False,
            reported_active_cores="task mapping only; profiler not measured")
    if full_comparison:
        report["entries"]["installed_bf16"]=dict(window="full_bf16",reader="actual installed cloud_sparse_flash_attention",
            same_query_indices_table_lengths_scale=True,return_stats=False,
            attention_launches=1,reduction_launches=0,
            python_entry="base._read('bf16',base.indices,statistics=False)[0]",
            read_plan=dict(base.full_bf16_plan.summary))
        report["comparison_contract"].update(
            pairs=[dict(baseline=a,candidate=b) for a,b in FULL_PRECISION_PAIRS],
            quality_reference="shared original BF16 inputs, FP64 mathematical attention AND actual native BF16 output",
            metrics="relative L2 and cosine per query/head; different reference names never conflated",
            scope="single-layer controlled synthetic input, all 128 recent tokens selected; fixed TopK, no QLI")
    raw_arrays=dict(query=host(base.query),cold_packed=ring.cold_cache.view(torch.uint8).cpu().numpy(),
        block_table=table_cpu,topk=ids,positions=positions_cpu,
        hot_latent=host(ring.hot_cache),hot_rope=host(ring.hot_rope),hot_tags=tags_cpu)
    if full_comparison:
        raw_arrays["original_bf16_latent_paged"]=host(base.kv)
        raw_arrays["original_bf16_rope_paged"]=host(base.rope)
    if args.verify_comparison:
        report["input_identity"]={name:dict(shape=list(value.shape),dtype=str(value.dtype),
            sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
            for name,value in raw_arrays.items()}
        report["input_identity_scope"]="logical host snapshots shared by all arms; cold_packed is raw cache bytes"

    def reference(window,*,zero=False,prefix=None):
        # Only the kernel contract's declared inputs belong in this reference.
        arrays={name:raw_arrays[name] for name in
            ("query","cold_packed","block_table","topk","positions","hot_latent","hot_rope","hot_tags")}
        if zero:
            arrays["query"]=np.zeros_like(raw_arrays["query"])
        if prefix is not None:
            arrays["topk"]=ids.copy()
            arrays["topk"][:,:,prefix:]=-1
        contract=SinglePassContract(queries=args.queries,keys=args.keys,ring_capacity=int(ring.capacity),
            window=window,aic_count=args.aic_count,scale=float(base.scale))
        return dense_attention_reference(contract,**arrays)["output"]

    references={window:reference(window) for window in args.windows}
    if full_comparison:
        # synthetic_case rounds all original Q/K/RoPE to BF16 before any
        # quantization. This is not a second NPU attention masquerading as math.
        references["full_bf16"]=attention_reference(data)
    arrays={f"w{window}_matching_math":value for window,value in references.items()}
    report["controls"]=dict(mathematical={},partial_reduce={},zero_query={})
    report["cross_checks"]={}
    report["partial_diagnostics"]={}
    def save_partials(label,partials):
        record=partial_diagnostics(partials,args.output/"partial_diagnostics"/(label+".npz"))
        report["partial_diagnostics"][label]=record
        persist() # Preserve raw evidence even if the following gate raises.
    with torch.inference_mode():
        baseline_outputs={}
        for window in args.windows:
            name=f"w{window}_v4"
            tensor=calls[name]()
            require(tensor.dtype==torch.bfloat16,"V4 baseline must retain BF16 output")
            actual=host(tensor)
            baseline_outputs[window]=actual
            arrays[name]=actual
            metric=compare(actual,references[window])
            metric.update(passed=bool(acceptable(metric,.01)),tolerance=.01,
                reference="Independent FP64 whole-selected-set softmax over actual packed C4/original BF16 ring")
            report["controls"]["mathematical"][name]=metric
            persist()
        if args.verify_comparison:
            tensor=calls["installed_c4"]()
            require(tensor.dtype==torch.bfloat16,"Installed C4 must use the same BF16 final output contract")
            actual=host(tensor)
            arrays["installed_c4"]=actual
            metric=compare(actual,references[0])
            metric.update(passed=bool(acceptable(metric,.01)),tolerance=.01,
                reference="Same actual packed C4 and original TopK; independent FP64 whole-set attention")
            report["controls"]["mathematical"]["installed_c4"]=metric
            alignment=compare(actual,baseline_outputs[0])
            alignment.update(passed=bool(acceptable(alignment,.01)),tolerance=.01)
            report["controls"]["installed_c4_alignment"]=alignment
            persist()
        if full_comparison:
            tensor=calls["installed_bf16"]()
            require(tensor.dtype==torch.bfloat16,"Native BF16 comparison must retain BF16 final output")
            actual=host(tensor)
            arrays["installed_bf16"]=actual
            metric=compare(actual,references["full_bf16"])
            metric.update(passed=bool(acceptable(metric,.01)),tolerance=.01,
                reference="Original BF16 Q/K/V/RoPE, independent FP64 whole-selected-set attention")
            report["controls"]["mathematical"]["installed_bf16"]=metric
            persist()
        for name,spec in specs.items():
            tensor=calls[name]()
            partials=tuple(partial_host(value) for value in plans[name].partial_outputs())
            save_partials("main_"+name,partials)
            require(tensor.dtype==torch.bfloat16,"KV-split final output must be BF16")
            actual=host(tensor)
            arrays[name]=actual
            metric=compare(actual,references[spec["window"]])
            metric.update(passed=bool(acceptable(metric,.01)),tolerance=.01,
                reference="Independent FP64 whole-selected-set softmax; no intermediate BF16 partial O")
            report["controls"]["mathematical"][name]=metric
            gate,merged=host_partial_control(partials,actual,compare,acceptable)
            report["controls"]["partial_reduce"][name]=gate
            for suffix,value in zip(("partial_o","partial_max","partial_sum"),partials):
                arrays[name+"_"+suffix]=value
            check=compare(actual,baseline_outputs[spec["window"]])
            check.update(acceptance_gate=False,
                reason="Different segmentation/probability rounding; independent full-math gate remains authoritative")
            report["cross_checks"][name+"_vs_v4"]=check
            if args.return_stats:
                _,maximum,total=plans[name].outputs()
                m,s=host(maximum),host(total)
                healthy=bool(np.isfinite(m).all() and np.isfinite(s).all() and np.all(s>0))
                stat_ok=healthy and bool(np.allclose(m,merged.maximum,rtol=1e-5,atol=1e-5) and
                    np.allclose(s,merged.total,rtol=1e-5,atol=1e-5))
                gate["returned_stats_match_cpu_partial_merge"]=stat_ok
                gate["passed"]=gate["passed"] and stat_ok
                arrays[name+"_max"],arrays[name+"_sum"]=m,s
            persist()

        # Mandatory untimed count controls exercise empty tail partitions.
        # Keep the original causal membership/order for the active prefix.
        zero_query=torch.zeros_like(base.query)
        for label,prefix in (("zero_full",2048),("zero_padded257",257)):
            control_inputs=list(inputs)
            control_inputs[0]=zero_query
            control_ids=ids.copy()
            control_ids[:,:,prefix:]=-1
            control_inputs[2]=base.tensor(control_ids,torch.int32)
            expected={window:reference(window,zero=True,prefix=prefix) for window in args.windows}
            section={}
            report["controls"]["zero_query"][label]=section
            for name,spec in specs.items():
                actual=host(plans[name].run(*control_inputs,spec["window"])[0])
                partials=tuple(partial_host(value) for value in plans[name].partial_outputs())
                save_partials(label+"_"+name,partials)
                metric=compare(actual,expected[spec["window"]])
                metric.update(passed=bool(acceptable(metric,.01)),tolerance=.01)
                merge_gate,_=host_partial_control(partials,actual,compare,acceptable)
                count_gate=zero_query_partial_control(partials,prefix)
                section[name]=dict(passed=metric["passed"] and merge_gate["passed"] and count_gate["passed"],
                    mathematical=metric,cpu_partial_merge=merge_gate,zero_query_counts=count_gate,not_timed=True)
                persist()
        arm_reference_keys={name:specs[name]["window"] if name in specs else
            ("full_bf16" if name=="installed_bf16" else 128 if name=="w128_v4" else 0)
            for name in calls}
        if full_comparison:
            report["precision_comparison"]=build_accuracy_report(
                {name:arrays[name] for name in calls},
                {name:references[key] for name,key in arm_reference_keys.items()},
                references["full_bf16"])
        np.savez_compressed(args.output/"outputs.npz",**arrays)
        report["all_accuracy_controls_passed"]=all_passed(report["controls"])
        persist()
        require(report["all_accuracy_controls_passed"],"KV-split mathematical/partial/empty control failed")

        # Restore original complete inputs by calling their immutable closures;
        # no hot/cold cache, source query or original indices were mutated.
        if full_comparison:
            # W0 split is a diagnostic arm, not one of the three timed pairs.
            # Restore it BEFORE timing; after timing we only read saved buffers.
            for call in calls.values():
                call()
        report["graph_controls"]={}
        if args.timing=="graph":
            for window in args.windows:
                name=f"w{window}_v4"
                report["graph_controls"][name]=graph_replay_control(calls[name],baselines[window].outputs,
                    torch,host,dict(compare=compare,acceptable=acceptable))
                persist()
            if args.verify_comparison:
                report["graph_controls"]["installed_c4"]=graph_replay_control(calls["installed_c4"],
                    lambda:(),torch,host,dict(compare=compare,acceptable=acceptable))
                persist()
            if full_comparison:
                report["graph_controls"]["installed_bf16"]=graph_replay_control(calls["installed_bf16"],
                    lambda:(),torch,host,dict(compare=compare,acceptable=acceptable))
                persist()
            for name in specs:
                plan=plans[name]
                buffers=lambda plan=plan:(*plan.outputs(),*plan.partial_outputs())
                replay_states={}
                def replay_compare(*values,**kwargs):
                    # graph_replay_control invokes compare after replay/sync,
                    # before its strict acceptance gate can raise.
                    partials=tuple(partial_host(value) for value in plan.partial_outputs())
                    save_partials("graph_replay_"+name,partials)
                    replay_states["partials"]=partials
                    return compare(*values,**kwargs)
                output_gate=graph_replay_control(calls[name],buffers,torch,host,
                    dict(compare=replay_compare,acceptable=acceptable))
                partials=replay_states["partials"]
                partial_gate,_=host_partial_control(partials,
                    host(plan.outputs()[0]),compare,acceptable)
                report["graph_controls"][name]=dict(passed=output_gate["passed"] and partial_gate["passed"],
                    output_replay=output_gate,partial_states_after_poison_replay=partial_gate)
                persist()
            require(all_passed(report["graph_controls"]),"KV-split graph replay/partial poison control failed")
        report["timings"]={}
        if args.timing!="none":
            if not full_comparison:
                measure_candidates(calls,specs,report["controls"],report["graph_controls"],args,benchmark,
                    paired_benchmark,persist,results=report["timings"])
            if args.verify_comparison:
                report["verification_timings"]={}
                measure_verification(calls,report["controls"],report["graph_controls"],args,benchmark,
                    paired_benchmark,persist,results=report["verification_timings"],
                    pairs=FULL_PRECISION_PAIRS if full_comparison else VERIFICATION_PAIRS)
                torch.npu.synchronize()
                readers={f"w{window}_v4":lambda window=window:baselines[window].outputs()[0]
                    for window in args.windows}
                readers.update({name:lambda name=name:plans[name].outputs()[0] for name in specs})
                readers["installed_c4"]=lambda:installed_outputs["installed_c4"]
                if full_comparison:
                    readers["installed_bf16"]=lambda:installed_outputs["installed_bf16"]
                windows={name:arm_reference_keys[name] for name in readers}
                report["post_timing_controls"],snapshots=check_saved_verification_outputs(readers,
                    {name:plans[name].partial_outputs for name in specs},windows,references,
                    host,compare,acceptable,partial_host=partial_host)
                np.savez_compressed(args.output/"post_timing_outputs.npz",**snapshots)
                if full_comparison:
                    report["post_timing_precision_comparison"]=build_accuracy_report(
                        {name:snapshots[name] for name in readers},
                        {name:references[key] for name,key in windows.items()},references["full_bf16"])
                report["post_timing_scope"]=("Three primary arms: saved final timed graph/eager buffers; other diagnostic arms: "
                    "untimed restore/graph-control buffers. No refreshing operator call after timing."
                    if full_comparison else "saved final timed graph/eager buffers; no refreshing operator call")
                persist()
                require(all_passed(report["post_timing_controls"]),"Post-timing mathematical/partial controls failed")
    require(np.array_equal(base.indices.detach().cpu().numpy(),ids),"Original device TopK mutated")
    report.update(status="VALIDATED_ONLY" if args.timing=="none" else "MEASURED",
        performance_scope="NOT_MEASURED" if args.timing=="none" else
            "paired original-input prepared read including attention AND reduction; no writes or serving claim")


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--payload-dir",type=Path,required=True)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--sdk-root",type=Path)
    p.add_argument("--build-directory",type=Path)
    p.add_argument("--device",required=True,help="Explicit idle A5 device; do not stop another service")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--queries",type=int,default=4)
    p.add_argument("--keys",type=int,choices=(8192,32768),default=8192)
    p.add_argument("--aic-count",type=int,choices=(1,2,4,8,16),
        help="Original baseline AICs only; default largest divisor of Q. KV-split always uses Q*splits tasks")
    p.add_argument("--windows",type=int,nargs="+",choices=(0,128),default=[0,128])
    p.add_argument("--splits",type=int,nargs="+",choices=(1,2,4),default=[1,2,4])
    p.add_argument("--return-stats",action="store_true")
    p.add_argument("--verify-comparison",action="store_true",
        help="Add actual installed C4 and six direct AB/BA pairs; needs windows 0/128, split4, stats off, >=20 rounds")
    p.add_argument("--compare-bf16",action="store_true",
        help="Add native full BF16 and per-query/head L2/cosine; prefer run_attention_compare.py for its defaults")
    p.add_argument("--seed",type=int,default=20261120)
    p.add_argument("--timing",choices=("none","eager","graph"),default="none")
    p.add_argument("--warmup",type=int,default=20)
    p.add_argument("--iterations",type=int,default=100)
    p.add_argument("--repeats",type=int,default=10)
    p.add_argument("--fault-marker",type=Path)
    p.add_argument("--verbose",action="store_true")
    return p


def main(argv=None,*,argument_parser=None):
    cli=argument_parser or parser()
    args=cli.parse_args(argv)
    try:
        args.aic_count=select_aic_count(args.queries,args.aic_count)
        candidate_specs(args.windows,args.splits)
        if args.queries*max(args.splits)>16:
            raise ValueError("KV-split prototype requires Q*split_count <= 16")
        if args.warmup<0 or min(args.iterations,args.repeats)<1:
            raise ValueError("Invalid timing counts")
        validate_verification_options(args)
    except ValueError as error:
        cli.error(str(error))
    args.output=args.output.expanduser().resolve()
    marker=(args.fault_marker or args.output.parent/"DEVICE_FAULT.json").expanduser().resolve()
    if marker.exists():
        cli.error("Existing device-fault marker blocks another NPU run")
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(version="qsfa-attention-threeway-a5-v1" if args.compare_bf16 else
            "qsfa-recent128-kv-split-a5-v1.2-direct-baseline-verification",status="STARTED",
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        model_accuracy="NOT_MEASURED",serving_performance="NOT_MEASURED",
        scope="prepared read including partial attention plus device reduction; no writes")
    if args.compare_bf16:
        report["runner_sources"]={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            for name in ("run_attention_compare.py","run_kv_split.py","attention_compare_metrics.py",
                "attention_compare_report.py")}
    def persist():
        (args.output/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    persist()
    try:
        execute(args,report,persist)
    except Exception as error:
        detail=traceback.format_exc()
        report.update(status="CONTROL_FAILED" if isinstance(error,ControlFailed) else "ERROR",
                      traceback=detail)
        if any(code in detail for code in FAULT_CODES):
            marker.parent.mkdir(parents=True,exist_ok=True)
            marker.write_text(json.dumps(dict(status="DEVICE_FAULT",results=str(args.output/"results.json")))+"\n")
            report["device_fault_marker"]=str(marker)
    failed=report["status"] in ("ERROR","CONTROL_FAILED")
    if failed and not args.compare_bf16:
        report["failure_diagnostic_files"]=[str(path) for path in
            failure_diagnostic_paths(args.output/"results.json")]
    if args.verify_comparison:
        report["verification_evidence_directory"]=str(args.output/"verification_evidence")
    if args.compare_bf16:
        from attention_compare_report import write_summary
        write_summary(report,args.output)
        report["summary_files"]=[str(args.output/name) for name in ("summary.json","summary.csv","summary.txt")]
    persist() # Freeze source bytes before read-only collectors hash them.
    if failed and not args.compare_bf16:
        write_failure_diagnostics(args.output/"results.json")
    evidence_export={}
    if args.verify_comparison:
        try:
            from export_verification_evidence import export_evidence
            export_evidence(args.output/"results.json",args.output/"verification_evidence")
            evidence_export["verification_evidence_directory"]=report["verification_evidence_directory"]
        except Exception as error:
            evidence_export["verification_evidence_export_error"]=str(error)
    print(json.dumps(dict(status=report["status"],results=str(args.output/"results.json"),
        failure_diagnostic_files=report.get("failure_diagnostic_files",[]),
        summary_files=report.get("summary_files",[]),**evidence_export)))
    if args.compare_bf16:
        print((args.output/"summary.txt").read_text())
    return 0 if report["status"] in ("VALIDATED_ONLY","MEASURED") else 1


if __name__=="__main__":
    raise SystemExit(main())
