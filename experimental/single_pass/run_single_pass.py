#!/usr/bin/env python3
"""A5 smoke test of a real one-launch cold-C4/hot-BF16 reader.

Inputs/quantization come from the unchanged pinned experiment. Two-pass native
attention is used only as an independent control; the candidate calls one ASC
entry with the original unsplit TopK. Default: accuracy only, no timing claim.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"direct_bridge"))


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def select_aic_count(queries, requested=None):
    if type(queries) is not int or not 1 <= queries <= 1024:
        raise ValueError("Require Q1..1024")
    supported=(1,2,4,8,16)
    if requested is None:
        return max(count for count in supported if count <= queries and queries % count == 0)
    if type(requested) is not int or requested not in supported or requested > queries or queries % requested:
        raise ValueError("Require AIC1/2/4/8/16 dividing Q")
    return requested


def plan_modes(routed_ab=False,fast_pair_ab=False,early_hot_ab=False):
    """Explicit entries share one ABI/library; never load competing symbol sets."""
    modes=dict(original_c4_direct=(0,True,False,False,False),single_pass_w0=(0,False,False,False,False),
               single_pass_w128=(128,False,False,False,False))
    if routed_ab or fast_pair_ab or early_hot_ab:
        modes["single_pass_w128_routed"]=(128,False,True,False,False)
    if fast_pair_ab or early_hot_ab:
        modes["single_pass_w128_fast_pair"]=(128,False,False,True,False)
    if early_hot_ab:
        modes["single_pass_w128_early_hot"]=(128,False,False,False,True)
    return modes


def plan_calls(plans,inputs,modes):
    calls={}
    for name,mode in modes.items():
        calls[name]=lambda name=name,mode=mode:plans[name].run(*inputs,*mode)[0]
    return calls


def prepare_stats_ab(bridge,query,keys,capacity,aic,scale,inputs):
    """Four independent outputs; only runtime stats/reader fields differ."""
    specs={}
    for stats in (False,True):
        suffix="stats_on" if stats else "stats_off"
        specs["original_c4_"+suffix]=(stats,(0,True,False,False,False))
        specs["fast_pair_"+suffix]=(stats,(128,False,False,True,False))
    plans={name:bridge.prepare(query,keys,capacity,aic,scale,spec[0]) for name,spec in specs.items()}
    calls=plan_calls(plans,inputs,{name:spec[1] for name,spec in specs.items()})
    return plans,calls,specs


def stats_ab_pairs():
    return (("fast_pair_vs_original_stats_off","original_c4_stats_off","fast_pair_stats_off"),
            ("fast_pair_vs_original_stats_on","original_c4_stats_on","fast_pair_stats_on"),
            ("original_stats_on_vs_off","original_c4_stats_off","original_c4_stats_on"),
            ("fast_pair_stats_on_vs_off","fast_pair_stats_off","fast_pair_stats_on"))


def routed_index_patterns(indices,positions,window=128):
    """Correctness-only permutations of the exact selected set, never timed."""
    import numpy as np
    ids=np.asarray(indices)
    positions=np.asarray(positions)
    if ids.ndim!=2 or positions.shape!=(ids.shape[0],) or ids.shape[0]<1:
        raise ValueError("Require [Q,TopK] indices and [Q] positions")
    if not np.issubdtype(ids.dtype,np.integer) or not np.issubdtype(positions.dtype,np.integer):
        raise ValueError("Indices and positions must be integer arrays")
    if type(window) is not int or window<1 or np.any(ids<0) or np.any(ids>positions[:,None]):
        raise ValueError("Require positive window and nonnegative causal indices")
    grouped=[]
    alternating=[]
    for row,position in zip(ids,positions):
        is_hot=row>=max(0,int(position)-window+1)
        hot,cold=row[is_hot],row[~is_hot]
        grouped.append(np.concatenate((hot,cold)))
        paired=min(hot.size,cold.size)
        prefix=np.stack((hot[:paired],cold[:paired]),axis=1).reshape(-1)
        alternating.append(np.concatenate((prefix,hot[paired:],cold[paired:])))
    return dict(hot_then_cold=np.ascontiguousarray(grouped,dtype=ids.dtype),
                alternating_prefix=np.ascontiguousarray(alternating,dtype=ids.dtype))


def execute(args,report,persist):
    import numpy as np
    from pinned_payload import activate_payload
    report["payload"]=activate_payload(args.payload_dir)
    from kv_native_twopass import NativeTwoPassBackend
    from variable_bf16_runner import VariableBf16HotRunner,validate_variable_case
    from kv_window_common import attention_reference,save_case,synthetic_case
    from test_kv_hot_fused_a5 import compare,acceptable,paired_benchmark
    from kv_window_backend import benchmark
    from run_direct_ab import graph_replay_control
    from load_single_pass import load_single_pass
    use_fast_pair=args.fast_pair_ab or args.early_hot_ab
    use_routed=args.routed_ab or use_fast_pair

    data=validate_variable_case(synthetic_case(batch=1,queries=args.queries,keys=args.keys,
        heads=64,dim=512,rope_dim=64,topk=2048,page=128,seed=args.seed),
        queries=args.queries,keys=args.keys)
    save_case(args.output/"inputs.npz",data)
    backend=NativeTwoPassBackend(data,device=args.device)
    runner=VariableBf16HotRunner(backend,writer="native")
    base,torch=backend.base,backend.base.torch
    report["environment"]=backend.environment
    report["reference_implementation"]=runner.metadata
    # All conversion/allocations are preparation, outside every candidate call.
    tags_cpu=np.asarray(runner.plan["tags"][0],dtype=np.int64)
    positions_cpu=np.asarray(runner.positions_cpu,dtype=np.int64)
    tags=base.tensor(tags_cpu,torch.int64)
    positions=base.tensor(positions_cpu,torch.int64)
    require(np.array_equal(base.cu.cpu().numpy(),[args.queries]),"Query lengths mismatch")
    require(np.array_equal(base.lengths.cpu().numpy(),[args.keys]),"KV lengths mismatch")
    table=base.table.cpu().numpy()
    require(np.array_equal(np.sort(table[0]),np.arange(args.keys//128)),"Block table must be a physical-page permutation")
    ids=base.indices.cpu().numpy()[:,0]
    require(all(len(set(row.tolist()))==2048 for row in ids),"Every row requires unique original TopK")
    require(bool(np.all((ids>=0)&(ids<=positions_cpu[:,None]))),"TopK contains invalid/future IDs")
    hot=(ids>=positions_cpu[:,None]-127)
    require(bool(np.all(hot.sum(axis=1)==128)),"Controlled experiment requires 128 selected hot tokens per row")
    selected=ids[hot]
    require(np.array_equal(tags_cpu[selected%runner.capacity],selected),"Ring tags do not cover selected hot tokens")
    require(tuple(runner.cold_cache.shape)==(args.keys//128,128,1,400),"Unexpected native C4 packing")
    require(runner.cold_cache.dtype==torch.int8,"Native C4 cache must keep INT8 packed storage")
    report["preflight"]=dict(passed=True,queries=args.queries,keys=args.keys,
        ring_capacity=int(runner.capacity),window=128,selected_cold_per_query=1920,
        selected_hot_per_query=128,block_table="validated permutation; never assumed identity",
        quantizer="unchanged pinned baseline",indices="original unsplit TopK2048")
    bridge=load_single_pass(args.library,sdk_root=args.sdk_root,
        build_directory=args.build_directory,verbose=args.verbose)
    report["bridge_build"]=getattr(bridge,"_single_pass_build_identity",{})
    report["bridge_configuration"]=dict(bridge.configuration())
    if use_routed:
        require(report["bridge_configuration"].get("routed_reader_available") is True,
                "--routed-ab requires the rebuilt library with QsfaSinglePassLaunchRouted")
    if use_fast_pair or args.stats_ab:
        require(report["bridge_configuration"].get("fast_pair_reader_available") is True,
                "--fast-pair-ab requires the rebuilt library with QsfaSinglePassLaunchFastPair")
    if args.early_hot_ab:
        require(report["bridge_configuration"].get("early_hot_reader_available") is True,
                "--early-hot-ab requires the rebuilt library with QsfaSinglePassLaunchEarlyHot")
    report["launch_geometry"]=dict(queries=args.queries,aic_count=args.aic_count,
        aiv_count=2*args.aic_count,query_tiles_per_aic=args.queries//args.aic_count,
        aic_selection="auto" if args.aic_auto else "explicit")
    modes=plan_modes(args.routed_ab,args.fast_pair_ab,args.early_hot_ab)
    report["reader_entries"]={name:dict(window=mode[0],original_c4=mode[1],routed=mode[2],
                                      fast_pair=mode[3],early_hot=mode[4])
                              for name,mode in modes.items()}
    plans={name:bridge.prepare(base.query,args.keys,runner.capacity,args.aic_count,
                              base.scale,args.return_stats) for name in modes}
    inputs=(base.query,runner.cold_cache,base.indices,base.table,base.cu,base.lengths,
            runner.hot_cache,runner.hot_rope,tags,positions)
    calls=dict(installed_c4=lambda:base._read("c4",base.indices,cache=runner.cold_cache,statistics=False)[0])
    calls.update(plan_calls(plans,inputs,modes))
    calls["installed_twopass"]=lambda:runner.call()
    # The independent mathematical path decodes the actual original C4 bytes
    # and evaluates one FP64 softmax over the unchanged complete selected set.
    decoded=runner.decoded_kv()
    cold_math=attention_reference(data,kv_c4=decoded)
    mixed_math=attention_reference(data,kv_c4=decoded,window=128,hot_kv=data["kv"])
    bf16_math=attention_reference(data)
    mixed_names={name for name,mode in modes.items() if mode[0]==128}|{"installed_twopass"}
    references={name:(mixed_math if name in mixed_names else cold_math) for name in calls}
    report["controls"]={}
    outputs={}
    passed=True
    arrays=dict(cold_mathematical_reference=cold_math,mixed_mathematical_reference=mixed_math)
    stats_plans,stats_calls,stats_specs={},{},{}
    with torch.inference_mode():
        for name,call in calls.items():
            tensor=call()
            output=backend.host(tensor).copy()
            outputs[name]=output
            arrays[name]=output
            metric=compare(output,references[name])
            metric["passed"]=acceptable(metric,.01)
            metric["tolerance"]=.01
            metric["reference_contract"]="Independent FP64 whole-selected-set softmax; no intermediate BF16 states"
            report["controls"][name]=metric
            passed=passed and metric["passed"]
            report.setdefault("accuracy_vs_full_bf16",{})[name]=compare(output,bf16_math)
            if name in plans and args.return_stats:
                _,maximum,total=plans[name].outputs()
                m,s=backend.host(maximum),backend.host(total)
                arrays[name+"_maximum"],arrays[name+"_sum"]=m,s
                healthy=bool(np.isfinite(m).all() and np.isfinite(s).all() and np.all(s>0))
                report["controls"][name]["returned_stats_finite_positive_sum"]=healthy
                passed=passed and healthy
            persist()
        if args.stats_ab:
            stats_plans,stats_calls,stats_specs=prepare_stats_ab(bridge,base.query,args.keys,
                runner.capacity,args.aic_count,base.scale,inputs)
            section=dict(status="STARTED",scope="same-library 2x2 runtime stats toggle; prepared read only; no ASC rebuild",
                         configuration={name:dict(return_stats=spec[0],window=spec[1][0],
                             original_c4=spec[1][1],fast_pair=spec[1][3]) for name,spec in stats_specs.items()},
                         shared_inputs=True,independent_prepared_outputs=True,
                         controls={},cross_checks={},graph_controls={},timings={})
            report["stats_ab"]=section
            stats_outputs={}
            persist()
            for name,call in stats_calls.items():
                actual=backend.host(call()).copy()
                stats_outputs[name]=actual
                arrays["stats_ab_"+name]=actual
                reference=cold_math if stats_specs[name][1][1] else mixed_math
                metric=compare(actual,reference)
                metric.update(passed=acceptable(metric,.01),tolerance=.01,
                              reference_contract="Independent FP64 whole-selected-set softmax")
                if stats_specs[name][0]:
                    _,maximum,total=stats_plans[name].outputs()
                    m,s=backend.host(maximum),backend.host(total)
                    healthy=bool(np.isfinite(m).all() and np.isfinite(s).all() and np.all(s>0))
                    metric["returned_stats_finite_positive_sum"]=healthy
                    metric["passed"]=metric["passed"] and healthy
                    arrays["stats_ab_"+name+"_maximum"]=m
                    arrays["stats_ab_"+name+"_sum"]=s
                section["controls"][name]=metric
                persist()
            for reader in ("original_c4","fast_pair"):
                metric=compare(stats_outputs[reader+"_stats_on"],stats_outputs[reader+"_stats_off"])
                metric.update(passed=acceptable(metric,.001),tolerance=.001)
                section["cross_checks"][reader+"_stats_on_vs_off"]=metric
            section["all_accuracy_controls_passed"]=all(item["passed"] for group in
                (section["controls"],section["cross_checks"]) for item in group.values())
            passed=passed and section["all_accuracy_controls_passed"]
        report["cross_checks"]={
            "original_c4_direct_vs_installed_c4":compare(outputs["original_c4_direct"],outputs["installed_c4"]),
            "single_pass_w0_vs_original_c4_direct":compare(outputs["single_pass_w0"],outputs["original_c4_direct"]),
            "single_pass_w128_vs_twopass_diagnostic":compare(outputs["single_pass_w128"],outputs["installed_twopass"])}
        # Same original C4 kernel/inputs controls, independent of the mixed
        # mathematical acceptance. Mixed-vs-two-pass is diagnostic only.
        for name in ("original_c4_direct_vs_installed_c4","single_pass_w0_vs_original_c4_direct"):
            check=report["cross_checks"][name]
            check.update(passed=acceptable(check,.01),tolerance=.01)
            passed=passed and check["passed"]
        report["cross_checks"]["single_pass_w128_vs_twopass_diagnostic"]["acceptance_gate"]=False
        if use_routed:
            variant_names=["single_pass_w128_routed"]
            if use_fast_pair:
                variant_names.append("single_pass_w128_fast_pair")
            if args.early_hot_ab:
                variant_names.append("single_pass_w128_early_hot")
            for name in variant_names:
                check=compare(outputs[name],outputs["single_pass_w128"])
                check.update(passed=acceptable(check,.001),tolerance=.001,acceptance_gate=True)
                report["cross_checks"][name+"_vs_overlay"]=check
                passed=passed and check["passed"]
            # The unchanged original TopK may be ordered into homogeneous
            # blocks. Exercise both homogeneous and mixed 16-row loaders by
            # changing order only, with no membership/cache/query changes.
            # These controls are intentionally excluded from timing/graphs.
            report["routed_pattern_controls"]={}
            for pattern,pattern_ids in routed_index_patterns(ids,positions_cpu).items():
                require(bool(np.array_equal(np.sort(pattern_ids,axis=1),np.sort(ids,axis=1))),
                        "Routed pattern changed the selected set")
                permuted_inputs=list(inputs)
                permuted_inputs[2]=base.tensor(pattern_ids[:,None,:],torch.int32)
                pattern_outputs={}
                pattern_modes=dict(overlay=(128,False,False,False,False),
                                   routed=(128,False,True,False,False))
                if use_fast_pair:
                    pattern_modes["fast_pair"]=(128,False,False,True,False)
                if args.early_hot_ab:
                    pattern_modes["early_hot"]=(128,False,False,False,True)
                pattern_plans={arm:bridge.prepare(base.query,args.keys,runner.capacity,args.aic_count,
                                                 base.scale,args.return_stats)
                               for arm in pattern_modes}
                result=dict(scope="correctness-only order permutation; same selected set; not timed",
                            selected_set_unchanged=True,checks={})
                arrays["pattern_"+pattern+"_indices"]=pattern_ids
                for arm,plan in pattern_plans.items():
                    actual=backend.host(plan.run(*permuted_inputs,*pattern_modes[arm])[0]).copy()
                    pattern_outputs[arm]=actual
                    arrays["pattern_"+pattern+"_"+arm]=actual
                    metric=compare(actual,mixed_math)
                    metric.update(passed=acceptable(metric,.01),tolerance=.01,
                                  reference_contract="Independent FP64 whole-selected-set softmax")
                    result["checks"][arm+"_vs_matching_math"]=metric
                for arm in pattern_modes:
                    if arm=="overlay":
                        continue
                    metric=compare(pattern_outputs[arm],pattern_outputs["overlay"])
                    metric.update(passed=acceptable(metric,.001),tolerance=.001)
                    result["checks"][arm+"_vs_overlay"]=metric
                result["passed"]=all(check["passed"] for check in result["checks"].values())
                report["routed_pattern_controls"][pattern]=result
                passed=passed and result["passed"]
                persist()
        if args.early_hot_ab:
            # Untimed refresh stress: both readers must consume the current
            # BF16 ring on every invocation, not a stale DMA/previous tile.
            # Eight sign flips restore the original values before timing.
            # Do not add finally-based NPU restoration: any device exception
            # must propagate immediately to the outer fault-stop handler.
            require(runner.hot_cache.dtype==torch.bfloat16 and runner.hot_rope.dtype==torch.bfloat16,
                    "Early-hot refresh control requires BF16 latent and RoPE rings")
            refresh=dict(scope="untimed fast-pair vs early-hot on alternating BF16 hot latent/RoPE values; not compared to old mathematical output",
                         expected_iterations=8,completed_iterations=0,
                         hot_values_restored=False,checks=[])
            report["early_hot_refresh_controls"]=refresh
            persist()
            for iteration in range(8):
                runner.hot_cache.mul_(-1)
                runner.hot_rope.mul_(-1)
                fast_output=backend.host(plans["single_pass_w128_fast_pair"].run(
                    *inputs,128,False,False,True,False)[0]).copy()
                early_output=backend.host(plans["single_pass_w128_early_hot"].run(
                    *inputs,128,False,False,False,True)[0]).copy()
                metric=compare(early_output,fast_output)
                metric.update(iteration=iteration,hot_sign=-1 if iteration%2==0 else 1,
                              passed=acceptable(metric,.001),tolerance=.001)
                refresh["checks"].append(metric)
                refresh["completed_iterations"]=iteration+1
                refresh["hot_values_restored"]=bool((iteration+1)%2==0)
                persist()
            refresh["passed"]=all(check["passed"] for check in refresh["checks"])
            passed=passed and refresh["passed"]
        np.savez_compressed(args.output/"outputs.npz",**arrays)
        report["all_accuracy_controls_passed"]=bool(passed)
        persist()
        require(passed,"Single-pass accuracy control failed; see outputs.npz and results.json")
        if args.timing!="none":
            if args.timing=="graph":
                report["graph_controls"]={}
                for name,plan in plans.items():
                    report["graph_controls"][name]=graph_replay_control(calls[name],plan.outputs,torch,
                        backend.host,dict(compare=compare,acceptable=acceptable))
                    persist()
                if args.stats_ab:
                    for name,plan in stats_plans.items():
                        metric=graph_replay_control(stats_calls[name],plan.outputs,torch,
                            backend.host,dict(compare=compare,acceptable=acceptable))
                        if stats_specs[name][0]:
                            _,maximum,total=plan.outputs()
                            m,s=backend.host(maximum),backend.host(total)
                            healthy=bool(np.isfinite(m).all() and np.isfinite(s).all() and np.all(s>0))
                            metric["returned_stats_finite_positive_sum_after_replay"]=healthy
                            metric["passed"]=metric["passed"] and healthy
                        report["stats_ab"]["graph_controls"][name]=metric
                        persist()
                        require(metric["passed"],"Stats A/B graph replay control failed")
            report["timings"]={}
            for name in plans:
                report["timings"][name]=paired_benchmark(calls["installed_c4"],calls[name],args,benchmark)
                require(all(item.get("status")=="MEASURED" for item in report["timings"][name].values()),
                        "Paired measurement failed; partial timing is not MEASURED")
                persist()
            report["timings"]["single_pass_w128_vs_installed_twopass"]=paired_benchmark(
                calls["installed_twopass"],calls["single_pass_w128"],args,benchmark)
            require(all(item.get("status")=="MEASURED" for item in
                        report["timings"]["single_pass_w128_vs_installed_twopass"].values()),
                    "Paired measurement failed; partial timing is not MEASURED")
            if use_routed:
                report["timings"]["single_pass_w128_routed_vs_overlay"]=paired_benchmark(
                    calls["single_pass_w128"],calls["single_pass_w128_routed"],args,benchmark)
                require(all(item.get("status")=="MEASURED" for item in
                            report["timings"]["single_pass_w128_routed_vs_overlay"].values()),
                        "Routed/overlay paired measurement failed; partial timing is not MEASURED")
            if use_fast_pair:
                for label,baseline in (("overlay","single_pass_w128"),("routed","single_pass_w128_routed")):
                    key="single_pass_w128_fast_pair_vs_"+label
                    report["timings"][key]=paired_benchmark(
                        calls[baseline],calls["single_pass_w128_fast_pair"],args,benchmark)
                    require(all(item.get("status")=="MEASURED" for item in report["timings"][key].values()),
                            "Fast-pair paired measurement failed; partial timing is not MEASURED")
            if args.early_hot_ab:
                for label,baseline in (("overlay","single_pass_w128"),("fast_pair","single_pass_w128_fast_pair")):
                    key="single_pass_w128_early_hot_vs_"+label
                    report["timings"][key]=paired_benchmark(
                        calls[baseline],calls["single_pass_w128_early_hot"],args,benchmark)
                    require(all(item.get("status")=="MEASURED" for item in report["timings"][key].values()),
                            "Early-hot paired measurement failed; partial timing is not MEASURED")
            if args.stats_ab:
                for key,baseline,candidate in stats_ab_pairs():
                    report["stats_ab"]["timings"][key]=paired_benchmark(
                        stats_calls[baseline],stats_calls[candidate],args,benchmark)
                    persist()
                    require(all(item.get("status")=="MEASURED" for item in
                                report["stats_ab"]["timings"][key].values()),
                            "Stats A/B paired measurement failed; partial timing is not MEASURED")
        if args.stats_ab:
            report["stats_ab"]["status"]="VALIDATED_ONLY" if args.timing=="none" else "MEASURED"
    report["status"]="VALIDATED_ONLY" if args.timing=="none" else "MEASURED"
    report["performance_scope"]="NOT_MEASURED" if args.timing=="none" else "paired prepared read only; no writes, no serving claim"


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--payload-dir",type=Path,required=True)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--sdk-root",type=Path)
    p.add_argument("--build-directory",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",required=True,help="Explicit NPU already checked idle; never an implicit shared device")
    p.add_argument("--queries",type=int,default=4)
    p.add_argument("--keys",type=int,choices=(8192,32768),default=8192)
    p.add_argument("--aic-count",type=int,choices=(1,2,4,8,16),default=None,
        help="Default: largest supported divisor of Q, up to 16 (Q4 uses 4 AICs)")
    p.add_argument("--seed",type=int,default=20261120)
    p.add_argument("--return-stats",action="store_true")
    p.add_argument("--stats-ab",action="store_true",
        help="No ASC rebuild: same-library original C4/fast-pair x stats OFF/ON paired controls")
    p.add_argument("--routed-ab",action="store_true",
        help="Also validate and pair the distinct precision-routed entry against preserved overlay")
    p.add_argument("--fast-pair-ab",action="store_true",
        help="Also validate/pair fast-pair routing against both preserved overlay and prior routed entries")
    p.add_argument("--early-hot-ab",action="store_true",
        help="Also validate/pair early-hot scheduling against current fast-pair and preserved overlay")
    p.add_argument("--timing",choices=("none","eager","graph"),default="none")
    p.add_argument("--warmup",type=int,default=20)
    p.add_argument("--iterations",type=int,default=100)
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--fault-marker",type=Path)
    p.add_argument("--verbose",action="store_true")
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    args.aic_auto=args.aic_count is None
    try:
        args.aic_count=select_aic_count(args.queries,args.aic_count)
    except ValueError as error:
        parser().error(str(error))
    if args.warmup<0 or min(args.iterations,args.repeats)<1:
        parser().error("Invalid timing iteration counts")
    args.output=args.output.expanduser().resolve()
    marker=(args.fault_marker or args.output.parent/"DEVICE_FAULT.json").expanduser().resolve()
    if marker.exists():
        parser().error("Existing device-fault marker blocks another NPU run; recover the device first")
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(version="qsfa-recent128-single-pass-a5-v1.5-stats-ab",status="STARTED",
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        candidate_contract="one direct ASC attention launch, original unsplit TopK; no cold attention call, partition or external merge",
        model_accuracy="NOT_MEASURED",serving_performance="NOT_MEASURED")
    def persist():
        (args.output/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    persist()
    try:
        execute(args,report,persist)
    except Exception:
        detail=traceback.format_exc()
        report.update(status="CONTROL_FAILED" if "control failed" in detail else "ERROR",traceback=detail)
        if any(code in detail for code in ("507015","507035","507053","507036","507011")):
            marker.parent.mkdir(parents=True,exist_ok=True)
            marker.write_text(json.dumps(dict(status="DEVICE_FAULT",results=str(args.output/"results.json")))+"\n")
            report["device_fault_marker"]=str(marker)
    persist()
    # Always emit a machine-generated small handoff; a handwritten latency
    # summary cannot establish which library/core count was actually tested.
    from collect_run_evidence import collect,portable
    evidence=args.output/"evidence_summary.json"
    evidence.write_text(json.dumps(portable(collect(args.output/"results.json")),
        ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=report["status"],results=str(args.output/"results.json"),
                          evidence_summary=str(evidence))))
    return 0 if report["status"] in ("VALIDATED_ONLY","MEASURED") else 1


if __name__=="__main__":
    raise SystemExit(main())
