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
    report["launch_geometry"]=dict(queries=args.queries,aic_count=args.aic_count,
        aiv_count=2*args.aic_count,query_tiles_per_aic=args.queries//args.aic_count,
        aic_selection="auto" if args.aic_auto else "explicit")
    plans={name:bridge.prepare(base.query,args.keys,runner.capacity,args.aic_count,
                              base.scale,args.return_stats)
        for name in ("original_c4_direct","single_pass_w0","single_pass_w128")}
    inputs=(base.query,runner.cold_cache,base.indices,base.table,base.cu,base.lengths,
            runner.hot_cache,runner.hot_rope,tags,positions)
    calls=dict(installed_c4=lambda:base._read("c4",base.indices,cache=runner.cold_cache,statistics=False)[0],
        original_c4_direct=lambda:plans["original_c4_direct"].run(*inputs,0,True)[0],
        single_pass_w0=lambda:plans["single_pass_w0"].run(*inputs,0,False)[0],
        single_pass_w128=lambda:plans["single_pass_w128"].run(*inputs,128,False)[0],
        installed_twopass=lambda:runner.call())
    # The independent mathematical path decodes the actual original C4 bytes
    # and evaluates one FP64 softmax over the unchanged complete selected set.
    decoded=runner.decoded_kv()
    cold_math=attention_reference(data,kv_c4=decoded)
    mixed_math=attention_reference(data,kv_c4=decoded,window=128,hot_kv=data["kv"])
    bf16_math=attention_reference(data)
    references={name:(mixed_math if name in ("single_pass_w128","installed_twopass") else cold_math)
                for name in calls}
    report["controls"]={}
    outputs={}
    passed=True
    arrays=dict(cold_mathematical_reference=cold_math,mixed_mathematical_reference=mixed_math)
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
    report=dict(version="qsfa-recent128-single-pass-a5-v1.1-perf-defaults",status="STARTED",
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
