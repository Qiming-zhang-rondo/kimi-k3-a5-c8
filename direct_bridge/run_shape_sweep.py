#!/usr/bin/env python3
"""Sequential A5 shape/tile/core sweep; preserve every measured case and failure.

No device initialization here. The child harness owns unchanged full accuracy
and graph gates. Hardware faults stop the entire sweep, never just one case.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def integers(text):
    values = list(dict.fromkeys(int(x.strip()) for x in text.split(",")))
    if not values or any(x < 1 for x in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers")
    return values


def cases(queries, keys, merge_rows, aic_counts):
    """Large query count and per-AIV head-row batch are independent axes."""
    output = []
    for tile in merge_rows:
        for key in keys:
            for query in queries:
                for count in dict.fromkeys(min(query, n) for n in aic_counts):
                    if query > 1024 or query > key or key not in (8192, 32768):
                        raise ValueError("Supported scan: 1<=Q<=1024, K8192/32768, Q<=K")
                    if count not in (1, 2, 4, 8, 16) or query % count:
                        raise ValueError("AIC count must be a supported divisor of the query count")
                    if tile not in (4, 8, 16, 32, 64):
                        raise ValueError("Merge rows must be 4, 8, 16, 32 or 64")
                    output.append(dict(queries=query, keys=key, merge_rows=tile, aic_count=count,
                        name=f"q{query}_k{key}_r{tile}_aic{count}",
                        status="DEGENERATE_ALIAS_OF32" if tile == 64 else "PENDING",
                        skip_reason=("H64 / two AIVs: at most 32 head rows per AIV; Q increases query tasks, not head rows" if tile == 64 else None)))
    # Return broad Q coverage first, rather than finishing all tuning variants
    # of Q4 before the first large-chunk result becomes visible.
    def priority(case):
        if case["status"] != "PENDING":
            tier = 3
        elif case["aic_count"] == min(case["queries"], max(aic_counts)):
            tier = 0 if case["merge_rows"] == 16 else 1
        else:
            tier = 2
        return (tier, merge_rows.index(case["merge_rows"]), keys.index(case["keys"]),
                queries.index(case["queries"]), aic_counts.index(next(n for n in aic_counts if min(case["queries"],n)==case["aic_count"])))
    return sorted(output,key=priority)


def cache_options(directory):
    values = {}
    cache = Path(directory)/"CMakeCache.txt"
    if not cache.is_file():
        raise ValueError(f"Validated CMakeCache.txt missing: {cache}")
    allowed = {"SFA_SDK_ROOT", "SFA_ASC_ROOT", "SFA_SDK_INCLUDE_DIRS", "ASC_DIR",
               "CMAKE_ASC_COMPILER", "CMAKE_ASC_ARCHITECTURES", "CMAKE_ASC_FLAGS"}
    for line in cache.read_text().splitlines():
        if line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=",1)[0]:
            continue
        typed, value = line.split("=",1)
        key = typed.split(":",1)[0]
        if key in allowed and value and not value.endswith("-NOTFOUND"):
            values[key] = value
    if not values.get("SFA_SDK_ROOT"):
        raise ValueError("Validated cache has no SFA_SDK_ROOT; no toolkit guessing")
    return [f"-D{key}={value}" for key,value in sorted(values.items())]


def execute(command, log):
    with Path(log).open("w") as stream:
        stream.write(json.dumps(command,ensure_ascii=False)+"\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def summarize(report):
    result = dict(status=report.get("status"), version=report.get("version"),
                  bridge_configuration=report.get("bridge_configuration"), pairs={})
    for name, timing in report.get("timings",{}).items():
        rows = timing.get("graph",{}).get("rounds",[])
        if not rows:
            continue
        ratios = [row["event_us_ratio"] for row in rows]
        result["pairs"][name] = dict(rounds=len(rows),
            baseline_us=statistics.median(row["baseline"]["event_us"]["median"] for row in rows),
            candidate_us=statistics.median(row["candidate"]["event_us"]["median"] for row in rows),
            ratio=statistics.median(ratios), ratio_min=min(ratios), ratio_max=max(ratios))
    result["failure"] = report.get("traceback")
    return result


def write_summary(directory, report):
    (directory/"matrix.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    with (directory/"summary.csv").open("w",newline="") as stream:
        fields = ["name","queries","keys","merge_rows","aic_count","status",
                  "fused_us","installed_us","fused_installed_ratio","ratio_min","ratio_max"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for case in report["cases"]:
            row={key:case.get(key) for key in fields}
            pair=case.get("measurement",{}).get("pairs",{}).get("direct_fused_vs_installed_native_twopass",{})
            row.update(fused_us=pair.get("candidate_us"),installed_us=pair.get("baseline_us"),
                fused_installed_ratio=pair.get("ratio"),ratio_min=pair.get("ratio_min"),ratio_max=pair.get("ratio_max"))
            writer.writerow(row)
    best={}
    for case in report["cases"]:
        pair=case.get("measurement",{}).get("pairs",{}).get("direct_fused_vs_installed_native_twopass",{})
        if case["status"]!="MEASURED" or "ratio" not in pair:
            continue
        key=(case["queries"],case["keys"])
        if key not in best or pair["ratio"]<best[key][1]["ratio"]:
            best[key]=(case,pair)
    with (directory/"best_by_shape.csv").open("w",newline="") as stream:
        fields=["queries","keys","merge_rows","aic_count","fused_us","installed_us","paired_ratio","ratio_min","ratio_max"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for key,(case,pair) in sorted(best.items()):
            writer.writerow(dict(queries=key[0],keys=key[1],merge_rows=case["merge_rows"],aic_count=case["aic_count"],
                fused_us=pair["candidate_us"],installed_us=pair["baseline_us"],paired_ratio=pair["ratio"],
                ratio_min=pair["ratio_min"],ratio_max=pair["ratio_max"]))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-build",type=Path,default=Path("/tmp/sfa-direct-c4-opt-v4/build-r16"))
    parser.add_argument("--payload-dir",type=Path,default=Path("/tmp/sfa-direct-payload"))
    parser.add_argument("--work-dir",type=Path,required=True)
    parser.add_argument("--device",required=True)
    parser.add_argument("--fault-marker",type=Path,default=Path("/tmp/sfa-direct-c4-ab-v3/DEVICE_FAULT.json"))
    parser.add_argument("--queries",type=integers,default=integers("4,32,64,128,256,512,1024,1"))
    parser.add_argument("--keys",type=integers,default=integers("8192,32768"))
    parser.add_argument("--merge-rows",type=integers,default=integers("16,8,32,64"))
    parser.add_argument("--aic-counts",type=integers,default=integers("4,8,16"))
    parser.add_argument("--iterations",type=int,default=100)
    parser.add_argument("--repeats",type=int,default=10)
    parser.add_argument("--warmup",type=int,default=20)
    parser.add_argument("--packed-candidate",action="store_true")
    parser.add_argument("--dry-run",action="store_true",help="Print actual case plan without cache reads, build or device work")
    args=parser.parse_args(argv)
    plan=cases(args.queries,args.keys,args.merge_rows,args.aic_counts)
    if args.dry_run:
        print(json.dumps(dict(cases=plan,actual_runs=sum(x["status"]=="PENDING" for x in plan)),indent=2))
        return 0
    if args.fault_marker.exists():
        raise RuntimeError(f"Existing device fault marker: {args.fault_marker}; stop, do not reset or bypass")
    options=cache_options(args.validated_build)
    root=args.work_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    report=dict(version="sfa-shape-sweep-v1",status="RUNNING",cases=plan,
        scope="B1 H64 D512 R64 TopK2048 W128; full precision gates and per-shape same-round baselines; no serving claim",
        source_sha256=hashlib.sha256((ROOT/"direct_bridge/run_direct_ab.py").read_bytes()).hexdigest())
    builds={};write_summary(root,report)
    for case in plan:
        if case["status"] != "PENDING":
            continue
        if args.fault_marker.exists():
            report["status"]="STOPPED_DEVICE_FAULT";write_summary(root,report);return 1
        tile=case["merge_rows"];build=root/f"build-r{tile}"
        if tile not in builds:
            configure=["cmake","-S",str(ROOT/"standalone"),"-B",str(build),*options,
                "-DSFA_BUILD_PHASE_C=OFF","-DSFA_BUILD_TORCH_DIRECT=ON","-DSFA_PROBE_TEMPLATE_MODE=0",
                "-DSFA_MERGE_TILE_ROWS=16",f"-DSFA_BF16_MERGE_TILE_ROWS={tile}"]
            builds[tile]=execute(configure,root/f"configure-r{tile}.log")==0
            if builds[tile]:
                builds[tile]=execute(["cmake","--build",str(build),"--target","sfa_torch_direct","-j8"],root/f"build-r{tile}.log")==0
        if not builds[tile]:
            case["status"]="BUILD_ERROR";write_summary(root,report);continue
        output=root/case["name"]
        command=[sys.executable,str(ROOT/"direct_bridge/run_direct_ab.py"),
            "--payload-dir",str(args.payload_dir.resolve()),"--bridge-library",str(build/"libsfa_torch_direct.so"),
            "--queries",str(case["queries"]),"--keys",str(case["keys"]),
            "--direct-aic-count",str(case["aic_count"]),"--device",args.device,
            "--artifacts","summary","--timing","graph","--warmup",str(args.warmup),
            "--iterations",str(args.iterations),"--repeats",str(args.repeats),
            "--fault-marker",str(args.fault_marker.resolve()),"--output",str(output)]
        if args.packed_candidate:
            command.append("--packed-candidate")
        print(f"RUN {case['name']}",flush=True)
        started=time.monotonic();code=execute(command,root/(case["name"]+".log"))
        path=output/"results.json"
        case["exit_code"],case["elapsed_seconds"],case["results"]=code,time.monotonic()-started,str(path)
        if path.is_file():
            case["measurement"]=summarize(json.loads(path.read_text()))
            case["status"]=case["measurement"]["status"]
        else:
            case["status"]="PROCESS_ERROR"
        if code and case["status"]=="MEASURED":
            case["status"]="PROCESS_ERROR_AFTER_RESULTS"
        write_summary(root,report)
        print(f"{case['name']}: {case['status']} — {root/'summary.csv'}",flush=True)
        if case["status"]=="DEVICE_ERROR" or args.fault_marker.exists():
            report["status"]="STOPPED_DEVICE_FAULT";write_summary(root,report);return 1
    report["status"]="COMPLETE" if all(c["status"] in ("MEASURED","DEGENERATE_ALIAS_OF32") for c in plan) else "COMPLETE_WITH_FAILURES"
    write_summary(root,report)
    return 0 if report["status"]=="COMPLETE" else 1


if __name__=="__main__":
    raise SystemExit(main())
