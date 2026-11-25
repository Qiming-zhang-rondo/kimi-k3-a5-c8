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
        for name in ("relative_l2", "worst_row_relative_l2"):
            metric = layer.get("control_max", {}).get(name)
            if not isinstance(metric, (int, float)) or not math.isfinite(metric) or not 0 <= metric <= .001:
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
