#!/usr/bin/env python3
"""Read saved merge states on CPU; diagnose BF16 midpoint differences.

No Torch, SDK, device launch, or acceptance-threshold change. The FP64 reference
uses the saved native cold/hot states, not the original BF16 full-attention gold.
For zero-query counts 1920/128 its weights are exactly 15/16 and 1/16.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

LIMIT = .001


def bf16_neighbors(value):
    """Bracket FP64 directly, avoiding FP64->FP32->BF16 double rounding."""
    codes = np.arange(65536, dtype=np.uint32)
    codes = codes[(codes & 0x7f80) != 0x7f80]  # Drop Inf/NaN bit patterns first.
    values = (codes << 16).view(np.float32).astype(np.float64)
    good = np.isfinite(values)
    values, codes = values[good], codes[good]
    order = np.argsort(values, kind="stable")
    values, codes = values[order], codes[order]
    values, first = np.unique(values, return_index=True)
    codes = codes[first]
    codes[values == 0] = 0
    value = np.asarray(value, np.float64)
    if not np.isfinite(value).all() or np.any((value < values[0]) | (value > values[-1])):
        raise ValueError("Reference is nonfinite or exceeds finite BF16 range")
    upper = np.clip(np.searchsorted(values, value), 1, len(values)-1)
    lower = upper-1
    lo, hi = values[lower], values[upper]
    midpoint = (lo+hi)/2
    tie = value == midpoint
    take_hi = (value > midpoint) | (tie & ((codes[lower] & 1) != 0))
    rounded = np.where(take_hi, hi, lo).astype(np.float32)
    return rounded, midpoint, tie, hi-lo


def bits(value):
    return np.asarray(value, np.float32).view(np.uint32) >> 16


def ordered_codes(value):
    code = bits(value).astype(np.int64)
    magnitude = code & 0x7fff
    return np.where(code & 0x8000, 0x8000-magnitude, 0x8000+magnitude)


def metrics(actual, reference):
    a, b = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Output shapes differ: {a.shape} vs {b.shape}")
    rows = []
    for index, (x, y) in enumerate(zip(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1))):
        finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
        rows.append(dict(row=index, finite=finite,
            relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(y), 1e-30)) if finite else None,
            max_abs=float(np.max(np.abs(x-y))) if finite else None))
    finite = all(row["finite"] for row in rows)
    relative = float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30)) if finite else None
    worst = max(row["relative_l2"] for row in rows) if finite else None
    return dict(finite=finite, nonfinite_count=int(np.count_nonzero(~np.isfinite(a))),
        relative_l2=relative, max_row_relative_l2=worst,
        max_abs=float(np.max(np.abs(a-b))) if finite else None,
        strict_limit=LIMIT, strict_passed=bool(finite and relative <= LIMIT and worst <= LIMIT), rows=rows)


def ulp_summary(actual, reference):
    actual, reference = np.asarray(actual, np.float32), np.asarray(reference, np.float32)
    finite = np.isfinite(actual) & np.isfinite(reference)
    on_grid = ((actual.view(np.uint32) & 0xffff) == 0) & ((reference.view(np.uint32) & 0xffff) == 0)
    valid = finite & on_grid
    distance = np.abs(ordered_codes(actual)-ordered_codes(reference))
    values = distance[valid]
    return dict(bf16_grid_valid=bool(valid.all()), invalid_or_not_bf16_count=int(np.count_nonzero(~valid)),
        equal=int(np.count_nonzero(valid & (distance == 0))),
        one_ulp=int(np.count_nonzero(valid & (distance == 1))),
        more_than_one_ulp=int(np.count_nonzero(valid & (distance > 1))),
        max_bf16_ulp=int(values.max()) if values.size else None)


def classify_mismatches(actual, reference, tie):
    mismatch = np.asarray(actual) != np.asarray(reference)
    return dict(total=int(mismatch.sum()), at_exact_fp64_midpoint=int((mismatch & tie).sum()),
        not_at_exact_fp64_midpoint=int((mismatch & ~tie).sum()),
        per_query=[dict(row=i, total=int(row.sum()), at_exact_fp64_midpoint=int((row & tie[i]).sum()),
                       not_at_exact_fp64_midpoint=int((row & ~tie[i]).sum())) for i,row in enumerate(mismatch)])


def diagnose_group(data, prefix, samples=8):
    required = [prefix+suffix for suffix in ("_actual", "_native_reference", "_cold_out", "_cold_max", "_cold_sum",
        "_installed_hot_out", "_installed_hot_max", "_installed_hot_sum")]
    missing = [key for key in required if key not in data]
    if missing:
        return dict(status="MISSING_ARRAYS", missing=missing)
    actual, native, co, cm, cs, ho, hm, hs = (np.asarray(data[key]) for key in required)
    shape = co.shape
    if len(shape) != 3 or ho.shape != shape or actual.shape != shape or native.shape != shape:
        raise ValueError(f"{prefix}: expected matching [query,head,feature] arrays")
    if any(value.shape != shape[:2]+(1,) for value in (cm,cs,hm,hs)):
        raise ValueError(f"{prefix}: max/sum shape must be [query,head,1]")
    if not all(np.isfinite(v).all() for v in (co,cm,cs,ho,hm,hs)) or not ((cs > 0).all() and (hs > 0).all()):
        raise ValueError(f"{prefix}: present-side native states invalid; cannot explain as rounding")
    co, cm, cs, ho, hm, hs = (v.astype(np.float64) for v in (co,cm,cs,ho,hm,hs))
    common = np.maximum(cm,hm)
    cw, hw = cs*np.exp(cm-common), hs*np.exp(hm-common)
    denominator = cw+hw
    cw, hw = cw/denominator, hw/denominator
    precise = cw*co+hw*ho
    oracle, midpoint, tie, spacing = bf16_neighbors(precise)
    # Separate arithmetic model, not a replacement for the exact FP64 oracle:
    # rounded FP32 coefficients, two rounded multiplies, then rounded addition.
    staged = np.add(np.multiply(cw.astype(np.float32), co.astype(np.float32), dtype=np.float32),
                    np.multiply(hw.astype(np.float32), ho.astype(np.float32), dtype=np.float32), dtype=np.float32)
    staged_oracle, _, staged_tie, _ = bf16_neighbors(staged)
    zero_exact = bool((cm == 0).all() and (hm == 0).all() and (cs == 1920).all() and (hs == 128).all())
    differences = actual != native
    locations = np.argwhere(differences)
    examples = []
    # Include non-tie errors first: do not let a large tie population hide them.
    if len(locations):
        locations = sorted(locations, key=lambda pos: bool(tie[tuple(pos)]))
    for pos in locations[:samples]:
        p = tuple(pos)
        examples.append(dict(index=[int(v) for v in pos], cold=float(co[p]), hot=float(ho[p]),
            actual=float(actual[p]) if np.isfinite(actual[p]) else None,
            native=float(native[p]) if np.isfinite(native[p]) else None, oracle=float(oracle[p]),
            actual_bf16_bits=f"0x{int(bits(actual)[p]):04x}", native_bf16_bits=f"0x{int(bits(native)[p]):04x}",
            oracle_bf16_bits=f"0x{int(bits(oracle)[p]):04x}", exact_merge_fp64=float(precise[p]),
            midpoint=float(midpoint[p]), midpoint_distance=float(precise[p]-midpoint[p]),
            midpoint_distance_in_bf16_ulp=float((precise[p]-midpoint[p])/spacing[p]),
            exact_fp64_tie=bool(tie[p]), fp32_muladd_value=float(staged[p]),
            fp32_muladd_bf16_oracle=float(staged_oracle[p]), fp32_muladd_tie=bool(staged_tie[p])))
    return dict(status="CPU_DIAGNOSED", shape=list(shape), zero_query_1920_128_exact=zero_exact,
        cold_weight_range=[float(cw.min()),float(cw.max())], hot_weight_range=[float(hw.min()),float(hw.max())],
        oracle="FP64 normalized native-state merge, direct FP64-to-BF16 round-to-nearest-even",
        oracle_scope="For zeroQ1920/128, weights exactly15/16+1/16; nonzero exponentials are FP64 approximations",
        tie_definition="exact equality to a BF16 midpoint in the FP64 reference; not a tolerance band",
        exact_midpoint_elements=int(tie.sum()),
        actual_vs_native=metrics(actual,native), actual_vs_oracle=metrics(actual,oracle), native_vs_oracle=metrics(native,oracle),
        fp32_muladd_model=dict(
            scope="FP64 coefficients rounded to FP32; separate FP32 multiplies/add; then BF16 RNE. Not an Ascend exp/log or FMA emulation.",
            actual_vs_oracle=metrics(actual,staged_oracle), native_vs_oracle=metrics(native,staged_oracle),
            bf16_oracle_differs_from_fp64_count=int(np.count_nonzero(staged_oracle != oracle)),
            actual_oracle_ulp=ulp_summary(actual,staged_oracle), native_oracle_ulp=ulp_summary(native,staged_oracle)),
        actual_native_ulp=ulp_summary(actual,native), actual_oracle_ulp=ulp_summary(actual,oracle), native_oracle_ulp=ulp_summary(native,oracle),
        actual_native_mismatches=classify_mismatches(actual,native,tie),
        actual_oracle_mismatches=classify_mismatches(actual,oracle,tie),
        native_oracle_mismatches=classify_mismatches(native,oracle,tie), examples=examples)


def self_test():
    # Positive/negative exact halfway values and adjacent non-ties.
    x = np.asarray([1+1/256, 1+3/256, -1-1/256, -1-3/256, 1+1/256+1e-12])
    out, _, tie, _ = bf16_neighbors(x)
    np.testing.assert_array_equal(out, [1.,1+1/64,-1.,-1-1/64,1+1/128])
    np.testing.assert_array_equal(tie, [True,True,True,True,False])
    assert ulp_summary(np.asarray([1.]),np.asarray([1+1/128]))["one_ulp"] == 1
    print("CPU BF16 rounding fixtures passed; no device validation.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path)
    p.add_argument("--results", type=Path, help="Defaults to results.json next to NPZ")
    p.add_argument("--output", type=Path, help="Defaults to stdout; does not modify source results")
    p.add_argument("--list-keys", action="store_true")
    p.add_argument("--all-groups", action="store_true", help="Include direct-native diagnostic groups as well as fused")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.npz is None:
        p.error("--npz is required")
    results_path = args.results or args.npz.parent/"results.json"
    result = json.loads(results_path.read_text()) if results_path.is_file() else {}
    with np.load(args.npz, allow_pickle=False) as data:
        keys = sorted(data.files)
        if args.list_keys:
            print(json.dumps(keys, indent=2))
            return 0
        available = sorted(key[:-7] for key in keys if re.search(r"_(?:nonzero_query|zero_query)_actual$",key))
        fused = [key for key in available if "fused" in key]
        selected = available if args.all_groups or not fused else fused
        report = dict(version="merge-rounding-cpu-v1", source_npz=str(args.npz),
            source_results=str(results_path) if results_path.is_file() else None,
            original_status=result.get("status"), original_strict_limit=LIMIT,
            original_acceptance_unchanged=True, device_launched=False,
            available_groups=available, groups={})
        for prefix in selected:
            try:
                report["groups"][prefix] = diagnose_group(data,prefix)
            except ValueError as exc:
                report["groups"][prefix] = dict(status="INVALID_STATES",error=str(exc))
        if not available:
            report.update(status="NO_MATCHING_ARRAYS", available_keys=keys)
        else:
            report["status"] = "CPU_DIAGNOSED" if all(x["status"] == "CPU_DIAGNOSED" for x in report["groups"].values()) else "INCOMPLETE"
    text = json.dumps(report,indent=2,allow_nan=False)+"\n"
    if args.output:
        if args.output.resolve() in (args.npz.resolve(), results_path.resolve()):
            p.error("Diagnostic output must not overwrite inputs or original results")
        args.output.write_text(text,encoding="utf-8")
        print(f"{report['status']}: {args.output}")
    else:
        sys.stdout.write(text)
    return 0 if report["status"] == "CPU_DIAGNOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
