#!/usr/bin/env python3
"""A5 single-operator accuracy: BF16 Lightning Indexer vs Hadamard + MXFP4 QLI.

No model loading, main-KV changes, package installation, DSL, or explicit operator rebuild.
Run --self-test without Torch. Use --input for real BF16 indexer input snapshots.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

from qli_topk_metrics import compare_topk, inspect_indices, json_safe


VERSION = "qli-bf16-accuracy-v1"
BACKENDS = ("bf16_original", "bf16_rotated", "bf16_dequantized", "mxfp4_qli")
PAIRS = (
    ("total_mxfp4_vs_bf16", "bf16_original", "mxfp4_qli", "original"),
    ("hadamard_rounding", "bf16_original", "bf16_rotated", "original"),
    ("input_quantization", "bf16_rotated", "bf16_dequantized", "rotated"),
    ("qli_vs_dequant_bf16", "bf16_dequantized", "mxfp4_qli", "dequantized"),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def integer_scalar(value, name):
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be one integer")
    return int(array.reshape(-1)[0])


def bf16_round(array):
    """Round finite float32 inputs to BF16, storing the exact values in float32."""
    array = np.asarray(array, np.float32)
    bits = array.view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
    if not np.all(np.isfinite(rounded)):
        raise ValueError("Input cannot be represented as finite BF16")
    return rounded


def validate_input(data):
    """Only original, pre-Hadamard BF16 indexer activations belong in this format."""
    required = ("q", "k", "w", "k_lengths", "topk", "mask", "page")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing input fields: {missing}")
    result = dict(data)
    for name in ("q", "k", "w"):
        value = np.asarray(data[name])
        if value.dtype.kind != "f" or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain finite FLOAT values from original BF16 inputs, not packed quantized bytes")
        result[name] = np.ascontiguousarray(value, dtype=np.float32)
    q, k, w = (result[name] for name in ("q", "k", "w"))
    if q.ndim != 4 or q.shape[-1] != 128 or min(q.shape) < 1:
        raise ValueError(f"q shape must be [B,Q,H,128], got {q.shape}")
    batch, queries, heads, _ = q.shape
    if heads not in (8, 16, 24, 32, 64):
        raise ValueError("BF16 A5 head count must be one of 8,16,24,32,64")
    if k.ndim != 4 or k.shape[0] != batch or k.shape[2:] != (1, 128) or k.shape[1] < 1:
        raise ValueError(f"k shape must be [B,K,1,128], got {k.shape}")
    if w.shape != q.shape[:-1]:
        raise ValueError(f"w must be [B,Q,H]={q.shape[:-1]}, got {w.shape}")
    for name, maximum in (("q_lengths", queries), ("k_lengths", k.shape[1])):
        value = np.asarray(data.get(name, np.full(batch, maximum, dtype=np.int32)))
        if value.shape != (batch,) or value.dtype.kind not in "iu" or np.any(value < 1) or np.any(value > maximum):
            raise ValueError(f"{name} must be {batch} integers in [1,{maximum}]")
        result[name] = value.astype(np.int32)
    for name in ("topk", "mask", "page", "seed"):
        result[name] = integer_scalar(data.get(name, 0), name)
    # The pinned deployment uses TopK=2048. Keep this diagnostic in the jointly
    # supported small-TopK range instead of guessing another kernel's constraints.
    if not 1 <= result["topk"] <= 2048:
        raise ValueError("This harness supports TopK 1..2048; GLM-5.3 default is 2048")
    if result["mask"] not in (0, 3):
        raise ValueError("mask must be 0 (all visible) or 3 (right-aligned causal)")
    if result["page"] not in (16, 32, 64, 128, 256):
        raise ValueError("page must be one of 16,32,64,128,256")
    if np.any(result["q_lengths"] > result["k_lengths"]):
        raise ValueError("This trailing-query replay format requires query lengths <= key lengths")
    stage = str(np.asarray(data.get("input_stage", "pre_hadamard_bf16")).item())
    if stage != "pre_hadamard_bf16":
        raise ValueError("Need original pre_hadamard_bf16 Q/K, not dequantized or already-rotated inputs")
    result["input_stage"] = stage
    return result


def load_case(path):
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        if "format_version" not in archive or integer_scalar(archive["format_version"], "format_version") != 1:
            raise ValueError("Unsupported snapshot: expected format_version=1")
        if "input_stage" not in archive:
            raise ValueError("Snapshot must explicitly identify input_stage=pre_hadamard_bf16")
        data = {key: archive[key] for key in archive.files if key != "format_version"}
    data = validate_input(data)
    for field in ("q", "k", "w"):
        if not np.array_equal(data[field], bf16_round(data[field])):
            raise ValueError(f"Snapshot {field} is not an exact BF16 tensor stored in float32; capture original BF16 inputs")
    source_kind = str(np.asarray(data.get("source_kind", "unspecified")).item())
    kind = "synthetic_replay" if source_kind == "synthetic" else "model_input_replay" if source_kind == "model_capture" else "input_snapshot_replay"
    return data, {"kind": kind, "file": str(path), "sha256": sha256(path),
                  "note": "Snapshot replay; source kind is recorded by its producer, not independently verified. No model is run by this script"}


def save_paged_case(path, q, key_cache, weights, block_table, cu_seqlens_q,
                    k_lengths, topk=2048, mask=3, label=""):
    """Call explicitly in an eager BF16 indexer path to export one NPU snapshot.

    q:[T,H,128], key_cache:[pages,page,1,128], weights:[T,H],
    block_table:[B,pages_per_sequence], cu_seqlens_q:[B+1] INCLUDING leading 0.
    key_cache must contain the ORIGINAL BF16 indexer keys for the entire visible
    history. An MXFP4 cache or a current-token-only K is deliberately rejected.
    This helper does not install a hook, patch the model, or collect prompts.
    """
    def original_bf16(tensor, name):
        if str(getattr(tensor, "dtype", "")) != "torch.bfloat16":
            raise ValueError(f"{name} must be an original torch.bfloat16 tensor")
        return tensor.detach().float().cpu().numpy()

    def integer_array(tensor, name):
        array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        if array.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integer indices/lengths")
        return array.astype(np.int64)

    q_cpu = original_bf16(q, "q")
    keys = original_bf16(key_cache, "key_cache")
    w_cpu = original_bf16(weights, "weights")
    table = integer_array(block_table, "block_table")
    cu = integer_array(cu_seqlens_q, "cu_seqlens_q")
    lengths = integer_array(k_lengths, "k_lengths")
    if cu.ndim != 1 or len(cu) < 2 or cu[0] != 0 or np.any(np.diff(cu) <= 0):
        raise ValueError("cu_seqlens_q requires a leading 0 and positive per-sequence query lengths")
    batch, qlens = len(cu)-1, np.diff(cu)
    if q_cpu.ndim != 3 or q_cpu.shape[-1] != 128 or q_cpu.shape[0] != int(cu[-1]):
        raise ValueError("q must have TND layout matching cu_seqlens_q")
    if w_cpu.shape != q_cpu.shape[:2] or lengths.shape != (batch,) or np.any(lengths <= 0):
        raise ValueError("weights or key lengths do not match the queries")
    if keys.ndim != 4 or keys.shape[2:] != (1, 128) or table.ndim != 2 or table.shape[0] != batch:
        raise ValueError("Expected original BF16 PA_BSND indexer cache and [B,pages] table")
    page, max_q, max_k = keys.shape[1], int(max(qlens)), int(max(lengths))
    q_logical = np.zeros((batch, max_q, q_cpu.shape[1], 128), np.float32)
    k_logical = np.zeros((batch, max_k, 1, 128), np.float32)
    w_logical = np.zeros(q_logical.shape[:-1], np.float32)
    for b in range(batch):
        n = int(lengths[b])
        if math.ceil(n/page) > table.shape[1]:
            raise ValueError("Page table cannot address the entire key history")
        positions = np.arange(n)
        physical = table[b, positions//page]
        if np.any(physical < 0) or np.any(physical >= len(keys)):
            raise ValueError("Page table contains an invalid physical page")
        q_logical[b, :qlens[b]] = q_cpu[cu[b]:cu[b+1]]
        w_logical[b, :qlens[b]] = w_cpu[cu[b]:cu[b+1]]
        k_logical[b, :n] = keys[physical, positions % page]
    data = validate_input(dict(q=q_logical, k=k_logical, w=w_logical,
                               q_lengths=qlens, k_lengths=lengths, topk=topk, mask=mask,
                               page=page, seed=0, input_stage="pre_hadamard_bf16", source_kind="model_capture"))
    path = Path(path)
    # Exclusive creation: never overwrite a captured sample.
    with path.open("xb") as stream:
        np.savez_compressed(stream, **data, format_version=np.int32(1), label=str(label))
    return path


def make_case(batch, queries, keys, heads, topk, page, mask, seed, distribution):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((batch, queries, heads, 128)).astype(np.float32)
    k = rng.standard_normal((batch, keys, 1, 128)).astype(np.float32)
    w = (rng.standard_normal((batch, queries, heads))/math.sqrt(heads*128)).astype(np.float32)
    if distribution == "outlier":
        q[..., 0] *= 16
        k[..., 0] *= 16
    elif distribution != "normal":
        raise ValueError("Unknown synthetic distribution")
    lengths = np.array([max(queries, keys-17*b) for b in range(batch)], np.int32)
    return validate_input(dict(q=bf16_round(q), k=bf16_round(k), w=bf16_round(w), k_lengths=lengths,
                               topk=topk, page=page, mask=mask, seed=seed, source_kind="synthetic"))


def visible_lengths(data, row_map):
    lengths = []
    for b, row in row_map:
        n = int(data["k_lengths"][b])
        if data["mask"] == 3:
            n = max(0, min(n, n-int(data["q_lengths"][b])+int(row)+1))
        lengths.append(n)
    return np.asarray(lengths, np.int32)


def diagnostic_scores(q, k, w, row_map, lengths, chunk=1024):
    """Float64 sum_h w_h*ReLU(dot(Q_h,K)); NOT the BF16 kernel's accumulator.

    The full candidate domain is evaluated, not just the returned TopK. This
    diagnostic formula is used to explain boundary changes, never to invent
    actual kernel score outputs or claim bit-exact kernel correctness.
    """
    if chunk < 1:
        raise ValueError("chunk must be positive")
    scores = np.full((len(row_map), k.shape[1]), -np.inf, dtype=np.float64)
    for t, ((b, row), n) in enumerate(zip(row_map, lengths)):
        query = q[b, row].astype(np.float64)
        weight = w[b, row].astype(np.float64)
        for start in range(0, int(n), chunk):
            stop = min(int(n), start+chunk)
            key = k[b, start:stop, 0].astype(np.float64)
            scores[t, start:stop] = weight @ np.maximum(query @ key.T, 0)
    if any(not np.all(np.isfinite(scores[i, :int(n)])) for i, n in enumerate(lengths)):
        raise ValueError("Nonfinite diagnostic scores on valid candidates")
    return scores


def score_delta(a, b, lengths):
    mask = np.arange(a.shape[1])[None, :] < lengths[:, None]
    x, y = a[mask], b[mask]
    if not len(x):
        return {"count": 0, "max_abs": None, "mean_abs": None, "relative_l2": None}
    diff = y-x
    norm = float(np.linalg.norm(x))
    return {"count": len(x), "max_abs": float(np.max(np.abs(diff))),
            "mean_abs": float(np.mean(np.abs(diff))),
            "relative_l2": float(np.linalg.norm(diff)/norm) if norm else None}


def list_of_positive_ints(text):
    try:
        result = [int(x) for x in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers") from None
    if not result or any(x < 1 for x in result):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers")
    return result


def print_comparison(label, result, worst_rows=3):
    print(f"  [{label}] {result['status']}", flush=True)
    # Print the complete aggregation without assuming a score is a kernel value.
    print("    " + json.dumps(json_safe(result["summary"]), ensure_ascii=False, allow_nan=False), flush=True)
    rows = result.get("rows", [])
    changed = [r for r in rows if r.get("replaced_count", 0) or r.get("recall") == 0]
    changed.sort(key=lambda r: r.get("recall") if r.get("recall") is not None else -1)
    for row in changed[:worst_rows]:
        # Full dropped/added lists remain in results.json and arrays.npz.
        small = {k: v for k, v in row.items() if k not in ("dropped", "added", "dropped_indices", "added_indices")}
        print("    worst row: " + json.dumps(json_safe(small), ensure_ascii=False, allow_nan=False), flush=True)


def self_test():
    d = make_case(2, 3, 19, 32, 8, 128, 3, 42, "normal")
    rows = np.array([(b, q) for b in range(2) for q in range(3)], np.int32)
    lens = visible_lengths(d, rows)
    assert lens.tolist() == [17, 18, 19, 1, 2, 3]
    scores = diagnostic_scores(d["q"], d["k"], d["w"], rows, lens, 4)
    np.testing.assert_allclose(scores, diagnostic_scores(d["q"], d["k"], d["w"], rows, lens, 64), rtol=1e-12, atol=1e-12)
    ix = np.full((6, 1, 8), -1, np.int32)
    for r, n in enumerate(lens):
        count = min(8, int(n))
        ix[r, 0, :count] = np.argsort(scores[r, :n])[-count:]
    assert inspect_indices(ix, lens, 19)["passed"]
    comparison = compare_topk(ix, ix[..., ::-1], scores, lens)
    assert comparison["status"] == "VALID_COMPARISON"
    json.dumps(json_safe(comparison), allow_nan=False)
    bad = ix.copy()
    bad[0, 0, 0] = 19
    assert not inspect_indices(bad, lens, 19)["passed"]
    print("SELF_TEST PASS: causal lengths, full-domain score computation, set comparison, invalid-index rejection")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input", action="append", help="Original BF16 input NPZ from save_paged_case; repeat for multiple samples")
    parser.add_argument("--keys", type=list_of_positive_ints, default=[8192])
    parser.add_argument("--queries", type=list_of_positive_ints, default=[4])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--page", type=int, default=128)
    parser.add_argument("--mask", type=int, choices=(0, 3), default=3)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--distribution", choices=("normal", "outlier"), default="normal")
    parser.add_argument("--repeats", type=int, default=3, help="Repeated identical calls to expose set instability, not a performance benchmark")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--tie-rtol", type=float, default=0.0)
    parser.add_argument("--tie-atol", type=float, default=0.0)
    parser.add_argument("--min-recall", type=float, help="Optional user-specified total TopK recall gate; no universal accuracy threshold is assumed")
    parser.add_argument("--save-inputs", action="store_true", help="Also retain original, rotated, dequantized and packed inputs; may be large")
    parser.add_argument("--output", default="qli_bf16_"+time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if min(args.batch, args.repeats, args.chunk) < 1:
        parser.error("batch/repeats/chunk must be positive")
    if not all(math.isfinite(x) and x >= 0 for x in (args.tie_rtol, args.tie_atol)):
        parser.error("Tie tolerances must be finite and nonnegative")
    if args.min_recall is not None and not 0 <= args.min_recall <= 1:
        parser.error("--min-recall must be in [0,1]")
    out = Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory must be empty; existing results are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    report = {"version": VERSION, "args": vars(args), "status": "RUNNING", "cases": [],
              "python": sys.version, "platform": platform.platform(),
              "scripts": {p.name: sha256(p) for p in (
                  Path(__file__), Path(__file__).with_name("qli_topk_metrics.py"),
                  Path(__file__).with_name("qli_bf16_backend.py"))},
              "scope": "Single indexer operator only. No main KV, attention output, logits, model loss or end-to-end accuracy is tested.",
              "score_semantics": "All reported scores are independent Float64 diagnostic sum_h w_h*ReLU(Q_h dot K). Actual operators return indices only.",
              "comparisons": {name: {"reference": left, "test": right, "score_domain": domain}
                              for name, left, right, domain in PAIRS}}
    write_json(out/"results.json", report)
    failed = False
    differences = False
    try:
        cases = []
        if args.input:
            for path in args.input:
                cases.append(load_case(path))
            print("INPUT SNAPSHOT REPLAY: using saved TopK/mask/page/lengths and source kind; synthetic flags do not override snapshots.", flush=True)
        else:
            print("SYNTHETIC INPUT: this run measures operator behavior, NOT GLM-5.3 model accuracy.", flush=True)
            for keys in args.keys:
                for queries in args.queries:
                    cases.append((make_case(args.batch, queries, keys, args.heads, args.topk,
                                           args.page, args.mask, args.seed, args.distribution),
                                  {"kind": "synthetic", "distribution": args.distribution, "seed": args.seed}))
        from qli_bf16_backend import run_npu_case
        for case_index, (data, origin) in enumerate(cases):
            name = f"case_{case_index:03d}_q{data['q'].shape[1]}_k{data['k'].shape[1]}"
            case_out = out/name
            case_out.mkdir()
            entry = {"name": name, "source": origin, "status": "RUNNING",
                     "q_shape": list(data["q"].shape), "k_shape": list(data["k"].shape),
                     "topk": data["topk"], "mask": data["mask"], "page": data["page"],
                     "q_lengths": data["q_lengths"].tolist(), "k_lengths": data["k_lengths"].tolist()}
            report["cases"].append(entry)
            print(f"\n[{name}] q={entry['q_shape']}, k={entry['k_shape']}, "
                  f"TopK={data['topk']}, mask={data['mask']}, source={origin['kind']}", flush=True)
            write_json(out/"results.json", report)
            result = run_npu_case(data, device=args.device, repeats=args.repeats)
            entry["environment"] = result["environment"]
            row_map = np.asarray(result["row_map"], np.int32)
            lens = visible_lengths(data, row_map)
            entry["row_map"] = [{"row": i, "batch": int(b), "query": int(q), "visible_keys": int(lens[i])}
                                for i, (b, q) in enumerate(row_map)]
            entry["rows_with_real_selection"] = int(np.sum(lens > data["topk"]))
            if not entry["rows_with_real_selection"]:
                print("WARNING: all visible lengths <= TopK; this only checks completeness/padding, not selective TopK flips.", flush=True)
            inputs, outputs = result["inputs"], result["outputs"]
            arrays = {"row_map": row_map, "visible_lengths": lens, "block_table": result["block_table"],
                      **{f"indices_{key}": value for key, value in outputs.items()}}
            # Save actual outputs before any CPU analysis so failures remain inspectable.
            np.savez_compressed(case_out/"arrays.npz", **arrays)
            print("  Computing full-domain diagnostic scores (not kernel-returned scores)...", flush=True)
            score_domains = {}
            for domain, suffix in (("original", "ref"), ("rotated", "rot"), ("dequantized", "deq")):
                score_domains[domain] = diagnostic_scores(inputs[f"q_{suffix}"], inputs[f"k_{suffix}"],
                                                         inputs["w_ref"], row_map, lens, args.chunk)
                arrays[f"diagnostic_scores_{domain}"] = score_domains[domain]
            entry["index_validation"] = {key: inspect_indices(outputs[key], lens, data["k"].shape[1])
                                          for key in BACKENDS}
            for backend, validity in entry["index_validation"].items():
                print(f"  {backend}: index structure {'VALID' if validity['passed'] else 'INVALID'}", flush=True)
                if not validity["passed"]:
                    print(json.dumps(json_safe(validity), ensure_ascii=False, allow_nan=False), flush=True)
                    failed = True
            entry["score_deltas"] = {"hadamard_rounding": score_delta(score_domains["original"], score_domains["rotated"], lens),
                                     "input_quantization": score_delta(score_domains["rotated"], score_domains["dequantized"], lens),
                                     "total_input_transform": score_delta(score_domains["original"], score_domains["dequantized"], lens)}
            entry["comparisons"] = {}
            for label, left, right, domain in PAIRS:
                comparison = compare_topk(outputs[left], outputs[right], score_domains[domain], lens,
                                          tie_rtol=args.tie_rtol, tie_atol=args.tie_atol)
                test_domain = "rotated" if right == "bf16_rotated" else "dequantized"
                for row in comparison["rows"]:
                    for kind in ("dropped_examples", "added_examples"):
                        for example in row.get(kind) or []:
                            example["test_input_diagnostic_score"] = float(score_domains[test_domain][row["row"], example["index"]])
                comparison["test_input_score_domain"] = test_domain
                entry["comparisons"][label] = comparison
                print_comparison(label, comparison)
                if comparison["status"] != "VALID_COMPARISON":
                    failed = True
            total = entry["comparisons"]["total_mxfp4_vs_bf16"]
            valid_recalls = [r["recall"] for r in total["rows"] if r.get("recall") is not None]
            differences |= any(value < 1 for value in valid_recalls)
            if args.min_recall is not None and any(value < args.min_recall for value in valid_recalls):
                entry["recall_gate_failed"] = True
                failed = True
            entry["repeat_comparisons"] = {}
            entry["repeat_set_instability"] = {}
            for backend, snapshots in result.get("repeats", {}).items():
                domain = "original" if backend == "bf16_original" else "rotated" if backend == "bf16_rotated" else "dequantized"
                repeat_reports = []
                for repeat_index, indices in enumerate(snapshots[1:], start=1):
                    arrays[f"repeat_{backend}_{repeat_index}"] = indices
                    check = compare_topk(outputs[backend], indices, score_domains[domain], lens,
                                         tie_rtol=args.tie_rtol, tie_atol=args.tie_atol)
                    repeat_reports.append(check)
                    if check["status"] != "VALID_COMPARISON":
                        failed = True
                entry["repeat_comparisons"][backend] = repeat_reports
                changed = sum(any(r.get("recall") is not None and r["recall"] < 1 for r in rep["rows"])
                              for rep in repeat_reports)
                entry["repeat_set_instability"][backend] = {"changed_calls": changed, "compared_calls": len(repeat_reports)}
                differences |= changed > 0
                print(f"  {backend}: repeated calls with changed index SET={changed}/{len(repeat_reports)}", flush=True)
            if args.save_inputs:
                arrays.update(inputs)
                arrays.update({f"quant_{k}": v for k, v in result.get("quantized", {}).items()})
                with (case_out/"input_bf16.npz").open("xb") as stream:
                    snapshot = dict(data, q=inputs["q_ref"], k=inputs["k_ref"], w=inputs["w_ref"], format_version=np.int32(1))
                    np.savez_compressed(stream, **snapshot)
            np.savez_compressed(case_out/"arrays.npz", **arrays)
            invalid_case = (any(not v["passed"] for v in entry["index_validation"].values())
                            or any(c["status"] != "VALID_COMPARISON" for c in entry["comparisons"].values())
                            or any(c["status"] != "VALID_COMPARISON" for repeats in entry["repeat_comparisons"].values()
                                   for c in repeats))
            entry["status"] = "INVALID" if invalid_case else "RECALL_GATE_FAILED" if entry.get("recall_gate_failed") else "MEASURED"
            write_json(case_out/"details.json", entry)
            write_json(out/"results.json", report)
    except Exception:
        failed = True
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
        if report["cases"] and report["cases"][-1]["status"] == "RUNNING":
            report["cases"][-1]["status"] = "ERROR"
    report["status"] = "ERROR_OR_INVALID" if failed else "MEASURED_WITH_TOPK_DIFFERENCES" if differences else "MEASURED_IDENTICAL_TOPK"
    report["note"] = "Measured differences (initial cross-path or repeated-call selection) are not automatically an operator bug or proof of model accuracy loss. No NPU execution was performed on the author's local CPU-only validation environment."
    write_json(out/"results.json", report)
    print(f"\n{report['status']}: {out}/results.json", flush=True)
    print("Inspect case details.json for full dropped/added token lists. Scores in arrays.npz are diagnostic, not returned kernel scores.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
