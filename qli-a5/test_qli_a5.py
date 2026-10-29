#!/usr/bin/env python3
"""A5 MXFP4 QLI V2 / DSL accuracy and eager latency harness.

CPU-only checks: python test_qli_a5.py --self-test
See QLI_A5_README.txt for setup, interpretation and source revisions.
"""
import argparse
import csv
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
import types
from dataclasses import asdict, dataclass, replace

import numpy as np


DSL_MODULES = ("quant_lightning_indexer_metadata_dsl", "quant_lightning_indexer_dsl")


def _dsl_file_pairs(root):
    """Accept either upstream samples layout or an external flat ops directory."""
    root = Path(root).expanduser().resolve()
    layouts = [
        {name: root / "samples" / name / (name + ".py") for name in DSL_MODULES},
        {name: root / name / (name + ".py") for name in DSL_MODULES},
        {name: root / "ops" / (name + ".py") for name in DSL_MODULES},
        {name: root / (name + ".py") for name in DSL_MODULES},
    ]
    return [files for files in layouts if all(p.is_file() for p in files.values())]


def _choose_dsl_files(pairs):
    unique = {}
    for files in pairs:
        # Keep distinct locations separate: adjacent prebuilt metadata binaries may differ.
        key = tuple(str(files[n].resolve()) for n in DSL_MODULES)
        unique[key] = {n: files[n].resolve() for n in DSL_MODULES}
    if len(unique) > 1:
        raise RuntimeError("Multiple DSL source locations found; select one with --dsl-root: "
                           + "; ".join(key[0] for key in unique))
    return next(iter(unique.values())) if unique else None


def resolve_dsl_files(args):
    """Resolve the requested files, without installing packages or copying kernels."""
    explicit = args.dsl_root or os.environ.get("CANNBOTDSL_TEST_OPS_DIR")
    if explicit:
        files = _choose_dsl_files(_dsl_file_pairs(explicit))
        if files is None:
            raise RuntimeError(f"Incomplete DSL source at {explicit}: need both "
                               "quant_lightning_indexer_dsl.py and quant_lightning_indexer_metadata_dsl.py")
        return files
    if args.dsl_source == "ops":
        try:
            spec = importlib.util.find_spec("ops")
        except (ImportError, ValueError):
            spec = None
        pairs = []
        for loc in (getattr(spec, "submodule_search_locations", None) or []):
            pairs.extend(_dsl_file_pairs(loc))
        files = _choose_dsl_files(pairs)
        if files:
            return files
    # The original download placed the harness next to cannbot-source.
    roots = [Path(__file__).resolve().parent / "cannbot-source", Path.cwd() / "cannbot-source"]
    files = _choose_dsl_files([pair for root in roots for pair in _dsl_file_pairs(root)])
    if files is None:
        raise RuntimeError("DSL source files not found. Pass --dsl-root /path/to/cannbot-source "
                           "(or /path/to/ops). No package reinstall is required for path mapping.")
    return files


def install_dsl_modules(files, namespace="ops"):
    """Expose exact source files under the imports expected by the installed wrapper.

    Original __file__ paths are preserved for JIT source inspection and adjacent
    _aicpu binaries. Only this process changes; existing modules are never replaced.
    """
    targets = [(name, (namespace + "." if namespace else "") + name,
                Path(files[name]).resolve()) for name in DSL_MODULES]
    for _, target, path in targets:
        existing = sys.modules.get(target)
        if existing is not None and (not getattr(existing, "__file__", None)
                                     or Path(existing.__file__).resolve() != path):
            raise RuntimeError(f"Conflicting loaded DSL module {target}: "
                               f"{getattr(existing, '__file__', None)}; requested {path}")
    parent = None
    created_parent = False
    if namespace:
        parent = sys.modules.get(namespace)
        if parent is None:
            try:
                parent = importlib.import_module(namespace)
            except ModuleNotFoundError as exc:
                if exc.name != namespace:
                    raise
                parent = types.ModuleType(namespace)
                parent.__path__ = []
                parent.__package__ = namespace
                parent.__spec__ = importlib.machinery.ModuleSpec(namespace, loader=None, is_package=True)
                sys.modules[namespace] = parent
                created_parent = True
        if not hasattr(parent, "__path__"):
            raise RuntimeError(f"{namespace} is already a non-package module")
    added = []
    old_attrs = {}
    absent = object()
    try:
        for short, target, path in targets:
            module = sys.modules.get(target)
            # Importing an existing parent package can itself populate these names.
            if module is not None and (not getattr(module, "__file__", None)
                                       or Path(module.__file__).resolve() != path):
                raise RuntimeError(f"Conflicting loaded DSL module {target}: "
                                   f"{getattr(module, '__file__', None)}; requested {path}")
            if module is None:
                spec = importlib.util.spec_from_file_location(target, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[target] = module
                added.append(target)
                spec.loader.exec_module(module)
            if parent is not None:
                old_attrs[short] = getattr(parent, short, absent)
                setattr(parent, short, module)
        return [sys.modules[target] for _, target, _ in targets]
    except Exception:
        for target in added:
            sys.modules.pop(target, None)
        for short, previous in old_attrs.items():
            if previous is absent:
                delattr(parent, short)
            else:
                setattr(parent, short, previous)
        if created_parent:
            sys.modules.pop(namespace, None)
        raise


@dataclass
class Case:
    name: str
    batch: int = 2
    query: int = 3
    key: int = 257
    heads: int = 32
    page: int = 128
    topk: int = 32
    mask: int = 0
    ratio: int = 1
    variable: bool = False
    weight: str = "mixed"
    offset: int = 0


def cases_for(suite):
    cases = [Case("smoke"), Case("causal_tail", key=129, page=64,
                                 topk=256, mask=3, ratio=2, variable=True, offset=7),
             Case("negative_scores", key=65, topk=128, weight="negative")]
    if suite in ("standard", "full"):
        cases += [Case("varlen_4k", batch=3, query=6, key=4097, topk=512,
                       mask=3, ratio=4, variable=True),
                  Case("qli_64k", batch=12, query=6, key=65536,
                       topk=512, mask=3, ratio=2)]
    if suite == "full":
        cases += [Case("qli_128k", batch=12, query=6, key=131072,
                       topk=512, mask=3, ratio=2)]
    return cases


def select_cases(suite, case_name=None, topk=None):
    """Select cases before overriding TopK; keep names/seeds/input bytes unchanged."""
    if topk is not None and not 1 <= topk <= 8192:
        raise ValueError("--topk must be between 1 and 8192")
    cases = cases_for(suite)
    if case_name:
        cases = [c for c in cases if c.name == case_name]
        if not cases:
            raise ValueError("--case not found in selected suite")
    if topk is not None:
        cases = [replace(c, topk=topk) for c in cases]
    return cases


def make_data(c, seed):
    rng = np.random.default_rng(seed)
    pages = math.ceil(c.key / c.page)
    q = rng.integers(0, 256, (c.batch*c.query, c.heads, 64), dtype=np.uint8)
    k = rng.integers(0, 256, (c.batch*pages, c.page, 1, 64), dtype=np.uint8)
    # Finite E8M0 powers of two, independently varying across D groups.
    sq = rng.integers(123, 128, (c.batch*c.query, c.heads, 2, 2), dtype=np.uint8)
    sk = rng.integers(123, 128, (c.batch*pages, c.page, 1, 2, 2), dtype=np.uint8)
    w = rng.uniform(-0.03, 0.03, (c.batch*c.query, c.heads)).astype(np.float32)
    if c.weight == "negative":
        w = -np.abs(w) - np.float32(0.01)
    qlen = np.full(c.batch, c.query, dtype=np.int32)
    klen = np.full(c.batch, c.key, dtype=np.int32)
    if c.variable:
        qlen[-1] = max(1, c.query-1)
        klen[-1] = max(1, c.key//2)
    residual = (np.arange(c.batch, dtype=np.int32) % c.ratio
                if c.mask == 3 and c.ratio != 1 else None)
    return dict(q=q, k=k, w=w, sq=sq, sk=sk,
                cu_seqlens_q=np.arange(c.batch+1, dtype=np.int32)*c.query,
                seqused_q=qlen, seqused_k=klen, cmp_residual_k=residual,
                block_table=rng.permutation(c.batch*pages).reshape(c.batch, pages).astype(np.int32),
                output_idx_offset=np.full((c.batch*c.query, 1), c.offset, np.int32))


def bf16_rne(x):
    """Round directly from float64 to the BF16 grid (no FP32 double rounding)."""
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)
    normal = np.ldexp(np.rint(m*256), e-8)
    subnormal = np.rint(x / 2.0**-133) * 2.0**-133
    return np.where(np.abs(x) < 2.0**-126, subnormal, normal).astype(np.float32)


def decode(packed, scale):
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6,
                    -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    codes = np.stack((packed & 15, packed >> 4), axis=-1)
    values = lut[codes].reshape(*packed.shape[:-1], 128)
    factors = np.exp2(scale.astype(np.int16).reshape(*packed.shape[:-1], 4)-127)
    return values * np.repeat(factors.astype(np.float32), 32, axis=-1)


def visible_length(c, d, b, row):
    if row >= int(d["seqused_q"][b]):
        return 0
    n = int(d["seqused_k"][b])
    if c.mask == 0:
        return n
    residual = 0 if d["cmp_residual_k"] is None else int(d["cmp_residual_k"][b])
    return min(n, max(0, (c.ratio*n + residual-int(d["seqused_q"][b])+row+1)//c.ratio))


def reference(c, d, chunk=2048):
    """Full-token reference for EVERY query row, with bounded QK scratch memory."""
    q = decode(d["q"], d["sq"])
    w = bf16_rne(d["w"])
    scores = np.full((c.batch*c.query, c.key), -np.inf, np.float32)
    for b in range(c.batch):
        for start in range(0, int(d["seqused_k"][b]), chunk):
            ids = np.arange(start, min(start+chunk, int(d["seqused_k"][b])))
            physical = d["block_table"][b, ids//c.page]
            key = decode(d["k"][physical, ids % c.page, 0],
                         d["sk"][physical, ids % c.page, 0])
            for row in range(c.query):
                t = b*c.query+row
                # Inputs are dyadic and bounded; FP64 dot accumulation is exact here.
                dot = q[t].astype(np.float64) @ key.astype(np.float64).T
                dot = bf16_rne(np.maximum(dot, 0))
                acc = np.zeros(len(ids), np.float32)
                for h in range(c.heads):
                    acc = bf16_rne(acc.astype(np.float64) +
                                   np.float64(w[t, h])*dot[h].astype(np.float64))
                valid = ids < visible_length(c, d, b, row)
                scores[t, ids[valid]] = acc[valid]
    return scores


def json_number(value):
    """Represent nonfinite observed values explicitly, without invalid JSON NaN."""
    value = float(value)
    return value if math.isfinite(value) else ("NaN" if math.isnan(value) else "+Inf" if value > 0 else "-Inf")


def validate(c, d, scores, indices, values, rtol, atol, detail_limit=8):
    """Independent checks; a count failure must not hide duplicates or NaNs.

    values=None checks the indices-only call, without inventing observed scores.
    Output positions, query rows and logical indices are all zero-based.
    """
    expected = (c.batch*c.query, 1, c.topk)
    if indices.shape != expected or (values is not None and values.shape != expected):
        raise AssertionError(f"Output shape mismatch: indices={indices.shape}, "
                             f"values={None if values is None else values.shape}, expected {expected}")
    if detail_limit < 1:
        raise ValueError("detail_limit must be positive")
    errors, error_types, rows, recall = [], {}, [], []
    max_abs, compared_scores, nonfinite_scores, compared_rows = None, 0, 0, 0
    for t in range(expected[0]):
        ix = indices[t, 0].astype(np.int64)
        val = None if values is None else values[t, 0]
        n = visible_length(c, d, t//c.query, t % c.query)
        count = min(n, c.topk)
        valid = ix != -1
        offset = int(d["output_idx_offset"][t, 0])
        logical = ix-offset
        safe = valid & (logical >= 0) & (logical < n)
        positions = np.flatnonzero(safe)
        ids = logical[safe]
        unique = np.unique(ids)
        actual = int(np.count_nonzero(valid))
        row = dict(row=t, batch=t//c.query, query=t % c.query, topk=c.topk,
                   visible_keys=n, offset=offset, expected_count=count, actual_count=actual,
                   in_range_count=len(ids), unique_count=len(unique),
                   padding_count=int(np.count_nonzero(~valid)), expected_padding_count=c.topk-count,
                   missing_unique_count=max(0, count-len(unique)), issues=[])
        rows.append(row)

        def issue(code, message, **detail):
            errors.append(f"row {t} (batch={t//c.query}, query={t % c.query}): {message}")
            error_types[code] = error_types.get(code, 0)+1
            row["issues"].append(dict(code=code, message=message, **detail))

        def examples(pos, with_reference=False):
            result = []
            for p in pos[:detail_limit]:
                item = dict(position=int(p), index=int(ix[p]), logical_index=int(logical[p]))
                if val is not None:
                    item["value"] = json_number(val[p])
                if with_reference and safe[p]:
                    truth_value = scores[t, logical[p]]
                    item["reference"] = json_number(truth_value)
                    if val is not None and np.isfinite(val[p]) and np.isfinite(truth_value):
                        item["abs_error"] = abs(float(val[p])-float(truth_value))
                        item["tolerance"] = atol+rtol*abs(float(truth_value))
                result.append(item)
            return result

        if actual != count:
            issue("count_mismatch", f"有效数量不符: expected={count}, actual={actual}, "
                  f"padding={c.topk-actual} (expected={c.topk-count})", expected=count, actual=actual)
        bad = np.flatnonzero(valid & ~safe)
        row["out_of_range_count"] = len(bad)
        if len(bad):
            issue("out_of_range", f"索引越界: count={len(bad)}, logical range=[0, {n}), offset={offset}",
                  count=len(bad), logical_range=[0, n], examples=examples(bad))
        raw_ids, occurrences = np.unique(ix[valid], return_counts=True)
        duplicate_ids = raw_ids[occurrences > 1]
        row["duplicate_extra_count"] = int(np.sum(occurrences-1))
        if len(duplicate_ids):
            duplicates = []
            for index in duplicate_ids[:detail_limit]:
                pos = np.flatnonzero(valid & (ix == index))
                duplicates.append(dict(index=int(index), logical_index=int(index)-offset,
                                       occurrences=len(pos), positions=pos[:detail_limit].tolist()))
            issue("duplicate_index", f"重复索引: {len(duplicate_ids)} 个索引重复, "
                  f"多占 {row['duplicate_extra_count']} 个输出位置", count=len(duplicate_ids), examples=duplicates)
        if val is not None:
            bad_padding = np.flatnonzero(~valid & ~np.isneginf(val))
            if len(bad_padding):
                issue("padding_score", f"padding 分数错误: {len(bad_padding)} 项 indices=-1 但 values 不是 -Inf",
                      count=len(bad_padding), examples=examples(bad_padding))
            bad_nonfinite = np.flatnonzero(valid & ~np.isfinite(val))
            nonfinite_scores += len(bad_nonfinite)
            if len(bad_nonfinite):
                issue("nonfinite_score", f"有效位置出现 NaN/Inf: count={len(bad_nonfinite)}",
                      count=len(bad_nonfinite), examples=examples(bad_nonfinite, with_reference=True))
            # Compare only addressable, finite pairs; other checks still fail independently.
            finite = positions[np.isfinite(val[safe]) & np.isfinite(scores[t, ids])]
            row["finite_score_comparison_count"] = len(finite)
            compared_scores += len(finite)
            compared_rows += int(bool(len(finite)))
            if len(finite):
                diff = np.abs(val[finite].astype(np.float64)-scores[t, logical[finite]].astype(np.float64))
                row_max = float(np.max(diff))
                max_abs = row_max if max_abs is None else max(max_abs, row_max)
                mismatched = finite[~np.isclose(val[finite], scores[t, logical[finite]], rtol=rtol, atol=atol)]
                if len(mismatched):
                    issue("score_mismatch", f"有限分数不符: count={len(mismatched)}, max_abs_error={row_max:g}",
                          count=len(mismatched), examples=examples(mismatched, with_reference=True))
        if count:
            truth = scores[t, ids]
            threshold = float(np.partition(scores[t, :n], n-count)[n-count])
            tol = atol+rtol*abs(threshold)
            row["topk_reference_threshold"] = json_number(threshold)
            below = positions[truth < threshold-tol]
            if len(below):
                issue("below_topk", f"选中了低于 TopK 门槛的索引: count={len(below)}, "
                      f"threshold={threshold:g}, tolerance={tol:g}", count=len(below),
                      examples=examples(below, with_reference=True))
            omitted = np.ones(n, dtype=bool)
            omitted[ids] = False
            # Keep the original tie-aware ranking check, even on incomplete rows.
            if len(truth):
                omitted_ids = np.flatnonzero(omitted & (scores[t, :n] > float(np.min(truth))+tol))
                if len(omitted_ids):
                    best = omitted_ids[np.argsort(scores[t, omitted_ids])[-detail_limit:][::-1]]
                    issue("omitted_higher_score", f"漏选了更高分索引: count={len(omitted_ids)}",
                          count=len(omitted_ids), examples=[dict(index=int(i)+offset, logical_index=int(i),
                              reference=json_number(scores[t, i])) for i in best])
            canonical = np.argpartition(scores[t, :n], n-count)[n-count:]
            overlap = len(np.intersect1d(unique, canonical))/count
            recall.append(overlap)
        else:
            overlap = None
        row["canonical_set_overlap"] = overlap
    return dict(passed=not errors, errors=errors[:20], error_count=len(errors),
                errors_truncated=len(errors) > 20, error_types=error_types, row_details=rows,
                failed_row_count=sum(bool(row["issues"]) for row in rows), checked_row_count=len(rows),
                values_checked=values is not None, detail_limit=detail_limit,
                finite_score_comparison_count=compared_scores, finite_score_comparison_row_count=compared_rows,
                nonfinite_selected_score_count=nonfinite_scores,
                max_abs_finite_score_error=max_abs,
                canonical_set_overlap_mean=float(np.mean(recall)) if recall else None,
                canonical_set_overlap_row_count=len(recall))


def print_validation(label, result):
    """Print every failed row, with bounded examples per issue (full arrays in NPZ)."""
    print(f"  {label}: {'PASS' if result['passed'] else 'FAIL'}; "
          f"failed_rows={result['failed_row_count']}/{result['checked_row_count']}", flush=True)
    for row in result["row_details"]:
        if not row["issues"]:
            continue
        print(f"    row={row['row']} batch={row['batch']} query={row['query']} (均从0开始), "
              f"TopK={row['topk']}, 可见key={row['visible_keys']}, offset={row['offset']}\n"
              f"      应有效={row['expected_count']}, 实际非-1={row['actual_count']}, "
              f"越界={row['out_of_range_count']}, 范围内去重后={row['unique_count']}, "
              f"缺少有效且不重复的项={row['missing_unique_count']}\n"
              f"      padding={row['padding_count']} (应为{row['expected_padding_count']})", flush=True)
        for issue in row["issues"]:
            print(f"      [{issue['code']}] {issue['message']}", flush=True)
            for sample in issue.get("examples", []):
                print("        " + json.dumps(sample, ensure_ascii=False, allow_nan=False), flush=True)
            if issue.get("count", 0) > len(issue.get("examples", [])) and "examples" in issue:
                print(f"        (示例仅显示 {len(issue['examples'])}/{issue['count']} 项；完整输出保存在 NPZ)", flush=True)
    if not result["passed"]:
        print(f"    finite_score_comparisons={result['finite_score_comparison_count']}, "
              f"max_abs_finite_score_error={result['max_abs_finite_score_error']}; "
              "有限分数误差小不代表数量、索引或 NaN 检查通过。", flush=True)


def inspect_results(path, detail_limit=8):
    """Recheck saved failure arrays using saved parameters, without Torch/NPU or writes."""
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source/"results.json" if (source/"results.json").is_file() else source/"results/results.json"
    if not source.is_file():
        raise FileNotFoundError(f"Saved results.json not found: {source}")
    report = json.loads(source.read_text())
    print(f"离线分析: {source}\n仅分析已保存的失败数组，不调用算子，不修改原始报告。", flush=True)
    inspected, failed = 0, False
    for entry in report["cases"]:
        c = Case(**entry["case"])
        print(f"[{c.name}] SAVED input={asdict(c)}, seed={entry.get('seed')}", flush=True)
        for name, previous in entry.get("backends", {}).items():
            filename = f"{c.name}_{name}_failure.npz"
            if Path(filename).name != filename:
                raise ValueError("Invalid saved case/backend filename")
            archive = source.parent/filename
            if not archive.is_file():
                print(f"  {name}: 原记录={previous.get('status')}; 无失败 NPZ，未重新校验。", flush=True)
                failed |= previous.get("status") != "PASS"
                continue
            with np.load(archive, allow_pickle=False) as saved:
                d = {k: saved[k] for k in ("seqused_q", "seqused_k", "output_idx_offset")}
                d["cmp_residual_k"] = saved["cmp_residual_k"] if "cmp_residual_k" in saved else None
                scores = saved["reference"]
                print(f"  失败数组: {archive}", flush=True)
                for label, indices, values in (("return_value=True", saved["indices"], saved["values"]),
                                               ("return_value=False", saved["indices_only"], None)):
                    result = validate(c, d, scores, indices, values,
                                      report.get("args", {}).get("rtol", 1/128),
                                      report.get("args", {}).get("atol", 2.5e-5), detail_limit=detail_limit)
                    print_validation(f"{name} {label}", result)
                    failed |= not result["passed"]
            inspected += 1
    if not inspected:
        print("没有找到可重新校验的失败 NPZ；未执行精度测试。", flush=True)
        return 1
    print(f"离线重检 {'FAIL' if failed else 'PASS'}: {inspected} 个失败数组文件；原报告保持不变。", flush=True)
    return 1 if failed else 0


def oracle_outputs(c, d, scores):
    ix = np.full((c.batch*c.query, 1, c.topk), -1, np.int32)
    val = np.full(ix.shape, -np.inf, np.float32)
    for t, row in enumerate(scores):
        ids = np.flatnonzero(np.isfinite(row))
        ids = ids[np.argsort(row[ids])[-c.topk:]]
        ix[t, 0, :len(ids)] = ids + d["output_idx_offset"][t, 0]
        val[t, 0, :len(ids)] = row[ids]
    return ix, val


def self_test():
    # Known nibble order and group scale, not just reference-versus-itself.
    p = np.full((1, 64), 0xA2, np.uint8)
    decoded = decode(p, np.array([[[127, 128], [129, 130]]], np.uint8))
    assert np.array_equal(decoded[0, :4], [1, -1, 1, -1])
    assert decoded[0, 32] == 2 and decoded[0, 96] == 8
    assert bf16_rne([1+1/256, 1+3/256]).tolist() == [1.0, 1+2/128]
    for c in cases_for("smoke"):
        d = make_data(c, 42)
        scores = reference(c, d, chunk=64)
        assert np.array_equal(scores, reference(c, d, chunk=2048))
        ix, val = oracle_outputs(c, d, scores)
        assert validate(c, d, scores, ix, val, 1/128, 2.5e-5)["passed"]
        bad = ix.copy(); bad[0, 0, 0] = 1000000
        assert not validate(c, d, scores, bad, val, 1/128, 2.5e-5)["passed"]
        bad = val.copy(); bad[0, 0, 0] += 100
        assert not validate(c, d, scores, ix, bad, 1/128, 2.5e-5)["passed"]
        bad = val.copy(); bad[0, 0, 0] = np.inf
        failure = validate(c, d, scores, ix, bad, 1/128, 2.5e-5)
        assert not failure["passed"]
        json.dumps(failure, allow_nan=False)
        duplicate = ix.copy(); duplicate[0, 0, 0] = duplicate[0, 0, 1]
        assert not validate(c, d, scores, duplicate, val, 1/128, 2.5e-5)["passed"]
        if c.name == "smoke":
            bad_ix, bad_val = ix.copy(), val.copy()
            lowest = int(np.argmin(scores[0]))
            bad_ix[0, 0, 0] = lowest+c.offset
            bad_val[0, 0, 0] = scores[0, lowest]
            assert not validate(c, d, scores, bad_ix, bad_val, 1/128, 2.5e-5)["passed"]
        if np.any(ix == -1):
            bad = val.copy(); bad[ix == -1] = 0
            assert not validate(c, d, scores, ix, bad, 1/128, 2.5e-5)["passed"]
    c = Case("known", batch=1, query=1, key=2, topk=1)
    d = make_data(c, 0)
    d["q"].fill(0x22); d["k"].fill(0x22)
    d["sq"].fill(127); d["sk"].fill(127); d["w"].fill(1)
    assert np.all(reference(c, d) == 4096)  # 128 per head, 32 heads.
    d["w"].fill(-1)
    assert np.all(reference(c, d) == -4096)
    d["seqused_q"].fill(0)
    scores = reference(c, d)
    ix, val = oracle_outputs(c, d, scores)
    assert validate(c, d, scores, ix, val, 1/128, 2.5e-5)["passed"]
    c = Case("causal_formula", batch=1, query=3, key=4, mask=3, ratio=2)
    d = make_data(c, 0)
    d["cmp_residual_k"].fill(1)
    assert [visible_length(c, d, 0, r) for r in range(3)] == [3, 4, 4]
    print("PASS: decode, BF16 rounding, known scores, page/chunk consistency, validator rejection checks")


def load_backends(args):
    result = {}
    use_dsl = args.backend in ("both", "dsl")
    use_installed_dsl = use_dsl and args.dsl_source == "ops"
    # The experimental wheel's __init__ reads sys.modules["cann_ops_transformer.ops"].
    # find_spec() is not enough: finish importing Native before any DSL package.
    # This prerequisite also applies when only the DSL backend is requested.
    if args.backend in ("both", "native") or use_installed_dsl:
        native = importlib.import_module("cann_ops_transformer.ops")
        if args.backend in ("both", "native"):
            result["native"] = (native.quant_lightning_indexer, native.quant_lightning_indexer_metadata)
    if use_installed_dsl:
        importlib.import_module("cann_ops_transformer_dsl")
        registered = importlib.import_module(
            "cann_ops_transformer_dsl.ops.attention.quant_lightning_indexer_dsl")
        # Keep the Torch dispatcher for timing; do not replace it with the Python wrapper.
        dispatcher = importlib.import_module("cann_ops_transformer.ops.ds41")
        print(f"Installed DSL wrapper loaded: {registered.__file__}", flush=True)
    if use_dsl:
        files = resolve_dsl_files(args)
        install_dsl_modules(files, namespace="ops" if args.dsl_source == "ops" else "")
        print("DSL source mapping (unmodified files):", flush=True)
        for name, path in files.items():
            print(f"  {name}: {path}", flush=True)
    if use_dsl:
        if args.dsl_source == "samples":
            mod = importlib.import_module("quant_lightning_indexer_dsl")
            meta = importlib.import_module("quant_lightning_indexer_metadata_dsl")
        else:
            mod = meta = dispatcher
        result["dsl"] = (mod.quant_lightning_indexer, meta.quant_lightning_indexer_metadata)
    return result


def bind_call(torch, c, d, name, funcs):
    op, meta = funcs
    shared = dict(cu_seqlens_q=d["cu_seqlens_q"], seqused_q=d["seqused_q"],
                  seqused_k=d["seqused_k"], cmp_residual_k=d["cmp_residual_k"],
                  max_seqlen_q=c.query, mask_mode=c.mask, cmp_ratio=c.ratio,
                  layout_q="TND", layout_k="PA_BBND")
    mk = dict(shared, batch_size=c.batch, max_seqlen_k=c.key,
              num_heads_q=c.heads, num_heads_k=1, head_dim=128, topk=c.topk)
    mode = 5 if name == "native" else 1
    if name == "native":
        mk["quant_mode"] = mode
        if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "float8_e8m0fnu"):
            raise RuntimeError("PyTorch must expose float4_e2m1fn_x2 and float8_e8m0fnu for native MXFP4")
        inputs = (d["q"].view(torch.float4_e2m1fn_x2), d["k"].view(torch.float4_e2m1fn_x2),
                  d["w"], d["sq"].view(torch.float8_e8m0fnu), d["sk"].view(torch.float8_e8m0fnu))
    else:
        inputs = tuple(d[k] for k in ("q", "k", "w", "sq", "sk"))
    metadata = meta(**mk)
    torch.npu.synchronize()
    # Native V2 currently allocates 2048 entries although its README says 1024.
    # Treat its own metadata producer as authoritative; DSL requires exactly 1024.
    if (metadata.dtype != torch.int32 or metadata.ndim != 1 or metadata.numel() == 0
            or metadata.device != d["q"].device or not metadata.is_contiguous()
            or (name == "dsl" and metadata.numel() != 1024)):
        raise RuntimeError(f"Unexpected {name} metadata: {metadata.shape}, {metadata.dtype}")
    kwargs = dict(shared, block_table=d["block_table"], output_idx_offset=d["output_idx_offset"],
                  metadata=metadata, topk=c.topk, quant_mode=mode)
    def call(return_value):
        return op(*inputs, **kwargs, return_value=return_value)
    call.metadata_shape = list(metadata.shape)
    return call


def bench(torch, call, warmup, iterations, repeats):
    for _ in range(warmup):
        output = call(False)
    torch.npu.synchronize()
    event_us, wall_us = [], []
    for _ in range(repeats):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        tic = time.perf_counter()
        start.record()
        for _ in range(iterations):
            output = call(False)
        end.record()
        torch.npu.synchronize()
        wall_us.append((time.perf_counter()-tic)*1e6/iterations)
        event_us.append(start.elapsed_time(end)*1000/iterations)
    return dict(event_batch_mean_us=event_us, event_median_us=statistics.median(event_us),
                event_min_us=min(event_us), event_max_us=max(event_us),
                wall_batch_mean_us=wall_us, wall_median_us=statistics.median(wall_us),
                iterations=iterations, repeats=repeats, warmup=warmup, return_value=False,
                timing_scope="eager operator dispatch; metadata prepared outside; events may include launch gaps")


def profile(torch, torch_npu, call, path):
    p = torch_npu.profiler
    with p.profile(activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
                   on_trace_ready=p.tensorboard_trace_handler(str(path)),
                   record_shapes=True) as prof:
        for _ in range(5):
            output = call(False)
            torch.npu.synchronize()
            prof.step()


def file_identity(fn):
    try:
        path = inspect.getsourcefile(fn)
        return dict(module=fn.__module__, file=path,
                    sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None)
    except (TypeError, OSError, AttributeError):
        return dict(repr=str(fn))


def save_report(out, report):
    out.mkdir(parents=True, exist_ok=True)
    tmp = out/"results.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(out/"results.json")
    with (out/"summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "backend", "status", "event_median_us", "wall_median_us"])
        writer.writeheader()
        for entry in report["cases"]:
            for name, result in entry.get("backends", {}).items():
                latency = result.get("latency", {})
                writer.writerow(dict(case=entry["case"]["name"], backend=name,
                                     status=result["status"], event_median_us=latency.get("event_median_us"),
                                     wall_median_us=latency.get("wall_median_us")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--inspect-results", help="Recheck saved results.json and failure NPZ files offline; no NPU or source imports")
    parser.add_argument("--suite", choices=["smoke", "standard", "full"], default="smoke")
    parser.add_argument("--case", help="Run one named case from the selected suite")
    parser.add_argument("--topk", type=int, help="Override TopK (1..8192) without changing case names, seeds or inputs")
    parser.add_argument("--backend", choices=["both", "native", "dsl"], default="both")
    parser.add_argument("--dsl-source", choices=["ops", "samples"], default="ops")
    parser.add_argument("--dsl-root", help="Existing cannbot-dsl repo, samples directory or flat ops directory; works with both sources")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=1/128)
    parser.add_argument("--atol", type=float, default=2.5e-5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after a case fails; skip its timings and subsequent cases")
    parser.add_argument("--output", default="qli_results_"+time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if args.inspect_results:
        return inspect_results(args.inspect_results)
    if min(args.warmup, args.iterations, args.repeats) < 1 or min(args.rtol, args.atol) < 0:
        parser.error("Timing counts must be positive and tolerances nonnegative")
    try:
        cases = select_cases(args.suite, args.case, args.topk)
    except ValueError as exc:
        parser.error(str(exc))
    out = Path(args.output).resolve()
    if (out/"results.json").exists():
        parser.error("Output already contains results.json; choose a fresh --output directory")
    report = dict(args=vars(args), cases=[], environment=dict(python=sys.version, platform=platform.platform()),
                  inspected_reference_revisions=dict(ops_transformer="a18ec2631ccf4b8ed08027d7d38bc3c1816f929a",
                                                     cannbot_dsl="606a737579356baa31da5674452f60673681cff0"))
    save_report(out, report)
    try:
        import torch
        import torch_npu
        torch.npu.set_device(args.device)
        report["environment"].update(torch=torch.__version__, torch_npu=torch_npu.__version__,
                                     device=torch.npu.get_device_name(torch.npu.current_device()),
                                     numpy=np.__version__)
        for package in ("cann-ops-transformer", "cann-ops-transformer-dsl", "cannbot-dsl"):
            try:
                report["environment"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report["environment"][package] = "not recorded in package metadata"
        try:
            info = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15)
            (out/"npu-smi.txt").write_text(info.stdout+info.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["environment"]["npu_smi_error"] = str(exc)
        backends = load_backends(args)
        report["implementations"] = {n: [file_identity(f) for f in fs] for n, fs in backends.items()}
        # Resolve and record the actual DSL module in addition to its Torch wrapper.
        dsl_module = "ops.quant_lightning_indexer_dsl" if args.dsl_source == "ops" else "quant_lightning_indexer_dsl"
        if "dsl" in backends:
            report["dsl_implementation"] = file_identity(importlib.import_module(dsl_module).quant_lightning_indexer)
            metadata_module = ("ops." if args.dsl_source == "ops" else "") + "quant_lightning_indexer_metadata_dsl"
            report["dsl_metadata_implementation"] = file_identity(
                importlib.import_module(metadata_module).quant_lightning_indexer_metadata)
        for ci, c in enumerate(cases):
            print(f"[{c.name}] input={asdict(c)}", flush=True)
            entry = dict(case=asdict(c), backends={})
            report["cases"].append(entry)
            seed = args.seed + int.from_bytes(hashlib.sha256(c.name.encode()).digest()[:4], "little")
            entry["seed"] = seed
            cpu = make_data(c, seed)
            print("  Computing full CPU reference ...", flush=True)
            scores = reference(c, cpu)
            tensors = {k: torch.from_numpy(v).to(args.device) if v is not None else None for k, v in cpu.items()}
            outputs, calls = {}, {}
            with torch.inference_mode():
                for name, funcs in backends.items():
                    result = entry["backends"][name] = dict(status="RUNNING")
                    try:
                        call = bind_call(torch, c, tensors, name, funcs)
                        result["metadata_shape"] = call.metadata_shape
                        # First invocation compiles/initializes; excluded from timings.
                        raw = call(True)
                        torch.npu.synchronize()
                        if raw[0].dtype != torch.int32 or raw[1].dtype != torch.bfloat16:
                            raise AssertionError("Expected INT32 indices and BF16 scores")
                        if name == "dsl" and (len(raw) != 4 or raw[2].numel() or raw[3].numel()):
                            raise AssertionError("Candidate-disabled DSL must return two empty candidate tensors")
                        ix, val = raw[0].cpu().numpy(), raw[1].float().cpu().numpy()
                        result["accuracy"] = validate(c, cpu, scores, ix, val, args.rtol, args.atol)
                        print_validation(f"{name} return_value=True", result["accuracy"])
                        outputs[name] = ix
                        no_value = call(False)
                        torch.npu.synchronize()
                        if no_value[1].numel() != 0 or no_value[0].dtype != torch.int32:
                            raise AssertionError("return_value=False contract mismatch")
                        ix_false = no_value[0].cpu().numpy()
                        if ix_false.shape != ix.shape:
                            raise AssertionError("return_value=False index shape mismatch")
                        result["indices_only_accuracy"] = validate(c, cpu, scores, ix_false, None, args.rtol, args.atol)
                        print_validation(f"{name} return_value=False", result["indices_only_accuracy"])
                        passed = result["accuracy"]["passed"] and result["indices_only_accuracy"]["passed"]
                        result["status"] = "PASS" if passed else "FAIL"
                        if passed:
                            calls[name] = call
                        else:
                            failure_path = out/f"{c.name}_{name}_failure.npz"
                            np.savez_compressed(failure_path, indices=ix, values=val,
                                                indices_only=ix_false, reference=scores, **{k:v for k,v in cpu.items() if v is not None})
                            print(f"  完整失败数组: {failure_path}", flush=True)
                    except Exception:
                        result.update(status="ERROR", error=traceback.format_exc())
                        print(result["error"], file=sys.stderr, flush=True)
                    print(f"  {name}: {result['status']}", flush=True)
                    save_report(out, report)
                if args.fail_fast and any(r["status"] != "PASS" for r in entry["backends"].values()):
                    report["stopped_early"] = c.name
                    print("Stopping after failed case; no further reference or timing runs.", flush=True)
                    break
                if len(outputs) == 2:
                    overlaps = []
                    for a, b in zip(outputs["native"][:, 0], outputs["dsl"][:, 0]):
                        a, b = a[a != -1], b[b != -1]
                        overlaps.append(len(np.intersect1d(a, b))/max(1, len(a)) if len(a) else float(len(b) == 0))
                    entry["native_dsl_index_overlap_mean"] = float(np.mean(overlaps))
                # Alternate which backend runs first across cases; timings use the False path checked above.
                order = list(calls)
                if ci % 2: order.reverse()
                for name in order:
                    result = entry["backends"][name]
                    try:
                        result["latency"] = bench(torch, calls[name], args.warmup, args.iterations, args.repeats)
                        if args.profile:
                            profile(torch, torch_npu, calls[name], out/"profiler"/c.name/name)
                        print(f"  {name}: eager event median {result['latency']['event_median_us']:.3f} us", flush=True)
                    except Exception:
                        result.update(status="ERROR", error=traceback.format_exc())
                        print(result["error"], file=sys.stderr, flush=True)
                    save_report(out, report)
                if len(calls) == 2 and all("latency" in entry["backends"][n] for n in calls):
                    a = entry["backends"]["native"]["latency"]["event_median_us"]
                    b = entry["backends"]["dsl"]["latency"]["event_median_us"]
                    entry["eager_event_speedup_native_over_dsl"] = a/b if b > 0 else None
            del calls, tensors, cpu, scores
            save_report(out, report)
    except Exception:
        report["fatal_error"] = traceback.format_exc()
        print(report["fatal_error"], file=sys.stderr)
    failed = bool(report.get("fatal_error")) or any(r["status"] != "PASS" for e in report["cases"] for r in e["backends"].values())
    report["status"] = "FAIL" if failed else "PASS"
    save_report(out, report)
    print(f"{report['status']}: {out}/results.json", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
