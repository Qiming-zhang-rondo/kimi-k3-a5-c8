#!/usr/bin/env python3
"""CPU diagnostics for comparing two TopK selections over the same candidates.

No operator-specific rounding model or universal accuracy threshold is assumed.
The supplied scores describe the baseline candidate domain, not necessarily the
exact internal arithmetic of either device operator. All positions are zero-based.
"""
from collections import Counter
import math

import numpy as np


def json_safe(value):
    """Convert NumPy values and nonfinite numbers to strict JSON-compatible data."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "+Inf" if value > 0 else "-Inf"
    return value


def _lengths(visible_lengths, candidate_count):
    if isinstance(candidate_count, (bool, np.bool_)) or not isinstance(candidate_count, (int, np.integer)):
        raise ValueError("candidate_count must be a nonnegative integer")
    if candidate_count < 0:
        raise ValueError("candidate_count must be a nonnegative integer")
    lengths = np.asarray(visible_lengths)
    if lengths.ndim != 1 or lengths.dtype.kind not in "iu":
        raise ValueError("visible_lengths must be a one-dimensional integer array")
    if np.any(lengths < 0) or np.any(lengths > candidate_count):
        raise ValueError("visible_lengths must lie in [0, candidate_count]")
    return lengths


def _index_matrix(indices):
    array = np.asarray(indices)
    if array.ndim == 3 and array.shape[1] == 1:
        return array[:, 0, :]
    return array if array.ndim == 2 else None


def inspect_indices(indices, visible_lengths, candidate_count):
    """Validate counts, padding, bounds and uniqueness without hiding bad entries.

    ``-1`` is the only padding sentinel. Values in ``[visible_length, N)``
    are invalid too: they address masked/invisible candidates. Floating point
    indices are rejected even if every value happens to be integral.
    """
    lengths = _lengths(visible_lengths, candidate_count)
    array = np.asarray(indices)
    matrix = _index_matrix(array)
    result = dict(passed=False, shape=list(array.shape), dtype=str(array.dtype),
                  candidate_count=int(candidate_count), errors=[], error_types={}, rows=[])
    global_issues = []
    if matrix is None:
        global_issues.append(dict(code="index_shape", message="Expected indices [R,K] or [R,1,K]"))
    else:
        if matrix.shape[0] != len(lengths):
            global_issues.append(dict(code="row_count", message="Index row count differs from visible_lengths"))
        if matrix.shape[1] < 1:
            global_issues.append(dict(code="empty_topk", message="TopK dimension must be positive"))
        if matrix.dtype.kind not in "iu":
            global_issues.append(dict(code="index_dtype", message="Indices must have an integer dtype"))
    result["errors"].extend(global_issues)
    counts = Counter(issue["code"] for issue in global_issues)
    if global_issues:
        for t, length in enumerate(lengths):
            result["rows"].append(dict(row=t, visible_length=int(length), valid=False,
                                       issues=[dict(issue) for issue in global_issues],
                                       expected_count=None, actual_count=None,
                                       in_range_count=None, unique_count=None, padding_count=None,
                                       duplicate_extra_count=None, out_of_range_count=None))
        result["error_types"] = dict(counts)
        return result

    topk = matrix.shape[1]
    for t, length in enumerate(lengths):
        length = int(length)
        raw = matrix[t]
        live = raw != -1
        safe = live & (raw >= 0) & (raw < length)
        unique, occurrence = np.unique(raw[live], return_counts=True)
        duplicate_ids = unique[occurrence > 1]
        bad_positions = np.flatnonzero(live & ~safe)
        count = int(np.count_nonzero(live))
        expected = min(topk, length)
        row = dict(row=t, visible_length=length, topk=topk, expected_count=expected,
                   actual_count=count, in_range_count=int(np.count_nonzero(safe)),
                   unique_count=int(len(np.unique(raw[safe]))),
                   padding_count=int(np.count_nonzero(~live)), expected_padding_count=topk-expected,
                   duplicate_extra_count=int(np.sum(occurrence-1)),
                   out_of_range_count=len(bad_positions), issues=[])
        row["missing_unique_count"] = max(0, expected-row["unique_count"])
        if count != expected:
            row["issues"].append(dict(code="count_mismatch", expected=expected, actual=count))
        if len(bad_positions):
            row["issues"].append(dict(code="out_of_range", allowed_range=[0, length],
                                     count=len(bad_positions), positions=bad_positions.tolist(),
                                     indices=raw[bad_positions].tolist()))
        if len(duplicate_ids):
            row["issues"].append(dict(code="duplicate_index", count=len(duplicate_ids),
                                     extra_slots=row["duplicate_extra_count"],
                                     duplicates=[dict(index=int(index),
                                                      positions=np.flatnonzero(live & (raw == index)).tolist())
                                                 for index in duplicate_ids]))
        row["valid"] = not row["issues"]
        for issue in row["issues"]:
            counts[issue["code"]] += 1
            result["errors"].append(dict(row=t, **issue))
        result["rows"].append(row)
    result["error_types"] = dict(counts)
    result["passed"] = not result["errors"]
    return result


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(count=0, mean=None, min=None, max=None, p05=None, p50=None, p95=None)
    return dict(count=len(values), mean=float(np.mean(values)), min=float(np.min(values)),
                max=float(np.max(values)), p05=float(np.percentile(values, 5)),
                p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)))


def compare_topk(reference_indices, test_indices, reference_scores, visible_lengths,
                 tie_rtol=0.0, tie_atol=0.0, detail_limit=16):
    """Compare index SETS and explain differences in baseline-score coordinates.

    Structural failures invalidate the row's selection metrics, rather than
    deleting invalid indices and reporting an inflated overlap. Boundary-tie
    counts are descriptive only; a TopK mismatch is not itself a kernel failure.

    Full dropped/added sets are retained. Examples are bounded and sorted by
    descending baseline score. ``baseline_score_regret`` is the mean baseline
    score of the reference selection minus that of the test selection. It can be
    negative when the reference operator does not select the exact mathematical
    TopK under the supplied diagnostic scores.
    """
    if not math.isfinite(tie_rtol) or not math.isfinite(tie_atol) or min(tie_rtol, tie_atol) < 0:
        raise ValueError("Tie tolerances must be finite and nonnegative")
    if isinstance(detail_limit, bool) or not isinstance(detail_limit, (int, np.integer)) or detail_limit < 1:
        raise ValueError("detail_limit must be a positive integer")
    scores = np.asarray(reference_scores)
    if scores.ndim != 2 or scores.dtype.kind not in "iuf":
        raise ValueError("reference_scores must be a real numeric [R,N] array")
    lengths = _lengths(visible_lengths, scores.shape[1])
    ref_validation = inspect_indices(reference_indices, lengths, scores.shape[1])
    test_validation = inspect_indices(test_indices, lengths, scores.shape[1])
    ref = _index_matrix(reference_indices)
    test = _index_matrix(test_indices)
    global_errors = []
    if scores.shape[0] != len(lengths):
        global_errors.append(dict(code="score_row_count", message="Score row count differs from visible_lengths"))
    if ref is not None and test is not None and ref.shape[1] != test.shape[1]:
        global_errors.append(dict(code="topk_mismatch", message="The two TopK dimensions differ"))
    result = dict(status="VALID_COMPARISON", reference_validation=ref_validation,
                  test_validation=test_validation, tie_rtol=float(tie_rtol), tie_atol=float(tie_atol),
                  detail_limit=int(detail_limit), errors=list(global_errors), rows=[], summary={})
    for t, n in enumerate(lengths):
        n = int(n)
        rv, tv = ref_validation["rows"][t], test_validation["rows"][t]
        issues = [dict(issue) for issue in global_errors]
        if not rv["valid"]:
            issues.append(dict(code="invalid_reference_indices", details=rv["issues"]))
        if not tv["valid"]:
            issues.append(dict(code="invalid_test_indices", details=tv["issues"]))
        if t < scores.shape[0]:
            nonfinite = np.flatnonzero(~np.isfinite(scores[t, :n]))
            if len(nonfinite):
                issues.append(dict(code="nonfinite_reference_scores", count=len(nonfinite),
                                   examples=[dict(index=int(i), score=json_safe(scores[t, i]))
                                             for i in nonfinite[:detail_limit]]))
        row = dict(row=t, visible_length=n, expected_count=rv["expected_count"],
                   valid=not issues, status="INVALID" if issues else "COMPARABLE", issues=issues,
                   overlap_count=None, recall=None, replaced_count=None, replacement_fraction=None,
                   exact_set_match=None, dropped=None, added=None, dropped_examples=None, added_examples=None,
                   topk_threshold=None, boundary_gap=None, tie_tolerance=None,
                   dropped_above_boundary_count=None, added_below_boundary_count=None,
                   boundary_near_dropped_count=None, boundary_near_added_count=None,
                   boundary_only_change=None, baseline_score_regret=None,
                   dropped_min_minus_added_max=None)
        result["rows"].append(row)
        if issues:
            result["errors"].extend(dict(row=t, **issue) for issue in issues)
            continue
        count = rv["expected_count"]
        if not count:
            row["status"] = "EMPTY"
            continue
        ref_ids = np.sort(ref[t][ref[t] != -1])
        test_ids = np.sort(test[t][test[t] != -1])
        common = np.intersect1d(ref_ids, test_ids, assume_unique=True)
        dropped = np.setdiff1d(ref_ids, test_ids, assume_unique=True)
        added = np.setdiff1d(test_ids, ref_ids, assume_unique=True)
        partitioned = np.partition(scores[t, :n], n-count)
        threshold = float(partitioned[n-count])
        next_score = float(np.max(partitioned[:n-count])) if n > count else None
        tolerance = tie_atol+tie_rtol*abs(threshold)

        def examples(ids):
            ordered = sorted((int(index) for index in ids), key=lambda index: (-float(scores[t, index]), index))
            return [dict(index=index, reference_score=float(scores[t, index]),
                         distance_from_boundary=float(scores[t, index])-threshold)
                    for index in ordered[:detail_limit]]

        dropped_scores, added_scores = scores[t, dropped], scores[t, added]
        near_dropped = int(np.count_nonzero(np.abs(dropped_scores-threshold) <= tolerance))
        near_added = int(np.count_nonzero(np.abs(added_scores-threshold) <= tolerance))
        regret = float(np.mean(scores[t, ref_ids].astype(np.float64))-
                       np.mean(scores[t, test_ids].astype(np.float64)))
        row.update(overlap_count=len(common), recall=len(common)/count,
                   replaced_count=len(dropped), replacement_fraction=len(dropped)/count,
                   exact_set_match=len(dropped) == 0, dropped=dropped.tolist(), added=added.tolist(),
                   dropped_examples=examples(dropped), added_examples=examples(added),
                   topk_threshold=threshold, boundary_gap=threshold-next_score if next_score is not None else None,
                   tie_tolerance=tolerance,
                   dropped_above_boundary_count=int(np.count_nonzero(dropped_scores > threshold+tolerance)),
                   added_below_boundary_count=int(np.count_nonzero(added_scores < threshold-tolerance)),
                   boundary_near_dropped_count=near_dropped, boundary_near_added_count=near_added,
                   boundary_only_change=bool(len(dropped) and near_dropped == len(dropped) and near_added == len(added)),
                   baseline_score_regret=regret,
                   dropped_min_minus_added_max=(float(np.min(dropped_scores))-float(np.max(added_scores)))
                   if len(dropped) else None)

    selected = [row for row in result["rows"] if row["valid"] and row["expected_count"]]
    invalid_rows = sum(not row["valid"] for row in result["rows"])
    result["status"] = "INVALID" if invalid_rows or global_errors else "VALID_COMPARISON"
    result["summary"] = dict(
        rows_total=len(lengths), invalid_rows=invalid_rows,
        comparable_rows=len(selected), empty_rows=sum(row["status"] == "EMPTY" for row in result["rows"]),
        changed_rows=sum(not row["exact_set_match"] for row in selected),
        exact_set_match_rows=sum(row["exact_set_match"] for row in selected),
        boundary_only_changed_rows=sum(row["boundary_only_change"] for row in selected),
        recall=_stats([row["recall"] for row in selected]),
        replacement_fraction=_stats([row["replacement_fraction"] for row in selected]),
        baseline_score_regret=_stats([row["baseline_score_regret"] for row in selected]),
        boundary_gap=_stats([row["boundary_gap"] for row in selected if row["boundary_gap"] is not None]),
        replaced_count_total=sum(row["replaced_count"] for row in selected),
        note="Selection differences are measurements, not a universal accuracy PASS/FAIL criterion.",
    )
    return json_safe(result)
