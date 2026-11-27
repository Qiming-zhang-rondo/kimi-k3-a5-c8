"""CPU output metrics, with each last-axis vector treated as one query/head.

This module sets no accuracy acceptance threshold and does not replace the
runner's independent kernel controls. NumPy is imported only when used.
"""

RELATIVE_L2_DENOMINATOR_FLOOR = 1e-30
_METRICS = (
    "relative_l2", "max_row_relative_l2", "cosine_mean", "cosine_min", "cosine_p05",
    "one_minus_cosine_mean", "one_minus_cosine_max", "one_minus_cosine_p95",
    "mae", "max_abs",
)


def compare_outputs(actual, reference):
    """Compare equal-shape real arrays without dropping any nonfinite value.

    All leading axes identify separate rows: [B,Q,H,D] gives B*Q*H cosine
    samples, never just B*Q samples. Both zero vectors have cosine 1; exactly
    one zero vector has cosine 0. Relative L2 uses max(reference norm, 1e-30),
    matching the existing CPU accuracy helpers; the floor is reported.

    Nonfinite input returns INVALID_NONFINITE and None for every metric. A
    finite result is descriptive only; no PASS threshold is invented here.
    Returned values are standard Python objects suitable for finite-only JSON.
    """
    import numpy as np

    raw_actual, raw_reference = np.asarray(actual), np.asarray(reference)
    if raw_actual.shape != raw_reference.shape:
        raise ValueError("comparison shapes differ: " + str(raw_actual.shape) +
                         " versus " + str(raw_reference.shape))
    if raw_actual.ndim < 1 or not raw_actual.size or raw_actual.shape[-1] == 0:
        raise ValueError("comparison requires nonempty last-axis vectors")
    if np.iscomplexobj(raw_actual) or np.iscomplexobj(raw_reference):
        raise ValueError("attention output comparison requires real arrays")
    actual = np.asarray(raw_actual, dtype=np.float64)
    reference = np.asarray(raw_reference, dtype=np.float64)
    rows = actual.reshape(-1, actual.shape[-1])
    ref_rows = reference.reshape(rows.shape)
    bad_actual = ~np.isfinite(actual)
    bad_reference = ~np.isfinite(reference)
    actual_bad = int(bad_actual.sum())
    reference_bad = int(bad_reference.sum())
    result = dict(shape=list(actual.shape), row_shape=list(actual.shape[:-1]),
        row_count=int(len(rows)), vector_dimension=int(actual.shape[-1]),
        comparison_unit="last-axis query-head vector",
        relative_l2_denominator_floor=RELATIVE_L2_DENOMINATOR_FLOOR,
        finite=not (actual_bad or reference_bad),
        actual_nonfinite_count=actual_bad, reference_nonfinite_count=reference_bad,
        nonfinite_count=actual_bad + reference_bad,
        nonfinite_row_count=int(np.any(
            (bad_actual | bad_reference).reshape(rows.shape), axis=-1).sum()),
        status="INVALID_NONFINITE" if actual_bad or reference_bad else "FINITE",
        metrics_valid=False)
    result.update({name: None for name in _METRICS})
    if not result["finite"]:
        return result

    # Row-wise rescaling keeps cosine stable for tiny or large finite values;
    # multiplying the original row norms together would under/overflow sooner.
    def unit_vectors(values):
        magnitude = np.max(np.abs(values), axis=-1)
        scaled = np.divide(values, magnitude[:, None], out=np.zeros_like(values),
                           where=magnitude[:, None] != 0)
        length = np.sqrt(np.sum(scaled * scaled, axis=-1))
        unit = np.divide(scaled, length[:, None], out=np.zeros_like(scaled),
                         where=length[:, None] != 0)
        return unit, magnitude == 0

    actual_unit, actual_zero = unit_vectors(rows)
    reference_unit, reference_zero = unit_vectors(ref_rows)
    cosine = np.clip(np.sum(actual_unit * reference_unit, axis=-1), -1., 1.)
    cosine[actual_zero & reference_zero] = 1.
    cosine_error = 1. - cosine

    def stable_norm(values, axis=None):
        scale = np.max(np.abs(values), axis=axis, keepdims=True)
        scaled = np.divide(values, scale, out=np.zeros_like(values), where=scale != 0)
        norm = scale * np.sqrt(np.sum(scaled * scaled, axis=axis, keepdims=True))
        return norm.reshape(()) if axis is None else np.squeeze(norm, axis=axis)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta = actual - reference
        row_error = stable_norm(delta.reshape(rows.shape), axis=-1)
        row_reference = stable_norm(ref_rows, axis=-1)
        metrics = dict(
            relative_l2=float(stable_norm(delta) / max(
                float(stable_norm(reference)), RELATIVE_L2_DENOMINATOR_FLOOR)),
            max_row_relative_l2=float(np.max(row_error / np.maximum(
                row_reference, RELATIVE_L2_DENOMINATOR_FLOOR))),
            cosine_mean=float(np.mean(cosine)), cosine_min=float(np.min(cosine)),
            cosine_p05=float(np.percentile(cosine, 5)),
            one_minus_cosine_mean=float(np.mean(cosine_error)),
            one_minus_cosine_max=float(np.max(cosine_error)),
            one_minus_cosine_p95=float(np.percentile(cosine_error, 95)),
            mae=float(np.mean(np.abs(delta))), max_abs=float(np.max(np.abs(delta))))
    if not all(np.isfinite(value) for value in metrics.values()):
        # Finite source arrays can still overflow FP64 error arithmetic. Keep
        # the failure explicit and the report JSON-safe instead of emitting NaN.
        result["status"] = "INVALID_NUMERIC_OVERFLOW"
        return result
    result.update(metrics, metrics_valid=True)
    return result
