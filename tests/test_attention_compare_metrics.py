"""Meaningful CPU metrics contracts, independent of Torch and device kernels."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "experimental/single_pass/attention_compare_metrics.py"
spec = importlib.util.spec_from_file_location("attention_compare_metrics_test", SOURCE)
metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_module)
compare_outputs = metrics_module.compare_outputs


class AttentionCompareMetricsTests(unittest.TestCase):
    def test_identity_preserves_original_shape_and_last_axis_row_count(self):
        values = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        result = compare_outputs(values, values.copy())
        self.assertEqual(result["shape"], [2, 3, 4, 5])
        self.assertEqual(result["row_shape"], [2, 3, 4])
        self.assertEqual(result["row_count"], 24)
        self.assertEqual(result["vector_dimension"], 5)
        self.assertTrue(result["finite"] and result["metrics_valid"])
        self.assertEqual(result["relative_l2"], 0)
        self.assertEqual(result["max_row_relative_l2"], 0)
        self.assertAlmostEqual(result["cosine_mean"], 1)
        self.assertEqual(result["mae"], 0)
        self.assertEqual(result["max_abs"], 0)
        self.assertNotIn("passed", result)  # The runner retains its own gates.
        json.dumps(result, allow_nan=False)

    def test_cosine_is_per_query_head_rather_than_flattening_heads(self):
        actual = np.array([[[1., 0.], [0., 0.]]])
        reference = np.array([[[1., 0.], [0., 1.]]])
        result = compare_outputs(actual, reference)
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["cosine_mean"], .5)
        self.assertEqual(result["cosine_min"], 0)
        self.assertAlmostEqual(result["cosine_p05"], .05)
        self.assertAlmostEqual(result["relative_l2"], math.sqrt(.5))
        self.assertEqual(result["max_row_relative_l2"], 1)
        flattened_cosine = 1 / math.sqrt(2)
        self.assertNotAlmostEqual(result["cosine_mean"], flattened_cosine)

    def test_analytic_norms_cosine_percentiles_and_absolute_errors(self):
        actual = np.array([[2., 0.], [0., 1.], [-1., 0.], [0., 0.]])
        reference = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 0.]])
        result = compare_outputs(actual, reference)
        self.assertAlmostEqual(result["relative_l2"], math.sqrt(7 / 3))
        self.assertEqual(result["max_row_relative_l2"], 2)
        self.assertEqual(result["cosine_mean"], .25)
        self.assertEqual(result["cosine_min"], -1)
        self.assertAlmostEqual(result["cosine_p05"], -.85)
        self.assertEqual(result["one_minus_cosine_mean"], .75)
        self.assertEqual(result["one_minus_cosine_max"], 2)
        self.assertAlmostEqual(result["one_minus_cosine_p95"], 1.85)
        self.assertEqual(result["mae"], .625)
        self.assertEqual(result["max_abs"], 2)

    def test_zero_vectors_have_explicit_contract(self):
        for actual, reference, cosine in (([0., 0.], [0., 0.], 1.),
                                          ([1., 0.], [0., 0.], 0.),
                                          ([0., 0.], [1., 0.], 0.)):
            with self.subTest(actual=actual, reference=reference):
                result = compare_outputs(actual, reference)
                self.assertEqual(result["cosine_mean"], cosine)
                self.assertEqual(result["row_count"], 1)
                self.assertTrue(result["metrics_valid"])
                json.dumps(result, allow_nan=False)
        self.assertEqual(compare_outputs([0., 0.], [0., 0.])["relative_l2"], 0)

    def test_nonfinite_anywhere_invalidates_all_metrics_without_filtering(self):
        actual = np.array([[[1., 2.], [np.nan, np.inf]]])
        reference = np.array([[[1., 2.], [3., -np.inf]]])
        result = compare_outputs(actual, reference)
        self.assertEqual(result["status"], "INVALID_NONFINITE")
        self.assertFalse(result["finite"])
        self.assertFalse(result["metrics_valid"])
        self.assertEqual(result["actual_nonfinite_count"], 2)
        self.assertEqual(result["reference_nonfinite_count"], 1)
        self.assertEqual(result["nonfinite_count"], 3)
        self.assertEqual(result["nonfinite_row_count"], 1)
        self.assertEqual(result["shape"], [1, 2, 2])
        for name in metrics_module._METRICS:
            self.assertIsNone(result[name])
        json.dumps(result, allow_nan=False)

    def test_tiny_cosine_is_scale_invariant_and_bad_shapes_do_not_broadcast(self):
        actual = np.array([[1e-200, 0.], [0., 1e-200]])
        reference = np.array([[1e-200, 0.], [1e-200, 0.]])
        result = compare_outputs(actual, reference)
        self.assertEqual(result["cosine_mean"], .5)
        self.assertAlmostEqual(result["relative_l2"] / 1e-170, math.sqrt(2))
        for actual, reference in ((np.zeros((2, 3)), np.zeros((1, 3))),
                                  (np.array(1.), np.array(1.)),
                                  (np.zeros((0, 3)), np.zeros((0, 3))),
                                  (np.zeros((3, 0)), np.zeros((3, 0))),
                                  (np.array([1j]), np.array([1j]))):
            with self.subTest(shape=np.shape(actual)), self.assertRaises(ValueError):
                compare_outputs(actual, reference)

    def test_large_finite_values_do_not_overflow_squared_norms(self):
        result = compare_outputs([[2e200, 0.]], [[1e200, 0.]])
        self.assertTrue(result["metrics_valid"])
        self.assertEqual(result["relative_l2"], 1.)
        self.assertEqual(result["max_row_relative_l2"], 1.)
        self.assertEqual(result["cosine_mean"], 1.)
        self.assertEqual(result["max_abs"], 1e200)
        json.dumps(result, allow_nan=False)

    def test_import_is_lazy_and_does_not_import_numpy_or_torch(self):
        script = """
import builtins, importlib.util
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'torch', 'torch_npu'):
        raise AssertionError('Unexpected eager import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
spec = importlib.util.spec_from_file_location('isolated_metrics', %r)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
""" % str(SOURCE)
        outcome = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(outcome.returncode, 0, outcome.stderr)


if __name__ == "__main__":
    unittest.main()
