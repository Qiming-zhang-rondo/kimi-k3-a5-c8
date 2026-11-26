"""CPU-only contract tests for preparation; never import torch or start an NPU."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_dsl_recent128 as runner


class PartitionTests(unittest.TestCase):
    def test_original_union_is_preserved_with_query_specific_boundaries(self):
        indices = np.array([[0, 8, 12, 13, 14, 15],
                            [0, 8, 12, 13, 15, 16]], dtype=np.int32)
        positions = np.array([15, 16], dtype=np.int32)
        split = runner.partition(indices, positions, window=4, ring_capacity=8,
                                 cold_capacity=6, hot_capacity=4)
        np.testing.assert_array_equal(split['ori_topk_length'][:, 0], [4, 3])
        np.testing.assert_array_equal(split['cmp_topk_length'][:, 0], [2, 3])
        np.testing.assert_array_equal(split['ori_sparse_indices'][0, 0], [4, 5, 6, 7])
        np.testing.assert_array_equal(split['ori_sparse_indices'][1, 0], [5, 7, 0, -1])
        np.testing.assert_array_equal(split['cmp_sparse_indices'][0, 0], [0, 8, -1, -1, -1, -1])
        np.testing.assert_array_equal(split['cmp_sparse_indices'][1, 0], [0, 8, 12, -1, -1, -1])
        # Slot zero in query1 means token16, not old token0. Decode using the
        # query's visible recent interval, and prove exact original selection.
        for t, pos in enumerate(positions):
            slot_to_token = {token % 8: token for token in range(int(pos) - 3, int(pos) + 1)}
            hot_len, cold_len = (int(split[name][t, 0])
                                 for name in ('ori_topk_length', 'cmp_topk_length'))
            hot_slots = split['ori_sparse_indices'][t, 0, :hot_len]
            hot_tokens = np.array([slot_to_token[int(slot)] for slot in hot_slots])
            cold_tokens = split['cmp_sparse_indices'][t, 0, :cold_len]
            self.assertEqual(set(hot_tokens) & set(cold_tokens), set())
            np.testing.assert_array_equal(np.sort(np.concatenate((hot_tokens, cold_tokens))), indices[t])

    def test_window_zero_returns_original_cold_selection(self):
        ids = np.array([[0, 7, 15]], np.int32)
        result = runner.partition(ids, np.array([15], np.int32), 0, 8, 3, 4)
        np.testing.assert_array_equal(result['cmp_sparse_indices'][0, 0], ids[0])
        np.testing.assert_array_equal(result['ori_sparse_indices'], np.full((1, 1, 4), -1))
        self.assertEqual(int(result['ori_topk_length'][0, 0]), 0)
        self.assertEqual(int(result['cmp_topk_length'][0, 0]), 3)

    def test_duplicate_future_and_capacity_fail_before_device(self):
        pos = np.array([15], np.int32)
        for ids in ([[13, 13]], [[14, 16]], [[-1, 15]]):
            with self.assertRaisesRegex(ValueError, 'unique visible'):
                runner.partition(np.array(ids, np.int32), pos, 4, 8, 2, 4)
        with self.assertRaisesRegex(ValueError, 'ring capacity'):
            runner.partition(np.array([[12, 15]], np.int32), pos, 4, 2, 2, 4)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            runner.partition(np.array([[12, 13, 14, 15]], np.int32), pos, 4, 8, 4, 3)


class PreparationTests(unittest.TestCase):
    @staticmethod
    def small_args(*extra):
        # Retain the default W128/Q4/page128, but avoid allocating an 8K cache.
        return runner.arguments(['--keys', '384', '--topk', '160', *extra])

    def test_default_ring256_and_wraparound_provenance(self):
        args = self.small_args()
        host = runner.prepare_host(args)
        self.assertEqual(host['ring_capacity'], 256)
        self.assertEqual(host['hot_ring_bytes'].shape, (256, 544))
        first = int(host['positions'][0]) - args.window + 1
        # Q4 needs W+Q-1=131 retained tokens. This shape crosses modulo256.
        logical = np.arange(first, args.keys, dtype=np.int32)
        self.assertEqual(len(logical), args.window + args.queries - 1)
        np.testing.assert_array_equal(host['tags'][logical % 256], logical)
        unused = np.ones(256, dtype=bool)
        unused[logical % 256] = False
        np.testing.assert_array_equal(host['tags'][unused], np.full(unused.sum(), -1))
        for t, row in enumerate(host['indices']):
            valid_hot = row[row >= int(host['positions'][t]) - args.window + 1]
            np.testing.assert_array_equal(host['tags'][valid_hot % 256], valid_hot)

    def test_covered_hot_selects_exactly128_without_changing_topk(self):
        args = self.small_args('--selection', 'covered_hot')
        host = runner.prepare_host(args)
        arm = host['arms']['fp4_fp8_w128']
        np.testing.assert_array_equal(arm['ori_topk_length'], np.full((4, 1), 128))
        np.testing.assert_array_equal(arm['cmp_topk_length'], np.full((4, 1), 32))
        for t, pos in enumerate(host['positions']):
            expected_hot = np.arange(int(pos) - 127, int(pos) + 1)
            actual_hot = host['indices'][t][host['indices'][t] >= int(pos) - 127]
            np.testing.assert_array_equal(actual_hot, expected_hot)
            self.assertEqual(len(np.unique(host['indices'][t])), args.topk)
            np.testing.assert_array_equal(host['tags'][arm['ori_sparse_indices'][t, 0]], expected_hot)

    def test_uniform_selection_and_bytes_reproduce_same_seed(self):
        a = runner.prepare_host(self.small_args('--selection', 'uniform', '--seed', '37'))
        b = runner.prepare_host(self.small_args('--selection', 'uniform', '--seed', '37'))
        for key in ('q', 'kv', 'positions', 'indices', 'cold_bytes', 'hot_ring_bytes', 'tags'):
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        c_q, c_kv, c_pos, c_indices = runner.make_inputs(
            self.small_args('--selection', 'uniform', '--seed', '38'))
        self.assertFalse(np.array_equal(a['indices'], c_indices))
        np.testing.assert_array_equal(a['positions'], c_pos)
        for row, pos in zip(a['indices'], a['positions']):
            self.assertTrue(np.all(row <= pos))

    def test_nonfinite_output_or_reference_cannot_pass_gate(self):
        expected = np.ones((2, 3, 4), np.float32)
        for actual, reference in ((np.full_like(expected, np.nan), expected),
                                  (np.full_like(expected, np.inf), expected),
                                  (expected, np.full_like(expected, np.nan))):
            checked = runner.gate(actual, reference, .01)
            self.assertFalse(checked['passed'])
            self.assertFalse(checked['finite'])
            self.assertIsNone(checked['relative_l2'])
        self.assertTrue(runner.gate(expected, expected, .01)['passed'])
        self.assertFalse(runner.gate(expected * 2, expected, .01)['passed'])


if __name__ == '__main__':
    unittest.main()
