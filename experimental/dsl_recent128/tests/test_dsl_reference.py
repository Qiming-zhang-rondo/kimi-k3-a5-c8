import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsl_reference import (attention_reference, bf16_round, decode_fp4, decode_fp8,
                           encode_fp4, encode_fp8, metrics, mixed_reference,
                           quantize_rows, unpack_rows)


class FormatTests(unittest.TestCase):
    def test_bf16_ties_even_and_finite_guard(self):
        bits = np.array([0x3F808000, 0x3F818000, 0xBF808000], dtype=np.uint32)
        rounded = bf16_round(bits.view(np.float32)).view(np.uint32)
        np.testing.assert_array_equal(rounded, [0x3F800000, 0x3F820000, 0xBF800000])
        with self.assertRaises(ValueError):
            bf16_round([np.nan])
        with self.assertRaises(ValueError):
            bf16_round([np.finfo(np.float32).max])

    def test_fp4_known_codes_saturation_and_ties(self):
        values = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -.5, -6], np.float32)
        np.testing.assert_array_equal(encode_fp4(values), [0, 1, 2, 3, 4, 5, 6, 7, 9, 15])
        np.testing.assert_array_equal(decode_fp4(encode_fp4(values)), values)
        np.testing.assert_array_equal(encode_fp4([.25, .75, 1.25, 5, 100]), [0, 2, 2, 6, 7])
        self.assertEqual(int(encode_fp4(np.array([-0.0]))[0]), 8)

    def test_fp8_known_codes_and_every_finite_roundtrip(self):
        values = np.array([0, 2 ** -9, 2 ** -6, 1, 2, 448, -448], np.float32)
        np.testing.assert_array_equal(encode_fp8(values), [0, 1, 8, 56, 64, 126, 254])
        codes = np.concatenate((np.arange(127), np.arange(128, 255))).astype(np.uint8)
        np.testing.assert_array_equal(encode_fp8(decode_fp8(codes)), codes)
        # halfway 1.0625 rounds to even code 56 (1.0), 1.1875 to code58.
        np.testing.assert_array_equal(encode_fp8([1.0625, 1.1875, 1000]), [56, 58, 126])
        with self.assertRaises(ValueError):
            decode_fp8([127])

    def test_rows_pack_sizes_scale_positions_and_roundtrip(self):
        rng = np.random.default_rng(9)
        x = bf16_round(rng.normal(size=(3, 512)).astype(np.float32))
        x[0] = 0
        for kind, size in [('fp4', 320), ('fp8', 544)]:
            packed, decoded = quantize_rows(x, kind)
            self.assertEqual(packed.shape, (3, size))
            self.assertEqual(packed.dtype, np.uint8)
            np.testing.assert_array_equal(unpack_rows(packed, kind), decoded)
            np.testing.assert_array_equal(bf16_round(decoded), decoded)
            np.testing.assert_array_equal(decoded[0], np.zeros(512))
            self.assertTrue(np.isfinite(decoded).all())
        # Positive 6, negative 6 under exact scale1 => first byte 0xF7.
        x = np.tile(np.array([6, -6], np.float32), (1, 256))
        packed, decoded = quantize_rows(x, 'fp4')
        self.assertEqual(int(packed[0, 0]), 0xF7)
        np.testing.assert_array_equal(packed[0, 256:260], [0x80, 0x3F, 0x80, 0x3F])
        np.testing.assert_array_equal(decoded, x)

    def test_quantizer_uses_bf16_rounded_scale_and_decode_product(self):
        x = np.full((1, 512), 1.0, np.float32)
        packed, decoded = quantize_rows(x, 'fp4')
        scale = bf16_round(np.array(1 / 6, np.float32))
        expected = bf16_round(np.array(6, np.float32) * scale)
        self.assertEqual(float(decoded[0, 0]), float(expected))
        scales = packed[:, 256:].copy().view('<u2')
        self.assertEqual(int(scales[0, 0]), int(scale.view(np.uint32) >> 16))

    def test_rejects_bad_layout_nonfinite_and_invalid_scale(self):
        with self.assertRaises(ValueError):
            quantize_rows(np.zeros((2, 576)), 'fp4')
        with self.assertRaises(ValueError):
            quantize_rows(np.full((2, 512), np.inf), 'fp8')
        with self.assertRaises(ValueError):
            unpack_rows(np.zeros((1, 320), np.uint8), 'fp4')


class AttentionTests(unittest.TestCase):
    def test_zero_query_sink_counts_and_values(self):
        q = np.zeros((1, 2, 3), np.float32)
        kv = np.arange(12, dtype=np.float32).reshape(4, 3)
        indices = np.array([[0, 2, -1]], np.int32)
        out, lse = attention_reference(q, kv, indices, 1, np.array([0, -np.inf]))
        np.testing.assert_allclose(np.exp(lse), [[3, 2]])
        np.testing.assert_allclose(out[0, 0], (kv[0] + kv[2]) / 3)
        np.testing.assert_allclose(out[0, 1], (kv[0] + kv[2]) / 2)

    def test_empty_side_and_invalid_indices(self):
        q = np.ones((1, 1, 3), np.float32)
        kv = np.ones((4, 3), np.float32)
        out, lse = attention_reference(q, kv, np.array([[-1, -1]]), 1, np.array([0]))
        np.testing.assert_array_equal(out, np.zeros_like(q))
        np.testing.assert_array_equal(lse, [[0]])
        for index in ([[0, 0]], [[0, 4]], [[-1, 0]], [[-2, -1]]):
            with self.assertRaises(ValueError):
                attention_reference(q, kv, np.array(index), 1, np.array([0]))

    def test_mixed_is_exact_selected_replacement(self):
        rng = np.random.default_rng(10)
        q = rng.normal(size=(2, 3, 4)).astype(np.float32)
        cold = rng.normal(size=(8, 4)).astype(np.float32)
        hot = rng.normal(size=(8, 4)).astype(np.float32)
        indices = np.array([[0, 6, 7], [3, 5, 7]], np.int32)
        starts = np.array([6, 5])
        out, lse = mixed_reference(q, cold, hot, indices, starts, .5, np.zeros(3))
        for t in range(2):
            merged = cold.copy()
            merged[starts[t]:] = hot[starts[t]:]
            expected, expected_lse = attention_reference(q[t:t+1], merged, indices[t:t+1], .5, np.zeros(3))
            np.testing.assert_array_equal(out[t], expected[0])
            np.testing.assert_array_equal(lse[t], expected_lse[0])

    def test_stable_softmax_large_scores_and_metrics(self):
        q = np.full((1, 2, 3), 1000, np.float32)
        kv = np.array([[1000, 1000, 1000], [-1000, -1000, -1000]], np.float32)
        out, lse = attention_reference(q, kv, np.array([[0, 1]]), 1, np.zeros(2))
        self.assertTrue(np.isfinite(lse).all())
        np.testing.assert_array_equal(out, np.full_like(out, 1000))
        self.assertEqual(metrics(out, out)['relative_l2'], 0)
        self.assertEqual(metrics(out, out)['mean_row_cosine'], 1)
        bad = out.copy()
        bad[0, 0, 0] = np.nan
        result = metrics(bad, out)
        self.assertFalse(result['finite'])
        self.assertIsNone(result['relative_l2'])


if __name__ == '__main__':
    unittest.main()
