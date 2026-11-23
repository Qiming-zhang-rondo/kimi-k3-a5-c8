"""CPU orchestration checks; the fake hot reader is not an NPU kernel test."""
import copy
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hotkv"))
from kv_native_twopass import NativeTwoPassBackend, merge_torch, partition_numpy
from kv_window_common import attention_reference
from test_native_twopass_backend import FakeBase, FakeScales, NumpyTorch, Tensor, case_data


class NativeHotReaderAdapterTests(unittest.TestCase):
    def setUp(self):
        self.data = case_data()
        self.scales = FakeScales(self.data)
        self.backend = NativeTwoPassBackend(self.data, scale_backend=self.scales)
        self.instances = []
        base, instances = self.scales.base, self.instances

        class FakePreparedHotMerge:
            """Use the CPU reference native reader on an isolated log object.

            This deliberately does NOT claim to simulate Triton compilation,
            packed C8 decoding, its rounding, or kernel performance.
            """
            def __init__(self, q_nope, q_rope, ring_cache, ring_table, row_batch,
                         cold_template, **kwargs):
                self.base = copy.copy(base)
                self.base.q_nope, self.base.q_rope = q_nope, q_rope
                self.base.reads = []
                self.cache, self.table, self.row_batch = ring_cache, ring_table, row_batch
                self.kwargs, self.calls = kwargs, []
                self.metadata = {"implementation": "CPU_TEST_DOUBLE", "kernel_launches": 1,
                                 "block_keys": kwargs["block_keys"], "block_heads": kwargs["block_heads"]}
                self.output = Tensor(np.empty_like(cold_template[0]))
                self.hot_statistics = tuple(Tensor(np.empty_like(x)) for x in cold_template[1:])
                instances.append(self)

            def call(self, cold_state, recent_ring, cold_present, recent_present, *, statistics=False):
                self.calls.append((cold_state, recent_ring, cold_present, recent_present, statistics))
                hot = FakeBase._read(self.base, "c8", recent_ring, cache=self.cache,
                                     table=self.table, statistics=True)
                self.output[:] = merge_torch(NumpyTorch, cold_state, hot, cold_present, recent_present)
                if statistics:
                    self.hot_statistics[0][:] = np.where(np.asarray(recent_present)[:, None, None], hot[1], -np.inf)
                    self.hot_statistics[1][:] = np.where(np.asarray(recent_present)[:, None, None], hot[2], 0)
                return self.output

        self.patch = mock.patch.dict(sys.modules, {"kv_native_hot_merge":
                                     types.SimpleNamespace(PreparedHotMerge=FakePreparedHotMerge)})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def prepare(self, recipe="baseline", **kwargs):
        return self.backend.prepare(recipe, 2, index_capacity="bounded", merge="triton",
                                    reader="hot_merge", **kwargs)

    def test_hot_merge_runtime_uses_one_native_cold_read_and_original_writers(self):
        for recipe in ("baseline", "alg2_max7"):
            with self.subTest(recipe=recipe):
                runner = self.prepare(recipe)
                fake, base = runner._hot_merge, self.scales.base
                # Native cold/hot template reads happen only during preparation.
                self.assertEqual([r["kind"] for r in base.reads[-2:]], ["c4", "c8"])
                self.assertEqual(fake.kwargs["hot_output_dtype"], np.dtype(np.float32))
                base.reads.clear()
                base.writes.clear()
                reference = attention_reference(self.data, kv_c4=runner.decoded_kv(),
                                                hot_kv=runner.decoded_hot_kv(), window=2)
                with mock.patch.object(runner, "_hot_read", side_effect=AssertionError("Timed native hot read")), \
                     mock.patch.object(runner, "_merge", side_effect=AssertionError("Separate timed merge")):
                    for invoke in (runner.call, lambda: runner.run(write=False, dynamic=True),
                                   lambda: runner.run(write=True, dynamic=True)):
                        np.testing.assert_allclose(invoke(), reference, rtol=1e-6, atol=1e-6)
                self.assertEqual([r["kind"] for r in base.reads], ["c4"]*3)
                self.assertTrue(all(r["statistics"] for r in base.reads))
                self.assertEqual(len(fake.calls), 3)
                self.assertFalse(any(call[-1] for call in fake.calls))
                self.assertEqual([w[0] for w in base.writes], [recipe, "c8"])
                for _, kv, rope in base.writes:
                    self.assertIs(kv, base.current_kv)
                    self.assertIs(rope, base.current_rope)
                self.assertEqual(runner.metadata["native_attention_calls_per_read"], 1)
                self.assertEqual(runner.metadata["merge_calls_per_read"], 0)
                self.assertEqual(runner.metadata["custom_hot_attention_merge_launches"], 1)

    def test_preparation_cache_distinguishes_reader_and_both_tile_dimensions(self):
        default = self.prepare()
        self.assertIs(default, self.prepare())
        key32 = self.prepare(hot_block_keys=32)
        head32 = self.prepare(hot_block_heads=32)
        self.assertIsNot(default, key32)
        self.assertIsNot(default, head32)
        self.assertEqual(len(self.instances), 3)
        self.assertEqual(key32._hot_merge.kwargs["block_keys"], 32)
        self.assertEqual(head32._hot_merge.kwargs["block_heads"], 32)
        native = self.backend.prepare("baseline", 2, index_capacity="bounded", merge="torch", reader="native")
        self.assertIsNot(default, native)
        self.assertIsNone(native._hot_merge)

    def test_zero_window_bypasses_hot_import_ring_partition_and_custom_writer(self):
        with mock.patch("kv_native_twopass.fixed_ring_plan", side_effect=AssertionError("W0 ring setup")):
            runner = self.backend.prepare("baseline", 0, reader="hot_merge", merge="torch",
                                          partition="triton_scan", writer="fused_store")
            runner.call()
            runner.run(write=True, dynamic=True)
        self.assertFalse(self.instances)
        self.assertIsNone(runner._hot_merge)
        self.assertFalse(hasattr(runner, "cold_cache"))
        self.assertFalse(hasattr(runner, "hot_cache"))
        self.assertEqual(self.scales.base.native_calls, [("baseline", False), ("baseline", True)])
        self.assertEqual(self.scales.base.reads, [])
        self.assertEqual(self.scales.base.writes, [])

    def test_invalid_combinations_fail_before_preparing_a_device_reader(self):
        invalid = [dict(reader="unknown"), dict(reader="hot_merge", mode="split_c4", merge="triton"),
                   dict(reader="hot_merge", merge="torch"), dict(reader="hot_merge", merge="triton_log512"),
                   dict(hot_block_keys=8), dict(hot_block_keys=True), dict(hot_block_heads=64)]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.backend.prepare("baseline", 2, **options)
        self.assertFalse(self.instances)
        self.assertEqual(self.scales.base.reads, [])

    def test_component_hot_merge_does_not_hide_a_native_cold_or_hot_read(self):
        runner = self.prepare()
        base = self.scales.base
        base.reads.clear()
        stages = runner.component_calls()
        self.assertEqual([r["kind"] for r in base.reads], ["c4"])
        self.assertEqual(set(stages), {"partition", "cold_read", "hot_read_merge", "cold_write", "hot_write"})
        base.reads.clear()
        base.writes.clear()
        stages["hot_read_merge"]()
        stages["hot_read_merge"]()
        self.assertEqual(base.reads, [])
        self.assertEqual(base.writes, [])
        self.assertEqual(len(runner._hot_merge.calls), 2)
        stages["cold_read"]()
        self.assertEqual([r["kind"] for r in base.reads], ["c4"])

    def test_reader_diagnostics_pass_and_preserve_original_queries_and_caches(self):
        runner = self.prepare()
        base = self.scales.base
        q, qr = base.q_nope, base.q_rope
        snapshots = [np.array(x, copy=True) for x in (q, qr, runner.cold_cache, runner.hot_cache)]
        diagnostic = runner.reader_diagnostics()
        self.assertTrue(diagnostic["passed"])
        self.assertEqual(set(diagnostic["checks"]), {"combined_vs_native_twopass", "hot_only_vs_native_hot",
                          "cold_only_vs_native_cold", "both_empty", "zero_query_hot_counts"})
        self.assertTrue(all(check["passed"] for check in diagnostic["checks"].values()))
        self.assertEqual(diagnostic["checks"]["zero_query_hot_counts"]["max_abs_count_error"], 0.)
        self.assertGreater(diagnostic["checks"]["zero_query_hot_counts"]["ignored_empty_rows"], 0)
        self.assertIs(base.q_nope, q)
        self.assertIs(base.q_rope, qr)
        for actual, saved in zip((q, qr, runner.cold_cache, runner.hot_cache), snapshots):
            np.testing.assert_array_equal(actual, saved)
        self.assertEqual(len(self.instances), 2)
        self.assertFalse(np.any(np.asarray(self.instances[-1].base.q_nope)))
        self.assertTrue(self.instances[-1].calls[-1][-1])
        json.dumps(diagnostic, allow_nan=False)

    def test_noop_reader_fails_combined_and_independent_output_controls(self):
        runner = self.prepare()
        with mock.patch.object(runner._hot_merge, "call", return_value=Tensor(np.zeros_like(runner._hot_merge.output))):
            diagnostic = runner.reader_diagnostics()
        self.assertFalse(diagnostic["passed"])
        for name in ("combined_vs_native_twopass", "hot_only_vs_native_hot", "cold_only_vs_native_cold"):
            self.assertFalse(diagnostic["checks"][name]["passed"])
        self.assertTrue(diagnostic["checks"]["both_empty"]["passed"])
        self.assertTrue(diagnostic["checks"]["zero_query_hot_counts"]["passed"])

    def test_zero_query_statistics_bias_fails_even_when_attention_outputs_match(self):
        runner = self.prepare()
        cls, original_call = type(runner._hot_merge), type(runner._hot_merge).call
        original_q = np.array(self.scales.base.q_nope, copy=True)

        def biased_stats(instance, cold_state, recent_ring, cold_present, recent_present, *, statistics=False):
            output = original_call(instance, cold_state, recent_ring, cold_present, recent_present, statistics=statistics)
            if statistics:
                instance.hot_statistics[1][:] += np.asarray(recent_present)[:, None, None]
            return output

        with mock.patch.object(cls, "call", biased_stats):
            diagnostic = runner.reader_diagnostics()
        self.assertFalse(diagnostic["passed"])
        self.assertFalse(diagnostic["checks"]["zero_query_hot_counts"]["passed"])
        self.assertEqual(diagnostic["checks"]["zero_query_hot_counts"]["max_abs_count_error"], 1.)
        for name, check in diagnostic["checks"].items():
            if name != "zero_query_hot_counts":
                self.assertTrue(check["passed"])
        np.testing.assert_array_equal(self.scales.base.q_nope, original_q)
        json.dumps(diagnostic, allow_nan=False)

    def test_descending_partition_preserves_integer_ids_and_truncates_before_gather(self):
        runner = self.backend.prepare("baseline", 2, partition="argsort_desc", index_capacity="bounded")
        base = self.scales.base
        base.indices = Tensor(np.full(base.indices.shape, -1, np.int32))
        base.indices[:, 0, :4] = [2**24+3, 2**24+1, 2**24+4, 2**24+2]
        base.positions = Tensor(np.full(base.rows, 2**24+4, np.int32))
        expected = partition_numpy(base.indices, base.positions, 2, runner.capacity, "bounded")
        sort_calls, gather_calls = [], []

        def argsort(value, dim=-1, descending=False):
            sort_calls.append((np.array(value, copy=True), descending))
            return Tensor(np.argsort(-np.asarray(value) if descending else np.asarray(value), axis=dim))

        def gather(value, dim, order):
            gather_calls.append((value.shape, order.shape, value.dtype))
            return Tensor(np.take_along_axis(value, order, axis=dim))

        with mock.patch.object(NumpyTorch, "argsort", side_effect=argsort), \
             mock.patch.object(NumpyTorch, "gather", side_effect=gather):
            actual = runner.partition()
        self.assertEqual(len(sort_calls), 2)
        self.assertTrue(all(descending and key.dtype == np.float32 for key, descending in sort_calls))
        self.assertEqual([shape[-1] for _, shape, _ in gather_calls], [base.indices.shape[-1], 2])
        self.assertTrue(all(dtype == np.int32 for _, _, dtype in gather_calls))
        for name in ("old", "recent_global", "recent_ring", "old_present", "recent_present"):
            np.testing.assert_array_equal(actual[name], expected[name])
        self.assertEqual(actual["recent_global"][0, 0].tolist(), [2**24+3, 2**24+4])


if __name__ == "__main__":
    unittest.main()
