"""CPU native-adapter contracts, not an A5 compilation or latency claim."""
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hotkv"))
from kv_native_twopass import (NativeTwoPassBackend, merge_torch, partition_numpy,
                               validate_partition_width)
from kv_window_common import (attention_reference, flatten_queries, paged_array,
                              query_positions, row_map, synthetic_case, validate_case)


class Tensor(np.ndarray):
    """Small NumPy device double; no real Torch/NPU dependency."""
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def view(self, *args):
        if len(args) == 1 and not isinstance(args[0], (int, tuple)):
            return Tensor(np.asarray(self).view(args[0]))
        return Tensor(np.asarray(self).reshape(*args))

    def clone(self):
        return Tensor(self.copy())

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self))

    def to(self, dtype):
        return Tensor(self.astype(dtype))

    def float(self):
        return self.to(np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def numel(self):
        return self.size

    def element_size(self):
        return self.itemsize

    def any(self, dim=None, **kwargs):
        return Tensor(np.asarray(self).any(axis=dim, **kwargs))

    def index_copy_(self, dim, slots, source):
        if dim != 0:
            raise AssertionError("Only token-axis writes are expected")
        self[np.asarray(slots)] = source
        return self


class NumpyTorch:
    int8, uint8, int32, int64 = np.int8, np.uint8, np.int32, np.int64
    bool, float32, bfloat16, float8_e4m3fn = np.bool_, np.float32, np.float32, np.uint8
    @staticmethod
    def arange(*args, dtype=None, device=None):
        return Tensor(np.arange(*args, dtype=dtype))
    @staticmethod
    def where(condition, left, right):
        return Tensor(np.where(condition, left, right))
    @staticmethod
    def zeros_like(value):
        return Tensor(np.zeros_like(value))
    @staticmethod
    def ones_like(value):
        return Tensor(np.ones_like(value))
    @staticmethod
    def empty_like(value):
        return Tensor(np.empty_like(value))
    @staticmethod
    def full_like(value, fill):
        return Tensor(np.full_like(value, fill))
    @staticmethod
    def maximum(left, right):
        return Tensor(np.maximum(left, right))
    @staticmethod
    def exp(value):
        return Tensor(np.exp(value))
    @staticmethod
    def log(value):
        return Tensor(np.log(value))
    @staticmethod
    def isfinite(value):
        return Tensor(np.isfinite(value))
    @staticmethod
    def argsort(value, dim=-1):
        return Tensor(np.argsort(value, axis=dim))
    @staticmethod
    def gather(value, dim, order):
        return Tensor(np.take_along_axis(value, order, axis=dim))


def case_data():
    data = synthetic_case(batch=2, queries=3, keys=257, heads=2, dim=64,
                          rope_dim=0, topk=16, page=128)
    # Exercise short and long rows with both all-hot and all-cold subsets.
    data["q_lengths"][1] = 2
    data["k_lengths"][1] = 250
    data["indices"][:] = -1
    rows = [[0, 253, 254], [0, 2, 254, 255], [0, 128, 255, 256],
            [247, 248], [0, 1, 2]]
    for (b, q), ids in zip(row_map(data), rows):
        data["indices"][b, q, 0, :len(ids)] = ids
    return validate_case(data)


def byte_cache(values):
    values = np.asarray(values, np.float32)
    return values.view(np.uint8).reshape(*values.shape[:-1], values.shape[-1]*4)


class FakeBase:
    """Emulate native reader with exact softmax on a simple packed float cache.

    The byte payload is a test format only. Production format paths delegate
    unchanged to the original native adapters, checked through their kwargs.
    """
    def __init__(self, data):
        self.data, self.torch, self.device = data, NumpyTorch, "cpu"
        self.batch, self.queries, self.heads, self.dim = data["q_nope"].shape
        self.keys, self.page, self.scale = data["kv"].shape[1], data["page"], data["scale"]
        self.mapping = row_map(data)
        self.rows = len(self.mapping)
        self.indices = Tensor(flatten_queries(data, "indices"))
        self.positions = Tensor(query_positions(data).astype(np.int32))
        self.q_nope = Tensor(flatten_queries(data, "q_nope"))
        self.q_rope = Tensor(flatten_queries(data, "q_rope"))
        self.table_cpu = np.arange(self.batch*3, dtype=np.int32)[::-1].reshape(self.batch, 3)
        self.table = Tensor(self.table_cpu)
        self.lengths = Tensor(data["k_lengths"].astype(np.int32))
        self.c8_logical = byte_cache(data["kv"]*.999)
        self.current_kv = Tensor(np.stack([data["kv"][b, p] for (b, _), p in zip(self.mapping, self.positions)]))
        self.current_rope = Tensor(np.zeros((self.rows, 1, 0), np.float32))
        self.current_slots = Tensor(np.asarray([self.table_cpu[b, p//self.page]*self.page+p%self.page
                                                for (b, _), p in zip(self.mapping, self.positions)], np.int64))
        self.reads, self.writes, self.native_calls = [], [], []

    def tensor(self, value, dtype=None):
        return Tensor(np.asarray(value, dtype=dtype).copy())

    def host(self, value):
        return np.array(value, copy=True)

    def decoded_kv(self, kind):
        return self.data["kv"] * (.999 if kind == "c8" else .95)

    def call(self, kind):
        self.native_calls.append((kind, False))
        return attention_reference(self.data, kv_c4=self.decoded_kv(kind))

    def run(self, kind, write=False):
        self.native_calls.append((kind, write))
        return attention_reference(self.data, kv_c4=self.decoded_kv(kind))

    def quantize(self, kv, rope, kind):
        self.writes.append((kind, kv, rope))
        return Tensor(byte_cache(kv*.999).reshape(self.rows, -1)), None, None

    def _read(self, kind, indices, *, cache, table=None, lengths=None, statistics):
        self.reads.append(dict(kind=kind, indices=indices, cache=cache, table=table,
                               lengths=lengths, statistics=statistics))
        assert statistics
        table = self.table if table is None else table
        storage = np.asarray(cache).view(np.uint8).view(np.float32).reshape(-1, self.page, 1, self.dim)
        output = np.empty((self.rows, self.heads, self.dim), np.float32)
        maximum, total = np.empty((self.rows, self.heads, 1), np.float32), np.empty((self.rows, self.heads, 1), np.float32)
        for row, (b, _) in enumerate(self.mapping):
            ids = np.asarray(indices)[row, 0]
            ids = ids[ids >= 0]
            kv = storage[np.asarray(table)[b, ids//self.page], ids%self.page, 0].astype(np.float64)
            scores = np.asarray(self.q_nope[row], np.float64) @ kv.T * self.scale
            maximum[row] = scores.max(-1, keepdims=True)
            weights = np.exp(scores-maximum[row])
            total[row] = weights.sum(-1, keepdims=True)
            output[row] = (weights @ kv)/total[row]
        return Tensor(output), Tensor(maximum), Tensor(total)


class FakeScales:
    def __init__(self, data):
        self.base, self.environment, self.variants = FakeBase(data), {}, {}

    def prepare(self, recipe):
        if recipe not in self.variants:
            base = self.base
            factor = .95 if recipe == "baseline" else .97
            decoded = base.data["kv"]*factor
            cache = Tensor(paged_array(byte_cache(decoded), base.page, base.table_cpu)).view(np.int8)

            def quantize(kv, rope):
                base.writes.append((recipe, kv, rope))
                return Tensor(byte_cache(kv*factor).reshape(base.rows, -1)), None, None

            def native_call():
                base.native_calls.append((recipe, False))
                return attention_reference(base.data, kv_c4=decoded)

            def native_run(write=False):
                base.native_calls.append((recipe, write))
                return attention_reference(base.data, kv_c4=decoded)

            self.variants[recipe] = types.SimpleNamespace(cache=cache, metadata={"recipe": recipe},
                decoded_kv=lambda: decoded.copy(), _quantize=quantize, call=native_call, run=native_run)
        return self.variants[recipe]


class PartitionTests(unittest.TestCase):
    def test_float_sort_key_exactness_guard_at_float32_integer_boundary(self):
        width = 2**23
        validate_partition_width("argsort_float", width)
        keys = np.array([0, width-1, width, 2*width-2, 2*width-1], np.int64)
        np.testing.assert_array_equal(keys.astype(np.float32).astype(np.int64), keys)
        with self.assertRaisesRegex(ValueError, r"2\*S-1 <= 2\*\*24"):
            validate_partition_width("argsort_float", width+1)
        # The float restriction does not silently restrict the integer baseline.
        validate_partition_width("argsort_int", width+1)

    def test_bounded_partition_keeps_order_wraps_ring_and_handles_empty_rows(self):
        indices = np.array([[[8, 0, 9, -1]], [[8, 9, -1, -1]], [[0, 1, -1, -1]]], np.int32)
        full = partition_numpy(indices, np.array([9, 9, 9]), 2, 4, "full")
        bounded = partition_numpy(indices, np.array([9, 9, 9]), 2, 4, "bounded")
        self.assertEqual(bounded["recent_global"].shape, (3, 1, 2))
        np.testing.assert_array_equal(bounded["recent_global"][:, 0], [[8, 9], [8, 9], [0, -1]])
        np.testing.assert_array_equal(bounded["recent_ring"][:, 0], [[0, 1], [0, 1], [0, -1]])
        np.testing.assert_array_equal(bounded["old_present"], [True, False, True])
        np.testing.assert_array_equal(bounded["recent_present"], [True, True, False])
        np.testing.assert_array_equal(bounded["old"], full["old"])

    def test_duplicate_future_and_invalid_indices_rejected(self):
        for ids in ([1, 1], [1, 4], [1, -2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                partition_numpy(np.array([[ids]], np.int32), np.array([3]), 2, 4, "bounded")

    def test_bounded_width_never_exceeds_topk(self):
        parts = partition_numpy(np.array([[[0, 1, -1]]], np.int32), np.array([1]), 128, 128, "bounded")
        self.assertEqual(parts["recent_global"].shape[-1], 3)
        self.assertEqual(parts["old_counts"].tolist(), [0])


class NativeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.data = case_data()
        self.scales = FakeScales(self.data)
        self.backend = NativeTwoPassBackend(self.data, scale_backend=self.scales)

    def test_zero_window_does_no_partition_ring_merge_or_writer_work(self):
        with mock.patch("kv_native_twopass.fixed_ring_plan", side_effect=AssertionError("W0 allocated ring")):
            runner = self.backend.prepare("baseline", 0, merge="triton", partition="triton_scan")
            np.testing.assert_array_equal(runner.call(), attention_reference(self.data, kv_c4=runner.decoded_kv()))
            runner.run(write=True, dynamic=True)
            self.assertFalse(hasattr(runner, "cold_cache"))
            self.assertFalse(hasattr(runner, "hot_cache"))
            self.assertFalse(hasattr(runner, "_triton_merge"))
            self.assertFalse(hasattr(runner, "_partition_runner"))
        self.assertEqual(self.scales.base.native_calls, [("baseline", False), ("baseline", True)])
        self.assertEqual(self.scales.base.reads, [])
        self.assertEqual(self.scales.base.writes, [])
        self.assertTrue(runner.metadata["zero_window_native_bypass"])

    def test_device_partition_matches_cpu_for_full_and_bounded_mixed(self):
        for mode in ("mixed", "split_c4"):
            for policy in ("full", "bounded"):
                with self.subTest(mode=mode, policy=policy):
                    runner = self.backend.prepare("baseline", 2, mode=mode, index_capacity=policy)
                    actual = runner.partition()
                    for key in ("old", "recent_global", "old_present", "recent_present"):
                        np.testing.assert_array_equal(actual[key], runner.prepared[key])
                    if mode == "mixed":
                        np.testing.assert_array_equal(actual["recent_ring"], runner.prepared["recent_ring"])
                    self.assertEqual(actual["old"].dtype, np.int32)
                    self.assertEqual(actual["recent_global"].dtype, np.int32)

    def test_float_candidate_changes_sort_key_dtype_only_and_preserves_large_token_ids(self):
        runners = {algo: self.backend.prepare("baseline", 2, partition=algo)
                   for algo in ("argsort_int", "argsort_float")}
        base = self.scales.base
        # IDs above the float32 exact-integer range must never be cast. This
        # partition-only probe needs no KV allocation at those logical IDs.
        base.indices = Tensor(np.full(base.indices.shape, -1, np.int32))
        base.indices[:, 0, :3] = [2**24+1, 2**24+2, 2**24+3]
        base.positions = Tensor(np.full(base.rows, 2**24+3, np.int32))
        expected = partition_numpy(base.indices, base.positions, 2, runners["argsort_int"].capacity)
        original_argsort = NumpyTorch.argsort
        for algo, runner in runners.items():
            observed_keys = []
            def argsort(key, dim=-1):
                observed_keys.append(np.array(key, copy=True))
                return original_argsort(key, dim=dim)
            with mock.patch.object(NumpyTorch, "argsort", side_effect=argsort):
                actual = runner.partition()
            self.assertEqual(len(observed_keys), 2)
            for key in observed_keys:
                self.assertEqual(key.dtype, np.float32 if algo == "argsort_float" else np.int32)
                self.assertLessEqual(int(key.max()), 2*base.indices.shape[-1]-1)
            for key in ("old", "recent_global", "recent_ring", "old_present", "recent_present"):
                np.testing.assert_array_equal(actual[key], expected[key])
            self.assertEqual(actual["recent_global"][0, 0, :2].tolist(), [2**24+2, 2**24+3])
            self.assertEqual(actual["recent_global"].dtype, np.int32)
            runner.indices_cpu = np.asarray(base.indices).copy()
            runner.positions_cpu = np.asarray(base.positions).copy()
            with mock.patch.object(base, "host", side_effect=AssertionError("Integer diagnostic used float-converting host()")):
                diagnostic = runner.partition_diagnostics()
            self.assertTrue(diagnostic["passed"])
            self.assertEqual(diagnostic["fields"]["recent_global"]["actual_dtype"], "int32")
            self.assertEqual(diagnostic["fields"]["recent_global"]["mismatch_count"], 0)
        self.assertIsNot(runners["argsort_int"], runners["argsort_float"])

    def test_float_partition_keeps_full_pipeline_math_for_both_modes_and_capacities(self):
        for mode in ("mixed", "split_c4"):
            for policy in ("full", "bounded"):
                with self.subTest(mode=mode, policy=policy):
                    integer = self.backend.prepare("baseline", 2, mode=mode, index_capacity=policy)
                    floating = self.backend.prepare("baseline", 2, mode=mode, index_capacity=policy,
                                                    partition="argsort_float")
                    np.testing.assert_array_equal(floating.call(), integer.call())
                    np.testing.assert_array_equal(floating.run(write=True, dynamic=True), integer.run(write=True, dynamic=True))
                    self.assertEqual(floating.metadata["partition"], "argsort_float")
                    self.assertEqual(floating.metadata["partition_implementation"]["performance"], "NOT_VALIDATED")
                    self.assertTrue(floating.partition_diagnostics()["passed"])

    def test_partition_diagnostic_checks_hot_indices_presence_dtype_and_optional_invalid_flags(self):
        runner = self.backend.prepare("baseline", 2, index_capacity="bounded")
        clean = runner.partition()
        self.assertTrue(runner.partition_diagnostics()["passed"])
        changes = ("recent_global", "recent_ring", "recent_present", "dtype", "nonfinite_dtype", "invalid_rows")
        for changed in changes:
            with self.subTest(changed=changed):
                parts = {key: value.clone() for key, value in clean.items()}
                if changed in ("recent_global", "recent_ring"):
                    parts[changed][0, 0, 0] += 1
                elif changed == "recent_present":
                    parts[changed][0] = ~parts[changed][0]
                elif changed in ("dtype", "nonfinite_dtype"):
                    parts["recent_global"] = parts["recent_global"].to(np.float32)
                    if changed == "nonfinite_dtype":
                        parts["recent_global"][0, 0, 0] = np.nan
                else:
                    parts["invalid_rows"] = Tensor(np.array([False, True, False, False, False]))
                with mock.patch.object(runner, "partition", return_value=parts):
                    diagnostic = runner.partition_diagnostics()
                self.assertFalse(diagnostic["passed"])
                self.assertTrue(diagnostic["fields"]["old"]["passed"])
                json.dumps(diagnostic, allow_nan=False)
                if changed == "invalid_rows":
                    self.assertEqual(diagnostic["invalid_rows"], [1])
                elif changed in ("dtype", "nonfinite_dtype"):
                    self.assertEqual(diagnostic["fields"]["recent_global"]["actual_dtype"], "float32")
                else:
                    self.assertEqual(diagnostic["fields"][changed]["mismatch_count"], 1)
                    self.assertTrue(diagnostic["fields"][changed]["first_bad_positions"])

    def test_partition_diagnostic_handles_split_ring_alias_and_skips_native_zero_window(self):
        split = self.backend.prepare("baseline", 2, mode="split_c4", index_capacity="bounded")
        parts = split.partition()
        self.assertIs(parts["recent_ring"], parts["recent_global"])
        diagnostic = split.partition_diagnostics()
        self.assertTrue(diagnostic["passed"])
        self.assertFalse(diagnostic["invalid_rows_checked"])
        native = self.backend.prepare("baseline", 0, partition="triton_scan")
        with mock.patch.object(native, "partition", side_effect=AssertionError("W0 partition invoked")):
            self.assertEqual(native.partition_diagnostics(),
                             {"passed": True, "skipped": True, "scope": "W0 has no partition"})

    def test_triton_partition_factory_prepares_once_reuses_buffers_and_changes_no_readers(self):
        instances = []
        class FakePreparedPartition:
            def __init__(self, indices, positions, **kwargs):
                self.indices, self.positions, self.kwargs = indices, positions, kwargs
                self.calls = 0
                self.metadata = {"algorithm": "triton_scan", "status": "CPU_TEST_DOUBLE"}
                parts = partition_numpy(indices, positions, kwargs["window"], kwargs["ring_capacity"],
                                        "full" if kwargs["hot_width"] == indices.shape[-1] else "bounded")
                self.parts = {key: Tensor(value.copy()) for key, value in parts.items() if not key.endswith("counts")}
                if not kwargs["remap_ring"]:
                    self.parts["recent_ring"] = self.parts["recent_global"]
                instances.append(self)

            def call(self):
                self.calls += 1
                return self.parts

        module = types.SimpleNamespace(PreparedPartition=FakePreparedPartition)
        with mock.patch.dict(sys.modules, kv_native_partition=module):
            runner = self.backend.prepare("baseline", 2, index_capacity="bounded", partition="triton_scan")
            split = self.backend.prepare("baseline", 2, mode="split_c4", partition="triton_scan")
            self.assertIs(runner, self.backend.prepare("baseline", 2, index_capacity="bounded", partition="triton_scan"))
            self.assertEqual(len(instances), 2)
            self.assertEqual(instances[0].calls, 0)
            prepared_output = runner.call()
            self.assertEqual(instances[0].calls, 0)
            dynamic_output = runner.run(write=True, dynamic=True)
            self.assertEqual(instances[0].calls, 1)
            np.testing.assert_allclose(dynamic_output, prepared_output, atol=1e-6)
            first, second = runner.partition(), runner.partition()
            self.assertIs(first, second)
            self.assertIs(first["old"], second["old"])
            self.assertIs(first["recent_ring"], second["recent_ring"])
        self.assertIs(instances[0].indices, self.scales.base.indices)
        self.assertIs(instances[0].positions, self.scales.base.positions)
        self.assertEqual(instances[0].kwargs, {"window": 2, "ring_capacity": runner.capacity,
                                             "hot_width": 2, "remap_ring": True})
        self.assertFalse(instances[1].kwargs["remap_ring"])
        self.assertEqual([call["kind"] for call in self.scales.base.reads], ["c4", "c8", "c4", "c8"])
        self.assertTrue(all(call["statistics"] for call in self.scales.base.reads))

    def test_partition_candidate_failure_never_falls_back_to_argsort(self):
        def unavailable(*args, **kwargs):
            raise RuntimeError("TRITON_PARTITION_UNAVAILABLE")
        with mock.patch.dict(sys.modules, kv_native_partition=types.SimpleNamespace(PreparedPartition=unavailable)):
            with self.assertRaisesRegex(RuntimeError, "TRITON_PARTITION_UNAVAILABLE"):
                self.backend.prepare("baseline", 2, partition="triton_scan")
            runner = self.backend.prepare("baseline", 2, partition="argsort_float")
            self.assertIsNone(runner._partition_runner)
            self.assertTrue(np.isfinite(runner.run(write=True, dynamic=True)).all())

    def test_split_and_mixed_match_math_including_empty_subsets_ring_wrap_and_two_queries(self):
        for recipe in ("baseline", "alg2_max7"):
            for mode in ("mixed", "split_c4"):
                for policy in ("full", "bounded"):
                    with self.subTest(recipe=recipe, mode=mode, policy=policy):
                        runner = self.backend.prepare(recipe, 2, mode=mode, index_capacity=policy)
                        reference = attention_reference(self.data, kv_c4=runner.decoded_kv(),
                            hot_kv=runner.decoded_hot_kv() if mode == "mixed" else None,
                            window=2 if mode == "mixed" else 0)
                        for output in (runner.call(), runner.run(write=False, dynamic=True),
                                       runner.run(write=True, dynamic=True)):
                            np.testing.assert_allclose(output, reference, atol=1e-6, rtol=1e-6)
                        self.assertEqual(runner.metadata["attention_calls_per_read"], 2)
                        self.assertGreater(runner.metadata["old_empty_rows"], 0)
                        self.assertGreater(runner.metadata["hot_empty_rows"], 0)
                        self.assertEqual(hasattr(runner, "hot_cache"), mode == "mixed")

    def test_native_contract_ring_pages_and_independent_cold_caches(self):
        first = self.backend.prepare("baseline", 2, index_capacity="bounded")
        second = self.backend.prepare("baseline", 4, index_capacity="bounded")
        original = self.scales.prepare("baseline").cache.copy()
        self.assertIs(first, self.backend.prepare("baseline", 2, index_capacity="bounded"))
        self.assertFalse(np.shares_memory(first.cold_cache, second.cold_cache))
        self.assertFalse(np.shares_memory(first.cold_cache, self.scales.prepare("baseline").cache))
        first.call()
        cold, hot = self.scales.base.reads[-2:]
        self.assertEqual((cold["kind"], hot["kind"]), ("c4", "c8"))
        self.assertTrue(cold["statistics"] and hot["statistics"])
        self.assertIs(cold["cache"], first.cold_cache)
        self.assertIs(hot["cache"], first.hot_cache)
        self.assertIs(hot["table"], first.ring_table)
        self.assertIs(hot["lengths"], first.ring_lengths)
        self.assertEqual(first.hot_cache.shape[1], self.data["page"])
        self.assertEqual(first.hot_cache.shape[0], self.scales.base.batch*first.capacity//self.data["page"])
        np.testing.assert_array_equal(first.ring_lengths, [first.capacity]*2)
        first.run(write=True)
        np.testing.assert_array_equal(self.scales.prepare("baseline").cache, original)
        np.testing.assert_array_equal(second.cold_cache, original)

    def test_mixed_writer_quantizes_original_bf16_and_split_writes_only_once(self):
        base = self.scales.base
        mixed = self.backend.prepare("alg2_max7", 2)
        split = self.backend.prepare("alg2_max7", 2, mode="split_c4")
        mixed.run(write=True, dynamic=True)
        self.assertEqual([entry[0] for entry in base.writes], ["alg2_max7", "c8"])
        for _, kv, rope in base.writes:
            self.assertIs(kv, base.current_kv)
            self.assertIs(rope, base.current_rope)
        base.writes.clear()
        split.run(write=True, dynamic=True)
        self.assertEqual([entry[0] for entry in base.writes], ["alg2_max7"])
        self.assertFalse(hasattr(split, "hot_cache"))

    def test_dynamic_run_repartitions_and_prepared_call_does_not(self):
        runner = self.backend.prepare("baseline", 2)
        with mock.patch.object(runner, "partition", wraps=runner.partition) as partition:
            runner.call()
            runner.run(write=False, dynamic=False)
            self.assertEqual(partition.call_count, 0)
            runner.run(write=False, dynamic=True)
            runner.run(write=True, dynamic=True)
            self.assertEqual(partition.call_count, 2)

    def test_component_calls_have_no_hidden_attention_in_partition_write_or_merge(self):
        runner = self.backend.prepare("baseline", 2)
        components = runner.component_calls()
        self.assertEqual(set(components), {"partition", "cold_read", "hot_read", "merge", "cold_write", "hot_write"})
        self.scales.base.reads.clear()
        for name in ("partition", "merge", "cold_write", "hot_write"):
            self.assertIsInstance(components[name](), Tensor)
        self.assertEqual(self.scales.base.reads, [])
        components["cold_read"]()
        components["hot_read"]()
        self.assertEqual(len(self.scales.base.reads), 2)
        split = self.backend.prepare("baseline", 2, mode="split_c4")
        self.assertNotIn("hot_write", split.component_calls())

    def test_triton_merge_is_optional_and_receives_raw_states_masks_preallocated_out(self):
        calls = []
        def fake_merge(left, right, out=None, *, left_present, right_present):
            calls.append((left, right, out, left_present, right_present))
            out[:] = merge_torch(NumpyTorch, left, right, left_present, right_present)
            return out
        module = types.SimpleNamespace(merge_states=fake_merge)
        with mock.patch.dict(sys.modules, kv_native_merge=module):
            runner = self.backend.prepare("baseline", 2, merge="triton")
            output = runner.call()
            runner.run(write=False, dynamic=True)
        self.assertIs(output, runner.merge_output)
        self.assertIs(calls[0][2], runner.merge_output)
        self.assertIs(calls[0][3], runner.prepared["old_present"])
        self.assertIs(calls[0][4], runner.prepared["recent_present"])
        self.assertIs(calls[1][2], runner.merge_output)
        self.assertTrue(np.isfinite(output).all())

    def test_triton_output_allocation_uses_actual_native_output_dtype_not_query(self):
        base = self.scales.base
        base.q_nope = base.q_nope.to(np.float16)
        module = types.SimpleNamespace(merge_states=lambda *args, **kwargs: kwargs["out"])
        with mock.patch.dict(sys.modules, kv_native_merge=module):
            runner = self.backend.prepare("baseline", 2, merge="triton")
        self.assertEqual(base.q_nope.dtype, np.float16)
        self.assertEqual(runner.merge_output.dtype, np.float32)
        self.assertEqual(len(base.reads), 1)
        self.assertEqual(base.reads[0]["kind"], "c4")
        self.assertTrue(base.reads[0]["statistics"])
        self.assertIs(base.reads[0]["indices"], runner.prepared["old"])

    def test_invalid_configuration_rejected(self):
        for kwargs in ({"window": -1}, {"window": True}, {"window": 2.5}, {"window": 1, "mode": "other"},
                       {"window": 1, "index_capacity": "compact"}, {"window": 1, "merge": "other"},
                       {"window": 1, "partition": "other"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.backend.prepare("baseline", **kwargs)
        with self.assertRaises(ValueError):
            self.backend.prepare("unknown", 1)

    def test_zero_query_diagnostic_counts_subsets_without_mutating_original_queries(self):
        base = self.scales.base
        query_before = base.q_nope.copy()
        original_query = base.q_nope
        for mode in ("mixed", "split_c4"):
            runner = self.backend.prepare("baseline", 2, mode=mode, index_capacity="bounded")
            diagnostic = runner.zero_query_diagnostics()
            self.assertTrue(diagnostic["passed"])
            self.assertTrue(diagnostic["cold"]["finite"])
            self.assertTrue(diagnostic["hot"]["finite"])
            self.assertEqual(diagnostic["cold"]["max_abs_count_error"], 0.)
            self.assertEqual(diagnostic["hot"]["max_abs_count_error"], 0.)
            self.assertGreater(diagnostic["hot"]["ignored_dummy_rows"], 0)
            self.assertGreater(diagnostic["cold"]["ignored_dummy_rows"], 0)
            json.dumps(diagnostic, allow_nan=False)
        np.testing.assert_array_equal(base.q_nope, query_before)
        self.assertIs(base.q_nope, original_query)

    def test_zero_query_diagnostic_failure_is_json_safe(self):
        runner = self.backend.prepare("baseline", 2)
        original = FakeBase._read

        def corrupt(base, *args, **kwargs):
            output, maximum, total = original(base, *args, **kwargs)
            total[0] = np.nan
            return output, maximum, total

        with mock.patch.object(FakeBase, "_read", corrupt):
            diagnostic = runner.zero_query_diagnostics()
        self.assertFalse(diagnostic["passed"])
        self.assertFalse(diagnostic["cold"]["finite"])
        self.assertIsNone(diagnostic["cold"]["max_abs_count_error"])
        json.dumps(diagnostic, allow_nan=False)


class TorchMergeTests(unittest.TestCase):
    def test_explicit_present_empty_sentinel_is_invalid_not_silently_dropped(self):
        left = (Tensor([[[123., 456.]]]), Tensor([[[-np.inf]]]), Tensor([[[0.]]]))
        right = (Tensor([[[3., 4.]]]), Tensor([[[0.]]]), Tensor([[[1.]]]))
        with np.errstate(invalid="ignore", divide="ignore"):
            actual = merge_torch(NumpyTorch, left, right, Tensor([True]), Tensor([True]))
            ignored = merge_torch(NumpyTorch, left, right, Tensor([False]), Tensor([True]))
        self.assertTrue(np.isnan(actual).all())
        np.testing.assert_array_equal(ignored, right[0])

    def test_known_empty_nan_dummy_is_ignored_but_nonempty_nan_is_not_repaired(self):
        left = (Tensor([[[1., 2.]], [[np.nan, np.nan]], [[np.nan, np.nan]]]),
                Tensor([[[0.]], [[np.nan]], [[np.nan]]]), Tensor([[[2.]], [[np.nan]], [[np.nan]]]))
        right = (Tensor([[[9., 10.]], [[3., 4.]], [[np.nan, np.nan]]]),
                 Tensor([[[0.]], [[0.]], [[np.nan]]]), Tensor([[[2.]], [[1.]], [[np.nan]]]))
        with np.errstate(invalid="ignore"):
            merged = merge_torch(NumpyTorch, left, right, Tensor([True, False, False]), Tensor([False, True, False]))
            np.testing.assert_array_equal(merged, [[[1., 2.]], [[3., 4.]], [[0., 0.]]])
            corrupt = merge_torch(NumpyTorch, left, right, Tensor([True, True, False]), Tensor([True, True, False]))
            self.assertFalse(np.isfinite(corrupt[1]).all())

    def test_large_finite_statistics_do_not_overflow_and_invalid_nonempty_sums_fail(self):
        presence = Tensor([True])
        state = (Tensor([[[1e30, -1e30]]]).float(), Tensor([[[1e30]]]).float(), Tensor([[[3e38]]]).float())
        with np.errstate(over="raise", invalid="raise"):
            merged = merge_torch(NumpyTorch, state, state, presence, presence)
        np.testing.assert_array_equal(merged, state[0])
        for invalid in (0., -1., np.inf, np.nan):
            bad = (state[0], Tensor([[[0.]]]), Tensor([[[invalid]]]))
            with self.subTest(total=invalid), np.errstate(invalid="ignore", divide="ignore"):
                merged = merge_torch(NumpyTorch, state, bad, presence, presence)
                self.assertTrue(np.isnan(merged).all())


if __name__ == "__main__":
    unittest.main()
