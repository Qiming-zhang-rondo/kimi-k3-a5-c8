"""CPU adapter tests for fused hot preparation; no device-kernel claim."""

import sys
import types
import unittest
from unittest import mock

import numpy as np

from test_native_twopass_backend import (
    FakeScales, NativeTwoPassBackend, Tensor, case_data, merge_torch,
    partition_numpy,
)


def fill_tensor(tensor, value):
    tensor[...] = value
    return tensor


def copy_tensor(tensor, value):
    tensor[...] = np.asarray(value)
    return tensor


class FakePartition:
    """Independent CPU oracle, preserving the prepared output objects."""

    def __init__(self, indices, positions, *, window, ring_capacity, hot_width,
                 remap_ring, events, **kwargs):
        self.indices, self.positions = indices, positions
        self.window, self.capacity = window, ring_capacity
        self.policy = "full" if hot_width == indices.shape[-1] else "bounded"
        self.remap_ring, self.events, self.calls = remap_ring, events, 0
        values = self.expected()
        self.outputs = {name: Tensor(value.copy()) for name, value in values.items()}
        self.metadata = dict(implementation="CPU_PARTITION_DOUBLE", variant="two_scan")

    def expected(self):
        values = partition_numpy(self.indices, self.positions, self.window,
                                 self.capacity, self.policy)
        values = {name: value for name, value in values.items() if not name.endswith("counts")}
        if not self.remap_ring:
            values["recent_ring"] = values["recent_global"].copy()
        values["invalid_rows"] = np.zeros(len(self.indices), np.bool_)
        return values

    def produce(self):
        for name, value in self.expected().items():
            self.outputs[name][...] = value
        return self.outputs

    def call(self):
        self.calls += 1
        self.events.append("standalone_partition")
        return self.produce()


class FakeHotPrepare:
    """Use the existing fake native writer and a separate CPU partition producer.

    This verifies backend scheduling and gates, not FP8 or Triton arithmetic.
    """

    def __init__(self, base, hot_cache, hot_slots, partition, events):
        self.base, self.cache, self.slots = base, hot_cache, hot_slots
        self.partition, self.outputs, self.events = partition, partition.outputs, events
        self.calls = 0
        self.skip_partition = self.skip_store = False
        self.metadata = dict(implementation="CPU_HOT_PREPARE_DOUBLE",
                             standalone_partition_calls=0)

    def call(self):
        self.calls += 1
        self.events.append("hot_prepare")
        packed, _, _ = self.base.quantize(self.base.current_kv, self.base.current_rope, "c8")
        if not self.skip_store:
            self.cache.view(np.uint8).view(-1, self.cache.shape[-1]).index_copy_(0, self.slots, packed)
        if not self.skip_partition:
            self.partition.produce()
        return self.outputs


class TestHotPrepareBackend(unittest.TestCase):
    def setUp(self):
        self.data = case_data()
        self.scales = FakeScales(self.data)
        self.backend = NativeTwoPassBackend(self.data, scale_backend=self.scales)
        self.events, self.partitions, self.preparations = [], [], []

        def partition(*args, **kwargs):
            result = FakePartition(*args, events=self.events, **kwargs)
            self.partitions.append(result)
            return result

        def preparation(*args):
            result = FakeHotPrepare(*args, events=self.events)
            self.preparations.append(result)
            return result

        def merge(left, right, out, *, left_present, right_present, **kwargs):
            self.events.append("merge")
            out[...] = merge_torch(self.scales.base.torch, left, right,
                                  left_present, right_present)
            return out

        modules = {
            "kv_native_partition": types.SimpleNamespace(PreparedPartition=partition),
            "kv_native_hot_prepare": types.SimpleNamespace(PreparedHotStorePartition=preparation),
            "kv_native_merge": types.SimpleNamespace(merge_states=merge),
        }
        for patch in (mock.patch.dict(sys.modules, modules),
                      mock.patch.object(Tensor, "fill_", fill_tensor, create=True),
                      mock.patch.object(Tensor, "copy_", copy_tensor, create=True)):
            patch.start()
            self.addCleanup(patch.stop)

    def prepare(self, **kwargs):
        options = dict(mode="mixed", index_capacity="bounded", merge="triton",
                       partition="triton_scan", writer="hot_store_partition", reader="native")
        options.update(kwargs)
        return self.backend.prepare("baseline", 2, **options)

    @staticmethod
    def snapshots(runner):
        return {"cold": np.array(runner.cold_cache, copy=True),
                "hot": np.array(runner.hot_cache, copy=True),
                **{name: np.array(value, copy=True) for name, value in runner._hot_prepare.outputs.items()}}

    def assert_restored(self, runner, saved):
        np.testing.assert_array_equal(runner.cold_cache, saved["cold"])
        np.testing.assert_array_equal(runner.hot_cache, saved["hot"])
        for name, value in runner._hot_prepare.outputs.items():
            np.testing.assert_array_equal(value, saved[name])

    def test_write_dynamic_uses_one_hot_prepare_and_two_unchanged_native_reads(self):
        runner = self.prepare()
        base = self.scales.base
        expected = np.array(runner.call(), copy=True)
        base.reads.clear()
        base.writes.clear()
        self.events.clear()
        with mock.patch.object(runner, "partition", side_effect=AssertionError("duplicate partition")):
            actual = runner.run(write=True, dynamic=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(runner._hot_prepare.calls, 1)
        self.assertEqual(runner._partition_runner.calls, 0)
        self.assertEqual(self.events, ["hot_prepare", "merge"])
        self.assertEqual([record[0] for record in base.writes], ["baseline", "c8"])
        for _, kv, rope in base.writes:
            self.assertIs(kv, base.current_kv)
            self.assertIs(rope, base.current_rope)
        self.assertEqual([record["kind"] for record in base.reads], ["c4", "c8"])
        self.assertTrue(all(record["statistics"] for record in base.reads))
        self.assertIs(base.reads[0]["indices"], runner._hot_prepare.outputs["old"])
        self.assertIs(base.reads[1]["indices"], runner._hot_prepare.outputs["recent_ring"])
        self.assertFalse(np.shares_memory(runner.cold_cache, runner.variant.cache))
        self.assertEqual(runner.metadata["separate_partition_calls_per_write_dynamic_run"], 0)

    def test_prepared_and_no_write_dynamic_preserve_original_scheduling(self):
        runner = self.prepare()
        self.events.clear()
        self.scales.base.writes.clear()
        runner.call()
        runner.run(write=False, dynamic=False)
        self.assertEqual(self.events, ["merge", "merge"])
        self.assertEqual(runner._hot_prepare.calls, 0)
        runner.run(write=False, dynamic=True)
        self.assertEqual(self.events[-2:], ["standalone_partition", "merge"])
        self.assertEqual(runner._partition_runner.calls, 1)
        self.assertEqual(runner._hot_prepare.calls, 0)
        self.assertEqual(self.scales.base.writes, [])

    def test_writer_byte_gate_poisons_and_restores_cache_and_shared_outputs(self):
        runner = self.prepare()
        saved = self.snapshots(runner)
        diagnostic = runner.writer_diagnostics()
        self.assertTrue(diagnostic["passed"])
        self.assertEqual(set(diagnostic["fields"]), {"cold", "hot"})
        self.assertTrue(all(item["mismatch_count"] == 0 for item in diagnostic["fields"].values()))
        self.assert_restored(runner, saved)
        runner._hot_prepare.skip_store = True
        broken = runner.writer_diagnostics()
        self.assertFalse(broken["passed"])
        self.assertTrue(broken["fields"]["cold"]["passed"])
        self.assertFalse(broken["fields"]["hot"]["passed"])
        self.assert_restored(runner, saved)

    def test_fused_partition_gate_detects_noop_despite_correct_cache_and_old_outputs(self):
        runner = self.prepare()
        # Leave correct outputs behind, exactly the state that masked a noop
        # before the fused producer received its own sentinel test.
        runner.partition()
        saved = self.snapshots(runner)
        standalone_calls = runner._partition_runner.calls
        passing = runner.partition_diagnostics()
        self.assertTrue(passing["passed"])
        self.assertEqual(passing["producer"], "fused_hot_store_partition_after_output_poison")
        self.assertEqual(runner._partition_runner.calls, standalone_calls)
        self.assert_restored(runner, saved)
        runner._hot_prepare.skip_partition = True
        self.assertTrue(runner.writer_diagnostics()["passed"])
        failing = runner.partition_diagnostics()
        self.assertFalse(failing["passed"])
        self.assertFalse(failing["fields"]["old"]["passed"])
        self.assertFalse(failing["fields"]["recent_present"]["passed"])
        self.assertTrue(failing["invalid_rows"])
        self.assertEqual(runner._partition_runner.calls, standalone_calls)
        self.assert_restored(runner, saved)

    def test_zero_window_bypasses_new_imports_and_setup(self):
        with mock.patch.dict(sys.modules, {"kv_native_hot_prepare": None, "kv_native_partition": None}), \
                mock.patch("kv_native_twopass.fixed_ring_plan", side_effect=AssertionError("W0 ring")):
            runner = self.backend.prepare("baseline", 0, writer="hot_store_partition",
                                          partition="triton_scan", merge="triton")
            runner.call()
            runner.run(write=True, dynamic=True)
        self.assertFalse(self.partitions)
        self.assertFalse(self.preparations)
        self.assertFalse(hasattr(runner, "hot_cache"))
        self.assertEqual(self.scales.base.native_calls, [("baseline", False), ("baseline", True)])

    def test_invalid_new_writer_combinations_fail_before_device_setup(self):
        for override in (dict(mode="split_c4"), dict(partition="argsort_int"),
                         dict(partition="triton_scan_packed"), dict(reader="hot_merge"),
                         dict(merge="torch"), dict(merge="triton_direct512")):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.prepare(**override)
        self.assertFalse(self.partitions)
        self.assertFalse(self.preparations)
        self.assertEqual(self.scales.base.reads, [])

    def test_component_hot_prepare_returns_raw_tensor_without_cold_read_or_write(self):
        runner = self.prepare()
        stages = runner.component_calls()
        self.assertEqual(set(stages), {"cold_read", "hot_read", "merge", "cold_write", "hot_prepare"})
        base = self.scales.base
        base.reads.clear()
        base.writes.clear()
        cold = np.array(runner.cold_cache, copy=True)
        self.events.clear()
        output = stages["hot_prepare"]()
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(np.shares_memory(output, runner.hot_cache))
        self.assertEqual(base.reads, [])
        self.assertEqual([record[0] for record in base.writes], ["c8"])
        self.assertEqual(self.events, ["hot_prepare"])
        np.testing.assert_array_equal(runner.cold_cache, cold)


if __name__ == "__main__":
    unittest.main()
