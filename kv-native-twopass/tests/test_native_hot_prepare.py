"""Exact CPU byte/index and native-call checks; no A5 performance claim."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hotkv"))
import kv_native_hot_prepare as prepare
from test_native_partition import TL as ScanTL, Tensor as Array, Pointer, allocate_outputs, oracle
from test_native_store_contract import Tensor, Kernel, fixture as store_fixture


ORDER = ("old", "recent_global", "recent_ring", "old_present", "recent_present", "invalid_rows")


class TL(ScanTL):
    uint8 = np.uint8

    @staticmethod
    def full(shape, value, dtype):
        return Array(np.full(shape, value, dtype))

    @staticmethod
    def where(condition, left, right):
        return Array(np.where(condition, left, right))


def kernel_body(tl):
    tree = ast.parse(Path(prepare.__file__).read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == "_hot_prepare_kernel")
    function.decorator_list = []
    namespace = {"tl": tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 prepare.__file__, "exec"), namespace)
    return namespace["_hot_prepare_kernel"]


class ExactSourceTests(unittest.TestCase):
    def execute(self, indices, positions, slots, rope, *, outputs=None, cache=None, seed=87,
                window=8, hot_width=8, ring_capacity=16):
        rows, _, sparse = indices.shape
        rng, width, physical_capacity = np.random.default_rng(seed), 528+2*rope, 48
        raw = lambda size: rng.integers(0, 256, (rows, size), np.uint8)
        q8, s8, rbytes = raw(512), raw(16), raw(2*rope)
        outputs = allocate_outputs(rows, sparse, hot_width, True) if outputs is None else outputs
        cache = np.full((physical_capacity, width), 165, np.uint8) if cache is None else cache
        expected_cache = cache.copy()
        for row, slot in enumerate(slots):
            if 0 <= slot < physical_capacity:
                expected_cache[slot] = np.concatenate((q8[row], rbytes[row], s8[row]))
        arrays = [indices, positions, *(outputs[name] for name in ORDER), q8, s8, rbytes, slots, cache]
        names = ["indices", "positions", *ORDER, "q8", "s8", "rope", "slots", "cache"]
        pointers = [Pointer(value, name) for value, name in zip(arrays, names)]
        language = TL()
        fn = kernel_body(language)
        constants = dict(S=sparse, HW=hot_width, WINDOW=window, CAP=ring_capacity,
                         BLOCK=1 << (sparse-1).bit_length(), R=rope, HOT_WIDTH=width,
                         HOT_CAPACITY=physical_capacity, STORE_BLOCK=1 << (width-1).bit_length())
        for row in range(rows):
            language.row = row
            fn(*pointers, **constants)
        expected = oracle(indices, positions, window, ring_capacity, hot_width, True)
        expected["invalid_rows"] |= (slots < 0) | (slots >= physical_capacity)
        for name in ORDER:
            np.testing.assert_array_equal(outputs[name], expected[name], err_msg=name)
            np.testing.assert_array_equal(language.writes[name], np.ones(outputs[name].size, np.int32), err_msg=name)
        np.testing.assert_array_equal(cache, expected_cache)
        expected_writes = np.zeros_like(cache, np.int32)
        expected_writes[slots[(slots >= 0) & (slots < physical_capacity)]] = 1
        np.testing.assert_array_equal(language.writes["cache"], expected_writes.reshape(-1))
        self.assertEqual(language.scans, 2*rows)
        return outputs, cache

    def test_q4_unsorted_empty_and_ring_wrap_byte_exact(self):
        indices = np.array([[31, 1, 35, 30, 32, 2, 33, 34, 0, 29, 28, 27, 26, -1, -1, -1, -1],
                            [4, 0, -1, 3, 1, 2] + [-1]*11,
                            [0, 2, 1, 3, 4, 5] + [-1]*11,
                            [-1]*17], np.int32)[:, None, :]
        positions, slots = np.array([35, 4, 100, 0], np.int64), np.array([47, 0, 16, 31], np.int32)
        for rope in (0, 64):
            with self.subTest(rope=rope):
                outputs, _ = self.execute(indices, positions, slots, rope)
                self.assertFalse(outputs["invalid_rows"].any())

    def test_reused_outputs_cache_and_changed_current_bytes(self):
        slots, positions = np.array([47, 0, 16, 31], np.int64), np.full(4, 35, np.int64)
        outputs, cache = None, None
        for seed, entries in enumerate(([35, 30, 29, 34, 28, 31, 33, 32], [0, 2, 1, 7], [])):
            indices = np.full((4, 1, 17), -1, np.int32)
            indices[:, 0, :len(entries)] = entries
            outputs, cache = self.execute(indices, positions, slots, 64, outputs=outputs, cache=cache, seed=seed)

    def test_invalid_indices_overflow_and_slots_are_bounded_and_flagged(self):
        indices = np.array([[31, 32, 33, 34, 35, 30, 29, 28] + [-1]*9,
                            [2, -2, 99] + [-1]*14,
                            [-1]*17,
                            [1, 2] + [-1]*15], np.int32)[:, None, :]
        outputs, _ = self.execute(indices, np.array([35, 35, 35, -1], np.int64),
                                  np.array([47, -1, 48, 31], np.int64), 64, hot_width=4)
        self.assertTrue(outputs["invalid_rows"].all())


def fixture(rope=64):
    data = store_fixture(rope=rope)
    data.base.torch.bool = "bool"
    tensor = lambda shape, dtype: Tensor(shape, dtype, events=data.events)
    layout = SimpleNamespace(rows=3, sparse_capacity=17, hot_width=8, window=8, ring_capacity=16,
                             remap_ring=True, block=32, variant="two_scan")
    outputs = {name: tensor((3, 1, 17 if name == "old" else 8), "int32") if i < 3
               else tensor((3,), "bool") for i, name in enumerate(ORDER)}
    inputs = (tensor((3, 1, 17), "int32"), tensor((3,), "int64"), *(outputs[name] for name in ORDER))
    data.partition = SimpleNamespace(layout=layout, outputs=outputs, _inputs=inputs,
                                    metadata={"variant": "two_scan"},
                                    call=mock.Mock(side_effect=AssertionError("No standalone partition call")))
    return data


def prepared(data):
    kernel = Kernel()
    with mock.patch.object(prepare, "_load_kernel", return_value=kernel):
        runner = prepare.PreparedHotStorePartition(data.base, data.hot, data.slots, data.partition)
    return runner, kernel


class NativeCallContractTests(unittest.TestCase):
    def test_quantization_each_call_one_launch_and_shared_partition_buffers(self):
        for rope in (0, 64):
            data = fixture(rope)
            runner, kernel = prepared(data)
            data.base.torch_npu.npu_dynamic_block_quant.assert_not_called()
            self.assertIn("cpu", data.events)
            data.events.clear()
            for _ in range(2):
                self.assertIs(runner.call(), data.partition.outputs)
            data.base.torch_npu.npu_dynamic_block_quant.assert_called_with(
                runner.kv, dst_type="fp8_e4m3fn", row_block_size=1, col_block_size=128)
            self.assertEqual(data.base.torch_npu.npu_dynamic_block_quant.call_count, 2)
            data.base.torch_npu.npu_dynamic_mx_quant.assert_not_called()
            data.partition.call.assert_not_called()
            data.base.quantize.assert_not_called()
            data.base.torch.cat.assert_not_called()
            self.assertFalse({"detach", "cpu", "numpy"} & set(data.events))
            self.assertEqual(len(kernel.calls), 2)
            for grid, args, constants in kernel.calls:
                self.assertEqual(grid, (3,))
                self.assertTrue(all(a is b for a, b in zip(args[:8], data.partition._inputs)))
                for pointer in (args[8], args[9], args[10], args[12]):
                    self.assertEqual(pointer.dtype, "uint8")
                self.assertIs(args[8].storage, data.q8.storage)
                self.assertIs(args[9].storage, data.s8.storage)
                self.assertIs(args[10].storage, data.base.current_rope.storage)
                self.assertIs(args[11], data.slots)
                self.assertIs(args[12].storage, data.hot.storage)
                self.assertEqual(constants["R"], rope)
                self.assertEqual(constants["HOT_WIDTH"], 528+2*rope)

    def test_reject_unsupported_partition_bad_counts_formats_slots_and_device(self):
        for case in ("packed", "global", "rows", "dim", "rope", "kv_dtype", "cache_width",
                     "cache_dtype", "slots_duplicate", "slots_oob", "slots_dtype", "cache_device",
                     "partition_device", "partition_dtype", "stride", "buffer_identity"):
            with self.subTest(case=case):
                data = fixture()
                if case == "packed": data.partition.layout.variant = "packed_one_scan"
                elif case == "global": data.partition.layout.remap_ring = False
                elif case == "rows": data.partition.layout.rows = 4
                elif case == "dim": data.base.dim = 256
                elif case == "rope": data.base.rope_dim = 32
                elif case == "kv_dtype": data.base.current_kv.dtype = "fp32"
                elif case == "cache_width": data.hot.shape = (1, 16, 1, 655)
                elif case == "cache_dtype": data.hot.dtype = "bf16"
                elif case == "slots_duplicate": data.slots.values = np.array([0, 0, 1])
                elif case == "slots_oob": data.slots.values = np.array([0, 1, 16])
                elif case == "slots_dtype": data.slots.dtype = "fp32"
                elif case == "cache_device": data.hot.device = "npu:1"
                elif case == "partition_device": data.partition._inputs[0].device = "npu:1"
                elif case == "partition_dtype": data.partition._inputs[0].dtype = "int64"
                elif case == "stride": data.base.current_kv.contiguous = False
                else: data.partition.outputs["old"] = object()
                with self.assertRaises(ValueError):
                    prepared(data)

    def test_native_quantizer_validation_errors_do_not_launch_kernel(self):
        for case in ("error", "q_dtype", "s_dtype", "q_size", "s_size", "q_stride", "s_device"):
            with self.subTest(case=case):
                data = fixture()
                runner, kernel = prepared(data)
                if case == "error": data.base.torch_npu.npu_dynamic_block_quant.side_effect = RuntimeError("native failure")
                elif case == "q_dtype": data.q8.dtype = "int8"
                elif case == "s_dtype": data.s8.dtype = "uint8"
                elif case == "q_size": data.q8.shape = (2, 1, 512)
                elif case == "s_size": data.s8.shape = (3, 1, 3)
                elif case == "q_stride": data.q8.contiguous = False
                else: data.s8.device = "npu:1"
                with self.assertRaises(RuntimeError):
                    runner.call()
                self.assertEqual(kernel.calls, [])
                data.partition.call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
