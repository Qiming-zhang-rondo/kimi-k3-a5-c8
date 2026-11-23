"""CPU source-body and host-contract checks, not Ascend compiler/timing proof."""

import ast
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hotkv"))
import kv_native_partition as partition


class Metadata:
    def __init__(self, shape, dtype="torch.int32", device="npu:0", contiguous=True):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous

    def item(self):
        raise AssertionError("No device synchronization")

    def cpu(self):
        raise AssertionError("No device copies")


def metadata_inputs():
    return Metadata((6, 1, 2048)), Metadata((6,), "torch.int64")


class TestHostContract(unittest.TestCase):
    def test_default_bounded_capacity_and_full_capacity(self):
        inputs = metadata_inputs()
        bounded = partition.validate_inputs(*inputs, window=128, ring_capacity=160)
        self.assertEqual((bounded.rows, bounded.hot_width, bounded.block), (6, 128, 2048))
        full = partition.validate_inputs(*inputs, window=128, ring_capacity=160, hot_width=2048)
        self.assertEqual(full.hot_width, 2048)
        short = partition.validate_inputs(Metadata((2, 1, 17)), Metadata((2,)),
                                           window=32, ring_capacity=32)
        self.assertEqual((short.hot_width, short.block), (17, 32))

    def test_bad_metadata_rejected_without_reading_values(self):
        for index, attribute, value in ((0, "shape", (6, 2, 2048)), (0, "dtype", "torch.int64"),
                                         (0, "device", "cpu"), (1, "shape", (6, 1)),
                                         (1, "dtype", "torch.float32"), (1, "device", "npu:1"),
                                         (1, "contiguous", False)):
            with self.subTest(attribute=attribute, value=value):
                inputs = metadata_inputs()
                setattr(inputs[index], attribute, value)
                with self.assertRaises(ValueError):
                    partition.validate_inputs(*inputs, window=128, ring_capacity=160)
        for overrides in (dict(window=0), dict(window=True), dict(ring_capacity=127),
                          dict(hot_width=127), dict(hot_width=2049), dict(remap_ring="true")):
            kwargs = dict(window=128, ring_capacity=160)
            kwargs.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                partition.validate_inputs(*metadata_inputs(), **kwargs)

    def test_prepared_reuse_is_one_launch_without_timed_allocations(self):
        class Kernel:
            def __init__(self):
                self.calls = []

            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    self.calls.append((grid, args, kwargs))
                return launch

        for remap in (False, True):
            with self.subTest(remap=remap):
                kernel, torch = Kernel(), mock.Mock()
                torch.empty.side_effect = lambda *args, **kwargs: object()
                torch.empty_like.side_effect = lambda *args, **kwargs: object()
                inputs = metadata_inputs()
                with mock.patch.dict(sys.modules, {"torch": torch}), mock.patch.object(partition, "_load_kernel", return_value=kernel):
                    prepared = partition.PreparedPartition(*inputs, window=128, ring_capacity=160,
                                                            remap_ring=remap)
                    torch.empty.reset_mock()
                    torch.empty_like.reset_mock()
                    for _ in range(3):
                        self.assertIs(prepared.call(), prepared.outputs)
                    torch.empty.assert_not_called()
                    torch.empty_like.assert_not_called()
                self.assertEqual(len(kernel.calls), 3)
                for grid, args, constants in kernel.calls:
                    self.assertEqual(grid, (6,))
                    self.assertEqual(args[:2], inputs)
                    self.assertEqual(constants, dict(S=2048, HW=128, WINDOW=128, CAP=160,
                                                     REMAP_RING=remap, BLOCK=2048))
                self.assertEqual(prepared.metadata["kernel_launches"], 1)
                self.assertEqual(prepared.metadata["attention_launches"], 0)
                if not remap:
                    self.assertIs(prepared.outputs["recent_ring"], prepared.outputs["recent_global"])

    def test_lazy_import_has_no_torch_or_triton_dependency(self):
        code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'triton'):
            raise AssertionError('eager import: '+fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
import kv_native_partition
assert kv_native_partition._KERNEL is None
"""
        subprocess.run([sys.executable, "-c", code, str(ROOT / "hotkv")], check=True,
                       capture_output=True, text=True)


class Tensor(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def to(self, dtype):
        return Tensor(np.asarray(self).astype(dtype))


class Pointer:
    def __init__(self, array, name, offset=0):
        self.array, self.name, self.offset = array.reshape(-1), name, np.asarray(offset)

    def __add__(self, offset):
        return Pointer(self.array, self.name, self.offset + offset)


class TL:
    constexpr, int32, int64 = object, np.int32, np.int64

    def __init__(self):
        self.row = 0
        self.writes = {}
        self.scans = 0

    def program_id(self, axis):
        assert axis == 0
        return Tensor(np.int32(self.row))

    @staticmethod
    def arange(start, end):
        return Tensor(np.arange(start, end, dtype=np.int32))

    @staticmethod
    def load(pointer, mask=True, other=0):
        offsets = np.asarray(pointer.offset)
        active = np.broadcast_to(mask, offsets.shape)
        assert np.all(offsets[active] >= 0) and np.all(offsets[active] < pointer.array.size)
        result = np.full(offsets.shape, other, pointer.array.dtype)
        result[active] = pointer.array[offsets[active]]
        return Tensor(result)

    def store(self, pointer, values, mask=True):
        offsets = np.asarray(pointer.offset)
        active = np.broadcast_to(mask, offsets.shape)
        assert np.all(offsets[active] >= 0) and np.all(offsets[active] < pointer.array.size)
        indices = offsets[active]
        # Track every physical output slot, including repeated scatter offsets.
        # A correct partition writes each slot exactly once, independent of
        # instruction scheduling; this catches fill/scatter write races.
        counts = self.writes.setdefault(pointer.name, np.zeros(pointer.array.size, np.int32))
        np.add.at(counts, indices, 1)
        pointer.array[indices] = np.broadcast_to(values, offsets.shape)[active]

    def cumsum(self, value, axis):
        self.scans += 1
        return Tensor(np.cumsum(value, axis=axis, dtype=np.int32))

    @staticmethod
    def sum(value, axis):
        return Tensor(np.sum(value, axis=axis, dtype=np.int32))

    @staticmethod
    def maximum(left, right):
        return Tensor(np.maximum(left, right))


def kernel_body():
    tree = ast.parse(Path(partition.__file__).read_text())
    loader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_load_kernel")
    definition = next(node for node in loader.body if isinstance(node, ast.FunctionDef))
    definition.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    language = TL()
    scope = {"tl": language}
    exec(compile(module, partition.__file__, "exec"), scope)
    return language, scope["_partition_kernel"]


def allocate_outputs(rows, width, hot_width, remap):
    recent = np.full((rows, 1, hot_width), 777, np.int32)
    return dict(old=np.full((rows, 1, width), 777, np.int32), recent_global=recent,
                recent_ring=np.full_like(recent, 777) if remap else recent,
                old_present=np.zeros(rows, np.bool_), recent_present=np.zeros(rows, np.bool_),
                invalid_rows=np.zeros(rows, np.bool_))


def oracle(indices, positions, window, capacity, hot_width, remap):
    outputs = allocate_outputs(indices.shape[0], indices.shape[-1], hot_width, remap)
    for row, position in enumerate(positions):
        entries = [int(key) for key in indices[row, 0]]
        visible = [key for key in entries if 0 <= key <= position]
        old = [key for key in visible if key <= int(position) - window]
        recent = [key for key in visible if key > int(position) - window]
        outputs["invalid_rows"][row] = (position < 0 or len(recent) > hot_width
                                       or any(key < -1 or key > position for key in entries if key != -1))
        outputs["old_present"][row], outputs["recent_present"][row] = bool(old), bool(recent)
        for name, values, width in (("old", old, indices.shape[-1]), ("recent_global", recent, hot_width)):
            values = values[:width] if values else [0]
            outputs[name][row, 0] = values + [-1] * (width-len(values))
        if remap:
            values = [key % capacity for key in recent[:hot_width]] if recent else [0]
            outputs["recent_ring"][row, 0] = values + [-1] * (hot_width-len(values))
    return outputs


class TestActualKernelBody(unittest.TestCase):
    def execute(self, indices, positions, window=8, capacity=16, hot_width=None, remap=True, outputs=None):
        rows, _, width = indices.shape
        hot_width = min(window, width) if hot_width is None else hot_width
        outputs = allocate_outputs(rows, width, hot_width, remap) if outputs is None else outputs
        tl, kernel = kernel_body()
        order = ("old", "recent_global", "recent_ring", "old_present", "recent_present", "invalid_rows")
        pointers = [Pointer(indices, "indices"), Pointer(positions, "positions")]
        pointers += [Pointer(outputs[name], name) for name in order]
        for row in range(rows):
            tl.row = row
            kernel(*pointers, S=width, HW=hot_width, WINDOW=window, CAP=capacity,
                   REMAP_RING=remap, BLOCK=1 << (width-1).bit_length())
        self.assertEqual(tl.scans, rows * 2)
        for name in order:
            if name != "recent_ring" or remap:
                np.testing.assert_array_equal(tl.writes[name], np.ones(outputs[name].size, np.int32), err_msg=name)
        expected = oracle(indices, positions, window, capacity, hot_width, remap)
        for name in order:
            np.testing.assert_array_equal(outputs[name], expected[name], err_msg=name)
        return outputs

    def test_stable_unsorted_heterogeneous_queries_and_ring_wrap(self):
        rows = [[31, 1, 35, 30, 32, 2, 33, 34, 0, 29, 28, 27, 26, -1, -1, -1, -1],
                [4, 0, -1, 3, 1, 2] + [-1] * 11,
                [0, 2, 1, 3, 4, 5] + [-1] * 11,
                [-1] * 17]
        indices = np.asarray(rows, np.int32)[:, None, :]
        positions = np.array([35, 4, 100, 0], np.int64)
        for remap in (False, True):
            for hot_width in (8, 17):
                with self.subTest(remap=remap, hot_width=hot_width):
                    result = self.execute(indices, positions, hot_width=hot_width, remap=remap)
                    self.assertFalse(result["invalid_rows"].any())

    def test_reused_buffers_overwritten_all_hot_then_all_cold_then_empty(self):
        indices = np.array([[[14, 9, 15, 8, 13, 10, 12, 11] + [-1] * 9]], np.int32)
        positions = np.array([15], np.int32)
        outputs = self.execute(indices, positions)
        identities = {name: id(value) for name, value in outputs.items()}
        positions[:] = 100
        self.execute(indices, positions, outputs=outputs)
        indices.fill(-1)
        self.execute(indices, positions, outputs=outputs)
        self.assertEqual(identities, {name: id(value) for name, value in outputs.items()})

    def test_large_non_power_of_two_selection_has_no_missing_keys(self):
        rng = np.random.default_rng(94)
        for width in (1, 2048, 2051):
            keys = rng.choice(8192, size=width, replace=False).astype(np.int32)
            indices = np.stack([keys, keys[::-1]])[:, None, :]
            result = self.execute(indices, np.array([8191, 8191], np.int64), window=128, capacity=160)
            self.assertFalse(result["invalid_rows"].any())

    def test_invalid_inputs_flagged_and_stores_still_bounded(self):
        # Repeated recent keys violate the caller uniqueness contract and can
        # overflow bounded recent capacity. We check capacity, not duplicates.
        indices = np.array([[[9] * 17], [[-2, 11, 8] + [-1] * 14], [[-1] * 17]], np.int32)
        positions = np.array([9, 9, -1], np.int64)
        result = self.execute(indices, positions, window=8, capacity=16)
        np.testing.assert_array_equal(result["invalid_rows"], [True, True, True])

    def test_int64_positions_do_not_wrap_to_int32(self):
        indices = np.array([[[0, 2147483647, -1]]], np.int32)
        positions = np.array([2147483648], np.int64)
        result = self.execute(indices, positions, window=8, capacity=16)
        self.assertFalse(result["invalid_rows"].any())


if __name__ == "__main__":
    unittest.main()
