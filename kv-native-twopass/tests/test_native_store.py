"""CPU byte-addressing tests; no claim of Ascend compilation/performance."""
import ast
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hotkv"))
import kv_native_store as store
from test_native_partition import TL as PartitionTL, Pointer
from test_native_twopass_backend import Tensor, NativeTwoPassBackend, FakeScales, case_data


class TL(PartitionTL):
    uint8 = np.uint8

    @staticmethod
    def full(shape, value, dtype):
        return Tensor(np.full(shape, value, dtype))

    @staticmethod
    def where(condition, left, right):
        return Tensor(np.where(condition, left, right))


def kernel_body(tl):
    tree = ast.parse(Path(store.__file__).read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == "_store_kernel")
    function.decorator_list = []
    namespace = {"tl": tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 store.__file__, "exec"), namespace)
    return namespace["_store_kernel"]


class StoreAddressTests(unittest.TestCase):
    def test_store_kernel_exact_layout_and_only_target_slots(self):
        rng = np.random.default_rng(315)
        for dim, rope in ((128, 0), (128, 64), (512, 64), (1024, 0)):
            for mixed in (False, True):
                with self.subTest(dim=dim, rope=rope, mixed=mixed):
                    count, capacity = 7, 31
                    widths = dim//2+2*rope+dim//32, dim+2*rope+dim//32
                    raw = lambda width: rng.integers(0, 256, (count, width), np.uint8)
                    q4, s4, q8, s8, r = raw(dim//2), raw(dim//32), raw(dim), raw(dim//32), raw(2*rope)
                    cs, hs = np.array([30, 1, 19, 0, 10, 11, 25]), np.array([3, 0, 30, 27, 25, 19, 13])
                    cold, hot = np.full((capacity, widths[0]), 165, np.uint8), np.full((capacity, widths[1]), 165, np.uint8)
                    ec, eh = cold.copy(), hot.copy()
                    ec[cs] = np.concatenate((q4, r, s4), axis=-1)
                    if mixed:
                        eh[hs] = np.concatenate((q8, r, s8), axis=-1)
                    arrays = (q4, s4, q8, s8, r, cs, hs, cold, hot)
                    pointers = [Pointer(value, str(i)) for i, value in enumerate(arrays)]
                    tl = TL()
                    fn = kernel_body(tl)
                    block = 1 << ((widths[1] if mixed else widths[0])-1).bit_length()
                    for row in range(count):
                        tl.row = row
                        fn(*pointers, dim, rope, mixed, *widths, block)
                    np.testing.assert_array_equal(cold, ec)
                    np.testing.assert_array_equal(hot, eh)

    def test_invalid_slot_maps_rejected(self):
        for bad in ([-1, 1], [0, 9], [1, 1], [[0, 1]], [0.0, 1.0]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                store.validate_slots(bad, 2, 9, "test")
        store.validate_slots(np.array([8, 0]), 2, 9, "test")


class WriterGateTests(unittest.TestCase):
    def setUp(self):
        # Extend only the NumPy test double with in-place byte operations.
        self.fill = mock.patch.object(Tensor, "fill_", lambda a, v: (a.fill(v), a)[1], create=True)
        self.copy = mock.patch.object(Tensor, "copy_", lambda a, v: (a.__setitem__(slice(None), v), a)[1], create=True)
        self.fill.start()
        self.copy.start()
        self.addCleanup(self.fill.stop)
        self.addCleanup(self.copy.stop)

    def runner(self):
        data = case_data()
        runner = NativeTwoPassBackend(data, scale_backend=FakeScales(data)).prepare("baseline", 2)
        runner.writer_kind = "fused_store"
        return runner

    def test_gate_accepts_identical_writer_and_restores_entire_caches(self):
        runner = self.runner()
        saved = runner.cold_cache.copy(), runner.hot_cache.copy()
        self.assertTrue(runner.writer_diagnostics()["passed"])
        np.testing.assert_array_equal(runner.cold_cache, saved[0])
        np.testing.assert_array_equal(runner.hot_cache, saved[1])

    def test_gate_rejects_noop_and_outside_slot_corruption(self):
        for corruption in (False, True):
            with self.subTest(corruption=corruption):
                runner = self.runner()
                saved = runner.cold_cache.copy(), runner.hot_cache.copy()
                native = runner._write
                def bad_write():
                    if corruption:
                        native()
                        runner.cold_cache.view(np.uint8).reshape(-1)[0] ^= np.uint8(1)
                runner._write = bad_write
                self.assertFalse(runner.writer_diagnostics()["passed"])
                np.testing.assert_array_equal(runner.cold_cache, saved[0])
                np.testing.assert_array_equal(runner.hot_cache, saved[1])

    def test_gate_restores_on_compile_or_runtime_exception(self):
        runner = self.runner()
        saved = runner.cold_cache.copy(), runner.hot_cache.copy()
        def failed():
            raise RuntimeError("compile failed")
        runner._write = failed
        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            runner.writer_diagnostics()
        np.testing.assert_array_equal(runner.cold_cache, saved[0])
        np.testing.assert_array_equal(runner.hot_cache, saved[1])


if __name__ == "__main__":
    unittest.main()
