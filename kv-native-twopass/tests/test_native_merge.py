"""Host/actual-body CPU tests; these are not Ascend compile or timing tests."""

import ast
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hotkv"))
import kv_native_merge as merge


class Metadata:
    def __init__(self, shape, dtype="torch.bfloat16", device="npu:0", contiguous=True):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.contiguous, self.requires_grad = contiguous, False

    def is_contiguous(self):
        return self.contiguous

    def item(self):
        raise AssertionError("No synchronization")

    def cpu(self):
        raise AssertionError("No host copy")

    def numpy(self):
        raise AssertionError("No host copy")


def state_metadata(shape=(3, 2, 527), dtype="torch.bfloat16"):
    return (Metadata(shape, dtype), Metadata(shape[:2] + (1,), "torch.float32"),
            Metadata(shape[:2] + (1,), "torch.float32"))


class TestHostContract(unittest.TestCase):
    def test_static_shapes_and_presence_validation(self):
        layout = merge.validate_states(state_metadata(), state_metadata(), Metadata((3, 2, 527)),
                                       left_present=Metadata((3,), "torch.bool"))
        self.assertEqual((layout.tokens, layout.heads, layout.features), (3, 2, 527))

    def test_native_fp32_outputs_are_accepted_without_dtype_conversion(self):
        left, right = state_metadata(dtype="torch.float32"), state_metadata(dtype="torch.float32")
        out = Metadata((3, 2, 527), "torch.float32")
        merge.validate_states(left, right, out)
        with self.assertRaisesRegex(ValueError, "dtype"):
            merge.validate_states(left, right, Metadata((3, 2, 527)))
        with self.assertRaisesRegex(ValueError, "dtype"):
            merge.validate_states(left, state_metadata(), out)
        with self.assertRaisesRegex(ValueError, "dtype"):
            merge.validate_states(state_metadata(dtype="torch.float16"), state_metadata(dtype="torch.float16"))

    def test_bad_state_and_static_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError, "maximum"):
            merge.validate_states(state_metadata()[:2], state_metadata())
        for location, attribute, value, message in (
            (0, "device", "cpu", "NPU"),
            (0, "shape", (3, 2), "shape"),
            (0, "shape", (0, 2, 512), "positive"),
            (0, "dtype", "torch.float32", "dtype"),
            (1, "dtype", "torch.bfloat16", "dtype"),
            (1, "shape", (3, 2, 8), "shape"),
            (2, "device", "npu:1", "same NPU"),
            (2, "contiguous", False, "contiguous"),
            (0, "requires_grad", True, "inference"),
        ):
            with self.subTest(attribute=attribute, value=value):
                left = state_metadata()
                setattr(left[location], attribute, value)
                with self.assertRaisesRegex(ValueError, message):
                    merge.validate_states(left, state_metadata())
        for presence in (Metadata((3, 1), "torch.bool"), Metadata((3,), "torch.int32"),
                         Metadata((3,), "torch.bool", contiguous=False)):
            with self.assertRaises(ValueError):
                merge.validate_states(state_metadata(), state_metadata(), right_present=presence)
        with self.assertRaises(ValueError):
            merge.validate_states(state_metadata(), state_metadata(), Metadata((3, 2, 512)))

    def test_preallocated_call_is_exactly_one_launch_without_allocation(self):
        class Kernel:
            def __init__(self):
                self.calls = []

            def __getitem__(self, grid):
                def call(*args, **kwargs):
                    self.calls.append((grid, args, kwargs))
                return call

        kernel, torch = Kernel(), mock.Mock()
        left, right, out = state_metadata(), state_metadata(), Metadata((3, 2, 527))
        presence = Metadata((3,), "torch.bool")
        with mock.patch.dict(sys.modules, {"torch": torch}), mock.patch.object(merge, "_load_kernel", return_value=kernel):
            for _ in range(3):
                self.assertIs(merge.merge_states(left, right, out, left_present=presence), out)
            torch.empty_like.assert_not_called()
            self.assertIs(merge.merge_states(left, right), torch.empty_like.return_value)
            torch.empty_like.assert_called_once_with(left[0])
        self.assertEqual(len(kernel.calls), 4)
        for grid, args, kwargs in kernel.calls[:3]:
            self.assertEqual(grid, (3, 6))
            self.assertEqual(args[:6], left + right)
            self.assertIs(args[6], presence)
            self.assertIs(args[7], right[1])
            self.assertIs(args[-1], out)
            self.assertEqual(kwargs, dict(H=2, D=527, BLOCK=256,
                                         HAS_LEFT_PRESENT=True, HAS_RIGHT_PRESENT=False, DIRECT=False))

    def test_variants_are_explicit_and_only_change_requested_launch_options(self):
        launch = mock.Mock()
        kernel = mock.MagicMock()
        kernel.__getitem__.return_value = launch
        left, right, out = state_metadata(), state_metadata(), Metadata((3, 2, 527))
        with mock.patch.object(merge, "_load_kernel", return_value=kernel):
            for variant, direct in (("log512", False), ("direct512", True)):
                merge.merge_states(left, right, out, variant=variant)
                kernel.__getitem__.assert_called_with((2, 6))
                self.assertEqual(launch.call_args.kwargs["BLOCK"], 512)
                self.assertEqual(launch.call_args.kwargs["DIRECT"], direct)
        with self.assertRaisesRegex(ValueError, "variant"):
            merge.merge_states(left, right, out, variant="unknown")

    def test_import_has_no_accelerator_dependencies(self):
        code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'triton'):
            raise AssertionError('eager accelerator import: '+fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
import kv_native_merge
assert kv_native_merge._KERNEL is None
"""
        subprocess.run([sys.executable, "-c", code, str(ROOT / "hotkv")], check=True,
                       capture_output=True, text=True)


def bf16(value):
    value = np.asarray(value, np.float32)
    bits = value.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


class Tensor(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def to(self, dtype):
        return Tensor(bf16(self) if dtype == "bf16" else np.asarray(self).astype(dtype))


class Pointer:
    def __init__(self, values, offset=0, store_dtype=None):
        self.values, self.offset = values.reshape(-1), np.asarray(offset)
        self.store_dtype = store_dtype

    def __add__(self, other):
        return Pointer(self.values, self.offset + other, self.store_dtype)


class TL:
    constexpr, float32, int64, bfloat16 = object, np.float32, np.int64, "bf16"

    def __init__(self):
        self.pid = (0, 0)
        self.stores = 0
        self.log_calls = 0

    def program_id(self, axis):
        return Tensor(np.int32(self.pid[axis]))

    @staticmethod
    def arange(start, end):
        return Tensor(np.arange(start, end, dtype=np.int32))

    @staticmethod
    def load(pointer, mask=True, other=0):
        indices = np.asarray(pointer.offset)
        active = np.broadcast_to(mask, indices.shape)
        if np.any(indices[active] < 0) or np.any(indices[active] >= pointer.values.size):
            raise AssertionError("Unmasked out-of-range read")
        result = np.full(indices.shape, other, dtype=pointer.values.dtype)
        result[active] = pointer.values[indices[active]]
        return Tensor(result)

    def store(self, pointer, value, mask=True):
        indices = np.asarray(pointer.offset)
        active = np.broadcast_to(mask, indices.shape)
        if np.any(indices[active] < 0) or np.any(indices[active] >= pointer.values.size):
            raise AssertionError("Unmasked out-of-range write")
        # Model tl.store's implicit conversion to the destination pointer type.
        value = bf16(value) if pointer.store_dtype == "bf16" else np.asarray(value, np.float32)
        pointer.values[indices[active]] = np.broadcast_to(value, indices.shape)[active]
        self.stores += 1

    @staticmethod
    def where(mask, yes, no):
        result = np.where(mask, yes, no)
        # This kernel has FP32 arithmetic only. NumPy otherwise promotes
        # 0-D FP32 tensors plus Python float literals to FP64, hiding genuine
        # FP32 subtraction/denominator overflow in the extreme-stat tests.
        if np.issubdtype(result.dtype, np.floating):
            result = result.astype(np.float32)
        return Tensor(result)

    @staticmethod
    def maximum(left, right):
        return Tensor(np.maximum(left, right))

    @staticmethod
    def exp(value):
        return Tensor(np.exp(value))

    def log(self, value):
        self.log_calls += 1
        return Tensor(np.log(value))


def kernel_body():
    tree = ast.parse(Path(merge.__file__).read_text())
    loader = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "_load_kernel")
    definition = next(x for x in loader.body if isinstance(x, ast.FunctionDef))
    definition.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    language = TL()
    scope = {"tl": language}
    exec(compile(module, merge.__file__, "exec"), scope)
    return language, scope["_merge_kernel"]


def make_states(shape=(3, 2, 527), dtype="bf16"):
    rng = np.random.default_rng(28)
    result = []
    for _ in range(2):
        values = rng.normal(size=shape)
        values = bf16(values) if dtype == "bf16" else values.astype(np.float32)
        result.append([values,
                       rng.normal(0, 8, size=shape[:2] + (1,)).astype(np.float32),
                       rng.uniform(1, 2048, size=shape[:2] + (1,)).astype(np.float32)])
    return result


class TestActualKernelBody(unittest.TestCase):
    variant = "log256"

    def execute(self, left, right, left_present=None, right_present=None, output_dtype="bf16",
                variant=None, return_trace=False):
        variant = self.variant if variant is None else variant
        block, direct = merge.MERGE_VARIANTS[variant]
        tl, kernel = kernel_body()
        output = np.full(left[0].shape, np.nan, np.float32)
        t, h, d = output.shape
        lp = left[1] if left_present is None else left_present
        rp = right[1] if right_present is None else right_present
        pointers = [Pointer(x) for x in (*left, *right, lp, rp)]
        pointers.append(Pointer(output, store_dtype=output_dtype))
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            for row in range(t * h):
                for tile in range((d + block - 1) // block):
                    tl.pid = tile, row
                    kernel(*pointers, H=h, D=d, BLOCK=block, HAS_LEFT_PRESENT=left_present is not None,
                           HAS_RIGHT_PRESENT=right_present is not None, DIRECT=direct)
        self.assertEqual(tl.stores, t * h * ((d + block - 1) // block))
        return (output, tl) if return_trace else output

    def assert_reference(self, output, left, right):
        # Independent FP64 sum*exp(max-shift) reference, rather than reproducing
        # the kernel's log-weight construction. Inputs are already BF16 values.
        lm, ls, rm, rs = [x.astype(np.float64) for x in (left[1], left[2], right[1], right[2])]
        common = np.maximum(lm, rm)
        lw, rw = ls * np.exp(lm-common), rs * np.exp(rm-common)
        reference = (lw / (lw+rw)) * left[0] + (rw / (lw+rw)) * right[0]
        np.testing.assert_allclose(output, reference, rtol=.004, atol=.004)
        self.assertLess(np.linalg.norm(output-reference) / np.linalg.norm(reference), .004)

    def test_finite_states_and_partial_feature_tile(self):
        left, right = make_states()
        self.assert_reference(self.execute(left, right), left, right)

    def test_fp32_destination_does_not_round_through_bf16(self):
        left, right = make_states(dtype="fp32")
        left[1].fill(0)
        right[1].fill(0)
        left[2].fill(1)
        right[2].fill(1)
        output = self.execute(left, right, output_dtype="fp32")
        reference = (left[0].astype(np.float64) + right[0].astype(np.float64)) / 2
        np.testing.assert_allclose(output, reference, rtol=2e-7, atol=2e-7)
        self.assertTrue(np.any(output != bf16(output)))

    def test_extreme_maxima_and_denominators(self):
        left, right = make_states()
        left[1].reshape(-1)[:] = [-10000, 10000, 5e37, 5e37, -120, 0]
        right[1].reshape(-1)[:] = [10000, -10000, 5e37, 5e37, 0, 0]
        left[2].reshape(-1)[:] = [1, 2048, 3e38, 1, 1e30, 3e38]
        right[2].reshape(-1)[:] = [2048, 1, 3e38, 2048, 1e-30, 1e38]
        self.assert_reference(self.execute(left, right), left, right)

    def test_one_empty_ignores_nan_output_and_preserves_other_side_exactly(self):
        for empty_side in (0, 1):
            with self.subTest(empty_side=empty_side):
                states = make_states()
                states[empty_side][0].fill(np.nan)
                states[empty_side][1].fill(-np.inf)
                states[empty_side][2].fill(0)
                np.testing.assert_array_equal(self.execute(*states), states[1-empty_side][0])

    def test_both_empty_is_exact_zero(self):
        states = make_states()
        for state in states:
            state[0].fill(np.nan)
            state[1].fill(-np.inf)
            state[2].fill(0)
        self.assertTrue(np.array_equal(self.execute(*states), np.zeros(states[0][0].shape)))

    def test_invalid_nonempty_statistics_propagate_nan_even_at_zero_weight(self):
        for side in (0, 1):
            for field, value in ((1, np.nan), (1, np.inf), (1, -np.inf),
                                 (2, np.nan), (2, np.inf), (2, -np.inf), (2, -1), (2, 0)):
                with self.subTest(side=side, field=field, value=value):
                    states = make_states()
                    states[side][1].fill(-10000)
                    states[1-side][1].fill(10000)
                    states[side][field].fill(value)
                    self.assertTrue(np.isnan(self.execute(*states)).all())

    def test_nonempty_nan_output_is_not_hidden(self):
        left, right = make_states()
        left[0].fill(np.nan)
        left[1].fill(-10000)
        right[1].fill(10000)
        self.assertTrue(np.isnan(self.execute(left, right)).all())

    def test_presence_masks_canonicalize_native_dummy_rows_in_same_kernel(self):
        left, right = make_states()
        left_present = np.array([False, True, False])
        right_present = np.array([True, False, False])
        for present, state in ((left_present, left), (right_present, right)):
            for tensor in state:
                tensor[~present] = np.nan
        output = self.execute(left, right, left_present, right_present)
        np.testing.assert_array_equal(output[0], right[0][0])
        np.testing.assert_array_equal(output[1], left[0][1])
        self.assertTrue(np.array_equal(output[2], np.zeros_like(output[2])))

    def test_present_true_rejects_native_empty_sentinel(self):
        for side in (0, 1):
            with self.subTest(side=side):
                states = make_states()
                states[side][1].fill(-np.inf)
                states[side][2].fill(0)
                present = np.ones((3,), dtype=np.bool_)
                masks = {"left_present" if side == 0 else "right_present": present}
                self.assertTrue(np.isnan(self.execute(*states, **masks)).all())
                # Without explicit presence, the documented sentinel denotes
                # an empty segment and the other output survives unchanged.
                np.testing.assert_array_equal(self.execute(*states), states[1-side][0])


class TestLog512KernelBody(TestActualKernelBody):
    # The complete existing empty/malformed/FP32/extreme-stat contract must
    # still hold when two 256-feature programs become one 512-feature program.
    variant = "log512"

    def test_schedule_change_is_bitwise_equivalent(self):
        for dtype in ("bf16", "fp32"):
            left, right = make_states(dtype=dtype)
            a = self.execute(left, right, output_dtype=dtype)
            b = self.execute(left, right, output_dtype=dtype, variant="log256")
            np.testing.assert_array_equal(a, b)


class TestDirect512KernelBody(TestActualKernelBody):
    variant = "direct512"

    def test_normal_maxsum_domain_skips_logarithms(self):
        left, right = make_states()
        output, trace = self.execute(left, right, return_trace=True)
        self.assertEqual(trace.log_calls, 0)
        self.assert_reference(output, left, right)

    def test_fast_domain_boundary_and_subtraction_overflow_are_safe(self):
        left, right = make_states()
        left[1].reshape(-1)[:] = [0, 0, 3e38, -3e38, 3e38, -3e38]
        right[1].reshape(-1)[:] = [0, 0, -3e38, 3e38, 3e38, -3e38]
        left[2].reshape(-1)[:] = [1, 65536, 1, 65536, 1, 65536]
        right[2].reshape(-1)[:] = [65536, 1, 65536, 1, 65536, 1]
        output, trace = self.execute(left, right, return_trace=True)
        self.assertEqual(trace.log_calls, 0)
        self.assert_reference(output, left, right)

    def test_finite_outside_fast_domain_uses_original_log_branch(self):
        for denominator in (np.nextafter(np.float32(1), np.float32(0)),
                            np.nextafter(np.float32(65536), np.float32(np.inf)),
                            np.float32(1e-30), np.float32(3e38)):
            with self.subTest(denominator=denominator):
                left, right = make_states()
                left[2].fill(denominator)
                output, trace = self.execute(left, right, return_trace=True)
                self.assertEqual(trace.log_calls, trace.stores * 2)
                original = self.execute(left, right, variant="log512")
                np.testing.assert_array_equal(output, original)

    def test_randomized_direct_and_log_variants_match_fp64_reference(self):
        rng = np.random.default_rng(76)
        for _ in range(20):
            left, right = make_states(shape=(2, 3, 31), dtype="fp32")
            for state in (left, right):
                state[0][:] = rng.normal(size=state[0].shape)
                state[1][:] = rng.uniform(-80, 80, size=state[1].shape)
                state[2][:] = rng.uniform(1, 65536, size=state[2].shape)
            output = self.execute(left, right, output_dtype="fp32")
            self.assert_reference(output, left, right)
            log_output = self.execute(left, right, output_dtype="fp32", variant="log512")
            np.testing.assert_allclose(output, log_output, rtol=3e-5, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
