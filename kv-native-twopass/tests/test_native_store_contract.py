"""CPU mocks for the native quantizer/one-store launch contract, not NPU proof."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hotkv"))
import kv_native_store as store


SIZES = {"bf16": 2, "fp32": 4, "uint8": 1, "int8": 1, "int32": 4,
         "int64": 8, "fp8_e4m3fn": 1, "fp4_e2m1fn_x2": 1}


class Tensor:
    """Metadata-only tensor; only physical slot maps allow host value access."""

    def __init__(self, shape, dtype, *, device="npu:0", contiguous=True, values=None,
                 events=None, storage=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.contiguous, self.values = contiguous, values
        self.events = [] if events is None else events
        self.storage = object() if storage is None else storage

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return SIZES[self.dtype]

    def is_contiguous(self):
        return self.contiguous

    def view(self, *args):
        if len(args) == 1 and isinstance(args[0], str):
            dtype = args[0]
            nbytes = self.numel() * self.element_size()
            if nbytes % SIZES[dtype]:
                raise RuntimeError("Invalid byte view")
            shape = (nbytes // SIZES[dtype],)
        else:
            dtype, shape = self.dtype, list(args)
            if shape.count(-1) > 1:
                raise RuntimeError("Invalid reshape")
            if -1 in shape:
                known = math.prod(x for x in shape if x != -1)
                if known == 0 or self.numel() % known:
                    raise RuntimeError("Invalid reshape size")
                shape[shape.index(-1)] = self.numel() // known
            if math.prod(shape) != self.numel():
                raise RuntimeError("Invalid reshape size")
        self.events.append("view")
        return Tensor(shape, dtype, device=self.device, contiguous=self.contiguous,
                      events=self.events, storage=self.storage)

    def detach(self):
        self.events.append("detach")
        return self

    def cpu(self):
        self.events.append("cpu")
        return self

    def numpy(self):
        self.events.append("numpy")
        if self.values is None:
            raise AssertionError("Only setup slot maps may be copied to host")
        return self.values


class Kernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return launch


def fixture(recipe="baseline", mixed=True, dim=512, rope=64):
    events = []
    tensor = lambda shape, dtype, **kwargs: Tensor(shape, dtype, events=events, **kwargs)
    torch = SimpleNamespace(bfloat16="bf16", float32="fp32", uint8="uint8", int8="int8",
                            int32="int32", int64="int64", float8_e4m3fn="fp8_e4m3fn")
    torch.cat = mock.Mock(side_effect=AssertionError("Packing with torch.cat is forbidden"))
    q4, s4 = tensor((3, 1, dim//2), "fp4_e2m1fn_x2"), tensor((3, 1, dim//32), "uint8")
    q8, s8 = tensor((3, 1, dim), "fp8_e4m3fn"), tensor((3, 1, dim//128), "fp32")
    npu = SimpleNamespace(npu_dynamic_mx_quant=mock.Mock(return_value=(q4, s4)),
                          npu_dynamic_block_quant=mock.Mock(return_value=(q8, s8)))
    base = SimpleNamespace(dim=dim, rope_dim=rope, torch=torch, torch_npu=npu,
                           current_kv=tensor((1, 3, 1, dim), "bf16"),
                           current_rope=tensor((1, 3, 1, rope), "bf16"),
                           current_slots=tensor((3,), "int64", values=np.array([6, 1, 3], np.int64)),
                           quantize=mock.Mock(side_effect=AssertionError("Do not invoke packing wrapper")))
    quant_args = {"dst_type": "fp4_e2m1fn_x2"}
    if recipe == "alg2_max7":
        quant_args.update(axis=-1, round_mode="rint", block_size=32, scale_alg=2, dst_type_max=7.0)
    variant = SimpleNamespace(_quantizer_arguments=quant_args,
                              _quantize=mock.Mock(side_effect=AssertionError("Do not invoke packing wrapper")))
    cold = tensor((8, 1, 1, dim//2 + 2*rope + dim//32), "int8")
    hot = tensor((1, 16, 1, dim + 2*rope + dim//32), "uint8") if mixed else None
    slots = tensor((3,), "int32", values=np.array([15, 0, 8], np.int32)) if mixed else None
    return SimpleNamespace(base=base, variant=variant, cold=cold, hot=hot, slots=slots,
                           q4=q4, s4=s4, q8=q8, s8=s8, events=events)


def prepare(data, kernel=None):
    kernel = Kernel() if kernel is None else kernel
    with mock.patch.object(store, "_load_kernel", return_value=kernel):
        writer = store.NativeQuantizedStore(data.base, data.variant, data.cold,
                                            hot_cache=data.hot, hot_slots=data.slots)
    return writer, kernel


class NativeQuantizerCallTests(unittest.TestCase):
    def test_exact_recipe_arguments_native_quantizers_and_one_raw_store(self):
        for recipe in ("baseline", "alg2_max7"):
            for mixed in (False, True):
                with self.subTest(recipe=recipe, mixed=mixed):
                    data = fixture(recipe, mixed)
                    expected = {"dst_type": "fp4_e2m1fn_x2"}
                    if recipe == "alg2_max7":
                        expected.update(axis=-1, round_mode="rint", block_size=32,
                                        scale_alg=2, dst_type_max=7.0)
                    writer, kernel = prepare(data)
                    self.assertEqual(writer.kv.shape, (3, 1, 512))
                    self.assertIs(writer.kv.storage, data.base.current_kv.storage)
                    self.assertIn("cpu", data.events)  # Slot validation is setup-only.
                    data.events.clear()
                    for _ in range(2):
                        self.assertIs(writer.call(), writer.cold_bytes)
                    self.assertEqual(data.base.torch_npu.npu_dynamic_mx_quant.call_count, 2)
                    data.base.torch_npu.npu_dynamic_mx_quant.assert_called_with(writer.kv, **expected)
                    self.assertEqual(data.variant._quantizer_arguments, expected)
                    if mixed:
                        self.assertEqual(data.base.torch_npu.npu_dynamic_block_quant.call_count, 2)
                        data.base.torch_npu.npu_dynamic_block_quant.assert_called_with(
                            writer.kv, dst_type="fp8_e4m3fn", row_block_size=1, col_block_size=128)
                    else:
                        data.base.torch_npu.npu_dynamic_block_quant.assert_not_called()
                    data.base.quantize.assert_not_called()
                    data.variant._quantize.assert_not_called()
                    data.base.torch.cat.assert_not_called()
                    self.assertFalse({"detach", "cpu", "numpy"} & set(data.events))
                    self.assertEqual(len(kernel.calls), 2)
                    for grid, args, constants in kernel.calls:
                        self.assertEqual(grid, (3,))
                        self.assertEqual(constants, dict(D=512, R=64, MIXED=mixed,
                                                         COLD_WIDTH=400, HOT_WIDTH=656,
                                                         BLOCK=1024 if mixed else 512))
                        for index in (0, 1, 2, 3, 4, 7, 8):
                            self.assertEqual(args[index].dtype, "uint8")
                        self.assertIs(args[0].storage, data.q4.storage)
                        self.assertIs(args[1].storage, data.s4.storage)
                        self.assertIs(args[4].storage, data.base.current_rope.storage)
                        self.assertIs(args[5], data.base.current_slots)
                        self.assertIs(args[7].storage, data.cold.storage)
                        if mixed:
                            self.assertIs(args[2].storage, data.q8.storage)
                            self.assertIs(args[3].storage, data.s8.storage)
                            self.assertIs(args[6], data.slots)
                            self.assertIs(args[8].storage, data.hot.storage)
                        else:
                            self.assertIs(args[2], args[0])
                            self.assertIs(args[3], args[1])
                            self.assertIs(args[6], args[5])
                            self.assertIs(args[8], args[7])
                    self.assertEqual(writer.metadata["quantizer_api_calls"], 2 if mixed else 1)
                    self.assertEqual(writer.metadata["pack_scatter_launches"], 1)

    def test_zero_rope_and_other_supported_width_use_correct_native_byte_counts(self):
        for dim, rope in ((128, 0), (128, 64), (1024, 0)):
            with self.subTest(dim=dim, rope=rope):
                data = fixture(dim=dim, rope=rope)
                writer, kernel = prepare(data)
                writer.call()
                _, args, constants = kernel.calls[0]
                self.assertEqual(args[0].numel(), 3*dim//2)
                self.assertEqual(args[1].numel(), 3*dim//32)
                self.assertEqual(args[2].numel(), 3*dim)
                self.assertEqual(args[3].numel(), 3*dim//32)
                self.assertEqual(args[4].numel(), 3*2*rope)
                self.assertEqual(constants["R"], rope)

    def test_quantizer_errors_propagate_without_fallback_or_store(self):
        for quantizer in ("npu_dynamic_mx_quant", "npu_dynamic_block_quant"):
            data = fixture()
            getattr(data.base.torch_npu, quantizer).side_effect = RuntimeError("unsupported native recipe")
            writer, kernel = prepare(data)
            with self.assertRaisesRegex(RuntimeError, "unsupported native recipe"):
                writer.call()
            self.assertEqual(kernel.calls, [])


class NativeStorageValidationTests(unittest.TestCase):
    def test_constructor_rejects_bad_input_cache_and_slot_metadata(self):
        cases = (("kv_dtype", "BF16"), ("rope_dtype", "BF16"), ("rope_size", "RoPE size"),
                 ("cold_width", "Cold cache"), ("cold_element_size", "Cold cache"),
                 ("cold_device", "same NPU"), ("cold_stride", "contiguous"),
                 ("kv_cpu", "Ascend NPU"), ("kv_stride", "contiguous"),
                 ("cold_slot_dtype", "integer"), ("hot_width", "Hot cache"),
                 ("hot_element_size", "Hot cache"), ("hot_slot_dtype", "Hot cache"),
                 ("hot_slots_missing", "Hot cache"), ("hot_device", "same NPU"),
                 ("hot_stride", "contiguous"), ("hot_slot_device", "same NPU"))
        for case, message in cases:
            with self.subTest(case=case):
                data = fixture()
                if case == "kv_dtype": data.base.current_kv.dtype = "fp32"
                elif case == "rope_dtype": data.base.current_rope.dtype = "fp32"
                elif case == "rope_size": data.base.current_rope.shape = (1, 2, 1, 64)
                elif case == "cold_width": data.cold.shape = (8, 1, 1, 399)
                elif case == "cold_element_size": data.cold.dtype = "bf16"
                elif case == "cold_device": data.cold.device = "npu:1"
                elif case == "cold_stride": data.cold.contiguous = False
                elif case == "kv_cpu": data.base.current_kv.device = "cpu"
                elif case == "kv_stride": data.base.current_kv.contiguous = False
                elif case == "cold_slot_dtype": data.base.current_slots.dtype = "fp32"
                elif case == "hot_width": data.hot.shape = (1, 16, 1, 655)
                elif case == "hot_element_size": data.hot.dtype = "bf16"
                elif case == "hot_slot_dtype": data.slots.dtype = "fp32"
                elif case == "hot_slots_missing": data.slots = None
                elif case == "hot_device": data.hot.device = "npu:1"
                elif case == "hot_stride": data.hot.contiguous = False
                elif case == "hot_slot_device": data.slots.device = "npu:1"
                with self.assertRaisesRegex(ValueError, message):
                    prepare(data)

    def test_slots_require_unique_in_range_fixed_destinations(self):
        for hot in (False, True):
            for bad in ([-1, 1, 3], [1, 1, 3], [1, 2], [16, 1, 3], [[1, 2, 3]]):
                with self.subTest(hot=hot, bad=bad):
                    data = fixture()
                    target = data.slots if hot else data.base.current_slots
                    target.values = np.asarray(bad, np.int64)
                    with self.assertRaisesRegex(ValueError, "physical slot|unique"):
                        prepare(data)

    def test_quantized_outputs_reject_wrong_dtype_byte_sizes_layout_or_device(self):
        cases = (("q4", "dtype", "bf16", "one-byte"),
                 ("s4", "dtype", "bf16", "one-byte"),
                 ("q8", "dtype", "int8", "E4M3FN"),
                 ("s8", "dtype", "int32", "FP32"),
                 ("q4", "shape", (3, 1, 255), "byte count"),
                 ("s4", "shape", (3, 1, 15), "byte count"),
                 ("q8", "shape", (3, 1, 511), "byte count"),
                 ("s8", "shape", (3, 1, 5), "byte count"),
                 ("q4", "contiguous", False, "contiguous"),
                 ("s4", "device", "npu:1", "input NPU"),
                 ("q8", "contiguous", False, "contiguous"),
                 ("s8", "device", "npu:1", "input NPU"))
        for name, attr, value, message in cases:
            with self.subTest(name=name, attr=attr, value=value):
                data = fixture()
                writer, kernel = prepare(data)
                setattr(getattr(data, name), attr, value)
                with self.assertRaisesRegex(RuntimeError, message):
                    writer.call()
                self.assertEqual(kernel.calls, [])


if __name__ == "__main__":
    unittest.main()
