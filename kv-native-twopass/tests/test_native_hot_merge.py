"""CPU actual-source tests; no Ascend compilation or latency claim."""

import ast
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hotkv"))
import kv_native_hot_merge as hot_merge
from native_hot_merge_support import decode_c8_token, TensorMetadata, _Tensor, _TL, _bf16


def metadata(rope=64, output_dtype="torch.float32"):
    q = TensorMetadata((3, 17, 512), "torch.bfloat16")
    qr = TensorMetadata((3, 17, rope), "torch.bfloat16")
    cache = TensorMetadata((6, 16, 1, 528+2*rope), "torch.uint8")
    table = TensorMetadata((2, 3), "torch.int32")
    mapping = TensorMetadata((3,), "torch.int32")
    cold = (TensorMetadata((3, 17, 512), output_dtype),
            TensorMetadata((3, 17, 1), "torch.float32"), TensorMetadata((3, 17, 1), "torch.float32"))
    return [q, qr, cache, table, mapping, cold]


def options(**overrides):
    args = dict(page=16, ring_capacity=33, hot_width=19, scale=.05, hot_output_dtype="torch.float32")
    args.update(overrides)
    return args


class TestHostContract(unittest.TestCase):
    def test_static_supported_tiles_and_dtypes(self):
        for bk in (16, 32, 64):
            for bh in (16, 32):
                layout = hot_merge.validate_inputs(*metadata(), **options(block_keys=bk, block_heads=bh))
                self.assertEqual((layout.block_keys, layout.block_heads, layout.pages), (bk, bh, 6))
        self.assertEqual(hot_merge.validate_inputs(*metadata(0, "torch.bfloat16"),
                                                   **options(hot_output_dtype="torch.bfloat16")).rope_dim, 0)
        self.assertEqual(hot_merge.validate_inputs(*metadata(), **options(decode="native_cast")).decode,
                         "native_cast")

    def test_bad_shapes_formats_options_and_device_rejected(self):
        for index, attr, value in ((0, "dtype", "torch.float32"), (0, "shape", (3, 17, 256)),
                                   (1, "shape", (3, 17, 32)), (2, "shape", (6, 16, 1, 400)),
                                   (2, "dtype", "torch.float8_e4m3fn"), (3, "shape", (2, 2)),
                                   (4, "device", "npu:1"), (2, "contiguous", False)):
            with self.subTest(index=index, attr=attr):
                values = metadata()
                setattr(values[index], attr, value)
                with self.assertRaises(ValueError):
                    hot_merge.validate_inputs(*values, **options())
        for bad in (dict(block_keys=8), dict(block_heads=64), dict(hot_width=0),
                    dict(page=True), dict(scale=float("nan")), dict(hot_output_dtype="torch.float16"),
                    dict(decode="auto")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                hot_merge.validate_inputs(*metadata(), **options(**bad))

    def test_one_dispatch_preallocated_outputs_and_optional_same_kernel_stats(self):
        kernel = mock.MagicMock()
        launch = kernel.__getitem__.return_value
        values = metadata()
        torch = mock.Mock()
        torch.empty_like.return_value = TensorMetadata((3, 17, 512), "torch.float32")
        torch.empty.side_effect = lambda shape, **kwargs: TensorMetadata(shape, "torch.float32")
        with mock.patch.dict(sys.modules, {"torch": torch}), mock.patch.object(hot_merge, "_load_kernel", return_value=kernel):
            runner = hot_merge.PreparedHotMerge(*values, **options())
            self.assertFalse(runner._constants["NATIVE_CAST"])
            self.assertEqual(runner._constants["SCAN_BLOCK"], 32)
            torch.empty_like.reset_mock()
            torch.empty.reset_mock()
            indices = TensorMetadata((3, 1, 19), "torch.int32")
            cp = TensorMetadata((3,), "torch.bool")
            hp = TensorMetadata((3,), "torch.bool")
            for statistics in (False, True, False):
                self.assertIs(runner.call(values[-1], indices, cp, hp, statistics=statistics), runner.output)
                self.assertEqual(launch.call_args.kwargs["STATISTICS"], statistics)
                self.assertFalse(launch.call_args.kwargs["multibuffer"])
            torch.empty.assert_not_called()
            torch.empty_like.assert_not_called()
            self.assertEqual(launch.call_count, 3)
            kernel.__getitem__.assert_called_with((3, 2))
            with self.assertRaises(ValueError):
                runner.call(values[-1], TensorMetadata((3, 1, 20), "torch.int32"), cp, hp)
        self.assertEqual(runner.metadata["source_dots_per_hot_tile"], 3)
        self.assertFalse(runner.metadata["c4_decode"])

    def test_lazy_import_does_not_require_torch_or_ascend_extension(self):
        code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'triton'):
            raise AssertionError('unexpected eager dependency: '+fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
import kv_native_hot_merge
assert kv_native_hot_merge._KERNEL is None
"""
        subprocess.run([sys.executable, "-c", code, str(ROOT / "hotkv")], check=True,
                       capture_output=True, text=True)


class Pointer:
    def __init__(self, values, name, offset=0, dtype=None):
        self.values, self.name, self.offset, self.dtype = values.reshape(-1), name, np.asarray(offset), dtype

    def __add__(self, offset):
        return Pointer(self.values, self.name, self.offset+offset, self.dtype)


class TL(_TL):
    def __init__(self):
        super().__init__()
        self.stores = []

    @staticmethod
    def log(value):
        return _Tensor(np.log(value))

    @staticmethod
    def where(condition, left, right):
        value = np.where(condition, left, right)
        if np.issubdtype(value.dtype, np.floating):
            value = value.astype(np.float32)
        return _Tensor(value)

    def store(self, pointer, value, mask=True):
        if pointer.dtype == "bf16":
            value = _bf16(value)
        super().store(pointer, value, mask)
        self.stores.append(pointer.name)


class Extension:
    def __init__(self):
        self.inserts = []

    def insert_slice(self, full, part, offsets, sizes, strides):
        self.inserts.append((full.shape, part.shape))
        assert tuple(sizes) == part.shape and tuple(strides) == (1, 1)
        assert offsets[0] == 0 and full.shape[0] == part.shape[0]
        assert 0 <= offsets[1] and offsets[1]+sizes[1] <= full.shape[1]
        result = full.copy()
        result[:, offsets[1]:offsets[1]+sizes[1]] = part
        return _Tensor(result)


def kernel_body():
    tree = ast.parse(Path(hot_merge.__file__).read_text())
    loader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_load_kernel")
    definitions = [node for node in loader.body if isinstance(node, ast.FunctionDef)]
    for definition in definitions:
        definition.decorator_list = []
    scope = {"tl": TL(), "extension": Extension()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 hot_merge.__file__, "exec"), scope)
    return scope


def fixture(rope=64, cold_dtype="fp32", hot_dtype="fp32", zero_query=False):
    rng = np.random.default_rng(191)
    rows, heads, cap, page = 3, 17, 33, 16
    q = _bf16(rng.normal(0, .03, (rows, heads, 512)))
    qr = _bf16(rng.normal(0, .1, (rows, heads, rope)))
    if zero_query:
        q.fill(0)
        qr.fill(0)
    cache = np.zeros((6, page, 1, 528+2*rope), np.uint8)
    table = np.array([[4, 0, 2], [1, 5, 3]], np.int32)
    for batch in range(2):
        for key in range(cap):
            codes = rng.integers(0, 126, 512, np.uint8)
            codes |= rng.integers(0, 2, 512, np.uint8) << 7
            rope_values = _bf16(rng.normal(0, .1, rope))
            rope_bytes = b"".join(struct.pack("<f", value)[2:] for value in rope_values)
            raw = codes.tobytes() + rope_bytes + struct.pack("<4f", .003, .01, .02, .005)
            cache[table[batch, key//page], key % page, 0] = np.frombuffer(raw, np.uint8)
    mapping = np.array([1, 0, 1], np.int32)
    indices = np.array([[32, 0, 15, 31, 16, 2, 27] + [-1]*12,
                        [0] + [-1]*18,
                        [11, 13, 17, 3, 1, 29, 9, 4, 6, 8, 26, 22, 10, 14, 0, 5, 7] + [-1]*2], np.int32)[:, None, :]
    cold_output = rng.normal(0, .2, (rows, heads, 512)).astype(np.float32)
    if cold_dtype == "bf16":
        cold_output = _bf16(cold_output)
    cold_max = rng.normal(0, 2, (rows, heads, 1)).astype(np.float32)
    cold_sum = rng.uniform(1, 9, (rows, heads, 1)).astype(np.float32)
    cp, hp = np.array([True, True, False]), np.array([True, False, True])
    cold_output[2] = cold_max[2] = cold_sum[2] = np.nan
    values = [q, qr, cache, table, mapping, indices, cold_output, cold_max, cold_sum, cp, hp]
    constants = dict(H=heads, R=rope, B=2, PAGES=6, TABLE_PAGES=3, PAGE=page, CAP=cap,
                     HW=19, SCALE=.05, BK=16, BH=16, HOT_BF16=hot_dtype == "bf16", STATISTICS=False)
    return values, constants


def reference(values, constants):
    q, qr, cache, table, mapping, indices, cold, cm, cs, cp, hp = values
    output = np.zeros_like(cold, dtype=np.float64)
    maxima = np.full(cm.shape, -np.inf, np.float64)
    totals = np.zeros(cs.shape, np.float64)
    for row in range(len(q)):
        hot = np.zeros(cold.shape[1:], np.float64)
        if hp[row]:
            keys = [int(key) for key in indices[row, 0] if key >= 0]
            decoded = [decode_c8_token(cache[table[mapping[row], key//constants["PAGE"]],
                                                  key % constants["PAGE"], 0], constants["R"]) for key in keys]
            latent = _bf16(np.array([entry[0] for entry in decoded])).astype(np.float64)
            rope = np.array([entry[1] for entry in decoded], np.float64)
            score = (q[row].astype(np.float64) @ latent.T + qr[row].astype(np.float64) @ rope.T) * constants["SCALE"]
            maxima[row, :, 0] = score.max(axis=1)
            probability = np.exp(score-maxima[row])
            totals[row] = probability.sum(axis=1, keepdims=True)
            hot = probability / totals[row] @ latent
            if constants["HOT_BF16"]:
                hot = _bf16(hot).astype(np.float64)
        if not cp[row] and not hp[row]:
            continue
        cold_max = cm[row] if cp[row] else np.full(cm[row].shape, -np.inf)
        maximum = np.maximum(cold_max, maxima[row])
        cw = cs[row].astype(np.float64)*np.exp(cold_max-maximum) if cp[row] else np.zeros_like(maximum)
        hw = totals[row]*np.exp(maxima[row]-maximum) if hp[row] else np.zeros_like(maximum)
        left = cold[row] if cp[row] else np.zeros_like(cold[row])
        output[row] = left*(cw/(cw+hw)) + hot*(hw/(cw+hw))
    return output, maxima, totals


class TestActualBody(unittest.TestCase):
    decode = "software"

    def execute(self, values, constants, cold_dtype="fp32", decode=None):
        constants = dict(constants, NATIVE_CAST=(decode or self.decode) == "native_cast",
                         SCAN_BLOCK=1 << (constants["HW"]-1).bit_length())
        scope = kernel_body()
        language = scope["tl"]
        output = np.full(values[6].shape, np.nan, np.float32)
        hot_max = np.full(values[7].shape, 123, np.float32)
        hot_sum = np.full(values[8].shape, 123, np.float32)
        names = ("Q", "QR", "CACHE", "TABLE", "ROW_BATCH", "INDICES", "COLD", "CMAX", "CSUM", "CP", "HP")
        pointers = [Pointer(value, name) for value, name in zip(values, names)]
        pointers += [Pointer(output, "OUT", dtype=cold_dtype), Pointer(hot_max, "HOT_MAX"), Pointer(hot_sum, "HOT_SUM")]
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            for row in range(len(output)):
                for group in range((constants["H"] + constants["BH"]-1)//constants["BH"]):
                    language.pid = row, group
                    scope["_hot_merge"](*pointers, **constants)
        return output, hot_max, hot_sum, scope

    def test_all_tiles_precisions_rope_and_partial_head_tail_match_reference(self):
        for rope in (0, 64):
            for cold_dtype, hot_dtype in (("fp32", "fp32"), ("bf16", "bf16"), ("fp32", "bf16")):
                values, base = fixture(rope, cold_dtype, hot_dtype)
                expected, _, _ = reference(values, base)
                for bk in (16, 32, 64):
                    for bh in (16, 32):
                        with self.subTest(rope=rope, cold=cold_dtype, hot=hot_dtype, bk=bk, bh=bh):
                            constants = dict(base, BK=bk, BH=bh)
                            actual, _, _, _ = self.execute(values, constants, cold_dtype)
                            self.assertTrue(np.isfinite(actual).all())
                            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected), .006)

    def test_zero_query_stats_count_and_one_decode_three_dots(self):
        values, constants = fixture(zero_query=True)
        constants["STATISTICS"] = True
        actual, maximum, total, scope = self.execute(values, constants)
        expected, expected_max, expected_sum = reference(values, constants)
        np.testing.assert_array_equal(maximum, expected_max)
        np.testing.assert_array_equal(total, expected_sum)
        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected), .006)
        groups = 2
        active_rows = 2
        tiles = sum((np.count_nonzero(values[5][row] >= 0)+constants["BK"]-1)//constants["BK"]
                    for row in range(len(values[10])) if values[10][row])
        self.assertEqual(len(scope["tl"].dots), groups*tiles*3)
        self.assertEqual(len(scope["extension"].inserts), groups*tiles*8)
        q_loads = [entry for entry in scope["tl"].loads if entry[0] == "Q"]
        self.assertEqual(len(q_loads), active_rows*groups)

    def test_stats_disabled_does_not_write_diagnostic_buffers(self):
        values, constants = fixture()
        _, maximum, total, scope = self.execute(values, constants)
        self.assertTrue(np.all(maximum == 123) and np.all(total == 123))
        self.assertNotIn("HOT_MAX", scope["tl"].stores)
        self.assertNotIn("HOT_SUM", scope["tl"].stores)

    def test_nonzero_hot_log_partition_matches_independent_reference(self):
        values, constants = fixture()
        constants["STATISTICS"] = True
        _, maximum, total, _ = self.execute(values, constants)
        _, expected_max, expected_sum = reference(values, constants)
        present = values[10]
        actual_lse = maximum[present] + np.log(total[present])
        expected_lse = expected_max[present] + np.log(expected_sum[present])
        np.testing.assert_allclose(actual_lse, expected_lse, rtol=2e-5, atol=2e-5)
        self.assertTrue(np.isneginf(maximum[~present]).all())
        self.assertTrue((total[~present] == 0).all())

    def test_native_hot_bf16_rounding_occurs_before_fp32_final_merge(self):
        values, constants = fixture(zero_query=True)
        values[9].fill(False)
        values[10].fill(True)
        values[5].fill(-1)
        values[5][:, 0, :2] = [0, 1]
        # With two equal-score keys, probabilities in PV are exactly one;
        # normalization averages two BF16 values and may require FP32 bits.
        fp32, _, _, _ = self.execute(values, dict(constants, HOT_BF16=False))
        rounded, _, _, _ = self.execute(values, dict(constants, HOT_BF16=True))
        np.testing.assert_array_equal(rounded, _bf16(fp32))
        self.assertTrue(np.any(fp32 != rounded))

    def test_empty_hot_ignores_dummy_and_cache_even_when_poisoned(self):
        values, constants = fixture()
        values[10].fill(False)
        values[9].fill(True)
        values[6][2] = values[6][0]
        values[7][2] = 0
        values[8][2] = 1
        values[2].fill(255)
        values[4].fill(-1)
        actual, _, _, scope = self.execute(values, constants)
        np.testing.assert_array_equal(actual, values[6])
        self.assertEqual(len(scope["tl"].dots), 0)
        self.assertFalse(any(name in ("CACHE", "TABLE", "INDICES") for name, _, _ in scope["tl"].loads))

    def test_both_empty_ignores_poisoned_cold_state(self):
        values, constants = fixture()
        values[9].fill(False)
        values[10].fill(False)
        for tensor in values[6:9]:
            tensor.fill(np.nan)
        actual, _, _, _ = self.execute(values, constants)
        self.assertTrue(np.array_equal(actual, np.zeros_like(actual)))

    def test_invalid_hot_inputs_and_present_empty_return_nan_safely(self):
        for bad in ("index", "negative", "page", "batch", "empty", "hole", "tail_negative", "tail_oob"):
            with self.subTest(bad=bad):
                values, constants = fixture()
                if bad == "index": values[5][0, 0, 0] = constants["CAP"]
                elif bad == "negative": values[5][0, 0, 0] = -2
                elif bad == "page": values[3][1, 2] = 99
                elif bad == "batch": values[4][0] = 99
                elif bad == "hole": values[5][0, 0, 0] = -1
                elif bad == "tail_negative": values[5][0, 0, -1] = -2
                elif bad == "tail_oob": values[5][0, 0, -1] = constants["CAP"]
                else: values[5][0].fill(-1)
                actual, _, _, _ = self.execute(values, constants)
                self.assertTrue(np.isnan(actual[0]).all())

    def test_nonempty_cold_invalid_stats_propagate_and_empty_cold_is_ignored(self):
        for field, bad in ((7, np.nan), (7, np.inf), (7, -np.inf), (8, np.nan), (8, np.inf), (8, 0)):
            with self.subTest(field=field, value=bad):
                values, constants = fixture()
                values[field][0].fill(bad)
                actual, _, _, _ = self.execute(values, constants)
                self.assertTrue(np.isnan(actual[0]).all())
                self.assertTrue(np.isfinite(actual[2]).all())

    def test_decoder_all_256_codes_scale_before_bf16_rounding(self):
        scope = kernel_body()
        raw = bytes(range(256))*2 + struct.pack("<4f", 1.00390625, .03125, 5.25, .3)
        cache = np.frombuffer(raw, np.uint8).copy()
        active = _Tensor(np.array([True] + [False]*15))
        base = _Tensor(np.zeros(16, np.int64))
        actual = []
        for start in range(0, 512, 64):
            part = scope["_decode_c8"](Pointer(cache, "CACHE"), base, active, start, 0, 16,
                                       self.decode == "native_cast")
            actual.extend(part[0])
        expected = _bf16(decode_c8_token(raw, 0)[0])
        np.testing.assert_array_equal(np.asarray(actual), expected)
        self.assertTrue(np.signbit(actual[128]))

    def test_padding_capacity_does_not_add_attention_tiles(self):
        values, constants = fixture()
        narrow, _, _, narrow_scope = self.execute(values, constants)
        values[5] = np.pad(values[5], ((0, 0), (0, 0), (0, 109)), constant_values=-1)
        wide, _, _, wide_scope = self.execute(values, dict(constants, HW=128))
        np.testing.assert_array_equal(wide, narrow)
        self.assertEqual(len(wide_scope["tl"].dots), len(narrow_scope["tl"].dots))
        self.assertEqual(len(wide_scope["extension"].inserts), len(narrow_scope["extension"].inserts))
        self.assertTrue(any(name == "INDICES" and shape == (128,) for name, shape, _ in wide_scope["tl"].loads))


class TestNativeCastBody(TestActualBody):
    # Repeat every numerical, dtype, empty, malformed and six-tile contract
    # through the explicit hardware-cast source branch. CPU emulation cannot
    # establish support/accuracy of the installed A5 cast implementation.
    decode = "native_cast"

    def test_native_cast_and_software_match_on_identical_inputs(self):
        values, constants = fixture()
        native, native_max, native_sum, _ = self.execute(values, dict(constants, STATISTICS=True))
        software, software_max, software_sum, _ = self.execute(values, dict(constants, STATISTICS=True), decode="software")
        np.testing.assert_array_equal(native, software)
        np.testing.assert_array_equal(native_max, software_max)
        np.testing.assert_array_equal(native_sum, software_sum)


if __name__ == "__main__":
    unittest.main()
