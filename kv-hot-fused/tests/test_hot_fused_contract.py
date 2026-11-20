"""CPU contracts only: these tests do NOT establish A5 kernel correctness."""

import ast
import importlib.util
import math
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hotkv"))
import kv_hot_fused as fused


class TensorMetadata:
    """Minimal static metadata; any attempt to read values fails this test."""

    def __init__(self, shape, dtype, device="npu:0", contiguous=True):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.contiguous = contiguous
        self.requires_grad = False

    def is_contiguous(self):
        return self.contiguous

    def item(self):
        raise AssertionError("Host synchronization is forbidden")

    def cpu(self):
        raise AssertionError("Device copies are forbidden")

    def numpy(self):
        raise AssertionError("Device copies are forbidden")


def inputs(rope=64):
    return [
        TensorMetadata((4, 64, 512), "torch.bfloat16"),
        TensorMetadata((4, 64, rope), "torch.bfloat16"),
        TensorMetadata((4, 1, 2048), "torch.int32"),
        TensorMetadata((512, 128, 1, 272 + 2 * rope), "torch.uint8"),
        TensorMetadata((2, 256), "torch.int32"),
        TensorMetadata((2, 640, 1, 528 + 2 * rope), "torch.uint8"),
        TensorMetadata((2, 640), "torch.int64"),
        TensorMetadata((4,), "torch.int32"),
        TensorMetadata((4,), "torch.int32"),
        TensorMetadata((2,), "torch.int32"),
    ]


def validate(values, **overrides):
    options = dict(window=512, page=128, scale=1.0 / 24)
    options.update(overrides)
    return fused.validate_inputs(*values, **options)


def bf16_bytes(value):
    return struct.pack("<f", value)[2:]


class TestPackedByteFormats(unittest.TestCase):
    def test_c4_all_e2m1_levels_low_nibble_first(self):
        # Bytes 0x10,0x32,... encode scalar codes 0,1,2,...,15.
        token = bytes((hi << 4) | lo for lo, hi in zip(range(0, 16, 2), range(1, 16, 2))) * 32
        token += bf16_bytes(1.5) * 64 + bytes([127] * 16)
        latent, rope = fused.decode_c4_token(token)
        self.assertEqual(latent[:16], [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
        self.assertLess(math.copysign(1, latent[8]), 0)
        self.assertEqual(rope, [1.5] * 64)

    def test_c4_scale_group_boundaries_and_r0(self):
        # E2M1 code2 = 1.0. Distinct adjacent scale groups catch wrong 32/128 grouping.
        token = bytes([0x22] * 256) + bytes(range(119, 135))
        latent, rope = fused.decode_c4_token(token, rope_dim=0)
        self.assertEqual(rope, [])
        for group in range(16):
            self.assertEqual(latent[32 * group:32 * (group + 1)], [2 ** (group - 8)] * 32)
        with self.assertRaisesRegex(ValueError, "255"):
            fused.decode_c4_token(bytes([0] * 256 + [255] * 16), rope_dim=0)

    def test_c8_signed_subnormal_extrema_and_rope(self):
        values = bytes([0x00, 0x80, 0x01, 0x81, 0x38, 0xB8, 0x7E, 0xFE]) * 64
        token = values + bf16_bytes(-2.25) * 64 + struct.pack("<4f", 1, 2, .5, 4)
        latent, rope = fused.decode_c8_token(token)
        self.assertEqual(latent[:8], [0, 0, 1 / 512, -1 / 512, 1, -1, 448, -448])
        self.assertLess(math.copysign(1, latent[1]), 0)
        self.assertEqual(latent[132], 2)
        self.assertEqual(latent[260], .5)
        self.assertEqual(latent[388], 4)
        self.assertEqual(rope, [-2.25] * 64)

    def test_c8_nan_encoding_and_exact_float32_scale(self):
        token = bytes([0x7F, 0xFF] + [0x38] * 510) + struct.pack("<4f", 1.125, 1, 1, 1)
        latent, rope = fused.decode_c8_token(token, rope_dim=0)
        self.assertTrue(math.isnan(latent[0]))
        self.assertTrue(math.isnan(latent[1]))
        self.assertEqual(latent[2], 1.125)
        self.assertEqual(rope, [])

    def test_bad_byte_counts_rejected(self):
        for decoder in (fused.decode_c4_token, fused.decode_c8_token):
            with self.assertRaises(ValueError):
                decoder(b"\x00")
            with self.assertRaises(ValueError):
                decoder(b"", rope_dim=32)


class TestStaticPreparationContract(unittest.TestCase):
    def test_metadata_only_validation(self):
        layout = validate(inputs())
        self.assertEqual((layout.cold_width, layout.hot_width), (400, 656))
        self.assertEqual(layout.sparse_capacity, 2048)
        self.assertEqual(layout.ring_capacity, 640)
        self.assertEqual(layout.rows, 4)
        self.assertEqual(layout.block_keys, 16)
        r0 = validate(inputs(rope=0))
        self.assertEqual((r0.cold_width, r0.hot_width), (272, 528))

    def test_w0_must_bypass_fused_reader(self):
        with self.assertRaisesRegex(ValueError, "window"):
            validate(inputs(), window=0)

    def test_wrong_dtype_device_strides_shape_rejected(self):
        for index, attr, value, error in [
            (0, "device", "cpu", "NPU"),
            (3, "dtype", "torch.float8_e4m3fn", "dtype"),
            (5, "shape", (2, 640, 1, 400), "hot_cache"),
            (6, "shape", (2, 512), "hot_positions"),
            (1, "contiguous", False, "contiguous"),
            (9, "device", "npu:1", "same NPU"),
            (0, "shape", (4, 64, 256), "512"),
            (0, "requires_grad", True, "inference"),
        ]:
            with self.subTest(index=index, attribute=attr):
                tensors = inputs()
                setattr(tensors[index], attr, value)
                with self.assertRaisesRegex(ValueError, error):
                    validate(tensors)

    def test_invalid_static_options_rejected(self):
        for kwargs in (dict(window=True), dict(window=641), dict(scale=math.nan),
                       dict(scale=0), dict(page=64), dict(block_keys=8), dict(block_heads=32)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate(inputs(), **kwargs)

    def test_one_dispatch_reuses_preallocated_output(self):
        # This checks the host launch contract only, not kernel arithmetic.
        class Kernel:
            def __init__(self):
                self.calls = []

            def __getitem__(self, grid):
                def launch(*args, **constants):
                    self.calls.append((grid, args, constants))
                return launch

        kernel = Kernel()
        output = object()
        fake_torch = mock.Mock()
        fake_torch.empty.return_value = output
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), mock.patch.object(fused, "_load_kernel", return_value=kernel):
            tensors = inputs()
            runner = fused.FusedHotKVReader(*tensors, window=512, page=128, scale=1 / 24)
            fake_torch.empty.assert_called_once()
            fake_torch.empty.reset_mock()
            for _ in range(3):
                self.assertIs(runner.call(), output)
            fake_torch.empty.assert_not_called()
        self.assertEqual(len(kernel.calls), 3)
        for grid, args, constants in kernel.calls:
            self.assertEqual(grid, (4, 4))
            self.assertEqual(args[:-1], tuple(tensors))
            self.assertIs(args[-1], output)
            self.assertEqual(constants["WINDOW"], 512)
            self.assertEqual(constants["CAP"], 640)
            self.assertFalse(constants["multibuffer"])
            self.assertEqual(constants["BK"], 16)
            self.assertTrue(constants["HOT_ENABLED"])
        self.assertEqual(runner.metadata["attention_launches_per_call"], 1)
        self.assertEqual(runner.metadata["revision"], 3)
        self.assertEqual(runner.metadata["kv_decode_passes_per_key_tile"], 1)
        self.assertEqual(runner.metadata["source_dot_calls_per_key_tile"], 3)
        self.assertIsNone(runner.metadata["fallback"])

    def test_cold_diagnostic_is_explicit_and_does_not_enable_w0(self):
        with mock.patch.dict(sys.modules, {"torch": mock.Mock()}), mock.patch.object(fused, "_load_kernel"):
            runner = fused.FusedHotKVReader(*inputs(), window=512, page=128, scale=1 / 24,
                                            diagnostic_cold_only=True)
            self.assertFalse(runner._constants["HOT_ENABLED"])
            self.assertEqual(runner.metadata["semantic_window"], 0)
            self.assertTrue(runner.metadata["diagnostic_cold_only"])
            with self.assertRaisesRegex(ValueError, "window"):
                fused.FusedHotKVReader(*inputs(), window=0, page=128, scale=1 / 24,
                                       diagnostic_cold_only=True)
            with self.assertRaisesRegex(ValueError, "diagnostic_cold_only"):
                fused.FusedHotKVReader(*inputs(), window=512, page=128, scale=1 / 24,
                                       diagnostic_cold_only="true")

    def test_import_does_not_require_torch_or_triton(self):
        code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'triton', 'torch_npu'):
            raise AssertionError('Unexpected eager dependency import: '+fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
import kv_hot_fused
assert kv_hot_fused._KERNEL is None
"""
        subprocess.run([sys.executable, "-c", code, str(ROOT / "hotkv")], check=True,
                       capture_output=True, text=True)


# Execute the ACTUAL helper/kernel AST with a tiny numerical implementation of
# the used TL primitives. This validates feature-loop arithmetic and masked
# pointer addressing; it is not a Triton interpreter or a compiler/resource test.
def _bf16(value):
    value = np.asarray(value, np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


class _Tensor(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def to(self, dtype, bitcast=False):
        value = np.asarray(self)
        if dtype == "bf16":
            result = ((value.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
                      if bitcast else _bf16(value))
        else:
            result = value.view(dtype) if bitcast else value.astype(dtype)
        return _Tensor(result)


class _Pointer:
    def __init__(self, values, offset=0, name=None):
        self.values = values.reshape(-1)
        self.offset = np.asarray(offset)
        self.name = name

    def __add__(self, offset):
        return _Pointer(self.values, self.offset + offset, self.name)


class _TL:
    constexpr = object
    float32, int32, int64 = np.float32, np.int32, np.int64
    uint8, uint16, uint32 = np.uint8, np.uint16, np.uint32
    bfloat16 = "bf16"
    range = staticmethod(range)

    def __init__(self):
        self.pid = (0, 0)
        self.load_shapes = []
        self.loads = []
        self.dots = []

    def program_id(self, axis):
        return _Tensor(np.int32(self.pid[axis]))

    @staticmethod
    def arange(start, end):
        return _Tensor(np.arange(start, end, dtype=np.int32))

    @staticmethod
    def full(shape, fill, dtype):
        return _Tensor(np.full(shape, fill, np.float32 if dtype == "bf16" else dtype))

    def load(self, pointer, mask=True, other=0):
        indices = np.asarray(pointer.offset)
        self.load_shapes.append(indices.shape)
        active = np.broadcast_to(mask, indices.shape)
        self.loads.append((pointer.name, indices.shape, int(np.count_nonzero(active))))
        if np.any(indices[active] < 0) or np.any(indices[active] >= pointer.values.size):
            raise AssertionError("Unmasked out-of-range kernel load")
        result = np.full(indices.shape, other, dtype=pointer.values.dtype)
        result[active] = pointer.values[indices[active]]
        return _Tensor(result)

    @staticmethod
    def store(pointer, value, mask=True):
        indices = np.asarray(pointer.offset)
        active = np.broadcast_to(mask, indices.shape)
        if np.any(indices[active] < 0) or np.any(indices[active] >= pointer.values.size):
            raise AssertionError("Unmasked out-of-range kernel store")
        pointer.values[indices[active]] = np.broadcast_to(value, indices.shape)[active]

    @staticmethod
    def where(mask, yes, no):
        return _Tensor(np.where(mask, yes, no))

    @staticmethod
    def maximum(a, b):
        return _Tensor(np.maximum(a, b))

    @staticmethod
    def exp2(value):
        return _Tensor(np.exp2(value))

    @staticmethod
    def exp(value):
        return _Tensor(np.exp(value))

    @staticmethod
    def sum(value, axis):
        return _Tensor(np.sum(value, axis=axis))

    @staticmethod
    def max(value, axis):
        return _Tensor(np.max(value, axis=axis))

    def dot(self, a, b):
        self.dots.append((a.shape, b.shape))
        return _Tensor(np.matmul(a, b))

    @staticmethod
    def trans(value):
        return _Tensor(np.transpose(value))

    @staticmethod
    def reshape(value, shape):
        return _Tensor(np.reshape(value, shape))

def _kernel_body():
    tree = ast.parse(Path(fused.__file__).read_text())
    loader = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "_load_kernel")
    functions = [x for x in loader.body if isinstance(x, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    language = _TL()
    scope = {"tl": language}
    exec(compile(module, fused.__file__, "exec"), scope)
    return language, scope["_mixed_sparse_mla"]


class TestDecodeOnceKernelBody(unittest.TestCase):
    def fixture(self, rope=64, zero_query=False, hot_enabled=True):
        rng = np.random.default_rng(23)
        keys, heads, page, capacity, window = 47, 17, 16, 16, 9
        q = _bf16(rng.normal(0, .15, (1, heads, 512)))
        qr = _bf16(rng.normal(0, .1, (1, heads, rope)))
        if zero_query:
            q.fill(0)
            qr.fill(0)
        kr = _bf16(rng.normal(0, .1, (keys, rope)))
        cold_tokens, hot_tokens = [], []
        for token in range(keys):
            packed = rng.integers(0, 256, size=256, dtype=np.uint8).tobytes()
            rbytes = b"".join(bf16_bytes(x) for x in kr[token])
            cold_tokens.append(packed + rbytes + rng.integers(123, 127, size=16, dtype=np.uint8).tobytes())
            # Finite E4M3 codes, distinct scale/128 groups and original RoPE.
            fp8 = rng.integers(0, 126, size=512, dtype=np.uint8)
            fp8 |= (rng.integers(0, 2, size=512, dtype=np.uint8) << 7)
            hot_tokens.append(fp8.tobytes() + rbytes + struct.pack("<4f", .01, .025, .05, .1))
        cold_width, hot_width = len(cold_tokens[0]), len(hot_tokens[0])
        table = np.array([[2, 0, 1]], np.int32)
        cold = np.zeros((3, page, 1, cold_width), np.uint8)
        for token in range(keys):
            cold[table[0, token // page], token % page, 0] = np.frombuffer(cold_tokens[token], np.uint8)
        hot = np.zeros((1, capacity, 1, hot_width), np.uint8)
        tags = np.full((1, capacity), -1, np.int64)
        position = 43
        for token in range(position-window+1, position+1):
            hot[0, token % capacity, 0] = np.frombuffer(hot_tokens[token], np.uint8)
            tags[0, token % capacity] = token
        # Padding, a future token, and >BK keys exercise empty and mixed tiles.
        ids = np.array([0, 1, 2, 3, 7, 8, 12, 15, 16, 17, 19, 20, 21, 25, 29,
                        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                        44, -1, -1, -1], np.int32).reshape(1, 1, -1)
        values = [q, qr, ids, cold, table, hot, tags, np.array([0], np.int32),
                  np.array([position], np.int64), np.array([keys], np.int32)]
        constants = dict(H=heads, S=ids.shape[-1], R=rope, B=1, PAGES=3, TABLE_PAGES=3,
                         PAGE=page, CAP=capacity, CW=cold_width, HW=hot_width,
                         WINDOW=window, SCALE=.04, BK=16, BH=16, HOT_ENABLED=hot_enabled)
        active = [int(i) for i in ids.reshape(-1) if 0 <= i <= position]
        kv = np.array([fused.decode_c8_token(hot_tokens[i], rope)[0] if hot_enabled and i > position-window
                       else fused.decode_c4_token(cold_tokens[i], rope)[0] for i in active])
        # Match BF16 dot operands while keeping reference reductions in FP64.
        kv = _bf16(kv).astype(np.float64)
        scores = (q[0].astype(np.float64) @ kv.T + qr[0].astype(np.float64) @ kr[active].T) * constants["SCALE"]
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        reference = ((probs / probs.sum(axis=-1, keepdims=True)) @ kv)[None]
        return values, constants, reference

    def execute(self, values, constants, return_trace=False):
        language, kernel = _kernel_body()
        output = np.zeros((1, constants["H"], 512), np.float32)
        names = ("Q", "QR", "IDX", "COLD", "TABLE", "HOT", "TAGS", "ROW_BATCH", "POS", "LENGTHS")
        pointers = [_Pointer(x, name=name) for name, x in zip(names, values)] + [_Pointer(output, name="OUT")]
        with np.errstate(invalid="ignore", over="ignore"):
            for group in range((constants["H"] + constants["BH"] - 1) // constants["BH"]):
                language.pid = (0, group)
                kernel(*pointers, **constants)
        # Source-level transfer bound only; the actual compiler can still
        # allocate extra buffers. Catch accidental full-512 decode loads.
        self.assertTrue(all(len(shape) < 2 or shape[-1] <= 64
                            for name, shape, _ in language.loads if name in ("COLD", "HOT")))
        return (output, language) if return_trace else output

    def test_decode_once_qk_pv_agree_with_independent_reference(self):
        for rope in (0, 64):
            for bk in (16, 32):
                for zero_query in (False, True):
                    with self.subTest(rope=rope, keys=bk, zero_query=zero_query):
                        values, constants, reference = self.fixture(rope, zero_query)
                        constants["BK"] = bk
                        output = self.execute(values, constants)
                        relative_l2 = np.linalg.norm(output-reference) / np.linalg.norm(reference)
                        self.assertLess(relative_l2, .004)
                        self.assertTrue(np.isfinite(output).all())

    def test_unsorted_hot_cold_and_padding_remain_correct(self):
        values, constants, reference = self.fixture()
        np.random.default_rng(91).shuffle(values[2].reshape(-1))
        output = self.execute(values, constants)
        self.assertLess(np.linalg.norm(output-reference) / np.linalg.norm(reference), .004)

    def test_full_width_dots_and_q_load_count_do_not_scale_with_features(self):
        # This executes the actual kernel body, not a duplicate implementation.
        # Counts guard the source regression that caused 17 dots per key tile
        # and reloaded Q in every feature loop. They are NOT NPU timing claims.
        for rope in (0, 64):
            values, constants, _ = self.fixture(rope)
            _, trace = self.execute(values, constants, return_trace=True)
            groups = math.ceil(constants["H"] / constants["BH"])
            tiles = math.ceil(constants["S"] / constants["BK"])
            self.assertEqual(len(trace.dots), groups * tiles * (3 if rope else 2))
            q_loads = [shape for name, shape, _ in trace.loads if name == "Q"]
            self.assertEqual(q_loads, [(constants["BH"], 512)] * groups)
            self.assertEqual(sum(a[-1] == 512 and b[0] == 512 for a, b in trace.dots), groups * tiles)
            self.assertEqual(sum(b[-1] == 512 for _, b in trace.dots), groups * tiles)

    def test_compile_time_cold_control_matches_c4_and_never_reads_hot(self):
        values, constants, reference = self.fixture(hot_enabled=False)
        values[6].fill(-1)  # Diagnostic cannot depend on ring validity.
        output, trace = self.execute(values, constants, return_trace=True)
        self.assertLess(np.linalg.norm(output-reference) / np.linalg.norm(reference), .004)
        self.assertFalse(any(name in ("HOT", "TAGS") for name, _, _ in trace.loads))

    def test_all_cold_tile_skips_c8_decode_and_decodes_once(self):
        values, constants, _ = self.fixture()
        values[2] = np.arange(16, dtype=np.int32).reshape(1, 1, 16)
        constants["S"] = 16
        _, trace = self.execute(values, constants, return_trace=True)
        groups = math.ceil(constants["H"] / constants["BH"])
        self.assertFalse(any(name == "HOT" for name, _, _ in trace.loads))
        # Per feature: one packed-byte matrix and one 2-scale matrix. RoPE
        # adds two byte matrices. A second decode pass would double 16 here.
        cold_loads = [shape for name, shape, _ in trace.loads if name == "COLD"]
        self.assertEqual(len(cold_loads), groups * (8 * 2 + 2))

    def test_all_hot_tile_skips_c4_decode_but_retains_original_rope(self):
        values, constants, _ = self.fixture(zero_query=True)
        values[2] = np.arange(35, 44, dtype=np.int32).reshape(1, 1, 9)
        constants["S"] = 9
        output, trace = self.execute(values, constants, return_trace=True)
        # Zero Q makes the independent result the mean of valid decoded hot
        # vectors. This also catches wrong assembly order in the hot-only arm.
        decoded = [_bf16(fused.decode_c8_token(values[5][0, int(key) % constants["CAP"], 0],
                                               constants["R"])[0])
                   for key in values[2].reshape(-1)]
        reference = np.broadcast_to(np.mean(np.asarray(decoded, np.float64), axis=0), output.shape)
        self.assertLess(np.linalg.norm(output-reference) / np.linalg.norm(reference), .004)
        groups = math.ceil(constants["H"] / constants["BH"])
        cold_loads = [shape for name, shape, _ in trace.loads if name == "COLD"]
        self.assertEqual(len(cold_loads), groups * 2)  # Only original RoPE.
        hot_loads = [shape for name, shape, _ in trace.loads if name == "HOT"]
        self.assertEqual(len(hot_loads), groups * 8 * 5)  # FP8 + four scale bytes.

    def test_missing_hot_tag_and_invalid_page_surface_as_nan(self):
        for bad in ("tag", "page", "batch"):
            with self.subTest(bad=bad):
                values, constants, _ = self.fixture()
                if bad == "tag":
                    values[6][0, 43 % constants["CAP"]] = -1
                elif bad == "page":
                    values[4][0, 0] = 99
                else:
                    values[7][0] = -1
                self.assertTrue(np.isnan(self.execute(values, constants)).all())

    def test_all_invalid_selection_is_zero_and_safe(self):
        values, constants, _ = self.fixture()
        values[2].fill(-1)
        self.assertTrue(np.array_equal(self.execute(values, constants), np.zeros((1, 17, 512))))


if __name__ == "__main__":
    unittest.main()
