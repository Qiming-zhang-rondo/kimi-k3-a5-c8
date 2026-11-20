"""CPU contracts only: these tests do NOT establish A5 kernel correctness."""

import importlib.util
import math
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest import mock


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
                       dict(scale=0), dict(page=64), dict(block_keys=16), dict(block_heads=32)):
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
        self.assertEqual(runner.metadata["attention_launches_per_call"], 1)
        self.assertIsNone(runner.metadata["fallback"])

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


if __name__ == "__main__":
    unittest.main()
