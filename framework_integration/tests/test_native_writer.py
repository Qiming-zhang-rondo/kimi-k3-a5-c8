"""Host contracts only. These tests do not compile or execute an A5 kernel."""
import ast
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "framework_integration"))
from sfa_recent128.native_writer import NativeWriterError, PreparedNativeWriter
from sfa_recent128.prolog_provider import RingBinding


class DeviceTensor:
    """Tensor metadata whose payload and conversion/allocation paths are forbidden."""
    def __init__(self, shape, dtype, *, strides=None, device="npu:0"):
        self.shape, self.dtype, self.device = tuple(shape), "torch." + dtype, device
        steps, step = [], 1
        for dim in reversed(shape):
            steps.insert(0, step)
            step *= dim
        self._strides = tuple(steps) if strides is None else tuple(strides)
        self.payload = object()

    def stride(self):
        return self._strides

    def is_contiguous(self):
        expected = 1
        for dim, step in zip(reversed(self.shape), reversed(self._strides)):
            if dim > 1 and step != expected:
                return False
            expected *= dim
        return True

    def forbidden(self, *args, **kwargs):
        raise AssertionError("device contents must remain unread/unconverted")

    __bool__ = __iter__ = __getitem__ = item = cpu = numpy = tolist = to = copy_ = forbidden
    clone = contiguous = new_empty = forbidden


class RecordingKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **constants):
            self.calls.append((grid, args, constants))
        return launch


def buffers(rows=4):
    # Real Native CP gathers interleaved D512/R64 rows: both views stride 576.
    latent = DeviceTensor((rows, 1, 1, 512), "bfloat16", strides=(576, 576, 576, 1))
    rope = DeviceTensor((rows, 64), "bfloat16", strides=(576, 1))
    ring = RingBinding(
        DeviceTensor((2, 256, 512), "bfloat16"), DeviceTensor((2, 256, 64), "bfloat16"),
        DeviceTensor((2, 256), "int64"), DeviceTensor((2, 256), "int64"),
        DeviceTensor((rows,), "int32"), DeviceTensor((rows,), "int64"),
        DeviceTensor((rows,), "int64"), DeviceTensor((rows,), "bool"), 4)
    return latent, rope, ring


class NativeWriterTests(unittest.TestCase):
    def test_import_requires_no_framework(self):
        script = '''
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'triton', 'torch_npu'):
        raise AssertionError('framework imported while importing native_writer')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from sfa_recent128.native_writer import PreparedNativeWriter
'''
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT / "framework_integration",
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_original_buffers_actual_strides_single_launch_and_device_valid_unread(self):
        latent, rope, ring = buffers()
        kernel = RecordingKernel()
        originals = (latent, rope, *vars(ring).values())
        states = tuple(dict(vars(value)) for value in originals if isinstance(value, DeviceTensor))
        writer = PreparedNativeWriter(latent, rope, ring, kernel_factory=lambda: kernel)
        self.assertEqual(kernel.calls, [])  # Preparation does not write real ring contents.
        self.assertIs(writer.warmup(latent, rope), ring)
        self.assertIs(writer(latent, rope), ring)
        self.assertEqual(len(kernel.calls), 2)
        grid, args, constants = kernel.calls[-1]
        self.assertEqual(grid, (4,))
        self.assertTrue(all(a is b for a, b in zip(args, originals[:-1])))
        self.assertIs(args[-1], ring.token_valid)
        self.assertEqual(constants, dict(POOLS=2, CAPACITY=256, LATENT_ROW=576,
                                        LATENT_COL=1, ROPE_ROW=576, ROPE_COL=1))
        self.assertEqual(states, tuple(dict(vars(value)) for value in originals
                                       if isinstance(value, DeviceTensor)))
        self.assertEqual(writer.configuration()["device_validation"], "PENDING")
        self.assertFalse(writer.configuration()["performance_claim"])

    def test_flattened_leading_dimensions_and_column_strides(self):
        latent, rope, ring = buffers()
        latent = DeviceTensor((2, 2, 512), "bfloat16", strides=(2048, 1024, 2))
        kernel = RecordingKernel()
        writer = PreparedNativeWriter(latent, rope, ring, kernel_factory=lambda: kernel)
        writer.write(latent, rope)
        self.assertEqual(kernel.calls[0][2]["LATENT_ROW"], 1024)
        self.assertEqual(kernel.calls[0][2]["LATENT_COL"], 2)

    def test_rejects_bad_shape_dtype_device_and_nonflattenable_input(self):
        latent, rope, ring = buffers()
        bad_latents = (
            DeviceTensor((4, 511), "bfloat16"), DeviceTensor((4, 512), "float32"),
            DeviceTensor((4, 512), "bfloat16", device="cpu"),
            DeviceTensor((2, 2, 512), "bfloat16", strides=(2048, 512, 1)),
        )
        for bad in bad_latents:
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(NativeWriterError):
                PreparedNativeWriter(bad, rope, ring, kernel_factory=RecordingKernel)
        for changed in (
            replace(ring, token_positions=DeviceTensor((4,), "int32")),
            replace(ring, token_valid=DeviceTensor((3,), "bool")),
            replace(ring, recent_rope=DeviceTensor((2, 256, 64), "float32")),
            replace(ring, token_ring_slots=DeviceTensor((4,), "int32", device="npu:1")),
        ):
            with self.assertRaises(NativeWriterError):
                PreparedNativeWriter(latent, rope, changed, kernel_factory=RecordingKernel)

    def test_fixed_metadata_and_persistent_ring_identity_before_launch(self):
        latent, rope, ring = buffers()
        kernel = RecordingKernel()
        writer = PreparedNativeWriter(latent, rope, ring, kernel_factory=lambda: kernel)
        with self.assertRaises(NativeWriterError):
            writer.write(DeviceTensor((4, 512), "bfloat16"), rope)
        with self.assertRaises(NativeWriterError):
            writer.write(latent, rope, replace(ring, token_valid=DeviceTensor((4,), "bool")))
        self.assertEqual(kernel.calls, [])

    def test_first_launch_must_precede_capture_and_warmed_launch_can_capture(self):
        latent, rope, ring = buffers()
        capture = [False]
        kernel = RecordingKernel()
        writer = PreparedNativeWriter(latent, rope, ring, kernel_factory=lambda: kernel,
                                      check_is_capture=lambda: capture[0])
        capture[0] = True
        with self.assertRaisesRegex(NativeWriterError, "warmed outside capture"):
            writer.write(latent, rope)
        self.assertEqual(kernel.calls, [])
        capture[0] = False
        writer.warmup(latent, rope)
        capture[0] = True
        writer.write(latent, rope)
        self.assertEqual(len(kernel.calls), 2)

    def test_source_gates_all_payload_and_tag_stores_and_has_no_cast_or_allocation(self):
        tree = ast.parse((ROOT / "framework_integration/sfa_recent128/native_writer.py").read_text())
        kernel = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "native_bf16_ring_copy")
        stores = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == "store"]
        self.assertEqual(len(stores), 4)
        for store in stores:
            self.assertTrue(any(keyword.arg == "mask" and isinstance(keyword.value, ast.Name)
                                and keyword.value.id == "writable" for keyword in store.keywords))
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & {"item", "cpu", "tolist", "numpy", "empty", "zeros", "clone", "copy_", "tensor"})
        # Only the bank-index arithmetic is widened; BF16 values are never cast.
        casts = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "to"]
        self.assertEqual(len(casts), 1)
        self.assertEqual(ast.unparse(casts[0].func.value), "tl.load(Banks + token)")


if __name__ == "__main__":
    unittest.main()
