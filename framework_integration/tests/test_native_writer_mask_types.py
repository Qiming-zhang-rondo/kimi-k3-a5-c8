"""Execute the real kernel body with strict host mask semantics, not Triton/A5.

Valid deliberately loads as a byte, matching the reported target-stack ABI.
NumPy supplies bit patterns and broadcasting; this does not compile a kernel.
"""
import ast
from pathlib import Path
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "sfa_recent128/native_writer.py"


class Value(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Value)


class Pointer:
    def __init__(self, storage, offset=0):
        self.storage = storage.reshape(-1)
        self.offset = np.asarray(offset, dtype=np.int64)

    def __add__(self, offset):
        return Pointer(self.storage, self.offset + offset)


class StrictTL:
    int64 = np.int64

    def __init__(self):
        self.token = 0
        self.mask_shapes = []

    def program_id(self, axis):
        assert axis == 0
        return self.token

    @staticmethod
    def arange(start, stop):
        return np.arange(start, stop).view(Value)

    def boolean(self, mask):
        mask = np.asarray(mask)
        assert mask.dtype == np.bool_, "mask/condition must have boolean element dtype"
        self.mask_shapes.append(mask.shape)
        return mask

    def where(self, condition, left, right):
        return np.where(self.boolean(condition), left, right).view(Value)

    def indices(self, pointer, mask):
        offsets, active = np.broadcast_arrays(pointer.offset, self.boolean(mask))
        assert np.all((offsets[active] >= 0) & (offsets[active] < pointer.storage.size)), "active pointer out of bounds"
        return offsets, active

    def load(self, pointer, mask=None, other=0):
        if mask is None:
            offsets = pointer.offset
            assert np.all((offsets >= 0) & (offsets < pointer.storage.size))
            return np.asarray(pointer.storage[offsets]).view(Value)
        offsets, active = self.indices(pointer, mask)
        value = np.full(offsets.shape, other, dtype=pointer.storage.dtype)
        value[active] = pointer.storage[offsets[active]]
        return value.view(Value)

    def store(self, pointer, value, mask):
        offsets, active = self.indices(pointer, mask)
        values = np.broadcast_to(value, offsets.shape)
        pointer.storage[offsets[active]] = values[active]


def kernel_body(tl, *, byte_mask_mutant=False):
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "native_bf16_ring_copy")
    function.decorator_list = []
    for argument in function.args.args:
        argument.annotation = None
    function.returns = None
    if byte_mask_mutant:
        assignment = next(node for node in function.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "valid"
                                  for target in node.targets))
        assignment.value = ast.parse("tl.load(Valid + token)", mode="eval").body
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"tl": tl}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[function.name]


class NativeWriterMaskTypeTests(unittest.TestCase):
    def run_body(self, *, mutant=False):
        tl = StrictTL()
        kernel = kernel_body(tl, byte_mask_mutant=mutant)
        tokens, pools, capacity = 8, 2, 256
        # Raw BF16 words, including signed zero and NaN encodings, are copied.
        source = (np.arange(tokens * 576, dtype=np.uint16) * 37).reshape(tokens, 576)
        source[0, :3] = (0, 0x8000, 0x7FC1)
        payloads = [np.full((pools, capacity, width), 0x55AA, np.uint16) for width in (512, 64)]
        tags = [np.full((pools, capacity), -911, np.int64) for _ in range(2)]
        banks = np.array([1, -9, -1, pools, 1, 1, 0, 0], np.int32)
        positions = np.array([255, -1, 3, 4, -1, 9, 256, 257], np.int64)
        epochs = np.array([10, -1, 12, 13, 14, -1, 15, 16], np.int64)
        valid = np.array([1, 0, 1, 1, 1, 1, 1, 1], np.uint8)
        expected = [array.copy() for array in payloads + tags]
        original_source = source.copy()
        original_routes = [array.copy() for array in (banks, positions, epochs, valid)]
        arguments = [Pointer(source), Pointer(source, 512)]
        arguments += [Pointer(array) for array in payloads + tags + [banks, positions, epochs, valid]]
        for token in range(tokens):
            tl.token = token
            kernel(*arguments, POOLS=pools, CAPACITY=capacity,
                   LATENT_ROW=576, LATENT_COL=1, ROPE_ROW=576, ROPE_COL=1)
            bank, position, epoch = banks[token], positions[token], epochs[token]
            if valid[token] and 0 <= bank < pools and position >= 0 and epoch >= 0:
                slot = position % capacity
                expected[0][bank, slot] = source[token, :512]
                expected[1][bank, slot] = source[token, 512:]
                expected[2][bank, slot] = position
                expected[3][bank, slot] = epoch
        for actual, reference in zip(payloads + tags, expected):
            np.testing.assert_array_equal(actual, reference)
        np.testing.assert_array_equal(source, original_source)
        for actual, reference in zip((banks, positions, epochs, valid), original_routes):
            np.testing.assert_array_equal(actual, reference)
        self.assertEqual(len(tl.mask_shapes), tokens * 7)
        # Scalar predicates must broadcast to the 512/64 element payloads.
        self.assertTrue(all(shape == () for shape in tl.mask_shapes))

    def test_byte_validity_becomes_boolean_for_all_kernel_masks(self):
        self.run_body()

    def test_old_byte_validity_expression_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "boolean element dtype"):
            self.run_body(mutant=True)


if __name__ == "__main__":
    unittest.main()
