"""Standalone CPU-only support for hot-attention/merge source-body tests.

No production attention implementation, Torch or Triton is imported here.
The byte decoder is an independent scalar format oracle. The tiny language
emulator checks arithmetic/masked addressing, not Ascend compilation or UB.
"""

import math
import struct

import numpy as np


class TensorMetadata:
    def __init__(self, shape, dtype, device="npu:0", contiguous=True):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.contiguous, self.requires_grad = contiguous, False

    def is_contiguous(self):
        return self.contiguous

    def item(self):
        raise AssertionError("Host synchronization is forbidden")

    def cpu(self):
        raise AssertionError("Device copies are forbidden")


def _decode_fp8_byte(byte):
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    value = (math.ldexp(float(mantissa), -9) if exponent == 0 else
             math.ldexp(1.0+mantissa/8.0, exponent-7))
    if exponent == 15 and mantissa == 7:
        value = math.nan
    return math.copysign(value, -1.0 if byte & 128 else 1.0)


_FP8_VALUES = np.asarray([_decode_fp8_byte(byte) for byte in range(256)], np.float32)


def decode_c8_token(raw, rope_dim=64):
    """Scalar E4M3FN / original BF16 RoPE / FP32 scale-per-128 oracle."""
    raw = bytes(raw)
    if rope_dim not in (0, 64) or len(raw) != 528+2*rope_dim:
        raise ValueError("Incorrect C8 token width or RoPE dimension")
    scales = struct.unpack("<4f", raw[512+2*rope_dim:])
    latent = []
    for dim, byte in enumerate(raw[:512]):
        latent.append(_decode_fp8_byte(byte) * scales[dim//128])
    rope = [struct.unpack("<f", b"\x00\x00"+raw[512+2*d:514+2*d])[0] for d in range(rope_dim)]
    return latent, rope


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
        elif dtype == "fp8_e4m3fn":
            if not bitcast or value.dtype != np.uint8:
                raise AssertionError("E4M3FN bitcast requires original uint8 bytes")
            result = _FP8_VALUES[value]
        else:
            result = value.view(dtype) if bitcast else value.astype(dtype)
        return _Tensor(result)


class _TL:
    constexpr = object
    float32, int32, int64 = np.float32, np.int32, np.int64
    uint8, uint16, uint32 = np.uint8, np.uint16, np.uint32
    bfloat16 = "bf16"
    float8e4nv = "fp8_e4m3fn"
    range = staticmethod(range)

    def __init__(self):
        self.pid = (0, 0)
        self.loads, self.dots = [], []

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
    def maximum(left, right):
        return _Tensor(np.maximum(left, right))

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

    def dot(self, left, right):
        self.dots.append((left.shape, right.shape))
        return _Tensor(np.matmul(left, right))

    @staticmethod
    def trans(value):
        return _Tensor(np.transpose(value))
