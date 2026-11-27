"""Public handshake for a vendor-implemented PrologV3 BF16 ring producer.

This module contains no vendor kernel source and imports no device framework.
The proposed, explicit new variant is ``cloud_mla_prolog_v3_recent128``. Its
FUNCTION must declare ``recent128_abi_version = 1``; a module-level marker is
insufficient. This handshake alone does not establish NPU or graph support.

ABI1 preserves every original positional argument and keyword. It adds the nine
RingBinding field names below and ``recent_window=128``. The vendor produces
original post-RMSNorm/post-RoPE, pre-MXFP4 BF16 values, preserving the original
C4 cache bytes/quantizer and five query outputs. Host code checks only tensor
metadata; request IDs, positions, valid masks and tag contents stay on device.
"""
from dataclasses import dataclass, fields
import importlib

PROVIDER_NAME = "cloud_mla_prolog_v3_recent128"
PROVIDER_ABI = 1
RECENT_WINDOW = 128


class PrologProviderError(RuntimeError):
    """The explicit new producer contract is missing or inconsistent."""


@dataclass(frozen=True)
class RingBinding:
    recent_latent: object
    recent_rope: object
    recent_position_tags: object
    recent_epoch_tags: object
    token_ring_slots: object
    token_positions: object
    token_request_epochs: object
    token_valid: object
    max_query_per_request: int


RING_KEYWORDS = tuple(field.name for field in fields(RingBinding))
PROVIDER_KEYWORDS = RING_KEYWORDS + ("recent_window",)


def _require(condition, message):
    if not condition:
        raise PrologProviderError(message)


def _shape(tensor, name):
    try:
        result = tuple(tensor.shape)
    except (AttributeError, TypeError) as error:
        raise PrologProviderError(name + " must expose tensor shape metadata") from error
    _require(all(type(value) is int and value >= 0 for value in result),
             name + " requires concrete nonnegative shape dimensions")
    return result


def _tensor_metadata(tensor, name, expected_shape, dtype, device):
    _require(_shape(tensor, name) == expected_shape, name + " shape must be " + str(expected_shape))
    _require(str(getattr(tensor, "dtype", None)) == "torch." + dtype,
             name + " dtype must be torch." + dtype)
    _require(getattr(tensor, "device", None) == device, name + " must use the token_x device")
    contiguous = getattr(tensor, "is_contiguous", None)
    _require(callable(contiguous) and contiguous() is True, name + " must be contiguous")


def validate_ring_binding(token_x, ring):
    """Metadata-only validation; no allocation, host reads or scalar extraction.

    max_query_per_request is a configured bound, not a device-derived value.
    Ring capacity protects the earliest query against all writes in its chunk.
    Private kernels must additionally validate/predicate device IDs and padding.
    """
    _require(isinstance(ring, RingBinding), "ring must be a RingBinding")
    token_shape = _shape(token_x, "token_x")
    _require(len(token_shape) in (2, 3) and all(value > 0 for value in token_shape),
             "token_x must be nonempty [T,H] or [B,T,H]")
    token_rows = token_shape[0] if len(token_shape) == 2 else token_shape[0] * token_shape[1]
    device = getattr(token_x, "device", None)
    _require(device is not None, "token_x must expose device metadata")
    contiguous = getattr(token_x, "is_contiguous", None)
    _require(callable(contiguous) and contiguous() is True, "token_x must be contiguous")
    latent_shape = _shape(ring.recent_latent, "recent_latent")
    _require(len(latent_shape) == 3 and latent_shape[0] > 0 and latent_shape[1] > 0 and latent_shape[2] == 512,
             "recent_latent must be [P,C,512] with positive P/C")
    pools, capacity, _ = latent_shape
    bound = ring.max_query_per_request
    _require(type(bound) is int and bound > 0, "max_query_per_request must be a positive integer")
    _require(capacity % 128 == 0 and capacity >= RECENT_WINDOW + bound - 1,
             "ring C must be a multiple of 128 and >=128+max_query_per_request-1")
    shapes = {
        "recent_latent": ((pools, capacity, 512), "bfloat16"),
        "recent_rope": ((pools, capacity, 64), "bfloat16"),
        "recent_position_tags": ((pools, capacity), "int64"),
        "recent_epoch_tags": ((pools, capacity), "int64"),
        "token_ring_slots": ((token_rows,), "int32"),
        "token_positions": ((token_rows,), "int64"),
        "token_request_epochs": ((token_rows,), "int64"),
        "token_valid": ((token_rows,), "bool"),
    }
    for name, (shape, dtype) in shapes.items():
        _tensor_metadata(getattr(ring, name), name, shape, dtype, device)
    return dict(abi_version=PROVIDER_ABI, provider=PROVIDER_NAME,
        token_rows=token_rows, request_pool_size=pools, ring_capacity=capacity,
        max_query_per_request=bound, recent_window=RECENT_WINDOW,
        validation="shape/dtype/device/contiguity only; device contents remain vendor responsibility")


def call_prolog_with_ring(original, args, kwargs, ring=None, module=None):
    """OFF passes straight through; ON requires the exact new vendor variant.

    No existing operator is relabelled as the new producer. There is no fallback
    to C4 dequantization, NATIVE projection or a separate attention operation.
    Return the vendor's original five-element tuple/list object unchanged.
    """
    if ring is None:
        return original(*args, **kwargs)
    _require(isinstance(args, (tuple, list)) and len(args) >= 9,
             "ON requires the original PrologV3 positional arguments (at least nine)")
    _require(isinstance(kwargs, dict), "original PrologV3 kwargs must be a dictionary")
    conflicts = set(PROVIDER_KEYWORDS).intersection(kwargs)
    _require(not conflicts, "Recent128 keyword collision: " + ", ".join(sorted(conflicts)))
    validate_ring_binding(args[0], ring)
    if module is None:
        module = importlib.import_module("cloud_ops_turbo")
    provider = getattr(module, PROVIDER_NAME, None)
    _require(callable(provider), "Vendor lacks explicit new provider " + PROVIDER_NAME)
    marker = getattr(provider, "recent128_abi_version", None)
    _require(type(marker) is int and marker == PROVIDER_ABI,
             PROVIDER_NAME + ".recent128_abi_version must be integer 1")
    extended = dict(kwargs)
    extended.update({name: getattr(ring, name) for name in RING_KEYWORDS})
    extended["recent_window"] = RECENT_WINDOW
    result = provider(*args, **extended)
    _require(isinstance(result, (tuple, list)) and len(result) == 5,
             "Recent128 provider must return exactly five original query outputs")
    return result
