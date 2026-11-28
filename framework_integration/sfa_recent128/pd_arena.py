"""A bounded, startup-registered NPU arena for immutable PD tail slots.

Construct before the transport's first peer link. Request publication and
receive use views of this same region, never newly registered clone pointers.
Registered backing allocations intentionally remain owned for process lifetime;
release() only returns a quiescent slot, and never unregisters the region.
The caller releases source slots only after the exact ACK/cancellation protocol
and destination slots only after actual import completion.
"""
import heapq
import threading

from .serving_state import LayerTail


_REGISTERED_ARENAS = []
_TOKEN_LIMIT = 128
_FIELDS = (("latent", 512, 2), ("rope", 64, 2),
           ("position_tags", 1, 8), ("epoch_tags", 1, 8))


def _require(value, message):
    if not value:
        raise RuntimeError(message)


class Arena:
    """One aligned registration; fixed slots with variable-length field views."""
    def __init__(self, engine, torch_module, device, layer_keys, slots,
                 alignment=2 * 1024 * 1024):
        _require(type(slots) is int and slots > 0, "PD arena slots must be positive")
        _require(type(alignment) is int and alignment >= 8 and
                 alignment & (alignment - 1) == 0,
                 "PD arena alignment must be a power of two >= 8")
        keys = tuple(layer_keys)
        _require(bool(keys) and all(type(key) in (str, int) and key != "" for key in keys),
                 "PD arena needs stable nonempty layer keys")
        _require(len(set(keys)) == len(keys), "PD arena layer keys must be unique")
        _require(callable(getattr(engine, "register_memory", None)),
                 "PD arena requires actual engine.register_memory")
        self.engine, self.torch = engine, torch_module
        self.device = torch_module.device(device)
        self.layer_keys, self.slot_count, self.alignment = keys, slots, alignment
        self._layer_bytes = sum(_TOKEN_LIMIT * width * element_bytes
                                for _, width, element_bytes in _FIELDS)
        self._slot_bytes = self._layer_bytes * len(keys)
        required = self._slot_bytes * slots
        registered_bytes = ((required + alignment - 1) // alignment) * alignment
        self._backing = torch_module.empty((registered_bytes + alignment - 1,),
                                          dtype=torch_module.uint8, device=self.device)
        _require(self._backing.is_contiguous() and self._backing.element_size() == 1,
                 "PD arena backing must be contiguous bytes")
        offset = (-self._backing.data_ptr()) % alignment
        self._bytes = self._backing.narrow(0, offset, registered_bytes)
        pointer = self._bytes.data_ptr()
        _require(pointer % alignment == 0 and self._bytes.numel() == registered_bytes,
                 "PD arena alignment/allocation mismatch")
        self.registered_extent = dict(pointer=pointer, nbytes=registered_bytes,
                                      alignment=alignment, slots=slots, layers=len(keys))
        self._lock = threading.RLock()
        self._free = list(range(slots))
        self._active = {}
        # Retain before invoking native code: even an exception or an unusual
        # failure return must not leave a possibly registered allocation freed.
        _REGISTERED_ARENAS.append(self)
        result = engine.register_memory(pointer, registered_bytes)
        self.registration_result = result
        _require(type(result) is int and result == 0,
                 "PD arena register_memory failed: " + repr(result))

    def contains(self, pointer, nbytes):
        extent = self.registered_extent
        return (type(pointer) is int and type(nbytes) is int and nbytes > 0 and
                extent["pointer"] <= pointer and
                pointer + nbytes <= extent["pointer"] + extent["nbytes"])

    def acquire(self, key, count):
        """Reserve one slot, or return its unchanged retry views.

        This reads only tensor metadata. It does not initialize payloads or
        create any additional memory registration.
        """
        _require(type(count) is int and 1 <= count <= _TOKEN_LIMIT,
                 "PD arena tail length must be in [1, 128]")
        with self._lock:
            previous = self._active.get(key)
            if previous is not None:
                _require(previous[1] == count, "PD arena retry changed its tail length")
                return previous[2]
            _require(bool(self._free), "PD arena slot pool exhausted; complete outstanding transfers")
            slot = self._free[0]
            layers = {}
            for layer_index, layer in enumerate(self.layer_keys):
                offset = slot * self._slot_bytes + layer_index * self._layer_bytes
                values = []
                for field, width, element_bytes in _FIELDS:
                    dtype = self.torch.bfloat16 if element_bytes == 2 else self.torch.int64
                    shape = (count, width) if field in ("latent", "rope") else (count,)
                    value = self._bytes.narrow(0, offset, count * width * element_bytes)
                    value = value.view(dtype).view(shape)
                    _require(value.is_contiguous() and
                             self.contains(value.data_ptr(), value.numel() * value.element_size()),
                             "PD arena field escaped its registered extent")
                    values.append(value)
                    offset += _TOKEN_LIMIT * width * element_bytes
                layers[layer] = LayerTail(*values)
            heapq.heappop(self._free)
            self._active[key] = (slot, count, layers)
            return layers

    def release(self, key):
        """Recycle after caller proves every read/write on the slot complete."""
        with self._lock:
            previous = self._active.pop(key, None)
            if previous is None:
                return False
            heapq.heappush(self._free, previous[0])
            return True
