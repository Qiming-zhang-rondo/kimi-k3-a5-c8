"""Versioned BF16 PD sidecar over an injected registered-memory transport.

This module is a connector boundary, never a forward/Graph operation. It does
not discover a Mooncake session, serialize tensor payloads, or infer success
from the arrival of C4. The caller supplies the real transport engine, source
device validation/barrier, and destination stream-ordering completion.

``validate_snapshot(request_id, snapshot)`` must return literal ``True`` only
after checking *every* device position/epoch tag against the interval/source
epoch and making snapshot writes visible to remote reads. It runs outside
Graph and may synchronize there. The core checks metadata without reading
tensor contents. A completion factory must return an actual backend event
with ``wait()``; there is deliberately no default success event.
"""
from dataclasses import dataclass
import secrets

from .serving_state import LayerTail, TailSnapshot


PD_TAIL_VERSION = 1
_U64_MAX = (1 << 64) - 1
_I64_MAX = (1 << 63) - 1
_FIELDS = ("latent", "rope", "position_tags", "epoch_tags")
_DTYPES = ("bfloat16", "bfloat16", "int64", "int64")
_WIDTHS = (512, 64, None, None)


class PDTailError(RuntimeError):
    pass


def _check(condition, message):
    if not condition:
        raise PDTailError(message)


def _integer(value, name, minimum=0, maximum=_U64_MAX):
    _check(type(value) is int and minimum <= value <= maximum,
           name + " must be an integer in the supported range")
    return value


def _identity(value, name):
    _check(type(value) in (str, int) and value != "", name + " must be a stable string or integer")
    return value


def _nonce(value):
    _check(type(value) is str and len(value) == 32 and
           all(c in "0123456789abcdef" for c in value) and int(value, 16) != 0,
           "nonce must be a nonzero 128-bit lowercase hexadecimal string")
    return value


def _result(value, operation):
    _check(type(value) is int and value == 0, operation + " failed: " + repr(value))


def _wait(completion):
    _check(completion is not None and callable(getattr(completion, "wait", None)),
           "An actual stream-ordering completion with wait() is required")
    completion.wait()


def _shape(count, field_index):
    width = _WIDTHS[field_index]
    return (count,) if width is None else (count, width)


@dataclass(frozen=True)
class TensorDescriptor:
    layer: object
    field: str
    dtype: str
    shape: tuple
    nbytes: int
    pointer: int

    def to_dict(self):
        return dict(layer=self.layer, field=self.field, dtype=self.dtype,
                    shape=list(self.shape), nbytes=self.nbytes, pointer=self.pointer)

    @classmethod
    def from_dict(cls, data):
        _check(type(data) is dict and set(data) ==
               {"layer", "field", "dtype", "shape", "nbytes", "pointer"},
               "Invalid tensor descriptor schema")
        _check(type(data["shape"]) in (list, tuple), "shape must be an array")
        return cls(data["layer"], data["field"], data["dtype"],
                   tuple(data["shape"]), data["nbytes"], data["pointer"])


@dataclass(frozen=True)
class TailDescriptor:
    version: int
    request_id: object
    source_epoch: int
    start: int
    end: int
    rank: int
    generation: int
    nonce: str
    buffers: tuple

    def validate(self):
        _check(type(self.version) is int and self.version == PD_TAIL_VERSION,
               "Unsupported PD tail descriptor version")
        _identity(self.request_id, "request_id")
        _integer(self.source_epoch, "source_epoch", 1, _I64_MAX)
        _integer(self.start, "start", 0, _I64_MAX)
        _integer(self.end, "end", 1, _I64_MAX)
        _check(self.start == max(0, self.end - 128), "Descriptor must cover exactly the committed BF16 tail")
        _integer(self.rank, "rank")
        _integer(self.generation, "generation", 1)
        _nonce(self.nonce)
        _check(type(self.buffers) is tuple and bool(self.buffers), "Descriptor has no layer buffers")
        groups = {}
        for buffer in self.buffers:
            _check(isinstance(buffer, TensorDescriptor), "Invalid tensor descriptor")
            _identity(buffer.layer, "layer")
            _check(buffer.field in _FIELDS, "Unknown BF16 sidecar field")
            index = _FIELDS.index(buffer.field)
            _check(type(buffer.shape) is tuple and
                   all(type(v) is int for v in buffer.shape) and
                   buffer.shape == _shape(self.end - self.start, index),
                   "Invalid BF16 sidecar shape")
            _check(buffer.dtype == _DTYPES[index], "Invalid BF16 sidecar dtype")
            elements = 1
            for extent in buffer.shape:
                elements *= extent
            _integer(buffer.nbytes, "nbytes", 1)
            _check(buffer.nbytes == elements * (2 if index < 2 else 8), "Tensor byte length mismatch")
            _integer(buffer.pointer, "64-bit pointer", 1)
            _check(buffer.pointer + buffer.nbytes - 1 <= _U64_MAX, "64-bit address range overflow")
            fields = groups.setdefault(buffer.layer, set())
            _check(buffer.field not in fields, "Duplicate layer field")
            fields.add(buffer.field)
        _check(all(fields == set(_FIELDS) for fields in groups.values()), "Every layer requires all four BF16 sidecar fields")
        return self

    @property
    def layer_keys(self):
        return frozenset(buffer.layer for buffer in self.buffers)

    def to_dict(self):
        self.validate()
        return dict(version=self.version, request_id=self.request_id,
                    source_epoch=self.source_epoch, start=self.start, end=self.end,
                    rank=self.rank, generation=self.generation, nonce=self.nonce,
                    buffers=[buffer.to_dict() for buffer in self.buffers])

    @classmethod
    def from_dict(cls, data):
        keys = {"version", "request_id", "source_epoch", "start", "end", "rank", "generation", "nonce", "buffers"}
        _check(type(data) is dict and set(data) == keys, "Invalid PD tail descriptor schema")
        _check(type(data["buffers"]) in (list, tuple), "buffers must be an array")
        values = {key: value for key, value in data.items() if key != "buffers"}
        return cls(**values, buffers=tuple(TensorDescriptor.from_dict(item)
                                           for item in data["buffers"])).validate()


def _snapshot_metadata(snapshot, expected_layers=None):
    _check(isinstance(snapshot, TailSnapshot), "Expected an actual BF16 TailSnapshot")
    _integer(snapshot.source_epoch, "source_epoch", 1, _I64_MAX)
    _integer(snapshot.end, "end", 1, _I64_MAX)
    _integer(snapshot.start, "start", 0, _I64_MAX)
    _check(snapshot.start == max(0, snapshot.end - 128), "Snapshot must cover exactly the committed BF16 tail")
    _check(bool(snapshot.layers), "Snapshot has no layers")
    if expected_layers is not None:
        _check(set(snapshot.layers) == set(expected_layers), "Snapshot layer set mismatch")
    device = None
    for layer, tail in snapshot.layers.items():
        _identity(layer, "layer")
        for index, field in enumerate(_FIELDS):
            tensor = getattr(tail, field, None)
            _check(tensor is not None, "Missing sidecar tensor")
            _check(tuple(tensor.shape) == _shape(snapshot.end - snapshot.start, index), "Snapshot shape mismatch")
            _check(str(tensor.dtype) == "torch." + _DTYPES[index], "Snapshot dtype mismatch")
            _check(tensor.is_contiguous(), "Snapshot tensors must be contiguous")
            device = tensor.device if device is None else device
            _check(tensor.device == device, "Snapshot tensors must share the actual source device")
    return snapshot


@dataclass
class _Publication:
    descriptor: TailDescriptor
    snapshot: TailSnapshot
    registered: list


class PDSidecarPublisher:
    """Own private immutable clones and registrations until an exact nonce ACK.

    Re-send ``descriptor_for`` on retry; publishing again while an ACK is
    outstanding is rejected. If an engine cannot unregister, retired tensor
    owners remain alive for the publisher lifetime to avoid dangling MR entries.
    Keep this owner alive until its engine has been shut down.
    """
    def __init__(self, engine, local_session_id, *, validate_snapshot, rank=0,
                 expected_layers=None):
        _check(callable(validate_snapshot), "A real device snapshot validator/barrier is required")
        _identity(local_session_id, "local_session_id")
        _integer(rank, "rank")
        self.engine = engine
        self.local_session_id = local_session_id
        self.validate_snapshot = validate_snapshot
        self.rank = rank
        self.expected_layers = None if expected_layers is None else frozenset(expected_layers)
        self._active = {}
        self._generations = {}
        self._versions = {}
        self._retired = []

    def publish(self, request_id, snapshot, *, expected_source_epoch=None):
        _identity(request_id, "request_id")
        _check(request_id not in self._active, "Request already has an unacknowledged sidecar; retry its descriptor")
        _snapshot_metadata(snapshot, self.expected_layers)
        if expected_source_epoch is not None:
            _check(snapshot.source_epoch == expected_source_epoch, "Source request epoch mismatch")
        previous = self._versions.get(request_id)
        if previous is not None:
            old_epoch, old_end = previous
            _check(snapshot.source_epoch >= old_epoch, "Source request epoch cannot roll back")
            _check(snapshot.source_epoch != old_epoch or snapshot.end >= old_end,
                   "Source committed interval cannot roll back in the same epoch")
        if snapshot.completion is not None:
            _wait(snapshot.completion)
        # Source ring reuse/append cannot alter the registered publication.
        layers = {key: LayerTail(*(getattr(tail, field).clone() for field in _FIELDS))
                  for key, tail in snapshot.layers.items()}
        owned = TailSnapshot(snapshot.start, snapshot.end, snapshot.source_epoch, layers)
        _check(self.validate_snapshot(request_id, owned) is True,
               "Device position/epoch validation and publication barrier did not succeed")
        _snapshot_metadata(owned, self.expected_layers)
        generation = self._generations.get(request_id, 0) + 1
        nonce = secrets.token_hex(16)
        while int(nonce, 16) == 0:
            nonce = secrets.token_hex(16)
        buffers = tuple(TensorDescriptor(key, field, _DTYPES[index], tuple(tensor.shape),
                         tensor.numel() * tensor.element_size(), tensor.data_ptr())
                        for key, tail in owned.layers.items()
                        for index, field in enumerate(_FIELDS)
                        for tensor in (getattr(tail, field),))
        descriptor = TailDescriptor(PD_TAIL_VERSION, request_id, owned.source_epoch,
                                    owned.start, owned.end, self.rank, generation,
                                    nonce, buffers).validate()
        publication = _Publication(descriptor, owned, [])
        try:
            for buffer in buffers:
                _result(self.engine.register_memory(buffer.pointer, buffer.nbytes), "register_memory")
                publication.registered.append(buffer.pointer)
        except BaseException:
            self._retire(publication)
            raise
        self._generations[request_id] = generation
        self._versions[request_id] = (owned.source_epoch, owned.end)
        self._active[request_id] = publication
        return descriptor

    def descriptor_for(self, request_id, nonce):
        publication = self._active.get(request_id)
        _check(publication is not None and publication.descriptor.nonce == nonce,
               "Unknown or stale publication nonce")
        return publication.descriptor

    def _retire(self, publication):
        unregister = getattr(self.engine, "unregister_memory", None)
        if not callable(unregister):
            if publication.registered and not any(item is publication for item in self._retired):
                self._retired.append(publication)
            return
        try:
            while publication.registered:
                _result(unregister(publication.registered[-1]), "unregister_memory")
                publication.registered.pop()
            self._retired[:] = [item for item in self._retired if item is not publication]
        except BaseException:
            # Keep owners alive even if deregistration fails during cleanup.
            if not any(item is publication for item in self._retired):
                self._retired.append(publication)
            raise

    def acknowledge(self, request_id, nonce):
        _nonce(nonce)
        publication = self._active.get(request_id)
        if publication is None or publication.descriptor.nonce != nonce:
            return False
        self._retire(publication)
        del self._active[request_id]
        return True

    def cancel(self, request_id, nonce, *, remote_read_stopped=False):
        _check(remote_read_stopped is True, "Cancellation requires proof that remote reads have stopped")
        return self.acknowledge(request_id, nonce)


class PDSidecarReceiver:
    """Allocate actual destination tensors and perform registered-memory reads.

    The engine's synchronous result alone never becomes a fabricated device
    event. ``completion_factory(descriptor, snapshot)`` supplies the backend's
    real receive visibility/stream dependency after the synchronous transfer.
    Exact retries return the same owned snapshot; older generations fail.
    """
    def __init__(self, engine, remote_session_id, *, device, completion_factory,
                 torch_module=None, validate_received=None, expected_rank=None):
        if torch_module is None:
            import torch as torch_module
        _identity(remote_session_id, "remote_session_id")
        _check(callable(completion_factory), "A real receive completion factory is required")
        _check(validate_received is None or callable(validate_received), "Invalid receive validator")
        if expected_rank is not None:
            _integer(expected_rank, "expected_rank")
        self.engine, self.remote_session_id = engine, remote_session_id
        self.torch, self.device = torch_module, torch_module.device(device)
        self.completion_factory = completion_factory
        self.validate_received, self.expected_rank = validate_received, expected_rank
        self._received = {}
        self._imported = {}
        self._retired = []

    def receive(self, descriptor, *, state, request_id=None, expected_source_epoch=None):
        if type(descriptor) is dict:
            descriptor = TailDescriptor.from_dict(descriptor)
        _check(isinstance(descriptor, TailDescriptor), "Expected a PD tail descriptor")
        descriptor.validate()
        request_id = descriptor.request_id if request_id is None else request_id
        _check(request_id == descriptor.request_id and type(request_id) is type(descriptor.request_id),
               "Descriptor request identity mismatch")
        if expected_source_epoch is not None:
            _check(descriptor.source_epoch == expected_source_epoch, "Stale source epoch")
        if self.expected_rank is not None:
            _check(descriptor.rank == self.expected_rank, "Descriptor rank mismatch")
        _check(set(state.layers) == descriptor.layer_keys, "Descriptor must cover every local attention layer")
        _check(state.device == self.device, "Receiver and serving state device mismatch")
        key = (request_id, descriptor.rank)
        previous = self._received.get(key)
        if previous is not None:
            old, snapshot = previous
            old_version = (old.source_epoch, old.generation)
            new_version = (descriptor.source_epoch, descriptor.generation)
            _check(new_version >= old_version, "Stale tail generation")
            if new_version == old_version:
                _check(descriptor == old, "Generation retry changed nonce or descriptor")
                _check(snapshot is not None, "Received payload was released; a fresh publication is required")
                return snapshot
            _check(descriptor.source_epoch != old.source_epoch or descriptor.end >= old.end,
                   "Tail generation cannot roll back a committed source interval")
        layers, arrays, registered = {}, [], []
        for layer in descriptor.layer_keys:
            fields = {}
            for buffer in descriptor.buffers:
                if buffer.layer == layer:
                    tensor = self.torch.empty(buffer.shape, dtype=getattr(self.torch, buffer.dtype), device=self.device)
                    _check(tensor.is_contiguous() and tensor.numel() * tensor.element_size() == buffer.nbytes,
                           "Destination allocation does not match descriptor")
                    fields[buffer.field] = tensor
            layers[layer] = LayerTail(*(fields[field] for field in _FIELDS))
        arrays = [getattr(layers[buffer.layer], buffer.field) for buffer in descriptor.buffers]
        snapshot = TailSnapshot(descriptor.start, descriptor.end, descriptor.source_epoch, layers)
        publication = _Publication(descriptor, snapshot, registered)
        try:
            for tensor, buffer in zip(arrays, descriptor.buffers):
                _result(self.engine.register_memory(tensor.data_ptr(), buffer.nbytes), "register_memory")
                registered.append(tensor.data_ptr())
            _result(self.engine.batch_transfer_sync_read(self.remote_session_id,
                    [buffer.pointer for buffer in descriptor.buffers],
                    [tensor.data_ptr() for tensor in arrays],
                    [buffer.nbytes for buffer in descriptor.buffers]), "batch_transfer_sync_read")
            completion = self.completion_factory(descriptor, snapshot)
            _wait(completion)
            snapshot = TailSnapshot(descriptor.start, descriptor.end, descriptor.source_epoch, layers, completion)
            publication.snapshot = snapshot
            if self.validate_received is not None:
                _check(self.validate_received(request_id, snapshot) is True, "Received device tail validation failed")
        finally:
            PDSidecarPublisher._retire(self, publication)
        self._received[key] = (descriptor, snapshot)
        return snapshot

    def receive_and_import(self, descriptor, *, state, request_id=None, expected_source_epoch=None):
        if type(descriptor) is dict:
            descriptor = TailDescriptor.from_dict(descriptor)
        _check(isinstance(descriptor, TailDescriptor), "Expected a PD tail descriptor")
        descriptor.validate()
        target_id = descriptor.request_id if request_id is None else request_id
        before = state.requests.get(target_id)
        imported = self._imported.get((target_id, descriptor.rank))
        if imported is not None and imported[0] == descriptor:
            _check(before is imported[1], "Destination request generation changed since the previous import")
        if before is not None:
            _check(before.committed_end == descriptor.end, "Destination request boundary mismatch")
        snapshot = self.receive(descriptor, state=state, request_id=target_id,
                                expected_source_epoch=expected_source_epoch)
        _check(state.requests.get(target_id) is before, "Destination request generation changed during transfer")
        lease = state.acquire_request(target_id, initial_position=snapshot.end)
        try:
            state.import_tail(target_id, snapshot, source="pd", completion=snapshot.completion)
        except BaseException:
            if before is None:
                state.release_request(target_id)
            else:
                lease.tail_ready = False
            raise
        self._imported[(target_id, descriptor.rank)] = (descriptor, lease)
        return lease

    def release_received(self, request_id, nonce, *, rank=0):
        """Drop a copied payload after import; retain its anti-replay watermark.

        The caller must have ordered import copies on the consuming stream.
        This does not ACK the publisher: send that exact nonce separately only
        after successful receive/import. An old descriptor cannot be reread
        after its source memory has been acknowledged and released.
        """
        _nonce(nonce)
        key = (request_id, rank)
        previous = self._received.get(key)
        if previous is None or previous[0].nonce != nonce:
            return False
        descriptor, snapshot = previous
        if snapshot is None:
            return False
        self._received[key] = (descriptor, None)
        return True
