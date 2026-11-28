"""Host contract tests: actual address-to-address bytes, no device claims."""
import ctypes
from dataclasses import replace
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework_integration.sfa_recent128.pd_tail import (
    PDTailError, PDSidecarPublisher, PDSidecarReceiver, TailDescriptor,
)
from framework_integration.sfa_recent128.serving_state import LayerTail, TailSnapshot


class Tensor:
    """A contiguous, independently owned byte allocation with torch metadata."""
    def __init__(self, shape, dtype, device="cpu"):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.storage = ctypes.create_string_buffer(self.numel() * self.element_size())

    def numel(self):
        count = 1
        for size in self.shape:
            count *= size
        return count

    def element_size(self):
        return 2 if self.dtype == "torch.bfloat16" else 8

    def data_ptr(self):
        return ctypes.addressof(self.storage)

    def is_contiguous(self):
        return True

    def clone(self):
        result = Tensor(self.shape, self.dtype, self.device)
        ctypes.memmove(result.data_ptr(), self.data_ptr(), self.numel() * self.element_size())
        return result

    def raw(self):
        return ctypes.string_at(self.data_ptr(), self.numel() * self.element_size())

    def fill_bytes(self, value):
        ctypes.memset(self.data_ptr(), value, self.numel() * self.element_size())

    def set_tags(self, values):
        raw = struct.pack("<" + "q" * len(values), *values)
        ctypes.memmove(self.data_ptr(), raw, len(raw))

    def tags(self):
        return struct.unpack("<" + "q" * self.numel(), self.raw())

    def cpu(self):
        raise AssertionError("payload CPU serialization is prohibited")


class Torch:
    bfloat16, int64 = "torch.bfloat16", "torch.int64"

    @staticmethod
    def device(value):
        return value

    @staticmethod
    def empty(shape, *, dtype, device):
        return Tensor(shape, dtype, device)


class Event:
    def __init__(self, calls=None):
        self.waits, self.calls = 0, calls

    def wait(self):
        self.waits += 1
        if self.calls is not None:
            self.calls.append("wait")


class Engine:
    def __init__(self):
        self.registered = {}
        self.reads = []
        self.register_rc = self.read_rc = self.unregister_rc = 0

    def register_memory(self, pointer, size):
        if self.register_rc:
            return self.register_rc
        self.registered[pointer] = size
        return 0

    def unregister_memory(self, pointer):
        if self.unregister_rc:
            return self.unregister_rc
        del self.registered[pointer]
        return 0

    def batch_transfer_sync_read(self, session, sources, destinations, lengths):
        self.reads.append((session, sources, destinations, lengths))
        if self.read_rc:
            return self.read_rc
        for source, destination, length in zip(sources, destinations, lengths):
            assert self.registered[source] == length
            assert self.registered[destination] == length
            ctypes.memmove(destination, source, length)
        return 0


class State:
    def __init__(self, keys=("L0", "L1")):
        self.layers = dict.fromkeys(keys)
        self.device = "cpu"
        self.requests = {}
        self.imported = None
        self.next_epoch = 10
        self.fail_import = False

    def acquire_request(self, request_id, *, initial_position):
        if request_id not in self.requests:
            self.next_epoch += 1
            self.requests[request_id] = SimpleNamespace(
                committed_end=initial_position, epoch=self.next_epoch,
                tail_ready=False, tail_source="tail_import_required")
        return self.requests[request_id]

    def import_tail(self, request_id, snapshot, *, source, completion):
        completion.wait()
        if self.fail_import:
            raise RuntimeError("import failed")
        assert source == "pd"
        self.imported = snapshot
        lease = self.requests[request_id]
        lease.tail_ready, lease.tail_source = True, source

    def release_request(self, request_id):
        self.requests.pop(request_id)


def snapshot(end=135, epoch=7, keys=("L0", "L1"), completion=None):
    start = max(0, end - 128)
    count = end - start
    layers = {}
    for offset, key in enumerate(keys):
        latent = Tensor((count, 512), Torch.bfloat16)
        rope = Tensor((count, 64), Torch.bfloat16)
        positions = Tensor((count,), Torch.int64)
        epochs = Tensor((count,), Torch.int64)
        latent.fill_bytes(31 + offset)
        rope.fill_bytes(57 + offset)
        positions.set_tags(list(range(start, end)))
        epochs.set_tags([epoch] * count)
        layers[key] = LayerTail(latent, rope, positions, epochs)
    return TailSnapshot(start, end, epoch, layers, completion)


def validate(request_id, value):
    assert request_id == "req"
    return all(tail.position_tags.tags() == tuple(range(value.start, value.end)) and
               tail.epoch_tags.tags() == (value.source_epoch,) * (value.end - value.start)
               for tail in value.layers.values())


class PDTailTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.state = Engine(), State()
        self.publisher = PDSidecarPublisher(self.engine, "source", validate_snapshot=validate,
                                            expected_layers=self.state.layers)
        self.events = []

        def complete(descriptor, received):
            self.assertEqual(len(self.engine.reads), 1)
            event = Event()
            self.events.append(event)
            return event

        self.receiver = PDSidecarReceiver(self.engine, "source", device="cpu",
            torch_module=Torch, completion_factory=complete, validate_received=validate,
            expected_rank=0)

    def publish(self, value=None):
        return self.publisher.publish("req", snapshot() if value is None else value,
                                      expected_source_epoch=7)

    def test_real_address_byte_copy_roundtrip_and_import(self):
        original = snapshot()
        descriptor = self.publish(original)
        self.assertEqual(TailDescriptor.from_dict(descriptor.to_dict()), descriptor)
        result = self.receiver.receive(descriptor, state=self.state, expected_source_epoch=7)
        for layer in original.layers:
            for field in ("latent", "rope", "position_tags", "epoch_tags"):
                source = getattr(original.layers[layer], field)
                target = getattr(result.layers[layer], field)
                self.assertEqual(target.raw(), source.raw())
                self.assertNotEqual(source.data_ptr(), target.data_ptr())
        self.assertEqual(self.events[0].waits, 1)
        lease = self.receiver.receive_and_import(descriptor, state=self.state, expected_source_epoch=7)
        self.assertTrue(lease.tail_ready)
        self.assertEqual(lease.tail_source, "pd")
        self.assertEqual(len(self.engine.reads), 1)
        self.assertEqual(self.events[0].waits, 2)

    def test_source_is_private_immutable_clone_and_survives_request_reuse(self):
        original = snapshot()
        descriptor = self.publish(original)
        expected = original.layers["L0"].latent.raw()
        original.layers["L0"].latent.fill_bytes(250)
        del original
        result = self.receiver.receive(descriptor, state=self.state)
        self.assertEqual(result.layers["L0"].latent.raw(), expected)
        self.assertEqual(len(self.engine.registered), 8)
        self.assertTrue(self.publisher.acknowledge("req", descriptor.nonce))
        self.assertFalse(self.engine.registered)
        self.assertFalse(self.publisher.acknowledge("req", descriptor.nonce))

    def test_source_completion_and_all_layer_tag_validation_before_registration(self):
        event = Event()
        invalid = snapshot(completion=event)
        invalid.layers["L1"].position_tags.set_tags([0] * 128)
        with self.assertRaisesRegex(PDTailError, "validation"):
            self.publish(invalid)
        self.assertEqual(event.waits, 1)
        self.assertFalse(self.engine.registered)
        invalid = snapshot()
        invalid.layers["L1"].epoch_tags.set_tags([6] * 128)
        with self.assertRaisesRegex(PDTailError, "validation"):
            self.publish(invalid)
        self.assertFalse(self.engine.registered)

    def test_publish_requires_validator_and_expected_metadata(self):
        with self.assertRaises(PDTailError):
            PDSidecarPublisher(self.engine, "source", validate_snapshot=None)
        with self.assertRaisesRegex(PDTailError, "epoch"):
            self.publish(snapshot(epoch=8))
        with self.assertRaisesRegex(PDTailError, "layer set"):
            self.publish(snapshot(keys=("L0",)))
        invalid = snapshot()
        invalid.layers["L0"].latent.dtype = Torch.int64
        with self.assertRaisesRegex(PDTailError, "dtype"):
            self.publish(invalid)

    def test_retry_nonce_ack_and_generation(self):
        first = self.publish()
        self.assertIs(self.publisher.descriptor_for("req", first.nonce), first)
        with self.assertRaisesRegex(PDTailError, "unacknowledged"):
            self.publish()
        self.assertFalse(self.publisher.acknowledge("req", "1" * 32))
        self.assertEqual(len(self.engine.registered), 8)
        received = self.receiver.receive(first, state=self.state)
        self.assertIs(self.receiver.receive(first.to_dict(), state=self.state), received)
        with self.assertRaisesRegex(PDTailError, "retry changed"):
            self.receiver.receive(replace(first, nonce="1" * 32), state=self.state)
        self.publisher.acknowledge("req", first.nonce)
        second = self.publish(snapshot(end=136))
        self.assertEqual(second.generation, first.generation + 1)
        self.assertNotEqual(first.nonce, second.nonce)
        self.receiver.completion_factory = lambda *_: Event()
        self.receiver.receive(second, state=self.state)
        with self.assertRaisesRegex(PDTailError, "Stale tail generation"):
            self.receiver.receive(first, state=self.state)
        self.assertFalse(self.publisher.acknowledge("req", first.nonce))
        self.assertEqual(len(self.engine.registered), 8)

    def test_receiver_rejects_identity_epoch_rank_layers_device_before_io(self):
        descriptor = self.publish()
        cases = [dict(request_id="wrong"), dict(expected_source_epoch=8)]
        for kwargs in cases:
            with self.assertRaises(PDTailError):
                self.receiver.receive(descriptor, state=self.state, **kwargs)
        with self.assertRaises(PDTailError):
            self.receiver.receive(replace(descriptor, rank=1), state=self.state)
        with self.assertRaises(PDTailError):
            self.receiver.receive(descriptor, state=State(keys=("L0",)))
        self.state.device = "npu:0"
        with self.assertRaises(PDTailError):
            self.receiver.receive(descriptor, state=self.state)
        self.assertFalse(self.engine.reads)

    def test_schema_rejects_truncation_missing_or_extra_fields_and_nonzero_nonce(self):
        descriptor = self.publish()
        for changes in (dict(version=2), dict(nonce="0" * 32), dict(source_epoch=0),
                        dict(start=8), dict(generation=0), dict(rank=True),
                        dict(buffers=descriptor.buffers[:-1])):
            with self.assertRaises(PDTailError):
                replace(descriptor, **changes).validate()
        for changes in (dict(pointer=1 << 64), dict(pointer=-1), dict(nbytes=1),
                        dict(shape=(128, 511)), dict(dtype="float16")):
            with self.assertRaises(PDTailError):
                replace(descriptor, buffers=(replace(descriptor.buffers[0], **changes),) + descriptor.buffers[1:]).validate()
        for key in ("payload", "version"):
            data = descriptor.to_dict()
            if key == "payload":
                data[key] = "forbidden"
            else:
                del data[key]
            with self.assertRaises(PDTailError):
                TailDescriptor.from_dict(data)

    def test_transfer_failure_has_no_ready_request_no_completion_and_can_retry(self):
        descriptor = self.publish()
        self.engine.read_rc = -1
        with self.assertRaisesRegex(PDTailError, "sync_read failed"):
            self.receiver.receive_and_import(descriptor, state=self.state)
        self.assertFalse(self.state.requests)
        self.assertFalse(self.events)
        self.assertEqual(len(self.engine.registered), 8)
        self.engine.read_rc = 0
        self.receiver.completion_factory = lambda *_: Event()
        self.assertTrue(self.receiver.receive_and_import(descriptor, state=self.state).tail_ready)

    def test_missing_completion_fails_closed_and_destination_registration_is_cleaned(self):
        descriptor = self.publish()
        self.receiver.completion_factory = lambda *_: None
        with self.assertRaisesRegex(PDTailError, "actual stream-ordering"):
            self.receiver.receive_and_import(descriptor, state=self.state)
        self.assertFalse(self.state.requests)
        self.assertEqual(len(self.engine.registered), 8)

    def test_import_failure_releases_new_lease_and_existing_lease_is_not_ready(self):
        descriptor = self.publish()
        self.state.fail_import = True
        with self.assertRaisesRegex(RuntimeError, "import failed"):
            self.receiver.receive_and_import(descriptor, state=self.state)
        self.assertFalse(self.state.requests)
        lease = self.state.acquire_request("req", initial_position=descriptor.end)
        lease.tail_ready = True
        with self.assertRaises(RuntimeError):
            self.receiver.receive_and_import(descriptor, state=self.state)
        self.assertFalse(lease.tail_ready)

    def test_destination_lease_aba_during_transfer_is_rejected(self):
        descriptor = self.publish()
        self.state.acquire_request("req", initial_position=descriptor.end)
        def change_lease(*_):
            self.state.release_request("req")
            self.state.acquire_request("req", initial_position=descriptor.end)
            return Event()
        self.receiver.completion_factory = change_lease
        with self.assertRaisesRegex(PDTailError, "generation changed"):
            self.receiver.receive_and_import(descriptor, state=self.state)
        self.assertFalse(self.state.requests["req"].tail_ready)

    def test_registration_and_deregistration_failures_preserve_lifetime(self):
        self.engine.register_rc = -3
        with self.assertRaisesRegex(PDTailError, "register_memory failed"):
            self.publish()
        self.assertFalse(self.publisher._active)
        self.engine.register_rc = 0
        descriptor = self.publish()
        self.engine.unregister_rc = -2
        with self.assertRaisesRegex(PDTailError, "unregister_memory failed"):
            self.publisher.acknowledge("req", descriptor.nonce)
        self.assertIn("req", self.publisher._active)
        self.assertTrue(self.publisher._retired)
        self.engine.unregister_rc = 0
        self.assertTrue(self.publisher.acknowledge("req", descriptor.nonce))

    def test_optional_unregister_keeps_registered_owners_and_cancel_requires_stop(self):
        self.engine.unregister_memory = None
        descriptor = self.publish()
        with self.assertRaisesRegex(PDTailError, "reads have stopped"):
            self.publisher.cancel("req", descriptor.nonce)
        result = self.receiver.receive(descriptor, state=self.state)
        self.assertTrue(self.publisher.cancel("req", descriptor.nonce, remote_read_stopped=True))
        self.assertTrue(self.publisher._retired)
        self.assertTrue(self.receiver._retired)
        self.assertEqual(result.layers["L0"].position_tags.tags(), tuple(range(7, 135)))

    def test_pointer_metadata_preserves_64_bits(self):
        descriptor = self.publish()
        pointer = (1 << 40) + 99
        changed = replace(descriptor, buffers=(replace(descriptor.buffers[0], pointer=pointer),) + descriptor.buffers[1:])
        self.assertEqual(TailDescriptor.from_dict(changed.to_dict()).buffers[0].pointer, pointer)

    def test_source_epoch_and_committed_position_cannot_roll_back(self):
        first = self.publish()
        self.publisher.acknowledge("req", first.nonce)
        with self.assertRaisesRegex(PDTailError, "interval cannot roll back"):
            self.publish(snapshot(end=134))
        with self.assertRaisesRegex(PDTailError, "epoch cannot roll back"):
            self.publisher.publish("req", snapshot(epoch=6))
        fresh = self.publisher.publish("req", snapshot(end=5, epoch=8), expected_source_epoch=8)
        self.assertEqual(fresh.source_epoch, 8)
        self.assertEqual(fresh.end, 5)

    def test_destination_reuse_rejects_replay_of_previously_imported_descriptor(self):
        descriptor = self.publish()
        self.receiver.receive_and_import(descriptor, state=self.state)
        self.state.release_request("req")
        self.state.acquire_request("req", initial_position=descriptor.end)
        with self.assertRaisesRegex(PDTailError, "generation changed since"):
            self.receiver.receive_and_import(descriptor, state=self.state)

    def test_releasing_payload_retains_watermark_and_rejects_old_pointer_retry(self):
        descriptor = self.publish()
        self.receiver.receive_and_import(descriptor, state=self.state)
        self.assertTrue(self.receiver.release_received("req", descriptor.nonce))
        self.assertFalse(self.receiver.release_received("req", descriptor.nonce))
        self.publisher.acknowledge("req", descriptor.nonce)
        with self.assertRaisesRegex(PDTailError, "payload was released"):
            self.receiver.receive(descriptor, state=self.state)
        self.assertEqual(len(self.engine.reads), 1)

    def test_receiver_validation_failure_does_not_cache_or_admit(self):
        descriptor = self.publish()
        self.receiver.validate_received = lambda *_: False
        with self.assertRaisesRegex(PDTailError, "Received device tail validation failed"):
            self.receiver.receive_and_import(descriptor, state=self.state)
        self.assertFalse(self.receiver._received)
        self.assertFalse(self.state.requests)
        self.assertEqual(len(self.engine.registered), 8)


if __name__ == "__main__":
    unittest.main()
