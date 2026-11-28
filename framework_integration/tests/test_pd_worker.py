"""Host lifecycle tests only; these do not certify NPU execution."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sfa_recent128.pd_worker import WorkerTailRegistry, export_tail, receive_tail
from sfa_recent128.serving_state import ServingState, ServingStateError
from test_serving_state_lifecycle import HostTorch, Tensor


class TransferTensor(Tensor):
    def __getitem__(self, key):
        return TransferTensor(self.data[key], self.dtype, self.device)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.data.tolist()


class TransferTorch(HostTorch):
    def full(self, shape, value, *, dtype, device):
        tensor = super().full(shape, value, dtype=dtype, device=device)
        return TransferTensor(tensor.data, tensor.dtype, tensor.device)

    def index_select(self, value, dim, indices, *, out=None):
        tensor = super().index_select(value, dim, indices, out=out)
        return TransferTensor(tensor.data, tensor.dtype, tensor.device)


class Completion:
    def __init__(self):
        self.waits = 0
        self.syncs = 0

    def wait(self):
        self.waits += 1

    def synchronize(self):
        self.syncs += 1


class PDWorkerTests(unittest.TestCase):
    def state(self, keys=("layer0", "layer1")):
        state = ServingState(request_pool_size=4, max_tokens=8,
            max_query_per_request=4, device="npu:0", torch_module=TransferTorch())
        for key in keys:
            state.register_layer(key)
        return state

    def write(self, state, request_id, end):
        lease = state.acquire_request(request_id)
        for ring in state.layers.values():
            for position in range(max(0, end - 128), end):
                slot = position % state.capacity
                ring.latent.data[lease.bank, slot] = position + .5
                ring.rope.data[lease.bank, slot] = position + .25
                ring.position_tags.data[lease.bank, slot] = position
                ring.epoch_tags.data[lease.bank, slot] = lease.epoch
        return lease

    def test_actual_completion_commits_scheduler_end_and_all_layers(self):
        state = self.state()
        lease = self.write(state, "r", 140)
        event = Completion()
        tail = export_tail(state, "r", committed_end=140, completion=event)
        self.assertEqual((tail.start, tail.end), (12, 140))
        self.assertEqual(set(tail.layers), {"layer0", "layer1"})
        self.assertEqual(lease.committed_end, 140)
        self.assertEqual((event.waits, event.syncs), (1, 1))
        self.assertIsNone(tail.completion)

    def test_missing_or_stale_tags_reject_and_restore_old_commit(self):
        state = self.state()
        lease = self.write(state, "r", 8)
        state.layers["layer1"].epoch_tags.data[lease.bank, 4] = -1
        with self.assertRaisesRegex(ServingStateError, "epoch tags"):
            export_tail(state, "r", committed_end=8, completion=Completion())
        self.assertEqual(lease.committed_end, 0)
        state.layers["layer1"].epoch_tags.data[lease.bank, 4] = lease.epoch
        state.layers["layer0"].position_tags.data[lease.bank, 4] = 3
        with self.assertRaisesRegex(ServingStateError, "position tags"):
            export_tail(state, "r", committed_end=8, completion=Completion())

    def test_no_completion_and_pending_speculation_cannot_export(self):
        state = self.state()
        self.write(state, "r", 2)
        with self.assertRaisesRegex(ServingStateError, "completion"):
            export_tail(state, "r", committed_end=2, completion=None)
        state.begin_speculation("r", committed_end=0, draft_count=2)
        with self.assertRaisesRegex(ServingStateError, "speculative"):
            export_tail(state, "r", committed_end=2, completion=Completion())

    def test_receive_existing_empty_lease_preserves_bank_epoch_and_rebases_tags(self):
        producer, consumer = self.state(), self.state()
        self.write(producer, "r", 12)
        tail = export_tail(producer, "r", committed_end=12, completion=Completion())
        old = consumer.acquire_request("old")
        consumer.release_request("old")
        lease = consumer.acquire_request("r")
        identity = (lease.bank, lease.epoch)
        self.assertGreater(lease.epoch, tail.source_epoch)
        result = receive_tail(consumer, "r", tail, completion=Completion())
        self.assertIs(result, lease)
        self.assertEqual((result.bank, result.epoch), identity)
        self.assertEqual(result.committed_end, 12)
        self.assertTrue(result.tail_ready)
        for ring in consumer.layers.values():
            self.assertEqual(ring.epoch_tags.data[lease.bank, :12].tolist(), [lease.epoch] * 12)
            self.assertEqual(ring.position_tags.data[lease.bank, :12].tolist(), list(range(12)))

    def test_receive_cannot_overwrite_producer_rows_without_a_host_commit(self):
        producer, consumer = self.state(), self.state()
        self.write(producer, "r", 12)
        tail = export_tail(producer, "r", committed_end=12, completion=Completion())
        lease = self.write(consumer, "r", 1)
        self.assertEqual(lease.committed_end, 0)
        with self.assertRaisesRegex(ServingStateError, "producer rows"):
            receive_tail(consumer, "r", tail, completion=Completion())
        self.assertEqual(lease.committed_end, 0)

    def test_registry_deduplicates_routing_lanes_and_covers_every_owner(self):
        first, second = self.state(("layer0",)), self.state(("layer1",))
        self.write(first, "r", 4)
        self.write(second, "r", 4)
        registry = WorkerTailRegistry((first, first.routing_lane("draft"), second))
        self.assertEqual(len(registry.states), 2)
        snapshots = registry.export_request("r", committed_end=4, completion=Completion())
        target0, target1 = self.state(("layer0",)), self.state(("layer1",))
        receiver = WorkerTailRegistry((target0, target0.routing_lane("draft"), target1))
        leases = receiver.receive_request("r", snapshots, completion=Completion())
        self.assertEqual(len(leases), 2)
        self.assertEqual(len(receiver.request_identity("r")), 2)
        self.assertTrue(all(lease.tail_ready for lease in leases))
        with self.assertRaisesRegex(ServingStateError, "every local pool owner"):
            receiver.receive_request("other", snapshots[:1], completion=Completion())
        self.assertNotIn("other", target0.requests)

    def test_incomplete_layers_rejected_before_consumer_admission(self):
        producer, consumer = self.state(("layer0",)), self.state()
        self.write(producer, "r", 1)
        tail = export_tail(producer, "r", committed_end=1, completion=Completion())
        with self.assertRaisesRegex(ServingStateError, "every local attention layer"):
            receive_tail(consumer, "r", tail, completion=Completion())
        self.assertNotIn("r", consumer.requests)


if __name__ == "__main__":
    unittest.main()
