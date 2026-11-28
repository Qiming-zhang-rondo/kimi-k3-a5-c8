"""Host tests for request ownership and tensor routing; no Torch/NPU import."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfa_recent128.serving_state import ServingState, ServingStateError


class Tensor:
    def __init__(self, data, dtype, device="npu:0"):
        self.data, self.dtype, self.device = data, dtype, device

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, key):
        return Tensor(self.data[key], self.dtype, self.device)

    def __setitem__(self, key, value):
        self.data[key] = value.data if isinstance(value, Tensor) else value

    def fill_(self, value):
        self.data.fill(value)
        return self

    def copy_(self, value):
        self.data[...] = value.data
        return self

    def logical_and_(self, value):
        np.logical_and(self.data, value.data, out=self.data)
        return self

    def logical_not_(self):
        np.logical_not(self.data, out=self.data)
        return self

    def clamp_(self, low, high):
        np.clip(self.data, low, high, out=self.data)
        return self

    def masked_fill_(self, mask, value):
        self.data[mask.data] = value
        return self

    def remainder_(self, divisor):
        self.data %= divisor
        return self

    def index_copy_(self, dim, indices, value):
        assert dim == 0
        self.data[indices.data] = value.data
        return self

    def cpu(self):
        raise AssertionError("No device value may be read by serving state")

    def item(self):
        raise AssertionError("No device scalar extraction is allowed")

    def is_contiguous(self):
        return True


class HostTorch:
    int32 = "torch.int32"
    int64 = "torch.int64"
    bool = "torch.bool"
    bfloat16 = "torch.bfloat16"
    DTYPES = {int32: np.int32, int64: np.int64, bool: np.bool_, bfloat16: np.float32}

    def __init__(self):
        self.allocations = 0

    @staticmethod
    def device(value):
        return str(value)

    def full(self, shape, value, *, dtype, device):
        self.allocations += 1
        return Tensor(np.full(shape, value, dtype=self.DTYPES[dtype]), dtype, device)

    def empty(self, shape, *, dtype, device):
        return self.full(shape, 0, dtype=dtype, device=device)

    def arange(self, *args, dtype, device):
        self.allocations += 1
        return Tensor(np.arange(*args, dtype=self.DTYPES[dtype]), dtype, device)

    @staticmethod
    def _compare(function, left, right, out):
        function(left.data, right.data if isinstance(right, Tensor) else right, out=out.data)

    def ge(self, left, right, *, out):
        self._compare(np.greater_equal, left, right, out)

    def lt(self, left, right, *, out):
        self._compare(np.less, left, right, out)

    def eq(self, left, right, *, out):
        self._compare(np.equal, left, right, out)

    @staticmethod
    def searchsorted(boundaries, values, *, right, out):
        out.data[...] = np.searchsorted(boundaries.data, values.data, side="right" if right else "left")

    def index_select(self, value, dim, indices, *, out=None):
        assert dim == 0
        result = value.data[indices.data]
        if out is None:
            self.allocations += 1
            return Tensor(result.copy(), value.dtype, value.device)
        out.data[...] = result
        return out

    def tensor(self, values, *, dtype=None):
        dtype = dtype or self.int64
        return Tensor(np.asarray(values, dtype=self.DTYPES[dtype]), dtype)


class ServingStateLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.t = HostTorch()
        self.state = ServingState(request_pool_size=8, max_tokens=32,
            max_query_per_request=4, device="npu:0", torch_module=self.t)
        self.state.register_layer("layer0")
        self.state.register_layer("layer1")

    def step(self, ids, positions, boundaries, **kwargs):
        return self.state.bind_step(ids, positions=self.t.tensor(positions),
            query_start_loc=self.t.tensor(boundaries), **kwargs)

    def write_reference(self, request_id, start, end):
        lease = self.state.requests[request_id]
        for ring in self.state.layers.values():
            for pos in range(start, end):
                slot = pos % self.state.capacity
                ring.latent.data[lease.bank, slot] = pos + .5
                ring.rope.data[lease.bank, slot] = pos + .25
                ring.position_tags.data[lease.bank, slot] = pos
                ring.epoch_tags.data[lease.bank, slot] = lease.epoch

    def test_reorder_preserves_request_banks_and_release_changes_epoch(self):
        first = self.step(["a", "b"], [0, 1, 0, -1], [0, 2, 3], request_starts=[0, 0])
        self.assertEqual(first.token_ring_slots.data.tolist(), [0, 0, 1, -1])
        self.assertEqual(first.token_valid.data.tolist(), [True, True, True, False])
        second = self.step(["b", "a"], [2, 2], [0, 1, 2])
        self.assertEqual(second.token_ring_slots.data.tolist(), [1, 0])
        self.assertTrue(self.state.release_request("a"))
        third = self.step(["c"], [0], [0, 1], request_starts=[0])
        self.assertEqual(third.token_ring_slots.data.tolist(), [0])
        self.assertEqual(third.token_request_epochs.data.tolist(), [2])

    def test_b8_same_position_different_bank_and_dynamic_validity(self):
        ids = [str(i) for i in range(8)]
        positions = [254, 255, 256, 257] * 8
        bounds = list(range(0, 33, 4))
        mask = self.t.tensor([True] * 31 + [False], dtype=self.t.bool)
        first = self.step(ids, positions, bounds, valid=mask, request_starts=[0] * 8)
        expected = np.repeat(np.arange(8), 4)
        expected[-1] = -1
        np.testing.assert_array_equal(first.token_ring_slots.data, expected)
        mask.data[:] = True
        mask.data[0] = False
        again = self.step(ids[::-1], positions, bounds, valid=mask)
        self.assertFalse(again.token_valid.data[0])
        self.assertTrue(again.token_valid.data[-1])
        self.assertEqual(again.token_ring_slots.data[1], 7)

    def test_fixed_binding_has_no_per_layer_allocations_or_device_reads(self):
        self.step(["a"], [0, 1], [0, 2], request_starts=[0])
        first = self.state.binding("layer0")
        allocations = self.t.allocations
        for _ in range(5):
            self.assertIs(self.state.binding("layer0", 2), first)
            self.state.binding("layer1", 2)
            metadata = SimpleNamespace()
            self.state.bound_step(metadata)
            self.assertIs(metadata.recent128_token_positions, first.token_positions)
        self.assertEqual(allocations, self.t.allocations)
        self.step(["a"], [2, 3], [0, 2])
        self.assertIs(self.state.binding("layer0"), first)
        self.assertEqual(allocations, self.t.allocations)
        self.assertEqual(first.token_positions.data.tolist(), [2, 3])

    def test_rollback_uses_corrected_positions_in_same_bank(self):
        self.step(["a"], [100, 101, 102, 103], [0, 4], request_starts=[0])
        self.state.begin_speculation("a", committed_end=100, draft_count=4)
        self.assertEqual(self.state.resolve_speculation("a", accepted_tokens=1), 101)
        retried = self.step(["a"], [101, 102], [0, 2])
        self.assertEqual(retried.token_positions.data.tolist(), [101, 102])
        self.assertEqual(retried.token_request_epochs.data.tolist(), [1, 1])
        self.assertGreaterEqual(self.state.capacity, 128 + 4 - 1)
        with self.assertRaisesRegex(ServingStateError, "retention"):
            self.state.begin_speculation("a", committed_end=101, draft_count=5)

    def test_prefix_requires_actual_snapshot_and_pd_completion(self):
        self.state.acquire_request("source")
        self.write_reference("source", 0, 200)
        self.state.commit_request("source", end=200)
        snapshot = self.state.export_tail("source", end=200)
        target = self.state.acquire_request("prefix", initial_position=200)
        with self.assertRaisesRegex(ServingStateError, "not ready"):
            self.step(["prefix"], [200], [0, 1])
        self.state.import_tail("prefix", snapshot, source="prefix")
        self.step(["prefix"], [200], [0, 1])
        ring = self.state.layers["layer0"]
        self.assertEqual(ring.position_tags.data[target.bank, 199], 199)
        self.assertEqual(ring.epoch_tags.data[target.bank, 199], target.epoch)
        np.testing.assert_array_equal(ring.latent.data[target.bank, 199], np.full(512, 199.5))
        self.state.acquire_request("pd", initial_position=200)
        with self.assertRaisesRegex(ServingStateError, "receive-completion"):
            self.state.import_tail("pd", snapshot, source="pd")
        events = []
        completion = SimpleNamespace(wait=lambda: events.append("wait"))
        self.state.import_tail("pd", snapshot, source="pd", completion=completion)
        self.assertEqual(events, ["wait"])

    def test_import_retains_invalid_tags_instead_of_fabricating_readiness(self):
        self.state.acquire_request("a")
        self.write_reference("a", 0, 20)
        self.state.commit_request("a", end=20)
        snapshot = self.state.export_tail("a", end=20)
        snapshot.layers["layer0"].epoch_tags.data[3] = -1
        snapshot.layers["layer0"].position_tags.data[4] = -1
        lease = self.state.acquire_request("b", initial_position=20)
        self.state.import_tail("b", snapshot, source="prefix")
        ring = self.state.layers["layer0"]
        self.assertEqual(ring.epoch_tags.data[lease.bank, 3], -1)
        self.assertEqual(ring.position_tags.data[lease.bank, 4], -1)

    def test_scheduler_identity_cannot_be_inferred_or_evicted_implicitly(self):
        with self.assertRaisesRegex(ServingStateError, "scheduler initial"):
            self.step(["a"], [0], [0, 1])
        with self.assertRaisesRegex(ServingStateError, "Nonzero request"):
            self.step(["a"], [200], [0, 1], request_starts=[200])
        self.assertEqual(self.state.requests, {})
        self.step(["a"], [0], [0, 1], request_starts=[0])
        self.step([], [-1], [0])
        self.assertIn("a", self.state.requests)

    def test_padded_tables_independent_lanes_and_actual_count_mask(self):
        state = ServingState(request_pool_size=2, request_table_capacity=8,
            max_tokens=8, max_query_per_request=4, device="npu:0", torch_module=self.t)
        first = state.routing_lane(("builder", 0))
        second = state.routing_lane(("builder", 1))
        state.register_layer("layer")
        all_valid = self.t.tensor([True] * 4, dtype=self.t.bool)
        first.bind_step(["a"], positions=self.t.tensor([10, 11, 0, 0]),
            query_start_loc=self.t.tensor([0, 4]), valid=all_valid,
            num_actual_tokens=2, request_starts=[0])
        second.bind_step(["a"], positions=self.t.tensor([12]),
            query_start_loc=self.t.tensor([0, 1]))
        self.assertIs(first.pool_owner, state)
        self.assertIs(first.requests, second.requests)
        self.assertIs(first.layers, state.layers)
        self.assertEqual(first.step.token_positions.data.tolist(), [10, 11, -1, -1])
        self.assertEqual(second.step.token_positions.data.tolist(), [12])
        self.assertEqual(tuple(first.step.request_banks.shape), (8,))
        self.assertEqual(tuple(first.binding("layer").recent_latent.shape[:2]), (2, 256))
        self.assertIs(state.routing_lane(("builder", 0), allow_create=False), first)
        with self.assertRaisesRegex(ServingStateError, "preallocated"):
            state.routing_lane("new-capture-lane", allow_create=False)


if __name__ == "__main__":
    unittest.main()
