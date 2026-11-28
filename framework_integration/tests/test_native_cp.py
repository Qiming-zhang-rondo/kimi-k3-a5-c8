"""Host contract checks for raw capture, TP token exchange and local routing.

Fake BF16 stores uint16 bit patterns, including NaN encodings, rather than
silently passing FP32 numerical comparisons. This is not device validation.
"""
from contextvars import copy_context
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfa_recent128.native_cp import (
    DSACPLayout, NativeCaptureScope, NativeCPBuffers, NativeCPError,
    NativeQueryRoutes, capture_native, gather_native_rows, get_native_scope,
)
from sfa_recent128.prolog_provider import RingBinding


class Tensor:
    def __init__(self, values, dtype, owner, device="npu:0"):
        self.values, self.dtype, self.owner, self.device = values, dtype, owner, device

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, key):
        self.owner.views += 1
        return Tensor(self.values[key], self.dtype, self.owner, self.device)

    def view(self, shape):
        self.owner.views += 1
        values = self.values.reshape(shape)
        if values.size and not np.shares_memory(values, self.values):
            raise AssertionError("Preparing a view must not copy tensor storage")
        return Tensor(values, self.dtype, self.owner, self.device)

    def is_contiguous(self):
        return bool(self.values.flags.c_contiguous)

    def copy_(self, source):
        self.values[...] = source.values
        return self

    def fill_(self, value):
        self.values.fill(value)
        return self

    def cpu(self):
        raise AssertionError("No host tensor read")

    def item(self):
        raise AssertionError("No scalar extraction")


class HostTorch:
    bfloat16, int32, int64, bool = "torch.bfloat16", "torch.int32", "torch.int64", "torch.bool"
    dtypes = {bfloat16: np.uint16, int32: np.int32, int64: np.int64, bool: np.bool_}

    def __init__(self):
        self.allocations = self.views = 0
        self.collectives = []
        self.rank_payloads = None
        self.rank = None
        self.expected_group = object()
        self.distributed = SimpleNamespace(all_gather_into_tensor=self.all_gather_into_tensor)

    def empty(self, shape, *, dtype, device):
        self.allocations += 1
        return Tensor(np.empty(shape, dtype=self.dtypes[dtype]), dtype, self, device)

    def full(self, shape, fill, *, dtype, device):
        tensor = self.empty(shape, dtype=dtype, device=device)
        tensor.fill_(fill)
        return tensor

    def tensor(self, values, dtype):
        return Tensor(np.asarray(values, dtype=self.dtypes[dtype]), dtype, self)

    def all_gather_into_tensor(self, output, local, *, group):
        if group is not self.expected_group:
            raise AssertionError("Wrong device group")
        np.testing.assert_array_equal(local.values, self.rank_payloads[self.rank])
        output.values[...] = np.concatenate(self.rank_payloads, axis=0)
        self.collectives.append((output, local, group))


def layout(global_tokens=5, tp_size=2, tp_rank=0, actual_tokens=None):
    padded = ((global_tokens + tp_size - 1) // tp_size) * tp_size
    rows = padded // tp_size
    context = SimpleNamespace(num_tokens=global_tokens, num_tokens_pad=padded,
        local_start=tp_rank * rows, local_end_with_pad=(tp_rank + 1) * rows,
        local_end=min((tp_rank + 1) * rows,
                      global_tokens if actual_tokens is None else actual_tokens))
    return DSACPLayout.from_context(context, global_tokens=global_tokens,
                                   tp_size=tp_size, tp_rank=tp_rank)


def raw(t, row_ids, width, *, rank4=True):
    values = (np.asarray(row_ids, dtype=np.uint16)[:, None] * 997
              + np.arange(width, dtype=np.uint16)[None, :])
    # Explicit BF16 infinity/NaN/sign payloads must survive concatenation.
    values[:, :4] = np.asarray([0x7f80, 0x7fc1, 0xff80, 0x8000], dtype=np.uint16)
    if rank4:
        values = values.reshape(len(row_ids), 1, 1, width)
    return Tensor(values, t.bfloat16, t)


def binding(t, count):
    stores = tuple(object() for _ in range(4))
    return RingBinding(*stores,
        t.tensor(np.arange(count) + 10, t.int32),
        t.tensor(np.arange(count) + 100, t.int64),
        t.tensor(np.arange(count) + 1000, t.int64),
        t.tensor([True] * (count - 1) + [False], t.bool), 4)


class NativeCPTests(unittest.TestCase):
    def test_capture_exact_references_nesting_and_exception_cleanup(self):
        impl, other, metadata = object(), object(), object()
        latent, rope = object(), object()
        self.assertIsNone(get_native_scope())
        self.assertFalse(capture_native(impl, latent, rope))
        with NativeCaptureScope(impl, metadata) as outer:
            self.assertIs(outer.metadata, metadata)
            self.assertEqual(outer.writes, 0)
            self.assertFalse(capture_native(other, latent, rope))
            with self.assertRaisesRegex(NativeCPError, "exactly one"):
                outer.require_capture()
            with self.assertRaisesRegex(ValueError, "failure"):
                with NativeCaptureScope(other, object()) as inner:
                    self.assertIsNone(get_native_scope(impl))
                    self.assertTrue(capture_native(other, rope, latent))
                    self.assertIs(inner.require_capture()[0], rope)
                    raise ValueError("failure")
            self.assertIs(get_native_scope(impl), outer)
            self.assertTrue(capture_native(impl, latent, rope))
            self.assertIs(outer.require_capture()[0], latent)
            self.assertIs(outer.require_capture()[1], rope)
            self.assertEqual(outer.capture_count, 1)
        self.assertIsNone(get_native_scope())
        with self.assertRaisesRegex(NativeCPError, "single-use"):
            with outer:
                pass

    def test_capture_duplicate_rejected_and_independent_contexts_do_not_leak(self):
        impl = object()
        with NativeCaptureScope(impl, "outer") as outer:
            def isolated():
                with NativeCaptureScope(impl, "inner") as inner:
                    capture_native(impl, "inner-latent", "inner-rope")
                    self.assertEqual(inner.require_capture(), ("inner-latent", "inner-rope"))
            copy_context().run(isolated)
            self.assertEqual(outer.capture_count, 0)
            capture_native(impl, "outer-latent", "outer-rope")
            with self.assertRaisesRegex(NativeCPError, "more than once"):
                capture_native(impl, "duplicate", "duplicate")
            with self.assertRaisesRegex(NativeCPError, "exactly one"):
                outer.require_capture()
        self.assertIsNone(get_native_scope())

    def test_layout_is_exact_rank_contiguous_and_keeps_actual_padding_separate(self):
        last = layout(global_tokens=5, tp_size=8, tp_rank=7, actual_tokens=3)
        self.assertEqual((last.num_tokens_pad, last.local_rows, last.local_input_rows,
                          last.local_valid_rows), (8, 1, 0, 0))
        middle = layout(global_tokens=5, tp_size=2, tp_rank=1, actual_tokens=4)
        self.assertEqual((middle.local_start, middle.local_rows, middle.local_input_rows,
                          middle.local_valid_rows), (3, 3, 2, 1))
        with self.assertRaisesRegex(NativeCPError, "rank order"):
            DSACPLayout(5, 6, 0, 5, 3, 2, 1)
        with self.assertRaisesRegex(NativeCPError, "TP-aligned"):
            DSACPLayout(5, 8, 0, 3, 4, 2, 0)
        with self.assertRaisesRegex(NativeCPError, "num_tokens must match"):
            DSACPLayout.from_context(SimpleNamespace(num_tokens=4),
                                    global_tokens=5, tp_size=2, tp_rank=0)

    def test_gather_preserves_bf16_bits_rank_order_and_trims_only_tp_padding(self):
        t = HostTorch()
        raws = [(raw(t, [rank * 3 + i for i in range(3)], 512),
                 raw(t, [rank * 3 + i for i in range(3)], 64)) for rank in range(2)]
        t.rank_payloads = [np.concatenate((latent.values.reshape(3, 512),
                                          rope.values.reshape(3, 64)), axis=1)
                           for latent, rope in raws]
        expected = np.concatenate(t.rank_payloads, axis=0)[:5]
        for rank in range(2):
            current = layout(tp_rank=rank, actual_tokens=4)
            prepared = NativeCPBuffers.prepare(current, device="npu:0", torch_module=t)
            allocation_count, view_count = t.allocations, t.views
            t.rank = rank
            latent, rope = gather_native_rows(*raws[rank], current, prepared,
                                             torch_module=t, group=t.expected_group)
            self.assertIs(latent, prepared.latent)
            self.assertIs(rope, prepared.rope)
            self.assertEqual(latent.shape, (5, 512))
            self.assertEqual(rope.shape, (5, 64))
            self.assertEqual(latent.values.strides, (576 * 2, 2))
            np.testing.assert_array_equal(latent.values, expected[:, :512])
            np.testing.assert_array_equal(rope.values, expected[:, 512:])
            self.assertEqual((t.allocations, t.views), (allocation_count, view_count))
            self.assertEqual(current.extra_collectives_per_call, 1)
            self.assertEqual(current.local_payload_bytes, 3 * 576 * 2)
        self.assertEqual(len(t.collectives), 2)  # Exactly once per participating rank.

    def test_empty_local_query_rank_still_participates_in_bf16_collective(self):
        t = HostTorch()
        current = layout(global_tokens=1, tp_size=8, tp_rank=7)
        prepared = NativeCPBuffers.prepare(current, device="npu:0", torch_module=t)
        latent, rope = raw(t, [7], 512), raw(t, [7], 64)
        t.rank_payloads = [np.full((1, 576), rank, np.uint16) for rank in range(8)]
        t.rank_payloads[7] = np.concatenate((latent.values.reshape(1, 512),
                                           rope.values.reshape(1, 64)), axis=1)
        t.rank = 7
        result, _ = gather_native_rows(latent, rope, current, prepared,
                                      torch_module=t, group=t.expected_group)
        self.assertEqual(current.local_input_rows, 0)
        self.assertEqual(len(t.collectives), 1)
        np.testing.assert_array_equal(result.values, np.zeros((1, 512), np.uint16))

    def test_tp1_returns_original_refs_without_payload_allocation_or_collective(self):
        t = HostTorch()
        current = layout(global_tokens=5, tp_size=1)
        prepared = NativeCPBuffers.prepare(current, device="npu:0", torch_module=t)
        latent, rope = raw(t, range(5), 512), raw(t, range(5), 64)
        result = gather_native_rows(latent, rope, current, prepared, torch_module=t)
        self.assertIs(result[0], latent)
        self.assertIs(result[1], rope)
        self.assertEqual((t.allocations, t.views, len(t.collectives)), (0, 0, 0))
        self.assertEqual(current.extra_collectives_per_call, 0)
        self.assertEqual(current.local_payload_bytes, 0)

    def test_local_routes_preserve_global_identity_refresh_and_never_promote_padding(self):
        t = HostTorch()
        current = layout(tp_rank=1, actual_tokens=4)
        source = binding(t, 5)
        routes = NativeQueryRoutes.prepare(current, source, torch_module=t)
        for name in ("recent_latent", "recent_rope", "recent_position_tags", "recent_epoch_tags"):
            self.assertIs(getattr(routes.binding, name), getattr(source, name))
        allocation_count, view_count = t.allocations, t.views
        local = routes.update()
        self.assertIs(local, routes.binding)
        self.assertEqual(local.token_positions.values.tolist(), [103, 104, -1])
        self.assertEqual(local.token_ring_slots.values.tolist(), [13, 14, -1])
        self.assertEqual(local.token_request_epochs.values.tolist(), [1003, 1004, -1])
        self.assertEqual(local.token_valid.values.tolist(), [True, False, False])
        # Poison previous replay padding, then update current global identity.
        local.token_valid.fill_(True)
        local.token_positions.fill_(999)
        source.token_positions.values[:] += 10
        source.token_valid.values[:] = [False, False, False, False, True]
        self.assertIs(routes.update(), local)
        self.assertEqual(local.token_positions.values.tolist(), [113, 114, -1])
        self.assertEqual(local.token_valid.values.tolist(), [False, True, False])
        self.assertEqual((t.allocations, t.views), (allocation_count, view_count))

    def test_query_rank_beyond_global_tokens_is_all_invalid_and_tp1_aliases_source(self):
        t = HostTorch()
        source = binding(t, 1)
        current = layout(global_tokens=1, tp_size=8, tp_rank=7)
        routes = NativeQueryRoutes.prepare(current, source, torch_module=t)
        local = routes.update()
        self.assertEqual(local.token_ring_slots.values.tolist(), [-1])
        self.assertEqual(local.token_positions.values.tolist(), [-1])
        self.assertEqual(local.token_valid.values.tolist(), [False])
        tp1 = NativeQueryRoutes.prepare(layout(global_tokens=1, tp_size=1), source, torch_module=t)
        counts = t.allocations, t.views
        self.assertIs(tp1.binding, source)
        self.assertIs(tp1.update(), source)
        self.assertEqual((t.allocations, t.views), counts)

    def test_bad_layout_buffer_dtype_or_source_routes_fail_before_collective(self):
        t = HostTorch()
        current = layout()
        prepared = NativeCPBuffers.prepare(current, device="npu:0", torch_module=t)
        latent, rope = raw(t, range(3), 512), raw(t, range(3), 64)
        with self.assertRaisesRegex(NativeCPError, "exact layout"):
            gather_native_rows(latent, rope, layout(tp_rank=1), prepared,
                               torch_module=t, group=t.expected_group)
        latent.dtype = "torch.float32"
        with self.assertRaisesRegex(NativeCPError, "bfloat16"):
            gather_native_rows(latent, rope, current, prepared,
                               torch_module=t, group=t.expected_group)
        self.assertEqual(len(t.collectives), 0)
        with self.assertRaisesRegex(NativeCPError, "shape"):
            NativeQueryRoutes.prepare(current, binding(t, 4), torch_module=t)


if __name__ == "__main__":
    unittest.main()
