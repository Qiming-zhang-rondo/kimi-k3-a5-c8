"""Host controller contracts; fake frameworks cannot read tensor contents."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework_integration"))
from sfa_recent128 import native_runtime as native


class Tensor:
    def __init__(self, rows, width=None, dtype="bfloat16", stride=None):
        self.shape = (rows,) if width is None else (rows, width)
        self.dtype, self.device = "torch." + dtype, "npu:0"
        self._stride = stride or ((1,) if width is None else (width, 1))

    def stride(self):
        return self._stride

    def cpu(self):
        raise AssertionError("No host value reads")

    item = tolist = cpu


def binding(rows):
    return SimpleNamespace(
        recent_latent=Tensor(256, 512), recent_rope=Tensor(256, 64),
        recent_position_tags=Tensor(256, dtype="int64"),
        recent_epoch_tags=Tensor(256, dtype="int64"),
        token_ring_slots=Tensor(rows, dtype="int32"),
        token_positions=Tensor(rows, dtype="int64"),
        token_request_epochs=Tensor(rows, dtype="int64"),
        token_valid=Tensor(rows, dtype="bool"), max_query_per_request=32)


class NativeRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.capture = SimpleNamespace(capturing=False)
        self.runtime = SimpleNamespace(impl=object(), state=object(), layer_key="layer.0",
                                       config=SimpleNamespace(max_query_chunk=32), producer_calls=0)
        self.group = SimpleNamespace(world_size=2, rank_in_group=1, device_group=object())
        self.group_calls = 0
        self.torch = ModuleType("torch")
        context = ModuleType("vllm_ascend.ascend_forward_context")
        context._EXTRA_CTX = self.capture
        cp = ModuleType("vllm_ascend.attention.context_parallel.sfa_cp")

        def get_group():
            self.assertFalse(self.capture.capturing)
            self.group_calls += 1
            return self.group

        cp.get_tp_group = get_group
        self.modules = patch.dict(sys.modules, {
            "torch": self.torch, "vllm_ascend.ascend_forward_context": context,
            "vllm_ascend.attention.context_parallel.sfa_cp": cp})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        testcase = self

        class Buffers:
            @classmethod
            def prepare(cls, layout, *, device, torch_module):
                testcase.assertFalse(testcase.capture.capturing)
                testcase.assertIs(torch_module, testcase.torch)
                testcase.events.append(("buffers", layout))
                return SimpleNamespace(layout=layout, device=device)

        class Routes:
            @classmethod
            def prepare(cls, layout, global_binding, *, torch_module):
                testcase.assertFalse(testcase.capture.capturing)
                testcase.events.append(("routes", layout, global_binding))
                local = global_binding if layout.tp_size == 1 else binding(layout.local_rows)
                route = SimpleNamespace(binding=local)
                route.update = lambda: testcase.events.append(("update", route))
                return route

        def gather(latent, rope, layout, buffers, *, torch_module, group):
            testcase.assertEqual(latent.shape[0], layout.local_rows)
            testcase.assertEqual(rope.shape[0], layout.local_rows)
            testcase.events.append(("gather", latent, rope, layout, group))
            return ((latent, rope) if layout.tp_size == 1 else
                    (Tensor(layout.global_tokens, 512, stride=(576, 1)),
                     Tensor(layout.global_tokens, 64, stride=(576, 1))))

        class Writer:
            def __init__(self, latent, rope, ring, *, check_is_capture):
                testcase.assertFalse(check_is_capture())
                self.ring = ring
                self.check = check_is_capture
                testcase.events.append(("writer", self, latent, rope, ring))

            def write(self, latent, rope):
                testcase.events.append(("write", self, latent, rope))
                return self.ring

        self.patchers = [patch.object(native, "NativeCPBuffers", Buffers),
                         patch.object(native, "NativeQueryRoutes", Routes),
                         patch.object(native, "gather_native_rows", gather),
                         patch.object(native, "PreparedNativeWriter", Writer)]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.producer = native.NativeProducer(self.runtime)

    def metadata(self, count=5, *, cp=False, lane=None):
        if lane is None:
            ring = binding(count)
            lane = SimpleNamespace(pool_owner=self.runtime.state)
            lane.binding = lambda layer, *, token_rows: (
                self.assertEqual((layer, token_rows), ("layer.0", count)) or ring)
        context = (SimpleNamespace(num_tokens=count, num_tokens_pad=6,
                                   local_start=3, local_end=5, local_end_with_pad=6)
                   if cp else None)
        return SimpleNamespace(recent128_state=lane, num_input_tokens=count,
                               max_query_len=4, dsa_cp_context=context)

    def scope(self, metadata, *, rows=None):
        if rows is None:
            rows = 3 if metadata.dsa_cp_context else metadata.num_input_tokens
        latent, rope = Tensor(rows, 512), Tensor(rows, 64)
        return SimpleNamespace(impl=self.runtime.impl, metadata=metadata, writes=0,
                               require_capture=lambda: (latent, rope))

    def count(self, event):
        return sum(row[0] == event for row in self.events)

    def test_construction_imports_no_framework_and_allocates_nothing(self):
        self.assertIsNone(self.producer._torch)
        self.assertIsNone(self.producer._tp_group)
        self.assertEqual(self.events, [])
        self.assertEqual(self.group_calls, 0)

    def test_ordinary_native_reuses_lane_plan_and_does_not_resolve_tp(self):
        metadata = self.metadata()
        first = self.scope(metadata)
        global_binding = self.producer.write(first)
        second = self.scope(metadata)
        self.producer.write(second)
        self.assertEqual(self.count("buffers"), 1)
        self.assertEqual(self.count("writer"), 1)
        self.assertEqual(self.count("write"), 2)
        self.assertEqual(self.group_calls, 0)
        self.assertTrue(all(row[4] is None for row in self.events if row[0] == "gather"))
        self.assertIs(self.producer.reader_binding(metadata, 5), global_binding)
        self.assertEqual((first.writes, second.writes, self.runtime.producer_calls), (1, 1, 2))

    def test_dsacp_global_writer_and_local_reader_use_exact_device_group(self):
        metadata = self.metadata(cp=True)
        result = self.producer.write(self.scope(metadata))
        writer_event = next(row for row in self.events if row[0] == "writer")
        gather_event = next(row for row in self.events if row[0] == "gather")
        self.assertEqual(writer_event[2].shape, (5, 512))
        self.assertEqual(writer_event[2].stride(), (576, 1))
        self.assertIs(writer_event[4], result)
        self.assertIs(gather_event[4], self.group.device_group)
        local = self.producer.reader_binding(metadata, 3)
        self.assertEqual(local.token_positions.shape, (3,))
        self.assertIsNot(local, result)
        self.assertEqual(self.count("update"), 1)
        with self.assertRaisesRegex(native.NativeCPError, "local DSACP"):
            self.producer.reader_binding(metadata, 5)

    def test_capture_reuses_prepared_writer_and_never_resolves_group_again(self):
        metadata = self.metadata(cp=True)
        self.producer.write(self.scope(metadata))
        self.capture.capturing = True
        self.producer.write(self.scope(metadata))
        self.producer.reader_binding(metadata, 3)
        self.assertEqual(self.group_calls, 1)
        self.assertEqual(self.count("buffers"), 1)
        self.assertEqual(self.count("writer"), 1)
        self.assertEqual(self.count("write"), 2)

    def test_capture_actual_count_change_reuses_static_bucket_and_device_valid_routes(self):
        metadata=self.metadata(cp=True)
        self.producer.write(self.scope(metadata))
        metadata.dsa_cp_context.local_end=0  # capture dummy / replay validity differs
        self.capture.capturing=True
        self.producer.write(self.scope(metadata))
        self.producer.reader_binding(metadata,3)
        self.assertEqual(self.count("buffers"),1)
        self.assertEqual(self.count("writer"),1)
        self.assertEqual(self.count("write"),2)

    def test_unprepared_capture_fails_before_group_buffers_or_gather(self):
        self.capture.capturing = True
        for metadata in (self.metadata(), self.metadata(cp=True)):
            with self.assertRaisesRegex(native.NativeCPError, "before Graph capture"):
                self.producer.write(self.scope(metadata))
        self.assertEqual(self.events, [])
        self.assertEqual(self.group_calls, 0)
        self.assertEqual(self.runtime.producer_calls, 0)

    def test_lanes_and_changed_routing_storage_have_independent_plans(self):
        first, second = self.metadata(), self.metadata()
        self.producer.write(self.scope(first))
        self.producer.write(self.scope(second))
        self.assertEqual(self.count("writer"), 2)
        second.recent128_state.binding("layer.0", token_rows=5).token_positions = Tensor(5, dtype="int64")
        self.capture.capturing = True
        with self.assertRaisesRegex(native.NativeCPError, "no plan allocation"):
            self.producer.write(self.scope(second))
        self.assertEqual(self.count("writer"), 2)

    def test_duplicate_scope_and_reader_before_write_are_rejected(self):
        metadata = self.metadata()
        with self.assertRaisesRegex(native.NativeCPError, "global producer write"):
            self.producer.reader_binding(metadata, 5)
        scope = self.scope(metadata)
        self.producer.write(scope)
        with self.assertRaisesRegex(native.NativeCPError, "exactly one"):
            self.producer.write(scope)
        self.assertEqual(self.count("write"), 1)
        self.assertEqual(self.runtime.producer_calls, 1)

    def test_foreign_lane_and_layer_are_rejected(self):
        metadata = self.metadata()
        metadata.recent128_state.pool_owner = object()
        with self.assertRaisesRegex(native.NativeCPError, "bound request metadata"):
            self.producer.write(self.scope(metadata))
        scope = self.scope(self.metadata())
        scope.impl = object()
        with self.assertRaisesRegex(native.NativeCPError, "another layer"):
            self.producer.write(scope)
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
