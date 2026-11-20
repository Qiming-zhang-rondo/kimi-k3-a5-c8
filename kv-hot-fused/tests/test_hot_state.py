"""Lifecycle checks use real metadata and a tiny external payload store.

No NPU behavior/performance is claimed. The dict represents only the caller's
copy ordering; production must supply equivalent stream/event dependencies.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hotkv"))
from kv_hot_state import HotCacheStateError, MissingHotKVError, RecentKVState


def stage_payload(state, count, storage):
    plan = state.stage(count)
    for item in plan.writes:
        storage[item.slot] = (state.request_id, state.generation, item.position)
    state.publish_stage(plan)
    return plan


def commit_payload(state, accepted, storage):
    plan = state.plan_commit(accepted)
    values = [(x.destination_slot, storage[x.source_slot]) for x in plan.copies]
    for dst, value in values:
        storage[dst] = value
    state.complete_commit(plan)
    return plan


def seed_payload(state, end, storage):
    plan = state.prepare_seed(end)
    for item in plan.writes:
        storage[item.slot] = (state.request_id, state.generation, item.position)
    state.publish_restore(plan)


class HotStateTests(unittest.TestCase):
    def assert_payload(self, state, storage, position, query):
        route = state.resolve(position, query)
        expected_hot = state.window and max(0, query-state.window+1) <= position <= query
        if expected_hot:
            self.assertIn(route.source, ("ring", "scratch"))
            self.assertEqual(storage[route.slot], (state.request_id, state.generation, position))
        elif 0 <= position <= query:
            self.assertEqual(route.source, "cold")
        else:
            self.assertEqual(route.source, "invalid")

    def test_zero_window_has_no_hot_allocation_or_write_work(self):
        state = RecentKVState("a", 0, 0, 64)
        self.assertIsNone(state._tags)
        self.assertEqual(state.capacity, 0)
        for accepted in (0, 1, 64):
            plan = stage_payload(state, 64, {})
            self.assertEqual(plan.writes, ())
            self.assertEqual(state.resolve(plan.start, plan.start).source, "cold")
            committed = commit_payload(state, accepted, {})
            self.assertEqual(committed.copies, ())
        self.assertEqual(state.committed_end, 65)
        self.assertEqual(state.export_snapshot()["entries"], [])

    def test_uninitialized_existing_prefix_cannot_stage_resolve_or_export(self):
        state = RecentKVState("a", 0, 4, 8, committed_end=100)
        self.assertFalse(state.tail_ready)
        with self.assertRaises(MissingHotKVError): state.stage(1)
        with self.assertRaises(MissingHotKVError): state.resolve(99, 99)
        with self.assertRaises(MissingHotKVError): state.export_snapshot()
        storage = {}
        seed_payload(state, 100, storage)
        self.assertTrue(state.tail_ready)
        self.assert_payload(state, storage, 96, 99)

    def test_seed_not_readable_until_payload_published(self):
        state = RecentKVState("a", 0, 4, 8, committed_end=100)
        plan = state.prepare_seed()
        self.assertEqual([x.position for x in plan.writes], [96, 97, 98, 99])
        self.assertFalse(state.tail_ready)
        with self.assertRaises(HotCacheStateError): state.resolve(99, 99)
        with self.assertRaises(HotCacheStateError): state.stage(1)
        with self.assertRaises(ValueError):
            RecentKVState("a", 0, 4, 8, committed_end=100).prepare_seed(99)

    def test_stage_does_not_touch_committed_ring_and_requires_publication(self):
        state, storage = RecentKVState("a", 0, 4, 8), {}
        seed_payload(state, 8, storage)
        before = {x: storage[x] for x in range(4)}
        plan = state.stage(8)
        self.assertEqual([x.slot for x in plan.writes], list(range(4, 12)))
        with self.assertRaises(HotCacheStateError): state.resolve(8, 8)
        with self.assertRaises(HotCacheStateError): state.plan_commit(1)
        self.assertEqual(before, storage)
        for write in plan.writes: storage[write.slot] = ("a", 0, write.position)
        state.publish_stage(plan)
        self.assertEqual(before, {x: storage[x] for x in range(4)})
        for q in range(8, 16):
            for p in range(q-3, q+1): self.assert_payload(state, storage, p, q)

    def test_partial_accept_reject_and_replay_preserve_correct_prefix(self):
        state, storage = RecentKVState("a", 1, 4, 8), {}
        seed_payload(state, 8, storage)
        first = stage_payload(state, 6, storage)
        commit_payload(state, 2, storage)
        self.assertEqual(state.committed_end, 10)
        with self.assertRaises(HotCacheStateError): state.resolve(10, 10)
        replay = stage_payload(state, 4, storage)
        self.assertEqual(replay.start, 10)
        self.assertGreater(replay.step_id, first.step_id)
        for p in range(7, 11): self.assert_payload(state, storage, p, 10)
        commit_payload(state, 4, storage)
        for p in range(10, 14): self.assert_payload(state, storage, p, 13)

    def test_all_rejected_or_aborted_does_not_modify_committed_payload(self):
        for accept_zero in (False, True):
            state, storage = RecentKVState("a", 0, 3, 10), {}
            seed_payload(state, 17, storage)
            before = {s: storage[s] for s in range(3)}
            stage_payload(state, 10, storage)
            if accept_zero: commit_payload(state, 0, storage)
            else: state.abort_step()
            self.assertEqual(state.committed_end, 17)
            self.assertEqual(before, {s: storage[s] for s in range(3)})
            for p in range(14, 17): self.assert_payload(state, storage, p, 16)

    def test_window_one_and_large_accept_have_unique_copy_destinations(self):
        for window in (1, 3, 8):
            state, storage = RecentKVState("a", 0, window, 12), {}
            seed_payload(state, 9, storage)
            stage_payload(state, 12, storage)
            plan = state.plan_commit(12)
            self.assertEqual(len(plan.copies), window)
            self.assertEqual(len({x.destination_slot for x in plan.copies}), window)
            self.assertEqual([x.position for x in plan.copies], list(range(21-window, 21)))
            for item in plan.copies: storage[item.destination_slot] = storage[item.source_slot]
            state.complete_commit(plan)
            for p in range(21-window, 21): self.assert_payload(state, storage, p, 20)

    def test_ring_wrap_per_query_window_and_invalid_indices(self):
        state, storage = RecentKVState("a", 0, 3, 4), {}
        for _ in range(12):
            plan = stage_payload(state, 4, storage)
            for q in range(plan.start, plan.end):
                for p in range(-1, q+2): self.assert_payload(state, storage, p, q)
            commit_payload(state, 4, storage)
        with self.assertRaises(MissingHotKVError): state.resolve(0, 0)

    def test_commit_preparation_freezes_reads_and_cannot_be_aborted(self):
        state, storage = RecentKVState("a", 0, 4, 4), {}
        seed_payload(state, 6, storage)
        stage_payload(state, 2, storage)
        plan = state.plan_commit(1)
        self.assertFalse(state.tail_ready)
        with self.assertRaises(HotCacheStateError): state.resolve(5, 6)
        with self.assertRaises(HotCacheStateError): state.abort_step()
        with self.assertRaises(HotCacheStateError): state.plan_commit(1)
        state.complete_commit(plan)
        self.assertTrue(state.tail_ready)
        with self.assertRaises(HotCacheStateError): state.complete_commit(plan)

    def test_decode_updates_touch_only_accepted_ring_slots(self):
        class CountedTags(list):
            writes = 0

            def __iter__(self):
                raise AssertionError("Decode must not scan the full hot window")

            def __setitem__(self, index, value):
                self.writes += 1
                return super().__setitem__(index, value)

        state, storage = RecentKVState("a", 0, 4096, 8), {}
        state._tags = CountedTags(state._tags)
        for _ in range(8):
            plan = stage_payload(state, 8, storage)
            self.assertEqual(len(plan.writes), 8)
            for p in range(plan.start, plan.end): self.assert_payload(state, storage, p, p)
            commit_payload(state, 3, storage)
        self.assertEqual(state._tags.writes, 24)

    def test_foreign_copied_and_stale_plans_are_rejected(self):
        a, b = RecentKVState("a", 0, 4, 4), RecentKVState("a", 0, 4, 4)
        ap, bp = a.stage(1), b.stage(1)
        with self.assertRaises(HotCacheStateError): a.publish_stage(bp)
        with self.assertRaises(HotCacheStateError): a.publish_stage(replace(ap))
        a.abort_step()
        a.stage(1)
        with self.assertRaises(HotCacheStateError): a.publish_stage(ap)

    def test_pd_snapshot_json_roundtrip_and_explicit_payload_restore(self):
        source, source_store = RecentKVState("request", 7, 5, 8), {}
        seed_payload(source, 103, source_store)
        snapshot = json.loads(json.dumps(source.export_snapshot()))
        target, target_store = RecentKVState("request", 7, 5, 2), {}
        plan = target.prepare_restore(snapshot)
        self.assertEqual(target.committed_end, 0)
        self.assertFalse(target.tail_ready)
        with self.assertRaises(HotCacheStateError): target.resolve(102, 102)
        for item in plan.copies: target_store[item.destination_slot] = source_store[item.source_slot]
        target.publish_restore(plan)
        self.assertEqual(target.committed_end, 103)
        for p in range(98, 103): self.assert_payload(target, target_store, p, 102)
        self.assertEqual(target.export_snapshot(), snapshot)

    def test_pd_rejects_identity_format_window_holes_duplicates_and_old_prefix(self):
        source, storage = RecentKVState("a", 3, 4, 4), {}
        seed_payload(source, 17, storage)
        original = source.export_snapshot()
        mutations = [
            lambda x: x.update(request_id="b"),
            lambda x: x.update(generation=2),
            lambda x: x.update(generation=True),
            lambda x: x.update(window=8),
            lambda x: x.update(ring_capacity=8),
            lambda x: x.update(version=2),
            lambda x: x.update(metadata_only=False),
            lambda x: x["format"].update(hot_dtype="bf16"),
            lambda x: x["format"].update(rope_dim=0),
            lambda x: x["entries"].pop(),
            lambda x: x["entries"][1].update(position=13),
            lambda x: x["entries"][1].update(slot=3),
            lambda x: x.update(committed_end=16),
        ]
        for mutate in mutations:
            snapshot = copy.deepcopy(original)
            mutate(snapshot)
            with self.subTest(snapshot=snapshot):
                with self.assertRaises(ValueError):
                    RecentKVState("a", 3, 4, 4, committed_end=17).prepare_restore(snapshot)

    def test_snapshot_export_rejects_pending_proposals(self):
        state = RecentKVState("a", 0, 4, 4)
        state.stage(1)
        with self.assertRaises(HotCacheStateError): state.export_snapshot()
        state.abort_step()
        self.assertEqual(state.export_snapshot()["entries"], [])

    def test_release_reuse_invalidates_old_plans_and_snapshots(self):
        state, storage = RecentKVState("a", 1, 3, 4), {}
        seed_payload(state, 8, storage)
        snapshot = state.export_snapshot()
        old_plan = state.stage(2)
        state.release()
        with self.assertRaises(HotCacheStateError): state.resolve(7, 7)
        with self.assertRaises(HotCacheStateError): state.publish_stage(old_plan)
        with self.assertRaises(ValueError): state.reuse("a", 1)
        state.reuse("b", 2)
        self.assertEqual(state.committed_end, 0)
        with self.assertRaises(ValueError): state.prepare_restore(snapshot)
        new_plan = state.stage(1)
        with self.assertRaises(HotCacheStateError): state.publish_stage(old_plan)
        self.assertEqual(new_plan.request_id, "b")

    def test_no_live_reseed_restore_or_unbounded_proposals(self):
        state, storage = RecentKVState("a", 0, 3, 4), {}
        with self.assertRaises(ValueError): state.stage(5)
        seed_payload(state, 8, storage)
        snapshot = state.export_snapshot()
        with self.assertRaises(HotCacheStateError): state.prepare_seed(8)
        with self.assertRaises(HotCacheStateError): state.prepare_restore(snapshot)
        with self.assertRaises(HotCacheStateError): state.reuse("b", 1)

    def test_randomized_acceptance_matches_payload_reference(self):
        rng = random.Random(20261120)
        for window in (1, 2, 7, 31):
            state, storage = RecentKVState("a", 0, window, 9), {}
            for _ in range(150):
                count = rng.randint(1, 9)
                plan = stage_payload(state, count, storage)
                for q in range(plan.start, plan.end):
                    for p in range(max(0, q-window), q+1):
                        self.assert_payload(state, storage, p, q)
                accepted = rng.randint(0, count)
                commit_payload(state, accepted, storage)
                if state.committed_end:
                    q = state.committed_end - 1
                    for p in range(max(0, q-window+1), q+1):
                        self.assert_payload(state, storage, p, q)

    def test_input_validation(self):
        for kwargs in (dict(window=-1), dict(window=True), dict(max_step=0),
                       dict(hot_dtype="fp4"), dict(latent_dim=511),
                       dict(generation=-1), dict(request_id=""), dict(rope_dim=-1)):
            args = dict(request_id="a", generation=0, window=4, max_step=4)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): RecentKVState(**args)
        state = RecentKVState("a", 0, 4, 4)
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): state.stage(count)
        stage_payload(state, 2, {})
        with self.assertRaises(ValueError): state.plan_commit(3)


if __name__ == "__main__":
    unittest.main()
