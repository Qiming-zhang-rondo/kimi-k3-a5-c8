"""Host checks for real scheduler boundary and Multi child metadata wiring."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfa_recent128 import pd_connector


def config(enabled=True):
    return SimpleNamespace(additional_config={"sfa_recent128": dict(
        window=128, execution="serving", preprocess="preserve", reader="single_pass",
        single_pass_kernel="batched_v4", library_dir="/tmp/lib", source_root="/tmp/src",
        diagnostics_path="/tmp/report")} if enabled else {})


def classes():
    class Metadata:
        def __init__(self):
            self.requests = {}
            self.requests_to_send = {}

        def add_new_req(self, request_id, local_block_ids, num_external_tokens,
                        kv_transfer_params, local_full_block_ids=None):
            self.requests[request_id] = SimpleNamespace(remote_request_id=kv_transfer_params["remote_request_id"])
            return "native-add-result"

    class Scheduler:
        def __init__(self, enabled=True):
            self.vllm_config = config(enabled)
            self.send = {}
            self.recv = {}
            self.last_params = None
            self.last_meta = None

        def request_finished(self, request, block_ids):
            if not request.kv_transfer_params.get("do_remote_decode"):
                return False, None
            self.send[request.request_id] = 42.25
            self.last_params = dict(do_remote_prefill=True, do_remote_decode=False,
                                    remote_request_id=request.request_id, native_key="unchanged")
            return True, self.last_params

        def build_connector_meta(self, scheduler_output):
            self.last_meta = meta = Metadata()
            meta.requests_to_send = self.send
            self.send = {}
            for rid, params in self.recv.items():
                meta.add_new_req(request_id=rid, local_block_ids=([9],),
                    num_external_tokens=200, kv_transfer_params=params)
            self.recv.clear()
            return meta

    return Scheduler, Metadata


class PDConnectorTests(unittest.TestCase):
    def setUp(self):
        self.Scheduler, self.Metadata = classes()
        pd_connector.install_connector_hooks(self.Scheduler, self.Metadata)

    @staticmethod
    def request(rid="producer", end=199):
        return SimpleNamespace(request_id=rid, prompt_token_ids=list(range(end)),
                               kv_transfer_params={"do_remote_decode": True})

    def test_native_commit_round_trip_preserves_identity_and_timestamps(self):
        producer = self.Scheduler()
        delay, params = producer.request_finished(self.request(), ([1, 2],))
        self.assertTrue(delay)
        self.assertIs(params, producer.last_params)
        self.assertEqual(params["recent128_end"], 199)
        self.assertEqual(params["native_key"], "unchanged")
        meta = producer.build_connector_meta(SimpleNamespace(finished_req_ids={"producer"}))
        self.assertIs(type(meta), self.Metadata)
        self.assertIs(meta, producer.last_meta)
        self.assertEqual(meta.requests_to_send, {"producer": 42.25})
        self.assertEqual(meta.recent128_sends, {"producer": 199})
        self.assertEqual(producer._recent128_sends_pending, {})
        self.assertEqual(producer.build_connector_meta(None).recent128_sends, {})

        decoder = self.Scheduler()
        decoder.recv["decoder"] = params
        received = decoder.build_connector_meta(None)
        self.assertEqual(received.requests["decoder"].recent128_end, 199)
        self.assertEqual(received.requests["decoder"].remote_request_id, "producer")

    def test_off_and_add_outside_enabled_build_are_untouched(self):
        scheduler = self.Scheduler(enabled=False)
        _, params = scheduler.request_finished(self.request(), ([1],))
        self.assertNotIn("recent128_end", params)
        self.assertFalse(hasattr(scheduler, "_recent128_sends_pending"))
        scheduler.recv["off"] = dict(remote_request_id="p", recent128_end=9)
        meta = scheduler.build_connector_meta(None)
        self.assertFalse(hasattr(meta, "recent128_sends"))
        self.assertFalse(hasattr(meta.requests["off"], "recent128_end"))
        direct = self.Metadata()
        result = direct.add_new_req("direct", [], 0,
            dict(remote_request_id="p", recent128_end=4))
        self.assertEqual(result, "native-add-result")
        self.assertFalse(hasattr(direct.requests["direct"], "recent128_end"))

    def test_invalid_boundary_fails_and_build_context_is_restored(self):
        scheduler = self.Scheduler()
        scheduler.recv["bad"] = dict(remote_request_id="p", recent128_end=True)
        with self.assertRaisesRegex(RuntimeError, "committed integer"):
            scheduler.build_connector_meta(None)
        direct = self.Metadata()
        direct.add_new_req("direct", [], 0, dict(remote_request_id="p", recent128_end=4))
        self.assertFalse(hasattr(direct.requests["direct"], "recent128_end"))
        self.assertFalse(pd_connector._BUILDING.get())

    def test_install_is_idempotent_and_missing_send_boundary_fails(self):
        method = self.Scheduler.request_finished
        pd_connector.install_connector_hooks(self.Scheduler, self.Metadata)
        self.assertIs(method, self.Scheduler.request_finished)
        scheduler = self.Scheduler()
        scheduler.send["untracked"] = 42.25
        with self.assertRaisesRegex(RuntimeError, "lacks its recent128"):
            scheduler.build_connector_meta(None)

    def test_recursive_multi_selects_actual_legacy_and_pairs_exact_metadata(self):
        legacy_type = type("MooncakeConnector", (), {"__module__": pd_connector.LEGACY_MODULE})
        legacy, store = legacy_type(), SimpleNamespace()
        nested = SimpleNamespace(sub_connectors=[store, legacy])
        group = SimpleNamespace(_connectors=[SimpleNamespace(), nested])
        legacy_meta = self.Metadata()
        store_meta = object()
        metadata = SimpleNamespace(metadata=(object(), SimpleNamespace(metadata=(store_meta, legacy_meta))))
        self.assertEqual(list(pd_connector.iter_legacy_connectors(group, metadata)), [(legacy, legacy_meta)])
        self.assertEqual(list(pd_connector.iter_legacy_connectors(group)), [(legacy, None)])
        # A display name cannot turn an upstream connector into the VA child.
        fake = type("MooncakeConnector", (), {"__module__": "vllm.upstream"})()
        self.assertFalse(pd_connector.is_legacy_connector(fake))
        facade = SimpleNamespace(connector_worker=type("MooncakeConnectorWorker", (),
            {"__module__": pd_connector.LEGACY_MODULE})())
        self.assertTrue(pd_connector.is_legacy_connector(facade))

    def test_multi_mismatch_and_cycle_fail_instead_of_pairing_wrong_child(self):
        group = SimpleNamespace(_connectors=[SimpleNamespace()])
        with self.assertRaisesRegex(RuntimeError, "actual child count"):
            list(pd_connector.iter_connector_leaves(group, SimpleNamespace(metadata=())))
        group._connectors = [group]
        with self.assertRaisesRegex(RuntimeError, "cycle"):
            list(pd_connector.iter_connector_leaves(group))


if __name__ == "__main__":
    unittest.main()
