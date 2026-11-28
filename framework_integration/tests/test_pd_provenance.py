"""The same selected TP peer must own MR, control metadata and RDMA session."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfa_recent128.pd_mooncake import LegacyTailBridge


class ProvenanceTests(unittest.TestCase):
    def bridge(self):
        bridge=object.__new__(LegacyTailBridge)
        engine=NS(get_rpc_port=lambda:20598)
        bridge.worker=NS(engine=engine,engine_id="P",te_rpc_port=20598,handshake_port=40000,tp_rank=0)
        bridge.device="npu:0"
        bridge.arena=NS(engine=engine,registration_result=0,registered_extent={"pointer":2**21,"nbytes":2**21})
        return bridge

    def test_source_live_engine_and_metadata_must_match(self):
        bridge=self.bridge()
        module=NS(global_te=NS(transfer_engine=bridge.worker.engine))
        with patch("sfa_recent128.pd_mooncake.importlib.import_module",return_value=module):
            self.assertEqual(bridge.validate_source_metadata(dict(engine_id="P",te_rpc_port=20598,handshake_port=40000))["live_rpc_port"],20598)
            with self.assertRaisesRegex(RuntimeError,"SOURCE_BINDING_MISMATCH"):
                bridge.validate_source_metadata(dict(engine_id="P",te_rpc_port=15429,handshake_port=40000))

    def test_source_global_engine_alias_is_verified(self):
        bridge=self.bridge()
        with patch("sfa_recent128.pd_mooncake.importlib.import_module",return_value=NS(global_te=NS(transfer_engine=object()))):
            with self.assertRaisesRegex(RuntimeError,"ENGINE_MISMATCH"):
                bridge.source_identity()

    def test_selected_peer_uses_its_native_port_not_another_rank(self):
        bridge=self.bridge()
        thread=NS(remote_metadata_lock=threading.RLock(),remote_te_port={"P":{40000:20598,40007:15429}})
        peer=dict(remote_host="P-host",remote_engine_id="P",remote_handshake_port=40000,
                  recent128_source_identity=dict(live_rpc_port=20598))
        self.assertEqual(bridge.selected_session(thread,peer,dict(te_rpc_port=20598)),"P-host:20598")
        with self.assertRaisesRegex(RuntimeError,"SELECTED_PEER_RPC_MISMATCH"):
            bridge.selected_session(thread,peer,dict(te_rpc_port=15429))


if __name__=="__main__":unittest.main()
