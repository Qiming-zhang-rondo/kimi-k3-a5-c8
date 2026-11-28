"""Existing socket/control semantics and real legacy hook order, host only."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfa_recent128 import pd_mooncake as pd
from sfa_recent128.pd_lifetime import PublicationTracker


class Codec:
    @staticmethod
    def encode(value):
        # JSON fixtures stand in for the native msgpack envelope in this test.
        return json.dumps(value).encode()
    @staticmethod
    def decode(value):
        return json.loads(value)


class Socket:
    def __init__(self, frames):self.frames=frames;self.sent=[]
    def recv_multipart(self):return self.frames
    def send_multipart(self, frames, **kwargs):self.sent.append((frames,kwargs))


class TransportHookTests(unittest.TestCase):
    def test_control_extension_keeps_native_metadata_and_routing(self):
        cmd=dict(protocol=pd._EXT,action="describe",request_id="remote")
        socket=Socket((b"identity",b"",Codec.encode(("get",cmd))))
        bridge=SimpleNamespace(control=lambda command:dict(status="READY",request=command["request_id"]))
        proxy=pd.ControlSocket(socket,bridge,Codec,"get")
        self.assertIs(proxy.recv_multipart(),socket.frames)
        original=dict(engine_id="native",te_rpc_port=123,kv_caches_base_addr=[[987]])
        proxy.send_multipart((b"identity",b"",Codec.encode(original)),flags=2)
        frames,kwargs=socket.sent[0]
        result=Codec.decode(frames[-1])
        self.assertEqual({key:result[key] for key in original},original)
        self.assertEqual(result[pd._EXT]["status"],"READY")
        self.assertEqual(frames[:-1],(b"identity",b""))
        self.assertEqual(kwargs,{"flags":2})

    def test_native_bootstrap_and_done_are_byte_preserved(self):
        for message in (("get",""),("done","r",{})):
            socket=Socket((b"id",b"",Codec.encode(message)))
            bridge=SimpleNamespace(control=lambda command:self.fail("native control was intercepted"))
            proxy=pd.ControlSocket(socket,bridge,Codec,"get");proxy.recv_multipart()
            response=(b"id",b"",b"ACK")
            proxy.send_multipart(response)
            self.assertIs(socket.sent[0][0],response)

    def test_control_failure_replies_without_exporting_exception_details(self):
        def fail(command):raise RuntimeError("private device path")
        socket=Socket((b"id",b"",Codec.encode(("get",dict(protocol=pd._EXT)))))
        proxy=pd.ControlSocket(socket,SimpleNamespace(control=fail),Codec,"get")
        proxy.recv_multipart();proxy.send_multipart((b"id",b"",Codec.encode(dict(engine_id="native"))))
        reply=Codec.decode(socket.sent[0][0][-1])
        self.assertEqual(reply[pd._EXT],dict(status="ERROR",error_type="RuntimeError"))

    def test_legacy_wrappers_pin_peer_before_c4_and_import_before_done(self):
        calls=[]
        bridge=SimpleNamespace(
            prepare_receive=lambda meta:calls.append("prepare"),
            note_pull=lambda thread,meta:calls.append("pin-source"),
            finish_receive=lambda thread,rid:calls.append("import-tail"),
            discard_unrequested=lambda ids:calls.append("source-gc"))
        class Worker:
            def __init__(self):pass
            def register_kv_caches(self, caches):return caches
            def start_load_kv(self, metadata):calls.append("native-start")
        class Send:
            def __init__(self,vllm_config):pass
            def run_busy_loop(self,socket):return socket
            def get_and_clear_finished_requests(self):return {"r"}
        class Receive:
            def __init__(self,vllm_config):pass
            def _transfer_kv_cache_all_groups(self,meta):calls.append("native-c4")
            def _reformat_pending_kv_caches(self,rid):calls.append("native-reformat")
        module=SimpleNamespace(MooncakeConnectorWorker=Worker,KVCacheSendingThread=Send,
            KVCacheRecvingThread=Receive,msgspec=SimpleNamespace(msgpack=Codec),GET_META_MSG="get")
        pd.install_worker_hooks(module)
        vc=object();pd._BRIDGES[id(vc)]=bridge
        try:
            recv=Receive(vc)
            recv._transfer_kv_cache_all_groups(dict(request_id="r"))
            recv._reformat_pending_kv_caches("r")
        finally:pd._BRIDGES.pop(id(vc),None)
        self.assertEqual(calls,["pin-source","native-c4","native-reformat","import-tail"])

    def test_pending_source_is_not_released_after_descriptor_escapes(self):
        bridge=object.__new__(pd.LegacyTailBridge)
        bridge.lock=__import__("threading").RLock()
        freed=[]
        bridge.arena=SimpleNamespace(release=lambda rid:freed.append(rid))
        bridge.pending={"unused":object(),"active":object()};bridge.descriptors={}
        bridge.lifetime=PublicationTracker(2)
        bridge.lifetime.add("unused");bridge.lifetime.add("active")
        bridge.lifetime.subscribe("active","D1","nonce")
        bridge.lifetime.subscribe("active","D2","nonce")
        bridge.discard_unrequested({"unused","active"})
        self.assertEqual(set(bridge.pending),{"active"})
        self.assertEqual(freed,["unused"])
        self.assertEqual(bridge.control(dict(action="ack",request_id="active",receiver_id="D1",nonce="nonce"))["status"],"ACK")
        self.assertIn("active",bridge.pending)
        bridge.control(dict(action="ack",request_id="active",receiver_id="D2",nonce="nonce"))
        self.assertNotIn("active",bridge.pending)
        self.assertEqual(freed,["unused","active"])

    def test_arena_is_prepared_before_native_registration_creates_threads(self):
        calls=[];vc=object()
        bridge=SimpleNamespace(prepare_arena=lambda:calls.append("aligned-arena-register"))
        class Worker:
            def __init__(self,config):self.vllm_config=config
            def register_kv_caches(self,caches):calls.append("native-register-and-start-threads")
            def start_load_kv(self,metadata):pass
        class Send:
            def __init__(self,vllm_config):pass
            def run_busy_loop(self,socket):pass
            def get_and_clear_finished_requests(self):return set()
        class Receive:
            def __init__(self,vllm_config):pass
            def _transfer_kv_cache_all_groups(self,meta):pass
            def _reformat_pending_kv_caches(self,rid):pass
        module=SimpleNamespace(MooncakeConnectorWorker=Worker,KVCacheSendingThread=Send,
            KVCacheRecvingThread=Receive,msgspec=SimpleNamespace(msgpack=Codec),GET_META_MSG="get")
        # Simulate installation from inside the CURRENT original constructor:
        # this existing instance never passes through our new init wrapper.
        worker=Worker(vc)
        pd.install_worker_hooks(module)
        try:
            with patch.object(pd,"is_serving_config",return_value=True),patch.object(pd,"LegacyTailBridge",return_value=bridge):
                worker.register_kv_caches({})
            self.assertIs(worker._recent128_tail_bridge,bridge)
            self.assertIs(pd._BRIDGES[id(vc)],bridge)
        finally:pd._BRIDGES.pop(id(vc),None)
        self.assertEqual(calls,["aligned-arena-register","native-register-and-start-threads"])

    def test_runner_boundary_finds_only_legacy_child(self):
        calls=[]
        bridge=SimpleNamespace(pin=lambda rid,end:calls.append((rid,end)))
        leaf=SimpleNamespace(connector_worker=SimpleNamespace(_recent128_tail_bridge=bridge))
        with patch.object(pd,"iter_legacy_connectors",return_value=iter(((leaf,SimpleNamespace(recent128_sends={"r":97})),))):
            pd.pin_finished_tails(object(),object())
        self.assertEqual(calls,[("r",97)])


if __name__=="__main__":unittest.main()
