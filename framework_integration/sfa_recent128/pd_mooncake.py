"""BF16 tails on the existing VA legacy Mooncake child control socket.

The composite and its Store sibling are untouched. Source snapshots are pinned
before finished-bank release; destination import precedes native receive-done.
This is an A5 candidate: host protocol tests do not validate the NPU transport.
"""
from functools import wraps
import importlib
import inspect
import threading
import time

from .pd_connector import LEGACY_MODULE, is_serving_config, iter_legacy_connectors
from .pd_tail import PDSidecarPublisher, PDSidecarReceiver, TailDescriptor
from .pd_worker import WorkerTailRegistry, validate_tail

_BRIDGES = {}
_EXT = "recent128_tail_v1"


def _require(value, message):
    if not value:
        raise RuntimeError(message)


class ControlSocket:
    """Extend GET_META replies only; native metadata and DONE frames survive."""
    def __init__(self, socket, bridge, codec, get_meta):
        self.socket, self.bridge, self.codec, self.get_meta = socket, bridge, codec, get_meta
        self.command = None

    def __getattr__(self, name):
        return getattr(self.socket, name)

    def recv_multipart(self, *args, **kwargs):
        frames = self.socket.recv_multipart(*args, **kwargs)
        self.command = None
        payload = [frame for frame in frames[1:] if frame != b""]
        if len(payload) == 1:
            message = self.codec.decode(payload[0])
            if len(message) >= 2 and message[0] == self.get_meta and isinstance(message[1], dict):
                command = message[1]
                if command.get("protocol") == _EXT:
                    self.command = command
        return frames

    def send_multipart(self, frames, *args, **kwargs):
        if self.command is not None:
            metadata = self.codec.decode(frames[-1])
            _require(isinstance(metadata, dict), "Native Mooncake metadata must remain a mapping")
            try:
                metadata[_EXT] = self.bridge.control(self.command)
            except Exception as error:
                import logging
                logging.getLogger(__name__).exception("Recent128 BF16-tail control failed")
                metadata[_EXT] = dict(status="ERROR", error_type=type(error).__name__)
            frames = (*frames[:-1], self.codec.encode(metadata))
            self.command = None
        return self.socket.send_multipart(frames, *args, **kwargs)


class LegacyTailBridge:
    def __init__(self, worker):
        import torch
        self.worker = worker
        self.device = torch.device("npu", torch.npu.current_device())
        self.lock = threading.RLock()
        self.pending = {}
        self.incoming = {}
        self.publishers = {}
        self.receivers = {}
        self.completed = set()
        self.acknowledged = []

    def state(self):
        from .contract import parse_recent128_config
        from .serving_runtime import get_state
        vc = self.worker.vllm_config
        return get_state(vc, parse_recent128_config(vc.additional_config),
                         self.device)

    def completion(self, state):
        # Explicit transfer boundary, never layer forward or Graph replay.
        # Drain all producer streams before an immutable publication is made.
        t = state.torch
        t.npu.synchronize(state.device)
        event = t.npu.Event()
        event.record(t.npu.current_stream(state.device))
        event.synchronize()
        return event

    def pin(self, request_id, end):
        with self.lock:
            if request_id in self.pending:
                _require(self.pending[request_id].end == end, "Changed committed PD boundary")
                return
            state = self.state()
            _require(len(self.pending) < state.request_pool_size,
                     "BF16 PD sidecar pool full: unfinished transfers must be acknowledged")
            with state.torch.npu.device(state.device):
                snapshots = WorkerTailRegistry((state,)).export_request(request_id,
                    committed_end=end, completion=self.completion(state))
            _require(len(snapshots) == 1, "Expected one local recent128 pool owner")
            self.pending[request_id] = snapshots[0]

    def control(self, command):
        # Runs in the native source control thread; no extra listener/port.
        request_id = command.get("request_id")
        with self.lock:
            if command.get("action") == "ack":
                identity = (request_id, command.get("nonce"))
                if identity in self.acknowledged:
                    return dict(status="ACK")
                publisher = self.publishers.get(request_id)
                accepted = publisher is not None and publisher.acknowledge(request_id, command.get("nonce"))
                if accepted:
                    self.publishers.pop(request_id, None)
                    self.pending.pop(request_id, None)
                    self.acknowledged.append(identity)
                    del self.acknowledged[:-256]
                return dict(status="ACK" if accepted else "STALE_ACK")
            _require(command.get("action") == "describe", "Unknown BF16 PD control action")
            snapshot = self.pending.get(request_id)
            if snapshot is None:
                return dict(status="NOT_READY")
            state = self.state()
            with state.torch.npu.device(state.device):
                publisher = self.publishers.get(request_id)
                if publisher is None:
                    _require(callable(getattr(self.worker.engine, "unregister_memory", None)),
                             "BF16 PD adapter requires real unregister_memory for bounded registrations")
                    def validate(rid, value):
                        validate_tail(state, value)
                        self.completion(state)
                        return True
                    publisher = PDSidecarPublisher(self.worker.engine,
                        str(self.worker.engine_id), validate_snapshot=validate,
                        rank=int(self.worker.tp_rank), expected_layers=state.layers)
                    self.publishers[request_id] = publisher
                publication = publisher._active.get(request_id)
                descriptor = (publication.descriptor if publication is not None else
                              publisher.publish(request_id, snapshot))
            return dict(status="READY", descriptor=descriptor.to_dict())

    def discard_unrequested(self, request_ids):
        with self.lock:
            for request_id in request_ids:
                # No descriptor has escaped; no remote reader has its addresses.
                if request_id not in self.publishers:
                    self.pending.pop(request_id, None)

    def prepare_receive(self, metadata):
        with self.lock:
            for request_id, meta in metadata.requests.items():
                if getattr(meta, "do_virtual", False):
                    continue
                end = getattr(meta, "recent128_end", None)
                _require(type(end) is int and end > 0,
                         "P transfer parameters lack the actual recent128_end; update both P and D")
                self.incoming.setdefault(request_id, dict(end=end, peers={}))

    def note_pull(self, thread, meta):
        with self.lock:
            plan = self.incoming.get(meta["request_id"])
            _require(plan is not None, "BF16 PD receive has no scheduler boundary")
            key = (meta["remote_host"], meta["remote_handshake_port"])
            if key in plan["peers"]:
                return
            # Pin the remote descriptor BEFORE native C4 can send its per-shard
            # DONE. The source must not reclaim a tail before our final import.
            peer = dict(meta)
            command = dict(protocol=_EXT, action="describe", request_id=peer["remote_request_id"])
            deadline = time.monotonic() + min(float(getattr(thread, "timeout", 30)), 30.)
            while True:
                metadata, response = self.exchange(thread, peer, command)
                if response.get("status") != "NOT_READY":
                    break
                _require(time.monotonic() < deadline, "Source has not pinned the committed BF16 tail")
                time.sleep(.01)
            _require(response.get("status") == "READY", "BF16 source did not publish a tail")
            descriptor = TailDescriptor.from_dict(response["descriptor"])
            _require(descriptor.request_id == peer["remote_request_id"] and descriptor.end == plan["end"],
                     "BF16 PD request or committed boundary mismatch")
            plan["peers"][key] = (peer, metadata, descriptor)

    def exchange(self, thread, peer, command):
        module = importlib.import_module(LEGACY_MODULE)
        sock = None
        host, port = peer["remote_host"], peer["remote_handshake_port"]
        try:
            sock = thread._get_remote_socket(host, port)
            codec = module.msgspec.msgpack
            module.ensure_zmq_send(sock, codec.encode((module.GET_META_MSG, command)), f"{host}:{port}")
            response = codec.decode(module.ensure_zmq_recv(sock, f"{host}:{port}"))
            _require(_EXT in response, "Source Mooncake worker lacks BF16-tail adapter")
            return response, response[_EXT]
        except Exception:
            if sock is not None:
                sock.close()
                sock = None
            raise
        finally:
            if sock is not None:
                thread._return_remote_socket(sock, host, port)

    def finish_receive(self, thread, request_id):
        with self.lock:
            if request_id in self.completed:
                return
            plan = self.incoming.get(request_id)
            _require(plan is not None and bool(plan["peers"]), "Missing BF16 PD source peer")
            # Native exchanged MLA latent/RoPE are TP replicated. Pull one full
            # local-layer tail; retain every source until the final import.
            peer, metadata, descriptor = next(iter(plan["peers"].values()))
            state = self.state()
            session = f"{peer['remote_host']}:{metadata['te_rpc_port']}"
            receiver = self.receivers.get(session)
            if receiver is None:
                receiver = self.receivers[session] = PDSidecarReceiver(self.worker.engine, session,
                    device=state.device, torch_module=state.torch,
                    completion_factory=lambda desc, tail: self.completion(state),
                    validate_received=lambda rid, tail: (validate_tail(state, tail) is tail))
            with state.torch.npu.device(state.device):
                snapshot = receiver.receive(descriptor, state=state)
                WorkerTailRegistry((state,)).receive_request(request_id, (snapshot,),
                    completion=snapshot.completion, source="pd")
                self.completion(state)  # Import writes finish before ACK/source release.
            for source_peer, _, source_desc in plan["peers"].values():
                # Unselected TP replicas were never read; selected read and
                # import have completed. All source pointers are now quiescent.
                _, ack = self.exchange(thread, source_peer, dict(protocol=_EXT, action="ack",
                    request_id=source_desc.request_id, nonce=source_desc.nonce))
                _require(ack.get("status") == "ACK", "Source did not acknowledge exact BF16 import nonce")
            receiver.release_received(descriptor.request_id, descriptor.nonce, rank=descriptor.rank)
            self.completed.add(request_id)
            self.incoming.pop(request_id, None)


def pin_finished_tails(group, metadata):
    """Runner calls this before releasing finished requests, outside Graph."""
    for connector, child_meta in iter_legacy_connectors(group, metadata):
        worker = getattr(connector, "connector_worker", None)
        bridge = getattr(worker, "_recent128_tail_bridge", None)
        sends = getattr(child_meta, "recent128_sends", {})
        if sends:
            _require(bridge is not None, "Actual Mooncake child lacks the BF16-tail worker adapter")
            for request_id, end in sends.items():
                bridge.pin(request_id, end)


def install_worker_hooks(module=None):
    module = module or importlib.import_module(LEGACY_MODULE)
    worker_cls = module.MooncakeConnectorWorker
    if getattr(worker_cls, "_recent128_tail_transport", False):
        return
    original_init = worker_cls.__init__
    @wraps(original_init)
    def init(worker, *args, **kwargs):
        original_init(worker, *args, **kwargs)
        if is_serving_config(worker.vllm_config):
            bridge = worker._recent128_tail_bridge = LegacyTailBridge(worker)
            _BRIDGES[id(worker.vllm_config)] = bridge
    worker_cls.__init__ = init

    original_register = worker_cls.register_kv_caches
    @wraps(original_register)
    def register(worker, *args, **kwargs):
        # init_ascend_config can install us from inside the original worker
        # constructor. Ensure THIS instance is wired before threads are born.
        if is_serving_config(worker.vllm_config):
            bridge = getattr(worker, "_recent128_tail_bridge", None)
            if bridge is None:
                bridge = worker._recent128_tail_bridge = LegacyTailBridge(worker)
            _BRIDGES[id(worker.vllm_config)] = bridge
        return original_register(worker, *args, **kwargs)
    worker_cls.register_kv_caches = register

    for cls in (module.KVCacheSendingThread, module.KVCacheRecvingThread):
        def wrap_constructor(original):
            signature = inspect.signature(original)
            @wraps(original)
            def construct(thread, *args, **kwargs):
                arguments = signature.bind(thread, *args, **kwargs).arguments
                original(thread, *args, **kwargs)
                thread._recent128_tail_bridge = _BRIDGES.get(id(arguments.get("vllm_config")))
            return construct
        cls.__init__ = wrap_constructor(cls.__init__)

    original_loop = module.KVCacheSendingThread.run_busy_loop
    @wraps(original_loop)
    def run_busy_loop(thread, sock):
        bridge = getattr(thread, "_recent128_tail_bridge", None)
        return original_loop(thread, ControlSocket(sock, bridge, module.msgspec.msgpack,
                            module.GET_META_MSG) if bridge is not None else sock)
    module.KVCacheSendingThread.run_busy_loop = run_busy_loop

    original_finished = module.KVCacheSendingThread.get_and_clear_finished_requests
    @wraps(original_finished)
    def get_finished(thread):
        result = original_finished(thread)
        bridge = getattr(thread, "_recent128_tail_bridge", None)
        if bridge is not None:
            bridge.discard_unrequested(result)
        return result
    module.KVCacheSendingThread.get_and_clear_finished_requests = get_finished

    original_start = worker_cls.start_load_kv
    @wraps(original_start)
    def start_load(worker, metadata):
        bridge = getattr(worker, "_recent128_tail_bridge", None)
        if bridge is not None and worker.kv_role == "kv_consumer":
            bridge.prepare_receive(metadata)
        return original_start(worker, metadata)
    worker_cls.start_load_kv = start_load

    original_pull = module.KVCacheRecvingThread._transfer_kv_cache_all_groups
    @wraps(original_pull)
    def pull(thread, req_meta):
        bridge = getattr(thread, "_recent128_tail_bridge", None)
        if bridge is not None:
            bridge.note_pull(thread, req_meta)  # Full C4 prefix hits also need a tail.
        return original_pull(thread, req_meta)
    module.KVCacheRecvingThread._transfer_kv_cache_all_groups = pull

    original_reformat = module.KVCacheRecvingThread._reformat_pending_kv_caches
    @wraps(original_reformat)
    def reformat(thread, request_id):
        result = original_reformat(thread, request_id)
        bridge = getattr(thread, "_recent128_tail_bridge", None)
        if bridge is not None:
            bridge.finish_receive(thread, request_id)
        return result
    module.KVCacheRecvingThread._reformat_pending_kv_caches = reformat
    worker_cls._recent128_tail_transport = True


def install():
    from .pd_connector import install_connector_hooks
    result = install_connector_hooks()
    install_worker_hooks()
    return result
