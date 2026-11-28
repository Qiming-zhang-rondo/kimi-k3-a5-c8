"""BF16 tails on the existing VA legacy Mooncake child control socket.

The composite and its Store sibling are untouched. Source snapshots are pinned
before finished-bank release; destination import precedes native receive-done.
This is an A5 candidate: host protocol tests do not validate the NPU transport.
"""
from functools import wraps
import importlib
import inspect
import logging
import os
import secrets
import threading
import time

from .pd_connector import LEGACY_MODULE, is_serving_config, iter_legacy_connectors
from .pd_arena import Arena
from .pd_lifetime import PublicationTracker
from .pd_tail import PD_TAIL_VERSION, TensorDescriptor, TailDescriptor
from .pd_worker import WorkerTailRegistry, validate_tail
from .serving_state import TailSnapshot

_BRIDGES = {}
_EXT = "recent128_tail_v2"
_FIELDS = ("latent", "rope", "position_tags", "epoch_tags")
_LOGGER = logging.getLogger(__name__)


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
                validate = getattr(self.bridge, "validate_source_metadata", None)
                if callable(validate):
                    validate(metadata)
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
        self.descriptors = {}
        self.generations = {}
        self.arena = None
        self.lifetime = None
        self.completed = set()
        self.successful_reads = 0
        self.failed_reads = 0

    def diagnostics(self):
        extent = self.arena.registered_extent if self.arena is not None else {}
        return dict(protocol=_EXT, startup_arena_registered=self.arena is not None,
            arena_bytes=extent.get("nbytes"), arena_slots=extent.get("slots"),
            arena_layers=extent.get("layers"), pending_sources=len(self.pending),
            successful_reads=self.successful_reads, failed_reads=self.failed_reads,
            completed_imports=len(self.completed))

    def source_identity(self):
        worker = self.worker
        _require(self.arena is not None and self.arena.engine is worker.engine,
                 "RECENT128_PD_ENGINE_MISMATCH arena/worker engines differ")
        module = importlib.import_module(LEGACY_MODULE)
        global_engine = getattr(getattr(module, "global_te", None), "transfer_engine", None)
        _require(global_engine is None or global_engine is worker.engine,
                 "RECENT128_PD_ENGINE_MISMATCH native global_te and worker engines differ")
        live_rpc = int(worker.engine.get_rpc_port())
        _require(live_rpc == int(worker.te_rpc_port),
                 f"RECENT128_PD_RPC_MISMATCH live={live_rpc} worker_cached={worker.te_rpc_port}")
        return dict(pid=os.getpid(), engine_id=str(worker.engine_id),
            engine_object=id(worker.engine), tp_rank=int(worker.tp_rank), device=str(self.device),
            handshake_port=int(worker.handshake_port), live_rpc_port=live_rpc,
            arena_register_result=self.arena.registration_result,
            arena=dict(self.arena.registered_extent))

    def validate_source_metadata(self, metadata):
        identity = self.source_identity()
        _require(str(metadata.get("engine_id")) == identity["engine_id"] and
                 metadata.get("te_rpc_port") == identity["live_rpc_port"] and
                 metadata.get("handshake_port") == identity["handshake_port"],
                 "RECENT128_PD_SOURCE_BINDING_MISMATCH " +
                 f"pid={identity['pid']} tp={identity['tp_rank']} " +
                 f"live_rpc={identity['live_rpc_port']} metadata_rpc={metadata.get('te_rpc_port')} " +
                 f"worker_handshake={identity['handshake_port']} metadata_handshake={metadata.get('handshake_port')}")
        return identity

    def prepare_arena(self):
        """Must precede native KV registration, thread startup and peer links."""
        with self.lock:
            if self.arena is not None:
                return
            state = self.state()
            with state.torch.npu.device(state.device):
                self.arena = Arena(self.worker.engine, state.torch, state.device,
                    tuple(state.layers), state.request_pool_size)
            self.lifetime = PublicationTracker(state.request_pool_size)
            import logging
            logging.getLogger(__name__).info(
                "Recent128 BF16 PD arena registered before native KV startup: bytes=%d slots=%d layers=%d",
                self.arena.registered_extent["nbytes"], state.request_pool_size, len(state.layers))

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
            _require(self.arena is not None, "BF16 PD arena must be registered before native KV startup")
            with state.torch.npu.device(state.device):
                snapshots = WorkerTailRegistry((state,)).export_request(request_id,
                    committed_end=end, completion=self.completion(state))
                _require(len(snapshots) == 1, "Expected one local recent128 pool owner")
                snapshot = snapshots[0]
                layers = self.arena.acquire(request_id, snapshot.end - snapshot.start)
                try:
                    for layer, source in snapshot.layers.items():
                        for field in _FIELDS:
                            getattr(layers[layer], field).copy_(getattr(source, field))
                    owned = TailSnapshot(snapshot.start, snapshot.end, snapshot.source_epoch, layers)
                    validate_tail(state, owned)
                    self.completion(state)
                    self.lifetime.add(request_id)
                except BaseException:
                    self.completion(state)
                    self.arena.release(request_id)
                    raise
            self.pending[request_id] = owned

    def _reclaim(self, request_id):
        if request_id in self.pending and self.lifetime.releasable(request_id):
            self.arena.release(request_id)
            self.lifetime.release(request_id)
            self.pending.pop(request_id, None)
            self.descriptors.pop(request_id, None)

    def control(self, command):
        # Runs in the native source control thread; no extra listener/port.
        request_id = command.get("request_id")
        receiver_id = command.get("receiver_id")
        _require(isinstance(receiver_id, str) and bool(receiver_id), "BF16 control requires receiver identity")
        with self.lock:
            if command.get("action") == "ack":
                accepted = self.lifetime.ack(request_id, receiver_id, command.get("nonce"))
                if accepted:
                    self._reclaim(request_id)
                return dict(status="ACK" if accepted else "STALE_ACK")
            _require(command.get("action") == "describe", "Unknown BF16 PD control action")
            snapshot = self.pending.get(request_id)
            if snapshot is None:
                return dict(status="NOT_READY")
            descriptor = self.descriptors.get(request_id)
            if descriptor is None:
                generation = self.generations.get(request_id, 0) + 1
                buffers = []
                for layer, tail in snapshot.layers.items():
                    for field in _FIELDS:
                        tensor = getattr(tail, field)
                        nbytes = tensor.numel() * tensor.element_size()
                        _require(self.arena.contains(tensor.data_ptr(), nbytes),
                                 "BF16 descriptor escaped startup registration")
                        buffers.append(TensorDescriptor(layer, field,
                            "bfloat16" if field in ("latent", "rope") else "int64",
                            tuple(tensor.shape), nbytes, tensor.data_ptr()))
                descriptor = TailDescriptor(PD_TAIL_VERSION, request_id, snapshot.source_epoch,
                    snapshot.start, snapshot.end, int(self.worker.tp_rank), generation,
                    secrets.token_hex(16), tuple(buffers)).validate()
                self.generations[request_id] = generation
                self.descriptors[request_id] = descriptor
            subscribed = self.lifetime.subscribe(request_id, receiver_id, descriptor.nonce)
            identity = self.source_identity()
            if subscribed:
                extent = self.arena.registered_extent
                _LOGGER.info("RECENT128_PD_SOURCE_READY pid=%d device=%s tp=%d handshake=%d live_rpc=%d "
                    "engine=%d arena=%#x arena_bytes=%d register_ret=%d src0=%#x bytes0=%d",
                    identity["pid"], identity["device"], identity["tp_rank"], identity["handshake_port"],
                    identity["live_rpc_port"], identity["engine_object"], extent["pointer"], extent["nbytes"],
                    identity["arena_register_result"], descriptor.buffers[0].pointer, descriptor.buffers[0].nbytes)
            return dict(status="READY", descriptor=descriptor.to_dict(),
                        registered_extent=dict(self.arena.registered_extent),
                        source_identity=identity)

    def discard_unrequested(self, request_ids):
        with self.lock:
            for request_id in request_ids:
                # Every reader subscribes before its C4 DONE. Native aggregate
                # completion closes the subscriber set; all exact ACKs gate reuse.
                if self.lifetime is not None:
                    self.lifetime.native_done(request_id)
                    self._reclaim(request_id)

    def prepare_receive(self, metadata):
        with self.lock:
            for request_id, meta in metadata.requests.items():
                if getattr(meta, "do_virtual", False):
                    continue
                end = getattr(meta, "recent128_end", None)
                _require(type(end) is int and end > 0,
                         "P transfer parameters lack the actual recent128_end; update both P and D")
                self.incoming.setdefault(request_id, dict(end=end, peers={},
                    receiver_id=f"{self.worker.engine_id}:{self.worker.handshake_port}:{request_id}"))

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
            command = dict(protocol=_EXT, action="describe", request_id=peer["remote_request_id"],
                           receiver_id=plan["receiver_id"])
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
            extent = response.get("registered_extent", {})
            base, length = extent.get("pointer"), extent.get("nbytes")
            _require(type(base) is int and type(length) is int and length > 0 and
                     base % (2 * 1024 * 1024) == 0 and length % (2 * 1024 * 1024) == 0,
                     "Source lacks a startup-registered aligned BF16 arena; update and restart both peers")
            _require(all(base <= b.pointer and b.pointer + b.nbytes <= base + length
                         for b in descriptor.buffers), "Source descriptor is outside its registered arena")
            _require(metadata.get("engine_id") == peer["remote_engine_id"], "BF16 source engine identity mismatch")
            source = response.get("source_identity", {})
            _require(source.get("arena_register_result") == 0 and
                     source.get("handshake_port") == peer["remote_handshake_port"] and
                     source.get("live_rpc_port") == metadata.get("te_rpc_port"),
                     "RECENT128_PD_SOURCE_PROVENANCE_MISMATCH: source registration/control engine differ")
            peer["recent128_source_identity"] = source
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

    def selected_session(self, thread, peer, metadata):
        # Compare THIS exact native peer, not another TP rank's log line.
        remote_engine, remote_handshake = peer["remote_engine_id"], peer["remote_handshake_port"]
        with thread.remote_metadata_lock:
            native_rpc = thread.remote_te_port.get(remote_engine, {}).get(remote_handshake)
        if native_rpc is None:
            thread._get_remote_metadata(peer["remote_host"], remote_handshake)
            with thread.remote_metadata_lock:
                native_rpc = thread.remote_te_port[remote_engine][remote_handshake]
        identity = peer["recent128_source_identity"]
        _require(native_rpc == metadata["te_rpc_port"] == identity["live_rpc_port"],
            "RECENT128_PD_SELECTED_PEER_RPC_MISMATCH " +
            f"handshake={remote_handshake} native_rpc={native_rpc} " +
            f"control_rpc={metadata['te_rpc_port']} source_live_rpc={identity['live_rpc_port']}")
        return f"{peer['remote_host']}:{native_rpc}"

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
            session = self.selected_session(thread, peer, metadata)
            native_rpc = metadata["te_rpc_port"]
            remote_handshake = peer["remote_handshake_port"]
            source_identity = peer["recent128_source_identity"]
            _require(self.arena is not None, "Destination BF16 arena was not registered at startup")
            _require(descriptor.layer_keys == frozenset(state.layers), "BF16 source must cover every local layer")
            slot_key = (request_id, descriptor.nonce)
            with state.torch.npu.device(state.device):
                layers = self.arena.acquire(slot_key, descriptor.end - descriptor.start)
                arrays = [getattr(layers[buffer.layer], buffer.field) for buffer in descriptor.buffers]
                _require(all(self.arena.contains(tensor.data_ptr(), buffer.nbytes)
                             for tensor, buffer in zip(arrays, descriptor.buffers)),
                         "Destination payload escaped startup registration")
                try:
                    _LOGGER.info("RECENT128_PD_READ_BEGIN pid=%d device=%s remote=%s handshake=%d native_rpc=%d "
                        "source_pid=%d source_tp=%d source_rpc=%d src0=%#x dst0=%#x bytes0=%d buffers=%d",
                        os.getpid(), state.device, session, remote_handshake, native_rpc,
                        source_identity["pid"], source_identity["tp_rank"], source_identity["live_rpc_port"],
                        descriptor.buffers[0].pointer, arrays[0].data_ptr(),
                        descriptor.buffers[0].nbytes, len(arrays))
                    result = self.worker.engine.batch_transfer_sync_read(session,
                        [buffer.pointer for buffer in descriptor.buffers],
                        [tensor.data_ptr() for tensor in arrays],
                        [buffer.nbytes for buffer in descriptor.buffers])
                    _require(type(result) is int and result == 0, "BF16 registered-arena RDMA read failed: " + repr(result))
                    _LOGGER.info("RECENT128_PD_READ_END pid=%d remote=%s ret=%d", os.getpid(), session, result)
                except BaseException:
                    self.failed_reads += 1
                    # A failed transport may still own the destination slot.
                    # Keep it pinned rather than risking concurrent slot reuse.
                    raise
                self.successful_reads += 1
                completion = self.completion(state)
                snapshot = TailSnapshot(descriptor.start, descriptor.end, descriptor.source_epoch,
                                        layers, completion)
                try:
                    WorkerTailRegistry((state,)).receive_request(request_id, (snapshot,),
                        completion=completion, source="pd")
                finally:
                    self.completion(state)  # No reuse before import copies drain.
                    self.arena.release(slot_key)
            for source_peer, _, source_desc in plan["peers"].values():
                # Unselected TP replicas were never read; selected read and
                # import have completed. All source pointers are now quiescent.
                _, ack = self.exchange(thread, source_peer, dict(protocol=_EXT, action="ack",
                    request_id=source_desc.request_id, nonce=source_desc.nonce,
                    receiver_id=plan["receiver_id"]))
                _require(ack.get("status") == "ACK", "Source did not acknowledge exact BF16 import nonce")
            self.completed.add(request_id)
            self.incoming.pop(request_id, None)
            import logging
            logging.getLogger(__name__).info("Recent128 BF16 PD tail imported: tokens=%d layers=%d",
                descriptor.end - descriptor.start, len(layers))


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

    def ensure_early_arena(worker):
        if worker is None or not is_serving_config(worker.vllm_config):
            return
        bridge = getattr(worker, "_recent128_tail_bridge", None)
        if bridge is None:
            bridge = worker._recent128_tail_bridge = LegacyTailBridge(worker)
        _BRIDGES[id(worker.vllm_config)] = bridge
        # The exact c09 worker constructs connectors after model weight loading.
        # Do this before Multi starts constructing its Store sibling: Store
        # setup may already establish links on the shared transfer engine.
        fresh = getattr(bridge, "arena", None) is None
        bridge.prepare_arena()
        if fresh:
            _LOGGER.info("RECENT128_PD_EARLY_ARENA pid=%d before_store_constructor=1", os.getpid())

    @wraps(original_init)
    def init(worker, *args, **kwargs):
        original_init(worker, *args, **kwargs)
        ensure_early_arena(worker)
    worker_cls.__init__ = init

    facade_cls = getattr(module, "MooncakeConnector", None)
    if facade_cls is not None:
        original_facade_init = facade_cls.__init__
        original_facade_setattr = facade_cls.__setattr__

        @wraps(original_facade_setattr)
        def facade_setattr(connector, name, value):
            original_facade_setattr(connector, name, value)
            if name == "connector_worker" and value is not None:
                # First installation can happen inside the original worker
                # constructor, after the original facade constructor entered.
                # Its next assignment still dispatches through this new slot.
                ensure_early_arena(value)

        @wraps(original_facade_init)
        def facade_init(connector, *args, **kwargs):
            original_facade_init(connector, *args, **kwargs)
            ensure_early_arena(getattr(connector, "connector_worker", None))

        facade_cls.__setattr__ = facade_setattr
        facade_cls.__init__ = facade_init

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
            bridge.prepare_arena()
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
