"""Production-path wiring; device validation remains an explicit deployment gate.

Unlike the accuracy harness, this path never reads lengths/TopK/tags to CPU,
changes the selected preprocessing path, or performs a mathematical reference
inside forward. The vendor producer must be built in the internal environment.
"""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import json
import os
import re

from .contract import validate_runtime_contract
from .serving_prolog import install_prolog_route, prolog_scope
from .serving_state import ServingState

_STATES = {}


@lru_cache(maxsize=4)
def _preflight_serving(source_root, library_dir):
    """Host-only capability symbols, before loading the model's weight shards."""
    import ctypes
    library = Path(library_dir).expanduser().resolve() / "libqsfa_recent128_direct.so"
    root = Path(source_root).expanduser().resolve()
    if not library.is_file() or not (root / "experimental/single_pass/batched_serving_bridge.cpp").is_file():
        raise RuntimeError("Build the isolated batched_v4 reader and point source_root/library_dir at it before loading model weights")
    producer = install_prolog_route()
    try:
        handle = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
        abi = handle.QsfaBatchedServingAbiVersion
        size = handle.QsfaBatchedServingBuffersBytes
        abi.argtypes = size.argtypes = []
        abi.restype = size.restype = ctypes.c_uint32
        if abi() != 1 or size() != 160:
            raise RuntimeError("batched_v4 library requires serving ABI1/160B")
    except (OSError, AttributeError) as error:
        raise RuntimeError("Cannot load the new batched_v4 host exports before model loading: " + str(error)) from error
    # Compile/load the thin current-Torch-ABI host bridge here as well. A missing
    # compiler/ninja or mismatched SDK must not wait until after weight loading.
    from .serving_reader import load_bridge
    bridge = load_bridge(str(root), str(library.parent))
    # Retain the dlopen handle; this uses no stream or device-launch ABI.
    return dict(producer=producer, library=str(library), abi_version=1, buffers_bytes=160,
                bridge_configuration=dict(bridge.configuration())), handle


def preflight_serving(config):
    return _preflight_serving(config.source_root, config.library_dir)[0]


def state_key(vllm_config, config, device):
    return (str(device), str(getattr(vllm_config.model_config, "model", "")),
            int(vllm_config.scheduler_config.max_num_seqs),
            int(vllm_config.scheduler_config.max_num_batched_tokens),
            config.max_query_chunk, str(config.source_root), str(config.library_dir))


def get_state(vllm_config, config, device, *, create=False):
    key = state_key(vllm_config, config, device)
    state = _STATES.get(key)
    if state is None and create:
        import torch
        scheduler = vllm_config.scheduler_config
        captures = getattr(vllm_config.compilation_config, "cudagraph_capture_sizes", None) or []
        table_capacity = max(int(scheduler.max_num_seqs) + 1, *(int(x) for x in captures), 1)
        state = ServingState(request_pool_size=int(scheduler.max_num_seqs),
            max_tokens=int(scheduler.max_num_batched_tokens),
            max_query_per_request=config.max_query_chunk, device=device,
            request_table_capacity=table_capacity)
        state.slot_valid = torch.empty((state.max_tokens,), dtype=torch.bool, device=device)
        state.runtime_readers = []
        state.runtime_layers = []
        _STATES[key] = state
    if state is None:
        raise RuntimeError("recent128 serving state has not been allocated by model weight initialization")
    return state


class ServingRuntime:
    def __init__(self, impl, config):
        self.impl, self.config = impl, config
        vc = impl.vllm_config
        validate_runtime_contract(config, preprocess_type=impl.preprocess_type,
            max_num_seqs=vc.scheduler_config.max_num_seqs)
        if not getattr(impl, "enable_mxfp4_kv", False):
            raise RuntimeError("recent128 serving preserves and requires the original MXFP4 main KV path")
        if impl.num_kv_heads != 1 or impl.kv_lora_rank != 512 or impl.qk_rope_head_dim != 64:
            raise RuntimeError("batched_v4 requires the original D512/R64/KV-head1 packed400 cache")
        # Verify the REAL provider before allocating sidecars or accepting requests.
        self.producer = install_prolog_route()
        self.state = get_state(vc, config, impl.kv_a_layernorm.weight.device, create=True)
        self.layer_key = str(getattr(impl, "layer_name", ""))
        if not self.layer_key:
            raise RuntimeError("recent128 serving requires a stable unique layer_name")
        self.state.register_layer(self.layer_key)
        from .serving_reader import ServingReader
        self.reader = ServingReader(config, impl)
        self._readers_by_lane = {}
        self.state.runtime_readers.append(self.reader)
        self.state.runtime_layers.append(self)
        self.producer_calls = 0
        self.read_calls = 0
        self.report = Path(config.diagnostics_path) / (
            "serving-" + str(os.getpid()) + "-" + re.sub(r"[^A-Za-z0-9_.-]", "_", self.layer_key) + ".json")
        self.flush()

    def flush(self):
        self.report.parent.mkdir(parents=True, exist_ok=True)
        value = dict(execution="serving", status=("EXERCISED_NOT_DEVICE_VALIDATED" if self.read_calls else "WIRED_NOT_DEVICE_VALIDATED"),
            layer=self.layer_key, producer=self.producer, kernel="batched_v4",
            ring_capacity=self.state.capacity, request_pool_size=self.state.request_pool_size,
            producer_calls=self.producer_calls, read_calls=self.read_calls,
            old_accuracy_runtime_used=False, original_c4_layout_preserved=True,
            performance_claim=False)
        self.report.write_text(json.dumps(value, indent=2) + "\n")

    @contextmanager
    def prolog_scope(self, metadata, token_rows):
        lane = getattr(metadata, "recent128_state", None)
        if lane is None or getattr(lane, "pool_owner", lane) is not self.state:
            raise RuntimeError("recent128 runner did not bind this step's corrected device metadata")
        if getattr(metadata, "max_query_len", 0) > self.config.max_query_chunk:
            raise RuntimeError("Recent128 capacity bound is smaller than the scheduler query chunk; configure max_query_chunk before allocating rings")
        binding = lane.binding(self.layer_key, token_rows=token_rows)
        with prolog_scope(self.impl, binding) as route:
            yield route
        if route.provider_calls != 1:
            raise RuntimeError("The real MXFP4 Prolog call did not invoke exactly one recent128 producer")
        self.producer_calls += 1

    def capture(self, *args):
        raise RuntimeError("serving uses the fused Prolog producer, not the eager Native raw-KV tap")

    def read(self, query, rope, cold, topk, metadata, q_lengths, kv_lengths, block_table=None):
        import torch
        lane = getattr(metadata, "recent128_state", None)
        if lane is None or getattr(lane, "pool_owner", lane) is not self.state:
            raise RuntimeError("recent128 reader lacks this step's bound request metadata")
        cold = cold[0] if isinstance(cold, (tuple, list)) else cold
        page = int(metadata.block_size)
        if page != 128 or cold.element_size() != 1:
            raise RuntimeError("batched_v4 requires page128 packed C4 bytes")
        cold = cold.view(-1, page, *cold.shape[2:]).view(torch.int8)
        table = metadata.block_table if block_table is None else block_table
        binding = lane.binding(self.layer_key, token_rows=query.shape[0])
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        key = id(lane)
        if key not in self._readers_by_lane:
            if bool(_EXTRA_CTX.capturing):
                raise RuntimeError("Warm up this recent128 routing lane before Graph capture; no reader creation inside capture")
            from .serving_reader import ServingReader, ServingExecutionLane
            reservation = getattr(lane, "reader_execution", None)
            if reservation is None:
                reservation = ServingExecutionLane()
                lane.reader_execution = reservation
            reader = ServingReader(self.config, self.impl, execution_lane=reservation)
            # Keep each route alive with its reader. Stable routing lanes make
            # target/draft/ubatch workspaces independent across stream changes.
            self._readers_by_lane[key] = (lane, reader)
            self.state.runtime_readers.append(reader)
        reader = self._readers_by_lane[key][1]
        self.reader = reader
        if not reader.is_prepared(query, cold, table):
            if bool(_EXTRA_CTX.capturing):
                raise RuntimeError("Warm up this recent128 batch bucket before Graph capture; no plan allocation inside capture")
            reader.prepare(device=query.device, cold_pages=int(cold.shape[0]),
                token_capacity=int(query.shape[0]), request_capacity=int(table.shape[0]),
                ring_binding=binding, table_columns=int(table.shape[1]))
        result = reader.read(query, rope, cold, topk, table, q_lengths, kv_lengths,
            binding, lane.step.request_banks, lane.step.request_epochs)
        self.read_calls += 1
        return result


def validate_serving_faults(*, reset=False):
    """Worker RPC AFTER the experiment, never inside forward or timing.

    Missing/stale BF16 rows latch faults on device even if a safe C4 value was
    used to avoid an illegal memory read. Such a run cannot pass acceptance.
    Explicit synchronization here is outside the measured production path.
    """
    import torch
    seen = set()
    reports = []
    for state in _STATES.values():
        torch.npu.synchronize(state.device)
        for runtime in state.runtime_layers:
            runtime.flush()
        for reader in state.runtime_readers:
            for fault in reader.fault_buffers():
                pointer = (str(fault.device), fault.data_ptr())
                if pointer in seen:
                    continue
                seen.add(pointer)
                values = fault.detach().cpu().tolist()
                reports.append(dict(device=str(fault.device), fault_count=sum(int(x != 0) for x in values)))
        if reset:
            for reader in state.runtime_readers:
                reader.reset_faults()
    if any(item["fault_count"] for item in reports):
        raise RuntimeError("Recent128 reader encountered invalid bank/epoch/missing BF16 tail: " + str(reports))
    return dict(status="NO_LATCHED_READER_FAULTS" if reports else "NOT_EXERCISED", buffers=reports,
                model_accuracy_validated=False, performance_validated=False)
