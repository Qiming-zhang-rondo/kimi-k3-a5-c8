"""Preallocated multi-request reader for the additive serving ASC entry.

Construction/prepare happen before graph capture. Read performs fixed-shape
NPU operations only; it never inspects device values on CPU. Each plan's fault
latches survive replays until explicitly drained by request completion code.
This implementation is a device-validation candidate, not a performance claim.
"""
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile
import threading

_PLANS = {}
_LOCK = threading.Lock()


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def load_bridge(source_root, library_dir):
    source = Path(source_root).expanduser().resolve()/"experimental/single_pass"
    library = Path(library_dir).expanduser().resolve()/"libqsfa_recent128_direct.so"
    files = [source/name for name in ("batched_serving_bridge.cpp", "batched_serving_api.h",
                                      "single_pass_api.h", "single_pass_tiling.h")]
    _require(library.is_file() and all(p.is_file() for p in files),
             "Rebuild the additive batched serving library and provide its public bridge files")
    # Reuse the existing matched-toolkit discovery; no ASC compilation occurs.
    helper = source.parent/"torch/load_sfa_direct.py"
    spec = importlib.util.spec_from_file_location("recent128_serving_sdk", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorded = module._cache_sdk(library.parent)
    selected = os.environ.get("ASCEND_HOME_PATH") or recorded
    _require(selected is not None, "ASCEND_HOME_PATH or the library CMakeCache must identify its matching SDK")
    sdk = Path(selected).expanduser().resolve()
    _require(not recorded or sdk == recorded, "Serving library and active SDK differ")
    acl = module._sdk_include(sdk)
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load
    npu = Path(torch_npu.__file__).resolve().parent
    hashes = {p.name: _sha(p) for p in files}
    identity = dict(library=str(library), library_sha256=_sha(library), source_sha256=hashes,
                    sdk_root=str(sdk), device_validation="PENDING")
    key = json.dumps(identity, sort_keys=True)
    build = Path(tempfile.mkdtemp(prefix="qsfa-serving-bridge-"))
    link = [shlex.quote(str(library)), "-Xlinker", "-rpath", "-Xlinker", shlex.quote(str(library.parent)),
            shlex.quote("-L"+str(npu/"lib")), "-ltorch_npu", "-Xlinker", "-rpath", "-Xlinker", shlex.quote(str(npu/"lib"))]
    bridge = load(name="qsfa_serving_"+hashlib.sha256(key.encode()).hexdigest()[:20],
                  sources=[str(files[0])], extra_include_paths=[str(source), str(npu/"include"), str(acl)],
                  extra_cflags=["-O2", "-std=c++17"], extra_ldflags=link,
                  build_directory=str(build), verbose=False)
    actual = dict(bridge.configuration())
    _require(actual.get("abi_version") == 1 and actual.get("buffers_bytes") == 160,
             "Serving bridge requires additive ABI1/160B, not the legacy B1 ABI")
    identity["configuration"] = actual
    bridge._serving_build_identity = identity
    return bridge


class ServingReader:
    def __init__(self, config, impl):
        _require(config.window == 128 and config.execution == "serving", "Serving reader requires execution=serving/W128")
        _require(config.single_pass_kernel == "batched_v4",
                 "Serving uses the explicit batched_v4 entry; it does not claim batched kv_split4")
        _require(impl.kv_lora_rank == 512 and impl.qk_rope_head_dim == 64 and impl.num_kv_heads == 1,
                 "Serving ABI requires MLA latent512/RoPE64/KV-head1")
        _require(math.isfinite(float(impl.scale)) and float(impl.scale) > 0, "Invalid attention scale")
        self.config, self.impl = config, impl
        self.bridge = load_bridge(config.source_root, config.library_dir)
        self.heads = int(impl.local_num_heads)
        _require(self.heads > 0, "Positive local attention head count required")
        self.prepared = {}
        self.last_faults = ()
        self.last_call_metadata = {}

    def configuration(self):
        return dict(self.bridge.configuration(), execution="serving", performance_claim=False,
                    build_identity=getattr(self.bridge, "_serving_build_identity", {}),
                    original_topk_unchanged=True, cold_layout="INT8[pages,128,1,400]",
                    hot_layout="BF16[P,C,512]+BF16[P,C,64]", device_validation="PENDING")

    def prepare(self, *, device, cold_pages, token_capacity, request_capacity,
                ring_binding, table_columns):
        """Prepare one exact static shape before capture; shared by stream lane/layers."""
        import torch
        t, b = int(token_capacity), int(request_capacity)
        pages, columns = int(cold_pages), int(table_columns)
        pools, capacity, dim = tuple(ring_binding.recent_latent.shape)
        _require(t > 0 and b > 0 and pages > 0 and 0 < columns <= pages and dim == 512,
                 "Invalid serving bucket geometry")
        _require(capacity % 128 == 0 and capacity >= 128 + ring_binding.max_query_per_request - 1,
                 "Ring capacity must protect every query in the configured per-request chunk")
        stream = torch.npu.current_stream(device)
        lane = int(stream.npu_stream)
        local_key = (str(device), lane, t, b, pages, columns, pools, capacity)
        key = (str(self.config.library_dir), str(self.config.source_root), *local_key, self.heads, float(self.impl.scale))
        with _LOCK:
            if key not in _PLANS:
                groups = (self.heads + 63)//64
                query = torch.zeros((groups,t,64,576), device=device, dtype=torch.bfloat16)
                plans = tuple(self.bridge.prepare_batched(query[g], b, pages*128, columns,
                              capacity, pools, float(self.impl.scale)) for g in range(groups))
                _PLANS[key] = dict(query=query, plans=plans,
                    indices=torch.empty((t,1,2048), device=device, dtype=torch.int32),
                    table=torch.empty((b,columns), device=device, dtype=torch.int32),
                    qcum=torch.empty((b,), device=device, dtype=torch.int32),
                    kvlen=torch.empty((b,), device=device, dtype=torch.int32),
                    banks=torch.empty((b,), device=device, dtype=torch.int32),
                    epochs=torch.empty((b,), device=device, dtype=torch.int64),
                    valid=torch.empty((t,), device=device, dtype=torch.bool),
                    output=torch.empty((t,self.heads,512), device=device, dtype=torch.bfloat16))
                _PLANS[key]["valid_view"] = _PLANS[key]["valid"].view(t,1,1)
            self.prepared[local_key] = _PLANS[key]
        return dict(tokens=t, requests=b, table_columns=columns, pools=pools, ring_capacity=capacity,
                    kernel="batched_load_tile32", prepared_before_capture=True)

    def is_prepared(self, query, cold, table):
        import torch
        t = int(query.shape[0])
        b, columns = tuple(table.shape)
        lane = int(torch.npu.current_stream(query.device).npu_stream)
        prefix = (str(query.device), lane, t, b, int(cold.shape[0]), columns)
        return any(key[:6] == prefix for key in self.prepared)

    def fault_buffers(self):
        """All sticky shapes/groups, so a later bucket cannot hide an old fault."""
        seen, result = set(), []
        for scratch in self.prepared.values():
            for plan in scratch["plans"]:
                if id(plan) not in seen:
                    seen.add(id(plan))
                    result.append(plan.faults())
        return tuple(result)

    def read(self, query, q_rope, cold, topk, table, qcum, kvlen,
             ring_binding, request_banks, request_epochs):
        import torch
        t, heads, dim = tuple(query.shape)
        b, columns = tuple(table.shape)
        pools, capacity, _ = tuple(ring_binding.recent_latent.shape)
        _require(heads == self.heads and dim == 512 and query.dtype == torch.bfloat16,
                 "Original BF16 query shape mismatch")
        _require(tuple(q_rope.shape) == (t,heads,64) and q_rope.dtype == torch.bfloat16,
                 "Original BF16 query RoPE64 required")
        _require(len(cold.shape) == 4 and tuple(cold.shape[1:]) == (128,1,400) and
                 cold.dtype == torch.int8 and cold.is_contiguous(), "Original contiguous packed400 C4 pool required")
        _require(tuple(topk.shape) == (t,1,2048) and topk.dtype == torch.int32,
                 "Original INT32 TopK2048 required")
        _require(tuple(qcum.shape) == (b,) and tuple(kvlen.shape) == (b,), "Per-request GPU lengths required")
        _require(len(request_banks.shape) == 1 and request_banks.shape[0] >= b and
                 len(request_epochs.shape) == 1 and request_epochs.shape[0] >= b,
                 "Per-request stable bank and epoch required")
        _require(tuple(ring_binding.token_positions.shape) == (t,) and
                 tuple(ring_binding.token_valid.shape) == (t,), "Per-token GPU position and valid mask required")
        device = query.device
        tensors = (q_rope,cold,topk,table,qcum,kvlen,request_banks,request_epochs,
                   ring_binding.recent_latent,ring_binding.recent_rope,ring_binding.recent_position_tags,
                   ring_binding.recent_epoch_tags,ring_binding.token_positions,ring_binding.token_valid)
        _require(all(value.device == device for value in tensors), "All serving inputs must share the query device")
        lane = int(torch.npu.current_stream(device).npu_stream)
        key = (str(device), lane, t, b, int(cold.shape[0]), columns, pools, capacity)
        _require(key in self.prepared, "Serving shape was not prepared before capture; prepare it during warmup")
        scratch = self.prepared[key]
        # Bind existing contiguous device metadata directly whenever possible.
        # A fallback copy is captured on every call; no host-step cache can omit
        # required graph updates after warmup. Every destination was preallocated.
        def launch_value(name, value, dtype):
            if value.dtype == dtype and value.is_contiguous():
                return value
            scratch[name].copy_(value)
            return scratch[name]
        valid = launch_value("valid", ring_binding.token_valid, torch.bool)
        launch_table = launch_value("table", table, torch.int32)
        launch_qcum = launch_value("qcum", qcum, torch.int32)
        launch_kvlen = launch_value("kvlen", kvlen, torch.int32)
        launch_banks = launch_value("banks", request_banks[:b], torch.int32)
        launch_epochs = launch_value("epochs", request_epochs[:b], torch.int64)
        valid_view = valid.view(t,1,1)
        # Original selected IDs remain untouched. Only inactive/padding rows in
        # the private launch copy become -1, preserving valid-prefix semantics.
        torch.where(valid_view, topk, -1, out=scratch["indices"])
        for group, plan in enumerate(scratch["plans"]):
            start = group*64
            width = min(64, heads-start)
            merged = scratch["query"][group]
            merged[:,:width,:512].copy_(query[:,start:start+width])
            merged[:,:width,512:].copy_(q_rope[:,start:start+width])
            result = plan.run(merged,cold,scratch["indices"],launch_table,launch_qcum,launch_kvlen,
                ring_binding.recent_latent,ring_binding.recent_rope,ring_binding.recent_position_tags,
                ring_binding.token_positions,launch_banks,launch_epochs,ring_binding.recent_epoch_tags,
                valid)
            torch.where(valid_view, result[:,:width], 0,
                        out=scratch["output"][:,start:start+width])
        self.last_faults = tuple(plan.faults() for plan in scratch["plans"])
        self.last_call_metadata = dict(kernel="batched_load_tile32", tokens=t, requests=b,
            head_groups=len(scratch["plans"]), sticky_faults=True, original_topk_unchanged=True,
            cpu_device_value_reads=0, new_plans_created=0, validation="A5_PENDING")
        return scratch["output"]

    def reset_faults(self):
        """Explicit lifecycle operation after fault drain, outside capture/forward."""
        seen = set()
        for scratch in self.prepared.values():
            for plan in scratch["plans"]:
                if id(plan) not in seen:
                    seen.add(id(plan))
                    plan.reset_faults()
