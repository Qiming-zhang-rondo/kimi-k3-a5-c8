"""Real-model eager accuracy path, with measured direct fusion where eligible.

CPU TopK/count inspection, compact grouping, head padding and comparison are
intentional correctness instrumentation. This path MUST NOT be used to claim
graph/TPOT or production performance. Native C4 quantization/cache writes are
unchanged. All model-visible indices remain the original TopK set.
"""
from __future__ import annotations

import atexit
from collections import Counter
from functools import lru_cache
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from weakref import WeakSet

import numpy as np

from .contract import plan_step, partition_reference, validate_runtime_contract

_ACTIVE_RUNTIMES = WeakSet()


def flush_all():
    """Worker RPC after generation, outside forward/timing and before teardown."""
    paths = []
    for runtime in list(_ACTIVE_RUNTIMES):
        runtime.flush()
        paths.append(str(runtime.report))
    return sorted(paths)


@lru_cache(maxsize=4)
def load_bridge(source_root, library_dir):
    root = Path(source_root).expanduser().resolve()
    loader = root / "experimental/torch/load_sfa_direct.py"
    spec = importlib.util.spec_from_file_location("recent128_direct_loader", loader)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot locate the verified direct bridge loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_sfa_direct(library_dir)


@lru_cache(maxsize=4)
def load_single_pass_math(source_root):
    path=Path(source_root).expanduser().resolve()/"experimental/single_pass/host_contract.py"
    name="recent128_single_pass_math_"+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing independent single-pass mathematical reference")
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name,None)
        raise
    return module


def normalize_state(result, *, rows, heads, dim):
    """Only the observed Cloud MLA statistics contract is accepted."""
    import torch
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise RuntimeError("SFA must return output, FP32 softmax_max, FP32 softmax_sum")
    output, maximum, total = result
    if tuple(output.shape) != (rows, heads, dim) or output.dtype != torch.bfloat16:
        raise RuntimeError("Unexpected SFA output shape/dtype; refusing a guessed ABI")
    if maximum.dtype != torch.float32 or total.dtype != torch.float32:
        raise RuntimeError("SFA max/sum must be FP32")
    if tuple(maximum.shape) != (1, rows, heads) or tuple(total.shape) != (1, rows, heads):
        raise RuntimeError("Expected Cloud MLA statistics [1,T,H]; no inferred permutation")
    return output, maximum.permute(1, 2, 0).contiguous(), total.permute(1, 2, 0).contiguous()


def merge_states(cold, hot, cold_present, hot_present):
    """Stable normalized-state merge, explicitly sanitizing absent states."""
    import torch
    cp, hp = cold_present[:, None, None], hot_present[:, None, None]
    cm = torch.where(cp, cold[1], -torch.inf)
    hm = torch.where(hp, hot[1], -torch.inf)
    maximum = torch.maximum(cm, hm)
    maximum = torch.where(cp | hp, maximum, 0.0)
    cz = torch.where(cp, cold[2], 0.0) * torch.exp(cm - maximum)
    hz = torch.where(hp, hot[2], 0.0) * torch.exp(hm - maximum)
    z = cz + hz
    numerator = torch.where(cp, cold[0].float(), 0.0) * cz
    numerator += torch.where(hp, hot[0].float(), 0.0) * hz
    return (numerator / torch.where(z > 0, z, 1.0)).to(cold[0].dtype)


def comparison(actual, expected):
    """Full tensor gate; zero/tiny reference rows retain an absolute guard."""
    return comparison_arrays(actual.detach().float().cpu().numpy(),expected.detach().float().cpu().numpy())


def comparison_arrays(actual, expected, limit=.001):
    a = np.asarray(actual,dtype=np.float64)
    b = np.asarray(expected,dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"passed": False, "reason": "shape_or_nonfinite"}
    delta = a - b
    rows = delta.reshape(-1, delta.shape[-1])
    references = b.reshape(rows.shape)
    abs_norm = np.linalg.norm(rows, axis=1)
    ref_norm = np.linalg.norm(references, axis=1)
    rel = abs_norm / np.maximum(ref_norm, 1e-12)
    global_l2 = float(np.linalg.norm(delta) / max(float(np.linalg.norm(b)), 1e-12))
    worst = float(rel.max(initial=0))
    return {"passed": global_l2 <= limit and worst <= limit,
            "relative_l2": global_l2, "worst_row_relative_l2": worst,
            "max_abs": float(np.abs(delta).max(initial=0))}


class Recent128Runtime:
    def __init__(self, impl, config):
        import torch
        self.impl, self.config = impl, config
        vc = impl.vllm_config
        parallel, cache = vc.parallel_config, vc.cache_config
        validate_runtime_contract(config,
            preprocess_type=getattr(impl.preprocess_type, "name", str(impl.preprocess_type)),
            max_num_seqs=vc.scheduler_config.max_num_seqs,
            enforce_eager=vc.model_config.enforce_eager,
            prefix_caching=bool(getattr(cache, "enable_prefix_caching", False)),
            pd_enabled=getattr(vc, "kv_transfer_config", None) is not None,
            pcp_size=getattr(parallel, "prefill_context_parallel_size", 1),
            dcp_size=getattr(parallel, "decode_context_parallel_size", 1),
            speculative_enabled=getattr(vc, "speculative_config", None) is not None)
        if not getattr(impl, "enable_mxfp4_kv", False):
            raise RuntimeError("recent128 requires the unchanged MXFP4 main KV path")
        if impl.num_kv_heads != 1 or impl.kv_lora_rank != 512 or impl.qk_rope_head_dim not in (0, 64):
            raise RuntimeError("Initial integration requires MLA KV-head1/D512/R0 or R64")
        if impl.kv_a_layernorm.weight.dtype != torch.bfloat16:
            raise RuntimeError("Initial recent128 integration requires BF16 native activations")
        self.end = None
        self.step = None
        self.tags = np.full(config.capacity, -1, np.int64)
        self.hot = self.rope = self.table = self.kv_lengths = None
        self.bridge = None
        self.single_pass_reader = None
        self.plans = {}
        self.counts = Counter()
        self.reasons = Counter()
        self.control_max = {"relative_l2": 0.0, "worst_row_relative_l2": 0.0, "max_abs": 0.0}
        self.failed = None
        self.report_dir = Path(config.diagnostics_path).expanduser().resolve()
        self.report_dir.mkdir(parents=True, exist_ok=True)
        layer = re.sub(r"[^A-Za-z0-9_.-]", "_", str(getattr(impl, "layer_name", "unnamed")))
        self.report = self.report_dir / f"recent128-{os.getpid()}-{layer}.json"
        if config.reader == "single_pass":
            from .single_pass_reader import SinglePassReader
            self.single_pass_reader=SinglePassReader(config,impl)
        elif config.reader != "native":
            if not config.library_dir:
                raise RuntimeError("fused reader requires library_dir; native fallback is not a library substitute")
            self.bridge = load_bridge(config.source_root, config.library_dir)
            if self.bridge.configuration()["abi_version"] != 4:
                raise RuntimeError("Framework adapter requires verified direct ABI4")
        atexit.register(self.flush)
        _ACTIVE_RUNTIMES.add(self)
        self.flush()

    def flush(self):
        data = dict(version="recent128-framework-accuracy-v1", execution="accuracy_eager",
            status="FAILED" if self.failed else ("EXERCISED" if self.counts["read_calls"] else "NOT_EXERCISED"),
            layer=getattr(self.impl, "layer_name", None), pid=os.getpid(),
            window=self.config.window, ring_capacity=self.config.capacity,
            reader=self.config.reader, counters=dict(self.counts), reasons=dict(self.reasons),
            single_pass_kernel=(getattr(self.config,"single_pass_kernel","overlay")
                                if self.single_pass_reader is not None else None),
            fusion_exercised=self.counts["fused_rows"] > 0,
            fusion_status="EXERCISED" if self.counts["fused_rows"] else "NOT_EXERCISED",
            fusion_coverage=self.counts["fused_rows"] / max(self.counts["query_rows"], 1),
            control_max=self.control_max, failure=self.failed,
            control_limit=.01 if self.single_pass_reader is not None else .001,
            control_scope="full_FP64_whole_selected_set" if self.single_pass_reader is not None else "compact_native_twopass",
            prolog_changed=self.config.preprocess == "native_accuracy",
            preprocess=self.config.preprocess,
            native_quantizer_changed=False, topk_changed=False,
            performance_claim=False,
            library_identity=(self.single_pass_reader.bridge_identity if self.single_pass_reader is not None
                              else getattr(self.bridge, "_sfa_build_identity", None)),
            single_pass_last_call=(self.single_pass_reader.last_call_metadata if self.single_pass_reader is not None else None))
        temporary = self.report.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.report)

    def _error(self, message):
        self.failed = message
        self.flush()
        raise RuntimeError(message)

    def capture(self, latent, rope, metadata):
        import torch
        # Accuracy instrumentation only: one explicit sequence, no graph padding.
        lengths = metadata.seq_lens.detach().cpu().numpy().reshape(-1)
        if len(lengths) != 1:
            self._error("recent128 accuracy integration requires exactly one active request")
        latent = latent.reshape(-1, 1, 512).contiguous()
        count = latent.shape[0]
        if latent.dtype != torch.bfloat16:
            self._error("Raw KV tap must provide original BF16 values before quantization")
        self.step = plan_step(self.config, seq_len=int(lengths[0]), query_count=count, previous_end=self.end)
        slots = metadata.slot_mapping[:count].detach().cpu().numpy()
        if len(slots) != count or np.any(slots < 0):
            self._error("Padding/invalid cache slots are unsupported in this explicit eager experiment")
        if self.hot is None:
            cap = self.config.capacity
            self.hot = torch.zeros((cap // 128, 128, 1, 512), device=latent.device, dtype=torch.bfloat16)
            self.rope = torch.zeros((cap // 128, 128, 1, 64), device=latent.device, dtype=torch.bfloat16)
            self.table = torch.arange(cap // 128, device=latent.device, dtype=torch.int32).reshape(1, -1)
            self.kv_lengths = torch.tensor([cap], device=latent.device, dtype=torch.int32)
        if self.step.reset:
            self.tags.fill(-1)
            self.counts["request_resets"] += 1
        positions = self.step.positions
        indices = torch.as_tensor(positions % self.config.capacity, device=latent.device, dtype=torch.int64)
        self.hot.view(-1, 1, 512).index_copy_(0, indices, latent)
        if self.impl.qk_rope_head_dim:
            rope = rope.reshape(count, 1, 64).contiguous()
            if rope.dtype != torch.bfloat16:
                self._error("RoPE tap must preserve original BF16")
            self.rope.view(-1, 1, 64).index_copy_(0, indices, rope)
        self.tags[positions % self.config.capacity] = positions
        self.end = self.step.end
        self.counts["raw_bf16_write_rows"] += count

    def _native_hot(self, query, query_rope, indices):
        import cloud_ops_turbo
        import torch
        n, h, d = query.shape
        result = cloud_ops_turbo.cloud_sparse_flash_attention(
            query=query.contiguous(), key=self.hot, value=self.hot,
            query_rope=query_rope.contiguous(), key_rope=self.rope,
            sparse_indices=indices.contiguous(), block_table=self.table,
            actual_seq_lengths_query=torch.tensor([n], device=query.device, dtype=torch.int32),
            actual_seq_lengths_kv=self.kv_lengths, scale_value=self.impl.scale,
            sparse_block_size=1, layout_query="TND", layout_kv="PA_BSND",
            sparse_mode=0, attention_mode=2, return_softmax_lse=True)
        return normalize_state(result, rows=n, heads=h, dim=d)

    def _fused_group(self, query, query_rope, indices, cold, present):
        import torch
        n, heads, dim = query.shape
        output = torch.empty_like(query)
        cu = torch.tensor([n], device=query.device, dtype=torch.int32)
        hp = torch.ones(n, device=query.device, dtype=torch.bool)
        # Independent heads share the same MLA latent. Zero padding to H64 is
        # exact, but is accuracy-only overhead, not the measured microbenchmark.
        for start in range(0, heads, 64):
            size = min(64, heads - start)
            q = torch.zeros((n, 64, dim), device=query.device, dtype=query.dtype)
            qr = torch.zeros((n, 64, 64), device=query.device, dtype=query.dtype)
            co = torch.zeros_like(q)
            cm = torch.zeros((n, 64, 1), device=query.device, dtype=torch.float32)
            cs = torch.ones_like(cm)
            q[:, :size].copy_(query[:, start:start+size])
            qr[:, :size].copy_(query_rope[:, start:start+size])
            co[:, :size].copy_(cold[0][:, start:start+size])
            cm[:, :size].copy_(cold[1][:, start:start+size])
            cs[:, :size].copy_(cold[2][:, start:start+size])
            key = (n, float(self.impl.scale))
            if key not in self.plans:
                cores = max(x for x in (1, 2, 4, 8, 16) if x <= n and n % x == 0)
                self.plans[key] = self.bridge.prepare(q, float(self.impl.scale), True, False,
                    True, cores, False, self.config.capacity)
            result = self.plans[key].fused_hot(q, self.hot, self.hot, indices, self.table,
                cu, self.kv_lengths, qr, self.rope, co, cm, cs, present, hp)[0]
            # Prepared output is reused by the next group. Copy now on the same stream.
            output[:, start:start+size].copy_(result[:, :size])
            self.counts["fused_kernel_calls"] += 1
            self.counts["head_padding_rows"] += n * (64-size)
        return output

    def _read_single_pass(self, query, q_rope, cache, topk, table, q_lengths, k_lengths, ids_cpu):
        """One candidate attention per head group; CPU math is a full control.

        This is explicitly an accuracy harness. It gathers only selected cold
        pages for its independent FP64 oracle, and never invokes a second NPU
        attention to manufacture the candidate output. Requested kv_split4
        adds one device reduction per head group only for Q<=4; larger
        prefill chunks keep the measured load_tile4 reader explicitly.
        """
        import torch
        if q_rope is None or q_rope.shape[-1] != 64 or cache.shape[-1] != 400:
            self._error("Single-pass prototype requires real R64/packed400; R0/272B is not silently adapted")
        n,heads,_=query.shape
        positions=self.step.positions.copy()
        logical_keys=self.end
        logical_pages=(logical_keys+127)//128
        table_cpu=table.detach().cpu().numpy()
        if table_cpu.ndim!=2 or table_cpu.shape[0]!=1 or table_cpu.shape[1]<logical_pages:
            self._error("Single-pass page table does not cover the logical sequence")
        active=table_cpu[0,:logical_pages].astype(np.int64,copy=False)
        if np.any(active<0) or np.any(active>=cache.shape[0]):
            self._error("Single-pass logical page maps outside the original C4 allocation")
        if np.any((ids_cpu>=0)&(ids_cpu>positions[:,None,None])):
            self._error("Single-pass original TopK contains future IDs; refusing to alter its membership")
        valid=ids_cpu[ids_cpu>=0]
        if any(not np.any(row>=0) for row in ids_cpu[:,0]):
            self._error("First single-pass model prototype requires at least one valid selection in every query")
        selected_pages=np.unique(valid//128)
        physical_pages=active[selected_pages]
        gather=torch.as_tensor(physical_pages,device=query.device,dtype=torch.int64)
        cold_cpu=cache.index_select(0,gather).view(torch.uint8).cpu().numpy()
        compact_table=np.zeros((1,logical_pages),dtype=np.int32)
        compact_table[0,selected_pages]=np.arange(len(selected_pages),dtype=np.int32)
        hot_cpu=self.hot.detach().float().cpu().numpy()
        rope_cpu=self.rope.detach().float().cpu().numpy()
        q_cpu=query.detach().float().cpu().numpy()
        qr_cpu=q_rope.detach().float().cpu().numpy()
        math_module=load_single_pass_math(self.config.source_root)
        contract=math_module.SinglePassContract(queries=n,keys=logical_keys,topk=ids_cpu.shape[-1],
            ring_capacity=self.config.capacity,window=128,aic_count=1,scale=float(self.impl.scale))
        reference=np.empty((n,heads,512),dtype=np.float64)
        for start in range(0,heads,64):
            count=min(64,heads-start)
            q=np.zeros((n,64,576),dtype=np.float32)
            q[:,:count,:512]=q_cpu[:,start:start+count]
            q[:,:count,512:]=qr_cpu[:,start:start+count]
            values=math_module.dense_attention_reference(contract,query=q,cold_packed=cold_cpu,
                block_table=compact_table,topk=ids_cpu,positions=positions,
                hot_latent=hot_cpu,hot_rope=rope_cpu,hot_tags=self.tags)
            reference[:,start:start+count]=values["output"][:,:count]
        tags=torch.as_tensor(self.tags,device=query.device,dtype=torch.int64)
        pos=torch.as_tensor(positions,device=query.device,dtype=torch.int64)
        result=self.single_pass_reader.read(query,q_rope,cache,topk,table,q_lengths,k_lengths,
            self.hot,self.rope,tags,pos,self.config.capacity)
        actual=self.single_pass_reader.last_call_metadata
        requested=getattr(self.config,"single_pass_kernel","overlay")
        expected_entry={"overlay":"QsfaSinglePassLaunch","fast_pair":"QsfaSinglePassLaunchFastPair",
                        "load_tile4":"QsfaSinglePassLaunchLoadTile",
                        "kv_split4":"QsfaSinglePassLaunchKvSplit" if n<=4
                                     else "QsfaSinglePassLaunchLoadTile"}[requested]
        expected_calls=(heads+63)//64
        if (actual.get("kernel_variant")!=requested or actual.get("entry")!=expected_entry
                or actual.get("return_stats") is not False
                or actual.get("external_merge_calls")!=0
                or actual.get("topk_partitioned") is not (requested=="kv_split4" and n<=4)
                or actual.get("head_groups")!=expected_calls
                or self.single_pass_reader.last_attention_calls!=expected_calls):
            self._error("Single-pass actual reader entry/call contract differs from requested kernel")
        if requested=="load_tile4" and (actual.get("load_variant")!=4 or
                actual.get("device_entry")!="QsfaSinglePassLaunchLoadTile32"):
            self._error("Single-pass load_tile4 did not execute the requested compiled layout")
        if requested=="kv_split4":
            split=n<=4
            executed="kv_split4" if split else "load_tile4"
            reductions=expected_calls if split else 0
            aic=n*4 if split else max(value for value in (1,2,4,8,16) if value<=n and n%value==0)
            numeric=("queries","heads","head_groups","candidate_attention_calls",
                     "split_count","aic_count","device_reduction_calls")
            if (any(type(actual.get(name)) is not int for name in numeric)
                    or actual.get("status")!="EXERCISED" or actual.get("queries")!=n
                    or actual.get("heads")!=heads
                    or actual.get("executed_kernel_variant")!=executed
                    or actual.get("dispatch_reason")!=("q_le_4_kv_split4" if split else
                                                       "q_gt_4_prefill_load_tile4")
                    or actual.get("split_count")!=(4 if split else 1)
                    or actual.get("aic_count")!=aic
                    or actual.get("device_entry")!=("qsfa_recent128_kv_split_attention" if split else
                                                    "QsfaSinglePassLaunchLoadTile32")
                    or (not split and actual.get("load_variant")!=4)
                    or actual.get("original_topk_unchanged") is not True
                    or actual.get("candidate_attention_calls")!=expected_calls
                    or actual.get("device_reduction_calls")!=reductions
                    or getattr(self.single_pass_reader,"last_reduction_calls",None)!=reductions
                    or actual.get("validation_attention_calls")!=0):
                self._error("Requested kv_split4 actual dispatch/partial-reduction contract differs from query route")
        metrics=comparison_arrays(result.detach().float().cpu().numpy(),reference,limit=.01)
        if not metrics["passed"]:
            self._error("Single-pass model attention failed full FP64 matching math: "+json.dumps(metrics))
        for name in self.control_max:
            self.control_max[name]=max(self.control_max[name],metrics[name])
        self.counts["fused_rows"]+=n
        self.counts["fused_kernel_calls"]+=self.single_pass_reader.last_attention_calls
        self.counts["candidate_attention_calls"]+=self.single_pass_reader.last_attention_calls
        self.counts["single_pass_expected_attention_calls"]+=expected_calls
        self.counts["single_pass_"+requested+"_calls"]+=self.single_pass_reader.last_attention_calls
        if requested=="kv_split4":
            if n<=4:
                self.counts["kv_split_attention_calls"]+=self.single_pass_reader.last_attention_calls
                self.counts["kv_split_reduction_calls"]+=self.single_pass_reader.last_reduction_calls
                self.counts["kv_split_rows"]+=n
            else:
                self.counts["kv_split_prefill_fallback_calls"]+=self.single_pass_reader.last_attention_calls
        self.counts["full_math_checked_rows"]+=n
        # The mathematical reference is on CPU; validation_attention_calls=0.
        self.counts["validation_attention_calls"]+=0
        return result

    def read(self, query, q_rope, kv_cache, topk, metadata, q_lengths, k_lengths, block_table):
        import torch
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        if bool(_EXTRA_CTX.capturing):
            self._error("accuracy_eager integration must not enter graph capture")
        if self.step is None or query.shape[0] != len(self.step.positions):
            self._error("Read rows do not match this step's original BF16 KV writes")
        n, heads, dim = query.shape
        if query.dtype != torch.bfloat16 or dim != 512 or tuple(topk.shape[:2]) != (n, 1):
            self._error("Unexpected model query/TopK layout")
        lengths = k_lengths.detach().cpu().numpy().reshape(-1)
        ql = q_lengths.detach().cpu().numpy().reshape(-1)
        if lengths.tolist() != [self.end] or ql.tolist() != [n]:
            self._error("Attention sequence metadata differs from raw-KV write metadata")
        ids_cpu=topk.detach().cpu().numpy()
        parts = partition_reference(ids_cpu, positions=self.step.positions,
            tags=self.tags, window=128, capacity=self.config.capacity)
        if self.single_pass_reader is not None:
            cache=kv_cache[0]
            page=int(metadata.block_size)
            if page!=128:
                self._error("Single-pass model prototype requires page128 metadata")
            cache=cache.view(-1,page,*cache.shape[2:])
            if cache.element_size()!=1:
                self._error("Single-pass packed C4 storage must have one-byte elements")
            cache=cache.view(torch.int8) # raw-byte reinterpretation, never a numeric FP8 cast.
            table=metadata.block_table if block_table is None else block_table
            result=self._read_single_pass(query,q_rope,cache,topk,table,q_lengths,k_lengths,ids_cpu)
            self.counts["read_calls"]+=1
            self.counts["query_rows"]+=n
            self.counts["cold_only_rows"]+=int(np.sum(parts["recent_counts"]==0))
            self.counts["mixed_rows"]+=int(np.sum(parts["recent_present"]&parts["old_present"]))
            self.counts["hot_only_rows"]+=int(np.sum(parts["recent_present"]&~parts["old_present"]))
            self.step=None
            if self.counts["read_calls"]==1 or self.counts["read_calls"]%128==0:
                self.flush()
            return result
        from ascend_vllm.patch.worker.patch_device_op import _get_mxfp4_entry, _mxfp4_read
        device = query.device
        to_device = lambda a, dtype: torch.as_tensor(a, device=device, dtype=dtype).contiguous()
        old = to_device(parts["old"], torch.int32)
        cp, hp = (to_device(parts[name], torch.bool) for name in ("old_present", "recent_present"))
        # Quantized reader's all-empty contract is not relied on: provide a
        # safe dummy and discard its state with the explicit present mask.
        old[~cp, 0, 0] = 0
        cache = kv_cache[0]
        page = int(metadata.block_size)
        cache = cache.view(-1, page, *cache.shape[2:])
        table = metadata.block_table if block_table is None else block_table
        cold = normalize_state(_mxfp4_read(_get_mxfp4_entry(), self.impl,
            (query, q_rope, cache, table, old, q_lengths.to(torch.int32), k_lengths.to(torch.int32)),
            sparse_mode=0, return_lse=True), rows=n, heads=heads, dim=dim)
        qr = q_rope
        if qr is None:
            # NoPE semantic adapter: an all-zero64 RoPE contributes exactly zero
            # to QK, retaining the model's actual scale and 272B C4 cold format.
            qr = torch.zeros((n, heads, 64), device=device, dtype=query.dtype)
            self.counts["nope_zero_rope_rows"] += n
        hot = (torch.zeros_like(query), torch.full((n, heads, 1), -torch.inf, device=device),
               torch.zeros((n, heads, 1), device=device))
        groups = []
        for count in np.unique(parts["recent_counts"]):
            if not count:
                continue
            rows_cpu = np.flatnonzero(parts["recent_counts"] == count)
            rows = to_device(rows_cpu, torch.int64)
            ix = to_device(parts["recent_ring"][rows_cpu, :, :int(count)], torch.int32)
            hot_group = self._native_hot(query.index_select(0, rows), qr.index_select(0, rows), ix)
            for destination, value in zip(hot, hot_group):
                destination.index_copy_(0, rows, value)
            groups.append((int(count), rows, ix))
            self.counts["compact_hot_calls"] += 1
        reference = merge_states(cold, hot, cp, hp)
        result = reference.clone()
        for count, rows, ix in groups:
            if self.bridge is None or count != 128:
                self.reasons["reader_native" if self.bridge is None else "hot_count_not_128"] += len(rows)
                continue
            selected_cold = tuple(value.index_select(0, rows).contiguous() for value in cold)
            fused = self._fused_group(query.index_select(0, rows).contiguous(),
                qr.index_select(0, rows).contiguous(), ix, selected_cold, cp.index_select(0, rows).contiguous())
            metrics = comparison(fused, reference.index_select(0, rows))
            if not metrics["passed"]:
                self._error("Fused model attention failed independent compact native control: " + json.dumps(metrics))
            for name in self.control_max:
                self.control_max[name] = max(self.control_max[name], metrics[name])
            result.index_copy_(0, rows, fused)
            self.counts["fused_rows"] += len(rows)
        self.counts["read_calls"] += 1
        self.counts["query_rows"] += n
        self.counts["cold_only_rows"] += int(np.sum(parts["recent_counts"] == 0))
        self.counts["mixed_rows"] += int(np.sum(parts["recent_present"] & parts["old_present"]))
        self.counts["hot_only_rows"] += int(np.sum(parts["recent_present"] & ~parts["old_present"]))
        self.step = None
        # Explicit experiment output, not per-layer warning/.tolist() logging.
        if self.counts["read_calls"] == 1 or self.counts["read_calls"] % 128 == 0:
            self.flush()
        return result
