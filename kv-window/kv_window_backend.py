#!/usr/bin/env python3
"""A5 main-MLA KV cache microbenchmark adapters.

The adapters call installed Cloud sparse-attention operators. They do not
modify a model, install packages, or emulate an unavailable NPU operator.
The recent cache is a separate ring in addition to a complete C4 cache.
All sparse positions are fixed; partitioning changes precision only.
"""
import hashlib
import importlib
import importlib.metadata
import inspect
import math
import platform
import statistics
import sys
import time

import numpy as np

from kv_window_common import dequantize_mxfp4, flatten_queries, paged_array, query_positions, row_map, validate_case


def identity(function):
    result = {"module": getattr(function, "__module__", None), "repr": str(function)}
    try:
        filename = inspect.getsourcefile(function)
        if filename:
            from pathlib import Path
            result.update(file=str(Path(filename).resolve()), sha256=hashlib.sha256(Path(filename).read_bytes()).hexdigest())
    except (TypeError, OSError):
        pass
    return result


def partition_indices(indices, positions, window, capacity):
    """CPU contract for the fixed selection's recent/old partitions.

    Valid entries are compacted without changing relative order. Empty rows
    receive one safe dummy slot for the kernel; their outputs are masked out.
    """
    indices = np.asarray(indices)
    positions = np.asarray(positions)
    if indices.ndim != 3 or indices.shape[1] != 1 or positions.shape != (len(indices),):
        raise ValueError("indices must be [T,1,S] and positions [T]")
    if window < 0 or capacity < 1:
        raise ValueError("window must be nonnegative and capacity positive")
    valid = indices >= 0
    hot = valid & (indices >= (positions-window+1)[:, None, None]) & (indices <= positions[:, None, None])
    if window == 0:
        hot[:] = False
    old = valid & ~hot
    def pack(mask, remap=False):
        result = np.full_like(indices, -1)
        for row in range(len(indices)):
            selected = indices[row, 0, mask[row, 0]]
            result[row, 0, :len(selected)] = selected % capacity if remap else selected
            if not len(selected):
                result[row, 0, 0] = 0
        return result
    return {"old": pack(old), "recent_global": pack(hot), "recent_ring": pack(hot, True),
            "old_present": old.any(axis=(1, 2)), "recent_present": hot.any(axis=(1, 2)),
            "old_counts": old.sum(axis=(1, 2)), "recent_counts": hot.sum(axis=(1, 2))}


def benchmark(call, warmup=20, iterations=100, repeats=5, graph=False):
    """Device event and synchronized host latency, including every op in call.

    Event values are amortized per invocation; host latency is for the whole
    synchronized batch. Graph capture errors are reported to the caller.
    """
    import torch
    if warmup < 0 or min(iterations, repeats) < 1:
        raise ValueError("warmup >= 0, iterations/repeats >= 1 required")
    with torch.inference_mode():
        for _ in range(warmup):
            call()
        torch.npu.synchronize()
        runner = call
        graph_object = None
        graph_reference = graph_output = None
        if graph:
            if not hasattr(torch.npu, "NPUGraph") or not hasattr(torch.npu, "graph"):
                raise RuntimeError("Installed torch_npu does not provide NPUGraph/graph")
            eager_output = call()
            if not isinstance(eager_output,torch.Tensor):
                raise RuntimeError("Graph timing requires a Tensor output so replay correctness can be verified")
            graph_reference = eager_output.detach().clone()
            torch.npu.synchronize()
            graph_object = torch.npu.NPUGraph()
            with torch.npu.graph(graph_object):
                graph_output = call()
            graph_object.replay()
            torch.npu.synchronize()
            if not bool(torch.isfinite(graph_output).all()) or not bool(torch.isfinite(graph_reference).all()):
                raise RuntimeError("Nonfinite graph/eager output; graph timing rejected")
            torch.testing.assert_close(graph_output,graph_reference,rtol=.01,atol=.001)
            runner = graph_object.replay
        event_times, wall_times = [], []
        for _ in range(repeats):
            torch.npu.synchronize()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            t0 = time.perf_counter()
            start.record()
            for _ in range(iterations):
                runner()
            end.record()
            torch.npu.synchronize()
            wall_times.append((time.perf_counter()-t0)*1e6/iterations)
            event_times.append(float(start.elapsed_time(end))*1000/iterations)
        if graph:
            if not bool(torch.isfinite(graph_output).all()):
                raise RuntimeError("Nonfinite output after timed graph replay; timings rejected")
            torch.testing.assert_close(graph_output,graph_reference,rtol=.01,atol=.001)
    return {"mode": "aclgraph_replay" if graph else "eager", "iterations": iterations, "repeats": repeats,
            "warmup": warmup, "event_us": {"median": statistics.median(event_times), "samples": event_times},
            "wall_us": {"median": statistics.median(wall_times), "samples": wall_times},
            "graph_replay_checked": bool(graph),
            "scope": "amortized single-layer fixed-input invocations; excludes setup/initial cache quantization/capture"}


class NPUWindowCase:
    """One fixed-query, fixed-TopK MLA case with three real cache readers."""
    def __init__(self, data, device="npu:0"):
        self.data = validate_case(data)
        import torch
        import torch_npu
        self.torch, self.torch_npu = torch, torch_npu
        self.device = device
        torch.npu.set_device(device)
        importlib.import_module("cann_ops_transformer.ops")
        self.ops = importlib.import_module("cloud_ops_turbo")
        for name in ("cloud_kv_quant_sparse_flash_attention", "cloud_sparse_flash_attention"):
            if not callable(getattr(self.ops, name, None)):
                raise RuntimeError(f"Installed cloud_ops_turbo lacks {name}; no fallback was used")
        for name in ("npu_dynamic_mx_quant", "npu_dynamic_block_quant"):
            if not callable(getattr(torch_npu, name, None)):
                raise RuntimeError(f"Installed torch_npu lacks {name}")
        self.fp4_type = getattr(torch_npu, "float4_e2m1fn_x2", None)
        if self.fp4_type is None or not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("Installed Torch/Torch-NPU lacks required FP4/FP8 dtype support")
        data = self.data
        self.batch, self.queries, self.heads, self.dim = data["q_nope"].shape
        self.rope_dim = data["q_rope"].shape[-1]
        self.keys, self.page = data["kv"].shape[1], int(data["page"])
        if self.dim != 512 or self.rope_dim not in (0, 64):
            raise ValueError("Initial Cloud MLA adapter supports D=512 and RoPE D=0 or64 only")
        self.rows = int(np.sum(data["q_lengths"]))
        self.mapping = row_map(data)
        self.positions_cpu = query_positions(data)
        self.indices_cpu = flatten_queries(data, "indices").astype(np.int32)
        self.indices = self.tensor(self.indices_cpu, torch.int32)
        self.positions = self.tensor(self.positions_cpu, torch.int32)
        self.q_nope = self.tensor(flatten_queries(data, "q_nope"), torch.bfloat16)
        self.q_rope = self.tensor(flatten_queries(data, "q_rope"), torch.bfloat16)
        self.query = torch.cat((self.q_nope, self.q_rope), -1).contiguous()
        self.cu = self.tensor(np.cumsum(data["q_lengths"], dtype=np.int32), torch.int32)
        self.lengths = self.tensor(data["k_lengths"], torch.int32)
        self.scale = float(data["scale"])
        self.pages_per_batch = math.ceil(self.keys/self.page)
        self.table_cpu = np.random.default_rng(int(data.get("seed", 0))).permutation(self.batch*self.pages_per_batch).astype(np.int32).reshape(self.batch, self.pages_per_batch)
        self.table = self.tensor(self.table_cpu, torch.int32)
        self.one_scale = torch.ones(1, dtype=torch.float32, device=device)
        self.kv = self.tensor(paged_array(data["kv"], self.page, self.table_cpu), torch.bfloat16)
        self.rope = self.tensor(paged_array(data["k_rope"], self.page, self.table_cpu), torch.bfloat16)
        logical_kv = self.tensor(data["kv"], torch.bfloat16)
        logical_rope = self.tensor(data["k_rope"], torch.bfloat16)
        with torch.inference_mode():
            c4, q4, s4 = self.quantize(logical_kv, logical_rope, "c4")
            c8, q8, s8 = self.quantize(logical_kv, logical_rope, "c8")
        self.c4_logical = self.bytes_host(c4).reshape(self.batch, self.keys, 1, -1)
        self.c8_logical = self.bytes_host(c8).reshape(self.batch, self.keys, 1, -1)
        q4_bytes = self.bytes_host(q4).reshape(self.batch, self.keys, 1, self.dim//2)
        s4_bytes = self.bytes_host(s4).reshape(self.batch, self.keys, 1, self.dim//32)
        self._decoded_c4 = dequantize_mxfp4(q4_bytes, s4_bytes, dim=self.dim)
        self._decoded_c8 = (self.host(q8).reshape(self.batch,self.keys,1,self.dim) * np.repeat(self.host(s8).reshape(self.batch,self.keys,1,self.dim//128), 128, axis=-1))
        if not np.all(np.isfinite(self._decoded_c4)) or not np.all(np.isfinite(self._decoded_c8)):
            raise RuntimeError("Nonfinite dequantized cache; refusing accuracy/performance comparison")
        self.c4 = self.tensor(paged_array(self.c4_logical, self.page, self.table_cpu)).view(torch.int8)
        self.c8 = self.tensor(paged_array(self.c8_logical, self.page, self.table_cpu)).view(torch.float8_e4m3fn)
        self.raw_result_shapes = {}
        current_kv, current_rope, physical_slots = [], [], []
        for b, r in self.mapping:
            pos = int(data["k_lengths"][b]-data["q_lengths"][b]+r)
            current_kv.append(data["kv"][b,pos])
            current_rope.append(data["k_rope"][b,pos])
            physical_slots.append(int(self.table_cpu[b,pos//self.page])*self.page+pos%self.page)
        self.current_kv = self.tensor(np.stack(current_kv), torch.bfloat16)
        self.current_rope = self.tensor(np.stack(current_rope), torch.bfloat16)
        self.current_slots = self.tensor(np.asarray(physical_slots, np.int64), torch.int64)
        self.environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                            "torch_npu": torch_npu.__version__, "numpy": np.__version__,
                            "device": torch.npu.get_device_name(torch.npu.current_device()),
                            "implementations": {name: identity(getattr(self.ops,name)) for name in ("cloud_kv_quant_sparse_flash_attention", "cloud_sparse_flash_attention")},
                            "quantizers": {name: identity(getattr(torch_npu,name)) for name in ("npu_dynamic_mx_quant", "npu_dynamic_block_quant")},
                            "c4_format": "E2M1 + E8M0 scale/32; RoPE BF16; raw packed bytes",
                            "c8_format": "FP8 E4M3FN + FP32 scale/128; RoPE BF16 (not INT8 C8)",
                            "softmax_statistics_contract": "TND MLA [N2=1,T,H] max and sum; natural exponent",
                            "benchmark_limit": "single-layer fixed-input microbenchmark; no model throughput claim"}
        for name in ("cloud-ops-turbo", "cann-ops-transformer", "torch-npu"):
            try:
                self.environment[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                self.environment[name] = "not recorded"

    def tensor(self, value, dtype=None):
        result = self.torch.from_numpy(np.ascontiguousarray(value)).to(self.device)
        return result.to(dtype) if dtype is not None else result

    @staticmethod
    def host(tensor):
        return tensor.detach().float().cpu().numpy().copy()

    def bytes_host(self, tensor):
        return tensor.detach().contiguous().view(self.torch.uint8).cpu().numpy().copy()

    def quantize(self, kv, rope, kind):
        t = self.torch
        if kind == "c4":
            quant, scale = self.torch_npu.npu_dynamic_mx_quant(kv.contiguous().view(-1,1,self.dim), dst_type=self.fp4_type)
            expected_scale_bytes, quant_bytes = self.dim//32, self.dim//2
            if quant.element_size() != 1 or scale.element_size() != 1:
                raise RuntimeError("MXFP4 quantizer must expose packed FP4 and E8M0 as one-byte storage")
        elif kind == "c8":
            quant, scale = self.torch_npu.npu_dynamic_block_quant(kv.contiguous().view(-1,1,self.dim), dst_type=t.float8_e4m3fn, row_block_size=1, col_block_size=128)
            expected_scale_bytes, quant_bytes = 4*self.dim//128, self.dim
            if quant.dtype != t.float8_e4m3fn or scale.dtype != t.float32:
                raise RuntimeError(f"C8 quantizer dtype contract mismatch: {quant.dtype}, {scale.dtype}; expected E4M3FN/FP32")
        else:
            raise ValueError(kind)
        count = kv.numel()//self.dim
        if quant.numel()*quant.element_size() != count*quant_bytes or scale.numel()*scale.element_size() != count*expected_scale_bytes:
            raise RuntimeError(f"Unexpected {kind} quantizer output sizes: q={tuple(quant.shape)}, scale={tuple(scale.shape)}")
        rope_bytes = rope.contiguous().view(t.uint8).reshape(count, 2*self.rope_dim)
        packed = t.cat((quant.contiguous().view(t.uint8).reshape(count,quant_bytes), rope_bytes,
                        scale.contiguous().view(t.uint8).reshape(count,expected_scale_bytes)), -1)
        return packed, quant, scale

    def decoded_kv(self, kind="c4"):
        if kind not in ("c4", "c8"):
            raise ValueError(kind)
        return (self._decoded_c4 if kind == "c4" else self._decoded_c8).copy()

    def _read(self, kind, indices, *, cache=None, rope=None, table=None, lengths=None, statistics=True):
        t = self.torch
        table = self.table if table is None else table
        lengths = self.lengths if lengths is None else lengths
        common = dict(sparse_indices=indices, block_table=table, actual_seq_lengths_query=self.cu,
                      actual_seq_lengths_kv=lengths, scale_value=self.scale, sparse_block_size=1,
                      layout_query="TND", layout_kv="PA_BSND", sparse_mode=0, attention_mode=2,
                      return_softmax_lse=statistics)
        if kind == "bf16":
            cache = self.kv if cache is None else cache
            rope = self.rope if rope is None else rope
            result = self.ops.cloud_sparse_flash_attention(query=self.q_nope, key=cache, value=cache,
                         query_rope=self.q_rope, key_rope=rope, **common)
        else:
            cache = (self.c4 if kind == "c4" else self.c8) if cache is None else cache
            scale = self.one_scale if kind == "c4" else None
            result = self.ops.cloud_kv_quant_sparse_flash_attention(query=self.query, key=cache, value=cache,
                         key_dequant_scale=scale, value_dequant_scale=scale, key_quant_mode=2, value_quant_mode=2,
                         quant_scale_repo_mode=1, tile_size=128, rope_head_dim=self.rope_dim, **common)
        if not statistics:
            output = result[0] if isinstance(result,(tuple,list)) else result
            if tuple(output.shape) != (self.rows,self.heads,self.dim):
                raise RuntimeError(f"{kind} output shape {tuple(output.shape)} != {(self.rows,self.heads,self.dim)}")
            self.raw_result_shapes[kind+"_stats_off"] = [list(output.shape)]
            return output,None,None
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise RuntimeError(f"{kind} must return (output, softmax_max, softmax_sum); got {type(result)}")
        output, maximum, total = result
        self.raw_result_shapes[kind] = [list(x.shape) for x in result]
        if tuple(output.shape) != (self.rows,self.heads,self.dim):
            raise RuntimeError(f"{kind} output shape {tuple(output.shape)} != {(self.rows,self.heads,self.dim)}")
        expected = (1,self.rows,self.heads)
        if tuple(maximum.shape) != expected or tuple(total.shape) != expected:
            raise RuntimeError(f"{kind} softmax stats expected MLA [N2,T,H/N2] {expected}, got {tuple(maximum.shape)}, {tuple(total.shape)}; refusing an inferred permutation")
        return output, maximum.permute(1,0,2).reshape(self.rows,self.heads,1).float(), total.permute(1,0,2).reshape(self.rows,self.heads,1).float()

    def write(self, kind="c4"):
        """Rewrite current query-token slots; repeats measure fixed-input cost."""
        if kind == "bf16":
            self.kv.view(-1,1,self.dim).index_copy_(0,self.current_slots,self.current_kv)
            if self.rope_dim:
                self.rope.view(-1,1,self.rope_dim).index_copy_(0,self.current_slots,self.current_rope)
        elif kind in ("c4", "c8"):
            packed, _, _ = self.quantize(self.current_kv,self.current_rope,kind)
            target = self.c4 if kind == "c4" else self.c8
            target.view(self.torch.uint8).view(-1,target.shape[-1]).index_copy_(0,self.current_slots,packed)
        else:
            raise ValueError(kind)

    def call(self, kind="c4"):
        if kind not in ("bf16", "c4", "c8"):
            raise ValueError(kind)
        # Production unsplit readers do not request softmax statistics. Split
        # readers necessarily do; their cost must remain in the comparison.
        return self._read(kind,self.indices,statistics=False)[0]

    def run(self, kind="c4", write=False):
        if write:
            self.write(kind)
        return self.call(kind)

    def prepare(self, window, high_precision="bf16"):
        return WindowRunner(self,window,high_precision)


class WindowRunner:
    def __init__(self, case, window, high_precision="bf16"):
        if not isinstance(window,int) or isinstance(window,bool) or window < 0:
            raise ValueError("window must be a nonnegative integer")
        if high_precision not in ("bf16", "c8"):
            raise ValueError("high_precision must be bf16 or c8")
        self.case, self.window, self.high_precision = case, window, high_precision
        t, data = case.torch, case.data
        # Multiple trailing causal queries need history for the earliest query.
        self.capacity = max(case.page, math.ceil((window+int(np.max(data["q_lengths"]))-1)/case.page)*case.page)
        parts = partition_indices(case.indices_cpu,case.positions_cpu,window,self.capacity)
        self.any_old, self.any_recent = bool(parts["old_present"].any()), bool(parts["recent_present"].any())
        self.prepared = {name:case.tensor(parts[name], t.bool if "present" in name else t.int32)
                         for name in ("old","recent_global","recent_ring","old_present","recent_present")}
        self.ordinal = t.arange(case.indices.shape[-1],dtype=t.int32,device=case.device).reshape(1,1,-1)
        self.ring_table_cpu = np.arange(case.batch*self.capacity//case.page,dtype=np.int32).reshape(case.batch,self.capacity//case.page)
        self.ring_table = case.tensor(self.ring_table_cpu,t.int32)
        self.ring_lengths = case.tensor(np.full(case.batch,self.capacity,np.int32),t.int32)
        latent = np.zeros((case.batch,self.capacity,1,case.dim),np.float32)
        rope = np.zeros((case.batch,self.capacity,1,case.rope_dim),np.float32)
        byte_width = case.c8_logical.shape[-1]
        c8 = np.zeros((case.batch,self.capacity,1,byte_width),np.uint8)
        if self.window > 0:
            for b in range(case.batch):
                end = int(data["k_lengths"][b])
                begin = max(0,end-(window+int(data["q_lengths"][b])-1))
                for pos in range(begin,end):
                    slot = pos % self.capacity
                    latent[b,slot] = data["kv"][b,pos]
                    rope[b,slot] = data["k_rope"][b,pos]
                    c8[b,slot] = case.c8_logical[b,pos]
        self.hot_kv = case.tensor(paged_array(latent,case.page,self.ring_table_cpu),t.bfloat16)
        self.hot_rope = case.tensor(paged_array(rope,case.page,self.ring_table_cpu),t.bfloat16)
        self.hot_c8 = case.tensor(paged_array(c8,case.page,self.ring_table_cpu)).view(t.float8_e4m3fn)
        hot_slots = np.asarray([int(b)*self.capacity+int(pos)%self.capacity for (b,r),pos in zip(case.mapping,case.positions_cpu)],np.int64)
        self.hot_slots = case.tensor(hot_slots,t.int64)
        self.partition_summary = {"window":window,"recent_precision":high_precision,"ring_capacity":self.capacity,
            "recent_counts":parts["recent_counts"].tolist(),"old_counts":parts["old_counts"].tolist(),
            "recent_attention_calls":int(self.any_recent),"old_attention_calls":int(self.any_old),
            "selection": "same fixed TopK; no forced recent-token inclusion",
            "empty_rows": "one safe dummy slot; output/statistics masked before merge",
            "window_definition":"query_position-W+1 <= logical KV position <= query_position"}
        per_slot = 2*(case.dim+case.rope_dim) if high_precision=="bf16" else byte_width
        self.memory = {"global_c4_bytes":case.c4.numel()*case.c4.element_size(),
            "additional_ring_bytes":case.batch*self.capacity*per_slot if self.window > 0 else 0,
            "benchmark_ring_buffers_bytes":sum(x.numel()*x.element_size() for x in (self.hot_kv,self.hot_rope,self.hot_c8)),
            "logical_ring_bytes_per_slot":per_slot,
            "note":"production layout estimate includes a complete C4 cache plus a high-precision ring; benchmark also allocates BF16/C8 controls and scratch"}

    def partition(self):
        c, t = self.case, self.case.torch
        valid = c.indices >= 0
        recent = valid & (c.indices >= (c.positions-self.window+1)[:,None,None]) & (c.indices <= c.positions[:,None,None])
        if self.window == 0:
            recent = recent & False
        old = valid & ~recent
        def packed(mask, ring=False):
            # An explicit sort key preserves the original order of valid entries.
            sort_key = self.ordinal + (~mask).to(t.int32)*c.indices.shape[-1]
            order = t.argsort(sort_key,dim=-1)
            values = t.where(mask,c.indices,t.full_like(c.indices,-1))
            values = t.gather(values,-1,order)
            present = mask.any(dim=-1).reshape(-1)
            if ring:
                values = t.where(values>=0,values % self.capacity,values)
            dummy = (self.ordinal==0) & (~present)[:,None,None]
            values = t.where(dummy,t.zeros_like(values),values)
            return values.contiguous(), present
        old_indices,old_present = packed(old)
        hot_global,hot_present = packed(recent)
        hot_ring = t.where(hot_global>=0,hot_global % self.capacity,hot_global).contiguous()
        return dict(old=old_indices,recent_global=hot_global,recent_ring=hot_ring,old_present=old_present,recent_present=hot_present)

    def write_hot(self):
        if self.window == 0:
            return
        c = self.case
        if self.high_precision == "bf16":
            self.hot_kv.view(-1,1,c.dim).index_copy_(0,self.hot_slots,c.current_kv)
            if c.rope_dim:
                self.hot_rope.view(-1,1,c.rope_dim).index_copy_(0,self.hot_slots,c.current_rope)
        else:
            packed,_,_ = c.quantize(c.current_kv,c.current_rope,"c8")
            self.hot_c8.view(c.torch.uint8).view(-1,self.hot_c8.shape[-1]).index_copy_(0,self.hot_slots,packed)

    def _masked(self, state, present):
        output, maximum, total = state
        t = self.case.torch
        mask = present[:,None,None]
        return (t.where(mask,output,t.zeros_like(output)),
                t.where(mask,maximum,t.full_like(maximum,-float("inf"))),
                t.where(mask,total,t.zeros_like(total)))

    def _merge(self, left, right):
        t = self.case.torch
        lo,lm,ls = left
        ro,rm,rs = right
        maximum = t.maximum(lm,rm)
        # Each original row is nonempty, but this guard also avoids -inf--inf.
        maximum = t.where(t.isfinite(maximum),maximum,t.zeros_like(maximum))
        lw = t.exp(lm-maximum)*ls
        rw = t.exp(rm-maximum)*rs
        denominator = lw+rw
        safe = t.where(denominator>0,denominator,t.ones_like(denominator))
        return ((lo.float()*lw+ro.float()*rw)/safe).to(lo.dtype)

    def run(self, mode="mixed", dynamic=False, write=False):
        if mode not in ("mixed", "split_c4"):
            raise ValueError("mode must be mixed or split_c4")
        c = self.case
        if self.window == 0:
            # A disabled precision window is exactly the unsplit production
            # C4 reader, including stats-off behavior and update costs.
            return c.run("c4",write=write)
        if write:
            c.write("c4")
            if mode=="mixed":
                self.write_hot()
        parts = self.partition() if dynamic else self.prepared
        old = recent = None
        if self.any_old:
            old = self._masked(c._read("c4",parts["old"]),parts["old_present"])
        if self.any_recent:
            if mode=="split_c4":
                state = c._read("c4",parts["recent_global"])
            elif self.high_precision=="bf16":
                state = c._read("bf16",parts["recent_ring"],cache=self.hot_kv,rope=self.hot_rope,table=self.ring_table,lengths=self.ring_lengths)
            else:
                state = c._read("c8",parts["recent_ring"],cache=self.hot_c8,table=self.ring_table,lengths=self.ring_lengths)
            recent = self._masked(state,parts["recent_present"])
        if old is None:
            return recent[0]
        if recent is None:
            return old[0]
        return self._merge(old,recent)
