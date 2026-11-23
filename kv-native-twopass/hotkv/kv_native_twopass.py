#!/usr/bin/env python3
"""Native C4/C8 two-pass sparse attention for a fixed-query A5 experiment.

This module changes neither the installed reader nor quantizer. It measures
their two subset calls, native statistics, and a selectable state merge. It
does not implement serving lifecycle, speculative rollback, or PD transfer.
"""
import copy

import numpy as np

from kv_hot_backend import fixed_ring_plan
from kv_window_common import flatten_queries, paged_array, query_positions, validate_case


RECIPES = ("baseline", "alg2_max7")
MODES = ("split_c4", "mixed")
INDEX_CAPACITIES = ("full", "bounded")
MERGES = ("torch", "triton")


def partition_numpy(indices, positions, window, ring_capacity, index_capacity="full"):
    """Untimed oracle: retain input order, trailing -1, safe empty-row dummy.

    With unique visible token indices, at most min(W,S) entries can be hot;
    therefore bounded capacity never drops a selected token.
    """
    indices, positions = np.asarray(indices), np.asarray(positions)
    if indices.ndim != 3 or indices.shape[1] != 1 or positions.shape != (len(indices),):
        raise ValueError("Expected indices [T,1,S] and positions [T]")
    if not np.issubdtype(indices.dtype, np.integer) or indices.shape[-1] < 1:
        raise ValueError("Indices must have positive capacity and integer dtype")
    if window <= 0 or ring_capacity < 1 or index_capacity not in INDEX_CAPACITIES:
        raise ValueError("Positive window/ring capacity and full or bounded policy required")
    width = indices.shape[-1]
    hot_width = width if index_capacity == "full" else min(window, width)
    old = np.full(indices.shape, -1, np.int32)
    hot = np.full((len(indices), 1, hot_width), -1, np.int32)
    counts_old, counts_hot = [], []
    for row, position in enumerate(positions):
        ids = indices[row, 0]
        if np.any(ids < -1):
            raise ValueError("Only -1 is supported as padding")
        valid = ids[ids >= 0]
        if len(np.unique(valid)) != len(valid) or np.any(valid > position):
            raise ValueError("Bounded partition requires unique visible selected indices")
        recent = (valid >= position-window+1) & (valid <= position)
        left, right = valid[~recent], valid[recent]
        counts_old.append(len(left))
        counts_hot.append(len(right))
        if len(right) > hot_width:
            raise RuntimeError("Hot subset exceeds its proven capacity")
        old[row, 0, :len(left)] = left
        hot[row, 0, :len(right)] = right
        if not len(left):
            old[row, 0, 0] = 0
        if not len(right):
            hot[row, 0, 0] = 0
    old_counts, hot_counts = np.asarray(counts_old, np.int32), np.asarray(counts_hot, np.int32)
    return dict(old=old, recent_global=hot,
                recent_ring=np.where(hot >= 0, hot % ring_capacity, hot).astype(np.int32),
                old_present=old_counts > 0, recent_present=hot_counts > 0,
                old_counts=old_counts, recent_counts=hot_counts)


def merge_torch(torch, left, right, left_present, right_present):
    """Merge normalized native max/sum states; empty dummy rows contribute 0.

    Nonempty NaN/Inf values are intentionally not repaired. Correctness
    controls must reject them rather than accept a fabricated output.
    """
    def masked(state, present):
        output, maximum, total = state
        # The partition's presence mask is authoritative. A selected nonempty
        # segment returning (-inf, 0) is a reader failure, not an empty segment.
        empty = ~present[:, None, None]
        bad = ~empty & (~torch.isfinite(maximum) | ~torch.isfinite(total) | (total <= 0))
        return (torch.where(empty, torch.zeros_like(output), output),
                torch.where(empty, torch.full_like(maximum, -float("inf")), maximum),
                torch.where(empty, torch.ones_like(total), total), empty, bad)

    lo, lm, ls, le, lb = masked(left, left_present)
    ro, rm, rs, re, rb = masked(right, right_present)
    both_empty = le & re
    maximum = torch.maximum(lm, rm)
    maximum = torch.where(both_empty, torch.zeros_like(maximum), maximum)
    # Shift before log(sum), then shift again. Finite large maxima and sums
    # must not overflow, and outputs are multiplied by normalized weights.
    left_log, right_log = (lm-maximum)+torch.log(ls), (rm-maximum)+torch.log(rs)
    common_log = torch.maximum(left_log, right_log)
    common_log = torch.where(both_empty, torch.zeros_like(common_log), common_log)
    left_weight = torch.where(le, torch.zeros_like(ls), torch.exp(left_log-common_log))
    right_weight = torch.where(re, torch.zeros_like(rs), torch.exp(right_log-common_log))
    denominator = torch.where(both_empty, torch.ones_like(left_weight), left_weight+right_weight)
    output = lo.float()*(left_weight/denominator)+ro.float()*(right_weight/denominator)
    return torch.where(lb | rb, torch.full_like(output, float("nan")), output).to(lo.dtype)


class NativeTwoPassBackend:
    def __init__(self, data, device="npu:0", *, scale_backend=None):
        if scale_backend is None:
            from kv_c4_scale_backend import C4ScaleBackend
            scale_backend = C4ScaleBackend(data, device=device)
        self.scales, self.base = scale_backend, scale_backend.base
        self.data = validate_case(self.base.data)
        self.environment = dict(self.scales.environment)
        self.environment["native_twopass_experiment"] = {
            "selection": "unchanged unique causal TopK; no QLI or token insertion",
            "reader": "installed cloud_kv_quant_sparse_flash_attention; unmodified native contracts",
            "w0": "unchanged native C4 adapter; no partition/ring/merge",
            "statistics": "native normalized [T,H,1] max/sum; natural-log domain unchanged",
            "production_validation": "NOT_VALIDATED",
            "scope": "fixed trailing query batch; no serving/MTP/PD state transition"}
        self._runners = {}

    def host(self, tensor):
        return self.base.host(tensor)

    def baseline(self, kind):
        return self.base.call(kind)

    def baseline_run(self, kind, write=False):
        return self.base.run(kind, write=write)

    def baseline_decoded(self, kind):
        return self.data["kv"].copy() if kind == "bf16" else self.base.decoded_kv(kind)

    def prepare(self, recipe, window, mode="mixed", index_capacity="full", merge="torch"):
        if recipe not in RECIPES or mode not in MODES or index_capacity not in INDEX_CAPACITIES or merge not in MERGES:
            raise ValueError("Unsupported recipe/mode/index_capacity/merge")
        if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < 0:
            raise ValueError("window must be an integer >= 0")
        key = (recipe, int(window), mode, index_capacity, merge)
        if key not in self._runners:
            self._runners[key] = NativeTwoPassRunner(self, self.scales.prepare(recipe), int(window),
                                                     mode, index_capacity, merge)
        return self._runners[key]


class NativeTwoPassRunner:
    def __init__(self, backend, variant, window, mode, index_capacity, merge):
        self.backend, self.base, self.variant = backend, backend.base, variant
        self.window, self.mode = window, mode
        self.index_capacity, self.merge_kind = index_capacity, merge
        self.metadata = dict(variant.metadata)
        self.metadata.update(window=window, mode=mode, index_capacity=index_capacity, merge=merge,
                             production_validation="NOT_VALIDATED",
                             lifetime_scope="fixed trailing query batch only")
        if window == 0:
            self.metadata.update(reader="native_c4", zero_window_native_bypass=True,
                                 attention_calls_per_read=1, hot_cache_bytes=0,
                                 additional_ring_bytes=0, partition_calls_per_dynamic_run=0,
                                 merge_calls_per_read=0)
            return
        base, t = self.base, self.base.torch
        self.plan = fixed_ring_plan(backend.data, window)
        self.capacity = self.plan["capacity"]
        self.indices_cpu = flatten_queries(backend.data, "indices").astype(np.int32)
        self.positions_cpu = query_positions(backend.data)
        parts = partition_numpy(self.indices_cpu, self.positions_cpu, window, self.capacity, index_capacity)
        self.prepared = {name: base.tensor(value, t.bool if "present" in name else t.int32)
                         for name, value in parts.items() if not name.endswith("counts")}
        self.hot_width = parts["recent_global"].shape[-1]
        self.ordinal = t.arange(self.indices_cpu.shape[-1], dtype=t.int32,
                                device=base.device).reshape(1, 1, -1)
        self.cold_cache = variant.cache.clone()
        ring_bytes = 0
        if mode == "mixed":
            width = base.c8_logical.shape[-1]
            hot = np.zeros((base.batch, self.capacity, 1, width), np.uint8)
            for b in range(base.batch):
                slots = np.flatnonzero(self.plan["tags"][b] >= 0)
                hot[b, slots] = base.c8_logical[b, self.plan["tags"][b, slots]]
            table = np.arange(base.batch*self.capacity//base.page, dtype=np.int32).reshape(base.batch, -1)
            self.hot_cache = base.tensor(paged_array(hot, base.page, table)).view(t.float8_e4m3fn)
            self.ring_table = base.tensor(table, t.int32)
            self.ring_lengths = base.tensor(np.full(base.batch, self.capacity, np.int32), t.int32)
            self.hot_slots = base.tensor(self.plan["current_slots"], t.int64)
            ring_bytes = int(hot.nbytes)
        self._triton_merge = None
        if merge == "triton":
            from kv_native_merge import merge_states
            self._triton_merge = merge_states
            # Installed bindings may return FP32 even with BF16 queries. Use
            # the actual native output contract; this template read is setup.
            template = self._cold_read(self.prepared)[0]
            self.merge_output = t.empty_like(template)
        self.metadata.update(
            reader="native_c4_c8_twopass" if mode == "mixed" else "native_c4_split_control",
            role="CANDIDATE" if mode == "mixed" else "SPLIT_C4_CONTROL",
            zero_window_native_bypass=False, attention_calls_per_read=2,
            cold_attention_calls_per_read=1, hot_attention_calls_per_read=1,
            partition_calls_per_dynamic_run=1, merge_calls_per_read=1,
            statistics=True, statistics_layout="output[T,H,D], max/sum[T,H,1]",
            cold_format="E2M1 + E8M0 scale/32; BF16 RoPE",
            hot_format="E4M3FN + FP32 scale/128; BF16 RoPE" if mode == "mixed" else "same C4 global cache",
            old_counts=parts["old_counts"].tolist(), hot_counts=parts["recent_counts"].tolist(),
            old_indices_shape=list(parts["old"].shape), hot_indices_shape=list(parts["recent_global"].shape),
            old_empty_rows=int(np.count_nonzero(~parts["old_present"])),
            hot_empty_rows=int(np.count_nonzero(~parts["recent_present"])),
            empty_rows="safe dummy index 0; native states masked out in merge",
            ring_capacity=self.capacity if mode == "mixed" else 0,
            hot_cache_bytes=ring_bytes, additional_ring_bytes=ring_bytes,
            cold_cache_bytes=int(self.cold_cache.numel()*self.cold_cache.element_size()),
            prepared_indices_and_masks_bytes=int(sum(value.numel()*value.element_size() for value in self.prepared.values())),
            ring_metadata_bytes=int(self.ring_table.numel()*self.ring_table.element_size() +
                                    self.ring_lengths.numel()*self.ring_lengths.element_size() +
                                    self.hot_slots.numel()*self.hot_slots.element_size()) if mode == "mixed" else 0,
            merge_output_bytes=int(self.merge_output.numel()*self.merge_output.element_size()) if merge == "triton" else 0,
            independent_cold_cache=True,
            prepared_read_scope="two stats-on native subset reads + empty-aware merge; prepared indices",
            dynamic_read_scope="device-only partition + two stats-on native subset reads + merge",
            write_scope="current original BF16 -> C4 quantization/scatter" +
                        (" + original BF16 -> C8 quantization/ring scatter" if mode == "mixed" else ""),
            component_timing_scope="diagnostic stages with prepared inputs; do not sum as full pipeline latency")

    def decoded_kv(self):
        return self.variant.decoded_kv()

    def decoded_hot_kv(self):
        return self.base.decoded_kv("c8") if self.mode == "mixed" else self.variant.decoded_kv()

    def partition(self):
        """Fixed-shape device work; no CPU indices or synchronization."""
        base, t = self.base, self.base.torch
        valid = base.indices >= 0
        hot = valid & (base.indices >= (base.positions-self.window+1)[:, None, None])
        hot = hot & (base.indices <= base.positions[:, None, None])

        def pack(mask, width):
            key = self.ordinal+(~mask).to(t.int32)*base.indices.shape[-1]
            order = t.argsort(key, dim=-1)
            values = t.gather(t.where(mask, base.indices, t.full_like(base.indices, -1)), -1, order)
            values = values[:, :, :width]
            present = mask.any(dim=-1).reshape(-1)
            dummy = (self.ordinal[:, :, :width] == 0) & (~present)[:, None, None]
            return t.where(dummy, t.zeros_like(values), values).contiguous(), present

        old, old_present = pack(valid & ~hot, base.indices.shape[-1])
        recent, hot_present = pack(hot, self.hot_width)
        ring = (t.where(recent >= 0, recent % self.capacity, recent).contiguous()
                if self.mode == "mixed" else recent)
        return dict(old=old, recent_global=recent, recent_ring=ring,
                    old_present=old_present, recent_present=hot_present)

    def _cold_read(self, parts, base=None):
        base = self.base if base is None else base
        return base._read("c4", parts["old"], cache=self.cold_cache, statistics=True)

    def _hot_read(self, parts, base=None):
        base = self.base if base is None else base
        if self.mode == "split_c4":
            return base._read("c4", parts["recent_global"], cache=self.cold_cache, statistics=True)
        return base._read("c8", parts["recent_ring"], cache=self.hot_cache,
                         table=self.ring_table, lengths=self.ring_lengths, statistics=True)

    def _merge(self, left, right, parts):
        if self._triton_merge is not None:
            return self._triton_merge(left, right, out=self.merge_output,
                                      left_present=parts["old_present"], right_present=parts["recent_present"])
        return merge_torch(self.base.torch, left, right, parts["old_present"], parts["recent_present"])

    def _read(self, parts):
        return self._merge(self._cold_read(parts), self._hot_read(parts), parts)

    def _cold_write(self):
        base = self.base
        packed, _, _ = self.variant._quantize(base.current_kv, base.current_rope)
        self.cold_cache.view(base.torch.uint8).view(-1, self.cold_cache.shape[-1]).index_copy_(
            0, base.current_slots, packed)
        return self.cold_cache

    def _hot_write(self):
        base = self.base
        packed, _, _ = base.quantize(base.current_kv, base.current_rope, "c8")
        self.hot_cache.view(base.torch.uint8).view(-1, self.hot_cache.shape[-1]).index_copy_(
            0, self.hot_slots, packed)
        # Packed FP8 storage also contains arbitrary BF16/scale bytes. Return
        # raw bytes so component graph checks do not reinterpret them as NaN.
        return self.hot_cache.view(base.torch.uint8)

    def call(self):
        return self.variant.call() if self.window == 0 else self._read(self.prepared)

    def run(self, write=False, dynamic=True):
        if self.window == 0:
            return self.variant.run(write=write)
        if write:
            self._cold_write()
            if self.mode == "mixed":
                self._hot_write()
        return self._read(self.partition() if dynamic else self.prepared)

    def component_calls(self):
        """Return Tensor-producing stage callables; state setup is untimed."""
        if self.window == 0:
            return {"cold_read": self.variant.call}
        parts = self.prepared
        left, right = self._cold_read(parts), self._hot_read(parts)
        result = {
            "partition": lambda: self.partition()["old"],
            "cold_read": lambda: self._cold_read(parts)[0],
            "hot_read": lambda: self._hot_read(parts)[0],
            "merge": lambda: self._merge(left, right, parts),
            "cold_write": self._cold_write}
        if self.mode == "mixed":
            result["hot_write"] = self._hot_write
        return result

    def zero_query_diagnostics(self):
        """Untimed native stats check: zero-query exp(max)*sum equals count.

        Use a shallow adapter copy with new zero query tensors, preserving
        original query/cache objects. Dummy empty rows are explicitly ignored.
        Host copies and synchronization here must never enter timing.
        """
        if self.window == 0:
            return {"passed": True, "skipped": True, "scope": "W0 has no subset calls"}
        base, t = self.base, self.base.torch
        zero = copy.copy(base)
        zero.q_nope, zero.q_rope = t.zeros_like(base.q_nope), t.zeros_like(base.q_rope)
        if hasattr(base, "query"):
            zero.query = t.zeros_like(base.query)
        # _normalize_read_result records shapes; isolate that mutable log too.
        if hasattr(base, "raw_result_shapes"):
            zero.raw_result_shapes = {}
        result = {"scope": "untimed zero-query native subset stats: exp(max)*sum equals selected count; no original query mutation"}
        for name, state, count_key in (
                ("cold", self._cold_read(self.prepared, base=zero), "old_counts"),
                ("hot", self._hot_read(self.prepared, base=zero), "hot_counts")):
            counts = np.asarray(self.metadata[count_key], np.int64)
            maximum, total = (base.host(x).astype(np.float64) for x in state[1:])
            with np.errstate(over="ignore", invalid="ignore"):
                actual = np.exp(maximum)*total
            present = counts > 0
            valid_stats = np.isfinite(maximum) & np.isfinite(total) & (total > 0)
            finite = bool(np.isfinite(actual[present]).all() and valid_stats[present].all())
            errors = np.abs(actual[present]-counts[present, None, None])
            tolerance = np.maximum(.02, counts[present, None, None]*1e-4)
            passed = bool(finite and np.all(errors <= tolerance))
            per_row_min, per_row_max = [], []
            for row, active in enumerate(present):
                row_finite = active and np.isfinite(actual[row]).all()
                per_row_min.append(float(actual[row].min()) if row_finite else None)
                per_row_max.append(float(actual[row].max()) if row_finite else None)
            result[name] = {"expected_counts": counts.tolist(), "finite": finite, "passed": passed,
                            "ignored_dummy_rows": int(np.count_nonzero(~present)),
                            "actual_count_min_per_row": per_row_min,
                            "actual_count_max_per_row": per_row_max,
                            "max_abs_count_error": float(errors.max()) if errors.size and finite else (0. if not errors.size else None),
                            "tolerance": "absolute <= max(0.02, count*1e-4) on present rows"}
        result["passed"] = bool(result["cold"]["passed"] and result["hot"]["passed"])
        return result
