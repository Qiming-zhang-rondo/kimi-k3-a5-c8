#!/usr/bin/env python3
"""Native C4/C8 two-pass sparse attention for a fixed-query A5 experiment.

This module changes neither the installed reader nor quantizer. It measures
their two subset calls, native statistics, and a selectable state merge. It
does not implement serving lifecycle, speculative rollback, or PD transfer.
"""
import copy

import numpy as np

from kv_hot_backend import fixed_ring_plan
from kv_window_common import (compare_outputs, flatten_queries, paged_array,
                              query_positions, row_map, validate_case)


RECIPES = ("baseline", "alg2_max7")
MODES = ("split_c4", "mixed")
INDEX_CAPACITIES = ("full", "bounded")
MERGES = ("torch", "triton", "triton_log512", "triton_direct512")
PARTITIONS = ("argsort_int", "argsort_float", "argsort_desc", "triton_scan", "triton_scan_packed")
WRITERS = ("native", "fused_store", "hot_store_partition")
READERS = ("native", "hot_merge")


def diagnostic_range(value):
    """Compact, JSON-safe range; nonfinite values are counted, never repaired."""
    value = np.asarray(value, np.float64)
    finite = value[np.isfinite(value)]
    return dict(shape=list(value.shape), finite=bool(finite.size == value.size),
                nonfinite_count=int(value.size-finite.size),
                min=float(finite.min()) if finite.size else None,
                max=float(finite.max()) if finite.size else None)


def diagnostic_gain(actual, reference):
    """Fit actual=gain*reference independently along D for every [row,head].

    Zero reference coordinates add no weight; all-zero/nonfinite vectors are
    excluded explicitly. This diagnostic never changes a correctness gate.
    """
    actual, reference = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    finite = np.isfinite(actual).all(axis=-1) & np.isfinite(reference).all(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        norm2 = np.sum(reference*reference, axis=-1)
        gains = np.sum(actual*reference, axis=-1)/norm2
    valid = finite & (norm2 > 0) & np.isfinite(gains)

    def summary(values):
        return dict(count=int(values.size), min=float(values.min()) if values.size else None,
                    median=float(np.median(values)) if values.size else None,
                    max=float(values.max()) if values.size else None)

    return dict(method="per-row/head least squares along D: dot(actual,reference)/dot(reference,reference)",
                **summary(gains[valid]),
                excluded_zero_reference_vectors=int(np.count_nonzero(norm2 == 0)),
                excluded_nonfinite_vectors=int(np.count_nonzero(~finite | ~np.isfinite(norm2))),
                zero_reference_coordinates=int(np.count_nonzero(reference == 0)),
                per_row=[dict(row=row, **summary(gains[row][valid[row]])) for row in range(len(gains))])


def validate_partition_width(partition, width):
    """Metadata-only guard for exact FLOAT32 encoding of unique sort keys."""
    if partition not in PARTITIONS:
        raise ValueError("Unsupported partition algorithm")
    if isinstance(width, (bool, np.bool_)) or not isinstance(width, (int, np.integer)) or width < 1:
        raise ValueError("Partition index capacity must be a positive integer")
    if partition in ("argsort_float", "argsort_desc") and 2*int(width)-1 > 2**24:
        raise ValueError("argsort_float requires 2*S-1 <= 2**24 for exact unique sort keys")


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

    def prepare(self, recipe, window, mode="mixed", index_capacity="full", merge="torch", partition="argsort_int", writer="native",
                reader="native", hot_block_keys=16, hot_block_heads=16, hot_decode="software"):
        if recipe not in RECIPES or mode not in MODES or index_capacity not in INDEX_CAPACITIES or merge not in MERGES or partition not in PARTITIONS or writer not in WRITERS or reader not in READERS:
            raise ValueError("Unsupported recipe/mode/index_capacity/merge/partition/writer/reader")
        if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < 0:
            raise ValueError("window must be an integer >= 0")
        if reader == "hot_merge" and window > 0 and (mode != "mixed" or merge != "triton"):
            raise ValueError("hot_merge requires mixed mode and merge=triton; merge is inside the hot reader")
        if writer == "hot_store_partition" and window > 0 and (
                mode != "mixed" or reader != "native" or merge != "triton" or partition != "triton_scan"):
            raise ValueError("hot_store_partition requires mixed/native reader/triton merge/triton_scan")
        if isinstance(hot_block_keys, bool) or hot_block_keys not in (16, 32, 64) or isinstance(hot_block_heads, bool) or hot_block_heads not in (16, 32):
            raise ValueError("Experimental hot tiles require keys=16/32/64 and heads=16/32")
        if hot_decode not in ("software", "native_cast"):
            raise ValueError("hot_decode must be software or native_cast")
        key = (recipe, int(window), mode, index_capacity, merge, partition, writer, reader, hot_block_keys, hot_block_heads, hot_decode)
        if key not in self._runners:
            self._runners[key] = NativeTwoPassRunner(self, self.scales.prepare(recipe), int(window),
                                                     mode, index_capacity, merge, partition, writer, reader,
                                                     hot_block_keys, hot_block_heads, hot_decode)
        return self._runners[key]


class NativeTwoPassRunner:
    def __init__(self, backend, variant, window, mode, index_capacity, merge, partition="argsort_int", writer="native",
                 reader="native", hot_block_keys=16, hot_block_heads=16, hot_decode="software"):
        self.backend, self.base, self.variant = backend, backend.base, variant
        self.window, self.mode = window, mode
        self.index_capacity, self.merge_kind = index_capacity, merge
        self.partition_algo = partition
        self.writer_kind, self._store_runner = writer, None
        self._hot_prepare = None
        self.reader_kind, self._hot_merge = reader, None
        self.metadata = dict(variant.metadata)
        self.metadata.update(window=window, mode=mode, index_capacity=index_capacity, merge=merge,
                             partition=partition, writer=writer, reader_selector=reader,
                             hot_block_keys=hot_block_keys, hot_block_heads=hot_block_heads,
                             hot_decode=hot_decode,
                             production_validation="NOT_VALIDATED",
                             lifetime_scope="fixed trailing query batch only")
        if window == 0:
            self.metadata.update(reader="native_c4", zero_window_native_bypass=True,
                                 attention_calls_per_read=1, hot_cache_bytes=0,
                                 additional_ring_bytes=0, partition_calls_per_dynamic_run=0,
                                 merge_calls_per_read=0)
            return
        base, t = self.base, self.base.torch
        validate_partition_width(partition, base.indices.shape[-1])
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
        self._partition_runner = None
        partition_metadata = {
            "algorithm": partition, "sort_key_dtype": "float32" if partition in ("argsort_float", "argsort_desc") else "int32",
            "source_indices_dtype": "int32", "source_shape": list(self.indices_cpu.shape),
            "old_shape": list(parts["old"].shape), "hot_shape": list(parts["recent_global"].shape),
            "sort_key": "ordinal + (!selected)*S; token IDs are never cast to float",
            "max_unique_sort_key": 2*self.indices_cpu.shape[-1]-1,
            "device_execution": "NOT_VERIFIED; inspect profiler and full-pipeline measurement",
            "performance": "NOT_VALIDATED"}
        if partition == "argsort_desc":
            partition_metadata.update(sort_key="(2*S-1)-(ordinal+(!selected)*S); descending FP32 unique keys",
                                      descending=True, truncate_order_before_gather=True,
                                      comparison="colleague-inspired ideas, not a reproduction of unavailable colleague source")
        if partition in ("triton_scan", "triton_scan_packed"):
            from kv_native_partition import PreparedPartition
            options = dict(variant="packed_one_scan") if partition == "triton_scan_packed" else {}
            self._partition_runner = PreparedPartition(
                base.indices, base.positions, window=window, ring_capacity=self.capacity,
                hot_width=self.hot_width, remap_ring=mode == "mixed", **options)
            partition_metadata = dict(self._partition_runner.metadata)
        self.metadata["partition_implementation"] = partition_metadata
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
        if reader == "native" and merge.startswith("triton"):
            from kv_native_merge import merge_states
            self._triton_merge = merge_states
            # Installed bindings may return FP32 even with BF16 queries. Use
            # the actual native output contract; this template read is setup.
            template = self._cold_read(self.prepared)[0]
            self.merge_output = t.empty_like(template)
        if reader == "hot_merge":
            from kv_native_hot_merge import PreparedHotMerge
            cold_template, hot_template = self._cold_read(self.prepared), self._hot_read(self.prepared)
            self._native_hot_output_dtype = hot_template[0].dtype
            self.reader_row_batch = base.tensor(np.asarray([b for b, _ in row_map(backend.data)], np.int32), t.int32)
            self._hot_merge = PreparedHotMerge(
                base.q_nope, base.q_rope, self.hot_cache.view(t.uint8), self.ring_table,
                self.reader_row_batch, cold_template, page=base.page, ring_capacity=self.capacity,
                hot_width=self.hot_width, scale=base.scale, block_keys=hot_block_keys,
                block_heads=hot_block_heads, hot_output_dtype=hot_template[0].dtype, decode=hot_decode)
            self.metadata["hot_reader_implementation"] = dict(self._hot_merge.metadata)
        if writer == "fused_store":
            from kv_native_store import NativeQuantizedStore
            self._store_runner = NativeQuantizedStore(
                base, variant, self.cold_cache,
                hot_cache=self.hot_cache if mode == "mixed" else None,
                hot_slots=self.hot_slots if mode == "mixed" else None)
            self.metadata["writer_implementation"] = dict(self._store_runner.metadata)
        if writer == "hot_store_partition":
            from kv_native_hot_prepare import PreparedHotStorePartition
            self._hot_prepare = PreparedHotStorePartition(base, self.hot_cache, self.hot_slots,
                                                         self._partition_runner)
            self.metadata["writer_implementation"] = dict(self._hot_prepare.metadata)
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
            merge_output_bytes=int(self.merge_output.numel()*self.merge_output.element_size()) if self._triton_merge else 0,
            independent_cold_cache=True,
            prepared_read_scope="two stats-on native subset reads + empty-aware merge; prepared indices",
            dynamic_read_scope="device-only partition + two stats-on native subset reads + merge",
            write_scope="current original BF16 -> C4 quantization/scatter" +
                        (" + original BF16 -> C8 quantization/ring scatter" if mode == "mixed" else "") +
                        ("; native quantizers unchanged; fused packing/scatter" if writer == "fused_store" else ""),
            component_timing_scope="diagnostic stages with prepared inputs; do not sum as full pipeline latency")
        if reader == "hot_merge":
            self.metadata.update(reader="native_c4_plus_fused_c8_hot_merge", native_attention_calls_per_read=1,
                                 custom_hot_attention_merge_launches=1, merge_calls_per_read=0,
                                 merge="fused_hot_reader", merge_selector=merge,
                                 prepared_read_scope="one native C4 stats-on read + custom C8 hot attention/merge",
                                 dynamic_read_scope="device partition + native C4 read + custom C8 hot attention/merge")
        if self._hot_prepare is not None:
            self.metadata.update(
                hot_prepare_implementation=dict(self._hot_prepare.metadata),
                separate_partition_calls_per_write_dynamic_run=0,
                hot_prepare_calls_per_write_dynamic_run=1,
                partition_calls_per_dynamic_read_without_write=1,
                write_scope="unchanged native C4 quantize/pack/scatter + native C8 quantizer + fused C8 pack/store and current TopK partition",
                write_dynamic_scope="cold write + hot_prepare + two native subset reads + unchanged Triton merge",
                prepared_write_scope="hot_prepare also computes partition; prepared indices remain the read source")

    def decoded_kv(self):
        return self.variant.decoded_kv()

    def decoded_hot_kv(self):
        return self.base.decoded_kv("c8") if self.mode == "mixed" else self.variant.decoded_kv()

    def partition(self):
        """Fixed-shape device work; no CPU indices or synchronization."""
        if self._partition_runner is not None:
            return self._partition_runner.call()
        base, t = self.base, self.base.torch
        valid = base.indices >= 0
        hot = valid & (base.indices >= (base.positions-self.window+1)[:, None, None])
        hot = hot & (base.indices <= base.positions[:, None, None])

        def pack(mask, width):
            key = self.ordinal+(~mask).to(t.int32)*base.indices.shape[-1]
            if self.partition_algo == "argsort_desc":
                key = (2*base.indices.shape[-1]-1-key).to(t.float32)
                order = t.argsort(key, dim=-1, descending=True)[:, :, :width]
            else:
                if self.partition_algo == "argsort_float":
                    key = key.to(t.float32)
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

    def partition_diagnostics(self):
        """Untimed exact checks of every partition output, including raw IDs.

        Do not use base.host(): its float conversion could round int32 token
        IDs above 2**24. No host copies are added to partition() or run().
        """
        if self.window == 0:
            return {"passed": True, "skipped": True, "scope": "W0 has no partition"}
        expected = partition_numpy(self.indices_cpu, self.positions_cpu, self.window,
                                   self.capacity, self.index_capacity)
        if self.mode == "split_c4":
            expected["recent_ring"] = expected["recent_global"]
        if self._hot_prepare is None:
            actual = self.partition()
        else:
            # Exercise the NEW fused producer, not the already-validated
            # standalone partition. Poison shared outputs so a missing store
            # cannot pass using values left by an earlier call.
            outputs = self._hot_prepare.outputs
            saved_parts = {name: value.clone() for name, value in outputs.items()}
            hot_bytes = self.hot_cache.view(self.base.torch.uint8)
            saved_hot = hot_bytes.clone()
            try:
                for name, value in outputs.items():
                    if name == "invalid_rows":
                        value.fill_(True)
                    elif name.endswith("present"):
                        value.copy_(self.base.tensor(~expected[name], self.base.torch.bool))
                    else:
                        value.fill_(-17)
                actual = {name: value.clone() for name, value in self._hot_prepare.call().items()}
            finally:
                hot_bytes.copy_(saved_hot)
                for name, value in outputs.items():
                    value.copy_(saved_parts[name])
        result = {"scope": "untimed exact integer/bool device partition versus CPU oracle",
                  "passed": True, "fields": {}, "invalid_rows": [],
                  "invalid_rows_checked": "invalid_rows" in actual}
        if self._hot_prepare is not None:
            result["producer"] = "fused_hot_store_partition_after_output_poison"
        def json_scalar(value):
            value = value.item()
            return str(value) if isinstance(value, float) and not np.isfinite(value) else value

        for name in ("old", "recent_global", "recent_ring", "old_present", "recent_present"):
            reference = expected[name]
            # A missing output is an interface failure and propagates unchanged.
            observed = actual[name].detach().cpu().numpy().copy()
            same_shape = observed.shape == reference.shape
            same_dtype = observed.dtype == reference.dtype
            bad = np.argwhere(observed != reference) if same_shape else np.empty((0, reference.ndim), np.int64)
            mismatch_count = len(bad) if same_shape else max(observed.size, reference.size)
            field = {"passed": bool(same_shape and same_dtype and not mismatch_count),
                     "expected_shape": list(reference.shape), "actual_shape": list(observed.shape),
                     "expected_dtype": str(reference.dtype), "actual_dtype": str(observed.dtype),
                     "mismatch_count": int(mismatch_count),
                     "first_bad_positions": bad[:16].tolist()}
            if same_shape and len(bad):
                field["first_mismatches"] = [
                    {"position": position.tolist(), "expected": json_scalar(reference[tuple(position)]),
                     "actual": json_scalar(observed[tuple(position)])} for position in bad[:16]]
            result["fields"][name] = field
            result["passed"] &= field["passed"]
        if "invalid_rows" in actual:
            invalid = actual["invalid_rows"].detach().cpu().numpy().copy()
            valid_format = invalid.shape == (len(self.indices_cpu),) and invalid.dtype == np.bool_
            result["invalid_rows"] = np.flatnonzero(invalid.reshape(-1)).tolist()
            result["invalid_rows_format"] = {"passed": bool(valid_format),
                                             "shape": list(invalid.shape), "dtype": str(invalid.dtype)}
            result["passed"] &= bool(valid_format and not result["invalid_rows"])
        return result

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
            options = ({"variant": "log512"} if self.merge_kind == "triton_log512" else
                       {"variant": "direct512"} if self.merge_kind == "triton_direct512" else {})
            return self._triton_merge(left, right, out=self.merge_output,
                                      left_present=parts["old_present"], right_present=parts["recent_present"], **options)
        return merge_torch(self.base.torch, left, right, parts["old_present"], parts["recent_present"])

    def _read(self, parts):
        if self._hot_merge is not None:
            return self._hot_merge.call(self._cold_read(parts), parts["recent_ring"],
                                        parts["old_present"], parts["recent_present"])
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

    def _write(self):
        if self._store_runner is not None:
            return self._store_runner.call()
        self._cold_write()
        if self._hot_prepare is not None:
            self._hot_prepare.call()
        elif self.mode == "mixed":
            self._hot_write()
        return self.cold_cache.view(self.base.torch.uint8)

    def writer_diagnostics(self):
        """Untimed full-byte comparison including unwritten slots and RoPE.

        Poisoning prevents a no-op store from passing because current-token
        payloads already exist in the initialized cache. Restore both caches
        even on failure, so controls never contaminate subsequent candidates.
        """
        if self.window == 0 or self.writer_kind == "native":
            return dict(passed=True, skipped=True, scope="unchanged native writer")
        t = self.base.torch
        caches = {"cold": self.cold_cache.view(t.uint8)}
        if self.mode == "mixed":
            caches["hot"] = self.hot_cache.view(t.uint8)
        saved = {name: value.clone() for name, value in caches.items()}
        saved_parts = ({name: value.clone() for name, value in self._hot_prepare.outputs.items()}
                       if self._hot_prepare is not None else {})
        result = dict(passed=True, scope="all packed bytes vs native writers after sentinel poison; restored",
                      fields={})
        try:
            for value in caches.values():
                value.fill_(165)
            self._write()
            actual = {name: value.detach().cpu().numpy().copy() for name, value in caches.items()}
            for value in caches.values():
                value.fill_(165)
            self._cold_write()
            if self.mode == "mixed":
                self._hot_write()
            for name, value in caches.items():
                expected = value.detach().cpu().numpy().copy()
                bad = np.argwhere(actual[name] != expected)
                result["fields"][name] = dict(passed=not len(bad), bytes=int(expected.size),
                                             mismatch_count=int(len(bad)), first_bad_positions=bad[:16].tolist())
                result["passed"] &= not len(bad)
        finally:
            for name, value in caches.items():
                value.copy_(saved[name])
            for name, value in saved_parts.items():
                self._hot_prepare.outputs[name].copy_(value)
        return result

    def reader_diagnostics(self, control_l2=.01, pipeline_l2=.001):
        """Untimed fused-reader controls against native states and zero query.

        These checks never update KV or the original Q. Output copies happen
        immediately because the fused reader deliberately reuses its buffer.
        """
        if self.window == 0 or self._hot_merge is None:
            return dict(passed=True, skipped=True, scope="unchanged native readers")
        if not all(np.isfinite(x) and x > 0 for x in (control_l2, pipeline_l2)):
            raise ValueError("Reader control tolerances must be finite and positive")
        base, t, parts = self.base, self.base.torch, self.prepared
        left, right = self._cold_read(parts), self._hot_read(parts)
        old_present, hot_present = parts["old_present"], parts["recent_present"]
        zeros = t.zeros_like(old_present)
        native_merged = base.host(merge_torch(t, left, right, old_present, hot_present))
        native_states = {"cold": tuple(base.host(value) for value in left),
                         "hot": tuple(base.host(value) for value in right)}
        cold_reference = np.where(np.asarray(self.metadata["old_counts"])[:, None, None] > 0,
                                  native_states["cold"][0], 0.)
        hot_reference = np.where(np.asarray(self.metadata["hot_counts"])[:, None, None] > 0,
                                 native_states["hot"][0], 0.)
        result = dict(passed=True, scope="fused hot attention/merge vs native states; forced empty sides; zero-query counts",
                      native_state_ranges={}, checks={})
        normalizers = {}
        for side, (output, maximum, total) in native_states.items():
            with np.errstate(over="ignore", invalid="ignore"):
                normalizers[side] = np.exp(maximum.astype(np.float64))*total.astype(np.float64)
            result["native_state_ranges"][side] = {
                "output": diagnostic_range(output), "maximum": diagnostic_range(maximum),
                "sum": diagnostic_range(total), "exp_max_times_sum": diagnostic_range(normalizers[side])}
        for name, cp, hp, reference, tolerance in (
                ("combined_vs_native_twopass", old_present, hot_present, native_merged, pipeline_l2),
                ("hot_only_vs_native_hot", zeros, hot_present, hot_reference, control_l2),
                ("cold_only_vs_native_cold", old_present, zeros, cold_reference, pipeline_l2),
                ("both_empty", zeros, zeros, np.zeros_like(native_merged), pipeline_l2)):
            output = base.host(self._hot_merge.call(left, parts["recent_ring"], cp, hp))
            metric = compare_outputs(output, reference)
            passed = bool(metric["finite"] and metric["relative_l2"] <= tolerance and
                          metric["max_row_relative_l2"] <= tolerance)
            metric.update(passed=passed, tolerance=tolerance)
            cold_mask = base.host(cp).astype(bool)[:, None, None]
            hot_mask = base.host(hp).astype(bool)[:, None, None]
            with np.errstate(over="ignore", invalid="ignore"):
                expected_normalizer = (np.where(cold_mask, normalizers["cold"], 0.) +
                                       np.where(hot_mask, normalizers["hot"], 0.))
            metric.update(output_range=diagnostic_range(output), reference_range=diagnostic_range(reference),
                          fitted_gain=diagnostic_gain(output, reference),
                          expected_normalizer=diagnostic_range(expected_normalizer),
                          expected_normalizer_definition="sum of exp(native_max)*native_sum over presence-true sides")
            result["checks"][name] = metric
            result["passed"] &= passed
        zero = type(self._hot_merge)(
            t.zeros_like(base.q_nope), t.zeros_like(base.q_rope), self.hot_cache.view(t.uint8),
            self.ring_table, self.reader_row_batch, left, page=base.page, ring_capacity=self.capacity,
            hot_width=self.hot_width, scale=base.scale, block_keys=self.metadata["hot_block_keys"],
            block_heads=self.metadata["hot_block_heads"], hot_output_dtype=self._native_hot_output_dtype,
            decode=self.metadata["hot_decode"])
        zero_output = base.host(zero.call(left, parts["recent_ring"], zeros, hot_present, statistics=True))
        maximum, total = (base.host(value).astype(np.float64) for value in zero.hot_statistics)
        counts = np.asarray(self.metadata["hot_counts"], np.int64)
        present = counts > 0
        with np.errstate(over="ignore", invalid="ignore"):
            actual = np.exp(maximum)*total
        finite = bool(np.isfinite(zero_output).all() and np.isfinite(actual[present]).all() and
                      np.isfinite(maximum[present]).all() and np.isfinite(total[present]).all() and
                      (total[present] > 0).all())
        error = np.abs(actual[present]-counts[present, None, None])
        passed = bool(finite and np.all(error <= np.maximum(.02, counts[present, None, None]*1e-4)))
        result["checks"]["zero_query_hot_counts"] = dict(
            passed=passed, finite=finite, expected_counts=counts.tolist(),
            max_abs_count_error=float(error.max()) if error.size and finite else (0. if not error.size else None),
            ignored_empty_rows=int(np.count_nonzero(~present)), tolerance="max(0.02,count*1e-4)")
        result["passed"] &= passed
        return result

    def run(self, write=False, dynamic=True):
        if self.window == 0:
            return self.variant.run(write=write)
        if write:
            self._write()
            if dynamic and self._hot_prepare is not None:
                return self._read(self._hot_prepare.outputs)
        return self._read(self.partition() if dynamic else self.prepared)

    def component_calls(self):
        """Return Tensor-producing stage callables; state setup is untimed."""
        if self.window == 0:
            return {"cold_read": self.variant.call}
        parts = self.prepared
        left = self._cold_read(parts)
        result = {"cold_read": lambda: self._cold_read(parts)[0]}
        if self._hot_prepare is None:
            result["partition"] = lambda: self.partition()["old"]
        if self._hot_merge is not None:
            result["hot_read_merge"] = lambda: self._hot_merge.call(
                left, parts["recent_ring"], parts["old_present"], parts["recent_present"])
        else:
            right = self._hot_read(parts)
            result["hot_read"] = lambda: self._hot_read(parts)[0]
            result["merge"] = lambda: self._merge(left, right, parts)
        if self._store_runner is not None:
            result["quantize_and_fused_store"] = self._write
        else:
            result["cold_write"] = self._cold_write
        if self._hot_prepare is not None:
            def hot_prepare():
                self._hot_prepare.call()
                return self.hot_cache.view(self.base.torch.uint8)
            result["hot_prepare"] = hot_prepare
        elif self.mode == "mixed" and self._store_runner is None:
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
