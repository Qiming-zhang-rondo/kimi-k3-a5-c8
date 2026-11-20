#!/usr/bin/env python3
"""Prepared native-C4 / fused C4+C8 adapters for fixed-TopK A5 experiments.

The ring here covers a fixed query batch. It is not a serving scheduler or an
MTP commit/rollback implementation. All setup and host validation are untimed.
"""
import numpy as np

from kv_window_common import query_positions, row_map, validate_case


RECIPES = ("baseline", "alg2_max7")


def fixed_ring_plan(data, window):
    """Keep every hot token needed by all trailing queries, including Q > 1."""
    data = validate_case(data)
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window <= 0:
        raise ValueError("A hot ring requires an integer window > 0")
    window = int(window)
    page = data["page"]
    capacity = ((window + int(data["q_lengths"].max()) - 2) // page + 1) * page
    batch = data["kv"].shape[0]
    tags = np.full((batch, capacity), -1, np.int64)
    for b in range(batch):
        end = int(data["k_lengths"][b])
        start = max(0, end - int(data["q_lengths"][b]) - window + 1)
        positions = np.arange(start, end, dtype=np.int64)
        tags[b, positions % capacity] = positions
    mapping, positions = row_map(data), query_positions(data)
    for (b, r), position in zip(mapping, positions):
        ids = data["indices"][b, r, 0]
        hot = ids[(ids >= 0) & (ids >= position - window + 1)]
        if not np.array_equal(tags[b, hot % capacity], hot):
            raise RuntimeError("Ring plan overwrote a required hot token")
    return {"capacity": capacity, "tags": tags, "row_batch": mapping[:, 0].copy(),
            "query_positions": positions, "current_slots": mapping[:, 0].astype(np.int64) * capacity + positions % capacity}


class HotWindowBackend:
    def __init__(self, data, device="npu:0", *, scale_backend=None, reader_factory=None,
                 block_keys=32, block_heads=16):
        if scale_backend is None:
            from kv_c4_scale_backend import C4ScaleBackend
            scale_backend = C4ScaleBackend(data, device=device)
        self.scales, self.base = scale_backend, scale_backend.base
        self.data = self.base.data
        self.environment = dict(self.scales.environment)
        self.environment["hot_window_experiment"] = {
            "status": "EXPERIMENTAL_A5_KERNEL", "production_validation": "NOT_VALIDATED",
            "selection": "unchanged fixed TopK; no recent-token insertion or QLI call",
            "w0": "unchanged native C4 reader and writer",
            "w_positive": "one custom Triton attention launch; native C4 and C8 writers",
            "lifetime_scope": "fixed trailing query batch only; not serving/MTP/PD integration",
        }
        self.reader_factory = reader_factory
        self.block_keys, self.block_heads = block_keys, block_heads
        self._runners = {}

    def host(self, tensor):
        return self.base.host(tensor)

    def baseline(self, kind):
        return self.base.call(kind)

    def baseline_run(self, kind, write=False):
        return self.base.run(kind, write=write)

    def baseline_decoded(self, kind):
        return self.data["kv"].copy() if kind == "bf16" else self.base.decoded_kv(kind)

    def prepare(self, recipe, window):
        if recipe not in RECIPES:
            raise ValueError(f"Unknown cold C4 recipe: {recipe}")
        if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < 0:
            raise ValueError("window must be an integer >= 0")
        key = (recipe, int(window))
        if key not in self._runners:
            self._runners[key] = HotWindowRunner(self, self.scales.prepare(recipe), int(window))
        return self._runners[key]


class HotWindowRunner:
    def __init__(self, backend, variant, window):
        self.backend, self.base, self.variant, self.window = backend, backend.base, variant, window
        self.metadata = dict(variant.metadata)
        self.metadata.update(window=window, hot_precision="c8" if window else None,
                             lifetime_scope="fixed query batch; no serving state transition is timed")
        if window == 0:
            # This branch deliberately neither imports Triton nor allocates a ring.
            self.metadata.update(reader="native_c4", zero_window_native_bypass=True,
                                 attention_calls_per_read=1)
            return
        base, t = self.base, self.base.torch
        plan = fixed_ring_plan(base.data, window)
        self.plan = plan
        self.cold_cache = variant.cache.clone()
        width = base.c8_logical.shape[-1]
        hot = np.zeros((base.batch, plan["capacity"], 1, width), np.uint8)
        for b in range(base.batch):
            slots = np.flatnonzero(plan["tags"][b] >= 0)
            hot[b, slots] = base.c8_logical[b, plan["tags"][b, slots]]
        self.hot_cache = base.tensor(hot)
        self.hot_positions = base.tensor(plan["tags"], t.int64)
        self.current_hot_slots = base.tensor(plan["current_slots"], t.int64)
        self.current_positions = base.tensor(plan["query_positions"], t.int64)
        self.row_batch = base.tensor(plan["row_batch"], t.int32)
        self.query_positions = base.tensor(plan["query_positions"], t.int64)
        factory = backend.reader_factory
        if factory is None:
            from kv_hot_fused import FusedHotKVReader
            factory = FusedHotKVReader
        self.reader = factory(base.q_nope, base.q_rope, base.indices, self.cold_cache, base.table,
                              self.hot_cache, self.hot_positions, self.row_batch, self.query_positions,
                              base.lengths, window=window, page=base.page, scale=base.scale,
                              block_keys=backend.block_keys, block_heads=backend.block_heads)
        self.metadata.update(reader="triton_fused_c4_c8", zero_window_native_bypass=False,
                             attention_calls_per_read=1, ring_capacity=plan["capacity"],
                             hot_cache_bytes=int(hot.nbytes), hot_tags_bytes=int(plan["tags"].nbytes),
                             independent_cold_cache=True, cold_cache_bytes=int(self.cold_cache.numel()),
                             reader_metadata=self.reader.metadata,
                             write_scope="current BF16 -> C4 and C8 quantization; both cache scatters; tag scatter; one fused attention",
                             prepare_scope="native full C4 and C8 quantization, physical paging, ring build, allocation outside timing")

    def decoded_kv(self):
        return self.variant.decoded_kv()

    def decoded_hot_kv(self):
        return self.base.decoded_kv("c8")

    def call(self):
        return self.variant.call() if self.window == 0 else self.reader.call()

    def run(self, write=False):
        if self.window == 0:
            return self.variant.run(write=write)
        if write:
            base, t = self.base, self.base.torch
            # Full current-token work is included; do not quantize the hot tail
            # from C4, or reuse prepared bytes as a purported write benchmark.
            cold, _, _ = self.variant._quantize(base.current_kv, base.current_rope)
            self.cold_cache.view(t.uint8).view(-1, self.cold_cache.shape[-1]).index_copy_(0, base.current_slots, cold)
            hot, _, _ = base.quantize(base.current_kv, base.current_rope, "c8")
            self.hot_cache.view(-1, self.hot_cache.shape[-1]).index_copy_(0, self.current_hot_slots, hot)
            self.hot_positions.view(-1).index_copy_(0, self.current_hot_slots, self.current_positions)
        return self.reader.call()
