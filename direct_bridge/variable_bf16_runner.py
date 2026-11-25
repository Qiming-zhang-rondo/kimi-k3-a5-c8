"""Variable-Q/K adapter around the unmodified pinned BF16 hot implementation.

Only initialization replaces the old Q4/K8192 experiment restriction. Native
writers/readers, fused BF16 store/partition and diagnostics remain inherited.
No pinned validator or module is monkeypatched.
"""
import numpy as np

from kv_bf16_hot_ab import (Bf16HotRunner, PreparedBf16HotStorePartition,
                            merge_mixed_dtype)
from kv_native_twopass import NativeTwoPassRunner, partition_numpy
from kv_window_common import paged_array, validate_case, window_selection_stats


def validate_variable_case(data, *, queries=None, keys=None):
    data = validate_case(data)
    q, k = int(data["q_nope"].shape[1]), int(data["kv"].shape[1])
    if not 1 <= q <= 1024 or k not in (8192, 32768):
        raise ValueError("Variable experiment requires Q in [1,1024] and K8192/K32768")
    if queries is not None and q != queries or keys is not None and k != keys:
        raise ValueError("Input replay shape differs from --queries/--keys")
    expected = {"q_nope": (1,q,64,512), "q_rope": (1,q,64,64),
                "kv": (1,k,1,512), "k_rope": (1,k,1,64), "indices": (1,q,1,2048)}
    if any(tuple(data[name].shape) != shape for name, shape in expected.items()):
        raise ValueError("Require B1 H64 D512 R64 TopK2048 for every query")
    if data["page"] != 128 or data["q_lengths"].tolist() != [q] or data["k_lengths"].tolist() != [k]:
        raise ValueError("Require page128 and all trailing Q query rows of the full K sequence")
    if not np.all(data["indices"] >= 0):
        raise ValueError("Every query must select exactly 2048 valid tokens")
    selection = window_selection_stats(data, 128)
    if selection["recent_selected_min"] != 128 or selection["recent_selected_max"] != 128:
        raise ValueError("Every query must select all 128 recent tokens")
    return data


class VariableBf16HotRunner(Bf16HotRunner):
    """Same installed BF16 hot path with a ring sized for the actual query chunk."""
    def __init__(self, backend, recipe="baseline", window=128, writer="native"):
        if recipe != "baseline" or window != 128 or writer not in ("native", "hot_store_partition"):
            raise ValueError("Variable BF16 experiment keeps baseline recipe and W128")
        validate_variable_case(backend.data)
        if backend.base.bf16_indices != "compact":
            raise ValueError("Variable BF16 experiment requires compact hot indices")
        # Explicitly bypass only Bf16HotRunner's fixed-shape constructor. The
        # unchanged parent creates the real C4 cache, partition and ring plan.
        NativeTwoPassRunner.__init__(self, backend, backend.scales.prepare(recipe), window,
            "mixed", "bounded", "triton", "triton_scan", "native", "native")
        base, t = self.base, self.base.torch
        parts = partition_numpy(self.indices_cpu, self.positions_cpu, window, self.capacity, "bounded")
        if np.any(parts["recent_counts"] != 128) or np.any(parts["old_counts"] != 1920) or self.hot_width != 128:
            raise ValueError("Every query requires selected cold1920/hot128")
        self.bf16_plan = base.make_bf16_plan(parts["recent_ring"], counts=parts["recent_counts"],
                                           label="bf16_hot_ring_compact_variable_q")
        if self.bf16_plan.uniform_count != 128 or self.bf16_plan.summary["operator_calls"] != 1:
            raise ValueError("The variable experiment requires one compact BF16 hot call")
        table = np.arange(base.batch*self.capacity//base.page, dtype=np.int32).reshape(base.batch, -1)
        latent = np.zeros((base.batch, self.capacity, 1, base.dim), np.float32)
        rope = np.zeros((base.batch, self.capacity, 1, base.rope_dim), np.float32)
        for b in range(base.batch):
            slots = np.flatnonzero(self.plan["tags"][b] >= 0)
            tokens = self.plan["tags"][b, slots]
            latent[b, slots] = backend.data["kv"][b, tokens]
            rope[b, slots] = backend.data["k_rope"][b, tokens]
        self.hot_cache = base.tensor(paged_array(latent, base.page, table), t.bfloat16)
        self.hot_rope = base.tensor(paged_array(rope, base.page, table), t.bfloat16)
        self._triton_merge = merge_mixed_dtype
        self.writer_kind = writer
        if writer == "hot_store_partition":
            self._hot_prepare = PreparedBf16HotStorePartition(
                base, self.hot_cache, self.hot_rope, self.hot_slots, self._partition_runner)
        hot_template = self._hot_read(self.prepared)
        hot_bytes = sum(value.numel()*value.element_size() for value in (self.hot_cache, self.hot_rope))
        self.metadata.update(reader="native_c4_bf16_twopass", hot_precision="bf16", writer=writer,
            hot_format="original BF16 latent and separate BF16 RoPE; no hot quantization",
            hot_cache_bytes=int(hot_bytes), additional_ring_bytes=int(hot_bytes),
            hot_attention_calls_per_read=1, attention_calls_per_read=2,
            bf16_hot_read_plan=dict(self.bf16_plan.summary), fixed_hot_count_required=128,
            varying_hot_counts_supported=False, ring_capacity=int(self.capacity),
            query_chunk=int(base.queries), global_keys=int(base.keys),
            merge_input_dtypes={"cold":str(self.merge_output.dtype), "hot":str(hot_template[0].dtype)},
            merge_output_dtype=str(self.merge_output.dtype), merge_dtype_cast_kernel_calls=0,
            merge_implementation="unchanged pinned log256 Triton merge",
            write_scope="unchanged native C4 writer + original BF16 latent/RoPE ring stores",
            prepared_read_scope="native C4 + compact native BF16 hot + same log256 merge",
            dynamic_read_scope="stable device partition + native C4 + compact BF16 hot + same merge",
            writer_implementation=(dict(self._hot_prepare.metadata) if self._hot_prepare else
                dict(implementation="bf16_latent_and_rope_index_copy",quantizer_api_calls=0,index_copy_api_calls=2)),
            production_validation="NOT_VALIDATED")
        if self._hot_prepare is not None:
            self.metadata.update(hot_prepare_implementation=dict(self._hot_prepare.metadata),
                separate_partition_calls_per_write_dynamic_run=0, hot_prepare_calls_per_write_dynamic_run=1,
                partition_calls_per_dynamic_read_without_write=1,
                write_dynamic_scope="native C4 write + BF16 hot_store_partition + two native reads + unchanged merge")
