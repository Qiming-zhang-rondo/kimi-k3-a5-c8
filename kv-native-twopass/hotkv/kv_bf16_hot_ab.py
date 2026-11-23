"""One-shape BF16-versus-C8 hot-window experiment, independent of v4.2.

The cold reader/writer, stable partition and merge arithmetic remain the
validated implementations. BF16 hot storage comes directly from the original
latent and RoPE. The fixed experiment requires 128 valid hot indices in every
row, so the native BF16 reader sees a compact capacity of 128 with no padding.
It is not a general dynamic-count serving implementation.
"""

import numpy as np

from kv_native_twopass import NativeTwoPassRunner, partition_numpy
from kv_native_store import validate_slots
from kv_window_common import bf16_round, compare_outputs, paged_array


def validate_fixed_case(base, window):
    actual = (base.batch, base.queries, base.keys, base.heads, base.dim,
              base.rope_dim, base.indices.shape[-1], base.page, window)
    if actual != (1, 4, 8192, 64, 512, 64, 2048, 128, 128):
        raise ValueError("BF16 hot A/B requires B1 Q4 K8192 H64 D512 R64 TopK2048 page128 W128")
    if not np.array_equal(base.data["q_lengths"], [4]) or not np.array_equal(base.data["k_lengths"], [8192]):
        raise ValueError("BF16 hot A/B requires all four trailing queries of the full 8192-token sequence")
    if base.bf16_indices != "compact":
        raise ValueError("BF16 hot A/B requires compact native BF16 indices")


def merge_mixed_dtype(left, right, out, *, left_present=None, right_present=None):
    """Use the unchanged native-result merge kernel with independent dtypes.

    The original kernel already loads each pointer at its own element type
    and computes in FP32. Validate each state independently, then the shared
    layout/device. No output conversion, tensor allocation or extra kernel is
    introduced; the destination retains the native cold output dtype.
    """
    from kv_native_merge import validate_states, _load_kernel
    layout = validate_states(left, left, out, left_present=left_present)
    other = validate_states(right, right, right_present=right_present)
    if layout != other or str(left[0].device) != str(right[0].device):
        raise ValueError("Cold and BF16-hot states must have the same shape and device")
    block = 256
    lp = left_present if left_present is not None else left[1]
    rp = right_present if right_present is not None else right[1]
    _load_kernel()[((layout.features+block-1)//block, layout.tokens*layout.heads)](
        *left, *right, lp, rp, out, H=layout.heads, D=layout.features, BLOCK=block,
        HAS_LEFT_PRESENT=left_present is not None, HAS_RIGHT_PRESENT=right_present is not None,
        DIRECT=False)
    return out


def merge_reference_for_dtype(values, dtype):
    """Round the independent FP64 oracle to the native destination contract."""
    name = str(dtype)
    if name in ("torch.bfloat16", "bfloat16"):
        return bf16_round(values)
    if name in ("torch.float32", "float32"):
        return np.asarray(values, np.float32)
    raise ValueError("Mixed merge destination must be BF16 or FP32")


_KERNEL = None


def _load_prepare_kernel():
    global _KERNEL, tl
    if _KERNEL is not None:
        return _KERNEL
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("BF16 hot preparation requires the installed Ascend Triton build") from exc

    @triton.jit
    def _bf16_hot_prepare_kernel(INDICES, POSITIONS, OLD, RECENT_GLOBAL, RECENT_RING,
                                 OLD_PRESENT, RECENT_PRESENT, INVALID_ROWS,
                                 KV, ROPE, HOT_SLOTS, HOT_KV, HOT_ROPE,
                                 S: tl.constexpr, HW: tl.constexpr, WINDOW: tl.constexpr,
                                 CAP: tl.constexpr, BLOCK: tl.constexpr,
                                 KV_BYTES: tl.constexpr, ROPE_BYTES: tl.constexpr,
                                 HOT_CAPACITY: tl.constexpr, STORE_BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        byte = tl.arange(0, STORE_BLOCK)
        slot = tl.load(HOT_SLOTS+row).to(tl.int64)
        slot_ok = (slot >= 0) & (slot < HOT_CAPACITY)
        kv = tl.load(KV+row*KV_BYTES+byte, mask=byte < KV_BYTES, other=0)
        rope = tl.load(ROPE+row*ROPE_BYTES+byte, mask=byte < ROPE_BYTES, other=0)
        tl.store(HOT_KV+slot*KV_BYTES+byte, kv, mask=slot_ok & (byte < KV_BYTES))
        tl.store(HOT_ROPE+slot*ROPE_BYTES+byte, rope, mask=slot_ok & (byte < ROPE_BYTES))

        lane = tl.arange(0, BLOCK)
        key = tl.load(INDICES+row*S+lane, mask=lane < S, other=-1)
        position = tl.load(POSITIONS+row).to(tl.int64)
        visible = (lane < S) & (key >= 0) & (key.to(tl.int64) <= position)
        recent = visible & (key.to(tl.int64) > position-WINDOW)
        old = visible & (recent == 0)
        old_rank = tl.cumsum(old.to(tl.int32), 0)-1
        recent_rank = tl.cumsum(recent.to(tl.int32), 0)-1
        old_count = tl.sum(old.to(tl.int32), 0)
        recent_count = tl.sum(recent.to(tl.int32), 0)
        tl.store(OLD_PRESENT+row, old_count > 0)
        tl.store(RECENT_PRESENT+row, recent_count > 0)
        bad_key = (lane < S) & ((key < -1) | ((key >= 0) & (key.to(tl.int64) > position)))
        # This reader plan requires a full hot prefix. Changed counts must be
        # visible to the correctness gate, never silently treated as supported.
        invalid = ((position < 0) | (recent_count != HW) | (slot_ok == 0)
                   | (tl.sum(bad_key.to(tl.int32), 0) > 0))
        tl.store(INVALID_ROWS+row, invalid)
        tl.store(OLD+row*S+old_rank, key, mask=old)
        tl.store(OLD+row*S+lane, -1, mask=(lane < S) & (lane >= tl.maximum(old_count, 1)))
        tl.store(OLD+row*S, 0, mask=old_count == 0)
        hot_store = recent & (recent_rank < HW)
        hot_padding = (lane < HW) & (lane >= tl.maximum(recent_count, 1))
        tl.store(RECENT_GLOBAL+row*HW+recent_rank, key, mask=hot_store)
        tl.store(RECENT_GLOBAL+row*HW+lane, -1, mask=hot_padding)
        tl.store(RECENT_GLOBAL+row*HW, 0, mask=recent_count == 0)
        tl.store(RECENT_RING+row*HW+recent_rank, key % CAP, mask=hot_store)
        tl.store(RECENT_RING+row*HW+lane, -1, mask=hot_padding)
        tl.store(RECENT_RING+row*HW, 0, mask=recent_count == 0)

    _KERNEL = _bf16_hot_prepare_kernel
    return _KERNEL


class PreparedBf16HotStorePartition:
    """Raw BF16 latent/RoPE scatter and the existing two-scan partition."""

    def __init__(self, base, hot_cache, hot_rope, hot_slots, partition):
        self.base = base
        layout, t = partition.layout, base.torch
        if layout.variant != "two_scan" or not layout.remap_ring:
            raise ValueError("BF16 preparation requires two_scan with ring remapping")
        if (base.dim, base.rope_dim) != (512, 64):
            raise ValueError("BF16 preparation supports D512/R64")
        self.count = layout.rows
        shape = tuple(hot_cache.shape)
        if (len(shape) != 4 or min(shape[:2]) < 1 or shape[2:] != (1, 512)
                or tuple(hot_rope.shape) != shape[:3]+(64,)):
            raise ValueError("BF16 hot latent and RoPE must use matching paged layouts")
        if base.current_kv.numel() != self.count*512 or base.current_rope.numel() != self.count*64:
            raise ValueError("BF16 current token count must match partition rows")
        for value in (base.current_kv, base.current_rope, hot_cache, hot_rope):
            if value.dtype != t.bfloat16 or value.element_size() != 2:
                raise ValueError("BF16 preparation preserves original two-byte BF16 tensors")
        names = ("old", "recent_global", "recent_ring", "old_present", "recent_present", "invalid_rows")
        inputs = tuple(partition._inputs)
        if len(inputs) != 8 or any(inputs[n+2] is not partition.outputs[name] for n, name in enumerate(names)):
            raise ValueError("Prepared partition buffers do not match their output identities")
        expected = [((self.count, 1, layout.sparse_capacity), t.int32), ((self.count,), None),
                    ((self.count, 1, layout.sparse_capacity), t.int32),
                    ((self.count, 1, layout.hot_width), t.int32), ((self.count, 1, layout.hot_width), t.int32),
                    ((self.count,), t.bool), ((self.count,), t.bool), ((self.count,), t.bool)]
        for value, (wanted, dtype) in zip(inputs, expected):
            if tuple(value.shape) != wanted or (dtype is not None and value.dtype != dtype):
                raise ValueError("Prepared partition tensor shape/dtype mismatch")
        if inputs[1].dtype not in (t.int32, t.int64) or hot_slots.dtype not in (t.int32, t.int64):
            raise ValueError("Positions and physical hot slots must be integer tensors")
        device = str(base.current_kv.device)
        if device.split(":", 1)[0] not in ("npu", "privateuseone"):
            raise ValueError("BF16 preparation requires NPU tensors")
        for value in (*inputs, base.current_kv, base.current_rope, hot_cache, hot_rope, hot_slots):
            if str(value.device) != device or not value.is_contiguous() or getattr(value, "requires_grad", False):
                raise ValueError("BF16 preparation requires contiguous inference tensors on one NPU")
        capacity = hot_cache.numel()//512
        validate_slots(hot_slots.detach().cpu().numpy(), self.count, capacity, "hot_slots")
        self._inputs, self.outputs = inputs, partition.outputs
        self.kv_bytes, self.rope_bytes = base.current_kv.view(t.uint8), base.current_rope.view(t.uint8)
        self.hot_bytes, self.hot_rope_bytes = hot_cache.view(t.uint8), hot_rope.view(t.uint8)
        self.hot_slots = hot_slots
        self.constants = dict(S=layout.sparse_capacity, HW=layout.hot_width, WINDOW=layout.window,
                              CAP=layout.ring_capacity, BLOCK=layout.block,
                              KV_BYTES=1024, ROPE_BYTES=128, HOT_CAPACITY=capacity, STORE_BLOCK=1024)
        self.kernel = _load_prepare_kernel()
        self.metadata = dict(implementation="bf16_hot_store_two_scan_partition", quantizer_api_calls=0,
                             prepare_kernel_launches=1, standalone_partition_calls=0,
                             bf16_latent_and_rope_byte_preserving=True, partition_outputs_reused=True,
                             partition_implementation=dict(partition.metadata), cold_writer_included=False,
                             attention_launches=0, no_timed_host_sync=True,
                             status="EXPERIMENTAL_REQUIRES_A5_VALIDATION")

    def call(self):
        self.kernel[(self.count,)](*self._inputs, self.kv_bytes, self.rope_bytes,
                                   self.hot_slots, self.hot_bytes, self.hot_rope_bytes, **self.constants)
        return self.outputs


class Bf16HotRunner(NativeTwoPassRunner):
    """Native C4 cold plus compact native BF16 hot, with optional fused store."""

    def __init__(self, backend, recipe="baseline", window=128, writer="native"):
        if recipe not in ("baseline", "alg2_max7") or writer not in ("native", "hot_store_partition"):
            raise ValueError("Unsupported BF16 hot recipe or writer")
        validate_fixed_case(backend.base, window)
        # Initialize the shared cold state/partition/output without constructing
        # a C8 fused writer. Setup allocation is not part of any timed call.
        super().__init__(backend, backend.scales.prepare(recipe), window, "mixed", "bounded",
                         "triton", "triton_scan", "native", "native")
        base, t = self.base, self.base.torch
        parts = partition_numpy(self.indices_cpu, self.positions_cpu, window, self.capacity, "bounded")
        if np.any(parts["recent_counts"] != window) or self.hot_width != window:
            raise ValueError("This fixed BF16 A/B requires exactly W valid hot indices per query")
        self.bf16_plan = base.make_bf16_plan(parts["recent_ring"], counts=parts["recent_counts"],
                                            label="bf16_hot_ring_compact")
        if self.bf16_plan.uniform_count != window or self.bf16_plan.summary["operator_calls"] != 1:
            raise ValueError("BF16 A/B must use one compact native hot call")
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
        hot_bytes = sum(x.numel()*x.element_size() for x in (self.hot_cache, self.hot_rope))
        self.metadata.update(
            reader="native_c4_bf16_twopass", hot_precision="bf16", writer=writer,
            hot_format="original BF16 latent and separate BF16 RoPE; no hot quantization",
            hot_cache_bytes=int(hot_bytes), additional_ring_bytes=int(hot_bytes),
            hot_attention_calls_per_read=1, attention_calls_per_read=2,
            bf16_hot_read_plan=dict(self.bf16_plan.summary),
            fixed_hot_count_required=window, varying_hot_counts_supported=False,
            merge_input_dtypes={"cold": str(self.merge_output.dtype), "hot": str(hot_template[0].dtype)},
            merge_output_dtype=str(self.merge_output.dtype), merge_dtype_cast_kernel_calls=0,
            merge_implementation="unchanged log256 Triton kernel with independently validated input dtypes",
            write_scope="unchanged native C4 writer + original BF16 latent/RoPE ring stores",
            prepared_read_scope="native C4 + compact native BF16 hot + same log256 merge",
            dynamic_read_scope="same stable device partition + native C4 + compact BF16 hot + same log256 merge",
            writer_implementation=(dict(self._hot_prepare.metadata) if self._hot_prepare else
                                   dict(implementation="bf16_latent_and_rope_index_copy", quantizer_api_calls=0,
                                        index_copy_api_calls=2)),
            production_validation="NOT_VALIDATED")
        if self._hot_prepare is not None:
            self.metadata.update(hot_prepare_implementation=dict(self._hot_prepare.metadata),
                                 separate_partition_calls_per_write_dynamic_run=0,
                                 hot_prepare_calls_per_write_dynamic_run=1,
                                 partition_calls_per_dynamic_read_without_write=1,
                                 write_dynamic_scope="native C4 write + fused BF16 latent/RoPE stores and stable partition + two native reads + same merge")

    def decoded_hot_kv(self):
        return self.backend.data["kv"].copy()

    def _hot_read(self, parts, base=None):
        base = self.base if base is None else base
        return base._read("bf16", parts["recent_ring"], cache=self.hot_cache, rope=self.hot_rope,
                          table=self.ring_table, lengths=self.ring_lengths, statistics=True,
                          bf16_plan=self.bf16_plan)

    def _hot_write(self):
        base = self.base
        self.hot_cache.view(-1, 1, base.dim).index_copy_(0, self.hot_slots, base.current_kv)
        self.hot_rope.view(-1, 1, base.rope_dim).index_copy_(0, self.hot_slots, base.current_rope)
        return self.hot_cache.view(base.torch.uint8)

    def partition_diagnostics(self):
        # The inherited gate restores hot latent + partition; RoPE is a second
        # BF16 buffer, so preserve it explicitly around the fused producer.
        saved = self.hot_rope.clone()
        try:
            return super().partition_diagnostics()
        finally:
            self.hot_rope.copy_(saved)

    def writer_diagnostics(self):
        """Poison both BF16 buffers and independently scatter expected bytes."""
        base, t = self.base, self.base.torch
        caches = {"cold": self.cold_cache.view(t.uint8), "hot_latent": self.hot_cache.view(t.uint8),
                  "hot_rope": self.hot_rope.view(t.uint8)}
        saved = {name: value.clone() for name, value in caches.items()}
        parts = ({name: value.clone() for name, value in self._hot_prepare.outputs.items()}
                 if self._hot_prepare is not None else {})
        report = dict(passed=True, scope="poisoned cold cache and both BF16 ring buffers; independent CPU byte scatter; restored",
                      fields={})
        try:
            for value in caches.values():
                value.fill_(165)
            self._write()
            actual = {name: value.detach().cpu().numpy().copy() for name, value in caches.items()}
            caches["cold"].fill_(165)
            self._cold_write()
            expected = {"cold": caches["cold"].detach().cpu().numpy().copy()}
            slots = self.hot_slots.detach().cpu().numpy().copy()
            for name, source, width in (("hot_latent", base.current_kv, base.dim),
                                        ("hot_rope", base.current_rope, base.rope_dim)):
                expected[name] = np.full(actual[name].shape, 165, np.uint8)
                byte_width = width*source.element_size()
                expected[name].reshape(-1, byte_width)[slots] = source.view(t.uint8).detach().cpu().numpy().reshape(-1, byte_width)
            for name in caches:
                bad = np.argwhere(actual[name] != expected[name])
                report["fields"][name] = dict(passed=not len(bad), bytes=int(expected[name].size),
                                               mismatch_count=int(len(bad)), first_bad_positions=bad[:16].tolist())
                report["passed"] &= not len(bad)
        finally:
            for name, value in caches.items():
                value.copy_(saved[name])
            for name, value in parts.items():
                self._hot_prepare.outputs[name].copy_(value)
        return report

    def reader_diagnostics(self, control_l2=.01, pipeline_l2=.001):
        from kv_window_backend import bf16_subset_state_reference, diagnostic_merge_log_partitions
        if not all(np.isfinite(x) and x > 0 for x in (control_l2, pipeline_l2)):
            raise ValueError("Reader tolerances must be finite and positive")
        base, parts = self.base, self.prepared
        left, right = self._cold_read(parts), self._hot_read(parts)
        cp = parts["old_present"].detach().cpu().numpy().astype(bool)
        hp = parts["recent_present"].detach().cpu().numpy().astype(bool)
        global_hot = parts["recent_global"].detach().cpu().numpy().copy()
        expected = bf16_subset_state_reference(self.backend.data, global_hot, hp)
        lo, lm, ls = (base.host(x).astype(np.float64) for x in left)
        ro, rm, rs = (base.host(x).astype(np.float64) for x in right)
        report = dict(passed=True, scope="compact BF16 hot against independent FP64 subset math; mixed-dtype merge against native states",
                      checks={}, hot_plan=dict(self.bf16_plan.summary))
        def metric(name, actual, reference, tolerance):
            result = compare_outputs(actual, reference)
            result.update(tolerance=tolerance, passed=bool(result["finite"] and
                          result["relative_l2"] <= tolerance and result["max_row_relative_l2"] <= tolerance))
            report["checks"][name] = result
            report["passed"] &= result["passed"]
        metric("hot_output_vs_own_reference", ro, expected["output"], control_l2)
        with np.errstate(invalid="ignore", divide="ignore"):
            llse, rlse = lm+np.log(ls), rm+np.log(rs)
        good_stats = bool(np.isfinite(rlse[hp]).all() and np.all(rs[hp] > 0))
        lse_error = float(np.max(np.abs(rlse[hp]-expected["lse"][hp]))) if good_stats else None
        report["checks"]["hot_lse_vs_own_reference"] = dict(
            passed=bool(good_stats and lse_error <= .02), max_abs=lse_error, tolerance=.02)
        report["passed"] &= report["checks"]["hot_lse_vs_own_reference"]["passed"]
        host_merged, _ = diagnostic_merge_log_partitions(lo, llse, ro, rlse, cp, hp)
        rounded = merge_reference_for_dtype(host_merged, self.merge_output.dtype)
        actual = base.host(self._merge(left, right, parts))
        metric("mixed_dtype_merge_vs_host_native_states", actual, rounded, pipeline_l2)
        report["merge_reference_destination_dtype"] = str(self.merge_output.dtype)
        report["merge_destination_rounding_vs_unrounded_reference"] = compare_outputs(rounded, host_merged)
        return report
