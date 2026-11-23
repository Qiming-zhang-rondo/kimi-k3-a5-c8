"""Experimental C8 recent-attention plus native-cold-state merge in one launch.

The cold attention remains the installed native C4 reader. This kernel only
reads a prepartitioned recent C8 subset and merges its normalized result with
the supplied native cold output/max/sum. It never decodes or recomputes C4.
Indices and presence masks must come from the independently validated stable
partition. Presence is authoritative, including native dummy-zero rows.

C8 bytes: E4M3FN[512], original BF16 RoPE[2*R], four FP32 scale/128 values.
The proven 64-feature decoder is reused from kv_hot_fused's format arithmetic.
Each BK tile is decoded once into BF16, then used for one QK, one optional
RoPE QK and one PV. Dynamic 2-D insert_slice avoids the previous 3-D full-tile
where assembly; Q is loaded once per program. Source tile sizes do not prove
compiler UB usage. Larger BH/BK choices are explicit experiments, not fallbacks.

Required installed API (no package upgrades or replacement are performed):
triton.language.extra.cann.extension.insert_slice
https://triton-ascend.readthedocs.io/en/latest/python-api/generated/triton.language.extra.cann.extension.insert_slice.html

No A5 correctness or performance claim follows from the CPU tests. Warm up
before graph capture. Each call enqueues one kernel, with preallocated output;
optional diagnostic hot-stat writes stay inside that same kernel.
Stable recent prefixes are scanned/validated once, and attention stops at the
actual prefix count. Explicit native_cast decoding uses raw-byte E4M3FN
bitcast/conversion; software remains the default with no silent fallback.
"""

from dataclasses import dataclass
import math

from kv_native_merge import validate_states


@dataclass(frozen=True)
class HotMergeLayout:
    rows: int
    heads: int
    rope_dim: int
    batches: int
    pages: int
    table_pages: int
    page: int
    ring_capacity: int
    hot_width: int
    scale: float
    block_keys: int
    block_heads: int
    hot_output_dtype: str
    decode: str


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _metadata(tensor, name, shape, dtypes, device):
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if str(tensor.dtype) not in dtypes:
        raise ValueError(f"{name} dtype must be one of {dtypes}")
    if str(tensor.device) != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on the same NPU")
    if getattr(tensor, "requires_grad", False):
        raise ValueError("PreparedHotMerge is inference-only")


def validate_inputs(q_nope, q_rope, ring_cache, ring_table, row_batch, cold_template, *,
                    page, ring_capacity, hot_width, scale, hot_output_dtype,
                    block_keys=16, block_heads=16, decode="software"):
    cold = validate_states(cold_template, cold_template)
    page, ring_capacity, hot_width = (_positive(page, "page"), _positive(ring_capacity, "ring_capacity"),
                                     _positive(hot_width, "hot_width"))
    if block_keys not in (16, 32, 64) or isinstance(block_keys, bool):
        raise ValueError("block_keys must be 16, 32 or 64")
    if block_heads not in (16, 32) or isinstance(block_heads, bool):
        raise ValueError("block_heads must be 16 or 32")
    if decode not in ("software", "native_cast"):
        raise ValueError("decode must be software or native_cast")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if cold.features != 512:
        raise ValueError("Native cold output must have D=512")
    if str(hot_output_dtype) not in ("torch.bfloat16", "torch.float32"):
        raise ValueError("hot_output_dtype must match actual native BF16 or FP32 hot output")
    device = str(cold_template[0].device)
    rows, heads = cold.tokens, cold.heads
    _metadata(q_nope, "q_nope", (rows, heads, 512), ("torch.bfloat16",), device)
    rshape = tuple(q_rope.shape)
    if len(rshape) != 3 or rshape[:2] != (rows, heads) or rshape[-1] not in (0, 64):
        raise ValueError("q_rope must be [T,H,R], R in {0,64}")
    rope = rshape[-1]
    _metadata(q_rope, "q_rope", rshape, ("torch.bfloat16",), device)
    shape = tuple(ring_cache.shape)
    if len(shape) != 4 or shape[0] < 1 or shape[1:] != (page, 1, 528 + 2*rope):
        raise ValueError("ring_cache must be paged native raw C8 bytes [P,page,1,528+2*R]")
    _metadata(ring_cache, "ring_cache", shape, ("torch.uint8", "torch.int8"), device)
    table_shape = tuple(ring_table.shape)
    if len(table_shape) != 2 or min(table_shape) < 1 or table_shape[1] * page < ring_capacity:
        raise ValueError("ring_table must be [B,logical_pages] and cover ring_capacity")
    _metadata(ring_table, "ring_table", table_shape, ("torch.int32", "torch.int64"), device)
    _metadata(row_batch, "row_batch", (rows,), ("torch.int32", "torch.int64"), device)
    return HotMergeLayout(rows, heads, rope, table_shape[0], shape[0], table_shape[1], page,
                           ring_capacity, hot_width, float(scale), block_keys, block_heads,
                           str(hot_output_dtype), decode)


_KERNEL = None


def _load_kernel():
    global _KERNEL, tl, extension
    if _KERNEL is not None:
        return _KERNEL
    try:
        import triton
        import triton.language as tl
        import triton.language.extra.cann.extension as extension
    except ImportError as exc:
        raise RuntimeError("hot_merge requires the installed Ascend Triton API "
                           "triton.language.extra.cann.extension.insert_slice; use the unchanged native reader if unavailable") from exc
    if not hasattr(extension, "insert_slice"):
        raise RuntimeError("Installed Ascend Triton lacks extension.insert_slice required by explicit hot_merge candidate")

    @triton.jit
    def _decode_c8(HOT, base, active, feature_start, R: tl.constexpr, BK: tl.constexpr,
                    NATIVE_CAST: tl.constexpr):
        dd = feature_start + tl.arange(0, 64)
        raw = tl.load(HOT + base[:, None] + dd[None, :], mask=active[:, None], other=0).to(tl.uint8)
        if NATIVE_CAST:
            # Bitcast must use the original ONE-byte tensor, never the int32
            # software decoder temporary. Multiply FP32 scales before BF16.
            value = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            packed = raw.to(tl.int32)
            exponent, mantissa = (packed >> 3) & 15, packed & 7
            value = tl.where(exponent == 0, mantissa * (1.0/512.0),
                              (1.0 + mantissa * 0.125) * tl.exp2((exponent - 7).to(tl.float32)))
            value = tl.where((packed & 128) != 0, -value, value)
            value = tl.where((exponent == 15) & (mantissa == 7), float("nan"), value)
        pointer = HOT + base + 512 + 2*R + (feature_start // 128)*4
        s0 = tl.load(pointer, mask=active, other=0).to(tl.uint8).to(tl.uint32)
        s1 = tl.load(pointer+1, mask=active, other=0).to(tl.uint8).to(tl.uint32)
        s2 = tl.load(pointer+2, mask=active, other=0).to(tl.uint8).to(tl.uint32)
        s3 = tl.load(pointer+3, mask=active, other=0).to(tl.uint8).to(tl.uint32)
        scale = (s0 | (s1 << 8) | (s2 << 16) | (s3 << 24)).to(tl.float32, bitcast=True)
        return (value * scale[:, None]).to(tl.bfloat16)

    @triton.jit
    def _hot_merge(Q, QR, CACHE, TABLE, ROW_BATCH, INDICES, COLD, CMAX, CSUM,
                    COLD_PRESENT, HOT_PRESENT, OUT, HOT_MAX, HOT_SUM,
                    H: tl.constexpr, R: tl.constexpr, B: tl.constexpr, PAGES: tl.constexpr,
                    TABLE_PAGES: tl.constexpr, PAGE: tl.constexpr, CAP: tl.constexpr,
                    HW: tl.constexpr, SCALE: tl.constexpr, BK: tl.constexpr, BH: tl.constexpr,
                    HOT_BF16: tl.constexpr, STATISTICS: tl.constexpr,
                    NATIVE_CAST: tl.constexpr, SCAN_BLOCK: tl.constexpr):
        row = tl.program_id(0)
        hh = tl.program_id(1)*BH + tl.arange(0, BH)
        dd = tl.arange(0, 512)
        kk = tl.arange(0, BK)
        # Torch bool storage may reach the backend as a byte. Canonicalize
        # nonzero values to predicates; never bitwise-invert the loaded byte.
        cp = tl.load(COLD_PRESENT + row).to(tl.int32) != 0
        hp = tl.load(HOT_PRESENT + row).to(tl.int32) != 0
        acc = tl.full((BH, 512), 0, tl.float32)
        maximum = tl.full((BH,), float("-inf"), tl.float32)
        total = tl.full((BH,), 0, tl.float32)
        bad = False
        if hp:
            batch = tl.load(ROW_BATCH + row).to(tl.int64)
            batch_ok = (batch >= 0) & (batch < B)
            bad = batch_ok == 0
            scan_lane = tl.arange(0, SCAN_BLOCK)
            scan_ids = tl.load(INDICES + row.to(tl.int64)*HW + scan_lane,
                               mask=scan_lane < HW, other=-1).to(tl.int64)
            nonpad = (scan_lane < HW) & (scan_ids >= 0)
            prefix_count = tl.sum(nonpad.to(tl.int32), 0)
            invalid_ids = (scan_lane < HW) & ((scan_ids < -1) | (scan_ids >= CAP))
            holes = (scan_lane < HW) & (nonpad != (scan_lane < prefix_count))
            bad = bad | (prefix_count == 0) | (tl.sum((invalid_ids | holes).to(tl.int32), 0) > 0)
            q = tl.load(Q + (row.to(tl.int64)*H + hh[:, None])*512 + dd[None, :],
                        mask=hh[:, None] < H, other=0)
            if R > 0:
                rr = tl.arange(0, R)
                qr = tl.load(QR + (row.to(tl.int64)*H + hh[:, None])*R + rr[None, :],
                             mask=hh[:, None] < H, other=0)
            # The complete pre-scan checks malformed IDs and holes even when
            # attention skips the padded suffix. Prefix stores are guaranteed
            # by the independently validated stable partition.
            for start in tl.range(0, prefix_count, BK):
                ids = tl.load(INDICES + row.to(tl.int64)*HW + start + kk,
                              mask=start+kk < prefix_count, other=-1).to(tl.int64)
                valid = batch_ok & (start+kk < prefix_count) & (ids >= 0) & (ids < CAP)
                logical = tl.maximum(ids, 0) // PAGE
                physical = tl.load(TABLE + batch*TABLE_PAGES + logical,
                                   mask=valid & (logical < TABLE_PAGES), other=-1).to(tl.int64)
                page_ok = (physical >= 0) & (physical < PAGES)
                bad = bad | (tl.sum((valid & (page_ok == 0)).to(tl.int32), 0) > 0)
                valid = valid & page_ok
                base = (physical*PAGE + tl.maximum(ids, 0) % PAGE) * (528+2*R)
                latent = tl.full((BK, 512), 0, tl.bfloat16)
                for feature_start in tl.range(0, 512, 64):
                    part = _decode_c8(CACHE, base, valid, feature_start, R, BK, NATIVE_CAST)
                    latent = extension.insert_slice(latent, part, [0, feature_start], [BK, 64], [1, 1])
                scores = tl.dot(q, tl.trans(latent))
                if R > 0:
                    pointer = CACHE + base[:, None] + 512 + 2*rr[None, :]
                    lo = tl.load(pointer, mask=valid[:, None], other=0).to(tl.uint8).to(tl.uint16)
                    hi = tl.load(pointer+1, mask=valid[:, None], other=0).to(tl.uint8).to(tl.uint16)
                    rope = (lo | (hi << 8)).to(tl.bfloat16, bitcast=True)
                    scores = scores + tl.dot(qr, tl.trans(rope))
                scores = tl.where(valid[None, :] & (hh[:, None] < H), scores*SCALE, float("-inf"))
                new_max = tl.maximum(maximum, tl.max(scores, 1))
                safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
                alpha = tl.where(total > 0, tl.exp(maximum-safe_max), 0.0)
                prob = tl.where(valid[None, :] & (hh[:, None] < H),
                                tl.exp(scores-safe_max[:, None]), 0.0)
                acc = acc*alpha[:, None] + tl.dot(prob.to(tl.bfloat16), latent)
                total = total*alpha + tl.sum(prob, 1)
                maximum = new_max

        if STATISTICS:
            tl.store(HOT_MAX + row.to(tl.int64)*H + hh, maximum, mask=hh < H)
            tl.store(HOT_SUM + row.to(tl.int64)*H + hh, total, mask=hh < H)
        hot = acc / tl.where(total > 0, total, 1.0)[:, None]
        if HOT_BF16:
            hot = hot.to(tl.bfloat16).to(tl.float32)
        hot = tl.where(hp, hot, 0.0)
        cm = tl.load(CMAX + row.to(tl.int64)*H + hh, mask=hh < H, other=0)
        cs = tl.load(CSUM + row.to(tl.int64)*H + hh, mask=hh < H, other=1)
        cold_bad = cp & ((cm != cm) | (cm == float("inf")) | (cm == float("-inf"))
                         | (cs != cs) | (cs == float("inf")) | (cs <= 0))
        hot_bad = hp & ((maximum != maximum) | (maximum == float("inf")) | (maximum == float("-inf"))
                        | (total != total) | (total == float("inf")) | (total <= 0))
        cm = tl.where(cp, cm, float("-inf"))
        both_empty = (cp == 0) & (hp == 0)
        common = tl.maximum(cm, maximum)
        common = tl.where(both_empty, 0.0, common)
        clog = tl.where(cp, (cm-common) + tl.log(tl.where(cp, cs, 1.0)), float("-inf"))
        hlog = tl.where(hp, (maximum-common) + tl.log(tl.where(hp, total, 1.0)), float("-inf"))
        shift = tl.maximum(clog, hlog)
        shift = tl.where(both_empty, 0.0, shift)
        cw = tl.where(cp, tl.exp(clog-shift), 0.0)
        hw = tl.where(hp, tl.exp(hlog-shift), 0.0)
        denominator = tl.where(both_empty, 1.0, cw+hw)
        cold = tl.load(COLD + (row.to(tl.int64)*H + hh[:, None])*512 + dd[None, :],
                       mask=hh[:, None] < H, other=0).to(tl.float32)
        cold = tl.where(cp, cold, 0.0)
        result = cold*(cw/denominator)[:, None] + hot*(hw/denominator)[:, None]
        result = tl.where(bad | cold_bad[:, None] | hot_bad[:, None], float("nan"), result)
        tl.store(OUT + (row.to(tl.int64)*H + hh[:, None])*512 + dd[None, :], result, mask=hh[:, None] < H)

    _KERNEL = _hot_merge
    return _KERNEL


class PreparedHotMerge:
    """Fixed query/cache setup; call accepts current native cold state/indices."""

    def __init__(self, q_nope, q_rope, ring_cache, ring_table, row_batch, cold_template, *,
                 page, ring_capacity, hot_width, scale, hot_output_dtype, block_keys=16, block_heads=16,
                 decode="software"):
        self.layout = validate_inputs(q_nope, q_rope, ring_cache, ring_table, row_batch, cold_template,
                                      page=page, ring_capacity=ring_capacity, hot_width=hot_width, scale=scale,
                                      hot_output_dtype=hot_output_dtype, block_keys=block_keys, block_heads=block_heads,
                                      decode=decode)
        self._kernel = _load_kernel()
        import torch
        layout = self.layout
        self.output = torch.empty_like(cold_template[0])
        self.hot_statistics = (torch.empty((layout.rows, layout.heads, 1), dtype=torch.float32, device=self.output.device),
                               torch.empty((layout.rows, layout.heads, 1), dtype=torch.float32, device=self.output.device))
        self._inputs = (q_nope, q_rope, ring_cache, ring_table, row_batch)
        self._grid = (layout.rows, (layout.heads+block_heads-1)//block_heads)
        self._constants = dict(H=layout.heads, R=layout.rope_dim, B=layout.batches, PAGES=layout.pages,
                               TABLE_PAGES=layout.table_pages, PAGE=page, CAP=ring_capacity, HW=hot_width,
                               SCALE=layout.scale, BK=block_keys, BH=block_heads,
                               HOT_BF16=layout.hot_output_dtype == "torch.bfloat16",
                               NATIVE_CAST=decode == "native_cast", SCAN_BLOCK=1 << (hot_width-1).bit_length())
        self.metadata = dict(implementation="ascend_triton_c8_hot_attention_native_cold_merge",
                             status="EXPERIMENTAL_REQUIRES_A5_VALIDATION", kernel_launches=1,
                             cold_attention="unchanged native C4 reader before this kernel",
                             c4_decode=False, source_dots_per_hot_tile=3 if layout.rope_dim else 2,
                             kv_decode_passes=1, hot_output_dtype=layout.hot_output_dtype,
                             final_output_dtype=str(self.output.dtype), block_keys=block_keys,
                             block_heads=block_heads, multibuffer=False,
                             decode=decode, native_cast_type="tl.float8e4nv" if decode == "native_cast" else None,
                             hot_loop_bound="validated nonpadding prefix count, not allocated capacity",
                             prefix_validation="all HW entries checked; invalid IDs/holes/present-empty produce NaN",
                             presence_predicates="explicit int32 != 0; logical negation uses == 0 (v4.1)",
                             assembly_api="triton.language.extra.cann.extension.insert_slice",
                             statistics_default=False, fallback=None,
                             resource_validation="A5 compile required; tile arithmetic is not compiler UB accounting")

    def call(self, cold_state, recent_ring, cold_present, recent_present, *, statistics=False):
        if not isinstance(statistics, bool):
            raise ValueError("statistics must be a bool")
        validate_states(cold_state, cold_state, self.output)
        layout, device = self.layout, str(self.output.device)
        _metadata(recent_ring, "recent_ring", (layout.rows, 1, layout.hot_width),
                   ("torch.int32", "torch.int64"), device)
        _metadata(cold_present, "cold_present", (layout.rows,), ("torch.bool",), device)
        _metadata(recent_present, "recent_present", (layout.rows,), ("torch.bool",), device)
        self._kernel[self._grid](*self._inputs, recent_ring, *cold_state, cold_present, recent_present,
                                  self.output, *self.hot_statistics, **self._constants,
                                  STATISTICS=statistics, multibuffer=False)
        return self.output
