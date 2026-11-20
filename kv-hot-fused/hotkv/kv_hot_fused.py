"""Experimental single-launch C4 + C8 sparse MLA reader for Ascend Triton.

This is a real kernel, not a wrapper around two attention calls. It consumes
the existing packed caches directly and maintains one online softmax across
all selected keys. No A5 performance or compiler support is asserted here.

Revision 3 keeps decode/address temporaries at BK x 64, assembles one BF16
BK x 512 tile, then reuses it for full-width QK and PV. Q is loaded once per
program. Each key tile uses two dot operations (three with RoPE), instead of
the feature-streamed revision's sixteen/seventeen. All-cold/all-hot tiles
decode only their own format. Dynamic feature loops and disabled multi-
buffering keep decoder liveness bounded; compiler scratch remains unknown.
Neither A5 compilation nor performance is established by the source-level
operation counts. Hardware remeasurement is mandatory.

Contract: D=512, RoPE D=0/64, one shared KV head, unique fixed sparse indices.
The complete C4 history remains available. A recent key is read from the C8
ring iff ``query_position-window < key <= query_position``. Ring position
tags must match every requested recent key; stale/missing tags produce NaN
output rather than silently reading C4. All -1 and future keys are masked.
Queries must already use the same latent basis as the stored KV (including
any rotation folded into model weights). This module adds no rotation.

Layout (byte offsets within one token):
  C4: E2M1[256], BF16-RoPE[2*R], E8M0-scale[16] (one scale/32).
  C8: E4M3FN[512], BF16-RoPE[2*R], float32-scale[16] (one scale/128).
C4 nibbles are low-first. BF16 and float32 bytes are little-endian.

Only static host metadata is validated here. The caller validates index
uniqueness and logical lengths before placing input on device. All payload
loads are masked on device; invalid page/row metadata produces NaN safely.
Prepare and first-call compile are outside measurement/capture. After that,
``call()`` performs exactly one attention kernel launch, no allocations or
host copies. Keep input tensors alive and update them in-place for graph use.

Algorithm reference (Ascend's public implementation):
https://github.com/Ascend/triton-ascend/blob/main/docs/en/examples/04_fused_attention_example.md
"""

from dataclasses import dataclass
import math
import struct


@dataclass(frozen=True)
class FusedLayout:
    rows: int
    heads: int
    sparse_capacity: int
    rope_dim: int
    batches: int
    cold_pages: int
    table_pages: int
    page: int
    ring_capacity: int
    cold_width: int
    hot_width: int
    window: int
    scale: float
    block_keys: int
    block_heads: int


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_inputs(q_nope, q_rope, indices, cold_cache, cold_table,
                    hot_cache, hot_positions, row_batch, query_positions,
                    kv_lengths, *, window, page, scale,
                    block_keys=16, block_heads=16):
    """Shape/dtype/device validation; never examines device tensor values."""
    window = _integer(window, "window")
    page = _integer(page, "page")
    if block_keys not in (16, 32, 64, 128) or isinstance(block_keys, bool):
        raise ValueError("block_keys must be 16, 32, 64 or 128")
    if block_heads != 16 or isinstance(block_heads, bool):
        raise ValueError("block_heads must be 16 for this experimental kernel")
    if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive host number")
    names = ("q_nope", "q_rope", "indices", "cold_cache", "cold_table",
             "hot_cache", "hot_positions", "row_batch", "query_positions", "kv_lengths")
    tensors = (q_nope, q_rope, indices, cold_cache, cold_table,
               hot_cache, hot_positions, row_batch, query_positions, kv_lengths)
    device = str(q_nope.device)
    if device.split(":", 1)[0] not in ("npu", "privateuseone"):
        raise ValueError("FusedHotKVReader requires Ascend NPU tensors")
    shapes = {}
    for name, tensor in zip(names, tensors):
        shapes[name] = tuple(tensor.shape)
        if str(tensor.device) != device:
            raise ValueError(f"{name} must be on the same NPU as q_nope")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous; prepare outside the timed call")
        allowed = ("torch.bfloat16",) if name in ("q_nope", "q_rope") else (
            ("torch.uint8", "torch.int8") if name in ("cold_cache", "hot_cache")
            else ("torch.int32", "torch.int64"))
        if str(tensor.dtype) not in allowed:
            raise ValueError(f"{name} dtype must be one of {allowed}, got {tensor.dtype}")
        if getattr(tensor, "requires_grad", False):
            raise ValueError("FusedHotKVReader is inference-only")
    qshape = shapes["q_nope"]
    if len(qshape) != 3 or min(qshape) < 1 or qshape[2] != 512:
        raise ValueError("q_nope must have shape [T,H,512] with T,H > 0")
    rows, heads, _ = qshape
    rshape = shapes["q_rope"]
    if len(rshape) != 3 or rshape[:2] != (rows, heads) or rshape[-1] not in (0, 64):
        raise ValueError("q_rope must have shape [T,H,R], R in {0,64}")
    rope = rshape[-1]
    ishape = shapes["indices"]
    if len(ishape) != 3 or ishape[:2] != (rows, 1) or ishape[-1] < 1:
        raise ValueError("indices must have shape [T,1,S], S > 0")
    cwidth, hwidth = 272 + 2 * rope, 528 + 2 * rope
    cshape = shapes["cold_cache"]
    if len(cshape) != 4 or cshape[0] < 1 or cshape[1:] != (page, 1, cwidth):
        raise ValueError(f"cold_cache must be raw bytes [P,{page},1,{cwidth}]")
    hshape = shapes["hot_cache"]
    if len(hshape) != 4 or min(hshape[:2]) < 1 or hshape[2:] != (1, hwidth):
        raise ValueError(f"hot_cache must be raw bytes [B,capacity,1,{hwidth}]")
    batch, capacity = hshape[:2]
    if capacity < window:
        raise ValueError("ring capacity must be >= window; batched queries may need more retained history")
    if shapes["hot_positions"] != (batch, capacity):
        raise ValueError("hot_positions must be [B,ring_capacity]")
    tshape = shapes["cold_table"]
    if len(tshape) != 2 or tshape[0] != batch or tshape[1] < 1:
        raise ValueError("cold_table must have shape [B,number_of_logical_pages]")
    for name in ("row_batch", "query_positions"):
        if shapes[name] != (rows,):
            raise ValueError(f"{name} must have shape [T]")
    if shapes["kv_lengths"] != (batch,):
        raise ValueError("kv_lengths must have shape [B]")
    return FusedLayout(rows, heads, ishape[-1], rope, batch, cshape[0],
                       tshape[1], page, capacity, cwidth, hwidth, window,
                       float(scale), block_keys, block_heads)


def decode_c4_token(raw, rope_dim=64):
    """Independent host format oracle, for byte-layout checks, never timed."""
    raw = bytes(raw)
    if rope_dim not in (0, 64) or len(raw) != 272 + 2 * rope_dim:
        raise ValueError("incorrect C4 token byte count or RoPE dimension")
    lut = (0.0, .5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
           -0.0, -.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
    scales = raw[256 + 2 * rope_dim:]
    if 255 in scales:
        raise ValueError("E8M0 scale 255 is NaN")
    latent = [lut[(raw[d // 2] >> (4 * (d % 2))) & 15]
              * math.ldexp(1.0, scales[d // 32] - 127) for d in range(512)]
    rope = [_bf16_from_bytes(raw[256 + 2 * d:258 + 2 * d]) for d in range(rope_dim)]
    return latent, rope


def decode_c8_token(raw, rope_dim=64):
    """Independent host E4M3FN/block-scale layout oracle, never timed."""
    raw = bytes(raw)
    if rope_dim not in (0, 64) or len(raw) != 528 + 2 * rope_dim:
        raise ValueError("incorrect C8 token byte count or RoPE dimension")
    scales = struct.unpack("<4f", raw[512 + 2 * rope_dim:])
    latent = []
    for d, byte in enumerate(raw[:512]):
        exponent, mantissa = (byte >> 3) & 15, byte & 7
        value = (math.ldexp(float(mantissa), -9) if exponent == 0 else
                 math.ldexp(1.0 + mantissa / 8.0, exponent - 7))
        if exponent == 15 and mantissa == 7:
            value = math.nan
        latent.append(math.copysign(value, -1.0 if byte & 128 else 1.0) * scales[d // 128])
    rope = [_bf16_from_bytes(raw[512 + 2 * d:514 + 2 * d]) for d in range(rope_dim)]
    return latent, rope


def _bf16_from_bytes(pair):
    return struct.unpack("<f", b"\x00\x00" + pair)[0]


_KERNEL = None


def _load_kernel():
    """Load the installed Ascend Triton only when an NPU runner is prepared."""
    global _KERNEL, tl
    if _KERNEL is not None:
        return _KERNEL
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("Install the Triton build matching this Ascend environment; no fallback is used") from exc

    @triton.jit
    def _decode_c4_tile(COLD, cold_base, use_cold, feature_start,
                        R: tl.constexpr, BK: tl.constexpr):
        # Keep every expensive address/bitwise/float temporary at BK x 64.
        # A dynamic feature loop calls this helper; do not statically unroll
        # all eight tiles or construct full-width integer/FP32 decoder state.
        dd = feature_start + tl.arange(0, 64)
        packed4 = tl.load(COLD + cold_base[:, None] + dd[None, :] // 2,
                          mask=use_cold[:, None], other=0).to(tl.uint8).to(tl.int32)
        code = (packed4 >> ((dd[None, :] & 1) * 4)) & 15
        mag = code & 7
        c4val = tl.where(mag < 2, mag * 0.5,
                         tl.exp2(((mag >> 1) - 1).to(tl.float32)) * (1.0 + (mag & 1) * 0.5))
        c4val = tl.where((code & 8) != 0, -c4val, c4val)
        # A 64-feature tile has only two E8M0 scales. Load these once per
        # key, not once per decoded scalar, then broadcast within each group.
        groups = feature_start // 32 + tl.arange(0, 2)
        e8 = tl.load(COLD + cold_base[:, None] + 256 + 2 * R + groups[None, :],
                     mask=use_cold[:, None], other=127).to(tl.uint8).to(tl.int32)
        scales4 = tl.exp2((e8 - 127).to(tl.float32))
        scales4 = tl.where(e8 == 255, float("nan"), scales4)
        c4val = (tl.reshape(c4val, (BK, 2, 32)) * scales4[:, :, None]).reshape((BK, 64))
        return c4val.to(tl.bfloat16)

    @triton.jit
    def _decode_c8_tile(HOT, hot_base, use_hot, feature_start,
                        R: tl.constexpr, BK: tl.constexpr):
        dd = feature_start + tl.arange(0, 64)
        packed8 = tl.load(HOT + hot_base[:, None] + dd[None, :],
                          mask=use_hot[:, None], other=0).to(tl.uint8).to(tl.int32)
        exp8 = (packed8 >> 3) & 15
        mant8 = packed8 & 7
        c8val = tl.where(exp8 == 0, mant8 * (1.0 / 512.0),
                         (1.0 + mant8 * 0.125) * tl.exp2((exp8 - 7).to(tl.float32)))
        c8val = tl.where((packed8 & 128) != 0, -c8val, c8val)
        c8val = tl.where((exp8 == 15) & (mant8 == 7), float("nan"), c8val)
        # Every aligned 64-feature tile lies inside ONE 128-feature C8 scale
        # group. Its four raw bytes are BK-vectors, not BK x 64 matrices.
        scale_ptr = HOT + hot_base + 512 + 2 * R + (feature_start // 128) * 4
        s0 = tl.load(scale_ptr, mask=use_hot, other=0).to(tl.uint8).to(tl.uint32)
        s1 = tl.load(scale_ptr + 1, mask=use_hot, other=0).to(tl.uint8).to(tl.uint32)
        s2 = tl.load(scale_ptr + 2, mask=use_hot, other=0).to(tl.uint8).to(tl.uint32)
        s3 = tl.load(scale_ptr + 3, mask=use_hot, other=0).to(tl.uint8).to(tl.uint32)
        scale8 = (s0 | (s1 << 8) | (s2 << 16) | (s3 << 24)).to(tl.float32, bitcast=True)
        c8val = c8val * scale8[:, None]
        return c8val.to(tl.bfloat16)

    @triton.jit
    def _assemble_latent(COLD, HOT, cold_base, hot_base, use_cold, use_hot,
                          R: tl.constexpr, BK: tl.constexpr, FORMAT: tl.constexpr):
        # Retain only BF16 at full width: 16 KiB for BK=16. A dynamic loop
        # releases each feature decoder's wide integer/FP32 temporaries before
        # the next one. Overwrite the destination slice; avoid zero-expanded
        # FP32 accumulator additions (the expensive update pattern in v2).
        latent_tiles = tl.full((BK, 8, 64), 0, tl.bfloat16)
        feature_tiles = tl.arange(0, 8)
        for feature_start in tl.range(0, 512, 64):
            if FORMAT == 0:
                part = _decode_c4_tile(COLD, cold_base, use_cold, feature_start, R, BK)
            elif FORMAT == 1:
                part = _decode_c8_tile(HOT, hot_base, use_hot, feature_start, R, BK)
            else:
                cold_part = _decode_c4_tile(COLD, cold_base, use_cold, feature_start, R, BK)
                hot_part = _decode_c8_tile(HOT, hot_base, use_hot, feature_start, R, BK)
                part = tl.where(use_hot[:, None], hot_part, cold_part)
            latent_tiles = tl.where((feature_tiles == feature_start // 64)[None, :, None],
                                    part[:, None, :], latent_tiles)
        return tl.reshape(latent_tiles, (BK, 512))

    @triton.jit
    def _mixed_sparse_mla(Q, QR, IDX, COLD, TABLE, HOT, TAGS, ROW_BATCH,
                          POS, LENGTHS, OUT,
                          H: tl.constexpr, S: tl.constexpr, R: tl.constexpr,
                          B: tl.constexpr, PAGES: tl.constexpr, TABLE_PAGES: tl.constexpr,
                          PAGE: tl.constexpr, CAP: tl.constexpr,
                          CW: tl.constexpr, HW: tl.constexpr,
                          WINDOW: tl.constexpr, SCALE: tl.constexpr,
                          BK: tl.constexpr, BH: tl.constexpr, HOT_ENABLED: tl.constexpr):
        row = tl.program_id(0)
        hh = tl.program_id(1) * BH + tl.arange(0, BH)
        dd = tl.arange(0, 512)
        kk = tl.arange(0, BK)
        batch = tl.load(ROW_BATCH + row).to(tl.int64)
        pos = tl.load(POS + row).to(tl.int64)
        batch_ok = (batch >= 0) & (batch < B)
        length = tl.load(LENGTHS + batch, mask=batch_ok, other=0).to(tl.int64)
        row_ok = batch_ok & (pos >= 0) & (pos < length) & (length <= TABLE_PAGES * PAGE)
        q = tl.load(Q + (row.to(tl.int64) * H + hh[:, None]) * 512 + dd[None, :],
                    mask=hh[:, None] < H, other=0)
        if R > 0:
            rr = tl.arange(0, R)
            qr = tl.load(QR + (row.to(tl.int64) * H + hh[:, None]) * R + rr[None, :],
                         mask=hh[:, None] < H, other=0)
        # Source-level resident state at BH=BK=16: Q 16 KiB, accumulator
        # 32 KiB, assembled BF16 KV 16 KiB. These are NOT compiler UB totals;
        # decoder, dot-result and old/new loop buffers add to peak storage.
        acc = tl.full((BH, 512), 0, tl.float32)
        normalizer = tl.full((BH,), 0, tl.float32)
        maximum = tl.full((BH,), float("-inf"), tl.float32)
        bad = ~row_ok
        for start in tl.range(0, S, BK):
            ids = tl.load(IDX + row.to(tl.int64) * S + start + kk,
                          mask=start + kk < S, other=-1).to(tl.int64)
            valid = row_ok & (ids >= 0) & (ids <= pos) & (ids < length)
            if HOT_ENABLED:
                wanted_hot = valid & (ids > pos - WINDOW)
                slot = tl.maximum(ids, 0) % CAP
                tag = tl.load(TAGS + batch * CAP + slot, mask=wanted_hot, other=-1).to(tl.int64)
                bad = bad | (tl.sum((wanted_hot & (tag != ids)).to(tl.int32), 0) > 0)
                use_hot = wanted_hot & (tag == ids)
                hot_base = (batch * CAP + slot) * HW
            else:
                # Explicit diagnostic control only. This removes tag reads and
                # C8 decoding at compile time; native W0 remains a separate path.
                wanted_hot = valid & False
                use_hot = valid & False
                hot_base = tl.full((BK,), 0, tl.int64)
            logical_page = tl.maximum(ids, 0) // PAGE
            physical = tl.load(TABLE + batch * TABLE_PAGES + logical_page,
                               mask=valid & (logical_page < TABLE_PAGES), other=-1).to(tl.int64)
            page_ok = (physical >= 0) & (physical < PAGES)
            bad = bad | (tl.sum((valid & ~page_ok).to(tl.int32), 0) > 0)
            valid = valid & page_ok
            use_cold = valid & ~wanted_hot
            use_hot = use_hot & valid
            cold_base = (physical * PAGE + tl.maximum(ids, 0) % PAGE) * CW
            # Choose a decoder once per key tile, not once per feature. Old
            # tiles do not execute C8 conversion or load its payload/scales.
            if HOT_ENABLED:
                if tl.sum(use_hot.to(tl.int32), 0) == 0:
                    latent = _assemble_latent(COLD, HOT, cold_base, hot_base,
                                               use_cold, use_hot, R, BK, 0)
                elif tl.sum(use_cold.to(tl.int32), 0) == 0:
                    latent = _assemble_latent(COLD, HOT, cold_base, hot_base,
                                               use_cold, use_hot, R, BK, 1)
                else:
                    latent = _assemble_latent(COLD, HOT, cold_base, hot_base,
                                               use_cold, use_hot, R, BK, 2)
            else:
                latent = _assemble_latent(COLD, HOT, cold_base, hot_base,
                                           use_cold, use_hot, R, BK, 0)
            scores = tl.dot(q, tl.trans(latent))
            if R > 0:
                # Both copies retain the same unquantized RoPE bytes.
                rp = COLD + cold_base[:, None] + 256 + 2 * rr[None, :]
                r0 = tl.load(rp, mask=valid[:, None], other=0).to(tl.uint8).to(tl.uint16)
                r1 = tl.load(rp + 1, mask=valid[:, None], other=0).to(tl.uint8).to(tl.uint16)
                rope = (r0 | (r1 << 8)).to(tl.bfloat16, bitcast=True)
                scores = scores + tl.dot(qr, tl.trans(rope))
            scores = scores * SCALE
            scores = tl.where(valid[None, :] & (hh[:, None] < H), scores, float("-inf"))
            new_max = tl.maximum(maximum, tl.max(scores, 1))
            safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
            alpha = tl.where(normalizer > 0, tl.exp(maximum - safe_max), 0.0)
            prob = tl.where(valid[None, :] & (hh[:, None] < H),
                            tl.exp(scores - safe_max[:, None]), 0.0)
            probability = prob.to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(probability, latent)
            normalizer = normalizer * alpha + tl.sum(prob, 1)
            maximum = new_max
        result = acc / tl.where(normalizer > 0, normalizer, 1.0)[:, None]
        result = tl.where(bad, float("nan"), result)
        tl.store(OUT + (row.to(tl.int64) * H + hh[:, None]) * 512 + dd[None, :],
                 result.to(tl.bfloat16), mask=hh[:, None] < H)

    _KERNEL = _mixed_sparse_mla
    return _KERNEL


class FusedHotKVReader:
    """Prepared, experimental one-launch reader; constructor is not timed.

    Call once and synchronize before graph capture or benchmark warmup.
    Compiling/running on A5 is mandatory before interpreting performance.
    The returned tensor is reused and is overwritten by subsequent calls.
    """

    def __init__(self, q_nope, q_rope, indices, cold_cache, cold_table,
                 hot_cache, hot_positions, row_batch, query_positions,
                 kv_lengths, *, window, page, scale, block_keys=16, block_heads=16,
                 diagnostic_cold_only=False):
        if not isinstance(diagnostic_cold_only, bool):
            raise ValueError("diagnostic_cold_only must be a bool")
        self.layout = validate_inputs(q_nope, q_rope, indices, cold_cache,
                                      cold_table, hot_cache, hot_positions,
                                      row_batch, query_positions, kv_lengths,
                                      window=window, page=page, scale=scale,
                                      block_keys=block_keys, block_heads=block_heads)
        import torch
        self.output = torch.empty((self.layout.rows, self.layout.heads, 512),
                                  dtype=torch.bfloat16, device=q_nope.device)
        self._inputs = (q_nope, q_rope, indices, cold_cache, cold_table,
                        hot_cache, hot_positions, row_batch, query_positions, kv_lengths)
        self._kernel = _load_kernel()
        self._grid = (self.layout.rows, (self.layout.heads + block_heads - 1) // block_heads)
        self._constants = dict(H=self.layout.heads, S=self.layout.sparse_capacity,
                               R=self.layout.rope_dim, B=self.layout.batches,
                               PAGES=self.layout.cold_pages, TABLE_PAGES=self.layout.table_pages,
                               PAGE=page, CAP=self.layout.ring_capacity,
                               CW=self.layout.cold_width, HW=self.layout.hot_width,
                               WINDOW=window, SCALE=self.layout.scale, BK=block_keys, BH=block_heads,
                               HOT_ENABLED=not diagnostic_cold_only)
        self.metadata = {"implementation": "ascend_triton_fused_c4_c8",
                         "status": "EXPERIMENTAL_REQUIRES_A5_VALIDATION",
                         "attention_launches_per_call": 1,
                         "separate_merge_launches": 0,
                         "revision": 3, "feature_tile": 64, "multibuffer": False,
                         "kv_decode_passes_per_key_tile": 1,
                         "qk_passes_per_key_tile": 1,
                         "source_dot_calls_per_key_tile": 3 if self.layout.rope_dim else 2,
                         "source_q_loads_per_program": 1,
                         "format_dispatch": "once per key tile; cold/hot/mixed",
                         "resource_fix": "dynamic BKx64 decoding; one retained BF16 BKx512 tile reused by QK and PV",
                         "resource_validation": "A5 compile required; source-level bounds are not compiler UB accounting",
                         "block_keys": block_keys, "block_heads": block_heads,
                         "window": window, "ring_capacity": self.layout.ring_capacity,
                         "semantic_window": 0 if diagnostic_cold_only else window,
                         "diagnostic_cold_only": diagnostic_cold_only,
                         "missing_hot_policy": "NaN", "fallback": None,
                         "accumulation": "FP32; BF16 dot operands"}

    def call(self):
        """Enqueue exactly one attention kernel, with preallocated output."""
        self._kernel[self._grid](*self._inputs, self.output, **self._constants, multibuffer=False)
        return self.output
