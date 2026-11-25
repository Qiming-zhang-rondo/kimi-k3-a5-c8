"""CPU contracts and math for the new standalone single-attention prototype.

The installed C4 tiling ABI is unknown. single_pass_tiling.h deliberately
defines a NEW host/device ABI1. These references do not emulate the native
probability cast/FP32 accumulation, validate AscendC lowering, or claim speed.
"""
from dataclasses import dataclass
import math

import numpy as np


class ContractError(ValueError):
    pass


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ContractError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ContractError(f"{name} must be >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class SinglePassContract:
    queries: int = 4
    keys: int = 8192
    topk: int = 2048
    heads: int = 64
    ring_capacity: int = 256
    window: int = 128
    aic_count: int = 4
    scale: float = 1.0 / 24.0

    def __post_init__(self):
        for name in ("queries", "keys", "topk", "heads", "ring_capacity", "aic_count"):
            _integer(getattr(self, name), name)
        _integer(self.window, "window", 0)
        if self.queries > 1024 or self.keys < self.queries or self.keys > np.iinfo(np.int32).max:
            raise ContractError("Prototype requires Q1..1024 and Q<=K<=INT32_MAX")
        if self.heads != 64 or self.topk > 2048 or self.aic_count > self.queries:
            raise ContractError("Prototype fixes H64, TopK<=2048 and 1<=AIC<=Q")
        if self.window not in (0, 128):
            raise ContractError("window must be 0 or 128")
        if self.ring_capacity % 128 or self.ring_capacity < self.window+self.queries-1:
            raise ContractError("Ring must be page128-aligned and retain the earliest query's complete window")
        if not math.isfinite(self.scale) or self.scale <= 0 or not np.isfinite(np.float32(self.scale)):
            raise ContractError("scale must remain finite and positive in FP32")

    def input_specs(self):
        """Logical shapes; root's direct wrapper owns its new pointer ABI."""
        return dict(query=((self.queries, 64, 576), "bf16"),
            cold_packed=((math.ceil(self.keys/128), 128, 1, 400), "raw_uint8"),
            topk=((self.queries, 1, self.topk), "int32"),
            cold_block_table=((1, math.ceil(self.keys/128)), "int32"),
            actual_query_lengths=((1,), "int32"), actual_kv_lengths=((1,), "int32"),
            hot_latent=((self.ring_capacity//128, 128, 1, 512), "bf16"),
            hot_rope=((self.ring_capacity//128, 128, 1, 64), "bf16"),
            hot_tags=((self.ring_capacity,), "int64"), query_positions=((self.queries,), "int64"),
            output=((self.queries, 64, 512), "bf16"),
            maximum=((self.queries, 64, 1), "float32"), total=((self.queries, 64, 1), "float32"))

    def execution_contract(self):
        return dict(abi="new_standalone_single_pass_1_not_installed_tiling",
            query_storage="combined BF16 NoPE512+RoPE64", kv_storage="MXFP4 raw packed400",
            template=dict(query="bfloat16_t", kv="int8_t", accumulator="float", output="bfloat16_t",
                          flash_decode=False, page_attention=True, layout_query=1, layout_kv=2,
                          template_mode=1, split_g=False),
            c4_dispatch="dSizeVInput==400 selects AntiquantVFImplBIMxfp4",
            kernel_launches=1, attention_streams_per_query=1, softmax_streams_per_head=1,
            external_cold_attention=False, external_hot_attention=False, external_merge=False,
            selection="original supplied TopK order and membership unchanged",
            user_workspace_bytes=0, user_workspace_allocation_bytes=1,
            precision_reference="FP64 mathematical online softmax; not bit-exact native BF16/FP32")


def bf16_round(values):
    array = np.asarray(values, np.float32)
    if not np.isfinite(array).all():
        raise ContractError("BF16 oracle input must be finite")
    bits = array.view(np.uint32)
    result = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
    if not np.isfinite(result).all():
        raise ContractError("BF16 oracle overflow")
    return result.copy()


def _bf16_values(values, shape, name):
    values = np.asarray(values)
    if values.dtype != np.float32 or values.shape != shape:
        raise ContractError(f"{name} must hold exact BF16 values as float32 with shape {shape}")
    if not np.isfinite(values).all() or np.any(values.view(np.uint32) & np.uint32(0xffff)):
        raise ContractError(f"{name} must be finite original BF16, not dequantized C4 substitutes")
    return values


def decode_c4_rows(packed):
    """Actual source branch: 256 nibble bytes, 128 BF16 RoPE, 16 E8M0.

    GM stride is400. The kernel pads each fetched GM row to416 UB bytes; that
    UB stride is NOT a different cache layout. E2M1 is low-nibble first.
    """
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] != 400:
        raise ContractError("C4 rows must be raw uint8 [N,400], not FP8/TurboQuant")
    if np.any(packed[:, 384:400] == 255):
        raise ContractError("E8M0 byte255 is nonfinite")
    codes = np.empty((len(packed), 512), np.uint8)
    codes[:, 0::2], codes[:, 1::2] = packed[:, :256] & 15, packed[:, :256] >> 4
    magnitudes = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.], np.float32)
    latent = magnitudes[codes & 7]
    latent = np.where((codes & 8) != 0, -latent, latent)
    scales = np.ldexp(np.ones((len(packed),16), np.float32), packed[:,384:400].astype(np.int32)-127)
    latent = bf16_round(latent * np.repeat(scales, 32, axis=1))
    rope_bytes = np.ascontiguousarray(packed[:,256:384])
    rope_u16 = rope_bytes.view("<u2").reshape(len(packed),64).astype(np.uint32)
    rope = (rope_u16 << 16).view(np.float32)
    if not np.isfinite(rope).all():
        raise ContractError("Selected cold RoPE contains nonfinite values")
    return latent, rope


def _int_array(value, shape, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or array.shape != shape:
        raise ContractError(f"{name} must be an integer array with shape {shape}")
    if array.dtype.kind == "u" and array.size and array.max() > np.iinfo(np.int64).max:
        raise ContractError(f"{name} exceeds int64")
    return array.astype(np.int64, copy=False)


def validate_inputs(contract, *, query, cold_packed, block_table, topk, positions,
                    hot_latent, hot_rope, hot_tags):
    """Strict prototype host gate before a new device launch; no repairs."""
    q = _bf16_values(query, (contract.queries,64,576), "query")
    cold = np.asarray(cold_packed)
    if cold.dtype != np.uint8 or cold.ndim != 4 or cold.shape[1:] != (128,1,400):
        raise ContractError("cold_packed must be raw uint8 [blocks,128,1,400]")
    pages = math.ceil(contract.keys/128)
    table = _int_array(block_table, (1,pages), "block_table")
    if np.any(table < 0) or np.any(table >= cold.shape[0]):
        raise ContractError("Cold page table points outside the actual packed allocation")
    ids = _int_array(topk, (contract.queries,1,contract.topk), "topk")
    pos = _int_array(positions, (contract.queries,), "positions")
    if np.any(pos < 0) or np.any(pos >= contract.keys):
        raise ContractError("Query positions must be logical in-range positions")
    if np.any(ids < -1) or np.any(ids >= contract.keys):
        raise ContractError("TopK IDs out of range or invalid padding")
    latent = np.asarray(hot_latent)
    rope = np.asarray(hot_rope)
    if latent.shape == (contract.ring_capacity//128,128,1,512):
        latent = latent.reshape(contract.ring_capacity,512)
    if rope.shape == (contract.ring_capacity//128,128,1,64):
        rope = rope.reshape(contract.ring_capacity,64)
    if latent.dtype != np.float32 or latent.shape != (contract.ring_capacity,512):
        raise ContractError("hot_latent must be the separate original BF16 ring")
    if rope.dtype != np.float32 or rope.shape != (contract.ring_capacity,64):
        raise ContractError("hot_rope must be the separate original BF16 ring")
    tags = _int_array(hot_tags, (contract.ring_capacity,), "hot_tags")
    for row, p in enumerate(pos):
        raw = ids[row,0]
        missing = raw == -1
        if missing.any() and np.any(raw[np.argmax(missing):] >= 0):
            raise ContractError("TopK -1 padding must be a suffix")
        selected = raw[raw >= 0]
        if np.any(selected > p):
            raise ContractError("Future TopK ID: prototype rejects instead of changing the supplied TopK")
        if len(np.unique(selected)) != len(selected):
            raise ContractError("Duplicate TopK ID would double attention mass")
        hot = selected[selected >= p-contract.window+1] if contract.window else selected[:0]
        if len(hot):
            slots = hot % contract.ring_capacity
            if not np.array_equal(tags[slots], hot):
                raise ContractError("Selected window token lacks its exact original BF16 tag")
            _bf16_values(latent[slots], (len(hot),512), "selected hot_latent")
            _bf16_values(rope[slots], (len(hot),64), "selected hot_rope")
    return dict(query=q, cold_packed=cold, block_table=table, topk=ids, positions=pos,
                hot_latent=latent, hot_rope=rope, hot_tags=tags)


def _selected_row(contract, inputs, row):
    """Gather source precision per original index, preserving its exact order."""
    selected = inputs["topk"][row,0]
    selected = selected[selected >= 0]
    p = inputs["positions"][row]
    hot = selected >= p-contract.window+1 if contract.window else np.zeros(len(selected), bool)
    latent = np.empty((len(selected),512), np.float32)
    rope = np.empty((len(selected),64), np.float32)
    if (~hot).any():
        cold_ids = selected[~hot]
        physical_pages = inputs["block_table"][0,cold_ids//128]
        packed = inputs["cold_packed"][physical_pages,cold_ids % 128,0]
        latent[~hot], rope[~hot] = decode_c4_rows(packed)
    if hot.any():
        slots = selected[hot] % contract.ring_capacity
        latent[hot], rope[hot] = inputs["hot_latent"][slots], inputs["hot_rope"][slots]
    return selected, hot, np.concatenate((latent,rope), axis=1).astype(np.float64), latent.astype(np.float64)


def online_attention_reference(contract, *, block_keys=128, **arrays):
    """One running (m,z,numerator) per query/head across cold AND hot tokens."""
    block_keys = _integer(block_keys, "block_keys")
    inputs = validate_inputs(contract, **arrays)
    shape = (contract.queries,64)
    output = np.zeros(shape+(512,), np.float64)
    maxima = np.full(shape, -np.inf, np.float64)
    sums = np.zeros(shape, np.float64)
    counts, hot_counts = [], []
    for row in range(contract.queries):
        selected, hot, keys, values = _selected_row(contract, inputs, row)
        counts.append(len(selected)); hot_counts.append(int(hot.sum()))
        numerator = np.zeros((64,512), np.float64)
        q = inputs["query"][row].astype(np.float64)
        for begin in range(0, len(selected), block_keys):
            stop = min(begin+block_keys, len(selected))
            scores = (q @ keys[begin:stop].T)*contract.scale
            tile_max = scores.max(axis=1)
            previous_max = maxima[row].copy()
            new_max = np.maximum(previous_max, tile_max)
            old_weight = np.exp(previous_max-new_max)
            probabilities = np.exp(scores-new_max[:,None])
            numerator = numerator*old_weight[:,None] + probabilities @ values[begin:stop]
            sums[row] = sums[row]*old_weight + probabilities.sum(axis=1)
            maxima[row] = new_max
        if len(selected):
            output[row] = numerator/sums[row,:,None]
    lse = np.full(shape, -np.inf, np.float64)
    present = sums > 0
    lse[present] = maxima[present] + np.log(sums[present])
    return dict(output=output, maximum=maxima[:,:,None], total=sums[:,:,None], lse=lse[:,:,None],
                valid_counts=np.asarray(counts), hot_counts=np.asarray(hot_counts),
                selected_topk=inputs["topk"].copy(), attention_streams_per_query=1,
                external_merge=False, scope="FP64 mathematical oracle, not native cast/rounding emulation")


def dense_attention_reference(contract, **arrays):
    """Separate direct softmax expression used to test the online recurrence."""
    inputs = validate_inputs(contract, **arrays)
    output = np.zeros((contract.queries,64,512), np.float64)
    lse = np.full((contract.queries,64,1), -np.inf, np.float64)
    for row in range(contract.queries):
        selected, _, keys, values = _selected_row(contract, inputs, row)
        if not len(selected):
            continue
        scores = inputs["query"][row].astype(np.float64) @ keys.T * contract.scale
        maximum = scores.max(axis=1, keepdims=True)
        weights = np.exp(scores-maximum)
        output[row] = (weights/weights.sum(axis=1,keepdims=True)) @ values
        lse[row,:,0] = maximum[:,0] + np.log(weights.sum(axis=1))
    return dict(output=output, lse=lse)
