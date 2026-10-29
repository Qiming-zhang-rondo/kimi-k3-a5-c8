# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public QLI MXFP4 implementation.

The public ABI is the one in the public interface specification: q is ``uint8[T1,32,64]`` and
k is ``uint8[block_num,block_size,1,64]`` (two packed E2M1 values per byte).
The Torch-facing host layer only validates the public ABI, builds launch
metadata/workspaces, and allocates public outputs.  It never creates
test inputs or computes a CPU reference.  DSL kernels load native FP4 E2M1
into L1/L0 and perform the Cube MX QK product into an FP32 accumulator.
Cube reads paged K and scales directly from GM. Two directed F322BF16
FIXP copies deliver the M halves to their AIVs. VEC1 applies BF16 ReLU,
weighted reduction and uint16 streaming TopK, with optional candidate TopK.
"""

from __future__ import annotations

__all__ = ["quant_lightning_indexer"]

import os
import threading
from math import gcd
from typing import Optional

import cannbotdsl
import torch
from cannbotdsl.aot import export

from cannbotdsl import ChannelKind, MemLoc, Tensor, dtypes
from cannbotdsl.buffer import UB, make_buffer
from cannbotdsl.channel import Channel, make_channel
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl import TensorSpec
from cannbotdsl.lang.vf import vf
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.constexpr import range_constexpr
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.matmul import matmul as dsl_matmul
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
from cannbotdsl.ops.sync import (
    PIPE,
    global_sync_all,
    vec_sync_all,
    cube_sync_intra_wait,
    cube_sync_intra_arrive,
    vec_sync_intra_wait,
    vec_sync_intra_arrive,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.tensor import (
    reinterpret,
    make_pointer,
    make_tensor,
    make_layout,
    tile_slice,
)
from cannbotdsl.ops import reg as rr


# Group six N1=32 query rows into an M192 tile; N1=64 uses four rows.
QUERY_TILE_ROWS = 6
TILE_M = 32 * QUERY_TILE_ROWS
TILE_N = 256
QK_COLUMNS = TILE_N
CACHE_TOKENS = 25600
LOGICAL_D = 128
PACKED_D = LOGICAL_D // 2
N1 = 32
N2 = 1
VEC_TILE = 128
CANDIDATE_BLOCK_SIZE = 8
TOPK_TRUNK_LEN = 26624
MAX_CUBE_WORKERS = 32
# Metadata 的 LI 区占 36 行，LD/FD 归并记录从固定第 36 行开始。
LD_RECORD_START = 36
METADATA_ELEMENTS = 1024
METADATA_FIELDS = 8
METADATA_ROWS = METADATA_ELEMENTS // METADATA_FIELDS
# 该 TopK 容量采用缓存分数路径，其余容量使用通用流式路径。
CACHED_TOPK_COUNT = 512
BF16_NEG_INF_BITS = 0xFF80
STREAM_TOPK_TRUNK_LEN = 16384
MAX_MERGE_PARTS = 8
MAX_PA_BLOCK_SIZE = 1024
PA_BLOCK_ALIGNMENT = 16
QK_ROW_BYTES = TILE_N * 2
CACHED_QK_UB_BYTES = TILE_M * QK_ROW_BYTES
CACHED_SCORE_UB_BYTES = (QUERY_TILE_ROWS // 2) * CACHE_TOKENS * 2
CACHED_HISTOGRAM_UB_OFFSET = CACHED_QK_UB_BYTES + CACHED_SCORE_UB_BYTES
HISTOGRAM_SLOT_OFFSET = QK_ROW_BYTES // 2
CACHED_VECTOR_UB_OFFSET = (
    CACHED_HISTOGRAM_UB_OFFSET + QUERY_TILE_ROWS * QK_ROW_BYTES + QK_ROW_BYTES
)
VECTOR_AFTER_QK_BYTES = 12288
HISTOGRAM_AFTER_QK_BYTES = 16384
# 缓存 TopK 的临时索引与输出使用已排空的 QK 区，不覆盖缓存分数。
CACHED_TOPK_INDEX_OFFSET = CACHE_TOKENS * 2 + 1024
CACHED_TOPK_VALUE_OFFSET = CACHED_TOPK_INDEX_OFFSET + CACHED_TOPK_COUNT * 4
KEY_STORAGE_ROW_ELEMENTS = N2 * PACKED_D
KEY_SCALE_STORAGE_ROW_ELEMENTS = N2 * 2 * 2


@jit
def _need_topk_values(metadata, return_values):
    return dtypes.int64(1 if metadata[LD_RECORD_START, 0] != 0 else return_values)


UB_CAPACITY_BYTES = 256 * 1024
UB_ALIGNMENT_BYTES = 32
# TopK 阶段的直方图与阈值暂存；避免覆盖仍待选择的缓存分数。
TOPK_HISTOGRAM_UB_OFFSET = CACHED_HISTOGRAM_UB_OFFSET + QUERY_TILE_ROWS * QK_ROW_BYTES
TOPK_TARGET_UB_OFFSET = CACHED_VECTOR_UB_OFFSET


class _UbPool:
    """编译期 UB 区间规划：保留显式别名，为其余暂存分配空闲区间。"""

    def __init__(self, start=0):
        self.start = start
        self.regions = []

    def slots(self, shape, dtype, depth=1, stride=None, addr=None):
        if stride is None:
            strides = []
            element_stride = 1
            for axis_size in reversed(shape):
                strides.insert(0, element_stride)
                element_stride *= axis_size
            stride = tuple(strides)
        span_bytes = (
            (sum((size - 1) * step for size, step in zip(shape, stride)) + 1)
            * dtype.bits
            // 8
        )
        alignment_mask = UB_ALIGNMENT_BYTES - 1
        slot_pitch = (
            (span_bytes + alignment_mask) // UB_ALIGNMENT_BYTES * UB_ALIGNMENT_BYTES
        )
        start = (
            (self.start + alignment_mask) // UB_ALIGNMENT_BYTES * UB_ALIGNMENT_BYTES
            if addr is None
            else addr
        )
        if addr is None:
            for ub_begin, ub_end in sorted(self.regions):
                if start + slot_pitch * depth <= ub_begin:
                    break
                start = max(start, ub_end)
        self.regions.append((start, start + slot_pitch * depth))
        if (
            start < 0
            or start + span_bytes + (depth - 1) * slot_pitch > UB_CAPACITY_BYTES
        ):
            raise ValueError("QLI UB allocation exceeds the device capacity")
        ub_root = UB.view(UB_CAPACITY_BYTES)
        return [
            reinterpret(
                ub_root,
                dtype=dtype,
                offset=start + slot * slot_pitch,
                shape=shape,
                stride=stride,
            )
            for slot in range(depth)
        ]


def _ub_buffer(pool, memloc, shape, dtype, *, addr=None, stride=None):
    return make_buffer(pool.slots(shape, dtype, stride=stride, addr=addr)[0])


def _ub_channel(
    pool,
    memloc,
    shape,
    dtype,
    *,
    depth=1,
    addr=None,
    stride=None,
    kind=ChannelKind.SameCore,
):
    slots = pool.slots(shape, dtype, depth=depth, stride=stride, addr=addr)
    return make_channel(slots, kind=kind)


def _offset_view(tensor, offsets):
    """Rebase an ND input with the current DSL's zero-copy interval slicing."""
    if all(isinstance(offset, int) and offset == 0 for offset in offsets):
        return tensor
    return tensor[tuple(slice(offset, None) for offset in offsets)]


def ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _pa_axis0_storage_rows(
    tensor: torch.Tensor,
    name: str,
    expected_inner_strides: tuple[int, ...],
    storage_row_elements: int,
) -> tuple[int, int]:
    """Validate ASC's PA stride contract and expose its storage-row span.

    Only axis 0 may contain padding. The returned extent includes that
    padding without copying or repacking public input data; the fused kernel
    still selects physical pages with the original axis-0 stride.
    """
    strides = tuple(int(step) for step in tensor.stride())
    if strides[1:] != expected_inner_strides:
        raise ValueError(
            f"{name} only supports non-contiguous storage on axis 0 in PA_BBND layout"
        )
    stride0 = strides[0]
    if stride0 < 0:
        raise ValueError(f"{name} axis-0 stride must be non-negative")
    if stride0 % storage_row_elements != 0:
        raise ValueError(f"{name} axis-0 stride must preserve complete PA storage rows")
    blocks = int(tensor.shape[0])
    if blocks <= 0:
        raise ValueError(f"{name} must contain at least one physical PA block")
    storage_rows = (blocks - 1) * (stride0 // storage_row_elements) + int(
        tensor.shape[1]
    )
    return stride0, storage_rows


class QliRawTopKWorkspace:
    """UB scratch shared by sequential Sparse and Candidate TopK stages."""

    def __init__(self, max_trunk_len: int, max_topk: int, pipelined: bool = False):
        self.ub_pool = _UbPool()
        self.max_trunk_len = int(max_trunk_len)
        self.max_topk = int(max_topk)
        self.pipelined = bool(pipelined)
        self.topk_pad = ceil_div(self.max_topk, 256) * 256
        self.merge_len_pad = ceil_div(self.topk_pad + self.max_trunk_len, 128) * 128
        self.current_trunk = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (1, self.max_trunk_len),
            dtypes.uint16,
            depth=1,
            addr=4 * self.merge_len_pad,
        )
        if self.pipelined:
            self.merge_key = _ub_channel(
                self.ub_pool,
                MemLoc.UB,
                (1, self.merge_len_pad),
                dtypes.uint16,
                depth=2,
                addr=0,
            )
        else:
            self.merge_key = _ub_channel(
                self.ub_pool,
                MemLoc.UB,
                (1, self.merge_len_pad),
                dtypes.uint16,
                depth=1,
                addr=0,
            )
        self.tmp_idx = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.merge_len_pad,),
            dtypes.uint16,
        )
        self.histogram = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (256,),
            dtypes.uint16,
            addr=TOPK_HISTOGRAM_UB_OFFSET,
        )
        self.high_target_carrier = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (64,),
            dtypes.int32,
            addr=TOPK_TARGET_UB_OFFSET,
        )
        self.history_key = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (2, self.topk_pad),
            dtypes.uint16,
        )
        self.next_history_key = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (1, self.topk_pad),
            dtypes.uint16,
            depth=1,
        )
        self.history_idx = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (2, self.topk_pad),
            dtypes.int32,
        )
        self.next_idx = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.topk_pad,),
            dtypes.int32,
        )


class QliRawTopKSelector:
    """ASC Arch35 uint16 radix TopK with in-AIV trunk/history merge.

    This is not LD/S2 core splitting.  One AIV processes a row sequentially in
    16K trunks.  The first trunk creates a TopK history; every later pass runs
    the same radix selector over ``history + current trunk`` and maps its local
    uint16 positions back to global int32 token indices.
    """

    def __init__(
        self,
        topk: int,
        workspace: QliRawTopKWorkspace,
    ):
        self.ub_pool = workspace.ub_pool
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("ASC raw TopK requires positive topk")
        # The ASC VF works in fixed 16K trunks.  The number of complete trunks
        # and the final tail length are runtime values so one compiled kernel
        # can serve every supported S2.
        self.trunk_len = workspace.max_trunk_len
        # ASC reserves history in 256-element units before the next trunk.
        self.topk_pad = ceil_div(self.topk, 256) * 256
        self.merge_len_pad = ceil_div(self.topk_pad + self.trunk_len, 128) * 128
        self.workspace = workspace
        self.pipelined = bool(getattr(workspace, "pipelined", False))
        if (
            self.trunk_len > workspace.max_trunk_len
            or self.topk > workspace.max_topk
            or self.merge_len_pad > workspace.merge_len_pad
        ):
            raise ValueError("TopK selector exceeds the shared UB workspace")
        # Final V->MTE3 stages are selector-owned so Sparse and Candidate can
        # share computational scratch without racing each other's GM writes.
        self.output_idx_stage = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (1, self.topk),
            dtypes.int32,
            depth=1,
        )
        self.output_value_bits_stage = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (1, self.topk),
            dtypes.uint16,
            depth=1,
        )
        self.hist_slot0 = workspace.histogram
        self.hist_slot1 = workspace.histogram

    @jit
    def _build_merge(
        self,
        current_trunk,
        history_key,
        merge_key,
        current_len,
        current_len_pad,
        history_prefix=None,
    ):
        """Assemble a trunk, optionally preceded by its retained TopK."""
        if history_prefix is None:
            history_prefix = self.topk_pad
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range((history_prefix + 127) // 128, unroll=1):
                offset = chunk * 128
                rr.vstore(
                    merge_key,
                    offset,
                    rr.vload(history_key, offset),
                    mask16,
                )
            current_chunk_count = (current_len + 127) // 128
            zero = rr.vdups(0, dtypes.uint16, mask=mask16)
            for chunk in dsl_range(current_chunk_count, unroll=1):
                source_offset = chunk * 128
                count = current_len - source_offset
                current_mask = rr.update_mask(count, elem_bits=16)[0]
                selected = rr.vselect(
                    rr.vload(current_trunk, source_offset), zero, cond_mask=current_mask
                )
                rr.vstore(merge_key, history_prefix + source_offset, selected, mask16)
            for chunk in dsl_range(
                current_chunk_count, (current_len_pad + 127) // 128, unroll=1
            ):
                rr.vstore(merge_key, history_prefix + chunk * 128, zero, mask16)
            rr.vmem_bar("vst_vld")

    @jit
    def _load_merge(
        self, gm_keys, history, merge, source_offset, history_prefix, length
    ):
        copied = ((length + 127) // 128) * 128
        if not self.pipelined:
            merge_slot = merge.produce()
            mem_copy(
                reinterpret(
                    _offset_view(merge_slot, (0, history_prefix)), shape=(1, copied)
                ),
                gm_keys[0:1, source_offset : source_offset + copied],
            )
            with vf(mode="simd"):
                mask = rr.update_mask(128, elem_bits=16)[0]
                for chunk in dsl_range(history_prefix // 128, unroll=1):
                    rr.vstore(
                        merge_slot, chunk * 128, rr.vload(history, chunk * 128), mask
                    )
                rr.vmem_bar("vst_vld")
        elif history_prefix == 0:
            # Common case: no retained history -> load the trunk straight into
            # the double-buffered merge Channel (async, no extra V copy).
            mem_copy(
                reinterpret(merge.produce(), shape=(1, copied)),
                gm_keys[0:1, source_offset : source_offset + copied],
            )
        else:
            mem_copy(
                reinterpret(self.workspace.current_trunk.produce(), shape=(1, copied)),
                gm_keys[0:1, source_offset : source_offset + copied],
            )
            self._build_merge(
                self.workspace.current_trunk.consume(),
                history,
                merge.produce(),
                length,
                copied,
                history_prefix,
            )

    @jit
    def _find_indices(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            h0c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            h0c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            h0c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            h0c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            full8 = rr.update_mask(256, elem_bits=8)[0]
            for block in dsl_range(valid_len // 1024, unroll=1):
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 0, width="b8"
                )
                h0c0 = rr.vhistogram_accumulate(h0c0, high, mask=full8, bin=0)
                h1c0 = rr.vhistogram_accumulate(h1c0, high, mask=full8, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 256, width="b8"
                )
                h0c1 = rr.vhistogram_accumulate(h0c1, high, mask=full8, bin=0)
                h1c1 = rr.vhistogram_accumulate(h1c1, high, mask=full8, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 512, width="b8"
                )
                h0c2 = rr.vhistogram_accumulate(h0c2, high, mask=full8, bin=0)
                h1c2 = rr.vhistogram_accumulate(h1c2, high, mask=full8, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 768, width="b8"
                )
                h0c3 = rr.vhistogram_accumulate(h0c3, high, mask=full8, bin=0)
                h1c3 = rr.vhistogram_accumulate(h1c3, high, mask=full8, bin=1)
            for chunk in dsl_range(
                (valid_len // 1024) * 4, (valid_len + 255) // 256, unroll=1
            ):
                mask8 = rr.update_mask(valid_len - chunk * 256, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                h0c0 = rr.vhistogram_accumulate(h0c0, high, mask=mask8, bin=0)
                h1c0 = rr.vhistogram_accumulate(h1c0, high, mask=mask8, bin=1)
            histogram_low_reg = rr.vadd(h0c0, h0c1, mask=full16)
            histogram_low_reg = rr.vadd(histogram_low_reg, h0c2, mask=full16)
            histogram_low_reg = rr.vadd(histogram_low_reg, h0c3, mask=full16)
            histogram_high_reg = rr.vadd(h1c0, h1c1, mask=full16)
            histogram_high_reg = rr.vadd(histogram_high_reg, h1c2, mask=full16)
            histogram_high_reg = rr.vadd(histogram_high_reg, h1c3, mask=full16)

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            bottom_k = valid_len - self.topk + 1
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = histogram_low_reg
            hist1 = histogram_high_reg
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist0, bottom_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist1, bottom_k, mask=full16),
            )
            high_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
            one16 = rr.vdups(1, dtypes.uint16, mask=full16)
            bottom_k = valid_len - self.topk + 1
            high_target = rr.vdup(high_target, mask=full16)
            high_target8 = rr.vreinterpret_lanes(
                rr.vbitwise_or(
                    high_target, rr.vshl(high_target, 8, mask=full16), mask=full16
                ),
                dtypes.uint8,
            )
            prev = rr.vsub(high_target, one16, mask=full16)
            is_zero = rr.veqs(high_target, 0, mask=full16)
            prev = rr.vselect(zero16, prev, cond_mask=is_zero)
            prev_lane = rr.vbitwise_and(
                prev, rr.vdups(127, dtypes.uint16, mask=full16), mask=full16
            )
            prev0 = rr.vgather_reg(histogram_low_reg, prev_lane)
            prev1 = rr.vgather_reg(histogram_high_reg, prev_lane)
            prev_count = rr.vselect(
                prev0, prev1, cond_mask=rr.vlts(prev, 128, mask=full16)
            )
            prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
            next_k = rr.vsub(
                rr.vdups(bottom_k, dtypes.uint16, mask=full16),
                prev_count,
                mask=full16,
            )
            next_k_register = rr.vdup(next_k, mask=full16)

            l0c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            full8 = rr.update_mask(256, elem_bits=8)[0]
            for block in dsl_range(valid_len // 1024, unroll=1):
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 0, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 256, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c1 = rr.vhistogram_accumulate(l0c1, low, mask=eq_high, bin=0)
                l1c1 = rr.vhistogram_accumulate(l1c1, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 512, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c2 = rr.vhistogram_accumulate(l0c2, low, mask=eq_high, bin=0)
                l1c2 = rr.vhistogram_accumulate(l1c2, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 768, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c3 = rr.vhistogram_accumulate(l0c3, low, mask=eq_high, bin=0)
                l1c3 = rr.vhistogram_accumulate(l1c3, low, mask=eq_high, bin=1)
            for chunk in dsl_range(
                (valid_len // 1024) * 4, (valid_len + 255) // 256, unroll=1
            ):
                mask8 = rr.update_mask(valid_len - chunk * 256, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
            l0 = rr.vadd(l0c0, l0c1, mask=full16)
            l0 = rr.vadd(l0, l0c2, mask=full16)
            l0 = rr.vadd(l0, l0c3, mask=full16)
            l1 = rr.vadd(l1c0, l1c1, mask=full16)
            l1 = rr.vadd(l1, l1c2, mask=full16)
            l1 = rr.vadd(l1, l1c3, mask=full16)

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            next_k = next_k_register
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = l0
            hist1 = l1
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist0, next_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist1, next_k, mask=full16),
            )
            low_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            low_target = rr.vdup(low_target, mask=full16)
            kth = rr.vbitwise_or(
                rr.vshl(high_target, 8, mask=full16),
                low_target,
                mask=full16,
            )
            ureg = rr.vstore_unalign_begin(tmp_idx)
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.vgt(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.veq(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            rr.vstore_unalign_post(tmp_idx, 0, ureg)
            rr.vmem_bar("vst_vld")

    @jit
    def _find_indices_cached(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            histogram_low_reg = rr.vload(histogram, 0)
            histogram_high_reg = rr.vload(histogram, 128)

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            bottom_k = valid_len - self.topk + 1
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = histogram_low_reg
            hist1 = histogram_high_reg
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist0, bottom_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist1, bottom_k, mask=full16),
            )
            high_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
            one16 = rr.vdups(1, dtypes.uint16, mask=full16)
            bottom_k = valid_len - self.topk + 1
            high_target = rr.vdup(high_target, mask=full16)
            high_target8 = rr.vreinterpret_lanes(
                rr.vbitwise_or(
                    high_target, rr.vshl(high_target, 8, mask=full16), mask=full16
                ),
                dtypes.uint8,
            )
            prev = rr.vsub(high_target, one16, mask=full16)
            is_zero = rr.veqs(high_target, 0, mask=full16)
            prev = rr.vselect(zero16, prev, cond_mask=is_zero)
            prev_lane = rr.vbitwise_and(
                prev, rr.vdups(127, dtypes.uint16, mask=full16), mask=full16
            )
            prev0 = rr.vgather_reg(histogram_low_reg, prev_lane)
            prev1 = rr.vgather_reg(histogram_high_reg, prev_lane)
            prev_count = rr.vselect(
                prev0, prev1, cond_mask=rr.vlts(prev, 128, mask=full16)
            )
            prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
            next_k = rr.vsub(
                rr.vdups(bottom_k, dtypes.uint16, mask=full16),
                prev_count,
                mask=full16,
            )
            next_k_register = rr.vdup(next_k, mask=full16)

            l0c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            full8 = rr.update_mask(256, elem_bits=8)[0]
            for block in dsl_range(valid_len // 1024, unroll=1):
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 0, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 256, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c1 = rr.vhistogram_accumulate(l0c1, low, mask=eq_high, bin=0)
                l1c1 = rr.vhistogram_accumulate(l1c1, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 512, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c2 = rr.vhistogram_accumulate(l0c2, low, mask=eq_high, bin=0)
                l1c2 = rr.vhistogram_accumulate(l1c2, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 768, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c3 = rr.vhistogram_accumulate(l0c3, low, mask=eq_high, bin=0)
                l1c3 = rr.vhistogram_accumulate(l1c3, low, mask=eq_high, bin=1)
            for chunk in dsl_range(
                (valid_len // 1024) * 4, (valid_len + 255) // 256, unroll=1
            ):
                mask8 = rr.update_mask(valid_len - chunk * 256, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
            l0 = rr.vadd(l0c0, l0c1, mask=full16)
            l0 = rr.vadd(l0, l0c2, mask=full16)
            l0 = rr.vadd(l0, l0c3, mask=full16)
            l1 = rr.vadd(l1c0, l1c1, mask=full16)
            l1 = rr.vadd(l1, l1c2, mask=full16)
            l1 = rr.vadd(l1, l1c3, mask=full16)

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            next_k = next_k_register
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = l0
            hist1 = l1
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist0, next_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist1, next_k, mask=full16),
            )
            low_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            low_target = rr.vdup(low_target, mask=full16)
            kth = rr.vbitwise_or(
                rr.vshl(high_target, 8, mask=full16),
                low_target,
                mask=full16,
            )
            ureg = rr.vstore_unalign_begin(tmp_idx)
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.vgt(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.veq(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            rr.vstore_unalign_post(tmp_idx, 0, ureg)
            rr.vmem_bar("vst_vld")

    @jit
    def _find_indices_dispatch(
        self,
        sort_key,
        valid_len,
        tmp_idx,
        histogram,
        high_target_carrier,
        cached_hist,
        hist_row,
    ):
        if cached_hist:
            self._find_indices_dispatch_cached(
                sort_key, valid_len, tmp_idx, histogram, high_target_carrier, hist_row
            )
        else:
            self._find_indices_dispatch_scanned(
                sort_key, valid_len, tmp_idx, histogram, high_target_carrier, hist_row
            )

    @jit
    def _find_indices_dispatch_cached(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier, hist_row
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            histogram_low_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            histogram_high_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            histogram_low_reg = rr.vadd(
                rr.vload(self.hist_slot0, hist_row * 512),
                rr.vload(self.hist_slot1, hist_row * 512),
                mask=full16,
            )
            histogram_high_reg = rr.vadd(
                rr.vload(self.hist_slot0, hist_row * 512 + 256),
                rr.vload(self.hist_slot1, hist_row * 512 + 256),
                mask=full16,
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            bottom_k = valid_len - self.topk + 1
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = histogram_low_reg
            hist1 = histogram_high_reg
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist0, bottom_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist1, bottom_k, mask=full16),
            )
            high_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
            one16 = rr.vdups(1, dtypes.uint16, mask=full16)
            bottom_k = valid_len - self.topk + 1
            high_target = rr.vdup(high_target, mask=full16)
            high_target8 = rr.vreinterpret_lanes(
                rr.vbitwise_or(
                    high_target, rr.vshl(high_target, 8, mask=full16), mask=full16
                ),
                dtypes.uint8,
            )
            prev = rr.vsub(high_target, one16, mask=full16)
            is_zero = rr.veqs(high_target, 0, mask=full16)
            prev = rr.vselect(zero16, prev, cond_mask=is_zero)
            prev_lane = rr.vbitwise_and(
                prev, rr.vdups(127, dtypes.uint16, mask=full16), mask=full16
            )
            prev0 = rr.vgather_reg(histogram_low_reg, prev_lane)
            prev1 = rr.vgather_reg(histogram_high_reg, prev_lane)
            prev_count = rr.vselect(
                prev0, prev1, cond_mask=rr.vlts(prev, 128, mask=full16)
            )
            prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
            next_k = rr.vsub(
                rr.vdups(bottom_k, dtypes.uint16, mask=full16),
                prev_count,
                mask=full16,
            )
            next_k_register = rr.vdup(next_k, mask=full16)

            l0c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            full8 = rr.update_mask(256, elem_bits=8)[0]
            for block in dsl_range(valid_len // 1024, unroll=1):
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 0, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 256, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c1 = rr.vhistogram_accumulate(l0c1, low, mask=eq_high, bin=0)
                l1c1 = rr.vhistogram_accumulate(l1c1, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 512, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c2 = rr.vhistogram_accumulate(l0c2, low, mask=eq_high, bin=0)
                l1c2 = rr.vhistogram_accumulate(l1c2, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 768, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c3 = rr.vhistogram_accumulate(l0c3, low, mask=eq_high, bin=0)
                l1c3 = rr.vhistogram_accumulate(l1c3, low, mask=eq_high, bin=1)
            for chunk in dsl_range(
                (valid_len // 1024) * 4, (valid_len + 255) // 256, unroll=1
            ):
                mask8 = rr.update_mask(valid_len - chunk * 256, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
            l0 = rr.vadd(l0c0, l0c1, mask=full16)
            l0 = rr.vadd(l0, l0c2, mask=full16)
            l0 = rr.vadd(l0, l0c3, mask=full16)
            l1 = rr.vadd(l1c0, l1c1, mask=full16)
            l1 = rr.vadd(l1, l1c2, mask=full16)
            l1 = rr.vadd(l1, l1c3, mask=full16)

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            next_k = next_k_register
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = l0
            hist1 = l1
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist0, next_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist1, next_k, mask=full16),
            )
            low_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            low_target = rr.vdup(low_target, mask=full16)
            kth = rr.vbitwise_or(
                rr.vshl(high_target, 8, mask=full16),
                low_target,
                mask=full16,
            )
            ureg = rr.vstore_unalign_begin(tmp_idx)
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.vgt(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.veq(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            rr.vstore_unalign_post(tmp_idx, 0, ureg)
            rr.vmem_bar("vst_vld")

    @jit
    def _find_indices_dispatch_scanned(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier, hist_row
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            histogram_low_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            histogram_high_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            h0c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            h0c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            h0c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            h0c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            full8 = rr.update_mask(256, elem_bits=8)[0]
            for block in dsl_range(valid_len // 1024, unroll=1):
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 0, width="b8"
                )
                h0c0 = rr.vhistogram_accumulate(h0c0, high, mask=full8, bin=0)
                h1c0 = rr.vhistogram_accumulate(h1c0, high, mask=full8, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 256, width="b8"
                )
                h0c1 = rr.vhistogram_accumulate(h0c1, high, mask=full8, bin=0)
                h1c1 = rr.vhistogram_accumulate(h1c1, high, mask=full8, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 512, width="b8"
                )
                h0c2 = rr.vhistogram_accumulate(h0c2, high, mask=full8, bin=0)
                h1c2 = rr.vhistogram_accumulate(h1c2, high, mask=full8, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 768, width="b8"
                )
                h0c3 = rr.vhistogram_accumulate(h0c3, high, mask=full8, bin=0)
                h1c3 = rr.vhistogram_accumulate(h1c3, high, mask=full8, bin=1)
            for chunk in dsl_range(
                (valid_len // 1024) * 4, (valid_len + 255) // 256, unroll=1
            ):
                mask8 = rr.update_mask(valid_len - chunk * 256, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                h0c0 = rr.vhistogram_accumulate(h0c0, high, mask=mask8, bin=0)
                h1c0 = rr.vhistogram_accumulate(h1c0, high, mask=mask8, bin=1)
            histogram_low_reg = rr.vadd(h0c0, h0c1, mask=full16)
            histogram_low_reg = rr.vadd(histogram_low_reg, h0c2, mask=full16)
            histogram_low_reg = rr.vadd(histogram_low_reg, h0c3, mask=full16)
            histogram_high_reg = rr.vadd(h1c0, h1c1, mask=full16)
            histogram_high_reg = rr.vadd(histogram_high_reg, h1c2, mask=full16)
            histogram_high_reg = rr.vadd(histogram_high_reg, h1c3, mask=full16)

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            bottom_k = valid_len - self.topk + 1
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = histogram_low_reg
            hist1 = histogram_high_reg
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist0, bottom_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist1, bottom_k, mask=full16),
            )
            high_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
            one16 = rr.vdups(1, dtypes.uint16, mask=full16)
            bottom_k = valid_len - self.topk + 1
            high_target = rr.vdup(high_target, mask=full16)
            high_target8 = rr.vreinterpret_lanes(
                rr.vbitwise_or(
                    high_target, rr.vshl(high_target, 8, mask=full16), mask=full16
                ),
                dtypes.uint8,
            )
            prev = rr.vsub(high_target, one16, mask=full16)
            is_zero = rr.veqs(high_target, 0, mask=full16)
            prev = rr.vselect(zero16, prev, cond_mask=is_zero)
            prev_lane = rr.vbitwise_and(
                prev, rr.vdups(127, dtypes.uint16, mask=full16), mask=full16
            )
            prev0 = rr.vgather_reg(histogram_low_reg, prev_lane)
            prev1 = rr.vgather_reg(histogram_high_reg, prev_lane)
            prev_count = rr.vselect(
                prev0, prev1, cond_mask=rr.vlts(prev, 128, mask=full16)
            )
            prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
            next_k = rr.vsub(
                rr.vdups(bottom_k, dtypes.uint16, mask=full16),
                prev_count,
                mask=full16,
            )
            next_k_register = rr.vdup(next_k, mask=full16)

            l0c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            full8 = rr.update_mask(256, elem_bits=8)[0]
            for block in dsl_range(valid_len // 1024, unroll=1):
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 0, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 256, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c1 = rr.vhistogram_accumulate(l0c1, low, mask=eq_high, bin=0)
                l1c1 = rr.vhistogram_accumulate(l1c1, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 512, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c2 = rr.vhistogram_accumulate(l0c2, low, mask=eq_high, bin=0)
                l1c2 = rr.vhistogram_accumulate(l1c2, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 768, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c3 = rr.vhistogram_accumulate(l0c3, low, mask=eq_high, bin=0)
                l1c3 = rr.vhistogram_accumulate(l1c3, low, mask=eq_high, bin=1)
            for chunk in dsl_range(
                (valid_len // 1024) * 4, (valid_len + 255) // 256, unroll=1
            ):
                mask8 = rr.update_mask(valid_len - chunk * 256, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
            l0 = rr.vadd(l0c0, l0c1, mask=full16)
            l0 = rr.vadd(l0, l0c2, mask=full16)
            l0 = rr.vadd(l0, l0c3, mask=full16)
            l1 = rr.vadd(l1c0, l1c1, mask=full16)
            l1 = rr.vadd(l1, l1c2, mask=full16)
            l1 = rr.vadd(l1, l1c3, mask=full16)

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            next_k = next_k_register
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = l0
            hist1 = l1
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist0, next_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist1, next_k, mask=full16),
            )
            low_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            low_target = rr.vdup(low_target, mask=full16)
            kth = rr.vbitwise_or(
                rr.vshl(high_target, 8, mask=full16),
                low_target,
                mask=full16,
            )
            ureg = rr.vstore_unalign_begin(tmp_idx)
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.vgt(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.veq(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            rr.vstore_unalign_post(tmp_idx, 0, ureg)
            rr.vmem_bar("vst_vld")

    @jit
    def _select_pass(
        self,
        sort_key,
        valid_len,
        history_prefix,
        current_global_base,
        tmp_idx,
        histogram,
        high_target_carrier,
        history_idx_in,
        history_idx_out,
        next_history_key,
    ):
        """Run one ASC radix pass over one trunk or history+trunk."""
        self._find_indices(sort_key, valid_len, tmp_idx, histogram, high_target_carrier)

        with vf(mode="simd"):
            # ASC FindValueOutputVFImpl keeps the compacted positions as
            # uint16 and gathers 128 values per iteration.  Do not reuse the
            # uint16->uint32 unpack used by FindRealIndexVFImpl below: that
            # unpack exposes only 64 lanes and breaks value/index pairing
            # once history is merged with another trunk.
            for chunk in dsl_range(ceil_div(self.topk, 128), unroll=1):
                count = self.topk - chunk * 128
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                pos16 = rr.vload(tmp_idx, chunk * 128)
                selected_key = rr.vgather(sort_key, pos16, mask=mask16)
                rr.vstore(
                    next_history_key,
                    chunk * 128,
                    selected_key,
                    mask16,
                )

            # ASC FindRealIndexVFImpl loads the same compacted positions with
            # DIST_UNPACK_B16, yielding 64 uint32 lanes for history/current
            # index mapping.  Keep this separate from the value gather above.
            _full32 = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                pos16 = rr.vload(tmp_idx, chunk * 64)
                pos32 = rr.vunpack(pos16, dtypes.uint32, part="lower")
                is_current = rr.vges(pos32, history_prefix, mask=mask32)
                is_history = rr.vlts(pos32, history_prefix, mask=mask32)
                old_idx = rr.vgather(history_idx_in, pos32, mask=is_history)
                current_idx = rr.vadds(
                    rr.vreinterpret(pos32, dtypes.int32),
                    current_global_base - history_prefix,
                    mask=mask32,
                )
                global_idx = rr.vselect(current_idx, old_idx, cond_mask=is_current)
                rr.vstore(history_idx_out, chunk * 64, global_idx, mask32)
            rr.vmem_bar("vst_vld")

    @jit
    def _commit_history(self, next_idx, history_idx, next_history_key, history_key):
        with vf(mode="simd"):
            copy_mask32 = rr.update_mask(64, elem_bits=32)[0]
            copy_mask16 = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                rr.vstore(
                    history_idx,
                    chunk * 64,
                    rr.vload(next_idx, chunk * 64),
                    copy_mask32,
                )
            for chunk in dsl_range(ceil_div(self.topk, 128), unroll=1):
                rr.vstore(
                    history_key,
                    chunk * 128,
                    rr.vload(next_history_key, chunk * 128),
                    copy_mask16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def store_empty_row(
        self,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
    ):
        """Store the public empty-TopK representation without scanning GM."""
        idx_write = self.output_idx_stage.produce()
        val_write = self.output_value_bits_stage.produce()
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                rr.vstore(
                    idx_write,
                    chunk * 64,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    mask32,
                )
                rr.vstore(
                    val_write,
                    chunk * 64,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    mask16,
                )
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.output_idx_stage.consume())

        mem_copy(gm_selected_value_bits, self.output_value_bits_stage.consume())

    @jit
    def select_row(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
        need_values=1,
        cached_hist=0,
        hist_row=0,
    ):
        if (
            need_values == 0
            and total_tokens >= self.topk
            and total_tokens <= self.trunk_len
        ):
            merge_key = self.workspace.merge_key
            self._load_merge(
                gm_sort_key,
                self.workspace.history_key,
                merge_key,
                dtypes.int64(0),
                dtypes.int64(0),
                total_tokens,
            )
            merge_read = merge_key.consume()
            self._find_indices_dispatch(
                merge_read,
                total_tokens,
                self.workspace.tmp_idx,
                self.workspace.histogram,
                self.workspace.high_target_carrier,
                cached_hist,
                hist_row,
            )
            out = self.output_idx_stage.produce()
            tmp = self.workspace.tmp_idx
            with vf(mode="simd"):
                for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                    mask = rr.update_mask(self.topk - chunk * 64, elem_bits=32)[0]
                    positions = rr.vunpack(
                        rr.vload(tmp, chunk * 64), dtypes.uint32, part="lower"
                    )
                    indices = rr.vadds(
                        rr.vreinterpret(positions, dtypes.int32), index_base, mask=mask
                    )
                    valid = rr.vlts(
                        rr.varange(chunk * 64, dtypes.uint32),
                        output_valid_count,
                        mask=mask,
                    )
                    rr.vstore(
                        out,
                        chunk * 64,
                        rr.vselect(
                            indices,
                            rr.vdups(-1, dtypes.int32, mask=mask),
                            cond_mask=valid,
                        ),
                        mask,
                    )
                rr.vmem_bar("vst_vld")
            mem_copy(gm_sparse_indices, self.output_idx_stage.consume())
        elif (
            need_values != 0
            and total_tokens >= self.topk
            and total_tokens <= self.trunk_len
        ):
            self._select_row_single_trunk(
                gm_sort_key,
                total_tokens,
                output_valid_count,
                gm_sparse_indices,
                gm_selected_value_bits,
                index_base,
                cached_hist,
                hist_row,
            )
        else:
            self._select_row_values(
                gm_sort_key,
                total_tokens,
                output_valid_count,
                gm_sparse_indices,
                gm_selected_value_bits,
                index_base,
            )

    @jit
    def _select_row_single_trunk(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
        cached_hist=0,
        hist_row=0,
    ):
        """PV-C8 port: single-trunk fast path WITH values.

        When the entire input fits in one trunk (total_tokens <= 16384) and
        this is the first (and only) pass, history stays empty: skip its
        zero-init, skip the next_history_key/next_idx round-trip, and gather
        value bits directly from merge_key at the squeeze positions.
        Mathematically identical to _select_row_values for this case because
        history_prefix == 0 makes the position mapping an identity.
        """
        merge_key = self.workspace.merge_key
        tmp_idx = self.workspace.tmp_idx
        histogram = self.workspace.histogram
        high_target_carrier = self.workspace.high_target_carrier
        self._load_merge(
            gm_sort_key,
            self.workspace.history_key,
            merge_key,
            dtypes.int64(0),
            dtypes.int64(0),
            total_tokens,
        )
        merge_read = merge_key.consume()
        self._find_indices_dispatch(
            merge_read,
            total_tokens,
            tmp_idx,
            histogram,
            high_target_carrier,
            cached_hist,
            hist_row,
        )
        out_idx = self.output_idx_stage.produce()
        out_val = self.output_value_bits_stage.produce()
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                pos16 = rr.vload(tmp_idx, chunk * 64)
                pos32 = rr.vunpack(pos16, dtypes.uint32, part="lower")
                keys = rr.vgather(merge_read, pos16, mask=mask16)
                positive = rr.veq(
                    rr.vbitwise_and(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                value_bits = rr.vselect(
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0xFFFF, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    cond_mask=positive,
                )
                valid32 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                valid16 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                indices = rr.vselect(
                    rr.vadds(
                        rr.vreinterpret(pos32, dtypes.int32),
                        index_base,
                        mask=mask32,
                    ),
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=valid32,
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    cond_mask=valid16,
                )
                rr.vstore(out_idx, chunk * 64, indices, mask32)
                rr.vstore(out_val, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.output_idx_stage.consume())
        mem_copy(gm_selected_value_bits, self.output_value_bits_stage.consume())

    @jit
    def _select_row_cached_ub(
        self,
        sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
        cached_hist=0,
        hist_row=0,
    ):
        """PV-C8 port: single-trunk fast path WITH values.

        When the entire input fits in one trunk (total_tokens <= 16384) and
        this is the first (and only) pass, history stays empty: skip its
        zero-init, skip the next_history_key/next_idx round-trip, and gather
        value bits directly from merge_key at the squeeze positions.
        Mathematically identical to _select_row_values for this case because
        history_prefix == 0 makes the position mapping an identity.
        """
        merge_key = sort_key
        tmp_idx = self.cache_tmp
        histogram = self.workspace.histogram
        high_target_carrier = self.workspace.high_target_carrier
        self._find_indices_dispatch(
            merge_key,
            total_tokens,
            tmp_idx,
            histogram,
            high_target_carrier,
            cached_hist,
            hist_row,
        )
        out_idx = self.cache_output_idx_stage.produce()
        out_val = self.cache_output_value_bits_stage.produce()
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                pos16 = rr.vload(tmp_idx, chunk * 64)
                pos32 = rr.vunpack(pos16, dtypes.uint32, part="lower")
                keys = rr.vgather(merge_key, pos16, mask=mask16)
                positive = rr.veq(
                    rr.vbitwise_and(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                value_bits = rr.vselect(
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0xFFFF, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    cond_mask=positive,
                )
                valid32 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                valid16 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                indices = rr.vselect(
                    rr.vadds(
                        rr.vreinterpret(pos32, dtypes.int32),
                        index_base,
                        mask=mask32,
                    ),
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=valid32,
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    cond_mask=valid16,
                )
                rr.vstore(out_idx, chunk * 64, indices, mask32)
                rr.vstore(out_val, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.cache_output_idx_stage.consume())
        mem_copy(gm_selected_value_bits, self.cache_output_value_bits_stage.consume())

    @jit
    def _select_row_values(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
    ):
        # Rank masks use uint16 lanes; clamp the count before narrowing.
        if output_valid_count > self.topk:
            output_valid_count = self.topk
        merge_key = self.workspace.merge_key
        _current_trunk = self.workspace.current_trunk
        tmp_idx = self.workspace.tmp_idx
        histogram = self.workspace.histogram
        high_target_carrier = self.workspace.high_target_carrier
        history_key_storage = self.workspace.history_key
        history_key = history_key_storage
        next_history_key = self.workspace.next_history_key
        history_idx_storage = self.workspace.history_idx
        history_idx = history_idx_storage
        next_idx = self.workspace.next_idx
        output_idx_stage = self.output_idx_stage.produce()
        out_value_bits = self.output_value_bits_stage.produce()

        # Finite signed BF16 scores have positive sortable keys; zero marks
        # invalid tokens and empty history slots.
        with vf(mode="simd"):
            init_mask16 = rr.update_mask(128, elem_bits=16)[0]
            init_mask32 = rr.update_mask(64, elem_bits=32)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=init_mask16)
            zero32 = rr.vdups(0, dtypes.int32, mask=init_mask32)
            for chunk in dsl_range(ceil_div(self.topk_pad * 2, 128), unroll=1):
                rr.vstore(history_key, chunk * 128, zero16, init_mask16)
            for chunk in dsl_range(ceil_div(self.topk_pad * 2, 64), unroll=1):
                rr.vstore(history_idx, chunk * 64, zero32, init_mask32)
            rr.vmem_bar("vst_vld")

        full_trunk_count = total_tokens // self.trunk_len
        tail_len = total_tokens % self.trunk_len
        for pass_idx in dsl_range(full_trunk_count, unroll=1):
            parity = pass_idx % 2
            history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (parity, 0)
            )
            next_history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            history_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (parity, 0)
            )
            next_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            trunk_offset = pass_idx * self.trunk_len
            history_prefix = dtypes.int64((0 if pass_idx == 0 else self.topk_pad))
            self._load_merge(
                gm_sort_key,
                history_key,
                merge_key,
                trunk_offset,
                history_prefix,
                self.trunk_len,
            )
            self._select_pass(
                merge_key.consume(),
                history_prefix + self.trunk_len,
                history_prefix,
                trunk_offset,
                tmp_idx,
                histogram,
                high_target_carrier,
                history_idx,
                next_idx,
                next_history_key,
            )

        if tail_len > 0:
            parity = full_trunk_count % 2
            history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (parity, 0)
            )
            next_history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            history_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (parity, 0)
            )
            next_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            tail_offset = full_trunk_count * self.trunk_len
            history_prefix = dtypes.int64(self.topk_pad)
            if full_trunk_count == 0 and tail_len >= self.topk:
                history_prefix = dtypes.int64(0)
            _tail_len_pad = ((tail_len + 255) // 256) * 256
            self._load_merge(
                gm_sort_key,
                history_key,
                merge_key,
                tail_offset,
                history_prefix,
                tail_len,
            )
            self._select_pass(
                merge_key.consume(),
                history_prefix + tail_len,
                history_prefix,
                tail_offset,
                tmp_idx,
                histogram,
                high_target_carrier,
                history_idx,
                next_idx,
                next_history_key,
            )

        final_slot = (full_trunk_count + dtypes.int64((1 if tail_len > 0 else 0))) % 2
        final_idx = tile_slice(history_idx_storage, (1, self.topk_pad), (final_slot, 0))
        history_key = tile_slice(
            history_key_storage, (1, self.topk_pad), (final_slot, 0)
        )
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                indices = rr.vadds(
                    rr.vload(final_idx, chunk * 64), index_base, mask=mask32
                )
                keys = rr.vload(history_key, chunk * 64)
                positive = rr.veq(
                    rr.vbitwise_and(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                value_bits = rr.vselect(
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0xFFFF, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    cond_mask=positive,
                )
                valid_rank32 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                valid_rank16 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                indices = rr.vselect(
                    indices,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=valid_rank32,
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    cond_mask=valid_rank16,
                )
                rr.vstore(output_idx_stage, chunk * 64, indices, mask32)
                rr.vstore(out_value_bits, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.output_idx_stage.consume())
        mem_copy(gm_selected_value_bits, self.output_value_bits_stage.consume())


class LdTopKSelector(QliRawTopKSelector):
    def __init__(self, topk, workspace, splits):
        self.ub_pool = workspace.ub_pool
        super().__init__(topk, workspace)
        self.ld_length = int(splits) * self.topk
        self.ld_indices = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (1, self.ld_length),
            dtypes.int32,
            depth=1,
        )
        self.ld_bits = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (1, self.ld_length),
            dtypes.uint16,
            depth=1,
        )

    @jit
    def merge_ld(
        self,
        partial_indices,
        partial_bits,
        gm_sparse_indices,
        gm_selected_value_bits,
        valid_length,
        output_offset=0,
        preloaded=False,
        need_values=1,
        emit_gm=1,
    ):
        if not preloaded:
            mem_copy(
                reinterpret(self.ld_indices.produce(), shape=(1, valid_length)),
                partial_indices[0:1, 0:valid_length],
            )
            mem_copy(
                reinterpret(self.ld_bits.produce(), shape=(1, valid_length)),
                partial_bits[0:1, 0:valid_length],
            )
        current_trunk = self.workspace.merge_key
        trunk_write = current_trunk.produce()
        _history_key = self.workspace.history_key
        history_idx = self.workspace.history_idx
        ld_idx_read = self.ld_indices.consume()
        ld_bits_read = self.ld_bits.consume()
        with vf(mode="simd"):
            mask16 = rr.update_mask(64, elem_bits=16)[0]
            mask32 = rr.update_mask(64, elem_bits=32)[0]
            # LD may merge fewer than 64 entries for a small TopK capacity.
            for chunk in dsl_range((valid_length + 63) // 64, unroll=1):
                count = min(64, valid_length - chunk * 64)
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                bits = rr.vload(ld_bits_read, chunk * 64)
                indices = rr.vload(ld_idx_read, chunk * 64)
                positive_key = rr.vbitwise_xor(
                    bits, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
                )
                negative_key = rr.vbitwise_xor(
                    bits, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
                )
                negative = rr.veqs(
                    rr.vbitwise_and(
                        bits, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
                    ),
                    0x8000,
                    mask=mask16,
                )
                keys = rr.vselect(negative_key, positive_key, cond_mask=negative)
                valid32 = rr.vselect(
                    rr.vdups(1, dtypes.int32, mask=mask32),
                    rr.vdups(0, dtypes.int32, mask=mask32),
                    cond_mask=rr.vges(indices, 0, mask=mask32),
                )
                valid16 = rr.vpack(
                    rr.vreinterpret(valid32, dtypes.uint32), dtypes.uint16, part="lower"
                )
                keys = rr.vselect(
                    keys,
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    cond_mask=rr.veqs(valid16, 1, mask=mask16),
                )
                rr.vstore(trunk_write, chunk * 64, keys, mask16)
            rr.vmem_bar("vst_vld")
        trunk_read = current_trunk.consume()
        self._find_indices(
            trunk_read,
            valid_length,
            self.workspace.tmp_idx,
            self.workspace.histogram,
            self.workspace.high_target_carrier,
        )
        _output_idx_stage = self.output_idx_stage.produce()
        _out_value_bits = self.output_value_bits_stage.produce()
        _output_valid_count = dtypes.int32(self.topk)
        _final_idx = history_idx
        if need_values != 0:
            self._emit_ld_values(
                trunk_read,
                ld_idx_read,
                ld_bits_read,
                output_offset,
                gm_sparse_indices,
                gm_selected_value_bits,
                emit_gm,
            )
        else:
            self._emit_ld_indices(
                trunk_read, ld_idx_read, output_offset, gm_sparse_indices, emit_gm
            )

    @jit
    def _emit_ld_values(
        self,
        current_trunk,
        ld_indices,
        ld_bits,
        output_offset,
        gm_sparse_indices,
        gm_selected_value_bits,
        emit_gm=1,
    ):
        output_idx_stage = self.output_idx_stage.produce()
        out_value_bits = self.output_value_bits_stage.produce()
        output_valid_count = dtypes.int32(self.topk)
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                positions = rr.vunpack(
                    rr.vload(self.workspace.tmp_idx, chunk * 64),
                    dtypes.uint32,
                    part="lower",
                )
                indices = rr.vgather(
                    ld_indices, rr.vreinterpret(positions, dtypes.uint32), mask=mask32
                )
                positions16 = rr.vload(self.workspace.tmp_idx, chunk * 64)
                keys = rr.vgather(current_trunk, positions16, mask=mask16)
                value_bits = rr.vgather(ld_bits, positions16, mask=mask16)
                _valid_rank32 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                _valid_rank16 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                key_valid16 = rr.vselect(
                    rr.vdups(1, dtypes.uint16, mask=mask16),
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    cond_mask=rr.vges(keys, 1, mask=mask16),
                )
                key_valid32 = rr.vunpack(key_valid16, dtypes.uint32, part="lower")
                indices = rr.vselect(
                    indices,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.veqs(key_valid32, 1, mask=mask32),
                )
                valid_flags32 = rr.vselect(
                    rr.vdups(1, dtypes.int32, mask=mask32),
                    rr.vdups(0, dtypes.int32, mask=mask32),
                    cond_mask=rr.vges(indices, 0, mask=mask32),
                )
                valid_flags16 = rr.vpack(
                    rr.vreinterpret(valid_flags32, dtypes.uint32),
                    dtypes.uint16,
                    part="lower",
                )
                indices = rr.vselect(
                    indices,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.vges(indices, 0, mask=mask32),
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    cond_mask=rr.veqs(valid_flags16, 1, mask=mask16),
                )
                indices = rr.vselect(
                    rr.vadds(indices, output_offset, mask=mask32),
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.veqs(valid_flags32, 1, mask=mask32),
                )
                rr.vstore(output_idx_stage, chunk * 64, indices, mask32)
                rr.vstore(out_value_bits, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        # Intermediate LD rounds keep the carry in the on-core output stages;
        # only the final round drains them to GM.
        if emit_gm != 0:
            mem_copy(gm_sparse_indices, self.output_idx_stage.consume())
            mem_copy(gm_selected_value_bits, self.output_value_bits_stage.consume())

    @jit
    def _emit_ld_indices(
        self, current_trunk, ld_indices, output_offset, gm_sparse_indices, emit_gm=1
    ):
        output_idx_stage = self.output_idx_stage.produce()
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                positions16 = rr.vload(self.workspace.tmp_idx, chunk * 64)
                positions32 = rr.vunpack(positions16, dtypes.uint32, part="lower")
                indices = rr.vgather(ld_indices, positions32, mask=mask32)
                keys = rr.vgather(current_trunk, positions16, mask=mask16)
                valid16 = rr.vselect(
                    rr.vdups(1, dtypes.uint16, mask=mask16),
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    cond_mask=rr.vges(keys, 1, mask=mask16),
                )
                valid32 = rr.vunpack(valid16, dtypes.uint32, part="lower")
                indices = rr.vselect(
                    rr.vadds(indices, output_offset, mask=mask32),
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.veqs(valid32, 1, mask=mask32),
                )
                rr.vstore(output_idx_stage, chunk * 64, indices, mask32)
            rr.vmem_bar("vst_vld")
        if emit_gm != 0:
            mem_copy(gm_sparse_indices, self.output_idx_stage.consume())


class LdMergeStage:
    def __init__(self, topk, splits):
        self.topk = topk
        self.length = topk * splits
        self.workspace = QliRawTopKWorkspace(TOPK_TRUNK_LEN, topk)
        self.selector = LdTopKSelector(topk, self.workspace, splits)

    @jit
    def __call__(
        self,
        partial_indices: Tensor,
        partial_bits: Tensor,
        indices: Tensor,
        bits: Tensor,
        metadata: Tensor,
        rows,
        workers,
        cu_seqlens_q: Tensor,
        query_tile_rows,
        has_cu,
        output_offset_address=0,
        need_values=1,
    ):
        fd = dtypes.int64(LD_RECORD_START + get_block_idx() * 2 + get_subblock_id())
        if metadata[fd, 0] != 0:
            batch_idx = dtypes.int64(metadata[fd, 1])
            begin = dtypes.int64(0)
            if has_cu:
                begin = dtypes.int64(cu_seqlens_q[batch_idx])
            row_base = begin + dtypes.int64(metadata[fd, 2]) * query_tile_rows
            workspace_begin = dtypes.int64(metadata[fd, 3])
            part_count = dtypes.int64(metadata[fd, 4])
            m_begin = dtypes.int64(metadata[fd, 5])
            m_end = m_begin + dtypes.int64(metadata[fd, 6])
            for m in dsl_range(m_begin, m_end, unroll=1):
                row = row_base + m
                output_offset = dtypes.int32(0)
                if output_offset_address != 0:
                    offsets = make_tensor(
                        make_pointer(
                            dtypes.int32, dtypes.int64(output_offset_address), MemLoc.GM
                        ),
                        make_layout((rows, 1), stride=(1, 1)),
                    )
                    output_offset = offsets[row, 0]
                for first in dsl_range(
                    0, part_count, self.length // self.topk - 1, unroll=1
                ):
                    count = min(part_count - first, self.length // self.topk - 1)
                    carry = dtypes.int64((1 if first > 0 else 0))
                    ld_idx_write = self.selector.ld_indices.produce()
                    ld_bits_write = self.selector.ld_bits.produce()
                    if first > 0:
                        # Carry stays on-core: the previous round's output-stage
                        # slots are the sole producer of this round's carry.
                        mem_copy(
                            reinterpret(ld_idx_write, shape=(1, self.topk)),
                            self.selector.output_idx_stage.consume(),
                        )
                        mem_copy(
                            reinterpret(ld_bits_write, shape=(1, self.topk)),
                            self.selector.output_value_bits_stage.consume(),
                        )
                    for part in dsl_range(0, count, unroll=1):
                        source_row = (
                            workspace_begin + first + part
                        ) * query_tile_rows + m
                        mem_copy(
                            tile_slice(ld_idx_write, (1, self.topk), (0, part + carry)),
                            tile_slice(
                                partial_indices, (1, self.topk), (source_row, 0)
                            ),
                        )
                        mem_copy(
                            tile_slice(
                                ld_bits_write, (1, self.topk), (0, part + carry)
                            ),
                            tile_slice(partial_bits, (1, self.topk), (source_row, 0)),
                        )
                    offset = dtypes.int32(
                        (output_offset if first + count == part_count else 0)
                    )
                    # Intermediate rounds emit into the UB output stages only;
                    # the final round drains to GM, so no GM write-read ordering
                    # is needed between rounds.
                    self.selector.merge_ld(
                        partial_indices,
                        partial_bits,
                        tile_slice(indices, (1, self.topk), (row, 0)),
                        tile_slice(bits, (1, self.topk), (row, 0)),
                        (count + carry) * self.topk,
                        offset,
                        True,
                        dtypes.int64(
                            (need_values if first + count == part_count else 1)
                        ),
                        dtypes.int64((1 if first + count == part_count else 0)),
                    )


# Candidate LD selectors keep their independent 16K merge scratch layout.
CANDIDATE_LD_TRUNK_LEN = 16384


class CandidateQliRawTopKWorkspace:
    """UB scratch shared by sequential Sparse and Candidate TopK stages."""

    def __init__(
        self, max_trunk_len: int, max_topk: int, candidate_optimized: bool = False
    ):
        self.ub_pool = _UbPool()
        self.candidate_optimized = bool(candidate_optimized)
        self.max_trunk_len = int(max_trunk_len)
        self.max_topk = int(max_topk)
        self.topk_pad = ceil_div(self.max_topk, 256) * 256
        self.merge_len_pad = ceil_div(self.topk_pad + self.max_trunk_len, 128) * 128
        self.current_trunk = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (1, self.max_trunk_len),
            dtypes.uint16,
            depth=1,
        )
        self.merge_key = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (1, self.merge_len_pad),
            dtypes.uint16,
            depth=1,
        )
        self.tmp_idx = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.merge_len_pad,),
            dtypes.uint16,
        )
        self.histogram = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (256,),
            dtypes.uint16,
            addr=TOPK_HISTOGRAM_UB_OFFSET,
        )
        self.high_target_carrier = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (64,),
            dtypes.int32,
            addr=TOPK_TARGET_UB_OFFSET,
        )
        self.history_key = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (2, self.topk_pad),
            dtypes.uint16,
        )
        self.next_history_key = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (1, self.topk_pad),
            dtypes.uint16,
            depth=1,
        )
        self.history_idx = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (2, self.topk_pad),
            dtypes.int32,
        )
        self.next_idx = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.topk_pad,),
            dtypes.int32,
        )


class CandidateQliRawTopKSelector:
    """ASC Arch35 uint16 radix TopK with in-AIV trunk/history merge.

    This is not LD/S2 core splitting.  One AIV processes a row sequentially in
    16K trunks.  The first trunk creates a TopK history; every later pass runs
    the same radix selector over ``history + current trunk`` and maps its local
    uint16 positions back to global int32 token indices.
    """

    def __init__(
        self,
        topk: int,
        workspace: CandidateQliRawTopKWorkspace,
    ):
        self.ub_pool = workspace.ub_pool
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("ASC raw TopK requires positive topk")
        # The ASC VF works in fixed 16K trunks.  The number of complete trunks
        # and the final tail length are runtime values so one compiled kernel
        # can serve every supported S2.
        self.trunk_len = CANDIDATE_LD_TRUNK_LEN
        # ASC reserves history in 256-element units before the next trunk.
        self.topk_pad = ceil_div(self.topk, 256) * 256
        self.merge_len_pad = ceil_div(self.topk_pad + CANDIDATE_LD_TRUNK_LEN, 128) * 128
        self.workspace = workspace
        if (
            CANDIDATE_LD_TRUNK_LEN > workspace.max_trunk_len
            or self.topk > workspace.max_topk
            or self.merge_len_pad > workspace.merge_len_pad
        ):
            raise ValueError("TopK selector exceeds the shared UB workspace")
        # Final V->MTE3 stages are selector-owned so Sparse and Candidate can
        # share computational scratch without racing each other's GM writes.
        self.output_idx_stage = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (1, self.topk),
            dtypes.int32,
            depth=1,
        )
        self.output_value_bits_stage = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (1, self.topk),
            dtypes.uint16,
            depth=1,
        )

    @jit
    def _build_merge(
        self,
        current_trunk,
        history_key,
        merge_key,
        current_len,
        current_len_pad,
        history_prefix=None,
    ):
        """Assemble a trunk, optionally preceded by its retained TopK."""
        if history_prefix is None:
            history_prefix = self.topk_pad
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range((history_prefix + 127) // 128, unroll=1):
                offset = chunk * 128
                rr.vstore(
                    merge_key,
                    offset,
                    rr.vload(history_key, offset),
                    mask16,
                )
            current_chunk_count = (current_len + 127) // 128
            zero = rr.vdups(0, dtypes.uint16, mask=mask16)
            for chunk in dsl_range(current_chunk_count, unroll=1):
                source_offset = chunk * 128
                count = current_len - source_offset
                current_mask = rr.update_mask(count, elem_bits=16)[0]
                selected = rr.vselect(
                    rr.vload(current_trunk, source_offset), zero, cond_mask=current_mask
                )
                rr.vstore(merge_key, history_prefix + source_offset, selected, mask16)
            for chunk in dsl_range(
                current_chunk_count, (current_len_pad + 127) // 128, unroll=1
            ):
                rr.vstore(merge_key, history_prefix + chunk * 128, zero, mask16)
            rr.vmem_bar("vst_vld")

    @jit
    def _load_merge(
        self, gm_keys, history, merge, source_offset, history_prefix, length
    ):
        copied = ((length + 127) // 128) * 128
        merge_slot = merge.produce()
        mem_copy(
            reinterpret(
                _offset_view(merge_slot, (0, history_prefix)), shape=(1, copied)
            ),
            gm_keys[0:1, source_offset : source_offset + copied],
        )
        with vf(mode="simd"):
            mask = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range(history_prefix // 128, unroll=1):
                rr.vstore(merge_slot, chunk * 128, rr.vload(history, chunk * 128), mask)
            rr.vmem_bar("vst_vld")

    @jit
    def _find_indices(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        if self.workspace.candidate_optimized:
            self._find_indices_optimized(
                sort_key, valid_len, tmp_idx, histogram, high_target_carrier
            )
        else:
            self._find_indices_baseline(
                sort_key, valid_len, tmp_idx, histogram, high_target_carrier
            )

    @jit
    def _find_indices_baseline(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            histogram_low_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            histogram_high_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range((valid_len + 255) // 256, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                _, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                histogram_low_reg = rr.vhistogram_accumulate(
                    histogram_low_reg, high, mask=mask8, bin=0
                )
                histogram_high_reg = rr.vhistogram_accumulate(
                    histogram_high_reg, high, mask=mask8, bin=1
                )
            rr.vstore(histogram, 0, histogram_low_reg, full16)
            rr.vstore(histogram, 128, histogram_high_reg, full16)
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            bottom_k = valid_len - self.topk + 1
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = rr.vload(histogram, 0)
            hist1 = rr.vload(histogram, 128)
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist0, bottom_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist1, bottom_k, mask=full16),
            )
            high_target = rr.vmin(
                rr.vreduce_min(candidate0, mask=full16),
                rr.vreduce_min(candidate1, mask=full16),
                mask=lane0,
            )
            rr.vstore(
                high_target_carrier,
                0,
                rr.vreinterpret(
                    rr.vunpack(high_target, dtypes.uint32, part="lower"),
                    dtypes.int32,
                ),
                rr.update_mask(1, elem_bits=32)[0],
            )
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
            one16 = rr.vdups(1, dtypes.uint16, mask=full16)
            bottom_k = valid_len - self.topk + 1
            high_target = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b16"),
                dtypes.uint16,
            )
            high_target8 = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b8"),
                dtypes.uint8,
            )
            prev = rr.vsub(high_target, one16, mask=full16)
            is_zero = rr.veqs(high_target, 0, mask=full16)
            prev = rr.vselect(zero16, prev, cond_mask=is_zero)
            prev_count = zero16
            for base in range_constexpr(0, 256, 128):
                hv = rr.vload(histogram, base)
                idx = rr.varange(base, dtypes.uint16)
                at_prev = rr.veq(idx, prev, mask=full16)
                picked = rr.vselect(hv, zero16, cond_mask=at_prev)
                prev_count = rr.vadd(
                    prev_count,
                    rr.vreduce_max(picked, mask=full16),
                    mask=full16,
                )
            prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
            next_k = rr.vsub(
                rr.vdups(bottom_k, dtypes.uint16, mask=full16),
                prev_count,
                mask=full16,
            )
            rr.vstore(tmp_idx, 0, next_k, lane0)

            l0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1 = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range((valid_len + 255) // 256, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0 = rr.vhistogram_accumulate(l0, low, mask=eq_high, bin=0)
                l1 = rr.vhistogram_accumulate(l1, low, mask=eq_high, bin=1)
            rr.vstore(histogram, 0, l0, full16)
            rr.vstore(histogram, 128, l1, full16)
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            next_k = rr.vload_broadcast(tmp_idx, 0)
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = rr.vload(histogram, 0)
            hist1 = rr.vload(histogram, 128)
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist0, next_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist1, next_k, mask=full16),
            )
            low_target = rr.vmin(
                rr.vreduce_min(candidate0, mask=full16),
                rr.vreduce_min(candidate1, mask=full16),
                mask=lane0,
            )
            rr.vstore(tmp_idx, 0, low_target, lane0)
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            high_target = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b16"),
                dtypes.uint16,
            )
            low_target = rr.vload_broadcast(tmp_idx, 0)
            kth = rr.vbitwise_or(
                rr.vshl(high_target, 8, mask=full16),
                low_target,
                mask=full16,
            )
            ureg = rr.vstore_unalign_begin(tmp_idx)
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.vgt(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.veq(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            rr.vstore_unalign_post(tmp_idx, 0, ureg)
            rr.vmem_bar("vst_vld")

    @jit
    def _find_indices_optimized(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            h0a = rr.vdups(0, dtypes.uint16, mask=full16)
            h1a = rr.vdups(0, dtypes.uint16, mask=full16)
            h0b = rr.vdups(0, dtypes.uint16, mask=full16)
            h1b = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range(0, (valid_len + 255) // 256, 2, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                _, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                h0a = rr.vhistogram_accumulate(h0a, high, mask=mask8, bin=0)
                h1a = rr.vhistogram_accumulate(h1a, high, mask=mask8, bin=1)
            for chunk in dsl_range(1, (valid_len + 255) // 256, 2, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                _, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                h0b = rr.vhistogram_accumulate(h0b, high, mask=mask8, bin=0)
                h1b = rr.vhistogram_accumulate(h1b, high, mask=mask8, bin=1)
            histogram_low_reg = rr.vadd(h0a, h0b, mask=full16)
            histogram_high_reg = rr.vadd(h1a, h1b, mask=full16)
            rr.vstore(histogram, 0, histogram_low_reg, full16)
            rr.vstore(histogram, 128, histogram_high_reg, full16)
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            bottom_k = valid_len - self.topk + 1
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = rr.vload(histogram, 0)
            hist1 = rr.vload(histogram, 128)
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist0, bottom_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist1, bottom_k, mask=full16),
            )
            high_target = rr.vmin(
                rr.vreduce_min(candidate0, mask=full16),
                rr.vreduce_min(candidate1, mask=full16),
                mask=lane0,
            )
            rr.vstore(
                high_target_carrier,
                0,
                rr.vreinterpret(
                    rr.vunpack(high_target, dtypes.uint32, part="lower"),
                    dtypes.int32,
                ),
                rr.update_mask(1, elem_bits=32)[0],
            )
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
            one16 = rr.vdups(1, dtypes.uint16, mask=full16)
            bottom_k = valid_len - self.topk + 1
            high_target = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b16"),
                dtypes.uint16,
            )
            high_target8 = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b8"),
                dtypes.uint8,
            )
            prev = rr.vsub(high_target, one16, mask=full16)
            is_zero = rr.veqs(high_target, 0, mask=full16)
            prev = rr.vselect(zero16, prev, cond_mask=is_zero)
            prev_count = zero16
            for base in range_constexpr(0, 256, 128):
                hv = rr.vload(histogram, base)
                idx = rr.varange(base, dtypes.uint16)
                at_prev = rr.veq(idx, prev, mask=full16)
                picked = rr.vselect(hv, zero16, cond_mask=at_prev)
                prev_count = rr.vadd(
                    prev_count,
                    rr.vreduce_max(picked, mask=full16),
                    mask=full16,
                )
            prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
            next_k = rr.vsub(
                rr.vdups(bottom_k, dtypes.uint16, mask=full16),
                prev_count,
                mask=full16,
            )
            rr.vstore(tmp_idx, 0, next_k, lane0)

            l0a = rr.vdups(0, dtypes.uint16, mask=full16)
            l1a = rr.vdups(0, dtypes.uint16, mask=full16)
            l0b = rr.vdups(0, dtypes.uint16, mask=full16)
            l1b = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range(0, (valid_len + 255) // 256, 2, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0a = rr.vhistogram_accumulate(l0a, low, mask=eq_high, bin=0)
                l1a = rr.vhistogram_accumulate(l1a, low, mask=eq_high, bin=1)
            for chunk in dsl_range(1, (valid_len + 255) // 256, 2, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0b = rr.vhistogram_accumulate(l0b, low, mask=eq_high, bin=0)
                l1b = rr.vhistogram_accumulate(l1b, low, mask=eq_high, bin=1)
            l0 = rr.vadd(l0a, l0b, mask=full16)
            l1 = rr.vadd(l1a, l1b, mask=full16)
            rr.vstore(histogram, 0, l0, full16)
            rr.vstore(histogram, 128, l1, full16)
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            next_k = rr.vload_broadcast(tmp_idx, 0)
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = rr.vload(histogram, 0)
            hist1 = rr.vload(histogram, 128)
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist0, next_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist1, next_k, mask=full16),
            )
            low_target = rr.vmin(
                rr.vreduce_min(candidate0, mask=full16),
                rr.vreduce_min(candidate1, mask=full16),
                mask=lane0,
            )
            rr.vstore(tmp_idx, 0, low_target, lane0)
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            high_target = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b16"),
                dtypes.uint16,
            )
            low_target = rr.vload_broadcast(tmp_idx, 0)
            kth = rr.vbitwise_or(
                rr.vshl(high_target, 8, mask=full16),
                low_target,
                mask=full16,
            )
            ureg = rr.vstore_unalign_begin(tmp_idx)
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.vgt(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.veq(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            rr.vstore_unalign_post(tmp_idx, 0, ureg)
            rr.vmem_bar("vst_vld")

    @jit
    def _select_pass(
        self,
        sort_key,
        valid_len,
        history_prefix,
        current_global_base,
        tmp_idx,
        histogram,
        high_target_carrier,
        history_idx_in,
        history_idx_out,
        next_history_key,
    ):
        """Run one ASC radix pass over one trunk or history+trunk."""
        self._find_indices(sort_key, valid_len, tmp_idx, histogram, high_target_carrier)

        with vf(mode="simd"):
            # ASC FindValueOutputVFImpl keeps the compacted positions as
            # uint16 and gathers 128 values per iteration.  Do not reuse the
            # uint16->uint32 unpack used by FindRealIndexVFImpl below: that
            # unpack exposes only 64 lanes and breaks value/index pairing
            # once history is merged with another trunk.
            for chunk in dsl_range(ceil_div(self.topk, 128), unroll=1):
                count = self.topk - chunk * 128
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                pos16 = rr.vload(tmp_idx, chunk * 128)
                selected_key = rr.vgather(sort_key, pos16, mask=mask16)
                rr.vstore(
                    next_history_key,
                    chunk * 128,
                    selected_key,
                    mask16,
                )

            # ASC FindRealIndexVFImpl loads the same compacted positions with
            # DIST_UNPACK_B16, yielding 64 uint32 lanes for history/current
            # index mapping.  Keep this separate from the value gather above.
            _full32 = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                pos16 = rr.vload(tmp_idx, chunk * 64)
                pos32 = rr.vunpack(pos16, dtypes.uint32, part="lower")
                is_current = rr.vges(pos32, history_prefix, mask=mask32)
                is_history = rr.vlts(pos32, history_prefix, mask=mask32)
                old_idx = rr.vgather(history_idx_in, pos32, mask=is_history)
                current_idx = rr.vadds(
                    rr.vreinterpret(pos32, dtypes.int32),
                    current_global_base - history_prefix,
                    mask=mask32,
                )
                global_idx = rr.vselect(current_idx, old_idx, cond_mask=is_current)
                rr.vstore(history_idx_out, chunk * 64, global_idx, mask32)
            rr.vmem_bar("vst_vld")

    @jit
    def _commit_history(self, next_idx, history_idx, next_history_key, history_key):
        with vf(mode="simd"):
            copy_mask32 = rr.update_mask(64, elem_bits=32)[0]
            copy_mask16 = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                rr.vstore(
                    history_idx,
                    chunk * 64,
                    rr.vload(next_idx, chunk * 64),
                    copy_mask32,
                )
            for chunk in dsl_range(ceil_div(self.topk, 128), unroll=1):
                rr.vstore(
                    history_key,
                    chunk * 128,
                    rr.vload(next_history_key, chunk * 128),
                    copy_mask16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def store_empty_row(
        self,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
    ):
        """Store the public empty-TopK representation without scanning GM."""
        idx_write = self.output_idx_stage.produce()
        val_write = self.output_value_bits_stage.produce()
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                rr.vstore(
                    idx_write,
                    chunk * 64,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    mask32,
                )
                rr.vstore(
                    val_write,
                    chunk * 64,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    mask16,
                )
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.output_idx_stage.consume())

        mem_copy(gm_selected_value_bits, self.output_value_bits_stage.consume())

    @jit
    def select_row(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
        need_values=1,
    ):
        if (
            need_values == 0
            and total_tokens >= self.topk
            and total_tokens <= self.trunk_len
        ):
            merge_key = self.workspace.merge_key
            self._load_merge(
                gm_sort_key,
                self.workspace.history_key,
                merge_key,
                dtypes.int64(0),
                dtypes.int64(0),
                total_tokens,
            )
            self._find_indices(
                merge_key.consume(),
                total_tokens,
                self.workspace.tmp_idx,
                self.workspace.histogram,
                self.workspace.high_target_carrier,
            )
            out = self.output_idx_stage.produce()
            tmp = self.workspace.tmp_idx
            with vf(mode="simd"):
                for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                    mask = rr.update_mask(self.topk - chunk * 64, elem_bits=32)[0]
                    positions = rr.vunpack(
                        rr.vload(tmp, chunk * 64), dtypes.uint32, part="lower"
                    )
                    indices = rr.vadds(
                        rr.vreinterpret(positions, dtypes.int32), index_base, mask=mask
                    )
                    valid = rr.vlts(
                        rr.varange(chunk * 64, dtypes.uint32),
                        output_valid_count,
                        mask=mask,
                    )
                    rr.vstore(
                        out,
                        chunk * 64,
                        rr.vselect(
                            indices,
                            rr.vdups(-1, dtypes.int32, mask=mask),
                            cond_mask=valid,
                        ),
                        mask,
                    )
                rr.vmem_bar("vst_vld")
            mem_copy(gm_sparse_indices, self.output_idx_stage.consume())
        elif (
            self.workspace.candidate_optimized
            and need_values != 0
            and total_tokens >= self.topk
            and total_tokens <= self.trunk_len
        ):
            self._select_row_single_trunk(
                gm_sort_key,
                total_tokens,
                output_valid_count,
                gm_sparse_indices,
                gm_selected_value_bits,
                index_base,
            )
        else:
            self._select_row_values(
                gm_sort_key,
                total_tokens,
                output_valid_count,
                gm_sparse_indices,
                gm_selected_value_bits,
                index_base,
            )

    @jit
    def _select_row_single_trunk(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
    ):
        """PV-C8 port: single-trunk fast path WITH values.

        When the entire input fits in one trunk (total_tokens <= 16384) and
        this is the first (and only) pass, history stays empty: skip its
        zero-init, skip the next_history_key/next_idx round-trip, and gather
        value bits directly from merge_key at the squeeze positions.
        Mathematically identical to _select_row_values for this case because
        history_prefix == 0 makes the position mapping an identity.
        """
        merge_key = self.workspace.merge_key
        tmp_idx = self.workspace.tmp_idx
        histogram = self.workspace.histogram
        high_target_carrier = self.workspace.high_target_carrier
        self._load_merge(
            gm_sort_key,
            self.workspace.history_key,
            merge_key,
            dtypes.int64(0),
            dtypes.int64(0),
            total_tokens,
        )
        merge_read = merge_key.consume()
        self._find_indices(
            merge_read, total_tokens, tmp_idx, histogram, high_target_carrier
        )
        out_idx = self.output_idx_stage.produce()
        out_val = self.output_value_bits_stage.produce()
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                pos16 = rr.vload(tmp_idx, chunk * 64)
                pos32 = rr.vunpack(pos16, dtypes.uint32, part="lower")
                keys = rr.vgather(merge_read, pos16, mask=mask16)
                positive = rr.veq(
                    rr.vbitwise_and(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                value_bits = rr.vselect(
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0xFFFF, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    cond_mask=positive,
                )
                valid32 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                valid16 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                indices = rr.vselect(
                    rr.vadds(
                        rr.vreinterpret(pos32, dtypes.int32),
                        index_base,
                        mask=mask32,
                    ),
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=valid32,
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    cond_mask=valid16,
                )
                rr.vstore(out_idx, chunk * 64, indices, mask32)
                rr.vstore(out_val, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.output_idx_stage.consume())
        mem_copy(gm_selected_value_bits, self.output_value_bits_stage.consume())

    @jit
    def _select_row_values(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
    ):
        # Rank masks use uint16 lanes; clamp the count before narrowing.
        if output_valid_count > self.topk:
            output_valid_count = self.topk
        merge_key = self.workspace.merge_key
        _current_trunk = self.workspace.current_trunk
        tmp_idx = self.workspace.tmp_idx
        histogram = self.workspace.histogram
        high_target_carrier = self.workspace.high_target_carrier
        history_key_storage = self.workspace.history_key
        history_key = history_key_storage
        next_history_key = self.workspace.next_history_key
        history_idx_storage = self.workspace.history_idx
        history_idx = history_idx_storage
        next_idx = self.workspace.next_idx
        output_idx_stage = self.output_idx_stage.produce()
        out_value_bits = self.output_value_bits_stage.produce()

        # Finite signed BF16 scores have positive sortable keys; zero marks
        # invalid tokens and empty history slots.
        with vf(mode="simd"):
            init_mask16 = rr.update_mask(128, elem_bits=16)[0]
            init_mask32 = rr.update_mask(64, elem_bits=32)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=init_mask16)
            zero32 = rr.vdups(0, dtypes.int32, mask=init_mask32)
            for chunk in dsl_range(ceil_div(self.topk_pad * 2, 128), unroll=1):
                rr.vstore(history_key, chunk * 128, zero16, init_mask16)
            for chunk in dsl_range(ceil_div(self.topk_pad * 2, 64), unroll=1):
                rr.vstore(history_idx, chunk * 64, zero32, init_mask32)
            rr.vmem_bar("vst_vld")

        full_trunk_count = total_tokens // CANDIDATE_LD_TRUNK_LEN
        tail_len = total_tokens % CANDIDATE_LD_TRUNK_LEN
        for pass_idx in dsl_range(full_trunk_count, unroll=1):
            parity = pass_idx % 2
            history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (parity, 0)
            )
            next_history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            history_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (parity, 0)
            )
            next_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            trunk_offset = pass_idx * self.trunk_len
            history_prefix = dtypes.int64((0 if pass_idx == 0 else self.topk_pad))
            self._load_merge(
                gm_sort_key,
                history_key,
                merge_key,
                trunk_offset,
                history_prefix,
                self.trunk_len,
            )
            self._select_pass(
                merge_key.consume(),
                history_prefix + self.trunk_len,
                history_prefix,
                trunk_offset,
                tmp_idx,
                histogram,
                high_target_carrier,
                history_idx,
                next_idx,
                next_history_key,
            )

        if tail_len > 0:
            parity = full_trunk_count % 2
            history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (parity, 0)
            )
            next_history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            history_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (parity, 0)
            )
            next_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            tail_offset = full_trunk_count * self.trunk_len
            history_prefix = dtypes.int64(self.topk_pad)
            if full_trunk_count == 0 and tail_len >= self.topk:
                history_prefix = dtypes.int64(0)
            _tail_len_pad = ((tail_len + 255) // 256) * 256
            self._load_merge(
                gm_sort_key,
                history_key,
                merge_key,
                tail_offset,
                history_prefix,
                tail_len,
            )
            self._select_pass(
                merge_key.consume(),
                history_prefix + tail_len,
                history_prefix,
                tail_offset,
                tmp_idx,
                histogram,
                high_target_carrier,
                history_idx,
                next_idx,
                next_history_key,
            )

        final_slot = (full_trunk_count + dtypes.int64((1 if tail_len > 0 else 0))) % 2
        final_idx = tile_slice(history_idx_storage, (1, self.topk_pad), (final_slot, 0))
        history_key = tile_slice(
            history_key_storage, (1, self.topk_pad), (final_slot, 0)
        )
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                indices = rr.vadds(
                    rr.vload(final_idx, chunk * 64), index_base, mask=mask32
                )
                keys = rr.vload(history_key, chunk * 64)
                positive = rr.veq(
                    rr.vbitwise_and(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                value_bits = rr.vselect(
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0xFFFF, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    cond_mask=positive,
                )
                valid_rank32 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                valid_rank16 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                indices = rr.vselect(
                    indices,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=valid_rank32,
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    cond_mask=valid_rank16,
                )
                rr.vstore(output_idx_stage, chunk * 64, indices, mask32)
                rr.vstore(out_value_bits, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.output_idx_stage.consume())
        mem_copy(gm_selected_value_bits, self.output_value_bits_stage.consume())


class CandidateLdTopKSelector(CandidateQliRawTopKSelector):
    def __init__(self, topk, workspace, splits):
        self.ub_pool = workspace.ub_pool
        super().__init__(topk, workspace)
        self.ld_length = int(splits) * self.topk
        self.ld_indices = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (1, self.ld_length),
            dtypes.int32,
            depth=1,
        )
        self.ld_bits = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (1, self.ld_length),
            dtypes.uint16,
            depth=1,
        )

    @jit
    def merge_ld(
        self,
        partial_indices,
        partial_bits,
        gm_sparse_indices,
        gm_selected_value_bits,
        valid_length,
        output_offset=0,
        preloaded=False,
        emit_gm=1,
    ):
        if not preloaded:
            mem_copy(
                reinterpret(self.ld_indices.produce(), shape=(1, valid_length)),
                partial_indices[0:1, 0:valid_length],
            )
            mem_copy(
                reinterpret(self.ld_bits.produce(), shape=(1, valid_length)),
                partial_bits[0:1, 0:valid_length],
            )
        current_trunk = self.workspace.merge_key
        trunk_write = current_trunk.produce()
        _history_key = self.workspace.history_key
        history_idx = self.workspace.history_idx
        ld_idx_read = self.ld_indices.consume()
        ld_bits_read = self.ld_bits.consume()
        with vf(mode="simd"):
            mask16 = rr.update_mask(64, elem_bits=16)[0]
            mask32 = rr.update_mask(64, elem_bits=32)[0]
            # LD may merge fewer than 64 entries for a small TopK capacity.
            for chunk in dsl_range((valid_length + 63) // 64, unroll=1):
                count = min(64, valid_length - chunk * 64)
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                bits = rr.vload(ld_bits_read, chunk * 64)
                indices = rr.vload(ld_idx_read, chunk * 64)
                positive_key = rr.vbitwise_xor(
                    bits, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
                )
                negative_key = rr.vbitwise_xor(
                    bits, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
                )
                negative = rr.veqs(
                    rr.vbitwise_and(
                        bits, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
                    ),
                    0x8000,
                    mask=mask16,
                )
                keys = rr.vselect(negative_key, positive_key, cond_mask=negative)
                valid32 = rr.vselect(
                    rr.vdups(1, dtypes.int32, mask=mask32),
                    rr.vdups(0, dtypes.int32, mask=mask32),
                    cond_mask=rr.vges(indices, 0, mask=mask32),
                )
                valid16 = rr.vpack(
                    rr.vreinterpret(valid32, dtypes.uint32), dtypes.uint16, part="lower"
                )
                keys = rr.vselect(
                    keys,
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    cond_mask=rr.veqs(valid16, 1, mask=mask16),
                )
                rr.vstore(trunk_write, chunk * 64, keys, mask16)
            rr.vmem_bar("vst_vld")
        trunk_read = current_trunk.consume()
        self._find_indices(
            trunk_read,
            valid_length,
            self.workspace.tmp_idx,
            self.workspace.histogram,
            self.workspace.high_target_carrier,
        )
        output_idx_stage = self.output_idx_stage.produce()
        out_value_bits = self.output_value_bits_stage.produce()
        output_valid_count = dtypes.int32(self.topk)
        _final_idx = history_idx
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                positions = rr.vunpack(
                    rr.vload(self.workspace.tmp_idx, chunk * 64),
                    dtypes.uint32,
                    part="lower",
                )
                indices = rr.vgather(
                    ld_idx_read, rr.vreinterpret(positions, dtypes.uint32), mask=mask32
                )
                positions16 = rr.vload(self.workspace.tmp_idx, chunk * 64)
                keys = rr.vgather(trunk_read, positions16, mask=mask16)
                value_bits = rr.vgather(ld_bits_read, positions16, mask=mask16)
                _valid_rank32 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                _valid_rank16 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                key_valid16 = rr.vselect(
                    rr.vdups(1, dtypes.uint16, mask=mask16),
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    cond_mask=rr.vges(keys, 1, mask=mask16),
                )
                key_valid32 = rr.vunpack(key_valid16, dtypes.uint32, part="lower")
                indices = rr.vselect(
                    indices,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.veqs(key_valid32, 1, mask=mask32),
                )
                valid_flags32 = rr.vselect(
                    rr.vdups(1, dtypes.int32, mask=mask32),
                    rr.vdups(0, dtypes.int32, mask=mask32),
                    cond_mask=rr.vges(indices, 0, mask=mask32),
                )
                valid_flags16 = rr.vpack(
                    rr.vreinterpret(valid_flags32, dtypes.uint32),
                    dtypes.uint16,
                    part="lower",
                )
                indices = rr.vselect(
                    indices,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.vges(indices, 0, mask=mask32),
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    cond_mask=rr.veqs(valid_flags16, 1, mask=mask16),
                )
                indices = rr.vselect(
                    rr.vadds(indices, output_offset, mask=mask32),
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.veqs(valid_flags32, 1, mask=mask32),
                )
                rr.vstore(output_idx_stage, chunk * 64, indices, mask32)
                rr.vstore(out_value_bits, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        # Intermediate LD rounds keep the carry in the on-core output stages;
        # only the final round drains them to GM.
        if emit_gm != 0:
            mem_copy(gm_sparse_indices, self.output_idx_stage.consume())
            mem_copy(gm_selected_value_bits, self.output_value_bits_stage.consume())


class CandidateLdMergeStage:
    def __init__(self, topk, splits, candidate_optimized=False):
        self.topk = topk
        self.length = topk * splits
        self.workspace = CandidateQliRawTopKWorkspace(
            CANDIDATE_LD_TRUNK_LEN, topk, candidate_optimized=candidate_optimized
        )
        self.selector = CandidateLdTopKSelector(topk, self.workspace, splits)

    @jit
    def __call__(
        self,
        partial_indices: Tensor,
        partial_bits: Tensor,
        indices: Tensor,
        bits: Tensor,
        metadata: Tensor,
        rows,
        workers,
        cu_seqlens_q: Tensor,
        query_tile_rows,
        has_cu,
        output_offset_address=0,
    ):
        fd = dtypes.int64(LD_RECORD_START + get_block_idx() * 2 + get_subblock_id())
        if metadata[fd, 0] != 0:
            batch_idx = dtypes.int64(metadata[fd, 1])
            begin = dtypes.int64(0)
            if has_cu:
                begin = dtypes.int64(cu_seqlens_q[batch_idx])
            row_base = begin + dtypes.int64(metadata[fd, 2]) * query_tile_rows
            workspace_begin = dtypes.int64(metadata[fd, 3])
            part_count = dtypes.int64(metadata[fd, 4])
            m_begin = dtypes.int64(metadata[fd, 5])
            m_end = m_begin + dtypes.int64(metadata[fd, 6])
            for m in dsl_range(m_begin, m_end, unroll=1):
                row = row_base + m
                output_offset = dtypes.int32(0)
                if output_offset_address != 0:
                    offsets = make_tensor(
                        make_pointer(
                            dtypes.int32, dtypes.int64(output_offset_address), MemLoc.GM
                        ),
                        make_layout((rows, 1), stride=(1, 1)),
                    )
                    output_offset = offsets[row, 0]
                for first in dsl_range(
                    0, part_count, self.length // self.topk - 1, unroll=1
                ):
                    count = min(part_count - first, self.length // self.topk - 1)
                    carry = dtypes.int64((1 if first > 0 else 0))
                    ld_idx_write = self.selector.ld_indices.produce()
                    ld_bits_write = self.selector.ld_bits.produce()
                    if first > 0:
                        # Carry stays on-core: the previous round's output-stage
                        # slots are the sole producer of this round's carry.
                        mem_copy(
                            reinterpret(ld_idx_write, shape=(1, self.topk)),
                            self.selector.output_idx_stage.consume(),
                        )
                        mem_copy(
                            reinterpret(ld_bits_write, shape=(1, self.topk)),
                            self.selector.output_value_bits_stage.consume(),
                        )
                    for part in dsl_range(0, count, unroll=1):
                        source_row = (
                            workspace_begin + first + part
                        ) * query_tile_rows + m
                        mem_copy(
                            tile_slice(ld_idx_write, (1, self.topk), (0, part + carry)),
                            tile_slice(
                                partial_indices, (1, self.topk), (source_row, 0)
                            ),
                        )
                        mem_copy(
                            tile_slice(
                                ld_bits_write, (1, self.topk), (0, part + carry)
                            ),
                            tile_slice(partial_bits, (1, self.topk), (source_row, 0)),
                        )
                    offset = dtypes.int32(
                        (output_offset if first + count == part_count else 0)
                    )
                    # Intermediate rounds emit into the UB output stages only;
                    # the final round drains to GM, so no GM write-read ordering
                    # is needed between rounds.
                    self.selector.merge_ld(
                        partial_indices,
                        partial_bits,
                        tile_slice(indices, (1, self.topk), (row, 0)),
                        tile_slice(bits, (1, self.topk), (row, 0)),
                        (count + carry) * self.topk,
                        offset,
                        True,
                        dtypes.int64((1 if first + count == part_count else 0)),
                    )


class QliCube:
    """AIC-owned native MXFP4 QK resources."""

    def __init__(self, pa_block_size, n1, query_rows):
        self.n1 = int(n1)
        self.tile_m = int(n1) * int(query_rows)
        self.pa_block_size = int(pa_block_size)
        self.copy_rows = gcd(QK_COLUMNS, self.pa_block_size)
        self.q_l1 = Channel(
            MemLoc.L1,
            (self.tile_m, LOGICAL_D),
            dtypes.fp4x2_e2m1,
            depth=2,
            data_format="nz",
        )
        self.k_l1 = Channel(
            MemLoc.L1,
            (QK_COLUMNS, LOGICAL_D),
            dtypes.fp4x2_e2m1,
            depth=4,
            data_format="nz",
        )
        self.q_scale_l1 = Channel(
            MemLoc.L1,
            (self.tile_m, 4),
            dtypes.float8_e8m0,
            depth=2,
            data_format="zn",
        )
        self.k_scale_l1 = Channel(
            MemLoc.L1, (4, QK_COLUMNS), dtypes.float8_e8m0, depth=4, data_format="nz"
        )
        self.l0a0 = Channel(
            MemLoc.L0A, (self.tile_m // 2, LOGICAL_D), dtypes.fp4x2_e2m1, depth=1
        )
        self.l0a1 = Channel(
            MemLoc.L0A, (self.tile_m // 2, LOGICAL_D), dtypes.fp4x2_e2m1, depth=1
        )
        self.l0b = Channel(
            MemLoc.L0B,
            (QK_COLUMNS, LOGICAL_D),
            dtypes.fp4x2_e2m1,
            depth=2,
            data_format="nz",
        )
        self.l0c = Channel(
            MemLoc.L0C, (self.tile_m // 2, QK_COLUMNS), dtypes.float32, depth=2
        )
        self.nd2nz_fp4 = make_copy_engine(format_transform="nd2nz")
        self.nd2nz_key = make_copy_engine(format_transform="nd2nz")
        self.scale_a = make_copy_engine(format_transform="mx_scale_and")
        self.scale_b = make_copy_engine(format_transform="mx_scale_bdn")
        self.fixpipe_aiv0 = make_copy_engine()
        self.fixpipe_aiv1 = make_copy_engine()

    @jit
    def load_query(self, query, query_scale, query_token_start):
        source_row = dtypes.int64(query_token_start) * self.n1
        # MQSMLA-style dynamic Q load: the tile window on the produced slot
        # pins the full tile_m-row NZ frame (static C0 stride) while a plain
        # dynamic slice keeps the copy height clamped to the available rows.
        # Rows past the task span keep stale L1 content; the vector/output
        # path masks them by task_active_count.
        avail = dtypes.int64(query.shape[0]) - source_row
        avail = avail if avail > 0 else dtypes.int64(0)
        rows_to_copy = avail if avail < self.tile_m else dtypes.int64(self.tile_m)
        mem_copy(
            tile_slice(self.q_l1.produce(), (self.tile_m, LOGICAL_D), (0, 0)),
            query[source_row : source_row + rows_to_copy, None],
            engine=self.nd2nz_fp4,
        )
        mem_copy(
            self.q_scale_l1.produce(),
            tile_slice(
                _offset_view(query_scale, (source_row, 0, 0)),
                (self.tile_m, 2, 2),
                (0, 0, 0),
            ),
            engine=self.scale_a,
        )
        # Keep the Cube M capacity fixed for split-M Channel consumers.
        # ND2NZ uses the same fixed C0 stride; rows outside task_span_count
        # are ignored by the vector/output path.
        query_read = self.q_l1.consume()
        scale_read = self.q_scale_l1.consume()
        mem_copy(
            self.l0a0.produce(),
            tile_slice(
                reinterpret(query_read, shape=(self.tile_m, LOGICAL_D)),
                (self.tile_m // 2, LOGICAL_D),
                (0, 0),
            ),
            mx_scale=tile_slice(
                reinterpret(scale_read, shape=(self.tile_m, 4)),
                (self.tile_m // 2, 4),
                (0, 0),
            ),
        )
        mem_copy(
            self.l0a1.produce(),
            tile_slice(
                reinterpret(query_read, shape=(self.tile_m, LOGICAL_D)),
                (self.tile_m // 2, LOGICAL_D),
                (1, 0),
            ),
            mx_scale=tile_slice(
                reinterpret(scale_read, shape=(self.tile_m, 4)),
                (self.tile_m // 2, 4),
                (1, 0),
            ),
        )

    @jit
    def compute_qk(
        self,
        key,
        key_scale,
        block_table,
        batch_idx,
        actual_s2,
        key_stride0,
        key_dequant_scale_stride0,
        has_block_table,
        token_tile,
        qk_handoff,
    ):
        key_write = self.k_l1.produce()
        scale_write = self.k_scale_l1.produce()
        for piece in range(QK_COLUMNS // self.copy_rows):
            logical_row = token_tile * TILE_N + piece * self.copy_rows
            safe_row = logical_row
            if logical_row >= dtypes.int64(actual_s2):
                safe_row = 0
            logical_page = safe_row // self.pa_block_size
            page_offset = safe_row % self.pa_block_size
            physical_page = logical_page
            if has_block_table:
                physical_page = dtypes.int64(block_table[batch_idx, logical_page])
            source_row = (
                physical_page * (key_stride0 // KEY_STORAGE_ROW_ELEMENTS) + page_offset
            )
            scale_row = (
                physical_page
                * (key_dequant_scale_stride0 // KEY_SCALE_STORAGE_ROW_ELEMENTS)
                + page_offset
            )
            mem_copy(
                tile_slice(key_write, (self.copy_rows, LOGICAL_D), (piece, 0)),
                key[source_row : source_row + self.copy_rows, None],
                engine=self.nd2nz_key,
            )
            mem_copy(
                tile_slice(scale_write, (4, self.copy_rows), (0, piece)),
                key_scale[scale_row : scale_row + self.copy_rows, None, None],
                engine=self.scale_b,
            )
        mem_copy(
            self.l0b.produce(), self.k_l1.consume(), mx_scale=self.k_scale_l1.consume()
        )
        key_read = self.l0b.consume()
        dsl_matmul(
            self.l0c.produce(), self.l0a0.consume(), key_read, init=True, unit_flag=3
        )
        qk_l0c_read0 = self.l0c.consume()
        mem_copy(
            reinterpret(qk_handoff, shape=(self.tile_m // 2, 128), stride=(256, 1)),
            reinterpret(qk_l0c_read0, shape=(self.tile_m // 2, 128)),
            engine=self.fixpipe_aiv0,
            dst_subblock=0,
            unit_flag=3,
        )
        mem_copy(
            reinterpret(
                qk_handoff,
                shape=(self.tile_m // 2, 128),
                stride=(256, 1),
                offset=(self.tile_m // 2) * 512,
            ),
            reinterpret(
                qk_l0c_read0,
                shape=(self.tile_m // 2, 128),
                offset=(self.tile_m // 2) * 128 * 4,
            ),
            engine=self.fixpipe_aiv0,
            dst_subblock=0,
            unit_flag=3,
        )
        dsl_matmul(
            self.l0c.produce(), self.l0a1.consume(), key_read, init=True, unit_flag=3
        )
        qk_l0c_read1 = self.l0c.consume()
        mem_copy(
            reinterpret(qk_handoff, shape=(self.tile_m // 2, 128), stride=(256, 1)),
            reinterpret(qk_l0c_read1, shape=(self.tile_m // 2, 128)),
            engine=self.fixpipe_aiv1,
            dst_subblock=1,
            unit_flag=3,
        )
        mem_copy(
            reinterpret(
                qk_handoff,
                shape=(self.tile_m // 2, 128),
                stride=(256, 1),
                offset=(self.tile_m // 2) * 512,
            ),
            reinterpret(
                qk_l0c_read1,
                shape=(self.tile_m // 2, 128),
                offset=(self.tile_m // 2) * 128 * 4,
            ),
            engine=self.fixpipe_aiv1,
            dst_subblock=1,
            unit_flag=3,
        )


class QliVector:
    """Each AIV consumes its 64-row split and produces two QLI rows."""

    def __init__(
        self,
        subblock_idx,
        mask_mode,
        cmp_ratio,
        candidate_enabled,
        candidate_topk_count,
        n1,
        query_rows,
        weight_addr,
        cache_enabled,
    ):
        self.ub_pool = _UbPool(weight_addr)
        self.cache_enabled = bool(cache_enabled)
        self.n1 = int(n1)
        self.query_rows = int(query_rows)
        self.vec_rows = int(query_rows) // 2
        self.subblock_idx = subblock_idx
        self.mask_mode = int(mask_mode)
        self.cmp_ratio = int(cmp_ratio)
        self.candidate_enabled = bool(candidate_enabled)
        self.candidate_topk_count = int(candidate_topk_count)
        self.weight_ub = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (self.vec_rows, self.n1),
            dtypes.float32,
            depth=1,
            addr=weight_addr,
        )
        self.packed_weight = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.vec_rows, 128),
            dtypes.bfloat16,
        )
        self.score_ub = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.vec_rows, QK_COLUMNS),
            dtypes.bfloat16,
        )
        self.key_ub = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (self.vec_rows, QK_COLUMNS),
            dtypes.uint16,
            depth=2,
        )
        if self.candidate_enabled:
            # Each row is written by VF before one batched DMA reads them;
            # the Channel slots own the V->MTE3 ordering.
            self.candidate_key_ub = _ub_channel(
                self.ub_pool,
                MemLoc.UB,
                (self.vec_rows, QK_COLUMNS // CANDIDATE_BLOCK_SIZE),
                dtypes.uint16,
                depth=1,
            )
        self.candidate_length_ub = _ub_channel(
            self.ub_pool,
            MemLoc.UB,
            (1,),
            dtypes.int32,
            depth=1,
        )

    @jit
    def load_weights(self, weights, public_row_base, task_span_count):
        weight_row = (
            dtypes.int64(public_row_base)
            + dtypes.int64(self.subblock_idx) * self.vec_rows
        )
        if dtypes.int32(self.subblock_idx) * self.vec_rows >= task_span_count:
            weight_row = dtypes.int64(public_row_base)
        mem_copy(
            self.weight_ub.produce(),
            tile_slice(
                _offset_view(weights, (weight_row, 0)), (self.vec_rows, self.n1), (0, 0)
            ),
        )
        weight_read = self.weight_ub.consume()
        with vf(mode="simd"):
            mask32 = rr.update_mask(self.n1, elem_bits=32)[0]
            mask16 = rr.update_mask(self.n1, elem_bits=16)[0]
            for row in range_constexpr(self.vec_rows):
                weight = rr.vcast(
                    rr.vload(weight_read, row * self.n1),
                    dtypes.bfloat16,
                    mask=mask32,
                    reg_layout=rr.RegLayout.ZERO,
                )
                packed = rr.vpack(
                    rr.vreinterpret_lanes(weight, dtypes.uint32),
                    dtypes.uint16,
                    part="lower",
                )
                rr.vstore(
                    self.packed_weight,
                    row * 128,
                    rr.vreinterpret(packed, dtypes.bfloat16),
                    mask16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def store_empty_candidate_lengths(
        self, candidate_length_out, public_row_base, task_span_count
    ):
        """Write zero Candidate length for both rows owned by this AIV."""
        for row_in_subblock in range_constexpr(self.vec_rows):
            local_row = dtypes.int32(
                self.subblock_idx * self.vec_rows + row_in_subblock
            )
            if local_row < task_span_count:
                length_write = self.candidate_length_ub.produce()
                with vf(mode="simd"):
                    length_mask = rr.update_mask(1, elem_bits=32)[0]
                    rr.vstore(
                        length_write,
                        0,
                        rr.vdups(0, dtypes.int32, mask=length_mask),
                        length_mask,
                    )
                    rr.vmem_bar("vst_vld")

                mem_copy(
                    tile_slice(
                        candidate_length_out,
                        (1,),
                        (dtypes.int64(public_row_base) + dtypes.int64(local_row),),
                    ),
                    self.candidate_length_ub.consume(),
                )

    @jit
    def compute_vector1(
        self,
        qk_handoff,
        weights,
        key_out,
        candidate_key_out,
        public_row_base,
        task_span_count,
        task_active_count,
        workspace_task_idx,
        token_offset,
        actual_s2,
        task_q_offset,
        task_used_q,
        cmp_residual,
        total_candidate_tokens,
        candidate_length_out,
        workspace_token_offset,
        tile_tokens,
        stream_hist,
        hist,
        cache_task,
    ):
        group_count = dtypes.int32(0)
        if dtypes.int32(self.subblock_idx) * self.vec_rows < task_active_count:
            group_count = dtypes.int32(self.n1)
        s2_len_v0 = actual_s2
        if self.mask_mode == 3:
            s2_len_v0 = max(
                0,
                min(
                    actual_s2,
                    (
                        actual_s2 * self.cmp_ratio
                        + cmp_residual
                        - task_used_q
                        + task_q_offset
                        + dtypes.int32(self.subblock_idx) * self.vec_rows
                        + 0
                        + 1
                    )
                    // self.cmp_ratio,
                ),
            )
        if dtypes.int32(self.subblock_idx) * self.vec_rows + 0 >= task_active_count:
            s2_len_v0 = 0
        s2_len_v0 = max(
            0, min(dtypes.int32(tile_tokens), s2_len_v0 - dtypes.int32(token_offset))
        )
        s2_len_v1 = actual_s2
        if self.mask_mode == 3:
            s2_len_v1 = max(
                0,
                min(
                    actual_s2,
                    (
                        actual_s2 * self.cmp_ratio
                        + cmp_residual
                        - task_used_q
                        + task_q_offset
                        + dtypes.int32(self.subblock_idx) * self.vec_rows
                        + 1
                        + 1
                    )
                    // self.cmp_ratio,
                ),
            )
        if dtypes.int32(self.subblock_idx) * self.vec_rows + 1 >= task_active_count:
            s2_len_v1 = 0
        s2_len_v1 = max(
            0, min(dtypes.int32(tile_tokens), s2_len_v1 - dtypes.int32(token_offset))
        )
        s2_len_v2 = actual_s2
        if self.mask_mode == 3:
            s2_len_v2 = max(
                0,
                min(
                    actual_s2,
                    (
                        actual_s2 * self.cmp_ratio
                        + cmp_residual
                        - task_used_q
                        + task_q_offset
                        + dtypes.int32(self.subblock_idx) * self.vec_rows
                        + 2
                        + 1
                    )
                    // self.cmp_ratio,
                ),
            )
        if dtypes.int32(self.subblock_idx) * self.vec_rows + 2 >= task_active_count:
            s2_len_v2 = 0
        s2_len_v2 = max(
            0, min(dtypes.int32(tile_tokens), s2_len_v2 - dtypes.int32(token_offset))
        )
        if self.cache_enabled and cache_task:
            if self.vec_rows == 3:
                self._score_three_cache(
                    qk_handoff,
                    group_count,
                    hist,
                    workspace_token_offset,
                    s2_len_v0,
                    s2_len_v1,
                    s2_len_v2,
                )
            else:
                self._score_two_cache(
                    qk_handoff,
                    group_count,
                    hist,
                    workspace_token_offset,
                    s2_len_v0,
                    s2_len_v1,
                )
        else:
            if self.candidate_enabled:
                if self.vec_rows == 3:
                    self._score_three(qk_handoff, group_count)
                else:
                    self._score_two(qk_handoff, group_count)
            elif self.vec_rows == 3:
                self._score_three_hist(
                    qk_handoff,
                    group_count,
                    hist,
                    workspace_token_offset,
                    s2_len_v0,
                    s2_len_v1,
                    s2_len_v2,
                )
            else:
                self._score_two_hist(
                    qk_handoff,
                    group_count,
                    hist,
                    workspace_token_offset,
                    s2_len_v0,
                    s2_len_v1,
                )
            if self.candidate_enabled:
                cand_key_write = self.candidate_key_ub.produce()
                for row_in_subblock in range_constexpr(self.vec_rows):
                    local_row = dtypes.int32(
                        self.subblock_idx * self.vec_rows + row_in_subblock
                    )
                    public_row = dtypes.int64(public_row_base) + dtypes.int64(local_row)
                    _workspace_row = (
                        workspace_task_idx * self.query_rows
                        + self.subblock_idx * self.vec_rows
                        + row_in_subblock
                    )
                    valid_s2 = actual_s2
                    if self.mask_mode == 3:
                        query_in_batch = (
                            dtypes.int32(task_q_offset)
                            + dtypes.int32(self.subblock_idx) * self.vec_rows
                            + dtypes.int32(row_in_subblock)
                        )
                        valid_s2 = (
                            actual_s2 * self.cmp_ratio
                            + cmp_residual
                            - task_used_q
                            + query_in_batch
                            + 1
                        ) // self.cmp_ratio
                        if valid_s2 < 0:
                            valid_s2 = 0
                        if valid_s2 > actual_s2:
                            valid_s2 = actual_s2
                    if local_row >= task_active_count:
                        valid_s2 = 0
                    valid_candidate_count = (
                        valid_s2 + CANDIDATE_BLOCK_SIZE - 1
                    ) // CANDIDATE_BLOCK_SIZE
                    local_candidate_count = (
                        valid_candidate_count
                        - dtypes.int32(token_offset) // CANDIDATE_BLOCK_SIZE
                    )
                    if local_candidate_count < 0:
                        local_candidate_count = 0
                    if local_candidate_count > QK_COLUMNS // CANDIDATE_BLOCK_SIZE:
                        local_candidate_count = QK_COLUMNS // CANDIDATE_BLOCK_SIZE
                    partial_target = dtypes.int32(QK_COLUMNS // CANDIDATE_BLOCK_SIZE)
                    if valid_s2 % CANDIDATE_BLOCK_SIZE != 0:
                        if valid_s2 > dtypes.int32(token_offset):
                            if valid_s2 < dtypes.int32(token_offset) + QK_COLUMNS:
                                partial_target = (
                                    valid_s2 - dtypes.int32(token_offset)
                                ) // CANDIDATE_BLOCK_SIZE

                    if self.candidate_enabled:
                        with vf(mode="simd"):
                            candidate_count = QK_COLUMNS // CANDIDATE_BLOCK_SIZE
                            candidate_mask16 = rr.update_mask(
                                candidate_count, elem_bits=16
                            )[0]
                            candidate_idx = rr.vadds(
                                rr.vmuls(
                                    rr.varange(0, dtypes.uint16),
                                    CANDIDATE_BLOCK_SIZE,
                                    mask=candidate_mask16,
                                ),
                                row_in_subblock * QK_COLUMNS,
                                mask=candidate_mask16,
                            )
                            candidate_max = rr.vgather(
                                self.score_ub,
                                candidate_idx,
                                mask=candidate_mask16,
                            )
                            for lane in range_constexpr(1, CANDIDATE_BLOCK_SIZE):
                                candidate_max = rr.vmax(
                                    candidate_max,
                                    rr.vgather(
                                        self.score_ub,
                                        rr.vadds(
                                            candidate_idx,
                                            lane,
                                            mask=candidate_mask16,
                                        ),
                                        mask=candidate_mask16,
                                    ),
                                    mask=candidate_mask16,
                                )
                            candidate_global_idx = rr.varange(0, dtypes.uint16)
                            candidate_max = rr.vselect(
                                candidate_max,
                                rr.vreinterpret(
                                    rr.vdups(
                                        0xFF80,
                                        dtypes.uint16,
                                        mask=candidate_mask16,
                                    ),
                                    dtypes.bfloat16,
                                ),
                                cond_mask=rr.vlts(
                                    candidate_global_idx,
                                    local_candidate_count,
                                    mask=candidate_mask16,
                                ),
                            )
                            candidate_max = rr.vselect(
                                rr.vreinterpret(
                                    rr.vdups(
                                        0x7F80,
                                        dtypes.uint16,
                                        mask=candidate_mask16,
                                    ),
                                    dtypes.bfloat16,
                                ),
                                candidate_max,
                                cond_mask=rr.veqs(
                                    candidate_global_idx,
                                    partial_target,
                                    mask=candidate_mask16,
                                ),
                            )
                            candidate_bits16 = rr.vreinterpret(
                                candidate_max, dtypes.uint16
                            )
                            candidate_key = rr.vbitwise_xor(
                                candidate_bits16,
                                rr.vdups(0x8000, dtypes.uint16, mask=candidate_mask16),
                                mask=candidate_mask16,
                            )
                            # Negative IEEE values require inverted bits to preserve numeric order.
                            candidate_negative = rr.veq(
                                rr.vbitwise_and(
                                    candidate_bits16,
                                    rr.vdups(
                                        0x8000, dtypes.uint16, mask=candidate_mask16
                                    ),
                                    mask=candidate_mask16,
                                ),
                                rr.vdups(0x8000, dtypes.uint16, mask=candidate_mask16),
                                mask=candidate_mask16,
                            )
                            candidate_key = rr.vselect(
                                rr.vbitwise_xor(
                                    candidate_bits16,
                                    rr.vdups(
                                        0xFFFF, dtypes.uint16, mask=candidate_mask16
                                    ),
                                    mask=candidate_mask16,
                                ),
                                candidate_key,
                                cond_mask=candidate_negative,
                            )
                            candidate_key = rr.vselect(
                                candidate_key,
                                rr.vdups(0, dtypes.uint16, mask=candidate_mask16),
                                cond_mask=rr.vlts(
                                    rr.varange(0, dtypes.uint16),
                                    local_candidate_count,
                                    mask=candidate_mask16,
                                ),
                            )
                            rr.vstore(
                                cand_key_write,
                                row_in_subblock * (QK_COLUMNS // CANDIDATE_BLOCK_SIZE),
                                candidate_key,
                                candidate_mask16,
                            )
                            rr.vmem_bar("vst_vld")

                    if self.candidate_enabled:
                        if local_row < task_span_count:
                            if token_offset == 0:
                                length = (
                                    valid_s2 + CANDIDATE_BLOCK_SIZE - 1
                                ) // CANDIDATE_BLOCK_SIZE
                                if length > self.candidate_topk_count:
                                    length = dtypes.int32(self.candidate_topk_count)
                                length_write = self.candidate_length_ub.produce()
                                with vf(mode="simd"):
                                    mask = rr.update_mask(1, elem_bits=32)[0]
                                    rr.vstore(
                                        length_write,
                                        0,
                                        rr.vdups(length, dtypes.int32, mask=mask),
                                        mask,
                                    )
                                mem_copy(
                                    tile_slice(
                                        candidate_length_out, (1,), (public_row,)
                                    ),
                                    self.candidate_length_ub.consume(),
                                )

            rows_to_store = (
                task_span_count - dtypes.int32(self.subblock_idx) * self.vec_rows
            )
            if rows_to_store < 0:
                rows_to_store = 0
            if rows_to_store > self.vec_rows:
                rows_to_store = self.vec_rows
            first_workspace_row = (
                workspace_task_idx * self.query_rows + self.subblock_idx * self.vec_rows
            )
            mem_copy(
                tile_slice(
                    _offset_view(
                        key_out, (first_workspace_row, workspace_token_offset)
                    ),
                    (self.vec_rows, QK_COLUMNS),
                    (0, 0),
                ),
                reinterpret(
                    self.key_ub.consume(),
                    shape=(dtypes.int64(rows_to_store), QK_COLUMNS),
                ),
            )
            if self.candidate_enabled:
                mem_copy(
                    tile_slice(
                        _offset_view(
                            candidate_key_out,
                            (
                                first_workspace_row,
                                workspace_token_offset // CANDIDATE_BLOCK_SIZE,
                            ),
                        ),
                        (self.vec_rows, QK_COLUMNS // CANDIDATE_BLOCK_SIZE),
                        (0, 0),
                    ),
                    reinterpret(
                        self.candidate_key_ub.consume(),
                        shape=(
                            dtypes.int64(rows_to_store),
                            QK_COLUMNS // CANDIDATE_BLOCK_SIZE,
                        ),
                    ),
                )

    @jit
    def _score_two(self, qk_handoff, group_count):
        if self.candidate_enabled:
            self._score_two_candidate(qk_handoff, group_count)
        else:
            self._score_two_plain(qk_handoff, group_count)

    @jit
    def _score_two_candidate(self, qk_handoff, group_count):
        key_write = self.key_ub.produce()
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            _weight_mask = rr.update_mask(self.n1, elem_bits=32)[0]
            bf16_weight0 = rr.vload(self.packed_weight, 0 * 128)
            query0_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query0_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            bf16_weight1 = rr.vload(self.packed_weight, 1 * 128)
            query1_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query1_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            for group in dsl_range(group_count, unroll=4):
                lane = rr.vdups(group, dtypes.uint16, mask=mask16)
                w0 = rr.vgather_reg(bf16_weight0, lane)
                w1 = rr.vgather_reg(bf16_weight1, lane)
                qk0_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half0_acc_reg = rr.vmadd(
                    qk0_0, w0, query0_s2_half0_acc_reg, mask=mask16
                )
                qk0_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half1_acc_reg = rr.vmadd(
                    qk0_1, w0, query0_s2_half1_acc_reg, mask=mask16
                )
                qk1_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half0_acc_reg = rr.vmadd(
                    qk1_0, w1, query1_s2_half0_acc_reg, mask=mask16
                )
                qk1_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half1_acc_reg = rr.vmadd(
                    qk1_1, w1, query1_s2_half1_acc_reg, mask=mask16
                )
            query0_s2_half0_acc_reg = rr.vadds(
                query0_s2_half0_acc_reg, 0.0, mask=mask16
            )
            rr.vstore(
                self.score_ub, 0 * QK_COLUMNS + 0 * 128, query0_s2_half0_acc_reg, mask16
            )
            bits0_0 = rr.vreinterpret(query0_s2_half0_acc_reg, dtypes.uint16)
            sign0_0 = rr.vbitwise_and(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_0 = rr.veqs(sign0_0, 0x8000, mask=mask16)
            positive_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                0 * QK_COLUMNS + 0 * 128,
                rr.vselect(negative_key0_0, positive_key0_0, cond_mask=negative0_0),
                mask16,
            )
            query0_s2_half1_acc_reg = rr.vadds(
                query0_s2_half1_acc_reg, 0.0, mask=mask16
            )
            rr.vstore(
                self.score_ub, 0 * QK_COLUMNS + 1 * 128, query0_s2_half1_acc_reg, mask16
            )
            bits0_1 = rr.vreinterpret(query0_s2_half1_acc_reg, dtypes.uint16)
            sign0_1 = rr.vbitwise_and(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_1 = rr.veqs(sign0_1, 0x8000, mask=mask16)
            positive_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                0 * QK_COLUMNS + 1 * 128,
                rr.vselect(negative_key0_1, positive_key0_1, cond_mask=negative0_1),
                mask16,
            )
            query1_s2_half0_acc_reg = rr.vadds(
                query1_s2_half0_acc_reg, 0.0, mask=mask16
            )
            rr.vstore(
                self.score_ub, 1 * QK_COLUMNS + 0 * 128, query1_s2_half0_acc_reg, mask16
            )
            bits1_0 = rr.vreinterpret(query1_s2_half0_acc_reg, dtypes.uint16)
            sign1_0 = rr.vbitwise_and(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_0 = rr.veqs(sign1_0, 0x8000, mask=mask16)
            positive_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                1 * QK_COLUMNS + 0 * 128,
                rr.vselect(negative_key1_0, positive_key1_0, cond_mask=negative1_0),
                mask16,
            )
            query1_s2_half1_acc_reg = rr.vadds(
                query1_s2_half1_acc_reg, 0.0, mask=mask16
            )
            rr.vstore(
                self.score_ub, 1 * QK_COLUMNS + 1 * 128, query1_s2_half1_acc_reg, mask16
            )
            bits1_1 = rr.vreinterpret(query1_s2_half1_acc_reg, dtypes.uint16)
            sign1_1 = rr.vbitwise_and(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_1 = rr.veqs(sign1_1, 0x8000, mask=mask16)
            positive_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                1 * QK_COLUMNS + 1 * 128,
                rr.vselect(negative_key1_1, positive_key1_1, cond_mask=negative1_1),
                mask16,
            )
            rr.vmem_bar("vst_vld")

    @jit
    def _score_two_plain(self, qk_handoff, group_count):
        key_write = self.key_ub.produce()
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            _weight_mask = rr.update_mask(self.n1, elem_bits=32)[0]
            bf16_weight0 = rr.vload(self.packed_weight, 0 * 128)
            query0_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query0_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            bf16_weight1 = rr.vload(self.packed_weight, 1 * 128)
            query1_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query1_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            for group in dsl_range(group_count, unroll=4):
                lane = rr.vdups(group, dtypes.uint16, mask=mask16)
                w0 = rr.vgather_reg(bf16_weight0, lane)
                w1 = rr.vgather_reg(bf16_weight1, lane)
                qk0_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half0_acc_reg = rr.vmadd(
                    qk0_0, w0, query0_s2_half0_acc_reg, mask=mask16
                )
                qk0_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half1_acc_reg = rr.vmadd(
                    qk0_1, w0, query0_s2_half1_acc_reg, mask=mask16
                )
                qk1_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half0_acc_reg = rr.vmadd(
                    qk1_0, w1, query1_s2_half0_acc_reg, mask=mask16
                )
                qk1_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half1_acc_reg = rr.vmadd(
                    qk1_1, w1, query1_s2_half1_acc_reg, mask=mask16
                )
            query0_s2_half0_acc_reg = rr.vadds(
                query0_s2_half0_acc_reg, 0.0, mask=mask16
            )
            bits0_0 = rr.vreinterpret(query0_s2_half0_acc_reg, dtypes.uint16)
            sign0_0 = rr.vbitwise_and(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_0 = rr.veqs(sign0_0, 0x8000, mask=mask16)
            positive_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                0 * QK_COLUMNS + 0 * 128,
                rr.vselect(negative_key0_0, positive_key0_0, cond_mask=negative0_0),
                mask16,
            )
            query0_s2_half1_acc_reg = rr.vadds(
                query0_s2_half1_acc_reg, 0.0, mask=mask16
            )
            bits0_1 = rr.vreinterpret(query0_s2_half1_acc_reg, dtypes.uint16)
            sign0_1 = rr.vbitwise_and(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_1 = rr.veqs(sign0_1, 0x8000, mask=mask16)
            positive_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                0 * QK_COLUMNS + 1 * 128,
                rr.vselect(negative_key0_1, positive_key0_1, cond_mask=negative0_1),
                mask16,
            )
            query1_s2_half0_acc_reg = rr.vadds(
                query1_s2_half0_acc_reg, 0.0, mask=mask16
            )
            bits1_0 = rr.vreinterpret(query1_s2_half0_acc_reg, dtypes.uint16)
            sign1_0 = rr.vbitwise_and(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_0 = rr.veqs(sign1_0, 0x8000, mask=mask16)
            positive_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                1 * QK_COLUMNS + 0 * 128,
                rr.vselect(negative_key1_0, positive_key1_0, cond_mask=negative1_0),
                mask16,
            )
            query1_s2_half1_acc_reg = rr.vadds(
                query1_s2_half1_acc_reg, 0.0, mask=mask16
            )
            bits1_1 = rr.vreinterpret(query1_s2_half1_acc_reg, dtypes.uint16)
            sign1_1 = rr.vbitwise_and(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_1 = rr.veqs(sign1_1, 0x8000, mask=mask16)
            positive_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                1 * QK_COLUMNS + 1 * 128,
                rr.vselect(negative_key1_1, positive_key1_1, cond_mask=negative1_1),
                mask16,
            )
            rr.vmem_bar("vst_vld")

    @jit
    def _score_two_hist(
        self, qk_handoff, group_count, hist, workspace_offset, valid0, valid1
    ):
        key_write = self.key_ub.produce()
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            zero_hist = rr.vdups(0, dtypes.uint16, mask=mask16)
            reset16 = rr.veqs(
                rr.vdups(
                    dtypes.int32(workspace_offset < 512), dtypes.uint16, mask=mask16
                ),
                1,
                mask=mask16,
            )
            _weight_mask = rr.update_mask(self.n1, elem_bits=32)[0]
            bf16_weight0 = rr.vload(self.packed_weight, 0 * 128)
            query0_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query0_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            bf16_weight1 = rr.vload(self.packed_weight, 1 * 128)
            query1_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query1_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            for group in dsl_range(group_count, unroll=4):
                lane = rr.vdups(group, dtypes.uint16, mask=mask16)
                w0 = rr.vgather_reg(bf16_weight0, lane)
                w1 = rr.vgather_reg(bf16_weight1, lane)
                qk0_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half0_acc_reg = rr.vmadd(
                    qk0_0, w0, query0_s2_half0_acc_reg, mask=mask16
                )
                qk0_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half1_acc_reg = rr.vmadd(
                    qk0_1, w0, query0_s2_half1_acc_reg, mask=mask16
                )
                qk1_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half0_acc_reg = rr.vmadd(
                    qk1_0, w1, query1_s2_half0_acc_reg, mask=mask16
                )
                qk1_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half1_acc_reg = rr.vmadd(
                    qk1_1, w1, query1_s2_half1_acc_reg, mask=mask16
                )
            query0_s2_half0_acc_reg = rr.vadds(
                query0_s2_half0_acc_reg, 0.0, mask=mask16
            )
            bits0_0 = rr.vreinterpret(query0_s2_half0_acc_reg, dtypes.uint16)
            sign0_0 = rr.vbitwise_and(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_0 = rr.veqs(sign0_0, 0x8000, mask=mask16)
            positive_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                0 * QK_COLUMNS + 0 * 128,
                rr.vselect(negative_key0_0, positive_key0_0, cond_mask=negative0_0),
                mask16,
            )
            query0_s2_half1_acc_reg = rr.vadds(
                query0_s2_half1_acc_reg, 0.0, mask=mask16
            )
            bits0_1 = rr.vreinterpret(query0_s2_half1_acc_reg, dtypes.uint16)
            sign0_1 = rr.vbitwise_and(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_1 = rr.veqs(sign0_1, 0x8000, mask=mask16)
            positive_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                0 * QK_COLUMNS + 1 * 128,
                rr.vselect(negative_key0_1, positive_key0_1, cond_mask=negative0_1),
                mask16,
            )
            histogram_low_reg = rr.vselect(
                zero_hist, rr.vload(hist, 0), cond_mask=reset16
            )
            histogram_high_reg = rr.vselect(
                zero_hist, rr.vload(hist, 256), cond_mask=reset16
            )
            key_high = rr.vshr(
                rr.vselect(negative_key0_0, positive_key0_0, cond_mask=negative0_0),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid0 - 0)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            key_high = rr.vshr(
                rr.vselect(negative_key0_1, positive_key0_1, cond_mask=negative0_1),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid0 - 128)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            rr.vstore(hist, 0, histogram_low_reg, mask16)
            rr.vstore(hist, 256, histogram_high_reg, mask16)

            query1_s2_half0_acc_reg = rr.vadds(
                query1_s2_half0_acc_reg, 0.0, mask=mask16
            )
            bits1_0 = rr.vreinterpret(query1_s2_half0_acc_reg, dtypes.uint16)
            sign1_0 = rr.vbitwise_and(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_0 = rr.veqs(sign1_0, 0x8000, mask=mask16)
            positive_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                1 * QK_COLUMNS + 0 * 128,
                rr.vselect(negative_key1_0, positive_key1_0, cond_mask=negative1_0),
                mask16,
            )
            query1_s2_half1_acc_reg = rr.vadds(
                query1_s2_half1_acc_reg, 0.0, mask=mask16
            )
            bits1_1 = rr.vreinterpret(query1_s2_half1_acc_reg, dtypes.uint16)
            sign1_1 = rr.vbitwise_and(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_1 = rr.veqs(sign1_1, 0x8000, mask=mask16)
            positive_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                1 * QK_COLUMNS + 1 * 128,
                rr.vselect(negative_key1_1, positive_key1_1, cond_mask=negative1_1),
                mask16,
            )
            histogram_low_reg = rr.vselect(
                zero_hist, rr.vload(hist, 512), cond_mask=reset16
            )
            histogram_high_reg = rr.vselect(
                zero_hist, rr.vload(hist, 768), cond_mask=reset16
            )
            key_high = rr.vshr(
                rr.vselect(negative_key1_0, positive_key1_0, cond_mask=negative1_0),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid1 - 0)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            key_high = rr.vshr(
                rr.vselect(negative_key1_1, positive_key1_1, cond_mask=negative1_1),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid1 - 128)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            rr.vstore(hist, 512, histogram_low_reg, mask16)
            rr.vstore(hist, 768, histogram_high_reg, mask16)

            rr.vmem_bar("vst_vld")

    @jit
    def _score_two_cache(
        self, qk_handoff, group_count, hist, workspace_offset, valid0, valid1
    ):
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            zero_hist = rr.vdups(0, dtypes.uint16, mask=mask16)
            reset16 = rr.veqs(
                rr.vdups(
                    dtypes.int32(workspace_offset < 512), dtypes.uint16, mask=mask16
                ),
                1,
                mask=mask16,
            )
            _weight_mask = rr.update_mask(self.n1, elem_bits=32)[0]
            bf16_weight0 = rr.vload(self.packed_weight, 0 * 128)
            query0_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query0_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            bf16_weight1 = rr.vload(self.packed_weight, 1 * 128)
            query1_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query1_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            for group in dsl_range(group_count, unroll=4):
                lane = rr.vdups(group, dtypes.uint16, mask=mask16)
                w0 = rr.vgather_reg(bf16_weight0, lane)
                w1 = rr.vgather_reg(bf16_weight1, lane)
                qk0_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half0_acc_reg = rr.vmadd(
                    qk0_0, w0, query0_s2_half0_acc_reg, mask=mask16
                )
                qk0_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half1_acc_reg = rr.vmadd(
                    qk0_1, w0, query0_s2_half1_acc_reg, mask=mask16
                )
                qk1_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half0_acc_reg = rr.vmadd(
                    qk1_0, w1, query1_s2_half0_acc_reg, mask=mask16
                )
                qk1_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half1_acc_reg = rr.vmadd(
                    qk1_1, w1, query1_s2_half1_acc_reg, mask=mask16
                )
            query0_s2_half0_acc_reg = rr.vadds(
                query0_s2_half0_acc_reg, 0.0, mask=mask16
            )
            bits0_0 = rr.vreinterpret(query0_s2_half0_acc_reg, dtypes.uint16)
            sign0_0 = rr.vbitwise_and(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_0 = rr.veqs(sign0_0, 0x8000, mask=mask16)
            positive_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                self.full_key_cache,
                0 * CACHE_TOKENS + dtypes.int32(workspace_offset) + 0 * 128,
                rr.vselect(negative_key0_0, positive_key0_0, cond_mask=negative0_0),
                mask16,
            )
            query0_s2_half1_acc_reg = rr.vadds(
                query0_s2_half1_acc_reg, 0.0, mask=mask16
            )
            bits0_1 = rr.vreinterpret(query0_s2_half1_acc_reg, dtypes.uint16)
            sign0_1 = rr.vbitwise_and(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_1 = rr.veqs(sign0_1, 0x8000, mask=mask16)
            positive_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                self.full_key_cache,
                0 * CACHE_TOKENS + dtypes.int32(workspace_offset) + 1 * 128,
                rr.vselect(negative_key0_1, positive_key0_1, cond_mask=negative0_1),
                mask16,
            )
            histogram_low_reg = rr.vselect(
                zero_hist, rr.vload(hist, 0), cond_mask=reset16
            )
            histogram_high_reg = rr.vselect(
                zero_hist, rr.vload(hist, 256), cond_mask=reset16
            )
            key_high = rr.vshr(
                rr.vselect(negative_key0_0, positive_key0_0, cond_mask=negative0_0),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid0 - 0)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            key_high = rr.vshr(
                rr.vselect(negative_key0_1, positive_key0_1, cond_mask=negative0_1),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid0 - 128)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            rr.vstore(hist, 0, histogram_low_reg, mask16)
            rr.vstore(hist, 256, histogram_high_reg, mask16)

            query1_s2_half0_acc_reg = rr.vadds(
                query1_s2_half0_acc_reg, 0.0, mask=mask16
            )
            bits1_0 = rr.vreinterpret(query1_s2_half0_acc_reg, dtypes.uint16)
            sign1_0 = rr.vbitwise_and(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_0 = rr.veqs(sign1_0, 0x8000, mask=mask16)
            positive_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                self.full_key_cache,
                1 * CACHE_TOKENS + dtypes.int32(workspace_offset) + 0 * 128,
                rr.vselect(negative_key1_0, positive_key1_0, cond_mask=negative1_0),
                mask16,
            )
            query1_s2_half1_acc_reg = rr.vadds(
                query1_s2_half1_acc_reg, 0.0, mask=mask16
            )
            bits1_1 = rr.vreinterpret(query1_s2_half1_acc_reg, dtypes.uint16)
            sign1_1 = rr.vbitwise_and(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_1 = rr.veqs(sign1_1, 0x8000, mask=mask16)
            positive_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                self.full_key_cache,
                1 * CACHE_TOKENS + dtypes.int32(workspace_offset) + 1 * 128,
                rr.vselect(negative_key1_1, positive_key1_1, cond_mask=negative1_1),
                mask16,
            )
            histogram_low_reg = rr.vselect(
                zero_hist, rr.vload(hist, 512), cond_mask=reset16
            )
            histogram_high_reg = rr.vselect(
                zero_hist, rr.vload(hist, 768), cond_mask=reset16
            )
            key_high = rr.vshr(
                rr.vselect(negative_key1_0, positive_key1_0, cond_mask=negative1_0),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid1 - 0)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            key_high = rr.vshr(
                rr.vselect(negative_key1_1, positive_key1_1, cond_mask=negative1_1),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid1 - 128)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            rr.vstore(hist, 512, histogram_low_reg, mask16)
            rr.vstore(hist, 768, histogram_high_reg, mask16)

            rr.vmem_bar("vst_vld")

    @jit
    def _score_three(self, qk_handoff, group_count):
        if self.candidate_enabled:
            self._score_three_candidate(qk_handoff, group_count)
        else:
            self._score_three_plain(qk_handoff, group_count)

    @jit
    def _score_three_candidate(self, qk_handoff, group_count):
        key_write = self.key_ub.produce()
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            _weight_mask = rr.update_mask(self.n1, elem_bits=32)[0]
            bf16_weight0 = rr.vload(self.packed_weight, 0 * 128)
            query0_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query0_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            bf16_weight1 = rr.vload(self.packed_weight, 1 * 128)
            query1_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query1_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            bf16_weight2 = rr.vload(self.packed_weight, 2 * 128)
            query2_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query2_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            for group in dsl_range(group_count, unroll=4):
                lane = rr.vdups(group, dtypes.uint16, mask=mask16)
                w0 = rr.vgather_reg(bf16_weight0, lane)
                w1 = rr.vgather_reg(bf16_weight1, lane)
                w2 = rr.vgather_reg(bf16_weight2, lane)
                qk0_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half0_acc_reg = rr.vmadd(
                    qk0_0, w0, query0_s2_half0_acc_reg, mask=mask16
                )
                qk0_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half1_acc_reg = rr.vmadd(
                    qk0_1, w0, query0_s2_half1_acc_reg, mask=mask16
                )
                qk1_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half0_acc_reg = rr.vmadd(
                    qk1_0, w1, query1_s2_half0_acc_reg, mask=mask16
                )
                qk1_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half1_acc_reg = rr.vmadd(
                    qk1_1, w1, query1_s2_half1_acc_reg, mask=mask16
                )
                qk2_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (2 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query2_s2_half0_acc_reg = rr.vmadd(
                    qk2_0, w2, query2_s2_half0_acc_reg, mask=mask16
                )
                qk2_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (2 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query2_s2_half1_acc_reg = rr.vmadd(
                    qk2_1, w2, query2_s2_half1_acc_reg, mask=mask16
                )
            query0_s2_half0_acc_reg = rr.vadds(
                query0_s2_half0_acc_reg, 0.0, mask=mask16
            )
            rr.vstore(
                self.score_ub, 0 * QK_COLUMNS + 0 * 128, query0_s2_half0_acc_reg, mask16
            )
            bits0_0 = rr.vreinterpret(query0_s2_half0_acc_reg, dtypes.uint16)
            sign0_0 = rr.vbitwise_and(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_0 = rr.veqs(sign0_0, 0x8000, mask=mask16)
            positive_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                0 * QK_COLUMNS + 0 * 128,
                rr.vselect(negative_key0_0, positive_key0_0, cond_mask=negative0_0),
                mask16,
            )
            query0_s2_half1_acc_reg = rr.vadds(
                query0_s2_half1_acc_reg, 0.0, mask=mask16
            )
            rr.vstore(
                self.score_ub, 0 * QK_COLUMNS + 1 * 128, query0_s2_half1_acc_reg, mask16
            )
            bits0_1 = rr.vreinterpret(query0_s2_half1_acc_reg, dtypes.uint16)
            sign0_1 = rr.vbitwise_and(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_1 = rr.veqs(sign0_1, 0x8000, mask=mask16)
            positive_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                0 * QK_COLUMNS + 1 * 128,
                rr.vselect(negative_key0_1, positive_key0_1, cond_mask=negative0_1),
                mask16,
            )
            query1_s2_half0_acc_reg = rr.vadds(
                query1_s2_half0_acc_reg, 0.0, mask=mask16
            )
            rr.vstore(
                self.score_ub, 1 * QK_COLUMNS + 0 * 128, query1_s2_half0_acc_reg, mask16
            )
            bits1_0 = rr.vreinterpret(query1_s2_half0_acc_reg, dtypes.uint16)
            sign1_0 = rr.vbitwise_and(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_0 = rr.veqs(sign1_0, 0x8000, mask=mask16)
            positive_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                1 * QK_COLUMNS + 0 * 128,
                rr.vselect(negative_key1_0, positive_key1_0, cond_mask=negative1_0),
                mask16,
            )
            query1_s2_half1_acc_reg = rr.vadds(
                query1_s2_half1_acc_reg, 0.0, mask=mask16
            )
            rr.vstore(
                self.score_ub, 1 * QK_COLUMNS + 1 * 128, query1_s2_half1_acc_reg, mask16
            )
            bits1_1 = rr.vreinterpret(query1_s2_half1_acc_reg, dtypes.uint16)
            sign1_1 = rr.vbitwise_and(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_1 = rr.veqs(sign1_1, 0x8000, mask=mask16)
            positive_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                1 * QK_COLUMNS + 1 * 128,
                rr.vselect(negative_key1_1, positive_key1_1, cond_mask=negative1_1),
                mask16,
            )
            query2_s2_half0_acc_reg = rr.vadds(
                query2_s2_half0_acc_reg, 0.0, mask=mask16
            )
            rr.vstore(
                self.score_ub, 2 * QK_COLUMNS + 0 * 128, query2_s2_half0_acc_reg, mask16
            )
            bits2_0 = rr.vreinterpret(query2_s2_half0_acc_reg, dtypes.uint16)
            sign2_0 = rr.vbitwise_and(
                bits2_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative2_0 = rr.veqs(sign2_0, 0x8000, mask=mask16)
            positive_key2_0 = rr.vbitwise_xor(
                bits2_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key2_0 = rr.vbitwise_xor(
                bits2_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                2 * QK_COLUMNS + 0 * 128,
                rr.vselect(negative_key2_0, positive_key2_0, cond_mask=negative2_0),
                mask16,
            )
            query2_s2_half1_acc_reg = rr.vadds(
                query2_s2_half1_acc_reg, 0.0, mask=mask16
            )
            rr.vstore(
                self.score_ub, 2 * QK_COLUMNS + 1 * 128, query2_s2_half1_acc_reg, mask16
            )
            bits2_1 = rr.vreinterpret(query2_s2_half1_acc_reg, dtypes.uint16)
            sign2_1 = rr.vbitwise_and(
                bits2_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative2_1 = rr.veqs(sign2_1, 0x8000, mask=mask16)
            positive_key2_1 = rr.vbitwise_xor(
                bits2_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key2_1 = rr.vbitwise_xor(
                bits2_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                2 * QK_COLUMNS + 1 * 128,
                rr.vselect(negative_key2_1, positive_key2_1, cond_mask=negative2_1),
                mask16,
            )
            rr.vmem_bar("vst_vld")

    @jit
    def _score_three_plain(self, qk_handoff, group_count):
        key_write = self.key_ub.produce()
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            _weight_mask = rr.update_mask(self.n1, elem_bits=32)[0]
            bf16_weight0 = rr.vload(self.packed_weight, 0 * 128)
            query0_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query0_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            bf16_weight1 = rr.vload(self.packed_weight, 1 * 128)
            query1_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query1_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            bf16_weight2 = rr.vload(self.packed_weight, 2 * 128)
            query2_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query2_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            for group in dsl_range(group_count, unroll=4):
                lane = rr.vdups(group, dtypes.uint16, mask=mask16)
                w0 = rr.vgather_reg(bf16_weight0, lane)
                w1 = rr.vgather_reg(bf16_weight1, lane)
                w2 = rr.vgather_reg(bf16_weight2, lane)
                qk0_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half0_acc_reg = rr.vmadd(
                    qk0_0, w0, query0_s2_half0_acc_reg, mask=mask16
                )
                qk0_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half1_acc_reg = rr.vmadd(
                    qk0_1, w0, query0_s2_half1_acc_reg, mask=mask16
                )
                qk1_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half0_acc_reg = rr.vmadd(
                    qk1_0, w1, query1_s2_half0_acc_reg, mask=mask16
                )
                qk1_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half1_acc_reg = rr.vmadd(
                    qk1_1, w1, query1_s2_half1_acc_reg, mask=mask16
                )
                qk2_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (2 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query2_s2_half0_acc_reg = rr.vmadd(
                    qk2_0, w2, query2_s2_half0_acc_reg, mask=mask16
                )
                qk2_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (2 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query2_s2_half1_acc_reg = rr.vmadd(
                    qk2_1, w2, query2_s2_half1_acc_reg, mask=mask16
                )
            query0_s2_half0_acc_reg = rr.vadds(
                query0_s2_half0_acc_reg, 0.0, mask=mask16
            )
            bits0_0 = rr.vreinterpret(query0_s2_half0_acc_reg, dtypes.uint16)
            sign0_0 = rr.vbitwise_and(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_0 = rr.veqs(sign0_0, 0x8000, mask=mask16)
            positive_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                0 * QK_COLUMNS + 0 * 128,
                rr.vselect(negative_key0_0, positive_key0_0, cond_mask=negative0_0),
                mask16,
            )
            query0_s2_half1_acc_reg = rr.vadds(
                query0_s2_half1_acc_reg, 0.0, mask=mask16
            )
            bits0_1 = rr.vreinterpret(query0_s2_half1_acc_reg, dtypes.uint16)
            sign0_1 = rr.vbitwise_and(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_1 = rr.veqs(sign0_1, 0x8000, mask=mask16)
            positive_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                0 * QK_COLUMNS + 1 * 128,
                rr.vselect(negative_key0_1, positive_key0_1, cond_mask=negative0_1),
                mask16,
            )
            query1_s2_half0_acc_reg = rr.vadds(
                query1_s2_half0_acc_reg, 0.0, mask=mask16
            )
            bits1_0 = rr.vreinterpret(query1_s2_half0_acc_reg, dtypes.uint16)
            sign1_0 = rr.vbitwise_and(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_0 = rr.veqs(sign1_0, 0x8000, mask=mask16)
            positive_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                1 * QK_COLUMNS + 0 * 128,
                rr.vselect(negative_key1_0, positive_key1_0, cond_mask=negative1_0),
                mask16,
            )
            query1_s2_half1_acc_reg = rr.vadds(
                query1_s2_half1_acc_reg, 0.0, mask=mask16
            )
            bits1_1 = rr.vreinterpret(query1_s2_half1_acc_reg, dtypes.uint16)
            sign1_1 = rr.vbitwise_and(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_1 = rr.veqs(sign1_1, 0x8000, mask=mask16)
            positive_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                1 * QK_COLUMNS + 1 * 128,
                rr.vselect(negative_key1_1, positive_key1_1, cond_mask=negative1_1),
                mask16,
            )
            query2_s2_half0_acc_reg = rr.vadds(
                query2_s2_half0_acc_reg, 0.0, mask=mask16
            )
            bits2_0 = rr.vreinterpret(query2_s2_half0_acc_reg, dtypes.uint16)
            sign2_0 = rr.vbitwise_and(
                bits2_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative2_0 = rr.veqs(sign2_0, 0x8000, mask=mask16)
            positive_key2_0 = rr.vbitwise_xor(
                bits2_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key2_0 = rr.vbitwise_xor(
                bits2_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                2 * QK_COLUMNS + 0 * 128,
                rr.vselect(negative_key2_0, positive_key2_0, cond_mask=negative2_0),
                mask16,
            )
            query2_s2_half1_acc_reg = rr.vadds(
                query2_s2_half1_acc_reg, 0.0, mask=mask16
            )
            bits2_1 = rr.vreinterpret(query2_s2_half1_acc_reg, dtypes.uint16)
            sign2_1 = rr.vbitwise_and(
                bits2_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative2_1 = rr.veqs(sign2_1, 0x8000, mask=mask16)
            positive_key2_1 = rr.vbitwise_xor(
                bits2_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key2_1 = rr.vbitwise_xor(
                bits2_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                2 * QK_COLUMNS + 1 * 128,
                rr.vselect(negative_key2_1, positive_key2_1, cond_mask=negative2_1),
                mask16,
            )
            rr.vmem_bar("vst_vld")

    @jit
    def _score_three_hist(
        self, qk_handoff, group_count, hist, workspace_offset, valid0, valid1, valid2
    ):
        key_write = self.key_ub.produce()
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            zero_hist = rr.vdups(0, dtypes.uint16, mask=mask16)
            reset16 = rr.veqs(
                rr.vdups(
                    dtypes.int32(workspace_offset < 512), dtypes.uint16, mask=mask16
                ),
                1,
                mask=mask16,
            )
            _weight_mask = rr.update_mask(self.n1, elem_bits=32)[0]
            bf16_weight0 = rr.vload(self.packed_weight, 0 * 128)
            query0_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query0_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            bf16_weight1 = rr.vload(self.packed_weight, 1 * 128)
            query1_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query1_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            bf16_weight2 = rr.vload(self.packed_weight, 2 * 128)
            query2_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query2_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            for group in dsl_range(group_count, unroll=4):
                lane = rr.vdups(group, dtypes.uint16, mask=mask16)
                w0 = rr.vgather_reg(bf16_weight0, lane)
                w1 = rr.vgather_reg(bf16_weight1, lane)
                w2 = rr.vgather_reg(bf16_weight2, lane)
                qk0_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half0_acc_reg = rr.vmadd(
                    qk0_0, w0, query0_s2_half0_acc_reg, mask=mask16
                )
                qk0_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half1_acc_reg = rr.vmadd(
                    qk0_1, w0, query0_s2_half1_acc_reg, mask=mask16
                )
                qk1_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half0_acc_reg = rr.vmadd(
                    qk1_0, w1, query1_s2_half0_acc_reg, mask=mask16
                )
                qk1_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half1_acc_reg = rr.vmadd(
                    qk1_1, w1, query1_s2_half1_acc_reg, mask=mask16
                )
                qk2_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (2 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query2_s2_half0_acc_reg = rr.vmadd(
                    qk2_0, w2, query2_s2_half0_acc_reg, mask=mask16
                )
                qk2_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (2 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query2_s2_half1_acc_reg = rr.vmadd(
                    qk2_1, w2, query2_s2_half1_acc_reg, mask=mask16
                )
            query0_s2_half0_acc_reg = rr.vadds(
                query0_s2_half0_acc_reg, 0.0, mask=mask16
            )
            bits0_0 = rr.vreinterpret(query0_s2_half0_acc_reg, dtypes.uint16)
            sign0_0 = rr.vbitwise_and(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_0 = rr.veqs(sign0_0, 0x8000, mask=mask16)
            positive_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                0 * QK_COLUMNS + 0 * 128,
                rr.vselect(negative_key0_0, positive_key0_0, cond_mask=negative0_0),
                mask16,
            )
            query0_s2_half1_acc_reg = rr.vadds(
                query0_s2_half1_acc_reg, 0.0, mask=mask16
            )
            bits0_1 = rr.vreinterpret(query0_s2_half1_acc_reg, dtypes.uint16)
            sign0_1 = rr.vbitwise_and(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_1 = rr.veqs(sign0_1, 0x8000, mask=mask16)
            positive_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                0 * QK_COLUMNS + 1 * 128,
                rr.vselect(negative_key0_1, positive_key0_1, cond_mask=negative0_1),
                mask16,
            )
            histogram_low_reg = rr.vselect(
                zero_hist, rr.vload(hist, 0), cond_mask=reset16
            )
            histogram_high_reg = rr.vselect(
                zero_hist, rr.vload(hist, 256), cond_mask=reset16
            )
            key_high = rr.vshr(
                rr.vselect(negative_key0_0, positive_key0_0, cond_mask=negative0_0),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid0 - 0)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            key_high = rr.vshr(
                rr.vselect(negative_key0_1, positive_key0_1, cond_mask=negative0_1),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid0 - 128)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            rr.vstore(hist, 0, histogram_low_reg, mask16)
            rr.vstore(hist, 256, histogram_high_reg, mask16)

            query1_s2_half0_acc_reg = rr.vadds(
                query1_s2_half0_acc_reg, 0.0, mask=mask16
            )
            bits1_0 = rr.vreinterpret(query1_s2_half0_acc_reg, dtypes.uint16)
            sign1_0 = rr.vbitwise_and(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_0 = rr.veqs(sign1_0, 0x8000, mask=mask16)
            positive_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                1 * QK_COLUMNS + 0 * 128,
                rr.vselect(negative_key1_0, positive_key1_0, cond_mask=negative1_0),
                mask16,
            )
            query1_s2_half1_acc_reg = rr.vadds(
                query1_s2_half1_acc_reg, 0.0, mask=mask16
            )
            bits1_1 = rr.vreinterpret(query1_s2_half1_acc_reg, dtypes.uint16)
            sign1_1 = rr.vbitwise_and(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_1 = rr.veqs(sign1_1, 0x8000, mask=mask16)
            positive_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                1 * QK_COLUMNS + 1 * 128,
                rr.vselect(negative_key1_1, positive_key1_1, cond_mask=negative1_1),
                mask16,
            )
            histogram_low_reg = rr.vselect(
                zero_hist, rr.vload(hist, 512), cond_mask=reset16
            )
            histogram_high_reg = rr.vselect(
                zero_hist, rr.vload(hist, 768), cond_mask=reset16
            )
            key_high = rr.vshr(
                rr.vselect(negative_key1_0, positive_key1_0, cond_mask=negative1_0),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid1 - 0)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            key_high = rr.vshr(
                rr.vselect(negative_key1_1, positive_key1_1, cond_mask=negative1_1),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid1 - 128)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            rr.vstore(hist, 512, histogram_low_reg, mask16)
            rr.vstore(hist, 768, histogram_high_reg, mask16)

            query2_s2_half0_acc_reg = rr.vadds(
                query2_s2_half0_acc_reg, 0.0, mask=mask16
            )
            bits2_0 = rr.vreinterpret(query2_s2_half0_acc_reg, dtypes.uint16)
            sign2_0 = rr.vbitwise_and(
                bits2_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative2_0 = rr.veqs(sign2_0, 0x8000, mask=mask16)
            positive_key2_0 = rr.vbitwise_xor(
                bits2_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key2_0 = rr.vbitwise_xor(
                bits2_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                2 * QK_COLUMNS + 0 * 128,
                rr.vselect(negative_key2_0, positive_key2_0, cond_mask=negative2_0),
                mask16,
            )
            query2_s2_half1_acc_reg = rr.vadds(
                query2_s2_half1_acc_reg, 0.0, mask=mask16
            )
            bits2_1 = rr.vreinterpret(query2_s2_half1_acc_reg, dtypes.uint16)
            sign2_1 = rr.vbitwise_and(
                bits2_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative2_1 = rr.veqs(sign2_1, 0x8000, mask=mask16)
            positive_key2_1 = rr.vbitwise_xor(
                bits2_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key2_1 = rr.vbitwise_xor(
                bits2_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                key_write,
                2 * QK_COLUMNS + 1 * 128,
                rr.vselect(negative_key2_1, positive_key2_1, cond_mask=negative2_1),
                mask16,
            )
            histogram_low_reg = rr.vselect(
                zero_hist, rr.vload(hist, 1024), cond_mask=reset16
            )
            histogram_high_reg = rr.vselect(
                zero_hist, rr.vload(hist, 1280), cond_mask=reset16
            )
            key_high = rr.vshr(
                rr.vselect(negative_key2_0, positive_key2_0, cond_mask=negative2_0),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid2 - 0)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            key_high = rr.vshr(
                rr.vselect(negative_key2_1, positive_key2_1, cond_mask=negative2_1),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid2 - 128)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            rr.vstore(hist, 1024, histogram_low_reg, mask16)
            rr.vstore(hist, 1280, histogram_high_reg, mask16)

            rr.vmem_bar("vst_vld")

    @jit
    def _score_three_cache(
        self, qk_handoff, group_count, hist, workspace_offset, valid0, valid1, valid2
    ):
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            zero_hist = rr.vdups(0, dtypes.uint16, mask=mask16)
            reset16 = rr.veqs(
                rr.vdups(
                    dtypes.int32(workspace_offset < 512), dtypes.uint16, mask=mask16
                ),
                1,
                mask=mask16,
            )
            _weight_mask = rr.update_mask(self.n1, elem_bits=32)[0]
            bf16_weight0 = rr.vload(self.packed_weight, 0 * 128)
            query0_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query0_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            bf16_weight1 = rr.vload(self.packed_weight, 1 * 128)
            query1_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query1_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            bf16_weight2 = rr.vload(self.packed_weight, 2 * 128)
            query2_s2_half0_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            query2_s2_half1_acc_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            for group in dsl_range(group_count, unroll=4):
                lane = rr.vdups(group, dtypes.uint16, mask=mask16)
                w0 = rr.vgather_reg(bf16_weight0, lane)
                w1 = rr.vgather_reg(bf16_weight1, lane)
                w2 = rr.vgather_reg(bf16_weight2, lane)
                qk0_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half0_acc_reg = rr.vmadd(
                    qk0_0, w0, query0_s2_half0_acc_reg, mask=mask16
                )
                qk0_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (0 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query0_s2_half1_acc_reg = rr.vmadd(
                    qk0_1, w0, query0_s2_half1_acc_reg, mask=mask16
                )
                qk1_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half0_acc_reg = rr.vmadd(
                    qk1_0, w1, query1_s2_half0_acc_reg, mask=mask16
                )
                qk1_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (1 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query1_s2_half1_acc_reg = rr.vmadd(
                    qk1_1, w1, query1_s2_half1_acc_reg, mask=mask16
                )
                qk2_0 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (2 * self.n1 + group) * 256
                        + 0 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query2_s2_half0_acc_reg = rr.vmadd(
                    qk2_0, w2, query2_s2_half0_acc_reg, mask=mask16
                )
                qk2_1 = rr.vmaxs(
                    rr.vload(
                        qk_handoff,
                        (2 * self.n1 + group) * 256
                        + 1 * (self.n1 * self.vec_rows) * 256,
                    ),
                    0.0,
                    mask=mask16,
                )
                query2_s2_half1_acc_reg = rr.vmadd(
                    qk2_1, w2, query2_s2_half1_acc_reg, mask=mask16
                )
            query0_s2_half0_acc_reg = rr.vadds(
                query0_s2_half0_acc_reg, 0.0, mask=mask16
            )
            bits0_0 = rr.vreinterpret(query0_s2_half0_acc_reg, dtypes.uint16)
            sign0_0 = rr.vbitwise_and(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_0 = rr.veqs(sign0_0, 0x8000, mask=mask16)
            positive_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_0 = rr.vbitwise_xor(
                bits0_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                self.full_key_cache,
                0 * CACHE_TOKENS + dtypes.int32(workspace_offset) + 0 * 128,
                rr.vselect(negative_key0_0, positive_key0_0, cond_mask=negative0_0),
                mask16,
            )
            query0_s2_half1_acc_reg = rr.vadds(
                query0_s2_half1_acc_reg, 0.0, mask=mask16
            )
            bits0_1 = rr.vreinterpret(query0_s2_half1_acc_reg, dtypes.uint16)
            sign0_1 = rr.vbitwise_and(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative0_1 = rr.veqs(sign0_1, 0x8000, mask=mask16)
            positive_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key0_1 = rr.vbitwise_xor(
                bits0_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                self.full_key_cache,
                0 * CACHE_TOKENS + dtypes.int32(workspace_offset) + 1 * 128,
                rr.vselect(negative_key0_1, positive_key0_1, cond_mask=negative0_1),
                mask16,
            )
            histogram_low_reg = rr.vselect(
                zero_hist, rr.vload(hist, 0), cond_mask=reset16
            )
            histogram_high_reg = rr.vselect(
                zero_hist, rr.vload(hist, 256), cond_mask=reset16
            )
            key_high = rr.vshr(
                rr.vselect(negative_key0_0, positive_key0_0, cond_mask=negative0_0),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid0 - 0)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            key_high = rr.vshr(
                rr.vselect(negative_key0_1, positive_key0_1, cond_mask=negative0_1),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid0 - 128)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            rr.vstore(hist, 0, histogram_low_reg, mask16)
            rr.vstore(hist, 256, histogram_high_reg, mask16)

            query1_s2_half0_acc_reg = rr.vadds(
                query1_s2_half0_acc_reg, 0.0, mask=mask16
            )
            bits1_0 = rr.vreinterpret(query1_s2_half0_acc_reg, dtypes.uint16)
            sign1_0 = rr.vbitwise_and(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_0 = rr.veqs(sign1_0, 0x8000, mask=mask16)
            positive_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_0 = rr.vbitwise_xor(
                bits1_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                self.full_key_cache,
                1 * CACHE_TOKENS + dtypes.int32(workspace_offset) + 0 * 128,
                rr.vselect(negative_key1_0, positive_key1_0, cond_mask=negative1_0),
                mask16,
            )
            query1_s2_half1_acc_reg = rr.vadds(
                query1_s2_half1_acc_reg, 0.0, mask=mask16
            )
            bits1_1 = rr.vreinterpret(query1_s2_half1_acc_reg, dtypes.uint16)
            sign1_1 = rr.vbitwise_and(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative1_1 = rr.veqs(sign1_1, 0x8000, mask=mask16)
            positive_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key1_1 = rr.vbitwise_xor(
                bits1_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                self.full_key_cache,
                1 * CACHE_TOKENS + dtypes.int32(workspace_offset) + 1 * 128,
                rr.vselect(negative_key1_1, positive_key1_1, cond_mask=negative1_1),
                mask16,
            )
            histogram_low_reg = rr.vselect(
                zero_hist, rr.vload(hist, 512), cond_mask=reset16
            )
            histogram_high_reg = rr.vselect(
                zero_hist, rr.vload(hist, 768), cond_mask=reset16
            )
            key_high = rr.vshr(
                rr.vselect(negative_key1_0, positive_key1_0, cond_mask=negative1_0),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid1 - 0)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            key_high = rr.vshr(
                rr.vselect(negative_key1_1, positive_key1_1, cond_mask=negative1_1),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid1 - 128)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            rr.vstore(hist, 512, histogram_low_reg, mask16)
            rr.vstore(hist, 768, histogram_high_reg, mask16)

            query2_s2_half0_acc_reg = rr.vadds(
                query2_s2_half0_acc_reg, 0.0, mask=mask16
            )
            bits2_0 = rr.vreinterpret(query2_s2_half0_acc_reg, dtypes.uint16)
            sign2_0 = rr.vbitwise_and(
                bits2_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative2_0 = rr.veqs(sign2_0, 0x8000, mask=mask16)
            positive_key2_0 = rr.vbitwise_xor(
                bits2_0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key2_0 = rr.vbitwise_xor(
                bits2_0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                self.full_key_cache,
                2 * CACHE_TOKENS + dtypes.int32(workspace_offset) + 0 * 128,
                rr.vselect(negative_key2_0, positive_key2_0, cond_mask=negative2_0),
                mask16,
            )
            query2_s2_half1_acc_reg = rr.vadds(
                query2_s2_half1_acc_reg, 0.0, mask=mask16
            )
            bits2_1 = rr.vreinterpret(query2_s2_half1_acc_reg, dtypes.uint16)
            sign2_1 = rr.vbitwise_and(
                bits2_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative2_1 = rr.veqs(sign2_1, 0x8000, mask=mask16)
            positive_key2_1 = rr.vbitwise_xor(
                bits2_1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
            )
            negative_key2_1 = rr.vbitwise_xor(
                bits2_1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
            )
            rr.vstore(
                self.full_key_cache,
                2 * CACHE_TOKENS + dtypes.int32(workspace_offset) + 1 * 128,
                rr.vselect(negative_key2_1, positive_key2_1, cond_mask=negative2_1),
                mask16,
            )
            histogram_low_reg = rr.vselect(
                zero_hist, rr.vload(hist, 1024), cond_mask=reset16
            )
            histogram_high_reg = rr.vselect(
                zero_hist, rr.vload(hist, 1280), cond_mask=reset16
            )
            key_high = rr.vshr(
                rr.vselect(negative_key2_0, positive_key2_0, cond_mask=negative2_0),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid2 - 0)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            key_high = rr.vshr(
                rr.vselect(negative_key2_1, positive_key2_1, cond_mask=negative2_1),
                8,
                mask=mask16,
            )
            packed_high = rr.vpack(key_high, dtypes.uint8, part="lower")
            hist_mask = rr.update_mask(max(0, min(128, valid2 - 128)), elem_bits=8)[0]
            histogram_low_reg = rr.vhistogram_accumulate(
                histogram_low_reg, packed_high, mask=hist_mask, bin=0
            )
            histogram_high_reg = rr.vhistogram_accumulate(
                histogram_high_reg, packed_high, mask=hist_mask, bin=1
            )
            rr.vstore(hist, 1024, histogram_low_reg, mask16)
            rr.vstore(hist, 1280, histogram_high_reg, mask16)

            rr.vmem_bar("vst_vld")


class QliFusedKernel:
    def __init__(
        self,
        topk,
        candidate_enabled,
        candidate_topk,
        mask_mode,
        cmp_ratio,
        has_cu_seqlens_q,
        has_seqused_q,
        has_seqused_k,
        has_block_table,
        pa_block_size,
        n1,
        query_rows,
        metadata,
        output_offset_address,
        return_values,
    ):
        self.ub_pool = _UbPool()
        self.output_offset_address = output_offset_address
        self.metadata = metadata
        self.need_topk_values = _need_topk_values(metadata, return_values)
        self.n1 = int(n1)
        self.query_rows = int(query_rows)
        self.vec_rows = int(query_rows) // 2
        self.tile_m = int(n1) * int(query_rows)
        self.topk_count = int(topk)
        self.cache_enabled = (
            not candidate_enabled
            and self.topk_count == CACHED_TOPK_COUNT
            and self.tile_m <= 192
        )
        self.candidate_enabled = bool(candidate_enabled)
        self.candidate_topk_count = int(candidate_topk)
        self.mask_mode = int(mask_mode)
        self.cmp_ratio = int(cmp_ratio)
        self.has_cu_seqlens_q = bool(has_cu_seqlens_q)
        self.has_seqused_q = bool(has_seqused_q)
        self.has_seqused_k = bool(has_seqused_k)
        self.has_cmp_residual_k = self.mask_mode == 3 and self.cmp_ratio != 1
        self.has_block_table = bool(has_block_table)
        self.qk_slot0 = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.tile_m, 128),
            dtypes.bfloat16,
            stride=(256, 1),
            addr=0,
        )
        self.qk_slot1 = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.tile_m, 128),
            dtypes.bfloat16,
            stride=(256, 1),
            addr=256,
        )
        self.cube = QliCube(pa_block_size, n1, query_rows)
        self.topk_workspace = QliRawTopKWorkspace(
            TOPK_TRUNK_LEN if self.cache_enabled else STREAM_TOPK_TRUNK_LEN,
            max(
                self.topk_count,
                self.candidate_topk_count if self.candidate_enabled else 1,
            ),
            pipelined=self.cache_enabled,
        )
        self.topk = QliRawTopKSelector(self.topk_count, self.topk_workspace)
        if self.candidate_enabled:
            self.candidate_topk = QliRawTopKSelector(
                self.candidate_topk_count,
                self.topk_workspace,
            )

        self.vector = QliVector(
            get_subblock_id(),
            self.mask_mode,
            self.cmp_ratio,
            self.candidate_enabled,
            self.candidate_topk_count,
            n1,
            query_rows,
            CACHED_VECTOR_UB_OFFSET
            if self.cache_enabled
            else self.tile_m * QK_ROW_BYTES + VECTOR_AFTER_QK_BYTES,
            self.cache_enabled,
        )
        # Dedicated 512B-strided bank-separated histogram slots, main phase only.
        self.hist_slot0 = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.vec_rows * 2, 128),
            dtypes.uint16,
            stride=(256, 1),
            addr=CACHED_HISTOGRAM_UB_OFFSET
            if self.cache_enabled
            else self.tile_m * QK_ROW_BYTES + HISTOGRAM_AFTER_QK_BYTES,
        )
        self.hist_slot1 = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.vec_rows * 2, 128),
            dtypes.uint16,
            stride=(256, 1),
            addr=CACHED_HISTOGRAM_UB_OFFSET + HISTOGRAM_SLOT_OFFSET
            if self.cache_enabled
            else self.tile_m * QK_ROW_BYTES
            + HISTOGRAM_AFTER_QK_BYTES
            + HISTOGRAM_SLOT_OFFSET,
        )
        self.topk.hist_slot0 = self.hist_slot0
        self.topk.hist_slot1 = self.hist_slot1
        # Main phase: QK [0,98304), cached keys [98304,251904),
        # histogram [251904,254976), weights and fallback key stages above255488.
        # Cached TopK phase: temporary/output buffers reuse the drained QK area.
        if self.cache_enabled:
            self.full_key_cache = _ub_buffer(
                self.ub_pool,
                MemLoc.UB,
                (self.vec_rows, CACHE_TOKENS),
                dtypes.uint16,
                addr=CACHED_QK_UB_BYTES,
            )
            self.vector.full_key_cache = self.full_key_cache
            self.topk.cache_tmp = _ub_buffer(
                self.ub_pool,
                MemLoc.UB,
                (CACHE_TOKENS,),
                dtypes.uint16,
                addr=0,
            )
            self.topk.cache_output_idx_stage = _ub_channel(
                self.ub_pool,
                MemLoc.UB,
                (1, self.topk_count),
                dtypes.int32,
                depth=1,
                addr=CACHED_TOPK_INDEX_OFFSET,
            )
            self.topk.cache_output_value_bits_stage = _ub_channel(
                self.ub_pool,
                MemLoc.UB,
                (1, self.topk_count),
                dtypes.uint16,
                depth=1,
                addr=CACHED_TOPK_VALUE_OFFSET,
            )

    @jit
    def _combine_hist(self, row):
        with vf(mode="simd"):
            mask = rr.update_mask(128, elem_bits=16)[0]
            histogram_low_reg = rr.vadd(
                rr.vload(self.hist_slot0, row * 512),
                rr.vload(self.hist_slot1, row * 512),
                mask=mask,
            )
            histogram_high_reg = rr.vadd(
                rr.vload(self.hist_slot0, row * 512 + 256),
                rr.vload(self.hist_slot1, row * 512 + 256),
                mask=mask,
            )
            rr.vstore(self.topk_workspace.histogram, 0, histogram_low_reg, mask)
            rr.vstore(self.topk_workspace.histogram, 128, histogram_high_reg, mask)
            rr.vmem_bar("vst_vld")

    @jit
    def _compute_nonempty_task(
        self,
        query: Tensor,
        key: Tensor,
        weights: Tensor,
        query_scale: Tensor,
        key_scale: Tensor,
        key_stride0,
        key_dequant_scale_stride0,
        block_table: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices: Tensor,
        sparse_value_bits: Tensor,
        candidate_indices: Tensor,
        candidate_value_bits: Tensor,
        candidate_length: Tensor,
        batch_idx,
        public_row_base,
        query_offset,
        task_span_count,
        task_active_count,
        used_q,
        actual_s2,
        cmp_residual,
        workspace_task_idx,
        tokens,
        tile_begin,
        tile_end,
        row_ld,
    ):
        # Previous TopK output DMA may alias the next record's score cache/QK.
        vec_sync_all()
        self.cube.load_query(query, query_scale, public_row_base)
        self.vector.load_weights(weights, public_row_base, task_span_count)
        # The streaming-histogram slots are read-modify-write accumulators;
        # zero them at record start so the first tile never reads stale UB
        # (the verifier rejects uninitialized reads).
        with vf(mode="simd"):
            zero_mask16 = rr.update_mask(128, elem_bits=16)[0]
            zero_hist16 = rr.vdups(0, dtypes.uint16, mask=zero_mask16)
            for hist_row in range_constexpr(self.vector.vec_rows * 2):
                rr.vstore(self.hist_slot0, hist_row * 256, zero_hist16, zero_mask16)
                rr.vstore(self.hist_slot1, hist_row * 256, zero_hist16, zero_mask16)
            rr.vmem_bar("vst_vld")
        tokens = (tile_end - tile_begin) * TILE_N
        stream_hist = dtypes.int32(0)
        if not self.candidate_enabled:
            stream_hist = dtypes.int32(
                tokens >= self.topk_count
                and tokens <= self.topk_workspace.max_trunk_len
            )
        cache_min_valid = actual_s2
        if self.mask_mode == 3:
            cache_min_valid = max(
                0,
                min(
                    actual_s2,
                    (
                        actual_s2 * self.cmp_ratio
                        + cmp_residual
                        - used_q
                        + query_offset
                        + 1
                    )
                    // self.cmp_ratio,
                ),
            )
        cache_task = dtypes.int32(0)
        if self.cache_enabled:
            cache_task = dtypes.int32(
                self.topk_count == CACHED_TOPK_COUNT
                and tokens >= self.topk_count
                and tokens <= CACHE_TOKENS
                and task_active_count == self.query_rows
                and task_span_count == self.query_rows
                and self.need_topk_values != 0
                and dtypes.int64(cache_min_valid) - tile_begin * TILE_N
                >= self.topk_count
            )
        _token_tiles = tokens // TILE_N
        candidate_tokens = tokens // CANDIDATE_BLOCK_SIZE
        vec_sync_intra_arrive(PIPE.V, 8)
        vec_sync_intra_arrive(PIPE.V, 9)
        for token_tile in dsl_range(tile_begin, tile_end, unroll=1):
            if (token_tile - tile_begin) % 2 == 0:
                logical_start = token_tile * TILE_N
                cube_sync_intra_wait(PIPE.FIXPIPE, 8)
                cube_sync_intra_wait(PIPE.FIXPIPE, 24)
                qk_write = self.qk_slot0
                self.cube.compute_qk(
                    key,
                    key_scale,
                    block_table,
                    batch_idx,
                    actual_s2,
                    key_stride0,
                    key_dequant_scale_stride0,
                    self.has_block_table,
                    token_tile,
                    qk_write,
                )
                cube_sync_intra_arrive(PIPE.FIXPIPE, 8)
                cube_sync_intra_arrive(PIPE.FIXPIPE, 24)
                vec_sync_intra_wait(PIPE.V, 8)
                qk_read = self.qk_slot0
                self.vector.compute_vector1(
                    qk_read,
                    weights,
                    key_out,
                    candidate_key_out,
                    public_row_base,
                    task_span_count,
                    task_active_count,
                    workspace_task_idx,
                    logical_start,
                    actual_s2,
                    query_offset,
                    used_q,
                    cmp_residual,
                    candidate_tokens,
                    candidate_length,
                    logical_start - tile_begin * TILE_N,
                    min(QK_COLUMNS, (tile_end - token_tile) * TILE_N),
                    stream_hist,
                    self.hist_slot0,
                    cache_task,
                )
                vec_sync_intra_arrive(PIPE.V, 8)
            else:
                logical_start = token_tile * TILE_N
                cube_sync_intra_wait(PIPE.FIXPIPE, 9)
                cube_sync_intra_wait(PIPE.FIXPIPE, 25)
                qk_write = self.qk_slot1
                self.cube.compute_qk(
                    key,
                    key_scale,
                    block_table,
                    batch_idx,
                    actual_s2,
                    key_stride0,
                    key_dequant_scale_stride0,
                    self.has_block_table,
                    token_tile,
                    qk_write,
                )
                cube_sync_intra_arrive(PIPE.FIXPIPE, 9)
                cube_sync_intra_arrive(PIPE.FIXPIPE, 25)
                vec_sync_intra_wait(PIPE.V, 9)
                qk_read = self.qk_slot1
                self.vector.compute_vector1(
                    qk_read,
                    weights,
                    key_out,
                    candidate_key_out,
                    public_row_base,
                    task_span_count,
                    task_active_count,
                    workspace_task_idx,
                    logical_start,
                    actual_s2,
                    query_offset,
                    used_q,
                    cmp_residual,
                    candidate_tokens,
                    candidate_length,
                    logical_start - tile_begin * TILE_N,
                    min(QK_COLUMNS, (tile_end - token_tile) * TILE_N),
                    stream_hist,
                    self.hist_slot1,
                    cache_task,
                )
                vec_sync_intra_arrive(PIPE.V, 9)
        cube_sync_intra_wait(PIPE.FIXPIPE, 8)
        cube_sync_intra_wait(PIPE.FIXPIPE, 24)
        cube_sync_intra_wait(PIPE.FIXPIPE, 9)
        cube_sync_intra_wait(PIPE.FIXPIPE, 25)
        # Order GM score writes before TopK reloads, as in ASC ProcessTopK.
        vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 0)
        vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 0)
        # Reuse the verified ASC vf_topk_16 implementation through a genuine
        # JIT boundary.  It emits into this top-level kernel and does not
        # launch a second kernel.
        for row_in_subblock in dsl_range(self.query_rows // 2, unroll=1):
            local_row = dtypes.int32(
                get_subblock_id() * (self.query_rows // 2) + row_in_subblock
            )
            public_row = dtypes.int64(public_row_base) + dtypes.int64(local_row)
            workspace_row = (
                workspace_task_idx * self.query_rows
                + get_subblock_id() * (self.query_rows // 2)
                + row_in_subblock
            )
            valid_s2 = actual_s2
            if self.mask_mode == 3:
                query_in_batch = (
                    query_offset
                    + dtypes.int32(get_subblock_id()) * (self.query_rows // 2)
                    + dtypes.int32(row_in_subblock)
                )
                valid_s2 = (
                    actual_s2 * self.cmp_ratio
                    + cmp_residual
                    - used_q
                    + query_in_batch
                    + 1
                ) // self.cmp_ratio
                if valid_s2 < 0:
                    valid_s2 = 0
                if valid_s2 > actual_s2:
                    valid_s2 = actual_s2
            if local_row >= task_active_count:
                valid_s2 = 0
            valid_s2 = valid_s2 - dtypes.int32(tile_begin * TILE_N)
            if valid_s2 < 0:
                valid_s2 = dtypes.int32(0)
            if valid_s2 > dtypes.int32(tokens):
                valid_s2 = dtypes.int32(tokens)
            if local_row < task_span_count:
                output_offset = dtypes.int32(0)
                if self.output_offset_address != 0 and row_ld == 0:
                    offsets = make_tensor(
                        make_pointer(
                            dtypes.int32,
                            dtypes.int64(self.output_offset_address),
                            MemLoc.GM,
                        ),
                        make_layout((weights.shape[0], 1), stride=(1, 1)),
                    )
                    output_offset = offsets[public_row, 0]
                if self.cache_enabled and cache_task:
                    self.topk._select_row_cached_ub(
                        tile_slice(
                            self.full_key_cache, (1, CACHE_TOKENS), (row_in_subblock, 0)
                        ),
                        dtypes.int64(valid_s2),
                        valid_s2,
                        tile_slice(
                            sparse_indices,
                            (1, self.topk_count),
                            (public_row, 0),
                        ),
                        tile_slice(
                            sparse_value_bits,
                            (1, self.topk_count),
                            (public_row, 0),
                        ),
                        dtypes.int32(tile_begin * TILE_N) + output_offset,
                        1,
                        row_in_subblock,
                    )
                else:
                    self.topk.select_row(
                        _offset_view(key_out, (workspace_row, 0)),
                        dtypes.int64(valid_s2),
                        valid_s2,
                        tile_slice(
                            sparse_indices,
                            (1, self.topk_count),
                            (public_row, 0),
                        ),
                        tile_slice(
                            sparse_value_bits,
                            (1, self.topk_count),
                            (public_row, 0),
                        ),
                        dtypes.int32(tile_begin * TILE_N) + output_offset,
                        self.need_topk_values,
                        stream_hist,
                        row_in_subblock,
                    )
                if self.candidate_enabled:
                    valid_candidate_count = (
                        valid_s2 + CANDIDATE_BLOCK_SIZE - 1
                    ) // CANDIDATE_BLOCK_SIZE
                    self.candidate_topk.select_row(
                        _offset_view(candidate_key_out, (workspace_row, 0)),
                        candidate_tokens,
                        valid_candidate_count,
                        tile_slice(
                            candidate_indices,
                            (1, self.candidate_topk_count),
                            (public_row, 0),
                        ),
                        tile_slice(
                            candidate_value_bits,
                            (1, self.candidate_topk_count),
                            (public_row, 0),
                        ),
                        dtypes.int32(tile_begin * TILE_N // CANDIDATE_BLOCK_SIZE),
                    )

    @jit
    def _compute_task(
        self,
        query: Tensor,
        key: Tensor,
        weights: Tensor,
        query_scale: Tensor,
        key_scale: Tensor,
        key_stride0,
        key_dequant_scale_stride0,
        block_table: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices: Tensor,
        sparse_value_bits: Tensor,
        candidate_indices: Tensor,
        candidate_value_bits: Tensor,
        candidate_length: Tensor,
        batch_idx,
        public_row_base,
        query_offset,
        task_span_count,
        task_active_count,
        used_q,
        actual_s2,
        cmp_residual,
        workspace_task_idx,
        tokens,
        tile_begin,
        tile_end,
        task_valid_s2,
        row_ld,
    ):
        if task_valid_s2 > dtypes.int32(tile_begin * TILE_N) and tile_end > tile_begin:
            self._compute_nonempty_task(
                query,
                key,
                weights,
                query_scale,
                key_scale,
                key_stride0,
                key_dequant_scale_stride0,
                block_table,
                key_out,
                candidate_key_out,
                sparse_indices,
                sparse_value_bits,
                candidate_indices,
                candidate_value_bits,
                candidate_length,
                batch_idx,
                public_row_base,
                query_offset,
                task_span_count,
                task_active_count,
                used_q,
                actual_s2,
                cmp_residual,
                workspace_task_idx,
                tokens,
                tile_begin,
                tile_end,
                row_ld,
            )
        else:
            for row_in_subblock in range_constexpr(self.query_rows // 2):
                local_row = dtypes.int32(
                    get_subblock_id() * (self.query_rows // 2) + row_in_subblock
                )
                public_row = dtypes.int64(public_row_base) + dtypes.int64(local_row)
                _workspace_row = (
                    workspace_task_idx * self.query_rows
                    + get_subblock_id() * (self.query_rows // 2)
                    + row_in_subblock
                )
                if local_row < task_span_count:
                    self.topk.store_empty_row(
                        tile_slice(
                            sparse_indices,
                            (1, self.topk_count),
                            (public_row, 0),
                        ),
                        tile_slice(
                            sparse_value_bits,
                            (1, self.topk_count),
                            (public_row, 0),
                        ),
                    )
                    if self.candidate_enabled:
                        self.candidate_topk.store_empty_row(
                            tile_slice(
                                candidate_indices,
                                (1, self.candidate_topk_count),
                                (public_row, 0),
                            ),
                            tile_slice(
                                candidate_value_bits,
                                (1, self.candidate_topk_count),
                                (public_row, 0),
                            ),
                        )
            if self.candidate_enabled:
                self.vector.store_empty_candidate_lengths(
                    candidate_length, public_row_base, task_span_count
                )

    @jit
    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        weights: Tensor,
        query_scale: Tensor,
        key_scale: Tensor,
        key_stride0,
        key_dequant_scale_stride0,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        block_table: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices: Tensor,
        sparse_value_bits: Tensor,
        candidate_indices: Tensor,
        candidate_value_bits: Tensor,
        candidate_length: Tensor,
        worker_count,
        total_q,
        batch_count,
        tokens,
        max_tasks_per_batch,
        metadata: Tensor,
        splits,
        task_count,
        partial_idx_address,
        partial_bits_address,
        partial_candidate_address,
        partial_candidate_bits_address,
        final_idx_address,
        final_bits_address,
        final_candidate_address,
        final_candidate_bits_address,
    ):
        worker = dtypes.int64(get_block_idx())
        workspace_cursor = dtypes.int64(metadata[worker, 7])
        if metadata[worker, 0] != 0:
            first_b = dtypes.int64(metadata[worker, 1])
            last_b = dtypes.int64(metadata[worker, 4])
            for batch_idx in dsl_range(first_b, min(last_b + 1, batch_count), unroll=1):
                query_begin = dtypes.int64(0)
                query_end = dtypes.int64(total_q)
                if self.has_cu_seqlens_q:
                    query_begin = dtypes.int64(cu_seqlens_q[batch_idx])
                    query_end = dtypes.int64(cu_seqlens_q[batch_idx + 1])
                used_q = dtypes.int32(query_end - query_begin)
                if self.has_seqused_q:
                    used_q = seqused_q[batch_idx]
                actual_s2 = dtypes.int32(tokens)
                if self.has_seqused_k:
                    actual_s2 = seqused_k[batch_idx]
                remainder = dtypes.int32(0)
                if self.has_cmp_residual_k:
                    remainder = cmp_residual_k[batch_idx]
                first_m = dtypes.int64(0)
                last_m = (
                    query_end - query_begin + self.query_rows - 1
                ) // self.query_rows
                if batch_idx == first_b:
                    first_m = dtypes.int64(metadata[worker, 2])
                if batch_idx == last_b:
                    last_m = dtypes.int64(metadata[worker, 5]) + dtypes.int64(
                        (1 if metadata[worker, 6] > 0 else 0)
                    )
                for m_idx in dsl_range(first_m, last_m, unroll=1):
                    query_offset = dtypes.int32(m_idx * self.query_rows)
                    public_row = dtypes.int32(query_begin + dtypes.int64(query_offset))
                    span = dtypes.int32(
                        min(self.query_rows, query_end - dtypes.int64(public_row))
                    )
                    active = min(span, max(dtypes.int32(0), used_q - query_offset))
                    visible = actual_s2
                    if self.mask_mode == 3:
                        visible = min(
                            actual_s2,
                            max(
                                dtypes.int32(0),
                                (
                                    actual_s2 * self.cmp_ratio
                                    + remainder
                                    - used_q
                                    + query_offset
                                    + active
                                )
                                // self.cmp_ratio,
                            ),
                        )
                    if active == 0:
                        visible = dtypes.int32(0)
                    tile_begin = dtypes.int64(0)
                    tile_end = (dtypes.int64(visible) + TILE_N - 1) // TILE_N
                    if batch_idx == first_b and m_idx == first_m:
                        tile_begin = dtypes.int64(metadata[worker, 3])
                    if batch_idx == last_b and m_idx == dtypes.int64(
                        metadata[worker, 5]
                    ):
                        tile_end = min(tile_end, dtypes.int64(metadata[worker, 6]))
                    row_ld = (
                        tile_begin > 0
                        or tile_end < (dtypes.int64(visible) + TILE_N - 1) // TILE_N
                    )
                    delta = workspace_cursor * self.query_rows - dtypes.int64(
                        public_row
                    )
                    idx_address = dtypes.int64(
                        (
                            dtypes.int64(partial_idx_address)
                            + delta * self.topk_count * 4
                            if row_ld
                            else final_idx_address
                        )
                    )
                    bits_address = dtypes.int64(
                        (
                            dtypes.int64(partial_bits_address)
                            + delta * self.topk_count * 2
                            if row_ld
                            else final_bits_address
                        )
                    )
                    candidate_address = dtypes.int64(
                        (
                            dtypes.int64(partial_candidate_address)
                            + delta * self.candidate_topk_count * 4
                            if row_ld
                            else final_candidate_address
                        )
                    )
                    candidate_bits_address = dtypes.int64(
                        (
                            dtypes.int64(partial_candidate_bits_address)
                            + delta * self.candidate_topk_count * 2
                            if row_ld
                            else final_candidate_bits_address
                        )
                    )
                    row_indices = make_tensor(
                        make_pointer(dtypes.int32, idx_address, MemLoc.GM),
                        make_layout(
                            (total_q, self.topk_count), stride=(self.topk_count, 1)
                        ),
                    )
                    row_bits = make_tensor(
                        make_pointer(dtypes.uint16, bits_address, MemLoc.GM),
                        make_layout(
                            (total_q, self.topk_count), stride=(self.topk_count, 1)
                        ),
                    )
                    row_candidates = make_tensor(
                        make_pointer(dtypes.int32, candidate_address, MemLoc.GM),
                        make_layout(
                            (total_q, self.candidate_topk_count),
                            stride=(self.candidate_topk_count, 1),
                        ),
                    )
                    row_candidate_bits = make_tensor(
                        make_pointer(dtypes.uint16, candidate_bits_address, MemLoc.GM),
                        make_layout(
                            (total_q, self.candidate_topk_count),
                            stride=(self.candidate_topk_count, 1),
                        ),
                    )
                    self._compute_task(
                        query,
                        key,
                        weights,
                        query_scale,
                        key_scale,
                        key_stride0,
                        key_dequant_scale_stride0,
                        block_table,
                        key_out,
                        candidate_key_out,
                        row_indices,
                        row_bits,
                        row_candidates,
                        row_candidate_bits,
                        candidate_length,
                        batch_idx,
                        public_row,
                        query_offset,
                        span,
                        active,
                        used_q,
                        actual_s2,
                        remainder,
                        worker,
                        tokens,
                        tile_begin,
                        tile_end,
                        visible,
                        row_ld,
                    )
                    # FD tasks cover active rows only. The first partial owner
                    # initializes inactive public rows directly in the final output.
                    if row_ld and tile_begin == 0:
                        final_indices = make_tensor(
                            make_pointer(
                                dtypes.int32, dtypes.int64(final_idx_address), MemLoc.GM
                            ),
                            make_layout(
                                (total_q, self.topk_count), stride=(self.topk_count, 1)
                            ),
                        )
                        final_bits = make_tensor(
                            make_pointer(
                                dtypes.uint16,
                                dtypes.int64(final_bits_address),
                                MemLoc.GM,
                            ),
                            make_layout(
                                (total_q, self.topk_count), stride=(self.topk_count, 1)
                            ),
                        )
                        final_candidates = make_tensor(
                            make_pointer(
                                dtypes.int32,
                                dtypes.int64(final_candidate_address),
                                MemLoc.GM,
                            ),
                            make_layout(
                                (total_q, self.candidate_topk_count),
                                stride=(self.candidate_topk_count, 1),
                            ),
                        )
                        final_candidate_bits = make_tensor(
                            make_pointer(
                                dtypes.uint16,
                                dtypes.int64(final_candidate_bits_address),
                                MemLoc.GM,
                            ),
                            make_layout(
                                (total_q, self.candidate_topk_count),
                                stride=(self.candidate_topk_count, 1),
                            ),
                        )
                        for local in range_constexpr(self.query_rows // 2):
                            r = dtypes.int32(
                                get_subblock_id() * (self.query_rows // 2) + local
                            )
                            if r >= active and r < span:
                                self.topk.store_empty_row(
                                    tile_slice(
                                        final_indices,
                                        (1, self.topk_count),
                                        (dtypes.int64(public_row) + dtypes.int64(r), 0),
                                    ),
                                    tile_slice(
                                        final_bits,
                                        (1, self.topk_count),
                                        (dtypes.int64(public_row) + dtypes.int64(r), 0),
                                    ),
                                )
                                if self.candidate_enabled:
                                    self.candidate_topk.store_empty_row(
                                        tile_slice(
                                            final_candidates,
                                            (1, self.candidate_topk_count),
                                            (
                                                dtypes.int64(public_row)
                                                + dtypes.int64(r),
                                                0,
                                            ),
                                        ),
                                        tile_slice(
                                            final_candidate_bits,
                                            (1, self.candidate_topk_count),
                                            (
                                                dtypes.int64(public_row)
                                                + dtypes.int64(r),
                                                0,
                                            ),
                                        ),
                                    )
                    if row_ld:
                        workspace_cursor += 1


_COMPILED_KERNEL: dict[tuple[object, ...], object] = {}
_COMPILED_KERNEL_LOCK = threading.Lock()


def _build_compiled_fused_runner(
    topk,
    candidate_enabled,
    candidate_topk,
    mask_mode,
    cmp_ratio,
    has_cu_seqlens_q,
    has_seqused_q,
    has_seqused_k,
    has_block_table,
    pa_block_size,
    split_count,
    n1=32,
    query_tile_rows=6,
):
    """Compile the dynamic TensorSpec contract using the standard DSL entry."""
    query_head_count = int(n1)
    query_tile_m = query_head_count * int(query_tile_rows)
    has_cmp_residual_k = int(mask_mode) == 3 and int(cmp_ratio) != 1
    topk = int(topk)
    candidate_enabled = bool(candidate_enabled)
    candidate_topk = int(candidate_topk)
    mask_mode = int(mask_mode)
    cmp_ratio = int(cmp_ratio)

    @kernel
    def QuantLightningIndexer(
        query_address,
        key_address,
        weights: Tensor,
        query_scale,
        key_scale_address,
        key_stride0,
        key_dequant_scale_stride0,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        block_table: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices_address,
        sparse_value_bits_address,
        candidate_indices_address,
        candidate_value_bits_address,
        candidate_length: Tensor,
        worker_count,
        total_q,
        batch_count,
        tokens,
        max_tasks_per_batch,
        key_rows,
        scale_rows,
        metadata: Tensor,
        task_count,
        final_sparse_indices_address,
        final_sparse_bits_address,
        final_candidate_indices_address,
        final_candidate_bits_address,
        return_values,
        merge_workers,
        output_offset_address,
    ):
        ld_enabled = metadata[LD_RECORD_START, 0] != 0
        final_sparse_indices = make_tensor(
            make_pointer(
                dtypes.int32, dtypes.int64(final_sparse_indices_address), MemLoc.GM
            ),
            make_layout((total_q, topk), stride=(topk, 1)),
        )
        sparse_indices = make_tensor(
            make_pointer(dtypes.int32, dtypes.int64(sparse_indices_address), MemLoc.GM),
            make_layout((64 * query_tile_rows, topk), stride=(topk, 1)),
        )
        sparse_value_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(sparse_value_bits_address), MemLoc.GM
            ),
            make_layout((64 * query_tile_rows, topk), stride=(topk, 1)),
        )
        final_sparse_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(final_sparse_bits_address), MemLoc.GM
            ),
            make_layout((total_q, topk), stride=(topk, 1)),
        )
        final_candidate_indices = make_tensor(
            make_pointer(
                dtypes.int32, dtypes.int64(final_candidate_indices_address), MemLoc.GM
            ),
            make_layout((total_q, candidate_topk), stride=(candidate_topk, 1)),
        )
        final_candidate_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(final_candidate_bits_address), MemLoc.GM
            ),
            make_layout((total_q, candidate_topk), stride=(candidate_topk, 1)),
        )
        candidate_indices = make_tensor(
            make_pointer(
                dtypes.int32, dtypes.int64(candidate_indices_address), MemLoc.GM
            ),
            make_layout(
                (64 * query_tile_rows, candidate_topk), stride=(candidate_topk, 1)
            ),
        )
        candidate_value_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(candidate_value_bits_address), MemLoc.GM
            ),
            make_layout(
                (64 * query_tile_rows, candidate_topk), stride=(candidate_topk, 1)
            ),
        )
        query = make_tensor(
            make_pointer(dtypes.fp4x2_e2m1, dtypes.int64(query_address), MemLoc.GM),
            make_layout((total_q * query_head_count, LOGICAL_D), stride=(LOGICAL_D, 1)),
        )
        key = make_tensor(
            make_pointer(dtypes.fp4x2_e2m1, dtypes.int64(key_address), MemLoc.GM),
            make_layout((key_rows, LOGICAL_D), stride=(LOGICAL_D, 1)),
        )
        key_scale = make_tensor(
            make_pointer(
                dtypes.float8_e8m0, dtypes.int64(key_scale_address), MemLoc.GM
            ),
            make_layout((scale_rows, 2, 2), stride=(4, 2, 1)),
        )
        QliFusedKernel(
            topk,
            candidate_enabled,
            candidate_topk,
            mask_mode,
            cmp_ratio,
            has_cu_seqlens_q,
            has_seqused_q,
            has_seqused_k,
            has_block_table,
            pa_block_size,
            query_head_count,
            query_tile_rows,
            metadata,
            output_offset_address,
            return_values,
        )(
            query,
            key,
            weights,
            query_scale,
            key_scale,
            key_stride0,
            key_dequant_scale_stride0,
            cu_seqlens_q,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            block_table,
            key_out,
            candidate_key_out,
            sparse_indices,
            sparse_value_bits,
            candidate_indices,
            candidate_value_bits,
            candidate_length,
            worker_count,
            total_q,
            batch_count,
            tokens,
            max_tasks_per_batch,
            metadata,
            split_count,
            task_count,
            sparse_indices_address,
            sparse_value_bits_address,
            candidate_indices_address,
            candidate_value_bits_address,
            final_sparse_indices_address,
            final_sparse_bits_address,
            final_candidate_indices_address,
            final_candidate_bits_address,
        )
        if ld_enabled:
            global_sync_all()
            # 显式 UB 视图复用已排空的阶段空间，不重置 Channel 编译期资源。
            if candidate_enabled:
                CandidateLdMergeStage(topk, split_count, candidate_optimized=True)(
                    sparse_indices,
                    sparse_value_bits,
                    final_sparse_indices,
                    final_sparse_bits,
                    metadata,
                    total_q,
                    worker_count * 2,
                    cu_seqlens_q,
                    query_tile_rows,
                    has_cu_seqlens_q,
                    output_offset_address,
                )
            else:
                LdMergeStage(topk, split_count)(
                    sparse_indices,
                    sparse_value_bits,
                    final_sparse_indices,
                    final_sparse_bits,
                    metadata,
                    total_q,
                    worker_count * 2,
                    cu_seqlens_q,
                    query_tile_rows,
                    has_cu_seqlens_q,
                    output_offset_address,
                    return_values,
                )

            if candidate_enabled:
                vec_sync_all()
                # 显式 UB 视图复用已排空的阶段空间，不重置 Channel 编译期资源。
                CandidateLdMergeStage(
                    candidate_topk, split_count, candidate_optimized=True
                )(
                    candidate_indices,
                    candidate_value_bits,
                    final_candidate_indices,
                    final_candidate_bits,
                    metadata,
                    total_q,
                    worker_count * 2,
                    cu_seqlens_q,
                    query_tile_rows,
                    has_cu_seqlens_q,
                )

    @host
    def run_fused_cv(
        query_address,
        key_address,
        weights: Tensor,
        query_scale,
        key_scale_address,
        key_stride0,
        key_dequant_scale_stride0,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        block_table: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices_address,
        sparse_value_bits_address,
        candidate_indices_address,
        candidate_value_bits_address,
        candidate_length: Tensor,
        worker_count,
        total_q,
        batch_count,
        tokens,
        max_tasks_per_batch,
        key_rows,
        scale_rows,
        metadata: Tensor,
        task_count,
        final_sparse_indices_address,
        final_sparse_bits_address,
        final_candidate_indices_address,
        final_candidate_bits_address,
        return_values,
        merge_workers,
        output_offset_address,
    ):
        QuantLightningIndexer[worker_count](
            query_address,
            key_address,
            weights,
            query_scale,
            key_scale_address,
            key_stride0,
            key_dequant_scale_stride0,
            cu_seqlens_q,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            block_table,
            key_out,
            candidate_key_out,
            sparse_indices_address,
            sparse_value_bits_address,
            candidate_indices_address,
            candidate_value_bits_address,
            candidate_length,
            worker_count,
            total_q,
            batch_count,
            tokens,
            max_tasks_per_batch,
            key_rows,
            scale_rows,
            metadata,
            task_count,
            final_sparse_indices_address,
            final_sparse_bits_address,
            final_candidate_indices_address,
            final_candidate_bits_address,
            return_values,
            merge_workers,
            output_offset_address,
        )

    query_rows_dim = cannbotdsl.Dim("T1")
    batch_dim = cannbotdsl.Dim("B")
    block_table_width_dim = cannbotdsl.Dim("MAX_BLOCKS")
    token_capacity_dim = cannbotdsl.Dim("S2_PAD")
    candidate_capacity_dim = cannbotdsl.Dim("CANDIDATE_PAD")
    _sparse_width = cannbotdsl.Dim("SPARSE_WIDTH")
    candidate_width = cannbotdsl.Dim("CANDIDATE_WIDTH")
    partial_rows = cannbotdsl.Dim("PARTIAL_ROWS")
    _metadata_tasks = cannbotdsl.Dim("METADATA_TASKS")
    workspace_rows = MAX_CUBE_WORKERS * (query_tile_m // query_head_count)
    fake = TensorSpec
    dummy_i32_spec = fake((1,), dtypes.int32)
    cu_spec = (
        fake((batch_dim + 1,), dtypes.int32) if has_cu_seqlens_q else dummy_i32_spec
    )
    seqused_q_spec = (
        fake((batch_dim,), dtypes.int32) if has_seqused_q else dummy_i32_spec
    )
    seqused_k_spec = (
        fake((batch_dim,), dtypes.int32) if has_seqused_k else dummy_i32_spec
    )
    residual_spec = (
        fake((batch_dim,), dtypes.int32) if has_cmp_residual_k else dummy_i32_spec
    )
    block_table_spec = (
        fake((batch_dim, block_table_width_dim), dtypes.int32)
        if has_block_table
        else fake((1, 1), dtypes.int32)
    )
    candidate_key_spec = (
        fake(
            (
                workspace_rows,
                candidate_capacity_dim,
            ),
            dtypes.uint16,
        )
        if candidate_enabled
        else fake((workspace_rows, 1), dtypes.uint16)
    )
    _candidate_index_spec = (
        fake((query_rows_dim, candidate_width), dtypes.int32)
        if candidate_enabled
        else fake((1, candidate_width), dtypes.int32)
    )

    compiled = cannbotdsl.compile(
        run_fused_cv,
        dtypes.int64,
        dtypes.int64,
        fake((query_rows_dim, query_head_count), dtypes.float32),
        fake((query_rows_dim * query_head_count, 2, 2), dtypes.float8_e8m0),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        cu_spec,
        seqused_q_spec,
        seqused_k_spec,
        residual_spec,
        block_table_spec,
        fake((workspace_rows, token_capacity_dim), dtypes.uint16),
        candidate_key_spec,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        fake((partial_rows,), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        fake((METADATA_ROWS, METADATA_FIELDS), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )
    return compiled


def clear_caches():
    """Close cached dynamic executables before and after native export."""
    with _COMPILED_KERNEL_LOCK:
        for compiled in _COMPILED_KERNEL.values():
            compiled.close()
        _COMPILED_KERNEL.clear()


def _get_compiled_fused_runner(*config):
    """Cache the dynamic QLI executable, independently of B/T/S2."""
    # DSL native policy is process-wide; never compile a missing runtime binary.
    cache_key = tuple(config)
    with _COMPILED_KERNEL_LOCK:
        compiled = _COMPILED_KERNEL.get(cache_key)
        if compiled is None:
            compiled = _build_compiled_fused_runner(*config)
            _COMPILED_KERNEL[cache_key] = compiled
        return compiled


def _validate_qli_pa_sequences(
    query,
    cu_seqlens_q,
    seqused_q,
    seqused_k,
    cmp_residual_k,
    block_table,
    output_idx_offset,
    mask_mode,
    cmp_ratio,
):
    """校验 PA 分页路径的序列边界、输出偏移与页表元信息。"""
    t = int(query.shape[0])
    if t <= 0:
        raise ValueError("Q must contain at least one T1 row")
    has_cu_seqlens_q = cu_seqlens_q is not None
    batch = int(cu_seqlens_q.numel()) - 1 if has_cu_seqlens_q else 1
    if batch <= 0:
        raise ValueError("Cu_seqlens_q must have shape (B+1,)")

    def validate_metadata(name, tensor, shape):
        if tensor is None:
            return
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)}")
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32")
        if tensor.device != query.device:
            raise ValueError(f"{name} must be on the same NPU as q")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if has_cu_seqlens_q:
        validate_metadata("cu_seqlens_q", cu_seqlens_q, (batch + 1,))
    validate_metadata("output_idx_offset", output_idx_offset, (t, 1))
    validate_metadata("seqused_q", seqused_q, (batch,))
    validate_metadata("seqused_k", seqused_k, (batch,))
    if int(mask_mode) == 3 and int(cmp_ratio) != 1:
        if cmp_residual_k is None:
            raise ValueError(
                "Cmp_residual_k is required for mask_mode=3 and cmp_ratio!=1"
            )
        validate_metadata("cmp_residual_k", cmp_residual_k, (batch,))
    elif cmp_residual_k is not None:
        raise ValueError(
            "Cmp_residual_k must be None unless mask_mode=3 and cmp_ratio!=1"
        )
    if block_table is not None:
        if block_table.ndim != 2 or int(block_table.shape[0]) != batch:
            raise ValueError("Block_table must have shape (B,max_num_blocks_per_seq)")
        if block_table.dtype != torch.int32:
            raise TypeError("Block_table must be int32")
        if block_table.device != query.device:
            raise ValueError("Block_table must be on the same NPU as q")
        if not block_table.is_contiguous():
            raise ValueError("Block_table must be contiguous")

    return t, batch, has_cu_seqlens_q


def _validate_qli_candidate(candidate_topk_blocks, candidate_block_size):
    """Candidate 关闭与开启时分别校验配套属性。"""
    if candidate_topk_blocks == -1:
        if candidate_block_size != -1:
            raise ValueError(
                "Candidate_block_size must be -1 when candidate is disabled"
            )
    elif candidate_topk_blocks <= 0 or candidate_block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError("Candidate requires positive topk_blocks and block_size=8")


def _validate_qli_compute_attributes(topk, mask_mode, cmp_ratio):
    """校验选择容量、掩码模式与压缩比例的支持范围。"""
    if int(topk) < 1 or int(topk) > 8192:
        raise ValueError("Topk must be in ASC-supported range [1,8192]")
    if int(mask_mode) not in (0, 3):
        raise ValueError("Mask_mode must be 0 or 3")
    if int(cmp_ratio) < 1 or int(cmp_ratio) > 128:
        raise ValueError("Cmp_ratio must be in [1,128]")


def quant_lightning_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    descale_q: torch.Tensor,
    descale_k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    *,
    topk: int,
    quant_mode: int,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "TND",
    return_value: bool = False,
    candidate_topk_blocks: int = -1,
    candidate_block_size: int = -1,
) -> tuple[torch.Tensor, ...]:
    """Run the xlsx QLI MXFP4 contract on NPU tensors.

    ``cu_seqlens_q`` maps TND query rows to batches, ``block_table`` maps each
    batch's logical K pages to physical PA_BBND blocks, and ``seqused_k[b]``
    gives that batch's direct (non-cumulative) usable S2 length. Internally,
    A query group contains six rows for N1=32 (four for N1=64).
    Each N256 step uses two half-M Cube results, one per AIV. AICPU
    schedules S2 shards; AIVs merge partial TopK in the same kernel.
    """
    if q.ndim != 3 or tuple(q.shape[1:]) != (32, 64):
        raise ValueError("QLI requires N1=32 and logical D=128")
    if layout_k == "TND":
        return _run_tnd(
            q,
            k,
            w,
            descale_q,
            descale_k,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            block_table,
            output_idx_offset,
            metadata,
            topk=topk,
            quant_mode=quant_mode,
            max_seqlen_q=max_seqlen_q,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            layout_q=layout_q,
            layout_k=layout_k,
            return_value=return_value,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
        )
    if metadata is None:
        raise ValueError(
            "Metadata is required; call quant_lightning_indexer_metadata before this operator"
        )
    if q.ndim != 3 or int(q.shape[1]) not in (32, 64):
        raise ValueError("Q must have shape [T1,N1,64], N1=32 or 64")
    query_heads = int(q.shape[1])
    query_tile_rows = min(QUERY_TILE_ROWS, 256 // query_heads)
    tile_m = query_heads * query_tile_rows
    query, key, weights = q, k, w
    query_dequant_scale, key_dequant_scale = descale_q, descale_k
    if int(quant_mode) != 1:
        raise ValueError(
            "Quant_lightning_indexer public contract supports quant_mode=1 only"
        )
    if layout_q != "TND":
        raise ValueError("Layout_q must be TND under the current xlsx contract")
    if layout_k != "PA_BBND":
        raise ValueError("Layout_k must be PA_BBND under the current xlsx contract")
    if query.device.type != "npu" or key.device.type != "npu":
        raise ValueError("Quant_lightning_indexer requires NPU q/k tensors")
    if query.dtype != torch.uint8 or key.dtype != torch.uint8:
        raise TypeError("Public q/k storage must be uint8 packed MXFP4")
    if query.ndim != 3 or key.ndim != 4:
        raise ValueError("Q must be [T1,32,64], k must be [block_num,block_size,1,64]")
    if (
        query.shape[1] not in (32, 64)
        or key.shape[2] != N2
        or query.shape[2] != PACKED_D
        or key.shape[3] != PACKED_D
    ):
        raise ValueError("Public static axes require N1=32, N2=1 and logical D=128")
    if tuple(weights.shape) != tuple(query.shape[:2]) or weights.dtype != torch.float32:
        raise ValueError("W must be FP32 with shape (T1,N1)")
    if query_dequant_scale is None:
        raise ValueError("Descale_q is a required xlsx input")
    expected_q_scale = (*query.shape[:2], LOGICAL_D // 64, 2)
    if tuple(query_dequant_scale.shape) != expected_q_scale:
        raise ValueError("Descale_q must have corrected xlsx shape (T1,N1,D/64,2)")
    if key_dequant_scale is None:
        raise ValueError("Descale_k is a required xlsx input")
    expected = tuple(key.shape[:3]) + (2, 2)
    if tuple(key_dequant_scale.shape) != expected:
        raise ValueError("Descale_k must have xlsx shape (block_num,block_size,N2,2,2)")
    key_stride0, key_storage_rows = _pa_axis0_storage_rows(
        key,
        "k",
        (PACKED_D, PACKED_D, 1),
        KEY_STORAGE_ROW_ELEMENTS,
    )
    (
        key_dequant_scale_stride0,
        key_dequant_scale_storage_rows,
    ) = _pa_axis0_storage_rows(
        key_dequant_scale,
        "descale_k",
        (KEY_SCALE_STORAGE_ROW_ELEMENTS, 2 * 2, 2, 1),
        KEY_SCALE_STORAGE_ROW_ELEMENTS,
    )
    # CANNBotDSL's packed-FP4 host carrier is compact-only. Expose the
    # allocation span, including axis-0 gaps, as a zero-copy compact carrier;
    # the original page strides are passed separately and consumed on NPU.
    key_storage = torch.as_strided(
        key,
        (key_storage_rows, PACKED_D),
        (KEY_STORAGE_ROW_ELEMENTS, 1),
    )
    key_dequant_scale_storage = torch.as_strided(
        key_dequant_scale,
        (key_dequant_scale_storage_rows, 4),
        (KEY_SCALE_STORAGE_ROW_ELEMENTS, 1),
    )
    _validate_qli_candidate(candidate_topk_blocks, candidate_block_size)
    # Keep this as a Python/JIT specialization constant. Disabled launches
    # compile out Candidate block-max, key generation, TopK and UB resources.
    candidate_enabled = int(candidate_topk_blocks) > 0
    pa_block_size = int(key.shape[1])
    if (
        pa_block_size <= 0
        or pa_block_size > MAX_PA_BLOCK_SIZE
        or pa_block_size % PA_BLOCK_ALIGNMENT != 0
    ):
        raise ValueError("PA block_size must be a multiple of 16 in (0, 1024]")
    _validate_qli_compute_attributes(topk, mask_mode, cmp_ratio)
    if query_dequant_scale.dtype != torch.uint8:
        raise TypeError("Descale_q must use uint8 E8M0 storage")
    if key_dequant_scale.dtype != torch.uint8:
        raise TypeError("Descale_k must use uint8 E8M0 storage")
    for name, tensor in (
        ("w", weights),
        ("descale_q", query_dequant_scale),
    ):
        if tensor.device != query.device:
            raise ValueError(f"{name} must be on the same NPU as q")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not query.is_contiguous():
        raise ValueError("Q must be contiguous")

    t, batch, has_cu_seqlens_q = _validate_qli_pa_sequences(
        query,
        cu_seqlens_q,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        mask_mode,
        cmp_ratio,
    )
    has_seqused_q = seqused_q is not None
    has_seqused_k = seqused_k is not None
    has_cmp_residual_k = cmp_residual_k is not None
    has_block_table = block_table is not None
    logical_k_capacity = (
        int(block_table.shape[1]) if has_block_table else int(key.shape[0])
    ) * pa_block_size
    tokens_pad = ceil_div(logical_k_capacity, TILE_N) * TILE_N
    if tokens_pad <= 0:
        raise ValueError("K storage must expose at least one PA page")
    # Optional xlsx tensors remain optional at the public boundary.  One
    # uninitialized carrier is sufficient because the JIT specialization
    # compiles out every load when the corresponding tensor is absent.
    dummy_i32 = torch.empty((1,), dtype=torch.int32, device=query.device)
    dummy_table = torch.empty((1, 1), dtype=torch.int32, device=query.device)
    cu_kernel = cu_seqlens_q if has_cu_seqlens_q else dummy_i32
    seqused_q_kernel = seqused_q if has_seqused_q else dummy_i32
    seqused_k_kernel = seqused_k if has_seqused_k else dummy_i32
    cmp_residual_kernel = cmp_residual_k if has_cmp_residual_k else dummy_i32
    block_table_kernel = block_table if has_block_table else dummy_table

    max_query_length = int(max_seqlen_q) if int(max_seqlen_q) > 0 else t
    max_tasks_per_batch = ceil_div(max_query_length, tile_m // query_heads)
    # Workspace capacity; AICPU metadata owns the actual shard boundaries.
    split_count = max(1, min(MAX_MERGE_PARTS, STREAM_TOPK_TRUNK_LEN // int(topk)))
    if (
        metadata.dtype != torch.int32
        or metadata.ndim != 1
        or metadata.numel() != METADATA_ELEMENTS
        or not metadata.is_contiguous()
    ):
        raise ValueError(
            "Metadata must be contiguous int32 [1024] in ASC LI/LD boundary format"
        )
    if metadata.device != q.device:
        raise ValueError("Metadata must be on the same device as q")
    metadata = metadata.view(METADATA_ROWS, METADATA_FIELDS)
    worker_count = MAX_CUBE_WORKERS
    workspace_row_count = MAX_CUBE_WORKERS * (tile_m // query_heads)
    shard_pad = tokens_pad
    # Public outputs and GM scratch are allocation-only Torch calls.  The
    # fused kernel writes every public row, including invalid/padded rows.
    sparse_indices_2d = torch.empty(
        (64 * query_tile_rows, int(topk)), dtype=torch.int32, device=query.device
    )
    sparse_values_2d = torch.empty(
        (64 * query_tile_rows, int(topk)), dtype=torch.bfloat16, device=weights.device
    )
    candidate_count = max(0, int(candidate_topk_blocks))
    candidate_indices_2d = torch.empty(
        (64 * query_tile_rows, candidate_count),
        dtype=torch.int32,
        device=query.device,
    )
    candidate_length_1d = torch.empty(
        (t * split_count,), dtype=torch.int32, device=query.device
    )

    key_workspace = torch.empty(
        (workspace_row_count, shard_pad),
        dtype=torch.uint16,
        device=query.device,
    )
    candidate_key_workspace = torch.empty(
        (
            workspace_row_count,
            shard_pad // CANDIDATE_BLOCK_SIZE if candidate_enabled else 1,
        ),
        dtype=torch.uint16,
        device=query.device,
    )
    kernel_candidate_count = candidate_count if candidate_enabled else 1
    candidate_indices_kernel = (
        candidate_indices_2d
        if candidate_enabled
        else torch.empty((1, split_count), dtype=torch.int32, device=query.device)
    )
    candidate_value_workspace = torch.empty(
        (64 * query_tile_rows, kernel_candidate_count),
        dtype=torch.uint16,
        device=query.device,
    )

    # GM aliases describe native packed formats without device copies.
    query_address = query.data_ptr()
    query_scale_storage = query_dequant_scale.reshape(t * query_heads, 2, 2).view(
        torch.float8_e8m0fnu
    )

    final_sparse_indices = torch.empty(
        (t, int(topk)), dtype=torch.int32, device=query.device
    )
    final_sparse_values = torch.empty(
        (t, int(topk)), dtype=torch.bfloat16, device=query.device
    )
    final_candidate_indices = torch.empty(
        (t, kernel_candidate_count), dtype=torch.int32, device=query.device
    )
    final_candidate_bits = torch.empty(
        (t, kernel_candidate_count), dtype=torch.uint16, device=query.device
    )
    _get_compiled_fused_runner(
        int(topk),
        candidate_enabled,
        kernel_candidate_count,
        int(mask_mode),
        int(cmp_ratio),
        has_cu_seqlens_q,
        has_seqused_q,
        has_seqused_k,
        has_block_table,
        pa_block_size,
        split_count,
        query_heads,
        query_tile_rows,
    )(
        query_address,
        key_storage.data_ptr(),
        weights,
        query_scale_storage,
        key_dequant_scale_storage.data_ptr(),
        key_stride0,
        key_dequant_scale_stride0,
        cu_kernel,
        seqused_q_kernel,
        seqused_k_kernel,
        cmp_residual_kernel,
        block_table_kernel,
        key_workspace,
        candidate_key_workspace,
        sparse_indices_2d.data_ptr(),
        sparse_values_2d.view(torch.uint16).data_ptr(),
        candidate_indices_kernel.data_ptr(),
        candidate_value_workspace.data_ptr(),
        candidate_length_1d,
        worker_count,
        t,
        batch,
        logical_k_capacity,
        max_tasks_per_batch,
        key_storage_rows,
        key_dequant_scale_storage.shape[0],
        metadata,
        metadata.shape[0],
        final_sparse_indices.data_ptr(),
        final_sparse_values.view(torch.uint16).data_ptr(),
        final_candidate_indices.data_ptr(),
        final_candidate_bits.data_ptr(),
        int(return_value),
        min(64, t),
        0 if output_idx_offset is None else output_idx_offset.data_ptr(),
    )

    sparse_indices_2d, sparse_values_2d = final_sparse_indices, final_sparse_values
    if candidate_enabled:
        candidate_indices_2d = final_candidate_indices
    candidate_length_1d = candidate_length_1d[:t]
    return (
        sparse_indices_2d.view(t, N2, int(topk)),
        sparse_values_2d.view(t, N2, int(topk))
        if return_value
        else sparse_values_2d.reshape(-1)[:0],
        candidate_indices_2d.view(t, N2, candidate_count)
        if candidate_enabled
        else candidate_indices_2d.reshape(-1),
        candidate_length_1d.view(t, N2)
        if candidate_enabled
        else candidate_length_1d[:0],
    )


@export("qli")
def export_qli():
    """Export static profiles with dynamic tensor axes, without NPU execution."""
    from itertools import product
    import json

    defaults = {
        "topk": 512,
        "candidate_enabled": True,
        "candidate_topk": 2048,
        "mask_mode": 3,
        "cmp_ratio": 1,
        "has_cu_seqlens_q": True,
        "has_seqused_q": True,
        "has_seqused_k": True,
        "has_block_table": True,
        "pa_block_size": 128,
        "split_count": 8,
        "n1": 32,
        "query_tile_rows": 6,
        "layout_k": "PA_BBND",
    }
    profiles = [
        dict(
            candidate_enabled=enabled,
            candidate_topk=2048 if enabled else 1,
            mask_mode=mask,
            cmp_ratio=ratio,
            pa_block_size=page,
            layout_k=layout,
        )
        for enabled, mask, ratio, (layout, page) in product(
            (False, True),
            (0, 3),
            (1, 2),
            (("PA_BBND", 64), ("PA_BBND", 128), ("TND", TILE_N)),
        )
    ]
    profile_path = os.environ.get("CANNBOTDSL_DS41_PROFILES")
    if profile_path:
        with open(profile_path, encoding="utf-8") as stream:
            profiles = json.load(stream).get("qli", profiles)
    clear_caches()
    clear_tnd_caches()
    try:
        exported = set()
        for profile in profiles:
            unknown = set(profile) - set(defaults)
            if unknown:
                raise ValueError(f"Unknown export profile fields: {sorted(unknown)}")
            config = dict(defaults, **profile)
            # cmp_ratio is unused without the causal mask; export one binary.
            identity = tuple(
                (name, value)
                for name, value in config.items()
                if name != "cmp_ratio" or config["mask_mode"] == 3
            )
            if identity in exported:
                continue
            exported.add(identity)
            runner_config = tuple(
                config[name] for name in defaults if name != "layout_k"
            )
            if config["layout_k"] == "TND":
                _get_tnd_compiled_fused_runner(*runner_config)
            elif config["layout_k"] == "PA_BBND":
                _get_compiled_fused_runner(*runner_config)
            else:
                raise ValueError("Layout_k export profile must be PA_BBND or TND")
    finally:
        clear_caches()
        clear_tnd_caches()


class QliTndCube(QliCube):
    @jit
    def compute_qk(
        self,
        key,
        key_scale,
        block_table,
        batch_idx,
        actual_s2,
        key_stride0,
        key_dequant_scale_stride0,
        has_block_table,
        token_tile,
        qk_handoff,
    ):
        key_write = self.k_l1.produce()
        scale_write = self.k_scale_l1.produce()
        base = dtypes.int64(block_table[batch_idx])
        logical_row = token_tile * TILE_N
        count = min(dtypes.int64(QK_COLUMNS), dtypes.int64(actual_s2) - logical_row)
        source_row = base + logical_row
        # Tile windows pin the full NZ frame while the dynamic source slice
        # keeps the copy height clamped to the available key rows.
        mem_copy(
            tile_slice(key_write, (QK_COLUMNS, LOGICAL_D), (0, 0)),
            key[source_row : source_row + count, None],
            engine=self.nd2nz_key,
        )
        mem_copy(
            tile_slice(scale_write, (4, QK_COLUMNS), (0, 0)),
            key_scale[source_row : source_row + count, None, None],
            engine=self.scale_b,
        )
        mem_copy(
            self.l0b.produce(), self.k_l1.consume(), mx_scale=self.k_scale_l1.consume()
        )
        key_read = self.l0b.consume()
        dsl_matmul(
            self.l0c.produce(), self.l0a0.consume(), key_read, init=True, unit_flag=3
        )
        qk_l0c_read0 = self.l0c.consume()
        mem_copy(
            reinterpret(qk_handoff, shape=(self.tile_m // 2, 128), stride=(256, 1)),
            reinterpret(qk_l0c_read0, shape=(self.tile_m // 2, 128)),
            engine=self.fixpipe_aiv0,
            dst_subblock=0,
            unit_flag=3,
        )
        mem_copy(
            reinterpret(
                qk_handoff,
                shape=(self.tile_m // 2, 128),
                stride=(256, 1),
                offset=(self.tile_m // 2) * 512,
            ),
            reinterpret(
                qk_l0c_read0,
                shape=(self.tile_m // 2, 128),
                offset=(self.tile_m // 2) * 128 * 4,
            ),
            engine=self.fixpipe_aiv0,
            dst_subblock=0,
            unit_flag=3,
        )
        dsl_matmul(
            self.l0c.produce(), self.l0a1.consume(), key_read, init=True, unit_flag=3
        )
        qk_l0c_read1 = self.l0c.consume()
        mem_copy(
            reinterpret(qk_handoff, shape=(self.tile_m // 2, 128), stride=(256, 1)),
            reinterpret(qk_l0c_read1, shape=(self.tile_m // 2, 128)),
            engine=self.fixpipe_aiv1,
            dst_subblock=1,
            unit_flag=3,
        )
        mem_copy(
            reinterpret(
                qk_handoff,
                shape=(self.tile_m // 2, 128),
                stride=(256, 1),
                offset=(self.tile_m // 2) * 512,
            ),
            reinterpret(
                qk_l0c_read1,
                shape=(self.tile_m // 2, 128),
                offset=(self.tile_m // 2) * 128 * 4,
            ),
            engine=self.fixpipe_aiv1,
            dst_subblock=1,
            unit_flag=3,
        )


class QliTndFusedKernel(QliFusedKernel):
    def __init__(
        self,
        topk,
        candidate_enabled,
        candidate_topk,
        mask_mode,
        cmp_ratio,
        has_cu_seqlens_q,
        has_seqused_q,
        has_seqused_k,
        has_block_table,
        pa_block_size,
        n1,
        query_rows,
        metadata,
        output_offset_address,
        return_values,
    ):
        self.ub_pool = _UbPool()
        self.output_offset_address = output_offset_address
        self.metadata = metadata
        self.need_topk_values = _need_topk_values(metadata, return_values)
        self.n1 = int(n1)
        self.query_rows = int(query_rows)
        self.vec_rows = int(query_rows) // 2
        self.tile_m = int(n1) * int(query_rows)
        self.topk_count = int(topk)
        self.cache_enabled = (
            not candidate_enabled
            and self.topk_count == CACHED_TOPK_COUNT
            and self.tile_m <= 192
        )
        self.candidate_enabled = bool(candidate_enabled)
        self.candidate_topk_count = int(candidate_topk)
        self.mask_mode = int(mask_mode)
        self.cmp_ratio = int(cmp_ratio)
        self.has_cu_seqlens_q = bool(has_cu_seqlens_q)
        self.has_seqused_q = bool(has_seqused_q)
        self.has_seqused_k = bool(has_seqused_k)
        self.has_cmp_residual_k = self.mask_mode == 3 and self.cmp_ratio != 1
        self.has_block_table = bool(has_block_table)
        self.qk_slot0 = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.tile_m, 128),
            dtypes.bfloat16,
            stride=(256, 1),
            addr=0,
        )
        self.qk_slot1 = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.tile_m, 128),
            dtypes.bfloat16,
            stride=(256, 1),
            addr=256,
        )
        self.cube = QliTndCube(pa_block_size, n1, query_rows)
        self.topk_workspace = QliRawTopKWorkspace(
            TOPK_TRUNK_LEN if self.cache_enabled else STREAM_TOPK_TRUNK_LEN,
            max(
                self.topk_count,
                self.candidate_topk_count if self.candidate_enabled else 1,
            ),
            pipelined=self.cache_enabled,
        )
        self.topk = QliRawTopKSelector(self.topk_count, self.topk_workspace)
        if self.candidate_enabled:
            self.candidate_topk = QliRawTopKSelector(
                self.candidate_topk_count,
                self.topk_workspace,
            )

        self.vector = QliVector(
            get_subblock_id(),
            self.mask_mode,
            self.cmp_ratio,
            self.candidate_enabled,
            self.candidate_topk_count,
            n1,
            query_rows,
            CACHED_VECTOR_UB_OFFSET
            if self.cache_enabled
            else self.tile_m * QK_ROW_BYTES + VECTOR_AFTER_QK_BYTES,
            self.cache_enabled,
        )
        # Dedicated 512B-strided bank-separated histogram slots, main phase only.
        self.hist_slot0 = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.vec_rows * 2, 128),
            dtypes.uint16,
            stride=(256, 1),
            addr=CACHED_HISTOGRAM_UB_OFFSET
            if self.cache_enabled
            else self.tile_m * QK_ROW_BYTES + HISTOGRAM_AFTER_QK_BYTES,
        )
        self.hist_slot1 = _ub_buffer(
            self.ub_pool,
            MemLoc.UB,
            (self.vec_rows * 2, 128),
            dtypes.uint16,
            stride=(256, 1),
            addr=CACHED_HISTOGRAM_UB_OFFSET + HISTOGRAM_SLOT_OFFSET
            if self.cache_enabled
            else self.tile_m * QK_ROW_BYTES
            + HISTOGRAM_AFTER_QK_BYTES
            + HISTOGRAM_SLOT_OFFSET,
        )
        self.topk.hist_slot0 = self.hist_slot0
        self.topk.hist_slot1 = self.hist_slot1
        # Main phase: QK [0,98304), cached keys [98304,251904),
        # histogram [251904,254976), weights and fallback key stages above255488.
        # Cached TopK phase: temporary/output buffers reuse the drained QK area.
        if self.cache_enabled:
            self.full_key_cache = _ub_buffer(
                self.ub_pool,
                MemLoc.UB,
                (self.vec_rows, CACHE_TOKENS),
                dtypes.uint16,
                addr=CACHED_QK_UB_BYTES,
            )
            self.vector.full_key_cache = self.full_key_cache
            self.topk.cache_tmp = _ub_buffer(
                self.ub_pool,
                MemLoc.UB,
                (CACHE_TOKENS,),
                dtypes.uint16,
                addr=0,
            )
            self.topk.cache_output_idx_stage = _ub_channel(
                self.ub_pool,
                MemLoc.UB,
                (1, self.topk_count),
                dtypes.int32,
                depth=1,
                addr=CACHED_TOPK_INDEX_OFFSET,
            )
            self.topk.cache_output_value_bits_stage = _ub_channel(
                self.ub_pool,
                MemLoc.UB,
                (1, self.topk_count),
                dtypes.uint16,
                depth=1,
                addr=CACHED_TOPK_VALUE_OFFSET,
            )


def _build_tnd_compiled_fused_runner(
    topk,
    candidate_enabled,
    candidate_topk,
    mask_mode,
    cmp_ratio,
    has_cu_seqlens_q,
    has_seqused_q,
    has_seqused_k,
    has_block_table,
    pa_block_size,
    split_count,
    n1=32,
    query_tile_rows=6,
):
    """Compile the dynamic TensorSpec contract using the standard DSL entry."""
    query_head_count = int(n1)
    query_tile_m = query_head_count * int(query_tile_rows)
    has_cmp_residual_k = int(mask_mode) == 3 and int(cmp_ratio) != 1
    topk = int(topk)
    candidate_enabled = bool(candidate_enabled)
    candidate_topk = int(candidate_topk)
    mask_mode = int(mask_mode)
    cmp_ratio = int(cmp_ratio)

    @kernel
    def QuantLightningIndexer(
        query_address,
        key_address,
        weights: Tensor,
        query_scale,
        key_scale_address,
        key_stride0,
        key_dequant_scale_stride0,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        block_table: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices_address,
        sparse_value_bits_address,
        candidate_indices_address,
        candidate_value_bits_address,
        candidate_length: Tensor,
        worker_count,
        total_q,
        batch_count,
        tokens,
        max_tasks_per_batch,
        key_rows,
        scale_rows,
        metadata: Tensor,
        task_count,
        final_sparse_indices_address,
        final_sparse_bits_address,
        final_candidate_indices_address,
        final_candidate_bits_address,
        return_values,
        merge_workers,
        output_offset_address,
    ):
        ld_enabled = metadata[LD_RECORD_START, 0] != 0
        final_sparse_indices = make_tensor(
            make_pointer(
                dtypes.int32, dtypes.int64(final_sparse_indices_address), MemLoc.GM
            ),
            make_layout((total_q, topk), stride=(topk, 1)),
        )
        sparse_indices = make_tensor(
            make_pointer(dtypes.int32, dtypes.int64(sparse_indices_address), MemLoc.GM),
            make_layout((64 * query_tile_rows, topk), stride=(topk, 1)),
        )
        sparse_value_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(sparse_value_bits_address), MemLoc.GM
            ),
            make_layout((64 * query_tile_rows, topk), stride=(topk, 1)),
        )
        final_sparse_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(final_sparse_bits_address), MemLoc.GM
            ),
            make_layout((total_q, topk), stride=(topk, 1)),
        )
        final_candidate_indices = make_tensor(
            make_pointer(
                dtypes.int32, dtypes.int64(final_candidate_indices_address), MemLoc.GM
            ),
            make_layout((total_q, candidate_topk), stride=(candidate_topk, 1)),
        )
        final_candidate_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(final_candidate_bits_address), MemLoc.GM
            ),
            make_layout((total_q, candidate_topk), stride=(candidate_topk, 1)),
        )
        candidate_indices = make_tensor(
            make_pointer(
                dtypes.int32, dtypes.int64(candidate_indices_address), MemLoc.GM
            ),
            make_layout(
                (64 * query_tile_rows, candidate_topk), stride=(candidate_topk, 1)
            ),
        )
        candidate_value_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(candidate_value_bits_address), MemLoc.GM
            ),
            make_layout(
                (64 * query_tile_rows, candidate_topk), stride=(candidate_topk, 1)
            ),
        )
        query = make_tensor(
            make_pointer(dtypes.fp4x2_e2m1, dtypes.int64(query_address), MemLoc.GM),
            make_layout((total_q * query_head_count, LOGICAL_D), stride=(LOGICAL_D, 1)),
        )
        key = make_tensor(
            make_pointer(dtypes.fp4x2_e2m1, dtypes.int64(key_address), MemLoc.GM),
            make_layout((key_rows, LOGICAL_D), stride=(key_stride0 * 2, 1)),
        )
        key_scale = make_tensor(
            make_pointer(
                dtypes.float8_e8m0, dtypes.int64(key_scale_address), MemLoc.GM
            ),
            make_layout((scale_rows, 2, 2), stride=(key_dequant_scale_stride0, 2, 1)),
        )
        QliTndFusedKernel(
            topk,
            candidate_enabled,
            candidate_topk,
            mask_mode,
            cmp_ratio,
            has_cu_seqlens_q,
            has_seqused_q,
            has_seqused_k,
            has_block_table,
            pa_block_size,
            query_head_count,
            query_tile_rows,
            metadata,
            output_offset_address,
            return_values,
        )(
            query,
            key,
            weights,
            query_scale,
            key_scale,
            key_stride0,
            key_dequant_scale_stride0,
            cu_seqlens_q,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            block_table,
            key_out,
            candidate_key_out,
            sparse_indices,
            sparse_value_bits,
            candidate_indices,
            candidate_value_bits,
            candidate_length,
            worker_count,
            total_q,
            batch_count,
            tokens,
            max_tasks_per_batch,
            metadata,
            split_count,
            task_count,
            sparse_indices_address,
            sparse_value_bits_address,
            candidate_indices_address,
            candidate_value_bits_address,
            final_sparse_indices_address,
            final_sparse_bits_address,
            final_candidate_indices_address,
            final_candidate_bits_address,
        )
        if ld_enabled:
            global_sync_all()
            # 显式 UB 视图复用已排空的阶段空间，不重置 Channel 编译期资源。
            if candidate_enabled:
                CandidateLdMergeStage(topk, split_count, candidate_optimized=True)(
                    sparse_indices,
                    sparse_value_bits,
                    final_sparse_indices,
                    final_sparse_bits,
                    metadata,
                    total_q,
                    worker_count * 2,
                    cu_seqlens_q,
                    query_tile_rows,
                    has_cu_seqlens_q,
                    output_offset_address,
                )
            else:
                LdMergeStage(topk, split_count)(
                    sparse_indices,
                    sparse_value_bits,
                    final_sparse_indices,
                    final_sparse_bits,
                    metadata,
                    total_q,
                    worker_count * 2,
                    cu_seqlens_q,
                    query_tile_rows,
                    has_cu_seqlens_q,
                    output_offset_address,
                    return_values,
                )

            if candidate_enabled:
                vec_sync_all()
                # 显式 UB 视图复用已排空的阶段空间，不重置 Channel 编译期资源。
                CandidateLdMergeStage(
                    candidate_topk, split_count, candidate_optimized=True
                )(
                    candidate_indices,
                    candidate_value_bits,
                    final_candidate_indices,
                    final_candidate_bits,
                    metadata,
                    total_q,
                    worker_count * 2,
                    cu_seqlens_q,
                    query_tile_rows,
                    has_cu_seqlens_q,
                )

    @host
    def run_fused_cv(
        query_address,
        key_address,
        weights: Tensor,
        query_scale,
        key_scale_address,
        key_stride0,
        key_dequant_scale_stride0,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        block_table: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices_address,
        sparse_value_bits_address,
        candidate_indices_address,
        candidate_value_bits_address,
        candidate_length: Tensor,
        worker_count,
        total_q,
        batch_count,
        tokens,
        max_tasks_per_batch,
        key_rows,
        scale_rows,
        metadata: Tensor,
        task_count,
        final_sparse_indices_address,
        final_sparse_bits_address,
        final_candidate_indices_address,
        final_candidate_bits_address,
        return_values,
        merge_workers,
        output_offset_address,
    ):
        QuantLightningIndexer[worker_count](
            query_address,
            key_address,
            weights,
            query_scale,
            key_scale_address,
            key_stride0,
            key_dequant_scale_stride0,
            cu_seqlens_q,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            block_table,
            key_out,
            candidate_key_out,
            sparse_indices_address,
            sparse_value_bits_address,
            candidate_indices_address,
            candidate_value_bits_address,
            candidate_length,
            worker_count,
            total_q,
            batch_count,
            tokens,
            max_tasks_per_batch,
            key_rows,
            scale_rows,
            metadata,
            task_count,
            final_sparse_indices_address,
            final_sparse_bits_address,
            final_candidate_indices_address,
            final_candidate_bits_address,
            return_values,
            merge_workers,
            output_offset_address,
        )

    query_rows_dim = cannbotdsl.Dim("T1")
    batch_dim = cannbotdsl.Dim("B")
    _block_table_width_dim = cannbotdsl.Dim("MAX_BLOCKS")
    token_capacity_dim = cannbotdsl.Dim("S2_PAD")
    candidate_capacity_dim = cannbotdsl.Dim("CANDIDATE_PAD")
    _sparse_width = cannbotdsl.Dim("SPARSE_WIDTH")
    candidate_width = cannbotdsl.Dim("CANDIDATE_WIDTH")
    partial_rows = cannbotdsl.Dim("PARTIAL_ROWS")
    _metadata_tasks = cannbotdsl.Dim("METADATA_TASKS")
    workspace_rows = MAX_CUBE_WORKERS * (query_tile_m // query_head_count)
    fake = TensorSpec
    dummy_i32_spec = fake((1,), dtypes.int32)
    cu_spec = (
        fake((batch_dim + 1,), dtypes.int32) if has_cu_seqlens_q else dummy_i32_spec
    )
    seqused_q_spec = (
        fake((batch_dim,), dtypes.int32) if has_seqused_q else dummy_i32_spec
    )
    seqused_k_spec = (
        fake((batch_dim,), dtypes.int32) if has_seqused_k else dummy_i32_spec
    )
    residual_spec = (
        fake((batch_dim,), dtypes.int32) if has_cmp_residual_k else dummy_i32_spec
    )
    block_table_spec = (
        fake((batch_dim + 1,), dtypes.int32)
        if has_block_table
        else fake((1, 1), dtypes.int32)
    )
    candidate_key_spec = (
        fake(
            (
                workspace_rows,
                candidate_capacity_dim,
            ),
            dtypes.uint16,
        )
        if candidate_enabled
        else fake((workspace_rows, 1), dtypes.uint16)
    )
    _candidate_index_spec = (
        fake((query_rows_dim, candidate_width), dtypes.int32)
        if candidate_enabled
        else fake((1, candidate_width), dtypes.int32)
    )

    compiled = cannbotdsl.compile(
        run_fused_cv,
        dtypes.int64,
        dtypes.int64,
        fake((query_rows_dim, query_head_count), dtypes.float32),
        fake((query_rows_dim * query_head_count, 2, 2), dtypes.float8_e8m0),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        cu_spec,
        seqused_q_spec,
        seqused_k_spec,
        residual_spec,
        block_table_spec,
        fake((workspace_rows, token_capacity_dim), dtypes.uint16),
        candidate_key_spec,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        fake((partial_rows,), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        fake((METADATA_ROWS, METADATA_FIELDS), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )
    return compiled


_TND_COMPILED_KERNEL = {}
_TND_COMPILED_KERNEL_LOCK = threading.Lock()


def clear_tnd_caches():
    """Close cached dynamic executables before and after native export."""
    with _TND_COMPILED_KERNEL_LOCK:
        for compiled in _TND_COMPILED_KERNEL.values():
            compiled.close()
        _TND_COMPILED_KERNEL.clear()


def _get_tnd_compiled_fused_runner(*config):
    """Cache the dynamic QLI executable, independently of B/T/S2."""
    # DSL native policy is process-wide; never compile a missing runtime binary.
    cache_key = tuple(config)
    with _TND_COMPILED_KERNEL_LOCK:
        compiled = _TND_COMPILED_KERNEL.get(cache_key)
        if compiled is None:
            compiled = _build_tnd_compiled_fused_runner(*config)
            _TND_COMPILED_KERNEL[cache_key] = compiled
        return compiled


def _tnd_axis0_stride(tensor, name, inner_strides, row_bytes):
    if tensor.shape[0] <= 0 or tuple(tensor.stride()[1:]) != inner_strides:
        raise ValueError(f"{name} requires positive T2 and contiguous inner axes")
    stride = int(tensor.stride(0))
    if stride < row_bytes:
        raise ValueError(f"{name} requires non-overlapping positive axis-0 stride")
    return stride


def _run_tnd(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    descale_q: torch.Tensor,
    descale_k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    *,
    topk: int,
    quant_mode: int,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "TND",
    return_value: bool = False,
    candidate_topk_blocks: int = -1,
    candidate_block_size: int = -1,
) -> tuple[torch.Tensor, ...]:
    if metadata is None:
        raise ValueError(
            "Metadata is required; call quant_lightning_indexer_metadata before this operator"
        )
    if int(max_seqlen_q) < -1:
        raise ValueError("Max_seqlen_q must be -1 or non-negative")
    if q.ndim != 3 or int(q.shape[1]) not in (32, 64):
        raise ValueError("Q must have shape [T1,N1,64], N1=32 or 64")
    query_heads = int(q.shape[1])
    query_tile_rows = min(QUERY_TILE_ROWS, 256 // query_heads)
    tile_m = query_heads * query_tile_rows
    query, key, weights = q, k, w
    query_dequant_scale, key_dequant_scale = descale_q, descale_k
    if int(quant_mode) != 1:
        raise ValueError(
            "Quant_lightning_indexer public contract supports quant_mode=1 only"
        )
    if layout_q != "TND":
        raise ValueError("Layout_q must be TND under the current xlsx contract")
    if layout_k != "TND":
        raise ValueError("Layout_k must be TND")
    if query.device.type != "npu" or key.device.type != "npu":
        raise ValueError("Quant_lightning_indexer requires NPU q/k tensors")
    if query.dtype != torch.uint8 or key.dtype != torch.uint8:
        raise TypeError("Public q/k storage must be uint8 packed MXFP4")
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("Q must be [T1,32,64], k must be [T2,1,64]")
    if (
        query.shape[1] not in (32, 64)
        or key.shape[1] != N2
        or query.shape[2] != PACKED_D
        or key.shape[2] != PACKED_D
    ):
        raise ValueError("Public static axes require N1=32, N2=1 and logical D=128")
    if tuple(weights.shape) != tuple(query.shape[:2]) or weights.dtype != torch.float32:
        raise ValueError("W must be FP32 with shape (T1,N1)")
    if query_dequant_scale is None:
        raise ValueError("Descale_q is a required xlsx input")
    expected_q_scale = (*query.shape[:2], LOGICAL_D // 64, 2)
    if tuple(query_dequant_scale.shape) != expected_q_scale:
        raise ValueError("Descale_q must have corrected xlsx shape (T1,N1,D/64,2)")
    if key_dequant_scale is None:
        raise ValueError("Descale_k is a required xlsx input")
    expected = tuple(key.shape[:2]) + (2, 2)
    if tuple(key_dequant_scale.shape) != expected:
        raise ValueError("Descale_k must have shape (T2,1,2,2)")
    key_stride0 = _tnd_axis0_stride(key, "k", (PACKED_D, 1), PACKED_D)
    key_dequant_scale_stride0 = _tnd_axis0_stride(
        key_dequant_scale, "descale_k", (4, 2, 1), 4
    )
    if key.device != query.device or key_dequant_scale.device != query.device:
        raise ValueError("Q, k and descale_k must share an NPU device")
    key_storage_rows = int(key.shape[0])
    key_storage = key
    key_dequant_scale_storage = key_dequant_scale
    if block_table is not None:
        raise ValueError("Block_table must be None for TND K")
    if candidate_topk_blocks == -1:
        if candidate_block_size != -1:
            raise ValueError(
                "Candidate_block_size must be -1 when candidate is disabled"
            )
    elif candidate_topk_blocks <= 0 or candidate_block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError("Candidate requires positive topk_blocks and block_size=8")
    # Keep this as a Python/JIT specialization constant. Disabled launches
    # compile out Candidate block-max, key generation, TopK and UB resources.
    candidate_enabled = int(candidate_topk_blocks) > 0
    pa_block_size = TILE_N
    if int(topk) < 1 or int(topk) > 8192:
        raise ValueError("Topk must be in ASC-supported range [1,8192]")
    if int(mask_mode) not in (0, 3):
        raise ValueError("Mask_mode must be 0 or 3")
    if int(cmp_ratio) < 1 or int(cmp_ratio) > 128:
        raise ValueError("Cmp_ratio must be in [1,128]")
    if query_dequant_scale.dtype != torch.uint8:
        raise TypeError("Descale_q must use uint8 E8M0 storage")
    if key_dequant_scale.dtype != torch.uint8:
        raise TypeError("Descale_k must use uint8 E8M0 storage")
    for name, tensor in (
        ("w", weights),
        ("descale_q", query_dequant_scale),
    ):
        if tensor.device != query.device:
            raise ValueError(f"{name} must be on the same NPU as q")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not query.is_contiguous():
        raise ValueError("Q must be contiguous")

    t = int(query.shape[0])
    if t <= 0:
        raise ValueError("Q must contain at least one T1 row")
    has_cu_seqlens_q = cu_seqlens_q is not None
    batch = int(cu_seqlens_q.numel()) - 1 if has_cu_seqlens_q else 1
    if batch <= 0:
        raise ValueError("Cu_seqlens_q must have shape (B+1,)")

    def validate_metadata(name, tensor, shape):
        if tensor is None:
            return
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)}")
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32")
        if tensor.device != query.device:
            raise ValueError(f"{name} must be on the same NPU as q")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if has_cu_seqlens_q:
        validate_metadata("cu_seqlens_q", cu_seqlens_q, (batch + 1,))
    validate_metadata("output_idx_offset", output_idx_offset, (t, 1))
    validate_metadata("seqused_q", seqused_q, (batch,))
    validate_metadata("seqused_k", seqused_k, (batch,))
    if int(mask_mode) == 3 and int(cmp_ratio) != 1:
        if cmp_residual_k is None:
            raise ValueError(
                "Cmp_residual_k is required for mask_mode=3 and cmp_ratio!=1"
            )
        validate_metadata("cmp_residual_k", cmp_residual_k, (batch,))
    elif cmp_residual_k is not None:
        raise ValueError(
            "Cmp_residual_k must be None unless mask_mode=3 and cmp_ratio!=1"
        )
    if cu_seqlens_k is None:
        if batch != 1:
            raise ValueError("Cu_seqlens_k is required for multiple TND batches")
        cu_seqlens_k = torch.tensor(
            [0, key_storage_rows], dtype=torch.int32, device=q.device
        )
    validate_metadata("cu_seqlens_k", cu_seqlens_k, (batch + 1,))
    effective_k = (
        seqused_k if seqused_k is not None else cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    )
    seqused_k = effective_k
    has_seqused_q = seqused_q is not None
    has_seqused_k = seqused_k is not None
    has_cmp_residual_k = cmp_residual_k is not None
    has_block_table = True
    logical_k_capacity = key_storage_rows
    tokens_pad = ceil_div(logical_k_capacity, TILE_N) * TILE_N
    if tokens_pad <= 0:
        raise ValueError("K storage must contain at least one token")
    # Optional xlsx tensors remain optional at the public boundary.  One
    # uninitialized carrier is sufficient because the JIT specialization
    # compiles out every load when the corresponding tensor is absent.
    dummy_i32 = torch.empty((1,), dtype=torch.int32, device=query.device)
    _dummy_table = torch.empty((1, 1), dtype=torch.int32, device=query.device)
    cu_kernel = cu_seqlens_q if has_cu_seqlens_q else dummy_i32
    seqused_q_kernel = seqused_q if has_seqused_q else dummy_i32
    seqused_k_kernel = seqused_k if has_seqused_k else dummy_i32
    cmp_residual_kernel = cmp_residual_k if has_cmp_residual_k else dummy_i32
    block_table_kernel = cu_seqlens_k

    max_query_length = int(max_seqlen_q) if int(max_seqlen_q) > 0 else t
    max_tasks_per_batch = ceil_div(max_query_length, tile_m // query_heads)
    # Workspace capacity; AICPU metadata owns the actual shard boundaries.
    split_count = max(1, min(MAX_MERGE_PARTS, STREAM_TOPK_TRUNK_LEN // int(topk)))
    if (
        metadata.dtype != torch.int32
        or metadata.ndim != 1
        or metadata.numel() != METADATA_ELEMENTS
        or not metadata.is_contiguous()
    ):
        raise ValueError(
            "Metadata must be contiguous int32 [1024] in ASC LI/LD boundary format"
        )
    if metadata.device != q.device:
        raise ValueError("Metadata must be on the same device as q")
    metadata = metadata.view(METADATA_ROWS, METADATA_FIELDS)
    worker_count = MAX_CUBE_WORKERS
    workspace_row_count = MAX_CUBE_WORKERS * (tile_m // query_heads)
    shard_pad = tokens_pad
    # Public outputs and GM scratch are allocation-only Torch calls.  The
    # fused kernel writes every public row, including invalid/padded rows.
    sparse_indices_2d = torch.empty(
        (64 * query_tile_rows, int(topk)), dtype=torch.int32, device=query.device
    )
    sparse_values_2d = torch.empty(
        (64 * query_tile_rows, int(topk)), dtype=torch.bfloat16, device=weights.device
    )
    candidate_count = max(0, int(candidate_topk_blocks))
    candidate_indices_2d = torch.empty(
        (64 * query_tile_rows, candidate_count),
        dtype=torch.int32,
        device=query.device,
    )
    candidate_length_1d = torch.empty(
        (t * split_count,), dtype=torch.int32, device=query.device
    )

    key_workspace = torch.empty(
        (workspace_row_count, shard_pad),
        dtype=torch.uint16,
        device=query.device,
    )
    candidate_key_workspace = torch.empty(
        (
            workspace_row_count,
            shard_pad // CANDIDATE_BLOCK_SIZE if candidate_enabled else 1,
        ),
        dtype=torch.uint16,
        device=query.device,
    )
    kernel_candidate_count = candidate_count if candidate_enabled else 1
    candidate_indices_kernel = (
        candidate_indices_2d
        if candidate_enabled
        else torch.empty((1, split_count), dtype=torch.int32, device=query.device)
    )
    candidate_value_workspace = torch.empty(
        (64 * query_tile_rows, kernel_candidate_count),
        dtype=torch.uint16,
        device=query.device,
    )

    # GM aliases describe native packed formats without device copies.
    query_address = query.data_ptr()
    query_scale_storage = query_dequant_scale.reshape(t * query_heads, 2, 2).view(
        torch.float8_e8m0fnu
    )

    final_sparse_indices = torch.empty(
        (t, int(topk)), dtype=torch.int32, device=query.device
    )
    final_sparse_values = torch.empty(
        (t, int(topk)), dtype=torch.bfloat16, device=query.device
    )
    final_candidate_indices = torch.empty(
        (t, kernel_candidate_count), dtype=torch.int32, device=query.device
    )
    final_candidate_bits = torch.empty(
        (t, kernel_candidate_count), dtype=torch.uint16, device=query.device
    )
    _get_tnd_compiled_fused_runner(
        int(topk),
        candidate_enabled,
        kernel_candidate_count,
        int(mask_mode),
        int(cmp_ratio),
        has_cu_seqlens_q,
        has_seqused_q,
        has_seqused_k,
        has_block_table,
        pa_block_size,
        split_count,
        query_heads,
        query_tile_rows,
    )(
        query_address,
        key_storage.data_ptr(),
        weights,
        query_scale_storage,
        key_dequant_scale_storage.data_ptr(),
        key_stride0,
        key_dequant_scale_stride0,
        cu_kernel,
        seqused_q_kernel,
        seqused_k_kernel,
        cmp_residual_kernel,
        block_table_kernel,
        key_workspace,
        candidate_key_workspace,
        sparse_indices_2d.data_ptr(),
        sparse_values_2d.view(torch.uint16).data_ptr(),
        candidate_indices_kernel.data_ptr(),
        candidate_value_workspace.data_ptr(),
        candidate_length_1d,
        worker_count,
        t,
        batch,
        logical_k_capacity,
        max_tasks_per_batch,
        key_storage_rows,
        key_dequant_scale_storage.shape[0],
        metadata,
        metadata.shape[0],
        final_sparse_indices.data_ptr(),
        final_sparse_values.view(torch.uint16).data_ptr(),
        final_candidate_indices.data_ptr(),
        final_candidate_bits.data_ptr(),
        int(return_value),
        min(64, t),
        0 if output_idx_offset is None else output_idx_offset.data_ptr(),
    )

    effective_k.record_stream(torch.npu.current_stream(q.device))

    sparse_indices_2d, sparse_values_2d = final_sparse_indices, final_sparse_values
    if candidate_enabled:
        candidate_indices_2d = final_candidate_indices
    candidate_length_1d = candidate_length_1d[:t]
    return (
        sparse_indices_2d.view(t, N2, int(topk)),
        sparse_values_2d.view(t, N2, int(topk))
        if return_value
        else sparse_values_2d.reshape(-1)[:0],
        candidate_indices_2d.view(t, N2, candidate_count)
        if candidate_enabled
        else candidate_indices_2d.reshape(-1),
        candidate_length_1d.view(t, N2)
        if candidate_enabled
        else candidate_length_1d[:0],
    )
