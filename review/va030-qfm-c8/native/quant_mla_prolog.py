# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A5 quant_mla_prolog fused Kernel, Torch interface and Native export."""

import json as _json
import math as _math
import os as _os
import struct as _struct
from enum import Enum as _Enum
from functools import lru_cache as _lru_cache
from pathlib import Path as _Path

import cannbotdsl as _dsl
import cannbotdsl.aot as _aot
from cannbotdsl import Tensor as _Tensor, dtypes as _dtypes
import torch as _torch
import torch_npu as _torch_npu

import cannbotdsl as dsl
from cannbotdsl import (
    MemLoc,
    ProfileSpec,
    Tensor,
    cv_valid_extent,
    dtypes,
    jit,
    kernel,
    vf,
)
from cannbotdsl.buffer import Buffer as _DslBuffer
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options as _copy_options
from cannbotdsl.ops.memcpy import mem_copy as _dsl_mem_copy
from cannbotdsl.ops.reg import RegLayout
from cannbotdsl.ops.sync import (
    cube_sync_block_arrive,
    onchip_rewind,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
)
from cannbotdsl.ops import sync as _sync_ops
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import make_layout, reinterpret as _dsl_reinterpret, tile_slice


_NATIVE_DIRECTORY = _Path(__file__).resolve().parent / "_native"
if __name__ == "quant_mla_prolog" and _NATIVE_DIRECTORY.is_dir():
    _aot.register_directory(_NATIVE_DIRECTORY)


STRIPE_PARTS = 6
COLUMNS_PER_PART = 128
STRIPE_COLUMNS = STRIPE_PARTS * COLUMNS_PER_PART
REDUCTION_PANEL = 256
MX_GROUP = 32
SCALE_PAIR = 2
NZ_COLUMNS = 16

PROJECTION_ROWS = 16
PROJECTION_COLUMNS = 64
PROJECTION_REDUCTION = 128
QA_PROJECTION_COLUMNS = 128
QA_PROJECTION_REDUCTION = 256
KVA_PROJECTION_COLUMNS = 64
KVA_PROJECTION_REDUCTION = 256
WQB_PROJECTION_REDUCTION = 256
WQB_L0_REDUCTION = 128
WQB_WIDE_COLUMNS = 512
WQB_WIDE_MAX_ROWS = 32
WQB_WIDE_MIN_COLUMN_TASKS = 32
WQB_HEAD_L1_REDUCTION = 512
WQB_TAIL_HEAD_ROWS = 80
WQB_LARGE_HEAD_ROWS = 128
WQB_SINGLE_TAIL_ROWS = 144
MX_GROUP_ELEMENTS = 32
MAX_REDUCTION_ELEMENTS = 8192
MX_SCALE_PAIR = 2
NZ_SCALE_COLUMNS = 16
VECTOR_FLOAT_ELEMENTS = 64
VECTOR_SUBBLOCKS = 2
HEAD_READY_EVENT = 3
HEAD_CONSUMED_EVENT = 4
PREPARE_READY_EVENT = 5
HEAD_PIPELINE_DEPTH = 2
COMPACT_HANDOFF_MAX_ROWS = 128
COMPACT_HANDOFF_MAX_SLOTS = 8
COMPACT_BOUNDARY_REDUCTION_PANEL = 128
CROSS_CORE_EVENT_COUNT = 16
MIX_JOIN_INPUT_EVENT = 0
MIX_JOIN_GRID_EVENT = 1
MIX_JOIN_OUTPUT_EVENT = 2
KVA_READY_GRID_EVENT = 6
KVA_READY_VECTOR_EVENT = 7
QR_READY_CUBE_EVENT = 8
FP32_EXPONENT_SHIFT = 23
FP32_EXPONENT_MASK = 255
E4M3_EXPONENT_HEADROOM = 8
E4M3_MAX_VALUE = 448.0
HIF8_MAX_VALUE = 32768.0
MAXIMUM_PARTIAL_ELEMENTS = 8
MAXIMUM_PARALLEL_ROWS = 16
HIF_QR_RESIDENT_ROWS = 2
HIF_QR_EXTENDED_ROWS = 4
HIF_PREPARE_BATCH_ROWS = 8
HIF_WQB_L1_REDUCTION = 512
WKB_WEIGHT_L2_CACHE_CTL = 4
MX_NOPE_L1_HANDOFF = True
HIF_QR_ROWS_PER_VECTOR = 8
BF16_RNA_BIAS = 32768
BF16_FP32_MASK = 4294901760
BF16_MAX_VALUE = 3.3895313892515355e38
WKB_C2V_MIN_ROWS = 16
WKB_C2V_MAX_ROWS = 64
WKB_DIRECT_MAX_ROWS = 96
WKB_C2V_COLUMNS = 256
WKB_C2V_REDUCTION = 64
LATENT = 512
VECTOR = 64
E4_MAX = 448.0
KV_ELEMENTS = 512
GROUP_ELEMENTS = 128
GROUP_COUNT = KV_ELEMENTS // GROUP_ELEMENTS
FLOAT_BYTES = 4
BF16_BYTES = 2
FRACTAL_BYTES = 32
VECTOR_ELEMENTS = 64
E4M3_MAX = 448.0
CACHE_AMAX_FLOOR = 0.0001


def _packed_strides(shape):
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    return tuple(strides)


def _buffer(mem_loc, shape, dtype, *, capacity=None, data_format=None):
    physical_layout = make_layout(capacity) if capacity is not None else None
    layout = make_layout(
        shape,
        stride=_packed_strides(capacity) if capacity is not None else None,
    )
    return _DslBuffer(
        mem_loc,
        dtype,
        layout,
        physical_layout=physical_layout,
        data_format=data_format,
    )


def _ring_buffer(
    mem_loc,
    shape,
    dtype,
    *,
    depth,
    capacity=None,
    data_format=None,
    cross_core=False,
    raw_bytes=False,
):
    logical_shape = shape
    view_format = data_format
    if data_format == "zn":
        if mem_loc == MemLoc.L0B:
            shape = (shape[1], shape[0])
            if capacity is not None:
                capacity = (capacity[1], capacity[0])
            data_format = None
            view_format = None
        else:
            data_format = "nz"
    if not isinstance(cross_core, bool):
        raise TypeError("cross_core must be a compile-time bool")
    if (
        (view_format == "zn" or raw_bytes and view_format == "nz")
        and dtype == dtypes.float8_e8m0
        and mem_loc == MemLoc.L1
    ):
        storage_shape = capacity if capacity is not None else logical_shape
        if view_format == "zn":
            storage_elements = (
                ((storage_shape[0] + 15) // 16) * ((storage_shape[1] + 1) // 2) * 32
            )
        else:
            storage_elements = (
                ((storage_shape[1] + 15) // 16) * ((storage_shape[0] + 1) // 2) * 32
            )
        storage_bytes = storage_elements * (dtype.width // 8)
        ring = RingBuffer(
            mem_loc,
            dtypes.uint8,
            make_layout((storage_bytes,)),
            depth=depth,
            data_format="nd",
        )
        return _RingBufferView(ring, logical_shape, dtype, view_format)
    physical_layout = None
    layout = make_layout(shape)
    if capacity is not None:
        if data_format == "nd" or (data_format is None and mem_loc == MemLoc.UB):
            physical_layout = make_layout(capacity)
            layout = make_layout(shape, stride=_packed_strides(capacity))
        else:
            shape = capacity
            layout = make_layout(shape)
    ring = RingBuffer(
        mem_loc,
        dtype,
        layout,
        depth=depth,
        physical_layout=physical_layout,
        data_format=data_format,
    )
    return ring


class _RingBufferView:
    def __init__(self, ring, shape, dtype, data_format):
        self._ring = ring
        self._shape = shape
        self._dtype = dtype
        self._data_format = data_format

    def next_write(self):
        return _view(
            self._ring.next_write(),
            dtype=self._dtype,
            shape=self._shape,
            data_format=self._data_format,
        )

    def next_read(self):
        return _view(
            self._ring.next_read(),
            dtype=self._dtype,
            shape=self._shape,
            data_format=self._data_format,
        )

    def next_write_raw(self):
        return self._ring.next_write()


def _view(
    value,
    dtype=None,
    shape=None,
    *,
    layout=None,
    physical_layout=None,
    offset=0,
    data_format=None,
):
    if data_format == "zn" and value.memloc == MemLoc.L0B:
        if shape is None:
            raise ValueError("ZN migration requires an explicit shape")
        shape = (shape[1], shape[0])
        data_format = "nz"
    if shape is not None:
        if layout is not None:
            raise ValueError("shape and layout are mutually exclusive")
        layout = make_layout(shape)
    return _dsl_reinterpret(
        value,
        dtype=dtype,
        layout=layout,
        physical_layout=physical_layout,
        offset=offset,
        data_format=data_format,
    )


def _copy_plan(*, format_transform=None, split_axis=None):
    if format_transform not in (
        None,
        "identity",
        "nd2nz",
        "mx_scale_and",
        "mx_scale_bdn",
    ):
        raise ValueError(f"unsupported copy transform: {format_transform}")
    if split_axis not in (None, 0, 1):
        raise ValueError(f"unsupported split axis: {split_axis}")
    return format_transform, split_axis


def _accumulator(ring, rows, columns):
    slot = ring.next_write(bare=True)
    view = _view(
        slot.view(),
        shape=(rows, columns),
        physical_layout=make_layout(((columns + 15) // 16, (rows + 15) // 16, 16, 16)),
    )
    return slot, view


def _copy(
    destination,
    source,
    *,
    plan=None,
    transpose=False,
    mx_scale=None,
    l2_cache_ctl=None,
    actual_shape=None,
    part_id=None,
    dst_subblock=None,
    unit_flag=None,
):
    if transpose:
        source = source.permute(1, 0)
    if plan is not None:
        transform, split_axis = plan
        if transform == "mx_scale_and":
            source = source.flatten(1, 2)
        elif transform == "mx_scale_bdn":
            source = source.permute(1, 2, 0).flatten(0, 1)
        if actual_shape is not None:
            source = _view(source, shape=actual_shape)
        if split_axis is not None:
            if part_id is None:
                source = source.partition(dim=split_axis)
            else:
                destination = destination.partition(dim=split_axis, part_id=part_id)
    if source.memloc == MemLoc.L0C and destination.memloc == MemLoc.GM:
        # DSL 0.8 FIXPIPE uses the source domain. Clip to the GM tile's
        # valid extent without changing the accumulator's physical pitch.
        source = _view(
            source, shape=destination.shape, physical_layout=source.physical_layout
        )
    if source.memloc == MemLoc.UB and destination.memloc == MemLoc.GM:
        if source.rank == destination.rank:
            source = _view(
                source,
                layout=make_layout(destination.shape, stride=source.stride),
                physical_layout=source.physical_layout,
            )
    options = None
    if any(
        value is not None for value in (mx_scale, l2_cache_ctl, dst_subblock, unit_flag)
    ):
        option_values = {
            "mx_scale": mx_scale,
            "l2_cache_ctl": l2_cache_ctl,
            "dst_subblock": dst_subblock,
            "unit_flag": unit_flag,
        }
        options = _copy_options(
            **{
                name: value
                for name, value in option_values.items()
                if value is not None
            }
        )
    _dsl_mem_copy(destination, source, options=options)


@jit
def _mix_join(wait_vectors: bool = True):
    # The initial projection has no preceding Vector writes to join.
    _sync_ops.cube_sync_all()
    if dsl.const_expr(wait_vectors):
        vec_sync_all()
        vec_sync_block_arrive(_sync_ops.PIPE.MTE3, MIX_JOIN_INPUT_EVENT, mode=2)
        _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, MIX_JOIN_INPUT_EVENT, mode=2)
    cube_sync_block_arrive(_sync_ops.PIPE.FIXPIPE, MIX_JOIN_GRID_EVENT, mode=0)
    _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, MIX_JOIN_GRID_EVENT, mode=0)
    cube_sync_block_arrive(_sync_ops.PIPE.MTE3, MIX_JOIN_OUTPUT_EVENT, mode=2)
    vec_sync_block_wait(_sync_ops.PIPE.S, MIX_JOIN_OUTPUT_EVENT, mode=2)


@jit
def _publish_kva_and_qr():
    _sync_ops.cube_sync_all()
    cube_sync_block_arrive(_sync_ops.PIPE.FIXPIPE, KVA_READY_GRID_EVENT, mode=0)
    # CANN 9.2.1 omits AIC MTE3 waits; use S to retain the FIXPIPE dependency.
    _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, KVA_READY_GRID_EVENT, mode=0)
    cube_sync_block_arrive(_sync_ops.PIPE.MTE3, KVA_READY_VECTOR_EVENT, mode=2)
    vec_sync_all()
    vec_sync_block_arrive(_sync_ops.PIPE.MTE3, QR_READY_CUBE_EVENT, mode=2)
    vec_sync_block_wait(_sync_ops.PIPE.S, KVA_READY_VECTOR_EVENT, mode=2)


@jit
def _wait_qr_for_wqb():
    _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, QR_READY_CUBE_EVENT, mode=2)
    cube_sync_block_arrive(_sync_ops.PIPE.FIXPIPE, MIX_JOIN_GRID_EVENT, mode=0)
    _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, MIX_JOIN_GRID_EVENT, mode=0)


@jit
def _quantize_cast(value, mask, hif):
    """SAT semantics at the quantization point without a kernel-wide SAT mode.

    The target DSL configures FP8 SAT through a kernel-wide CTRL bit. In a fused
    program that also changes how the preceding MX Matmul consumes input
    NaN/E8M0 special values. Clamp only the cast operand here, never a Matmul
    or RMSNorm operand, and keep the hardware's requested rounding mode.
    """
    maximum = E4M3_MAX_VALUE
    dtype = dtypes.float8_e4m3fn
    rounding = rr.RoundingMode.RN
    if dsl.const_expr(hif):
        maximum = HIF8_MAX_VALUE
        dtype = dtypes.hifloat8
        rounding = rr.RoundingMode.RNA
    zero = rr.vdups(0.0, dtypes.float32, mask=mask)
    finite = rr.vselect(value, zero, cond_mask=rr.veq(value, value, mask=mask))
    finite = rr.vmins(rr.vmaxs(finite, -maximum, mask=mask), maximum, mask=mask)
    if dsl.const_expr(hif):
        # HiF8 has no negative zero (0x80 is NaN). Smallest magnitude is
        # 2^-22; RNA rounds strictly below its half-way point to +0.
        finite = rr.vselect(
            zero,
            finite,
            cond_mask=rr.vlts(rr.vabs(finite, mask=mask), 2.0**-23, mask=mask),
        )
    return rr.vcast(
        finite,
        dtype,
        mask=mask,
        rounding=rounding,
        saturate=False,
        reg_layout=RegLayout.ZERO,
    )


QA_PIPELINE_COLUMNS = 128
QA_PIPELINE_REDUCTION = 256
QA_BALANCED_COLUMNS = 96
QA_BALANCED_MAX_ROWS = 32
QA_BALANCED_L1_REDUCTION = 512
QA_L1_REDUCTION = 1024


class QaRowPipeline:
    def __init__(
        self,
        rows,
        width,
        reduction,
        mode,
        nz,
        bm,
        splits,
        row_major=False,
        columns=QA_PIPELINE_COLUMNS,
    ):
        self.rows = rows
        self.width = width
        self.reduction = reduction
        self.mode = mode
        self.nz = nz
        self.bm = bm
        self.splits = splits
        self.row_major = row_major
        self.columns = columns
        self.nc = width // self.columns
        self.nm = (rows + bm - 1) // bm
        self.nt = self.nm * self.nc * splits
        self.l1_reduction = QA_PIPELINE_REDUCTION

    @jit
    def project(
        self,
        x: Tensor,
        weight: Tensor,
        descale_activation: Tensor,
        descale_weight: Tensor,
        partial: Tensor,
    ):
        dtype = dtypes.float8_e4m3fn
        cube_dtype = dtypes.float8_e4m3fn
        if dsl.const_expr(self.mode == 2):
            dtype = dtypes.uint8
            cube_dtype = dtypes.hifloat8
        a1 = _ring_buffer(
            MemLoc.L1, (self.bm, self.l1_reduction), dtype, depth=2, data_format="nz"
        )
        b1 = _ring_buffer(
            MemLoc.L1,
            (self.columns, self.l1_reduction),
            dtype,
            depth=2,
            data_format="nz",
        )
        a0 = _ring_buffer(
            MemLoc.L0A,
            (self.bm, QA_PIPELINE_REDUCTION),
            cube_dtype,
            depth=2,
            data_format="nz",
        )
        b0 = _ring_buffer(
            MemLoc.L0B,
            (self.columns, QA_PIPELINE_REDUCTION),
            cube_dtype,
            depth=2,
            data_format="nz",
        )
        c0 = _ring_buffer(MemLoc.L0C, (self.bm, self.columns), dtypes.float32, depth=1)
        ae = _copy_plan(format_transform="nd2nz")
        be = _copy_plan(format_transform="nd2nz")
        if dsl.const_expr(self.nz):
            be = _copy_plan(format_transform="identity")
        if dsl.const_expr(self.mode == 1):
            sa = _ring_buffer(
                MemLoc.L1,
                (self.bm, self.reduction // 32),
                dtypes.float8_e8m0,
                depth=1,
                data_format="zn",
            )
            sb = _ring_buffer(
                MemLoc.L1,
                (self.reduction // 32, self.columns),
                dtypes.float8_e8m0,
                depth=1,
                data_format="nz",
                raw_bytes=self.nz,
            )
            sae = _copy_plan(format_transform="mx_scale_and")
            sbe = _copy_plan(format_transform="mx_scale_bdn")
        # Small-T: fixed K+N owner per core, M tiles advance locally.
        task_start = get_block_idx()
        task_stride = get_block_num()
        for task in range(task_start, self.nt, task_stride):
            split = task % self.splits
            tile = task // self.splits
            mt = tile // self.nc
            nt = tile % self.nc
            valid_rows = min(self.bm, self.rows - mt * self.bm)
            acc_slot, acc = _accumulator(c0, valid_rows, self.columns)
            sa_full = None
            sb_full = None
            if dsl.const_expr(self.mode == 1):
                _copy(
                    _view(sa.next_write(), shape=(valid_rows, self.reduction // 32)),
                    tile_slice(
                        descale_activation,
                        dsl.make_tiler(
                            (self.bm, self.reduction // 64, 2),
                            alignment=(self.bm, 2, 2),
                        ),
                        (mt, 0, 0),
                    ),
                    plan=sae,
                )
                if dsl.const_expr(self.nz):
                    pw = self.reduction // 64 * 16 * 2
                    packed = descale_weight.view((self.width // 16, pw))
                    _copy(
                        _view(
                            sb.next_write_raw(),
                            shape=(self.columns // 16, pw),
                            data_format="nd",
                        ),
                        tile_slice(packed, (self.columns // 16, pw), (nt, 0)),
                    )
                else:
                    _copy(
                        sb.next_write(),
                        tile_slice(
                            descale_weight,
                            dsl.make_tiler(
                                (self.columns, self.reduction // 64, 2),
                                alignment=(self.columns, 2, 2),
                            ),
                            (nt, 0, 0),
                        ),
                        plan=sbe,
                    )
                sa_full = _view(
                    sa.next_read(), shape=(valid_rows, self.reduction // 32)
                )
                sb_full = sb.next_read()
            start = split * (self.reduction // self.l1_reduction) // self.splits
            stop = (split + 1) * (self.reduction // self.l1_reduction) // self.splits
            for kt in range(start, stop + 1):
                if kt < stop:
                    _copy(
                        _view(a1.next_write(), shape=(valid_rows, self.l1_reduction)),
                        tile_slice(x, (self.bm, self.l1_reduction), (mt, kt)),
                        plan=ae,
                    )
                    _copy(
                        b1.next_write(),
                        tile_slice(weight, (self.columns, self.l1_reduction), (nt, kt)),
                        plan=be,
                    )
                if kt > start:
                    av = a1.next_read()
                    bv = b1.next_read()
                    if dsl.const_expr(self.l1_reduction > QA_PIPELINE_REDUCTION):
                        for inner in range(self.l1_reduction // QA_PIPELINE_REDUCTION):
                            scale_tile = (kt - 1) * (
                                self.l1_reduction // QA_PIPELINE_REDUCTION
                            ) + inner
                            _copy(
                                _view(
                                    a0.next_write(),
                                    shape=(valid_rows, QA_PIPELINE_REDUCTION),
                                ),
                                _view(
                                    tile_slice(
                                        av, (self.bm, QA_PIPELINE_REDUCTION), (0, inner)
                                    ),
                                    shape=(valid_rows, QA_PIPELINE_REDUCTION),
                                ),
                                mx_scale=tile_slice(
                                    sa_full,
                                    (self.bm, QA_PIPELINE_REDUCTION // 32),
                                    (0, scale_tile),
                                ),
                            )
                            _copy(
                                b0.next_write(),
                                tile_slice(
                                    bv,
                                    (self.columns, QA_PIPELINE_REDUCTION),
                                    (0, inner),
                                ),
                                mx_scale=tile_slice(
                                    sb_full,
                                    (QA_PIPELINE_REDUCTION // 32, self.columns),
                                    (scale_tile, 0),
                                ),
                            )
                            matmul(
                                acc_slot,
                                _view(
                                    a0.next_read(),
                                    shape=(valid_rows, QA_PIPELINE_REDUCTION),
                                ),
                                b0.next_read(),
                                init=(kt == start + 1 and inner == 0),
                                mx_mode=True,
                            )
                    elif dsl.const_expr(self.mode == 1):
                        _copy(
                            _view(
                                a0.next_write(),
                                shape=(valid_rows, QA_PIPELINE_REDUCTION),
                            ),
                            _view(av, shape=(valid_rows, QA_PIPELINE_REDUCTION)),
                            mx_scale=tile_slice(
                                sa_full,
                                (self.bm, QA_PIPELINE_REDUCTION // 32),
                                (0, kt - 1),
                            ),
                        )
                        _copy(
                            b0.next_write(),
                            bv,
                            mx_scale=tile_slice(
                                sb_full,
                                (QA_PIPELINE_REDUCTION // 32, self.columns),
                                (kt - 1, 0),
                            ),
                        )
                    else:
                        _copy(
                            _view(
                                a0.next_write(),
                                shape=(valid_rows, QA_PIPELINE_REDUCTION),
                            ),
                            _view(
                                _view(av, dtype=dtypes.hifloat8),
                                shape=(valid_rows, QA_PIPELINE_REDUCTION),
                            ),
                        )
                        _copy(b0.next_write(), _view(bv, dtype=dtypes.hifloat8))
                    if dsl.const_expr(self.l1_reduction == QA_PIPELINE_REDUCTION):
                        matmul(
                            acc_slot,
                            _view(
                                a0.next_read(),
                                shape=(valid_rows, QA_PIPELINE_REDUCTION),
                            ),
                            b0.next_read(),
                            init=kt == start + 1,
                            mx_mode=self.mode == 1,
                        )
            if dsl.const_expr(self.row_major):
                row_major_output = partial
                if dsl.const_expr(self.splits > 1):
                    row_major_output = partial.view(
                        (self.splits * self.nm * self.bm, self.width)
                    )
                dest = tile_slice(
                    row_major_output,
                    (self.bm, self.columns),
                    (split * self.nm + mt, nt),
                )
            else:
                dest = tile_slice(partial, (1, self.bm * self.columns), (task, 0)).view(
                    (self.bm, self.columns)
                )
            c0.next_read()
            _copy(dest[0:valid_rows, None], acc)


class _WkbProjection:
    def __init__(self, tile_rows):
        self.tile_rows = tile_rows
        self.tile_columns = (
            128 if tile_rows == WQB_SINGLE_TAIL_ROWS else WKB_C2V_COLUMNS
        )
        self.vector_rows = tile_rows // VECTOR_SUBBLOCKS

    @jit
    def run_direct(
        self,
        q_nope: Tensor,
        wkb: Tensor,
        q: Tensor,
        publish_heads: bool = False,
        coalesce_b: bool = False,
        token_major_output: bool = False,
        input_prepared: bool = False,
        head_group: int = 1,
        shared_l1_b=None,
        shared_l0_b=None,
        shared_l0_c=None,
        nope_l1=None,
    ):
        if dsl.const_expr(shared_l1_b is None):
            self.run_direct_original(
                q_nope,
                wkb,
                q,
                publish_heads,
                coalesce_b,
                token_major_output,
                input_prepared,
                head_group,
            )
            return
        N = wkb.shape[0]
        reduction = wkb.shape[1]
        rows = q_nope.shape[0] // N
        row_tiles = (rows + self.tile_rows - 1) // self.tile_rows
        a_l1 = _ring_buffer(
            MemLoc.L1, (self.tile_rows, WKB_C2V_REDUCTION), dtypes.bfloat16, depth=2
        )
        b_l1 = shared_l1_b
        a_l0 = _ring_buffer(
            MemLoc.L0A, (self.tile_rows, WKB_C2V_REDUCTION), dtypes.bfloat16, depth=2
        )
        b_l0 = shared_l0_b
        if dsl.const_expr(shared_l0_b is None):
            b_l0 = _ring_buffer(
                MemLoc.L0B,
                (WKB_C2V_REDUCTION, self.tile_columns),
                dtypes.bfloat16,
                depth=2,
                data_format="zn",
            )
        c_l0 = shared_l0_c
        if dsl.const_expr(shared_l0_c is None):
            c_l0 = _ring_buffer(
                MemLoc.L0C,
                (self.tile_rows, self.tile_columns),
                dtypes.float32,
                depth=2 if self.tile_rows <= WKB_C2V_MAX_ROWS else 1,
            )
        load_engine = _copy_plan(format_transform="nd2nz")
        input_tokens = q_nope.view((rows, N * reduction))
        published_heads = 0
        for group_task in range(
            get_block_idx(), N // head_group * row_tiles, get_block_num()
        ):
            group_nope = None
            if dsl.const_expr(nope_l1 is not None):
                group_nope = _view(
                    nope_l1.next_read(),
                    shape=(self.tile_rows, head_group * reduction),
                    data_format="nz",
                )
            for local_head in range(head_group):
                task = group_task * head_group + local_head
                row_tile = task // N
                head = task % N
                # Earlier Cast completions were consumed by the WQB credit window.
                if dsl.const_expr(publish_heads):
                    if published_heads >= HEAD_PIPELINE_DEPTH:
                        _sync_ops.cube_sync_block_wait(
                            _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                        )
                    if dsl.const_expr(not input_prepared):
                        _sync_ops.cube_sync_block_wait(
                            _sync_ops.PIPE.S, PREPARE_READY_EVENT, mode=2
                        )
                elif dsl.const_expr(not input_prepared):
                    if task + HEAD_PIPELINE_DEPTH * get_block_num() >= N * row_tiles:
                        _sync_ops.cube_sync_block_wait(
                            _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                        )
                head_weight = wkb[head, None, None].view((reduction, KV_ELEMENTS))
                if dsl.const_expr(publish_heads and not token_major_output):
                    output_heads = q.view((N, rows, KV_ELEMENTS))
                else:
                    output_heads = dsl.permute(q.view((rows, N, q.shape[1])), (1, 0, 2))
                # Keep the token stride, including other heads and the RoPE tail.
                head_output = output_heads[head, None, 0:KV_ELEMENTS]
                _copy(
                    _view(
                        b_l1.next_write(),
                        dtype=dtypes.bfloat16,
                        shape=(reduction, self.tile_columns),
                        data_format="nz",
                    ),
                    tile_slice(head_weight, (reduction, self.tile_columns), (0, 0)),
                    plan=load_engine,
                    l2_cache_ctl=WKB_WEIGHT_L2_CACHE_CTL,
                )
                for column_tile in range(KV_ELEMENTS // self.tile_columns):
                    output_tile = tile_slice(
                        head_output,
                        (self.tile_rows, self.tile_columns),
                        (row_tile, column_tile),
                    )
                    # MMAD and Fixpipe must share the tail's packed NZ row stride.
                    acc_slot, acc = _accumulator(
                        c_l0, output_tile.shape[0], self.tile_columns
                    )
                    if column_tile + 1 < KV_ELEMENTS // self.tile_columns:
                        _copy(
                            _view(
                                b_l1.next_write(),
                                dtype=dtypes.bfloat16,
                                shape=(reduction, self.tile_columns),
                                data_format="nz",
                            ),
                            tile_slice(
                                head_weight,
                                (reduction, self.tile_columns),
                                (0, column_tile + 1),
                            ),
                            plan=load_engine,
                            l2_cache_ctl=WKB_WEIGHT_L2_CACHE_CTL,
                        )
                    resident_b = _view(
                        b_l1.next_read(),
                        dtype=dtypes.bfloat16,
                        shape=(reduction, self.tile_columns),
                        data_format="nz",
                    )
                    for k_tile in range(reduction // WKB_C2V_REDUCTION):
                        if dsl.const_expr(nope_l1 is None):
                            _copy(
                                a_l1.next_write(),
                                tile_slice(
                                    input_tokens,
                                    (self.tile_rows, WKB_C2V_REDUCTION),
                                    (
                                        row_tile,
                                        head * (reduction // WKB_C2V_REDUCTION)
                                        + k_tile,
                                    ),
                                ),
                                plan=load_engine,
                            )
                        a_write = _view(
                            a_l0.next_write(),
                            shape=(output_tile.shape[0], WKB_C2V_REDUCTION),
                        )
                        if dsl.const_expr(nope_l1 is None):
                            _copy(a_write, a_l1.next_read())
                        else:
                            _copy(
                                a_write,
                                tile_slice(
                                    group_nope,
                                    (self.tile_rows, WKB_C2V_REDUCTION),
                                    (
                                        0,
                                        local_head * (reduction // WKB_C2V_REDUCTION)
                                        + k_tile,
                                    ),
                                ),
                            )
                        _copy(
                            _view(
                                b_l0.next_write(),
                                dtype=dtypes.bfloat16,
                                shape=(WKB_C2V_REDUCTION, self.tile_columns),
                                data_format="zn",
                            ),
                            tile_slice(
                                resident_b,
                                (WKB_C2V_REDUCTION, self.tile_columns),
                                (k_tile, 0),
                            ),
                            transpose=True,
                        )
                        matmul(
                            acc_slot,
                            _view(
                                a_l0.next_read(),
                                shape=(output_tile.shape[0], WKB_C2V_REDUCTION),
                            ),
                            _view(
                                b_l0.next_read(),
                                dtype=dtypes.bfloat16,
                                shape=(WKB_C2V_REDUCTION, self.tile_columns),
                                data_format="zn",
                            ),
                            init=k_tile == 0,
                        )
                    c_l0.next_read()
                    _copy(output_tile, acc)
                if dsl.const_expr(publish_heads):
                    cube_sync_block_arrive(
                        _sync_ops.PIPE.FIXPIPE, HEAD_READY_EVENT, mode=2
                    )
                    published_heads += 1
        if dsl.const_expr(publish_heads):
            for pending in range(min(published_heads, HEAD_PIPELINE_DEPTH)):
                _sync_ops.cube_sync_block_wait(
                    _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                )

    @jit
    def run_direct_original(
        self,
        q_nope: Tensor,
        wkb: Tensor,
        q: Tensor,
        publish_heads: bool = False,
        coalesce_b: bool = False,
        token_major_output: bool = False,
        input_prepared: bool = False,
        head_group: int = 1,
    ):
        if dsl.const_expr(coalesce_b):
            self.run_direct_resident_b(
                q_nope, wkb, q, publish_heads, input_prepared, head_group
            )
            return
        N = wkb.shape[0]
        reduction = wkb.shape[1]
        rows = q_nope.shape[0] // N
        row_tiles = (rows + self.tile_rows - 1) // self.tile_rows
        a_l1 = _ring_buffer(
            MemLoc.L1, (self.tile_rows, WKB_C2V_REDUCTION), dtypes.bfloat16, depth=2
        )
        b_l1 = _ring_buffer(
            MemLoc.L1, (WKB_C2V_REDUCTION, self.tile_columns), dtypes.bfloat16, depth=2
        )
        a_l0 = _ring_buffer(
            MemLoc.L0A, (self.tile_rows, WKB_C2V_REDUCTION), dtypes.bfloat16, depth=2
        )
        b_l0 = _ring_buffer(
            MemLoc.L0B,
            (WKB_C2V_REDUCTION, self.tile_columns),
            dtypes.bfloat16,
            depth=2,
            data_format="zn",
        )
        c_l0 = _ring_buffer(
            MemLoc.L0C,
            (self.tile_rows, self.tile_columns),
            dtypes.float32,
            depth=2 if self.tile_rows <= WKB_C2V_MAX_ROWS else 1,
        )
        load_engine = _copy_plan(format_transform="nd2nz")
        input_tokens = q_nope.view((rows, N * reduction))
        published_heads = 0
        for group_task in range(
            get_block_idx(), N // head_group * row_tiles, get_block_num()
        ):
            for local_head in range(head_group):
                task = group_task * head_group + local_head
                row_tile = task // N
                head = task % N
                # Earlier Cast completions were consumed by the WQB credit window.
                if dsl.const_expr(publish_heads):
                    if published_heads >= HEAD_PIPELINE_DEPTH:
                        _sync_ops.cube_sync_block_wait(
                            _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                        )
                    if dsl.const_expr(not input_prepared):
                        _sync_ops.cube_sync_block_wait(
                            _sync_ops.PIPE.S, PREPARE_READY_EVENT, mode=2
                        )
                else:
                    if task + HEAD_PIPELINE_DEPTH * get_block_num() >= N * row_tiles:
                        _sync_ops.cube_sync_block_wait(
                            _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                        )
                head_weight = wkb[head, None, None].view((reduction, KV_ELEMENTS))
                if dsl.const_expr(publish_heads and not token_major_output):
                    output_heads = q.view((N, rows, KV_ELEMENTS))
                else:
                    output_heads = dsl.permute(q.view((rows, N, q.shape[1])), (1, 0, 2))
                # Keep the token stride, including other heads and the RoPE tail.
                head_output = output_heads[head, None, 0:KV_ELEMENTS]
                for column_tile in range(KV_ELEMENTS // self.tile_columns):
                    output_tile = tile_slice(
                        head_output,
                        (self.tile_rows, self.tile_columns),
                        (row_tile, column_tile),
                    )
                    # MMAD and Fixpipe must share the tail's packed NZ row stride.
                    acc_slot, acc = _accumulator(
                        c_l0, output_tile.shape[0], self.tile_columns
                    )
                    for k_tile in range(reduction // WKB_C2V_REDUCTION):
                        _copy(
                            a_l1.next_write(),
                            tile_slice(
                                input_tokens,
                                (self.tile_rows, WKB_C2V_REDUCTION),
                                (
                                    row_tile,
                                    head * (reduction // WKB_C2V_REDUCTION) + k_tile,
                                ),
                            ),
                            plan=load_engine,
                        )
                        _copy(
                            b_l1.next_write(),
                            tile_slice(
                                head_weight,
                                (WKB_C2V_REDUCTION, self.tile_columns),
                                (k_tile, column_tile),
                            ),
                            plan=load_engine,
                            l2_cache_ctl=WKB_WEIGHT_L2_CACHE_CTL,
                        )
                        _copy(
                            _view(
                                a_l0.next_write(),
                                shape=(output_tile.shape[0], WKB_C2V_REDUCTION),
                            ),
                            a_l1.next_read(),
                        )
                        _copy(b_l0.next_write(), b_l1.next_read(), transpose=True)
                        matmul(
                            acc_slot,
                            _view(
                                a_l0.next_read(),
                                shape=(output_tile.shape[0], WKB_C2V_REDUCTION),
                            ),
                            b_l0.next_read(),
                            init=k_tile == 0,
                        )
                    c_l0.next_read()
                    _copy(output_tile, acc)
                if dsl.const_expr(publish_heads):
                    cube_sync_block_arrive(
                        _sync_ops.PIPE.FIXPIPE, HEAD_READY_EVENT, mode=2
                    )
                    published_heads += 1
        if dsl.const_expr(publish_heads):
            for pending in range(min(published_heads, HEAD_PIPELINE_DEPTH)):
                _sync_ops.cube_sync_block_wait(
                    _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                )

    @jit
    def run_direct_resident_b(
        self,
        q_nope: Tensor,
        wkb: Tensor,
        q: Tensor,
        publish_heads: bool = False,
        input_prepared: bool = False,
        head_group: int = 1,
    ):
        N = wkb.shape[0]
        reduction = wkb.shape[1]
        rows = q_nope.shape[0] // N
        row_tiles = (rows + self.tile_rows - 1) // self.tile_rows
        a_l1 = _ring_buffer(
            MemLoc.L1, (self.tile_rows, WKB_C2V_REDUCTION), dtypes.bfloat16, depth=2
        )
        b_l1 = _ring_buffer(
            MemLoc.L1, (reduction, self.tile_columns), dtypes.bfloat16, depth=2
        )
        a_l0 = _ring_buffer(
            MemLoc.L0A, (self.tile_rows, WKB_C2V_REDUCTION), dtypes.bfloat16, depth=2
        )
        b_l0 = _ring_buffer(
            MemLoc.L0B,
            (WKB_C2V_REDUCTION, self.tile_columns),
            dtypes.bfloat16,
            depth=2,
            data_format="zn",
        )
        c_l0 = _ring_buffer(
            MemLoc.L0C,
            (self.tile_rows, self.tile_columns),
            dtypes.float32,
            depth=2 if self.tile_rows <= WKB_C2V_MAX_ROWS else 1,
        )
        load_engine = _copy_plan(format_transform="nd2nz")
        input_tokens = q_nope.view((rows, N * reduction))
        published_heads = 0
        for group_task in range(
            get_block_idx(), N // head_group * row_tiles, get_block_num()
        ):
            for local_head in range(head_group):
                task = group_task * head_group + local_head
                row_tile = task // N
                head = task % N
                # Earlier Cast completions were consumed by the WQB credit window.
                if dsl.const_expr(publish_heads):
                    if published_heads >= HEAD_PIPELINE_DEPTH:
                        _sync_ops.cube_sync_block_wait(
                            _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                        )
                    if dsl.const_expr(not input_prepared):
                        _sync_ops.cube_sync_block_wait(
                            _sync_ops.PIPE.S, PREPARE_READY_EVENT, mode=2
                        )
                else:
                    if task + HEAD_PIPELINE_DEPTH * get_block_num() >= N * row_tiles:
                        _sync_ops.cube_sync_block_wait(
                            _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                        )
                head_weight = wkb[head, None, None].view((reduction, KV_ELEMENTS))
                if dsl.const_expr(publish_heads):
                    output_heads = q.view((N, rows, KV_ELEMENTS))
                else:
                    output_heads = dsl.permute(q.view((rows, N, q.shape[1])), (1, 0, 2))
                # Keep the token stride, including other heads and the RoPE tail.
                head_output = output_heads[head, None, 0:KV_ELEMENTS]
                for column_tile in range(KV_ELEMENTS // self.tile_columns):
                    output_tile = tile_slice(
                        head_output,
                        (self.tile_rows, self.tile_columns),
                        (row_tile, column_tile),
                    )
                    # MMAD and Fixpipe must share the tail's packed NZ row stride.
                    acc_slot, acc = _accumulator(
                        c_l0, output_tile.shape[0], self.tile_columns
                    )
                    # Coalesce the full Dn panel; keep the original L0 K64 order.
                    _copy(
                        b_l1.next_write(),
                        tile_slice(
                            head_weight,
                            (reduction, self.tile_columns),
                            (0, column_tile),
                        ),
                        plan=load_engine,
                        l2_cache_ctl=WKB_WEIGHT_L2_CACHE_CTL,
                    )
                    resident_b = b_l1.next_read()
                    for k_tile in range(reduction // WKB_C2V_REDUCTION):
                        _copy(
                            a_l1.next_write(),
                            tile_slice(
                                input_tokens,
                                (self.tile_rows, WKB_C2V_REDUCTION),
                                (
                                    row_tile,
                                    head * (reduction // WKB_C2V_REDUCTION) + k_tile,
                                ),
                            ),
                            plan=load_engine,
                        )
                        _copy(
                            _view(
                                a_l0.next_write(),
                                shape=(output_tile.shape[0], WKB_C2V_REDUCTION),
                            ),
                            a_l1.next_read(),
                        )
                        _copy(
                            b_l0.next_write(),
                            tile_slice(
                                resident_b,
                                (WKB_C2V_REDUCTION, self.tile_columns),
                                (k_tile, 0),
                            ),
                            transpose=True,
                        )
                        matmul(
                            acc_slot,
                            _view(
                                a_l0.next_read(),
                                shape=(output_tile.shape[0], WKB_C2V_REDUCTION),
                            ),
                            b_l0.next_read(),
                            init=k_tile == 0,
                        )
                    c_l0.next_read()
                    _copy(output_tile, acc)
                if dsl.const_expr(publish_heads):
                    cube_sync_block_arrive(
                        _sync_ops.PIPE.FIXPIPE, HEAD_READY_EVENT, mode=2
                    )
                    published_heads += 1
        if dsl.const_expr(publish_heads):
            for pending in range(min(published_heads, HEAD_PIPELINE_DEPTH)):
                _sync_ops.cube_sync_block_wait(
                    _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                )

    @jit
    def run(self, q_nope: Tensor, wkb: Tensor, output: Tensor):
        N = wkb.shape[0]
        reduction = wkb.shape[1]
        rows = q_nope.shape[0] // N
        row_tiles = (rows + self.tile_rows - 1) // self.tile_rows
        a_l1 = _ring_buffer(
            MemLoc.L1, (self.tile_rows, WKB_C2V_REDUCTION), dtypes.bfloat16, depth=2
        )
        b_l1 = _ring_buffer(
            MemLoc.L1, (WKB_C2V_REDUCTION, WKB_C2V_COLUMNS), dtypes.bfloat16, depth=2
        )
        a_l0 = _ring_buffer(
            MemLoc.L0A, (self.tile_rows, WKB_C2V_REDUCTION), dtypes.bfloat16, depth=2
        )
        b_l0 = _ring_buffer(
            MemLoc.L0B,
            (WKB_C2V_REDUCTION, WKB_C2V_COLUMNS),
            dtypes.bfloat16,
            depth=2,
            data_format="zn",
        )
        c_l0 = _ring_buffer(
            MemLoc.L0C, (self.tile_rows, WKB_C2V_COLUMNS), dtypes.float32, depth=1
        )
        c_to_v = _ring_buffer(
            MemLoc.UB,
            (self.vector_rows, WKB_C2V_COLUMNS),
            dtypes.float32,
            depth=2,
            cross_core=True,
        )
        load_engine = _copy_plan(format_transform="nd2nz")
        split_engine = _copy_plan(split_axis=0)
        result = _buffer(
            MemLoc.UB, (self.vector_rows, WKB_C2V_COLUMNS), dtypes.bfloat16
        )
        input_tokens = q_nope.view((rows, N * reduction))
        output_heads = output.view((N, rows, 512))
        for task in range(get_block_idx(), N * row_tiles * 2, get_block_num()):
            column_tile = task % 2
            row_tile = (task // 2) % row_tiles
            head = task // (2 * row_tiles)
            head_weight = wkb[head, None, None].view((reduction, 512))
            head_output = output_heads[head, None, None].view((rows, 512))
            output_tile = tile_slice(
                head_output, (self.tile_rows, WKB_C2V_COLUMNS), (row_tile, column_tile)
            )
            acc_slot, acc = _accumulator(c_l0, output_tile.shape[0], WKB_C2V_COLUMNS)
            for k_tile in range(reduction // WKB_C2V_REDUCTION):
                _copy(
                    a_l1.next_write(),
                    tile_slice(
                        input_tokens,
                        (self.tile_rows, WKB_C2V_REDUCTION),
                        (row_tile, head * (reduction // WKB_C2V_REDUCTION) + k_tile),
                    ),
                    plan=load_engine,
                )
                _copy(
                    b_l1.next_write(),
                    tile_slice(
                        head_weight,
                        (WKB_C2V_REDUCTION, WKB_C2V_COLUMNS),
                        (k_tile, column_tile),
                    ),
                    plan=load_engine,
                    l2_cache_ctl=WKB_WEIGHT_L2_CACHE_CTL,
                )
                _copy(
                    _view(
                        a_l0.next_write(),
                        shape=(output_tile.shape[0], WKB_C2V_REDUCTION),
                    ),
                    a_l1.next_read(),
                )
                _copy(b_l0.next_write(), b_l1.next_read(), transpose=True)
                matmul(
                    acc_slot,
                    _view(
                        a_l0.next_read(),
                        shape=(output_tile.shape[0], WKB_C2V_REDUCTION),
                    ),
                    b_l0.next_read(),
                    init=k_tile == 0,
                )
            c_l0.next_read()
            _copy(c_to_v.next_write(), acc, plan=split_engine)
            values = c_to_v.next_read()
            vector_rows = cv_valid_extent(
                output_tile, axis=0, part_id=get_subblock_id()
            )
            with vf(mode="simd"):
                cast_mask = rr.full_mask()
                for offset in range(
                    0, vector_rows * WKB_C2V_COLUMNS, VECTOR_FLOAT_ELEMENTS
                ):
                    encoded = rr.vcast(
                        rr.vload(values, offset),
                        dtypes.bfloat16,
                        mask=cast_mask,
                        rounding=rr.RoundingMode.RN,
                        reg_layout=RegLayout.ZERO,
                    )
                    rr.vstore_pack(
                        result,
                        offset,
                        encoded,
                        cast_mask,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
            _copy(output_tile, result, plan=split_engine, part_id=get_subblock_id())


MM34_SHARED_COLUMNS = 256


class WqbStripeProjection:
    def __init__(self, tile_rows, ready_event, grid_event):
        self.tile_rows = tile_rows
        self.ready_event = ready_event
        self.grid_event = grid_event
        self.head_pipeline = False

    @jit
    def run(
        self,
        activation: Tensor,
        weight: Tensor,
        activation_scale: Tensor,
        weight_scale: Tensor,
        output: Tensor,
        c_to_v=None,
        shared_l1_b=None,
        shared_l0_b=None,
        shared_l0_c=None,
    ):
        rows = activation.shape[0]
        reduction = activation.shape[1]
        stripes_per_row = weight.shape[0] // STRIPE_COLUMNS
        task_count = ((rows + self.tile_rows - 1) // self.tile_rows) * stripes_per_row
        panels = reduction // REDUCTION_PANEL
        a_l1 = _ring_buffer(
            MemLoc.L1,
            (self.tile_rows, REDUCTION_PANEL),
            dtypes.float8_e4m3fn,
            depth=2,
            data_format="nz",
        )
        b_l1 = shared_l1_b
        if dsl.const_expr(shared_l1_b is None):
            b_l1 = _ring_buffer(
                MemLoc.L1,
                (STRIPE_COLUMNS, REDUCTION_PANEL),
                dtypes.float8_e4m3fn,
                depth=2,
                data_format="nz",
            )
        a_l0 = _ring_buffer(
            MemLoc.L0A,
            (self.tile_rows, REDUCTION_PANEL),
            dtypes.float8_e4m3fn,
            depth=2,
            data_format="nz",
        )
        b_l0 = shared_l0_b
        if dsl.const_expr(shared_l0_b is None):
            b_l0 = _ring_buffer(
                MemLoc.L0B,
                (COLUMNS_PER_PART, REDUCTION_PANEL),
                dtypes.float8_e4m3fn,
                depth=2,
                data_format="nz",
            )
        c_l0 = shared_l0_c
        if dsl.const_expr(shared_l0_c is None):
            c_l0 = _ring_buffer(
                MemLoc.L0C,
                (self.tile_rows, COLUMNS_PER_PART),
                dtypes.float32,
                depth=STRIPE_PARTS,
            )
        sa_l1 = _ring_buffer(
            MemLoc.L1,
            (self.tile_rows, reduction // MX_GROUP),
            dtypes.float8_e8m0,
            depth=1,
            data_format="zn",
        )
        sb_l1 = _ring_buffer(
            MemLoc.L1,
            (reduction // MX_GROUP, STRIPE_COLUMNS),
            dtypes.float8_e8m0,
            depth=1,
            data_format="nz",
            raw_bytes=True,
        )
        nd_engine = _copy_plan(format_transform="nd2nz")
        weight_engine = _copy_plan(format_transform="identity")
        scale_engine = _copy_plan(format_transform="mx_scale_and")
        packed_width = reduction // (MX_GROUP * SCALE_PAIR) * NZ_COLUMNS * SCALE_PAIR
        packed_scales = weight_scale.view((weight.shape[0] // NZ_COLUMNS, packed_width))

        if get_block_idx() < task_count:
            stripe = get_block_idx() % stripes_per_row
            _copy(
                b_l1.next_write(),
                tile_slice(weight, (STRIPE_COLUMNS, REDUCTION_PANEL), (stripe, 0)),
                plan=weight_engine,
            )
            scale_target = _view(
                sb_l1.next_write_raw(),
                shape=(STRIPE_COLUMNS // NZ_COLUMNS, packed_width),
                data_format="nd",
            )
            _copy(
                scale_target,
                tile_slice(
                    packed_scales,
                    (STRIPE_COLUMNS // NZ_COLUMNS, packed_width),
                    (stripe, 0),
                ),
            )
        # All launch blocks must participate, including blocks with no stripes.
        _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, self.ready_event, mode=2)
        _sync_ops.cube_sync_block_arrive(
            _sync_ops.PIPE.FIXPIPE, self.grid_event, mode=0
        )
        _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, self.grid_event, mode=0)

        if dsl.const_expr(self.head_pipeline):
            split_engine = _copy_plan(split_axis=0)
        published_groups = 0
        for task in range(get_block_idx(), task_count, get_block_num()):
            if dsl.const_expr(self.head_pipeline):
                if published_groups >= HEAD_PIPELINE_DEPTH:
                    _sync_ops.cube_sync_block_wait(
                        _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                    )
            row_tile = task // stripes_per_row
            stripe = task % stripes_per_row
            if dsl.const_expr(shared_l0_c is None):
                acc_first = c_l0.next_write()
                acc_second = c_l0.next_write()
                acc_third = c_l0.next_write()
                acc_fourth = c_l0.next_write()
                acc_fifth = c_l0.next_write()
                acc_sixth = c_l0.next_write()
            else:
                valid_m = self.tile_rows
                stripe_acc = _view(c_l0.next_write(), shape=(valid_m, STRIPE_COLUMNS))
                acc_first = tile_slice(
                    stripe_acc, (self.tile_rows, COLUMNS_PER_PART), (0, 0)
                )
                acc_second = tile_slice(
                    stripe_acc, (self.tile_rows, COLUMNS_PER_PART), (0, 1)
                )
                acc_third = tile_slice(
                    stripe_acc, (self.tile_rows, COLUMNS_PER_PART), (0, 2)
                )
                acc_fourth = tile_slice(
                    stripe_acc, (self.tile_rows, COLUMNS_PER_PART), (0, 3)
                )
                acc_fifth = tile_slice(
                    stripe_acc, (self.tile_rows, COLUMNS_PER_PART), (0, 4)
                )
                acc_sixth = tile_slice(
                    stripe_acc, (self.tile_rows, COLUMNS_PER_PART), (0, 5)
                )
            _copy(
                sa_l1.next_write(),
                tile_slice(
                    activation_scale,
                    dsl.make_tiler(
                        (
                            self.tile_rows,
                            reduction // (MX_GROUP * SCALE_PAIR),
                            SCALE_PAIR,
                        ),
                        alignment=(self.tile_rows, SCALE_PAIR, SCALE_PAIR),
                    ),
                    (row_tile, 0, 0),
                ),
                plan=scale_engine,
            )
            if task != get_block_idx():
                scale_target = _view(
                    sb_l1.next_write_raw(),
                    shape=(STRIPE_COLUMNS // NZ_COLUMNS, packed_width),
                    data_format="nd",
                )
                _copy(
                    scale_target,
                    tile_slice(
                        packed_scales,
                        (STRIPE_COLUMNS // NZ_COLUMNS, packed_width),
                        (stripe, 0),
                    ),
                )
            sa_full = sa_l1.next_read()
            sb_full = sb_l1.next_read()
            for panel in range(panels + 1):
                if panel < panels:
                    _copy(
                        a_l1.next_write(),
                        tile_slice(
                            activation,
                            (self.tile_rows, REDUCTION_PANEL),
                            (row_tile, panel),
                        ),
                        plan=nd_engine,
                    )
                    if task != get_block_idx() or panel != 0:
                        _copy(
                            b_l1.next_write(),
                            tile_slice(
                                weight,
                                (STRIPE_COLUMNS, REDUCTION_PANEL),
                                (stripe, panel),
                            ),
                            plan=weight_engine,
                        )
                if panel > 0:
                    a = a_l1.next_read()
                    b = b_l1.next_read()
                    _copy(
                        a_l0.next_write(),
                        a,
                        mx_scale=tile_slice(
                            sa_full,
                            (self.tile_rows, REDUCTION_PANEL // MX_GROUP),
                            (0, panel - 1),
                        ),
                    )
                    a_local = a_l0.next_read()
                    _copy(
                        b_l0.next_write(),
                        tile_slice(b, (COLUMNS_PER_PART, REDUCTION_PANEL), (0, 0)),
                        mx_scale=tile_slice(
                            sb_full,
                            (REDUCTION_PANEL // MX_GROUP, COLUMNS_PER_PART),
                            (panel - 1, 0),
                        ),
                    )
                    matmul(
                        acc_first,
                        a_local,
                        b_l0.next_read(),
                        init=panel == 1,
                        mx_mode=True,
                    )
                    _copy(
                        b_l0.next_write(),
                        tile_slice(b, (COLUMNS_PER_PART, REDUCTION_PANEL), (1, 0)),
                        mx_scale=tile_slice(
                            sb_full,
                            (REDUCTION_PANEL // MX_GROUP, COLUMNS_PER_PART),
                            (panel - 1, 1),
                        ),
                    )
                    matmul(
                        acc_second,
                        a_local,
                        b_l0.next_read(),
                        init=panel == 1,
                        mx_mode=True,
                    )
                    _copy(
                        b_l0.next_write(),
                        tile_slice(b, (COLUMNS_PER_PART, REDUCTION_PANEL), (2, 0)),
                        mx_scale=tile_slice(
                            sb_full,
                            (REDUCTION_PANEL // MX_GROUP, COLUMNS_PER_PART),
                            (panel - 1, 2),
                        ),
                    )
                    matmul(
                        acc_third,
                        a_local,
                        b_l0.next_read(),
                        init=panel == 1,
                        mx_mode=True,
                    )
                    _copy(
                        b_l0.next_write(),
                        tile_slice(b, (COLUMNS_PER_PART, REDUCTION_PANEL), (3, 0)),
                        mx_scale=tile_slice(
                            sb_full,
                            (REDUCTION_PANEL // MX_GROUP, COLUMNS_PER_PART),
                            (panel - 1, 3),
                        ),
                    )
                    matmul(
                        acc_fourth,
                        a_local,
                        b_l0.next_read(),
                        init=panel == 1,
                        mx_mode=True,
                    )
                    _copy(
                        b_l0.next_write(),
                        tile_slice(b, (COLUMNS_PER_PART, REDUCTION_PANEL), (4, 0)),
                        mx_scale=tile_slice(
                            sb_full,
                            (REDUCTION_PANEL // MX_GROUP, COLUMNS_PER_PART),
                            (panel - 1, 4),
                        ),
                    )
                    matmul(
                        acc_fifth,
                        a_local,
                        b_l0.next_read(),
                        init=panel == 1,
                        mx_mode=True,
                    )
                    _copy(
                        b_l0.next_write(),
                        tile_slice(b, (COLUMNS_PER_PART, REDUCTION_PANEL), (5, 0)),
                        mx_scale=tile_slice(
                            sb_full,
                            (REDUCTION_PANEL // MX_GROUP, COLUMNS_PER_PART),
                            (panel - 1, 5),
                        ),
                    )
                    matmul(
                        acc_sixth,
                        a_local,
                        b_l0.next_read(),
                        init=panel == 1,
                        mx_mode=True,
                    )
            if dsl.const_expr(self.head_pipeline):
                if published_groups == 0:
                    _sync_ops.cube_sync_block_wait(
                        _sync_ops.PIPE.S, HEAD_READY_EVENT, mode=2
                    )
                stripe_values = c_to_v.next_write()
                if dsl.const_expr(shared_l0_c is None):
                    for part in range(STRIPE_PARTS):
                        _copy(
                            tile_slice(
                                stripe_values,
                                (self.tile_rows // VECTOR_SUBBLOCKS, COLUMNS_PER_PART),
                                (0, part),
                            ),
                            c_l0.next_read(),
                            plan=split_engine,
                            actual_shape=(self.tile_rows, COLUMNS_PER_PART),
                        )
                else:
                    stripe_result = _view(
                        c_l0.next_read(), shape=(valid_m, STRIPE_COLUMNS)
                    )
                    for part in range(STRIPE_PARTS):
                        _copy(
                            tile_slice(
                                stripe_values,
                                (self.tile_rows // VECTOR_SUBBLOCKS, COLUMNS_PER_PART),
                                (0, part),
                            ),
                            tile_slice(
                                stripe_result,
                                (self.tile_rows, COLUMNS_PER_PART),
                                (0, part),
                            ),
                            plan=split_engine,
                            actual_shape=(valid_m, COLUMNS_PER_PART),
                        )
                published_groups += 1
            else:
                _copy(
                    tile_slice(
                        output,
                        (self.tile_rows, COLUMNS_PER_PART),
                        (row_tile, stripe * STRIPE_PARTS),
                    ),
                    c_l0.next_read(),
                )
                _copy(
                    tile_slice(
                        output,
                        (self.tile_rows, COLUMNS_PER_PART),
                        (row_tile, stripe * STRIPE_PARTS + 1),
                    ),
                    c_l0.next_read(),
                )
                _copy(
                    tile_slice(
                        output,
                        (self.tile_rows, COLUMNS_PER_PART),
                        (row_tile, stripe * STRIPE_PARTS + 2),
                    ),
                    c_l0.next_read(),
                )
                _copy(
                    tile_slice(
                        output,
                        (self.tile_rows, COLUMNS_PER_PART),
                        (row_tile, stripe * STRIPE_PARTS + 3),
                    ),
                    c_l0.next_read(),
                )
                _copy(
                    tile_slice(
                        output,
                        (self.tile_rows, COLUMNS_PER_PART),
                        (row_tile, stripe * STRIPE_PARTS + 4),
                    ),
                    c_l0.next_read(),
                )
                _copy(
                    tile_slice(
                        output,
                        (self.tile_rows, COLUMNS_PER_PART),
                        (row_tile, stripe * STRIPE_PARTS + 5),
                    ),
                    c_l0.next_read(),
                )
        if dsl.const_expr(self.head_pipeline):
            if published_groups == 0:
                _sync_ops.cube_sync_block_wait(
                    _sync_ops.PIPE.S, HEAD_READY_EVENT, mode=2
                )
            for pending in range(min(published_groups, HEAD_PIPELINE_DEPTH)):
                _sync_ops.cube_sync_block_wait(
                    _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                )
            _sync_ops.cube_sync_all()


class WqbCompactHeadProjection(WqbStripeProjection):
    def __init__(
        self,
        tile_rows,
        ready_event,
        grid_event,
        head_group,
        head_width,
        weight_is_nz,
        reduction_panel=REDUCTION_PANEL,
    ):
        self.tile_rows = tile_rows
        self.reduction_panel = reduction_panel
        self.ready_event = ready_event
        self.grid_event = grid_event
        self.head_pipeline = False
        self.stripe_columns = head_group * head_width
        self.part_columns = (
            COLUMNS_PER_PART
            if self.stripe_columns % COLUMNS_PER_PART == 0
            else VECTOR_FLOAT_ELEMENTS
        )
        self.weight_is_nz = weight_is_nz
        self.stripe_parts = self.stripe_columns // self.part_columns

    @jit
    def run(
        self,
        activation: Tensor,
        weight: Tensor,
        activation_scale: Tensor,
        weight_scale: Tensor,
        output: Tensor,
        c_to_v=None,
        shared_l1_b=None,
        shared_l0_b=None,
        shared_l0_c=None,
    ):
        rows = activation.shape[0]
        reduction = activation.shape[1]
        stripes_per_row = weight.shape[0] // self.stripe_columns
        task_count = ((rows + self.tile_rows - 1) // self.tile_rows) * stripes_per_row
        panels = reduction // self.reduction_panel
        a_l1 = _ring_buffer(
            MemLoc.L1,
            (self.tile_rows, self.reduction_panel),
            dtypes.float8_e4m3fn,
            depth=2,
            data_format="nz",
        )
        b_l1 = shared_l1_b
        a_l0 = _ring_buffer(
            MemLoc.L0A,
            (self.tile_rows, self.reduction_panel),
            dtypes.float8_e4m3fn,
            depth=2,
            data_format="nz",
        )
        b_l0 = shared_l0_b
        c_l0 = shared_l0_c
        sa_l1 = _ring_buffer(
            MemLoc.L1,
            (self.tile_rows, reduction // MX_GROUP),
            dtypes.float8_e8m0,
            depth=1,
            data_format="zn",
        )
        sb_l1 = _ring_buffer(
            MemLoc.L1,
            (reduction // MX_GROUP, self.stripe_columns),
            dtypes.float8_e8m0,
            depth=1,
            data_format="nz",
            raw_bytes=self.weight_is_nz,
        )
        nd_engine = _copy_plan(format_transform="nd2nz")
        weight_engine = _copy_plan(format_transform="nd2nz")
        if dsl.const_expr(self.weight_is_nz):
            weight_engine = _copy_plan(format_transform="identity")
        weight_scale_engine = _copy_plan(format_transform="mx_scale_bdn")
        scale_engine = _copy_plan(format_transform="mx_scale_and")
        packed_width = reduction // (MX_GROUP * SCALE_PAIR) * NZ_COLUMNS * SCALE_PAIR
        if dsl.const_expr(self.weight_is_nz):
            packed_scales = weight_scale.view(
                (weight.shape[0] // NZ_COLUMNS, packed_width)
            )

        if get_block_idx() < task_count:
            stripe = get_block_idx() % stripes_per_row
            _copy(
                _view(
                    b_l1.next_write(), shape=(self.stripe_columns, self.reduction_panel)
                ),
                tile_slice(
                    weight, (self.stripe_columns, self.reduction_panel), (stripe, 0)
                ),
                plan=weight_engine,
            )
            if dsl.const_expr(self.weight_is_nz):
                scale_target = _view(
                    sb_l1.next_write_raw(),
                    shape=(self.stripe_columns // NZ_COLUMNS, packed_width),
                    data_format="nd",
                )
                _copy(
                    scale_target,
                    tile_slice(
                        packed_scales,
                        (self.stripe_columns // NZ_COLUMNS, packed_width),
                        (stripe, 0),
                    ),
                )
            else:
                _copy(
                    sb_l1.next_write(),
                    tile_slice(
                        weight_scale,
                        dsl.make_tiler(
                            (
                                self.stripe_columns,
                                reduction // (MX_GROUP * SCALE_PAIR),
                                SCALE_PAIR,
                            ),
                            alignment=(self.stripe_columns, SCALE_PAIR, SCALE_PAIR),
                        ),
                        (stripe, 0, 0),
                    ),
                    plan=weight_scale_engine,
                )
        # All launch blocks must participate, including blocks with no stripes.
        _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, self.ready_event, mode=2)
        _sync_ops.cube_sync_block_arrive(
            _sync_ops.PIPE.FIXPIPE, self.grid_event, mode=0
        )
        _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, self.grid_event, mode=0)

        if dsl.const_expr(self.head_pipeline):
            split_engine = _copy_plan(split_axis=0)
        published_groups = 0
        for task in range(get_block_idx(), task_count, get_block_num()):
            if dsl.const_expr(self.head_pipeline):
                if published_groups >= HEAD_PIPELINE_DEPTH:
                    _sync_ops.cube_sync_block_wait(
                        _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                    )
            row_tile = task // stripes_per_row
            stripe = task % stripes_per_row
            valid_m = self.tile_rows
            stripe_acc = _view(c_l0.next_write(), shape=(valid_m, self.stripe_columns))
            _copy(
                sa_l1.next_write(),
                tile_slice(
                    activation_scale,
                    dsl.make_tiler(
                        (
                            self.tile_rows,
                            reduction // (MX_GROUP * SCALE_PAIR),
                            SCALE_PAIR,
                        ),
                        alignment=(self.tile_rows, SCALE_PAIR, SCALE_PAIR),
                    ),
                    (row_tile, 0, 0),
                ),
                plan=scale_engine,
            )
            if task != get_block_idx():
                if dsl.const_expr(self.weight_is_nz):
                    scale_target = _view(
                        sb_l1.next_write_raw(),
                        shape=(self.stripe_columns // NZ_COLUMNS, packed_width),
                        data_format="nd",
                    )
                    _copy(
                        scale_target,
                        tile_slice(
                            packed_scales,
                            (self.stripe_columns // NZ_COLUMNS, packed_width),
                            (stripe, 0),
                        ),
                    )
                else:
                    _copy(
                        sb_l1.next_write(),
                        tile_slice(
                            weight_scale,
                            dsl.make_tiler(
                                (
                                    self.stripe_columns,
                                    reduction // (MX_GROUP * SCALE_PAIR),
                                    SCALE_PAIR,
                                ),
                                alignment=(self.stripe_columns, SCALE_PAIR, SCALE_PAIR),
                            ),
                            (stripe, 0, 0),
                        ),
                        plan=weight_scale_engine,
                    )
            sa_full = sa_l1.next_read()
            sb_full = sb_l1.next_read()
            for panel in range(panels + 1):
                if panel < panels:
                    _copy(
                        a_l1.next_write(),
                        tile_slice(
                            activation,
                            (self.tile_rows, self.reduction_panel),
                            (row_tile, panel),
                        ),
                        plan=nd_engine,
                    )
                    if task != get_block_idx() or panel != 0:
                        _copy(
                            _view(
                                b_l1.next_write(),
                                shape=(self.stripe_columns, self.reduction_panel),
                            ),
                            tile_slice(
                                weight,
                                (self.stripe_columns, self.reduction_panel),
                                (stripe, panel),
                            ),
                            plan=weight_engine,
                        )
                if panel > 0:
                    a = a_l1.next_read()
                    b = _view(
                        b_l1.next_read(),
                        shape=(self.stripe_columns, self.reduction_panel),
                    )
                    _copy(
                        a_l0.next_write(),
                        a,
                        mx_scale=tile_slice(
                            sa_full,
                            (self.tile_rows, self.reduction_panel // MX_GROUP),
                            (0, panel - 1),
                        ),
                    )
                    a_local = a_l0.next_read()
                    for part in range(self.stripe_parts):
                        _copy(
                            _view(
                                b_l0.next_write(),
                                shape=(self.part_columns, self.reduction_panel),
                            ),
                            tile_slice(
                                b, (self.part_columns, self.reduction_panel), (part, 0)
                            ),
                            mx_scale=tile_slice(
                                sb_full,
                                (self.reduction_panel // MX_GROUP, self.part_columns),
                                (panel - 1, part),
                            ),
                        )
                        matmul(
                            tile_slice(
                                stripe_acc,
                                (self.tile_rows, self.part_columns),
                                (0, part),
                            ),
                            a_local,
                            _view(
                                b_l0.next_read(),
                                shape=(self.part_columns, self.reduction_panel),
                            ),
                            init=panel == 1,
                            mx_mode=True,
                        )
            if dsl.const_expr(self.head_pipeline):
                if published_groups == 0:
                    _sync_ops.cube_sync_block_wait(
                        _sync_ops.PIPE.S, HEAD_READY_EVENT, mode=2
                    )
                stripe_values = c_to_v.next_write()
                stripe_result = _view(
                    c_l0.next_read(), shape=(valid_m, self.stripe_columns)
                )
                for part in range(self.stripe_parts):
                    _copy(
                        tile_slice(
                            stripe_values,
                            (self.tile_rows // VECTOR_SUBBLOCKS, self.part_columns),
                            (0, part),
                        ),
                        tile_slice(
                            stripe_result,
                            (self.tile_rows, self.part_columns),
                            (0, part),
                        ),
                        plan=split_engine,
                        actual_shape=(valid_m, self.part_columns),
                    )
                published_groups += 1
        if dsl.const_expr(self.head_pipeline):
            if published_groups == 0:
                _sync_ops.cube_sync_block_wait(
                    _sync_ops.PIPE.S, HEAD_READY_EVENT, mode=2
                )
            for pending in range(min(published_groups, HEAD_PIPELINE_DEPTH)):
                _sync_ops.cube_sync_block_wait(
                    _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                )
            _sync_ops.cube_sync_all()


class _Projection:
    def __init__(
        self,
        columns,
        reduction,
        w_qr_quant_mode,
        weight_is_nz,
        descale_weight_is_nz,
        tile_rows=PROJECTION_ROWS,
        tile_columns=PROJECTION_COLUMNS,
        tile_reduction=PROJECTION_REDUCTION,
        head_pipeline=False,
        join_after_prefetch=False,
        force_split_l0=False,
        hif_c2v=False,
        hif_split_l0=False,
        drain_handoff=False,
    ):
        self.drain_handoff = drain_handoff
        self.columns = columns
        self.reduction = reduction
        self.w_qr_quant_mode = w_qr_quant_mode
        self.weight_is_nz = weight_is_nz
        self.descale_weight_is_nz = descale_weight_is_nz
        self.tile_rows = tile_rows
        self.tile_columns = tile_columns
        self.tile_reduction = tile_reduction
        self.split_l0 = (
            head_pipeline
            and (tile_rows >= 128 or force_split_l0)
            and w_qr_quant_mode == 1
        )
        if w_qr_quant_mode == 2 and hif_split_l0:
            self.split_l0 = True
        self.l0_reduction = (
            min(tile_reduction, WQB_L0_REDUCTION) if self.split_l0 else tile_reduction
        )
        self.head_pipeline = head_pipeline
        self.join_after_prefetch = join_after_prefetch
        self.hif_c2v = hif_c2v
        self.weight_lookahead = False

    @jit
    def run(
        self,
        activation: Tensor,
        weight_input: Tensor,
        descale_activation: Tensor,
        descale_weight: Tensor,
        projection_fp32: Tensor,
        core_begin: int = 0,
        core_count: int = 0,
        defer_qr: bool = False,
        c_to_v=None,
    ):
        rows = activation.shape[0]
        columns = weight_input.shape[0]
        reduction = activation.shape[1]
        dtype = dtypes.float8_e4m3fn
        if dsl.const_expr(self.w_qr_quant_mode == 2):
            dtype = dtypes.uint8
        a_l1 = _ring_buffer(
            MemLoc.L1,
            (self.tile_rows, self.tile_reduction),
            dtype,
            depth=2,
            data_format="nz",
        )
        b_l1 = _ring_buffer(
            MemLoc.L1,
            (self.tile_columns, self.tile_reduction),
            dtype,
            depth=3 if self.weight_lookahead else 2,
            data_format="nz",
        )
        cube_dtype = dtypes.float8_e4m3fn
        if dsl.const_expr(self.w_qr_quant_mode == 2):
            cube_dtype = dtypes.hifloat8
        a_l0 = _ring_buffer(
            MemLoc.L0A,
            (self.tile_rows, self.l0_reduction),
            cube_dtype,
            depth=2,
            data_format="nz",
        )
        b_l0_depth = 1 if self.tile_columns * self.l0_reduction > 32768 else 2
        b_l0 = _ring_buffer(
            MemLoc.L0B,
            (self.tile_columns, self.l0_reduction),
            cube_dtype,
            depth=b_l0_depth,
            data_format="nz",
        )
        c_l0 = _ring_buffer(
            MemLoc.L0C, (self.tile_rows, self.tile_columns), dtypes.float32, depth=1
        )
        if dsl.const_expr(self.weight_is_nz):
            # Packed NZ scale copies require the same physical stride as
            # the GM payload; NZ K is statically specialized by the host.
            sa_l1 = _ring_buffer(
                MemLoc.L1,
                (self.tile_rows, reduction // MX_GROUP_ELEMENTS),
                dtypes.float8_e8m0,
                depth=1,
                data_format="zn",
            )
            sb_l1 = _ring_buffer(
                MemLoc.L1,
                (reduction // MX_GROUP_ELEMENTS, self.tile_columns),
                dtypes.float8_e8m0,
                depth=1,
                data_format="nz",
                raw_bytes=True,
            )
        else:
            sa_l1 = _ring_buffer(
                MemLoc.L1,
                (self.tile_rows, reduction // MX_GROUP_ELEMENTS),
                dtypes.float8_e8m0,
                depth=1,
                data_format="zn",
                capacity=(self.tile_rows, MAX_REDUCTION_ELEMENTS // MX_GROUP_ELEMENTS),
            )
            sb_l1 = _ring_buffer(
                MemLoc.L1,
                (reduction // MX_GROUP_ELEMENTS, self.tile_columns),
                dtypes.float8_e8m0,
                depth=1,
                data_format="nz",
                capacity=(
                    MAX_REDUCTION_ELEMENTS // MX_GROUP_ELEMENTS,
                    self.tile_columns,
                ),
            )
        sa_engine = _copy_plan(format_transform="mx_scale_and")
        sb_engine = _copy_plan(format_transform="mx_scale_bdn")
        nd_engine = _copy_plan(format_transform="nd2nz")
        b_engine = _copy_plan(format_transform="nd2nz")
        if dsl.const_expr(self.weight_is_nz):
            b_engine = _copy_plan(format_transform="identity")
        tasks_per_row = columns // self.tile_columns
        task_count = (rows + self.tile_rows - 1) // self.tile_rows * tasks_per_row
        task_stride = get_block_num()
        if core_count > 0:
            task_stride = core_count
        if defer_qr:
            if get_block_idx() - core_begin < task_count:
                first_col = (get_block_idx() - core_begin) % tasks_per_row
                _copy(
                    b_l1.next_write(),
                    tile_slice(
                        weight_input,
                        (self.tile_columns, self.tile_reduction),
                        (first_col, 0),
                    ),
                    plan=b_engine,
                )
            _wait_qr_for_wqb()
        if dsl.const_expr(self.join_after_prefetch):
            if get_block_idx() < task_count:
                col_tile = get_block_idx() % tasks_per_row
                _copy(
                    b_l1.next_write(),
                    tile_slice(
                        weight_input,
                        (self.tile_columns, self.tile_reduction),
                        (col_tile, 0),
                    ),
                    plan=b_engine,
                )
                if dsl.const_expr(self.descale_weight_is_nz):
                    packed_width = reduction // 64 * NZ_SCALE_COLUMNS * MX_SCALE_PAIR
                    packed_rows = descale_weight.view(
                        (columns // NZ_SCALE_COLUMNS, packed_width)
                    )
                    packed = tile_slice(
                        packed_rows,
                        (self.tile_columns // NZ_SCALE_COLUMNS, packed_width),
                        (col_tile, 0),
                    )
                    raw_scale_tile = _view(
                        sb_l1.next_write_raw(),
                        shape=(self.tile_columns // NZ_SCALE_COLUMNS, packed_width),
                        data_format="nd",
                    )
                    _copy(raw_scale_tile, packed.view(dtype=dtypes.uint8))
                else:
                    _copy(
                        sb_l1.next_write(),
                        tile_slice(
                            descale_weight,
                            dsl.make_tiler(
                                (self.tile_columns, reduction // 64, MX_SCALE_PAIR),
                                alignment=(
                                    self.tile_columns,
                                    MX_SCALE_PAIR,
                                    MX_SCALE_PAIR,
                                ),
                            ),
                            (col_tile, 0, 0),
                        ),
                        plan=sb_engine,
                    )
            _wait_qr_for_wqb()
        if dsl.const_expr(self.hif_c2v):
            split_engine = _copy_plan(split_axis=0)
        published_heads = 0
        for task in range(get_block_idx() - core_begin, task_count, task_stride):
            if dsl.const_expr(self.head_pipeline):
                if published_heads >= HEAD_PIPELINE_DEPTH:
                    _sync_ops.cube_sync_block_wait(
                        _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                    )
            row_tile = task // tasks_per_row
            col_tile = task % tasks_per_row
            acc = c_l0.next_write()
            if dsl.const_expr(self.w_qr_quant_mode == 1):
                _copy(
                    sa_l1.next_write(),
                    tile_slice(
                        descale_activation,
                        dsl.make_tiler(
                            (self.tile_rows, reduction // 64, MX_SCALE_PAIR),
                            alignment=(self.tile_rows, MX_SCALE_PAIR, MX_SCALE_PAIR),
                        ),
                        (row_tile, 0, 0),
                    ),
                    plan=sa_engine,
                )
                if not self.join_after_prefetch or task != get_block_idx():
                    if dsl.const_expr(self.descale_weight_is_nz):
                        packed_width = (
                            reduction // 64 * NZ_SCALE_COLUMNS * MX_SCALE_PAIR
                        )
                        packed_rows = descale_weight.view(
                            (columns // NZ_SCALE_COLUMNS, packed_width)
                        )
                        packed = tile_slice(
                            packed_rows,
                            (self.tile_columns // NZ_SCALE_COLUMNS, packed_width),
                            (col_tile, 0),
                        )
                        raw_scale_tile = _view(
                            sb_l1.next_write_raw(),
                            shape=(self.tile_columns // NZ_SCALE_COLUMNS, packed_width),
                            data_format="nd",
                        )
                        _copy(raw_scale_tile, packed.view(dtype=dtypes.uint8))
                    else:
                        _copy(
                            sb_l1.next_write(),
                            tile_slice(
                                descale_weight,
                                dsl.make_tiler(
                                    (self.tile_columns, reduction // 64, MX_SCALE_PAIR),
                                    alignment=(
                                        self.tile_columns,
                                        MX_SCALE_PAIR,
                                        MX_SCALE_PAIR,
                                    ),
                                ),
                                (col_tile, 0, 0),
                            ),
                            plan=sb_engine,
                        )
                sa_full = sa_l1.next_read()
                sb_full = sb_l1.next_read()
            if dsl.const_expr(self.weight_lookahead):
                if (not self.join_after_prefetch or task != get_block_idx()) and (
                    not defer_qr or task != get_block_idx() - core_begin
                ):
                    _copy(
                        b_l1.next_write(),
                        tile_slice(
                            weight_input,
                            (self.tile_columns, self.tile_reduction),
                            (col_tile, 0),
                        ),
                        plan=b_engine,
                    )
            for reduction_tile in range(reduction // self.tile_reduction + 1):
                if reduction_tile < reduction // self.tile_reduction:
                    _copy(
                        a_l1.next_write(),
                        tile_slice(
                            activation,
                            (self.tile_rows, self.tile_reduction),
                            (row_tile, reduction_tile),
                        ),
                        plan=nd_engine,
                    )
                    if dsl.const_expr(self.weight_lookahead):
                        if reduction_tile + 1 < reduction // self.tile_reduction:
                            _copy(
                                b_l1.next_write(),
                                tile_slice(
                                    weight_input,
                                    (self.tile_columns, self.tile_reduction),
                                    (col_tile, reduction_tile + 1),
                                ),
                                plan=b_engine,
                            )
                    else:
                        if (
                            not self.join_after_prefetch
                            or task != get_block_idx()
                            or reduction_tile != 0
                        ) and (
                            not defer_qr
                            or task != get_block_idx() - core_begin
                            or reduction_tile != 0
                        ):
                            _copy(
                                b_l1.next_write(),
                                tile_slice(
                                    weight_input,
                                    (self.tile_columns, self.tile_reduction),
                                    (col_tile, reduction_tile),
                                ),
                                plan=b_engine,
                            )
                if reduction_tile > 0:
                    a = a_l1.next_read()
                    b = b_l1.next_read()
                    if dsl.const_expr(self.split_l0):
                        for l0_tile in range(self.tile_reduction // self.l0_reduction):
                            scale_tile = (reduction_tile - 1) * (
                                self.tile_reduction // self.l0_reduction
                            ) + l0_tile
                            if dsl.const_expr(self.w_qr_quant_mode == 1):
                                _copy(
                                    a_l0.next_write(),
                                    tile_slice(
                                        a,
                                        (self.tile_rows, self.l0_reduction),
                                        (0, l0_tile),
                                    ),
                                    mx_scale=tile_slice(
                                        sa_full,
                                        (
                                            self.tile_rows,
                                            self.l0_reduction // MX_GROUP_ELEMENTS,
                                        ),
                                        (0, scale_tile),
                                    ),
                                )
                                _copy(
                                    b_l0.next_write(),
                                    tile_slice(
                                        b,
                                        (self.tile_columns, self.l0_reduction),
                                        (0, l0_tile),
                                    ),
                                    mx_scale=tile_slice(
                                        sb_full,
                                        (
                                            self.l0_reduction // MX_GROUP_ELEMENTS,
                                            self.tile_columns,
                                        ),
                                        (scale_tile, 0),
                                    ),
                                )
                            else:
                                _copy(
                                    a_l0.next_write(),
                                    tile_slice(
                                        _view(a, dtype=dtypes.hifloat8),
                                        (self.tile_rows, self.l0_reduction),
                                        (0, l0_tile),
                                    ),
                                )
                                _copy(
                                    b_l0.next_write(),
                                    tile_slice(
                                        _view(b, dtype=dtypes.hifloat8),
                                        (self.tile_columns, self.l0_reduction),
                                        (0, l0_tile),
                                    ),
                                )
                            matmul(
                                acc,
                                a_l0.next_read(),
                                b_l0.next_read(),
                                init=scale_tile == 0,
                                mx_mode=self.w_qr_quant_mode == 1,
                            )
                    else:
                        if dsl.const_expr(self.w_qr_quant_mode == 1):
                            _copy(
                                a_l0.next_write(),
                                a,
                                mx_scale=tile_slice(
                                    sa_full,
                                    (
                                        self.tile_rows,
                                        self.tile_reduction // MX_GROUP_ELEMENTS,
                                    ),
                                    (0, reduction_tile - 1),
                                ),
                            )
                            _copy(
                                b_l0.next_write(),
                                b,
                                mx_scale=tile_slice(
                                    sb_full,
                                    (
                                        self.tile_reduction // MX_GROUP_ELEMENTS,
                                        self.tile_columns,
                                    ),
                                    (reduction_tile - 1, 0),
                                ),
                            )
                        else:
                            _copy(a_l0.next_write(), _view(a, dtype=dtypes.hifloat8))
                            _copy(b_l0.next_write(), _view(b, dtype=dtypes.hifloat8))
                        matmul(
                            acc,
                            a_l0.next_read(),
                            b_l0.next_read(),
                            init=reduction_tile == 1,
                            mx_mode=self.w_qr_quant_mode == 1,
                        )
            if dsl.const_expr(self.drain_handoff):
                if published_heads == 0:
                    _sync_ops.cube_sync_block_wait(
                        _sync_ops.PIPE.S, HEAD_READY_EVENT, mode=2
                    )
            if dsl.const_expr(self.hif_c2v):
                c_l0.next_read()
                _copy(
                    c_to_v.next_write(),
                    acc,
                    plan=split_engine,
                    actual_shape=(self.tile_rows, self.tile_columns),
                )
            else:
                c_l0.next_read()
                _copy(
                    tile_slice(
                        projection_fp32,
                        (self.tile_rows, self.tile_columns),
                        (row_tile, col_tile),
                    ),
                    acc,
                )
            if dsl.const_expr(self.head_pipeline):
                if dsl.const_expr(not self.hif_c2v):
                    cube_sync_block_arrive(
                        _sync_ops.PIPE.FIXPIPE, HEAD_READY_EVENT, mode=2
                    )
                published_heads += 1
        if dsl.const_expr(self.drain_handoff):
            if published_heads == 0:
                _sync_ops.cube_sync_block_wait(
                    _sync_ops.PIPE.S, HEAD_READY_EVENT, mode=2
                )
            for pending in range(min(published_heads, HEAD_PIPELINE_DEPTH)):
                _sync_ops.cube_sync_block_wait(
                    _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                )
        if dsl.const_expr(self.head_pipeline):
            # Complete Cube accesses before WKB reuses its on-chip storage.
            _sync_ops.cube_sync_all()

    @jit
    def run_mm34(
        self,
        activation: Tensor,
        weight_input: Tensor,
        descale_activation: Tensor,
        descale_weight: Tensor,
        projection_fp32: Tensor,
        q_nope: Tensor,
        wkb: Tensor,
        q: Tensor,
        core_begin: int = 0,
        core_count: int = 0,
        defer_qr: bool = False,
    ):
        rows = activation.shape[0]
        columns = weight_input.shape[0]
        reduction = activation.shape[1]
        dtype = dtypes.float8_e4m3fn
        if dsl.const_expr(self.w_qr_quant_mode == 2):
            dtype = dtypes.uint8
        a_l1 = _ring_buffer(
            MemLoc.L1,
            (self.tile_rows, self.tile_reduction),
            dtype,
            depth=2,
            data_format="nz",
        )
        b_l1 = _ring_buffer(
            MemLoc.L1,
            (self.tile_columns, self.tile_reduction),
            dtype,
            depth=2,
            data_format="nz",
        )
        cube_dtype = dtypes.float8_e4m3fn
        if dsl.const_expr(self.w_qr_quant_mode == 2):
            cube_dtype = dtypes.hifloat8
        a_l0 = _ring_buffer(
            MemLoc.L0A,
            (self.tile_rows, self.l0_reduction),
            cube_dtype,
            depth=2,
            data_format="nz",
        )
        b_l0 = _ring_buffer(
            MemLoc.L0B,
            (MM34_SHARED_COLUMNS, self.l0_reduction),
            cube_dtype,
            depth=2,
            data_format="nz",
        )
        c_l0 = _ring_buffer(
            MemLoc.L0C, (self.tile_rows, MM34_SHARED_COLUMNS), dtypes.float32, depth=2
        )
        wkb_l1_reduction = WKB_C2V_REDUCTION
        if dsl.const_expr(self.w_qr_quant_mode == 2):
            wkb_l1_reduction = wkb.shape[1]
        wkb_a_l1 = _ring_buffer(
            MemLoc.L1, (self.tile_rows, wkb_l1_reduction), dtypes.bfloat16, depth=2
        )
        wkb_b_l1 = _ring_buffer(
            MemLoc.L1, (wkb_l1_reduction, WKB_C2V_COLUMNS), dtypes.bfloat16, depth=2
        )
        wkb_load_engine = _copy_plan(format_transform="nd2nz")
        head_count = wkb.shape[0]
        wkb_reduction = wkb.shape[1]
        input_tokens = q_nope.view((rows, head_count * wkb_reduction))
        output_heads = dsl.permute(q.view((rows, head_count, q.shape[1])), (1, 0, 2))
        if dsl.const_expr(self.w_qr_quant_mode == 1):
            if dsl.const_expr(self.weight_is_nz):
                sa_l1 = _ring_buffer(
                    MemLoc.L1,
                    (self.tile_rows, reduction // MX_GROUP_ELEMENTS),
                    dtypes.float8_e8m0,
                    depth=1,
                    data_format="zn",
                )
                sb_l1 = _ring_buffer(
                    MemLoc.L1,
                    (reduction // MX_GROUP_ELEMENTS, self.tile_columns),
                    dtypes.float8_e8m0,
                    depth=1,
                    data_format="nz",
                    raw_bytes=True,
                )
            else:
                sa_l1 = _ring_buffer(
                    MemLoc.L1,
                    (self.tile_rows, reduction // MX_GROUP_ELEMENTS),
                    dtypes.float8_e8m0,
                    depth=1,
                    data_format="zn",
                    capacity=(
                        self.tile_rows,
                        MAX_REDUCTION_ELEMENTS // MX_GROUP_ELEMENTS,
                    ),
                )
                sb_l1 = _ring_buffer(
                    MemLoc.L1,
                    (reduction // MX_GROUP_ELEMENTS, self.tile_columns),
                    dtypes.float8_e8m0,
                    depth=1,
                    data_format="nz",
                    capacity=(
                        MAX_REDUCTION_ELEMENTS // MX_GROUP_ELEMENTS,
                        self.tile_columns,
                    ),
                )
            sa_engine = _copy_plan(format_transform="mx_scale_and")
            sb_engine = _copy_plan(format_transform="mx_scale_bdn")
        nd_engine = _copy_plan(format_transform="nd2nz")
        b_engine = _copy_plan(format_transform="nd2nz")
        if dsl.const_expr(self.weight_is_nz):
            b_engine = _copy_plan(format_transform="identity")
        tasks_per_row = columns // self.tile_columns
        task_count = (rows + self.tile_rows - 1) // self.tile_rows * tasks_per_row
        task_stride = get_block_num()
        if core_count > 0:
            task_stride = core_count
        if dsl.const_expr(self.w_qr_quant_mode == 1):
            if dsl.const_expr(self.join_after_prefetch):
                if get_block_idx() < task_count:
                    col_tile = get_block_idx() % tasks_per_row
                    _copy(
                        b_l1.next_write(),
                        tile_slice(
                            weight_input,
                            (self.tile_columns, self.tile_reduction),
                            (col_tile, 0),
                        ),
                        plan=b_engine,
                    )
                    if dsl.const_expr(self.descale_weight_is_nz):
                        packed_width = (
                            reduction // 64 * NZ_SCALE_COLUMNS * MX_SCALE_PAIR
                        )
                        packed_rows = descale_weight.view(
                            (columns // NZ_SCALE_COLUMNS, packed_width)
                        )
                        packed = tile_slice(
                            packed_rows,
                            (self.tile_columns // NZ_SCALE_COLUMNS, packed_width),
                            (col_tile, 0),
                        )
                        raw_scale_tile = _view(
                            sb_l1.next_write_raw(),
                            shape=(self.tile_columns // NZ_SCALE_COLUMNS, packed_width),
                            data_format="nd",
                        )
                        _copy(raw_scale_tile, packed.view(dtype=dtypes.uint8))
                    else:
                        _copy(
                            sb_l1.next_write(),
                            tile_slice(
                                descale_weight,
                                dsl.make_tiler(
                                    (self.tile_columns, reduction // 64, MX_SCALE_PAIR),
                                    alignment=(
                                        self.tile_columns,
                                        MX_SCALE_PAIR,
                                        MX_SCALE_PAIR,
                                    ),
                                ),
                                (col_tile, 0, 0),
                            ),
                            plan=sb_engine,
                        )
                _wait_qr_for_wqb()
        if defer_qr:
            if get_block_idx() - core_begin < task_count:
                first_col = (get_block_idx() - core_begin) % tasks_per_row
                _copy(
                    b_l1.next_write(),
                    tile_slice(
                        weight_input,
                        (self.tile_columns, self.tile_reduction),
                        (first_col, 0),
                    ),
                    plan=b_engine,
                )
            _wait_qr_for_wqb()
        published_heads = 0
        for task in range(get_block_idx() - core_begin, task_count, task_stride):
            if dsl.const_expr(self.head_pipeline):
                if published_heads >= HEAD_PIPELINE_DEPTH:
                    _sync_ops.cube_sync_block_wait(
                        _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                    )
            row_tile = task // tasks_per_row
            col_tile = task % tasks_per_row
            acc = _view(c_l0.next_write(), shape=(self.tile_rows, self.tile_columns))
            if dsl.const_expr(self.w_qr_quant_mode == 1):
                _copy(
                    sa_l1.next_write(),
                    tile_slice(
                        descale_activation,
                        dsl.make_tiler(
                            (self.tile_rows, reduction // 64, MX_SCALE_PAIR),
                            alignment=(self.tile_rows, MX_SCALE_PAIR, MX_SCALE_PAIR),
                        ),
                        (row_tile, 0, 0),
                    ),
                    plan=sa_engine,
                )
                if not self.join_after_prefetch or task != get_block_idx():
                    if dsl.const_expr(self.descale_weight_is_nz):
                        packed_width = (
                            reduction // 64 * NZ_SCALE_COLUMNS * MX_SCALE_PAIR
                        )
                        packed_rows = descale_weight.view(
                            (columns // NZ_SCALE_COLUMNS, packed_width)
                        )
                        packed = tile_slice(
                            packed_rows,
                            (self.tile_columns // NZ_SCALE_COLUMNS, packed_width),
                            (col_tile, 0),
                        )
                        raw_scale_tile = _view(
                            sb_l1.next_write_raw(),
                            shape=(self.tile_columns // NZ_SCALE_COLUMNS, packed_width),
                            data_format="nd",
                        )
                        _copy(raw_scale_tile, packed.view(dtype=dtypes.uint8))
                    else:
                        _copy(
                            sb_l1.next_write(),
                            tile_slice(
                                descale_weight,
                                dsl.make_tiler(
                                    (self.tile_columns, reduction // 64, MX_SCALE_PAIR),
                                    alignment=(
                                        self.tile_columns,
                                        MX_SCALE_PAIR,
                                        MX_SCALE_PAIR,
                                    ),
                                ),
                                (col_tile, 0, 0),
                            ),
                            plan=sb_engine,
                        )
                sa_full = sa_l1.next_read()
                sb_full = sb_l1.next_read()
            for reduction_tile in range(reduction // self.tile_reduction + 1):
                if reduction_tile < reduction // self.tile_reduction:
                    _copy(
                        a_l1.next_write(),
                        tile_slice(
                            activation,
                            (self.tile_rows, self.tile_reduction),
                            (row_tile, reduction_tile),
                        ),
                        plan=nd_engine,
                    )
                    if dsl.const_expr(self.w_qr_quant_mode == 2):
                        if (
                            not defer_qr
                            or task != get_block_idx() - core_begin
                            or reduction_tile != 0
                        ):
                            _copy(
                                b_l1.next_write(),
                                tile_slice(
                                    weight_input,
                                    (self.tile_columns, self.tile_reduction),
                                    (col_tile, reduction_tile),
                                ),
                                plan=b_engine,
                            )
                    else:
                        if (
                            not self.join_after_prefetch
                            or task != get_block_idx()
                            or reduction_tile != 0
                        ):
                            _copy(
                                b_l1.next_write(),
                                tile_slice(
                                    weight_input,
                                    (self.tile_columns, self.tile_reduction),
                                    (col_tile, reduction_tile),
                                ),
                                plan=b_engine,
                            )
                if reduction_tile > 0:
                    a = a_l1.next_read()
                    b = b_l1.next_read()
                    if dsl.const_expr(self.split_l0):
                        for l0_tile in range(self.tile_reduction // self.l0_reduction):
                            scale_tile = (reduction_tile - 1) * (
                                self.tile_reduction // self.l0_reduction
                            ) + l0_tile
                            if dsl.const_expr(self.w_qr_quant_mode == 1):
                                _copy(
                                    a_l0.next_write(),
                                    tile_slice(
                                        a,
                                        (self.tile_rows, self.l0_reduction),
                                        (0, l0_tile),
                                    ),
                                    mx_scale=tile_slice(
                                        sa_full,
                                        (
                                            self.tile_rows,
                                            self.l0_reduction // MX_GROUP_ELEMENTS,
                                        ),
                                        (0, scale_tile),
                                    ),
                                )
                                _copy(
                                    _view(
                                        b_l0.next_write(),
                                        shape=(self.tile_columns, self.l0_reduction),
                                        data_format="nz",
                                    ),
                                    tile_slice(
                                        b,
                                        (self.tile_columns, self.l0_reduction),
                                        (0, l0_tile),
                                    ),
                                    mx_scale=tile_slice(
                                        sb_full,
                                        (
                                            self.l0_reduction // MX_GROUP_ELEMENTS,
                                            self.tile_columns,
                                        ),
                                        (scale_tile, 0),
                                    ),
                                )
                            else:
                                _copy(
                                    a_l0.next_write(),
                                    tile_slice(
                                        _view(a, dtype=dtypes.hifloat8),
                                        (self.tile_rows, self.l0_reduction),
                                        (0, l0_tile),
                                    ),
                                )
                                _copy(
                                    _view(
                                        b_l0.next_write(),
                                        shape=(self.tile_columns, self.l0_reduction),
                                        data_format="nz",
                                    ),
                                    tile_slice(
                                        _view(b, dtype=dtypes.hifloat8),
                                        (self.tile_columns, self.l0_reduction),
                                        (0, l0_tile),
                                    ),
                                )
                            matmul(
                                acc,
                                a_l0.next_read(),
                                _view(
                                    b_l0.next_read(),
                                    shape=(self.tile_columns, self.l0_reduction),
                                    data_format="nz",
                                ),
                                init=scale_tile == 0,
                                mx_mode=self.w_qr_quant_mode == 1,
                            )
                    else:
                        if dsl.const_expr(self.w_qr_quant_mode == 1):
                            _copy(
                                a_l0.next_write(),
                                a,
                                mx_scale=tile_slice(
                                    sa_full,
                                    (
                                        self.tile_rows,
                                        self.tile_reduction // MX_GROUP_ELEMENTS,
                                    ),
                                    (0, reduction_tile - 1),
                                ),
                            )
                            _copy(
                                _view(
                                    b_l0.next_write(),
                                    shape=(self.tile_columns, self.l0_reduction),
                                    data_format="nz",
                                ),
                                b,
                                mx_scale=tile_slice(
                                    sb_full,
                                    (
                                        self.tile_reduction // MX_GROUP_ELEMENTS,
                                        self.tile_columns,
                                    ),
                                    (reduction_tile - 1, 0),
                                ),
                            )
                        else:
                            _copy(a_l0.next_write(), _view(a, dtype=dtypes.hifloat8))
                            _copy(
                                _view(
                                    b_l0.next_write(),
                                    shape=(self.tile_columns, self.l0_reduction),
                                    data_format="nz",
                                ),
                                _view(b, dtype=dtypes.hifloat8),
                            )
                        matmul(
                            acc,
                            a_l0.next_read(),
                            _view(
                                b_l0.next_read(),
                                shape=(self.tile_columns, self.l0_reduction),
                                data_format="nz",
                            ),
                            init=reduction_tile == 1,
                            mx_mode=self.w_qr_quant_mode == 1,
                        )
            _copy(
                tile_slice(
                    projection_fp32,
                    (self.tile_rows, self.tile_columns),
                    (row_tile, col_tile),
                ),
                _view(c_l0.next_read(), shape=(self.tile_rows, self.tile_columns)),
            )
            if dsl.const_expr(self.head_pipeline):
                cube_sync_block_arrive(_sync_ops.PIPE.FIXPIPE, HEAD_READY_EVENT, mode=2)
                published_heads += 1
        for task in range(get_block_idx() - core_begin, task_count, task_stride):
            row_tile = task // head_count
            head = task % head_count
            if task + HEAD_PIPELINE_DEPTH * task_stride >= task_count:
                _sync_ops.cube_sync_block_wait(
                    _sync_ops.PIPE.S, HEAD_CONSUMED_EVENT, mode=2
                )
            head_weight = wkb[head, None, None].view((wkb_reduction, KV_ELEMENTS))
            head_output = output_heads[head, None, 0:KV_ELEMENTS]
            valid_rows = min(self.tile_rows, rows - row_tile * self.tile_rows)
            for column_tile in range(KV_ELEMENTS // WKB_C2V_COLUMNS):
                output_tile = tile_slice(
                    head_output,
                    (self.tile_rows, WKB_C2V_COLUMNS),
                    (row_tile, column_tile),
                )
                wkb_acc_slot, wkb_acc = _accumulator(c_l0, valid_rows, WKB_C2V_COLUMNS)
                if dsl.const_expr(self.w_qr_quant_mode == 2):
                    _copy(
                        _view(wkb_a_l1.next_write(), shape=(valid_rows, wkb_reduction)),
                        tile_slice(
                            input_tokens,
                            (self.tile_rows, wkb_reduction),
                            (row_tile, head),
                        ),
                        plan=wkb_load_engine,
                    )
                    a_full = _view(
                        wkb_a_l1.next_read(), shape=(valid_rows, wkb_reduction)
                    )
                    _copy(
                        wkb_b_l1.next_write(),
                        tile_slice(
                            head_weight,
                            (wkb_reduction, WKB_C2V_COLUMNS),
                            (0, column_tile),
                        ),
                        plan=wkb_load_engine,
                        l2_cache_ctl=WKB_WEIGHT_L2_CACHE_CTL,
                    )
                    b_full = wkb_b_l1.next_read()
                    for k_tile in range(wkb_reduction // WKB_C2V_REDUCTION):
                        _copy(
                            _view(
                                _view(
                                    a_l0.next_write(),
                                    dtype=dtypes.bfloat16,
                                    shape=(self.tile_rows, WKB_C2V_REDUCTION),
                                    data_format="nd",
                                ),
                                shape=(valid_rows, WKB_C2V_REDUCTION),
                                data_format="nz",
                            ),
                            tile_slice(
                                a_full, (self.tile_rows, WKB_C2V_REDUCTION), (0, k_tile)
                            ),
                        )
                        _copy(
                            _view(
                                b_l0.next_write(),
                                dtype=dtypes.bfloat16,
                                shape=(WKB_C2V_REDUCTION, WKB_C2V_COLUMNS),
                                data_format="zn",
                            ),
                            tile_slice(
                                b_full,
                                (WKB_C2V_REDUCTION, WKB_C2V_COLUMNS),
                                (k_tile, 0),
                            ),
                            transpose=True,
                        )
                        matmul(
                            wkb_acc_slot,
                            _view(
                                _view(
                                    a_l0.next_read(),
                                    dtype=dtypes.bfloat16,
                                    shape=(self.tile_rows, WKB_C2V_REDUCTION),
                                    data_format="nd",
                                ),
                                shape=(valid_rows, WKB_C2V_REDUCTION),
                                data_format="nz",
                            ),
                            _view(
                                b_l0.next_read(),
                                dtype=dtypes.bfloat16,
                                shape=(WKB_C2V_REDUCTION, WKB_C2V_COLUMNS),
                                data_format="zn",
                            ),
                            init=k_tile == 0,
                        )
                else:
                    for k_tile in range(wkb_reduction // WKB_C2V_REDUCTION):
                        _copy(
                            wkb_a_l1.next_write(),
                            tile_slice(
                                input_tokens,
                                (self.tile_rows, WKB_C2V_REDUCTION),
                                (
                                    row_tile,
                                    head * (wkb_reduction // WKB_C2V_REDUCTION)
                                    + k_tile,
                                ),
                            ),
                            plan=wkb_load_engine,
                        )
                        _copy(
                            wkb_b_l1.next_write(),
                            tile_slice(
                                head_weight,
                                (WKB_C2V_REDUCTION, WKB_C2V_COLUMNS),
                                (k_tile, column_tile),
                            ),
                            plan=wkb_load_engine,
                            l2_cache_ctl=WKB_WEIGHT_L2_CACHE_CTL,
                        )
                        _copy(
                            _view(
                                _view(
                                    a_l0.next_write(),
                                    dtype=dtypes.bfloat16,
                                    shape=(self.tile_rows, WKB_C2V_REDUCTION),
                                    data_format="nd",
                                ),
                                shape=(valid_rows, WKB_C2V_REDUCTION),
                                data_format="nz",
                            ),
                            wkb_a_l1.next_read(),
                        )
                        _copy(
                            _view(
                                b_l0.next_write(),
                                dtype=dtypes.bfloat16,
                                shape=(WKB_C2V_REDUCTION, WKB_C2V_COLUMNS),
                                data_format="zn",
                            ),
                            wkb_b_l1.next_read(),
                            transpose=True,
                        )
                        matmul(
                            wkb_acc_slot,
                            _view(
                                _view(
                                    a_l0.next_read(),
                                    dtype=dtypes.bfloat16,
                                    shape=(self.tile_rows, WKB_C2V_REDUCTION),
                                    data_format="nd",
                                ),
                                shape=(valid_rows, WKB_C2V_REDUCTION),
                                data_format="nz",
                            ),
                            _view(
                                b_l0.next_read(),
                                dtype=dtypes.bfloat16,
                                shape=(WKB_C2V_REDUCTION, WKB_C2V_COLUMNS),
                                data_format="zn",
                            ),
                            init=k_tile == 0,
                        )
                c_l0.next_read()
                _copy(output_tile, wkb_acc)
        _sync_ops.cube_sync_all()


class _Normalize:
    def __init__(
        self,
        width,
        w_qr_quant_mode,
        partial_bm=0,
        hif_resident_rows=HIF_QR_RESIDENT_ROWS,
    ):
        self.width = width
        self.w_qr_quant_mode = w_qr_quant_mode
        self.partial_bm = partial_bm
        self.hif_resident_rows = hif_resident_rows
        self.maximum = _GlobalMaximum(width)

    @jit
    def run_hif_quantized(
        self,
        projection_fp32: Tensor,
        norm_weight: Tensor,
        descale_activation: Tensor,
        descale_weight: Tensor,
        qr: Tensor,
        descale_qr: Tensor,
        norm_eps: float,
        post_norm_scale: float,
        max_scratch: Tensor,
        distributed: bool = False,
    ):
        # The caller's resident/distributed/fallback branches are exclusive.
        # Give each branch its own 0.8 allocation epoch.
        onchip_rewind(reset_sync_id=False)
        resident_rows = self.hif_resident_rows
        worker = 0
        worker_count = 1
        if dsl.const_expr(distributed):
            resident_rows = HIF_QR_ROWS_PER_VECTOR
            worker = get_block_idx() * VECTOR_SUBBLOCKS + get_subblock_id()
            worker_count = get_block_num() * VECTOR_SUBBLOCKS
        input_rows = _ring_buffer(MemLoc.UB, (1, self.width), dtypes.float32, depth=2)
        if dsl.const_expr(self.partial_bm > 0):
            incoming_rows = _ring_buffer(
                MemLoc.UB, (1, self.width), dtypes.float32, depth=2
            )
            partial_rows = (
                (qr.shape[0] + self.partial_bm - 1) // self.partial_bm * self.partial_bm
            )
            partial_view = projection_fp32.view((2 * partial_rows, self.width))
        normalized_rows = _buffer(
            MemLoc.UB, (resident_rows, self.width), dtypes.float32
        )
        norm_weight_buffer = _buffer(MemLoc.UB, (1, self.width), dtypes.float32)
        weight_scale_buffer = _buffer(MemLoc.UB, (1, self.width), dtypes.float32)
        reduction = _buffer(MemLoc.UB, (2 * VECTOR_FLOAT_ELEMENTS,), dtypes.float32)
        scalar = _buffer(MemLoc.UB, (1,), dtypes.float32)
        encoded_rows = _ring_buffer(MemLoc.UB, (1, self.width), dtypes.uint8, depth=2)
        _copy(norm_weight_buffer, norm_weight)
        _copy(weight_scale_buffer, tile_slice(descale_weight, (1, self.width), (0, 0)))
        activation_scale = descale_activation[0]
        with vf(mode="simd"):
            initial_mask = rr.full_mask()
            initial_zero = rr.vdups(0.0, dtypes.float32, mask=initial_mask)
            rr.vstore(reduction, 0, initial_zero, initial_mask)
            rr.vstore(reduction, VECTOR_FLOAT_ELEMENTS, initial_zero, initial_mask)
        for row in range(worker, qr.shape[0], worker_count):
            if dsl.const_expr(self.partial_bm > 0):
                _copy(
                    input_rows.next_write(),
                    tile_slice(partial_view, (1, self.width), (row, 0)),
                )
                _copy(
                    incoming_rows.next_write(),
                    tile_slice(partial_view, (1, self.width), (partial_rows + row, 0)),
                )
                values = input_rows.next_read()
                incoming = incoming_rows.next_read()
                with vf(mode="simd"):
                    merge_mask = rr.full_mask()
                    for offset in range(0, self.width, VECTOR_FLOAT_ELEMENTS):
                        merged = rr.vadd(
                            rr.vload(values, offset),
                            rr.vload(incoming, offset),
                            mask=merge_mask,
                        )
                        rr.vstore(values, offset, merged, merge_mask)
                    rr.vmem_bar("vst_vld")
            else:
                _copy(
                    input_rows.next_write(),
                    tile_slice(projection_fp32, (1, self.width), (row, 0)),
                )
                values = input_rows.next_read()
            normalized = tile_slice(
                normalized_rows, (1, self.width), ((row - worker) // worker_count, 0)
            )
            with vf(mode="simd"):
                norm_mask = rr.full_mask()
                squared_sum = rr.vdups(0.0, dtypes.float32, mask=norm_mask)
                for offset in range(0, self.width, VECTOR_FLOAT_ELEMENTS):
                    raw_value = rr.vload(values, offset)
                    raw_value = rr.vmul(
                        raw_value, rr.vload(weight_scale_buffer, offset), mask=norm_mask
                    )
                    raw_value = rr.vmuls(raw_value, activation_scale, mask=norm_mask)
                    rr.vstore(values, offset, raw_value, norm_mask)
                    squared_sum = rr.vadd(
                        squared_sum,
                        rr.vreduce_sum(
                            rr.vmul(raw_value, raw_value, mask=norm_mask),
                            mask=norm_mask,
                        ),
                        mask=norm_mask,
                    )
                rr.vmem_bar("vst_vld")
                total = rr.vdup(squared_sum, mask=norm_mask)
                mean_square = rr.vdiv(
                    total,
                    rr.vdups(float(self.width), dtypes.float32, mask=norm_mask),
                    mask=norm_mask,
                    precision_mode=True,
                )
                norm_denominator = rr.vsqrt(
                    rr.vadds(mean_square, norm_eps, mask=norm_mask), mask=norm_mask
                )
                row_maximum = rr.vload(reduction, 0)
                row_nan = rr.vload(reduction, VECTOR_FLOAT_ELEMENTS)
                one = rr.vdups(1.0, dtypes.float32, mask=norm_mask)
                zero = rr.vdups(0.0, dtypes.float32, mask=norm_mask)
                for offset in range(0, self.width, VECTOR_FLOAT_ELEMENTS):
                    norm_value = rr.vdiv(
                        rr.vload(values, offset),
                        norm_denominator,
                        mask=norm_mask,
                        precision_mode=True,
                    )
                    norm_value = rr.vmul(
                        norm_value, rr.vload(norm_weight_buffer, offset), mask=norm_mask
                    )
                    norm_value = rr.vmuls(norm_value, post_norm_scale, mask=norm_mask)
                    rr.vstore(normalized, offset, norm_value, norm_mask)
                    row_maximum = rr.vmax(
                        row_maximum, rr.vabs(norm_value, mask=norm_mask), mask=norm_mask
                    )
                    row_nan = rr.vmax(
                        row_nan,
                        rr.vselect(
                            one,
                            zero,
                            cond_mask=rr.vne(norm_value, norm_value, mask=norm_mask),
                        ),
                        mask=norm_mask,
                    )
                rr.vstore(reduction, 0, row_maximum, norm_mask)
                rr.vstore(reduction, VECTOR_FLOAT_ELEMENTS, row_nan, norm_mask)
        if dsl.const_expr(not distributed):
            with vf(mode="simd"):
                scale_mask = rr.full_mask()
                rr.vmem_bar("vst_vld")
                maximum = rr.vdup(
                    rr.vreduce_max(rr.vload(reduction, 0), mask=scale_mask),
                    mask=scale_mask,
                )
                has_nan = rr.vdup(
                    rr.vreduce_max(
                        rr.vload(reduction, VECTOR_FLOAT_ELEMENTS), mask=scale_mask
                    ),
                    mask=scale_mask,
                )
                scale_value = rr.vdiv(
                    maximum,
                    rr.vdups(HIF8_MAX_VALUE, dtypes.float32, mask=scale_mask),
                    mask=scale_mask,
                    precision_mode=True,
                )
                nan_value = rr.vreinterpret(
                    rr.vdups(2143289344, dtypes.uint32, mask=scale_mask), dtypes.float32
                )
                scale_value = rr.vselect(
                    nan_value,
                    scale_value,
                    cond_mask=rr.vgts(has_nan, 0.0, mask=scale_mask),
                )
                rr.vstore_first(scalar, 0, scale_value)
        if dsl.const_expr(distributed):
            partials = max_scratch.view(
                (
                    max_scratch.shape[0]
                    * max_scratch.shape[1]
                    // MAXIMUM_PARTIAL_ELEMENTS,
                    MAXIMUM_PARTIAL_ELEMENTS,
                )
            )
            partial = _buffer(MemLoc.UB, (1, MAXIMUM_PARTIAL_ELEMENTS), dtypes.float32)
            with vf(mode="simd"):
                partial_mask = rr.update_mask(MAXIMUM_PARTIAL_ELEMENTS, elem_bits=32)[0]
                full_mask = rr.full_mask()
                rr.vstore(
                    partial,
                    0,
                    rr.vdups(0.0, dtypes.float32, mask=partial_mask),
                    partial_mask,
                )
                rr.vmem_bar("vst_vld")
                maximum = rr.vdup(
                    rr.vreduce_max(rr.vload(reduction, 0), mask=full_mask),
                    mask=full_mask,
                )
                has_nan = rr.vdup(
                    rr.vreduce_max(
                        rr.vload(reduction, VECTOR_FLOAT_ELEMENTS), mask=full_mask
                    ),
                    mask=full_mask,
                )
                rr.vstore_first(partial, 0, maximum)
                rr.vstore_first(partial, 1, has_nan)
            _copy(
                tile_slice(partials, (1, MAXIMUM_PARTIAL_ELEMENTS), (worker, 0)),
                partial,
            )
            # All workers, including empty ones, publish before reduction.
            # Do not rewind: normalized_rows remains live across this join.
            _mix_join()
            self.maximum.reduce_partials(
                max_scratch,
                scalar,
                worker_count,
                self.width // MAXIMUM_PARTIAL_ELEMENTS,
            )
        if worker == 0:
            _copy(descale_qr, scalar)
        for row in range(worker, qr.shape[0], worker_count):
            quant_input = tile_slice(
                normalized_rows, (1, self.width), ((row - worker) // worker_count, 0)
            )
            payload = encoded_rows.next_write()
            with vf(mode="simd"):
                quant_mask = rr.full_mask()
                rr.vmem_bar("vst_vld")
                qr_descale_value = rr.vload_broadcast(scalar, 0)
                zero_scale = rr.veqs(qr_descale_value, 0.0, mask=quant_mask)
                quant_denominator = rr.vselect(
                    rr.vdups(1.0, dtypes.float32, mask=quant_mask),
                    qr_descale_value,
                    cond_mask=zero_scale,
                )
                for offset in range(0, self.width, VECTOR_FLOAT_ELEMENTS):
                    quant_value = rr.vdiv(
                        rr.vload(quant_input, offset),
                        quant_denominator,
                        mask=quant_mask,
                        precision_mode=True,
                    )
                    quant_value = rr.vselect(
                        rr.vdups(0.0, dtypes.float32, mask=quant_mask),
                        quant_value,
                        cond_mask=zero_scale,
                    )
                    encoded = _quantize_cast(quant_value, quant_mask, True)
                    rr.vstore_pack(
                        payload,
                        offset,
                        rr.vreinterpret(encoded, dtypes.float8_e4m3fn),
                        quant_mask,
                        pack_mode=rr.PackMode.B32_TO_B8,
                    )
            _copy(tile_slice(qr, (1, self.width), (row, 0)), encoded_rows.next_read())

    @jit
    def run(
        self,
        projection_fp32: Tensor,
        norm_weight: Tensor,
        descale_activation: Tensor,
        descale_weight: Tensor,
        normalized_fp32: Tensor,
        norm_eps: float,
        post_norm_scale: float,
        max_scratch: Tensor,
        collect_max: bool = False,
    ):
        onchip_rewind(reset_sync_id=False)
        input_rows = _ring_buffer(MemLoc.UB, (1, self.width), dtypes.float32, depth=2)
        output_rows = _ring_buffer(MemLoc.UB, (1, self.width), dtypes.float32, depth=2)
        norm_weight_buffer = _buffer(MemLoc.UB, (1, self.width), dtypes.float32)
        _copy(norm_weight_buffer, norm_weight)
        if dsl.const_expr(self.w_qr_quant_mode == 2):
            weight_scale_buf = _buffer(MemLoc.UB, (1, self.width), dtypes.float32)
            _copy(weight_scale_buf, tile_slice(descale_weight, (1, self.width), (0, 0)))
            activation_scale = descale_activation[0]
        worker = get_block_idx() * VECTOR_SUBBLOCKS + get_subblock_id()
        worker_count = get_block_num() * VECTOR_SUBBLOCKS
        reduction = norm_weight_buffer
        if dsl.const_expr(collect_max):
            reduction = _buffer(MemLoc.UB, (2 * VECTOR_FLOAT_ELEMENTS,), dtypes.float32)
            with vf(mode="simd"):
                initial_mask = rr.full_mask()
                initial_zero = rr.vdups(0.0, dtypes.float32, mask=initial_mask)
                rr.vstore(reduction, 0, initial_zero, initial_mask)
                rr.vstore(reduction, VECTOR_FLOAT_ELEMENTS, initial_zero, initial_mask)
        if dsl.const_expr(self.partial_bm > 0):
            incoming_rows = _ring_buffer(
                MemLoc.UB, (1, self.width), dtypes.float32, depth=2
            )
            partial_rows = (
                (normalized_fp32.shape[0] + self.partial_bm - 1)
                // self.partial_bm
                * self.partial_bm
            )
            partial_view = projection_fp32.view((2 * partial_rows, self.width))
        for row in range(worker, normalized_fp32.shape[0], worker_count):
            if dsl.const_expr(self.partial_bm > 0):
                _copy(
                    input_rows.next_write(),
                    tile_slice(partial_view, (1, self.width), (row, 0)),
                )
                _copy(
                    incoming_rows.next_write(),
                    tile_slice(partial_view, (1, self.width), (partial_rows + row, 0)),
                )
                values = input_rows.next_read()
                incoming = incoming_rows.next_read()
                with vf(mode="simd"):
                    merge_mask = rr.full_mask()
                    for offset in range(0, self.width, VECTOR_FLOAT_ELEMENTS):
                        merged = rr.vadd(
                            rr.vload(values, offset),
                            rr.vload(incoming, offset),
                            mask=merge_mask,
                        )
                        rr.vstore(values, offset, merged, merge_mask)
                    rr.vmem_bar("vst_vld")
            else:
                _copy(
                    input_rows.next_write(),
                    tile_slice(projection_fp32, (1, self.width), (row, 0)),
                )
                values = input_rows.next_read()
            normalized = output_rows.next_write()
            if dsl.const_expr(self.w_qr_quant_mode == 2):
                with vf(mode="simd"):
                    mask = rr.full_mask()
                    for offset in range(0, self.width, VECTOR_FLOAT_ELEMENTS):
                        value = rr.vload(values, offset)
                        value = rr.vmul(
                            value, rr.vload(weight_scale_buf, offset), mask=mask
                        )
                        value = rr.vmuls(value, activation_scale, mask=mask)
                        rr.vstore(values, offset, value, mask)
            with vf(mode="simd"):
                mask = rr.full_mask()
                rr.vmem_bar("vst_vld")
                squared_sum = rr.vdups(0.0, dtypes.float32, mask=mask)
                for offset in range(0, self.width, VECTOR_FLOAT_ELEMENTS):
                    value = rr.vload(values, offset)
                    squared_sum = rr.vadd(
                        squared_sum,
                        rr.vreduce_sum(rr.vmul(value, value, mask=mask), mask=mask),
                        mask=mask,
                    )
                total = rr.vdup(squared_sum, mask=mask)
                mean_square = rr.vdiv(
                    total,
                    rr.vdups(float(self.width), dtypes.float32, mask=mask),
                    mask=mask,
                    precision_mode=True,
                )
                denominator = rr.vsqrt(
                    rr.vadds(mean_square, norm_eps, mask=mask), mask=mask
                )
                for offset in range(0, self.width, VECTOR_FLOAT_ELEMENTS):
                    value = rr.vdiv(
                        rr.vload(values, offset),
                        denominator,
                        mask=mask,
                        precision_mode=True,
                    )
                    value = rr.vmul(
                        value, rr.vload(norm_weight_buffer, offset), mask=mask
                    )
                    value = rr.vmuls(value, post_norm_scale, mask=mask)
                    rr.vstore(normalized, offset, value, mask)
            if dsl.const_expr(collect_max):
                with vf(mode="simd"):
                    maximum_mask = rr.full_mask()
                    rr.vmem_bar("vst_vld")
                    row_maximum = rr.vload(reduction, 0)
                    row_nan = rr.vload(reduction, VECTOR_FLOAT_ELEMENTS)
                    one = rr.vdups(1.0, dtypes.float32, mask=maximum_mask)
                    zero = rr.vdups(0.0, dtypes.float32, mask=maximum_mask)
                    for offset in range(0, self.width, VECTOR_FLOAT_ELEMENTS):
                        norm_value = rr.vload(normalized, offset)
                        row_maximum = rr.vmax(
                            row_maximum,
                            rr.vabs(norm_value, mask=maximum_mask),
                            mask=maximum_mask,
                        )
                        row_nan = rr.vmax(
                            row_nan,
                            rr.vselect(
                                one,
                                zero,
                                cond_mask=rr.vne(
                                    norm_value, norm_value, mask=maximum_mask
                                ),
                            ),
                            mask=maximum_mask,
                        )
                    rr.vstore(reduction, 0, row_maximum, maximum_mask)
                    rr.vstore(reduction, VECTOR_FLOAT_ELEMENTS, row_nan, maximum_mask)
            _copy(
                tile_slice(normalized_fp32, (1, self.width), (row, 0)),
                output_rows.next_read(),
            )
        if dsl.const_expr(collect_max):
            partials = max_scratch.view(
                (
                    max_scratch.shape[0]
                    * max_scratch.shape[1]
                    // MAXIMUM_PARTIAL_ELEMENTS,
                    MAXIMUM_PARTIAL_ELEMENTS,
                )
            )
            partial = _buffer(MemLoc.UB, (1, MAXIMUM_PARTIAL_ELEMENTS), dtypes.float32)
            with vf(mode="simd"):
                final_mask = rr.full_mask()
                partial_mask = rr.update_mask(MAXIMUM_PARTIAL_ELEMENTS, elem_bits=32)[0]
                rr.vstore(
                    partial,
                    0,
                    rr.vdups(0.0, dtypes.float32, mask=partial_mask),
                    partial_mask,
                )
                rr.vmem_bar("vst_vld")
                maximum = rr.vdup(
                    rr.vreduce_max(rr.vload(reduction, 0), mask=final_mask),
                    mask=final_mask,
                )
                has_nan = rr.vdup(
                    rr.vreduce_max(
                        rr.vload(reduction, VECTOR_FLOAT_ELEMENTS), mask=final_mask
                    ),
                    mask=final_mask,
                )
                rr.vstore_first(partial, 0, maximum)
                rr.vstore_first(partial, 1, has_nan)
            _copy(
                tile_slice(partials, (1, MAXIMUM_PARTIAL_ELEMENTS), (worker, 0)),
                partial,
            )

    @jit
    def run_mx_quantized(
        self,
        projection_fp32: Tensor,
        norm_weight: Tensor,
        qr: Tensor,
        descale_qr: Tensor,
        norm_eps: float,
        post_norm_scale: float,
    ):
        group_count = self.width // MX_GROUP_ELEMENTS
        padded_groups = (
            (group_count + VECTOR_FLOAT_ELEMENTS - 1)
            // VECTOR_FLOAT_ELEMENTS
            * VECTOR_FLOAT_ELEMENTS
        )
        input_rows = _ring_buffer(MemLoc.UB, (1, self.width), dtypes.float32, depth=2)
        if dsl.const_expr(self.partial_bm > 0):
            merged_rows = _ring_buffer(
                MemLoc.UB, (1, self.width), dtypes.float32, depth=2
            )
            incoming_rows = _ring_buffer(
                MemLoc.UB, (1, self.width), dtypes.float32, depth=2
            )
            partial_rows = (
                (qr.shape[0] + self.partial_bm - 1) // self.partial_bm * self.partial_bm
            )
            partial_view = projection_fp32.view((2 * partial_rows, self.width))
        normalized_rows = _ring_buffer(
            MemLoc.UB,
            (1, self.width),
            dtypes.float32,
            depth=2,
            capacity=(1, self.width + VECTOR_FLOAT_ELEMENTS),
        )
        output_rows = _ring_buffer(
            MemLoc.UB, (1, self.width), dtypes.float8_e4m3fn, depth=2
        )
        scales = _ring_buffer(MemLoc.UB, (1, group_count), dtypes.float8_e8m0, depth=2)
        norm_weight_buffer = _buffer(MemLoc.UB, (1, self.width), dtypes.float32)
        exponent_buffer = _buffer(MemLoc.UB, (padded_groups,), dtypes.uint32)
        _copy(norm_weight_buffer, norm_weight)
        worker = get_block_idx() * VECTOR_SUBBLOCKS + get_subblock_id()
        worker_count = get_block_num() * VECTOR_SUBBLOCKS
        for row in range(worker, qr.shape[0], worker_count):
            if dsl.const_expr(self.partial_bm > 0):
                _copy(
                    merged_rows.next_write(),
                    tile_slice(partial_view, (1, self.width), (row, 0)),
                )
                _copy(
                    incoming_rows.next_write(),
                    tile_slice(partial_view, (1, self.width), (partial_rows + row, 0)),
                )
                merged_row = merged_rows.next_read()
                incoming_row = incoming_rows.next_read()
                with vf(mode="simd"):
                    merge_mask = rr.full_mask()
                    for offset in range(0, self.width, VECTOR_FLOAT_ELEMENTS):
                        merged = rr.vadd(
                            rr.vload(merged_row, offset),
                            rr.vload(incoming_row, offset),
                            mask=merge_mask,
                        )
                        rr.vstore(merged_row, offset, merged, merge_mask)
                values = merged_row
            else:
                _copy(
                    input_rows.next_write(),
                    tile_slice(projection_fp32, (1, self.width), (row, 0)),
                )
                values = input_rows.next_read()
            normalized = normalized_rows.next_write()
            with vf(mode="simd"):
                mask = rr.full_mask()
                rr.vmem_bar("vst_vld")
                squared_sum = rr.vdups(0.0, dtypes.float32, mask=mask)
                for offset in range(0, self.width, VECTOR_FLOAT_ELEMENTS):
                    value = rr.vload(values, offset)
                    squared_sum = rr.vadd(
                        squared_sum,
                        rr.vreduce_sum(rr.vmul(value, value, mask=mask), mask=mask),
                        mask=mask,
                    )
                total = rr.vdup(squared_sum, mask=mask)
                mean_square = rr.vdiv(
                    total,
                    rr.vdups(float(self.width), dtypes.float32, mask=mask),
                    mask=mask,
                    precision_mode=True,
                )
                denominator = rr.vsqrt(
                    rr.vadds(mean_square, norm_eps, mask=mask), mask=mask
                )
                for offset in range(0, self.width, VECTOR_FLOAT_ELEMENTS):
                    value = rr.vdiv(
                        rr.vload(values, offset),
                        denominator,
                        mask=mask,
                        precision_mode=True,
                    )
                    value = rr.vmul(
                        value, rr.vload(norm_weight_buffer, offset), mask=mask
                    )
                    value = rr.vmuls(value, post_norm_scale, mask=mask)
                    rr.vstore(normalized, offset, value, mask)
            normalized_values = normalized_rows.next_read()
            payload = output_rows.next_write()
            scale = scales.next_write()
            with vf(mode="simd"):
                rr.vmem_bar("vst_vld")
                mask = rr.update_mask(MX_GROUP_ELEMENTS, elem_bits=32)[0]
                zero = rr.vdups(0.0, dtypes.float32, mask=mask)
                nan = rr.vreinterpret(
                    rr.vdups(2143289344, dtypes.uint32, mask=mask), dtypes.float32
                )
                for group in range(group_count):
                    value = rr.vload(normalized_values, group * MX_GROUP_ELEMENTS)
                    # Local SAT+ROUND semantics; vcast saturation changes kernel CTRL.
                    value = rr.vselect(
                        value, zero, cond_mask=rr.veq(value, value, mask=mask)
                    )
                    value = rr.vmins(
                        rr.vmaxs(value, -BF16_MAX_VALUE, mask=mask),
                        BF16_MAX_VALUE,
                        mask=mask,
                    )
                    rounded = rr.vcast(
                        value,
                        dtypes.bfloat16,
                        mask=mask,
                        rounding=rr.RoundingMode.RNA,
                        reg_layout=RegLayout.ZERO,
                    )
                    value = rr.vcast(rounded, dtypes.float32, mask=mask)
                    bits = rr.vreinterpret(value, dtypes.uint32)
                    field = rr.vbitwise_and(
                        rr.vshr(bits, FP32_EXPONENT_SHIFT, mask=mask),
                        rr.vdups(FP32_EXPONENT_MASK, dtypes.uint32, mask=mask),
                        mask=mask,
                    )
                    field = rr.vdup(
                        rr.vreduce_max(rr.vreinterpret(field, dtypes.int32), mask=mask),
                        mask=mask,
                    )
                    field_fp = rr.vcast(field, dtypes.float32, mask=mask)
                    raw_fp = rr.vmins(
                        rr.vmaxs(
                            rr.vadds(field_fp, -E4M3_EXPONENT_HEADROOM, mask=mask),
                            0.0,
                            mask=mask,
                        ),
                        254.0,
                        mask=mask,
                    )
                    special = rr.veqs(field_fp, 255.0, mask=mask)
                    zero_exponent = rr.veqs(field_fp, 0.0, mask=mask)
                    inverse_field = rr.vreinterpret(
                        rr.vcast(
                            rr.vsub(
                                rr.vdups(254.0, dtypes.float32, mask=mask),
                                raw_fp,
                                mask=mask,
                            ),
                            dtypes.int32,
                            mask=mask,
                        ),
                        dtypes.uint32,
                    )
                    inverse = rr.vreinterpret(
                        rr.vshl(inverse_field, FP32_EXPONENT_SHIFT, mask=mask),
                        dtypes.float32,
                    )
                    inverse = rr.vselect(zero, inverse, cond_mask=zero_exponent)
                    inverse = rr.vselect(nan, inverse, cond_mask=special)
                    raw_fp = rr.vselect(zero, raw_fp, cond_mask=zero_exponent)
                    raw_fp = rr.vselect(
                        rr.vdups(255.0, dtypes.float32, mask=mask),
                        raw_fp,
                        cond_mask=special,
                    )
                    raw = rr.vreinterpret(
                        rr.vcast(raw_fp, dtypes.int32, mask=mask), dtypes.uint32
                    )
                    rr.vstore_first(exponent_buffer, group, raw)
                    quant_value = rr.vmul(value, inverse, mask=mask)
                    rounded = rr.vcast(
                        quant_value,
                        dtypes.bfloat16,
                        mask=mask,
                        rounding=rr.RoundingMode.RN,
                        reg_layout=RegLayout.ZERO,
                    )
                    quant_value = rr.vcast(rounded, dtypes.float32, mask=mask)
                    encoded = _quantize_cast(quant_value, mask, False)
                    rr.vstore_pack(
                        payload,
                        group * MX_GROUP_ELEMENTS,
                        encoded,
                        mask,
                        pack_mode=rr.PackMode.B32_TO_B8,
                    )
            with vf(mode="simd"):
                rr.vmem_bar("vst_vld")
                for offset in range(0, padded_groups, VECTOR_FLOAT_ELEMENTS):
                    mask = rr.update_mask(
                        min(VECTOR_FLOAT_ELEMENTS, group_count - offset), elem_bits=32
                    )[0]
                    rr.vstore_pack(
                        scale,
                        offset,
                        rr.vload(exponent_buffer, offset),
                        mask,
                        pack_mode=rr.PackMode.B32_TO_B8,
                    )
            _copy(tile_slice(qr, (1, self.width), (row, 0)), output_rows.next_read())
            _copy(
                tile_slice(descale_qr, (1, group_count), (row, 0)), scales.next_read()
            )


class _Float32Storage:
    dtype = dtypes.float32

    @jit
    def load(self, values, offset):
        return rr.vload(values, offset)


class _Bfloat16Storage:
    dtype = dtypes.bfloat16

    @jit
    def load(self, values, offset):
        return rr.vcast(
            rr.vload_unpack(values, offset, unpack_mode=rr.UnpackMode.UNPACK2),
            dtypes.float32,
            mask=rr.full_mask(),
        )


class _GlobalMaximum:
    def __init__(self, width, storage=None):
        self.width = width
        self.storage = storage if storage is not None else _Float32Storage()

    @jit
    def run(self, source_values: Tensor, descale_output: Tensor):
        input_rows = _ring_buffer(
            MemLoc.UB, (1, self.width), self.storage.dtype, depth=2
        )
        reduction = _buffer(MemLoc.UB, (2 * VECTOR_FLOAT_ELEMENTS,), dtypes.float32)
        scalar = _buffer(MemLoc.UB, (1,), dtypes.float32)
        with vf(mode="simd"):
            init_mask = rr.full_mask()
            init_zero = rr.vdups(0.0, dtypes.float32, mask=init_mask)
            rr.vstore(reduction, 0, init_zero, init_mask)
            rr.vstore(reduction, VECTOR_FLOAT_ELEMENTS, init_zero, init_mask)
        for row in range(source_values.shape[0]):
            _copy(
                input_rows.next_write(),
                tile_slice(source_values, (1, self.width), (row, 0)),
            )
            values = input_rows.next_read()
            with vf(mode="simd"):
                row_mask = rr.full_mask()
                rr.vmem_bar("vst_vld")
                row_maximum = rr.vload(reduction, 0)
                row_nan = rr.vload(reduction, VECTOR_FLOAT_ELEMENTS)
                row_one = rr.vdups(1.0, dtypes.float32, mask=row_mask)
                row_zero = rr.vdups(0.0, dtypes.float32, mask=row_mask)
                for offset in range(0, self.width, VECTOR_FLOAT_ELEMENTS):
                    row_value = self.storage.load(values, offset)
                    row_maximum = rr.vmax(
                        row_maximum, rr.vabs(row_value, mask=row_mask), mask=row_mask
                    )
                    row_nan_value = rr.vselect(
                        row_one,
                        row_zero,
                        cond_mask=rr.vne(row_value, row_value, mask=row_mask),
                    )
                    row_nan = rr.vmax(row_nan, row_nan_value, mask=row_mask)
                rr.vstore(reduction, 0, row_maximum, row_mask)
                rr.vstore(reduction, VECTOR_FLOAT_ELEMENTS, row_nan, row_mask)
        with vf(mode="simd"):
            final_mask = rr.full_mask()
            rr.vmem_bar("vst_vld")
            final_maximum = rr.vdup(
                rr.vreduce_max(rr.vload(reduction, 0), mask=final_mask), mask=final_mask
            )
            final_nan = rr.vdup(
                rr.vreduce_max(
                    rr.vload(reduction, VECTOR_FLOAT_ELEMENTS), mask=final_mask
                ),
                mask=final_mask,
            )
            final_scale = rr.vdiv(
                final_maximum,
                rr.vdups(HIF8_MAX_VALUE, dtypes.float32, mask=final_mask),
                mask=final_mask,
                precision_mode=True,
            )
            nan_value = rr.vreinterpret(
                rr.vdups(2143289344, dtypes.uint32, mask=final_mask), dtypes.float32
            )
            final_scale = rr.vselect(
                nan_value,
                final_scale,
                cond_mask=rr.vgts(final_nan, 0.0, mask=final_mask),
            )
            rr.vstore_first(scalar, 0, final_scale)
        _copy(descale_output, scalar)

    @jit
    def run_parallel(
        self,
        source_values: Tensor,
        descale_output: Tensor,
        scratch_fp32: Tensor,
        partial_capacity: int,
    ):
        worker = get_block_idx() * VECTOR_SUBBLOCKS + get_subblock_id()
        partial_count = get_block_num() * VECTOR_SUBBLOCKS
        if partial_count > source_values.shape[0]:
            partial_count = source_values.shape[0]
        if partial_count > partial_capacity:
            partial_count = partial_capacity
        partials = scratch_fp32.view(
            (
                scratch_fp32.shape[0]
                * scratch_fp32.shape[1]
                // MAXIMUM_PARTIAL_ELEMENTS,
                MAXIMUM_PARTIAL_ELEMENTS,
            )
        )
        input_rows = _ring_buffer(
            MemLoc.UB, (1, self.width), self.storage.dtype, depth=2
        )
        reduction = _buffer(MemLoc.UB, (2 * VECTOR_FLOAT_ELEMENTS,), dtypes.float32)
        partial = _buffer(MemLoc.UB, (1, MAXIMUM_PARTIAL_ELEMENTS), dtypes.float32)
        if worker < partial_count:
            with vf(mode="simd"):
                init_mask = rr.full_mask()
                init_zero = rr.vdups(0.0, dtypes.float32, mask=init_mask)
                rr.vstore(reduction, 0, init_zero, init_mask)
                rr.vstore(reduction, VECTOR_FLOAT_ELEMENTS, init_zero, init_mask)
            for row in range(worker, source_values.shape[0], partial_count):
                _copy(
                    input_rows.next_write(),
                    tile_slice(source_values, (1, self.width), (row, 0)),
                )
                values = input_rows.next_read()
                with vf(mode="simd"):
                    row_mask = rr.full_mask()
                    rr.vmem_bar("vst_vld")
                    row_maximum = rr.vload(reduction, 0)
                    row_nan = rr.vload(reduction, VECTOR_FLOAT_ELEMENTS)
                    row_one = rr.vdups(1.0, dtypes.float32, mask=row_mask)
                    row_zero = rr.vdups(0.0, dtypes.float32, mask=row_mask)
                    for offset in range(0, self.width, VECTOR_FLOAT_ELEMENTS):
                        row_value = self.storage.load(values, offset)
                        row_maximum = rr.vmax(
                            row_maximum,
                            rr.vabs(row_value, mask=row_mask),
                            mask=row_mask,
                        )
                        row_nan = rr.vmax(
                            row_nan,
                            rr.vselect(
                                row_one,
                                row_zero,
                                cond_mask=rr.vne(row_value, row_value, mask=row_mask),
                            ),
                            mask=row_mask,
                        )
                    rr.vstore(reduction, 0, row_maximum, row_mask)
                    rr.vstore(reduction, VECTOR_FLOAT_ELEMENTS, row_nan, row_mask)
            with vf(mode="simd"):
                reduce_mask = rr.full_mask()
                partial_mask = rr.update_mask(MAXIMUM_PARTIAL_ELEMENTS, elem_bits=32)[0]
                rr.vstore(
                    partial,
                    0,
                    rr.vdups(0.0, dtypes.float32, mask=partial_mask),
                    partial_mask,
                )
                rr.vmem_bar("vst_vld")
                partial_maximum = rr.vdup(
                    rr.vreduce_max(rr.vload(reduction, 0), mask=reduce_mask),
                    mask=reduce_mask,
                )
                partial_nan = rr.vdup(
                    rr.vreduce_max(
                        rr.vload(reduction, VECTOR_FLOAT_ELEMENTS), mask=reduce_mask
                    ),
                    mask=reduce_mask,
                )
                rr.vstore_first(partial, 0, partial_maximum)
                rr.vstore_first(partial, 1, partial_nan)
            _copy(
                tile_slice(partials, (1, MAXIMUM_PARTIAL_ELEMENTS), (worker, 0)),
                partial,
            )
        # Every active Vector publishes a disjoint 32-byte slot.  QA Matmul
        # storage is reused only after all QA normalization readers finish.
        _mix_join()
        onchip_rewind()
        self.finish_partials(
            scratch_fp32, descale_output, partial_count, partial_capacity
        )

    @jit
    def finish_partials(
        self,
        scratch_fp32: Tensor,
        descale_output: Tensor,
        partial_count: int,
        partial_capacity: int,
    ):
        worker = get_block_idx() * VECTOR_SUBBLOCKS + get_subblock_id()
        if worker == 0:
            scalar = _buffer(MemLoc.UB, (1,), dtypes.float32)
            self.reduce_partials(scratch_fp32, scalar, partial_count, partial_capacity)
            _copy(descale_output, scalar)

    @jit
    def reduce_partials(
        self,
        scratch_fp32: Tensor,
        scalar: Tensor,
        partial_count: int,
        partial_capacity: int,
    ):
        partials = scratch_fp32.view(
            (
                scratch_fp32.shape[0]
                * scratch_fp32.shape[1]
                // MAXIMUM_PARTIAL_ELEMENTS,
                MAXIMUM_PARTIAL_ELEMENTS,
            )
        )
        inputs = _buffer(
            MemLoc.UB,
            (partial_count, MAXIMUM_PARTIAL_ELEMENTS),
            dtypes.float32,
            capacity=(partial_capacity, MAXIMUM_PARTIAL_ELEMENTS),
        )
        _copy(
            inputs,
            tile_slice(
                partials,
                dsl.make_tiler(
                    (partial_count, MAXIMUM_PARTIAL_ELEMENTS),
                    alignment=(1, MAXIMUM_PARTIAL_ELEMENTS),
                ),
                (0, 0),
            ),
        )
        with vf(mode="simd"):
            final_mask = rr.full_mask()
            final_maximum = rr.vdups(0.0, dtypes.float32, mask=final_mask)
            final_nan = rr.vdups(0.0, dtypes.float32, mask=final_mask)
            for partial_index in range(partial_count):
                final_maximum = rr.vmax(
                    final_maximum,
                    rr.vload_broadcast(
                        inputs, partial_index * MAXIMUM_PARTIAL_ELEMENTS
                    ),
                    mask=final_mask,
                )
                final_nan = rr.vmax(
                    final_nan,
                    rr.vload_broadcast(
                        inputs, partial_index * MAXIMUM_PARTIAL_ELEMENTS + 1
                    ),
                    mask=final_mask,
                )
            scale = rr.vdiv(
                final_maximum,
                rr.vdups(HIF8_MAX_VALUE, dtypes.float32, mask=final_mask),
                mask=final_mask,
                precision_mode=True,
            )
            nan_value = rr.vreinterpret(
                rr.vdups(2143289344, dtypes.uint32, mask=final_mask), dtypes.float32
            )
            scale = rr.vselect(
                nan_value, scale, cond_mask=rr.vgts(final_nan, 0.0, mask=final_mask)
            )
            rr.vstore_first(scalar, 0, scale)


class _Hif8Quantize:
    def __init__(self, width, precision_mode=True, storage=None):
        self.width = width
        self.precision_mode = precision_mode
        self.storage = storage if storage is not None else _Float32Storage()

    @jit
    def run(
        self, source_values: Tensor, quantized_values: Tensor, descale_output: Tensor
    ):
        scalar = _buffer(MemLoc.UB, (1,), dtypes.float32)
        _copy(scalar, descale_output)
        self.run_with_scale(source_values, quantized_values, scalar)

    @jit
    def run_with_scale(
        self, source_values: Tensor, quantized_values: Tensor, scalar: Tensor
    ):
        input_rows = _ring_buffer(
            MemLoc.UB, (1, self.width), self.storage.dtype, depth=2
        )
        output_rows = _ring_buffer(MemLoc.UB, (1, self.width), dtypes.uint8, depth=2)
        worker = get_block_idx() * VECTOR_SUBBLOCKS + get_subblock_id()
        worker_count = get_block_num() * VECTOR_SUBBLOCKS
        for row in range(worker, source_values.shape[0], worker_count):
            _copy(
                input_rows.next_write(),
                tile_slice(source_values, (1, self.width), (row, 0)),
            )
            values = input_rows.next_read()
            payload = output_rows.next_write()
            with vf(mode="simd"):
                mask = rr.full_mask()
                rr.vmem_bar("vst_vld")
                scale = rr.vload_broadcast(scalar, 0)
                zero_scale = rr.veqs(scale, 0.0, mask=mask)
                denominator = rr.vselect(
                    rr.vdups(1.0, dtypes.float32, mask=mask),
                    scale,
                    cond_mask=zero_scale,
                )
                for offset in range(0, self.width, VECTOR_FLOAT_ELEMENTS):
                    value = rr.vdiv(
                        self.storage.load(values, offset),
                        denominator,
                        mask=mask,
                        precision_mode=self.precision_mode,
                    )
                    value = rr.vselect(
                        rr.vdups(0.0, dtypes.float32, mask=mask),
                        value,
                        cond_mask=zero_scale,
                    )
                    encoded = _quantize_cast(value, mask, True)
                    rr.vstore_pack(
                        payload,
                        offset,
                        rr.vreinterpret(encoded, dtypes.float8_e4m3fn),
                        mask,
                        pack_mode=rr.PackMode.B32_TO_B8,
                    )
            _copy(
                tile_slice(quantized_values, (1, self.width), (row, 0)),
                output_rows.next_read(),
            )


@jit
def _bf16_sat_input(value, mask):
    zero = rr.vdups(0.0, dtypes.float32, mask=mask)
    value = rr.vselect(value, zero, cond_mask=rr.veq(value, value, mask=mask))
    return rr.vmins(
        rr.vmaxs(value, -BF16_MAX_VALUE, mask=mask), BF16_MAX_VALUE, mask=mask
    )


@jit
def _rope_row(rope_fp32, offset, rope_sin, rope_cos, rope_bf16, rope_mode, cache_sat=0):
    if dsl.const_expr(rope_mode == 0):
        with vf(mode="simd"):
            mask = rr.full_mask()
            value = rr.vload(rope_fp32, offset)
            for _ in range(cache_sat):
                value = _bf16_sat_input(value, mask)
            encoded = rr.vcast(
                value,
                dtypes.bfloat16,
                mask=mask,
                rounding=rr.RoundingMode.RN,
                reg_layout=RegLayout.ZERO,
            )
            rr.vstore_pack(
                rope_bf16, 0, encoded, mask, pack_mode=rr.PackMode.B32_TO_B16
            )
    else:
        even_step = 1
        odd_start = VECTOR_FLOAT_ELEMENTS // 2
        if dsl.const_expr(rope_mode == 1):
            even_step = 2
            odd_start = 1
        with vf(mode="simd"):
            full = rr.full_mask()
            half = rr.update_mask(VECTOR_FLOAT_ELEMENTS // 2, elem_bits=32)[0]
            values = rr.vload(rope_fp32, offset)
            positions = rr.varange(0, dtypes.uint32)
            first_idx = rr.vmuls(positions, even_step, mask=full)
            second_idx = rr.vadds(first_idx, odd_start, mask=full)
            first = rr.vgather_reg(values, first_idx)
            second = rr.vgather_reg(values, second_idx)
            sin_values = rr.vcast(
                rr.vload_unpack(rope_sin, 0, unpack_mode=rr.UnpackMode.UNPACK2),
                dtypes.float32,
                mask=full,
            )
            cos_values = rr.vcast(
                rr.vload_unpack(rope_cos, 0, unpack_mode=rr.UnpackMode.UNPACK2),
                dtypes.float32,
                mask=full,
            )
            upper_idx = rr.vadds(positions, VECTOR_FLOAT_ELEMENTS // 2, mask=full)
            sin_upper = rr.vgather_reg(sin_values, upper_idx)
            cos_upper = rr.vgather_reg(cos_values, upper_idx)
            lower = rr.vsub(
                rr.vmul(first, cos_values, mask=half),
                rr.vmul(second, sin_values, mask=half),
                mask=half,
            )
            upper = rr.vadd(
                rr.vmul(first, sin_upper, mask=half),
                rr.vmul(second, cos_upper, mask=half),
                mask=half,
            )
            for _ in range(cache_sat):
                lower = _bf16_sat_input(lower, half)
                upper = _bf16_sat_input(upper, half)
            lower_bf16 = rr.vcast(
                lower,
                dtypes.bfloat16,
                mask=half,
                rounding=rr.RoundingMode.RN,
                reg_layout=RegLayout.ZERO,
            )
            upper_bf16 = rr.vcast(
                upper,
                dtypes.bfloat16,
                mask=half,
                rounding=rr.RoundingMode.RN,
                reg_layout=RegLayout.ZERO,
            )
            rr.vstore_pack(
                rope_bf16, 0, lower_bf16, half, pack_mode=rr.PackMode.B32_TO_B16
            )
            rr.vstore_pack(
                rope_bf16,
                VECTOR_FLOAT_ELEMENTS // 2,
                upper_bf16,
                half,
                pack_mode=rr.PackMode.B32_TO_B16,
            )


class _QueryPrepare:
    def __init__(self, N, Dn, Dr, w_qr_quant_mode, rope_mode, mx_token_batch_rows=1):
        self.N = N
        self.Dn = Dn
        self.Dr = Dr
        self.w_qr_quant_mode = w_qr_quant_mode
        self.rope_mode = rope_mode
        self.mx_token_batch_rows = mx_token_batch_rows

    @jit
    def run(
        self,
        wqb_matmul_fp32: Tensor,
        descale_qr: Tensor,
        descale_wqb: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        q_nope_bf16: Tensor,
        q_rope_bf16: Tensor,
        selected_head: int = -1,
        token_begin: int = 0,
        token_end: int = -1,
        token_step: int = 1,
        selected_tile: bool = False,
        resident_head: bool = False,
        cast_nope: bool = True,
        apply_rope: bool = True,
    ):
        if dsl.const_expr(
            self.w_qr_quant_mode == 1 and self.N % 8 == 0 and not selected_tile
        ):
            self.run_mx_batch8(
                wqb_matmul_fp32, rope_sin, rope_cos, q_nope_bf16, q_rope_bf16
            )
            return
        width = self.Dn + self.Dr
        rows = wqb_matmul_fp32.shape[0]
        input_rows = _ring_buffer(MemLoc.UB, (1, width), dtypes.float32, depth=2)
        if dsl.const_expr(cast_nope):
            nope_rows = _ring_buffer(MemLoc.UB, (1, self.Dn), dtypes.bfloat16, depth=2)
        if dsl.const_expr(self.w_qr_quant_mode == 2):
            if dsl.const_expr(resident_head):
                head_scales = _buffer(MemLoc.UB, (1, width), dtypes.float32)
                _copy(
                    head_scales, tile_slice(descale_wqb, (1, width), (0, selected_head))
                )
            else:
                descale_wqb_rows = _ring_buffer(
                    MemLoc.UB, (1, width), dtypes.float32, depth=2
                )
            activation_scale = descale_qr[0]
        if dsl.const_expr(self.Dr != 0 and apply_rope):
            rope_rows = _ring_buffer(MemLoc.UB, (1, self.Dr), dtypes.bfloat16, depth=2)
            rope_sin_buffer = _buffer(MemLoc.UB, (1, self.Dr), dtypes.bfloat16)
            rope_cos_buffer = _buffer(MemLoc.UB, (1, self.Dr), dtypes.bfloat16)
            if dsl.const_expr(self.w_qr_quant_mode == 2):
                scaled_rope = _buffer(MemLoc.UB, (1, self.Dr), dtypes.float32)
        worker = get_block_idx() * VECTOR_SUBBLOCKS + get_subblock_id()
        worker_count = get_block_num() * VECTOR_SUBBLOCKS
        task_count = rows * self.N
        tasks_per_worker = (task_count + worker_count - 1) // worker_count
        task_begin = worker * tasks_per_worker
        task_end = task_begin + tasks_per_worker
        if task_end > task_count:
            task_end = task_count
        task_step = 1
        if selected_head >= 0:
            task_begin = token_begin * self.N + selected_head
            task_end = task_count
            if token_end >= 0:
                task_end = token_end * self.N + selected_head
            task_step = self.N * token_step
        for task in range(task_begin, task_end, task_step):
            row = task // self.N
            head = task % self.N
            _copy(
                input_rows.next_write(),
                tile_slice(wqb_matmul_fp32, (1, width), (row, head)),
            )
            values = input_rows.next_read()
            if dsl.const_expr(self.w_qr_quant_mode == 2):
                if dsl.const_expr(resident_head):
                    channel_scales = head_scales
                else:
                    _copy(
                        descale_wqb_rows.next_write(),
                        tile_slice(descale_wqb, (1, width), (0, head)),
                    )
                    channel_scales = descale_wqb_rows.next_read()
            if dsl.const_expr(cast_nope):
                nope_values = nope_rows.next_write()
                if dsl.const_expr(self.w_qr_quant_mode == 2):
                    with vf(mode="simd"):
                        mask = rr.full_mask()
                        for offset in range(0, self.Dn, VECTOR_FLOAT_ELEMENTS):
                            value = rr.vmul(
                                rr.vload(values, offset),
                                rr.vload(channel_scales, offset),
                                mask=mask,
                            )
                            value = rr.vmuls(value, activation_scale, mask=mask)
                            encoded = rr.vcast(
                                value,
                                dtypes.bfloat16,
                                mask=mask,
                                rounding=rr.RoundingMode.RN,
                                reg_layout=RegLayout.ZERO,
                            )
                            rr.vstore_pack(
                                nope_values,
                                offset,
                                encoded,
                                mask,
                                pack_mode=rr.PackMode.B32_TO_B16,
                            )
                else:
                    with vf(mode="simd"):
                        mask = rr.full_mask()
                        for offset in range(0, self.Dn, VECTOR_FLOAT_ELEMENTS):
                            encoded = rr.vcast(
                                rr.vload(values, offset),
                                dtypes.bfloat16,
                                mask=mask,
                                rounding=rr.RoundingMode.RN,
                                reg_layout=RegLayout.ZERO,
                            )
                            rr.vstore_pack(
                                nope_values,
                                offset,
                                encoded,
                                mask,
                                pack_mode=rr.PackMode.B32_TO_B16,
                            )
                _copy(
                    tile_slice(q_nope_bf16, (1, self.Dn), (task, 0)),
                    nope_rows.next_read(),
                )
            if dsl.const_expr(self.Dr != 0 and apply_rope):
                sin_values = rope_sin_buffer
                cos_values = rope_cos_buffer
                if dsl.const_expr(self.rope_mode != 0):
                    if dsl.const_expr(resident_head):
                        sin_values = tile_slice(rope_sin, (1, self.Dr), (row, 0))
                        cos_values = tile_slice(rope_cos, (1, self.Dr), (row, 0))
                    else:
                        _copy(
                            rope_sin_buffer,
                            tile_slice(rope_sin, (1, self.Dr), (row, 0)),
                        )
                        _copy(
                            rope_cos_buffer,
                            tile_slice(rope_cos, (1, self.Dr), (row, 0)),
                        )
                rope_values = rope_rows.next_write()
                if dsl.const_expr(self.w_qr_quant_mode == 2):
                    with vf(mode="simd"):
                        mask = rr.full_mask()
                        value = rr.vmul(
                            rr.vload(values, self.Dn),
                            rr.vload(channel_scales, self.Dn),
                            mask=mask,
                        )
                        value = rr.vmuls(value, activation_scale, mask=mask)
                        rr.vstore(scaled_rope, 0, value, mask)
                    _rope_row(
                        scaled_rope,
                        0,
                        sin_values,
                        cos_values,
                        rope_values,
                        self.rope_mode,
                    )
                else:
                    _rope_row(
                        values,
                        self.Dn,
                        sin_values,
                        cos_values,
                        rope_values,
                        self.rope_mode,
                    )
                _copy(
                    tile_slice(q_rope_bf16, (1, self.Dr), (task, 0)),
                    rope_rows.next_read(),
                )

    @jit
    def prepare_wkb_tile(
        self,
        wqb_matmul_fp32: Tensor,
        descale_qr: Tensor,
        descale_wqb: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        q_nope_bf16: Tensor,
        q_rope_bf16: Tensor,
        task: int,
        tile_rows: int,
    ):
        token_start = task // self.N * tile_rows
        token_stop = min(token_start + tile_rows, wqb_matmul_fp32.shape[0])
        token_middle = token_start + (token_stop - token_start) // VECTOR_SUBBLOCKS
        token_begin = token_start
        token_end = token_middle
        if get_subblock_id() == 1:
            token_begin = token_middle
            token_end = token_stop
        if dsl.const_expr(
            self.w_qr_quant_mode == 2 and tile_rows % HIF_PREPARE_BATCH_ROWS == 0
        ):
            self.run_hif_token_batch(
                wqb_matmul_fp32,
                descale_qr,
                descale_wqb,
                rope_sin,
                rope_cos,
                q_nope_bf16,
                q_rope_bf16,
                task % self.N,
                token_start,
                token_stop,
            )
        elif dsl.const_expr(self.mx_token_batch_rows > 1):
            self.run_mx_token_batch(
                wqb_matmul_fp32,
                rope_sin,
                rope_cos,
                q_nope_bf16,
                q_rope_bf16,
                task % self.N,
                token_start,
                token_stop,
            )
        else:
            if token_begin < token_end:
                self.run(
                    wqb_matmul_fp32,
                    descale_qr,
                    descale_wqb,
                    rope_sin,
                    rope_cos,
                    q_nope_bf16,
                    q_rope_bf16,
                    selected_head=task % self.N,
                    token_begin=token_begin,
                    token_end=token_end,
                    selected_tile=True,
                )
        # Both Vector roles publish, including an empty tail partition.
        vec_sync_block_arrive(_sync_ops.PIPE.MTE3, PREPARE_READY_EVENT, mode=2)

    @jit
    def run_hif_resident_pipeline(
        self,
        wqb_matmul_fp32: Tensor,
        descale_qr: Tensor,
        descale_wqb: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        q_nope_bf16: Tensor,
        q: Tensor,
        tile_rows: int,
    ):
        rows = wqb_matmul_fp32.shape[0]
        row_tiles = (rows + tile_rows - 1) // tile_rows
        vector_rows = tile_rows // VECTOR_SUBBLOCKS
        width = self.Dn + self.Dr
        values = _buffer(MemLoc.UB, (vector_rows, width), dtypes.float32)
        head_scales = _buffer(MemLoc.UB, (1, width), dtypes.float32)
        nope_values = _buffer(MemLoc.UB, (vector_rows, self.Dn), dtypes.bfloat16)
        nope_tokens = q_nope_bf16.view((rows, self.N * self.Dn))
        activation_scale = descale_qr[0]
        if dsl.const_expr(self.Dr != 0):
            sin_buffer = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
            cos_buffer = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
            rope_values = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
            scaled_rope = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.float32)
            output_heads = dsl.permute(q.view((rows, self.N, q.shape[1])), (1, 0, 2))
        for task in range(get_block_idx(), row_tiles * self.N, get_block_num()):
            head = task % self.N
            batch = task // self.N * VECTOR_SUBBLOCKS + get_subblock_id()
            valid_rows = min(vector_rows, max(0, rows - batch * vector_rows))
            vec_sync_block_wait(_sync_ops.PIPE.S, HEAD_READY_EVENT, mode=2)
            if valid_rows > 0:
                _copy(
                    values,
                    tile_slice(wqb_matmul_fp32, (vector_rows, width), (batch, head)),
                )
                _copy(head_scales, tile_slice(descale_wqb, (1, width), (0, head)))
                with vf(mode="simd"):
                    mask = rr.full_mask()
                    for row in range(valid_rows):
                        for offset in range(0, self.Dn, VECTOR_FLOAT_ELEMENTS):
                            value = rr.vmul(
                                rr.vload(values, row * width + offset),
                                rr.vload(head_scales, offset),
                                mask=mask,
                            )
                            value = rr.vmuls(value, activation_scale, mask=mask)
                            encoded = rr.vcast(
                                value,
                                dtypes.bfloat16,
                                mask=mask,
                                rounding=rr.RoundingMode.RN,
                                reg_layout=RegLayout.ZERO,
                            )
                            rr.vstore_pack(
                                nope_values,
                                row * self.Dn + offset,
                                encoded,
                                mask,
                                pack_mode=rr.PackMode.B32_TO_B16,
                            )
                _copy(
                    tile_slice(nope_tokens, (vector_rows, self.Dn), (batch, head)),
                    nope_values,
                )
            # Cube may consume NoPE while RoPE still reads the resident UB tile.
            vec_sync_block_arrive(_sync_ops.PIPE.MTE3, HEAD_CONSUMED_EVENT, mode=2)
            if dsl.const_expr(self.Dr != 0):
                if valid_rows > 0:
                    if dsl.const_expr(self.rope_mode != 0):
                        _copy(
                            sin_buffer,
                            tile_slice(rope_sin, (vector_rows, self.Dr), (batch, 0)),
                        )
                        _copy(
                            cos_buffer,
                            tile_slice(rope_cos, (vector_rows, self.Dr), (batch, 0)),
                        )
                    with vf(mode="simd"):
                        rope_mask = rr.full_mask()
                        for row in range(valid_rows):
                            value = rr.vmul(
                                rr.vload(values, row * width + self.Dn),
                                rr.vload(head_scales, self.Dn),
                                mask=rope_mask,
                            )
                            value = rr.vmuls(value, activation_scale, mask=rope_mask)
                            rr.vstore(scaled_rope, row * self.Dr, value, rope_mask)
                    for row in range(valid_rows):
                        _rope_row(
                            scaled_rope,
                            row * self.Dr,
                            tile_slice(sin_buffer, (1, self.Dr), (row, 0)),
                            tile_slice(cos_buffer, (1, self.Dr), (row, 0)),
                            tile_slice(rope_values, (1, self.Dr), (row, 0)),
                            self.rope_mode,
                        )
                    head_rope = output_heads[
                        head, None, KV_ELEMENTS : KV_ELEMENTS + self.Dr
                    ]
                    _copy(
                        tile_slice(head_rope, (vector_rows, self.Dr), (batch, 0)),
                        rope_values,
                    )

    @jit
    def run_hif_token_batch(
        self,
        wqb_matmul_fp32: Tensor,
        descale_qr: Tensor,
        descale_wqb: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        q_nope_bf16: Tensor,
        q_rope_bf16: Tensor,
        head: int,
        token_begin: int,
        token_end: int,
    ):
        batch_rows = HIF_PREPARE_BATCH_ROWS
        width = self.Dn + self.Dr
        rows = wqb_matmul_fp32.shape[0]
        first_batch = token_begin // batch_rows + get_subblock_id()
        stop_batch = (token_end + batch_rows - 1) // batch_rows
        nope_tokens = q_nope_bf16.view((rows, self.N * self.Dn))
        inputs = _ring_buffer(MemLoc.UB, (batch_rows, width), dtypes.float32, depth=2)
        nope_outputs = _ring_buffer(
            MemLoc.UB, (batch_rows, self.Dn), dtypes.bfloat16, depth=2
        )
        head_scales = _buffer(MemLoc.UB, (1, width), dtypes.float32)
        if dsl.const_expr(self.Dr != 0):
            rope_tokens = q_rope_bf16.view((rows, self.N * self.Dr))
            rope_outputs = _ring_buffer(
                MemLoc.UB, (batch_rows, self.Dr), dtypes.bfloat16, depth=2
            )
            sin_buffer = _buffer(MemLoc.UB, (batch_rows, self.Dr), dtypes.bfloat16)
            cos_buffer = _buffer(MemLoc.UB, (batch_rows, self.Dr), dtypes.bfloat16)
            scaled_rope = _buffer(MemLoc.UB, (batch_rows, self.Dr), dtypes.float32)
        if first_batch < stop_batch:
            _copy(head_scales, tile_slice(descale_wqb, (1, width), (0, head)))
            activation_scale = descale_qr[0]
            for batch in range(first_batch, stop_batch, VECTOR_SUBBLOCKS):
                valid_rows = min(batch_rows, token_end - batch * batch_rows)
                _copy(
                    inputs.next_write(),
                    tile_slice(wqb_matmul_fp32, (batch_rows, width), (batch, head)),
                )
                values = inputs.next_read()
                nope_values = nope_outputs.next_write()
                with vf(mode="simd"):
                    mask = rr.full_mask()
                    for row in range(valid_rows):
                        for offset in range(0, self.Dn, VECTOR_FLOAT_ELEMENTS):
                            value = rr.vmul(
                                rr.vload(values, row * width + offset),
                                rr.vload(head_scales, offset),
                                mask=mask,
                            )
                            value = rr.vmuls(value, activation_scale, mask=mask)
                            encoded = rr.vcast(
                                value,
                                dtypes.bfloat16,
                                mask=mask,
                                rounding=rr.RoundingMode.RN,
                                reg_layout=RegLayout.ZERO,
                            )
                            rr.vstore_pack(
                                nope_values,
                                row * self.Dn + offset,
                                encoded,
                                mask,
                                pack_mode=rr.PackMode.B32_TO_B16,
                            )
                _copy(
                    tile_slice(nope_tokens, (batch_rows, self.Dn), (batch, head)),
                    nope_outputs.next_read(),
                )
                if dsl.const_expr(self.Dr != 0):
                    if dsl.const_expr(self.rope_mode != 0):
                        _copy(
                            sin_buffer,
                            tile_slice(rope_sin, (batch_rows, self.Dr), (batch, 0)),
                        )
                        _copy(
                            cos_buffer,
                            tile_slice(rope_cos, (batch_rows, self.Dr), (batch, 0)),
                        )
                    with vf(mode="simd"):
                        rope_mask = rr.full_mask()
                        for row in range(valid_rows):
                            value = rr.vmul(
                                rr.vload(values, row * width + self.Dn),
                                rr.vload(head_scales, self.Dn),
                                mask=rope_mask,
                            )
                            value = rr.vmuls(value, activation_scale, mask=rope_mask)
                            rr.vstore(scaled_rope, row * self.Dr, value, rope_mask)
                    rope_values = rope_outputs.next_write()
                    for row in range(valid_rows):
                        _rope_row(
                            scaled_rope,
                            row * self.Dr,
                            tile_slice(sin_buffer, (1, self.Dr), (row, 0)),
                            tile_slice(cos_buffer, (1, self.Dr), (row, 0)),
                            tile_slice(rope_values, (1, self.Dr), (row, 0)),
                            self.rope_mode,
                        )
                    _copy(
                        tile_slice(rope_tokens, (batch_rows, self.Dr), (batch, head)),
                        rope_outputs.next_read(),
                    )

    @jit
    def run_mx_token_batch(
        self,
        wqb_matmul_fp32: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        q_nope_bf16: Tensor,
        q_rope_bf16: Tensor,
        head: int,
        token_begin: int,
        token_end: int,
    ):
        batch_rows = self.mx_token_batch_rows
        width = self.Dn + self.Dr
        rows = wqb_matmul_fp32.shape[0]
        nope_tokens = q_nope_bf16.view((rows, self.N * self.Dn))
        inputs = _ring_buffer(MemLoc.UB, (batch_rows, width), dtypes.float32, depth=2)
        nope_outputs = _ring_buffer(
            MemLoc.UB, (batch_rows, self.Dn), dtypes.bfloat16, depth=2
        )
        if dsl.const_expr(self.Dr != 0):
            rope_tokens = q_rope_bf16.view((rows, self.N * self.Dr))
            rope_outputs = _ring_buffer(
                MemLoc.UB, (batch_rows, self.Dr), dtypes.bfloat16, depth=2
            )
            sin_buffer = _buffer(MemLoc.UB, (batch_rows, self.Dr), dtypes.bfloat16)
            cos_buffer = _buffer(MemLoc.UB, (batch_rows, self.Dr), dtypes.bfloat16)
        for batch in range(
            token_begin // batch_rows + get_subblock_id(),
            (token_end + batch_rows - 1) // batch_rows,
            VECTOR_SUBBLOCKS,
        ):
            valid_rows = min(batch_rows, token_end - batch * batch_rows)
            _copy(
                inputs.next_write(),
                tile_slice(wqb_matmul_fp32, (batch_rows, width), (batch, head)),
            )
            values = inputs.next_read()
            nope_values = nope_outputs.next_write()
            with vf(mode="simd"):
                mask = rr.full_mask()
                for row in range(valid_rows):
                    for offset in range(0, self.Dn, VECTOR_FLOAT_ELEMENTS):
                        encoded = rr.vcast(
                            rr.vload(values, row * width + offset),
                            dtypes.bfloat16,
                            mask=mask,
                            rounding=rr.RoundingMode.RN,
                            reg_layout=RegLayout.ZERO,
                        )
                        rr.vstore_pack(
                            nope_values,
                            row * self.Dn + offset,
                            encoded,
                            mask,
                            pack_mode=rr.PackMode.B32_TO_B16,
                        )
            _copy(
                tile_slice(nope_tokens, (batch_rows, self.Dn), (batch, head)),
                nope_outputs.next_read(),
            )
            if dsl.const_expr(self.Dr != 0):
                if dsl.const_expr(self.rope_mode != 0):
                    _copy(
                        sin_buffer,
                        tile_slice(rope_sin, (batch_rows, self.Dr), (batch, 0)),
                    )
                    _copy(
                        cos_buffer,
                        tile_slice(rope_cos, (batch_rows, self.Dr), (batch, 0)),
                    )
                rope_values = rope_outputs.next_write()
                for row in range(valid_rows):
                    _rope_row(
                        values,
                        row * width + self.Dn,
                        tile_slice(sin_buffer, (1, self.Dr), (row, 0)),
                        tile_slice(cos_buffer, (1, self.Dr), (row, 0)),
                        tile_slice(rope_values, (1, self.Dr), (row, 0)),
                        self.rope_mode,
                    )
                _copy(
                    tile_slice(rope_tokens, (batch_rows, self.Dr), (batch, head)),
                    rope_outputs.next_read(),
                )

    @jit
    def run_pipeline(
        self,
        wqb_matmul_fp32: Tensor,
        descale_qr: Tensor,
        descale_wqb: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        q_nope_bf16: Tensor,
        q: Tensor,
        tile_rows: int,
    ):
        if dsl.const_expr(self.w_qr_quant_mode == 1):
            self.run_mx_resident_pipeline(
                wqb_matmul_fp32, rope_sin, rope_cos, q_nope_bf16, q, tile_rows
            )
            return
        if dsl.const_expr(self.w_qr_quant_mode == 2):
            self.run_hif_resident_pipeline(
                wqb_matmul_fp32,
                descale_qr,
                descale_wqb,
                rope_sin,
                rope_cos,
                q_nope_bf16,
                q,
                tile_rows,
            )
            return
        rows = wqb_matmul_fp32.shape[0]
        row_tiles = (rows + tile_rows - 1) // tile_rows
        q_rope = q_nope_bf16
        if dsl.const_expr(self.Dr != 0):
            q_rope = q[None, KV_ELEMENTS : KV_ELEMENTS + self.Dr]
        for task in range(get_block_idx(), row_tiles * self.N, get_block_num()):
            token_start = (task // self.N) * tile_rows
            token_stop = min(token_start + tile_rows, rows)
            token_middle = token_start + (token_stop - token_start) // VECTOR_SUBBLOCKS
            token_begin = token_start
            token_end = token_middle
            if get_subblock_id() == 1:
                token_begin = token_middle
                token_end = token_stop
            vec_sync_block_wait(_sync_ops.PIPE.S, HEAD_READY_EVENT, mode=2)
            if token_begin < token_end:
                self.run(
                    wqb_matmul_fp32,
                    descale_qr,
                    descale_wqb,
                    rope_sin,
                    rope_cos,
                    q_nope_bf16,
                    q_rope,
                    selected_head=task % self.N,
                    token_begin=token_begin,
                    token_end=token_end,
                    selected_tile=True,
                    apply_rope=False,
                )
            # Empty Vector partitions must also release the paired Cube wait.
            vec_sync_block_arrive(_sync_ops.PIPE.MTE3, HEAD_CONSUMED_EVENT, mode=2)
            if dsl.const_expr(self.Dr != 0):
                if token_begin < token_end:
                    self.run(
                        wqb_matmul_fp32,
                        descale_qr,
                        descale_wqb,
                        rope_sin,
                        rope_cos,
                        q_nope_bf16,
                        q_rope,
                        selected_head=task % self.N,
                        token_begin=token_begin,
                        token_end=token_end,
                        selected_tile=True,
                        cast_nope=False,
                    )

    @jit
    def run_mx_handoff(
        self,
        projection: Tensor,
        c_to_v,
        rope_sin: Tensor,
        rope_cos: Tensor,
        q_nope: Tensor,
        q_rope: Tensor,
        tile_rows: int,
        head_group: int,
        direct_ub: bool,
        nope_l1=None,
        packed_output: bool = False,
    ):
        rows = projection.shape[0]
        width = self.Dn + self.Dr
        columns = head_group * width
        vector_rows = tile_rows // VECTOR_SUBBLOCKS
        groups = self.N // head_group
        row_tiles = (rows + tile_rows - 1) // tile_rows
        gm_inputs = None
        if dsl.const_expr(not direct_ub):
            gm_inputs = _ring_buffer(
                MemLoc.UB, (vector_rows, columns), dtypes.float32, depth=2
            )
        for task in range(get_block_idx(), row_tiles * groups, get_block_num()):
            group = task % groups
            batch = task // groups * VECTOR_SUBBLOCKS + get_subblock_id()
            valid_rows = min(vector_rows, max(0, rows - batch * vector_rows))
            if dsl.const_expr(direct_ub):
                values = c_to_v.next_read()
                if dsl.const_expr(nope_l1 is None):
                    self.convert_mx_handoff(
                        values,
                        rope_sin,
                        rope_cos,
                        q_nope,
                        q_rope,
                        batch,
                        group,
                        head_group,
                        vector_rows,
                        valid_rows,
                    )
                else:
                    self.convert_mx_group_l1(
                        values,
                        rope_sin,
                        rope_cos,
                        q_rope,
                        batch,
                        group,
                        head_group,
                        vector_rows,
                        valid_rows,
                        nope_l1.next_write(),
                        packed_output,
                    )
            else:
                vec_sync_block_wait(_sync_ops.PIPE.S, HEAD_READY_EVENT, mode=2)
                values = gm_inputs.next_write()
                if valid_rows > 0:
                    _copy(
                        values,
                        tile_slice(projection, (vector_rows, columns), (batch, group)),
                    )
                values = gm_inputs.next_read()
                if valid_rows > 0:
                    self.convert_mx_handoff(
                        values,
                        rope_sin,
                        rope_cos,
                        q_nope,
                        q_rope,
                        batch,
                        group,
                        head_group,
                        vector_rows,
                        valid_rows,
                    )
            # Acknowledge only after both QNope and QRoPE consumers release the slot.
            vec_sync_block_arrive(_sync_ops.PIPE.MTE3, HEAD_CONSUMED_EVENT, mode=2)

    @jit
    def convert_mx_handoff(
        self,
        values: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        q_nope: Tensor,
        q_rope: Tensor,
        batch: int,
        group: int,
        head_group: int,
        vector_rows: int,
        valid_rows: int,
    ):
        if dsl.const_expr(
            self.Dn == 192 and self.Dr == 64 and head_group == 2 and self.rope_mode == 1
        ):
            self.convert_mx_group(
                values,
                rope_sin,
                rope_cos,
                q_nope,
                q_rope,
                batch,
                group,
                head_group,
                vector_rows,
                valid_rows,
            )
            return
        rows = q_nope.shape[0] // self.N
        width = self.Dn + self.Dr
        columns = head_group * width
        nope_values = _buffer(MemLoc.UB, (vector_rows, self.Dn), dtypes.bfloat16)
        nope_tokens = q_nope.view((rows, self.N * self.Dn))
        rope_tokens = q_rope.view((rows, self.N * self.Dr))
        rope_values = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
        sin_values = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
        cos_values = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
        if valid_rows > 0:
            if dsl.const_expr(self.rope_mode != 0):
                _copy(
                    sin_values, tile_slice(rope_sin, (vector_rows, self.Dr), (batch, 0))
                )
                _copy(
                    cos_values, tile_slice(rope_cos, (vector_rows, self.Dr), (batch, 0))
                )
        for local_head in range(head_group):
            head = group * head_group + local_head
            # Empty Vector partitions still acquire and release the CrossCore slot.
            with vf(mode="simd"):
                mask = rr.full_mask()
                for row in range(valid_rows):
                    for offset in range(0, self.Dn, VECTOR_FLOAT_ELEMENTS):
                        value = rr.vload(
                            values, row * columns + local_head * width + offset
                        )
                        encoded = rr.vcast(
                            value,
                            dtypes.bfloat16,
                            mask=mask,
                            rounding=rr.RoundingMode.RN,
                            reg_layout=RegLayout.ZERO,
                        )
                        rr.vstore_pack(
                            nope_values,
                            row * self.Dn + offset,
                            encoded,
                            mask,
                            pack_mode=rr.PackMode.B32_TO_B16,
                        )
            if valid_rows > 0:
                _copy(
                    tile_slice(nope_tokens, (vector_rows, self.Dn), (batch, head)),
                    nope_values,
                )
            for row in range(valid_rows):
                _rope_row(
                    values,
                    row * columns + local_head * width + self.Dn,
                    tile_slice(sin_values, (1, self.Dr), (row, 0)),
                    tile_slice(cos_values, (1, self.Dr), (row, 0)),
                    tile_slice(rope_values, (1, self.Dr), (row, 0)),
                    self.rope_mode,
                )
            if valid_rows > 0:
                _copy(
                    tile_slice(rope_tokens, (vector_rows, self.Dr), (batch, head)),
                    rope_values,
                )

    @jit
    def convert_mx_group(
        self,
        values: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        q_nope: Tensor,
        q_rope: Tensor,
        batch: int,
        group: int,
        head_group: int,
        vector_rows: int,
        valid_rows: int,
    ):
        rows = q_nope.shape[0] // self.N
        width = self.Dn + self.Dr
        columns = head_group * width
        nope_columns = head_group * self.Dn
        rope_columns = head_group * self.Dr
        nope_values = _buffer(MemLoc.UB, (vector_rows, nope_columns), dtypes.bfloat16)
        rope_values = _buffer(MemLoc.UB, (vector_rows, rope_columns), dtypes.bfloat16)
        sin_values = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
        cos_values = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
        nope_tokens = q_nope.view((rows, self.N * self.Dn))
        rope_tokens = q_rope.view((rows, self.N * self.Dr))
        if valid_rows > 0:
            _copy(sin_values, tile_slice(rope_sin, (vector_rows, self.Dr), (batch, 0)))
            _copy(cos_values, tile_slice(rope_cos, (vector_rows, self.Dr), (batch, 0)))
        with vf(mode="simd"):
            full = rr.full_mask()
            half = rr.update_mask(VECTOR_FLOAT_ELEMENTS // 2, elem_bits=32)[0]
            positions = rr.varange(0, dtypes.uint32)
            first_idx = rr.vmuls(positions, 2, mask=full)
            second_idx = rr.vadds(first_idx, 1, mask=full)
            upper_idx = rr.vadds(positions, VECTOR_FLOAT_ELEMENTS // 2, mask=full)
            for row in range(valid_rows):
                sin_row = rr.vcast(
                    rr.vload_unpack(
                        sin_values, row * self.Dr, unpack_mode=rr.UnpackMode.UNPACK2
                    ),
                    dtypes.float32,
                    mask=full,
                )
                cos_row = rr.vcast(
                    rr.vload_unpack(
                        cos_values, row * self.Dr, unpack_mode=rr.UnpackMode.UNPACK2
                    ),
                    dtypes.float32,
                    mask=full,
                )
                sin_upper = rr.vgather_reg(sin_row, upper_idx)
                cos_upper = rr.vgather_reg(cos_row, upper_idx)
                for local_head in range(head_group):
                    for offset in range(0, self.Dn, VECTOR_FLOAT_ELEMENTS):
                        value = rr.vload(
                            values, row * columns + local_head * width + offset
                        )
                        encoded = rr.vcast(
                            value,
                            dtypes.bfloat16,
                            mask=full,
                            rounding=rr.RoundingMode.RN,
                            reg_layout=RegLayout.ZERO,
                        )
                        rr.vstore_pack(
                            nope_values,
                            row * nope_columns + local_head * self.Dn + offset,
                            encoded,
                            full,
                            pack_mode=rr.PackMode.B32_TO_B16,
                        )
                    rope_row = rr.vload(
                        values, row * columns + local_head * width + self.Dn
                    )
                    first = rr.vgather_reg(rope_row, first_idx)
                    second = rr.vgather_reg(rope_row, second_idx)
                    lower = rr.vsub(
                        rr.vmul(first, cos_row, mask=half),
                        rr.vmul(second, sin_row, mask=half),
                        mask=half,
                    )
                    upper = rr.vadd(
                        rr.vmul(first, sin_upper, mask=half),
                        rr.vmul(second, cos_upper, mask=half),
                        mask=half,
                    )
                    lower_bf16 = rr.vcast(
                        lower,
                        dtypes.bfloat16,
                        mask=half,
                        rounding=rr.RoundingMode.RN,
                        reg_layout=RegLayout.ZERO,
                    )
                    upper_bf16 = rr.vcast(
                        upper,
                        dtypes.bfloat16,
                        mask=half,
                        rounding=rr.RoundingMode.RN,
                        reg_layout=RegLayout.ZERO,
                    )
                    rr.vstore_pack(
                        rope_values,
                        row * rope_columns + local_head * self.Dr,
                        lower_bf16,
                        half,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
                    rr.vstore_pack(
                        rope_values,
                        row * rope_columns
                        + local_head * self.Dr
                        + VECTOR_FLOAT_ELEMENTS // 2,
                        upper_bf16,
                        half,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
        if valid_rows > 0:
            _copy(
                tile_slice(nope_tokens, (vector_rows, nope_columns), (batch, group)),
                nope_values,
            )
            _copy(
                tile_slice(rope_tokens, (vector_rows, rope_columns), (batch, group)),
                rope_values,
            )

    @jit
    def convert_mx_group_l1(
        self,
        values: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        q_rope: Tensor,
        batch: int,
        group: int,
        head_group: int,
        vector_rows: int,
        valid_rows: int,
        destination: Tensor,
        packed_output: bool = False,
    ):
        width = self.Dn + self.Dr
        columns = head_group * width
        nope_columns = head_group * self.Dn
        rope_columns = head_group * self.Dr
        fractal_lanes = FRACTAL_BYTES // BF16_BYTES
        nope_values = _buffer(
            MemLoc.UB,
            (nope_columns // fractal_lanes, vector_rows * fractal_lanes),
            dtypes.bfloat16,
        )
        if dsl.const_expr(self.Dr != 0):
            rope_values = _buffer(
                MemLoc.UB, (vector_rows, rope_columns), dtypes.bfloat16
            )
            sin_values = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
            cos_values = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
            rows = q_rope.shape[0] // self.N
            if dsl.const_expr(packed_output):
                output_heads = dsl.permute(
                    q_rope.view((rows, self.N, q_rope.shape[1])), (1, 0, 2)
                )
            else:
                rope_tokens = q_rope.view((rows, self.N * self.Dr))
            if valid_rows > 0:
                _copy(
                    sin_values, tile_slice(rope_sin, (vector_rows, self.Dr), (batch, 0))
                )
                _copy(
                    cos_values, tile_slice(rope_cos, (vector_rows, self.Dr), (batch, 0))
                )
        with vf(mode="simd"):
            full = rr.full_mask()
            half = rr.update_mask(VECTOR_FLOAT_ELEMENTS // 2, elem_bits=32)[0]
            packed_mask = rr.update_mask(VECTOR_FLOAT_ELEMENTS, elem_bits=16)[0]
            zero = rr.vdups(0, dtypes.uint16, mask=packed_mask)
            for row in range(valid_rows, vector_rows):
                for offset in range(0, nope_columns, VECTOR_FLOAT_ELEMENTS):
                    rr.vstore_strided(
                        nope_values,
                        offset // fractal_lanes * vector_rows * fractal_lanes
                        + row * fractal_lanes,
                        rr.vreinterpret(zero, dtypes.bfloat16),
                        packed_mask,
                        block_stride=vector_rows,
                    )
            positions = rr.varange(0, dtypes.uint32)
            first_idx = rr.vmuls(positions, 2, mask=full)
            second_idx = rr.vadds(first_idx, 1, mask=full)
            upper_idx = rr.vadds(positions, VECTOR_FLOAT_ELEMENTS // 2, mask=full)
            for row in range(valid_rows):
                if dsl.const_expr(self.Dr != 0):
                    sin_row = rr.vcast(
                        rr.vload_unpack(
                            sin_values, row * self.Dr, unpack_mode=rr.UnpackMode.UNPACK2
                        ),
                        dtypes.float32,
                        mask=full,
                    )
                    cos_row = rr.vcast(
                        rr.vload_unpack(
                            cos_values, row * self.Dr, unpack_mode=rr.UnpackMode.UNPACK2
                        ),
                        dtypes.float32,
                        mask=full,
                    )
                    sin_upper = rr.vgather_reg(sin_row, upper_idx)
                    cos_upper = rr.vgather_reg(cos_row, upper_idx)
                for local_head in range(head_group):
                    for offset in range(0, self.Dn, VECTOR_FLOAT_ELEMENTS):
                        value = rr.vload(
                            values, row * columns + local_head * width + offset
                        )
                        encoded = rr.vcast(
                            value,
                            dtypes.bfloat16,
                            mask=full,
                            rounding=rr.RoundingMode.RN,
                            reg_layout=RegLayout.ZERO,
                        )
                        packed = rr.vpack(
                            rr.vreinterpret_lanes(encoded, dtypes.uint32),
                            dtypes.uint16,
                            part="lower",
                        )
                        rr.vstore_strided(
                            nope_values,
                            (local_head * self.Dn + offset)
                            // fractal_lanes
                            * vector_rows
                            * fractal_lanes
                            + row * fractal_lanes,
                            rr.vreinterpret(packed, dtypes.bfloat16),
                            packed_mask,
                            block_stride=vector_rows,
                        )
                    if dsl.const_expr(self.Dr != 0):
                        rope_row = rr.vload(
                            values, row * columns + local_head * width + self.Dn
                        )
                        first = rr.vgather_reg(rope_row, first_idx)
                        second = rr.vgather_reg(rope_row, second_idx)
                        lower = rr.vsub(
                            rr.vmul(first, cos_row, mask=half),
                            rr.vmul(second, sin_row, mask=half),
                            mask=half,
                        )
                        upper = rr.vadd(
                            rr.vmul(first, sin_upper, mask=half),
                            rr.vmul(second, cos_upper, mask=half),
                            mask=half,
                        )
                        lower_bf16 = rr.vcast(
                            lower,
                            dtypes.bfloat16,
                            mask=half,
                            rounding=rr.RoundingMode.RN,
                            reg_layout=RegLayout.ZERO,
                        )
                        upper_bf16 = rr.vcast(
                            upper,
                            dtypes.bfloat16,
                            mask=half,
                            rounding=rr.RoundingMode.RN,
                            reg_layout=RegLayout.ZERO,
                        )
                        rr.vstore_pack(
                            rope_values,
                            row * rope_columns + local_head * self.Dr,
                            lower_bf16,
                            half,
                            pack_mode=rr.PackMode.B32_TO_B16,
                        )
                        rr.vstore_pack(
                            rope_values,
                            row * rope_columns
                            + local_head * self.Dr
                            + VECTOR_FLOAT_ELEMENTS // 2,
                            upper_bf16,
                            half,
                            pack_mode=rr.PackMode.B32_TO_B16,
                        )
        _copy(
            tile_slice(
                destination,
                (nope_columns // fractal_lanes, vector_rows * fractal_lanes),
                (0, get_subblock_id()),
            ),
            nope_values,
        )
        if dsl.const_expr(self.Dr != 0):
            if valid_rows > 0:
                if dsl.const_expr(packed_output):
                    for local_head in range(head_group):
                        head_rope = output_heads[
                            group * head_group + local_head,
                            None,
                            KV_ELEMENTS : KV_ELEMENTS + self.Dr,
                        ]
                        # Bound the UB source too: the strided GM view does not
                        # truncate the 0.8 copy to the final token tail.
                        _copy(
                            tile_slice(head_rope, (vector_rows, self.Dr), (batch, 0)),
                            tile_slice(
                                rope_values, (vector_rows, self.Dr), (0, local_head)
                            )[0:valid_rows, None],
                        )
                else:
                    _copy(
                        tile_slice(
                            rope_tokens, (vector_rows, rope_columns), (batch, group)
                        ),
                        rope_values,
                    )

    @jit
    def run_hif_c2v_pipeline(
        self,
        c_to_v,
        descale_qr: Tensor,
        descale_wqb: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        q_nope_bf16: Tensor,
        q: Tensor,
        tile_rows: int,
    ):
        rows = q_nope_bf16.shape[0] // self.N
        row_tiles = (rows + tile_rows - 1) // tile_rows
        vector_rows = tile_rows // VECTOR_SUBBLOCKS
        width = self.Dn + self.Dr
        head_scales = _buffer(MemLoc.UB, (1, width), dtypes.float32)
        nope_values = _buffer(MemLoc.UB, (vector_rows, self.Dn), dtypes.bfloat16)
        nope_tokens = q_nope_bf16.view((rows, self.N * self.Dn))
        activation_scale = descale_qr[0]
        if dsl.const_expr(self.Dr != 0):
            sin_buffer = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
            cos_buffer = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
            rope_values = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
            scaled_rope = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.float32)
            output_heads = dsl.permute(q.view((rows, self.N, q.shape[1])), (1, 0, 2))
        for task in range(get_block_idx(), row_tiles * self.N, get_block_num()):
            head = task % self.N
            batch = task // self.N * VECTOR_SUBBLOCKS + get_subblock_id()
            valid_rows = min(vector_rows, max(0, rows - batch * vector_rows))
            values = c_to_v.next_read()
            _copy(head_scales, tile_slice(descale_wqb, (1, width), (0, head)))
            with vf(mode="simd"):
                mask = rr.full_mask()
                for row in range(vector_rows):
                    for offset in range(0, self.Dn, VECTOR_FLOAT_ELEMENTS):
                        value = rr.vmul(
                            rr.vload(values, row * width + offset),
                            rr.vload(head_scales, offset),
                            mask=mask,
                        )
                        value = rr.vmuls(value, activation_scale, mask=mask)
                        encoded = rr.vcast(
                            value,
                            dtypes.bfloat16,
                            mask=mask,
                            rounding=rr.RoundingMode.RN,
                            reg_layout=RegLayout.ZERO,
                        )
                        rr.vstore_pack(
                            nope_values,
                            row * self.Dn + offset,
                            encoded,
                            mask,
                            pack_mode=rr.PackMode.B32_TO_B16,
                        )
            if valid_rows > 0:
                _copy(
                    tile_slice(nope_tokens, (vector_rows, self.Dn), (batch, head)),
                    nope_values,
                )
            vec_sync_block_arrive(_sync_ops.PIPE.MTE3, HEAD_CONSUMED_EVENT, mode=2)
            if dsl.const_expr(self.Dr != 0):
                if valid_rows > 0:
                    if dsl.const_expr(self.rope_mode != 0):
                        _copy(
                            sin_buffer,
                            tile_slice(rope_sin, (vector_rows, self.Dr), (batch, 0)),
                        )
                        _copy(
                            cos_buffer,
                            tile_slice(rope_cos, (vector_rows, self.Dr), (batch, 0)),
                        )
                    with vf(mode="simd"):
                        rope_mask = rr.full_mask()
                        for row in range(valid_rows):
                            value = rr.vmul(
                                rr.vload(values, row * width + self.Dn),
                                rr.vload(head_scales, self.Dn),
                                mask=rope_mask,
                            )
                            value = rr.vmuls(value, activation_scale, mask=rope_mask)
                            rr.vstore(scaled_rope, row * self.Dr, value, rope_mask)
                    for row in range(valid_rows):
                        _rope_row(
                            scaled_rope,
                            row * self.Dr,
                            tile_slice(sin_buffer, (1, self.Dr), (row, 0)),
                            tile_slice(cos_buffer, (1, self.Dr), (row, 0)),
                            tile_slice(rope_values, (1, self.Dr), (row, 0)),
                            self.rope_mode,
                        )
                    head_rope = output_heads[
                        head, None, KV_ELEMENTS : KV_ELEMENTS + self.Dr
                    ]
                    _copy(
                        tile_slice(head_rope, (vector_rows, self.Dr), (batch, 0)),
                        rope_values,
                    )

    @jit
    def run_mx_resident_pipeline(
        self,
        wqb_matmul_fp32: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        q_nope_bf16: Tensor,
        q: Tensor,
        tile_rows: int,
    ):
        rows = wqb_matmul_fp32.shape[0]
        row_tiles = (rows + tile_rows - 1) // tile_rows
        vector_rows = tile_rows // VECTOR_SUBBLOCKS
        width = self.Dn + self.Dr
        values = _buffer(MemLoc.UB, (vector_rows, width), dtypes.float32)
        nope_values = _buffer(MemLoc.UB, (vector_rows, self.Dn), dtypes.bfloat16)
        nope_tokens = q_nope_bf16.view((rows, self.N * self.Dn))
        if dsl.const_expr(self.Dr != 0):
            sin_buffer = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
            cos_buffer = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
            rope_values = _buffer(MemLoc.UB, (vector_rows, self.Dr), dtypes.bfloat16)
            output_heads = dsl.permute(q.view((rows, self.N, q.shape[1])), (1, 0, 2))
        for task in range(get_block_idx(), row_tiles * self.N, get_block_num()):
            head = task % self.N
            batch = task // self.N * VECTOR_SUBBLOCKS + get_subblock_id()
            valid_rows = min(vector_rows, max(0, rows - batch * vector_rows))
            vec_sync_block_wait(_sync_ops.PIPE.S, HEAD_READY_EVENT, mode=2)
            if valid_rows > 0:
                _copy(
                    values,
                    tile_slice(wqb_matmul_fp32, (vector_rows, width), (batch, head)),
                )
                with vf(mode="simd"):
                    mask = rr.full_mask()
                    for row in range(valid_rows):
                        for offset in range(0, self.Dn, VECTOR_FLOAT_ELEMENTS):
                            encoded = rr.vcast(
                                rr.vload(values, row * width + offset),
                                dtypes.bfloat16,
                                mask=mask,
                                rounding=rr.RoundingMode.RN,
                                reg_layout=RegLayout.ZERO,
                            )
                            rr.vstore_pack(
                                nope_values,
                                row * self.Dn + offset,
                                encoded,
                                mask,
                                pack_mode=rr.PackMode.B32_TO_B16,
                            )
                _copy(
                    tile_slice(nope_tokens, (vector_rows, self.Dn), (batch, head)),
                    nope_values,
                )
            # Release Cube after NoPE; retain values until RoPE has consumed them.
            vec_sync_block_arrive(_sync_ops.PIPE.MTE3, HEAD_CONSUMED_EVENT, mode=2)
            if dsl.const_expr(self.Dr != 0):
                if valid_rows > 0:
                    if dsl.const_expr(self.rope_mode != 0):
                        _copy(
                            sin_buffer,
                            tile_slice(rope_sin, (vector_rows, self.Dr), (batch, 0)),
                        )
                        _copy(
                            cos_buffer,
                            tile_slice(rope_cos, (vector_rows, self.Dr), (batch, 0)),
                        )
                    for row in range(valid_rows):
                        _rope_row(
                            values,
                            row * width + self.Dn,
                            tile_slice(sin_buffer, (1, self.Dr), (row, 0)),
                            tile_slice(cos_buffer, (1, self.Dr), (row, 0)),
                            tile_slice(rope_values, (1, self.Dr), (row, 0)),
                            self.rope_mode,
                        )
                    head_rope = output_heads[
                        head, None, KV_ELEMENTS : KV_ELEMENTS + self.Dr
                    ]
                    _copy(
                        tile_slice(head_rope, (vector_rows, self.Dr), (batch, 0)),
                        rope_values,
                    )

    @jit
    def run_mx_batch8(
        self,
        wqb_matmul_fp32: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        q_nope_bf16: Tensor,
        q_rope_bf16: Tensor,
    ):
        batch_rows = 8
        width = self.Dn + self.Dr
        rows = wqb_matmul_fp32.shape[0]
        task_count = rows * self.N
        batch_count = task_count // batch_rows
        source_rows = wqb_matmul_fp32.view((task_count, width))
        input_batches = _ring_buffer(
            MemLoc.UB, (batch_rows, width), dtypes.float32, depth=2
        )
        nope_batches = _ring_buffer(
            MemLoc.UB, (batch_rows, self.Dn), dtypes.bfloat16, depth=2
        )
        if dsl.const_expr(self.Dr != 0):
            rope_batches = _ring_buffer(
                MemLoc.UB, (batch_rows, self.Dr), dtypes.bfloat16, depth=2
            )
            rope_sin_buffer = _buffer(MemLoc.UB, (1, self.Dr), dtypes.bfloat16)
            rope_cos_buffer = _buffer(MemLoc.UB, (1, self.Dr), dtypes.bfloat16)
        worker = get_block_idx() * VECTOR_SUBBLOCKS + get_subblock_id()
        worker_count = get_block_num() * VECTOR_SUBBLOCKS
        batches_per_worker = (batch_count + worker_count - 1) // worker_count
        batch_begin = worker * batches_per_worker
        batch_end = batch_begin + batches_per_worker
        if batch_end > batch_count:
            batch_end = batch_count
        for batch in range(batch_begin, batch_end):
            _copy(
                input_batches.next_write(),
                tile_slice(source_rows, (batch_rows, width), (batch, 0)),
            )
            values = input_batches.next_read()
            nope_values = nope_batches.next_write()
            with vf(mode="simd"):
                mask = rr.full_mask()
                for local_row in range(batch_rows):
                    source_base = local_row * width
                    target_base = local_row * self.Dn
                    for offset in range(0, self.Dn, VECTOR_FLOAT_ELEMENTS):
                        encoded = rr.vcast(
                            rr.vload(values, source_base + offset),
                            dtypes.bfloat16,
                            mask=mask,
                            rounding=rr.RoundingMode.RN,
                            reg_layout=RegLayout.ZERO,
                        )
                        rr.vstore_pack(
                            nope_values,
                            target_base + offset,
                            encoded,
                            mask,
                            pack_mode=rr.PackMode.B32_TO_B16,
                        )
            _copy(
                tile_slice(q_nope_bf16, (batch_rows, self.Dn), (batch, 0)),
                nope_batches.next_read(),
            )
            if dsl.const_expr(self.Dr != 0):
                rope_values = rope_batches.next_write()
                # N is divisible by eight, so the batch shares one token's RoPE.
                token = batch * batch_rows // self.N
                if dsl.const_expr(self.rope_mode != 0):
                    _copy(
                        rope_sin_buffer, tile_slice(rope_sin, (1, self.Dr), (token, 0))
                    )
                    _copy(
                        rope_cos_buffer, tile_slice(rope_cos, (1, self.Dr), (token, 0))
                    )
                for local_row in range(batch_rows):
                    _rope_row(
                        values,
                        local_row * width + self.Dn,
                        rope_sin_buffer,
                        rope_cos_buffer,
                        tile_slice(rope_values, (1, self.Dr), (local_row, 0)),
                        self.rope_mode,
                    )
                _copy(
                    tile_slice(q_rope_bf16, (batch_rows, self.Dr), (batch, 0)),
                    rope_batches.next_read(),
                )


class _Assemble:
    def __init__(
        self,
        N,
        Dr,
        qkv_quant_mode,
        batch_rows,
        absorb_tile_rows,
        query_prepare,
        logical_rows=0,
    ):
        self.N = N
        self.Dr = Dr
        self.qkv_quant_mode = qkv_quant_mode
        self.batch_rows = batch_rows
        self.has_tail = not logical_rows or logical_rows % batch_rows != 0
        self.absorb_tile_rows = absorb_tile_rows
        self.query_prepare = query_prepare
        self.assembly_storage = (
            _Bfloat16Storage() if qkv_quant_mode == 3 else _Float32Storage()
        )

    @jit
    def run(
        self,
        q_nope_absorbed_bf16: Tensor,
        q_rope_bf16: Tensor,
        q_fp32: Tensor,
        q: Tensor,
        descale_q: Tensor,
        scratch_fp32: Tensor,
        wqb_matmul_fp32: Tensor,
        descale_qr: Tensor,
        descale_wqb: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        q_nope_bf16: Tensor,
        input_prepared: bool = False,
        head_group: int = 1,
    ):
        T = q_nope_absorbed_bf16.shape[0] // self.N
        inputs = _ring_buffer(
            MemLoc.UB, (self.batch_rows, LATENT), dtypes.bfloat16, depth=2
        )
        outputs = _ring_buffer(
            MemLoc.UB,
            (self.batch_rows, LATENT + self.Dr),
            self.assembly_storage.dtype,
            depth=2,
        )
        output_dtype = dtypes.bfloat16
        if dsl.const_expr(self.qkv_quant_mode == 2):
            output_dtype = dtypes.float8_e4m3fn
        encoded_outputs = _ring_buffer(
            MemLoc.UB, (self.batch_rows, LATENT + self.Dr), output_dtype, depth=2
        )
        scale_outputs = _ring_buffer(
            MemLoc.UB, (self.batch_rows, 1), dtypes.float32, depth=2
        )
        scale_buffer = _buffer(MemLoc.UB, (self.batch_rows, 2 * VECTOR), dtypes.float32)
        if dsl.const_expr(self.has_tail):
            head_inputs = q_nope_absorbed_bf16.view((self.N, T, LATENT))
        q_tokens = q.view((T, self.N * (LATENT + self.Dr)))
        q_fp32_tokens = q_fp32.view((T, self.N * (LATENT + self.Dr)))
        if dsl.const_expr(self.qkv_quant_mode == 2):
            descale_tokens = descale_q.view((T, self.N))
        if dsl.const_expr(self.Dr != 0):
            rope_inputs = _ring_buffer(
                MemLoc.UB, (self.batch_rows, self.Dr), dtypes.bfloat16, depth=2
            )
            rope_tokens = q_rope_bf16.view((T, self.N * self.Dr))
        block = get_block_idx()
        core_count = get_block_num()
        subblock = get_subblock_id()
        use_partial_max = (
            core_count * VECTOR_SUBBLOCKS
            <= scratch_fp32.shape[1] // MAXIMUM_PARTIAL_ELEMENTS
        )
        q_maximum = _buffer(MemLoc.UB, (2 * VECTOR_FLOAT_ELEMENTS,), dtypes.float32)
        if dsl.const_expr(self.qkv_quant_mode == 3):
            if use_partial_max:
                with vf(mode="simd"):
                    initial_mask = rr.full_mask()
                    initial_zero = rr.vdups(0.0, dtypes.float32, mask=initial_mask)
                    rr.vstore(q_maximum, 0, initial_zero, initial_mask)
                    rr.vstore(
                        q_maximum, VECTOR_FLOAT_ELEMENTS, initial_zero, initial_mask
                    )
        token_tiles = (T + self.absorb_tile_rows - 1) // self.absorb_tile_rows
        if not input_prepared and block < token_tiles * self.N:
            self.query_prepare.prepare_wkb_tile(
                wqb_matmul_fp32,
                descale_qr,
                descale_wqb,
                rope_sin,
                rope_cos,
                q_nope_bf16,
                q_rope_bf16,
                block,
                self.absorb_tile_rows,
            )
        for group_task in range(
            block, token_tiles * (self.N // head_group), core_count
        ):
            for local_head in range(head_group):
                task = group_task * head_group + local_head
                # Prepare one task ahead before waiting for the current WKB result.
                if not input_prepared and task + core_count < token_tiles * self.N:
                    self.query_prepare.prepare_wkb_tile(
                        wqb_matmul_fp32,
                        descale_qr,
                        descale_wqb,
                        rope_sin,
                        rope_cos,
                        q_nope_bf16,
                        q_rope_bf16,
                        task + core_count,
                        self.absorb_tile_rows,
                    )
                head = task % self.N
                token_batch_begin = (
                    task // self.N * self.absorb_tile_rows // self.batch_rows
                )
                token_begin = token_batch_begin + subblock
                token_step = VECTOR_SUBBLOCKS
                if dsl.const_expr(self.has_tail):
                    token_batches = (T + self.batch_rows - 1) // self.batch_rows
                    token_end = min(
                        token_batch_begin + self.absorb_tile_rows // self.batch_rows,
                        token_batches,
                    )
                else:
                    token_end = min(
                        token_batch_begin + self.absorb_tile_rows // self.batch_rows,
                        T // self.batch_rows,
                    )
                vec_sync_block_wait(_sync_ops.PIPE.S, HEAD_READY_EVENT, mode=2)
                for token_batch in range(token_begin, token_end, token_step):
                    valid_rows = self.batch_rows
                    if dsl.const_expr(self.has_tail):
                        valid_rows = min(
                            self.batch_rows, T - token_batch * self.batch_rows
                        )
                    if dsl.const_expr(self.qkv_quant_mode == 3):
                        _copy(
                            inputs.next_write()[0:valid_rows, None],
                            tile_slice(
                                q_fp32_tokens,
                                (self.batch_rows, LATENT + self.Dr),
                                (token_batch, head),
                            )[None, 0:LATENT],
                        )
                    elif dsl.const_expr(self.has_tail):
                        _copy(
                            inputs.next_write()[0:valid_rows, None],
                            tile_slice(
                                head_inputs[head, None, None],
                                (self.batch_rows, LATENT),
                                (token_batch, 0),
                            ),
                        )
                    else:
                        _copy(
                            inputs.next_write(),
                            tile_slice(
                                q_nope_absorbed_bf16,
                                (self.batch_rows, LATENT),
                                (head * (T // self.batch_rows) + token_batch, 0),
                            ),
                        )
                    source = inputs.next_read()
                    target = outputs.next_write()
                    if dsl.const_expr(self.qkv_quant_mode == 2):
                        with vf(mode="simd"):
                            mask = rr.full_mask()
                            for local_row in range(valid_rows):
                                input_base = local_row * LATENT
                                output_base = local_row * (LATENT + self.Dr)
                                maximum = rr.vdups(0.0, dtypes.float32, mask=mask)
                                for offset in range(0, LATENT, VECTOR):
                                    values = rr.vcast(
                                        rr.vload_unpack(
                                            source,
                                            input_base + offset,
                                            unpack_mode=rr.UnpackMode.UNPACK2,
                                        ),
                                        dtypes.float32,
                                        mask=mask,
                                    )
                                    rr.vstore(
                                        target, output_base + offset, values, mask
                                    )
                                    maximum = rr.vmax(
                                        maximum, rr.vabs(values, mask=mask), mask=mask
                                    )
                                stats_base = local_row * 2 * VECTOR
                                rr.vstore(scale_buffer, stats_base, maximum, mask)
                    elif dsl.const_expr(self.qkv_quant_mode == 3):
                        with vf(mode="simd"):
                            mask = rr.full_mask()
                            for local_row in range(valid_rows):
                                input_base = local_row * LATENT
                                output_base = local_row * (LATENT + self.Dr)
                                for offset in range(0, LATENT, VECTOR):
                                    packed_values = rr.vload_unpack(
                                        source,
                                        input_base + offset,
                                        unpack_mode=rr.UnpackMode.UNPACK2,
                                    )
                                    rr.vstore_pack(
                                        target,
                                        output_base + offset,
                                        packed_values,
                                        mask,
                                        pack_mode=rr.PackMode.B32_TO_B16,
                                    )
                    else:
                        with vf(mode="simd"):
                            mask = rr.full_mask()
                            for local_row in range(valid_rows):
                                input_base = local_row * LATENT
                                output_base = local_row * (LATENT + self.Dr)
                                for offset in range(0, LATENT, VECTOR):
                                    values = rr.vcast(
                                        rr.vload_unpack(
                                            source,
                                            input_base + offset,
                                            unpack_mode=rr.UnpackMode.UNPACK2,
                                        ),
                                        dtypes.float32,
                                        mask=mask,
                                    )
                                    rr.vstore(
                                        target, output_base + offset, values, mask
                                    )
                    if dsl.const_expr(self.Dr != 0):
                        _copy(
                            rope_inputs.next_write()[0:valid_rows, None],
                            tile_slice(
                                rope_tokens,
                                (self.batch_rows, self.Dr),
                                (token_batch, head),
                            ),
                        )
                        rope_source = rope_inputs.next_read()
                        if dsl.const_expr(self.qkv_quant_mode == 2):
                            with vf(mode="simd"):
                                rope_mask = rr.full_mask()
                                rr.vmem_bar("vst_vld")
                                for local_row in range(valid_rows):
                                    rope_values = rr.vcast(
                                        rr.vload_unpack(
                                            rope_source,
                                            local_row * self.Dr,
                                            unpack_mode=rr.UnpackMode.UNPACK2,
                                        ),
                                        dtypes.float32,
                                        mask=rope_mask,
                                    )
                                    rr.vstore(
                                        target,
                                        local_row * (LATENT + self.Dr) + LATENT,
                                        rope_values,
                                        rope_mask,
                                    )
                                    stats_base = local_row * 2 * VECTOR
                                    maximum = rr.vmax(
                                        rr.vload(scale_buffer, stats_base),
                                        rr.vabs(rope_values, mask=rope_mask),
                                        mask=rope_mask,
                                    )
                                    rr.vstore(
                                        scale_buffer, stats_base, maximum, rope_mask
                                    )
                        elif dsl.const_expr(self.qkv_quant_mode == 3):
                            with vf(mode="simd"):
                                rope_mask = rr.full_mask()
                                for local_row in range(valid_rows):
                                    packed_rope = rr.vload_unpack(
                                        rope_source,
                                        local_row * self.Dr,
                                        unpack_mode=rr.UnpackMode.UNPACK2,
                                    )
                                    rr.vstore_pack(
                                        target,
                                        local_row * (LATENT + self.Dr) + LATENT,
                                        packed_rope,
                                        rope_mask,
                                        pack_mode=rr.PackMode.B32_TO_B16,
                                    )
                            _copy(
                                tile_slice(
                                    q_fp32_tokens,
                                    (self.batch_rows, LATENT + self.Dr),
                                    (token_batch, head),
                                )[None, LATENT : LATENT + self.Dr],
                                rope_source[0:valid_rows, None],
                            )
                        else:
                            with vf(mode="simd"):
                                rope_mask = rr.full_mask()
                                for local_row in range(valid_rows):
                                    rope_values = rr.vcast(
                                        rr.vload_unpack(
                                            rope_source,
                                            local_row * self.Dr,
                                            unpack_mode=rr.UnpackMode.UNPACK2,
                                        ),
                                        dtypes.float32,
                                        mask=rope_mask,
                                    )
                                    rr.vstore(
                                        target,
                                        local_row * (LATENT + self.Dr) + LATENT,
                                        rope_values,
                                        rope_mask,
                                    )
                    values = outputs.next_read()
                    if dsl.const_expr(self.qkv_quant_mode == 3):
                        if use_partial_max:
                            with vf(mode="simd"):
                                max_mask = rr.full_mask()
                                rr.vmem_bar("vst_vld")
                                q_row_maximum = rr.vload(q_maximum, 0)
                                q_row_nan = rr.vload(q_maximum, VECTOR_FLOAT_ELEMENTS)
                                one = rr.vdups(1.0, dtypes.float32, mask=max_mask)
                                zero = rr.vdups(0.0, dtypes.float32, mask=max_mask)
                                for local_row in range(valid_rows):
                                    for offset in range(0, LATENT + self.Dr, VECTOR):
                                        value = self.assembly_storage.load(
                                            values,
                                            local_row * (LATENT + self.Dr) + offset,
                                        )
                                        q_row_maximum = rr.vmax(
                                            q_row_maximum,
                                            rr.vabs(value, mask=max_mask),
                                            mask=max_mask,
                                        )
                                        q_row_nan = rr.vmax(
                                            q_row_nan,
                                            rr.vselect(
                                                one,
                                                zero,
                                                cond_mask=rr.vne(
                                                    value, value, mask=max_mask
                                                ),
                                            ),
                                            mask=max_mask,
                                        )
                                rr.vstore(q_maximum, 0, q_row_maximum, max_mask)
                                rr.vstore(
                                    q_maximum,
                                    VECTOR_FLOAT_ELEMENTS,
                                    q_row_nan,
                                    max_mask,
                                )
                    else:
                        encoded_target = encoded_outputs.next_write()
                        if dsl.const_expr(self.qkv_quant_mode == 2):
                            batch_scales = scale_outputs.next_write()
                            with vf(mode="simd"):
                                full = rr.full_mask()
                                rr.vmem_bar("vst_vld")
                                for local_row in range(valid_rows):
                                    stats_base = local_row * 2 * VECTOR
                                    maximum = rr.vload(scale_buffer, stats_base)
                                    amax = rr.vdup(
                                        rr.vreduce_max(maximum, mask=full), mask=full
                                    )
                                    scale = rr.vdiv(
                                        amax,
                                        rr.vdups(E4_MAX, dtypes.float32, mask=full),
                                        mask=full,
                                        precision_mode=True,
                                    )
                                    rr.vstore_first(scale_buffer, stats_base, scale)
                                    rr.vstore_first(batch_scales, local_row, scale)
                            _copy(
                                tile_slice(
                                    descale_tokens,
                                    (self.batch_rows, 1),
                                    (token_batch, head),
                                ),
                                scale_outputs.next_read()[0:valid_rows, None],
                            )
                            with vf(mode="simd"):
                                quant_mask = rr.full_mask()
                                rr.vmem_bar("vst_vld")
                                zero = rr.vdups(0.0, dtypes.float32, mask=quant_mask)
                                for local_row in range(valid_rows):
                                    row_base = local_row * (LATENT + self.Dr)
                                    broadcast_scale = rr.vload_broadcast(
                                        scale_buffer, local_row * 2 * VECTOR
                                    )
                                    is_zero = rr.veq(
                                        broadcast_scale, zero, mask=quant_mask
                                    )
                                    denominator = rr.vselect(
                                        rr.vdups(1.0, dtypes.float32, mask=quant_mask),
                                        broadcast_scale,
                                        cond_mask=is_zero,
                                    )
                                    for quant_offset in range(
                                        0, LATENT + self.Dr, VECTOR
                                    ):
                                        normalized = rr.vdiv(
                                            rr.vload(values, row_base + quant_offset),
                                            denominator,
                                            mask=quant_mask,
                                            precision_mode=False,
                                        )
                                        normalized = rr.vselect(
                                            zero, normalized, cond_mask=is_zero
                                        )
                                        encoded = _quantize_cast(
                                            normalized, quant_mask, False
                                        )
                                        rr.vstore_pack(
                                            encoded_target,
                                            row_base + quant_offset,
                                            encoded,
                                            quant_mask,
                                            pack_mode=rr.PackMode.B32_TO_B8,
                                        )
                        else:
                            with vf(mode="simd"):
                                cast_mask = rr.full_mask()
                                for local_row in range(valid_rows):
                                    row_base = local_row * (LATENT + self.Dr)
                                    for cast_offset in range(
                                        0, LATENT + self.Dr, VECTOR
                                    ):
                                        encoded_bf16 = rr.vcast(
                                            rr.vload(values, row_base + cast_offset),
                                            dtypes.bfloat16,
                                            mask=cast_mask,
                                            rounding=rr.RoundingMode.RN,
                                            reg_layout=RegLayout.ZERO,
                                        )
                                        rr.vstore_pack(
                                            encoded_target,
                                            row_base + cast_offset,
                                            encoded_bf16,
                                            cast_mask,
                                            pack_mode=rr.PackMode.B32_TO_B16,
                                        )
                        encoded_values = encoded_outputs.next_read()
                        _copy(
                            tile_slice(
                                q_tokens,
                                (self.batch_rows, LATENT + self.Dr),
                                (token_batch, head),
                            ),
                            encoded_values[0:valid_rows, None],
                        )
                # Empty subblocks must acknowledge the same task generation.
                vec_sync_block_arrive(_sync_ops.PIPE.MTE3, HEAD_CONSUMED_EVENT, mode=2)
        if dsl.const_expr(self.qkv_quant_mode == 3):
            if use_partial_max:
                partials = scratch_fp32.view(
                    (
                        scratch_fp32.shape[0]
                        * scratch_fp32.shape[1]
                        // MAXIMUM_PARTIAL_ELEMENTS,
                        MAXIMUM_PARTIAL_ELEMENTS,
                    )
                )
                partial = _buffer(
                    MemLoc.UB, (1, MAXIMUM_PARTIAL_ELEMENTS), dtypes.float32
                )
                with vf(mode="simd"):
                    final_partial_mask = rr.full_mask()
                    partial_mask = rr.update_mask(
                        MAXIMUM_PARTIAL_ELEMENTS, elem_bits=32
                    )[0]
                    rr.vstore(
                        partial,
                        0,
                        rr.vdups(0.0, dtypes.float32, mask=partial_mask),
                        partial_mask,
                    )
                    rr.vmem_bar("vst_vld")
                    final_partial_maximum = rr.vdup(
                        rr.vreduce_max(rr.vload(q_maximum, 0), mask=final_partial_mask),
                        mask=final_partial_mask,
                    )
                    final_partial_nan = rr.vdup(
                        rr.vreduce_max(
                            rr.vload(q_maximum, VECTOR_FLOAT_ELEMENTS),
                            mask=final_partial_mask,
                        ),
                        mask=final_partial_mask,
                    )
                    rr.vstore_first(partial, 0, final_partial_maximum)
                    rr.vstore_first(partial, 1, final_partial_nan)
                _copy(
                    tile_slice(
                        partials,
                        (1, MAXIMUM_PARTIAL_ELEMENTS),
                        (block * VECTOR_SUBBLOCKS + subblock, 0),
                    ),
                    partial,
                )


class _PerTokenHeadQStrategy:
    def __init__(
        self, N, Dr, batch_rows, absorb_tile_rows, query_prepare, logical_rows=0
    ):
        query_prepare = _QueryPrepare(
            N,
            query_prepare.Dn,
            Dr,
            query_prepare.w_qr_quant_mode,
            query_prepare.rope_mode,
            mx_token_batch_rows=8,
        )
        self.assemble = _Assemble(
            N, Dr, 2, batch_rows, absorb_tile_rows, query_prepare, logical_rows
        )

    @jit
    def finish(
        self, q_fp32: Tensor, q: Tensor, descale_q: Tensor, scratch_fp32: Tensor
    ):
        return


class _Hif8OutputQuantize:
    def __init__(self, width):
        self.width = width
        self.batch_rows = 8
        self.wide_batch_rows = 16
        self.single = _Hif8Quantize(width, False, _Bfloat16Storage())
        self.batch = _Hif8Quantize(width * self.batch_rows, False, _Bfloat16Storage())
        self.wide_batch = _Hif8Quantize(
            width * self.wide_batch_rows, False, _Bfloat16Storage()
        )

    @jit
    def run(
        self, values_fp32: Tensor, quantized_values: Tensor, descale_output: Tensor
    ):
        scalar = _buffer(MemLoc.UB, (1,), dtypes.float32)
        _copy(scalar, descale_output)
        self.run_with_scale(values_fp32, quantized_values, scalar)

    @jit
    def run_with_scale(
        self, values_fp32: Tensor, quantized_values: Tensor, scalar: Tensor
    ):
        rows = values_fp32.shape[0]
        worker_count = get_block_num() * VECTOR_SUBBLOCKS
        if (
            rows % self.wide_batch_rows == 0
            and rows >= worker_count * self.wide_batch_rows
        ):
            self.wide_batch.run_with_scale(
                values_fp32.view(
                    (rows // self.wide_batch_rows, self.width * self.wide_batch_rows)
                ),
                quantized_values.view(
                    (rows // self.wide_batch_rows, self.width * self.wide_batch_rows)
                ),
                scalar,
            )
        elif rows % self.batch_rows == 0 and rows >= worker_count * self.batch_rows:
            self.batch.run_with_scale(
                values_fp32.view(
                    (rows // self.batch_rows, self.width * self.batch_rows)
                ),
                quantized_values.view(
                    (rows // self.batch_rows, self.width * self.batch_rows)
                ),
                scalar,
            )
        else:
            self.single.run_with_scale(values_fp32, quantized_values, scalar)


class _PerTensorQStrategy:
    def __init__(
        self, N, Dr, batch_rows, absorb_tile_rows, query_prepare, logical_rows=0
    ):
        self.assemble = _Assemble(
            N, Dr, 3, batch_rows, absorb_tile_rows, query_prepare, logical_rows
        )
        self.maximum = _GlobalMaximum(KV_ELEMENTS + Dr, _Bfloat16Storage())
        self.quantize = _Hif8OutputQuantize(KV_ELEMENTS + Dr)

    @jit
    def finish(
        self, q_fp32: Tensor, q: Tensor, descale_q: Tensor, scratch_fp32: Tensor
    ):
        partial_capacity = scratch_fp32.shape[1] // MAXIMUM_PARTIAL_ELEMENTS
        if get_block_num() * VECTOR_SUBBLOCKS <= partial_capacity:
            local_q_scale = _buffer(MemLoc.UB, (1,), dtypes.float32)
            self.maximum.reduce_partials(
                scratch_fp32,
                local_q_scale,
                get_block_num() * VECTOR_SUBBLOCKS,
                partial_capacity,
            )
            if get_block_idx() == 0 and get_subblock_id() == 0:
                _copy(descale_q, local_q_scale)
            self.quantize.run_with_scale(q_fp32, q, local_q_scale)
        else:
            if q_fp32.shape[0] >= MAXIMUM_PARALLEL_ROWS:
                self.maximum.run_parallel(
                    q_fp32, descale_q, scratch_fp32, partial_capacity
                )
            elif get_block_idx() == 0 and get_subblock_id() == 0:
                self.maximum.run(q_fp32, descale_q)
            _mix_join()
            onchip_rewind()
            self.quantize.run(q_fp32, q, descale_q)
        _mix_join()


class _Cache:
    def __init__(self, Dr, w_qr_quant_mode, qkv_quant_mode, cache_mode, rope_mode):
        self.Dr = Dr
        self.w_qr_quant_mode = w_qr_quant_mode
        self.qkv_quant_mode = qkv_quant_mode
        self.cache_mode = cache_mode
        self.rope_mode = rope_mode
        self.quant_dtype = dtypes.float8_e4m3fn
        self.quant_rounding = rr.RoundingMode.RN
        if qkv_quant_mode == 3:
            self.quant_dtype = dtypes.hifloat8
            self.quant_rounding = rr.RoundingMode.RNA
        self.payload_bytes = KV_ELEMENTS + Dr
        if qkv_quant_mode == 0:
            self.payload_bytes = BF16_BYTES * (KV_ELEMENTS + Dr)
        elif qkv_quant_mode == 1:
            self.payload_bytes = (
                KV_ELEMENTS + BF16_BYTES * Dr + FLOAT_BYTES * GROUP_COUNT
            )

    @jit
    def run(
        self,
        kv_norm_fp32: Tensor,
        kva_matmul_fp32: Tensor,
        descale_x: Tensor,
        descale_wkva: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        quant_scale_kv: Tensor,
        kv_cache: Tensor,
        cache_index: Tensor,
    ):
        input_rows = _ring_buffer(MemLoc.UB, (1, KV_ELEMENTS), dtypes.float32, depth=2)
        payload_rows = _ring_buffer(
            MemLoc.UB, (1, self.payload_bytes), dtypes.uint8, depth=2
        )
        cache_scale = 1.0
        activation_scale = 1.0
        kr_rounded = None
        if dsl.const_expr(self.qkv_quant_mode in (2, 3)):
            cache_scale = quant_scale_kv[0]
        if dsl.const_expr(self.Dr != 0):
            kr_rows = _ring_buffer(MemLoc.UB, (1, self.Dr), dtypes.float32, depth=2)
            rope_sin_buffer = _buffer(MemLoc.UB, (1, self.Dr), dtypes.bfloat16)
            rope_cos_buffer = _buffer(MemLoc.UB, (1, self.Dr), dtypes.bfloat16)
            if dsl.const_expr(self.qkv_quant_mode in (2, 3)):
                kr_rounded = _buffer(MemLoc.UB, (1, self.Dr), dtypes.bfloat16)
            if dsl.const_expr(self.w_qr_quant_mode == 2):
                kr_scales = _buffer(MemLoc.UB, (1, self.Dr), dtypes.float32)
                _copy(
                    kr_scales,
                    tile_slice(descale_wkva, (1, self.Dr), (0, KV_ELEMENTS // self.Dr)),
                )
                activation_scale = descale_x[0]
        worker = get_block_idx() * VECTOR_SUBBLOCKS + get_subblock_id()
        worker_count = get_block_num() * VECTOR_SUBBLOCKS
        for token in range(worker, kv_norm_fp32.shape[0], worker_count):
            slot = cache_index[token]
            if slot >= 0 and slot < kv_cache.shape[0] * kv_cache.shape[1]:
                _copy(
                    input_rows.next_write(),
                    tile_slice(kv_norm_fp32, (1, KV_ELEMENTS), (token, 0)),
                )
                values = input_rows.next_read()
                payload = payload_rows.next_write()
                if dsl.const_expr(self.qkv_quant_mode == 0):
                    bf16_payload = _view(
                        payload, dtype=dtypes.bfloat16, shape=(1, KV_ELEMENTS + self.Dr)
                    )
                    with vf(mode="simd"):
                        mask = rr.full_mask()
                        for offset in range(0, KV_ELEMENTS, VECTOR_ELEMENTS):
                            value = _bf16_sat_input(rr.vload(values, offset), mask)
                            encoded = rr.vcast(
                                value,
                                dtypes.bfloat16,
                                mask=mask,
                                rounding=rr.RoundingMode.RN,
                                reg_layout=RegLayout.ZERO,
                            )
                            rr.vstore_pack(
                                bf16_payload,
                                offset,
                                encoded,
                                mask,
                                pack_mode=rr.PackMode.B32_TO_B16,
                            )
                elif dsl.const_expr(self.qkv_quant_mode == 1):
                    scale_payload = _view(
                        payload,
                        dtype=dtypes.float32,
                        shape=(1, GROUP_COUNT),
                        offset=KV_ELEMENTS + BF16_BYTES * self.Dr,
                    )
                    with vf(mode="simd"):
                        mask = rr.full_mask()
                        zero = rr.vdups(0.0, dtypes.float32, mask=mask)
                        one = rr.vdups(1.0, dtypes.float32, mask=mask)
                        nan = rr.vreinterpret(
                            rr.vdups(2143289344, dtypes.uint32, mask=mask),
                            dtypes.float32,
                        )
                        for group in range(GROUP_COUNT):
                            maximum = zero
                            has_nan = zero
                            for step in range(GROUP_ELEMENTS // VECTOR_ELEMENTS):
                                value = rr.vload(
                                    values,
                                    group * GROUP_ELEMENTS + step * VECTOR_ELEMENTS,
                                )
                                maximum = rr.vmax(
                                    maximum, rr.vabs(value, mask=mask), mask=mask
                                )
                                has_nan = rr.vmax(
                                    has_nan,
                                    rr.vselect(
                                        one,
                                        zero,
                                        cond_mask=rr.vne(value, value, mask=mask),
                                    ),
                                    mask=mask,
                                )
                            amax = rr.vdup(
                                rr.vreduce_max(maximum, mask=mask), mask=mask
                            )
                            nan_seen = rr.vdup(
                                rr.vreduce_max(has_nan, mask=mask), mask=mask
                            )
                            scale = rr.vdiv(
                                rr.vmaxs(amax, CACHE_AMAX_FLOOR, mask=mask),
                                rr.vdups(E4M3_MAX, dtypes.float32, mask=mask),
                                mask=mask,
                                precision_mode=True,
                            )
                            scale = rr.vselect(
                                nan, scale, cond_mask=rr.vgts(nan_seen, 0.0, mask=mask)
                            )
                            rr.vstore_first(scale_payload, group, scale)
                            for step in range(GROUP_ELEMENTS // VECTOR_ELEMENTS):
                                offset = group * GROUP_ELEMENTS + step * VECTOR_ELEMENTS
                                normalized = rr.vdiv(
                                    rr.vload(values, offset),
                                    scale,
                                    mask=mask,
                                    precision_mode=True,
                                )
                                encoded = _quantize_cast(normalized, mask, False)
                                rr.vstore_pack(
                                    payload,
                                    offset,
                                    encoded,
                                    mask,
                                    pack_mode=rr.PackMode.B32_TO_B8,
                                )
                else:
                    with vf(mode="simd"):
                        mask = rr.full_mask()
                        for offset in range(0, KV_ELEMENTS, VECTOR_ELEMENTS):
                            value = rr.vmuls(
                                rr.vload(values, offset), cache_scale, mask=mask
                            )
                            encoded = _quantize_cast(
                                value, mask, self.qkv_quant_mode == 3
                            )
                            rr.vstore_pack(
                                payload,
                                offset,
                                rr.vreinterpret(encoded, dtypes.float8_e4m3fn),
                                mask,
                                pack_mode=rr.PackMode.B32_TO_B8,
                            )
                if dsl.const_expr(self.Dr != 0):
                    _copy(
                        kr_rows.next_write(),
                        tile_slice(
                            kva_matmul_fp32,
                            (1, self.Dr),
                            (token, KV_ELEMENTS // self.Dr),
                        ),
                    )
                    kr = kr_rows.next_read()
                    if dsl.const_expr(self.w_qr_quant_mode == 2):
                        with vf(mode="simd"):
                            mask = rr.full_mask()
                            value = rr.vmul(
                                rr.vload(kr, 0), rr.vload(kr_scales, 0), mask=mask
                            )
                            rr.vstore(
                                kr,
                                0,
                                rr.vmuls(value, activation_scale, mask=mask),
                                mask,
                            )
                            rr.vmem_bar("vst_vld")
                    if dsl.const_expr(self.rope_mode != 0):
                        _copy(
                            rope_sin_buffer,
                            tile_slice(rope_sin, (1, self.Dr), (token, 0)),
                        )
                        _copy(
                            rope_cos_buffer,
                            tile_slice(rope_cos, (1, self.Dr), (token, 0)),
                        )
                    if dsl.const_expr(self.qkv_quant_mode in (0, 1)):
                        kr_offset_bytes = KV_ELEMENTS
                        if dsl.const_expr(self.qkv_quant_mode == 0):
                            kr_offset_bytes = KV_ELEMENTS * BF16_BYTES
                        kr_target = _view(
                            payload,
                            dtype=dtypes.bfloat16,
                            shape=(1, self.Dr),
                            offset=kr_offset_bytes,
                        )
                        _rope_row(
                            kr,
                            0,
                            rope_sin_buffer,
                            rope_cos_buffer,
                            kr_target,
                            self.rope_mode,
                            1,
                        )
                    else:
                        _rope_row(
                            kr,
                            0,
                            rope_sin_buffer,
                            rope_cos_buffer,
                            kr_rounded,
                            self.rope_mode,
                            1,
                        )
                        with vf(mode="simd"):
                            mask = rr.full_mask()
                            value = rr.vcast(
                                rr.vload_unpack(
                                    kr_rounded, 0, unpack_mode=rr.UnpackMode.UNPACK2
                                ),
                                dtypes.float32,
                                mask=mask,
                            )
                            encoded = _quantize_cast(
                                rr.vmuls(value, cache_scale, mask=mask),
                                mask,
                                self.qkv_quant_mode == 3,
                            )
                            rr.vstore_pack(
                                payload,
                                KV_ELEMENTS,
                                rr.vreinterpret(encoded, dtypes.float8_e4m3fn),
                                mask,
                                pack_mode=rr.PackMode.B32_TO_B8,
                            )
                if dsl.const_expr(self.cache_mode == 0):
                    cache_rows = kv_cache.view(
                        (kv_cache.shape[0] * kv_cache.shape[1], self.payload_bytes)
                    )
                    _copy(
                        tile_slice(cache_rows, (1, self.payload_bytes), (slot, 0)),
                        payload_rows.next_read(),
                    )
                else:
                    page = slot // kv_cache.shape[1]
                    row = slot % kv_cache.shape[1]
                    physical = kv_cache.view(
                        (
                            kv_cache.shape[0],
                            self.payload_bytes // FRACTAL_BYTES,
                            kv_cache.shape[1],
                            FRACTAL_BYTES,
                        )
                    )
                    ready = payload_rows.next_read()
                    for chunk in range(self.payload_bytes // FRACTAL_BYTES):
                        destination = physical[page, chunk, row, None].view(
                            (1, FRACTAL_BYTES)
                        )
                        _copy(
                            destination,
                            tile_slice(ready, (1, FRACTAL_BYTES), (0, chunk)),
                        )


@kernel(profile=ProfileSpec(name="quant_mla_prolog", op_type="quant_mla_prolog"))
class _QuantMlaPrologKernel:
    def __init__(
        self,
        q_lora,
        N,
        Dn,
        Dr,
        w_qr_quant_mode,
        qkv_quant_mode,
        weight_is_nz,
        cache_mode,
        rope_mode,
        input_tile_rows,
        wqb_tile_rows,
        absorb_tile_rows,
        qa_rows=0,
        qa_dim=0,
        qa_config=(0, 1, 128),
        compact_head_handoff=(0, 0),
    ):
        self.qa_enabled = qa_rows > 0
        self.qa_rows = qa_rows
        self.hif_qr_resident_rows = (
            HIF_QR_EXTENDED_ROWS
            if HIF_QR_RESIDENT_ROWS < qa_rows <= HIF_QR_EXTENDED_ROWS
            else HIF_QR_RESIDENT_ROWS
        )
        self.qa_bm = qa_config[0] if qa_rows else input_tile_rows
        self.qa_splits = qa_config[1]
        self.qa_bn = qa_config[2] if self.qa_enabled else QA_PROJECTION_COLUMNS
        self.qa_fused_norm = self.qa_enabled and self.qa_splits == 2
        self.qa_nc = q_lora // 128
        self.qa_nm = (qa_rows + self.qa_bm - 1) // self.qa_bm
        qa_columns = QA_PIPELINE_COLUMNS
        if (
            w_qr_quant_mode == 1
            and qkv_quant_mode in (0, 2)
            and weight_is_nz
            and 0 < qa_rows <= QA_BALANCED_MAX_ROWS
            and self.qa_splits == 2
            and q_lora % QA_BALANCED_COLUMNS == 0
        ):
            qa_columns = QA_BALANCED_COLUMNS
        if self.qa_enabled:
            self.qa_pipeline = QaRowPipeline(
                qa_rows,
                q_lora,
                qa_dim,
                w_qr_quant_mode,
                weight_is_nz,
                self.qa_bm,
                self.qa_splits,
                True,
                qa_columns,
            )
            if (
                qa_columns == QA_BALANCED_COLUMNS
                and qa_dim % (self.qa_splits * QA_BALANCED_L1_REDUCTION) == 0
            ):
                self.qa_pipeline.l1_reduction = QA_BALANCED_L1_REDUCTION
            elif (
                w_qr_quant_mode == 1
                and qkv_quant_mode in (0, 2)
                and weight_is_nz
                and 0 < qa_rows <= QA_BALANCED_MAX_ROWS
                and qa_dim % (self.qa_splits * QA_L1_REDUCTION) == 0
            ):
                self.qa_pipeline.l1_reduction = QA_L1_REDUCTION
        self.q_lora = q_lora
        self.N = N
        self.Dn = Dn
        self.Dr = Dr
        self.w_qr_quant_mode = w_qr_quant_mode
        self.qkv_quant_mode = qkv_quant_mode
        self.rope_mode = rope_mode
        self.input_tile_rows = input_tile_rows
        self.wqb_tile_rows = wqb_tile_rows
        self.wkb_tile_rows = min(
            WKB_C2V_MAX_ROWS, max(WKB_C2V_MIN_ROWS, absorb_tile_rows)
        )
        if qkv_quant_mode in (0, 1):
            self.wqb_tile_rows = min(WKB_C2V_MAX_ROWS, wqb_tile_rows)
            if wqb_tile_rows == WKB_DIRECT_MAX_ROWS:
                self.wqb_tile_rows = WKB_DIRECT_MAX_ROWS
            if w_qr_quant_mode == 1 and wqb_tile_rows == WQB_TAIL_HEAD_ROWS:
                self.wqb_tile_rows = WQB_TAIL_HEAD_ROWS
            if w_qr_quant_mode == 1 and wqb_tile_rows == WQB_LARGE_HEAD_ROWS:
                self.wqb_tile_rows = WQB_LARGE_HEAD_ROWS
            if (
                w_qr_quant_mode == 1
                and Dn + Dr == 192
                and wqb_tile_rows == WQB_SINGLE_TAIL_ROWS
            ):
                self.wqb_tile_rows = WQB_SINGLE_TAIL_ROWS
            self.wkb_tile_rows = self.wqb_tile_rows
        if (
            w_qr_quant_mode == 1
            and qkv_quant_mode in (0, 1, 2)
            and WQB_WIDE_MAX_ROWS < qa_rows <= COMPACT_HANDOFF_MAX_ROWS
            and compact_head_handoff[0]
        ):
            self.wqb_tile_rows = wqb_tile_rows
            self.wkb_tile_rows = absorb_tile_rows
        wqb_reduction = (
            PROJECTION_REDUCTION if q_lora == 640 else WQB_PROJECTION_REDUCTION
        )
        hif_wqb_hierarchy = (
            w_qr_quant_mode == 2
            and qkv_quant_mode in (0, 3)
            and q_lora % HIF_WQB_L1_REDUCTION == 0
        )
        if hif_wqb_hierarchy:
            wqb_reduction = HIF_WQB_L1_REDUCTION
        wqb_output = N * (Dn + Dr)
        wqb_columns = (
            256
            if wqb_output % 256 == 0
            else QA_PROJECTION_COLUMNS
            if wqb_output % QA_PROJECTION_COLUMNS == 0
            else PROJECTION_COLUMNS
        )
        if qkv_quant_mode in (0, 1):
            wqb_columns = Dn + Dr
        # Experimental transport routing, not a public shape restriction.
        # Keep the scalar path for small T and partial row tiles.
        self.use_hif_c2v = (
            w_qr_quant_mode == 2
            and qkv_quant_mode == 0
            and qa_rows >= 128
            and qa_rows % self.wqb_tile_rows == 0
        )
        self.coalesce_wkb_b = (
            w_qr_quant_mode == 1
            and qkv_quant_mode == 2
            and 0 < qa_rows <= 32
            and N >= 64
            and Dn == 128
        )
        wkb_columns = (
            128 if self.wkb_tile_rows == WQB_SINGLE_TAIL_ROWS else WKB_C2V_COLUMNS
        )
        self.mode0_coalesce_wkb_b = (
            w_qr_quant_mode == 1
            and qkv_quant_mode == 0
            and 0 < qa_rows <= 32
            and Dn == 128
            and 2
            * BF16_BYTES
            * (self.wkb_tile_rows * WKB_C2V_REDUCTION + Dn * wkb_columns)
            <= dsl.get_mem_size("l1")
        )
        self.wqb_projection = _Projection(
            0,
            0,
            w_qr_quant_mode,
            weight_is_nz,
            weight_is_nz and w_qr_quant_mode == 1,
            tile_rows=self.wqb_tile_rows,
            tile_columns=wqb_columns,
            tile_reduction=wqb_reduction,
            head_pipeline=qkv_quant_mode in (0, 1),
            hif_c2v=self.use_hif_c2v,
            hif_split_l0=hif_wqb_hierarchy,
            join_after_prefetch=w_qr_quant_mode == 1,
            force_split_l0=(
                qa_rows == WQB_LARGE_HEAD_ROWS + 1
                and wqb_tile_rows == WQB_TAIL_HEAD_ROWS
                and Dn + Dr == 256
                and w_qr_quant_mode == 1
            ),
        )
        self.use_mm34_shared = (
            w_qr_quant_mode == 1
            and qkv_quant_mode in (0, 1)
            and self.wqb_projection.l0_reduction == 128
            and self.wqb_tile_rows <= 128
            and self.wkb_tile_rows == self.wqb_tile_rows
            and wqb_columns <= MM34_SHARED_COLUMNS
        )
        if (
            w_qr_quant_mode == 2
            and qkv_quant_mode == 0
            and not self.use_hif_c2v
            and self.wqb_tile_rows == PROJECTION_ROWS
            and self.wkb_tile_rows == self.wqb_tile_rows
            and hif_wqb_hierarchy
            and Dn + Dr <= MM34_SHARED_COLUMNS
            and (
                2 * (self.wqb_tile_rows + wqb_columns) * HIF_WQB_L1_REDUCTION
                + 2 * (self.wqb_tile_rows + WKB_C2V_COLUMNS) * Dn * BF16_BYTES
            )
            <= dsl.get_mem_size("l1")
        ):
            self.use_mm34_shared = True
        if (
            w_qr_quant_mode == 1
            and qkv_quant_mode == 2
            and weight_is_nz
            and self.wqb_tile_rows <= 32
            and q_lora % WQB_PROJECTION_REDUCTION == 0
            and wqb_output % STRIPE_COLUMNS == 0
        ):
            self.wqb_projection = WqbStripeProjection(
                self.wqb_tile_rows, QR_READY_CUBE_EVENT, MIX_JOIN_GRID_EVENT
            )
        wqb_wide_l1_bytes = (
            2 * (self.wqb_tile_rows + WQB_WIDE_COLUMNS) * WQB_PROJECTION_REDUCTION
            + (self.wqb_tile_rows + WQB_WIDE_COLUMNS) * q_lora // MX_GROUP_ELEMENTS
        )
        if (
            w_qr_quant_mode == 1
            and qkv_quant_mode == 2
            and weight_is_nz
            and 0 < qa_rows <= WQB_WIDE_MAX_ROWS
            and self.wqb_tile_rows <= WQB_WIDE_MAX_ROWS
            and q_lora % WQB_PROJECTION_REDUCTION == 0
            and wqb_output % WQB_WIDE_COLUMNS == 0
            and wqb_output // WQB_WIDE_COLUMNS >= WQB_WIDE_MIN_COLUMN_TASKS
            and wqb_output % STRIPE_COLUMNS != 0
            and wqb_wide_l1_bytes <= dsl.get_mem_size("l1")
        ):
            self.wqb_projection.tile_columns = WQB_WIDE_COLUMNS
            self.wqb_projection.tile_reduction = WQB_PROJECTION_REDUCTION
            self.wqb_projection.l0_reduction = WQB_L0_REDUCTION
            self.wqb_projection.split_l0 = True
        wqb_head_l1_bytes = (
            2 * (self.wqb_tile_rows + wqb_columns) * WQB_HEAD_L1_REDUCTION
            + (self.wqb_tile_rows + wqb_columns) * q_lora // MX_GROUP_ELEMENTS
        )
        if (
            w_qr_quant_mode == 1
            and qkv_quant_mode == 0
            and Dn == 192
            and weight_is_nz
            and 0 < qa_rows <= WQB_WIDE_MAX_ROWS
            and self.wqb_tile_rows <= WQB_WIDE_MAX_ROWS
            and q_lora % WQB_HEAD_L1_REDUCTION == 0
            and wqb_columns <= MM34_SHARED_COLUMNS
            and self.wqb_projection.l0_reduction == WQB_PROJECTION_REDUCTION
            and not self.use_mm34_shared
            and wqb_head_l1_bytes <= dsl.get_mem_size("l1")
        ):
            self.wqb_projection.tile_reduction = WQB_HEAD_L1_REDUCTION
            self.wqb_projection.split_l0 = True
        large_m_l1_bytes = (
            2 * WQB_LARGE_HEAD_ROWS + 3 * wqb_columns
        ) * wqb_reduction + (
            WQB_LARGE_HEAD_ROWS + wqb_columns
        ) * q_lora // MX_GROUP_ELEMENTS
        if (
            w_qr_quant_mode == 1
            and qkv_quant_mode == 0
            and weight_is_nz
            and qa_rows >= 2 * WQB_LARGE_HEAD_ROWS
            and qa_rows % WQB_LARGE_HEAD_ROWS == 0
            and self.wqb_tile_rows == WKB_C2V_MAX_ROWS
            and wqb_columns <= MM34_SHARED_COLUMNS
            and self.wqb_projection.l0_reduction == WQB_PROJECTION_REDUCTION
            and not self.use_mm34_shared
            and large_m_l1_bytes <= dsl.get_mem_size("l1")
        ):
            self.wqb_tile_rows = WQB_LARGE_HEAD_ROWS
            self.wkb_tile_rows = WQB_LARGE_HEAD_ROWS
            self.wqb_projection.tile_rows = WQB_LARGE_HEAD_ROWS
        if (
            w_qr_quant_mode == 1
            and qkv_quant_mode == 0
            and weight_is_nz
            and not self.use_mm34_shared
            and qa_rows > 0
            and (
                (2 * self.wqb_tile_rows + 3 * wqb_columns)
                * self.wqb_projection.tile_reduction
                + (self.wqb_tile_rows + wqb_columns) * q_lora // MX_GROUP_ELEMENTS
            )
            <= dsl.get_mem_size("l1")
        ):
            self.wqb_projection.weight_lookahead = True
        self.use_mx_handoff = (
            w_qr_quant_mode == 1
            and qkv_quant_mode == 2
            and weight_is_nz
            and 0 < qa_rows <= WQB_WIDE_MAX_ROWS
            and Dn == 192
            and Dr == 64
            and N % 2 == 0
            and isinstance(self.wqb_projection, _Projection)
            and self.wqb_projection.tile_columns == WQB_WIDE_COLUMNS
            and self.wqb_projection.split_l0
        )
        self.nope_l1_depth = 1
        self.handoff_reduction_panel = REDUCTION_PANEL
        self.shared_weight_columns = STRIPE_COLUMNS
        self.shared_acc_columns = STRIPE_COLUMNS
        if compact_head_handoff[0]:
            head_group, handoff_slots = compact_head_handoff
            stripe_columns = head_group * (Dn + Dr)
            reduction_panel = (
                COMPACT_BOUNDARY_REDUCTION_PANEL
                if qkv_quant_mode in (0, 1, 2) and qa_rows > WQB_WIDE_MAX_ROWS
                else REDUCTION_PANEL
            )
            weight_columns = max(
                stripe_columns, Dn * WKB_C2V_COLUMNS * BF16_BYTES // reduction_panel
            )
            l1_bytes = (
                2 * weight_columns * reduction_panel
                + 2 * self.wqb_tile_rows * reduction_panel
                + (self.wqb_tile_rows + stripe_columns) * q_lora // MX_GROUP
                + handoff_slots * self.wqb_tile_rows * head_group * Dn * BF16_BYTES
                + 2 * self.wkb_tile_rows * WKB_C2V_REDUCTION * BF16_BYTES
            )
            if (
                w_qr_quant_mode == 1
                and qkv_quant_mode in (0, 1, 2)
                and (rope_mode == 1 or Dr == 0)
                and (
                    0 < qa_rows <= WQB_WIDE_MAX_ROWS
                    or (
                        qkv_quant_mode in (0, 1, 2)
                        and WQB_WIDE_MAX_ROWS < qa_rows <= COMPACT_HANDOFF_MAX_ROWS
                    )
                )
                and self.wqb_tile_rows == self.wkb_tile_rows
                and q_lora % WQB_PROJECTION_REDUCTION == 0
                and l1_bytes <= dsl.get_mem_size("l1")
                and 2
                * self.wqb_tile_rows
                * max(stripe_columns, WKB_C2V_COLUMNS)
                * FLOAT_BYTES
                <= dsl.get_mem_size("l0c")
            ):
                self.wqb_projection = WqbCompactHeadProjection(
                    self.wqb_tile_rows,
                    QR_READY_CUBE_EVENT,
                    MIX_JOIN_GRID_EVENT,
                    head_group,
                    Dn + Dr,
                    weight_is_nz,
                    reduction_panel,
                )
                self.handoff_reduction_panel = reduction_panel
                self.nope_l1_depth = handoff_slots
                self.shared_weight_columns = weight_columns
                self.shared_acc_columns = max(stripe_columns, WKB_C2V_COLUMNS)
        self.use_nope_shared = (
            w_qr_quant_mode == 1
            and qkv_quant_mode == 2
            and weight_is_nz
            and rope_mode == 1
            and Dn == 128
            and Dr == 64
            and 0 < qa_rows <= 32
            and 0 < N <= 128
            and isinstance(self.wqb_projection, WqbStripeProjection)
            and self.wqb_tile_rows == self.wkb_tile_rows
        )
        if isinstance(self.wqb_projection, WqbCompactHeadProjection):
            self.use_nope_shared = True
        self.handoff_ub = True
        self.handoff_heads = 2
        self.handoff_rows = self.wqb_tile_rows
        self.use_stripe_local = False
        if (
            isinstance(self.wqb_projection, WqbStripeProjection)
            and 0 < qa_rows <= WQB_WIDE_MAX_ROWS
            and self.wqb_tile_rows == self.wkb_tile_rows
            and Dn == 128
            and Dr == 64
            and N % (STRIPE_COLUMNS // (Dn + Dr)) == 0
        ):
            self.use_mx_handoff = True
            self.handoff_heads = STRIPE_COLUMNS // (Dn + Dr)
            self.use_stripe_local = True
        if isinstance(self.wqb_projection, WqbCompactHeadProjection):
            self.use_mx_handoff = True
            self.handoff_heads = self.wqb_projection.stripe_columns // (Dn + Dr)
            self.use_stripe_local = True
        if self.use_mx_handoff:
            self.wqb_projection.head_pipeline = True
            self.wqb_projection.hif_c2v = self.handoff_ub
            self.wqb_projection.drain_handoff = True
        self.qa_projection = _Projection(
            q_lora,
            0,
            w_qr_quant_mode,
            weight_is_nz,
            weight_is_nz and w_qr_quant_mode == 1,
            tile_rows=self.qa_bm,
            tile_columns=self.qa_bn,
            tile_reduction=QA_PROJECTION_REDUCTION,
        )
        self.kva_projection = _Projection(
            KV_ELEMENTS + Dr,
            0,
            w_qr_quant_mode,
            weight_is_nz,
            weight_is_nz and w_qr_quant_mode == 1,
            tile_rows=input_tile_rows,
            tile_columns=KVA_PROJECTION_COLUMNS,
            tile_reduction=KVA_PROJECTION_REDUCTION,
        )
        self.qa_rmsnorm = _Normalize(
            q_lora,
            w_qr_quant_mode,
            self.qa_bm if self.qa_fused_norm else 0,
            hif_resident_rows=self.hif_qr_resident_rows,
        )
        self.kva_rmsnorm = _Normalize(KV_ELEMENTS, w_qr_quant_mode)
        self.qr_max = _GlobalMaximum(q_lora)
        self.qr_hif = _Hif8Quantize(q_lora)
        self.prepare = _QueryPrepare(N, Dn, Dr, w_qr_quant_mode, rope_mode)
        tail_batch_rows = 8 if absorb_tile_rows >= 32 else 1
        if qkv_quant_mode == 2:
            self.q_strategy = _PerTokenHeadQStrategy(
                N, Dr, tail_batch_rows, self.wkb_tile_rows, self.prepare, qa_rows
            )
        elif qkv_quant_mode == 3:
            self.q_strategy = _PerTensorQStrategy(
                N, Dr, tail_batch_rows, self.wkb_tile_rows, self.prepare, qa_rows
            )
        self.cache_writer = _Cache(
            Dr, w_qr_quant_mode, qkv_quant_mode, cache_mode, rope_mode
        )

    def __call__(
        self,
        x: Tensor,
        wqa: Tensor,
        wqb: Tensor,
        wkva: Tensor,
        wkb: Tensor,
        descale_x: Tensor,
        descale_wqa: Tensor,
        descale_wqb: Tensor,
        descale_wkva: Tensor,
        norm_weight_qa: Tensor,
        norm_weight_kva: Tensor,
        kv_cache: Tensor,
        cache_index: Tensor,
        quant_scale_kv: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        qa_matmul_fp32: Tensor,
        kva_matmul_fp32: Tensor,
        qa_norm_fp32: Tensor,
        kv_norm_fp32: Tensor,
        qr: Tensor,
        descale_qr: Tensor,
        wqb_matmul_fp32: Tensor,
        q_nope_bf16: Tensor,
        q_rope_bf16: Tensor,
        q_nope_absorbed_bf16: Tensor,
        q_fp32: Tensor,
        q: Tensor,
        descale_q: Tensor,
        norm_eps: float,
        qr_scale: float,
        kvr_scale: float,
        qa_partial: Tensor,
    ):
        if dsl.const_expr(self.qa_enabled):
            self.qa_pipeline.project(x, wqa, descale_x, descale_wqa, qa_partial)
            _mix_join(False)
        else:
            self.qa_projection.run(x, wqa, descale_x, descale_wqa, qa_matmul_fp32)
            _mix_join(False)
        onchip_rewind(reset_sync_id=False)
        self.kva_projection.run(x, wkva, descale_x, descale_wkva, kva_matmul_fp32)
        use_hif_norm_summary = (
            get_block_num() * VECTOR_SUBBLOCKS
            <= self.q_lora // MAXIMUM_PARTIAL_ELEMENTS
            and get_block_num() * VECTOR_SUBBLOCKS
            <= wqb_matmul_fp32.shape[0]
            * wqb_matmul_fp32.shape[1]
            // MAXIMUM_PARTIAL_ELEMENTS
        )
        use_hif_resident = (
            use_hif_norm_summary
            and x.shape[0]
            <= get_block_num() * VECTOR_SUBBLOCKS * HIF_QR_ROWS_PER_VECTOR
        )
        if dsl.const_expr(self.w_qr_quant_mode == 1):
            if dsl.const_expr(self.qa_fused_norm):
                self.qa_rmsnorm.run_mx_quantized(
                    qa_partial, norm_weight_qa, qr, descale_qr, norm_eps, qr_scale
                )
            else:
                self.qa_rmsnorm.run_mx_quantized(
                    qa_matmul_fp32, norm_weight_qa, qr, descale_qr, norm_eps, qr_scale
                )
        else:
            hif_projection = qa_matmul_fp32
            if dsl.const_expr(self.qa_fused_norm):
                hif_projection = qa_partial
            if x.shape[0] <= self.hif_qr_resident_rows:
                if get_block_idx() == 0 and get_subblock_id() == 0:
                    self.qa_rmsnorm.run_hif_quantized(
                        hif_projection,
                        norm_weight_qa,
                        descale_x,
                        descale_wqa,
                        qr,
                        descale_qr,
                        norm_eps,
                        qr_scale,
                        wqb_matmul_fp32,
                    )
            elif use_hif_resident:
                self.qa_rmsnorm.run_hif_quantized(
                    hif_projection,
                    norm_weight_qa,
                    descale_x,
                    descale_wqa,
                    qr,
                    descale_qr,
                    norm_eps,
                    qr_scale,
                    wqb_matmul_fp32,
                    True,
                )
            else:
                if use_hif_norm_summary:
                    self.qa_rmsnorm.run(
                        hif_projection,
                        norm_weight_qa,
                        descale_x,
                        descale_wqa,
                        qa_norm_fp32,
                        norm_eps,
                        qr_scale,
                        wqb_matmul_fp32,
                        True,
                    )
                else:
                    self.qa_rmsnorm.run(
                        hif_projection,
                        norm_weight_qa,
                        descale_x,
                        descale_wqa,
                        qa_norm_fp32,
                        norm_eps,
                        qr_scale,
                        wqb_matmul_fp32,
                    )
        if dsl.const_expr(self.w_qr_quant_mode == 1):
            _publish_kva_and_qr()
        else:
            if x.shape[0] <= self.hif_qr_resident_rows or use_hif_resident:
                _publish_kva_and_qr()
            else:
                _mix_join()
        # Complete both roles at the merged control-flow boundary so 0.8 can
        # release buffer IDs from every preceding normalization branch.
        _sync_ops.cube_sync_all()
        vec_sync_all()
        onchip_rewind(reset_sync_id=False)
        if dsl.const_expr(self.w_qr_quant_mode == 1):
            self.kva_rmsnorm.run(
                kva_matmul_fp32,
                norm_weight_kva,
                descale_x,
                descale_wkva,
                kv_norm_fp32,
                norm_eps,
                kvr_scale,
                kv_norm_fp32,
            )
            vec_sync_all()
        onchip_rewind(reset_sync_id=False)
        if dsl.const_expr(self.w_qr_quant_mode == 2):
            if x.shape[0] > self.hif_qr_resident_rows and not use_hif_resident:
                if use_hif_norm_summary:
                    local_qr_scale = _buffer(MemLoc.UB, (1,), dtypes.float32)
                    self.qr_max.reduce_partials(
                        wqb_matmul_fp32,
                        local_qr_scale,
                        get_block_num() * VECTOR_SUBBLOCKS,
                        self.q_lora // MAXIMUM_PARTIAL_ELEMENTS,
                    )
                    if get_block_idx() == 0 and get_subblock_id() == 0:
                        _copy(descale_qr, local_qr_scale)
                    self.qr_hif.run_with_scale(qa_norm_fp32, qr, local_qr_scale)
                else:
                    if qa_norm_fp32.shape[0] >= MAXIMUM_PARALLEL_ROWS:
                        self.qr_max.run_parallel(
                            qa_norm_fp32,
                            descale_qr,
                            qa_matmul_fp32,
                            self.q_lora // MAXIMUM_PARTIAL_ELEMENTS,
                        )
                    elif get_block_idx() == 0 and get_subblock_id() == 0:
                        self.qr_max.run(qa_norm_fp32, descale_qr)
                    _mix_join()
                    onchip_rewind()
                    self.qr_hif.run(qa_norm_fp32, qr, descale_qr)
                _mix_join()
                onchip_rewind()
        if dsl.const_expr(self.w_qr_quant_mode == 2):
            self.kva_rmsnorm.run(
                kva_matmul_fp32,
                norm_weight_kva,
                descale_x,
                descale_wkva,
                kv_norm_fp32,
                norm_eps,
                kvr_scale,
                kv_norm_fp32,
            )
            vec_sync_all()
        _sync_ops.cube_sync_all()
        onchip_rewind(reset_sync_id=False)
        descale_qr_view = descale_qr
        if dsl.const_expr(self.w_qr_quant_mode == 1):
            descale_qr_view = descale_qr.view((x.shape[0], self.q_lora // 64, 2))
        if dsl.const_expr(self.qkv_quant_mode == 3):
            if (
                get_block_num() * VECTOR_SUBBLOCKS
                <= self.q_lora // MAXIMUM_PARTIAL_ELEMENTS
            ):
                partials = qa_matmul_fp32.view(
                    (
                        qa_matmul_fp32.shape[0]
                        * self.q_lora
                        // MAXIMUM_PARTIAL_ELEMENTS,
                        MAXIMUM_PARTIAL_ELEMENTS,
                    )
                )
                initial_partial = _buffer(
                    MemLoc.UB, (1, MAXIMUM_PARTIAL_ELEMENTS), dtypes.float32
                )
                with vf(mode="simd"):
                    mask = rr.full_mask()
                    zero = rr.vdups(0.0, dtypes.float32, mask=mask)
                    partial_mask = rr.update_mask(
                        MAXIMUM_PARTIAL_ELEMENTS, elem_bits=32
                    )[0]
                    rr.vstore(initial_partial, 0, zero, partial_mask)
                # Inactive Vector roles also publish a neutral summary.
                _copy(
                    tile_slice(
                        partials,
                        (1, MAXIMUM_PARTIAL_ELEMENTS),
                        (get_block_idx() * VECTOR_SUBBLOCKS + get_subblock_id(), 0),
                    ),
                    initial_partial,
                )

        if dsl.const_expr(self.use_mx_handoff):
            vec_sync_all()
            self.cache_writer.run(
                kv_norm_fp32,
                kva_matmul_fp32,
                descale_x,
                descale_wkva,
                rope_sin,
                rope_cos,
                quant_scale_kv,
                kv_cache,
                cache_index,
            )
            vec_sync_block_arrive(_sync_ops.PIPE.MTE3, HEAD_READY_EVENT, mode=2)
            mx_handoff = None
            if dsl.const_expr(self.handoff_ub):
                mx_handoff = _ring_buffer(
                    MemLoc.UB,
                    (
                        self.handoff_rows // VECTOR_SUBBLOCKS,
                        self.handoff_heads * (self.Dn + self.Dr),
                    ),
                    dtypes.float32,
                    depth=HEAD_PIPELINE_DEPTH,
                    cross_core=True,
                )
            shared_l1_b = None
            shared_l0_b = None
            shared_l0_c = None
            nope_l1 = None
            if dsl.const_expr(self.use_nope_shared):
                shared_l1_b = _ring_buffer(
                    MemLoc.L1,
                    (self.shared_weight_columns, self.handoff_reduction_panel),
                    dtypes.float8_e4m3fn,
                    depth=2,
                    data_format="nz",
                )
                shared_l0_b = _ring_buffer(
                    MemLoc.L0B,
                    (COLUMNS_PER_PART, REDUCTION_PANEL),
                    dtypes.float8_e4m3fn,
                    depth=2,
                    data_format="nz",
                )
                shared_l0_c = _ring_buffer(
                    MemLoc.L0C,
                    (self.handoff_rows, self.shared_acc_columns),
                    dtypes.float32,
                    depth=2,
                )
                if dsl.const_expr(MX_NOPE_L1_HANDOFF):
                    nope_l1 = _ring_buffer(
                        MemLoc.L1,
                        (
                            self.handoff_heads
                            * self.Dn
                            // (FRACTAL_BYTES // BF16_BYTES),
                            self.handoff_rows * (FRACTAL_BYTES // BF16_BYTES),
                        ),
                        dtypes.bfloat16,
                        depth=self.nope_l1_depth,
                        data_format="nd",
                        cross_core=True,
                    )
                self.wqb_projection.run(
                    qr,
                    wqb,
                    descale_qr_view,
                    descale_wqb,
                    wqb_matmul_fp32,
                    c_to_v=mx_handoff,
                    shared_l1_b=shared_l1_b,
                    shared_l0_b=shared_l0_b,
                    shared_l0_c=shared_l0_c,
                )
            else:
                self.wqb_projection.run(
                    qr,
                    wqb,
                    descale_qr_view,
                    descale_wqb,
                    wqb_matmul_fp32,
                    c_to_v=mx_handoff,
                )
            if dsl.const_expr(self.qkv_quant_mode in (0, 1)):
                self.prepare.run_mx_handoff(
                    wqb_matmul_fp32,
                    mx_handoff,
                    rope_sin,
                    rope_cos,
                    q_nope_bf16,
                    q,
                    self.handoff_rows,
                    self.handoff_heads,
                    self.handoff_ub,
                    nope_l1=nope_l1,
                    packed_output=True,
                )
            else:
                self.prepare.run_mx_handoff(
                    wqb_matmul_fp32,
                    mx_handoff,
                    rope_sin,
                    rope_cos,
                    q_nope_bf16,
                    q_rope_bf16,
                    self.handoff_rows,
                    self.handoff_heads,
                    self.handoff_ub,
                    nope_l1=nope_l1,
                )
            if dsl.const_expr(
                self.use_stripe_local
                or (
                    self.handoff_heads == 2
                    and self.handoff_rows == self.wkb_tile_rows
                    and self.prepare.rope_mode == 1
                )
            ):
                _sync_ops.cube_sync_all()
                vec_sync_all()
                vec_sync_block_arrive(_sync_ops.PIPE.MTE3, MIX_JOIN_INPUT_EVENT, mode=2)
                _sync_ops.cube_sync_block_wait(
                    _sync_ops.PIPE.S, MIX_JOIN_INPUT_EVENT, mode=2
                )
                cube_sync_block_arrive(
                    _sync_ops.PIPE.MTE3, MIX_JOIN_OUTPUT_EVENT, mode=2
                )
                vec_sync_block_wait(_sync_ops.PIPE.S, MIX_JOIN_OUTPUT_EVENT, mode=2)
            else:
                _mix_join()
            if dsl.const_expr(not self.use_nope_shared):
                onchip_rewind(reset_sync_id=True)
            wkb_projection = _WkbProjection(self.wkb_tile_rows)
            if dsl.const_expr(self.qkv_quant_mode in (0, 1)):
                wkb_projection.run_direct(
                    q_nope_bf16,
                    wkb,
                    q,
                    input_prepared=True,
                    head_group=self.handoff_heads,
                    shared_l1_b=shared_l1_b,
                    shared_l0_b=shared_l0_b,
                    shared_l0_c=shared_l0_c,
                    nope_l1=nope_l1,
                )
                _mix_join()
                return
            wkb_projection.run_direct(
                q_nope_bf16,
                wkb,
                q_nope_absorbed_bf16,
                publish_heads=True,
                coalesce_b=self.coalesce_wkb_b,
                input_prepared=True,
                head_group=self.handoff_heads,
                shared_l1_b=shared_l1_b,
                shared_l0_b=shared_l0_b,
                shared_l0_c=shared_l0_c,
                nope_l1=nope_l1,
            )
            self.q_strategy.assemble.run(
                q_nope_absorbed_bf16,
                q_rope_bf16,
                q_fp32,
                q,
                descale_q,
                qa_matmul_fp32,
                wqb_matmul_fp32,
                descale_qr,
                descale_wqb,
                rope_sin,
                rope_cos,
                q_nope_bf16,
                input_prepared=True,
                head_group=self.handoff_heads,
            )
            _mix_join()
            return
        if dsl.const_expr(self.use_hif_c2v):
            # Do not reuse the KVA Vector UB before its writes complete.
            vec_sync_block_arrive(_sync_ops.PIPE.MTE3, HEAD_READY_EVENT, mode=2)
            _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, HEAD_READY_EVENT, mode=2)
            wqb_to_prepare = _ring_buffer(
                MemLoc.UB,
                (self.wqb_tile_rows // VECTOR_SUBBLOCKS, self.Dn + self.Dr),
                dtypes.float32,
                depth=HEAD_PIPELINE_DEPTH,
                cross_core=True,
            )
            self.wqb_projection.run(
                qr,
                wqb,
                descale_qr_view,
                descale_wqb,
                wqb_matmul_fp32,
                defer_qr=(x.shape[0] <= self.hif_qr_resident_rows or use_hif_resident),
                c_to_v=wqb_to_prepare,
            )
        elif dsl.const_expr(self.use_mm34_shared):
            self.wqb_projection.run_mm34(
                qr,
                wqb,
                descale_qr_view,
                descale_wqb,
                wqb_matmul_fp32,
                q_nope_bf16,
                wkb,
                q,
                defer_qr=(
                    self.w_qr_quant_mode == 2
                    and (x.shape[0] <= self.hif_qr_resident_rows or use_hif_resident)
                ),
            )
        elif dsl.const_expr(self.w_qr_quant_mode == 2):
            self.wqb_projection.run(
                qr,
                wqb,
                descale_qr_view,
                descale_wqb,
                wqb_matmul_fp32,
                defer_qr=(x.shape[0] <= self.hif_qr_resident_rows or use_hif_resident),
            )
        else:
            self.wqb_projection.run(
                qr, wqb, descale_qr_view, descale_wqb, wqb_matmul_fp32
            )
        if dsl.const_expr(not self.use_hif_c2v and self.qkv_quant_mode in (0, 1)):
            # The remaining Cast credits are consumed per head by WKB.
            onchip_rewind()
        vec_sync_all()
        self.cache_writer.run(
            kv_norm_fp32,
            kva_matmul_fp32,
            descale_x,
            descale_wkva,
            rope_sin,
            rope_cos,
            quant_scale_kv,
            kv_cache,
            cache_index,
        )
        if dsl.const_expr(self.qkv_quant_mode in (0, 1)):
            if dsl.const_expr(self.use_hif_c2v):
                self.prepare.run_hif_c2v_pipeline(
                    wqb_to_prepare,
                    descale_qr,
                    descale_wqb,
                    rope_sin,
                    rope_cos,
                    q_nope_bf16,
                    q,
                    self.wqb_tile_rows,
                )
                onchip_rewind(reset_sync_id=False)
            else:
                self.prepare.run_pipeline(
                    wqb_matmul_fp32,
                    descale_qr,
                    descale_wqb,
                    rope_sin,
                    rope_cos,
                    q_nope_bf16,
                    q,
                    self.wqb_tile_rows,
                )
            if dsl.const_expr(not self.use_mm34_shared):
                wkb_projection = _WkbProjection(self.wkb_tile_rows)
                wkb_projection.run_direct(
                    q_nope_bf16, wkb, q, coalesce_b=self.mode0_coalesce_wkb_b
                )
            _mix_join()
            return
        _mix_join()
        onchip_rewind(reset_sync_id=True)
        wkb_projection = _WkbProjection(self.wkb_tile_rows)
        if dsl.const_expr(self.qkv_quant_mode == 3):
            wkb_projection.run_direct(
                q_nope_bf16, wkb, q_fp32, publish_heads=True, token_major_output=True
            )
        else:
            wkb_projection.run_direct(
                q_nope_bf16,
                wkb,
                q_nope_absorbed_bf16,
                publish_heads=True,
                coalesce_b=self.coalesce_wkb_b,
            )
        self.q_strategy.assemble.run(
            q_nope_absorbed_bf16,
            q_rope_bf16,
            q_fp32,
            q,
            descale_q,
            qa_matmul_fp32,
            wqb_matmul_fp32,
            descale_qr,
            descale_wqb,
            rope_sin,
            rope_cos,
            q_nope_bf16,
        )
        _mix_join()
        onchip_rewind(reset_sync_id=True)
        self.q_strategy.finish(q_fp32, q, descale_q, qa_matmul_fp32)


__all__ = [
    "quant_mla_prolog",
    "quant_mla_prolog_precheck",
    "quant_mla_prolog_native",
    "WQrQuantMode",
    "QkvQuantMode",
    "CacheMode",
    "RopeMode",
]


class WQrQuantMode(_Enum):
    MXFP8 = 1
    HIF8 = 2


class QkvQuantMode(_Enum):
    NO_QUANT = 0
    KV_FP8_PER_TOKEN_GROUP = 1
    Q_KV_FP8 = 2
    Q_KV_HIF8 = 3


class CacheMode(_Enum):
    PA_BSND = 0
    PA_NZ = 1


class RopeMode(_Enum):
    NONE = 0
    INTERLEAVE_HALF = 1
    HALF = 2


_VALID_PAIRS = {(0, 1), (0, 2), (1, 1), (2, 1), (3, 2)}
_E4 = _torch.float8_e4m3fn
_E8 = _torch.float8_e8m0fnu
_HIF_CARRIERS = (_torch.uint8, _torch.int8)
_NZ_FORMAT = 29
_TAIL_PIPELINE_ROW_CHOICES = (4, 8)
_TAIL_HEAD_ROW_CHOICES = (32, 64)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _mode_value(name, value, enum_type):
    if type(value) is enum_type:
        return value.value
    if type(value) is not int:
        raise TypeError(f"{name} must be {enum_type.__name__} or int")
    if value not in tuple(member.value for member in enum_type):
        raise ValueError(f"{name} has unsupported value {value}")
    return value


def _validate_modes(w_qr_quant_mode, qkv_quant_mode, cache_mode, rope_mode):
    for name, value, enum_type in (
        ("w_qr_quant_mode", w_qr_quant_mode, WQrQuantMode),
        ("qkv_quant_mode", qkv_quant_mode, QkvQuantMode),
        ("cache_mode", cache_mode, CacheMode),
        ("rope_mode", rope_mode, RopeMode),
    ):
        _mode_value(name, value, enum_type)
    _require(
        (qkv_quant_mode, w_qr_quant_mode) in _VALID_PAIRS,
        "unsupported qkv_quant_mode/w_qr_quant_mode combination",
    )
    _require(
        not (
            cache_mode == CacheMode.PA_NZ.value
            and qkv_quant_mode == QkvQuantMode.KV_FP8_PER_TOKEN_GROUP.value
        ),
        "qkv_quant_mode=1 does not support PA_NZ",
    )


def _output_specs(T, N, q_lora, Dr, w_qr_quant_mode, qkv_quant_mode, return_qr):
    q_dtype = (
        _torch.bfloat16
        if qkv_quant_mode in (0, 1)
        else _E4
        if qkv_quant_mode == 2
        else _torch.uint8
    )
    qr_dtype = _E4 if w_qr_quant_mode == 1 else _torch.uint8
    descale_qr_dtype = _E8 if w_qr_quant_mode == 1 else _torch.float32
    descale_q_shape = (
        (T, N, 1)
        if qkv_quant_mode == 2
        else (int(T != 0),)
        if qkv_quant_mode == 3
        else (0,)
    )
    qr_shape = (T, q_lora) if return_qr else (0,)
    descale_qr_shape = (
        ((T, q_lora // 32) if w_qr_quant_mode == 1 else (int(T != 0),))
        if return_qr
        else (0,)
    )
    return (
        ((T, N, KV_ELEMENTS + Dr), q_dtype),
        (descale_q_shape, _torch.float32),
        (qr_shape, qr_dtype),
        (descale_qr_shape, descale_qr_dtype),
    )


def _tensor(name, value, shape, dtype, device, *, weight_is_nz=False):
    if not isinstance(value, _torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    _require(
        tuple(value.shape) == tuple(shape),
        f"{name} shape must be {tuple(shape)}, got {tuple(value.shape)}",
    )
    allowed = dtype if isinstance(dtype, tuple) else (dtype,)
    if value.dtype not in allowed:
        raise TypeError(f"{name} dtype must be {allowed}, got {value.dtype}")
    _require(value.device == device, f"{name} must be on {device}")
    _require(value.is_contiguous(), f"{name} must be contiguous")
    if not weight_is_nz:
        # Torch tags ordinary contiguous rank-4 carriers NCHW(0). Both 0/2
        # are linear storage here, unlike a blocked device format such as 29.
        _require(
            _torch_npu.get_npu_format(value) in (0, 2),
            f"{name} must use linear ND storage",
        )


def _validate_scalars(norm_eps, qr_scale, kvr_scale):
    for name, value in (
        ("norm_eps", norm_eps),
        ("qr_scale", qr_scale),
        ("kvr_scale", kvr_scale),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a Python float or int, not bool or Tensor")
        try:
            fp32_value = _struct.unpack("f", _struct.pack("f", float(value)))[0]
        except (OverflowError, _struct.error) as error:
            raise ValueError(f"{name} must be finite in FP32") from error
        _require(_math.isfinite(fp32_value), f"{name} must be finite in FP32")
        if name == "norm_eps":
            _require(fp32_value > 0, f"{name} must be strictly positive in FP32")


def _validate_rope_inputs(rope_sin, rope_cos, T, Dr, rope_mode, device, *, fake=False):
    if rope_sin is None and rope_cos is None:
        _require(
            not (Dr and rope_mode), "rope_sin/rope_cos are required for active RoPE"
        )
        return
    _require(
        rope_sin is not None and rope_cos is not None,
        "rope_sin/rope_cos must be supplied together",
    )
    for name, value in (("rope_sin", rope_sin), ("rope_cos", rope_cos)):
        if fake:
            # Fake tensors expose metadata only; device format is checked at execution.
            _require(isinstance(value, _torch.Tensor), f"{name} must be a Tensor")
            _require(tuple(value.shape) == (T, Dr), f"{name} shape must be {(T, Dr)}")
            _require(value.dtype == _torch.bfloat16, f"{name} must be BF16")
            _require(value.device == device, f"{name} must be on {device}")
            _require(value.is_contiguous(), f"{name} must be contiguous")
        else:
            _tensor(name, value, (T, Dr), _torch.bfloat16, device)


def _validate_inputs(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    quant_scale_kv,
    rope_sin,
    rope_cos,
    *,
    norm_eps,
    w_qr_quant_mode,
    qkv_quant_mode,
    cache_mode,
    qr_scale,
    kvr_scale,
    rope_mode,
    return_qr,
):
    """Validate metadata without reading, copying or mutating device data."""
    _validate_modes(w_qr_quant_mode, qkv_quant_mode, cache_mode, rope_mode)
    _require(type(return_qr) is bool, "return_qr must be bool")
    _validate_scalars(norm_eps, qr_scale, kvr_scale)
    _require(isinstance(x, _torch.Tensor) and x.ndim == 2, "x shape must be (T,dim)")
    _require(x.device.type == "npu", "x must be on NPU")
    T, dim = x.shape
    _require(
        1024 <= dim <= 8192 and dim % 256 == 0, "dim must be [1024,8192], 256-aligned"
    )
    _require(
        isinstance(norm_weight_qa, _torch.Tensor) and norm_weight_qa.ndim == 1,
        "norm_weight_qa shape must be (q_lora,)",
    )
    q_lora = norm_weight_qa.shape[0]
    _require(q_lora in (640, 1024, 1536, 2048), "q_lora must be 640/1024/1536/2048")
    _require(
        isinstance(wkb, _torch.Tensor) and wkb.ndim == 3, "wkb shape must be (N,Dn,512)"
    )
    N, Dn, kv_lora = wkb.shape
    _require(1 <= N <= 128, "N must be [1,128]")
    _require(kv_lora == 512, "wkb.shape[2]/kv_lora must be 512")
    for name, weight in (("wqa", wqa), ("wqb", wqb), ("wkva", wkva)):
        _require(
            isinstance(weight, _torch.Tensor) and weight.ndim == 2,
            f"{name} Torch logical shape must use (O,K); a rank-4 ND carrier is not NZ",
        )
    Dr = wkva.shape[0] - 512
    _require(
        (Dn, Dr) in ((128, 64), (192, 64), (256, 0)),
        "(Dn,Dr) must be (128,64)/(192,64)/(256,0)",
    )
    _require(Dr != 0 or rope_mode == 0, "Dr=0 requires rope_mode=0")
    formats = tuple(_torch_npu.get_npu_format(w) for w in (wqa, wqb, wkva))
    _require(
        formats in ((2, 2, 2), (_NZ_FORMAT,) * 3),
        "wqa/wqb/wkva must be uniformly ND or true NZ",
    )
    weight_is_nz = formats[0] == _NZ_FORMAT
    device = x.device
    data_dtype = _E4 if w_qr_quant_mode == 1 else _HIF_CARRIERS
    _tensor("x", x, (T, dim), data_dtype, device)
    columns = (q_lora, N * (Dn + Dr), 512 + Dr)
    reductions = (dim, q_lora, dim)
    for name, weight, width, reduction in zip(
        ("wqa", "wqb", "wkva"), (wqa, wqb, wkva), columns, reductions
    ):
        _tensor(
            name,
            weight,
            (width, reduction),
            data_dtype,
            device,
            weight_is_nz=weight_is_nz,
        )
    _tensor("wkb", wkb, (N, Dn, 512), _torch.bfloat16, device)
    _tensor("norm_weight_qa", norm_weight_qa, (q_lora,), _torch.float32, device)
    _tensor("norm_weight_kva", norm_weight_kva, (512,), _torch.float32, device)
    descale_x_shape = (T, dim // 32) if w_qr_quant_mode == 1 else (1,)
    descale_x_dtype = _E8 if w_qr_quant_mode == 1 else _torch.float32
    _tensor("descale_x", descale_x, descale_x_shape, descale_x_dtype, device)
    for name, scale, width, reduction in zip(
        ("descale_wqa", "descale_wqb", "descale_wkva"),
        (descale_wqa, descale_wqb, descale_wkva),
        columns,
        reductions,
    ):
        shape = (1, width)
        if w_qr_quant_mode == 1:
            shape = (
                (width // 16, reduction // 64, 16, 2)
                if weight_is_nz
                else (width, reduction // 64, 2)
            )
        _tensor(name, scale, shape, descale_x_dtype, device)
    _require(
        isinstance(kv_cache, _torch.Tensor) and kv_cache.ndim == 4,
        "kv_cache shape must be (block_number,block_size,1,Dtile)",
    )
    block_number, block_size, cache_heads, dtile = kv_cache.shape
    _require(block_number > 0, "block_number must be positive")
    _require(
        16 <= block_size <= 1024 and block_size % 16 == 0,
        "block_size must be [16,1024], 16-aligned",
    )
    expected_dtile = 512 + 2 * Dr + 16 if qkv_quant_mode == 1 else 512 + Dr
    cache_dtype = _torch.bfloat16 if qkv_quant_mode == 0 else _E4
    if qkv_quant_mode == 3:
        cache_dtype = _HIF_CARRIERS
    _tensor(
        "kv_cache",
        kv_cache,
        (block_number, block_size, 1, expected_dtile),
        cache_dtype,
        device,
    )
    _require(
        cache_heads == 1 and dtile == expected_dtile, "kv_cache Dtile/heads mismatch"
    )
    _tensor("cache_index", cache_index, (T,), _torch.int64, device)
    if qkv_quant_mode in (2, 3):
        _tensor("quant_scale_kv", quant_scale_kv, (1,), _torch.float32, device)
    else:
        _require(
            quant_scale_kv is None, "quant_scale_kv must be None in qkv_quant_mode=0/1"
        )
    _validate_rope_inputs(rope_sin, rope_cos, T, Dr, rope_mode, device)
    return T, dim, q_lora, N, Dn, Dr, weight_is_nz


class _QuantMlaPrologProgram:
    def __init__(
        self,
        q_lora,
        N,
        Dn,
        Dr,
        w_qr_quant_mode,
        qkv_quant_mode,
        weight_is_nz,
        cache_mode,
        rope_mode,
        input_tile_rows,
        wqb_tile_rows,
        absorb_tile_rows,
        qa_rows=0,
        qa_dim=0,
        qa_config=(0, 1, 128),
        compact_head_handoff=(0, 0),
    ):
        self.op = _QuantMlaPrologKernel(
            q_lora,
            N,
            Dn,
            Dr,
            w_qr_quant_mode,
            qkv_quant_mode,
            weight_is_nz,
            cache_mode,
            rope_mode,
            input_tile_rows,
            wqb_tile_rows,
            absorb_tile_rows,
            qa_rows,
            qa_dim,
            qa_config,
            compact_head_handoff,
        )

    @_dsl.host
    def run(
        self,
        x: _Tensor,
        wqa: _Tensor,
        wqb: _Tensor,
        wkva: _Tensor,
        wkb: _Tensor,
        descale_x: _Tensor,
        descale_wqa: _Tensor,
        descale_wqb: _Tensor,
        descale_wkva: _Tensor,
        norm_weight_qa: _Tensor,
        norm_weight_kva: _Tensor,
        kv_cache: _Tensor,
        cache_index: _Tensor,
        quant_scale_kv: _Tensor,
        rope_sin: _Tensor,
        rope_cos: _Tensor,
        qa_matmul_fp32: _Tensor,
        kva_matmul_fp32: _Tensor,
        qa_norm_fp32: _Tensor,
        kv_norm_fp32: _Tensor,
        qr: _Tensor,
        descale_qr: _Tensor,
        wqb_matmul_fp32: _Tensor,
        q_nope_bf16: _Tensor,
        q_rope_bf16: _Tensor,
        q_nope_absorbed_bf16: _Tensor,
        q_fp32: _Tensor,
        q: _Tensor,
        descale_q: _Tensor,
        norm_eps: float,
        qr_scale: float,
        kvr_scale: float,
        qa_partial: _Tensor,
        core_num: int,
    ):
        self.op[core_num](
            x,
            wqa,
            wqb,
            wkva,
            wkb,
            descale_x,
            descale_wqa,
            descale_wqb,
            descale_wkva,
            norm_weight_qa,
            norm_weight_kva,
            kv_cache,
            cache_index,
            quant_scale_kv,
            rope_sin,
            rope_cos,
            qa_matmul_fp32,
            kva_matmul_fp32,
            qa_norm_fp32,
            kv_norm_fp32,
            qr,
            descale_qr,
            wqb_matmul_fp32,
            q_nope_bf16,
            q_rope_bf16,
            q_nope_absorbed_bf16,
            q_fp32,
            q,
            descale_q,
            norm_eps,
            qr_scale,
            kvr_scale,
            qa_partial,
        )


def _build_fused_program(
    dim,
    q_lora,
    N,
    Dn,
    Dr,
    w_qr_quant_mode,
    qkv_quant_mode,
    weight_is_nz,
    cache_mode,
    rope_mode,
    input_tile_rows,
    wqb_tile_rows,
    absorb_tile_rows,
    single_token=False,
    qa_rows=0,
    qa_config=(0, 1, 128),
    compact_head_handoff=(0, 0),
):
    T = 1 if single_token else _dsl.Dim("tokens", min=1)
    block_number = _dsl.Dim("cache_blocks", min=1)
    block_size = _dsl.Dim("cache_block_size", min=16, max=1024)
    reduction = (
        dim
        if weight_is_nz or qa_rows
        else _dsl.Dim("dim", min=1024, max=8192, multiple_of=256)
    )
    data_dtype = _dtypes.float8_e4m3fn if w_qr_quant_mode == 1 else _dtypes.uint8
    q_dtype = _dtypes.bfloat16 if qkv_quant_mode in (0, 1) else data_dtype
    x_spec = _dsl.TensorSpec((T, reduction), data_dtype)
    weight_specs = tuple(
        _dsl.TensorSpec(
            shape, data_dtype, storage_format="nz" if weight_is_nz else "nd"
        )
        for shape in (
            (q_lora, reduction),
            (N * (Dn + Dr), q_lora),
            (KV_ELEMENTS + Dr, reduction),
        )
    )
    wqa_spec, wqb_spec, wkva_spec = weight_specs
    wkb_spec = _dsl.TensorSpec((N, Dn, KV_ELEMENTS), _dtypes.bfloat16)
    descale_x_spec = _dsl.TensorSpec((T, reduction // 64, 2), _dtypes.float8_e8m0)
    if w_qr_quant_mode == 2:
        descale_x_spec = _dsl.TensorSpec((1,), _dtypes.float32)
    descale_weight_specs = []
    for outputs, k in (
        (q_lora, reduction),
        (N * (Dn + Dr), q_lora),
        (KV_ELEMENTS + Dr, reduction),
    ):
        shape = (1, outputs)
        dtype = _dtypes.float32
        if w_qr_quant_mode == 1:
            shape = (
                (outputs // 16, k // 64, 16, 2)
                if weight_is_nz
                else (outputs, k // 64, 2)
            )
            dtype = _dtypes.uint8 if weight_is_nz else _dtypes.float8_e8m0
        descale_weight_specs.append(_dsl.TensorSpec(shape, dtype))
    descale_wqa_spec, descale_wqb_spec, descale_wkva_spec = descale_weight_specs
    norm_weight_qa_spec = _dsl.TensorSpec((1, q_lora), _dtypes.float32)
    norm_weight_kva_spec = _dsl.TensorSpec((1, KV_ELEMENTS), _dtypes.float32)
    payload_bytes = (
        2 * (KV_ELEMENTS + Dr)
        if qkv_quant_mode == 0
        else KV_ELEMENTS + 2 * Dr + 16
        if qkv_quant_mode == 1
        else KV_ELEMENTS + Dr
    )
    kv_cache_bytes_spec = _dsl.TensorSpec(
        (block_number, block_size, 1, payload_bytes), _dtypes.uint8
    )
    cache_index_spec = _dsl.TensorSpec((T,), _dtypes.int64)
    qa_matmul_fp32_spec = _dsl.TensorSpec((T, q_lora), _dtypes.float32)
    qa_norm_fp32_spec = qa_matmul_fp32_spec
    kva_matmul_fp32_spec = _dsl.TensorSpec((T, KV_ELEMENTS + Dr), _dtypes.float32)
    kv_norm_fp32_spec = _dsl.TensorSpec((T, KV_ELEMENTS), _dtypes.float32)
    quant_scale_kv_spec = (
        _dsl.TensorSpec((1,), _dtypes.float32)
        if qkv_quant_mode in (2, 3)
        else kv_norm_fp32_spec
    )
    qr_spec = _dsl.TensorSpec((T, q_lora), data_dtype)
    descale_qr_spec = (
        _dsl.TensorSpec((T, q_lora // 32), _dtypes.float8_e8m0)
        if w_qr_quant_mode == 1
        else _dsl.TensorSpec((1,), _dtypes.float32)
    )
    wqb_matmul_fp32_spec = _dsl.TensorSpec((T, N * (Dn + Dr)), _dtypes.float32)
    q_nope_bf16_spec = _dsl.TensorSpec((N * T, Dn), _dtypes.bfloat16)
    q_rope_bf16_spec = (
        _dsl.TensorSpec((N * T, Dr), _dtypes.bfloat16) if Dr else q_nope_bf16_spec
    )
    rope_sin_spec = (
        _dsl.TensorSpec((T, Dr), _dtypes.bfloat16)
        if Dr and rope_mode
        else q_nope_bf16_spec
    )
    rope_cos_spec = rope_sin_spec
    q_nope_absorbed_bf16_spec = _dsl.TensorSpec((N * T, KV_ELEMENTS), _dtypes.bfloat16)
    q_fp32_spec = _dsl.TensorSpec(
        (N * T, KV_ELEMENTS + Dr),
        _dtypes.bfloat16 if qkv_quant_mode == 3 else _dtypes.float32,
    )
    q_spec = _dsl.TensorSpec((N * T, KV_ELEMENTS + Dr), q_dtype)
    descale_q_spec = (
        _dsl.TensorSpec((N * T, 1), _dtypes.float32)
        if qkv_quant_mode == 2
        else _dsl.TensorSpec((1,), _dtypes.float32)
        if qkv_quant_mode == 3
        else q_fp32_spec
    )
    entry = _QuantMlaPrologProgram(
        q_lora,
        N,
        Dn,
        Dr,
        w_qr_quant_mode,
        qkv_quant_mode,
        weight_is_nz,
        cache_mode,
        rope_mode,
        input_tile_rows,
        wqb_tile_rows,
        absorb_tile_rows,
        qa_rows,
        dim,
        qa_config,
        compact_head_handoff,
    )
    bm = qa_config[0] if qa_rows else input_tile_rows
    splits = qa_config[1]
    nm, nc = (qa_rows + bm - 1) // bm, q_lora // 128
    qa_partial_spec = (
        _dsl.TensorSpec((nm * nc * splits, bm * 128), _dtypes.float32)
        if splits == 2
        else qa_matmul_fp32_spec
    )
    return _dsl.compile(
        entry.run,
        x_spec,
        wqa_spec,
        wqb_spec,
        wkva_spec,
        wkb_spec,
        descale_x_spec,
        descale_wqa_spec,
        descale_wqb_spec,
        descale_wkva_spec,
        norm_weight_qa_spec,
        norm_weight_kva_spec,
        kv_cache_bytes_spec,
        cache_index_spec,
        quant_scale_kv_spec,
        rope_sin_spec,
        rope_cos_spec,
        qa_matmul_fp32_spec,
        kva_matmul_fp32_spec,
        qa_norm_fp32_spec,
        kv_norm_fp32_spec,
        qr_spec,
        descale_qr_spec,
        wqb_matmul_fp32_spec,
        q_nope_bf16_spec,
        q_rope_bf16_spec,
        q_nope_absorbed_bf16_spec,
        q_fp32_spec,
        q_spec,
        descale_q_spec,
        _dtypes.float32,
        _dtypes.float32,
        _dtypes.float32,
        qa_partial_spec,
        _dtypes.int64,
    )


@_lru_cache(maxsize=256)
def _program(
    dim,
    q_lora,
    N,
    Dn,
    Dr,
    w_qr_quant_mode,
    qkv_quant_mode,
    weight_is_nz,
    cache_mode,
    rope_mode,
    input_tile_rows,
    wqb_tile_rows,
    absorb_tile_rows,
    single_token=False,
    qa_rows=0,
    qa_config=(0, 1, 128),
    compact_head_handoff=(0, 0),
):
    return _build_fused_program(
        dim,
        q_lora,
        N,
        Dn,
        Dr,
        w_qr_quant_mode,
        qkv_quant_mode,
        weight_is_nz,
        cache_mode,
        rope_mode,
        input_tile_rows,
        wqb_tile_rows,
        absorb_tile_rows,
        single_token,
        qa_rows,
        qa_config,
        compact_head_handoff,
    )


def _select_qa_config(rows, width, reduction, core_count, mode, weight_is_nz):
    column_tiles = width // QA_PIPELINE_COLUMNS
    if (
        rows <= 128
        and not (reduction == 1024 and rows <= 16)
        and column_tiles * 2 <= core_count
    ):
        return (((rows + 15) // 16) * 16, 2, QA_PIPELINE_COLUMNS)
    row_owners = max(1, core_count // column_tiles)
    balanced_rows = (rows + row_owners - 1) // row_owners
    return (min(128, ((balanced_rows + 15) // 16) * 16), 1, QA_PIPELINE_COLUMNS)


def _select_small_head_handoff(
    rows,
    heads,
    nope_width,
    rope_width,
    reduction,
    core_count,
    reduction_panel=REDUCTION_PANEL,
    max_slots=HEAD_PIPELINE_DEPTH,
):
    head_width = nope_width + rope_width
    max_group = min(heads, STRIPE_COLUMNS // head_width)
    aligned_rows = ((rows + PROJECTION_ROWS - 1) // PROJECTION_ROWS) * PROJECTION_ROWS
    best = None
    choice = (0, 0, 0)
    for tile_rows in range(PROJECTION_ROWS, aligned_rows + 1, PROJECTION_ROWS):
        row_tiles = (rows + tile_rows - 1) // tile_rows
        for group in range(1, max_group + 1):
            if heads % group:
                continue
            columns = group * head_width
            tasks = row_tiles * (heads // group)
            waves = (tasks + core_count - 1) // core_count
            if waves > max_slots:
                continue
            weight_columns = max(
                columns, nope_width * WKB_C2V_COLUMNS * BF16_BYTES // reduction_panel
            )
            l1_bytes = (
                2 * weight_columns * reduction_panel
                + 2 * tile_rows * reduction_panel
                + (tile_rows + columns) * reduction // MX_GROUP
                + waves * tile_rows * group * nope_width * BF16_BYTES
                + 2 * tile_rows * WKB_C2V_REDUCTION * BF16_BYTES
            )
            l0a_bytes = max(
                2 * tile_rows * reduction_panel,
                2 * tile_rows * WKB_C2V_REDUCTION * BF16_BYTES,
            )
            l0b_bytes = 2 * COLUMNS_PER_PART * REDUCTION_PANEL
            l0c_bytes = 2 * tile_rows * max(columns, WKB_C2V_COLUMNS) * FLOAT_BYTES
            vector_rows = tile_rows // VECTOR_SUBBLOCKS
            ub_bytes = (
                HEAD_PIPELINE_DEPTH * vector_rows * columns * FLOAT_BYTES
                + vector_rows * group * head_width * BF16_BYTES
                + 2 * vector_rows * rope_width * BF16_BYTES
            )
            if (
                l1_bytes > _dsl.get_mem_size("l1")
                or l0a_bytes > _dsl.get_mem_size("l0a")
                or l0b_bytes > _dsl.get_mem_size("l0b")
                or l0c_bytes > _dsl.get_mem_size("l0c")
                or ub_bytes > _dsl.get_mem_size("ub")
            ):
                continue
            projection_bytes = (tile_rows + columns) * reduction
            scale_bytes = (tile_rows + columns) * reduction // MX_GROUP
            absorb_weight_bytes = group * nope_width * KV_ELEMENTS * BF16_BYTES
            task_bytes = projection_bytes + scale_bytes + absorb_weight_bytes
            rank = (
                waves * task_bytes,
                tasks * task_bytes,
                waves * tile_rows * group,
                -tile_rows,
                group,
            )
            if best is None or rank < best:
                best = rank
                choice = (tile_rows, group, waves)
    return choice


def _program_parameters(
    T,
    dim,
    q_lora,
    N,
    Dn,
    Dr,
    w_qr_quant_mode,
    qkv_quant_mode,
    weight_is_nz,
    cache_mode,
    rope_mode,
    selected_core_num,
):
    input_tile_rows = (
        128 if T >= 128 and T % 128 == 0 else 32 if T >= 32 and T % 32 == 0 else 16
    )
    wqb_tile_rows = (
        128 if T >= 128 and T % 128 == 0 else 32 if T >= 32 and T % 32 == 0 else 16
    )
    absorb_tile_rows = (
        128 if T >= 128 and T % 128 == 0 else 32 if T >= 32 and T % 32 == 0 else 16
    )
    tail_head_min_rows = _TAIL_HEAD_ROW_CHOICES[0]
    if (
        w_qr_quant_mode == 1
        and qkv_quant_mode in (0, 1)
        and T > tail_head_min_rows
        and T % tail_head_min_rows
    ):
        for rows_per_tile in _TAIL_HEAD_ROW_CHOICES:
            if (
                N * ((T + rows_per_tile - 1) // rows_per_tile)
                >= HEAD_PIPELINE_DEPTH * selected_core_num
            ):
                wqb_tile_rows = rows_per_tile
                absorb_tile_rows = rows_per_tile
    if Dn + Dr == 256 and selected_core_num >= 8:
        for tail_rows in _TAIL_PIPELINE_ROW_CHOICES:
            if (
                T > tail_rows
                and N * ((T + tail_rows - 1) // tail_rows) <= selected_core_num
            ):
                wqb_tile_rows = tail_rows
                absorb_tile_rows = tail_rows
                break
    if w_qr_quant_mode == 1 and qkv_quant_mode in (0, 1):
        wqb_tile_rows = min(wqb_tile_rows, WKB_C2V_MAX_ROWS)
        absorb_tile_rows = wqb_tile_rows
    if (
        w_qr_quant_mode == 1
        and qkv_quant_mode in (0, 1)
        and T == WKB_DIRECT_MAX_ROWS
        and N >= selected_core_num
    ):
        wqb_tile_rows = WKB_DIRECT_MAX_ROWS
        absorb_tile_rows = WKB_DIRECT_MAX_ROWS
    if (
        w_qr_quant_mode == 1
        and qkv_quant_mode in (0, 1)
        and T in (WKB_C2V_MAX_ROWS, WKB_C2V_MAX_ROWS + 1)
        and N >= selected_core_num
    ):
        wqb_tile_rows = ((T + PROJECTION_ROWS - 1) // PROJECTION_ROWS) * PROJECTION_ROWS
        absorb_tile_rows = wqb_tile_rows
    if (
        w_qr_quant_mode == 1
        and qkv_quant_mode in (0, 1)
        and T in (WQB_LARGE_HEAD_ROWS, WQB_LARGE_HEAD_ROWS + 1)
        and N >= selected_core_num
    ):
        wqb_tile_rows = WQB_LARGE_HEAD_ROWS
        absorb_tile_rows = WQB_LARGE_HEAD_ROWS
    if T == 33 and w_qr_quant_mode == 1 and qkv_quant_mode in (0, 1):
        if N == 16:
            wqb_tile_rows = 32
            absorb_tile_rows = 32
        elif 24 <= N < 64:
            wqb_tile_rows = 48
            absorb_tile_rows = 48
    if (
        w_qr_quant_mode == 1
        and qkv_quant_mode in (0, 1)
        and N < selected_core_num
        and N * 4 >= selected_core_num * 3
        and T in (WKB_C2V_MAX_ROWS, WQB_LARGE_HEAD_ROWS, WQB_LARGE_HEAD_ROWS + 1)
    ):
        wqb_tile_rows = (
            WKB_C2V_MAX_ROWS if T == WKB_C2V_MAX_ROWS else WQB_LARGE_HEAD_ROWS
        )
        absorb_tile_rows = wqb_tile_rows
    if (
        w_qr_quant_mode == 1
        and qkv_quant_mode in (0, 1)
        and T == WQB_LARGE_HEAD_ROWS + 1
        and Dn + Dr == 192
        and wqb_tile_rows == WQB_LARGE_HEAD_ROWS
    ):
        wqb_tile_rows = WQB_SINGLE_TAIL_ROWS
        absorb_tile_rows = WQB_SINGLE_TAIL_ROWS
    if (
        w_qr_quant_mode == 1
        and qkv_quant_mode in (0, 1)
        and T == WQB_LARGE_HEAD_ROWS + 1
        and Dn + Dr == 256
        and wqb_tile_rows == WQB_LARGE_HEAD_ROWS
    ):
        wqb_tile_rows = WQB_TAIL_HEAD_ROWS
        absorb_tile_rows = WQB_TAIL_HEAD_ROWS
    if (
        w_qr_quant_mode == 1
        and qkv_quant_mode in (0, 1)
        and T == 6 * PROJECTION_ROWS
        and N * 2 == selected_core_num
        and Dn + Dr == WKB_C2V_COLUMNS
    ):
        wqb_tile_rows = T // 2
        absorb_tile_rows = wqb_tile_rows
    qa_config = _select_qa_config(
        T, q_lora, dim, selected_core_num, w_qr_quant_mode, weight_is_nz
    )
    if (
        w_qr_quant_mode == 1
        and qkv_quant_mode == 2
        and WQB_WIDE_MAX_ROWS < T <= WQB_LARGE_HEAD_ROWS
        and N * (Dn + Dr) >= selected_core_num * MM34_SHARED_COLUMNS
    ):
        wqb_tile_rows = ((T + PROJECTION_ROWS - 1) // PROJECTION_ROWS) * PROJECTION_ROWS
    compact_head_handoff = (0, 0)
    if (
        w_qr_quant_mode == 1
        and qkv_quant_mode in (0, 1, 2)
        and 0 < T <= WQB_WIDE_MAX_ROWS
        and 0 < N <= 128
        and (Dn, Dr) in ((128, 64), (192, 64), (256, 0))
        and (rope_mode == 1 or Dr == 0)
        and q_lora % WQB_PROJECTION_REDUCTION == 0
    ):
        compact_rows, group, slots = _select_small_head_handoff(
            T, N, Dn, Dr, q_lora, selected_core_num
        )
        if group:
            compact_head_handoff = (group, slots)
            wqb_tile_rows = compact_rows
            absorb_tile_rows = compact_rows
    if (
        w_qr_quant_mode == 1
        and qkv_quant_mode == 2
        and WQB_WIDE_MAX_ROWS < T <= COMPACT_HANDOFF_MAX_ROWS
        and 0 < N <= 128
        and (Dn, Dr) in ((128, 64), (192, 64))
        and (rope_mode == 1 or Dr == 0)
        and q_lora % WQB_PROJECTION_REDUCTION == 0
    ):
        compact_rows, group, slots = _select_small_head_handoff(
            T,
            N,
            Dn,
            Dr,
            q_lora,
            selected_core_num,
            COMPACT_BOUNDARY_REDUCTION_PANEL,
            COMPACT_HANDOFF_MAX_SLOTS,
        )
        if group:
            compact_head_handoff = (group, slots)
            wqb_tile_rows = compact_rows
            absorb_tile_rows = compact_rows
    return (
        dim,
        q_lora,
        N,
        Dn,
        Dr,
        w_qr_quant_mode,
        qkv_quant_mode,
        weight_is_nz,
        cache_mode,
        rope_mode,
        input_tile_rows,
        wqb_tile_rows,
        absorb_tile_rows,
        T == 1 and w_qr_quant_mode == 2,
        T,
        qa_config,
        compact_head_handoff,
    )


def _cores(device, core_num=None):
    properties = _torch.npu.get_device_properties(device)
    cube = int(properties.cube_core_num)
    vector = int(properties.vector_core_num)
    get_limit = getattr(_torch.npu, "get_stream_limit", None)
    if not callable(get_limit):
        raise RuntimeError(
            "current Stream quota query requires torch.npu.get_stream_limit"
        )
    try:
        limits = get_limit(_torch.npu.current_stream(device))
        cube_limit, vector_limit = limits["cube_core_num"], limits["vector_core_num"]
        if any(
            type(value) is not int or value < 0 for value in (cube_limit, vector_limit)
        ):
            raise ValueError("Stream quotas must be nonnegative integers")
    except Exception as error:
        raise RuntimeError(
            "failed to query current Stream quota; refusing full-device fallback"
        ) from error
    cube = min(cube, cube_limit)
    vector = min(vector, vector_limit)
    available_core_num = min(cube, vector // 2)
    _require(
        available_core_num >= 1, "current stream needs at least one AIC and two AIV"
    )
    if core_num is not None:
        if type(core_num) is not int:
            raise TypeError("core_num must be an integer or None")
        _require(
            1 <= core_num <= available_core_num,
            f"core_num must be in [1,{available_core_num}] for the current stream",
        )
        return core_num
    return available_core_num


def _validate_quant_scale(quant_scale_kv):
    if _torch.npu.is_current_stream_capturing():
        raise RuntimeError(
            "quant_mla_prolog_precheck cannot read quant_scale_kv during capture; "
            "precheck quant_scale_kv outside capture"
        )
    value = float(quant_scale_kv.detach().cpu().item())
    _require(
        _math.isfinite(value) and value >= 0,
        "quant_scale_kv must be finite and nonnegative",
    )


def quant_mla_prolog_precheck(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    quant_scale_kv=None,
    rope_sin=None,
    rope_cos=None,
    *,
    norm_eps=1e-5,
    w_qr_quant_mode,
    qkv_quant_mode,
    cache_mode,
    qr_scale=1.0,
    kvr_scale=1.0,
    rope_mode=1,
    return_qr=False,
):
    """Validate inputs explicitly before execution; do not launch or update Cache.

    Callers guarantee valid cache_index values; only its metadata is checked.
    Nonempty quant_scale_kv checks synchronize with the current stream. Repeat
    this precheck after changing quant_scale_kv contents.
    """
    if type(return_qr) is not bool:
        raise TypeError("return_qr must be bool")
    _validate_scalars(norm_eps, qr_scale, kvr_scale)
    w_qr_quant_mode = _mode_value("w_qr_quant_mode", w_qr_quant_mode, WQrQuantMode)
    qkv_quant_mode = _mode_value("qkv_quant_mode", qkv_quant_mode, QkvQuantMode)
    cache_mode = _mode_value("cache_mode", cache_mode, CacheMode)
    rope_mode = _mode_value("rope_mode", rope_mode, RopeMode)
    values = (
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        quant_scale_kv,
        rope_sin,
        rope_cos,
    )
    attributes = dict(
        norm_eps=norm_eps,
        w_qr_quant_mode=w_qr_quant_mode,
        qkv_quant_mode=qkv_quant_mode,
        cache_mode=cache_mode,
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=rope_mode,
        return_qr=return_qr,
    )
    _validate_inputs(*values, **attributes)
    if x.shape[0] != 0 and quant_scale_kv is not None:
        _validate_quant_scale(quant_scale_kv)


def _execute_device(*values, **attributes):
    # The shared execution path validates metadata without device-content reads.
    return _execute_async(*values, **attributes)


def _execute_async(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    quant_scale_kv=None,
    rope_sin=None,
    rope_cos=None,
    *,
    norm_eps=1e-5,
    w_qr_quant_mode,
    qkv_quant_mode,
    cache_mode,
    qr_scale=1.0,
    kvr_scale=1.0,
    rope_mode=1,
    return_qr=False,
    core_num=None,
):
    T, dim, q_lora, N, Dn, Dr, weight_is_nz = _validate_inputs(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        quant_scale_kv,
        rope_sin,
        rope_cos,
        norm_eps=norm_eps,
        w_qr_quant_mode=w_qr_quant_mode,
        qkv_quant_mode=qkv_quant_mode,
        cache_mode=cache_mode,
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=rope_mode,
        return_qr=return_qr,
    )

    def empty(shape, dtype):
        return _torch.empty(shape, dtype=dtype, device=x.device)

    output_specs = _output_specs(
        T, N, q_lora, Dr, w_qr_quant_mode, qkv_quant_mode, return_qr
    )
    q_dtype, qr_dtype, descale_qr_dtype = (output_specs[slot][1] for slot in (0, 2, 3))
    if T == 0:
        return tuple(empty(shape, dtype) for shape, dtype in output_specs)
    selected_core_num = _cores(x.device, core_num)
    descale_x_view = (
        descale_x.view(T, dim // 64, 2) if w_qr_quant_mode == 1 else descale_x
    )
    x_view = x if w_qr_quant_mode == 1 else x.view(_torch.uint8)
    wqa_view, wqb_view, wkva_view = (
        (wqa, wqb, wkva)
        if w_qr_quant_mode == 1
        else tuple(weight.view(_torch.uint8) for weight in (wqa, wqb, wkva))
    )
    qa_matmul_fp32 = empty((T, q_lora), _torch.float32)
    kva_matmul_fp32 = empty((T, 512 + Dr), _torch.float32)
    qa_norm_fp32 = empty((T, q_lora), _torch.float32)
    kv_norm_fp32 = empty((T, 512), _torch.float32)
    qr = empty((T, q_lora), qr_dtype)
    descale_qr = empty(
        (T, q_lora // 32) if w_qr_quant_mode == 1 else (1,), descale_qr_dtype
    )
    wqb_matmul_fp32 = empty((T, N * (Dn + Dr)), _torch.float32)
    q_nope_bf16 = empty((N * T, Dn), _torch.bfloat16)
    q_rope_bf16 = empty((N * T, Dr), _torch.bfloat16) if Dr else q_nope_bf16
    q_nope_absorbed_bf16 = empty((N * T, 512), _torch.bfloat16)
    q_fp32 = empty(
        (N * T, 512 + Dr), _torch.bfloat16 if qkv_quant_mode == 3 else _torch.float32
    )
    q = empty((N * T, 512 + Dr), q_dtype)
    descale_q = empty(
        (N * T, 1) if qkv_quant_mode == 2 else (1,) if qkv_quant_mode == 3 else (0,),
        _torch.float32,
    )
    norm_weight_qa_view = norm_weight_qa.view(1, q_lora)
    norm_weight_kva_view = norm_weight_kva.view(1, 512)
    kv_cache_bytes = kv_cache.view(_torch.uint8)
    parameters = _program_parameters(
        T,
        dim,
        q_lora,
        N,
        Dn,
        Dr,
        w_qr_quant_mode,
        qkv_quant_mode,
        weight_is_nz,
        cache_mode,
        rope_mode,
        selected_core_num,
    )
    qa_bm, qa_splits, qa_bn = parameters[-2]
    qa_tiles = ((T + qa_bm - 1) // qa_bm) * (q_lora // qa_bn)
    qa_partial = (
        empty((qa_tiles * qa_splits, qa_bm * qa_bn), _torch.float32)
        if qa_splits == 2
        else qa_matmul_fp32
    )
    if w_qr_quant_mode == WQrQuantMode.MXFP8.value and weight_is_nz:
        descale_wqa, descale_wqb, descale_wkva = (
            scale.view(_torch.uint8)
            for scale in (descale_wqa, descale_wqb, descale_wkva)
        )
    program = _program(*parameters)
    program(
        x_view,
        wqa_view,
        wqb_view,
        wkva_view,
        wkb,
        descale_x_view,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa_view,
        norm_weight_kva_view,
        kv_cache_bytes,
        cache_index,
        quant_scale_kv if qkv_quant_mode in (2, 3) else kv_norm_fp32,
        rope_sin if Dr and rope_mode else q_nope_bf16,
        rope_cos if Dr and rope_mode else q_nope_bf16,
        qa_matmul_fp32,
        kva_matmul_fp32,
        qa_norm_fp32,
        kv_norm_fp32,
        qr,
        descale_qr,
        wqb_matmul_fp32,
        q_nope_bf16,
        q_rope_bf16,
        q_nope_absorbed_bf16,
        q_fp32,
        q,
        descale_q if qkv_quant_mode in (2, 3) else q_fp32,
        norm_eps,
        qr_scale,
        kvr_scale,
        qa_partial,
        selected_core_num,
    )
    if qkv_quant_mode == 2:
        descale_q = descale_q.view(T, N, 1)
    return (
        q.view(output_specs[0][0]),
        descale_q,
        qr if return_qr else empty(*output_specs[2]),
        descale_qr if return_qr else empty(*output_specs[3]),
    )


# Retain the registration owner across module reloads.
_LIBRARY = globals().get("_LIBRARY")
_TORCH_SCHEMA = (
    "quant_mla_prolog(Tensor x, Tensor wqa, Tensor wqb, Tensor wkva, Tensor wkb, "
    "Tensor descale_x, Tensor descale_wqa, Tensor descale_wqb, Tensor descale_wkva, "
    "Tensor norm_weight_qa, Tensor norm_weight_kva, Tensor(a!) kv_cache, Tensor cache_index, "
    "Tensor? quant_scale_kv=None, Tensor? rope_sin=None, Tensor? rope_cos=None, *, "
    "float norm_eps=1e-5, int w_qr_quant_mode, int qkv_quant_mode, int cache_mode, "
    "float qr_scale=1., float kvr_scale=1., int rope_mode=1, bool return_qr=False) "
    "-> (Tensor, Tensor, Tensor, Tensor)"
)


def _fake(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    quant_scale_kv=None,
    rope_sin=None,
    rope_cos=None,
    *,
    norm_eps=1e-5,
    w_qr_quant_mode,
    qkv_quant_mode,
    cache_mode,
    qr_scale=1.0,
    kvr_scale=1.0,
    rope_mode=1,
    return_qr=False,
):
    _validate_modes(w_qr_quant_mode, qkv_quant_mode, cache_mode, rope_mode)
    _require(type(return_qr) is bool, "return_qr must be bool")
    _validate_scalars(norm_eps, qr_scale, kvr_scale)
    T = x.shape[0]
    N = wkb.shape[0]
    Dr = wkva.shape[0] - 512
    q_lora = norm_weight_qa.shape[0]
    _require(Dr != 0 or rope_mode == RopeMode.NONE.value, "Dr=0 requires rope_mode=0")
    _validate_rope_inputs(rope_sin, rope_cos, T, Dr, rope_mode, x.device, fake=True)
    return tuple(
        x.new_empty(shape, dtype=dtype)
        for shape, dtype in _output_specs(
            T, N, q_lora, Dr, w_qr_quant_mode, qkv_quant_mode, return_qr
        )
    )


def _reject_functionalization(*args, **kwargs):
    raise RuntimeError(
        "quant_mla_prolog requires a backend preserving the mutable cache op; "
        "functionalization with cache clone/copy-back is disabled"
    )


def _execute_graph(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    quant_scale_kv=None,
    rope_sin=None,
    rope_cos=None,
    *,
    norm_eps=1e-5,
    w_qr_quant_mode,
    qkv_quant_mode,
    cache_mode,
    qr_scale=1.0,
    kvr_scale=1.0,
    rope_mode=1,
    return_qr=False,
    core_num=None,
):
    # E8M0 blocks auto_functionalized_v2 decomposition in Torch 2.12. Byte views
    # are zero-copy; the shared program still receives the original E8M0 dtype.
    if w_qr_quant_mode == WQrQuantMode.MXFP8.value:
        descale_x, descale_wqa, descale_wqb, descale_wkva = (
            value.view(_E8)
            for value in (descale_x, descale_wqa, descale_wqb, descale_wkva)
        )
    q, descale_q, qr, descale_qr = _execute_async(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        quant_scale_kv,
        rope_sin,
        rope_cos,
        norm_eps=norm_eps,
        w_qr_quant_mode=w_qr_quant_mode,
        qkv_quant_mode=qkv_quant_mode,
        cache_mode=cache_mode,
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=rope_mode,
        return_qr=return_qr,
        core_num=core_num,
    )
    if w_qr_quant_mode == WQrQuantMode.MXFP8.value:
        descale_qr = descale_qr.view(_torch.uint8)
    return q, descale_q, qr, descale_qr


def _graph_fake(*args, **attrs):
    q, descale_q, qr, descale_qr = _fake(*args, **attrs)
    if attrs["w_qr_quant_mode"] == WQrQuantMode.MXFP8.value:
        descale_qr = descale_qr.view(_torch.uint8)
    return q, descale_q, qr, descale_qr


def _register_torch():
    global _LIBRARY
    registrations = (
        ("quant_mla_prolog", _TORCH_SCHEMA, _execute_device, _fake),
        (
            "_quant_mla_prolog_graph",
            _TORCH_SCHEMA.replace("quant_mla_prolog(", "_quant_mla_prolog_graph(", 1),
            _execute_graph,
            _graph_fake,
        ),
    )
    existing = getattr(_torch.ops.cannbotdsl_quant_mla_prolog, "quant_mla_prolog", None)
    if existing is not None:
        for name, schema, _, _ in registrations:
            registered = getattr(_torch.ops.cannbotdsl_quant_mla_prolog, name, None)
            expected = _torch._C.parse_schema("cannbotdsl_quant_mla_prolog::" + schema)
            if registered is None or str(registered.default._schema) != str(expected):
                raise RuntimeError(
                    "quant_mla_prolog is already registered with another schema"
                )
        return
    _LIBRARY = _torch.library.Library("cannbotdsl_quant_mla_prolog", "FRAGMENT")
    for name, schema, implementation, fake in registrations:
        _LIBRARY.define(schema)
        _LIBRARY.impl(name, implementation, "PrivateUse1")
        _torch.library.register_fake("cannbotdsl_quant_mla_prolog::" + name)(fake)
    _LIBRARY.impl("quant_mla_prolog", _reject_functionalization, "Functionalize")


_register_torch()


def quant_mla_prolog(
    x: _torch.Tensor,
    wqa: _torch.Tensor,
    wqb: _torch.Tensor,
    wkva: _torch.Tensor,
    wkb: _torch.Tensor,
    descale_x: _torch.Tensor,
    descale_wqa: _torch.Tensor,
    descale_wqb: _torch.Tensor,
    descale_wkva: _torch.Tensor,
    norm_weight_qa: _torch.Tensor,
    norm_weight_kva: _torch.Tensor,
    kv_cache: _torch.Tensor,
    cache_index: _torch.Tensor,
    quant_scale_kv: _torch.Tensor | None = None,
    rope_sin: _torch.Tensor | None = None,
    rope_cos: _torch.Tensor | None = None,
    *,
    norm_eps: float = 1e-5,
    w_qr_quant_mode: WQrQuantMode | int,
    qkv_quant_mode: QkvQuantMode | int,
    cache_mode: CacheMode | int,
    qr_scale: float = 1.0,
    kvr_scale: float = 1.0,
    rope_mode: RopeMode | int = RopeMode.INTERLEAVE_HALF,
    return_qr: bool = False,
) -> tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor]:
    if type(return_qr) is not bool:
        raise TypeError("return_qr must be bool")
    _validate_scalars(norm_eps, qr_scale, kvr_scale)
    w_qr_quant_mode = _mode_value("w_qr_quant_mode", w_qr_quant_mode, WQrQuantMode)
    qkv_quant_mode = _mode_value("qkv_quant_mode", qkv_quant_mode, QkvQuantMode)
    cache_mode = _mode_value("cache_mode", cache_mode, CacheMode)
    rope_mode = _mode_value("rope_mode", rope_mode, RopeMode)
    graph_execution = _torch.compiler.is_compiling() or (
        isinstance(x, _torch.Tensor)
        and x.device.type in ("npu", "privateuseone")
        and x.shape[0] != 0
        and _torch.npu.is_current_stream_capturing()
    )
    if graph_execution and w_qr_quant_mode == WQrQuantMode.MXFP8.value:
        for name, value in (
            ("descale_x", descale_x),
            ("descale_wqa", descale_wqa),
            ("descale_wqb", descale_wqb),
            ("descale_wkva", descale_wkva),
        ):
            if not isinstance(value, _torch.Tensor) or value.dtype != _E8:
                raise TypeError(f"{name} must be an E8M0 Tensor")
        descale_x, descale_wqa, descale_wqb, descale_wkva = (
            value.view(_torch.uint8)
            for value in (descale_x, descale_wqa, descale_wqb, descale_wkva)
        )
    args = (
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        quant_scale_kv,
        rope_sin,
        rope_cos,
    )
    attrs = dict(
        norm_eps=norm_eps,
        w_qr_quant_mode=w_qr_quant_mode,
        qkv_quant_mode=qkv_quant_mode,
        cache_mode=cache_mode,
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=rope_mode,
        return_qr=return_qr,
    )
    dispatch = (
        _torch.ops.cannbotdsl_quant_mla_prolog._quant_mla_prolog_graph.default
        if graph_execution
        else _torch.ops.cannbotdsl_quant_mla_prolog.quant_mla_prolog.default
    )
    result = dispatch(*args, **attrs)
    if graph_execution and w_qr_quant_mode == WQrQuantMode.MXFP8.value:
        return result[0], result[1], result[2], result[3].view(_E8)
    return result


def quant_mla_prolog_native(
    x: _torch.Tensor,
    wqa: _torch.Tensor,
    wqb: _torch.Tensor,
    wkva: _torch.Tensor,
    wkb: _torch.Tensor,
    descale_x: _torch.Tensor,
    descale_wqa: _torch.Tensor,
    descale_wqb: _torch.Tensor,
    descale_wkva: _torch.Tensor,
    norm_weight_qa: _torch.Tensor,
    norm_weight_kva: _torch.Tensor,
    kv_cache: _torch.Tensor,
    cache_index: _torch.Tensor,
    quant_scale_kv: _torch.Tensor | None = None,
    rope_sin: _torch.Tensor | None = None,
    rope_cos: _torch.Tensor | None = None,
    *,
    norm_eps: float = 1e-5,
    w_qr_quant_mode: WQrQuantMode | int,
    qkv_quant_mode: QkvQuantMode | int,
    cache_mode: CacheMode | int,
    qr_scale: float = 1.0,
    kvr_scale: float = 1.0,
    rope_mode: RopeMode | int = RopeMode.INTERLEAVE_HALF,
    return_qr: bool = False,
) -> tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor]:
    """Async entry: caller guarantees valid indices and prechecks quant_scale_kv.

    Metadata checks and the shared program remain active. Direct invocation is
    not proof of Native binary installation; require mode must be tested separately.
    """
    return _execute_async(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        quant_scale_kv,
        rope_sin,
        rope_cos,
        norm_eps=norm_eps,
        w_qr_quant_mode=_mode_value("w_qr_quant_mode", w_qr_quant_mode, WQrQuantMode),
        qkv_quant_mode=_mode_value("qkv_quant_mode", qkv_quant_mode, QkvQuantMode),
        cache_mode=_mode_value("cache_mode", cache_mode, CacheMode),
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=_mode_value("rope_mode", rope_mode, RopeMode),
        return_qr=return_qr,
    )


def _default_profiles():
    """Small deployment baseline covers every legal mode/layout/RoPE branch."""
    profiles = []
    for qkv_quant_mode, w_qr_quant_mode in sorted(_VALID_PAIRS):
        for weight_format in ("ND", "NZ"):
            for Dn, Dr in ((128, 64), (192, 64), (256, 0)):
                for rope_mode in (0, 1, 2) if Dr else (0,):
                    for cache_mode in (0,) if qkv_quant_mode == 1 else (0, 1):
                        profiles.append(
                            dict(
                                T=1,
                                dim=1024,
                                q_lora=640,
                                N=1,
                                Dn=Dn,
                                Dr=Dr,
                                w_qr_quant_mode=w_qr_quant_mode,
                                qkv_quant_mode=qkv_quant_mode,
                                cache_mode=cache_mode,
                                rope_mode=rope_mode,
                                weight_format=weight_format,
                            )
                        )
    return profiles


def _program_configs(profile):
    fields = {
        "T",
        "dim",
        "q_lora",
        "N",
        "Dn",
        "Dr",
        "w_qr_quant_mode",
        "qkv_quant_mode",
        "cache_mode",
        "rope_mode",
        "weight_format",
    }
    if set(profile) != fields:
        raise ValueError(
            "deployment profiles accept only public shape/mode/format fields; "
            "core_num and tiling parameters are not supported"
        )
    T = profile["T"]
    if (
        any(type(profile[name]) is not int for name in fields - {"weight_format"})
        or T < 0
    ):
        raise ValueError(f"invalid deployment metadata: {profile}")
    dim, q_lora, N, Dn, Dr = (profile[k] for k in ("dim", "q_lora", "N", "Dn", "Dr"))
    w_qr_quant_mode, qkv_quant_mode, cache_mode, rope_mode = (
        profile[k]
        for k in ("w_qr_quant_mode", "qkv_quant_mode", "cache_mode", "rope_mode")
    )
    weight_format = profile["weight_format"]
    if not (
        1024 <= dim <= 8192
        and dim % 256 == 0
        and q_lora in (640, 1024, 1536, 2048)
        and 1 <= N <= 128
        and (Dn, Dr) in ((128, 64), (192, 64), (256, 0))
    ):
        raise ValueError(f"invalid deployment shape profile: {profile}")
    if not (
        (qkv_quant_mode, w_qr_quant_mode) in _VALID_PAIRS
        and weight_format in ("ND", "NZ")
        and cache_mode in (0, 1)
        and not (qkv_quant_mode == 1 and cache_mode == 1)
        and rope_mode in (0, 1, 2)
        and (Dr != 0 or rope_mode == 0)
    ):
        raise ValueError(f"invalid deployment mode profile: {profile}")
    if T == 0:
        return []
    # Cover every resource-threshold interval without binding the package to
    # the build device's core count or accepting caller-supplied tiling.
    column_tiles = q_lora // QA_PIPELINE_COLUMNS
    resource_boundaries = {
        1,
        8,
        column_tiles * 2,
        N,
        N + 1,
        (N * 4) // 3 + 1,
        N * 2,
        N * 2 + 1,
    }
    resource_boundaries.update(
        column_tiles * owners for owners in range(1, (T + 15) // 16 + 1)
    )
    for rows in _TAIL_HEAD_ROW_CHOICES:
        tasks = N * ((T + rows - 1) // rows)
        resource_boundaries.add(tasks // HEAD_PIPELINE_DEPTH + 1)
    for rows in _TAIL_PIPELINE_ROW_CHOICES:
        resource_boundaries.add(N * ((T + rows - 1) // rows))
    if (
        w_qr_quant_mode == 1
        and qkv_quant_mode in (0, 1, 2)
        and T <= COMPACT_HANDOFF_MAX_ROWS
    ):
        max_slots = (
            HEAD_PIPELINE_DEPTH if T <= WQB_WIDE_MAX_ROWS else COMPACT_HANDOFF_MAX_SLOTS
        )
        for rows in range(
            PROJECTION_ROWS,
            ((T + PROJECTION_ROWS - 1) // PROJECTION_ROWS) * PROJECTION_ROWS + 1,
            PROJECTION_ROWS,
        ):
            for group in range(1, min(N, STRIPE_COLUMNS // (Dn + Dr)) + 1):
                if N % group == 0:
                    tasks = ((T + rows - 1) // rows) * (N // group)
                    resource_boundaries.update(
                        (tasks + waves - 1) // waves
                        for waves in range(1, max_slots + 1)
                    )
        resource_boundaries.add(N * (Dn + Dr) // MM34_SHARED_COLUMNS + 1)
    return sorted(
        {
            _program_parameters(
                T,
                dim,
                q_lora,
                N,
                Dn,
                Dr,
                w_qr_quant_mode,
                qkv_quant_mode,
                weight_format == "NZ",
                cache_mode,
                rope_mode,
                resources,
            )
            for resources in resource_boundaries
        }
    )


@_aot.export("quant_mla_prolog")
def _export_quant_mla_prolog():
    """Export the same complete fused program used by the public operator."""
    path = _os.environ.get("CANNBOTDSL_QMP_PROFILES")
    profiles = _json.loads(_Path(path).read_text()) if path else _default_profiles()
    configs = {config for profile in profiles for config in _program_configs(profile)}
    for parameters in sorted(configs):
        with _build_fused_program(*parameters):
            pass
