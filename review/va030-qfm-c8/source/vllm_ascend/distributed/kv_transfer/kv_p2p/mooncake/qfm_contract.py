# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Versioned wire contract for raw QFM pages; no device operations."""

QFM_WIRE_FORMAT = "k3-qfm-v1:fp8_e4m3fn:pa_nz:128x576:descale=1"
QFM_BLOCK_SHAPE = (1, 18, 128, 32)
QFM_BLOCK_BYTES = 128 * 576


def layer_wire_format(metadata, index):
    formats = getattr(metadata, "cache_formats", None)
    return formats[index] if formats is not None else ""


def validate_qfm_transfer(local, remote, local_index, remote_index):
    """Reject mixed versions/formats before any transfer is submitted.

    Padding/strides may differ across P/D; transfer only the contiguous
    128-token payload. KDA/draft layers keep the existing connector protocol.
    """
    local_format = layer_wire_format(local, local_index)
    remote_format = layer_wire_format(remote, remote_index)
    if not (local_format or remote_format):
        return
    if local_format != QFM_WIRE_FORMAT or remote_format != QFM_WIRE_FORMAT:
        raise ValueError("QFM P/D cache format mismatch; install the same QFM PD patch on both endpoints")
    for metadata, index in ((local, local_index), (remote, remote_index)):
        if metadata.block_shapes[index] != [QFM_BLOCK_SHAPE]:
            raise ValueError("QFM PD requires one PA_NZ full576 cache per MLA layer")
        if metadata.block_lens[index] != [QFM_BLOCK_BYTES]:
            raise ValueError("QFM PD cache payload must be 128 * 576 FP8 bytes")
        scales = metadata.block_size_scales[index]
        if len(scales) != 1 or scales[0] <= 0 or metadata.layer_block_sizes[index] != 128 * scales[0]:
            raise ValueError("QFM PD manager/kernel page mapping is inconsistent")
        strides = metadata.block_strides[index]
        if len(strides) != 1 or strides[0] < QFM_BLOCK_BYTES or strides[0] % 32:
            raise ValueError("QFM PD kernel stride must preserve payload and 32-byte alignment")
