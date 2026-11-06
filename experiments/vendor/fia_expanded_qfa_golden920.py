# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: RUF059, PLR1730
"""Pinned official QFA MXFP8 CPU golden and result comparison functions.

Pure functions extracted by AST from the official v9.2.0 source. NPU imports,
CLI/global example configuration and logging setup are deliberately omitted.
The probe sets these globals for each native partial call. Its query sampling
mask is injected separately, preserving each sampled query's original index.
"""

import logging
import math

import torch

SOURCE_REVISION = "dff221330eecd2132233b1e73ed1072cd8925140"
GOLDEN_SOURCE_SHA256 = "793bf7d1adc1b28fa69bc5c87ea47f8caebbd10cf44cd8e37d5b07f37eb2c7b0"
COMPARE_SOURCE_SHA256 = "c5345e38c6a52cf12afb0da7563ea4283cdf9fc7f57abd6c0d978437cec36edc"
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
_EPS = 1e-10
D = 256
D_rope = 0
ENABLE_ROPE = False
QUANT_GROUP_SIZE = 32
FP8_DTYPE = torch.float8_e4m3fn
N_q = 1
N_kv = 1
SPARSE_MODE = 0
SEQUSED_Q = None
SEQUSED_KV = None
CU_SEQLENS_Q = None
CU_SEQLENS_KV = None

def _derive_seqused(cu_seqlens):
    if cu_seqlens is None:
        return None
    return [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]

def _get_seqused_q():
    return SEQUSED_Q if SEQUSED_Q is not None else _derive_seqused(CU_SEQLENS_Q)

def _get_seqused_kv():
    return SEQUSED_KV if SEQUSED_KV is not None else _derive_seqused(CU_SEQLENS_KV)

def broadcast_kv(num_heads, num_kv_heads, kv_tensor):
    factor = num_heads // num_kv_heads
    B, _, S, D = kv_tensor.shape
    result = torch.zeros([B, num_heads, S, D], dtype=kv_tensor.dtype)
    for i in range(num_heads):
        result[:, i : i + 1, :, :] = kv_tensor[:, i // factor : i // factor + 1, :, :]
    return result


def _build_attention_mask(b, Sq, Skv, actual_seq_q, actual_seq_kv, sparse_mode):
    """构建全局 attention mask
    sparse_mode=3: causal + padding mask (左下三角 + 右上 padding)
    其他: 仅 padding mask
    """
    q_lens_t = torch.tensor(actual_seq_q, dtype=torch.int32)
    k_lens_t = torch.tensor(actual_seq_kv, dtype=torch.int32)
    q_lens_acl = q_lens_t.view(b, 1, 1, 1)
    k_lens_acl = k_lens_t.view(b, 1, 1, 1)

    q_range = torch.arange(Sq).view(1, 1, -1, 1)
    k_range = torch.arange(Skv).view(1, 1, 1, -1)
    q_padding_mask = q_range >= q_lens_acl
    k_padding_mask = k_range >= k_lens_acl

    if sparse_mode == 3:
        delta = k_lens_acl - q_lens_acl
        causal_mask = k_range > (q_range + delta)
        return causal_mask | q_padding_mask | k_padding_mask
    else:
        return q_padding_mask | k_padding_mask


def _compute_s_block(
    Qi, Kj, deq_scale_q_i, deq_scale_k_j, softmax_scale, Qri=None, Krj=None
):
    """计算单个 S block (attention score)"""
    S_ij = torch.matmul(Qi * deq_scale_q_i, (Kj * deq_scale_k_j).permute(0, 1, 3, 2))
    if Qri is not None and Krj is not None:
        S_ij += torch.matmul(
            Qri.to(torch.float32), Krj.to(torch.float32).permute(0, 1, 3, 2)
        )
    return S_ij * softmax_scale


def _online_softmax_update(S_ij, mask_j, mi, si, oi, ln_p_scale, LN2):
    """Online softmax: 计算 m, P, s 更新 (MXFP8: stored max 不含 -ln(p_scale), P 含 p_scale 因子)
    1. mask 位置填 -inf
    2. 求 block 内 max (m_block_j)
    3. m 对齐到 ln2 整数倍 (ceil)，模拟 NPU MXFP8 量化精度损失
    4. 与前一个 block 的 m 取 max (stored max 不含 -ln(p_scale))
    5. 计算 P = exp(S - m + ln_p_scale)，模拟 NPU: exp 用 adjusted max (含 -ln(p_scale))，但 stored max 不含
    6. 求 s = sum(P)
    7. P 转 FP8 再转回 FP32，模拟 NPU 侧 P 的量化损失
    """
    S_ij = S_ij.masked_fill(mask_j, float("-inf"))

    m_block_j, _ = torch.max(S_ij, dim=-1, keepdims=True)
    m_block_j = torch.ceil(m_block_j / LN2) * LN2
    m_block_j = m_block_j - ln_p_scale
    m_block_j = torch.max(mi, m_block_j)

    P_ij_raw = torch.exp(S_ij - m_block_j)
    s_block_j = torch.sum(P_ij_raw, dim=-1, keepdims=True)
    P_ij_drop = P_ij_raw.to(FP8_DTYPE).to(torch.float32)

    return m_block_j, s_block_j, P_ij_drop


def cpu_mxfp8_golden(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    v_descale,
    p_scale,
    actual_seq_q,
    actual_seq_kv,
    softmax_scale=None,
    qr_bf16=None,
    kr_bf16=None,
):
    """CPU Flash Attention golden with MXFP8, C1V1C1V1C2V2 流水"""
    EPSILON = 1e-20
    LN2 = math.log(2.0)
    Q_BLOCK_SIZE = 128
    if D == 256:
        K_BLOCK_SIZE = 128
        V_BLOCK_SIZE = 256
    else:
        K_BLOCK_SIZE = 256
        V_BLOCK_SIZE = 512

    if actual_seq_q is None:
        actual_seq_q = _get_seqused_q()
    if actual_seq_kv is None:
        actual_seq_kv = _get_seqused_kv()

    # FP8 → FP32 反量化
    q_tensor = q_fp8.to(torch.float32)
    k_tensor = k_fp8.to(torch.float32)
    v_tensor = v_fp8.to(torch.float32)

    # GQA: 广播 K/V 到与 Q 相同的 head 数
    if N_q != N_kv:
        logger.info("[INFO] GQA 广播")
        k_tensor = broadcast_kv(N_q, N_kv, k_tensor)
        v_tensor = broadcast_kv(N_q, N_kv, v_tensor)
        dequant_scale_k = broadcast_kv(N_q, N_kv, dequant_scale_k)
        v_descale = broadcast_kv(N_q, N_kv, v_descale)
        if kr_bf16 is not None:
            kr_bf16 = broadcast_kv(N_q, N_kv, kr_bf16)

    b, n, s, d = q_tensor.shape
    d_total = d + (D_rope if ENABLE_ROPE else 0)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d_total)
    dv = v_tensor.shape[-1]
    Sq, Skv = q_tensor.shape[2], k_tensor.shape[2]

    minValue = torch.tensor(-3.402823466e38, dtype=torch.float32)
    out = torch.zeros([b, n, Sq, dv], dtype=torch.float32)
    o_sum = torch.zeros(q_tensor.shape[:-1])[..., None]
    o_max = torch.full(q_tensor.shape[:-1], minValue.item(), dtype=torch.float32)[
        ..., None
    ]

    TILES_Q = (Sq + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    TILES_KV = (Skv + K_BLOCK_SIZE - 1) // K_BLOCK_SIZE

    mask_global = _build_attention_mask(
        b, Sq, Skv, actual_seq_q, actual_seq_kv, SPARSE_MODE
    )

    Q_BLOCKS = list(torch.split(q_tensor, Q_BLOCK_SIZE, dim=2))
    K_BLOCKS = list(torch.split(k_tensor, K_BLOCK_SIZE, dim=2))
    V_BLOCKS = list(torch.split(v_tensor, V_BLOCK_SIZE, dim=2))
    o_BLOCKS = list(torch.split(out, Q_BLOCK_SIZE, dim=2))
    s_BLOCKS = list(torch.split(o_sum, Q_BLOCK_SIZE, dim=2))
    m_BLOCKS = list(torch.split(o_max, Q_BLOCK_SIZE, dim=2))

    Qr_BLOCKS = None
    Kr_BLOCKS = None
    if ENABLE_ROPE and qr_bf16 is not None:
        Qr_BLOCKS = list(torch.split(qr_bf16, Q_BLOCK_SIZE, dim=2))
        Kr_BLOCKS = list(torch.split(kr_bf16, K_BLOCK_SIZE, dim=2))

    ln_p_scale = torch.tensor([math.log(p_scale)], dtype=torch.float32)

    # dequant_scale 按 group_size 扩展，用于逐元素反量化
    dequant_scale_q_expanded = dequant_scale_q.repeat_interleave(
        QUANT_GROUP_SIZE, dim=-1
    )[:, :, :, :D]
    dequant_scale_k_expanded = dequant_scale_k.repeat_interleave(
        QUANT_GROUP_SIZE, dim=-1
    )[:, :, :, :D]
    v_descale_expanded = v_descale.repeat_interleave(QUANT_GROUP_SIZE, dim=2)

    logger.info(
        "[CPU Golden] TILES_Q=%d, TILES_KV=%d, Sq=%d, Skv=%d",
        TILES_Q,
        TILES_KV,
        Sq,
        Skv,
    )

    for i in range(TILES_Q):
        Qi = Q_BLOCKS[i]
        Sq_start = i * Q_BLOCK_SIZE
        Sq_end = min(Sq_start + Q_BLOCK_SIZE, Sq)
        deq_scale_q_i = dequant_scale_q_expanded[:, :, Sq_start:Sq_end, :]

        Qri = Qr_BLOCKS[i] if Qr_BLOCKS is not None else None

        for j in range(0, TILES_KV, 2):
            # C1V1C1V1C2V2: 每次迭代处理 2 个 K block + 1 个 V block
            oi, si, mi = o_BLOCKS[i], s_BLOCKS[i], m_BLOCKS[i]

            Kj = K_BLOCKS[j]
            Sk_start = j * K_BLOCK_SIZE
            Sk_end = min(Sk_start + K_BLOCK_SIZE, Skv)
            deq_scale_k_j = dequant_scale_k_expanded[:, :, Sk_start:Sk_end, :]
            Krj = Kr_BLOCKS[j] if Kr_BLOCKS is not None else None

            S_ij = _compute_s_block(
                Qi, Kj, deq_scale_q_i, deq_scale_k_j, softmax_scale, Qri, Krj
            )
            mask_j = mask_global[:, :, Sq_start:Sq_end, Sk_start:Sk_end]
            m_block_j, s_block_j, P_ij_drop = _online_softmax_update(
                S_ij, mask_j, mi, si, oi, ln_p_scale, LN2
            )

            if j + 1 < TILES_KV:
                # --- 第二个 K block (j+1) ---
                Kj1 = K_BLOCKS[j + 1]
                Sk1_start = (j + 1) * K_BLOCK_SIZE
                Sk1_end = min(Sk1_start + K_BLOCK_SIZE, Skv)
                deq_scale_k_j1 = dequant_scale_k_expanded[:, :, Sk1_start:Sk1_end, :]
                Krj1 = Kr_BLOCKS[j + 1] if Kr_BLOCKS is not None else None

                S_ij1 = _compute_s_block(
                    Qi, Kj1, deq_scale_q_i, deq_scale_k_j1, softmax_scale, Qri, Krj1
                )
                mask_j1 = mask_global[:, :, Sq_start:Sq_end, Sk1_start:Sk1_end]
                m_block_j1, s_block_j1, P_ij1_drop = _online_softmax_update(
                    S_ij1, mask_j1, m_block_j, s_block_j, oi, ln_p_scale, LN2
                )

                # V block: 一个 V_BLOCK_SIZE 对应两个 K_BLOCK_SIZE
                Vj = V_BLOCKS[j // 2]
                Sv_start = (j // 2) * V_BLOCK_SIZE
                Sv_end = min(Sv_start + V_BLOCK_SIZE, Skv)
                deq_scale_v_j = v_descale_expanded[:, :, Sv_start:Sv_end, :]
                Vj_dequant = Vj * deq_scale_v_j[:, :, : Vj.shape[2], :]

                V_part1 = Vj_dequant[:, :, : Kj.shape[2], :]
                V_part2 = Vj_dequant[:, :, Kj.shape[2] : Kj.shape[2] + Kj1.shape[2], :]
                P_ij_Vj = torch.matmul(
                    P_ij_drop * torch.exp(m_block_j - m_block_j1), V_part1
                ) + torch.matmul(P_ij1_drop, V_part2)

                update_mul_si = torch.exp(mi - m_block_j1)
                si_new = (
                    update_mul_si * si
                    + s_block_j * torch.exp(m_block_j - m_block_j1)
                    + s_block_j1
                )
                o_BLOCKS[i] = update_mul_si * oi + P_ij_Vj
                s_BLOCKS[i] = si_new
                m_BLOCKS[i] = m_block_j1
            else:
                Vj = V_BLOCKS[j // 2]
                Sv_start = j * K_BLOCK_SIZE
                Sv_end = min(Sv_start + K_BLOCK_SIZE, Skv)
                deq_scale_v_j = v_descale_expanded[:, :, Sv_start:Sv_end, :]
                Vj_expanded = Vj[:, :, : Kj.shape[2], :]
                Vj_dequant = Vj_expanded * deq_scale_v_j[:, :, : Kj.shape[2], :]

                P_ij_Vj = torch.matmul(P_ij_drop, Vj_dequant)
                update_mul_si = torch.exp(mi - m_block_j)
                si_new = update_mul_si * si + s_block_j
                o_BLOCKS[i] = update_mul_si * oi + P_ij_Vj
                s_BLOCKS[i] = si_new
                m_BLOCKS[i] = m_block_j

    out = torch.cat(o_BLOCKS, dim=2)
    out_sum = torch.cat(s_BLOCKS, dim=2)
    out = out / (out_sum + EPSILON)

    o_max = torch.cat(m_BLOCKS, dim=2)
    all_masked = o_max <= minValue.item()
    lse = torch.where(
        all_masked,
        torch.full_like(o_max, float("inf")),
        o_max + torch.log(out_sum + EPSILON),
    )
    out = torch.where(all_masked, torch.zeros_like(out), out)
    logger.info("[CPU Golden] output=%s", out.shape)
    return out, lse


def cal_relative_diff_torch(real_data, expect_data):
    diff = abs(float(real_data) - float(expect_data))
    return diff / (abs(float(expect_data)) + _EPS)


def display_output_torch(real_data, expect_data, start, end, expect_fp32_data=None):
    has_fp32 = expect_fp32_data is not None

    logger.info(
        "---------------------------------------------------------------------------------------"
    )
    logger.info(
        "Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff"
        if has_fp32
        else "Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff"
    )
    logger.info(
        "---------------------------------------------------------------------------------------"
    )

    split_count = int(end - start)

    def display_row(i):
        j = start + i
        real_value = float(real_data[j])
        expect_value = float(expect_data[j])
        diff_rate = cal_relative_diff_torch(real_value, expect_value)
        idx = start + i + 1

        if not torch.isfinite(expect_data[j]).item():
            diff_abs = "inf" if torch.isinf(expect_data[j]).item() else "nan"
            if has_fp32:
                logger.info(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s",
                    idx,
                    float(expect_fp32_data[j]),
                    expect_value,
                    real_value,
                    diff_abs,
                    diff_rate,
                )
            else:
                logger.info(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s",
                    idx,
                    expect_value,
                    real_value,
                    diff_abs,
                    diff_rate,
                )
        else:
            diff_abs = abs(expect_value - real_value)
            if has_fp32:
                logger.info(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f",
                    idx,
                    float(expect_fp32_data[j]),
                    expect_value,
                    real_value,
                    diff_abs,
                    diff_rate,
                )
            else:
                logger.info(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f",
                    idx,
                    expect_value,
                    real_value,
                    diff_abs,
                    diff_rate,
                )

    if split_count <= 20:
        for i in range(split_count + 1):
            display_row(i)
    else:
        for i in range(10):
            display_row(i)
        logger.info("...   \t   ...   \t   ...   \t   ...    \t   ...")
        for i in range(split_count - 10 + 1, split_count + 1):
            display_row(i)


def display_error_output_torch(real_data, expect_data, err_idx, relative_diff):
    logger.info(
        "Error Line-----------------------------------------------------------------------------"
    )
    logger.info("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    logger.info(
        "---------------------------------------------------------------------------------------"
    )

    len_err = int(err_idx.numel())
    for count, i in enumerate(err_idx.tolist(), 1):
        if count < 10 or (90 < count < 100):
            expect_value = float(expect_data[i])
            real_value = float(real_data[i])
            logger.info(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f",
                i,
                expect_value,
                real_value,
                abs(expect_value - real_value),
                float(relative_diff[count - 1].item()),
            )
        elif count == 10 or (count == 100 and len_err > 100):
            logger.info(
                "%08s \t %07s \t %07s \t %07s \t %07s",
                "...",
                "...",
                "...",
                "...",
                "...",
            )
        elif count > 100:
            break

    logger.info(
        "Max-RE line:---------------------------------------------------------------------------"
    )
    max_error = (
        float(torch.max(relative_diff).item()) if relative_diff.numel() > 0 else 0.0
    )
    m_idx_list = err_idx[relative_diff == max_error]
    for m_count, m_idx in enumerate(m_idx_list.tolist()):
        if m_count >= 3:
            break
        expect_value = float(expect_data[m_idx])
        real_value = float(real_data[m_idx])
        logger.info(
            "%08d \t %.7f \t %.7f \t %.7f \t %.7f",
            m_idx,
            expect_value,
            real_value,
            abs(expect_value - real_value),
            max_error,
        )
    logger.info(
        "---------------------------------------------------------------------------------------"
    )


def check_result(
    expect,
    npu_result,
    rtol=None,
    atol=None,
    diff_thd=None,
    pct_thd=None,
    max_diff_hd=None,
):
    # 默认值与 ttk result_compare_method.check_result 保持一致:
    # diff_thd=0.005, pct_thd=0.005, max_diff_hd=10
    # rtol/atol 按 dtype 分支(bf16 0.0078125/0.0001, 否则 0.005/0.000025)
    # 调用方可显式传入 rtol/atol 覆盖默认值(例如 LSE 对齐 ttk replay 模式)
    _diff_thd_default = 0.005
    _pct_thd_default = 0.005
    _max_diff_hd_default = 10
    _rtol_default = 0.005
    _atol_default = 0.000025

    diff_thd = _diff_thd_default if diff_thd is None else diff_thd
    pct_thd = _pct_thd_default if pct_thd is None else pct_thd
    max_diff_hd = _max_diff_hd_default if max_diff_hd is None else max_diff_hd
    rtol = _rtol_default if rtol is None else rtol
    atol = _atol_default if atol is None else atol

    real_data = npu_result.detach().cpu().flatten()
    expect_data = (
        expect.detach().cpu().flatten()
        if isinstance(expect, torch.Tensor)
        else torch.as_tensor(expect).flatten()
    )

    if real_data.numel() == 0:
        logger.info(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"'
        )
        return "Pass", 100.0, 0

    if real_data.numel() != expect_data.numel():
        logger.info(
            "Error,the size of npu output[%s] and benchmark[%s] is not equal.",
            real_data.numel(),
            expect_data.numel(),
        )
        return "Failed", 0.0, 0

    start = 0
    end = real_data.numel() - 1
    if end < start:
        end = start
    split_count = end - start + 1 if end != start else 1
    max_error = 0.0

    overflows_count = int(
        torch.isinf(expect_data).sum().item() + torch.isnan(expect_data).sum().item()
    )
    if overflows_count > 0:
        logger.info(
            "Overflow,size:%s,benchmark_output:%s, %s",
            overflows_count,
            expect_data[torch.isinf(expect_data)][0:10],
            expect_data[torch.isnan(expect_data)][0:10],
        )

    logger.info("split_count:%s; max_diff_hd:%s;", float(split_count), max_diff_hd)

    real_float = real_data.to(torch.float32)
    expect_float = expect_data.to(torch.float32)

    if npu_result.dtype == torch.bfloat16:
        rtol = 0.0078125
        atol = 0.0001

    diff_result = torch.isclose(
        real_float, expect_float, rtol=rtol, atol=atol, equal_nan=True
    )
    err_idx = torch.nonzero(~diff_result, as_tuple=False).flatten()

    if expect_data.dtype == torch.bool:
        expect_data = expect_data.to(torch.int8)
        real_data = real_data.to(torch.int8)
        real_float = real_data.to(torch.float32)
        expect_float = expect_data.to(torch.float32)

    diff_abs = torch.abs(expect_float - real_float)
    b1 = torch.maximum(torch.abs(real_float), torch.abs(expect_float))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = torch.maximum(b1, torch.tensor(b2, dtype=torch.float32)) + _EPS
    err_diff = (diff_abs / b)[err_idx]
    err_diff = torch.where(
        torch.isnan(err_diff) | torch.isinf(err_diff),
        torch.tensor(float("inf"), dtype=torch.float32),
        err_diff,
    )

    fulfill_percent = float(split_count - err_idx.numel()) / float(split_count) * 100.0

    display_output_torch(real_data, expect_data, start, end)

    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"

    if err_diff.numel() > 0:
        max_error = float(torch.max(err_diff).item())
        if max_error >= max_diff_hd:
            result = "Failed"

    logger.info(
        "---------------------------------------------------------------------------------------"
    )
    logger.info("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    logger.info(
        "---------------------------------------------------------------------------------------"
    )
    logger.info(
        "%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s",
        rtol,
        atol,
        pct_thd,
        fulfill_percent,
        result,
    )

    if err_diff.numel() > 0:
        logger.info(
            "Max-RelativeError is: %s. Threshold is: %s.", max_error, max_diff_hd
        )

    if result == "Failed":
        display_error_output_torch(real_data, expect_data, err_idx, err_diff)

    return result, fulfill_percent, max_error
