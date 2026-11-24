/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


// Source candidate only: compile and device validation remain mandatory.
#include "kernel_operator.h"
#if (__CCE_AICORE__ != 310)
#error "SparseFlashAttentionColdMerge is an arch35-only experimental candidate"
#endif
// Private template registration: same numeric argument layout as native SFA.
// Only the fixed-shape PA/TND candidate is registered here.
#include "ascendc/host_api/tiling/template_argument.h"
#define ASCENDC_TPL_4_BW 4
ASCENDC_TPL_ARGS_DECL(SparseFlashAttentionColdMerge,
    ASCENDC_TPL_BOOL_DECL(FLASH_DECODE, 0),
    ASCENDC_TPL_BOOL_DECL(PAGE_ATTENTION, 1),
    ASCENDC_TPL_UINT_DECL(LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 1),
    ASCENDC_TPL_UINT_DECL(KV_LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 2),
    ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_BOOL_DECL(IS_SPLIT_G, 0, 1),
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(FLASH_DECODE, 0),
        ASCENDC_TPL_BOOL_SEL(PAGE_ATTENTION, 1),
        ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, 2),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_SPLIT_G, 0, 1),
    ),
);

#include "arch35/sparse_flash_attention_kernel_mla.h"
using namespace AscendC;

#if defined(__DAV_C310_CUBE__)
#define SFA_OP_IMPL(templateClass, tilingdataClass, ...)                                          \
    do {                                                                                          \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,               \
            BaseApi::SFAMatmulService<__VA_ARGS__>, BaseApi::SFAMatmulServiceDummy<__VA_ARGS__>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,                \
            BaseApi::SFAVectorServiceDummy<__VA_ARGS__>, BaseApi::SFAVectorService<__VA_ARGS__>>::type;   \
        templateClass<CubeBlockType, VecBlockType> op;                                            \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                       \
        op.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV,      \
	    blocktable, queryRope, keyRope, attentionOut, softmaxMax, softmaxSum, coldOutput, coldMax, coldSum, coldPresent, hotPresent, user, nullptr, tiling, &tPipe);         \
        op.Process();                                                                             \
    } while (0)
#else
#define SFA_OP_IMPL(templateClass, tilingdataClass, ...)                                          \
    do {                                                                                          \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,               \
            BaseApi::SFAMatmulService<__VA_ARGS__>, BaseApi::SFAMatmulServiceDummy<__VA_ARGS__>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,                \
            BaseApi::SFAVectorServiceDummy<__VA_ARGS__>, BaseApi::SFAVectorService<__VA_ARGS__>>::type;   \
        templateClass<CubeBlockType, VecBlockType> op;                                            \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                       \
        const tilingdataClass *__restrict tilingData = &tiling_data_in;                                     \
        op.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV,      \
	    blocktable, queryRope, keyRope, attentionOut, softmaxMax, softmaxSum, coldOutput, coldMax, coldSum, coldPresent, hotPresent, user, tilingData, tiling, &tPipe);         \
        op.Process();                                                                             \
    } while (0)
#endif

template<int FLASH_DECODE, int PAGE_ATTENTION, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int IS_SPLIT_G>
__global__ __aicore__ void sparse_flash_attention_cold_merge(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *actualSeqLengthsQuery, __gm__ uint8_t *actualSeqLengthsKV,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *coldOutput, __gm__ uint8_t *coldMax, __gm__ uint8_t *coldSum,
    __gm__ uint8_t *coldPresent, __gm__ uint8_t *hotPresent,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    static_assert(FLASH_DECODE == 0 && PAGE_ATTENTION == 1 && LAYOUT_T == 1 && KV_LAYOUT_T == 2,
        "Candidate requires non-flash-decode TND / PA_BSND");
    static_assert(ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 &&
        ORIG_DTYPE_VALUE == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT,
        "Candidate requires BF16 Q/K/V and FP32 fused attentionOut");
    static_assert(ORIG_DTYPE_QUERY_ROPE == DT_BF16 && ORIG_DTYPE_KEY_ROPE == DT_BF16,
        "Candidate keeps query/key RoPE in BF16");
    static_assert(ORIG_DTYPE_COLD_OUTPUT == DT_FLOAT && ORIG_DTYPE_COLD_MAX == DT_FLOAT &&
        ORIG_DTYPE_COLD_SUM == DT_FLOAT,
        "Cold normalized output/max/sum must be FP32");
    static_assert((ORIG_DTYPE_COLD_PRESENT == DT_UINT8 || ORIG_DTYPE_COLD_PRESENT == DT_BOOL) &&
        (ORIG_DTYPE_HOT_PRESENT == DT_UINT8 || ORIG_DTYPE_HOT_PRESENT == DT_BOOL),
        "Presence tensors must use one byte per query");
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    // Never inherit native's BF16-output-or-half fallback for FP32 output.
    SFA_OP_IMPL(BaseApi::SparseFlashAttentionKernelMla, SparseFlashAttentionTilingDataMla,
        bfloat16_t, bfloat16_t, float, float, FLASH_DECODE, PAGE_ATTENTION,
        static_cast<SFA_LAYOUT>(LAYOUT_T), static_cast<SFA_LAYOUT>(KV_LAYOUT_T),
        static_cast<SFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G);
}
