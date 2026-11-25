// Independent standalone prototype ABI1. Never call installed ACLNN with this.
#pragma once
#include <stdint.h>
#if defined(__GNUC__) || defined(__clang__)
#define QSFA_SINGLE_PASS_EXPORT __attribute__((visibility("default")))
#else
#define QSFA_SINGLE_PASS_EXPORT
#endif

typedef struct SinglePassBuffers {
    uint8_t *query;
    uint8_t *cold;
    uint8_t *indices;
    uint8_t *blockTable;
    uint8_t *actualQuery;
    uint8_t *actualKv;
    uint8_t *hotLatent;
    uint8_t *hotRope;
    uint8_t *ringTags;
    uint8_t *queryPositions;
    uint8_t *output;
    uint8_t *maximum;
    uint8_t *sum;
    uint8_t *workspace;
    uint8_t *tiling;
} SinglePassBuffers;

#ifdef __cplusplus
extern "C" {
#endif
QSFA_SINGLE_PASS_EXPORT uint32_t QsfaSinglePassAbiVersion(void);
QSFA_SINGLE_PASS_EXPORT uint32_t QsfaSinglePassBuffersBytes(void);
QSFA_SINGLE_PASS_EXPORT int32_t QsfaSinglePassLaunch(void *stream, const SinglePassBuffers *buffers,
    uint32_t aicCount, uint32_t ringCapacity, uint32_t window);
// Additive entry: same ABI1 descriptor; independent precision-routed kernel.
QSFA_SINGLE_PASS_EXPORT int32_t QsfaSinglePassLaunchRouted(void *stream, const SinglePassBuffers *buffers,
    uint32_t aicCount, uint32_t ringCapacity, uint32_t window);
// Additive fast-pair routing variant; descriptor layout remains ABI1.
QSFA_SINGLE_PASS_EXPORT int32_t QsfaSinglePassLaunchFastPair(void *stream, const SinglePassBuffers *buffers,
    uint32_t aicCount, uint32_t ringCapacity, uint32_t window);
// Independent original native C4 kernel, compiled in a separate ASC TU.
QSFA_SINGLE_PASS_EXPORT int32_t QsfaSinglePassLaunchCold(void *stream, const SinglePassBuffers *buffers,
    uint32_t aicCount, uint32_t ringCapacity, uint32_t window);
#ifdef __cplusplus
}
#endif
