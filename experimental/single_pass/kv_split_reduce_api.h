// Independent second launch for the standalone KV-split prototype.
#pragma once
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define QSFA_KV_SPLIT_REDUCE_EXPORT __attribute__((visibility("default")))
#else
#define QSFA_KV_SPLIT_REDUCE_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif
// task = query * splitCount + split. Inputs are read-only device allocations:
// partialOutput: float[queryCount, splitCount, 64, 512], normalized partial O.
// maximum/sum:   float[queryCount, splitCount, 64], compact softmax statistics.
// output:        bfloat16[queryCount, 64, 512].
// finalMaximum/finalSum: optional float[queryCount,64], required iff returnStats.
// Empty partials have sum=0; their max and O allocations are not consumed.
// The caller queues this after the producer on the same stream. No global
// barrier or synchronization with other streams is performed by this launch.
QSFA_KV_SPLIT_REDUCE_EXPORT int32_t QsfaSinglePassLaunchKvSplitReduce(
    void *stream, uint8_t *partialOutput, uint8_t *maximum, uint8_t *sum,
    uint8_t *output, uint8_t *finalMaximum, uint8_t *finalSum,
    uint32_t queryCount, uint32_t splitCount, uint32_t vectorBlocks,
    uint32_t returnStats);
#ifdef __cplusplus
}
#endif
