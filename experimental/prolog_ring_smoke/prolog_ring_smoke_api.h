#pragma once
#include <cstdint>
#include <cstddef>
#if defined(__GNUC__)
#define QSFA_RING_EXPORT __attribute__((visibility("default")))
#else
#define QSFA_RING_EXPORT
#endif
struct PrologRingSmokeBuffers {
    uint8_t *inputLatent;
    uint8_t *inputRope;
    uint8_t *latentRing;
    uint8_t *ropeRing;
    uint8_t *positionTags;
    uint8_t *epochTags;
    uint8_t *tokenBanks;
    uint8_t *tokenPositions;
    uint8_t *tokenEpochs;
    uint8_t *tokenValid;
};
static_assert(sizeof(PrologRingSmokeBuffers)==80);
extern "C" QSFA_RING_EXPORT uint32_t QsfaPrologRingSmokeAbiVersion();
extern "C" QSFA_RING_EXPORT uint32_t QsfaPrologRingSmokeBuffersBytes();
extern "C" QSFA_RING_EXPORT int32_t QsfaPrologRingSmokeLaunch(void *stream,
    const PrologRingSmokeBuffers *buffers, uint32_t rows, uint32_t pools,
    uint32_t capacity, uint32_t blocks, uint32_t enabled);
