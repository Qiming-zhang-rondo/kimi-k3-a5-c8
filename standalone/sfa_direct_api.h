// Fixed compact128 direct-launch C ABI. No CANN/Torch/device headers here.
// Caller validates every tensor, allocation, tiling and stream before launch.
// This layer never initializes, changes, synchronizes or resets a device.
#ifndef SFA_DIRECT_API_H
#define SFA_DIRECT_API_H

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define SFA_DIRECT_EXPORT __attribute__((visibility("default")))
#else
#define SFA_DIRECT_EXPORT
#endif

typedef struct SfaDirectBuffers {
    uint8_t *query;
    uint8_t *key;
    uint8_t *value;
    uint8_t *indices;
    uint8_t *blockTable;
    uint8_t *actualQuery;
    uint8_t *actualKv;
    uint8_t *queryRope;
    uint8_t *keyRope;
    uint8_t *output;
    uint8_t *maximum;
    uint8_t *sum;
    uint8_t *workspace;
    uint8_t *tiling;
    uint8_t *coldOutput;
    uint8_t *coldMax;
    uint8_t *coldSum;
    uint8_t *coldPresent;
    uint8_t *hotPresent;
} SfaDirectBuffers;

#ifdef __cplusplus
extern "C" {
#endif

// stream is the caller's current aclrtStream, represented as an opaque pointer.
// buffers and all fields used by the selected kernel must be nonnull and valid.
// Launch is asynchronous; allocation lifetimes belong to the caller's stream.
SFA_DIRECT_EXPORT void SfaLaunchOriginal(void *stream, const SfaDirectBuffers *buffers);
SFA_DIRECT_EXPORT void SfaLaunchFused(void *stream, const SfaDirectBuffers *buffers);
// BF16 cold output and BF16 final output; max/sum and internal arithmetic FP32.
SFA_DIRECT_EXPORT void SfaLaunchFusedBf16(void *stream, const SfaDirectBuffers *buffers);
SFA_DIRECT_EXPORT uint32_t SfaDirectAbiVersion(void);
SFA_DIRECT_EXPORT uint32_t SfaDirectBuffersBytes(void);

#ifdef __cplusplus
}
#endif
#endif
