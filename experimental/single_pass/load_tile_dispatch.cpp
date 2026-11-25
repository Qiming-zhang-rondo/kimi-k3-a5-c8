// Host-only dispatch; every layout has independent device/helper symbols.
#include "single_pass_api.h"
extern "C" int32_t QsfaSinglePassLaunchLoadTile8(void*, const SinglePassBuffers*, uint32_t, uint32_t, uint32_t);
extern "C" int32_t QsfaSinglePassLaunchLoadStride416(void*, const SinglePassBuffers*, uint32_t, uint32_t, uint32_t);
extern "C" int32_t QsfaSinglePassLaunchLoadOut1(void*, const SinglePassBuffers*, uint32_t, uint32_t, uint32_t);
extern "C" int32_t QsfaSinglePassLaunchLoadTile32(void*, const SinglePassBuffers*, uint32_t, uint32_t, uint32_t);
extern "C" int32_t QsfaSinglePassLaunchLoadTile32In1(void*, const SinglePassBuffers*, uint32_t, uint32_t, uint32_t);
extern "C" QSFA_SINGLE_PASS_EXPORT int32_t QsfaSinglePassLaunchLoadTile(
    void *stream, const SinglePassBuffers *b, uint32_t aic, uint32_t capacity,
    uint32_t window, uint32_t variant)
{
    switch (variant) {
    case 1: return QsfaSinglePassLaunchLoadTile8(stream, b, aic, capacity, window);
    case 2: return QsfaSinglePassLaunchLoadStride416(stream, b, aic, capacity, window);
    case 3: return QsfaSinglePassLaunchLoadOut1(stream, b, aic, capacity, window);
    case 4: return QsfaSinglePassLaunchLoadTile32(stream, b, aic, capacity, window);
    case 5: return QsfaSinglePassLaunchLoadTile32In1(stream, b, aic, capacity, window);
    default: return -1;
    }
}
