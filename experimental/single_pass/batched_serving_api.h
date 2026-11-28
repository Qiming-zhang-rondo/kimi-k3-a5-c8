#pragma once
#include "single_pass_api.h"
#include "single_pass_tiling.h"

// Additive serving ABI1. Never pass this descriptor to an old entry.
struct BatchedServingBuffers {
    SinglePassBuffers base;
    uint8_t *requestBanks;   // INT32 [B], stable bank for each current request row
    uint8_t *requestEpochs;  // INT64 [B]
    uint8_t *ringEpochs;     // INT64 [P,C]
    uint8_t *queryValid;     // BOOL/UINT8 [T]
    uint8_t *faults;         // INT32 [T], sticky until explicit caller drain
};
static_assert(sizeof(BatchedServingBuffers)==20*sizeof(uint8_t*));
extern "C" QSFA_SINGLE_PASS_EXPORT uint32_t QsfaBatchedServingAbiVersion();
extern "C" QSFA_SINGLE_PASS_EXPORT uint32_t QsfaBatchedServingBuffersBytes();
extern "C" QSFA_SINGLE_PASS_EXPORT int32_t QsfaBatchedServingLaunch(
    void *stream,const BatchedServingBuffers *buffers,uint32_t aicCount,
    uint32_t ringCapacity,uint32_t pools,uint32_t queries);

#if !defined(__NPU_ARCH__) || __NPU_ARCH__ == 0
namespace recent128_single_pass {
inline Tiling MakeBatchedServingTiling(uint32_t queries,uint32_t batches,uint32_t keys,
    uint32_t tableColumns,uint32_t aicCount,float scale)
{
    if (!queries || !batches || keys<128 || keys%128 ||
        keys>INT32_MAX || !tableColumns || tableColumns>keys/128 || !aicCount ||
        aicCount>16 || aicCount>queries || !std::isfinite(scale) || scale<=0)
        throw std::invalid_argument("Invalid batched serving geometry");
    Tiling result{};
    auto &b=result.baseParams;
    b.batchSize=batches; b.seqSize=keys; b.qSeqSize=queries;
    b.blockSize=128; b.maxBlockNumPerBatch=tableColumns; b.scaleValue=scale;
    b.nNumOfQInOneGroup=64; b.sparseMode=0; b.outputLayout=1;
    b.dSizeVInput=400; b.sparseBlockCount=2048;
    b.isActualLenDimsNull=0; b.isActualLenDimsKVNull=0; b.returnSoftmaxLse=0;
    result.singleCoreParams.usedCoreNum=aicCount;
    return result;
}
}
#endif
