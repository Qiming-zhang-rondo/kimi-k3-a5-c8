// Phase C: Three-arm paired performance benchmark
// W0: Original SFA (all-C4, TopK 2048)
// Fused: Fusion SFA (BF16 hot 128 + C4 cold 1920, native merge)
// Two-pass reference: from existing framework (126.17 µs baseline)
//
// Timing: warmup 20, iterations 100, 10 rounds AB/BA, ACL event timer
// Same device, same process, same inputs for both arms

#include "kernel_operator.h"
#include "sparse_flash_attention_tiling_data.h"

// Original kernel entry (for W0 arm)
extern "C" __global__ __aicore__ void sfa_arch35_bf16_compile_probe(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *actualSeqLengthsQuery, __gm__ uint8_t *actualSeqLengthsKV,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling);

// Fusion kernel entry (for Fused arm)
extern "C" __global__ __aicore__ void sfa_cold_merge_probe(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *actualSeqLengthsQuery, __gm__ uint8_t *actualSeqLengthsKV,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling,
    __gm__ uint8_t *coldOutput, __gm__ uint8_t *coldMax, __gm__ uint8_t *coldSum,
    __gm__ uint8_t *coldPresent, __gm__ uint8_t *hotPresent);

#include "acl/acl.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// Fixed shape
static constexpr uint32_t B = 1, Q = 4, H = 64, D = 512, R = 64;
static constexpr uint32_t TOPK = 2048, KV = 8192, PAGE = 128;
static constexpr int WARMUP = 20, ITERS = 100, ROUNDS = 10;

// Buffer sizes (BF16 for KV/Q, FP32 for output in fusion)
static constexpr size_t Q_BYTES = B * Q * H * D * 2;  // bf16
static constexpr size_t QR_BYTES = B * Q * H * R * 2;
static constexpr size_t KV_BYTES = KV * H * D * 2;     // bf16 KV
static constexpr size_t KR_BYTES = KV * R * 2;
static constexpr size_t IDX_BYTES = B * Q * TOPK * 4;   // int32
static constexpr size_t OUT_BF16_BYTES = B * Q * H * D * 2;
static constexpr size_t OUT_FP32_BYTES = B * Q * H * D * 4;
static constexpr size_t STATS_BYTES = B * Q * H * 4;
static constexpr size_t SEQ_BYTES = B * 4;
static constexpr size_t TILING_BYTES = 144;
static constexpr size_t WS_BYTES = 442368;  // verified user-scratch size

static void FillTiling(SparseFlashAttentionTilingDataMla &t)
{
    memset(&t, 0, sizeof(t));
    t.baseParams.batchSize = B;
    t.baseParams.seqSize = KV;
    t.baseParams.qSeqSize = Q;
    t.baseParams.blockSize = 128;
    t.baseParams.maxBlockNumPerBatch = (KV + 127) / 128;
    t.baseParams.scaleValue = 1.0f / sqrtf((float)(D + R));
    t.baseParams.nNumOfQInOneGroup = 1;
    t.baseParams.actualLenDimsQ = Q;
    t.baseParams.actualLenDimsKV = KV;
    t.baseParams.outputLayout = 1;  // TND
    t.baseParams.sparseMode = 0;
    t.baseParams.attentionMode = 2;
    t.baseParams.returnSoftmaxLse = 0;
    t.baseParams.sparseBlockCount = TOPK;
    t.splitKVParams.s2 = 128;
    t.splitKVParams.accumOutSize = 1 * 2 * 1 * 64 * 576;
    t.splitKVParams.logSumExpSize = 2 * 1 * 2 * 1 * 64;
    t.singleCoreParams.usedCoreNum = 1;
    t.singleCoreTensorSize.mmResUbSize = 128 * 64;
    t.singleCoreTensorSize.bmm2ResUbSize = 576 * 64;
    t.innerSplitParams.mBaseSize = 64;
    t.innerSplitParams.s2BaseSize = 128;
}

struct Buffers {
    uint8_t *q, *k, *v, *qr, *kr, *idx, *bt;
    uint8_t *seqQ, *seqKV, *outBf16, *outFp32, *maxBuf, *sumBuf;
    uint8_t *ws, *tiling;
    uint8_t *coldOut, *coldMax, *coldSum, *coldPres, *hotPres;
};

static double TimeKernel(aclrtStream stream, aclrtEvent start, aclrtEvent end,
                         int iterations, void (*launch)(Buffers&, aclrtStream), Buffers &buf)
{
    // Warmup
    for (int i = 0; i < WARMUP; i++) launch(buf, stream);
    aclrtSynchronizeStream(stream);

    // Timed
    double totalMs = 0;
    for (int i = 0; i < iterations; i++) {
        aclrtRecordEvent(start, stream);
        launch(buf, stream);
        aclrtRecordEvent(end, stream);
        aclrtSynchronizeEvent(end);
        float ms;
        aclrtElapsedTime(&ms, start, end);
        totalMs += ms;
    }
    return totalMs / iterations * 1000.0;  // µs per iteration
}

static void LaunchW0(Buffers &b, aclrtStream s)
{
    sfa_arch35_bf16_compile_probe<<<1, nullptr, s>>>(
        b.q, b.k, b.v, b.idx, b.bt, b.seqQ, b.seqKV,
        b.qr, b.kr, b.outBf16, b.maxBuf, b.sumBuf, b.ws, b.tiling);
}

static void LaunchFused(Buffers &b, aclrtStream s)
{
    sfa_cold_merge_probe<<<1, nullptr, s>>>(
        b.q, b.k, b.v, b.idx, b.bt, b.seqQ, b.seqKV,
        b.qr, b.kr, b.outFp32, b.maxBuf, b.sumBuf, b.ws, b.tiling,
        b.coldOut, b.coldMax, b.coldSum, b.coldPres, b.hotPres);
}

int main(int argc, char **argv)
{
    printf("=== Phase C: Three-Arm Paired Performance ===\n");
    printf("Shape: B=%u Q=%u K=%u H=%u D=%u TopK=%u\n", B, Q, KV, H, D, TOPK);
    printf("Timing: warmup=%d iter=%d rounds=%d\n\n", WARMUP, ITERS, ROUNDS);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    aclrtEvent evStart, evEnd;
    aclrtCreateEvent(&evStart);
    aclrtCreateEvent(&evEnd);

    // Allocate buffers
    Buffers buf = {};
    struct { const char *name; size_t bytes; uint8_t **ptr; } allocs[] = {
        {"query", Q_BYTES, &buf.q}, {"key", KV_BYTES, &buf.k},
        {"value", KV_BYTES, &buf.v}, {"queryRope", QR_BYTES, &buf.qr},
        {"keyRope", KR_BYTES, &buf.kr}, {"indices", IDX_BYTES, &buf.idx},
        {"blockTable", 256, &buf.bt}, {"seqQ", SEQ_BYTES, &buf.seqQ},
        {"seqKV", SEQ_BYTES, &buf.seqKV}, {"outBf16", OUT_BF16_BYTES, &buf.outBf16},
        {"outFp32", OUT_FP32_BYTES, &buf.outFp32},
        {"max", STATS_BYTES, &buf.maxBuf}, {"sum", STATS_BYTES, &buf.sumBuf},
        {"workspace", WS_BYTES, &buf.ws}, {"tiling", TILING_BYTES, &buf.tiling},
        {"coldOut", OUT_FP32_BYTES, &buf.coldOut},
        {"coldMax", STATS_BYTES, &buf.coldMax}, {"coldSum", STATS_BYTES, &buf.coldSum},
        {"coldPresent", Q, &buf.coldPres}, {"hotPresent", Q, &buf.hotPres},
    };
    for (auto &a : allocs) {
        if (aclrtMalloc((void**)a.ptr, a.bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            printf("ERROR: alloc %s failed\n", a.name); return 1;
        }
        aclrtMemset(*a.ptr, a.bytes, 0, a.bytes);
    }
    printf("Buffers allocated (%zu KB total)\n",
           (Q_BYTES+KV_BYTES*2+QR_BYTES+KR_BYTES+IDX_BYTES+OUT_BF16_BYTES+OUT_FP32_BYTES+STATS_BYTES*5+WS_BYTES)/1024);

    // Fill tiling
    SparseFlashAttentionTilingDataMla tiling;
    FillTiling(tiling);
    aclrtMemcpy(buf.tiling, TILING_BYTES, &tiling, TILING_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);

    // Fill seq lengths and block table
    int32_t seqQ[B] = {(int32_t)Q}, seqKV[B] = {(int32_t)KV};
    aclrtMemcpy(buf.seqQ, SEQ_BYTES, seqQ, SEQ_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(buf.seqKV, SEQ_BYTES, seqKV, SEQ_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    int32_t bt[64]; for (int i = 0; i < 64; i++) bt[i] = i;
    aclrtMemcpy(buf.bt, 256, bt, 256, ACL_MEMCPY_HOST_TO_DEVICE);

    // Fill indices (all 2048 keys, zero-based for fair W0 comparison)
    std::vector<int32_t> idx(B * Q * TOPK);
    for (uint32_t i = 0; i < idx.size(); i++) idx[i] = i % KV;
    aclrtMemcpy(buf.idx, IDX_BYTES, idx.data(), IDX_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);

    // Fill presence flags (both present)
    uint8_t pres[Q]; memset(pres, 1, Q);
    aclrtMemcpy(buf.coldPres, Q, pres, Q, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(buf.hotPres, Q, pres, Q, ACL_MEMCPY_HOST_TO_DEVICE);

    // Two-pass reference (from existing framework baseline)
    const double twoPassUs = 126.17;  // historical measured value

    // Paired AB/BA rounds
    std::vector<double> w0Times, fusedTimes;
    for (int round = 0; round < ROUNDS; round++) {
        double w0Us, fusedUs;
        if (round % 2 == 0) {  // AB: W0 first
            w0Us = TimeKernel(stream, evStart, evEnd, ITERS, LaunchW0, buf);
            fusedUs = TimeKernel(stream, evStart, evEnd, ITERS, LaunchFused, buf);
        } else {  // BA: Fused first
            fusedUs = TimeKernel(stream, evStart, evEnd, ITERS, LaunchFused, buf);
            w0Us = TimeKernel(stream, evStart, evEnd, ITERS, LaunchW0, buf);
        }
        w0Times.push_back(w0Us);
        fusedTimes.push_back(fusedUs);
        printf("Round %2d: W0=%.2f µs  Fused=%.2f µs  ratio=%.3f\n",
               round, w0Us, fusedUs, fusedUs / w0Us);
    }

    // Statistics
    std::vector<double> ratios;
    for (int i = 0; i < ROUNDS; i++) ratios.push_back(fusedTimes[i] / w0Times[i]);
    std::sort(ratios.begin(), ratios.end());
    std::sort(w0Times.begin(), w0Times.end());
    std::sort(fusedTimes.begin(), fusedTimes.end());

    double w0Med = w0Times[ROUNDS/2];
    double fusedMed = fusedTimes[ROUNDS/2];
    double ratioMed = ratios[ROUNDS/2];

    printf("\n=== RESULTS ===\n");
    printf("W0 (original SFA):       median=%.2f µs  [%.2f, %.2f]\n",
           w0Med, w0Times[0], w0Times[ROUNDS-1]);
    printf("Fused (cold-state merge): median=%.2f µs  [%.2f, %.2f]\n",
           fusedMed, fusedTimes[0], fusedTimes[ROUNDS-1]);
    printf("Two-pass (reference):    %.2f µs (existing framework baseline)\n", twoPassUs);
    printf("Fused/W0 ratio:          median=%.3f  [%.3f, %.3f]\n",
           ratioMed, ratios[0], ratios[ROUNDS-1]);
    printf("Fused vs two-pass:      %.2f vs %.2f µs (%.1f%% of two-pass)\n",
           fusedMed, twoPassUs, fusedMed / twoPassUs * 100);

    // Cleanup
    for (auto &a : allocs) aclrtFree(*a.ptr);
    aclrtDestroyEvent(evStart);
    aclrtDestroyEvent(evEnd);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    printf("\nPhase C complete.\n");
    return 0;
}
