"""Actual routed loader/VF placement/UB-to-L1 bodies for 8/16/32 load rows.

Host DMA and allocation accounting only: no claim about SDK resource peaks,
hardware events, FP4 arithmetic or device performance.
"""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("load_tiles_host_model",
    ROOT / "tests/test_single_pass_routed_loader.py")
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)
CONFIG = ROOT / "experimental/single_pass/load_tile_config.h"
KERNEL = ROOT / "experimental/single_pass/arch35/kv_quant_sparse_flash_attention_kernel_mla.h"


def kernel_allocation_body(text):
    start = text.index("template <typename CubeBlockType, typename VecBlockType>\n"
                       "__aicore__ inline void Recent128KvQuantSparseFlashAttentionMla<"
                       "CubeBlockType, VecBlockType>::InitMMResBuf(")
    brace = text.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end].replace("template <typename CubeBlockType, typename VecBlockType>", "").replace(
        "Recent128KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::", "MockKernel::").replace(
        "__aicore__", "").replace("__gm__", "")


SUPPORT = r'''
using T = float;
constexpr int CV_RATIO = 2;
constexpr bool IS_SPLIT_G = false;
#define ASCEND_IS_AIC (false)
#define ASCEND_IS_AIV (true)
constexpr int INVALID_CROSS_CORE_EVENT_ID = -1;
int Align16Func(int n) { return (n + 15) / 16 * 16; }
enum class BufferType { L1, GM };
enum class SyncType { CROSS_CORE_SYNC_FORWARD, CROSS_CORE_SYNC_BACKWARD };
template<BufferType, SyncType> struct Buffer {
    std::vector<uint8_t>* bytes;
    template<class U> Tensor<U> GetTensor() { return {bytes}; }
};
struct DataCopyParams { uint32_t blockCount, blockLen, srcGap, dstGap; };
template<class U> void DataCopy(Tensor<U> dst, Tensor<U> src, DataCopyParams p) {
    assert(p.srcGap == EXPECTED_NZ - p.blockLen);
    const size_t srcStep = (p.blockLen + p.srcGap) * 32;
    const size_t dstStep = (p.blockLen + p.dstGap) * 32;
    for (uint32_t block = 0; block < p.blockCount; ++block) {
        const size_t source = src.offset + block * srcStep;
        const size_t destination = dst.offset + block * dstStep;
        assert(source + p.blockLen * 32 <= src.storage->size());
        assert(destination + p.blockLen * 32 <= dst.storage->size());
        std::memcpy(dst.storage->data() + destination,
                    src.storage->data() + source, p.blockLen * 32);
    }
}
struct Buf {
    std::vector<uint8_t> bytes;
    template<class U> Tensor<U> Get() { return {&bytes}; }
};
struct Queue {};
struct TPipe {
    uint64_t ubBytes = 0;
    void InitBuffer(Buf& b, uint32_t size) { ubBytes += size; b.bytes.resize(size); }
    void InitBuffer(Queue&, uint32_t count, uint32_t size) { ubBytes += count * size; }
    template<HardEvent E> int AllocEventID() { return 0; }
};
TPipe* selectedPipe = nullptr;
TPipe* GetTPipePtr() { return selectedPipe; }
struct Manager {
    bool ub;
    void Init(TPipe* pipe, uint32_t bytes) { if (ub) pipe->ubBytes += bytes; }
    void Init(uint8_t*) {}
};
struct Policy {
    template<class M> void Init(M&, uint32_t) {}
    Policy& Get() { return *this; }
    void SetCrossCoreID(int, int) {}
    void SetCrossCore() {}
};
struct MockKernel {
    ConstInfo constInfo;
    TPipe* pipe;
    Manager l1BufferManager{false}, ubBufferManager{true}, gmBufferManager{false};
    Policy l1RightBuffers, bmm1Buffers, bmm2Buffers, v0ResGmBuffers;
    int crossCoreSyncBufId = 0, aicIdx = 0;
    void InitMMResBuf(uint8_t*);
};
'''

VECTOR_EXTRA = r'''
    static constexpr uint32_t dVTemplateTypeInput = QSFA_RECENT_INPUT_ROW_BYTES;
    static constexpr uint32_t vec1Srcstride = 33, s2BaseSize = 128, s1BaseSize = 64;
    static constexpr uint32_t qsfaDTemplateAlign64 = 512;
    TPipe* tPipe;
    Buf softmaxSumBuf[2], softmaxMaxBuf[2], softmaxExpBuf[2], commonTBuf, lseBuf, stage2OutBuf;
    Tensor<float> lseUb;
    Queue stage0InQue, stage0OutQue, stage1OutQue[2];
    int mte3ToVId[2], vToMte3Id[2], mte3ToVLseOutId, vToMte3LseOutId;
    void SoftmaxInitBuffer();
    void InitLocalBuffer(TPipe*, ConstInfo&);
    void CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>&,
        LocalTensor<Q_T>, int64_t, int64_t, const RunInfo&, ConstInfo&);
    void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>&,
        LocalTensor<Q_T>, int64_t, int64_t, const RunInfo&, ConstInfo&);
'''


class LoadTilesTest(unittest.TestCase):
    def test_actual_load_decode_copyout_and_complete_explicit_ub_budget(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            self.skipTest("No host C++ compiler")
        text = MODEL.HEADER.read_text()
        names = ("CopyInSingleKv", "CopyInKvSparse", "PrepareRecentQuery",
                 "ResolveRecentHotSlot", "ReadRecentHotRows", "WaitRecentHotRows", "DequantKvRouted",
                 "CopyOutKvUb2L1", "CopyOutKvUb2Gm", "SoftmaxInitBuffer", "InitLocalBuffer")
        bodies = "\n".join(MODEL._EXTRACT.actual_function(text, name) for name in names)
        prefix = MODEL.CPP_MODEL.replace("HardEvent { V_MTE2, MTE2_MTE3 }",
            "HardEvent { V_MTE2, MTE2_MTE3, MTE3_V, V_MTE3 }").replace(
            "struct RunInfo { int64_t boIdx = 0, s1oIdx = 0; };",
            "struct RunInfo { int64_t boIdx = 0, s1oIdx = 0, s2RealSize = 128; };").replace(
            "int64_t dSizeVInput = 400, sparseBlockSize = 1, dSizeNope = 512, dSizeRope = 64;",
            "int64_t dSizeVInput = 400, sparseBlockSize = 1, dSizeNope = 512, dSizeRope = 64;"
            " int64_t s1BaseSize = 64, s2BaseSize = 128;").replace(
            "(d / 16) * 17 * 16", "(d / 16) * EXPECTED_NZ * 16")
        vector = MODEL.CPP_VECTOR.replace("struct MockVector {", "struct MockVector {" + VECTOR_EXTRA).replace(
            "int64_t recentTileTokens[16]", "int64_t recentTileTokens[EXPECTED_ROWS]").replace(
            "int64_t recentTileSlots[16]", "int64_t recentTileSlots[EXPECTED_ROWS]").replace(
            "uint32_t recentHotMask = 0;", "uint64_t slotsGuard = 0x123456789abcdef0ULL; uint32_t recentHotMask = 0;").replace(
            "dst[512 * 17]", "dst[512 * EXPECTED_NZ]").replace("{17, 1, 1, 13}", "{EXPECTED_NZ, 1, 1, 13}")
        configurations = ((8, 672, 2, 2, 214784), (16, 416, 2, 2, 235776),
                          (16, 416, 2, 1, 216192), (32, 416, 2, 1, 247936),
                          (32, 416, 1, 1, 234624))
        for rows, stride, ins, outs, total in configurations:
            with self.subTest(rows=rows, stride=stride, ins=ins, outs=outs), tempfile.TemporaryDirectory() as temp:
                source, binary = Path(temp) / "load_tile.cpp", Path(temp) / "load_tile"
                source.write_text(f"""
#define QSFA_RECENT_LOAD_ROWS {rows}
#define QSFA_RECENT_INPUT_ROW_BYTES {stride}
#define QSFA_RECENT_INPUT_BUFFERS {ins}
#define QSFA_RECENT_OUTPUT_BUFFERS {outs}
#define QSFA_RECENT_ROUTE_FAST_PAIR 1
#define QSFA_RECENT_EARLY_HOT 0
#include "{CONFIG}"
constexpr int EXPECTED_ROWS = {rows}, EXPECTED_NZ = {rows + 1};
constexpr unsigned long long EXPECTED_UB = {total};
""" + prefix + MODEL.actual_vf(text) + SUPPORT + vector + bodies
                    + kernel_allocation_body(KERNEL.read_text()) + CASES)
                result = subprocess.run([compiler, "-std=c++17", "-O0", "-fsanitize=undefined",
                    "-fno-sanitize-recover=all", str(source), "-o", str(binary)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                run = subprocess.run([str(binary)], capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)


CASES = r'''
void put16(std::vector<uint8_t>& b, size_t at, uint16_t value) { std::memcpy(b.data() + at, &value, 2); }
void put64(std::vector<uint8_t>& b, size_t index, int64_t value) { std::memcpy(b.data() + index * 8, &value, 8); }
uint16_t get16(const std::vector<uint8_t>& b, size_t at) { uint16_t x; std::memcpy(&x,b.data()+at,2); return x; }
int main() {
    TPipe pipe;
    selectedPipe = &pipe;
    ConstInfo info;
    MockVector v;
    v.tPipe = &pipe;
    v.InitLocalBuffer(&pipe, info);
    const uint64_t vectorBytes = 84992 + EXPECTED_ROWS * QSFA_RECENT_INPUT_ROW_BYTES * QSFA_RECENT_INPUT_BUFFERS
        + 576 * EXPECTED_NZ * 2 * QSFA_RECENT_OUTPUT_BUFFERS;
    assert(pipe.ubBytes == vectorBytes);
    MockKernel kernel;
    kernel.pipe = &pipe;
    kernel.InitMMResBuf(nullptr);
    assert(pipe.ubBytes - vectorBytes == 98304 && pipe.ubBytes == EXPECTED_UB);

    std::vector<uint8_t> cold(512 * 400), packed(EXPECTED_ROWS * 416);
    std::vector<uint8_t> latent(256 * 512 * 2), rope(256 * 64 * 2), nz(576 * EXPECTED_NZ * 2);
    std::vector<uint8_t> tags(256 * 8), positions(8), l1(576 * 128 * 2), gm(576 * 128 * 2);
    coldStorage = &cold; packedStorage = &packed; tagStorage = &tags;
    forbiddenPhysical.resize(512);
    v.keyGm = {&cold}; v.recentLatentGm = {&latent}; v.recentRopeGm = {&rope};
    v.recentTagsGm = {&tags}; v.recentPositionsGm = {&positions};
    v.physical.resize(512);
    for (int physical = 0; physical < 512; ++physical) {
        std::fill(cold.begin() + physical * 400, cold.begin() + (physical + 1) * 400, physical % 127);
        for (int d = 0; d < 64; ++d) put16(cold, physical * 400 + 256 + d * 2, 7000 + physical % 127);
    }
    for (int slot = 0; slot < 256; ++slot) {
        for (int d = 0; d < 512; ++d) put16(latent, (slot * 512 + d) * 2, 20000 + slot + d);
        for (int d = 0; d < 64; ++d) put16(rope, (slot * 64 + d) * 2, 40000 + slot + d);
    }
    RunInfo run;
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> outL1{&l1};
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> outGm{&gm};
    for (int count : {1, EXPECTED_ROWS-1, EXPECTED_ROWS, EXPECTED_ROWS+1, EXPECTED_ROWS*2+3})
    for (int kind = 0; kind < 3; ++kind)
    for (int reverse = 0; reverse < 2; ++reverse) {
        std::vector<int64_t> ids, expected;
        for (int i = 0; i < count; ++i) ids.push_back(kind == 0 ? i : kind == 1 ? 80+i : (i%2 ? 80+i/2 : i/2));
        put64(positions, 0, 200);
        for (int slot = 0; slot < 256; ++slot) put64(tags, slot, -1);
        for (int token = 73; token <= 200; ++token) put64(tags, token % 256, token);
        for (int token = 0; token < 512; ++token) v.physical[token] = reverse ? 511 - token : token;
        std::fill(forbiddenPhysical.begin(), forbiddenPhysical.end(), false);
        for (size_t at = 0; at < ids.size(); at += 2) {
            const int64_t a = ids[at], b = at+1 < ids.size() ? ids[at+1] : -1;
            const bool swap = b >= 0 && v.physical[b] < v.physical[a];
            expected.push_back(swap ? b : a);
            if (b >= 0) expected.push_back(swap ? a : b);
        }
        for (int64_t token : expected) if (token >= 73 && token <= 200) forbiddenPhysical[v.physical[token]] = true;
        std::fill(l1.begin(), l1.end(), 0x55); std::fill(gm.begin(), gm.end(), 0x55);
        v.PrepareRecentQuery(run, info);
        size_t at = 0;
        while (at < ids.size()) {
            const int start = at;
            int rows = 0;
            v.recentHotMask = 0;
            std::fill(v.recentTileSlots, v.recentTileSlots + EXPECTED_ROWS, -888);
            std::fill(packed.begin(), packed.end(), 0xee); std::fill(nz.begin(), nz.end(), 0x33);
            allowedColdRows = coldReadBytes = vfReadRows = originalDecodeCalls = 0;
            while (rows < EXPECTED_ROWS && at < ids.size()) {
                rows += v.CopyInKvSparse({&packed}, rows, ids[at], at+1 < ids.size() ? ids[at+1] : -1, run, info);
                at += 2;
            }
            const uint32_t validMask = rows == 32 ? 0xffffffffu : (1u << rows) - 1;
            allowedColdRows = validMask & ~v.recentHotMask;
            v.ReadRecentHotRows({&nz}, rows);
            v.DequantKvRouted({&nz}, {&packed}, rows, info);
            v.WaitRecentHotRows();
            assert(v.slotsGuard == 0x123456789abcdef0ULL);
            assert(coldReadBytes == __builtin_popcount(allowedColdRows) * 400);
            assert(vfReadRows == __builtin_popcount(allowedColdRows));
            if (rows == 32 && v.recentHotMask == 0xffffffffu) assert(originalDecodeCalls == 0 && vfReadRows == 0);
            v.CopyOutKvUb2L1(outL1, {&nz}, rows, start, run, info);
            v.CopyOutKvUb2Gm(outGm, {&nz}, rows, start, run, info);
            // The pad row and all unused tail rows stay untouched.
            for (int row = rows; row < EXPECTED_NZ; ++row) for (int d = 0; d < 576; ++d)
                assert(get16(nz, ((d/16)*EXPECTED_NZ*16 + row*16 + d%16)*2) == 0x3333);
        }
        assert(l1 == gm);
        for (int row = 0; row < 128; ++row) for (int d = 0; d < 576; ++d) {
            uint16_t wanted = 0x5555;
            if (row < expected.size()) {
                const int token = expected[row];
                const bool hot = token >= 73 && token <= 200;
                wanted = hot ? (d < 512 ? 20000+token+d : 40000+token+d-512)
                    : (d < 512 ? 9000 : 7000) + v.physical[token] % 127;
            }
            assert(get16(l1, ((d/16)*128*16 + row*16 + d%16)*2) == wanted);
        }
    }
}
'''


if __name__ == "__main__":
    unittest.main()
