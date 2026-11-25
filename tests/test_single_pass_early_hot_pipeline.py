"""Compile actual ProcessSparseKv against an ordered host DMA/queue model.

Pending MTE2 writes become visible only at the mocked input queue join or the
explicit old hot wait. This checks source ordering and row preservation, not
real AscendC event semantics, device arithmetic, compilation or performance.
"""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "early_hot_routed_model", ROOT / "tests/test_single_pass_routed_loader.py")
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


def replace_function(text, needle, replacement):
    start = text.index(needle)
    brace = text.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[:start] + replacement(text[start:end]) + text[end:]


STATE = r'''
#include <cassert>
#include <functional>
#include <string>
#include <vector>
std::vector<std::function<void()>> pendingWrites;
std::vector<std::string> trace;
bool outputMte2Owned = false, inputJoined = false;
int inputEnqueueAt = -1, inputDequeueAt = -1, outputAllocAt = -1, firstHotAt = -1;
int decodeAt = -1, tileWaitStart = 0;
int mark(const char* name) { trace.emplace_back(name); return trace.size() - 1; }
void flushMte2() {
    for (auto& write : pendingWrites) write();
    pendingWrites.clear();
}
'''

ASYNC_DMA = r'''
template<class T> void DataCopyPad(Tensor<T> dst, Tensor<T> src,
    const DataCopyExtParams& p, const DataCopyPadExtParams<T>& pad) {
    if (src.storage != coldStorage) {
        assert(outputMte2Owned); // Allocation ownership must cross V->MTE2.
        const int at = mark("hot_dma_issue");
        if (firstHotAt < 0) firstHotAt = at;
    } else mark("cold_dma_issue");
    pendingWrites.emplace_back([=]{DataCopyPadImmediate(dst, src, p, pad);});
}
'''

QUEUES = r'''
enum class BufferType { L1, GM };
enum class SyncType { CROSS_CORE_SYNC_FORWARD, CROSS_CORE_SYNC_BACKWARD };
template<BufferType, SyncType> struct Buffer {};
constexpr bool IS_SPLIT_G = false;
int64_t Min(int64_t a, int64_t b) { return a < b ? a : b; }
std::vector<uint8_t>* nzStorage = nullptr;
struct MockQueue {
    bool input;
    template<class T> LocalTensor<T> AllocTensor() {
        if (input) {
            assert(pendingWrites.empty());
            mark("input_alloc");
            inputJoined = false;
            firstHotAt = inputEnqueueAt = inputDequeueAt = outputAllocAt = decodeAt = -1;
            tileWaitStart = dmaWaits;
            std::fill(packedStorage->begin(), packedStorage->end(), 0xee);
            return {packedStorage};
        }
        outputAllocAt = mark("output_alloc");
        // Model VECOUT's previous MTE3->V buffer reuse ownership. MTE2 must
        // still wait for its own V->MTE2 handoff before writing this buffer.
        outputMte2Owned = false;
        std::fill(nzStorage->begin(), nzStorage->end(), 0x33);
        return {nzStorage};
    }
    template<class T> void EnQue(LocalTensor<T>) {
        if (input) inputEnqueueAt = mark("input_enqueue");
        else mark("output_enqueue");
    }
    template<class T> LocalTensor<T> DeQue() {
        if (input) {
            inputDequeueAt = mark("input_dequeue");
            flushMte2();
            inputJoined = true;
            return {packedStorage};
        }
        mark("output_dequeue");
        // Output's V->MTE3 join does not magically finish foreign MTE2 writes.
        assert(pendingWrites.empty());
        return {nzStorage};
    }
    template<class T> void FreeTensor(LocalTensor<T>) {
        mark(input ? "input_free" : "output_free");
    }
};
'''

VECTOR_ADDITIONS = r'''
    MockQueue stage0InQue{true}, stage0OutQue{false};
    int64_t qsfaSparseCalSize = 0, sparseS2Start = 0, sparseS2End = 0;
    std::vector<int64_t> ids;
    std::vector<uint16_t> result;
    void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>&,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>&,
        const RunInfo&, ConstInfo&);
    void GetRealCmpS2Idx(int64_t& a, int64_t& b, int64_t start,
        const RunInfo&, ConstInfo&) {
        a = start < ids.size() ? ids[start] : -1;
        b = start + 1 < ids.size() ? ids[start + 1] : -1;
    }
    void SaveOutput(LocalTensor<Q_T> tile, int64_t rows, int64_t start) {
        assert(pendingWrites.empty() && inputJoined);
        assert(decodeAt > inputDequeueAt && result.size() == start * 576);
        if (firstHotAt >= 0) {
            assert(outputAllocAt < firstHotAt);
#if QSFA_RECENT_EARLY_HOT
            assert(firstHotAt < inputEnqueueAt && inputEnqueueAt < inputDequeueAt);
            assert(dmaWaits == tileWaitStart); // No redundant explicit hot join.
#else
            assert(outputAllocAt > inputDequeueAt && firstHotAt > inputDequeueAt);
            assert(dmaWaits == tileWaitStart + 1);
#endif
        } else {
            assert(outputAllocAt > inputDequeueAt); // Original cold-only path.
            assert(dmaWaits == tileWaitStart);
        }
        for (int64_t row = 0; row < rows; ++row)
            for (int64_t d = 0; d < 576; ++d)
                result.push_back(tile.GetValue((d / 16) * 17 * 16 + row * 16 + d % 16));
    }
    void CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>&,
        LocalTensor<Q_T> tile, int64_t rows, int64_t start, const RunInfo&, ConstInfo&) {
        SaveOutput(tile, rows, start);
    }
    void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>&,
        LocalTensor<Q_T> tile, int64_t rows, int64_t start, const RunInfo&, ConstInfo&) {
        SaveOutput(tile, rows, start);
    }
'''


class EarlyHotPipelineTest(unittest.TestCase):
    def test_actual_process_sparse_kv_pending_dma_order_and_data(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            self.skipTest("No host C++ compiler")
        text = MODEL.HEADER.read_text()
        names = ("CopyInSingleKv", "CopyInKvSparse", "PrepareRecentQuery",
                 "ResolveRecentHotSlot", "ReadRecentHotRows", "WaitRecentHotRows",
                 "DequantKvRouted", "ProcessSparseKv")
        actual = []
        for name in names:
            body = MODEL._EXTRACT.actual_function(text, name)
            if name == "DequantKvRouted":
                body = body.replace("{", """{
                    decodeAt = mark("decode");
                    assert(inputJoined);
#if QSFA_RECENT_EARLY_HOT
                    assert(pendingWrites.empty());
#endif
                    allowedColdRows = ((uint32_t(1) << rows) - 1) & ~recentHotMask;
""", 1)
            actual.append(body)
        prefix = MODEL.CPP_MODEL
        prefix = replace_function(prefix, "template<class T> void DataCopyPad(",
            lambda old: old.replace("void DataCopyPad(", "void DataCopyPadImmediate(", 1) + ASYNC_DMA)
        prefix = replace_function(prefix, "template<HardEvent E> void WaitFlag(int)", lambda old: r'''
template<HardEvent E> void WaitFlag(int) {
    if constexpr (E == HardEvent::V_MTE2) {
        ++vWaits;
        mark("v_mte2");
        outputMte2Owned = true;
    } else {
        ++dmaWaits;
        mark("mte2_mte3");
        flushMte2();
    }
}
''')
        vector = MODEL.CPP_VECTOR.replace("struct MockVector {", "struct MockVector {" + VECTOR_ADDITIONS)
        for early_hot in (0, 1):
            with self.subTest(early_hot=early_hot), tempfile.TemporaryDirectory() as temp:
                source = Path(temp) / "actual_pipeline.cpp"
                binary = Path(temp) / "actual_pipeline"
                source.write_text(f"#define QSFA_RECENT_EARLY_HOT {early_hot}\n"
                    + "#define QSFA_RECENT_ROUTE_FAST_PAIR 1\n" + STATE + prefix
                    + MODEL.actual_vf(text) + QUEUES + vector + "\n".join(actual) + CASES)
                compile_run = subprocess.run([compiler, "-std=c++17", "-O0", str(source), "-o", str(binary)],
                                             capture_output=True, text=True)
                self.assertEqual(compile_run.returncode, 0, compile_run.stderr)
                run = subprocess.run([str(binary)], capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)


CASES = r'''
void put16(std::vector<uint8_t>& b, size_t at, uint16_t value) {
    std::memcpy(b.data() + at, &value, 2);
}
void put64(std::vector<uint8_t>& b, size_t index, int64_t value) {
    std::memcpy(b.data() + index * 8, &value, 8);
}
int main() {
    std::vector<uint8_t> cold(512 * 400), packed(16 * 416);
    std::vector<uint8_t> latent(256 * 512 * 2), rope(256 * 64 * 2), nz(576 * 17 * 2);
    std::vector<uint8_t> tags(256 * 8), positions(8);
    coldStorage = &cold; packedStorage = &packed; tagStorage = &tags; nzStorage = &nz;
    forbiddenPhysical.resize(512);
    MockVector v;
    v.keyGm = {&cold}; v.recentLatentGm = {&latent}; v.recentRopeGm = {&rope};
    v.recentTagsGm = {&tags}; v.recentPositionsGm = {&positions};
    v.physical.resize(512);
    for (int physical = 0; physical < 512; ++physical) {
        std::fill(cold.begin() + physical * 400, cold.begin() + (physical + 1) * 400,
                  physical % 127);
        for (int d = 0; d < 64; ++d) put16(cold, physical * 400 + 256 + d * 2, 7000 + physical % 127);
    }
    for (int slot = 0; slot < 256; ++slot) {
        for (int d = 0; d < 512; ++d) put16(latent, (slot * 512 + d) * 2, 20000 + slot + d);
        for (int d = 0; d < 64; ++d) put16(rope, (slot * 64 + d) * 2, 40000 + slot + d);
    }
    const std::vector<std::vector<int64_t>> cases = {
        {-1,-1}, {0,-1}, {100,-1}, {0,100,101,-1},
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
        {100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115},
        {0,100,1,101,2,102,3,103,4,104,5,105,6,106,7,107,108,109},
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,100,101},
        {100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,0,1}
    };
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> output;
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> scratch;
    RunInfo run;
    ConstInfo info;
    for (int reverse = 0; reverse < 2; ++reverse)
    for (int stale = 0; stale < 2; ++stale)
    for (const auto& ids : cases) {
        assert(pendingWrites.empty());
        put64(positions, 0, 150);
        for (int slot = 0; slot < 256; ++slot) put64(tags, slot, -1);
        for (int token = 23; token <= 150; ++token)
            put64(tags, token % 256, stale && token % 2 ? token + 256 : token);
        for (int token = 0; token < 512; ++token)
            v.physical[token] = reverse ? 511 - token : token;
        std::fill(forbiddenPhysical.begin(), forbiddenPhysical.end(), false);
        std::vector<int64_t> expected;
        for (size_t at = 0; at < ids.size(); at += 2) {
            const int64_t a = ids[at], b = at + 1 < ids.size() ? ids[at + 1] : -1;
            if (a < 0) break;
            const bool swap = b >= 0 && v.physical[b] < v.physical[a];
            expected.push_back(swap ? b : a);
            if (b >= 0) expected.push_back(swap ? a : b);
        }
        int coldRows = 0;
        for (int64_t token : expected) {
            const bool hot = token >= 23 && token <= 150 && !(stale && token % 2);
            forbiddenPhysical[v.physical[token]] = hot;
            coldRows += !hot;
        }
        coldReadBytes = vfReadRows = originalDecodeCalls = 0;
        vWaits = dmaWaits = 0;
        trace.clear(); v.result.clear(); v.ids = ids;
        v.qsfaSparseCalSize = ids.size(); v.sparseS2Start = 0; v.sparseS2End = ids.size();
        v.ProcessSparseKv(output, scratch, run, info);
        assert(pendingWrites.empty());
        assert(coldReadBytes == coldRows * 400 && vfReadRows == coldRows);
        assert(v.result.size() == expected.size() * 576);
        for (size_t row = 0; row < expected.size(); ++row) {
            const int64_t token = expected[row];
            const bool hot = token >= 23 && token <= 150 && !(stale && token % 2);
            for (int d = 0; d < 576; ++d) {
                const uint16_t wanted = hot ? (d < 512 ? 20000 + token + d : 40000 + token + d - 512)
                    : (d < 512 ? 9000 : 7000) + v.physical[token] % 127;
                assert(v.result.at(row * 576 + d) == wanted);
            }
        }
    }
}
'''


if __name__ == "__main__":
    unittest.main()
