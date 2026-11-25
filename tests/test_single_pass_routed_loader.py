"""Host checks of the shipped routed loader, not an AscendC/device emulator.

The DMA model runs actual scalar loader/helper bodies and the VF row-routing
loop. MicroAPI/FP4 arithmetic is stubbed while memory accesses are guarded;
actual MicroAPI arithmetic and synchronization require the A5 tests.
"""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "experimental/single_pass/arch35/kv_quant_sparse_flash_attention_service_vector_mla.h"
_SPEC = importlib.util.spec_from_file_location(
    "routed_actual_loader_extract", ROOT / "tests/test_single_pass_recent_loader.py")
_EXTRACT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXTRACT)


CPP_MODEL = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#define QSFA_RECENT_ROUTE_HOT 1
#define __ubuf__
#define unlikely(x) (x)
using KV_T = int8_t;
using Q_T = uint16_t; // BF16 bits; this model deliberately does no FP4 math.
constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
template<class T> struct Tensor {
    std::vector<uint8_t>* storage = nullptr;
    size_t offset = 0;
    Tensor operator[](int64_t index) const {
        assert(index >= 0 && offset + index * sizeof(T) <= storage->size());
        return {storage, offset + static_cast<size_t>(index) * sizeof(T)};
    }
    T GetValue(int64_t index) const {
        T value;
        const size_t address = offset + index * sizeof(T);
        assert(address + sizeof(T) <= storage->size());
        std::memcpy(&value, storage->data() + address, sizeof(T));
        return value;
    }
    T* GetPhyAddr() const {
        return reinterpret_cast<T*>(storage->data() + offset);
    }
    template<class U> Tensor<U> ReinterpretCast() const { return {storage, offset}; }
};
template<class T> using LocalTensor = Tensor<T>;
template<class T> using GlobalTensor = Tensor<T>;
struct DataCopyExtParams { uint32_t blockCount, blockLen, srcStride, dstStride; };
template<class T> struct DataCopyPadExtParams {
    bool isPad = false;
    uint32_t leftPadding = 0, rightPadding = 0;
    T paddingValue = 0;
};
std::vector<uint8_t>* coldStorage = nullptr;
std::vector<bool> forbiddenPhysical;
uint32_t coldReadBytes = 0, vfReadRows = 0, originalDecodeCalls = 0;
std::vector<uint8_t>* packedStorage = nullptr;
uint32_t allowedColdRows = 0;
template<class T> void DataCopyPad(Tensor<T> dst, Tensor<T> src,
    const DataCopyExtParams& p, const DataCopyPadExtParams<T>& pad) {
    const size_t sourceStep = p.blockLen + p.srcStride;
    const size_t destinationStep = p.blockLen + p.dstStride * 32
        + (pad.isPad ? (pad.leftPadding + pad.rightPadding) * sizeof(T) : 0);
    for (uint32_t block = 0; block < p.blockCount; ++block) {
        const size_t d = dst.offset + block * destinationStep;
        const size_t s = src.offset + block * sourceStep;
        assert(d + p.blockLen <= dst.storage->size());
        assert(s + p.blockLen <= src.storage->size());
        if (src.storage == coldStorage) {
            assert(!forbiddenPhysical.at(s / 400));
            coldReadBytes += p.blockLen;
        }
        std::memcpy(dst.storage->data() + d, src.storage->data() + s, p.blockLen);
        if (pad.isPad) for (uint32_t i = 0; i < pad.rightPadding; ++i)
            std::memcpy(dst.storage->data() + d + p.blockLen + i * sizeof(T),
                        &pad.paddingValue, sizeof(T));
    }
}
struct CopyParams { uint32_t dstBlkStride, srcBlkStride, dstRepStride, srcRepStride; };
template<class T> void Copy(Tensor<T> dst, Tensor<T> src,
    uint32_t mask, uint8_t repeats, CopyParams p) {
    assert(mask % (32 / sizeof(T)) == 0);
    const uint32_t blocks = mask / (32 / sizeof(T));
    for (uint32_t row = 0; row < repeats; ++row) {
        const size_t source = src.offset + row * p.srcRepStride * 32;
        if (src.storage == packedStorage) {
            const uint32_t packedRow = source / 416;
            assert(allowedColdRows & (1u << packedRow));
        }
        for (uint32_t block = 0; block < blocks; ++block) {
            const size_t s = source + block * p.srcBlkStride * 32;
            const size_t d = dst.offset + row * p.dstRepStride * 32
                + block * p.dstBlkStride * 32;
            assert(s + 32 <= src.storage->size() && d + 32 <= dst.storage->size());
            std::memcpy(dst.storage->data() + d, src.storage->data() + s, 32);
        }
    }
}
enum class HardEvent { V_MTE2, MTE2_MTE3 };
int vWaits = 0, dmaWaits = 0;
template<HardEvent E> void SetFlag(int) {}
template<HardEvent E> void WaitFlag(int) {
    if constexpr (E == HardEvent::V_MTE2) ++vWaits; else ++dmaWaits;
}
uint32_t CeilAlign(uint32_t n, uint32_t a) { return (n + a - 1) / a * a; }
struct RunInfo { int64_t boIdx = 0, s1oIdx = 0; };
struct ConstInfo {
    int64_t dSizeVInput = 400, sparseBlockSize = 1, dSizeNope = 512, dSizeRope = 64;
};
// Stub arithmetic only, retaining the actual VF function's loop and hot mask.
// Any scale/tile read from a hot packed row fails before reading its poison.
using bfloat16_t = uint16_t;
struct fp8_e8m0_t { uint8_t bits; };
constexpr int castTraitBIE8M0ToBf16 = 0;
#define __simd_vf__
namespace MicroAPI {
template<class T> struct RegTensor {};
struct MaskReg {};
enum class MaskPattern { ALL };
enum class LoadDist { DIST_UNPACK_B8 };
template<class T, MaskPattern P> MaskReg CreateMask() { return {}; }
template<class... A> void Arange(A&&...) {}
template<class... A> void ShiftRights(A&&...) {}
template<class... A> void Adds(A&&...) {}
template<class T, LoadDist L> void LoadAlign(RegTensor<T>&, uint8_t* address) {
    const size_t offset = address - packedStorage->data();
    assert(offset % 416 == 384 && offset < packedStorage->size());
    const uint32_t row = offset / 416;
    assert(allowedColdRows & (1u << row));
    ++vfReadRows;
}
template<class To, class From, int Trait>
void Cast(RegTensor<To>&, const RegTensor<From>&, const MaskReg&) {}
}
template<class T> void AntiquantBIMxfp4Tile(
    uint8_t* src, T* dst, MicroAPI::RegTensor<bfloat16_t>&,
    MicroAPI::RegTensor<int16_t>&, MicroAPI::MaskReg&) {
    const size_t offset = src - packedStorage->data();
    assert(offset < packedStorage->size());
    const uint32_t row = offset / 416;
    assert(allowedColdRows & (1u << row));
    const uint8_t physicalTag = packedStorage->at(row * 416);
    assert(physicalTag != 0xee);
    for (uint32_t d = 0; d < 128; ++d)
        dst[(d / 16) * 17 * 16 + d % 16] = 9000 + physicalTag;
}
'''


def actual_vf(text):
    """Keep the actual VF scalar loop while the host stubs arithmetic APIs."""
    start = text.index("template <typename Q_T>\n__simd_vf__ void AntiquantVFImplBIMxfp4Routed(")
    brace = text.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


CPP_VECTOR = r'''
struct MockVector {
    int64_t recentTileTokens[16] = {};
    int64_t recentTileSlots[16] = {};
    uint32_t recentHotMask = 0;
    bool recentEnabled = true, recentRouteActive = false;
    int64_t recentRingCapacity = 256, recentWindow = 128;
    int64_t recentQueryPosition = -1, recentQueryFirst = 0;
    int recentVToMte2 = 1, recentMte2ToMte3 = 2;
    Tensor<KV_T> keyGm;
    Tensor<Q_T> recentLatentGm, recentRopeGm;
    Tensor<int64_t> recentTagsGm, recentPositionsGm;
    std::vector<int64_t> physical;
    int64_t GetkeyOffset(int64_t token, const RunInfo&, ConstInfo&) {
        return token < 0 ? -1 : physical.at(token) * 400;
    }
    void CopyInSingleKv(LocalTensor<KV_T>, int64_t, int64_t, uint32_t);
    uint32_t CopyInKvSparse(LocalTensor<KV_T>, int64_t, int64_t, int64_t,
                           const RunInfo&, ConstInfo&);
    void PrepareRecentQuery(const RunInfo&, const ConstInfo&);
    int64_t ResolveRecentHotSlot(int64_t);
    void ReadRecentHotRows(LocalTensor<Q_T>, int64_t);
    void WaitRecentHotRows();
    void DequantKvRouted(LocalTensor<Q_T>, LocalTensor<KV_T>, int64_t, ConstInfo&);
    void DequantKv(LocalTensor<Q_T> dst, LocalTensor<KV_T> src, int64_t rows, ConstInfo&) {
        ++originalDecodeCalls;
        AntiquantVFImplBIMxfp4Routed<Q_T>(reinterpret_cast<uint8_t*>(src.GetPhyAddr()),
                                       dst.GetPhyAddr(), rows, 0);
        Copy(dst[512 * 17], src[256].ReinterpretCast<Q_T>(), 64,
             static_cast<uint8_t>(rows), {17, 1, 1, 13});
    }
};
'''


class RoutedLoaderTest(unittest.TestCase):
    def test_actual_routed_scalar_loader_dma_and_cold_row_contract(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            self.skipTest("host C++ compiler unavailable")
        text = HEADER.read_text()
        names = ("CopyInSingleKv", "CopyInKvSparse", "PrepareRecentQuery",
                 "ResolveRecentHotSlot", "ReadRecentHotRows", "WaitRecentHotRows",
                 "DequantKvRouted")
        bodies = "\n".join(_EXTRACT.actual_function(text, name) for name in names)
        code = CPP_MODEL + actual_vf(text) + CPP_VECTOR + bodies + CPP_CASES
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "routed_actual.cpp"
            binary = Path(temp) / "routed_actual"
            source.write_text(code)
            compile_run = subprocess.run([compiler, "-std=c++17", "-O0", str(source),
                                          "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(compile_run.returncode, 0, compile_run.stderr)
            execution = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(execution.returncode, 0, execution.stderr)


CPP_CASES = r'''
void put16(std::vector<uint8_t>& buffer, size_t address, uint16_t value) {
    assert(address + sizeof(value) <= buffer.size());
    std::memcpy(buffer.data() + address, &value, sizeof(value));
}
void put64(std::vector<uint8_t>& buffer, size_t index, int64_t value) {
    std::memcpy(buffer.data() + index * 8, &value, 8);
}
uint16_t get16(const std::vector<uint8_t>& buffer, size_t address) {
    uint16_t value;
    std::memcpy(&value, buffer.data() + address, 2);
    return value;
}
int main() {
    std::vector<uint8_t> cold(512 * 400), packed(16 * 416);
    std::vector<uint8_t> latent(256 * 512 * 2), rope(256 * 64 * 2), nz(576 * 17 * 2);
    std::vector<uint8_t> tags(256 * 8), positions(8);
    coldStorage = &cold; packedStorage = &packed;
    forbiddenPhysical.resize(512);
    MockVector v;
    v.keyGm = {&cold}; v.recentLatentGm = {&latent}; v.recentRopeGm = {&rope};
    v.recentTagsGm = {&tags}; v.recentPositionsGm = {&positions};
    v.physical.resize(512);
    for (int64_t token = 0; token < 512; ++token) v.physical[token] = token;
    for (int physical = 0; physical < 512; ++physical) {
        const uint8_t tag = physical % 127;
        std::fill(cold.begin() + physical * 400, cold.begin() + (physical + 1) * 400, tag);
        for (int d = 0; d < 64; ++d)
            put16(cold, physical * 400 + 256 + d * 2, 7000 + tag);
    }
    for (int slot = 0; slot < 256; ++slot) {
        for (int d = 0; d < 512; ++d) put16(latent, (slot * 512 + d) * 2, 20000 + slot + d);
        for (int d = 0; d < 64; ++d) put16(rope, (slot * 64 + d) * 2, 40000 + slot + d);
    }
    RunInfo run;
    ConstInfo info;
    const std::vector<std::vector<int64_t>> cases = {
        {-1,-1}, {0, 1}, {100, 101}, {0, 100}, {100, 0}, {100, -1},
        {22, 23}, {150, 151}, {100, 101, 0, 1, 110, 2, 111, 3,
            112, 4, 113, 5, 114, 6, 115, 7},
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
        {110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125}
    };
    // Normal and reverse physical order; stale/missing tags; both extremes of
    // the hot window; all-hot/all-cold/mixed 16-row tiles; a one-row tail.
    for (int order = 0; order < 3; ++order)
    for (int stale = 0; stale < 2; ++stale)
    for (const auto& tokens : cases) {
        put64(positions, 0, 150);
        for (int slot = 0; slot < 256; ++slot) put64(tags, slot, -1);
        for (int token = 23; token <= 150; ++token)
            put64(tags, token % 256, stale && token % 2 ? token + 256 : token);
        for (int token = 0; token < 512; ++token)
            v.physical[token] = order == 1 ? 511 - token : token;
        if (order == 2) v.physical[100] = v.physical[101]; // Native fallback: equal addresses.
        std::fill(packed.begin(), packed.end(), 0xee);
        std::fill(nz.begin(), nz.end(), 0x33);
        std::fill(forbiddenPhysical.begin(), forbiddenPhysical.end(), false);
        allowedColdRows = 0; coldReadBytes = 0; vfReadRows = 0;
        originalDecodeCalls = 0; vWaits = 0; dmaWaits = 0;
        v.recentHotMask = 0;
        v.PrepareRecentQuery(run, info);
        std::vector<int64_t> expected;
        for (size_t i = 0; i < tokens.size(); i += 2) {
            const int64_t a = tokens[i], b = i + 1 < tokens.size() ? tokens[i + 1] : -1;
            if (a < 0 && b < 0) break;
            const bool swap = a >= 0 && b >= 0 && v.physical[b] < v.physical[a];
            expected.push_back(swap ? b : a);
            if (b >= 0) expected.push_back(swap ? a : b);
        }
        uint32_t hotMask = 0;
        for (size_t row = 0; row < expected.size(); ++row) {
            const int64_t token = expected[row];
            const bool hot = token >= 23 && token <= 150 && !(stale && token % 2);
            if (hot) {
                hotMask |= 1u << row;
                forbiddenPhysical[v.physical[token]] = true;
            } else allowedColdRows |= 1u << row;
        }
        // Equal-address cold/hot alias may legitimately read this physical
        // source for its cold row. Total DMA bytes still catch any hot copy.
        for (size_t row = 0; row < expected.size(); ++row)
            if (allowedColdRows & (1u << row)) forbiddenPhysical[v.physical[expected[row]]] = false;
        int rows = 0;
        for (size_t i = 0; i < tokens.size(); i += 2) {
            rows += v.CopyInKvSparse({&packed}, rows, tokens[i],
                i + 1 < tokens.size() ? tokens[i + 1] : -1, run, info);
        }
        assert(rows == expected.size());
        assert(v.recentHotMask == hotMask);
        assert(coldReadBytes == 400 * (rows - __builtin_popcount(hotMask)));
        for (int row = 0; row < rows; ++row) {
            assert(v.recentTileTokens[row] == expected[row]);
            if (hotMask & (1u << row))
                assert(std::all_of(packed.begin() + row * 416,
                    packed.begin() + (row + 1) * 416, [](uint8_t x){return x == 0xee;}));
        }
        v.ReadRecentHotRows({&nz}, rows);
        v.DequantKvRouted({&nz}, {&packed}, rows, info);
        v.WaitRecentHotRows();
        assert(vfReadRows == rows - __builtin_popcount(hotMask));
        assert(originalDecodeCalls == (hotMask == 0));
        assert(vWaits == bool(hotMask) && dmaWaits == bool(hotMask));
        // Check all features in every row, including untouched pad row16.
        for (int row = 0; row < 17; ++row) for (int d = 0; d < 576; ++d) {
            uint16_t wanted = 0x3333;
            if (row < rows) {
                const int64_t token = expected[row];
                if (hotMask & (1u << row)) {
                    const int slot = token % 256;
                    wanted = d < 512 ? 20000 + slot + d : 40000 + slot + d - 512;
                } else wanted = (d < 512 ? 9000 : 7000) + v.physical[token] % 127;
            }
            const size_t address = ((d / 16) * 17 * 16 + row * 16 + d % 16) * 2;
            assert(get16(nz, address) == wanted);
        }
    }
    // Ring wrap: a current absolute token must resolve its modulo slot, not a
    // same-shaped older occupant. Position is cached once per query run.
    put64(positions, 0, 400);
    put64(tags, 390 % 256, 390);
    v.PrepareRecentQuery(run, info);
    assert(v.ResolveRecentHotSlot(390) == 390 % 256);
    put64(positions, 0, 900); // Must not change the already prepared query.
    assert(v.ResolveRecentHotSlot(390) == 390 % 256);
    put64(tags, 390 % 256, 390 - 256);
    assert(v.ResolveRecentHotSlot(390) < 0);
    assert(v.ResolveRecentHotSlot(272) < 0 && v.ResolveRecentHotSlot(401) < 0);
    v.recentEnabled = false;
    v.PrepareRecentQuery(run, info);
    assert(!v.recentRouteActive);
    std::fill(forbiddenPhysical.begin(), forbiddenPhysical.end(), false);
    v.recentHotMask = 0;
    allowedColdRows = 3;
    coldReadBytes = vfReadRows = originalDecodeCalls = 0;
    vWaits = dmaWaits = 0;
    assert(v.CopyInKvSparse({&packed}, 0, 100, 101, run, info) == 2);
    assert(coldReadBytes == 800 && v.recentHotMask == 0);
    v.ReadRecentHotRows({&nz}, 2);
    v.DequantKvRouted({&nz}, {&packed}, 2, info);
    v.WaitRecentHotRows();
    assert(vfReadRows == 2 && originalDecodeCalls == 1 && vWaits == 0 && dmaWaits == 0);
}
'''


if __name__ == "__main__":
    unittest.main()
