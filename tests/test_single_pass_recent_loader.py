"""Host-only byte-DMA checks of actual candidate loader/overlay function bodies.

This does not compile AscendC or emulate C4 arithmetic. It checks the important
pair-order and NZ hot-row placement contracts using the source we ship.
"""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADER = ROOT / 'experimental/single_pass/arch35/kv_quant_sparse_flash_attention_service_vector_mla.h'


def actual_function(text, name):
    needle = 'Recent128QSFAVectorService<TEMPLATE_ARGS>::' + name + '('
    position = text.index(needle)
    start = text.rfind('TEMPLATES_DEF_NO_DEFAULT', 0, position)
    brace = text.index('{', position)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return (text[start:end].replace('TEMPLATES_DEF_NO_DEFAULT', '')
            .replace('__aicore__', '').replace('inline', '')
            .replace('Recent128QSFAVectorService<TEMPLATE_ARGS>::', 'MockVector::'))


class TestSinglePassRecentLoader(unittest.TestCase):
    def test_actual_pair_loader_and_nz_overlay_bytes(self):
        compiler = shutil.which('clang++') or shutil.which('c++')
        if not compiler:
            self.skipTest('host C++ compiler unavailable')
        text = HEADER.read_text()
        bodies = '\n'.join(actual_function(text, name) for name in (
            'CopyInSingleKv', 'CopyInKvSparse', 'OverlayRecentHotRows'))
        code = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <climits>
#include <vector>
using KV_T = int8_t;
using Q_T = uint16_t; // BF16 bits only: no simulated arithmetic/SDK claim.
constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
template<class T> struct Tensor {
    std::vector<T>* storage = nullptr;
    size_t offset = 0;
    Tensor operator[](int64_t index) const {
        assert(index >= 0 && offset + index <= storage->size());
        return {storage, offset + static_cast<size_t>(index)};
    }
    T GetValue(int64_t index) const { return storage->at(offset + index); }
};
template<class T> using LocalTensor = Tensor<T>;
struct DataCopyExtParams { uint32_t blockCount, blockLen, srcStride, dstStride; };
template<class T> struct DataCopyPadExtParams {
    bool isPad = false;
    uint32_t leftPadding = 0, rightPadding = 0;
    T paddingValue = 0;
};
template<class T> void DataCopyPad(Tensor<T> dst, Tensor<T> src,
    const DataCopyExtParams& p, const DataCopyPadExtParams<T>& pad) {
    assert(p.blockLen % sizeof(T) == 0 && p.srcStride % sizeof(T) == 0);
    const size_t count = p.blockLen / sizeof(T);
    const size_t sourceStep = count + p.srcStride / sizeof(T);
    const size_t destinationStep = count + p.dstStride * 32 / sizeof(T)
        + (pad.isPad ? pad.leftPadding + pad.rightPadding : 0);
    for (size_t block = 0; block < p.blockCount; ++block) {
        const size_t d = dst.offset + block * destinationStep;
        const size_t s = src.offset + block * sourceStep;
        for (size_t i = 0; i < count; ++i)
            dst.storage->at(d + i) = src.storage->at(s + i);
        if (pad.isPad) for (size_t i = 0; i < pad.rightPadding; ++i)
            dst.storage->at(d + count + i) = pad.paddingValue;
    }
}
enum class HardEvent { V_MTE2, MTE2_MTE3 };
int vWaits = 0, dmaWaits = 0;
template<HardEvent E> void SetFlag(int) {}
template<HardEvent E> void WaitFlag(int) {
    if constexpr (E == HardEvent::V_MTE2) ++vWaits; else ++dmaWaits;
}
#define unlikely(x) (x)
uint32_t CeilAlign(uint32_t n, uint32_t a) { return (n + a - 1) / a * a; }
struct RunInfo { int64_t boIdx = 0, s1oIdx = 0; };
struct ConstInfo {
    int64_t dSizeVInput = 400, sparseBlockSize = 1, dSizeNope = 512, dSizeRope = 64;
};
struct MockVector {
    int64_t recentTileTokens[16] = {};
    bool recentEnabled = true;
    int64_t recentRingCapacity = 256, recentWindow = 128;
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
    void OverlayRecentHotRows(LocalTensor<Q_T>, int64_t, const RunInfo&, const ConstInfo&);
};
''' + bodies + r'''
int main() {
    std::vector<KV_T> cold(256 * 400), packed(16 * 416);
    std::vector<Q_T> latent(256 * 512), rope(256 * 64), nz(576 * 17);
    std::vector<int64_t> tags(256, -1), positions{150};
    MockVector v;
    v.keyGm = {&cold}; v.recentLatentGm = {&latent}; v.recentRopeGm = {&rope};
    v.recentTagsGm = {&tags}; v.recentPositionsGm = {&positions};
    v.physical.resize(256);
    for (int64_t i = 0; i < 256; ++i) {
        v.physical[i] = i;
        std::fill(cold.begin() + i * 400, cold.begin() + (i + 1) * 400,
                  static_cast<KV_T>(i % 127));
        for (int d = 0; d < 512; ++d) latent[i * 512 + d] = i * 128 + d;
        for (int d = 0; d < 64; ++d) rope[i * 64 + d] = 40000 + i * 64 + d;
    }
    RunInfo run;
    ConstInfo info;
    // Native coalesced ascending and descending physical pages, plus duplicate
    // physical addresses forcing the two-single-DMA fallback and a tail token.
    for (int scenario = 0; scenario < 4; ++scenario) {
        v.physical[100] = scenario == 1 ? 7 : 3;
        v.physical[101] = scenario == 2 ? 3 : 5;
        const int64_t second = scenario == 3 ? -1 : 101;
        const int count = v.CopyInKvSparse({&packed}, 0, 100, second, run, info);
        assert(count == (scenario == 3 ? 1 : 2));
        assert(v.recentTileTokens[0] == (scenario == 1 ? 101 : 100));
        for (int row = 0; row < count; ++row)
            assert(packed.at(row * 416) == v.physical[v.recentTileTokens[row]] % 127);
        for (int hotMask = 0; hotMask < 4; ++hotMask) {
            std::fill(tags.begin(), tags.end(), -1);
            if (hotMask & 1) tags[100] = 100;
            if (hotMask & 2) tags[101] = 101;
            std::fill(nz.begin(), nz.end(), 0x3333);
            vWaits = dmaWaits = 0;
            v.OverlayRecentHotRows({&nz}, count, run, info);
            bool hasHot = false;
            for (int row = 0; row < 17; ++row) for (int d = 0; d < 576; ++d) {
                const size_t at = (d / 16) * 17 * 16 + row * 16 + d % 16;
                Q_T expected = 0x3333;
                if (row < count) {
                    const int token = v.recentTileTokens[row];
                    if (tags[token] == token) {
                        hasHot = true;
                        expected = d < 512 ? latent[token * 512 + d]
                                          : rope[token * 64 + d - 512];
                    }
                }
                assert(nz.at(at) == expected);
            }
            assert(vWaits == hasHot && dmaWaits == hasHot);
        }
    }
    // Fill all sixteen tile rows, including row15 next to the NZ pad row16.
    std::fill(tags.begin(), tags.end(), -1);
    for (int row = 0; row < 16; row += 2) {
        const int firstToken = 110 + row;
        v.physical[firstToken] = 200 - row;
        v.physical[firstToken + 1] = 190 - row;
        assert(v.CopyInKvSparse({&packed}, row, firstToken, firstToken + 1, run, info) == 2);
        tags[firstToken] = firstToken;
    }
    std::fill(nz.begin(), nz.end(), 0x3333);
    v.OverlayRecentHotRows({&nz}, 16, run, info);
    for (int row = 0; row < 17; ++row) for (int d = 0; d < 576; ++d) {
        const size_t at = (d / 16) * 17 * 16 + row * 16 + d % 16;
        Q_T expected = 0x3333;
        if (row < 16 && tags[v.recentTileTokens[row]] == v.recentTileTokens[row]) {
            const int token = v.recentTileTokens[row];
            expected = d < 512 ? latent[token * 512 + d] : rope[token * 64 + d - 512];
        }
        assert(nz.at(at) == expected);
    }
    // Window/tag validity: stale resident slots, old and future tokens do not
    // overwrite C4. Caller must reject absent required hot tags before launch.
    for (const int64_t token : {22, 151, 100}) {
        v.recentTileTokens[0] = token;
        std::fill(tags.begin(), tags.end(), -1);
        tags[token] = token == 100 ? token + 256 : token;
        std::fill(nz.begin(), nz.end(), 0x3333);
        vWaits = dmaWaits = 0;
        v.OverlayRecentHotRows({&nz}, 1, run, info);
        assert(std::all_of(nz.begin(), nz.end(), [](Q_T x){ return x == 0x3333; }));
        assert(vWaits == 0 && dmaWaits == 0);
    }
}
'''
        with tempfile.TemporaryDirectory() as temp:
            source = pathlib.Path(temp) / 'actual_loader.cpp'
            binary = pathlib.Path(temp) / 'actual_loader'
            source.write_text(code)
            subprocess.run([compiler, '-std=c++17', '-O0', str(source), '-o', str(binary)],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
