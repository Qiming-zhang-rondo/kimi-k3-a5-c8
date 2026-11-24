"""Host-compile and invoke the actual standalone .asc entry across compiler passes.

The compiler sees the real entry and pinned SGL POD, with lightweight host mocks
for AscendC and the arch35 services. These tests validate dispatch, argument
wiring, local tiling copying, and Init/Process execution. They do NOT compile
AscendC, emulate device address spaces, validate the real attention implementation,
or establish compatibility with an installed/private tiling serializer. Failures
use exceptions/normal exit codes; no fatal signals or process crashes are used.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "standalone/sfa_standalone_entry.asc"

KERNEL_OPERATOR = r'''
#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#define __gm__
#define __global__
#define __aicore__
namespace Probe {
inline std::array<uint8_t*, 14> expected{};
inline std::array<uint8_t, 144> tilingSnapshot{};
inline unsigned initCalls = 0, processCalls = 0, workspaceCalls = 0, taskCalls = 0;
inline bool pipeAlive = false;
inline void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
}
namespace AscendC {
struct bfloat16_t { uint16_t value; };
constexpr int AIC = 0, AIV = 1;
#ifdef __DAV_C310_CUBE__
constexpr int g_coreType = AIC;
#else
constexpr int g_coreType = AIV;
#endif
struct TPipe {
    TPipe() { Probe::pipeAlive = true; }
    ~TPipe() { Probe::pipeAlive = false; }
};
inline uint8_t *GetUserWorkspace(uint8_t *workspace) {
    Probe::Require(workspace == Probe::expected[12], "workspace argument was reordered");
    ++Probe::workspaceCalls;
    return workspace + 32;
}
}
#define KERNEL_TYPE_MIX_AIC_1_2 42
#define KERNEL_TASK_TYPE_DEFAULT(value) \
    do { Probe::Require((value) == 42, "wrong mixed task kind"); ++Probe::taskCalls; } while (0)
'''

ARCH35_KERNEL = r'''
#pragma once
// The real pinned POD was included by the actual entry before this header.
enum class SFA_LAYOUT { BSND=0, TND=1, PA_BSND=2 };
enum class SFATemplateMode { C=0, V=1 };
namespace BaseApi {
template<typename Q, typename KV, typename Acc, typename Out,
         bool Fd, bool Pa, SFA_LAYOUT Layout, SFA_LAYOUT KVLayout,
         SFATemplateMode Mode, bool Split>
struct TypeContract {
    static_assert(std::is_same<Q, AscendC::bfloat16_t>::value, "Q must remain BF16");
    static_assert(std::is_same<KV, AscendC::bfloat16_t>::value, "KV must remain BF16");
    static_assert(std::is_same<Acc, float>::value, "accumulator must be FP32");
    static_assert(std::is_same<Out, AscendC::bfloat16_t>::value, "native baseline output must be BF16");
    static_assert(!Fd && Pa && !Split, "wrong boolean template axes");
    static_assert(Layout == SFA_LAYOUT::TND && KVLayout == SFA_LAYOUT::PA_BSND,
                  "wrong layout template axes");
    static_assert(static_cast<int>(Mode) == PROBE_EXPECTED_MODE, "wrong selected C/V template mode");
};
#define PROBE_TEMPLATE \
    template<typename Q, typename KV, typename Acc, typename Out, bool Fd, bool Pa, \
             SFA_LAYOUT Layout, SFA_LAYOUT KVLayout, SFATemplateMode Mode, bool Split>
#define PROBE_ARGS Q, KV, Acc, Out, Fd, Pa, Layout, KVLayout, Mode, Split
PROBE_TEMPLATE struct SFAMatmulService : TypeContract<PROBE_ARGS> { static constexpr int kind = 0; };
PROBE_TEMPLATE struct SFAMatmulServiceDummy : TypeContract<PROBE_ARGS> { static constexpr int kind = 1; };
PROBE_TEMPLATE struct SFAVectorService : TypeContract<PROBE_ARGS> { static constexpr int kind = 2; };
PROBE_TEMPLATE struct SFAVectorServiceDummy : TypeContract<PROBE_ARGS> { static constexpr int kind = 3; };
#undef PROBE_ARGS
#undef PROBE_TEMPLATE

template<typename Cube, typename Vec>
class SparseFlashAttentionKernelMla {
    static constexpr bool isAic = AscendC::g_coreType == AscendC::AIC;
    static_assert(Cube::kind == (isAic ? 0 : 1), "wrong real/dummy cube service");
    static_assert(Vec::kind == (isAic ? 3 : 2), "wrong real/dummy vector service");
    const SparseFlashAttentionTilingDataMla *savedTiling = nullptr;
    AscendC::TPipe *savedPipe = nullptr;
public:
    void Init(uint8_t *query, uint8_t *key, uint8_t *value, uint8_t *indices,
              uint8_t *seqQ, uint8_t *seqKV, uint8_t *blocktable,
              uint8_t *queryRope, uint8_t *keyRope, uint8_t *out,
              uint8_t *maximum, uint8_t *sum, uint8_t *userWorkspace,
              const SparseFlashAttentionTilingDataMla *localTiling,
              uint8_t *gmTiling, AscendC::TPipe *pipe) {
        Probe::Require(Probe::initCalls == 0, "Init called more than once");
        ++Probe::initCalls;
        const std::array<uint8_t*, 12> actual{{query,key,value,indices,seqQ,seqKV,
                                              blocktable,queryRope,keyRope,out,maximum,sum}};
        const std::array<unsigned, 12> order{{0,1,2,3,5,6,4,7,8,9,10,11}};
        for (unsigned i = 0; i < actual.size(); ++i)
            Probe::Require(actual[i] == Probe::expected[order[i]], "Init tensor argument order mismatch");
        Probe::Require(userWorkspace == Probe::expected[12], "direct user scratch was not forwarded unchanged");
        Probe::Require(gmTiling == Probe::expected[13], "original GM tiling pointer was not preserved");
        Probe::Require(pipe != nullptr && Probe::pipeAlive, "TPipe is missing/not live in Init");
        if constexpr (isAic) {
            Probe::Require(localTiling == nullptr, "AIC must receive null local tiling");
        } else {
            Probe::Require(localTiling != nullptr, "AIV local tiling is missing");
            Probe::Require(reinterpret_cast<const void*>(localTiling) != gmTiling,
                           "AIV must copy GM bytes rather than aliasing GM");
            Probe::Require(std::memcmp(localTiling, Probe::tilingSnapshot.data(), 144) == 0,
                           "AIV did not copy all 144 POD bytes, including padding");
        }
        savedTiling = localTiling;
        savedPipe = pipe;
        // Poison the GM source after Init. Process must still see its own copy.
        std::memset(gmTiling, 0xa5, 144);
    }
    void Process() {
        Probe::Require(Probe::initCalls == 1 && Probe::processCalls == 0,
                       "Process ordering/count mismatch");
        Probe::Require(savedPipe != nullptr && Probe::pipeAlive, "TPipe ended before Process");
        if constexpr (!isAic) {
            Probe::Require(savedTiling != nullptr, "local copy disappeared before Process");
            Probe::Require(std::memcmp(savedTiling, Probe::tilingSnapshot.data(), 144) == 0,
                           "local tiling was changed or out of scope during Process");
        }
        ++Probe::processCalls;
    }
};
}
'''

DRIVER = r'''
#include <iostream>
#include "sfa_standalone_entry.asc"
int main() {
    try {
        std::array<std::array<uint8_t,64>,13> buffers{};
        alignas(8) std::array<uint32_t,36> gmWords{};
        for (unsigned i=0; i<buffers.size(); ++i) Probe::expected[i] = buffers[i].data();
        Probe::expected[13] = reinterpret_cast<uint8_t*>(gmWords.data());
        for (unsigned i=0; i<144; ++i)
            Probe::expected[13][i] = static_cast<uint8_t>((i*37+11)&0xff);
        std::memcpy(Probe::tilingSnapshot.data(), Probe::expected[13], 144);
        // Calling the actual exported non-template entry forces its template body
        // and both mock service Init/Process methods to be compiled and exercised.
        auto &e = Probe::expected;
        sfa_arch35_bf16_user_scratch_probe(e[0],e[1],e[2],e[3],e[4],e[5],e[6],
                                     e[7],e[8],e[9],e[10],e[11],e[12],e[13]);
        Probe::Require(Probe::initCalls == 1 && Probe::processCalls == 1,
                       "entry did not execute Init and Process exactly once");
        Probe::Require(Probe::workspaceCalls == 0 && Probe::taskCalls == 1,
                       "entry skipped workspace/task setup");
        Probe::Require(!Probe::pipeAlive, "TPipe survived the completed entry scope");
        std::cout << "HOST_ENTRY_PASS core=" << AscendC::g_coreType
                  << " mode=" << PROBE_EXPECTED_MODE << " init=1 process=1 bytes=144\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "HOST_ENTRY_FAIL: " << error.what() << "\n";
        return 1;
    }
}
'''


class StandaloneEntryHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if cls.compiler is None:
            raise unittest.SkipTest("A host C++ compiler is required; no AscendC claim is made")

    def run_path(self, cube, mode):
        with tempfile.TemporaryDirectory(prefix="sfa-entry-host-") as temp:
            directory = Path(temp)
            (directory / "arch35").mkdir()
            (directory / "kernel_operator.h").write_text(KERNEL_OPERATOR)
            (directory / "arch35/sparse_flash_attention_kernel_mla.h").write_text(ARCH35_KERNEL)
            (directory / "driver.cpp").write_text(DRIVER)
            binary = directory / "entry_probe"
            command = [self.compiler, "-std=c++17", "-O0", "-Wno-unknown-pragmas",
                       "-D__CCE_AICORE__=310", "-D__NPU_ARCH__=3510",
                       f"-DPROBE_EXPECTED_MODE={mode}",
                       "-I", str(directory), "-I", str(ENTRY.parent),
                       str(directory / "driver.cpp"), "-o", str(binary)]
            if cube:
                command.append("-D__DAV_C310_CUBE__=1")
            else:
                command.append("-D__DAV_C310_VEC__=1")
            # Mode 0 intentionally also tests the actual entry's default.
            if mode:
                command.append(f"-DSFA_PROBE_TEMPLATE_MODE={mode}")
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.strip(),
                             f"HOST_ENTRY_PASS core={0 if cube else 1} mode={mode} init=1 process=1 bytes=144")

    def compile_host_or_rejected_pass(self, defines, expect_success):
        """Poison device-only headers so an empty host pass cannot hide parsing.

        This checks preprocessing/entry wiring with an ordinary C++ compiler,
        not SDK host or device code generation.
        """
        with tempfile.TemporaryDirectory(prefix="sfa-entry-pass-") as temp:
            directory = Path(temp)
            (directory / "arch35").mkdir()
            (directory / "kernel_operator.h").write_text(KERNEL_OPERATOR)
            poison = '#error "HOST_PASS_PARSED_DEVICE_ONLY_HEADER"\n'
            (directory / "arch35/sparse_flash_attention_kernel_mla.h").write_text(poison)
            (directory / "sparse_flash_attention_tiling_data.h").write_text(poison)
            # Preserve the complete real entry bytes, but place them beside the
            # poisoned quoted includes so its own directory cannot bypass them.
            (directory / "sfa_standalone_entry.asc").write_bytes(ENTRY.read_bytes())
            driver = r'''
#include <iostream>
#include "sfa_standalone_entry.asc"
int main() {
    try {
        std::array<uint8_t, 64> sentinel{};
        sentinel.fill(0x5a);
        uint8_t *p = sentinel.data();
        sfa_arch35_bf16_user_scratch_probe(p,p,p,p,p,p,p,p,p,p,p,p,p,p);
        Probe::Require(Probe::initCalls == 0 && Probe::processCalls == 0,
                       "host pass ran native services");
        Probe::Require(Probe::workspaceCalls == 0 && Probe::taskCalls == 1,
                       "host pass ran device setup");
        Probe::Require(!Probe::pipeAlive, "host pass constructed a pipe");
        for (uint8_t byte : sentinel)
            Probe::Require(byte == 0x5a, "host entry touched argument memory");
        std::cout << "HOST_ONLY_PASS no_device_headers init=0 process=0\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
'''
            (directory / "driver.cpp").write_text(driver)
            binary = directory / "host_probe"
            command = [self.compiler, "-std=c++17", "-O0", "-Wno-unknown-pragmas",
                       "-I", str(directory), "-I", str(ENTRY.parent),
                       *[f"-D{define}" for define in defines],
                       str(directory / "driver.cpp"), "-o", str(binary)]
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=30)
            output = compiled.stdout + compiled.stderr
            if not expect_success:
                self.assertNotEqual(compiled.returncode, 0, "unsupported device pass compiled")
                self.assertNotIn("HOST_PASS_PARSED_DEVICE_ONLY_HEADER", output,
                                 "reject the target before including device headers")
                self.assertIn("3510", output, "expected explicit target guard diagnostic")
                return
            self.assertEqual(compiled.returncode, 0, output)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.strip(),
                             "HOST_ONLY_PASS no_device_headers init=0 process=0")

    def test_aic_c_mode(self):
        self.run_path(cube=True, mode=0)

    def test_aiv_c_mode(self):
        self.run_path(cube=False, mode=0)

    def test_aic_v_mode(self):
        self.run_path(cube=True, mode=1)

    def test_aiv_v_mode(self):
        self.run_path(cube=False, mode=1)

    def test_host_pass_without_npu_arch(self):
        self.compile_host_or_rejected_pass([], expect_success=True)

    def test_host_pass_with_npu_arch_zero(self):
        self.compile_host_or_rejected_pass(["__NPU_ARCH__=0"], expect_success=True)

    def test_unsupported_device_arch_is_rejected(self):
        self.compile_host_or_rejected_pass(["__NPU_ARCH__=2201"], expect_success=False)

    def test_device_markers_require_matching_npu_arch(self):
        for marker in ("__DAV_C310_CUBE__=1", "__DAV_C310_VEC__=1"):
            with self.subTest(marker=marker, architecture="missing"):
                self.compile_host_or_rejected_pass([marker], expect_success=False)
            with self.subTest(marker=marker, architecture="zero"):
                self.compile_host_or_rejected_pass([marker, "__NPU_ARCH__=0"], expect_success=False)


if __name__ == "__main__":
    unittest.main()
