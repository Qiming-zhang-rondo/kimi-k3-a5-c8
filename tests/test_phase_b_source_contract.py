"""Host-only regressions for the Phase B 2168be3 integration defects.

Compile the real entry against mocked native services, not an AscendC SDK.
The real merge header's numerical/DMA tests live in test_device_merge_reference.
No test here establishes CANN compatibility or device correctness/performance.
"""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "standalone/sfa_fusion_entry.asc"
spec = importlib.util.spec_from_file_location("original_entry_test_helpers", ROOT / "tests/test_standalone_entry.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

OPERATOR = helpers.KERNEL_OPERATOR.replace("std::array<uint8_t*, 14>", "std::array<uint8_t*, 19>")
KERNEL = helpers.ARCH35_KERNEL.replace(
    'std::is_same<Out, AscendC::bfloat16_t>::value, "native baseline output must be BF16"',
    'std::is_same<Out, float>::value, "fused output must remain FP32"',
).replace(
    "uint8_t *maximum, uint8_t *sum, uint8_t *userWorkspace,",
    "uint8_t *maximum, uint8_t *sum, uint8_t *coldOutput, uint8_t *coldMax, "
    "uint8_t *coldSum, uint8_t *coldPresent, uint8_t *hotPresent, uint8_t *userWorkspace,",
).replace(
    "savedTiling = localTiling;",
    "const std::array<uint8_t*,5> cold{{coldOutput,coldMax,coldSum,coldPresent,hotPresent}};\n"
    '        for (unsigned i=0; i<5; ++i) Probe::Require(cold[i]==Probe::expected[14+i], "cold-state input order mismatch");\n'
    "        savedTiling = localTiling;",
)

DRIVER = r'''
#include <iostream>
#include "standalone/sfa_fusion_entry.asc"
int main() {
    try {
        std::array<std::array<uint8_t,64>,19> buffers{};
        alignas(8) std::array<uint32_t,36> gmWords{};
        auto &e=Probe::expected;
        for(unsigned i=0;i<buffers.size();++i) e[i]=buffers[i].data();
        e[13]=reinterpret_cast<uint8_t*>(gmWords.data());
        for(unsigned i=0;i<144;++i) e[13][i]=static_cast<uint8_t>((i*37+11)&0xff);
        std::memcpy(Probe::tilingSnapshot.data(),e[13],144);
        sfa_cold_merge_probe(e[0],e[1],e[2],e[3],e[4],e[5],e[6],e[7],e[8],
            e[9],e[10],e[11],e[12],e[13],e[14],e[15],e[16],e[17],e[18]);
        Probe::Require(Probe::initCalls==1 && Probe::processCalls==1,"missing Init/Process");
        Probe::Require(Probe::workspaceCalls==0 && Probe::taskCalls==1,"wrong scratch/task ABI");
        Probe::Require(!Probe::pipeAlive,"TPipe lifetime");
        std::cout << "HOST_FUSION_ENTRY_PASS\n";
        return 0;
    } catch(const std::exception &e) {std::cerr << e.what();return 1;}
}
'''

HOST_DRIVER = r'''
#include "standalone/sfa_fusion_entry.asc"
int main() {
    uint8_t x=0;
    sfa_cold_merge_probe(&x,&x,&x,&x,&x,&x,&x,&x,&x,&x,&x,&x,&x,&x,&x,&x,&x,&x,&x);
    return Probe::taskCalls==1 && Probe::initCalls==0 && Probe::processCalls==0 ? 0:1;
}
'''


class PhaseBSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if not cls.compiler:
            raise unittest.SkipTest("Host C++ compiler unavailable")

    def compile_entry(self, defines, device=False, expect_success=True):
        with tempfile.TemporaryDirectory(prefix="sfa-fusion-entry-host-") as temp:
            root=Path(temp)
            (root/"standalone").mkdir()
            (root/"experimental/arch35").mkdir(parents=True)
            (root/"standalone/sfa_fusion_entry.asc").write_bytes(ENTRY.read_bytes())
            (root/"kernel_operator.h").write_text(OPERATOR)
            poison='#error "HOST_PASS_PARSED_DEVICE_HEADER"\n'
            (root/"experimental/arch35/sparse_flash_attention_kernel_mla.h").write_text(KERNEL if device else poison)
            if device:
                (root/"standalone/sparse_flash_attention_tiling_data.h").write_bytes(
                    (ROOT/"standalone/sparse_flash_attention_tiling_data.h").read_bytes())
            else:
                (root/"standalone/sparse_flash_attention_tiling_data.h").write_text(poison)
            (root/"driver.cpp").write_text(DRIVER if device else HOST_DRIVER)
            command=[self.compiler,"-std=c++17","-O0","-Wno-unknown-pragmas",
                     "-DPROBE_EXPECTED_MODE=0",*defines,"-I",str(root),"-I",str(ROOT/"standalone"),
                     str(root/"driver.cpp"),"-o",str(root/"probe")]
            compiled=subprocess.run(command,capture_output=True,text=True,timeout=30)
            if not expect_success:
                self.assertNotEqual(compiled.returncode,0)
                self.assertIn("SFA fusion device",compiled.stderr)
                return
            self.assertEqual(compiled.returncode,0,compiled.stdout+compiled.stderr)
            ran=subprocess.run([str(root/"probe")],capture_output=True,text=True,timeout=10)
            self.assertEqual(ran.returncode,0,ran.stdout+ran.stderr)
            if device:
                self.assertEqual(ran.stdout.strip(),"HOST_FUSION_ENTRY_PASS")

    def test_aic_fp32_output_and_all_five_cold_inputs(self):
        self.compile_entry(["-D__NPU_ARCH__=3510","-D__DAV_C310_CUBE__=1"],device=True)

    def test_aiv_fp32_output_and_local_tiling_lifetime(self):
        self.compile_entry(["-D__NPU_ARCH__=3510","-D__DAV_C310_VEC__=1"],device=True)

    def test_host_scan_does_not_parse_microapi(self):
        self.compile_entry([])
        self.compile_entry(["-D__NPU_ARCH__=0"])

    def test_wrong_or_missing_device_arch_rejected(self):
        self.compile_entry(["-D__NPU_ARCH__=3002"],expect_success=False)
        self.compile_entry(["-D__DAV_C310_VEC__=1"],expect_success=False)

    def test_original_and_candidate_headers_cannot_shadow_each_other(self):
        for name in ("sparse_flash_attention_kernel_mla.h",
                     "sparse_flash_attention_service_vector_mla.h","cold_state_merge_device.h"):
            self.assertFalse((ROOT/"standalone/arch35"/name).exists(),"duplicate candidate shadows original entry")
            self.assertTrue((ROOT/"standalone/archive/2168be3"/(name+".snapshot")).is_file())
        cmake=(ROOT/"standalone/CMakeLists.txt").read_text()
        shared=cmake.split("function(configure_sfa target)",1)[1].split("endfunction()",1)[0]
        self.assertNotIn("experimental/arch35",shared)
        self.assertIn("add_library(sfa_fusion_probe STATIC sfa_fusion_entry.asc)",cmake)
        self.assertIn("SFA_BUILD_FUSION_PROBE",cmake)
        self.assertIn('"../experimental/arch35/sparse_flash_attention_kernel_mla.h"',ENTRY.read_text())

    def test_small_statistics_keep_byte_length_padded_dma(self):
        merge=(ROOT/"experimental/arch35/cold_state_merge_device.h").read_text()
        self.assertIn("copy.blockLen = count * sizeof(float)",merge)
        self.assertIn("DataCopyPad(stats, coldMaxGm[flatOffset + start], copy, pad)",merge)
        self.assertIn("DataCopyPad(stats[8], coldSumGm[flatOffset + start], copy, pad)",merge)
        self.assertNotIn("DataCopy(stats",merge)

    def test_fused_store_uses_same_type_three_argument_ub_to_gm_api(self):
        service=(ROOT/"experimental/arch35/sparse_flash_attention_service_vector_mla.h").read_text()
        store=service.split("::Bmm2DataCopyOut",1)[1].split("::CopyOutAttentionOut",1)[0]
        self.assertIn("IsSameType<VEC2_RES_T, float>::value && IsSameType<OUTPUT_T, float>::value",store)
        self.assertIn("DataCopyExtParams dataCopyParams = {}",store)
        self.assertIn("DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], vec2ResUb, dataCopyParams)",store)
        self.assertNotIn("DataCopyPadExtParams<",store)


if __name__=="__main__":
    unittest.main()
