"""The type-free loader copies exact caller bytes, never fabricating a layout."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
HEADER=ROOT/"experimental/prolog_inplace_build/direct_tiling_load.h"
SHIM='''#pragma once
#include <cstdint>
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
'''
PROGRAM=r'''
#include "direct_tiling_load.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
// Deliberately a TEST type. No private operator size/field claims are made.
struct alignas(8) NestedPod {uint32_t n;float scale;uint64_t stride;uint16_t lanes[4];};
struct alignas(8) TestActualPod {NestedPod base;uint32_t route[3];uint32_t flags;uint64_t nestedStride;};
int main(){
 alignas(8) uint32_t sourceWords[sizeof(TestActualPod)/sizeof(uint32_t)];
 auto* source=reinterpret_cast<unsigned char*>(sourceWords);
 for(unsigned i=0;i<sizeof(sourceWords);++i)source[i]=static_cast<unsigned char>(i*37+11);
 TestActualPod actual{};
 qsfa_direct::LoadTilingWords(actual,source);
 if(std::memcmp(source,&actual,sizeof(actual)))return 2;
 unsigned char before[sizeof(actual)];std::memcpy(before,&actual,sizeof(actual));
 // A second independent destination models the other device phase. The
 // helper never inserts a nullptr assumption or changes the caller's T.
 TestActualPod second{};
 qsfa_direct::LoadTilingWords(second,source);
 if(std::memcmp(&second,&actual,sizeof(actual)))return 5;
 std::cout<<"HOST_EXACT_POD_COPY_PASS\n";return 0;
}
'''


class ActualTilingWordsTest(unittest.TestCase):
    def compiler(self):
        compiler=shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if not compiler:self.skipTest("Host C++ compiler absent; no device claim")
        return compiler

    def test_real_header_copies_word_and_subword_fields_on_both_phase_models(self):
        compiler=self.compiler()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"kernel_operator.h").write_text(SHIM);(root/"main.cpp").write_text(PROGRAM)
            for phase in ("__DAV_C310_CUBE__","__DAV_C310_VEC__"):
                with self.subTest(phase=phase):
                    exe=root/phase
                    result=subprocess.run([compiler,"-std=c++17","-O2","-fstrict-aliasing","-fsanitize=undefined",
                        "-fno-sanitize-recover=all","-DQSFA_ASC_DIRECT_BUILD=1","-D__NPU_ARCH__=3510","-D"+phase,
                        "-I"+str(root),"-I"+str(HEADER.parent),str(root/"main.cpp"),"-o",str(exe)],capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stderr)
                    run=subprocess.run([str(exe)],capture_output=True,text=True)
                    self.assertEqual(run.returncode,0,run.stderr)
                    self.assertIn("HOST_EXACT_POD_COPY_PASS",run.stdout)

    def test_real_header_rejects_nontrivial_serialization_type_at_compile_time(self):
        compiler=self.compiler()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"kernel_operator.h").write_text(SHIM)
            source='#include "direct_tiling_load.h"\nstruct HostOnly {virtual ~HostOnly() {} int n;};\nint main(){HostOnly value;qsfa_direct::LoadTilingWords(value,nullptr);}\n'
            (root/"bad.cpp").write_text(source)
            result=subprocess.run([compiler,"-std=c++17","-DQSFA_ASC_DIRECT_BUILD=1","-D__NPU_ARCH__=3510","-D__DAV_C310_VEC__",
                "-I"+str(root),"-I"+str(HEADER.parent),str(root/"bad.cpp"),"-o",str(root/"bad")],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("device tiling POD",result.stderr)


if __name__=="__main__":unittest.main()
