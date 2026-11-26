"""Compile the actual host/device POD and two C API declarations with UBSan."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"experimental/single_pass"


class SplitHostContractTests(unittest.TestCase):
    def test_real_pod_offsets_allocation_and_legacy_abi(self):
        compiler=shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            self.skipTest("No C++17 compiler")
        code=r'''
#include "kv_split_tiling.h"
#include "single_pass_api.h"
#include "kv_split_reduce_api.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>
using namespace recent128_single_pass;
static_assert(sizeof(Tiling)==72 && sizeof(SplitTiling)==112);
static_assert(offsetof(SplitTiling,splitParams)==72);
static_assert(sizeof(SinglePassBuffers)==15*sizeof(uint8_t*));
#if QSFA_RECENT_KV_SPLIT
static_assert(std::is_same<KvQuantSparseFlashAttentionTilingDataMla,SplitTiling>::value);
#else
static_assert(std::is_same<KvQuantSparseFlashAttentionTilingDataMla,Tiling>::value);
#endif
using ReduceApi=int32_t(*)(void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
                          uint32_t,uint32_t,uint32_t,uint32_t);
static_assert(std::is_same<decltype(&QsfaSinglePassLaunchKvSplitReduce),ReduceApi>::value);
int main() {
 for (uint32_t q : {1U,4U}) for (uint32_t s : {1U,2U,4U}) for (bool stats : {false,true}) {
   const auto t=MakeSplitTiling(q,8192,s,1.0f/std::sqrt(576.f),stats);
   assert(t.splitParams.abiVersion==2 && t.splitParams.splitCount==s);
   assert(t.singleCoreParams.usedCoreNum==q*s && t.baseParams.qSeqSize==q);
   assert(t.baseParams.dSizeVInput==400 && t.baseParams.sparseBlockCount==2048);
   assert(t.baseParams.returnSoftmaxLse==uint32_t(stats));
   const auto &p=t.splitParams;
   assert(p.partialOutputOffsetBytes==0);
   assert(p.partialMaximumOffsetBytes==uint64_t(q)*s*64*512*4);
   assert(p.partialSumOffsetBytes==p.partialMaximumOffsetBytes+uint64_t(q)*s*64*4);
   assert(p.workspaceBytes==uint64_t(q)*s*64*(512+2)*4);
   assert(p.workspaceBytes%32==0);
   std::vector<float> storage(p.workspaceBytes/4+16,-99.f);
   for (uint64_t i=0;i<p.workspaceBytes/4;++i) storage[i]=float(i%257);
   for (uint64_t i=p.workspaceBytes/4;i<storage.size();++i) assert(storage[i]==-99.f);
   assert(storage[p.partialMaximumOffsetBytes/4-1]!=-99.f);
   assert(storage[p.partialSumOffsetBytes/4]!=-99.f);
 }
 auto rejected=[](uint32_t q,uint32_t k,uint32_t s,float scale) {
   try { (void)MakeSplitTiling(q,k,s,scale,false); return false; }
   catch(const std::invalid_argument&) { return true; }
 };
 assert(rejected(0,8192,2,1) && rejected(4,8192,3,1));
 assert(rejected(8,8192,4,1) && rejected(4,8191,2,1));
 assert(rejected(4,1024,2,1) && rejected(4,8192,2,0));
 assert(rejected(4,8192,2,std::numeric_limits<float>::infinity()));
 const auto old=MakeTiling(4,8192,4,1.f,false);
 assert(old.singleCoreParams.usedCoreNum==4 && userWorkspaceAllocationBytes==1);
}
'''
        with tempfile.TemporaryDirectory(prefix="qsfa-split-host-") as temp:
            path=Path(temp)
            source=path/"check.cpp"
            source.write_text(code)
            for split in (0,1):
                binary=path/f"check{split}"
                build=subprocess.run([compiler,"-std=c++17","-fsanitize=undefined",
                    "-fno-sanitize-recover=all",f"-DQSFA_RECENT_KV_SPLIT={split}",
                    "-I",str(SOURCE),str(source),"-o",str(binary)],text=True,capture_output=True)
                self.assertEqual(build.returncode,0,build.stderr)
                run=subprocess.run([str(binary)],text=True,capture_output=True)
                self.assertEqual(run.returncode,0,run.stderr)


if __name__=="__main__":
    unittest.main()
