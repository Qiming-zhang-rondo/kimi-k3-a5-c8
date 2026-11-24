import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class FixedContractHostTests(unittest.TestCase):
    def test_real_pod_geometry_bounds_and_rejections(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("Host C++ compiler unavailable; no A5 validation")
        source_text = r'''
#include "sfa_fixed_contract.h"
#include <iostream>
#include <limits>
#include <vector>
#define CHECK(x) do { if (!(x)) { std::cerr << "failed line " << __LINE__ << "\n"; return 1; } } while (0)
int main() {
 using namespace sfa_fixed;
 auto t=MakeTiling(); auto b=GetBufferBytes();
 CHECK(sizeof(t)==144 && alignof(Tiling)==8);
 CHECK(ValidateTiling(t) && ValidateBufferBytes(b));
 CHECK(batch==1 && queries==4 && heads==64 && kvHeads==1);
 CHECK(physicalKv==256 && pageSize==128 && physicalBlocks==2 && sparseCompact==128);
 CHECK(launchAic==1 && launchAiv==2);
 CHECK(t.baseParams.nNumOfQInOneGroup==64 && t.baseParams.actualLenDimsQ==1 && t.baseParams.actualLenDimsKV==1);
 CHECK(t.baseParams.returnSoftmaxLse==1 && t.singleCoreParams.usedCoreNum==1);
 CHECK(b.query==262144 && b.key==262144 && b.output==262144);
 CHECK(b.queryRope==32768 && b.keyRope==32768);
 CHECK(b.softmaxMax==1024 && b.softmaxSum==1024 && b.indices==2048);
 CHECK(b.blockTable==8 && b.actualQuery==4 && b.actualKv==4 && b.tiling==144);
 CHECK(b.userWorkspace==442368);
 // Every opt-in launch count has matching actual POD and independent scratch.
 for(uint32_t n:{1u,2u,4u}) {
  auto parallel=MakeTiling(softmaxScale,n);auto bytes=GetBufferBytes(n);
  CHECK(IsSupportedAicCount(n)&&ValidateTiling(parallel,n)&&ValidateBufferBytes(bytes,n));
  CHECK(parallel.singleCoreParams.usedCoreNum==n&&bytes.userWorkspace==442368*n);
  if(n!=1) CHECK(!ValidateTiling(parallel)); // Default remains the verified single-AIC contract.
  for(size_t core=0;core<n;++core) {
   const size_t begin=core*442368,end=begin+442368;
   CHECK(end<=bytes.userWorkspace);
   for(size_t slot=0;slot<3;++slot) CHECK(begin+(slot+1)*128*576*2<=end);
  }
  auto shortParallel=bytes;--shortParallel.userWorkspace;CHECK(!ValidateBufferBytes(shortParallel,n));
 }
 for(uint32_t n:{0u,3u,5u,8u}) {
  CHECK(!IsSupportedAicCount(n)&&WorkspaceBytesForAic(n)==0&&!ValidateBufferBytes(b,n));
  bool rejected=false;try{MakeTiling(softmaxScale,n);}catch(const std::invalid_argument&){rejected=true;}
  CHECK(rejected);
 }
 // Bound every consumed query/head row, sparse-index row and physical paged KV row.
 for(size_t q=0;q<queries;++q) for(size_t h=0;h<heads;++h) {
  size_t row=q*heads+h;
  CHECK((row*latentDim+latentDim)*2<=b.query);
  CHECK((row*latentDim+latentDim)*2<=b.output);
  CHECK((row*ropeDim+ropeDim)*2<=b.queryRope);
  CHECK((row+1)*4<=b.softmaxMax && (row+1)*4<=b.softmaxSum);
  for(size_t j=0;j<sparseCompact;++j) CHECK((q*sparseCompact+j+1)*4<=b.indices);
 }
 for(size_t block=0;block<physicalBlocks;++block) for(size_t slot=0;slot<pageSize;++slot) {
  size_t token=block*pageSize+slot;
  CHECK((token*kvHeads*latentDim+latentDim)*2<=b.key);
  CHECK((token*kvHeads*ropeDim+ropeDim)*2<=b.keyRope);
 }
 // Actual archived InitMMResBuf uses three 128*576 BF16 buffers per AIC.
 for(size_t aic=0;aic<launchAic;++aic) for(size_t slot=0;slot<3;++slot)
  CHECK((aic*3+slot+1)*128*576*2<=b.userWorkspace);
 auto bad=t; bad.baseParams.nNumOfQInOneGroup=1; CHECK(!ValidateTiling(bad));
 bad=t; bad.baseParams.actualLenDimsQ=4; CHECK(!ValidateTiling(bad));
 bad=t; bad.baseParams.actualLenDimsKV=256; CHECK(!ValidateTiling(bad));
 bad=t; bad.baseParams.returnSoftmaxLse=0; CHECK(!ValidateTiling(bad));
 bad=t; bad.singleCoreParams.usedCoreNum=2; CHECK(!ValidateTiling(bad));
 bad=t; bad.baseParams.sparseBlockCount=2048; CHECK(!ValidateTiling(bad));
 CHECK(!ValidateTiling(MakeTiling(0)) && !ValidateTiling(MakeTiling(std::numeric_limits<float>::infinity())));
 auto shortBuffer=b; --shortBuffer.keyRope; CHECK(!ValidateBufferBytes(shortBuffer));
 shortBuffer=b; --shortBuffer.userWorkspace; CHECK(!ValidateBufferBytes(shortBuffer));
 std::vector<int32_t> indices(512); int32_t table[2]={0,1}, qlen[1]={4}, kvlen[1]={256};
 for(size_t q=0;q<4;++q) for(size_t j=0;j<128;++j) indices[q*128+j]=int32_t((q*41+j*3)%256);
 CHECK(ValidateIndicesAndLengths(indices.data(),512,table,2,qlen,1,kvlen,1));
 CHECK(!ValidateIndicesAndLengths(indices.data(),511,table,2,qlen,1,kvlen,1));
 auto old=indices.back(); indices.back()=256; CHECK(!ValidateIndicesAndLengths(indices.data(),512,table,2,qlen,1,kvlen,1));
 indices.back()=-1; CHECK(!ValidateIndicesAndLengths(indices.data(),512,table,2,qlen,1,kvlen,1));
 indices.back()=indices[384]; CHECK(!ValidateIndicesAndLengths(indices.data(),512,table,2,qlen,1,kvlen,1));
 indices.back()=old; table[1]=0; CHECK(!ValidateIndicesAndLengths(indices.data(),512,table,2,qlen,1,kvlen,1));
 table[1]=1; qlen[0]=3; CHECK(!ValidateIndicesAndLengths(indices.data(),512,table,2,qlen,1,kvlen,1));
 return 0;
}
'''
        with tempfile.TemporaryDirectory() as temp:
            source = pathlib.Path(temp) / "contract.cpp"
            binary = pathlib.Path(temp) / "contract"
            source.write_text(source_text)
            compiled = subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT / "standalone"), str(source), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            ran = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(ran.returncode, 0, ran.stderr)


if __name__ == "__main__":
    unittest.main()
