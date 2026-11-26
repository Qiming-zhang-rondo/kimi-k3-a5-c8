"""Compile real split scalar/DMA function bodies in a bounded CPU model.

This is not an AscendC emulator or an SDK/device compilation. It checks task
mapping, original TopK column addressing, packed-row bounds and FP32 workspace
stores. Cube arithmetic, MicroAPI arithmetic and real synchronization remain
the responsibility of the independent A5 mathematical/graph controls.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / 'experimental/single_pass/arch35'


def actual_body(text, marker):
    start = text.index('{', text.index(marker))
    depth, end = 1, start + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


MODEL = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>
#define QSFA_RECENT_KV_SPLIT 1
#define unlikely(x) (x)
#define __gm__
constexpr int LAYOUT_T=1;
enum class QSFA_LAYOUT { TND=1, BSND=0 };
constexpr bool operator==(int a, QSFA_LAYOUT b){return a==static_cast<int>(b);}
template<class T>T Min(T a,T b){return std::min(a,b);}
constexpr uint32_t CV_RATIO=2, s1BaseSize=64, dVTemplateType=512;
constexpr uint32_t qsfaDTemplateAlign64=512, BUFFER_SIZE_BYTE_32B=32;
using T=float;
using Q_T=uint16_t;
using OUTPUT_T=uint16_t;
struct RunInfo{
 int64_t boIdx=0,s1oIdx=0,s2LoopCount=0,s2StartIdx=0,s2EndIdx=0;
 int64_t s2LoopLimit=0;
 uint32_t vec2MRealSize=32,vec2MBaseSize=32,vec2S1RealSize=1,vec2S1BaseSize=1;
 uint32_t halfMRealSize=32,multiCoreIdxMod2=0,taskIdMod2=0;
};
struct ConstInfo{
 uint32_t s1Size=4,sparseBlockCount=2048,s2BaseSize=128,dSizeV=512,subBlockIdx=0;
 uint32_t bN2Start=0,bN2End=0,gS1Start=0,gS1End=0,s2Start=0,s2End=0;
 float softmaxScale=1.0f;
};
struct IndexTensor{std::vector<int32_t> v;int32_t GetValue(int64_t i)const{return v.at(i);}};
uint8_t *workspaceBegin=nullptr,*workspaceEnd=nullptr;
template<class T>struct Tensor{
 T*pointer=nullptr;uint64_t count=0;
 void SetGlobalBuffer(T*p){
  auto address=reinterpret_cast<uint8_t*>(p);
  assert(address>=workspaceBegin&&address<=workspaceEnd);
  pointer=p;count=(workspaceEnd-address)/sizeof(T);
 }
 Tensor operator[](uint64_t offset)const{assert(offset<=count);return {pointer+offset,count-offset};}
};
template<class T>using GlobalTensor=Tensor<T>;
template<class T>using LocalTensor=Tensor<T>;
struct FloatBuffer{
 Tensor<float> tensor;
 template<class U>Tensor<U> Get(){
  static_assert(std::is_same<U,float>::value,"FP32 model only");return tensor;
 }
};
struct CubeBuffer{
 Tensor<float> tensor;
 void WaitCrossCore(){}
 void SetCrossCore(){}
 template<class U>Tensor<U> GetTensor(){
  static_assert(std::is_same<U,float>::value,"FP32 model only");return tensor;
 }
};
namespace matmul{
 template<class T>void InitOutput(Tensor<T>dst,uint64_t n,T value){
  assert(n<=dst.count);std::fill(dst.pointer,dst.pointer+n,value);
 }
}
enum class HardEvent{V_MTE3,MTE3_V};
template<HardEvent E>void SetFlag(int){}
template<HardEvent E>void WaitFlag(int){}
struct DataCopyExtParams{uint32_t blockLen=0,srcStride=0,dstStride=0,blockCount=0;};
template<class T>void DataCopyPad(Tensor<T>dst,Tensor<T>src,const DataCopyExtParams&p){
 assert(p.blockLen%sizeof(T)==0);
 uint64_t length=p.blockLen/sizeof(T);
 for(uint32_t i=0;i<p.blockCount;i++){
  uint64_t source=i*(length+p.srcStride*32/sizeof(T));
  uint64_t target=i*(length+p.dstStride/sizeof(T));
  assert(source+length<=src.count&&target+length<=dst.count);
  std::copy(src.pointer+source,src.pointer+source+length,dst.pointer+target);
 }
}
template<class T>void DataCopy(Tensor<T>dst,Tensor<T>src,uint64_t n){
 assert(n<=dst.count&&n<=src.count);std::copy(src.pointer,src.pointer+n,dst.pointer);
}
template<class A,class B,class C,int D,bool E,bool F,class...Args>
void FlashUpdateNew(Args...){assert(false&&"not exercised by one-tile empty-piece regression");}
template<class A,class B,class C,int D,bool E,bool F,class...Args>
void FlashUpdateLastNew(Args...){assert(false&&"not exercised by one-tile empty-piece regression");}
template<class A,class B,class C,int D,bool E>
void LastDivNew(Tensor<float>dst,Tensor<float>src,Tensor<float>sum,
 uint32_t rows,uint32_t columns,float){
 for(uint32_t row=0;row<rows;row++)for(uint32_t d=0;d<columns;d++)
  dst.pointer[row*columns+d]=src.pointer[row*columns+d]/sum.pointer[row];
}
void ZeroInvalidAttnOut(Tensor<float>out,Tensor<float>maximum,float sentinel,
 uint32_t rows,uint32_t columns,uint32_t stride){
 for(uint32_t row=0;row<rows;row++)if(maximum.pointer[row]==sentinel)
  std::fill(out.pointer+row*stride,out.pointer+row*stride+columns,0.0f);
}
void CleanNaNAttnOut(Tensor<float>out,uint32_t rows,uint32_t columns,uint32_t stride){
 for(uint32_t row=0;row<rows;row++)for(uint32_t d=0;d<columns;d++)
  if(std::isnan(out.pointer[row*stride+d]))out.pointer[row*stride+d]=0.0f;
}
void ZeroInvalidLse(Tensor<float>maximum,Tensor<float>sum,float sentinel,uint32_t rows){
 for(uint32_t row=0;row<rows;row++)if(maximum.pointer[row]==sentinel){
  maximum.pointer[row]=0.0f;sum.pointer[row]=0.0f;
 }
}
// Only fields consumed by the extracted device bodies are modeled here.
// Full serialized POD size/offset validation belongs to the host ABI tests.
struct Tiling{
 struct Base{uint32_t sparseBlockCount=2048;}baseParams;
 struct Split{uint32_t splitCount=1;uint64_t partialOutputOffsetBytes=0,
  partialMaximumOffsetBytes=0,partialSumOffsetBytes=0;}splitParams;
};
struct Vector{
 uint32_t kvSplitTaskId=0,kvSplitCount=1,kvSplitColumns=0,kvSplitColumnBegin=0;
 int64_t sparseS2End=128;
 IndexTensor SparseIndicesGm,cuSeqlensQGm;
 Tiling*tilingData=nullptr;
 GlobalTensor<float>kvSplitOutputGm,softmaxMaxGm,softmaxSumGm;
 int vToMte3Id[1]={0};
 int mte3ToVId[1]={0},mte3ToVLseOutId=0,vToMte3LseOutId=0;
 float negativeFloatScalar=-9999.0f;
 FloatBuffer stage2OutBuf,softmaxExpBuf[2],softmaxSumBuf[2],softmaxMaxBuf[2];
 LocalTensor<float>lseUb;
 void GetRealCmpS2Idx(int64_t&token0Idx,int64_t&token1Idx,int64_t s2IdxInBase,
  const RunInfo&runInfo,ConstInfo&constInfo)
 BODY_REAL
 int64_t CalcSparseValidRows(const RunInfo&runInfo,ConstInfo&constInfo)
 BODY_VALID
 void InitKvSplitWorkspace(uint8_t*workspace)
 BODY_WORKSPACE
 void InitEmptyKvSplit(const ConstInfo&constInfo)
 BODY_EMPTY
 template<class VEC2_RES_T>void Bmm2DataCopyOut(RunInfo&runInfo,ConstInfo&constInfo,
  LocalTensor<VEC2_RES_T>&vec2ResUb,int64_t vec2S1Idx,int64_t qsfaVec2CalcSize)
 BODY_OUTPUT
 void CopyOutAttentionOut(RunInfo&runInfo,ConstInfo&constInfo,LocalTensor<float>&out,
  int64_t start,int64_t size){Bmm2DataCopyOut(runInfo,constInfo,out,start,size);}
 void CopyFALseToGm(RunInfo&runInfo,ConstInfo&constInfo)
 BODY_STATS
 void ProcessVec2(CubeBuffer&bmm2ResBuf,RunInfo&runInfo,ConstInfo&constInfo)
 BODY_VEC2
};
struct Kernel{
 int64_t maxS2LoopCnt=0;
 uint32_t usedCoreNum=0,kvSplitCount=1,kvSplitColumnBegin=0,kvSplitColumns=0,aicIdx=0;
 ConstInfo constInfo;
 void InitCalcParamsEach()
 BODY_KERNEL
};
void check_indices(unsigned queries,unsigned parts){
 Vector vec;ConstInfo ci;ci.s1Size=queries;
 vec.SparseIndicesGm.v.resize(queries*2048);vec.cuSeqlensQGm.v={static_cast<int32_t>(queries)};
 for(unsigned q=0;q<queries;q++)for(int c=0;c<2048;c++)
  vec.SparseIndicesGm.v[q*2048+c]=(q*2048+c*29)%8192;
 for(unsigned task=0;task<queries*parts;task++){
  Kernel kernel;kernel.constInfo.s1Size=queries;kernel.usedCoreNum=queries*parts;kernel.aicIdx=task;
  kernel.InitCalcParamsEach();
  assert(kernel.kvSplitCount==parts&&kernel.kvSplitColumns==2048/parts);
  assert(kernel.constInfo.gS1Start==task/parts&&kernel.constInfo.gS1End==task/parts);
  assert(kernel.kvSplitColumnBegin==(task%parts)*(2048/parts));
  vec.kvSplitColumnBegin=kernel.kvSplitColumnBegin;
  RunInfo ri;ri.s1oIdx=task/parts;ri.s2EndIdx=2048/parts;
  // Deliberate nonzero poison: splitting must not add this to token IDs.
  ri.s2StartIdx=777;
  for(unsigned tile=0;tile<2048/parts/128;tile++){
   ri.s2LoopCount=tile;assert(vec.CalcSparseValidRows(ri,ci)==128);
   for(int pair=0;pair<128;pair+=2){
    int64_t x=-2,y=-2;vec.GetRealCmpS2Idx(x,y,pair,ri,ci);
    auto base=ri.s1oIdx*2048+vec.kvSplitColumnBegin+tile*128+pair;
    assert(x==vec.SparseIndicesGm.v.at(base));assert(y==vec.SparseIndicesGm.v.at(base+1));
   }
  }
 }
 for(unsigned q=0;q<queries;q++)for(int c=1988;c<2048;c++)vec.SparseIndicesGm.v[q*2048+c]=-1;
 RunInfo ri;ri.s1oIdx=queries-1;ri.s2EndIdx=2048/parts;ri.s2LoopCount=2048/parts/128-1;
 vec.kvSplitColumnBegin=2048-2048/parts;
 assert(vec.CalcSparseValidRows(ri,ci)==68);
 // Entire invalid final tile must be reported empty, not read stale KV.
 for(unsigned q=0;q<queries;q++)for(int c=1920;c<2048;c++)vec.SparseIndicesGm.v[q*2048+c]=-1;
 assert(vec.CalcSparseValidRows(ri,ci)==0);
}
void check_workspace(unsigned queries,unsigned parts){
 const uint64_t rows=queries*parts*64,guard=8;
 std::vector<float>memory(guard+rows*(512+2)+guard,-1337.0f);
 workspaceBegin=reinterpret_cast<uint8_t*>(memory.data());
 workspaceEnd=reinterpret_cast<uint8_t*>(memory.data()+memory.size());
 Tiling tiling;tiling.splitParams.splitCount=parts;
 tiling.splitParams.partialOutputOffsetBytes=guard*sizeof(float);
 tiling.splitParams.partialMaximumOffsetBytes=(guard+rows*512)*sizeof(float);
 tiling.splitParams.partialSumOffsetBytes=(guard+rows*(512+1))*sizeof(float);
 for(unsigned task=0;task<queries*parts;task++)for(unsigned half=0;half<2;half++){
  Vector vector;vector.tilingData=&tiling;vector.kvSplitTaskId=task;
  vector.InitKvSplitWorkspace(workspaceBegin);
  assert(vector.kvSplitColumnBegin==(task%parts)*(2048/parts));
  ConstInfo ci;ci.subBlockIdx=half;
  vector.InitEmptyKvSplit(ci);
 }
 assert(std::all_of(memory.begin()+guard,memory.end()-guard,[](float v){return v==0.0f;}));
 // Write real FP32 detail smaller than a BF16 ulp and check it survives DMA.
 for(unsigned task=0;task<queries*parts;task++)for(unsigned half=0;half<2;half++){
  Vector vector;vector.tilingData=&tiling;vector.kvSplitTaskId=task;
  vector.InitKvSplitWorkspace(workspaceBegin);
  ConstInfo ci;ci.subBlockIdx=half;RunInfo ri;
  float value=1.0f+(task*2+half+1)/4096.0f;
  std::vector<float>input(32*512,value);LocalTensor<float>tensor{input.data(),input.size()};
  vector.Bmm2DataCopyOut(ri,ci,tensor,0,input.size());
  uint64_t first=guard+(task*64+half*32)*512;
  for(uint64_t j=0;j<32*512;j++)assert(memory.at(first+j)==value);
 }
 assert(std::all_of(memory.begin()+guard+rows*512,memory.end()-guard,
  [](float v){return v==0.0f;}));
 for(unsigned i=0;i<guard;i++)assert(memory[i]==-1337.0f&&memory[memory.size()-i-1]==-1337.0f);
}
void check_empty_stats_order(){
 const uint64_t rows=64,guard=8;
 for(bool empty:{false,true}){
  std::vector<float>memory(guard+rows*(512+2)+guard,-1337.0f);
  workspaceBegin=reinterpret_cast<uint8_t*>(memory.data());
  workspaceEnd=reinterpret_cast<uint8_t*>(memory.data()+memory.size());
  Tiling tiling;tiling.splitParams.splitCount=1;
  tiling.splitParams.partialOutputOffsetBytes=guard*sizeof(float);
  tiling.splitParams.partialMaximumOffsetBytes=(guard+rows*512)*sizeof(float);
  tiling.splitParams.partialSumOffsetBytes=(guard+rows*(512+1))*sizeof(float);
  for(unsigned half=0;half<2;half++){
   Vector vector;vector.tilingData=&tiling;vector.kvSplitTaskId=0;
   vector.InitKvSplitWorkspace(workspaceBegin);
   std::vector<float>output(32*512,-55.0f),pv(32*512,1.0f),sum(32,empty?0.0f:2.0f);
   std::vector<float>maximum(32,empty?-9999.0f:4.0f),exponential(32,1.0f),lse(128,-88.0f);
   vector.stage2OutBuf.tensor={output.data(),output.size()};
   vector.softmaxSumBuf[0].tensor={sum.data(),sum.size()};
   vector.softmaxMaxBuf[0].tensor={maximum.data(),maximum.size()};
   vector.softmaxExpBuf[0].tensor={exponential.data(),exponential.size()};
   vector.lseUb={lse.data(),lse.size()};
   CubeBuffer buffer{{pv.data(),pv.size()}};RunInfo run;ConstInfo ci;ci.subBlockIdx=half;
   // Empty piece uses deliberately stale nonzero PV and sum0. LastDiv makes
   // Inf; only ZeroInvalidAttnOut with the original sentinel can remove it.
   // CleanNaNAttnOut intentionally does not erase Inf in this scalar model.
   vector.ProcessVec2(buffer,run,ci);
   const uint64_t first=guard+half*32*512;
   for(uint64_t j=0;j<32*512;j++)assert(memory[first+j]==(empty?0.0f:0.5f));
  }
  for(uint64_t head=0;head<64;head++){
   assert(memory[guard+rows*512+head]==(empty?0.0f:4.0f));
   assert(memory[guard+rows*(512+1)+head]==(empty?0.0f:2.0f));
  }
  for(unsigned i=0;i<guard;i++)assert(memory[i]==-1337.0f&&memory[memory.size()-i-1]==-1337.0f);
 }
}
int main(){
 for(unsigned queries:{1u,4u})for(unsigned parts:{1u,2u,4u}){
  check_indices(queries,parts);check_workspace(queries,parts);
 }
 check_empty_stats_order();
 std::cout<<"PASS: real split bodies; Q1/Q4; S1/S2/S4; TopK2048; tail68/empty; FP32 workspace fences\n";
}
'''


class KvSplitDeviceBodiesTest(unittest.TestCase):
    def test_actual_split_task_gather_and_partial_dma_with_ubsan(self):
        compiler = shutil.which('c++')
        if compiler is None:
            self.skipTest('C++ compiler unavailable')
        vector = (ARCH / 'kv_quant_sparse_flash_attention_service_vector_mla.h').read_text()
        kernel = (ARCH / 'kv_quant_sparse_flash_attention_kernel_mla.h').read_text()
        substitutions = {
            'BODY_REAL': actual_body(vector, '::GetRealCmpS2Idx('),
            'BODY_VALID': actual_body(vector, '::CalcSparseValidRows('),
            'BODY_WORKSPACE': actual_body(vector, 'void InitKvSplitWorkspace('),
            'BODY_EMPTY': actual_body(vector, 'void InitEmptyKvSplit('),
            'BODY_OUTPUT': actual_body(vector, '::Bmm2DataCopyOut ('),
            'BODY_STATS': actual_body(vector, '::CopyFALseToGm('),
            'BODY_VEC2': actual_body(vector, '::ProcessVec2('),
            'BODY_KERNEL': actual_body(kernel, '::InitCalcParamsEach('),
        }
        source = MODEL
        for marker, body in substitutions.items():
            source = source.replace(marker, body)
        with tempfile.TemporaryDirectory(prefix='qsfa-kv-split-device-') as directory:
            cpp = Path(directory) / 'split.cpp'
            executable = Path(directory) / 'split'
            cpp.write_text(source)
            compiled = subprocess.run([compiler, '-std=c++17', '-O2', '-fsanitize=undefined',
                                       '-fno-sanitize-recover=all', str(cpp), '-o', str(executable)],
                                      text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            tested = subprocess.run([str(executable)], text=True, capture_output=True)
            self.assertEqual(tested.returncode, 0, tested.stdout + tested.stderr)
            self.assertIn('S1/S2/S4', tested.stdout)

    def test_only_split_candidate_defers_inplace_stats_canonicalization(self):
        compiler = shutil.which('c++')
        if compiler is None:
            self.skipTest('C++ compiler unavailable')
        vector = (ARCH / 'kv_quant_sparse_flash_attention_service_vector_mla.h').read_text()
        for split in (0, 1):
            bodies = []
            for name in ('ProcessVec1', 'ProcessVec2'):
                source = ('#define QSFA_RECENT_KV_SPLIT ' + str(split) + '\n'
                          + actual_body(vector, '::' + name + '('))
                processed = subprocess.run([compiler, '-E', '-P', '-x', 'c++', '-'],
                                           input=source, text=True, capture_output=True)
                self.assertEqual(processed.returncode, 0, processed.stderr)
                bodies.append(processed.stdout)
            if split:
                self.assertNotIn('CopyFALseToGm(', bodies[0])
                self.assertGreater(bodies[1].index('CopyFALseToGm('),
                                   bodies[1].index('CopyOutAttentionOut('))
                self.assertGreater(bodies[1].index('CopyOutAttentionOut('),
                                   bodies[1].index('ZeroInvalidAttnOut('))
            else:
                self.assertIn('CopyFALseToGm(', bodies[0])
                self.assertNotIn('CopyFALseToGm(', bodies[1])


if __name__ == '__main__':
    unittest.main()
