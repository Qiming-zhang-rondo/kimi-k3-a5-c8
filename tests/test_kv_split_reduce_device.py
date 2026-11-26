"""Run actual KV-split VF/controller bodies under a checked C++64-lane model.

This verifies arithmetic, compact indexing, skipped empty inputs, DMA extents,
events and the final rounding boundary. It does not validate AscendC lowering,
MicroAPI accuracy, device synchronization or NPU performance.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "experimental/single_pass/kv_split_reduce.asc"


def actual_function(text, signature):
    start = text.index("template <uint32_t Splits>\n" + signature)
    brace = text.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


STUB = r'''
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#define __aicore__
#define __simd_vf__
#define __ubuf__
#define __gm__
void require(bool good, const char* what) { if (!good) throw std::runtime_error(what); }
namespace AscendC {
constexpr unsigned PIPE_V=1, EVENT_ID0=0;
enum class HardEvent { MTE2_V,V_MTE3,MTE3_V,MTE2_S,S_MTE2 };
enum class TPosition { VECCALC };
enum class RoundMode { CAST_RINT };
enum class CMPMODE { GT };
unsigned blockIndex=0, blockCount=1, currentSplits=1;
size_t currentRowHead=0, inputFloats=0, scalarCount=0;
float *inputBase=nullptr, *maximumBase=nullptr, *sumBase=nullptr;
std::vector<bool> activeHeads;
std::vector<float> beforeCast;
unsigned pending[5]={}, maximumReads=0, outputReads=0, exponentials=0, vfLoads=0;
unsigned gmScalarReads=0,statsDmaReads=0,statsDmaEpoch=0,scalarReadyEpoch=0;
struct ScalarSlot { bool maximum;size_t index;unsigned epoch; };
std::map<const float*,ScalarSlot> scalarSlots;
unsigned statWrites=0;
uint32_t GetBlockIdx() { return blockIndex; }
uint32_t GetBlockNum() { return blockCount; }
template<HardEvent E> void SetFlag(unsigned id) {
 require(id==0 && pending[unsigned(E)]++==0,"double/invalid event set");
}
template<HardEvent E> void WaitFlag(unsigned id) {
 require(id==0 && pending[unsigned(E)]--==1,"unmatched event wait");
 if constexpr(E==HardEvent::MTE2_S) scalarReadyEpoch=statsDmaEpoch;
}
template<unsigned> void PipeBarrier() {}
struct bfloat16_t { uint16_t bits=0; };
uint16_t bf16(float value) {
 uint32_t bits; std::memcpy(&bits,&value,4);
 if ((bits&0x7f800000U)==0x7f800000U)
  return uint16_t((bits>>16) | ((bits&0x007fffffU)?0x40U:0));
 return uint16_t((bits+0x7fffU+((bits>>16)&1U))>>16);
}
float unbf16(uint16_t value) { uint32_t bits=uint32_t(value)<<16; float x;std::memcpy(&x,&bits,4);return x; }
template<class T> struct LocalTensor {
 T* p=nullptr; size_t count=0;
 void* GetPhyAddr() const { require(p!=nullptr,"absent UB"); return p; }
 LocalTensor operator[](size_t i) const { require(i<=count,"UB index");return {p+i,count-i}; }
 T GetValue(size_t i) const {
  require(p && i<count,"UB scalar bounds");
  if constexpr(std::is_same<T,float>::value) {
   auto slot=scalarSlots.find(p+i);
   require(slot!=scalarSlots.end() && slot->second.epoch<=scalarReadyEpoch,
           "UB stats read without completed MTE2_S");
   if(slot->second.maximum) {
    require(activeHeads[slot->second.index],"empty max entered scalar arithmetic"); ++maximumReads;
   } else {
    const size_t index=slot->second.index;
    currentRowHead=(index/(currentSplits*64))*64+index%64;
   }
  }
  return p[i];
 }
};
template<class T> struct GlobalTensor {
 T* p=nullptr;
 void SetGlobalBuffer(T* value) { p=value; }
 GlobalTensor operator[](size_t i) const { require(p!=nullptr,"absent GM");return {p+i}; }
 T GetValue(size_t i) const {
  require(p!=nullptr,"absent GM scalar");
  if constexpr(std::is_same<T,float>::value) {
   require((p==maximumBase || p==sumBase) && i<scalarCount,"compact stats bounds");
   // Simulate an old DCache line surviving a preceding producer overwrite.
   // DMA sees f.maximum/f.sum, but an uninvalidated scalar GM read sees zero.
   ++gmScalarReads;return T(0);
  }
  return p[i];
 }
};
template<TPosition> struct TBuf {
 std::vector<uint32_t> storage;
 template<class T> LocalTensor<T> Get() { return {reinterpret_cast<T*>(storage.data()),storage.size()*4/sizeof(T)}; }
};
struct TPipe {
 template<TPosition P> void InitBuffer(TBuf<P>& value,size_t bytes) {
  require(bytes%4==0,"unaligned allocation");value.storage.assign(bytes/4,0x7fc00000U);
 }
};
template<class T> void DataCopy(LocalTensor<T> destination,GlobalTensor<T> source,unsigned count) {
 require(destination.p && source.p && destination.count>=count,"copy-in extent");
 if constexpr(std::is_same<T,float>::value) {
  const size_t offset=source.p-inputBase;
  require(offset<inputFloats && offset+count<=inputFloats && offset%512==0 && count==512,"partial O bounds");
  require(activeHeads[offset/512],"empty/poison O was loaded"); ++outputReads;
 }
 std::memcpy(destination.p,source.p,count*sizeof(T));
}
template<class T> void DataCopy(GlobalTensor<T> destination,LocalTensor<T> source,unsigned count) {
 require(destination.p && source.p && source.count>=count,"copy-out extent");
 std::memcpy(destination.p,source.p,count*sizeof(T));
}
struct DataCopyExtParams { uint16_t blockCount;uint32_t blockLen,srcStride,dstStride,rsv; };
template<class T> struct DataCopyPadExtParams { bool isPad;uint8_t leftPadding,rightPadding;T paddingValue; };
template<class T> void DataCopyPad(LocalTensor<T> destination,GlobalTensor<T> source,
 const DataCopyExtParams& c,const DataCopyPadExtParams<T>& pad) {
 require(destination.p && source.p && c.blockCount==1 && c.blockLen==4 && c.srcStride==0 && c.dstStride==0,
         "compact input stats DMA");
 require(pad.isPad && pad.leftPadding==0 && pad.rightPadding==7 && destination.count>=8,
         "input stats UB 32B padding");
 const bool isMaximum=source.p>=maximumBase && source.p<maximumBase+scalarCount;
 const bool isSum=source.p>=sumBase && source.p<sumBase+scalarCount;
 require(isMaximum || isSum,"stats input GM bounds");
 const size_t index=source.p-(isMaximum?maximumBase:sumBase);
 destination.p[0]=source.p[0];
 std::fill(destination.p+1,destination.p+8,pad.paddingValue);
 scalarSlots[destination.p]={isMaximum,index,++statsDmaEpoch};++statsDmaReads;
}
template<class T> void DataCopyPad(GlobalTensor<T> destination,LocalTensor<T> source,const DataCopyExtParams& c) {
 require(destination.p && source.p && c.blockCount==1 && c.blockLen==4 && c.srcStride==0 && c.dstStride==0,
         "compact scalar statistics DMA");
 std::memcpy(destination.p,source.p,4);++statWrites;
}
template<class T> void Cast(LocalTensor<bfloat16_t> destination,LocalTensor<T> source,RoundMode mode,unsigned count) {
 require(mode==RoundMode::CAST_RINT && destination.count>=count && source.count>=count && count==512,"final cast");
 for(unsigned i=0;i<count;++i) {
  beforeCast[currentRowHead*512+i]=source.p[i];
  destination.p[i].bits=bf16(source.p[i]);
 }
}
namespace MicroAPI {
enum class MaskPattern { ALL };
enum class LoadDist { DIST_NORM };
enum class StoreDist { DIST_NORM_B32 };
struct MaskReg { std::array<bool,64> x{}; };
template<class T> struct RegTensor {
 std::array<T,64> x;
 RegTensor() { x.fill(std::numeric_limits<T>::quiet_NaN()); }
};
template<class T,MaskPattern> MaskReg CreateMask() { MaskReg mask;mask.x.fill(true);return mask; }
template<class T> void Duplicate(RegTensor<T>& output,T value) { output.x.fill(value); }
template<class T,LoadDist> void LoadAlign(RegTensor<T>& output,T* source) {
 require(source!=nullptr,"absent VF input");++vfLoads;
 for(unsigned i=0;i<64;++i) output.x[i]=source[i];
}
template<class T,StoreDist> void StoreAlign(T* destination,RegTensor<T>& value,MaskReg& mask) {
 require(destination!=nullptr,"absent VF output");
 for(unsigned i=0;i<64;++i) if(mask.x[i]) destination[i]=value.x[i];
}
template<class T> void Exp(RegTensor<T>& output,RegTensor<T>& input,MaskReg& mask) {
 ++exponentials;for(unsigned i=0;i<64;++i) if(mask.x[i]) output.x[i]=std::exp(input.x[i]);
}
template<class T> void Muls(RegTensor<T>& output,RegTensor<T>& input,T scalar,MaskReg& mask) {
 for(unsigned i=0;i<64;++i) if(mask.x[i]) output.x[i]=input.x[i]*scalar;
}
#define BINARY(NAME,EXPR) template<class T> void NAME(RegTensor<T>& o,RegTensor<T>& a,RegTensor<T>& b,MaskReg& mask) { \
 for(unsigned i=0;i<64;++i) if(mask.x[i]) { const T x=a.x[i],y=b.x[i];o.x[i]=(EXPR); } }
BINARY(Add,x+y)
BINARY(Mul,x*y)
BINARY(Div,x/y)
#undef BINARY
template<class T,CMPMODE C> void Compares(MaskReg& output,RegTensor<T>& a,T b,MaskReg& mask) {
 for(unsigned i=0;i<64;++i) output.x[i]=mask.x[i] && a.x[i]>b;
}
template<class T> void Select(RegTensor<T>& output,RegTensor<T>& a,RegTensor<T>& b,MaskReg& mask) {
 for(unsigned i=0;i<64;++i) output.x[i]=mask.x[i]?a.x[i]:b.x[i];
}
} // MicroAPI
} // AscendC
namespace QsfaKvSplitReduction {
using namespace AscendC;
constexpr uint32_t kHeads=64,kDim=512,kFp32Lanes=64;
'''


DRIVER = r'''
} // Actual QsfaKvSplitReduction function bodies above, unchanged.
using namespace AscendC;
struct Fixture {
 unsigned splits=1,queries=2;
 std::vector<float> partial,maximum,sum;
 std::vector<double> expected;
};
Fixture make(unsigned splits,unsigned present,float offset,bool normalized=false) {
 Fixture f;f.splits=splits;
 const unsigned count=f.queries*splits*64;
 f.partial.assign(count*512,std::numeric_limits<float>::quiet_NaN());
 f.maximum.assign(count,0.0f);f.sum.assign(count,0.0f);
 f.expected.assign(f.queries*64*512,0.0);
 for(unsigned q=0;q<f.queries;++q) for(unsigned h=0;h<64;++h) {
  double m=-std::numeric_limits<double>::infinity(),denominator=0;
  std::vector<double> scores;std::vector<std::array<double,512>> values;
  for(unsigned s=0;s<splits;++s) {
   const unsigned at=(q*splits+s)*64+h;
   if(!(present&(1U<<s))) continue;
   const unsigned n=normalized? (s==0?3:s==1?7:s==2?2:5):3+s*2;
   std::vector<float> logits(n);
   for(unsigned j=0;j<n;++j) logits[j]=offset+(normalized?0.0f:float(s)*0.75f+float(j)*0.25f-float(q+h%3)*0.5f);
   const float localMax=*std::max_element(logits.begin(),logits.end());
   double sum=0;for(float score:logits) sum+=std::exp(double(score)-localMax);
   f.maximum[at]=localMax;f.sum[at]=float(sum);
   for(unsigned d=0;d<512;++d) {
    double numerator=0;
    for(unsigned j=0;j<n;++j) {
     const float value=normalized? (s==0?2.0f:s==1?10.0f:s==2?-4.0f:1.0f):
       float(1.001+0.002*s+0.001*q+0.00002*h+0.000002*d+0.0001*j);
     numerator+=std::exp(double(logits[j])-localMax)*value;
    }
    f.partial[at*512+d]=float(numerator/sum);
   }
   for(unsigned j=0;j<n;++j) {
    scores.push_back(logits[j]);m=std::max(m,double(logits[j]));
    std::array<double,512> row;
    for(unsigned d=0;d<512;++d) row[d]=normalized?(s==0?2.0f:s==1?10.0f:s==2?-4.0f:1.0f):
       float(1.001+0.002*s+0.001*q+0.00002*h+0.000002*d+0.0001*j);
    values.push_back(row);
   }
  }
  for(double score:scores) denominator+=std::exp(score-m);
  for(unsigned d=0;d<512;++d) {
   double numerator=0;
   for(unsigned j=0;j<scores.size();++j) numerator+=std::exp(scores[j]-m)*values[j][d];
   f.expected[(q*64+h)*512+d]=denominator>0?numerator/denominator:0;
  }
 }
 return f;
}
template<unsigned Splits> std::vector<float> execute(Fixture f,unsigned blocks,bool stats,unsigned emptyMax) {
 require(f.splits==Splits,"fixture split count");
 activeHeads.assign(f.sum.size(),false);
 for(unsigned i=0;i<f.sum.size();++i) {
  activeHeads[i]=f.sum[i]>0;
  if(!activeHeads[i]) f.maximum[i]=emptyMax==0?0.0f:emptyMax==1?
    -std::numeric_limits<float>::infinity():std::numeric_limits<float>::quiet_NaN();
 }
 const unsigned rowHeads=f.queries*64;
 const uint16_t bfGuard=0x4321;
 std::vector<bfloat16_t> output(rowHeads*512+2);output.front().bits=output.back().bits=bfGuard;
 std::vector<float> outM(rowHeads+2,-12345),outS(rowHeads+2,-12345);
 beforeCast.assign(rowHeads*512,std::numeric_limits<float>::quiet_NaN());
 inputBase=f.partial.data();inputFloats=f.partial.size();maximumBase=f.maximum.data();sumBase=f.sum.data();
 scalarCount=f.sum.size();currentSplits=Splits;blockCount=blocks;
 maximumReads=outputReads=exponentials=vfLoads=statWrites=gmScalarReads=statsDmaReads=0;
 statsDmaEpoch=scalarReadyEpoch=0;scalarSlots.clear();
 for(blockIndex=0;blockIndex<blocks;++blockIndex) {
  QsfaKvSplitReduction::Run<Splits>(reinterpret_cast<uint8_t*>(f.partial.data()),
    reinterpret_cast<uint8_t*>(f.maximum.data()),reinterpret_cast<uint8_t*>(f.sum.data()),
    reinterpret_cast<uint8_t*>(output.data()+1),stats?reinterpret_cast<uint8_t*>(outM.data()+1):nullptr,
    stats?reinterpret_cast<uint8_t*>(outS.data()+1):nullptr,f.queries,unsigned(stats));
  for(unsigned event:pending) require(event==0,"pending event after controller");
 }
 require(output.front().bits==bfGuard && output.back().bits==bfGuard,"output guard corrupted");
 require(outM.front()==-12345 && outM.back()==-12345 && outS.front()==-12345 && outS.back()==-12345,
         "compact stats guard corrupted");
 const unsigned active=std::count(activeHeads.begin(),activeHeads.end(),true);
 require(maximumReads==active && outputReads==active && exponentials==active && vfLoads==active*8,
         "empty input consumed or active partial omitted");
 require(gmScalarReads==0 && statsDmaReads==2*rowHeads*Splits,
         "producer statistics used stale scalar GM cache instead of DMA");
 require(statWrites==(stats?2*rowHeads:0),"optional stats copy count");
 for(unsigned row=0;row<rowHeads;++row) {
  const unsigned q=row/64,h=row%64;
  double maximum=-std::numeric_limits<double>::infinity(),sum=0;
  for(unsigned s=0;s<Splits;++s) {
   const unsigned at=(q*Splits+s)*64+h;
   if(f.sum[at]>0) maximum=std::max(maximum,double(f.maximum[at]));
  }
  for(unsigned s=0;s<Splits;++s) {
   const unsigned at=(q*Splits+s)*64+h;
   if(f.sum[at]>0) sum+=f.sum[at]*std::exp(double(f.maximum[at])-maximum);
  }
  if(stats) {
   require((std::isinf(maximum)&&outM[row+1]==-std::numeric_limits<float>::infinity()) || outM[row+1]==maximum,
           "final maximum mismatch");
   require(std::abs(outS[row+1]-sum)<=2e-6*std::max(sum,1.0),"final normalized sum mismatch");
  } else require(outM[row+1]==-12345 && outS[row+1]==-12345,"disabled stats changed");
  for(unsigned d=0;d<512;++d) {
   const unsigned at=row*512+d;
   require(std::isfinite(beforeCast[at]),"nonfinite reduced O");
   require(std::abs(beforeCast[at]-f.expected[at])<=2e-6*std::max(1.0,std::abs(f.expected[at])),
           "actual FP32 VF differs from independently concatenated full attention");
   require(output[at+1].bits==bf16(beforeCast[at]),"BF16 final rounding boundary mismatch");
  }
 }
 return beforeCast;
}
template<unsigned S> void empty_suite() {
 for(unsigned bits=0;bits<(1U<<S);++bits) for(unsigned empty=0;empty<3;++empty)
  execute<S>(make(S,bits,-10000.0f),3,true,empty);
}
template<unsigned S> void shifts_suite() {
 const unsigned full=(1U<<S)-1;
 auto original=execute<S>(make(S,full,0.0f),16,true,0);
 for(float shift:{-1024.0f,1024.0f}) {
  auto shifted=execute<S>(make(S,full,shift),16,true,1);
  require(original==shifted,"global logit shift changed output");
 }
 auto noStats=execute<S>(make(S,full,0.0f),7,false,0);
 require(original==noStats,"optional statistics changed output");
}
int main(int argc,char** argv) {
 try {
  require(argc==2,"suite argument");std::string suite=argv[1];
  if(suite=="empty") { empty_suite<1>();empty_suite<2>();empty_suite<4>(); }
  else if(suite=="shifts") { shifts_suite<1>();shifts_suite<2>();shifts_suite<4>(); }
  else if(suite=="normalized") {
   execute<1>(make(1,1,0.0f,true),1,true,0);
   execute<2>(make(2,3,0.0f,true),32,true,0);
   execute<4>(make(4,15,0.0f,true),32,true,0);
  } else throw std::runtime_error("unknown suite");
  std::cout<<"PASS "<<suite<<" actual VF+controller S1/S2/S4\n";
 } catch(const std::exception& e) { std::cerr<<e.what()<<"\n";return 2; }
}
'''


class KvSplitReduceDeviceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("Host C++ compiler unavailable; no device claim")
        cls.temp = tempfile.TemporaryDirectory(prefix="qsfa-split-reduce-model-")
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        actual = SOURCE.read_text()
        functions = actual_function(actual, "__simd_vf__ void ReduceVf(") + "\n" + actual_function(
            actual, "__aicore__ inline void Run(")
        cpp = root / "actual_reduce.cpp"
        cpp.write_text(STUB + functions + DRIVER)
        cls.binary = root / "actual_reduce"
        built = subprocess.run([compiler, "-std=c++17", "-O1", "-ffp-contract=off",
                                "-fsanitize=undefined", "-fno-sanitize-recover=all",
                                str(cpp), "-o", str(cls.binary)], capture_output=True, text=True)
        if built.returncode:
            raise AssertionError(built.stderr)

    def run_suite(self, suite):
        result = subprocess.run([str(self.binary), suite], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS " + suite, result.stdout)

    def test_all_empty_masks_skip_poison_max_and_o_even_at_negative_active_max(self):
        self.run_suite("empty")

    def test_global_score_shift_and_optional_stats_preserve_fp32_result(self):
        self.run_suite("shifts")

    def test_actual_vf_consumes_normalized_partials_and_final_stats(self):
        self.run_suite("normalized")

    def test_dynamic_stats_ignore_simulated_stale_global_scalar_cache(self):
        self.run_suite("normalized")


if __name__ == "__main__":
    unittest.main()
