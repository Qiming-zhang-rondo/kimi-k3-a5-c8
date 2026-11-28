"""Compile the real inline header against a bounded CPU API shim, not an NPU.

The shim checks arithmetic/byte-copy/RNE/control ordering only. It does not
establish Ascend lowering, real event resources, UB usage or vendor integration.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
HEADER=ROOT/"experimental/prolog_ring_smoke/prolog_ring_write_device.h"
SHIM=r'''
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <set>
#include <utility>
#include <type_traits>
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
namespace AscendC {
inline void Require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
struct bfloat16_t{uint16_t bits;};
enum class HardEvent{MTE2_S,S_MTE2,V_MTE3,MTE3_V,S_MTE3,MTE3_S};
enum class TPosition{VECCALC};
enum class RoundMode{CAST_RINT};
// The SDK allocates dynamic IDs as TEventID; event_t is a distinct enum.
// Keeping them distinct prevents this host model from hiding API type errors.
enum class event_t:uint16_t{EVENT_ID0=0};
using TEventID=int32_t;
static_assert(!std::is_convertible<TEventID,event_t>::value);
static_assert(!std::is_convertible<event_t,TEventID>::value);
inline std::set<std::pair<int,int>> reserved;
inline uint32_t initBytes=0,events=0,releases=0,loads=0;
inline bool metaReady=false,payloadPending=false,tagsPending=false;
inline std::vector<int> trace;
template<HardEvent E>void SetFlag(TEventID id){Require(reserved.count({int(E),id}),"event not owned");trace.push_back(100+int(E));}
template<HardEvent E>void WaitFlag(TEventID id){
 Require(reserved.count({int(E),id}),"event not owned");trace.push_back(200+int(E));
 if constexpr(E==HardEvent::MTE2_S)metaReady=true;
 if constexpr(E==HardEvent::MTE3_V)payloadPending=false;
 if constexpr(E==HardEvent::MTE3_S)tagsPending=false;
}
template<class T>struct LocalTensor{
 T* p;size_t n;
 LocalTensor operator[](size_t i)const{Require(i<=n,"local index bounds");return{p+i,n-i};}
 T GetValue(size_t i)const{Require(metaReady && i<n,"metadata read without MTE2_S");return p[i];}
 void SetValue(size_t i,T x)const{Require(!tagsPending && i<n,"tag source reused before MTE3_S");p[i]=x;}
};
template<class T>struct GlobalRef{T* p;};
template<class T>class GlobalTensor{
 T* p_=nullptr;
public:void SetGlobalBuffer(T* p){p_=p;}GlobalRef<T>operator[](uint64_t i)const{return{p_+i};}
};
template<TPosition P>struct TBuf{
 void* p=nullptr;size_t bytes=0;
 ~TBuf(){std::free(p);}
 template<class T>LocalTensor<T>Get(){return{reinterpret_cast<T*>(p),bytes/sizeof(T)};}
};
class TPipe{
public:
 template<TPosition P>void InitBuffer(TBuf<P>& b,uint32_t bytes){
  Require(bytes%32==0,"unaligned UB allocation");Require(posix_memalign(&b.p,32,bytes)==0,"allocation failed");
  std::memset(b.p,0,bytes);b.bytes=bytes;initBytes+=bytes;
 }
 template<HardEvent E>TEventID AllocEventID(){auto id=TEventID(++events+10);reserved.insert({int(E),id});return id;}
 template<HardEvent E>void ReleaseEventID(TEventID id){Require(reserved.erase({int(E),id})==1,"event release mismatch");++releases;}
};
struct DataCopyExtParams{uint16_t blockCount;uint32_t blockLen;uint32_t srcStride,dstStride,rsv;};
template<class T>struct DataCopyPadExtParams{bool isPad;uint8_t left,right;T value;};
template<class T>void DataCopyPad(LocalTensor<T> dst,GlobalRef<T> src,DataCopyExtParams bytes,DataCopyPadExtParams<T>){
 Require(bytes.blockCount==1 && bytes.blockLen<=dst.n*sizeof(T),"metadata copy bounds");
 std::memcpy(dst.p,src.p,bytes.blockLen);++loads;metaReady=false;
}
template<class T>void DataCopyPad(GlobalRef<T> dst,LocalTensor<T> src,DataCopyExtParams bytes){
 Require(bytes.blockCount==1 && bytes.blockLen<=src.n*sizeof(T),"store copy bounds");
 std::memcpy(dst.p,src.p,bytes.blockLen);
 if constexpr(sizeof(T)==2)payloadPending=true;
 else tagsPending=true;
}
inline uint16_t Rne(float value){uint32_t b;std::memcpy(&b,&value,4);return uint16_t((b+0x7fff+((b>>16)&1))>>16);}
inline void Cast(LocalTensor<bfloat16_t> dst,const LocalTensor<float>& src,RoundMode,uint32_t count){
 Require(!payloadPending,"cast scratch reused before MTE3_V");Require(count<=src.n && count<=dst.n,"Cast count bounds");
 for(uint32_t i=0;i<count;++i)dst.p[i].bits=Rne(src.p[i]);
}
}
'''
PROGRAM=r'''
#include "prolog_ring_write_device.h"
#include <iostream>
#include <array>
using namespace AscendC;
using namespace QsfaPrologRingWrite;
template<class T>GM_ADDR Gm(T* p){return reinterpret_cast<GM_ADDR>(p);}
int main(){try{
 TPipe pipe;
 {Writer off;Binding b;Require(off.Init(pipe,b),"OFF init rejected");off.Release();
  Require(initBytes==0 && events==0,"OFF allocated UB/events");Require(!off.ResolveRow(0).active,"OFF route active");}
 constexpr uint32_t P=2,C=128,T=4;
 std::vector<bfloat16_t> latent(P*C*512,bfloat16_t{0xbeef}),rope(P*C*64,bfloat16_t{0x1234});
 std::vector<int64_t> posTags(P*C,-1),epochTags(P*C,-1);
 int32_t banks[T]={0,1,0,-1};int64_t positions[T]={129,129,130,3},epochs[T]={7,8,7,9};uint8_t valid[T]={1,1,0,1};
 Binding b{Gm(latent.data()),Gm(rope.data()),Gm(posTags.data()),Gm(epochTags.data()),Gm(banks),Gm(positions),Gm(epochs),Gm(valid),T,P,C,1};
 {Writer invalid;auto bad=b;bad.capacity=129;Require(!invalid.Init(pipe,bad),"unaligned capacity accepted");Require(initBytes==0,"bad geometry allocated");}
 Writer writer;Require(writer.Init(pipe,b),"ON init rejected");Require(initBytes==1216 && events==6,"UB/event budget mismatch");
 Require(!writer.Init(pipe,b),"double init accepted");
 auto a=writer.ResolveRow(0),other=writer.ResolveRow(1);
 Require(a.active && a.ringRow==1 && other.ringRow==129,"bank and position confused");
 Require(!writer.ResolveRow(2).active && !writer.ResolveRow(3).active,"padding/negative bank active");
 auto beforeLoads=loads;Require(!writer.ResolveRow(T).active && loads==beforeLoads,"out-of-range metadata read");
 alignas(32) float src[512];alignas(32)bfloat16_t kr[64];
 for(uint32_t i=0;i<512;++i)src[i]=float(i-256)/32.0f;
 src[0]=1.00390625f;src[1]=1.01171875f;
 for(uint32_t i=0;i<64;++i)kr[i].bits=uint16_t(0x3f00+i);
 auto saved=std::array<float,512>{};std::memcpy(saved.data(),src,sizeof(src));
 LocalTensor<float> source{src,512};LocalTensor<bfloat16_t> ropeSource{kr,64};
 Require(writer.WriteLatentSlice(a,source,0,512),"full latent failed");
 Require(writer.WriteRopeSlice(a,ropeSource,0,64),"full rope failed");
 for(uint32_t i=0;i<512;++i)Require(latent[512+i].bits==Rne(src[i]),"wrong RNE latent");
 Require(latent[512].bits==0x3f80 && latent[513].bits==0x3f82,"BF16 ties not even");
 writer.PublishTags(a,false);Require(posTags[1]==-1,"nonowner published tags");
 writer.PublishTags(a,true);Require(posTags[1]==129 && epochTags[1]==7,"tags wrong");
 Require(writer.WriteLatentSlice(other,source,7,17),"unaligned GM partial latent failed");
 Require(writer.WriteRopeSlice(other,ropeSource,3,5),"unaligned GM partial rope failed");
 writer.PublishTags(other,true);
 for(uint32_t i=0;i<512;++i)Require(latent[129*512+i].bits==(i>=7&&i<24?Rne(src[i-7]):uint16_t(0xbeef)),"partial latent stomped adjacent columns");
 for(uint32_t i=0;i<64;++i)Require(rope[129*64+i].bits==(i>=3&&i<8?kr[i-3].bits:uint16_t(0x1234)),"partial rope stomped adjacent columns");
 Require(posTags[129]==129 && epochTags[129]==8,"other bank tags wrong");
 Require(!writer.WriteLatentSlice(a,source,511,2) && !writer.WriteRopeSlice(a,ropeSource,63,2),"slice overrun accepted");
 Require(std::memcmp(src,saved.data(),sizeof(src))==0,"private quantizer input changed");
 positions[0]=300;epochs[0]=22;banks[0]=1;auto fresh=writer.ResolveRow(0);
 Require(fresh.active && fresh.ringRow==172 && fresh.position==300 && fresh.epoch==22,"stale dynamic metadata");
 writer.PublishTags(fresh,true);Require(posTags[172]==300 && epochTags[172]==22,"fresh tags wrong");
 valid[0]=0;Require(!writer.ResolveRow(0).active,"new padding metadata stale");
 writer.Release();Require(releases==6 && reserved.empty(),"event reservation leaked");
 Require(!payloadPending && !tagsPending,"outstanding source DMA at release");
 std::cout<<"HOST_HEADER_MODEL_PASS\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
'''


class PrologRingDeviceHelperTest(unittest.TestCase):
    def test_actual_header_runs_bounded_host_model_with_ubsan(self):
        compiler=shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if not compiler:self.skipTest("Host C++ compiler unavailable; no NPU result implied")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"kernel_operator.h").write_text(SHIM)
            (root/"test.cpp").write_text(PROGRAM)
            executable=root/"check"
            build=subprocess.run([compiler,"-std=c++17","-O1","-fsanitize=undefined","-fno-sanitize-recover=all",
                "-D__NPU_ARCH__=3510","-D__DAV_C310_VEC__","-I"+str(root),"-I"+str(HEADER.parent),
                str(root/"test.cpp"),"-o",str(executable)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            result=subprocess.run([str(executable)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("HOST_HEADER_MODEL_PASS",result.stdout)
            # Reintroducing the deployed type mistake must fail against this
            # strict shim, rather than silently passing through an integer alias.
            (root/"wrong_event_header.h").write_text(HEADER.read_text().replace("TEventID","event_t"))
            (root/"wrong_event.cpp").write_text(PROGRAM.replace(
                '#include "prolog_ring_write_device.h"','#include "wrong_event_header.h"'))
            wrong=subprocess.run([compiler,"-std=c++17","-fsyntax-only",
                "-D__NPU_ARCH__=3510","-D__DAV_C310_VEC__","-I"+str(root),
                str(root/"wrong_event.cpp")],capture_output=True,text=True)
            self.assertNotEqual(wrong.returncode,0,"Strict shim accepted dynamic IDs as event_t")
            self.assertIn("event_t",wrong.stderr)

    def test_header_is_inline_vector_component_without_new_launch_or_host_abi(self):
        text=HEADER.read_text()
        self.assertIn("__NPU_ARCH__ == 3510 && defined(__DAV_C310_VEC__)",text)
        self.assertNotIn("__global__",text)
        self.assertNotIn("KERNEL_TASK_TYPE",text)
        self.assertNotIn("<<<",text)
        self.assertNotIn("extern \"C\"",text)
        self.assertNotIn("EVENT_ID0",text.replace("never steal EVENT_ID0", "never steal an unowned ID"))
        self.assertIn("DataCopyPad",text)
        self.assertIn("uniqueTagOwner",text)
        self.assertIn("WHOLE producer kernel",text)


if __name__=="__main__":unittest.main()
