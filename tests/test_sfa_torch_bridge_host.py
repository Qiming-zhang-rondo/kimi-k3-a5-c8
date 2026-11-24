"""Compile and execute the actual Torch bridge using small host API fakes.

No Torch/CANN SDK, NPU, ACL synchronization, fatal signals or model messages.
This verifies Prepared's real control flow, pointer/stream wiring and POD, not
PyTorch ABI compatibility, ASC compilation or device correctness/performance.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "experimental/torch/sfa_direct_bridge.cpp"

# Intentionally no Tensor.cpu/item or synchronization/reset API is provided.
TORCH = r'''
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(condition,...) do {if(!(condition)) throw std::runtime_error("TORCH_CHECK");} while(0)
namespace Probe {
inline int emptyCalls=0,toCalls=0,guardCalls=0,launches=0,expectedDevice=3;
inline bool flushed=false;
inline void* rawStream=reinterpret_cast<void*>(uintptr_t{123});
inline std::vector<void*> records;
inline std::vector<std::string> order;
inline std::map<void*,size_t> allocations;
}
namespace c10 {
enum class DeviceType {CPU,PrivateUse1};
struct Device {
 DeviceType kind=DeviceType::CPU; int ordinal=0;
 Device()=default;Device(DeviceType type,int index=0):kind(type),ordinal(index){}
 DeviceType type()const{return kind;} int index()const{return ordinal;}
 bool operator==(Device other)const{return kind==other.kind&&ordinal==other.ordinal;}
};
}
namespace at {
enum ScalarType {kByte,kBool,kBFloat16,kFloat,kInt};
inline constexpr auto kCPU=c10::DeviceType::CPU;
inline size_t Width(ScalarType t){return t==kBFloat16?2:(t==kFloat||t==kInt?4:1);}
struct IntArrayRef {
 std::vector<int64_t> values;
 IntArrayRef(std::initializer_list<int64_t> v):values(v){}
 explicit IntArrayRef(std::vector<int64_t> v):values(std::move(v)){}
 bool equals(const IntArrayRef& other)const{return values==other.values;}
};
struct TensorOptions {
 c10::Device dev;ScalarType type=kFloat;
 TensorOptions device(c10::Device d)const{auto copy=*this;copy.dev=d;return copy;}
 TensorOptions device(c10::DeviceType d)const{return device(c10::Device(d));}
 TensorOptions dtype(ScalarType d)const{auto copy=*this;copy.type=d;return copy;}
};
struct Storage {void* pointer;void* data_ptr()const{return pointer;}};
struct Tensor {
 std::shared_ptr<std::vector<uint8_t>> bytes;
 std::vector<int64_t> shape;c10::Device dev;ScalarType type=kFloat;
 bool contiguous=true,grad=false;size_t byteOffset=0;
 bool defined()const{return bool(bytes);}c10::Device device()const{return dev;}
 bool is_contiguous()const{return contiguous;}bool requires_grad()const{return grad;}
 ScalarType scalar_type()const{return type;}IntArrayRef sizes()const{return IntArrayRef(shape);}
 int64_t numel()const{int64_t count=1;for(auto n:shape)count*=n;return count;}
 size_t element_size()const{return Width(type);}
 void* data_ptr()const{return bytes?bytes->data()+byteOffset:nullptr;}
 Storage storage()const{return {bytes?bytes->data():nullptr};}
 TensorOptions options()const{return TensorOptions().device(dev).dtype(type);}
 Tensor to(c10::Device device,ScalarType dtype,bool,bool)const;
};
inline Tensor empty(std::initializer_list<int64_t> shape,TensorOptions options){
 ++Probe::emptyCalls;Tensor t;t.shape=shape;t.dev=options.dev;t.type=options.type;
 size_t count=1;for(auto n:shape)count*=static_cast<size_t>(n);
 t.bytes=std::make_shared<std::vector<uint8_t>>(count*Width(t.type));Probe::allocations[t.data_ptr()]=t.bytes->size();return t;
}
inline Tensor empty_like(const Tensor& input){
 ++Probe::emptyCalls;Tensor t=input;t.bytes=std::make_shared<std::vector<uint8_t>>(input.bytes->size());return t;
}
inline Tensor Tensor::to(c10::Device device,ScalarType dtype,bool,bool)const{
 ++Probe::toCalls;Tensor t=*this;t.dev=device;t.type=dtype;
 t.bytes=std::make_shared<std::vector<uint8_t>>(*bytes);return t;
}
}
namespace pybind11 {
struct dict {std::map<std::string,int64_t> values;int64_t& operator[](const char* name){return values[name];}};
struct arg {explicit arg(const char*){}template<class T>arg operator=(T)const{return *this;}};
struct module_ {template<class...Args>void def(const char*,Args...) {}};
template<class T,class Holder>struct class_ {
 class_(module_&,const char*){}template<class...Args>class_& def(const char*,Args...){return *this;}
};
}
#define PYBIND11_MODULE(name,module) void FakePythonModule(pybind11::module_& module)
'''

NPU = r'''
#pragma once
#include <torch/extension.h>
namespace c10_npu {
struct NPUGuard {explicit NPUGuard(c10::Device){++Probe::guardCalls;}};
struct NPUStream {
 void* stream(bool drain=true)const{
  if(!drain)throw std::runtime_error("host queue was not flushed");
  Probe::flushed=true;Probe::order.push_back("flush");return Probe::rawStream;
 }
};
inline NPUStream getCurrentNPUStream(int device){
 if(device!=Probe::expectedDevice)throw std::runtime_error("wrong current device");return {};
}
namespace NPUCachingAllocator {
inline void recordStream(void* pointer,const NPUStream&){
 if(!Probe::flushed)throw std::runtime_error("record before stream flush");
 Probe::records.push_back(pointer);Probe::order.push_back("record");
}
}
}
'''

DRIVER = r'''
#include "ACTUAL_BRIDGE"
#include <functional>
#include <iostream>
#include <limits>
#include <array>
#include <set>
namespace Probe {
inline uint32_t abi=3,descriptorBytes=sizeof(SfaDirectBuffers),lastCount=1;
inline SfaDirectBuffers last{};inline bool lastFused=false,lastBf16=false;
void Require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void Begin(){flushed=false;records.clear();order.clear();}
void Launched(void* stream,const SfaDirectBuffers* buffers,bool fused,bool bf16=false){
 Require(stream==rawStream,"launch did not use current stream");
 Require(flushed&&!order.empty()&&order.front()=="flush","launch before stream drain");
 Require(order.back()=="record","missing allocator records before launch");
 last=*buffers;lastFused=fused;lastBf16=bf16;++launches;order.push_back("launch");
 const std::array<uint8_t*,19> fields{{buffers->query,buffers->key,buffers->value,buffers->indices,
  buffers->blockTable,buffers->actualQuery,buffers->actualKv,buffers->queryRope,buffers->keyRope,
  buffers->output,buffers->maximum,buffers->sum,buffers->workspace,buffers->tiling,buffers->coldOutput,
  buffers->coldMax,buffers->coldSum,buffers->coldPresent,buffers->hotPresent}};
 for(unsigned i=0;i<(fused?19:14);++i)
  Require(fields[i]&&std::find(records.begin(),records.end(),fields[i])!=records.end(),"unrecorded tensor storage");
 Require(records.size()==size_t(fused?19:14),"wrong allocator record count");
}
}
extern "C" uint32_t SfaDirectAbiVersion(void){return Probe::abi;}
extern "C" uint32_t SfaDirectBuffersBytes(void){return Probe::descriptorBytes;}
extern "C" uint32_t SfaDirectMergeTileRows(void){return 8;}
extern "C" int32_t SfaLaunchOriginal(void* stream,const SfaDirectBuffers* buffers,uint32_t n){Probe::lastCount=n;Probe::Launched(stream,buffers,false);return 0;}
extern "C" int32_t SfaLaunchFused(void* stream,const SfaDirectBuffers* buffers,uint32_t n){Probe::lastCount=n;Probe::Launched(stream,buffers,true);return 0;}
extern "C" int32_t SfaLaunchFusedBf16(void* stream,const SfaDirectBuffers* buffers,uint32_t n){Probe::lastCount=n;Probe::Launched(stream,buffers,true,true);return 0;}
at::Tensor Make(std::initializer_list<int64_t> shape,at::ScalarType dtype){
 return at::empty(shape,at::TensorOptions().device(c10::Device(c10::DeviceType::PrivateUse1,3)).dtype(dtype));
}
struct Inputs {
 at::Tensor q=Make({4,64,512},at::kBFloat16),k=Make({2,128,1,512},at::kBFloat16),v=k;
 at::Tensor idx=Make({4,1,128},at::kInt),table=Make({1,2},at::kInt);
 at::Tensor aq=Make({1},at::kInt),ak=Make({1},at::kInt);
 at::Tensor qr=Make({4,64,64},at::kBFloat16),kr=Make({2,128,1,64},at::kBFloat16);
 at::Tensor co=Make({4,64,512},at::kFloat),cm=Make({4,64,1},at::kFloat),cs=Make({4,64,1},at::kFloat);
 at::Tensor cp=Make({4},at::kBool),hp=Make({4},at::kBool);
};
State Native(Prepared& plan,Inputs& x){return plan.NativeHot(x.q,x.k,x.v,x.idx,x.table,x.aq,x.ak,x.qr,x.kr);}
State Fused(Prepared& plan,Inputs& x){return plan.FusedHot(x.q,x.k,x.v,x.idx,x.table,x.aq,x.ak,x.qr,x.kr,x.co,x.cm,x.cs,x.cp,x.hp);}
void Reject(const std::function<void()>& operation){
 const int before=Probe::launches;bool rejected=false;Probe::Begin();
 try{operation();}catch(const std::exception&){rejected=true;}
 Probe::Require(rejected,"invalid arguments accepted");
 Probe::Require(Probe::launches==before,"invalid arguments launched kernel");
 Probe::Require(!Probe::flushed&&Probe::records.empty(),"rejection happened after stream drain/record");
}
void CheckTiling(bool stats){
 const auto& t=*reinterpret_cast<const sfa_fixed::Tiling*>(Probe::last.tiling);
 Probe::Require(sizeof(t)==144,"wrong POD bytes");
 Probe::Require(t.baseParams.nNumOfQInOneGroup==64&&t.baseParams.seqSize==256&&
  t.baseParams.sparseBlockCount==128&&t.baseParams.returnSoftmaxLse==stats,"wrong actual tiling values");
 auto original=t;original.baseParams.returnSoftmaxLse=1;
 Probe::Require(sfa_fixed::ValidateTiling(original,Probe::lastCount),"actual prepared POD fails real fixed validator");
 Probe::Require(Probe::allocations[Probe::last.workspace]==442368*Probe::lastCount,"scratch allocation differs from actual launch count");
}
void Layout(){
 Inputs x;Prepared native(x.q,1./24.,false,true),fused(x.q,1./24.,true,false);
 Prepared bf16(x.q,1./24.,true,false,true);
 auto n=native.Outputs(),f=fused.Outputs();
 Probe::Require(std::get<0>(n).bytes->size()==262144,"native output is not BF16-sized");
 Probe::Require(std::get<0>(f).bytes->size()==524288,"fused output is not FP32-sized");
 Probe::Require(std::get<0>(bf16.Outputs()).bytes->size()==262144&&
  std::get<0>(bf16.Outputs()).type==at::kBFloat16,"BF16 fused output allocation or dtype mismatch");
 Probe::Require(std::get<1>(n).shape==std::vector<int64_t>({4,64,1})&&
  std::get<2>(n).shape==std::vector<int64_t>({4,64,1}),"native stats shape");
 Probe::Require(std::get<1>(f).shape==std::vector<int64_t>({0})&&
  std::get<2>(f).shape==std::vector<int64_t>({0}),"fused stats must be empty");
 Probe::Begin();Native(native,x);CheckTiling(true);
 Probe::Begin();Fused(fused,x);CheckTiling(false);
 Probe::Require(Probe::lastFused&&!Probe::lastBf16,"FP32 mode selected wrong entry");
 Inputs xb;xb.co=Make({4,64,512},at::kBFloat16);
 Probe::Begin();Fused(bf16,xb);CheckTiling(false);
 Probe::Require(Probe::lastFused&&Probe::lastBf16,"BF16 mode did not select BF16 kernel entry");
 Probe::Require(std::get<1>(bf16.Outputs()).shape==std::vector<int64_t>({0})&&
  std::get<2>(bf16.Outputs()).shape==std::vector<int64_t>({0}),"BF16 fused stats must be empty");
 Prepared noStats(x.q,1./24.,false,false);Probe::Begin();Native(noStats,x);CheckTiling(false);
 Probe::Require(std::get<1>(noStats.Outputs()).shape==std::vector<int64_t>({0}),"native no-stats result");
}
void Dynamic(bool bf16=false){
 Inputs first,second;
 if(bf16){first.co=Make({4,64,512},at::kBFloat16);second.co=Make({4,64,512},at::kBFloat16);}
 Prepared plan(first.q,1./24.,true,false,bf16);
 const int allocations=Probe::emptyCalls,copies=Probe::toCalls;
 Probe::Begin();Fused(plan,first);const auto saved=Probe::last;
 Probe::rawStream=reinterpret_cast<void*>(uintptr_t{456});Probe::Begin();Fused(plan,second);
 Probe::Require(Probe::last.query==second.q.data_ptr()&&Probe::last.coldOutput==second.co.data_ptr()&&
  Probe::last.coldMax==second.cm.data_ptr()&&Probe::last.coldSum==second.cs.data_ptr()&&
  Probe::last.coldPresent==second.cp.data_ptr()&&Probe::last.hotPresent==second.hp.data_ptr(),"stale input pointer");
 Probe::Require(Probe::last.coldOutput!=saved.coldOutput&&Probe::last.output==saved.output&&
  Probe::last.workspace==saved.workspace&&Probe::last.tiling==saved.tiling,"prepared/dynamic ownership mismatch");
 Probe::Require(Probe::emptyCalls==allocations&&Probe::toCalls==copies,"allocation or H2D copy in run path");
 Probe::Require(Probe::lastBf16==bf16,"dynamic call selected wrong cold/output dtype entry");
}
void Invalid(){
 Inputs good;Prepared native(good.q,1./24.,false,true),fused(good.q,1./24.,true,false);
 for(int kind=0;kind<6;++kind){Inputs x;switch(kind){
  case 0:x.q.type=at::kFloat;break;case 1:x.q.shape={4,32,512};break;
  case 2:x.q.dev=c10::Device(c10::DeviceType::CPU);break;case 3:x.q.grad=true;break;
  case 4:x.q.contiguous=false;break;case 5:x.q=at::Tensor{};break;}
  Reject([&]{Prepared bad(x.q,1./24.,false,true);});
 }
 Reject([&]{Prepared bad(good.q,1./24.,true,true);});
 Reject([&]{Prepared bad(good.q,1./24.,true,true,true);});
 Reject([&]{Prepared bad(good.q,1./24.,false,false,true);});
 for(double scale:{0.,-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::max(),std::numeric_limits<double>::denorm_min()})
  Reject([&]{Prepared bad(good.q,scale,false,true);});
 for(uint32_t old:{1u,2u}) {Probe::abi=old;Reject([&]{Prepared bad(good.q,1./24.,false,true);});}Probe::abi=3;
 for(int n:{-1,0,3,5,8}) Reject([&]{Prepared bad(good.q,1./24.,true,false,false,n);});
 Probe::descriptorBytes=1;Reject([&]{Prepared bad(good.q,1./24.,false,true);});Probe::descriptorBytes=sizeof(SfaDirectBuffers);
 for(int kind=0;kind<12;++kind){Inputs x;switch(kind){
  case 0:x.q.type=at::kFloat;break;case 1:x.idx.shape={4,1,2048};break;
  case 2:x.k.dev=c10::Device(c10::DeviceType::PrivateUse1,4);break;case 3:x.q.grad=true;break;
  case 4:x.q.contiguous=false;break;case 5:x.v=Make({2,128,1,512},at::kBFloat16);break;
  case 6:x.co.type=at::kBFloat16;break;case 7:x.cm.shape={4,64};break;
  case 8:x.cp.type=at::kByte;break;case 9:x.hp.dev=c10::Device(c10::DeviceType::CPU);break;
  case 10:x.co=std::get<0>(fused.Outputs());break;
  case 11:x.co=std::get<0>(fused.Outputs());x.co.bytes->resize(x.co.bytes->size()+16);x.co.byteOffset=16;break;}
  Reject([&]{Fused(fused,x);});
 }
 Reject([&]{Native(fused,good);});Reject([&]{Fused(native,good);});
 Prepared bf16(good.q,1./24.,true,false,true);
 Reject([&]{Fused(bf16,good);}); // FP32 cold cannot silently enter BF16 mode.
 Inputs xb;xb.co=Make({4,64,512},at::kBFloat16);
 Reject([&]{Fused(fused,xb);}); // Nor can BF16 enter the legacy FP32 mode.
 xb.co=std::get<0>(bf16.Outputs());Reject([&]{Fused(bf16,xb);});
 xb.co=Make({4,64,512},at::kBFloat16);xb.cm.type=at::kBFloat16;
 Reject([&]{Fused(bf16,xb);}); // Statistics remain FP32 in both modes.
 Probe::Require(Probe::launches==0,"rejected matrix must never launch");
}
void Parallel(){
 Inputs x,xb;xb.co=Make({4,64,512},at::kBFloat16);
 auto library=LibraryConfiguration();Probe::Require(library["abi_version"]==3&&library["merge_tile_rows"]==8,"library configuration was guessed");
 for(uint32_t n:{1u,2u,4u}) {
  Prepared native(x.q,1./24.,false,true,false,n),fused(x.q,1./24.,true,false,false,n),bf16(x.q,1./24.,true,false,true,n);
  for(auto* plan:{&native,&fused,&bf16}) {
   auto config=plan->Configuration();
   Probe::Require(config["aic_count"]==n&&config["aiv_count"]==2*n&&config["workspace_bytes"]==442368*n,"prepared parallel configuration mismatch");
  }
  Probe::Begin();Native(native,x);CheckTiling(true);Probe::Require(Probe::lastCount==n,"native count not forwarded");
  Probe::Begin();Fused(fused,x);CheckTiling(false);Probe::Require(Probe::lastCount==n,"FP32 count not forwarded");
  Probe::Begin();Fused(bf16,xb);CheckTiling(false);Probe::Require(Probe::lastCount==n,"BF16 count not forwarded");
 }
}
int main(int argc,char** argv){try{
 Probe::Require(argc==2,"choose host test case");const std::string mode=argv[1];
 if(mode=="layout")Layout();else if(mode=="dynamic")Dynamic();else if(mode=="dynamic_bf16")Dynamic(true);
 else if(mode=="reject")Invalid();else if(mode=="parallel")Parallel();else return 2;
 std::cout<<"HOST_REAL_BRIDGE_"<<mode<<"_PASS\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what();return 1;}}
'''


class RealTorchBridgeHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("Host C++ compiler unavailable")
        if len((TORCH + NPU).splitlines()) > 250:
            raise AssertionError("Keep host API fakes below the agreed 250-line budget")
        cls.temp = tempfile.TemporaryDirectory(prefix="sfa-real-torch-bridge-host-")
        cls.addClassCleanup(cls.temp.cleanup)
        directory = Path(cls.temp.name)
        (directory / "torch").mkdir()
        (directory / "torch/extension.h").write_text(TORCH)
        npu = directory / "torch_npu/csrc/core/npu"
        npu.mkdir(parents=True)
        (npu / "NPUStream.h").write_text(NPU)
        for name in ("NPUGuard.h", "NPUCachingAllocator.h"):
            (npu / name).write_text('#pragma once\n#include "NPUStream.h"\n')
        (directory / "driver.cpp").write_text(DRIVER.replace("ACTUAL_BRIDGE", str(BRIDGE)))
        cls.binary = directory / "real_bridge_host"
        result = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(directory),
                                 "-I", str(ROOT / "standalone"), str(directory / "driver.cpp"),
                                 "-o", str(cls.binary)], capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def run_case(self, mode):
        result = subprocess.run([str(self.binary), mode], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), "HOST_REAL_BRIDGE_" + mode + "_PASS")

    def test_real_prepared_uses_pinned_pod_output_bytes_and_stats_shapes(self):
        self.run_case("layout")

    def test_current_step_pointers_stream_drain_and_allocator_records(self):
        self.run_case("dynamic")

    def test_bf16_variant_uses_current_cold_pointers_without_cast_or_allocation(self):
        self.run_case("dynamic_bf16")

    def test_metadata_tensor_alias_and_abi_errors_reject_before_launch(self):
        self.run_case("reject")

    def test_parallel_count_matches_tiling_allocation_launch_and_real_configuration(self):
        self.run_case("parallel")


if __name__ == "__main__":
    unittest.main()
