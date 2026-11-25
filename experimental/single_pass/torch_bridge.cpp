// Thin Torch host bridge. Only the separate ASC shared library launches a
// kernel: there is no Python/ACLNN C4 reader, second attention, or LSE merge.
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <tuple>
#include <type_traits>
#include "single_pass_api.h"
#include "single_pass_tiling.h"

// The additive routed export is optional for older measured ABI1 libraries.
// Linux weak references keep framework/default-overlay runs compatible; only
// an explicit routed run requires the rebuilt three-entry library.
#if defined(__ELF__)
#pragma weak QsfaSinglePassLaunchRouted
#pragma weak QsfaSinglePassLaunchFastPair
#pragma weak QsfaSinglePassLaunchEarlyHot
#endif

namespace {
using State = std::tuple<at::Tensor,at::Tensor,at::Tensor>;
static_assert(std::is_standard_layout<SinglePassBuffers>::value, "Shared C ABI required");
static_assert(sizeof(SinglePassBuffers)==15*sizeof(uint8_t*), "Fifteen-pointer ABI required");

uint8_t* Ptr(const at::Tensor& tensor) { return reinterpret_cast<uint8_t*>(tensor.data_ptr()); }

void Check(const at::Tensor& tensor,c10::Device device,std::initializer_list<int64_t> shape,
           at::ScalarType type,const char* name) {
    TORCH_CHECK(tensor.defined() && tensor.device()==device,name," device mismatch");
    TORCH_CHECK(tensor.scalar_type()==type && tensor.is_contiguous(),name," dtype/contiguity mismatch");
    TORCH_CHECK(!tensor.requires_grad(),name," is inference-only");
    TORCH_CHECK(tensor.sizes().equals(at::IntArrayRef(shape)),name," shape mismatch");
}

pybind11::dict Configuration() {
    pybind11::dict out;
    out["abi_version"]=QsfaSinglePassAbiVersion();
    out["buffers_bytes"]=QsfaSinglePassBuffersBytes();
    out["attention_launches_per_run"]=1;
    out["external_partition_launches"]=0;
    out["external_merge_launches"]=0;
    out["overlay_reader_entry"]="QsfaSinglePassLaunch";
    out["routed_reader_entry"]="QsfaSinglePassLaunchRouted";
    out["routed_reader_available"]=QsfaSinglePassLaunchRouted != nullptr;
    out["fast_pair_reader_entry"]="QsfaSinglePassLaunchFastPair";
    out["fast_pair_reader_available"]=QsfaSinglePassLaunchFastPair != nullptr;
    out["early_hot_reader_entry"]="QsfaSinglePassLaunchEarlyHot";
    out["early_hot_reader_available"]=QsfaSinglePassLaunchEarlyHot != nullptr;
    out["default_reader"]="overlay";
    return out;
}

class Prepared {
public:
    Prepared(const at::Tensor& query,int64_t keys,int64_t capacity,int64_t aic,double scale,bool stats)
        : device_(query.device()),q_(query.size(0)),keys_(keys),capacity_(capacity),aic_(aic),stats_(stats) {
        TORCH_CHECK(device_.type()==c10::DeviceType::PrivateUse1,"Ascend NPU tensor required");
        TORCH_CHECK(q_>=1 && q_<=1024,"Require Q1..1024");
        Check(query,device_,{q_,64,576},at::kBFloat16,"query");
        TORCH_CHECK(keys>=2048 && keys<=INT32_MAX && keys%128==0,"Page-aligned K>=2048 required");
        TORCH_CHECK(capacity>=128+q_-1 && capacity<=INT32_MAX && capacity%128==0,
                    "Ring must be page-aligned and retain W+Q-1 tokens");
        TORCH_CHECK((aic==1 || aic==2 || aic==4 || aic==8 || aic==16) && aic<=q_ && q_%aic==0,
                    "AIC count must be 1/2/4/8/16 and divide Q");
        TORCH_CHECK(std::isfinite(scale) && scale>0 && std::isfinite(static_cast<float>(scale)) &&
                    static_cast<float>(scale)>0,"Finite positive FP32 scale required");
        TORCH_CHECK(QsfaSinglePassAbiVersion()==1 && QsfaSinglePassBuffersBytes()==sizeof(SinglePassBuffers),
                    "Single-pass library ABI mismatch; rebuild matching sources");
        c10_npu::NPUGuard guard(device_);
        output_=at::empty({q_,64,512},query.options());
        maximum_=at::empty({q_,64,1},query.options().dtype(at::kFloat));
        sum_=at::empty_like(maximum_);
        empty_=at::empty({0},query.options().dtype(at::kFloat));
        workspace_=at::empty({1},query.options().dtype(at::kByte));
        const auto value=recent128_single_pass::MakeTiling(
            q_,keys_,aic_,static_cast<float>(scale),stats_,2048);
        auto host=at::empty({static_cast<int64_t>(sizeof(value))},at::TensorOptions().device(at::kCPU).dtype(at::kByte));
        std::memcpy(host.data_ptr(),&value,sizeof(value));
        tiling_=host.to(device_,at::kByte,false,true);
    }

    State Run(const at::Tensor& query,const at::Tensor& cold,const at::Tensor& indices,
              const at::Tensor& table,const at::Tensor& actualQuery,const at::Tensor& actualKv,
              const at::Tensor& hotLatent,const at::Tensor& hotRope,const at::Tensor& tags,
              const at::Tensor& positions,int64_t window=128,bool coldOnly=false,bool routed=false,
              bool fastPair=false,bool earlyHot=false) {
        Check(query,device_,{q_,64,576},at::kBFloat16,"query");
        Check(cold,device_,{keys_/128,128,1,400},at::kChar,"cold");
        Check(indices,device_,{q_,1,2048},at::kInt,"indices");
        Check(table,device_,{1,keys_/128},at::kInt,"block_table");
        Check(actualQuery,device_,{1},at::kInt,"actual_query");
        Check(actualKv,device_,{1},at::kInt,"actual_kv");
        Check(hotLatent,device_,{capacity_/128,128,1,512},at::kBFloat16,"hot_latent");
        Check(hotRope,device_,{capacity_/128,128,1,64},at::kBFloat16,"hot_rope");
        Check(tags,device_,{capacity_},at::kLong,"ring_tags");
        Check(positions,device_,{q_},at::kLong,"query_positions");
        TORCH_CHECK(window==0 || window==128,"Prototype supports W0/W128 only");
        TORCH_CHECK(!coldOnly || window==0,"Original C4 control requires window=0");
        TORCH_CHECK(static_cast<int>(coldOnly)+static_cast<int>(routed)+static_cast<int>(fastPair)+
                    static_cast<int>(earlyHot)<=1,
                    "Choose one of original C4, routed, fast-pair, or early-hot entries");
        TORCH_CHECK(!routed || QsfaSinglePassLaunchRouted != nullptr,
                    "Routed reader requires the rebuilt three-entry shared library");
        TORCH_CHECK(!fastPair || QsfaSinglePassLaunchFastPair != nullptr,
                    "Fast-pair reader requires the rebuilt four-entry shared library");
        TORCH_CHECK(!earlyHot || QsfaSinglePassLaunchEarlyHot != nullptr,
                    "Early-hot reader requires the rebuilt five-entry shared library");
        const std::initializer_list<at::Tensor> inputs={query,cold,indices,table,actualQuery,actualKv,
            hotLatent,hotRope,tags,positions};
        for (const auto& written:{output_,maximum_,sum_,workspace_,tiling_}) {
            const auto begin=reinterpret_cast<uintptr_t>(written.data_ptr());
            const auto end=begin+written.numel()*written.element_size();
            for (const auto& input:inputs) {
                const auto ib=reinterpret_cast<uintptr_t>(input.data_ptr());
                const auto ie=ib+input.numel()*input.element_size();
                TORCH_CHECK(ie<=begin || end<=ib,"Input overlaps Prepared output/scratch/tiling");
            }
        }
        SinglePassBuffers buffers{};
        buffers.query=Ptr(query); buffers.cold=Ptr(cold); buffers.indices=Ptr(indices);
        buffers.blockTable=Ptr(table); buffers.actualQuery=Ptr(actualQuery); buffers.actualKv=Ptr(actualKv);
        buffers.hotLatent=Ptr(hotLatent); buffers.hotRope=Ptr(hotRope); buffers.ringTags=Ptr(tags);
        buffers.queryPositions=Ptr(positions); buffers.output=Ptr(output_);
        buffers.maximum=Ptr(maximum_); buffers.sum=Ptr(sum_);
        buffers.workspace=Ptr(workspace_); buffers.tiling=Ptr(tiling_);
        c10_npu::NPUGuard guard(device_);
        const auto stream=c10_npu::getCurrentNPUStream(device_.index());
        void* raw=stream.stream(); // Drain host dispatch queue, never device-sync.
        for (const auto& input:inputs)
            c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(),stream);
        for (const auto& value:{output_,maximum_,sum_,workspace_,tiling_})
            c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(),stream);
        const int code=coldOnly ? QsfaSinglePassLaunchCold(raw,&buffers,aic_,capacity_,window)
            : earlyHot ? QsfaSinglePassLaunchEarlyHot(raw,&buffers,aic_,capacity_,window)
            : fastPair ? QsfaSinglePassLaunchFastPair(raw,&buffers,aic_,capacity_,window)
            : routed ? QsfaSinglePassLaunchRouted(raw,&buffers,aic_,capacity_,window)
                     : QsfaSinglePassLaunch(raw,&buffers,aic_,capacity_,window);
        TORCH_CHECK(code==0,"Single-pass C entry rejected launch descriptor");
        return Outputs();
    }

    State Outputs() const { return {output_,stats_?maximum_:empty_,stats_?sum_:empty_}; }

private:
    c10::Device device_;
    int64_t q_,keys_,capacity_;
    uint32_t aic_;
    bool stats_;
    at::Tensor output_,maximum_,sum_,workspace_,tiling_,empty_;
};
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
    module.def("configuration",&Configuration);
    pybind11::class_<Prepared,std::shared_ptr<Prepared>>(module,"Prepared")
        .def("run",&Prepared::Run,pybind11::arg("query"),pybind11::arg("cold"),
            pybind11::arg("indices"),pybind11::arg("block_table"),pybind11::arg("actual_query"),
            pybind11::arg("actual_kv"),pybind11::arg("hot_latent"),pybind11::arg("hot_rope"),
            pybind11::arg("ring_tags"),pybind11::arg("query_positions"),
            pybind11::arg("window")=128,pybind11::arg("cold_only")=false,
            pybind11::arg("routed")=false,pybind11::arg("fast_pair")=false,
            pybind11::arg("early_hot")=false)
        .def("outputs",&Prepared::Outputs);
    module.def("prepare",[](const at::Tensor& query,int64_t keys,int64_t capacity,
                            int64_t aic,double scale,bool stats) {
        TORCH_CHECK(query.defined() && query.dim()==3,"Query must be defined rank3");
        return std::make_shared<Prepared>(query,keys,capacity,aic,scale,stats);
    },pybind11::arg("query"),pybind11::arg("keys"),pybind11::arg("ring_capacity"),
      pybind11::arg("aic_count"),pybind11::arg("scale"),pybind11::arg("return_stats")=false);
}
