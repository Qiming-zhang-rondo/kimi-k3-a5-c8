// Host-only bridge for additive serving ABI. All data-dependent routing stays
// on device; preparation owns every output/scratch allocation before capture.
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include "batched_serving_api.h"

namespace {
uint8_t *Ptr(const at::Tensor &v){return reinterpret_cast<uint8_t*>(v.data_ptr());}
void Check(const at::Tensor &v,c10::Device device,std::initializer_list<int64_t> shape,
           at::ScalarType type,const char *name)
{
    TORCH_CHECK(v.defined() && v.device()==device && v.scalar_type()==type &&
        v.is_contiguous() && !v.requires_grad(),name," device/dtype/contiguity mismatch");
    TORCH_CHECK(v.sizes().equals(at::IntArrayRef(shape)),name," shape mismatch");
}
pybind11::dict Configuration()
{
    pybind11::dict out;
    out["abi_version"]=QsfaBatchedServingAbiVersion();
    out["buffers_bytes"]=QsfaBatchedServingBuffersBytes();
    out["kernel_entry"]="qsfa_recent128_batched_serving";
    out["geometry"]="load32_stride416_input2_output1";
    out["batched_tnd"]=true;
    out["epoch_checked"]=true;
    out["sticky_device_faults"]=true;
    out["fault_reset_automatic"]=false;
    out["kv_split"]=false;
    out["device_validation"]="PENDING";
    return out;
}
class Prepared {
public:
    Prepared(const at::Tensor &query,int64_t batches,int64_t keys,int64_t tableColumns,
             int64_t capacity,int64_t pools,double scale)
        : device_(query.device()),q_(query.size(0)),b_(batches),keys_(keys),
          columns_(tableColumns),capacity_(capacity),pools_(pools)
    {
        TORCH_CHECK(device_.type()==c10::DeviceType::PrivateUse1,"NPU query required");
        TORCH_CHECK(q_>0 && q_<=INT32_MAX && b_>0 && b_<=INT32_MAX,"Invalid T/B geometry");
        Check(query,device_,{q_,64,576},at::kBFloat16,"query");
        TORCH_CHECK(keys_>=128 && keys_<=INT32_MAX && keys_%128==0 &&
                    columns_>0 && columns_<=keys_/128,"Invalid cold pool/table width");
        TORCH_CHECK(capacity_>=128 && capacity_<=INT32_MAX && capacity_%128==0 &&
                    pools_>0 && pools_<=INT32_MAX,"Invalid hot geometry");
        TORCH_CHECK(QsfaBatchedServingAbiVersion()==1 &&
            QsfaBatchedServingBuffersBytes()==sizeof(BatchedServingBuffers),"Serving ABI1/160B mismatch");
        aic_=static_cast<uint32_t>(std::min<int64_t>(16,q_));
        const auto value=recent128_single_pass::MakeBatchedServingTiling(
            q_,b_,keys_,columns_,aic_,static_cast<float>(scale));
        c10_npu::NPUGuard guard(device_);
        output_=at::empty({q_,64,512},query.options());
        faults_=at::zeros({q_},query.options().dtype(at::kInt));
        workspace_=at::empty({1},query.options().dtype(at::kByte));
        auto host=at::empty({static_cast<int64_t>(sizeof(value))},
            at::TensorOptions().device(at::kCPU).dtype(at::kByte));
        std::memcpy(host.data_ptr(),&value,sizeof(value));
        tiling_=host.to(device_,at::kByte,false,true);
    }
    at::Tensor Run(const at::Tensor &query,const at::Tensor &cold,const at::Tensor &indices,
        const at::Tensor &table,const at::Tensor &qcum,const at::Tensor &kvlen,
        const at::Tensor &hot,const at::Tensor &rope,const at::Tensor &tags,
        const at::Tensor &positions,const at::Tensor &banks,const at::Tensor &epochs,
        const at::Tensor &epochTags,const at::Tensor &valid)
    {
        Check(query,device_,{q_,64,576},at::kBFloat16,"query");
        Check(cold,device_,{keys_/128,128,1,400},at::kChar,"cold");
        Check(indices,device_,{q_,1,2048},at::kInt,"indices");
        Check(table,device_,{b_,columns_},at::kInt,"table");
        Check(qcum,device_,{b_},at::kInt,"qcum");
        Check(kvlen,device_,{b_},at::kInt,"kvlen");
        Check(hot,device_,{pools_,capacity_,512},at::kBFloat16,"hot");
        Check(rope,device_,{pools_,capacity_,64},at::kBFloat16,"rope");
        Check(tags,device_,{pools_,capacity_},at::kLong,"tags");
        Check(epochTags,device_,{pools_,capacity_},at::kLong,"epoch_tags");
        Check(positions,device_,{q_},at::kLong,"positions");
        Check(banks,device_,{b_},at::kInt,"request_banks");
        Check(epochs,device_,{b_},at::kLong,"request_epochs");
        Check(valid,device_,{q_},at::kBool,"query_valid");
        const std::initializer_list<at::Tensor> inputs={query,cold,indices,table,qcum,kvlen,
            hot,rope,tags,positions,banks,epochs,epochTags,valid};
        for(const auto &written:{output_,faults_,workspace_,tiling_}){
            const auto begin=reinterpret_cast<uintptr_t>(written.data_ptr());
            const auto end=begin+written.numel()*written.element_size();
            for(const auto &input:inputs){
                const auto ib=reinterpret_cast<uintptr_t>(input.data_ptr());
                const auto ie=ib+input.numel()*input.element_size();
                TORCH_CHECK(ie<=begin || end<=ib,"Serving input/output overlap");
            }
        }
        c10_npu::NPUGuard guard(device_);
        // Fixed-shape device operation. Padding/empty requests never expose a
        // previous replay's output even when native scheduling performs no work.
        output_.zero_();
        BatchedServingBuffers s{};
        auto &p=s.base;
        p.query=Ptr(query);p.cold=Ptr(cold);p.indices=Ptr(indices);p.blockTable=Ptr(table);
        p.actualQuery=Ptr(qcum);p.actualKv=Ptr(kvlen);p.hotLatent=Ptr(hot);p.hotRope=Ptr(rope);
        p.ringTags=Ptr(tags);p.queryPositions=Ptr(positions);p.output=Ptr(output_);
        p.workspace=Ptr(workspace_);p.tiling=Ptr(tiling_);
        s.requestBanks=Ptr(banks);s.requestEpochs=Ptr(epochs);s.ringEpochs=Ptr(epochTags);
        s.queryValid=Ptr(valid);s.faults=Ptr(faults_);
        const auto stream=c10_npu::getCurrentNPUStream(device_.index());
        void *raw=stream.stream();
        for(const auto &value:inputs)c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(),stream);
        for(const auto &value:{output_,faults_,workspace_,tiling_})
            c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(),stream);
        TORCH_CHECK(QsfaBatchedServingLaunch(raw,&s,aic_,capacity_,pools_,q_)==0,"Serving launch rejected");
        return output_;
    }
    at::Tensor Output()const{return output_;}
    at::Tensor Faults()const{return faults_;}
    void ResetFaults(){c10_npu::NPUGuard guard(device_);faults_.zero_();}
private:
    c10::Device device_;
    int64_t q_,b_,keys_,columns_,capacity_,pools_;
    uint32_t aic_;
    at::Tensor output_,faults_,workspace_,tiling_;
};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,module){
    module.def("configuration",&Configuration);
    pybind11::class_<Prepared,std::shared_ptr<Prepared>>(module,"BatchedServingPlan")
        .def("run",&Prepared::Run).def("output",&Prepared::Output)
        .def("faults",&Prepared::Faults).def("reset_faults",&Prepared::ResetFaults);
    module.def("prepare_batched",[](const at::Tensor &query,int64_t batches,int64_t keys,
        int64_t columns,int64_t capacity,int64_t pools,double scale){
        return std::make_shared<Prepared>(query,batches,keys,columns,capacity,pools,scale);
    });
}
