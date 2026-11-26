// Included inside torch_bridge.cpp's anonymous namespace. No installed ABI.
class KvSplitPrepared {
public:
    KvSplitPrepared(const at::Tensor &query, int64_t keys, int64_t capacity,
                    int64_t splits, double scale, bool stats)
        : device_(query.device()), q_(query.size(0)), keys_(keys), capacity_(capacity),
          splits_(splits), stats_(stats)
    {
        TORCH_CHECK(device_.type()==c10::DeviceType::PrivateUse1,"Ascend NPU query required");
        TORCH_CHECK(q_>=1 && (splits==1 || splits==2 || splits==4) && q_<=16/splits,
                    "KV-split requires S1/2/4 and Q*S<=16");
        Check(query,device_,{q_,64,576},at::kBFloat16,"query");
        TORCH_CHECK(keys>=2048 && keys<=INT32_MAX && keys%128==0,"Page-aligned K>=2048 required");
        TORCH_CHECK(capacity>=128+q_-1 && capacity<=INT32_MAX && capacity%128==0,
                    "Ring must retain W+Q-1 and be page aligned");
        TORCH_CHECK(std::isfinite(scale) && scale>0 && std::isfinite(static_cast<float>(scale)) &&
                    static_cast<float>(scale)>0,"Finite positive FP32 scale required");
        TORCH_CHECK(QsfaSinglePassLaunchKvSplit != nullptr &&
                    QsfaSinglePassLaunchKvSplitReduce != nullptr &&
                    QsfaSinglePassKvSplitAbiVersion != nullptr &&
                    QsfaSinglePassKvSplitTilingBytes != nullptr,
                    "Library lacks real KV-split exports; rebuild the isolated candidate");
        TORCH_CHECK(QsfaSinglePassAbiVersion()==1 && QsfaSinglePassBuffersBytes()==sizeof(SinglePassBuffers) &&
                    QsfaSinglePassKvSplitAbiVersion()==2 &&
                    QsfaSinglePassKvSplitTilingBytes()==sizeof(recent128_single_pass::SplitTiling),
                    "KV-split ABI2/112B mismatch; never pass ABI1 tiling to this entry");
        c10_npu::NPUGuard guard(device_);
        const auto value=recent128_single_pass::MakeSplitTiling(
            q_,keys_,splits_,static_cast<float>(scale),stats_);
        aic_=value.singleCoreParams.usedCoreNum;
        output_=at::empty({q_,64,512},query.options());
        maximum_=at::empty({q_,64,1},query.options().dtype(at::kFloat));
        sum_=at::empty_like(maximum_);
        empty_=at::empty({0},query.options().dtype(at::kFloat));
        // All partial slices share one FP32 allocation, whose raw byte pointer
        // matches the device descriptor. No dtype reinterpretation op needed.
        workspace_=at::empty({static_cast<int64_t>(value.splitParams.workspaceBytes/sizeof(float))},
                             query.options().dtype(at::kFloat));
        const auto &p=value.splitParams;
        partialOutput_=workspace_.narrow(0,p.partialOutputOffsetBytes/4,q_*splits_*64*512)
            .view({q_,splits_,64,512});
        partialMaximum_=workspace_.narrow(0,p.partialMaximumOffsetBytes/4,q_*splits_*64)
            .view({q_,splits_,64});
        partialSum_=workspace_.narrow(0,p.partialSumOffsetBytes/4,q_*splits_*64)
            .view({q_,splits_,64});
        auto host=at::empty({static_cast<int64_t>(sizeof(value))},
                            at::TensorOptions().device(at::kCPU).dtype(at::kByte));
        std::memcpy(host.data_ptr(),&value,sizeof(value));
        tiling_=host.to(device_,at::kByte,false,true);
    }

    State Run(const at::Tensor &query,const at::Tensor &cold,const at::Tensor &indices,
              const at::Tensor &table,const at::Tensor &actualQuery,const at::Tensor &actualKv,
              const at::Tensor &hot,const at::Tensor &rope,const at::Tensor &tags,
              const at::Tensor &positions,int64_t window=128)
    {
        Check(query,device_,{q_,64,576},at::kBFloat16,"query");
        Check(cold,device_,{keys_/128,128,1,400},at::kChar,"cold");
        Check(indices,device_,{q_,1,2048},at::kInt,"indices");
        Check(table,device_,{1,keys_/128},at::kInt,"block_table");
        Check(actualQuery,device_,{1},at::kInt,"actual_query");
        Check(actualKv,device_,{1},at::kInt,"actual_kv");
        Check(hot,device_,{capacity_/128,128,1,512},at::kBFloat16,"hot_latent");
        Check(rope,device_,{capacity_/128,128,1,64},at::kBFloat16,"hot_rope");
        Check(tags,device_,{capacity_},at::kLong,"ring_tags");
        Check(positions,device_,{q_},at::kLong,"query_positions");
        TORCH_CHECK(window==0 || window==128,"KV-split supports W0/W128");
        const std::initializer_list<at::Tensor> inputs={query,cold,indices,table,actualQuery,
            actualKv,hot,rope,tags,positions};
        for (const auto &written:{output_,maximum_,sum_,workspace_,tiling_}) {
            const auto begin=reinterpret_cast<uintptr_t>(written.data_ptr());
            const auto end=begin+written.numel()*written.element_size();
            for (const auto &input:inputs) {
                const auto ib=reinterpret_cast<uintptr_t>(input.data_ptr());
                const auto ie=ib+input.numel()*input.element_size();
                TORCH_CHECK(ie<=begin || end<=ib,"Input overlaps split output/workspace/tiling");
            }
        }
        SinglePassBuffers b{};
        b.query=Ptr(query); b.cold=Ptr(cold); b.indices=Ptr(indices); b.blockTable=Ptr(table);
        b.actualQuery=Ptr(actualQuery); b.actualKv=Ptr(actualKv);
        b.hotLatent=Ptr(hot); b.hotRope=Ptr(rope); b.ringTags=Ptr(tags); b.queryPositions=Ptr(positions);
        b.output=Ptr(output_); b.maximum=Ptr(maximum_); b.sum=Ptr(sum_);
        b.workspace=Ptr(workspace_); b.tiling=Ptr(tiling_);
        c10_npu::NPUGuard guard(device_);
        const auto stream=c10_npu::getCurrentNPUStream(device_.index());
        void *raw=stream.stream();
        for (const auto &input:inputs)
            c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(),stream);
        for (const auto &value:{output_,maximum_,sum_,workspace_,tiling_})
            c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(),stream);
        const auto code=QsfaSinglePassLaunchKvSplit(raw,&b,aic_,capacity_,window,q_,splits_,stats_?1:0);
        TORCH_CHECK(code==0,"KV-split C entry rejected descriptor");
        return Outputs();
    }

    State Outputs() const { return {output_,stats_?maximum_:empty_,stats_?sum_:empty_}; }
    State Partials() const { return {partialOutput_,partialMaximum_,partialSum_}; }

private:
    c10::Device device_;
    int64_t q_,keys_,capacity_,splits_;
    uint32_t aic_;
    bool stats_;
    at::Tensor output_,maximum_,sum_,workspace_,tiling_,empty_;
    at::Tensor partialOutput_,partialMaximum_,partialSum_;
};
