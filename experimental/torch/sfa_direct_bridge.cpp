// Direct bridge to the compiled standalone kernels, not an unimplemented ACLNN.
// Prepared objects own output/tiling/scratch. Every call uses the CURRENT input
// tensor pointers, including newly allocated C4 output/max/sum from that step.
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
#include "sfa_direct_api.h"
#include "sfa_fixed_contract.h"

namespace {
using State = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
static_assert(std::is_standard_layout<SfaDirectBuffers>::value,
              "Direct launch arguments must have one shared C layout");
static_assert(sizeof(SfaDirectBuffers) == 19 * sizeof(uint8_t *),
              "Do not silently change the nineteen-pointer direct ABI");

void CheckTensor(const at::Tensor &tensor, c10::Device device,
                 std::initializer_list<int64_t> shape, at::ScalarType dtype,
                 const char *name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device() == device, name, " device mismatch");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(!tensor.requires_grad(), name, " is inference-only");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " dtype mismatch");
    TORCH_CHECK(tensor.sizes().equals(at::IntArrayRef(shape)), name, " fixed-shape mismatch");
}

uint8_t *Address(const at::Tensor &tensor)
{
    return reinterpret_cast<uint8_t *>(tensor.data_ptr());
}

class Prepared {
public:
    Prepared(const at::Tensor &query, double scale, bool fused, bool returnStats)
        : device_(query.device()), fused_(fused), returnStats_(returnStats)
    {
        TORCH_CHECK(device_.type() == c10::DeviceType::PrivateUse1,
                    "An Ascend NPU query tensor is required");
        CheckTensor(query, device_, {4,64,512}, at::kBFloat16, "query");
        TORCH_CHECK(std::isfinite(scale) && scale > 0 &&
                    std::isfinite(static_cast<float>(scale)) && static_cast<float>(scale) > 0,
                    "Scale must remain finite and positive in FP32");
        TORCH_CHECK(!fused || !returnStats,
                    "The fused direct entry returns final output only; no combined stats ABI");
        TORCH_CHECK(SfaDirectAbiVersion() == 1 && SfaDirectBuffersBytes() == sizeof(SfaDirectBuffers),
                    "Compiled direct-library ABI does not match this bridge; rebuild the exact sources");
        c10_npu::NPUGuard guard(device_);
        output_ = at::empty({4,64,512}, query.options().dtype(fused ? at::kFloat : at::kBFloat16));
        // Keep valid full-size pointers even when the entry is not returning stats.
        maximum_ = at::empty({4,64,1}, query.options().dtype(at::kFloat));
        sum_ = at::empty_like(maximum_);
        empty_ = at::empty({0}, query.options().dtype(at::kFloat));
        scratch_ = at::empty({static_cast<int64_t>(sfa_fixed::userWorkspaceBytes)},
                              query.options().dtype(at::kByte));
        auto tiling = sfa_fixed::MakeTiling(static_cast<float>(scale));
        TORCH_CHECK(sfa_fixed::ValidateTiling(tiling), "Fixed original POD validation failed");
        tiling.baseParams.returnSoftmaxLse = returnStats ? 1 : 0;
        auto host = at::empty({static_cast<int64_t>(sizeof(tiling))},
                               at::TensorOptions().device(at::kCPU).dtype(at::kByte));
        std::memcpy(host.data_ptr(), &tiling, sizeof(tiling));
        // Construction happens before capture/timing. There is no H2D tiling
        // copy, CPU value read, output allocation or device sync in run methods.
        tiling_ = host.to(device_, at::kByte, false, true);
    }

    State NativeHot(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                    const at::Tensor &indices, const at::Tensor &table,
                    const at::Tensor &queryLengths, const at::Tensor &kvLengths,
                    const at::Tensor &queryRope, const at::Tensor &keyRope)
    {
        TORCH_CHECK(!fused_, "This Prepared object is for fused_hot");
        auto arguments = HotArguments(query,key,value,indices,table,queryLengths,kvLengths,queryRope,keyRope);
        CheckWritableAliasing({query,key,value,indices,table,queryLengths,kvLengths,queryRope,keyRope});
        c10_npu::NPUGuard guard(device_);
        const auto stream = c10_npu::getCurrentNPUStream(device_.index());
        // stream() defaults to draining the NPU HOST dispatch queue, so the
        // preceding Torch/ACLNN work is submitted before this raw ASC launch.
        // This is not a device synchronization. Do not replace with stream(false).
        void *raw = stream.stream();
        Record(stream,{query,key,value,indices,table,queryLengths,kvLengths,queryRope,keyRope});
        SfaLaunchOriginal(raw, &arguments);
        return Outputs();
    }

    State FusedHot(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                   const at::Tensor &indices, const at::Tensor &table,
                   const at::Tensor &queryLengths, const at::Tensor &kvLengths,
                   const at::Tensor &queryRope, const at::Tensor &keyRope,
                   const at::Tensor &coldOutput, const at::Tensor &coldMax,
                   const at::Tensor &coldSum, const at::Tensor &coldPresent,
                   const at::Tensor &hotPresent)
    {
        TORCH_CHECK(fused_, "This Prepared object is for native_hot");
        auto arguments = HotArguments(query,key,value,indices,table,queryLengths,kvLengths,queryRope,keyRope);
        // Never hide an old BF16-output binding by converting the cold result.
        CheckTensor(coldOutput,device_,{4,64,512},at::kFloat,"cold_output");
        CheckTensor(coldMax,device_,{4,64,1},at::kFloat,"cold_max");
        CheckTensor(coldSum,device_,{4,64,1},at::kFloat,"cold_sum");
        CheckTensor(coldPresent,device_,{4},at::kBool,"cold_present");
        CheckTensor(hotPresent,device_,{4},at::kBool,"hot_present");
        arguments.coldOutput=Address(coldOutput);
        arguments.coldMax=Address(coldMax);
        arguments.coldSum=Address(coldSum);
        arguments.coldPresent=Address(coldPresent);
        arguments.hotPresent=Address(hotPresent);
        CheckWritableAliasing({query,key,value,indices,table,queryLengths,kvLengths,queryRope,keyRope,
                               coldOutput,coldMax,coldSum,coldPresent,hotPresent});
        c10_npu::NPUGuard guard(device_);
        const auto stream = c10_npu::getCurrentNPUStream(device_.index());
        void *raw = stream.stream();
        Record(stream,{query,key,value,indices,table,queryLengths,kvLengths,queryRope,keyRope,
                       coldOutput,coldMax,coldSum,coldPresent,hotPresent});
        SfaLaunchFused(raw, &arguments);
        return Outputs();
    }

    State Outputs() const
    {
        return {output_, returnStats_ ? maximum_ : empty_, returnStats_ ? sum_ : empty_};
    }

private:
    SfaDirectBuffers HotArguments(const at::Tensor &query, const at::Tensor &key,
                                 const at::Tensor &value, const at::Tensor &indices,
                                 const at::Tensor &table, const at::Tensor &queryLengths,
                                 const at::Tensor &kvLengths, const at::Tensor &queryRope,
                                 const at::Tensor &keyRope) const
    {
        CheckTensor(query,device_,{4,64,512},at::kBFloat16,"query");
        CheckTensor(key,device_,{2,128,1,512},at::kBFloat16,"key");
        CheckTensor(value,device_,{2,128,1,512},at::kBFloat16,"value");
        TORCH_CHECK(key.data_ptr() == value.data_ptr(), "The fixed MLA value must alias the latent key cache");
        CheckTensor(indices,device_,{4,1,128},at::kInt,"compact_indices");
        CheckTensor(table,device_,{1,2},at::kInt,"identity_block_table");
        CheckTensor(queryLengths,device_,{1},at::kInt,"query_lengths");
        CheckTensor(kvLengths,device_,{1},at::kInt,"kv_lengths");
        CheckTensor(queryRope,device_,{4,64,64},at::kBFloat16,"query_rope");
        CheckTensor(keyRope,device_,{2,128,1,64},at::kBFloat16,"key_rope");
        // Before timing, the harness must validate the actual device VALUES:
        // lengths=4/256, identity table, unique in-range compact128 for each
        // query and valid nonempty hot work. This bridge never reads them on CPU.
        SfaDirectBuffers result{};
        result.query=Address(query); result.key=Address(key); result.value=Address(value);
        result.indices=Address(indices); result.blockTable=Address(table);
        result.actualQuery=Address(queryLengths); result.actualKv=Address(kvLengths);
        result.queryRope=Address(queryRope); result.keyRope=Address(keyRope);
        result.output=Address(output_); result.maximum=Address(maximum_); result.sum=Address(sum_);
        result.workspace=Address(scratch_); result.tiling=Address(tiling_);
        return result;
    }

    void Record(const c10_npu::NPUStream &stream, std::initializer_list<at::Tensor> inputs) const
    {
        for (const auto &tensor : inputs) {
            c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
        }
        for (const auto &tensor : {output_,maximum_,sum_,scratch_,tiling_}) {
            c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
        }
    }

    void CheckWritableAliasing(std::initializer_list<at::Tensor> inputs) const
    {
        // Key/value may intentionally alias each other; no input may alias an
        // output or scratch that this launch will overwrite. No CPU value read.
        for (const auto &written : {output_,maximum_,sum_,scratch_}) {
            const auto begin = reinterpret_cast<uintptr_t>(written.data_ptr());
            const auto end = begin + written.numel() * written.element_size();
            for (const auto &input : inputs) {
                const auto inputBegin = reinterpret_cast<uintptr_t>(input.data_ptr());
                const auto inputEnd = inputBegin + input.numel() * input.element_size();
                TORCH_CHECK(inputEnd <= begin || end <= inputBegin,
                            "An input overlaps the Prepared output/statistics/workspace");
            }
        }
    }

    c10::Device device_;
    bool fused_, returnStats_;
    at::Tensor output_, maximum_, sum_, empty_, scratch_, tiling_;
};
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    pybind11::class_<Prepared,std::shared_ptr<Prepared>>(module,"Prepared")
        .def("native_hot",&Prepared::NativeHot)
        .def("fused_hot",&Prepared::FusedHot)
        .def("outputs",&Prepared::Outputs);
    module.def("prepare",[](const at::Tensor &query,double scale,bool fused,bool returnStats) {
        TORCH_CHECK(query.defined(), "Query must be defined");
        return std::make_shared<Prepared>(query,scale,fused,returnStats);
    },pybind11::arg("query"),pybind11::arg("scale"),pybind11::arg("fused"),
      pybind11::arg("return_stats")=false);
}
