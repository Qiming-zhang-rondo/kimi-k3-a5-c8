#!/usr/bin/env python3
"""Validate the isolated ring writer; this is NOT a Prolog producer integration.

Build the independent ASC .so explicitly first. This runner only compiles a
thin Torch host bridge, loads the .so RTLD_LOCAL and uses the real NPU stream.
No vendor installation, package change, Prolog/C4 call or performance claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import traceback

HERE = Path(__file__).resolve().parent
FAULT_CODES = ("507015", "507035", "507053", "507036", "507011", "ACL_ERROR_RT_AICORE_EXCEPTION")
TENSOR_NAMES = ("inputLatent", "inputRope", "latentRing", "ropeRing", "positionTags", "epochTags",
                "tokenBanks", "tokenPositions", "tokenEpochs", "tokenValid")
BRIDGE_SOURCE = r'''
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <dlfcn.h>
#include <memory>
#include <type_traits>
#include "prolog_ring_smoke_api.h"
namespace {
static_assert(sizeof(PrologRingSmokeBuffers)==80,"Smoke ABI is ten pointers / 80 bytes");
static_assert(std::is_standard_layout<PrologRingSmokeBuffers>::value,"C layout required");
class Library {
  void *handle_=nullptr;
  using Query=uint32_t(*)();
  using Launch=int32_t(*)(void*,const PrologRingSmokeBuffers*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
  Launch launch_=nullptr;
public:
  explicit Library(const std::string& path) {
    handle_=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    TORCH_CHECK(handle_!=nullptr,"Smoke dlopen failed: ",dlerror());
    auto abi=reinterpret_cast<Query>(dlsym(handle_,"QsfaPrologRingSmokeAbiVersion"));
    auto bytes=reinterpret_cast<Query>(dlsym(handle_,"QsfaPrologRingSmokeBuffersBytes"));
    launch_=reinterpret_cast<Launch>(dlsym(handle_,"QsfaPrologRingSmokeLaunch"));
    TORCH_CHECK(abi && bytes && launch_,"Required smoke exports missing");
    TORCH_CHECK(abi()==1 && bytes()==80,"Smoke ABI/version mismatch");
  }
  ~Library(){ if(handle_) dlclose(handle_); }
  pybind11::dict Configuration() const {
    pybind11::dict result;result["abi_version"]=1;result["buffers_bytes"]=80;
    result["scope"]="isolated_ring_writer_not_producer";return result;
  }
  at::Tensor Run(const std::vector<at::Tensor>& x,int64_t blocks,bool enabled) {
    TORCH_CHECK(x.size()==10,"Ten tensors required in documented C ABI order");
    const auto device=x[0].device();
    TORCH_CHECK(device.type()==c10::DeviceType::PrivateUse1,"NPU tensors required");
    TORCH_CHECK(x[0].dim()==2 && x[0].size(1)==512,"FP32 latent [T,512] required");
    TORCH_CHECK(x[2].dim()==3 && x[2].size(2)==512,"BF16 ring [P,C,512] required");
    int64_t t=x[0].size(0),p=x[2].size(0),c=x[2].size(1);
    TORCH_CHECK(t>0 && t<=32 && p>0 && c>0 && c%128==0,"Smoke geometry invalid");
    TORCH_CHECK(blocks>0 && blocks<=t && blocks<=64,"Smoke block count invalid");
    const at::ScalarType types[]={at::kFloat,at::kBFloat16,at::kBFloat16,at::kBFloat16,
        at::kLong,at::kLong,at::kInt,at::kLong,at::kLong,at::kBool};
    const std::vector<std::vector<int64_t>> shapes={{t,512},{t,64},{p,c,512},{p,c,64},
        {p,c},{p,c},{t},{t},{t},{t}};
    for(size_t i=0;i<x.size();++i) {
      TORCH_CHECK(x[i].device()==device && x[i].is_contiguous() && !x[i].requires_grad(),"Smoke tensor device/contiguity/grad mismatch");
      TORCH_CHECK(x[i].scalar_type()==types[i] && x[i].sizes().equals(at::IntArrayRef(shapes[i])),"Smoke tensor shape/dtype mismatch at ",i);
    }
    c10_npu::NPUGuard guard(device);
    auto stream=c10_npu::getCurrentNPUStream(device.index());
    for(const auto& v:x)c10_npu::NPUCachingAllocator::recordStream(v.storage().data_ptr(),stream);
    PrologRingSmokeBuffers b{};
    b.inputLatent=reinterpret_cast<uint8_t*>(x[0].data_ptr());b.inputRope=reinterpret_cast<uint8_t*>(x[1].data_ptr());
    b.latentRing=reinterpret_cast<uint8_t*>(x[2].data_ptr());b.ropeRing=reinterpret_cast<uint8_t*>(x[3].data_ptr());
    b.positionTags=reinterpret_cast<uint8_t*>(x[4].data_ptr());b.epochTags=reinterpret_cast<uint8_t*>(x[5].data_ptr());
    b.tokenBanks=reinterpret_cast<uint8_t*>(x[6].data_ptr());b.tokenPositions=reinterpret_cast<uint8_t*>(x[7].data_ptr());
    b.tokenEpochs=reinterpret_cast<uint8_t*>(x[8].data_ptr());b.tokenValid=reinterpret_cast<uint8_t*>(x[9].data_ptr());
    TORCH_CHECK(launch_(stream.stream(),&b,t,p,c,blocks,enabled?1:0)==0,"Smoke launch rejected C descriptor");
    return x[2];
  }
};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){pybind11::class_<Library,std::shared_ptr<Library>>(m,"Library")
 .def(pybind11::init<const std::string&>()).def("run",&Library::Run).def("configuration",&Library::Configuration);}
'''


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def bf16_round(value):
    import numpy as np
    x=np.asarray(value,dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError("Finite smoke inputs required")
    bits=x.view(np.uint32)
    result=((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
    if not np.isfinite(result).all():raise ValueError("BF16 overflow")
    return result.copy()


def make_case(pools=8,capacity=256,phase=0):
    import numpy as np
    if pools not in (1,8) or capacity%128 or capacity<128:raise ValueError("Smoke case geometry invalid")
    per_request=1 if pools==1 else 4
    rows=per_request*pools
    # Same logical positions in every request expose bank/position confusion.
    banks=np.repeat(np.arange(pools,dtype=np.int32),per_request)
    positions=np.tile(np.array([256] if pools==1 else [254,255,256,257],np.int64)+phase*4,pools)
    epochs=(100+banks.astype(np.int64)+phase).astype(np.int64)
    valid=np.ones(rows,dtype=np.bool_)
    if pools==8:
        order=np.arange(rows).reshape(pools,per_request)[np.roll(np.arange(pools)[::-1],phase)].reshape(-1)
        banks,positions,epochs=(a[order].copy() for a in (banks,positions,epochs))
        valid[-1]=False
        banks[-1]=-1;positions[-1]=-1;epochs[-1]=-1
    raw=(np.arange(rows*512,dtype=np.float32).reshape(rows,512)%127-63)/16
    raw+=np.float32(phase/8)
    # Add exactly representable BF16 halfway values to verify RNE conversion.
    raw[:,0]=np.float32(1.00390625)
    raw[:,1]=np.float32(1.01171875)
    rope=bf16_round((np.arange(rows*64,dtype=np.float32).reshape(rows,64)%97-48)/32+phase)
    return dict(inputLatent=raw,inputRope=rope,tokenBanks=banks,tokenPositions=positions,
        tokenEpochs=epochs,tokenValid=valid)


def blank_rings(pools,capacity):
    import numpy as np
    return dict(latentRing=np.full((pools,capacity,512),-7.,np.float32),
        ropeRing=np.full((pools,capacity,64),-11.,np.float32),
        positionTags=np.full((pools,capacity),-1,np.int64),epochTags=np.full((pools,capacity),-1,np.int64))


def expected_rings(case,initial,enabled=True):
    import numpy as np
    out={name:value.copy() for name,value in initial.items()}
    pools,capacity=out["positionTags"].shape
    if not enabled:return out
    seen=set()
    for row,(bank,position,epoch,valid) in enumerate(zip(case["tokenBanks"],case["tokenPositions"],case["tokenEpochs"],case["tokenValid"])):
        if not valid or not 0<=bank<pools or position<0:continue
        slot=int(position%capacity)
        key=int(bank),slot
        if key in seen:raise ValueError("Duplicate ring writes in one kernel would race")
        seen.add(key)
        out["latentRing"][bank,slot]=bf16_round(case["inputLatent"][row])
        out["ropeRing"][bank,slot]=case["inputRope"][row]
        out["positionTags"][bank,slot]=position;out["epochTags"][bank,slot]=epoch
    return out


def _recorded_sdk(library):
    for parent in (library.parent,library.parent.parent):
        path=parent/"CMakeCache.txt"
        if path.is_file():
            for line in path.read_text(errors="replace").splitlines():
                if line.startswith("SFA_SDK_ROOT:PATH="):return Path(line.split("=",1)[1]).resolve()
    return None


def load_bridge(library,output,sdk_root=None,verbose=False):
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load
    library=Path(library).resolve()
    if library.name!="libqsfa_prolog_ring_smoke.so" or not library.is_file() or not library.stat().st_size:
        raise ValueError("Use explicitly built nonempty libqsfa_prolog_ring_smoke.so")
    recorded=_recorded_sdk(library)
    selected=sdk_root or os.getenv("ASCEND_HOME_PATH") or recorded
    if not selected:raise ValueError("Pass matched --sdk-root or ASCEND_HOME_PATH")
    sdk=Path(selected).resolve()
    if recorded and recorded!=sdk:raise ValueError("Selected SDK differs from library CMakeCache")
    includes=[sdk/"aarch64-linux/include",sdk/"include"]
    acl=next((path for path in includes if (path/"acl/acl.h").is_file()),None)
    if acl is None:raise ValueError("Matched ACL headers unavailable")
    npu=Path(torch_npu.__file__).resolve().parent
    if not (npu/"lib/libtorch_npu.so").is_file():raise ValueError("Active torch_npu library unavailable")
    header=HERE/"prolog_ring_smoke_api.h"
    if not header.is_file():raise ValueError("Smoke C ABI header missing")
    identity=dict(library=str(library),library_sha256=sha256(library),api_sha256=sha256(header),
        bridge_sha256=hashlib.sha256(BRIDGE_SOURCE.encode()).hexdigest(),sdk_root=str(sdk),
        torch=str(torch.__version__),torch_npu=str(getattr(torch_npu,"__version__","unknown")),
        torch_cxx11_abi=getattr(torch._C,"_GLIBCXX_USE_CXX11_ABI",None),asc_compiled_by_runner=False)
    build=Path(output)/"torch_bridge";build.mkdir()
    source=build/"torch_bridge.cpp";source.write_text(BRIDGE_SOURCE)
    links=["-ldl",shlex.quote("-L"+str(npu/"lib")),"-ltorch_npu","-Xlinker","-rpath","-Xlinker",shlex.quote(str(npu/"lib"))]
    module=load(name="qsfa_prolog_ring_smoke_"+identity["library_sha256"][:16],sources=[str(source)],
        extra_include_paths=[str(HERE),str(npu/"include"),str(acl)],extra_cflags=["-O2","-std=c++17"],
        extra_ldflags=links,build_directory=str(build),verbose=verbose)
    instance=module.Library(str(library))
    identity["configuration"]=dict(instance.configuration())
    return instance,identity


def execute(args,report,persist):
    import numpy as np
    if args.host_only:
        for pools in (1,8):
            case=make_case(pools);initial=blank_rings(pools,256)
            expected=expected_rings(case,initial)
            report.setdefault("cases",{})[f"p{pools}"]=dict(status="HOST_REFERENCE_ONLY",rows=len(case["tokenBanks"]),
                valid_rows=int(case["tokenValid"].sum()),written_slots=int(np.count_nonzero(expected["positionTags"]>=0)))
        report["status"]="HOST_REFERENCE_ONLY";return
    import torch
    import torch_npu
    torch.npu.set_device(args.device)
    bridge,identity=load_bridge(args.library,args.output,args.sdk_root,args.verbose)
    report["build_identity"]=identity;persist()
    dtypes={"inputLatent":torch.float32,"inputRope":torch.bfloat16,"latentRing":torch.bfloat16,"ropeRing":torch.bfloat16,
        "positionTags":torch.int64,"epochTags":torch.int64,"tokenBanks":torch.int32,"tokenPositions":torch.int64,
        "tokenEpochs":torch.int64,"tokenValid":torch.bool}
    def device(value,name):return torch.from_numpy(np.ascontiguousarray(value)).to(device=args.device,dtype=dtypes[name])
    def host(value):return value.detach().float().cpu().numpy() if value.is_floating_point() else value.detach().cpu().numpy()
    def check(name,buffers,case,expected):
        torch.npu.synchronize()
        checks={key:bool(np.array_equal(host(buffers[key]),value)) for key,value in expected.items()}
        checks["input_latent_unchanged"]=bool(np.array_equal(host(buffers["inputLatent"]),case["inputLatent"]))
        checks["input_rope_unchanged"]=bool(np.array_equal(host(buffers["inputRope"]),case["inputRope"]))
        report.setdefault("cases",{})[name]=dict(passed=all(checks.values()),checks=checks)
        persist()
        if not all(checks.values()):raise RuntimeError("Smoke correctness failed: "+name)
    with torch.inference_mode():
        for pools in (1,8):
            case=make_case(pools);initial=blank_rings(pools,256)
            buffers={name:device(value,name) for name,value in {**case,**initial}.items()}
            ordered=lambda:[buffers[name] for name in TENSOR_NAMES]
            blocks=min(len(case["tokenBanks"]),8)
            bridge.run(ordered(),blocks,False)
            check(f"p{pools}_off",buffers,case,initial)
            bridge.run(ordered(),blocks,True)
            expected=expected_rings(case,initial)
            check(f"p{pools}_eager",buffers,case,expected)
            if args.graph or args.timing=="graph":
                graph=torch.npu.NPUGraph()
                with torch.npu.graph(graph):bridge.run(ordered(),blocks,True)
                for phase in (0,1):
                    updated=make_case(pools,phase=phase)
                    for name,value in updated.items():buffers[name].copy_(device(value,name))
                    poison=blank_rings(pools,256)
                    poison["latentRing"].fill(np.nan);poison["ropeRing"].fill(np.nan)
                    poison["positionTags"].fill(-999);poison["epochTags"].fill(-999)
                    for name,value in poison.items():buffers[name].copy_(device(value,name))
                    graph.replay()
                    expected=expected_rings(updated,poison)
                    # Untouched poison must remain poison; compare with equal_nan.
                    torch.npu.synchronize()
                    checks={name:bool(np.array_equal(host(buffers[name]),value,equal_nan=True)) for name,value in expected.items()}
                    checks["input_latent_unchanged"]=bool(np.array_equal(host(buffers["inputLatent"]),updated["inputLatent"]))
                    checks["input_rope_unchanged"]=bool(np.array_equal(host(buffers["inputRope"]),updated["inputRope"]))
                    name=f"p{pools}_graph_phase{phase}"
                    report.setdefault("cases",{})[name]=dict(passed=all(checks.values()),checks=checks,
                        captured_once=True,dynamic_metadata_updated=True,untouched_poison_retained=True)
                    persist()
                    if not all(checks.values()):raise RuntimeError("Poisoned/dynamic graph replay failed: "+name)
    report["status"]="DEVICE_RING_WRITER_VALIDATED"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path)
    parser.add_argument("--sdk-root",type=Path)
    parser.add_argument("--device",default="npu:0")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--host-only",action="store_true")
    parser.add_argument("--graph",action="store_true",help="Validate capture, poisoning and changed device metadata; no latency measurement")
    parser.add_argument("--timing",choices=("none","graph"),default="none",help="graph is a replay correctness check only; no timing claim")
    parser.add_argument("--verbose",action="store_true")
    args=parser.parse_args(argv)
    if args.output.exists():parser.error("Use a new isolated output directory")
    args.output.mkdir(parents=True)
    report=dict(status="STARTED",scope="isolated_ring_writer_not_producer",prolog_validated=False,c4_validated=False,
        native_quantizer_tested=False,vendor_modified=False,latency="NOT_MEASURED",cases={})
    def persist():
        path=args.output/"results.json";temporary=path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n");temporary.replace(path)
    try:
        if not args.host_only and args.library is None:raise ValueError("Device smoke requires --library")
        execute(args,report,persist);persist();print(str(args.output/"results.json"));return 0
    except Exception as error:
        detail=traceback.format_exc()
        report.update(status="ERROR",error=str(error),traceback=detail,stop_after_first_failure=True,
            device_fault_suspected=any(code in detail for code in FAULT_CODES))
        persist();print(str(args.output/"results.json"));return 1


if __name__=="__main__":raise SystemExit(main())
