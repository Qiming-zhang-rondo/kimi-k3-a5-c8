"""Compile only the Torch host bridge around an explicitly built ASC library."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import threading

_LOCK=threading.Lock()
_MODULES={}
_LIBRARY_HASHES={}


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_single_pass(library, *, sdk_root=None, build_directory=None, verbose=False):
    library=Path(library).expanduser().resolve()
    if library.name!="libqsfa_recent128_direct.so" or not library.is_file() or not library.stat().st_size:
        raise ValueError("Pass the explicitly built nonempty libqsfa_recent128_direct.so")
    directory=Path(__file__).resolve().parent
    files=[directory/name for name in ("torch_bridge.cpp","single_pass_api.h","single_pass_tiling.h")]
    if any(not path.is_file() for path in files):
        raise ValueError("Single-pass bridge source/API/tiling headers are incomplete")
    previous=sys.path[:]
    try:
        sys.path.insert(0,str(directory.parent/"torch"))
        from load_sfa_direct import _cache_sdk, _sdk_include
    finally:
        sys.path[:]=previous
    recorded=_cache_sdk(library.parent)
    selected=sdk_root or os.environ.get("ASCEND_HOME_PATH") or recorded
    if not selected:
        raise ValueError("Pass --sdk-root matching the ASC shared-library build")
    sdk=Path(selected).expanduser().resolve()
    if recorded and recorded!=sdk:
        raise ValueError("SDK differs from this library's recorded SFA_SDK_ROOT")
    acl=_sdk_include(sdk)
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load
    npu=Path(torch_npu.__file__).resolve().parent
    include,lib=npu/"include",npu/"lib"
    required=("NPUGuard.h","NPUStream.h","NPUCachingAllocator.h")
    if any(not (include/"torch_npu/csrc/core/npu"/name).is_file() for name in required):
        raise ValueError("Active Torch-NPU lacks direct stream/allocator headers")
    if not (lib/"libtorch_npu.so").is_file():
        raise ValueError("Active Torch-NPU lacks libtorch_npu.so")
    hashes={path.name:sha256(path) for path in files}
    lib_hash=sha256(library)
    key=(str(library),lib_hash,json.dumps(hashes,sort_keys=True))
    with _LOCK:
        if str(library) in _LIBRARY_HASHES and _LIBRARY_HASHES[str(library)]!=lib_hash:
            raise RuntimeError("Shared library changed in this process; use a fresh Python process")
        if key in _MODULES:
            return _MODULES[key]
        build=Path(build_directory).expanduser().resolve() if build_directory else Path(tempfile.mkdtemp(prefix="qsfa-single-pass-"))
        build.mkdir(parents=True,exist_ok=True)
        link=[shlex.quote(str(library)),"-Xlinker","-rpath","-Xlinker",shlex.quote(str(library.parent)),
              shlex.quote("-L"+str(lib)),"-ltorch_npu","-Xlinker","-rpath","-Xlinker",shlex.quote(str(lib))]
        module=load(name="qsfa_single_pass_"+hashlib.sha256(repr(key).encode()).hexdigest()[:20],
            sources=[str(files[0])],extra_include_paths=[str(directory),str(include),str(acl)],
            extra_cflags=["-O2","-std=c++17"],extra_ldflags=link,
            build_directory=str(build),verbose=bool(verbose))
        actual=dict(module.configuration())
        if actual.get("abi_version")!=1 or actual.get("buffers_bytes")!=15*8:
            raise RuntimeError("Shared single-pass library has incompatible C ABI")
        identity=dict(library=str(library),library_sha256=lib_hash,bridge_sha256=hashes,
            sdk_root=str(sdk),torch=str(torch.__version__),torch_npu=str(getattr(torch_npu,"__version__","unknown")),
            torch_cxx11_abi=getattr(torch._C,"_GLIBCXX_USE_CXX11_ABI",None),configuration=actual,
            asc_compiled_by_loader=False,npu_launched_by_loader=False,build_directory=str(build))
        (build/"single_pass_bridge_build.json").write_text(json.dumps(identity,indent=2)+"\n")
        module._single_pass_build_identity=identity
        _LIBRARY_HASHES[str(library)]=lib_hash
        _MODULES[key]=module
        return module
