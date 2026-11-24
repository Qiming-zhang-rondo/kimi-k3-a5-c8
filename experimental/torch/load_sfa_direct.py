"""Load the real ASC direct-launch library through a small Torch C++ bridge.

No network, package installation, vendor registration, or device launch occurs
here. The ASC shared library must already have been built by the matching SDK.
Only sfa_direct_bridge.cpp is compiled by torch.utils.cpp_extension.load.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile
import threading


_LOCK = threading.Lock()
_LOADED_LIBRARY_HASHES: dict[str, str] = {}
_MODULES: dict[tuple[str, str, str], object] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_sdk(library_dir: Path) -> Path | None:
    # The CMake target normally writes into the build root; lib/ is also
    # accepted. Do not search unrelated SDKs or the user's vendor installation.
    for parent in (library_dir, library_dir.parent):
        cache = parent / "CMakeCache.txt"
        if not cache.is_file():
            continue
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("SFA_SDK_ROOT:PATH="):
                value = line.split("=", 1)[1]
                if value:
                    return Path(value).expanduser().resolve()
    return None


def _sdk_include(sdk_root: Path) -> Path:
    candidates = (sdk_root / "aarch64-linux/include", sdk_root / "include")
    for include in candidates:
        if (include / "acl/acl.h").is_file() and (include / "acl/acl_op.h").is_file():
            return include.resolve()
    raise ValueError("The selected SDK has no matching acl/acl.h and acl/acl_op.h include directory")


def load_sfa_direct(library_dir, *, sdk_root=None, build_directory=None, verbose=False):
    """Return the compiled bridge module; this function performs no NPU work.

    `library_dir` must contain libsfa_torch_direct.so, built with
    SFA_BUILD_TORCH_DIRECT=ON. Its two ASC translation units own the original
    and fused kernel launch wrappers; this loader never recompiles their ASC
    implementation or substitutes the old ACLNN declaration-only bridge.
    """
    directory = Path(library_dir).expanduser().resolve()
    library = directory / "libsfa_torch_direct.so"
    if not library.is_file() or library.stat().st_size == 0:
        raise ValueError("library_dir must contain a nonempty libsfa_torch_direct.so built by the matching ASC SDK")
    source = Path(__file__).resolve().with_name("sfa_direct_bridge.cpp")
    if not source.is_file():
        raise ValueError("sfa_direct_bridge.cpp is missing from this source bundle")
    recorded_sdk = _cache_sdk(directory)
    selected_sdk = sdk_root or os.environ.get("ASCEND_HOME_PATH") or recorded_sdk
    if not selected_sdk:
        raise ValueError("Pass sdk_root or set ASCEND_HOME_PATH to the same SDK used to build the direct library")
    sdk = Path(selected_sdk).expanduser().resolve()
    if recorded_sdk is not None and sdk != recorded_sdk:
        raise ValueError("The selected SDK differs from SFA_SDK_ROOT in the library's CMakeCache.txt")
    acl_include = _sdk_include(sdk)

    # Delay heavy imports so help, source inspection and host-only tests do
    # not initialize torch_npu, discover devices, or need a toolkit installed.
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load

    npu_root = Path(torch_npu.__file__).resolve().parent
    npu_include = npu_root / "include"
    npu_library_dir = npu_root / "lib"
    required_headers = (
        "torch_npu/csrc/core/npu/NPUGuard.h",
        "torch_npu/csrc/core/npu/NPUStream.h",
        "torch_npu/csrc/core/npu/NPUCachingAllocator.h",
    )
    if any(not (npu_include / name).is_file() for name in required_headers):
        raise ValueError("The active torch_npu installation does not provide the required direct-bridge headers")
    if not (npu_library_dir / "libtorch_npu.so").is_file():
        raise ValueError("The active torch_npu installation has no lib/libtorch_npu.so")
    repository = source.parents[2]
    contracts = (
        "standalone/sfa_direct_api.h", "standalone/sfa_fixed_contract.h",
        "standalone/sparse_flash_attention_tiling_data.h",
        "upstream/sgl-kernel-npu-597250fc/csrc/sparse_flash_attention/op_kernel/sparse_flash_attention_tiling_data.h",
    )
    if any(not (repository / name).is_file() for name in contracts):
        raise ValueError("The pinned host ABI/tiling headers are missing from the direct-bridge source bundle")
    contract_hashes = {name: _sha256(repository / name) for name in contracts}
    library_hash, source_hash = _sha256(library), _sha256(source)
    source_identity = hashlib.sha256(json.dumps(
        {"source": source_hash, "contracts": contract_hashes}, sort_keys=True).encode()).hexdigest()
    key = (str(library), library_hash, source_identity)
    with _LOCK:
        previous = _LOADED_LIBRARY_HASHES.get(str(library))
        if previous is not None and previous != library_hash:
            raise RuntimeError("The direct library changed after being loaded; start a fresh Python process before testing the new binary")
        if key in _MODULES:
            return _MODULES[key]
        name = "sfa_direct_" + hashlib.sha256("\n".join(key).encode()).hexdigest()[:20]
        build = (Path(build_directory).expanduser().resolve() if build_directory is not None else
                 Path(tempfile.mkdtemp(prefix="sfa-direct-torch-")))
        build.mkdir(parents=True, exist_ok=True)
        # cpp_extension joins these flags into a shell command. Quote path
        # arguments, including spaces; JSON stringification is not shell quoting.
        link = [shlex.quote(str(library)),
                "-Xlinker", "-rpath", "-Xlinker", shlex.quote(str(directory)),
                shlex.quote("-L" + str(npu_library_dir)), "-ltorch_npu",
                "-Xlinker", "-rpath", "-Xlinker", shlex.quote(str(npu_library_dir))]
        module = load(
            name=name, sources=[str(source)],
            extra_include_paths=[str(source.parents[2] / "standalone"), str(npu_include), str(acl_include)],
            extra_cflags=["-O2", "-std=c++17"], extra_ldflags=link,
            build_directory=str(build), verbose=bool(verbose),
        )
        # The C++ Prepared constructor checks the shared library's explicit ABI
        # and buffer geometry before any launch. Record only reproducibility
        # facts here; do not claim that import performed an accuracy test.
        identity = {
            "library": str(library), "library_sha256": library_hash,
            "bridge_source": str(source), "bridge_source_sha256": source_hash,
            "bridge_host_contract_sha256": contract_hashes,
            "sdk_root": str(sdk), "torch": str(torch.__version__),
            "torch_npu": str(getattr(torch_npu, "__version__", "unknown")),
            "torch_cxx11_abi": getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
            "build_directory": str(build), "asc_compiled_by_loader": False,
            "npu_launched_by_loader": False,
        }
        (build / "sfa_direct_bridge_build.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        module._sfa_build_identity = identity
        _LOADED_LIBRARY_HASHES[str(library)] = library_hash
        _MODULES[key] = module
        return module
