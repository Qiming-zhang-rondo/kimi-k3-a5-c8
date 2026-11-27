#!/usr/bin/env python3
"""Install an opt-in SFA runtime overlay into an explicit mxllm source checkout.

Default operation is read-only checking. No pip, environment changes, imports
of the runtime, NPU compilation or device execution occur here. Restore paths
and original bytes are recorded under the caller's external experiment output.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

PATCH = Path("ascend_vllm/patch/worker/patch_sfa_v1_mxfp4.py")
WORKER = Path("ascend_vllm/patch/worker/__init__.py")
PLATFORM = Path("ascend_vllm/patch/platform/__init__.py")
PINNED_SOURCE_SHA256 = "b209150bd23ff6401ac8b69901c06582bef5515b577581d7022a6e292d1a507c"
PINNED_WORKER_SHA256 = "6fa56c057930b2de8f447f6418836bd1ecf74a8287f8f8d8c51b2e96a3368879"
PINNED_PLATFORM_SHA256 = "f3d13c2a09404a0b124e88e744cf7682681a9d188d069035fba9d8ed7822661c"
SOURCE_PROFILES = {
    "legacy": (PINNED_SOURCE_SHA256, PINNED_WORKER_SHA256, PINNED_PLATFORM_SHA256),
    # Reviewed original files at mxllm PR199 head 342680fb. This selects exact
    # source bytes only; it does not certify the installed VA/CANN stack.
    "pr199": ("8d9f93cdd1d374d0fcbc309c1a0047c5f9fc6cb068fda271f6ee07be26798c5c",
              "e388c47319c6029f8904140b16977558b00ec95faa00360a403b26085c00bf2d",
              "35e7fd676da66e15f1c68c97cc3a875bc112644b5270e8de0d86c01acff879b7"),
}
STATE = ".sfa_recent128_overlay.json"
TAP = ("    # BEGIN SFA_RECENT128_RAW_KV_TAP\n"
       "    from sfa_recent128.hooks import capture_native_kv\n"
       "    capture_native_kv(impl, k_nope, k_rope)\n"
       "    # END SFA_RECENT128_RAW_KV_TAP\n")
INSTALL = ("\n# BEGIN SFA_RECENT128_WORKER_INSTALL\n"
           "from sfa_recent128.hooks import install\n"
           "install()\n"
           "# END SFA_RECENT128_WORKER_INSTALL\n")
PLATFORM_INSTALL = ("\n# BEGIN SFA_RECENT128_CONFIG_BRIDGE_INSTALL\n"
                    "from sfa_recent128.config_bridge import install as _install_recent128_config_bridge\n"
                    "_install_recent128_config_bridge()\n"
                    "# END SFA_RECENT128_CONFIG_BRIDGE_INSTALL\n")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def expected_digest(value):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("Expected SHA256 must be exactly 64 hexadecimal characters")
    return value.lower()


def patch_source(data):
    source = data.decode("utf-8")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_mxfp4_custom_kv_rmsnorm_rope"]
    if len(functions) != 1:
        raise ValueError("Expected exactly one MXFP4 native normalization helper")
    body = functions[0].body
    def assignment(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    rope = [i for i, node in enumerate(body) if assignment(node, "k_rope")]
    prefix = [i for i, node in enumerate(body) if assignment(node, "prefix_shape")]
    if len(rope) != 1 or len(prefix) != 1 or prefix[0] != rope[0] + 1:
        raise ValueError("Reviewed k_rope -> prefix_shape insertion boundary does not match")
    point = body[prefix[0]]
    if point.col_offset != 4:
        raise ValueError("Unexpected helper indentation; review source before installing")
    lines = source.splitlines(keepends=True)
    lines.insert(point.lineno-1, TAP)
    result = "".join(lines)
    ast.parse(result)
    return result.encode("utf-8")


def worker_source(data):
    source = data.decode("utf-8")
    tree = ast.parse(source)
    imported = any(isinstance(node, ast.ImportFrom) and
                   node.module == "ascend_vllm.patch.worker" and
                   any(alias.name == "patch_sfa_v1_mxfp4" for alias in node.names)
                   for node in tree.body)
    if not imported:
        raise ValueError("Worker must import the mxllm MXFP4 patch before installing the overlay")
    result = source + INSTALL
    ast.parse(result)
    return result.encode("utf-8")


def patch_platform_source(data):
    source = data.decode("utf-8")
    ast.parse(source)
    result = source + PLATFORM_INSTALL
    ast.parse(result)
    return result.encode("utf-8")


def package_files(directory):
    if not directory.is_dir():
        raise ValueError(f"Runtime package is missing: {directory}")
    files = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if "__pycache__" in relative.parts or path.suffix in (".pyc", ".pyo") or relative.as_posix() == STATE:
            continue
        if path.is_symlink():
            raise ValueError(f"Runtime package must not contain symlinks: {relative}")
        if path.is_file():
            files[relative.as_posix()] = path.read_bytes()
    if not {"__init__.py", "hooks.py", "config_bridge.py"} <= files.keys():
        raise ValueError("Runtime package must provide __init__.py, hooks.py and config_bridge.py")
    return files


def _write(path, data):
    with tempfile.NamedTemporaryFile(prefix=".sfa-overlay-", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        if path.exists():
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install_overlay(mxllm_root, *, apply=False, output=None,
                    expected_source_sha256=PINNED_SOURCE_SHA256,
                    expected_worker_sha256=PINNED_WORKER_SHA256,
                    expected_platform_sha256=PINNED_PLATFORM_SHA256, runtime_root=None):
    root = Path(mxllm_root).resolve()
    if any(name in root.parts for name in ("site-packages", "dist-packages")):
        raise ValueError("Target an explicit source checkout, not installed site-packages")
    runtime = Path(runtime_root) if runtime_root is not None else Path(__file__).parent/"sfa_recent128"
    contents = package_files(runtime)
    hashes = {name: sha(data) for name, data in contents.items()}
    expected = {str(PATCH): expected_digest(expected_source_sha256),
                str(WORKER): expected_digest(expected_worker_sha256),
                str(PLATFORM): expected_digest(expected_platform_sha256)}
    paths = {str(path): root/path for path in (PATCH, WORKER, PLATFORM)}
    for path in paths.values():
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Required source file missing or linked outside checkout: {path}")
    current = {name: path.read_bytes() for name, path in paths.items()}
    target = root/"sfa_recent128"
    if target.is_symlink():
        raise ValueError("Refusing an existing linked runtime package")
    state_path = target/STATE
    marked = any(b"SFA_RECENT128_" in data for data in current.values())
    if marked:
        if not state_path.is_file():
            raise ValueError("Partial/unmanaged overlay: restore original source before installing")
        state = json.loads(state_path.read_text())
        if str(PLATFORM) not in state.get("original_sha256", {}):
            raise ValueError("Older two-file overlay detected: restore all original files and remove its runtime "
                             "using its backup manifest, then reinstall the three-file overlay")
        original = {}
        for name, block in ((str(PATCH), TAP), (str(WORKER), INSTALL), (str(PLATFORM), PLATFORM_INSTALL)):
            data = current[name]
            token = block.encode()
            if data.count(token) != 1:
                raise ValueError(f"Overlay marker altered or duplicated: {name}")
            original[name] = data.replace(token, b"", 1)
        regenerated = {str(PATCH): patch_source(original[str(PATCH)]),
                       str(WORKER): worker_source(original[str(WORKER)]),
                       str(PLATFORM): patch_platform_source(original[str(PLATFORM)])}
        actual = {name: sha(data) for name, data in current.items()}
        installed_hashes = {name: sha(data) for name, data in package_files(target).items()}
        if (state.get("original_sha256") != expected or
                any(sha(original[name]) != expected[name] for name in expected) or
                regenerated != current or state.get("installed_source_sha256") != actual or
                state.get("runtime_sha256") != hashes or installed_hashes != hashes):
            raise ValueError("Installed source/runtime differs from the reviewed manifest; restore and review before reinstall")
        return dict(status="ALREADY_INSTALLED",changed=False,mode="apply" if apply else "check",
                    mxllm_root=str(root),backup_directory=state.get("backup_directory"),
                    source_sha256=actual,runtime_sha256=hashes,device_launched=False)
    if target.exists():
        raise ValueError("Unmanaged sfa_recent128 package already exists; no overwrite performed")
    actual = {name: sha(data) for name, data in current.items()}
    for name in expected:
        if actual[name] != expected[name]:
            flag = {str(PATCH): "--expected-source-sha256", str(WORKER): "--expected-worker-sha256",
                    str(PLATFORM): "--expected-platform-sha256"}[name]
            raise ValueError(f"Source SHA256 mismatch for {name}: expected={expected[name]} actual={actual[name]}. "
                             f"Review the actual file, then pass its exact {flag}; there is no force option.")
    modified = {str(PATCH): patch_source(current[str(PATCH)]), str(WORKER): worker_source(current[str(WORKER)]),
                str(PLATFORM): patch_platform_source(current[str(PLATFORM)])}
    result = dict(status="READY",changed=False,mode="check",mxllm_root=str(root),
                  original_sha256=actual,installed_source_sha256={name:sha(data) for name,data in modified.items()},
                  runtime_sha256=hashes,device_launched=False)
    if not apply:
        return result
    if output is None:
        raise ValueError("--apply requires --output for recoverable experiment backups")
    output = Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Backups must be in an experiment output outside the target source checkout")
    output.mkdir(parents=True,exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="sfa-overlay-",dir=output))
    for name,path in paths.items():
        destination = backup/"original"/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    manifest = dict(result,mode="apply",status="BACKED_UP",backup_directory=str(backup),
                    restore=dict(source_root=str(backup/"original"),target_root=str(root),
                                 remove_runtime=str(target)),runtime_package="sfa_recent128")
    manifest_path = backup/"manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    staged = Path(tempfile.mkdtemp(prefix=".sfa-runtime-",dir=root))
    try:
        for name,data in contents.items():
            path=staged/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(data)
        state=dict(original_sha256=actual,installed_source_sha256=result["installed_source_sha256"],
                   runtime_sha256=hashes,backup_directory=str(backup))
        (staged/STATE).write_text(json.dumps(state,indent=2)+"\n")
        os.replace(staged,target)
        for name,data in modified.items():
            _write(paths[name],data)
        manifest.update(status="INSTALLED",changed=True)
    except Exception:
        for name,data in current.items():
            _write(paths[name],data)
        if target.exists():
            shutil.rmtree(target)
        manifest.update(status="ROLLED_BACK",changed=False)
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
        raise
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mxllm-root",type=Path,required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--check",action="store_true",help="Read-only validation (default)")
    mode.add_argument("--apply",action="store_true",help="Copy runtime and patch this source checkout")
    parser.add_argument("--output",type=Path,help="External experiment output for original-source backups/manifest")
    parser.add_argument("--source-profile",choices=tuple(SOURCE_PROFILES),default="legacy",
        help="Exact reviewed MXLLM three-file source set; pr199 is head 342680fb, not a compatibility override")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-worker-sha256")
    parser.add_argument("--expected-platform-sha256")
    args=parser.parse_args()
    try:
        source_hash,worker_hash,platform_hash=SOURCE_PROFILES[args.source_profile]
        result=install_overlay(args.mxllm_root,apply=args.apply,output=args.output,
            expected_source_sha256=args.expected_source_sha256 or source_hash,
            expected_worker_sha256=args.expected_worker_sha256 or worker_hash,
            expected_platform_sha256=args.expected_platform_sha256 or platform_hash)
        result["source_profile"]=args.source_profile
    except (ValueError,OSError,SyntaxError,json.JSONDecodeError) as error:
        parser.exit(2,f"Overlay refused: {error}\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
