#!/usr/bin/env python3
"""Resolve the installed experimental QLI environment and run its Torch entry point.

No installs, upgrades, compiler patches, or substitution of sample implementations.
"""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import runpy
import ssl
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request

HARNESS_URL = "https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/0884ceac8d243d369261ad4ea8a566a0aedd47f4/qli-a5/test_qli_a5.py"
HARNESS_SHA256 = "5287df84caa50d77cc8d2260c48607efbc6f38addd78ba5040c0da56f5a6eba8"
NAMES = ("quant_lightning_indexer_dsl.py", "quant_lightning_indexer_metadata_dsl.py")


def valid_ops(path):
    path = Path(path).resolve()
    return path.name == "ops" and all((path / n).is_file() for n in NAMES)


def normalize_ops(value):
    path = Path(value).expanduser().resolve()
    for candidate in (path, path / "ops"):
        if valid_ops(candidate):
            return candidate
    return None


def scan_sources(roots, seconds=25):
    deadline = time.monotonic() + seconds
    candidates, notes = set(), []
    skip = {".git", "__pycache__", ".cache", "node_modules", "proc", "sys", "dev"}
    for root in roots:
        if not Path(root).is_dir():
            continue
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in skip]
            if Path(current).name == "ops" and all(n in files for n in NAMES):
                candidates.add(str(Path(current).resolve()))
            if time.monotonic() > deadline:
                notes.append("Source scan time limit reached; inventory is incomplete")
                return sorted(candidates), notes
    return sorted(candidates), notes


def identify(path):
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def resolve_sources(args, report):
    explicit = args.ops_dir or os.environ.get("CANNBOTDSL_TEST_OPS_DIR")
    if explicit:
        path = normalize_ops(explicit)
        if path is None:
            raise RuntimeError(f"Specified source has no ops package containing BOTH implementations: {explicit}")
        return path
    # Prefer an implementation already configured in this exact Python environment.
    try:
        spec = importlib.util.find_spec("ops")
    except (ImportError, ValueError):
        spec = None
    if spec and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            if valid_ops(loc):
                return Path(loc).resolve()
    candidates, notes = scan_sources(args.search_root or ["/mnt", "/workspace", "/opt", "/home", "/root", "/usr/local/Ascend"])
    report["source_candidates"], report["source_scan_notes"] = candidates, notes
    if not candidates:
        raise RuntimeError(
            "The Torch wheel is installed but its required ops/quant_lightning_indexer_dsl.py and "
            "ops/quant_lightning_indexer_metadata_dsl.py implementation pair was not found. "
            "The inspected upstream wheel delegates to these external files. "
            "Obtain the exact ops source directory used by the verified build from your mentor/image provider; "
            "then rerun with --ops-dir /path/to/ops. This runner will not silently replace that implementation "
            "with the previously downloaded cannbot samples. See diagnostics.json for inventory."
        )
    fingerprints = {}
    for candidate in candidates:
        path = Path(candidate)
        key = tuple(identify(path / n)["sha256"] for n in NAMES)
        fingerprints.setdefault(key, path)
    if len(fingerprints) != 1:
        raise RuntimeError("Multiple different DSL source versions found. Select the verified one with --ops-dir. "
                           + " Candidates: " + ", ".join(candidates))
    return next(iter(fingerprints.values()))


def diagnostic_toolchain(report):
    # Read-only inventory: never alter include paths or mix host/CANN toolchains.
    home = os.environ.get("ASCEND_HOME_PATH", "")
    report["ascend_home"] = home
    if not home:
        return
    roots = [Path(home), Path(home).parent]
    headers = set()
    for root in roots:
        hcc = root / "toolkit/toolchain/hcc"
        if hcc.is_dir():
            for name in ("cstdint", "c++config.h"):
                headers.update(str(p) for p in hcc.rglob(name))
    report["hcc_headers"] = sorted(headers)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--insecure", action="store_true", help="Allow unverified HTTPS for the pinned, hash-checked test script")
    p.add_argument("--ops-dir", help="Verified ops directory, or its parent")
    p.add_argument("--search-root", action="append")
    p.add_argument("--work-dir", default="/mnt")
    p.add_argument("--suite", choices=("smoke", "standard", "full"), default="smoke")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--worker-dir", help=argparse.SUPPRESS)
    args = p.parse_args()
    if not args.worker_dir:
        Path(args.work_dir).mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="qli-installed-", dir=args.work_dir)).resolve()
        print(f"Logs and results: {run}", flush=True)
        # Capture Python, compiler subprocess, and native runtime diagnostics in one file.
        cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker-dir", str(run)]
        with (run / "run.log").open("w") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED="1"))
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line); log.flush()
            return proc.wait()
    run = Path(args.worker_dir)
    report = {"python": sys.executable, "stage": "installed-package-check"}
    def save():
        (run / "diagnostics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    try:
        try:
            dist = metadata.distribution("cann-ops-transformer-dsl")
            report["wheel_version"] = dist.version
            report["wheel_entries"] = [str(f) for f in dist.files or [] if "indexer" in str(f) or "entry_points" in str(f)]
            report["entry_points"] = [str(ep) for ep in dist.entry_points]
        except metadata.PackageNotFoundError:
            raise RuntimeError("The experimental wheel is not installed in this Python environment. "
                               "Use the same python used to install your mentor's built wheel.") from None
        diagnostic_toolchain(report)
        save()
        report["stage"] = "resolve-verified-implementation"
        source = resolve_sources(args, report)
        report["implementation"] = [identify(source / n) for n in NAMES]
        sys.path.insert(0, str(source.parent))
        os.environ["PYTHONPATH"] = str(source.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
        print(f"Using installed Torch wheel and DSL implementation: {source}", flush=True)
        save()
        report["stage"] = "import-and-metadata-preflight"
        import torch
        import torch_npu
        import cann_ops_transformer.ops
        torch.npu.set_device(args.device)
        from cann_ops_transformer.ops.ds41 import quant_lightning_indexer_metadata
        for name in ("ops.quant_lightning_indexer_dsl", "ops.quant_lightning_indexer_metadata_dsl"):
            module = importlib.import_module(name)
            if Path(module.__file__).resolve().parent != source:
                raise RuntimeError(f"Import shadowing: {name} loaded from {module.__file__}, expected {source}")
        cu = torch.tensor([0, 3], dtype=torch.int32, device=args.device)
        qlen = torch.tensor([3], dtype=torch.int32, device=args.device)
        klen = torch.tensor([257], dtype=torch.int32, device=args.device)
        meta = quant_lightning_indexer_metadata(
            cu_seqlens_q=cu, seqused_q=qlen, seqused_k=klen,
            batch_size=1, max_seqlen_q=3, max_seqlen_k=257,
            num_heads_q=32, num_heads_k=1, head_dim=128, topk=32,
            mask_mode=0, cmp_ratio=1, layout_q="TND", layout_k="PA_BBND")
        torch.npu.synchronize()
        report["metadata"] = {"shape": list(meta.shape), "dtype": str(meta.dtype)}
        print("PASS: installed entry point, source imports and metadata execution", flush=True)
        report["stage"] = "comparison"
        save()
        context = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()
        with urllib.request.urlopen(HARNESS_URL, context=context, timeout=120) as response:
            body = response.read()
        if hashlib.sha256(body).hexdigest() != HARNESS_SHA256:
            raise RuntimeError("Test script SHA256 mismatch")
        script = run / "test_qli_a5.py"
        script.write_bytes(body)
        sys.argv = [str(script), "--backend", "both", "--dsl-source", "ops", "--suite", args.suite,
                    "--device", args.device, "--output", str(run / "results")]
        if args.profile:
            sys.argv.append("--profile")
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            code = int(exc.code or 0)
            report["status"] = "PASS" if code == 0 else "FAIL"
            save()
            return code
        report["status"] = "PASS"; save(); return 0
    except Exception:
        report["status"] = "BLOCKED"
        report["error"] = traceback.format_exc()
        save()
        print(report["error"], file=sys.stderr, flush=True)
        print(f"Stopped at: {report['stage']}. Full report: {run}/diagnostics.json", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
