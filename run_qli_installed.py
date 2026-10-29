#!/usr/bin/env python3
"""Test the installed experimental QLI wrapper with an existing DSL source tree.

No package installs, upgrades, source downloads or compiler patches. The pinned
test harness maps existing source files to the wrapper's two ops.* imports.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import traceback
import urllib.request

HARNESS_URL = "https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/51d02efe90eb7ff93aada7bca3658117dbb55449/qli-a5/test_qli_a5.py"
HARNESS_SHA256 = "84637c150efef40a5eb0c1e9dbd4bd7d22fe83fb6b027f7f7ead20da2429c822"
RUNNER_VERSION = "qli-topk-override-v3"


def identify(path):
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_harness(run, insecure):
    context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
    with urllib.request.urlopen(HARNESS_URL, context=context, timeout=120) as response:
        body = response.read()
    if hashlib.sha256(body).hexdigest() != HARNESS_SHA256:
        raise RuntimeError("Test script SHA256 mismatch")
    script = run / "test_qli_a5.py"
    script.write_bytes(body)
    spec = importlib.util.spec_from_file_location("qli_a5_harness", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def diagnostic_toolchain(report):
    # Read-only inventory; never alter include paths or mix host/CANN toolchains.
    home = os.environ.get("ASCEND_HOME_PATH", "")
    report["ascend_home"] = home
    if not home:
        return
    headers = set()
    for root in (Path(home), Path(home).parent):
        hcc = root / "toolkit/toolchain/hcc"
        if hcc.is_dir():
            for name in ("cstdint", "c++config.h"):
                headers.update(str(p) for p in hcc.rglob(name))
    report["hcc_headers"] = sorted(headers)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--insecure", action="store_true", help="Allow unverified HTTPS for the pinned, hash-checked test script")
    p.add_argument("--dsl-root", "--ops-dir", dest="dsl_root",
                   help="Existing cannbot-dsl repo, samples directory or flat ops directory")
    p.add_argument("--work-dir", default="/mnt")
    p.add_argument("--suite", choices=("smoke", "standard", "full"), default="smoke")
    p.add_argument("--case", help="Run just this case from the selected suite")
    p.add_argument("--topk", type=int, help="Override TopK (1..8192), preserving the original input data")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--worker-dir", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.topk is not None and not 1 <= args.topk <= 8192:
        p.error("--topk must be between 1 and 8192")
    if not args.worker_dir:
        Path(args.work_dir).mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="qli-installed-", dir=args.work_dir)).resolve()
        print(f"Runner: {RUNNER_VERSION} ({Path(__file__).resolve()})", flush=True)
        print(f"Logs and results: {run}", flush=True)
        # Capture Python, compiler subprocess and native runtime output together.
        cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker-dir", str(run)]
        with (run / "run.log").open("w") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED="1"))
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return proc.wait()
    run = Path(args.worker_dir)
    report = {"python": sys.executable, "stage": "installed-package-check",
              "runner_version": RUNNER_VERSION, "runner": identify(Path(__file__).resolve()),
              "harness_url": HARNESS_URL, "harness_sha256": HARNESS_SHA256}

    def save():
        (run / "diagnostics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    try:
        for package in ("cann-ops-transformer-dsl", "cannbot-dsl"):
            try:
                dist = metadata.distribution(package)
            except metadata.PackageNotFoundError:
                raise RuntimeError(f"{package} is not visible to {sys.executable}. "
                                   "Use the existing qli-a5-ready venv/bin/python, which was created "
                                   "with system-site-packages; no new install is performed.") from None
            report[package] = {"version": dist.version, "location": str(dist.locate_file(""))}
        diagnostic_toolchain(report)
        save()
        report["stage"] = "download-verified-test-script"
        harness = load_harness(run, args.insecure)
        report["stage"] = "map-existing-dsl-source"
        args.dsl_source, args.backend = "ops", "both"
        files = harness.resolve_dsl_files(args)
        report["implementation"] = {name: identify(path) for name, path in files.items()}
        metadata_path = files["quant_lightning_indexer_metadata_dsl"]
        source_root = (metadata_path.parent.parent if metadata_path.parent.name == "quant_lightning_indexer_metadata_dsl"
                       else metadata_path.parent)
        args.dsl_root = str(source_root)
        save()
        report["stage"] = "import-installed-dispatcher-and-source"
        import torch
        import torch_npu
        torch.npu.set_device(args.device)
        backends = harness.load_backends(args)
        report["dispatchers"] = {name: [harness.file_identity(fn) for fn in funcs]
                                 for name, funcs in backends.items()}
        print("PASS: installed dispatcher and mapped source imports (NPU execution not yet checked)", flush=True)
        report["stage"] = "dsl-metadata-compile-and-execute"
        save()
        cu = torch.tensor([0, 3], dtype=torch.int32, device=args.device)
        qlen = torch.tensor([3], dtype=torch.int32, device=args.device)
        klen = torch.tensor([257], dtype=torch.int32, device=args.device)
        with torch.inference_mode():
            meta = backends["dsl"][1](
                cu_seqlens_q=cu, seqused_q=qlen, seqused_k=klen,
                batch_size=1, max_seqlen_q=3, max_seqlen_k=257,
                num_heads_q=32, num_heads_k=1, head_dim=128,
                topk=args.topk if args.topk is not None else 32,
                mask_mode=0, cmp_ratio=1, layout_q="TND", layout_k="PA_BBND")
            torch.npu.synchronize()
        if meta.dtype != torch.int32 or tuple(meta.shape) != (1024,) or meta.device != cu.device:
            raise RuntimeError(f"Invalid DSL metadata: {meta.shape}, {meta.dtype}, {meta.device}")
        report["metadata"] = {"shape": list(meta.shape), "dtype": str(meta.dtype)}
        print("PASS: DSL metadata compile/execution; now checking QLI kernels and accuracy", flush=True)
        report["stage"] = "comparison"
        save()
        sys.argv = [harness.__file__, "--backend", "both", "--dsl-source", "ops",
                    "--dsl-root", args.dsl_root, "--suite", args.suite, "--fail-fast",
                    "--device", args.device, "--output", str(run / "results")]
        if args.profile:
            sys.argv.append("--profile")
        if args.case:
            sys.argv.extend(["--case", args.case])
        if args.topk is not None:
            sys.argv.extend(["--topk", str(args.topk)])
        code = harness.main()
        report["status"] = "PASS" if code == 0 else "FAIL"
        save()
        return code
    except Exception:
        report["status"] = "BLOCKED"
        report["error"] = traceback.format_exc()
        if "'cstdint' file not found" in report["error"]:
            report["diagnosis"] = (
                "Source imports were resolved. This is the separate AICPU C++ header/toolchain failure "
                "seen earlier, not a missing ops module. QLI kernel comparison has not completed. "
                "No include paths, compiler, CANN installation or package versions were changed.")
        save()
        print(report["error"], file=sys.stderr, flush=True)
        if "diagnosis" in report:
            print(report["diagnosis"], flush=True)
        print(f"Stopped at: {report['stage']}. Full report: {run}/diagnostics.json", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
