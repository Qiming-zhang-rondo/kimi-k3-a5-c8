#!/usr/bin/env python3
"""Download SHA-pinned DSL recent128 experiment; no installs or CANN changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import tempfile
import traceback
import urllib.request

REPOSITORY = "Qiming-zhang-rondo/kimi-k3-a5-c8"
PAYLOAD_COMMIT = "fad3c596a69abf8c76d831a8b1d3673fb855497f"
FILES = {
    "dsl_reference.py": "8bfbd1c0c4fcb7c63384a9f1ec0d95201552569969f0030a1ed7c54bd3d5f1ae",
    "test_dsl_recent128.py": "51c060add9de60517f7ca13d65f8e8e6ac5f69b6cf123977b2c2d84fa8f76287",
    "README.md": "70b611b0cf7e66856ee15ce2e49078bdb4c9ff01f4aa69f7e457ff5a34cb4dba",
    "EXECUTION_TASK.md": "055636503c43eb7823bee06cbae465fac16c7959d16aeba7c64b825881fd6a01"
}
PREFIX = "experimental/dsl_recent128"


def download(run, insecure=False):
    if len(PAYLOAD_COMMIT) != 40 or len(FILES) != 4:
        raise RuntimeError("Launcher payload is not pinned")
    context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
    records = {}
    for name, expected in FILES.items():
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{PAYLOAD_COMMIT}/{PREFIX}/{name}"
        with urllib.request.urlopen(url, context=context, timeout=120) as response:
            data = response.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("Payload size/SHA256 mismatch: " + name)
        with (run / name).open("xb") as handle:
            handle.write(data)
        records[name] = dict(url=url, sha256=expected, bytes=len(data))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--work-dir", default="/tmp/dsl-recent128")
    args, forwarded = parser.parse_known_args(argv)
    if any(arg == "--output" or arg.startswith("--output=") for arg in forwarded):
        parser.error("Use --work-dir; launcher creates an isolated output directory")
    parent = Path(args.work_dir).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="dsl-recent128-", dir=parent))
    report = dict(status="RUNNING", payload_commit=PAYLOAD_COMMIT,
                  python=sys.executable, arguments=forwarded, run=str(run), stage="download")
    print(f"Python: {sys.executable}\nRun directory: {run}", flush=True)
    try:
        report["files"] = download(run, args.insecure)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        ninja = Path("/tmp/cloudops/bin/ninja")
        if not shutil.which("ninja", path=env.get("PATH", os.defpath)) and ninja.is_file():
            env["PATH"] = str(ninja.parent) + os.pathsep + env.get("PATH", os.defpath)
        env.setdefault("TORCH_EXTENSIONS_DIR", str(run / "torch_extensions"))
        report["stage"] = "test"
        command = [sys.executable, str(run / "test_dsl_recent128.py"), *forwarded,
                   "--output", str(run / "results")]
        with (run / "run.log").open("x", encoding="utf-8") as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", env=env) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
        report.update(returncode=code, status="COMPLETED" if code == 0 else "FAILED_OR_BLOCKED")
        return code
    except Exception:
        report.update(status="ERROR", traceback=traceback.format_exc())
        print(report["traceback"], file=sys.stderr, flush=True)
        return 1
    finally:
        (run / "launcher.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Summary: {run}/results/summary.txt\nDependency/build task: {run}/EXECUTION_TASK.md", flush=True)


if __name__ == "__main__":
    sys.exit(main())
