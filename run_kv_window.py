#!/usr/bin/env python3
"""Fetch a pinned main-KV precision-window experiment; use current Python."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import traceback
import urllib.request

PAYLOAD_COMMIT = "d2fcd69362788775b59a61b011cb8de712760d41"
REPOSITORY = "Qiming-zhang-rondo/kimi-k3-a5-c8"
FILES = {
    "test_kv_window_a5.py": "2473340ac4222331a46ca4c6d3ec586363baf4a3458eeb160cc06d2d5efcc790",
    "kv_window_backend.py": "20c33c19612da73484ea7c57e05b2a78eba68e7bc886db6daf0997976a530bc4",
    "kv_window_common.py": "7357ca29ee22f71445b1d9a3d5eddf8c2d6c0a0dab134b0e2e4d42c36e3dd724",
    "kv_window_capture.py": "b872f9a378a22a27c3ff970426c33718d3c20958724ee018d383b4bb51216bb7",
    "KV_WINDOW_README.txt": "944065efa08858fb8728fb4a5a08f2bb3cde2606d89c78581179ec420e40d309"
}


def download_payload(directory, insecure=False):
    if len(PAYLOAD_COMMIT) != 40 or len(FILES) < 4:
        raise RuntimeError("Launcher has not been pinned; refusing to run")
    context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
    records = {}
    for filename, expected in FILES.items():
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{PAYLOAD_COMMIT}/kv-window/{filename}"
        with urllib.request.urlopen(url, context=context, timeout=120) as response:
            body = response.read(2*1024*1024+1)
        if len(body) > 2*1024*1024 or hashlib.sha256(body).hexdigest() != expected:
            raise RuntimeError(f"Invalid size or SHA256: {filename}; refusing to execute")
        with (directory/filename).open("xb") as target:
            target.write(body)
        records[filename] = {"url": url, "sha256": expected, "bytes": len(body)}
    return records


def main(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--insecure", action="store_true")
    p.add_argument("--work-dir", default="/mnt")
    args, forwarded = p.parse_known_args(argv)
    if any(x == "--output" or x.startswith("--output=") for x in forwarded):
        p.error("Launcher creates an isolated results directory; use --work-dir for its parent")
    root = Path(args.work_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="kv-window-", dir=root))
    print(f"Python: {sys.executable}\nTest, log and results: {run}", flush=True)
    report = dict(python=sys.executable, payload_commit=PAYLOAD_COMMIT, arguments=forwarded,
                  status="RUNNING", stage="download")
    try:
        report["files"] = download_payload(run, args.insecure)
        report["stage"] = "test"
        print("SHA256 verified. No installs, upgrades or model changes.", flush=True)
        command = [sys.executable, str(run/"test_kv_window_a5.py"), *forwarded,
                   "--output", str(run/"results")]
        with (run/"run.log").open("x", encoding="utf-8") as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1,
                                  env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
        report.update(returncode=code, status="COMPLETED" if code == 0 else "ERROR_OR_CONTROL_FAILED")
        return code
    except Exception:
        report.update(status="ERROR", traceback=traceback.format_exc())
        print(report["traceback"], file=sys.stderr, flush=True)
        return 1
    finally:
        (run/"launcher.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Full log: {run}/run.log", flush=True)


if __name__ == "__main__":
    sys.exit(main())
