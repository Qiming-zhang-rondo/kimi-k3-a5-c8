#!/usr/bin/env python3
"""Download a pinned, SHA256-verified single-indexer accuracy test and run it.

Uses the current Python environment. Does not install or upgrade any package.
All arguments except --insecure and --work-dir are forwarded to the test.
"""
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


VERSION = "qli-bf16-launcher-v1"
PAYLOAD_COMMIT = "834fbf9d05a50fef8e2c4e7be478af8c936b3f60"
REPOSITORY = "Qiming-zhang-rondo/kimi-k3-a5-c8"
FILES = {
    "test_qli_bf16_accuracy.py": "92d01e0e67c2558c1b776ea86b8f121cf888aa7cac304f781d71aae672bc9532",
    "qli_bf16_backend.py": "86d9995e12248e46e35d537b6f66d41792e59955dc3f5fed041c866ea68f829a",
    "qli_topk_metrics.py": "1b476a429ca4f39c63cd0e533c4130f5831576c392a56f826542cc3b1779fe04",
}


def download_payload(directory, insecure=False):
    context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
    records = {}
    for filename, expected in FILES.items():
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{PAYLOAD_COMMIT}/qli-bf16/{filename}"
        with urllib.request.urlopen(url, context=context, timeout=120) as response:
            body = response.read(2*1024*1024+1)
        if len(body) > 2*1024*1024:
            raise RuntimeError(f"Payload too large: {filename}")
        actual = hashlib.sha256(body).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {filename}; refusing to execute")
        with (directory/filename).open("xb") as target:
            target.write(body)
        records[filename] = {"url": url, "sha256": actual, "bytes": len(body)}
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--work-dir", default="/mnt")
    args, forwarded = parser.parse_known_args(argv)
    if any(arg == "--output" or arg.startswith("--output=") for arg in forwarded):
        parser.error("The launcher creates a fresh results directory. Use --work-dir to choose its parent.")
    root = Path(args.work_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="qli-bf16-", dir=root))
    print(f"{VERSION}; Python: {sys.executable}\nLogs and results: {run}", flush=True)
    report = {"launcher_version": VERSION, "python": sys.executable, "payload_commit": PAYLOAD_COMMIT,
              "arguments": forwarded, "stage": "download", "status": "RUNNING"}
    try:
        report["files"] = download_payload(run, args.insecure)
        print("Verified all test scripts. No package installs or CANN changes.", flush=True)
        command = [sys.executable, str(run/"test_qli_bf16_accuracy.py"), *forwarded,
                   "--output", str(run/"results")]
        report["stage"] = "test"
        with (run/"run.log").open("x", encoding="utf-8") as logfile:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1,
                                  env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    logfile.write(line)
                    logfile.flush()
                code = process.wait()
        report["returncode"] = code
        report["status"] = "COMPLETED" if code == 0 else "ERROR_OR_INVALID"
        print(f"Full log: {run}/run.log", flush=True)
        return code
    except Exception:
        report["status"] = "ERROR"
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        (run/"launcher.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
