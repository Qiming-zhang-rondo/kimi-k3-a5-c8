#!/usr/bin/env python3
"""Fetch the pinned native two-pass C4/C8 experiment; use current Python."""
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

PAYLOAD_COMMIT = "b0e252ee7de7b77593a68f1c76fd03d46ff68289"
REPOSITORY = "Qiming-zhang-rondo/kimi-k3-a5-c8"
FILES = {
    "NATIVE_TWOPASS_README.txt": "b65b26d0223b9acf0d2bab7c9b2aa3640e809ae560f9bb044690e25a22e415d3",
    "inspect_kv_fusion_capability.py": "d0f7bb3052a29380b363e4d177a505299c4d5a91f3d607e2e2f66872a462e6f9",
    "kv_c4_scale_backend.py": "5002af47b6f7394f84f6b5d2d65eeeaf4de8097df279f9cf0328b252f87149a5",
    "kv_hot_backend.py": "90a9a005458ebb2c8b837f103641b3897a8c99d29fa4f22ae4254dc6d9cb61fa",
    "kv_native_merge.py": "55c5bad52891dd539e9e5a1acde942aca7decf4019e20d658a08b12b7ad4468a",
    "kv_native_partition.py": "1b83122344bd1fd2ff07544f9fcf8678e0edd6e4f37e583a71940a32b363d7db",
    "kv_native_store.py": "94e33ae49f72bfcf5efc0c1914e5ca9f312e6afce1ed9442b22decea56ef85d2",
    "kv_native_twopass.py": "2f7929472abe4b5257f7d8a274539474be0bb28a9064f7771c96e457e31b669c",
    "kv_window_backend.py": "20c33c19612da73484ea7c57e05b2a78eba68e7bc886db6daf0997976a530bc4",
    "kv_window_capture.py": "b872f9a378a22a27c3ff970426c33718d3c20958724ee018d383b4bb51216bb7",
    "kv_window_common.py": "7357ca29ee22f71445b1d9a3d5eddf8c2d6c0a0dab134b0e2e4d42c36e3dd724",
    "test_kv_hot_fused_a5.py": "a64183fa77fbdaa8085c2ff03b90723de6e9f31deffdfb2b91fb9a42ce22a2e2",
    "test_kv_native_twopass_a5.py": "0b209ee57cf1e355ce2762ab6300717e0eeb1451ceea806b251daa32859f66be"
}


def download_payload(directory, insecure=False):
    if len(PAYLOAD_COMMIT) != 40 or len(FILES) != 13:
        raise RuntimeError("Launcher has not been pinned; refusing to run")
    context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
    records = {}
    for filename, expected in FILES.items():
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{PAYLOAD_COMMIT}/kv-native-twopass/hotkv/{filename}"
        with urllib.request.urlopen(url, context=context, timeout=120) as response:
            body = response.read(2*1024*1024+1)
        if len(body) > 2*1024*1024 or hashlib.sha256(body).hexdigest() != expected:
            raise RuntimeError(f"Invalid size or SHA256: {filename}; refusing to execute")
        with (directory/filename).open("xb") as target:
            target.write(body)
        records[filename] = {"url": url, "sha256": expected, "bytes": len(body)}
    return records


def test_environment(run, cloudops_root="/tmp/cloudops", torch_extensions_dir=None):
    """Configure the child process only; preserve existing tool/proxy choices.

    A reported Cloud deployment keeps ninja in bin. A new run gets its own JIT
    cache unless the caller explicitly supplies an existing location. This
    does not source a vendor environment, install dependencies or clear caches.
    """
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    cloud_root = Path(cloudops_root).expanduser().resolve()
    ninja = shutil.which("ninja", path=env.get("PATH", os.defpath))
    ninja_source = "existing_PATH" if ninja else "not_found"
    candidate = cloud_root/"bin"/"ninja"
    if ninja is None and candidate.is_file() and os.access(candidate, os.X_OK):
        previous = env.get("PATH", os.defpath)
        env["PATH"] = str(candidate.parent) + (os.pathsep+previous if previous else "")
        ninja, ninja_source = str(candidate), "cloudops_bin"
    if torch_extensions_dir is not None:
        cache, cache_source = Path(torch_extensions_dir).expanduser().resolve(), "launcher_argument"
        env["TORCH_EXTENSIONS_DIR"] = str(cache)
    elif env.get("TORCH_EXTENSIONS_DIR"):
        cache, cache_source = Path(env["TORCH_EXTENSIONS_DIR"]).expanduser(), "existing_environment"
    else:
        cache = Path(run)/"torch_extensions"
        cache_source = "isolated_run"
        env["TORCH_EXTENSIONS_DIR"] = str(cache)
    cache.mkdir(parents=True, exist_ok=True)
    return env, dict(ninja=ninja, ninja_source=ninja_source,
                     torch_extensions_dir=env["TORCH_EXTENSIONS_DIR"], cache_source=cache_source)


def main(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--insecure", action="store_true")
    p.add_argument("--work-dir", default="/tmp/kv-native-twopass")
    p.add_argument("--cloudops-root", default="/tmp/cloudops")
    p.add_argument("--torch-extensions-dir", default=None)
    args, forwarded = p.parse_known_args(argv)
    if any(x == "--output" or x.startswith("--output=") for x in forwarded):
        p.error("Launcher creates an isolated results directory; use --work-dir for its parent")
    root = Path(args.work_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="kv-native-twopass-", dir=root))
    print(f"Python: {sys.executable}\nTest, log and results: {run}", flush=True)
    report = dict(python=sys.executable, payload_commit=PAYLOAD_COMMIT, arguments=forwarded,
                  status="RUNNING", stage="download")
    try:
        report["files"] = download_payload(run, args.insecure)
        report["stage"] = "environment"
        child_env, report["test_environment"] = test_environment(
            run, args.cloudops_root, args.torch_extensions_dir)
        print(f"Ninja: {report['test_environment']['ninja'] or 'not found in PATH'}\n"
              f"JIT cache: {report['test_environment']['torch_extensions_dir']}", flush=True)
        report["stage"] = "test"
        print("SHA256 verified. No installs, upgrades or model changes.", flush=True)
        command = [sys.executable, str(run/"test_kv_native_twopass_a5.py"), *forwarded,
                   "--output", str(run/"results")]
        with (run/"run.log").open("x", encoding="utf-8") as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1,
                                  env=child_env) as process:
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
        if (run/"results"/"brief.txt").is_file():
            print(f"Short summary: {run}/results/brief.txt", flush=True)


if __name__ == "__main__":
    sys.exit(main())
