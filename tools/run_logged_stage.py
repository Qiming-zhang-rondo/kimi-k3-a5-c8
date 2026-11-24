#!/usr/bin/env python3
"""Run one explicit command and preserve its exit status and output as evidence.

No shell, environment collection, installation, or implicit follow-up commands.
An existing DEVICE_FAULT.json blocks accuracy/performance commands in this root.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


STAGES = ("baseline_build", "candidate_build", "accuracy", "performance")
DEVICE_STAGES = ("accuracy", "performance")
FAULT_SIGNATURES = ("507015", "ACL_ERROR_RT_AICORE_EXCEPTION", "GM invalid",
                    "The address for scalar to access GM is invalid")


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def save_json(path, data):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run_stage(stage, work_dir, command, cwd=None, console=None):
    """Return the recorded status; only a zero exit with no detected fault passes."""
    if stage not in STAGES or not command:
        raise ValueError("A supported stage and a nonempty command are required")
    if not all(isinstance(arg, str) for arg in command):
        raise ValueError("Command must be a list of strings")
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=stage + "_", dir=root))
    log_path = run_dir / "stage.log"
    marker_path = root / "DEVICE_FAULT.json"
    report = {
        "stage": stage, "argv": list(command),
        "cwd": str(Path(cwd or os.getcwd()).resolve()), "pid": None,
        "started_utc": utc_now(), "ended_utc": None, "returncode": None,
        "status": "ERROR", "run_dir": str(run_dir), "log": str(log_path),
        "device_fault": False, "blocked": False,
    }
    digest = hashlib.sha256()
    tail = b""
    matches = set()
    process = None
    if console is None:
        console = getattr(sys.stdout, "buffer", sys.stdout)

    with log_path.open("wb") as logfile:
        def record(chunk):
            nonlocal tail
            logfile.write(chunk)
            logfile.flush()
            digest.update(chunk)
            try:
                console.write(chunk)
            except TypeError:
                console.write(chunk.decode("utf-8", errors="replace"))
            console.flush()
            scan = (tail + chunk).lower()
            for signature in FAULT_SIGNATURES:
                if signature.lower().encode("ascii") in scan:
                    matches.add(signature)
            tail = scan[-(max(map(len, FAULT_SIGNATURES)) - 1):]
            if matches:
                report["device_fault"] = True
                marker = {
                    "detected_utc": utc_now(), "stage": stage,
                    "run_dir": str(run_dir), "log": str(log_path),
                    "signatures": sorted(matches),
                }
                # Preserve the first fault. Existence alone is sufficient to block.
                try:
                    with marker_path.open("x", encoding="utf-8") as stream:
                        json.dump(marker, stream, ensure_ascii=False, indent=2)
                        stream.write("\n")
                except FileExistsError:
                    pass

        if stage in DEVICE_STAGES and marker_path.exists():
            report["blocked"] = True
            report["error"] = "Existing DEVICE_FAULT.json blocks this device stage"
            report["device_fault_marker"] = str(marker_path)
            record(("NOT EXECUTED: " + report["error"] + "\n").encode())
        else:
            try:
                process = subprocess.Popen(
                    command, cwd=report["cwd"], shell=False,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
                report["pid"] = process.pid
                with process.stdout:
                    while True:
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if not chunk:
                            break
                        record(chunk)
                report["returncode"] = process.wait()
                if report["returncode"] == 0 and not report["device_fault"]:
                    report["status"] = "PASS"
            except (OSError, ValueError, KeyboardInterrupt) as error:
                report["error"] = type(error).__name__ + ": " + str(error)
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                    report["returncode"] = process.wait()
                record(("RUNNER ERROR: " + report["error"] + "\n").encode())
        if matches:
            report["device_fault_signatures"] = sorted(matches)
            report["device_fault_marker"] = str(marker_path)
    report["ended_utc"] = utc_now()
    report["log_sha256"] = digest.hexdigest()
    save_json(run_dir / "status.json", report)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--work-dir", required=True, help="Isolated evidence root")
    parser.add_argument("--cwd", help="Command working directory (default: current directory)")
    prefix = argv[:argv.index("--")] if "--" in argv else argv
    if "--help" in prefix or "-h" in prefix:
        parser.parse_args(["--help"])
    if "--" not in argv:
        parser.error("Supply an explicit command after --")
    separator = argv.index("--")
    args = parser.parse_args(argv[:separator])
    command = argv[separator + 1:]
    if not command:
        parser.error("Command after -- must not be empty")
    report = run_stage(args.stage, args.work_dir, command, args.cwd)
    print("\n{}: {}".format(report["status"], Path(report["run_dir"]) / "status.json"))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
