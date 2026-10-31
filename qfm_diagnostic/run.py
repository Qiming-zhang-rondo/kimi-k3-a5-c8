"""Isolated, fail-closed supervisor for the standalone QFM Stage 1 probes.

Examples (the card must be an idle physical NPU you are authorized to use)::

    python run.py --card 3 --plan-only
    python run.py --card 3 --suite inspect
    python run.py --card 3 --suite preflight
    python run.py --card 3 --suite all --timeout-seconds 600

The supervisor imports only Python's standard library.  Each prerequisite or
test gets a fresh process, a separate log, and the same single visible physical
card; the worker addresses it as logical card 0.  No packages are installed and
no vLLM files are patched.  The runner does not verify that the selected card is
idle or authorized.  Plan-only performs no worker launches or NPU operations.
The default all-suite runs preflight, small, graph, dcp, then performance.
Inspect is dependency inventory only and does not require a built QFM operator.
The separate optional prolog suite requires the DSL/reference source; neither
inspect, preflight nor all requires it.  None of these claims MLAPrologV4's fused
FP8 path is validated.  The memory estimate is an admission guard, not a device
memory reservation, hard limit, or guarantee against unknown native workspaces.

A successful process exit is NOT enough: the worker must emit exactly one
matching JSON suite_complete event with pass=true.  Failures, missing results,
and timeouts stop the run immediately.  Killing the host process group cannot
guarantee that a wedged device has recovered; inspect it before any later run.
"""

import argparse
import codecs
import datetime
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SUITES = (
    "inspect",
    "preflight",
    "small",
    "graph",
    "dcp",
    "prolog",
    "performance",
    "all",
)
HEADS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
SCRIPT_DIR = Path(__file__).resolve().parent
MIB = 2**20
DEVICE_WARNING = (
    "Host process-group termination does not guarantee that NPU work stopped; "
    "the device may still be hung. Inspect the selected card before another run."
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--card",
        type=int,
        required=True,
        help="authorized idle PHYSICAL NPU index; workers see logical card 0",
    )
    parser.add_argument("--suite", choices=SUITES, default="all")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600,
        help="maximum wall time for EACH isolated worker (default: 600)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
        help="parent directory; a unique run subdirectory is always created",
    )
    parser.add_argument(
        "--memory-limit-mib",
        type=float,
        default=4096,
        help="reject estimated device residency above this budget; not a hard limit",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=8192,
        help="performance-suite query chunk length (default: 8192)",
    )
    parser.add_argument(
        "--kv-length",
        type=int,
        default=131072,
        help="performance-suite final logical KV length (default: 131072)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        choices=HEADS,
        default=12,
        help="performance-suite local query heads (default: 12)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="performance-suite timing repetitions (default: 10)",
    )
    parser.add_argument(
        "--prolog-source",
        type=Path,
        help="optional pinned prolog reference source; otherwise worker/env default",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write the launch/memory plan without importing NPU modules or launching workers",
    )
    args = parser.parse_args(argv)
    if args.card < 0:
        parser.error("--card must be a nonnegative physical NPU index")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    if not math.isfinite(args.memory_limit_mib) or args.memory_limit_mib <= 0:
        parser.error("--memory-limit-mib must be finite and positive")
    if not 1 <= args.queries <= args.kv_length:
        parser.error("need 1 <= --queries <= --kv-length")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    return args


def child_environment(physical_card, environ=None):
    """Pin one physical card, independent of an inherited logical device ID."""
    result = dict(os.environ if environ is None else environ)
    result["ASCEND_RT_VISIBLE_DEVICES"] = str(physical_card)
    result.pop("ASCEND_DEVICE_ID", None)
    result["PYTHONUNBUFFERED"] = "1"
    return result


def build_jobs(args, script_dir=SCRIPT_DIR):
    """Inspect needs no QFM; only an explicit prolog request needs the DSL."""
    if args.suite == "all":
        names = ["preflight", "small", "graph", "dcp", "performance"]
    elif args.suite in ("inspect", "preflight"):
        names = [args.suite]
    elif args.suite == "prolog":
        names = ["prolog_preflight", "prolog"]
    else:
        names = ["preflight", args.suite]
    jobs = []
    for name in names:
        prolog = name.startswith("prolog")
        worker = Path(script_dir) / ("prolog_probe.py" if prolog else "qfm_probe.py")
        command = [sys.executable, "-u", str(worker), "--card", "0"]
        if prolog:
            if name == "prolog_preflight":
                command += ["--preflight"]
            if args.prolog_source is not None:
                command += ["--prolog-source", str(args.prolog_source.resolve())]
        else:
            command += ["--suite", name]
            if name == "performance":
                command += [
                    "--queries",
                    str(args.queries),
                    "--kv-length",
                    str(args.kv_length),
                    "--heads",
                    str(args.heads),
                    "--iterations",
                    str(args.iterations),
                ]
        jobs.append({"suite": name, "command": command, "worker": str(worker)})
    return jobs


def estimate_qfm_memory(queries, kv_length, heads):
    """Approximate synthetic probe residency, not a native workspace bound.

    The probe retains test-only original BF16 inputs and runs an FIA comparison;
    those are included here, not proposed as a production second KV cache.
    Quantization temporaries are conservatively added even when lifetimes may
    not overlap.  Allocator/runtime/vendor scratch is only a rough reserve.
    """
    pages = (kv_length + 127) // 128
    virtual_sequences = (queries + 15) // 16
    padded_heads = 1 << (heads - 1).bit_length()
    components = {
        "fp8_cache": pages * 128 * 576,
        "test_kv_bf16_device_input": kv_length * 576 * 2,
        "query_bf16": queries * heads * 576 * 2,
        "query_fp8_and_scales": queries * heads * (576 + 4),
        "qfm_outputs_and_lse": queries * heads * (512 * 2 + 4) * 2,
        "quantization_temporaries": max(kv_length * 576, queries * heads * 576) * 9,
        "virtual_block_tables": (virtual_sequences + 1) * pages * 4,
        "test_fia_bf16_inputs": queries * padded_heads * 576 * 4 + kv_length * 576 * 2,
        "test_fia_outputs": queries * padded_heads * 512 * 4,
        "masks": 2048 * 2048 * 2,
    }
    subtotal = sum(components.values())
    components["allocation_margin"] = (subtotal + 3) // 4
    components["runtime_workspace_reserve"] = 512 * MIB
    return {"estimated_bytes": sum(components.values()), "components_bytes": components}


def memory_precheck(args, jobs):
    estimates = {}
    for job in jobs:
        name = job["suite"]
        if name == "performance":
            estimates[name] = estimate_qfm_memory(
                args.queries, args.kv_length, args.heads
            )
        elif name in ("small", "graph", "dcp"):
            # Upper envelope of the fixed small cases, not their simultaneous sum.
            estimates[name] = estimate_qfm_memory(48, 1024, 48)
        elif name == "prolog":
            estimates[name] = {
                "estimated_bytes": 768 * MIB,
                "components_bytes": {
                    "fixed_prolog_inputs_and_margin": 256 * MIB,
                    "runtime_workspace_reserve": 512 * MIB,
                },
            }
        else:
            estimates[name] = {
                "estimated_bytes": 512 * MIB,
                "components_bytes": {"runtime_workspace_reserve": 512 * MIB},
            }
    # Workers are sequential and isolated: compare the largest, do not sum.
    peak = max((entry["estimated_bytes"] for entry in estimates.values()), default=0)
    limit = args.memory_limit_mib * MIB
    return {
        "pass": peak <= limit,
        "estimated_peak_mib": peak / MIB,
        "limit_mib": args.memory_limit_mib,
        "per_worker": estimates,
        "caveat": (
            "Approximate device-memory admission check, not a reservation or hard cap. "
            "Unknown native/compiler workspaces, allocator retention and device availability "
            "can invalidate it. Prolog golden-reference host RAM may exceed 1 GiB."
        ),
    }


def _stop_process_group(process, grace_seconds=0.5):
    """Bound every wait, including a worker that survives SIGKILL."""

    def send(sig):
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except (ProcessLookupError, PermissionError):
            pass

    send(signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # The group can still contain descendants after its leader has exited.
    send(signal.SIGKILL)
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        return False
    return True


def run_child(command, suite, log_path, *, timeout_seconds, env, stdout=None):
    """Stream one worker and accept only an explicit, matching completion."""
    stdout = sys.stdout if stdout is None else stdout
    log_path = Path(log_path)
    started = time.monotonic()
    completions = []
    negative_events = []
    line_buffer = ""
    overlong_line = False
    reason = None
    returncode = None
    warning = None
    process = None

    def parse_line(line):
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(event, dict):
            return
        if event.get("event") == "suite_complete":
            completions.append(event)
        if event.get("pass") is False:
            negative_events.append(event)

    with log_path.open("w", encoding="utf-8") as log:

        def stream_text(text):
            nonlocal line_buffer, overlong_line
            log.write(text)
            log.flush()
            stdout.write(text)
            stdout.flush()
            for piece in text.splitlines(keepends=True):
                line_buffer += piece
                if len(line_buffer) > 1024 * 1024:
                    # Keep logs intact but bound supervisor parser memory.
                    line_buffer = ""
                    overlong_line = True
                if piece.endswith(("\n", "\r")):
                    if not overlong_line:
                        parse_line(line_buffer)
                    line_buffer = ""
                    overlong_line = False

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=(os.name == "posix"),
                bufsize=0,
            )
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            deadline = started + timeout_seconds
            exited_at = None
            with selectors.DefaultSelector() as selector:
                os.set_blocking(process.stdout.fileno(), False)
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    now = time.monotonic()
                    returncode = process.poll()
                    if returncode is not None and not selector.get_map():
                        break
                    if now >= deadline:
                        reason = "timeout"
                        warning = DEVICE_WARNING
                        _stop_process_group(process)
                        returncode = process.poll()
                        break
                    if returncode is not None:
                        exited_at = now if exited_at is None else exited_at
                        if now - exited_at >= 1.0:
                            reason = "stdout_held_open_after_worker_exit"
                            warning = DEVICE_WARNING
                            _stop_process_group(process)
                            break
                    for key, _ in selector.select(min(0.1, max(0.0, deadline - now))):
                        try:
                            data = os.read(key.fileobj.fileno(), 65536)
                        except BlockingIOError:
                            continue
                        if data:
                            stream_text(decoder.decode(data))
                        else:
                            selector.unregister(key.fileobj)
                stream_text(decoder.decode(b"", final=True))
                if line_buffer and not overlong_line:
                    parse_line(line_buffer)
        except KeyboardInterrupt:
            reason = "interrupted"
            warning = DEVICE_WARNING
            if process is not None:
                _stop_process_group(process)
                returncode = process.poll()
        except (OSError, ValueError) as exc:
            reason = f"supervisor_or_launch_error: {type(exc).__name__}: {exc}"
            if process is not None:
                warning = DEVICE_WARNING
                _stop_process_group(process)
                returncode = process.poll()
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()

        if reason is not None:
            status = (
                "failed"
                if reason.startswith("supervisor_or_launch_error:")
                else "incomplete"
            )
        elif returncode != 0:
            status, reason = "failed", "nonzero_exit"
        elif negative_events:
            status, reason = "failed", "worker_reported_failure"
        elif len(completions) != 1:
            status, reason = "incomplete", "missing_or_ambiguous_suite_complete"
        elif (
            completions[0].get("suite") != suite
            or completions[0].get("pass") is not True
        ):
            status, reason = "incomplete", "mismatched_or_invalid_suite_complete"
        else:
            status, reason = "passed", "explicit_completion_and_zero_exit"
        if warning:
            message = f"\nWARNING: {warning}\n"
            log.write(message)
            log.flush()
            stdout.write(message)
            stdout.flush()

    return {
        "suite": suite,
        "status": status,
        "reason": reason,
        "command": list(command),
        "log_path": str(log_path),
        "returncode": returncode,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "completion": completions[0] if len(completions) == 1 else None,
        "warning": warning,
    }


def _write_summary(path, summary):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    output_parent = args.output_dir.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(tempfile.mkdtemp(prefix=f"qfm-{stamp}-", dir=output_parent))
    jobs = build_jobs(args)
    admission = memory_precheck(args, jobs)
    summary = {
        "status": "incomplete",
        "all_passed": False,
        "requested_suite": args.suite,
        "plan_only": args.plan_only,
        "physical_card": args.card,
        "worker_logical_card": 0,
        "child_environment": {
            "ASCEND_RT_VISIBLE_DEVICES": str(args.card),
            "ASCEND_DEVICE_ID": None,
        },
        "timeout_seconds": args.timeout_seconds,
        "output_dir": str(output_dir),
        "memory_precheck": admission,
        "fused_prolog_validated": False,
        "prolog_status": "not_run"
        if args.suite == "prolog"
        else "prolog_not_requested",
        "native_computation_validated": False,
        "scope": (
            "Standalone synthetic probes only; no VA integration or distributed HCCL validation. "
            "Inspect/preflight inventory does not validate native computation or fused MLAPrologV4. "
            "Optional DSL prolog results are separate from the MLAPrologV4 path."
        ),
        "results": [
            {
                **job,
                "status": "incomplete",
                "reason": "not_run",
                "log_path": str(output_dir / f"{job['suite']}.log"),
            }
            for job in jobs
        ],
    }
    summary_path = output_dir / "summary.json"
    if not admission["pass"]:
        summary["reason"] = "estimated_memory_exceeds_budget"
        for result in summary["results"]:
            result["reason"] = "memory_precheck_rejected"
    elif args.plan_only:
        summary["reason"] = "plan_only_no_hardware_tests"
        for result in summary["results"]:
            result["reason"] = "plan_only_no_hardware_tests"
    else:
        env = child_environment(args.card)
        for index, job in enumerate(jobs):
            print(
                json.dumps(
                    {
                        "event": "worker_begin",
                        "suite": job["suite"],
                        "physical_card": args.card,
                        "log": summary["results"][index]["log_path"],
                    }
                ),
                flush=True,
            )
            result = run_child(
                job["command"],
                job["suite"],
                summary["results"][index]["log_path"],
                timeout_seconds=args.timeout_seconds,
                env=env,
            )
            summary["results"][index] = result
            _write_summary(summary_path, summary)
            if result["status"] != "passed":
                for remaining in summary["results"][index + 1 :]:
                    remaining["reason"] = (
                        f"not_run_after_{job['suite']}_{result['status']}"
                    )
                break
        statuses = [result["status"] for result in summary["results"]]
        summary["status"] = (
            "failed"
            if "failed" in statuses
            else "passed"
            if statuses and all(status == "passed" for status in statuses)
            else "incomplete"
        )
        summary["all_passed"] = summary["status"] == "passed"
        summary["native_computation_validated"] = summary[
            "all_passed"
        ] and args.suite not in ("inspect", "preflight")
        if args.suite == "prolog":
            summary["prolog_status"] = summary["status"]
        summary["reason"] = (
            "all_requested_workers_passed"
            if summary["all_passed"]
            else "stopped_on_first_failed_or_incomplete_worker"
        )
    _write_summary(summary_path, summary)
    print(
        json.dumps(
            {
                "event": "run_complete",
                "status": summary["status"],
                "all_passed": summary["all_passed"],
                "summary": str(summary_path),
                "reason": summary["reason"],
            }
        ),
        flush=True,
    )
    return 0 if summary["all_passed"] or (args.plan_only and admission["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
