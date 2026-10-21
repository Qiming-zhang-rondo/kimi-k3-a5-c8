#!/usr/bin/env bash
# A5 container acceptance smoke test. No dependencies beyond the active Python.
# Example: bash run-k3-c8-model-matrix.sh --native-passed --tp '8'
# Broader matrix: add --tp '2 4 8'. Stop existing services yourself first.
set -euo pipefail
exec python - "$@" <<'PY'
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


PLUGINS = (
    "ascend_vllm,ascend_model,ascend_kv_connector,ascend_model_loader,"
    "ascend_service_profiling,modelarts_general_plugin_patch"
)
STAGES = (1, 3, 8, 1)


class UnsafeCleanup(RuntimeError):
    pass


def emit(message):
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="K3 A5 C8 ordinary/MLAPO x eager/FULL_DECODE_ONLY model smoke matrix. "
        "Run inside the patched VA container after native tests PASS. "
        "No profiling, weight changes, precision claims, pkill, or automatic SIGKILL."
    )
    parser.add_argument("--native-passed", action="store_true", required=True,
                        help="Required acknowledgement: native C8 and graph tests passed on this image.")
    parser.add_argument("--tp", default="8", help="Space-separated TP sizes: '8' or '2 4 8'.")
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7", help="Allocated NPU devices; first TP devices are used.")
    parser.add_argument("--model", default="/mnt/paas/models/Kimi-K3-5L")
    parser.add_argument("--nic", default="enp39s0f3")
    parser.add_argument("--local-ip", help="Default: IPv4 address of --nic, not an arbitrary hostname address.")
    parser.add_argument("--port", type=int, default=7700)
    parser.add_argument("--rpc-port", type=int, default=13397)
    parser.add_argument("--ready-timeout", type=int, default=7200)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--stop-timeout", type=int, default=60)
    parser.add_argument("--memory-utilization", type=float, default=0.90)
    parser.add_argument("--work-dir", default="/myWorkspace")
    parser.add_argument("--log-dir", default="/myWorkspace/log/k3_c8_matrix")
    parser.add_argument("--plugins", default=PLUGINS)
    parser.add_argument("--dry-run", action="store_true", help="Only print commands; do not create files or start servers.")
    args = parser.parse_args()
    try:
        args.tp_sizes = [int(value) for value in args.tp.split()]
        args.device_ids = [int(value) for value in args.devices.split(",")]
    except ValueError:
        parser.error("--tp and --devices must contain integers.")
    if not args.tp_sizes or any(tp not in (2, 4, 8) for tp in args.tp_sizes):
        parser.error("This acceptance matrix supports TP 2, 4, and 8 only.")
    if len(set(args.tp_sizes)) != len(args.tp_sizes):
        parser.error("Do not repeat TP sizes.")
    if len(set(args.device_ids)) != len(args.device_ids) or min(args.device_ids, default=-1) < 0:
        parser.error("--devices must be nonnegative, unique device IDs.")
    if len(args.device_ids) < max(args.tp_sizes):
        parser.error("Not enough allocated --devices for the largest requested TP.")
    if not 0 < args.memory_utilization < 1:
        parser.error("--memory-utilization must be between 0 and 1.")
    if not (0 < args.port < 65536 and 0 < args.rpc_port < 65536) or args.port == args.rpc_port:
        parser.error("API/RPC ports must be distinct valid ports.")
    if min(args.ready_timeout, args.request_timeout, args.stop_timeout) <= 0:
        parser.error("Timeouts must be positive seconds.")
    return args


def interface_ip(name):
    if len(name.encode()) > 15:
        raise ValueError("NIC name is longer than Linux IFNAMSIZ.")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            info = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", name.encode()))
        except OSError as exc:
            raise RuntimeError(f"Cannot get IPv4 for NIC {name}; pass --local-ip explicitly: {exc}") from exc
    return socket.inet_ntoa(info[20:24])


def check_ports(args):
    for port in (args.port, args.rpc_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError as exc:
                raise UnsafeCleanup(f"Port {port} is occupied/unavailable. Existing services were NOT stopped: {exc}") from exc


def build_case(args, tp, mlapo, graph_mode, ip):
    env = os.environ.copy()
    env.update({
        "MODEL_PATH": args.model, "LOCAL_IP": ip, "NIC_NAME": args.nic,
        "HCCL_IF_IP": ip, "GLOO_SOCKET_IFNAME": args.nic,
        "TP_SOCKET_IFNAME": args.nic, "HCCL_SOCKET_IFNAME": args.nic,
        "VLLM_ENGINE_READY_TIMEOUT_S": str(args.ready_timeout),
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "OMP_PROC_BIND": "false", "OMP_NUM_THREADS": "1", "TASK_QUEUE_ENABLE": "1",
        "HCCL_BUFFSIZE": "800", "HCCL_OP_EXPANSION_MODE": "AIV",
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, args.device_ids[:tp])),
        "VLLM_USE_FASTOKENS": "1", "ENABLE_ASYNC_REJECT_SAMPLE": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_PLUGINS": args.plugins,
        "KDA_CLOUD_DEBUG": "1",
    })
    additional = {
        "enable_flashcomm1": True, "enable_shared_expert_dp": False,
        "enable_kimi_k3_sfa_c8": True, "enable_mlapo": mlapo, "enable_kv_nz": False,
    }
    command = [
        "vllm", "serve", args.model, "--served-model-name", "kimi-k3",
        "--port", str(args.port), "--api-server-count", "1", "--trust-remote-code",
        "--safetensors-load-strategy", "prefetch", "--tensor-parallel-size", str(tp),
        "--data-parallel-size", "1", "--data-parallel-size-local", "1",
        "--data-parallel-address", ip, "--data-parallel-rpc-port", str(args.rpc_port),
        "--no-enable-prefix-caching", "--enable-expert-parallel",
        "--dtype", "bfloat16", "--kv-cache-dtype", "auto", "--block-size", "128",
        "--max-num-seqs", "8", "--max-model-len", "4096", "--max-num-batched-tokens", "512",
        "--gpu-memory-utilization", str(args.memory_utilization), "--tokenizer-mode", "kimi_k3",
        "--mm-processor-cache-gb", "0", "--additional-config", json.dumps(additional),
        "--mm-encoder-tp-mode", "data", "--limit-mm-per-prompt", '{"vision_chunk":40}',
        "--enable-auto-tool-choice", "--reasoning-parser", "kimi_k3", "--tool-call-parser", "kimi_k3",
    ]
    if graph_mode:
        command += ["--compilation-config", json.dumps({
            "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8],
        })]
    else:
        command.append("--enforce-eager")
    label = f"tp{tp}_{'mlapo' if mlapo else 'ordinary'}_{'decode_graph' if graph_mode else 'eager'}"
    return label, command, env


def proc_stat(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        return {"state": fields[0], "parent": int(fields[1]), "group": int(fields[2]),
                "session": int(fields[3]), "start": fields[19]}
    except (OSError, IndexError, ValueError):
        return None


def live_session_members(session):
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        info = proc_stat(entry.name)
        if info and info["session"] == session and info["state"] != "Z":
            members.append(int(entry.name))
    return members


def stop_owned_server(proc, identity, timeout):
    """Signal only our still-identifiable session leader's process group."""
    running = proc.poll() is None
    descendants = {}
    if running:
        current = proc_stat(proc.pid)
        if (not current or not identity or current["start"] != identity["start"]
                or current["group"] != proc.pid or current["session"] != proc.pid):
            raise UnsafeCleanup(f"Cannot verify ownership of PID {proc.pid}; no signal sent. Inspect it manually.")
        # Track descendants before the leader exits, including workers that
        # may have selected a different process group/session. We never signal
        # those other groups, but must not launch a new case over live workers.
        process_info = {}
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                info = proc_stat(entry.name)
                if info:
                    process_info[int(entry.name)] = info
        owned = {proc.pid}
        while True:
            found = {pid for pid, info in process_info.items() if info["parent"] in owned}
            if found.issubset(owned):
                break
            owned.update(found)
        descendants = {pid: process_info[pid]["start"] for pid in owned if pid in process_info}
        emit(f"Stopping only this run's process group PGID={proc.pid} with SIGTERM.")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout

    def remaining():
        members = set(live_session_members(proc.pid))
        for pid, stamp in descendants.items():
            info = proc_stat(pid)
            if info and info["start"] == stamp and info["state"] != "Z":
                members.add(pid)
        return sorted(members)

    while time.monotonic() < deadline:
        proc.poll()
        members = remaining()
        if not members:
            return
        time.sleep(0.5)
    members = remaining()
    raise UnsafeCleanup(
        f"Owned session {proc.pid} still has live PIDs {members}. Matrix stopped; no SIGKILL/pkill used. "
        "Inspect and clean up these workers manually before rerunning."
    )


def copy_server_logs(proc, logfile, label):
    for line in proc.stdout:
        try:
            logfile.write(line)
            logfile.flush()
        except ValueError:
            return  # Safety-stop can leave workers alive for manual cleanup.
        print(f"[{label}] {line}", end="", flush=True)


def opener():
    # Local container HTTP must not be sent through corporate HTTP proxies.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def wait_ready(proc, args):
    deadline = time.monotonic() + args.ready_timeout
    url = f"http://127.0.0.1:{args.port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited before readiness, returncode={proc.returncode}; inspect its log.")
        try:
            with opener().open(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    raise TimeoutError(f"Service not healthy after {args.ready_timeout}s: {url}")


def one_request(args, stage, index, barrier):
    repeats = (0, 16, 48, 96, 8, 32, 64, 112)[index % 8] if stage in (1, 2) else 0
    prompt = ("请继续补全这段文本。\n" + "北京是中国的首都，上海位于中国东部。\n" * repeats
              + f"请求编号 {stage}-{index}：介绍一种常见的交通工具。")
    body = {
        "model": "kimi-k3", "prompt": prompt, "max_tokens": 32, "temperature": 0,
        "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
    }
    record = {"stage": stage, "request": index, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    barrier.wait()
    started = time.perf_counter()
    first = None
    texts, usage, done = [], None, False
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with opener().open(request, timeout=args.request_timeout) as response:
            for raw_line in response:
                if time.perf_counter() - started > args.request_timeout:
                    raise TimeoutError("Streaming request exceeded its total time budget.")
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                payload = json.loads(data)
                if "error" in payload:
                    raise RuntimeError(str(payload["error"]))
                for choice in payload.get("choices", []):
                    text = choice.get("text", "")
                    if text:
                        if first is None:
                            first = time.perf_counter() - started
                        texts.append(text)
                if payload.get("usage") is not None:
                    usage = payload["usage"]
        if not done or not texts or first is None:
            raise RuntimeError("Missing complete SSE response or nonempty generated text.")
        if not isinstance(usage, dict):
            raise RuntimeError("Missing streaming usage; cannot verify multi-token Decode.")
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise RuntimeError(f"Invalid usage.{key}: {value!r}")
        if usage["completion_tokens"] != 32:
            raise RuntimeError(f"Expected 32 tokens with ignore_eos=true, received {usage['completion_tokens']}.")
        record.update(ok=True, usage=usage, text="".join(texts))
    except Exception as exc:
        record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    record.update(ttft_seconds=first, elapsed_seconds=time.perf_counter() - started)
    return record


def smoke(args, proc, output_dir):
    all_ok = True
    with (output_dir / "requests.jsonl").open("w") as records:
        for stage, concurrency in enumerate(STAGES):
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited during requests, returncode={proc.returncode}.")
            emit(f"Smoke stage {stage + 1}/{len(STAGES)}: concurrent requests={concurrency}, output_tokens=32")
            barrier = threading.Barrier(concurrency)
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(one_request, args, stage, i, barrier) for i in range(concurrency)]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    records.write(json.dumps(result, ensure_ascii=False) + "\n")
                    records.flush()
                    all_ok = all_ok and result["ok"]
                    emit(json.dumps({k: v for k, v in result.items() if k != "text"}, ensure_ascii=False))
    return all_ok


def run_case(args, label, command, env, output_dir):
    check_ports(args)
    output_dir.mkdir(parents=True)
    (output_dir / "command.txt").write_text(shlex.join(command) + "\n")
    selected_env = {key: env[key] for key in (
        "ASCEND_RT_VISIBLE_DEVICES", "VLLM_PLUGINS", "VLLM_USE_V2_MODEL_RUNNER",
        "HCCL_IF_IP", "HCCL_SOCKET_IFNAME", "VLLM_ENGINE_READY_TIMEOUT_S",
    )}
    (output_dir / "environment.json").write_text(json.dumps(selected_env, indent=2) + "\n")
    emit(f"\n=== {label}: devices={env['ASCEND_RT_VISIBLE_DEVICES']} log={output_dir / 'server.log'} ===")
    emit(shlex.join(command))
    record = {"case": label, "ok": False, "log_directory": str(output_dir)}
    started = time.monotonic()
    with (output_dir / "server.log").open("w", buffering=1) as logfile:
        proc = subprocess.Popen(command, env=env, cwd=args.work_dir, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                                start_new_session=True)
        identity = proc_stat(proc.pid)
        (output_dir / "process.json").write_text(json.dumps({"pid": proc.pid, "identity": identity}) + "\n")
        thread = threading.Thread(target=copy_server_logs, args=(proc, logfile, label), daemon=True)
        thread.start()
        try:
            wait_ready(proc, args)
            record["ready_seconds"] = time.monotonic() - started
            emit(f"{label} ready after {record['ready_seconds']:.1f}s; running real requests.")
            record["ok"] = smoke(args, proc, output_dir)
            # Marker selection is evidence of framework routing, not kernel-level profiling.
            text = (output_dir / "server.log").read_text(errors="replace")
            if "K3 native QSFA C8 enabled" not in text:
                raise RuntimeError("Expected K3 C8 implementation log missing; AV may not be using patched VA.")
            if "mlapo" in label and "K3 C8 MLAPO: Prolog V4 No-RoPE" not in text:
                raise RuntimeError("Expected Prolog V4 path log missing.")
            if "decode_graph" in label and "ACL graph replay is active for the V2 target model" not in text:
                raise RuntimeError("MRV2 graph replay marker missing; capture/startup alone is not a graph pass.")
        except KeyboardInterrupt:
            record["error"] = "Interrupted by user."
            raise
        except Exception as exc:
            record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            emit(f"FAILED {label}: {record['error']}")
        finally:
            try:
                stop_owned_server(proc, identity, args.stop_timeout)
            except UnsafeCleanup as exc:
                record.update(ok=False, cleanup_error=str(exc))
                raise
            finally:
                thread.join(timeout=3)
                record["elapsed_seconds"] = time.monotonic() - started
                (output_dir / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    check_ports(args)
    return record


def main():
    args = parse_args()
    ip = args.local_ip or ("127.0.0.1" if args.dry_run else interface_ip(args.nic))
    socket.inet_aton(ip)
    cases = [build_case(args, tp, mlapo, graph_mode, ip)
             for tp in args.tp_sizes for mlapo in (False, True) for graph_mode in (False, True)]
    if args.dry_run:
        for label, command, env in cases:
            emit(f"{label}: ASCEND_RT_VISIBLE_DEVICES={env['ASCEND_RT_VISIBLE_DEVICES']}")
            emit(shlex.join(command))
        return 0
    if sys.platform != "linux" or not Path("/proc/self/stat").exists():
        raise RuntimeError("Run inside the Linux A5 container; process ownership checks require /proc.")
    if not shutil.which("vllm"):
        raise RuntimeError("vllm not found in the current Python environment.")
    if not Path(args.model, "config.json").is_file():
        raise RuntimeError(f"Model config missing: {args.model}/config.json")
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_root = Path(args.log_dir).absolute() / stamp
    output_root.mkdir(parents=True)
    emit(f"Results: {output_root}")
    emit("Only run on NPUs allocated to you. Existing services are never stopped automatically.")
    emit("This is a functional smoke matrix, NOT a precision evaluation or a controlled performance benchmark.")
    emit("TTFT is client-observed first nonempty text latency, including queue/network overhead.")
    results = []
    try:
        for label, command, env in cases:
            results.append(run_case(args, label, command, env, output_root / label))
            (output_root / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    except (UnsafeCleanup, KeyboardInterrupt) as exc:
        emit(f"MATRIX STOPPED SAFELY: {exc}")
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
    passed = sum(result["ok"] for result in results)
    emit(f"Matrix complete: {passed}/{len(results)} cases passed. Results: {output_root}")
    emit("Graph configuration + successful smoke requests does not replace profiler evidence of actual graph replay.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit(f"ERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
PY
