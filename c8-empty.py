"""One idle A5: installed QSFA reader, empty local KV, capture and replay.

No model loading, framework modification, HCCL, Prolog or performance claims.
The parent imports no torch and supervises one isolated worker at a time.
On failure/timeout it stops the suite; a killed worker does NOT guarantee that
an already-hung NPU task has been cleared. Never run on a shared busy card.
"""

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import ModuleType

PATTERNS = ("nonempty", "empty", "mixed")
MODES = ("eager", "graph")
BLOCK = 128
PACKED = 656
SCOPE = "one-card QSFA empty-shard correctness; NOT HCCL, Prolog or model acceptance"


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def stage(name, status="begin", **fields):
    emit("stage", stage=name, status=status, **fields)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--card",
        type=int,
        required=True,
        help="One authorized idle physical A5 card ID",
    )
    parser.add_argument(
        "--va-source",
        type=Path,
        help="Editable VA source root; otherwise locate installed package",
    )
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument(
        "--tp",
        type=int,
        default=8,
        help="Service TP, used only to derive Query head shape",
    )
    parser.add_argument(
        "--dcp",
        type=int,
        default=2,
        help="Shape calculation only; no real distributed group",
    )
    parser.add_argument("--total-heads", type=int, default=96)
    parser.add_argument(
        "--heads", type=int, help="Override Query heads AFTER DCP AllGather"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=160000,
        help="Use service value: controls full index width",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="Wall-time limit per child, including imports",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Parent directory for a new unique log directory",
    )
    parser.add_argument("--only", choices=("all", *PATTERNS), default="all")
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print shapes/matrix without importing torch or using NPU",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.card < 0 or args.tokens < 2 or args.max_model_len < 4 or args.timeout <= 0:
        parser.error("card must be >=0; tokens >=2; max-model-len >=4; timeout >0")
    if args.tp < 1 or args.dcp < 1 or args.total_heads < 1:
        parser.error("tp, dcp and total-heads must be positive")
    if args.heads is None:
        if args.total_heads % args.tp or args.tp % args.dcp:
            parser.error(
                "total-heads must divide by tp, and tp must divide by dcp; or supply --heads"
            )
        args.heads = args.total_heads // args.tp * args.dcp
    if not 1 <= args.heads <= 128:
        parser.error("post-gather heads must be in [1,128]")
    if args.worker and (args.only == "all" or args.mode == "all"):
        parser.error("Worker needs one --only and one --mode")
    return args


def case_matrix(args):
    patterns = PATTERNS if args.only == "all" else (args.only,)
    modes = MODES if args.mode == "all" else (args.mode,)
    return [(pattern, mode) for pattern in patterns for mode in modes]


def lengths_for(pattern, count):
    if pattern == "empty":
        return [0] * count
    if pattern == "nonempty":
        return [1] * count
    if pattern == "mixed":
        return [i % 2 for i in range(count)]
    if pattern == "changed":
        return [2 if i % 2 else 1 for i in range(count)]
    raise ValueError(pattern)


def resolve_source(explicit):
    if explicit is not None:
        root = explicit.resolve()
    else:
        spec = importlib.util.find_spec("vllm_ascend")
        if spec is None or spec.origin is None:
            raise RuntimeError("VA package not found; pass --va-source /path/to/va")
        root = Path(spec.origin).resolve().parent.parent
    if not (root / "vllm_ascend/attention/kimi_k3_sfa_c8_graph.py").is_file():
        raise RuntimeError(
            f"Installed C8 row reader missing under {root}; no patch will be applied"
        )
    return root


def load_readers(root, torch):
    """Execute installed source, removing only its cross-module import.

    No vllm/platform/MoE package initialization, no source-function rewrites.
    Record exact hashes so results can be tied to the installed patch.
    """
    modules, sources = {}, {}
    for name, filename in (
        ("ordinary", "kimi_k3_sfa_c8.py"),
        ("rows", "kimi_k3_sfa_c8_graph.py"),
    ):
        path = root / "vllm_ascend/attention" / filename
        source = path.read_bytes()
        tree = ast.parse(source, filename=str(path))
        namespace = {"torch": torch}
        if name == "rows":
            dependencies = [
                n
                for n in tree.body
                if isinstance(n, ast.ImportFrom)
                and n.module == "vllm_ascend.attention.kimi_k3_sfa_c8"
            ]
            if len(dependencies) != 1:
                raise RuntimeError(
                    "Installed reader imports changed; update diagnostic instead of modifying VA"
                )
            for alias in dependencies[0].names:
                namespace[alias.asname or alias.name] = getattr(
                    modules["ordinary"], alias.name
                )
            tree.body.remove(dependencies[0])
        for node in ast.walk(tree):
            names = (
                [node.module]
                if isinstance(node, ast.ImportFrom)
                else [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            if any(n and n.startswith("vllm") for n in names):
                raise RuntimeError(f"Unexpected framework dependency: {path}")
        module = ModuleType(f"c8_empty_diag_{name}")
        module.__dict__.update(namespace)
        exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102
        modules[name] = module
        sources[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(source).hexdigest(),
        }
    rows = modules["rows"]
    if "return_lse" not in inspect.signature(rows.kimi_k3_sfa_c8_decode).parameters:
        raise RuntimeError(
            "Installed reader has no return_lse: use the C8 DCP test patch first"
        )
    return modules["ordinary"], rows, sources


def fixture_cpu(torch, tokens, heads):
    """Independent legal packed bytes; no native quantizer or scatter involved.

    Four positive FP32 scales and exactly representable E4M3 values. Two KV
    tokens differ, so a stale length of 1 cannot pass a replay length-2 check.
    """
    scales = torch.tensor([0.125, 0.5, 2.0, 8.0], dtype=torch.float32)
    base = ((torch.arange(512) % 4) + 1).float()
    fp8_values = torch.stack((base, base.flip(0) + 1)).to(torch.float8_e4m3fn)
    latent = fp8_values.float() * scales.repeat_interleave(128)
    raw = torch.stack(
        (torch.linspace(-0.25, 0.25, 64), torch.linspace(0.3, -0.3, 64))
    ).bfloat16()
    packed = torch.zeros((BLOCK, 1, PACKED), dtype=torch.uint8)
    packed[:2, 0, :512] = fp8_values.view(torch.uint8)
    packed[:2, 0, 512:640] = raw.view(torch.uint8)
    packed[:, 0, 640:] = scales.view(torch.uint8)
    generator = torch.Generator().manual_seed(20261030)
    query = (torch.randn(tokens, heads, 576, generator=generator) * 0.015).bfloat16()
    return query, packed.unsqueeze(0), latent, raw.float()


def reference(torch, query, latent, raw, lengths, scale):
    query = query.float().cpu()
    keys = torch.cat((latent.float(), raw.float()), dim=-1)
    scores = torch.matmul(query, keys.T) * scale
    lens = torch.tensor(lengths)
    mask = torch.arange(2).view(1, 1, 2) < lens[:, None, None]
    masked_scores = scores.masked_fill(~mask, -float("inf"))
    expected_lse = torch.logsumexp(masked_scores, dim=-1, keepdim=True)
    # Empty softmax is undefined; its framework contribution is defined as 0.
    safe_scores = torch.where(
        (lens > 0)[:, None, None], masked_scores, torch.zeros_like(masked_scores)
    )
    output = torch.matmul(safe_scores.softmax(-1), latent.float())
    output[lens == 0] = 0
    return output, expected_lse


def check_result(
    torch, output, lse, expected, expected_lse, lengths, raw_result, heads
):
    actual, actual_lse = output.float().cpu(), lse.float().cpu()
    valid = torch.tensor(lengths) > 0
    delta = (actual - expected).abs()
    output_ok = bool(
        torch.isfinite(actual).all() and (delta <= 0.025 + 0.02 * expected.abs()).all()
    )
    empty_ok = bool(
        (actual[~valid] == 0).all() and torch.isneginf(actual_lse[~valid]).all()
    )
    lse_ok = bool(torch.isfinite(actual_lse[valid]).all())
    if bool(valid.any()):
        lse_ok = lse_ok and bool(
            (
                (actual_lse[valid] - expected_lse[valid]).abs()
                <= 0.025 + 0.01 * expected_lse[valid].abs()
            ).all()
        )
    raw_max = raw_result[1].float().cpu()[0, :, :heads]
    raw_sum = raw_result[2].float().cpu()[0, :, :heads]
    # Raw empty outputs are diagnostic only: the framework explicitly masks
    # them. Do not require undocumented zero-output native semantics.
    return {
        "pass": output_ok and empty_ok and lse_ok,
        "output_ok": output_ok,
        "empty_contribution_ok": empty_ok,
        "lse_ok": lse_ok,
        "valid_rows": int(valid.sum()),
        "empty_rows": int((~valid).sum()),
        "max_abs": float(delta.max()) if bool(torch.isfinite(delta).all()) else None,
        "raw_empty_max_nonfinite": int((~torch.isfinite(raw_max[~valid])).sum()),
        "raw_empty_sum_nonfinite": int((~torch.isfinite(raw_sum[~valid])).sum()),
    }


def worker(args):
    stage("import_runtime")
    import torch
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("No usable NPU in this container")
    torch.npu.set_device(0)
    ordinary, rows, sources = load_readers(resolve_source(args.va_source), torch)
    native_op = ordinary.require_kimi_k3_sfa_c8_ops()
    emit(
        "environment",
        torch=torch.__version__,
        torch_npu=getattr(torch_npu, "__version__", "unknown"),
        device=torch.npu.get_device_name(0),
        physical_card=args.card,
        sources=sources,
        tokens=args.tokens,
        query_heads_after_gather=args.heads,
        native_padded_heads=ordinary.get_kimi_k3_sfa_c8_padded_heads(args.heads),
        max_model_len=args.max_model_len,
        return_lse=True,
        scope=SCOPE,
    )
    stage("import_runtime", "complete")
    stage("prepare_fixture")
    query_cpu, cache_cpu, latent, raw = fixture_cpu(torch, args.tokens, args.heads)
    device = torch.device("npu:0")
    query = query_cpu.to(device)
    cache = cache_cpu.to(device).view(torch.float8_e4m3fn)
    table_width = math.ceil(args.max_model_len / BLOCK)
    state = rows.KimiK3C8DecodeState(
        args.tokens, args.max_model_len, table_width, device
    )
    tables = torch.zeros((args.tokens, table_width), dtype=torch.int32, device=device)
    slots = torch.full((args.tokens,), -1, dtype=torch.int64, device=device)
    raw_capture = {}

    def observe_native(*a, **kw):
        result = native_op(*a, **kw)
        raw_capture["result"] = result
        return result

    # Only redirect the diagnostic's isolated in-memory module; VA files and
    # other processes are unchanged. Exact installed reader still executes.
    rows.require_kimi_k3_sfa_c8_ops = lambda: observe_native
    scale = 192**-0.5

    def update(pattern):
        lengths = lengths_for(pattern, args.tokens)
        # Simulate rank1 of DCP2/interleave1 with one query per request.
        # Global length1 produces local0; global2/4 produce local1/2.
        global_lengths = torch.tensor(
            [max(1, 2 * n) for n in lengths], dtype=torch.int32
        )
        state.update(
            global_lengths,
            tables,
            slots,
            args.tokens,
            seq_lens_device=global_lengths.to(device),
            query_start_loc_cpu=torch.arange(args.tokens + 1, dtype=torch.int32),
            dcp_size=2,
            dcp_rank=1,
            interleave=1,
        )
        torch.npu.synchronize()
        if state.seq_lens.cpu().tolist() != lengths:
            raise RuntimeError(
                "Installed DCP metadata did not produce the intended local lengths"
            )
        return lengths

    def run():
        return rows.kimi_k3_sfa_c8_decode(
            query[..., :512], query[..., 512:], cache, state, scale, return_lse=True
        )

    def validate(result, pattern, lengths, phase):
        expected, expected_lse = reference(
            torch, query_cpu, latent, raw, lengths, scale
        )
        checks = check_result(
            torch,
            *result,
            expected,
            expected_lse,
            lengths,
            raw_capture["result"],
            args.heads,
        )
        checks["cache_unchanged"] = bool(
            torch.equal(cache.view(torch.uint8).cpu(), cache_cpu)
        )
        checks["pass"] = checks["pass"] and checks["cache_unchanged"]
        emit(
            "check",
            mode=args.mode,
            capture_pattern=args.only,
            pattern=pattern,
            phase=phase,
            **checks,
        )
        if not checks["pass"]:
            raise RuntimeError(
                "Numerical/cache check failed; see preceding check record"
            )

    lengths = update(args.only)
    stage("prepare_fixture", "complete", local_lengths=sorted(set(lengths)))
    if args.mode == "eager":
        stage("eager_call")
        result = run()
        stage("eager_submitted")
        torch.npu.synchronize()
        stage("eager_call", "complete")
        validate(result, args.only, lengths, "eager")
    else:
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        stage("warmup")
        with torch.npu.stream(stream):
            for _ in range(2):
                run()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        stage("warmup", "complete")
        graph = torch.npu.NPUGraph()
        stage("capture")
        with torch.npu.graph(graph, stream=stream):
            result = run()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        stage("capture", "complete")
        # Persistent buffers and the same graph must handle changed lengths,
        # including both nonempty->empty and empty->nonempty transitions.
        for step, pattern in enumerate(
            (args.only, "mixed", "empty", "changed", "nonempty")
        ):
            stage("update_metadata", step=step, pattern=pattern)
            lengths = update(pattern)
            stage("replay", step=step, pattern=pattern)
            graph.replay()
            stage("replay_submitted", step=step, pattern=pattern)
            torch.npu.synchronize()
            stage("replay", "complete", step=step, pattern=pattern)
            validate(result, pattern, lengths, f"replay_{step}")
    emit(
        "case_complete",
        pattern=args.only,
        mode=args.mode,
        passed=True,
        peak_allocated_mib=torch.npu.max_memory_allocated() / 2**20,
    )
    return 0


def stop_owned_child(process):
    """Signal only the process group created for our own diagnostic worker."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            return True
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return True
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
    return process.poll() is not None


def supervise(command, env, log_path, timeout):
    timed_out = False
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        try:
            with log_path.open(encoding="utf-8", errors="replace") as reader:
                while True:
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end="", flush=True)
                    if process.poll() is not None:
                        print(reader.read(), end="", flush=True)
                        break
                    if time.monotonic() >= deadline:
                        timed_out = True
                        emit(
                            "timeout",
                            seconds=timeout,
                            worker_pid=process.pid,
                            log=str(log_path),
                        )
                        reaped = stop_owned_child(process)
                        emit(
                            "worker_stop",
                            reaped=reaped,
                            warning="NPU may remain hung; suite stops. Do not start another test until device is checked.",
                        )
                        print(reader.read(), end="", flush=True)
                        break
                    time.sleep(0.2)
        except BaseException:
            stop_owned_child(process)
            raise
    return {"exit_code": process.poll(), "timeout": timed_out, "log": str(log_path)}


def parent(args):
    matrix = case_matrix(args)
    plan = {
        "tokens": args.tokens,
        "query_heads_after_gather": args.heads,
        "tp": args.tp,
        "dcp_shape": args.dcp,
        "max_model_len": args.max_model_len,
        "physical_card": args.card,
        "cases": matrix,
        "index_and_mask_mib": args.tokens * args.max_model_len * 5 / 2**20,
        "memory_note": "Index+mask only; NPU runtime, query, native workspace and graph pool are additional.",
        "scope": SCOPE,
    }
    emit("plan", **plan)
    if args.plan_only:
        return 0
    source = resolve_source(args.va_source)
    directory = args.output_dir or (
        Path("/mnt") if Path("/mnt").is_dir() else Path.cwd()
    )
    directory.mkdir(parents=True, exist_ok=True)
    logs = Path(tempfile.mkdtemp(prefix="c8-empty-", dir=directory))
    (logs / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    emit("logs", directory=str(logs), va_source=str(source))
    env = dict(
        os.environ, ASCEND_RT_VISIBLE_DEVICES=str(args.card), PYTHONUNBUFFERED="1"
    )
    results = []
    for pattern, mode in matrix:
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--card",
            str(args.card),
            "--va-source",
            str(source),
            "--tokens",
            str(args.tokens),
            "--heads",
            str(args.heads),
            "--max-model-len",
            str(args.max_model_len),
            "--only",
            pattern,
            "--mode",
            mode,
        ]
        emit("case_start", pattern=pattern, mode=mode)
        result = supervise(command, env, logs / f"{pattern}-{mode}.log", args.timeout)
        result.update(pattern=pattern, mode=mode)
        # Exit0 alone is insufficient: require the worker's final success marker.
        records = []
        for line in (
            Path(result["log"])
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        ):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        complete = any(
            isinstance(r, dict)
            and r.get("event") == "case_complete"
            and r.get("passed") is True
            for r in records
        )
        result["pass"] = result["exit_code"] == 0 and not result["timeout"] and complete
        results.append(result)
        if not result["pass"]:
            break
    summary = {
        "planned": len(matrix),
        "completed": len(results),
        "passed": sum(r["pass"] for r in results),
        "all_passed": len(results) == len(matrix) and all(r["pass"] for r in results),
        "results": results,
        "scope": SCOPE,
    }
    (logs / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    emit("summary", **summary)
    emit(
        "send_back",
        files=[str(logs / "summary.json"), str(logs)],
        note="Send summary plus failed case log; no dataset or weights needed.",
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    try:
        cli = parse_args()
        if cli.worker:
            # Keep selection in place even when a worker is explicitly rerun.
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(cli.card)
            raise SystemExit(worker(cli))
        raise SystemExit(parent(cli))
    except Exception as error:  # noqa: BLE001 - preserve native failures in diagnostic output
        emit("error", type=type(error).__name__, message=str(error))
        traceback.print_exc()
        raise SystemExit(1)
