"""Two idle A5s: real HCCL + installed C8 attention math, no model weights.

Reuses c8-empty.py from the verified previous diagnostic. Uses isolated AST
loads of installed DCP helpers and base device all_gather to avoid the VA
plugin/MoE import cycle. A fresh two-rank WORLD group replaces GroupCoordinator
lifecycle/custom-op dispatch: this is NOT full runner/TP/multi-node acceptance.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace


def helper_module():
    path = Path(__file__).with_name("c8-empty.py")
    if not path.exists():
        path = Path(__file__).with_name("diagnose-k3-c8-empty-kv.py")
    spec = importlib.util.spec_from_file_location("c8_empty_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = helper_module()
SCOPE = "two-card isolated HCCL plus installed C8 helpers; NOT Prolog, full runner or multi-node acceptance"
CASES = (("comm", "eager"), ("comm", "graph"), ("chain", "eager"), ("chain", "graph"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cards", required=True, help="Two authorized idle physical cards, e.g. 0,1"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=8,
        help="Service TP for head shape only; test always uses two cards",
    )
    parser.add_argument("--total-heads", type=int, default=96)
    parser.add_argument(
        "--local-heads", type=int, help="Override per-rank heads BEFORE DCP gather"
    )
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=160000)
    parser.add_argument(
        "--timeout",
        type=float,
        default=240,
        help="Wall seconds per two-rank test, including imports",
    )
    parser.add_argument("--va-source", type=Path)
    parser.add_argument("--vllm-source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", choices=("all", "comm", "chain"), default="all")
    parser.add_argument("--mode", choices=("all", "eager", "graph"), default="all")
    parser.add_argument("--pattern", choices=("all", "both_nonempty", "one_empty_rank"), default="all")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--rank", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument("--rendezvous", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        cards = [int(n) for n in args.cards.split(",")]
    except ValueError:
        parser.error("cards must be two nonnegative integers")
    if len(cards) != 2 or min(cards) < 0 or cards[0] == cards[1]:
        parser.error("Select two distinct nonnegative physical card IDs")
    args.cards = ",".join(map(str, cards))
    if (
        args.tokens < 2
        or args.max_model_len < 2
        or args.timeout <= 0
        or args.tp < 2
        or args.total_heads < 1
    ):
        parser.error("tokens/model-len/tp >=2, timeout/total-heads >0 required")
    if args.local_heads is None:
        if args.total_heads % args.tp or args.tp % 2:
            parser.error("total-heads must divide by even TP; or specify --local-heads")
        args.local_heads = args.total_heads // args.tp
    if not 1 <= args.local_heads <= 64:
        parser.error("local-heads must be in [1,64]")
    if args.rank is not None and (
        args.only == "all" or args.mode == "all" or args.rendezvous is None
    ):
        parser.error("Worker needs one case, one mode and rendezvous")
    return args


def read_ast(path, names, namespace, class_name=None):
    source = path.read_bytes()
    tree = ast.parse(source, filename=str(path))
    body = tree.body
    if class_name:
        matches = [
            n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name
        ]
        if len(matches) != 1:
            raise RuntimeError(f"No unique {class_name} in {path}")
        body = matches[0].body
    selected = [
        n
        for n in body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    if sorted(n.name for n in selected) != sorted(names):
        raise RuntimeError(f"Installed helper definitions changed in {path}")
    exec(  # noqa: S102
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(source).hexdigest(),
        "definitions": names,
    }


def resolve_vllm(explicit):
    if explicit:
        root = explicit.resolve()
    else:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("vLLM not found; use --vllm-source")
        root = Path(spec.origin).resolve().parent.parent
    return root


def load_dcp(va_root, vllm_root, torch, dist, torch_npu):
    def unexpected_group_lookup():
        raise RuntimeError("Diagnostic must use its explicit two-rank group")

    namespace = {
        "torch": torch,
        "dist": dist,
        "torch_npu": torch_npu,
        "get_dcp_group": unexpected_group_lookup,
        "get_decode_context_model_parallel_world_size": lambda: 2,
    }
    sources = [
        read_ast(
            va_root / "vllm_ascend/attention/context_parallel/common_cp.py",
            ["DCPImplMixin", "_process_attn_out_lse", "_npu_attention_update"],
            namespace,
        )
    ]
    gather_ns = {"torch": torch, "dist": dist}
    sources.append(
        read_ast(
            vllm_root
            / "vllm/distributed/device_communicators/base_device_communicator.py",
            ["all_gather"],
            gather_ns,
            "DeviceCommunicatorBase",
        )
    )
    group = SimpleNamespace(world_size=2, device_group=dist.group.WORLD)
    group.all_gather = lambda tensor, dim: gather_ns["all_gather"](group, tensor, dim)
    impl = namespace["DCPImplMixin"].__new__(namespace["DCPImplMixin"])
    impl.dcp_size, impl.dcp_rank = 2, dist.get_rank()
    impl.dcp_group, impl.dcp_device_group = group, dist.group.WORLD
    return impl, namespace, sources


def global_lengths(pattern, tokens):
    if pattern == "one_empty_rank":
        return [1] * tokens
    if pattern == "both_nonempty":
        return [2] * tokens
    if pattern == "mixed":
        return [1 + n % 2 for n in range(tokens)]
    if pattern == "reverse":
        return [2 - n % 2 for n in range(tokens)]
    if pattern == "padded":
        return [2] * (tokens // 2) + [0] * (tokens - tokens // 2)
    raise ValueError(pattern)


def worker(args):
    rank = args.rank

    def event(name, **fields):
        helper.emit(name, rank=rank, **fields)

    event("stage", stage="runtime_import", status="begin")
    import torch
    import torch.distributed as dist
    import torch_npu

    torch.npu.set_device(rank)
    event("stage", stage="group_init", status="begin")
    dist.init_process_group(
        "hccl",
        init_method=args.rendezvous.resolve().as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=args.timeout),
    )
    event("stage", stage="group_init", status="complete")
    va_root = helper.resolve_source(args.va_source)
    ordinary, rows, reader_sources = helper.load_readers(va_root, torch)
    impl, namespace, dcp_sources = load_dcp(
        va_root, resolve_vllm(args.vllm_source), torch, dist, torch_npu
    )
    if args.only == "chain":
        ordinary.require_kimi_k3_sfa_c8_ops()
    event(
        "environment",
        torch=torch.__version__,
        torch_npu=getattr(torch_npu, "__version__", "unknown"),
        physical_cards=args.cards,
        logical_device=rank,
        device=torch.npu.get_device_name(rank),
        tokens=args.tokens,
        local_heads=args.local_heads,
        gathered_heads=2 * args.local_heads,
        native_heads=ordinary.get_kimi_k3_sfa_c8_padded_heads(2 * args.local_heads),
        max_model_len=args.max_model_len,
        sources={"readers": reader_sources, "dcp": dcp_sources},
        scope=SCOPE,
        hccl_op_expansion_mode=os.environ.get("HCCL_OP_EXPANSION_MODE"),
    )

    phase = "setup"
    stage_sync = args.mode == "eager"

    def invoke(name, function, *a, **kw):
        # No device synchronization is inserted into warmup/capture/replay.
        # Host submitted != device complete; synchronized markers only in eager.
        operands = [
            {"shape": list(t.shape), "dtype": str(t.dtype), "stride": list(t.stride())}
            for t in a
            if isinstance(t, torch.Tensor)
        ]
        event("stage", phase=phase, stage=name, status="begin", operands=operands)
        result = function(*a, **kw)
        event("stage", phase=phase, stage=name, status="submitted")
        if stage_sync:
            torch.npu.synchronize()
            event("stage", phase=phase, stage=name, status="device_complete")
        return result

    # Instrument only this isolated module, never the installed source. Sync
    # before the native call separates query preparation from native execution.
    native_factory = rows.require_kimi_k3_sfa_c8_ops

    def traced_factory():
        native = invoke("qsfa_resolve", native_factory)

        def traced_native(*a, **kw):
            event("stage", phase=phase, stage="qsfa_prepare", status="submitted")
            if stage_sync:
                torch.npu.synchronize()
                event("stage", phase=phase, stage="qsfa_prepare", status="device_complete")
            result = invoke("qsfa_native", native, *a, **kw)
            event("stage", phase=phase, stage="qsfa_postprocess", status="begin")
            return result

        return traced_native

    rows.require_kimi_k3_sfa_c8_ops = traced_factory

    original_exchange = namespace["_process_attn_out_lse"]
    original_merge = namespace["_npu_attention_update"]
    namespace["_process_attn_out_lse"] = lambda *a, **kw: invoke(
        "alltoall", original_exchange, *a, **kw
    )
    namespace["_npu_attention_update"] = lambda *a, **kw: invoke(
        "merge", original_merge, *a, **kw
    )
    device = torch.device(f"npu:{rank}")
    full_query, full_packed, latent, raw = helper.fixture_cpu(
        torch, args.tokens, 2 * args.local_heads
    )
    head_slice = slice(rank * args.local_heads, (rank + 1) * args.local_heads)
    query = full_query[:, head_slice].contiguous().to(device)
    local_packed = full_packed.clone()
    local_packed[0, 0] = full_packed[0, rank]
    cache = local_packed.to(device).view(torch.float8_e4m3fn)
    width = math.ceil(args.max_model_len / helper.BLOCK)
    state = rows.KimiK3C8DecodeState(args.tokens, args.max_model_len, width, device)
    table = torch.zeros((args.tokens, width), dtype=torch.int32, device=device)
    slots = torch.full((args.tokens,), -1, dtype=torch.int64, device=device)

    def update(pattern):
        lengths = global_lengths(pattern, args.tokens)
        n = sum(length > 0 for length in lengths)
        tensor = torch.tensor(lengths, dtype=torch.int32)
        state.update(
            tensor,
            table,
            slots,
            n,
            seq_lens_device=tensor.to(device),
            query_start_loc_cpu=torch.arange(args.tokens + 1, dtype=torch.int32),
            dcp_size=2,
            dcp_rank=rank,
            interleave=1,
        )
        torch.npu.synchronize()
        event(
            "metadata",
            phase=phase,
            pattern=pattern,
            active_tokens=n,
            local_lengths=sorted(set(state.seq_lens.cpu().tolist())),
        )
        return lengths

    def run():
        ql, qr = invoke(
            "allgather",
            impl._dcp_all_gather_fragments,
            query[..., :512],
            query[..., 512:],
            dim=1,
        )
        if args.only == "chain":
            output, lse = invoke(
                "qsfa",
                rows.kimi_k3_sfa_c8_decode,
                ql,
                qr,
                cache,
                state,
                192**-0.5,
                return_lse=True,
            )
        else:
            # Independent synthetic per-rank contributions. Same output/LSE
            # shapes as QSFA, but no attention kernel can obscure comm failures.
            output = (ql.float() + (rank + 1)).bfloat16()
            lse = torch.zeros((*ql.shape[:2], 1), dtype=torch.float32, device=device)
        merged = impl._merge_dcp_attention_output(output, lse, 512).to(query.dtype)
        if args.only == "chain":
            merged.masked_fill_(~state.active[:, None, None], 0)
        return merged, ql, qr

    def validate(result, lengths, pattern):
        output, ql, qr = result
        gather_ok = torch.equal(ql.cpu(), full_query[..., :512]) and torch.equal(
            qr.cpu(), full_query[..., 512:]
        )
        if args.only == "chain":
            expected, _ = helper.reference(
                torch, full_query, latent, raw, lengths, 192**-0.5
            )
        else:
            a = (full_query[..., :512].float() + 1).bfloat16().float()
            b = (full_query[..., :512].float() + 2).bfloat16().float()
            expected = (a + b) * 0.5
        expected = expected[:, head_slice]
        actual = output.float().cpu()
        delta = (actual - expected).abs()
        finite = bool(torch.isfinite(actual).all())
        accurate = finite and bool((delta <= 0.04 + 0.025 * expected.abs()).all())
        cache_ok = torch.equal(cache.view(torch.uint8).cpu(), local_packed)
        event(
            "check",
            phase=phase,
            pattern=pattern,
            passed=bool(gather_ok and accurate and cache_ok),
            gathered_query_exact=bool(gather_ok),
            finite=finite,
            output_ok=accurate,
            cache_unchanged=bool(cache_ok),
            max_abs=float(delta.max()) if finite else None,
        )
        if not (gather_ok and accurate and cache_ok):
            raise RuntimeError("Distributed result/reference mismatch")

    patterns = (
        "one_empty_rank",
        "both_nonempty",
        "mixed",
        "padded",
        "reverse",
        "one_empty_rank",
    )
    if args.pattern != "all":
        patterns = (args.pattern,)
    if args.mode == "eager":
        for index, pattern in enumerate(
            patterns if args.only == "chain" else ("both_nonempty",)
        ):
            phase = f"eager_{index}"
            lengths = update(pattern)
            result = run()
            torch.npu.synchronize()
            validate(result, lengths, pattern)
    else:
        phase = "warmup"
        initial = patterns[0] if args.only == "chain" else "both_nonempty"
        update(initial)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(2):
                run()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        event("phase_complete", phase=phase)
        phase = "capture"
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            result = run()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        event("phase_complete", phase=phase)
        for index, pattern in enumerate(
            patterns if args.only == "chain" else ("both_nonempty",) * 3
        ):
            phase = f"replay_{index}"
            lengths = update(pattern)
            event("stage", phase=phase, stage="replay", status="begin")
            graph.replay()
            event("stage", phase=phase, stage="replay", status="submitted")
            torch.npu.synchronize()
            event("phase_complete", phase=phase)
            validate(result, lengths, pattern)
        del graph
    torch.npu.synchronize()
    event("stage", stage="group_destroy", status="begin")
    dist.destroy_process_group()
    event(
        "case_complete",
        test=args.only,
        mode=args.mode,
        passed=True,
        peak_allocated_mib=torch.npu.max_memory_allocated() / 2**20,
    )
    return 0


def supervise_pair(commands, envs, logs, timeout):
    processes, writers, readers = [], [], []
    reason = None
    try:
        for command, env, log in zip(commands, envs, logs):
            writer = log.open("x", encoding="utf-8")
            writers.append(writer)
            processes.append(
                subprocess.Popen(
                    command,
                    env=env,
                    stdout=writer,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
            readers.append(log.open(encoding="utf-8", errors="replace"))
        deadline = time.monotonic() + timeout
        while True:
            for rank, reader in enumerate(readers):
                chunk = reader.read()
                if chunk:
                    print(f"[rank {rank}] {chunk}", end="", flush=True)
            codes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in codes):
                reason = "worker_failed"
                break
            if all(code is not None for code in codes):
                break
            if time.monotonic() >= deadline:
                reason = "timeout"
                break
            time.sleep(0.2)
    finally:
        # Stop only the two owned process groups, never the user's services.
        for process in processes:
            if process.poll() is None:
                helper.stop_owned_child(process)
        for rank, reader in enumerate(readers):
            print(f"[rank {rank}] {reader.read()}", end="", flush=True)
            reader.close()
        for writer in writers:
            writer.close()
    return {
        "reason": reason,
        "exit_codes": [p.poll() for p in processes],
        "logs": [str(p) for p in logs],
    }


def parent(args):
    cases = [
        (test, mode)
        for test, mode in CASES
        if args.only in ("all", test) and args.mode in ("all", mode)
    ]
    plan = {
        "cards": args.cards,
        "tokens": args.tokens,
        "local_heads": args.local_heads,
        "gathered_heads": args.local_heads * 2,
        "max_model_len": args.max_model_len,
        "cases": cases,
        "pattern": args.pattern,
        "scope": SCOPE,
        "warning": "Only use two idle cards. A timed-out process may leave NPU work hung.",
    }
    helper.emit("plan", **plan)
    if args.plan_only:
        return 0
    source = helper.resolve_source(args.va_source)
    vllm_source = resolve_vllm(args.vllm_source)
    base = args.output_dir or (Path("/mnt") if Path("/mnt").is_dir() else Path.cwd())
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="c8-dcp-", dir=base))
    (directory / "plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    helper.emit("logs", directory=str(directory))
    results = []
    for test, mode in cases:
        commands, envs, logs = [], [], []
        for rank in (0, 1):
            commands.append(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--cards",
                    args.cards,
                    "--rank",
                    str(rank),
                    "--only",
                    test,
                    "--mode",
                    mode,
                    "--pattern",
                    args.pattern,
                    "--va-source",
                    str(source),
                    "--vllm-source",
                    str(vllm_source),
                    "--tokens",
                    str(args.tokens),
                    "--local-heads",
                    str(args.local_heads),
                    "--max-model-len",
                    str(args.max_model_len),
                    "--timeout",
                    str(args.timeout),
                    "--rendezvous",
                    str(directory / f"{test}-{mode}.store"),
                ]
            )
            envs.append(
                dict(
                    os.environ,
                    ASCEND_RT_VISIBLE_DEVICES=args.cards,
                    PYTHONUNBUFFERED="1",
                    RANK=str(rank),
                    LOCAL_RANK=str(rank),
                    WORLD_SIZE="2",
                    LOCAL_WORLD_SIZE="2",
                )
            )
            logs.append(directory / f"{test}-{mode}-rank{rank}.log")
        helper.emit("case_start", test=test, mode=mode)
        result = supervise_pair(commands, envs, logs, args.timeout)
        complete = []
        for rank, log in enumerate(logs):
            success = False
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(item, dict)
                    and item.get("event") == "case_complete"
                    and item.get("rank") == rank
                ):
                    success = item.get("passed") is True
            complete.append(success)
        result.update(
            test=test,
            mode=mode,
            passed=result["reason"] is None
            and result["exit_codes"] == [0, 0]
            and all(complete),
        )
        results.append(result)
        if not result["passed"]:
            helper.emit(
                "stop",
                reason=result["reason"],
                warning="No further cases. Check both NPU devices before retrying.",
            )
            break
    summary = {
        "planned": len(cases),
        "completed": len(results),
        "passed": sum(r["passed"] for r in results),
        "all_passed": len(results) == len(cases) and all(r["passed"] for r in results),
        "results": results,
        "scope": SCOPE,
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    helper.emit("summary", **summary)
    helper.emit(
        "send_back",
        directory=str(directory),
        note="Send summary and BOTH ranks' logs from the failing case",
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    try:
        cli = parse_args()
        if cli.rank is not None:
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = cli.cards
            raise SystemExit(worker(cli))
        raise SystemExit(parent(cli))
    except Exception as error:  # noqa: BLE001 - retain native failures and exit nonzero
        helper.emit(
            "error",
            rank=getattr(locals().get("cli"), "rank", None),
            type=type(error).__name__,
            message=str(error),
        )
        traceback.print_exc()
        raise SystemExit(1)
