#!/usr/bin/env python3
"""A5 K3 Decode-only BF16 FIA vs C8 QSFA benchmark; no model weights.

Run inside the patched VA environment. No fallback if native operators fail.
Timing is operator-call latency, not isolated device kernel duration: use
--profile-dir for kernel attribution. No prefill/MLAPO/whole-model claims.
"""

import argparse
import importlib.metadata
import json
import math
import statistics
import time
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384, 55000])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    p.add_argument(
        "--heads", type=int, default=12, help="Local heads; K3 TP8=12, TP4=24, TP2=48"
    )
    p.add_argument(
        "--device", type=int, default=0, help="Logical NPU index; one NPU only"
    )
    p.add_argument(
        "--scale",
        type=float,
        default=192**-0.5,
        help="Attention softmax scale; override to match deployed model",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--profile-dir", type=Path)
    p.add_argument(
        "--graph",
        action="store_true",
        help="Also time fixed-shape native FIA/QSFA graph replay (not MRV2)",
    )
    p.add_argument("--output", type=Path, default=Path("k3-c8-decode.jsonl"))
    a = p.parse_args(argv)
    if min(a.lengths + a.batches + [a.heads, a.iterations, a.repeats]) <= 0:
        p.error("Lengths, batches, heads, iterations and repeats must be positive")
    if a.heads > 128 or a.warmup < 0 or not math.isfinite(a.scale) or a.scale <= 0:
        p.error("Require heads <=128, warmup >=0 and finite positive scale")
    return a


def summary(samples):
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def measure(torch, operations, a):
    """Rotate operation order to reduce ordering/temperature bias."""
    samples = {name: {"event": [], "wall": []} for name in operations}
    for op in operations.values():
        for _ in range(a.warmup):
            op()
    torch.npu.synchronize()
    names = list(operations)
    for repeat in range(a.repeats):
        for name in names[repeat % len(names) :] + names[: repeat % len(names)]:
            start, end = (
                torch.npu.Event(enable_timing=True),
                torch.npu.Event(enable_timing=True),
            )
            torch.npu.synchronize()
            begin = time.perf_counter()
            start.record()
            for _ in range(a.iterations):
                operations[name]()
            end.record()
            torch.npu.synchronize()
            samples[name]["wall"].append(
                (time.perf_counter() - begin) * 1000 / a.iterations
            )
            samples[name]["event"].append(start.elapsed_time(end) / a.iterations)
    return {
        name: {kind: summary(values) for kind, values in data.items()}
        for name, data in samples.items()
    }


def run_case(torch, torch_npu, c8, a, batch, length):
    block = 128
    pages = math.ceil(length / block)
    total_pages = batch * pages
    torch.manual_seed(1234)
    # CPU-origin tensors make independent references deterministic.
    latent = torch.randn(batch, pages * block, 1, 512).to(torch.bfloat16)
    raw = torch.randn(batch, pages * block, 1, 64).to(torch.bfloat16)
    q = torch.randn(batch, a.heads, 512).to(torch.bfloat16)
    qr = torch.randn(batch, a.heads, 64).to(torch.bfloat16)
    k = latent.reshape(total_pages, block, 1, 512).npu()
    kr = raw.reshape(total_pages, block, 1, 64).npu()
    qd, qrd = q.npu(), qr.npu()
    table = torch.arange(total_pages, dtype=torch.int32).reshape(batch, pages).npu()
    packed = c8.pack_kimi_k3_sfa_c8(k.reshape(-1, 1, 512), kr.reshape(-1, 1, 64))
    cache = packed.reshape(total_pages, block, 1, c8.PACKED_DIM)
    # Match VA BF16 ND Decode padding; QSFA has a distinct supported-head set.
    fia_heads = 1 << (a.heads - 1).bit_length()
    sfa_heads = c8.get_kimi_k3_sfa_c8_padded_heads(a.heads)
    fq = torch.zeros(batch, fia_heads, 1, 512, dtype=torch.bfloat16, device=qd.device)
    fr = torch.zeros(batch, fia_heads, 1, 64, dtype=torch.bfloat16, device=qd.device)
    fq[:, : a.heads, 0].copy_(qd)
    fr[:, : a.heads, 0].copy_(qrd)
    sq = torch.zeros(batch, sfa_heads, 576, dtype=torch.bfloat16, device=qd.device)
    sq[:, : a.heads, :512].copy_(qd)
    sq[:, : a.heads, 512:].copy_(qrd)
    indices = torch.arange(length, dtype=torch.int32, device=qd.device).reshape(
        1, 1, -1
    )
    indices = indices.expand(batch, 1, length).contiguous()
    ends = torch.arange(1, batch + 1, dtype=torch.int32, device=qd.device)
    lens = torch.full((batch,), length, dtype=torch.int32, device=qd.device)
    sinks = torch.full(
        (sfa_heads,), -float("inf"), dtype=torch.float32, device=qd.device
    )
    native_sfa = c8.require_kimi_k3_sfa_c8_ops()
    fk, fkr = k.view(total_pages, 1, block, 512), kr.view(total_pages, 1, block, 64)
    lengths = [length] * batch
    starts = list(range(batch + 1))

    def fia():
        return torch_npu.npu_fused_infer_attention_score_v2(
            fq,
            fk,
            fk,
            query_rope=fr,
            key_rope=fkr,
            num_query_heads=fia_heads,
            num_key_value_heads=1,
            input_layout="BNSD_NBSD",
            sparse_mode=0,
            atten_mask=None,
            softmax_scale=a.scale,
            block_table=table,
            block_size=block,
            actual_seq_qlen=None,
            actual_seq_kvlen=lengths,
        )

    def qsfa():
        return native_sfa(
            sq,
            cache,
            cache,
            indices,
            a.scale,
            key_quant_mode=2,
            value_quant_mode=2,
            block_table=table,
            actual_seq_lengths_query=ends,
            actual_seq_lengths_kv=lens,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=128,
            rope_head_dim=64,
            sinks=sinks,
            return_softmax_lse=False,
        )

    def wrapper():
        return c8.kimi_k3_sfa_c8_attention(
            qd, qrd, cache, table, starts, lengths, a.scale
        )

    # Actual current-token write positions; keep token values constant across iterations.
    new_k = latent[:, length - 1].contiguous().npu()
    new_r = raw[:, length - 1].contiguous().npu()
    slots = (torch.arange(batch, dtype=torch.int64) * pages * block + length - 1).npu()

    def bf16_write():
        torch_npu.npu_scatter_pa_kv_cache(
            key=new_k,
            value=new_r,
            key_cache=k,
            value_cache=kr,
            slot_mapping=slots,
            cache_mode="Norm",
        )

    def c8_write():
        c8.scatter_kimi_k3_sfa_c8(c8.pack_kimi_k3_sfa_c8(new_k, new_r), slots, cache)

    # Writer and numerical checks precede all performance reporting.
    bf16_write()
    c8_write()
    fo = fia()[0]
    so = qsfa()[0]
    if tuple(fo.shape) != (fia_heads, batch, 1, 512):
        raise RuntimeError(
            f"Unexpected FIA NBSD output {fo.shape}; refuse ambiguous layout comparison"
        )
    if tuple(so.shape) != (batch, sfa_heads, 512):
        raise RuntimeError(f"Unexpected QSFA output {so.shape}")
    bfout = fo[: a.heads, :, 0].transpose(0, 1).float().cpu()
    c8out = so[:, : a.heads].float().cpu()
    wrapped = wrapper().float().cpu()
    torch.testing.assert_close(wrapped, c8out, atol=0.01, rtol=0.03)
    physical = cache.view(torch.uint8).cpu().reshape(batch, pages * block, 656)
    quant = physical[..., :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = physical[..., 640:].contiguous().view(torch.float32)
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise RuntimeError("Invalid C8 scale")
    decoded = (quant * scales.repeat_interleave(128, dim=-1)).to(torch.bfloat16).float()
    decoded_raw = physical[..., 512:640].contiguous().view(torch.bfloat16)
    torch.testing.assert_close(decoded_raw, raw.squeeze(2), atol=0, rtol=0)
    errors = []
    # Independent dense FP32 references, first/last request and first/last head.
    # Never form a prefill LxL matrix; references are outside timed regions.
    for b in sorted({0, batch - 1}):
        for h in sorted({0, a.heads - 1}):
            for label, keys, actual in [
                ("bf16", latent[b, :length, 0].float(), bfout),
                ("c8", decoded[b, :length], c8out),
            ]:
                logits = (
                    keys @ q[b, h].float()
                    + raw[b, :length, 0].float() @ qr[b, h].float()
                )
                ref = (logits * a.scale).softmax(-1) @ keys
                rel = float((actual[b, h] - ref).norm() / ref.norm().clamp_min(1e-8))
                if not math.isfinite(rel) or rel > 0.05:
                    raise RuntimeError(
                        f"{label} reference failed b={b} h={h} relative_L2={rel}"
                    )
                errors.append(
                    {"path": label, "batch": b, "head": h, "relative_l2": rel}
                )

    operations = {
        "bf16_fia_prebuilt": fia,
        "c8_qsfa_prebuilt": qsfa,
        "c8_eager_wrapper": wrapper,
        "bf16_cache_write": bf16_write,
        "c8_quant_pack_write": c8_write,
    }
    timing = measure(torch, operations, a)
    graph_timing = None
    graph_owners = []
    if a.graph:
        graph_operations = {}
        for name, op in (("bf16_fia_graph", fia), ("c8_qsfa_graph", qsfa)):
            for _ in range(max(3, a.warmup)):
                op()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            stream = torch.npu.Stream()
            with torch.npu.graph(graph, stream=stream):
                graph_output = op()
            torch.npu.current_stream().wait_stream(stream)
            graph.replay()
            torch.npu.synchronize()
            expected = fo if name.startswith("bf16") else so
            torch.testing.assert_close(graph_output[0], expected, atol=0.01, rtol=0.03)
            # Retain outputs and capture stream through timing and profiling.
            graph_owners.append((graph, stream, graph_output))
            graph_operations[name] = graph.replay
        graph_timing = measure(torch, graph_operations, a)
        operations.update(graph_operations)
    if a.profile_dir:
        folder = a.profile_dir / f"b{batch}_kv{length}_h{a.heads}"
        folder.mkdir(parents=True, exist_ok=True)
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            record_shapes=True,
            with_stack=False,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(folder)),
        ) as prof:
            for name, op in operations.items():
                with torch.profiler.record_function(name):
                    for _ in range(5):
                        op()
                    torch.npu.synchronize()
                prof.step()
    base = timing["bf16_fia_prebuilt"]["event"]["median_ms"]
    test = timing["c8_qsfa_prebuilt"]["event"]["median_ms"]
    return {
        "batch": batch,
        "kv_length": length,
        "query_length_per_request": 1,
        "local_heads": a.heads,
        "fia_heads": fia_heads,
        "qsfa_heads": sfa_heads,
        "scale": a.scale,
        "block_size": block,
        "cache_stride": list(cache.stride()),
        "bf16_bytes_per_token": 1152,
        "c8_bytes_per_token": 656,
        "reference_checks": errors,
        "c8_vs_bf16_output_relative_l2": float(
            (c8out - bfout).norm() / bfout.norm().clamp_min(1e-8)
        ),
        "latency": timing,
        "fixed_shape_graph_latency": graph_timing,
        "qsfa_vs_fia_latency_change_pct": (test / base - 1) * 100,
        "interpretation": "positive change = regression; eager operator calls, NOT graph replay or TTFT",
    }


def main():
    a = parse_args()
    # Lazy imports allow --help/argument checks on machines without torch_npu.
    import torch
    import torch_npu
    from vllm_ascend.attention import kimi_k3_sfa_c8 as c8

    torch.npu.set_device(a.device)
    c8.require_kimi_k3_sfa_c8_ops()
    versions = {}
    for package in (
        "torch",
        "torch-npu",
        "vllm",
        "vllm-ascend",
        "cann-ops-transformer",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite earlier evidence.
    with a.output.open("x") as out, torch.inference_mode():
        header = {
            "device": torch.npu.get_device_name(a.device),
            "versions": versions,
            "scope": "synthetic No-RoPE latent512+raw64 Decode, no weights or projections",
            "warmup": a.warmup,
            "iterations": a.iterations,
            "repeats": a.repeats,
            "timing_note": "NPU events include stream idle/launch gaps; profiler is required for kernel-only duration",
        }
        out.write(json.dumps(header, ensure_ascii=False) + "\n")
        print(json.dumps(header, ensure_ascii=False), flush=True)
        for length in a.lengths:
            for batch in a.batches:
                result = run_case(torch, torch_npu, c8, a, batch, length)
                line = json.dumps(result, ensure_ascii=False)
                out.write(line + "\n")
                out.flush()
                print(line, flush=True)
                torch.npu.empty_cache()


if __name__ == "__main__":
    main()
