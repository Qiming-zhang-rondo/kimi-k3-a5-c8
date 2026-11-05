# SPDX-License-Identifier: Apache-2.0
"""Standalone FIA Prefill probe. Simulated shards, NOT distributed acceptance."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--queries", type=int, default=33)
    parser.add_argument("--history", type=int, default=257)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--shards", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.queries < 1 or args.history < 0:
        parser.error("queries must be positive and history nonnegative")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    os.environ.pop("ASCEND_DEVICE_ID", None)
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    path = Path(__file__).resolve().parents[6] / "vllm_ascend/attention/kimi_k3_qfm.py"
    spec = importlib.util.spec_from_file_location("qfm_fia_probe", path)
    qfm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = qfm
    spec.loader.exec_module(qfm)
    torch.manual_seed(719)
    q_cpu = torch.randn(args.queries, args.heads, 576).to(torch.bfloat16)
    kv_cpu = torch.randn(args.history + args.queries, 576).to(torch.bfloat16)
    q, kv = q_cpu.npu(), kv_cpu.npu()
    scale = torch.ones(1, device="npu")
    result = None
    for rank in range(args.shards):
        local = [p for p in range(len(kv_cpu)) if (p // 16) % args.shards == rank]
        local_history = sum(p < args.history for p in local)
        pages = max(1, (len(local) + 127) // 128)
        stride, offset = 128 * 576 + 256, 256
        backing = torch.full((offset + pages * stride + 256,), 0x5A, dtype=torch.uint8, device="npu")
        cache = qfm.cache_view(backing[offset : offset + pages * stride], pages, stride)
        table = torch.arange(pages - 1, -1, -1, dtype=torch.int32, device="npu")
        position = torch.arange(len(local), device="npu")
        slots = table[position // 128].long() * 128 + position % 128
        if local:
            qfm.write_cache(kv[torch.tensor(local, device="npu", dtype=torch.long)], slots, cache, scale)
        before = backing.clone()
        impl = SimpleNamespace(dcp_enabled=args.shards > 1, dcp_rank=rank, qfm_scale=scale, scale=1 / 24)
        batch = SimpleNamespace(fia_requests=((0, args.queries, local_history, table),))
        out = torch.empty(args.queries, args.heads, 512, dtype=torch.bfloat16, device="npu")
        lse = torch.empty(args.queries, args.heads, 1, device="npu")
        qfm.fill_fia_prefill(impl, q, kv[args.history :], cache, batch, out, lse)
        torch.npu.synchronize()
        if not torch.equal(before, backing):
            raise AssertionError("FIA reader modified cache/guards")
        result = (out, lse) if result is None else qfm.merge_fia_parts(result, (out, lse))
    # Independent CPU FP32 reference: quantized history, original BF16 current.
    keys = kv_cpu.float()
    keys[: args.history] = kv_cpu[: args.history].to(torch.float8_e4m3fn).float()
    rows = sorted({0, args.queries // 2, args.queries - 1})
    scores = torch.einsum("thd,kd->thk", q_cpu[rows].float(), keys) / 24
    invalid = torch.arange(len(keys))[None, :] > (args.history + torch.tensor(rows))[:, None]
    scores.masked_fill_(invalid[:, None], -float("inf"))
    reference = torch.einsum("thk,kd->thd", scores.softmax(-1), keys[:, :512])
    actual = result[0].cpu()[rows].float()
    actual_lse = result[1].cpu()[rows, :, 0]
    torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)
    torch.testing.assert_close(actual_lse, scores.logsumexp(-1), atol=0.005, rtol=0.001)
    print(
        json.dumps(
            {
                "pass": True,
                "queries": args.queries,
                "history": args.history,
                "heads": args.heads,
                "simulated_shards": args.shards,
                "sampled_rows": rows,
                "max_abs": (actual - reference).abs().max().item(),
                "scope": "native FIA + cache read; NOT HCCL, graph, Prolog or model acceptance",
            }
        )
    )


if __name__ == "__main__":
    main()
