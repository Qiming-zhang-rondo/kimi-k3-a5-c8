"""One-A5 correctness diagnostic. Does not modify the installed framework.

Compare ordinary QSFA, the actual eager speculative verification wrapper,
and a CPU reference made from the SAME packed C8 bytes. No model weights,
DSpark model, graph capture, DCP collectives, or performance claims.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


def numbers(value):
    result = [int(x) for x in value.split(",")]
    if not result or min(result) <= 0:
        raise argparse.ArgumentTypeError("Use comma-separated positive integers")
    return result


def compare(torch, actual, expected):
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    finite = bool(torch.isfinite(actual).all())
    delta = (actual - expected).abs()
    allowed = 0.02 + 0.02 * expected.abs()
    passed = finite and bool((delta <= allowed).all())
    return {
        "pass": passed,
        "finite": finite,
        "max_abs": float(delta.max()) if finite else None,
        "relative_l2": float(
            (actual - expected).norm() / expected.norm().clamp_min(1e-12)
        )
        if finite
        else None,
        "bad_elements": int((~torch.isfinite(actual) | (delta > allowed)).sum()),
    }


def dense_reference(torch, query, packed, length, scale):
    # Interpret bytes independently of either attention wrapper. Rounding the
    # reconstructed latent to BF16 matches the existing native-test reference.
    data = packed.view(torch.uint8).cpu()
    latent = data[..., :512].contiguous().view(torch.float8_e4m3fn).float()
    raw = data[..., 512:640].contiguous().view(torch.bfloat16).float()
    scales = data[..., 640:].contiguous().view(torch.float32)
    if not bool(torch.isfinite(scales).all() and (scales > 0).all()):
        raise RuntimeError("Invalid packed cache scales before attention")
    latent = (latent * scales.repeat_interleave(128, -1)).bfloat16().float()[:, 0]
    keys = torch.cat((latent, raw[:, 0]), -1)
    q = query.cpu().float()
    outputs = []
    for row in range(q.shape[0]):
        visible = length - q.shape[0] + row + 1
        probabilities = (q[row] @ keys[:visible].T * scale).softmax(-1)
        outputs.append(probabilities @ latent[:visible])
    return torch.stack(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=numbers, default=[12])
    parser.add_argument("--queries", type=numbers, default=[1, 4, 9])
    parser.add_argument("--lengths", type=numbers, default=[127, 129, 1024])
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Logical device AFTER visibility filtering",
    )
    parser.add_argument("--seed", type=int, default=92029)
    args = parser.parse_args()

    import torch
    import torch_npu
    from vllm_ascend.attention import kimi_k3_sfa_c8 as ordinary
    from vllm_ascend.attention import kimi_k3_sfa_c8_graph as rows
    from vllm_ascend.attention import mla_v1
    from vllm_ascend.device.device_config import check_ascend_device_type, is_950

    torch.set_num_threads(1)
    if not torch.npu.is_available():
        raise RuntimeError("A real A5 is required; CPU mocks are not accepted")
    torch.npu.set_device(args.device)
    check_ascend_device_type()
    if not is_950():
        raise RuntimeError("This diagnostic requires the A5/950 build")
    ordinary.require_kimi_k3_sfa_c8_ops()
    verify = getattr(mla_v1.AscendMLAImpl, "_kimi_k3_c8_verify_attention", None)
    if verify is None:
        raise RuntimeError(
            "Installed VA has no C8 speculative verification wrapper; check the test patch version"
        )
    sources = {}
    for module in (ordinary, rows, mla_v1):
        path = Path(module.__file__)
        sources[module.__name__] = {
            "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    print(
        json.dumps(
            {
                "event": "environment",
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "device": args.device,
                "args": vars(args),
                "sources": sources,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    device = torch.device(f"npu:{args.device}")
    scale = 192**-0.5
    context = SimpleNamespace(
        dcp_size=1,
        scale=scale,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1)
        ),
    )
    failed = cases = 0
    with torch.inference_mode():
        for heads in args.heads:
            for length in args.lengths:
                for count in args.queries:
                    if count > length:
                        continue
                    cases += 1
                    case = {"heads": heads, "queries": count, "kv_length": length}
                    try:
                        generator = torch.Generator().manual_seed(
                            args.seed + length * 31 + heads * 7 + count
                        )
                        latent = torch.randn(length, 1, 512, generator=generator)
                        latent *= torch.tensor(
                            [0.125, 0.5, 2.0, 8.0]
                        ).repeat_interleave(128)
                        latent = latent.bfloat16().to(device)
                        raw = (
                            torch.randn(length, 1, 64, generator=generator)
                            .bfloat16()
                            .to(device)
                        )
                        query = (
                            (torch.randn(count, heads, 576, generator=generator) * 0.2)
                            .bfloat16()
                            .to(device)
                        )
                        packed = ordinary.pack_kimi_k3_sfa_c8(latent, raw)
                        # Reserve extra capacity for the intentionally loose CPU
                        # length bound. Poison unwritten rows to expose overreads.
                        bound = length + 3
                        pages = math.ceil(bound / 128)
                        cache = torch.full(
                            (pages + 2, 128, 1, 656),
                            127,
                            dtype=torch.uint8,
                            device=device,
                        ).view(torch.float8_e4m3fn)
                        table_cpu = torch.arange(pages, 0, -1, dtype=torch.int32).view(
                            1, -1
                        )
                        positions = torch.arange(length)
                        slots = (
                            table_cpu[0, positions // 128].long() * 128
                            + positions % 128
                        ).to(device)
                        table = table_cpu.to(device)
                        ordinary.scatter_kimi_k3_sfa_c8(packed, slots, cache)
                        before = cache.view(torch.uint8).cpu().clone()
                        slots_cpu = slots.cpu()
                        stored = before[slots_cpu // 128, slots_cpu % 128]
                        if not torch.equal(stored, packed.view(torch.uint8).cpu()):
                            raise RuntimeError(
                                "Scatter byte check failed before either attention reader"
                            )
                        q_latent, q_raw = query[..., :512], query[..., 512:]
                        reference = dense_reference(torch, query, packed, length, scale)
                        old = (
                            ordinary.kimi_k3_sfa_c8_attention(
                                q_latent,
                                q_raw,
                                cache,
                                table,
                                [0, count],
                                [length],
                                scale,
                            )
                            .float()
                            .cpu()
                        )
                        # Exercise the REAL eager wrapper, including the differing
                        # CPU upper bound. DCP=1 => return_softmax_lse=False.
                        new = (
                            verify(
                                context,
                                q_latent,
                                q_raw,
                                cache,
                                table,
                                [0, count],
                                [bound],
                                torch.tensor(
                                    [length], dtype=torch.int32, device=device
                                ),
                            )
                            .float()
                            .cpu()
                        )
                        checks = {
                            "ordinary_vs_reference": compare(torch, old, reference),
                            "verify_vs_reference": compare(torch, new, reference),
                            "verify_vs_ordinary": compare(torch, new, old),
                        }
                        cache_unchanged = torch.equal(
                            before, cache.view(torch.uint8).cpu()
                        )
                        passed = cache_unchanged and all(
                            item["pass"] for item in checks.values()
                        )
                        failed += not passed
                        print(
                            json.dumps(
                                {
                                    "event": "case",
                                    **case,
                                    "pass": passed,
                                    "cache_unchanged": cache_unchanged,
                                    "checks": checks,
                                }
                            ),
                            flush=True,
                        )
                    except Exception as exc:
                        # An asynchronous NPU error can poison the process; stop
                        # immediately rather than reporting misleading later cases.
                        print(
                            json.dumps({"event": "error", **case, "error": repr(exc)}),
                            flush=True,
                        )
                        raise
    print(
        json.dumps(
            {
                "event": "summary",
                "cases": cases,
                "failed": failed,
                "scope": "single-request eager QSFA correctness only; not DSpark/model acceptance",
            }
        ),
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
