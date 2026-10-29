"""One-A5 address/stride diagnostic; no model weights or service required.

Executes the installed C8 scatter and both installed eager QSFA readers.
Writer and reader tests are separated: readers receive independently seeded
cache bytes, so matching writer/reader address bugs cannot cancel each other.
This is not DSpark end-to-end acceptance, Prolog, graph, or performance testing.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import math
from itertools import pairwise
from pathlib import Path
from types import ModuleType, SimpleNamespace

PACKED = 656
BLOCK = 128
SENTINEL = 127


def report(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def isolated_function(path, name, namespace, cls=None):
    source = path.read_bytes()
    tree = ast.parse(source, filename=str(path))
    body = tree.body
    if cls:
        classes = [n for n in body if isinstance(n, ast.ClassDef) and n.name == cls]
        if len(classes) != 1:
            raise RuntimeError(f"No unique installed class {cls}: {path}")
        body = classes[0].body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1 or nodes[0].decorator_list:
        raise RuntimeError(f"No unique undecorated installed function {name}: {path}")
    namespace = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace[name], {
        "file": str(path),
        "sha256": hashlib.sha256(source).hexdigest(),
        "function": name,
        "line": nodes[0].lineno,
        "load_mode": "unchanged_installed_function_ast",
    }


def load_sources(root, torch):
    """Avoid package initialization and the DeviceOperator/MoE import cycle."""
    ordinary_path = root / "vllm_ascend/attention/kimi_k3_sfa_c8.py"
    rows_path = ordinary_path.with_name("kimi_k3_sfa_c8_graph.py")
    modules, sources = {}, {}
    for name, path in (("ordinary", ordinary_path), ("rows", rows_path)):
        source = path.read_bytes()
        tree = ast.parse(source, filename=str(path))
        namespace = {"torch": torch}
        if name == "rows":
            imports = [
                n
                for n in tree.body
                if isinstance(n, ast.ImportFrom)
                and n.module == "vllm_ascend.attention.kimi_k3_sfa_c8"
            ]
            if len(imports) != 1:
                raise RuntimeError(
                    "Installed row reader dependency changed; update diagnostic"
                )
            for alias in imports[0].names:
                namespace[alias.asname or alias.name] = getattr(
                    modules["ordinary"], alias.name
                )
            tree.body.remove(imports[0])
        for node in ast.walk(tree):
            names = (
                [node.module]
                if isinstance(node, ast.ImportFrom)
                else (
                    [a.name for a in node.names] if isinstance(node, ast.Import) else []
                )
            )
            if any(n and n.startswith("vllm") for n in names):
                raise RuntimeError(f"Unexpected framework import in isolated {path}")
        module = ModuleType(f"c8_address_diagnostic_{name}")
        module.__dict__.update(namespace)
        exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102
        modules[name] = module
        sources[name] = {
            "file": str(path),
            "sha256": hashlib.sha256(source).hexdigest(),
        }
    rows = modules["rows"]
    verify, sources["verify"] = isolated_function(
        ordinary_path.with_name("mla_v1.py"),
        "_kimi_k3_c8_verify_attention",
        {
            "torch": torch,
            "KimiK3C8DecodeState": rows.KimiK3C8DecodeState,
            "kimi_k3_sfa_c8_decode": rows.kimi_k3_sfa_c8_decode,
        },
        cls="AscendMLAImpl",
    )
    view, sources["page_view"] = isolated_function(
        root / "vllm_ascend/worker/v2/attn_utils.py",
        "_adjust_dsv4_kv_layout",
        {
            "torch": torch,
            "get_dtype_size": lambda dtype: torch.empty((), dtype=dtype).element_size(),
        },
    )
    return modules["ordinary"], verify, view, sources


def reference(torch, query, packed, length, scale):
    data = packed.view(torch.uint8).cpu()
    latent = data[..., :512].contiguous().view(torch.float8_e4m3fn).float()
    raw = data[..., 512:640].contiguous().view(torch.bfloat16).float()
    scales = data[..., 640:].contiguous().view(torch.float32)
    if not bool(torch.isfinite(scales).all() and (scales > 0).all()):
        raise RuntimeError("Quantizer generated invalid group scales")
    latent = (latent * scales.repeat_interleave(128, -1)).bfloat16().float()[:, 0]
    keys = torch.cat((latent, raw[:, 0]), -1)
    q = query.cpu().float()
    result = []
    for row in range(q.shape[0]):
        visible = length - q.shape[0] + row + 1
        result.append(
            (q[row] @ keys[:visible].T * scale).softmax(-1) @ latent[:visible]
        )
    result = torch.stack(result)
    if not bool(torch.isfinite(result).all()) or float(result.abs().max()) < 0.1:
        raise RuntimeError(
            "Reference is degenerate; zero-output detection would be meaningless"
        )
    return result


def compare(torch, actual, expected):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    finite = bool(torch.isfinite(actual).all())
    delta = (actual - expected).abs()
    allowed = 0.02 + 0.02 * expected.abs()
    all_zero = bool((actual == 0).all())
    return {
        "pass": finite and not all_zero and bool((delta <= allowed).all()),
        "finite": finite,
        "all_zero": all_zero,
        "max_abs": float(delta.max()) if finite else None,
        "relative_l2": float(
            (actual - expected).norm() / expected.norm().clamp_min(1e-12)
        )
        if finite
        else None,
        "bad_elements": int((~torch.isfinite(actual) | (delta > allowed)).sum()),
    }


def byte_check(torch, actual, expected):
    actual = actual.cpu()
    differences = actual != expected
    count = int(differences.sum())
    return {
        "pass": count == 0,
        "different_bytes": count,
        "first_bad_byte": int(differences.nonzero()[0]) if count else None,
    }


def cache_fixture(torch, view, blocks, page_bytes, offset, device):
    # The suffix/prefix are NOT inside the logical cache. Their sentinels
    # detect an operator incorrectly treating storage base as tensor data_ptr.
    pool = torch.full(
        (offset + blocks * page_bytes + 256,),
        SENTINEL,
        dtype=torch.uint8,
        device=device,
    )
    raw = pool[offset : offset + blocks * page_bytes]
    cache = view(raw, [(blocks, BLOCK, 1, PACKED)], [torch.float8_e4m3fn], page_bytes)[
        0
    ]
    if cache.storage_offset() != offset or cache.stride(0) != page_bytes:
        raise RuntimeError(
            "Installed page-view helper produced incorrect offset/stride"
        )
    return pool, raw, cache


def write_expected(
    torch, pool, packed, slots, page_bytes, offset, width=PACKED, component=0
):
    expected = pool.cpu().clone()
    data = packed.contiguous().view(torch.uint8).reshape(len(slots), width).cpu()
    for row, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        begin = offset + slot // BLOCK * page_bytes + component + slot % BLOCK * width
        expected[begin : begin + width] = data[row]
    return expected


def inputs(torch, length, queries, heads, device, seed):
    generator = torch.Generator().manual_seed(seed + length * 31 + queries * 7 + heads)
    # Positive, nonconstant latent guarantees a meaningful nonzero output.
    # Different group magnitudes also expose incorrect scale-byte addresses.
    latent = torch.randn(length, 1, 512, generator=generator) * 0.2 + 1.5
    latent *= torch.tensor([0.125, 0.5, 2.0, 8.0]).repeat_interleave(128)
    raw = torch.randn(length, 1, 64, generator=generator) * 0.25
    query = torch.randn(queries, heads, 576, generator=generator) * 0.1
    return (*(x.bfloat16().to(device) for x in (latent, raw, query)),)


def readers(torch, ordinary, verify, cache, table, query, expected, length):
    scale, count = 192**-0.5, query.shape[0]
    context = SimpleNamespace(
        dcp_size=1,
        scale=scale,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1)
        ),
    )
    q_latent, q_raw = query[..., :512], query[..., 512:]
    old = ordinary.kimi_k3_sfa_c8_attention(
        q_latent, q_raw, cache, table, [0, count], [length], scale
    )
    new = verify(
        context,
        q_latent,
        q_raw,
        cache,
        table,
        [0, count],
        [length + 3],
        torch.tensor([length], dtype=torch.int32, device=query.device),
    )
    return {
        "ordinary_vs_reference": compare(torch, old, expected),
        "verify_vs_reference": compare(torch, new, expected),
        "verify_vs_ordinary": compare(torch, new, old),
    }


def run_address_case(
    torch, ordinary, verify, view, device, name, page_bytes, offset, args, length, count
):
    pages = math.ceil((length + 3) / BLOCK)
    blocks = pages + 2
    pool, _, cache = cache_fixture(torch, view, blocks, page_bytes, offset, device)
    latent, raw, query = inputs(torch, length, count, args.heads, device, args.seed)
    packed = ordinary.pack_kimi_k3_sfa_c8(latent, raw)
    table_cpu = torch.arange(pages, 0, -1, dtype=torch.int32).view(1, -1)
    positions = torch.arange(length)
    slots_cpu = table_cpu[0, positions // BLOCK].long() * BLOCK + positions % BLOCK
    expected_pool = write_expected(torch, pool, packed, slots_cpu, page_bytes, offset)
    ordinary.scatter_kimi_k3_sfa_c8(packed, slots_cpu.to(device), cache)
    writer = byte_check(torch, pool, expected_pool)
    # Independently initialize the entire backing buffer from host-computed
    # byte addresses, not by reusing the tested scatter or the cache view.
    pool.copy_(expected_pool.to(device))
    expected_output = reference(torch, query, packed, length, 192**-0.5)
    checks = readers(
        torch,
        ordinary,
        verify,
        cache,
        table_cpu.to(device),
        query,
        expected_output,
        length,
    )
    guard = byte_check(torch, pool, expected_pool)
    passed = (
        writer["pass"] and guard["pass"] and all(c["pass"] for c in checks.values())
    )
    report(
        "address_case",
        layout=name,
        length=length,
        queries=count,
        heads=args.heads,
        storage_offset=cache.storage_offset(),
        byte_stride=list(cache.stride()),
        page_padding_bytes=page_bytes - BLOCK * PACKED,
        writer=writer,
        reader_storage_unchanged=guard,
        checks=checks,
        **{"pass": passed},
    )
    return passed


def load_shared_plan(root, torch):
    from collections import Counter
    from dataclasses import dataclass

    from vllm.v1.kv_cache_interface import (
        KVCacheGroupSpec,
        KVCacheSpec,
        MambaSpec,
        MLAAttentionSpec,
        UniformTypeKVCacheSpecs,
    )

    planner, source = isolated_function(
        root / "vllm_ascend/core/kimi_k3_cache_layout.py",
        "get_kimi_k3_c8_cache_layout",
        {
            "Counter": Counter,
            "KVCacheGroupSpec": KVCacheGroupSpec,
            "KVCacheSpec": KVCacheSpec,
            "MambaSpec": MambaSpec,
            "UniformTypeKVCacheSpecs": UniformTypeKVCacheSpecs,
        },
    )

    # A small real vLLM MLA spec subclass carries the Ascend marker. The
    # planner sees actual immutable specs, not a replacement planning formula.
    @dataclass(frozen=True, kw_only=True)
    class DiagnosticC8Spec(MLAAttentionSpec):
        cache_sparse_sfa_c8: bool = True

    page = BLOCK * (512 + 64) * 2
    main = DiagnosticC8Spec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=PACKED,
        dtype=torch.float8_e4m3fn,
        page_size_padded=page,
    )
    draft = MLAAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        non_causal_multi_token_decode=True,
    )
    state = MambaSpec(
        block_size=BLOCK,
        shapes=((16, 16),),
        dtypes=(torch.float32,),
        page_size_padded=page,
    )
    groups = [
        KVCacheGroupSpec(["state"], state),
        KVCacheGroupSpec(["target"], main),
        KVCacheGroupSpec(["draft"], draft),
    ]
    result = planner(groups)
    if result is None:
        raise RuntimeError(
            "Installed C8 shared-pool fix did not activate for the hybrid fixture"
        )
    return result, {"state": state, "target": main, "draft": draft}, source


def run_shared_pool_case(
    torch, torch_npu, ordinary, verify, view, device, args, plan, specs
):
    """Native use of exact installed planner offsets, not full runner startup."""
    length, count, blocks = 257, 4, 6
    guard = 256
    bytes_per_block, layer_offsets = plan
    pool = torch.full(
        (guard + blocks * bytes_per_block + guard,),
        SENTINEL,
        dtype=torch.uint8,
        device=device,
    )
    ranges = {
        name: (
            guard + layer_offsets[name] * blocks,
            guard + layer_offsets[name] * blocks + spec.page_size_bytes * blocks,
        )
        for name, spec in specs.items()
    }
    ranges_sorted = sorted(ranges.values())
    if any(first[1] > second[0] for first, second in pairwise(ranges_sorted)):
        raise RuntimeError(
            "Installed planner assigned overlapping incompatible cache regions"
        )
    offset, target_end = ranges["target"]
    page_bytes = specs["target"].page_size_bytes
    cache = view(
        pool[offset:target_end],
        [(blocks, BLOCK, 1, PACKED)],
        [torch.float8_e4m3fn],
        page_bytes,
    )[0]
    draft_start, draft_end = ranges["draft"]
    k_bytes, v_bytes = blocks * BLOCK * 512 * 2, blocks * BLOCK * 64 * 2
    # Preserve MRV2's existing whole-pool contiguous BF16 K and V planes.
    draft_begin = draft_end - k_bytes - v_bytes
    if draft_begin < draft_start:
        raise RuntimeError("Draft cache region is too small")
    draft_k = (
        pool[draft_begin : draft_begin + k_bytes]
        .view(torch.bfloat16)
        .view(blocks, BLOCK, 1, 512)
    )
    draft_v = (
        pool[draft_begin + k_bytes : draft_end]
        .view(torch.bfloat16)
        .view(blocks, BLOCK, 1, 64)
    )
    latent, raw, query = inputs(torch, length, count, args.heads, device, args.seed)
    packed = ordinary.pack_kimi_k3_sfa_c8(latent, raw)
    table_cpu = torch.tensor([[5, 3, 1]], dtype=torch.int32)
    positions = torch.arange(length)
    target_slots = table_cpu[0, positions // BLOCK].long() * BLOCK + positions % BLOCK
    expected = write_expected(torch, pool, packed, target_slots, page_bytes, offset)
    ordinary.scatter_kimi_k3_sfa_c8(packed, target_slots.to(device), cache)
    target_write = byte_check(torch, pool, expected)
    pool.copy_(expected.to(device))
    draft_slots = torch.tensor(
        [0, BLOCK * 1 + 127, BLOCK * 5 + 1, BLOCK * 4], dtype=torch.int64
    )
    draft_key = (
        torch.arange(4 * 512).reshape(4, 1, 512).remainder(29).bfloat16().to(device)
    )
    draft_value = (
        (torch.arange(4 * 64).reshape(4, 1, 64).remainder(17) + 1).bfloat16().to(device)
    )
    expected = write_expected(
        torch, pool, draft_key, draft_slots, BLOCK * 1024, draft_begin, width=1024
    )
    expected = write_expected(
        torch,
        expected,
        draft_value,
        draft_slots,
        BLOCK * 128,
        draft_begin + k_bytes,
        width=128,
    )
    torch_npu.npu_scatter_pa_kv_cache(
        key=draft_key,
        value=draft_value,
        key_cache=draft_k,
        value_cache=draft_v,
        slot_mapping=draft_slots.to(device),
        cache_mode="Norm",
    )
    draft_write = byte_check(torch, pool, expected)
    state_start, state_end = ranges["state"]
    pool[state_start:state_end].fill_(43)
    expected[state_start:state_end].fill_(43)
    state_write = byte_check(torch, pool, expected)
    # Do not repair the native writer's result before reading: this phase must
    # detect draft writes changing the target's actual attention result.
    checks = readers(
        torch,
        ordinary,
        verify,
        cache,
        table_cpu.to(device),
        query,
        reference(torch, query, packed, length, 192**-0.5),
        length,
    )
    before_rewrite = pool.cpu().clone()
    ordinary.scatter_kimi_k3_sfa_c8(packed, target_slots.to(device), cache)
    target_rewrite = byte_check(torch, pool, before_rewrite)
    passed = all(
        x["pass"]
        for x in (
            target_write,
            draft_write,
            state_write,
            target_rewrite,
            *checks.values(),
        )
    )
    report(
        "segregated_shared_pool",
        scope="installed_layout_planner_and_native_writers; not_full_runner_allocator",
        page_bytes=page_bytes,
        offset=offset,
        byte_regions=ranges,
        target_blocks=[5, 3, 1],
        draft_blocks=[0, 1, 5, 4],
        draft_k_byte_offset=draft_k.storage_offset() * 2,
        draft_v_byte_offset=draft_v.storage_offset() * 2,
        target_write=target_write,
        draft_write=draft_write,
        state_write=state_write,
        target_rewrite=target_rewrite,
        checks=checks,
        **{"pass": passed},
    )
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--va-root",
        type=Path,
        help="Source root containing vllm_ascend; default installed package",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Logical NPU ID after visibility filtering",
    )
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--lengths", default="129,257")
    parser.add_argument("--queries", default="1,4")
    parser.add_argument("--seed", type=int, default=92029)
    parser.add_argument(
        "--address-only",
        action="store_true",
        help="Skip the fixed shared-pool planner case; useful before installing the allocator fix",
    )
    args = parser.parse_args()
    import torch
    import torch_npu

    if args.va_root is None:
        spec = importlib.util.find_spec("vllm_ascend")
        if spec is None or spec.origin is None:
            raise RuntimeError("Cannot locate installed VA; provide --va-root")
        args.va_root = Path(spec.origin).resolve().parent.parent
    args.va_root = args.va_root.resolve()
    lengths = [int(n) for n in args.lengths.split(",")]
    queries = [int(n) for n in args.queries.split(",")]
    if min(*lengths, *queries, args.heads) <= 0:
        raise ValueError("Lengths, queries, and heads must be positive")
    torch.set_num_threads(1)
    if not torch.npu.is_available():
        raise RuntimeError(
            "A real A5 is required; this script has no CPU emulation mode"
        )
    torch.npu.set_device(args.device)
    ordinary, verify, view, sources = load_sources(args.va_root, torch)
    if not args.address_only:
        plan, specs, sources["shared_pool_plan"] = load_shared_plan(args.va_root, torch)
    ordinary.require_kimi_k3_sfa_c8_ops()
    report(
        "environment",
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        device=args.device,
        device_name=torch.npu.get_device_name(args.device),
        sources=sources,
        scope="installed_eager_kernel_addressing; not_model_or_graph_acceptance",
    )
    device = torch.device(f"npu:{args.device}")
    layouts = [
        ("tight", BLOCK * PACKED, 0),
        ("offset_only", BLOCK * PACKED, 147456),
        ("padding_not_row_aligned", 147456, 0),
        ("padding_and_offset", 147456, 147456),
    ]
    cases = failed = 0
    with torch.inference_mode():
        for name, page_bytes, offset in layouts:
            for length in lengths:
                for count in queries:
                    if count > length:
                        continue
                    cases += 1
                    try:
                        passed = run_address_case(
                            torch,
                            ordinary,
                            verify,
                            view,
                            device,
                            name,
                            page_bytes,
                            offset,
                            args,
                            length,
                            count,
                        )
                    except Exception as exc:
                        # Native asynchronous errors can poison the process.
                        report(
                            "error",
                            layout=name,
                            length=length,
                            queries=count,
                            error=repr(exc),
                        )
                        raise
                    failed += not passed
        if not args.address_only:
            cases += 1
            failed += not run_shared_pool_case(
                torch, torch_npu, ordinary, verify, view, device, args, plan, specs
            )
    report(
        "summary",
        cases=cases,
        failed=failed,
        scope="installed_scatter_and_QSFA_addressing_only; runtime_allocator_and_DSpark_not_accepted",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
