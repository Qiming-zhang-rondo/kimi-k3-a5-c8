#!/usr/bin/env python3
"""Single-operator, synthetic Cloud MLA Prolog C4 writer diagnostic for Ascend A5."""

import argparse
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import traceback

import numpy as np


HIDDEN, Q_RANK, KV_RANK, HEAD_DIM, ROPE_DIM = 6144, 2048, 512, 192, 64
PAGE = 128
Q_BYTES, ROPE_BYTES, SCALE_BYTES = KV_RANK // 2, ROPE_DIM * 2, KV_RANK // 32
PACKED_BYTES = Q_BYTES + ROPE_BYTES + SCALE_BYTES


def _byte_rows(value, label):
    data = np.asarray(value)
    if data.dtype not in (np.dtype("uint8"), np.dtype("int8")):
        raise ValueError(f"{label}: pass raw storage bytes, got {data.dtype}")
    if data.ndim < 2 or any(dim == 0 for dim in data.shape):
        raise ValueError(f"{label}: expected nonempty [active_rows, ...], got {data.shape}")
    return data.view(np.uint8).reshape(data.shape[0], -1)


def _like(value, expected, label):
    data = _byte_rows(value, label)
    if data.shape != expected.shape:
        raise ValueError(f"{label}: expected shape {expected.shape}, got {data.shape}")
    return data


def _count(mask):
    return int(np.count_nonzero(mask))


def compare_c4(actual_payload, actual_scale, actual_rope, references, reference_rope,
               *, min_groups=4, initial=None):
    """Compare C4 bytes in logical order, one E8M0 byte per 16 packed FP4 bytes.

    references maps each recipe name to its (packed_payload, raw_scale).
    RoPE tensors must be presented as raw storage bytes. Optional initial contains
    the (payload, scale, rope) of the same physical slots before the fused write.
    MATCH means exact equality on these samples and sufficient discrimination from
    each competing recipe. It does not identify a kernel implementation universally.
    """
    if len(references) < 2:
        raise ValueError("at least two native reference recipes are required")
    if not isinstance(min_groups, int) or min_groups < 1:
        raise ValueError("min_groups must be a positive integer")
    packed = _byte_rows(actual_payload, "actual payload")
    scale = _byte_rows(actual_scale, "actual scale")
    rope = _byte_rows(actual_rope, "actual RoPE")
    expected_rope = _like(reference_rope, rope, "reference RoPE")
    rows, groups_per_row = scale.shape
    if packed.shape != (rows, groups_per_row * 16) or rope.shape[0] != rows:
        raise ValueError("expected corresponding active rows and 16 payload bytes per K32 scale")
    groups = packed.reshape(rows, groups_per_row, 16)
    refs = {}
    for name, (ref_packed, ref_scale) in references.items():
        ref_groups = _like(ref_packed, packed, f"{name} payload").reshape(groups.shape)
        refs[name] = (ref_groups, _like(ref_scale, scale, f"{name} scale"))

    unchanged = None
    if initial is not None:
        ini_payload, ini_scale, ini_rope = (
            _like(value, expected, f"initial {label}")
            for value, expected, label in zip(initial, (packed, scale, rope),
                                              ("payload", "scale", "RoPE"), strict=True)
        )
        unchanged = _count((packed == ini_payload).all(axis=1)
                           & (scale == ini_scale).all(axis=1)
                           & (rope == ini_rope).all(axis=1))

    names = list(refs)
    scale_discriminating = np.zeros(scale.shape, dtype=bool)
    pairwise = {name: {} for name in names}
    for i, name in enumerate(names):
        q1, s1 = refs[name]
        for other in names[i + 1:]:
            q2, s2 = refs[other]
            scale_diff = s1 != s2
            scale_discriminating |= scale_diff
            pair = {
                "different_scale_groups": _count(scale_diff),
                "different_payload_groups": _count((q1 != q2).any(axis=-1)),
            }
            pairwise[name][other] = pair
            pairwise[other][name] = pair

    discriminating_groups = _count(scale_discriminating)
    rope_mismatches = _count(rope != expected_rope)
    metrics = {}
    exact = []
    for name, (ref_groups, ref_scales) in refs.items():
        scale_diff = scale != ref_scales
        byte_diff = groups != ref_groups
        group_diff = scale_diff | byte_diff.any(axis=-1)
        metrics[name] = {
            "scale_bytes_mismatched": _count(scale_diff),
            "payload_bytes_mismatched": _count(byte_diff),
            "groups_mismatched": _count(group_diff),
            "discriminating_scale_bytes_mismatched": _count(scale_diff[scale_discriminating]),
            "discriminating_payload_bytes_mismatched": _count(byte_diff[scale_discriminating]),
            "discriminating_groups_mismatched": _count(group_diff[scale_discriminating]),
        }
        if not group_diff.any():
            exact.append(name)

    status, matched = "INCONCLUSIVE", None
    if unchanged:
        reason = "one or more active slots were identical to their pre-write sentinel"
    elif rope_mismatches:
        reason = "RoPE control differs; check write slots, cache layout and mode0/mode4 arithmetic"
    elif discriminating_groups < min_groups:
        reason = "too few groups have distinct native reference scales"
    elif not exact:
        status = "NO_MATCH"
        reason = "no native recipe matched every payload and scale byte on these inputs"
    elif len(exact) > 1:
        reason = "multiple native recipes match exactly on these inputs"
    elif any(d["different_scale_groups"] < min_groups for d in pairwise[exact[0]].values()):
        reason = "too few scale differences between the exact match and at least one alternative"
    else:
        status, matched = "MATCH", exact[0]
        reason = "all bytes match and the reference scales differ from every alternative"
    return {
        "status": status, "matched_recipe": matched, "reason": reason,
        "active_rows": rows, "k32_groups": int(scale.size),
        "min_discriminating_groups": min_groups,
        "discriminating_groups": discriminating_groups,
        "rope_bytes_mismatched": rope_mismatches, "unchanged_active_rows": unchanged,
        "exact_payload_and_scale_matches": exact,
        "native_reference_pairs": pairwise, "recipes": metrics,
    }


def emit(event, **data):
    print(json.dumps(dict(event=event, **data), ensure_ascii=False), flush=True)


def bf16_inputs(torch, torch_npu, args):
    """A one-hot token selects one DKV row; the RMS-normalized latent is +/- gamma.

    For each K32 group, gamma sets amax safely between competing scale thresholds.
    Spread non-leading coordinates away from E2M1 ties even if fusion keeps FP32
    intermediate RMSNorm values. Distinct signs/positions expose slot permutations.
    """
    device = args.device
    dtype = torch.bfloat16
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    tokens = args.tokens
    rows = torch.arange(tokens)
    x = torch.zeros(tokens, HIDDEN)
    x[rows, rows] = 1
    gamma = torch.tensor([0.328125, 0.6875, 1.1875, 1.6875,
                          2.375, 2.75, 3.375, 4.375] * 4).repeat(16, 1)
    for group in range(16):
        gamma[group, 0] = (5.5, 6.5, 7.5, 4.75)[group % 4]
        gamma[group] *= 2 ** (group // 4 - 2)
    gamma = gamma.flatten()
    # Deterministic nonzero RoPE with independent signs for the two halves.
    angle = (rows[:, None] + 1) * torch.exp(-torch.arange(32)[None, :] / 8)
    sin = torch.cat((angle.sin(), angle.sin()), dim=1)
    cos = torch.cat((angle.cos(), angle.cos()), dim=1)
    signs = 2 * torch.randint(0, 2, (tokens, KV_RANK), generator=rng) - 1
    dkv = torch.zeros(HIDDEN, KV_RANK + ROPE_DIM, dtype=dtype)
    dkv[:tokens, :KV_RANK] = signs
    dkv[:tokens, KV_RANK:] = torch.randn(tokens, ROPE_DIM, generator=rng)
    expected_k = (signs * gamma).to(dtype)

    def to_device(tensor):
        return tensor.to(dtype).contiguous().to(device)

    common = dict(
        weight_uk=to_device(torch.randn(args.heads, HEAD_DIM, KV_RANK, generator=rng) * .0625),
        rmsnorm_gamma_cq=to_device(torch.ones(Q_RANK)),
        rmsnorm_gamma_ckv=to_device(gamma),
        rope_sin=to_device(sin), rope_cos=to_device(cos),
        rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5,
        cache_mode="PA_BSND", weight_quant_mode=args.weight_mode,
        query_quant_mode=0, query_norm_flag=False,
    )

    def add_operand(name, tensor, scale_name):
        if args.weight_mode == 3:
            # Exactly representable X and DKV NoPE, unit dynamic scales in
            # checkpoint [out,K/32] and token [T,He/32] layout for this fixture.
            data = tensor.to(torch.float8_e4m3fn).contiguous().to(device)
            scale_rows, k = tensor.shape if name == "token_x" else tensor.T.shape
            scales = torch.full((scale_rows, k // 32), 127, dtype=torch.uint8)
            common[scale_name] = scales.to(device).view(torch.float8_e8m0fnu)
        else:
            data = to_device(tensor)
        common[name] = data if name == "token_x" else torch_npu.npu_format_cast(data, 29)

    add_operand("token_x", x, "dequant_scale_x")
    add_operand("weight_dq", torch.randn(HIDDEN, Q_RANK, dtype=dtype, generator=rng), "dequant_scale_w_dq")
    add_operand("weight_uq_qr", torch.randn(Q_RANK, args.heads * (HEAD_DIM + ROPE_DIM),
                                             dtype=dtype, generator=rng) * .03125, "dequant_scale_w_uq_qr")
    add_operand("weight_dkv_kr", dkv, "dequant_scale_w_dkv_kr")
    return common, expected_k


def run_cloud(torch, op, common, args, mode, report):
    slots = np.array([PAGE * (i % 2) + 3 + i // 2 for i in range(args.tokens)])
    common = dict(common, cache_index=torch.tensor(slots, device=args.device, dtype=torch.int64))
    valid = np.zeros(2 * PAGE, dtype=bool)
    valid[slots] = True
    active_from_runs = []
    for sentinel in (0x35, 0xCA):
        def cache(width, dtype):
            t = torch.empty((2, PAGE, 1, width), device=args.device, dtype=dtype)
            t.view(torch.uint8).fill_(sentinel)
            return t

        if mode == 0:
            kv = cache(KV_RANK, torch.bfloat16)
            kr = cache(ROPE_DIM, torch.bfloat16)
            kw = dict(kv_cache_quant_mode=0, ckvkr_repo_mode=0, quant_scale_repo_mode=0)
            buffers = [kv, kr]
        else:
            kv = cache(PACKED_BYTES, torch.int8)
            # Combined mode writes RoPE inside kv. Cloud still validates the
            # rank of the empty kr argument against the PA_BSND layout.
            if args.empty_kr_axis == "pages":
                kr_shape = (0, kv.shape[1], kv.shape[2], ROPE_DIM)
            else:
                kr_shape = (*kv.shape[:3], 0)
            kr = torch.empty(kr_shape, device=kv.device, dtype=torch.bfloat16)
            kw = dict(kv_cache_quant_mode=4, ckvkr_repo_mode=1, quant_scale_repo_mode=1,
                      tile_size=args.tile_size)
            buffers = [kv]
        call = dict(mode=mode, sentinel=sentinel, cache_shape=list(kv.shape),
                    cache_dtype=str(kv.dtype), kr_cache_shape=list(kr.shape),
                    kr_cache_dtype=str(kr.dtype), kr_cache_numel=kr.numel(), options=kw)
        report["last_call"] = call
        emit("call", **call)
        output = op(**common, kv_cache=kv, kr_cache=kr, **kw)
        torch.npu.synchronize()
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise ValueError("Cloud wrapper did not return a query/query_rope tuple")
        expected_shapes = [(args.tokens, args.heads, KV_RANK), (args.tokens, args.heads, ROPE_DIM)]
        for q, shape in zip(output[:2], expected_shapes):
            if tuple(q.shape) != shape or not torch.isfinite(q.detach().float().cpu()).all().item():
                raise ValueError(f"Unexpected query output {q.shape} for expected {shape} or nonfinite data")
        active = []
        for t in buffers:
            raw = t.detach().contiguous().view(torch.uint8).cpu().numpy().reshape(2 * PAGE, -1)
            if not np.all(raw[~valid] == sentinel):
                raise ValueError(f"mode={mode} wrote outside requested physical slots")
            active.append(raw[slots].copy())
        active_from_runs.append(active)
    if not all(np.array_equal(x, y) for x, y in zip(*active_from_runs)):
        raise ValueError(f"mode={mode} active rows depended on their initial sentinel")
    report[f"mode{mode}_write_control"] = "two sentinels agree on active rows, inactive rows untouched"
    emit("write_verified", mode=mode, rows=args.tokens)
    return active_from_runs[0]


def scratch_reference(torch, raw):
    # Reconstitute exactly the active BF16 bytes emitted by the mode0 kernel.
    return torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(-1, 1, KV_RANK)


def classify(torch, torch_npu, k, rope_bytes, packed, args):
    recipes = {"default": {}, "alg2_max6": {"dst_type_max": 6.0}, "alg2_max7": {"dst_type_max": 7.0}}
    references = {}
    x = k.to(args.device).contiguous()
    for name, options in recipes.items():
        kwargs = dict(dst_type=torch_npu.float4_e2m1fn_x2)
        if options:
            kwargs.update(axis=-1, round_mode="rint", block_size=32, scale_alg=2, **options)
        q, s = torch_npu.npu_dynamic_mx_quant(x, **kwargs)
        torch.npu.synchronize()
        references[name] = tuple(t.detach().contiguous().view(torch.uint8).cpu().numpy().reshape(args.tokens, -1)
                                 for t in (q, s))
    return compare_c4(packed[:, :Q_BYTES], packed[:, Q_BYTES+ROPE_BYTES:],
                      packed[:, Q_BYTES:Q_BYTES+ROPE_BYTES], references, rope_bytes)


def run(args, result):
    result["stage"] = "load_runtime"
    import torch
    import torch_npu
    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU unavailable")
    torch.npu.set_device(args.device)
    if hasattr(torch.npu, "config") and hasattr(torch.npu.config, "allow_internal_format"):
        torch.npu.config.allow_internal_format = True
    # Some Cloud releases use the base package for op registration during import.
    if importlib.util.find_spec("cann_ops_transformer") is not None:
        try:
            importlib.import_module("cann_ops_transformer.ops")
        except (ImportError, OSError, RuntimeError) as exc:
            result["optional_base_import"] = f"{type(exc).__name__}: {exc}"
            emit("optional_base_import_failed", detail=result["optional_base_import"])
    cloud = importlib.import_module("cloud_ops_turbo")
    op = getattr(cloud, "cloud_mla_prolog_v3", None)
    if not callable(op):
        mod = importlib.import_module("cloud_ops_turbo.cloud_mla_prolog_v3")
        op = getattr(mod, "cloud_mla_prolog_v3")
    try:
        signature = str(inspect.signature(op))
    except (TypeError, ValueError):
        signature = "unavailable"
    result["runtime"] = dict(torch=torch.__version__, torch_npu=getattr(torch_npu, "__version__", None),
                             device=torch.npu.get_device_name(torch.npu.current_device()),
                             cloud_module=cloud.__file__, wrapper_module=getattr(op, "__module__", None),
                             signature=signature, registered_schemas=[str(s) for s in getattr(torch._C, "_jit_get_all_schemas", lambda: ())()
                                if "cloud" in s.name.lower() and "mla_prolog" in s.name.lower()])
    emit("runtime", **result["runtime"])

    with torch.inference_mode():
        result["stage"] = "fixture"
        common, expected_k = bf16_inputs(torch, torch_npu, args)
        # Run the questioned operator first: a mode0 failure must not conceal
        # whether this deployment actually wrote mode4 at these fixture settings.
        result["stage"] = "cloud_mode4"
        packed, = run_cloud(torch, op, common, args, 4, result)
        result["stage"] = "cloud_mode0"
        k_bytes, rope_bytes = run_cloud(torch, op, common, args, 0, result)
        k = scratch_reference(torch, k_bytes)
        if not torch.equal(k.reshape(args.tokens, KV_RANK), expected_k):
            diff = (k.reshape(args.tokens, KV_RANK).float() - expected_k.float()).abs()
            result["mode0_fixture_max_abs"] = float(diff.max())
            raise ValueError("mode0 BF16 NoPE disagrees with engineered post-RMSNorm input; check format/weights/scales")
        result["mode0_fixture_exact"] = True
        result["stage"] = "native_references_and_compare"
        result["comparison"] = classify(torch, torch_npu, k, rope_bytes, packed, args)
        result["status"] = result["comparison"]["status"]
        result["stage"] = "complete"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--weight-mode", type=int, choices=(0, 3), default=3,
                   help="3 uses synthetic MXFP8 with E8M0 unit scales; 0 uses BF16")
    p.add_argument("--heads", type=int, choices=(4, 8, 16, 32, 64), default=8)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=119)
    p.add_argument("--tile-size", type=int, choices=(32, 64, 128), default=128,
                   help="vendor mode4 tile_size. Independent of native MX group size (32)")
    p.add_argument("--empty-kr-axis", choices=("width", "pages"), default="width",
                   help="mode4 empty 4-D KR placeholder: zero width (default), or zero pages with RoPE width 64")
    p.add_argument("--output", type=Path, help="also write final JSON to this file")
    args = p.parse_args()
    if not 1 <= args.tokens <= 128:
        p.error("tokens must be in [1,128]")
    result = dict(status="ERROR", scope="single synthetic suboperator; exact sample comparison, no serving run",
                  device=args.device, tokens=args.tokens, heads=args.heads, seed=args.seed,
                  weight_quant_mode=args.weight_mode, mode4_tile_size=args.tile_size,
                  mode4_layout="PA_BSND, int8 400B = 256 NoPE + 128 BF16 RoPE + 16 E8M0")
    try:
        run(args, result)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    emit("result", **result)
    return {"MATCH": 0, "NO_MATCH": 2, "INCONCLUSIVE": 3, "ERROR": 1}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
