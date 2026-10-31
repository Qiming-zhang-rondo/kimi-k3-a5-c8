"""Pinned recipe-ABI Prolog -> PA_NZ FP8 -> QFM synthetic smoke probe.

No vLLM imports, model weights, CANN upgrades, source-JIT fallback, or downloads.
The independently maintained CPU golden is loaded from the pinned official
CANNBot-DSL checkout, not from the installed operator implementation.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path

RECIPE_COMMIT = "4447f2f0972cb9287e069e3994b1d46ffe244ebc"
PROLOG_COMMIT = "7425a9b998e9133096c734600f9a9453f4101afb"
GOLDEN_SHA256 = "df9db34440fa4a5d9269b2ee02e5a67fb943e876390e95081274c40cc1ded759"
OPERATOR_SHA256 = "ab20d00ec17d379d792abaad936ddc292cf81bb7c4b939270b9d170b870cd81b"
CASES = ("case_013_aw1_c1", "case_014_aw1_c1", "gen_t0064_c1", "case_016_aw1_c1")
GUARD_BYTES = 256


class Unavailable(RuntimeError):
    """Required runtime, native package, or pinned reference is unavailable."""


def emit(event, **fields):
    print(
        json.dumps({"event": event, **fields}, ensure_ascii=False, allow_nan=False),
        flush=True,
    )


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _version(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "not installed"


def load_reference(source=None):
    source = Path(
        source
        or os.environ.get("K3_PROLOG_SOURCE")
        or Path(__file__).parent / "sources" / "cannbot-dsl"
    ).resolve()
    path = source / "test" / "quant_mla_prolog" / "golden.py"
    if not path.is_file() or _digest(path) != GOLDEN_SHA256:
        raise Unavailable(
            f"Pinned CPU golden missing or changed: {path}. Check out CANNBot-DSL "
            f"{PROLOG_COMMIT}; pass --prolog-source or set K3_PROLOG_SOURCE. "
            "See BUILD.md. Current master is not an ABI-compatible substitute."
        )
    spec = importlib.util.spec_from_file_location("qfm_stage1_prolog_golden", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def load_runtime(card=0, source=None):
    # The old Native loader must not silently compile an untested source path.
    os.environ["CANNBOTDSL_NATIVE_BINARY_MODE"] = "require"
    try:
        torch = importlib.import_module("torch")
        importlib.import_module("torch_npu")
        if not torch.npu.is_available():
            raise Unavailable("An Ascend NPU is required; this is not a CPU pass.")
        torch.npu.set_device(card)
        dsl_version = _version("cannbotdsl", "cannbot-dsl")
        if not dsl_version.startswith("0.7."):
            raise Unavailable(
                f"Pinned native packaging requires DSL 0.7.x, got {dsl_version}."
            )
        importlib.import_module("cannbotdsl")
        module = importlib.import_module("ops.quant_mla_prolog")
        op = module.quant_mla_prolog_op
        if not {"quant_mode_aw", "quant_mode_c"}.issubset(
            inspect.signature(op).parameters
        ):
            raise Unavailable("Installed Prolog does not expose the recipe AW1/C1 ABI.")
        operator_file = Path(inspect.getsourcefile(op) or "")
        if not operator_file.is_file() or _digest(operator_file) != OPERATOR_SHA256:
            raise Unavailable(
                "Installed Prolog source is not the pinned 7425a9b implementation; "
                "rebuild its Native wheel as documented in BUILD.md."
            )
        golden, source = load_reference(source)
        importlib.import_module("cann_ops_transformer.ops")
        native = torch.ops.cann_ops_transformer
        schemas = {
            name: str(getattr(native, name).default._schema)
            for name in ("quant_flash_mla", "quant_flash_mla_metadata")
        }
    except Unavailable:
        raise
    except (ImportError, AttributeError, OSError) as exc:
        raise Unavailable(
            f"Required Prolog/QFM runtime unavailable: {exc}. See BUILD.md."
        ) from exc
    emit(
        "prolog_environment",
        card=card,
        device=torch.npu.get_device_name(card),
        torch=_version("torch"),
        torch_npu=_version("torch-npu"),
        cannbot_dsl=dsl_version,
        native_mode="require",
        prolog_source=str(source),
        prolog_commit=PROLOG_COMMIT,
        recipe_commit=RECIPE_COMMIT,
        operator_sha256=OPERATOR_SHA256,
        golden_sha256=GOLDEN_SHA256,
        qfm_schemas=schemas,
        prolog_supported_heads=[96],
        tp_local_prolog_supported=False,
        integration_blocker="Pinned Prolog hard-requires WKB [96,128,512]; "
        "TP-local 6/12/24/48 heads are not supported.",
    )
    return torch, golden, op, native


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _metrics(torch, actual, expected, *, rtol=0.03, atol=0.025, relative_l2=0.02):
    a, e = actual.float().cpu(), expected.float().cpu()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    if not finite:
        return {"pass": False, "finite": False}
    error = (a - e).abs()
    rel = float(error.norm() / e.norm().clamp_min(1e-12))
    bad = int((error > atol + rtol * e.abs()).sum())
    return {
        "pass": bad == 0 and rel <= relative_l2,
        "finite": True,
        "relative_l2": rel,
        "max_abs": float(error.max()),
        "bad_elements": bad,
        "rtol": rtol,
        "atol": atol,
        "relative_l2_limit": relative_l2,
    }


def qfm_roundtrip(torch, native, payload):
    """Consume the exact writer Q/scale/cache views; no repack or re-quantize.

    Each written token is a separate one-query request over its physical page
    prefix. This is a layout/scale handoff test, not a model sequence test.
    The independent CPU attention oracle reads a logical transpose of cache.
    """
    q, qs, cache = payload["q"], payload["q_descale"], payload["cache"]
    slots = payload["slots_cpu"]
    tokens, heads, _ = q.shape
    table_cpu = slots.div(128, rounding_mode="floor").to(torch.int32).view(-1, 1)
    lengths_cpu = (slots.remainder(128) + 1).to(torch.int32)
    table, lengths = table_cpu.to(q.device), lengths_cpu.to(q.device)
    cu = torch.arange(tokens + 1, dtype=torch.int32, device=q.device)
    used = torch.ones(tokens, dtype=torch.int32, device=q.device)
    metadata = native.quant_flash_mla_metadata(
        lengths,
        heads,
        1,
        1,
        cu_seqlens_q=cu,
        seqused_q=used,
        max_seqlen_q=1,
        max_seqlen_kv=128,
        head_dim_qk=576,
        head_dim_v=512,
        mask_mode=0,
        layout_q="TND",
    )
    output, lse = native.quant_flash_mla(
        q,
        cache,
        qs,
        payload["kv_scale"],
        table,
        lengths,
        1,
        cu_seqlens_q=cu,
        seqused_q=used,
        metadata=metadata,
        head_dim_v=512,
        softmax_scale=1 / math.sqrt(192),
        mask_mode=0,
        max_seqlen_q=1,
        max_seqlen_kv=128,
        layout_q="TND",
        layout_kv="PA_NZ",
        layout_out="TND",
        return_softmax_lse=True,
    )
    torch.npu.synchronize()
    _check(tuple(output.shape) == (tokens, heads, 512), "QFM output shape mismatch")
    _check(tuple(lse.shape) == (heads, tokens), "QFM LSE shape mismatch")
    # CPU-only dequantization is the oracle, not a resident BF16 cache.
    logical = (
        cache.cpu().view(-1, 18, 128, 32).permute(0, 2, 1, 3).reshape(-1, 128, 576)
    )
    logical = logical.float() * float(payload["kv_scale"].cpu().item())
    query = q.cpu().float() * qs.cpu()[..., None]
    expected, expected_lse = [], []
    for i in range(tokens):
        kv = logical[int(table_cpu[i, 0]), : int(lengths_cpu[i])]
        logits = query[i] @ kv.T / math.sqrt(192)
        expected.append(logits.softmax(-1) @ kv[:, :512])
        expected_lse.append(logits.logsumexp(-1))
    checks = {
        "output": _metrics(torch, output, torch.stack(expected)),
        "lse": _metrics(torch, lse.T, torch.stack(expected_lse)),
    }
    emit(
        "prolog_qfm_roundtrip",
        direct_writer_cache_alias=True,
        query_requantized=False,
        layout="PA_NZ",
        checks=checks,
        **{"pass": all(item["pass"] for item in checks.values())},
    )
    _check(
        all(item["pass"] for item in checks.values()), "Prolog -> QFM roundtrip failed"
    )
    return checks


def run_probe(card=0, prolog_source=None, *, case_id=CASES[0], preflight=False):
    """Return (JSON-safe report, device payload); preflight has an empty payload.

    Payload keys: q, q_descale, cache [B,1,18,128,32], backing_storage,
    kv_scale [1], slots_cpu. Its cache is an alias of the native writer output.
    """
    torch, golden, op, native = load_runtime(card, prolog_source)
    if preflight:
        return {
            "pass": True,
            "scope": "imports/source hashes/schema only; no kernel execution",
        }, {}
    case = golden.load_case(case_id, quant_mode_c=1)
    emit("stage", name="prolog_independent_cpu_golden", status="begin", case_id=case_id)
    inputs, expected = golden.run_case(case)
    emit("stage", name="prolog_independent_cpu_golden", status="complete")
    physical = inputs["kv_cache"]
    blocks, _, page, _ = physical.shape
    body_bytes = physical.numel()
    initial = torch.full((body_bytes + 2 * GUARD_BYTES,), 0xA5, dtype=torch.uint8)
    initial[GUARD_BYTES:-GUARD_BYTES].copy_(physical.view(torch.uint8).reshape(-1))
    backing = initial.to("npu")
    physical_device = (
        backing[GUARD_BYTES:-GUARD_BYTES].view(torch.float8_e4m3fn).view_as(physical)
    )
    cache = physical_device.view(blocks, page, 1, 576)
    pointer, storage_pointer = cache.data_ptr(), cache.untyped_storage().data_ptr()
    device_inputs = {
        name: value.to("npu")
        for name, value in inputs.items()
        if name != "kv_cache"
        and name not in ("descale_wqa", "descale_wqb", "descale_wkva")
    }
    scales = [
        golden.logical_descale(inputs[name], outer, inner).to("npu")
        for name, outer, inner in (
            ("descale_wqa", 1536, 7168),
            ("descale_wqb", 96 * 192, 1536),
            ("descale_wkva", 576, 7168),
        )
    ]
    emit("stage", name="native_prolog", status="begin", tokens=case["T"], heads=96)
    q, qscale = op(
        device_inputs["x"],
        device_inputs["wqa"],
        device_inputs["wqb"],
        device_inputs["wkva"],
        device_inputs["wkb"],
        device_inputs["descale_x"],
        *scales,
        device_inputs["norm_weight_qa"],
        device_inputs["norm_weight_kva"],
        cache,
        device_inputs["cache_index"],
        device_inputs["qscale_kv"],
        norm_eps=1e-6,
        quant_mode_aw=1,
        quant_mode_c=1,
    )
    torch.npu.synchronize()
    _check(
        cache.data_ptr() == pointer
        and cache.untyped_storage().data_ptr() == storage_pointer,
        "Native Prolog replaced cache storage",
    )
    after = backing.cpu()
    _check(
        torch.equal(after[:GUARD_BYTES], initial[:GUARD_BYTES])
        and torch.equal(after[-GUARD_BYTES:], initial[-GUARD_BYTES:]),
        "Cache boundary guard corruption",
    )
    actual = {
        "q": q.cpu(),
        "descale_q": qscale.cpu(),
        "kv_cache_out": physical_device.cpu(),
    }
    report = golden.compare_outputs(
        actual,
        expected,
        inputs["cache_index"],
        quant_mode_c=1,
        qscale_kv=inputs["qscale_kv"],
    )
    emit(
        "prolog_numeric",
        case_id=case_id,
        oracle="pinned independent official CPU golden",
        **report,
    )
    _check(report["passed"], "Prolog Q/scale/cache failed independent CPU golden")
    # Check untouched rows using a logical transpose, independent of scatter offsets.
    old_rows = (
        physical.permute(0, 2, 1, 3).contiguous().view(torch.uint8).reshape(-1, 576)
    )
    new_rows = (
        actual["kv_cache_out"]
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(torch.uint8)
        .reshape(-1, 576)
    )
    unwritten = torch.ones(blocks * page, dtype=torch.bool)
    unwritten[inputs["cache_index"]] = False
    _check(
        torch.equal(old_rows[unwritten], new_rows[unwritten]),
        "An unwritten cache row changed",
    )
    emit(
        "prolog_cache_guards",
        storage_offset_bytes=GUARD_BYTES,
        prefix_suffix_bytes=GUARD_BYTES,
        pointer_unchanged=True,
        untouched_rows=int(unwritten.sum()),
        **{"pass": True},
    )
    payload = {
        "q": q,
        "q_descale": qscale,
        "cache": physical_device.view(blocks, 1, 18, 128, 32),
        "backing_storage": backing,
        "kv_scale": device_inputs["qscale_kv"],
        "slots_cpu": inputs["cache_index"],
    }
    report["qfm_roundtrip"] = qfm_roundtrip(torch, native, payload)
    report.update(
        {
            "pass": True,
            "supported_prolog_heads": [96],
            "tp_local_prolog_supported": False,
            "model_accuracy_validated": False,
        }
    )
    return report, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, default=0, help="Logical visible NPU index")
    parser.add_argument(
        "--prolog-source", help="Pinned CANNBot-DSL checkout; see BUILD.md"
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--case", choices=CASES, default=CASES[0])
    args = parser.parse_args(argv)
    suite = "prolog_preflight" if args.preflight else "prolog"
    try:
        report, _ = run_probe(
            args.card, args.prolog_source, case_id=args.case, preflight=args.preflight
        )
        emit(
            "suite_complete",
            suite=suite,
            scope="fixed-96-head synthetic smoke only",
            tp_local_prolog_supported=False,
            **{"pass": report["pass"]},
        )
        return 0
    except Unavailable as exc:
        emit(
            "suite_complete",
            suite=suite,
            status="unavailable",
            error=str(exc),
            **{"pass": False},
        )
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI reports every native failure and returns nonzero.
        emit(
            "suite_complete",
            suite=suite,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            **{"pass": False},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
