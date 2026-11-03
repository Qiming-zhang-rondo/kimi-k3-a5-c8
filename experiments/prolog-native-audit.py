"""Read-only native coverage probe: no model, weights, inference or JIT fallback."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

SOURCE_SHA = "ab0876c2fc09b7cc2b0541063dc7bab16414b2046ea9030bc1fecce4020be1bc"
TARGET_KEY = "b2324aedf7041b9b1f196f7c5c4094e6eaea068214c387c1580e1d7effd6c945"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--card",
        type=int,
        default=0,
        help="Logical card in existing visible-device environment",
    )
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--tokens", type=int, nargs="+")
    parser.add_argument("--target-key", default=TARGET_KEY)
    parser.add_argument(
        "--prolog-python",
        type=Path,
        help="Private runtime/python directory containing ops and cannbotdsl",
    )
    args = parser.parse_args()
    if args.prolog_python:
        private = args.prolog_python.resolve()
        if (
            not (private / "ops/quant_mla_prolog.py").is_file()
            or not (private / "cannbotdsl").is_dir()
        ):
            parser.error(
                "--prolog-python must contain the installed Prolog and DSL wheels"
            )
        sys.path.insert(0, str(private))
    # Must be selected before importing the DSL. Never permit compilation.
    os.environ["CANNBOTDSL_NATIVE_BINARY_MODE"] = "require"
    output = Path(tempfile.mkdtemp(prefix="prolog-native-audit-", dir="/mnt"))
    os.environ["CANNBOTDSL_NATIVE_BINARY_REPORT"] = str(output / "native-report.json")
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.card)
    op = importlib.import_module("ops.quant_mla_prolog")
    if hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError(
            "Installed Prolog source differs from the pinned build; do not infer coverage"
        )
    native = Path(op.__file__).parent / "_native"
    if not native.is_dir():
        raise RuntimeError("Missing _native package directory")
    cores = op._cores(torch.device("npu", args.card))
    print(
        json.dumps(
            {
                "event": "inventory",
                "module": op.__file__,
                "cores": cores,
                "heads": args.heads,
                "directory": str(output),
                "mode": "require",
            }
        ),
        flush=True,
    )
    tokens = args.tokens or [
        *range(1, 129),
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        24576,
    ]
    results, unique, missing_profiles = [], {}, []
    target_seen = False
    for mode, layout in ((2, 1), (2, 0), (0, 0)):
        for token in tokens:
            profile = dict(
                T=token,
                dim=7168,
                q_lora=1536,
                N=args.heads,
                Dn=128,
                Dr=64,
                w_qr_quant_mode=1,
                qkv_quant_mode=mode,
                cache_mode=layout,
                rope_mode=0,
                weight_format="ND",
            )
            parameters = op._program_parameters(
                token, 7168, 1536, args.heads, 128, 64, 1, mode, False, layout, 0, cores
            )
            if parameters not in unique:
                print(
                    json.dumps(
                        {
                            "event": "checking",
                            "T": token,
                            "mode": mode,
                            "layout": layout,
                        }
                    ),
                    flush=True,
                )
                try:
                    op._program(*parameters)
                    status = dict(status="hit")
                except Exception as exc:
                    message = str(exc)
                    match = re.search(
                        r"Native binary required but not found: ([0-9a-f]{64})", message
                    )
                    if not match:
                        raise  # ABI/import/runtime failures are NOT missing profiles.
                    status = dict(status="missing", key=match.group(1))
                    missing_profiles.append(profile)
                unique[parameters] = status
                print(
                    json.dumps({"event": "variant", "profile": profile, **status}),
                    flush=True,
                )
            status = unique[parameters]
            if status.get("key") == args.target_key:
                target_seen = True
            results.append(dict(profile=profile, parameters=parameters, **status))
    (output / "coverage.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "missing-profiles.json").write_text(
        json.dumps(missing_profiles, indent=2) + "\n"
    )
    summary = dict(
        event="summary",
        tested_profiles=len(results),
        unique_variants=len(unique),
        missing_variants=len(missing_profiles),
        target_key_reproduced=target_seen,
        target_profiles=[
            r["profile"] for r in results if r.get("key") == args.target_key
        ][:16],
        target_profiles_truncated=sum(r.get("key") == args.target_key for r in results)
        > 16,
        missing_profiles=str(output / "missing-profiles.json"),
        scope="Native lookup only; default-stream cores. NOT kernel execution or graph acceptance",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
