"""Prepare or build pinned K3 Prolog AOT profiles. Run the build on A5.

This does not claim binary coverage until cold-process native=require probes
pass. No network downloads, global package installs or source resets.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REVISION = "9e01355ab8f809050579271c0eadcc5858ed20bc"
SOURCE_SHA = "ab0876c2fc09b7cc2b0541063dc7bab16414b2046ea9030bc1fecce4020be1bc"
WHEEL_SHA = "2d40711a02a12a72a6983004d96752556650f78892905471c3ead17eb7bf1302"


def profiles(heads, tokens):
    return [
        dict(
            T=t,
            dim=7168,
            q_lora=1536,
            N=h,
            Dn=128,
            Dr=64,
            w_qr_quant_mode=1,
            qkv_quant_mode=m,
            cache_mode=c,
            rope_mode=0,
            weight_format="ND",
        )
        for h in heads
        for t in tokens
        for m, c in ((0, 0), (2, 0), (2, 1))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--cann-env", type=Path, required=True)
    parser.add_argument("--heads", type=int, nargs="+", default=[12, 24, 48])
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 8, 9, 16, 128, 512, 8192]
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--archive-dir", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        parser.error("Expected pinned cann/cannbot-dsl source at " + REVISION)
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        parser.error(
            "Tracked source modifications detected; use a dedicated clean checkout"
        )
    vendor = source / "samples/quant_mla_prolog/quant_mla_prolog.py"
    for file, digest in ((vendor, SOURCE_SHA), (args.wheel, WHEEL_SHA)):
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            parser.error("Pinned SHA256 mismatch: " + str(file))
    if not args.cann_env.is_file():
        parser.error("CANN environment file is absent")
    if any(h < 1 or h > 128 for h in args.heads) or any(t < 1 for t in args.tokens):
        parser.error("Heads must be 1..128; token counts must be positive")
    work = Path(tempfile.mkdtemp(prefix="k3-prolog-aot-"))
    config = work / "profiles.json"
    config.write_text(json.dumps(profiles(args.heads, args.tokens), indent=2) + "\n")
    env = dict(
        os.environ,
        CANN_ENV=str(args.cann_env.resolve()),
        CANNBOTDSL_WHEEL=str(args.wheel.resolve()),
        PYTHON=sys.executable,
        CANNBOTDSL_QMP_PROFILES=str(config),
    )
    print(
        json.dumps(
            {
                "profiles": str(config),
                "source_revision": revision,
                "profile_count": len(profiles(args.heads, args.tokens)),
                "status": "building" if args.build else "plan_only",
                "scope": "Real K3 geometry; binary coverage still requires cold native=require tests",
            }
        ),
        flush=True,
    )
    if args.build:
        subprocess.run(
            [
                "bash",
                "./run-build.sh",
                "--operator",
                "quant_mla_prolog",
                "--target",
                "dav-3510",
            ],
            cwd=source / "net/native_package",
            env=env,
            check=True,
        )
        output = source / "net/native_package/output/operators/quant_mla_prolog"
        wheels = sorted(output.glob("wheels/*.whl"))
        if not wheels:
            raise RuntimeError("Build returned success without an operator wheel")
        archive_path = None
        if args.archive_dir:
            args.archive_dir.mkdir(parents=True, exist_ok=True)
            bundle = Path(
                tempfile.mkdtemp(
                    prefix="k3-prolog-aot-candidate-", dir=args.archive_dir
                )
            )
            for item in [*wheels, args.wheel, config, source / "LICENSE"]:
                shutil.copy2(item, bundle / item.name)
            (bundle / "manifest.json").write_text(
                json.dumps(
                    {
                        "source_revision": revision,
                        "dsl_wheel_sha256": WHEEL_SHA,
                        "target": "dav-3510",
                        "python": sys.version,
                        "cann_env": str(args.cann_env.resolve()),
                        "acceptance": "build only; cold native=require execution not yet validated",
                    },
                    indent=2,
                )
                + "\n"
            )
            checksums = [
                f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
                for p in sorted(bundle.iterdir())
                if p.is_file()
            ]
            (bundle / "SHA256SUMS").write_text("".join(checksums))
            archive_path = bundle.with_suffix(".tar.gz")
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(bundle, arcname=bundle.name)
            archive_path.with_name(archive_path.name + ".sha256").write_text(
                f"{hashlib.sha256(archive_path.read_bytes()).hexdigest()}  {archive_path.name}\n"
            )
        print(
            json.dumps(
                {
                    "status": "build_complete_not_accepted",
                    "wheels": [str(p) for p in wheels],
                    "archive": str(archive_path) if archive_path else None,
                    "next": "Install privately; register ops; run cold native=require probes before archiving",
                }
            )
        )


if __name__ == "__main__":
    main()
