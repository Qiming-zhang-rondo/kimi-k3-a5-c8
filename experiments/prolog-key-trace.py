"""CPU IR/key probe. Stops before native lookup, external compilation or execution."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

KEY = "b2324aedf7041b9b1f196f7c5c4094e6eaea068214c387c1580e1d7effd6c945"
SHA = "ab0876c2fc09b7cc2b0541063dc7bab16414b2046ea9030bc1fecce4020be1bc"


class KeyReady(BaseException):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prolog-python", type=Path, required=True)
    parser.add_argument("--cores", nargs="+", type=int, default=[32, 24, 25, 20, 16, 8])
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--key", default=KEY)
    args = parser.parse_args()
    root = args.prolog_python.resolve()
    source = root / "ops/quant_mla_prolog.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA:
        parser.error("Unexpected Prolog source SHA256")
    sys.path.insert(0, str(root))
    os.environ["CANNBOTDSL_NPU_ARCH"] = "dav-3510"
    os.environ["CANNBOTDSL_NATIVE_BINARY_MODE"] = "require"
    os.environ.pop("CANNBOTDSL_PIPE_STAGE", None)
    op = importlib.import_module("ops.quant_mla_prolog")
    key_module = importlib.import_module("cannbotdsl.core.compiler.binary_key")
    index = json.loads((root / "ops/_native/index.json").read_text())
    original = key_module.compute_binary_key
    captured = {}

    def inspect(*pos, **kw):
        key, inputs = original(*pos, **kw)
        captured.update(key=key, inputs=inputs, target=kw["target"])
        raise KeyReady()

    key_module.compute_binary_key = inspect
    out = Path(tempfile.mkdtemp(prefix="prolog-key-trace-", dir="/mnt"))
    checked = set()
    results = []
    tokens = list(
        dict.fromkeys(
            [1, 8, 16, 32, 64, 96, 128, 256, 512, *range(1, 129), 8192, 24576]
        )
    )
    try:
        for mode, layout in [(2, 1), (2, 0), (0, 0)]:
            for t in tokens:
                for cores in args.cores:
                    parameters = op._program_parameters(
                        t,
                        7168,
                        1536,
                        args.heads,
                        128,
                        64,
                        1,
                        mode,
                        False,
                        layout,
                        0,
                        cores,
                    )
                    if parameters in checked:
                        continue
                    checked.add(parameters)
                    profile = dict(
                        T=t,
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
                    print(
                        json.dumps(
                            dict(
                                event="checking_key",
                                T=t,
                                mode=mode,
                                layout=layout,
                                assumed_cores=cores,
                            )
                        ),
                        flush=True,
                    )
                    captured.clear()
                    try:
                        op._program(*parameters)
                    except KeyReady:
                        pass
                    else:
                        raise RuntimeError(
                            "Key interception did not run; stop rather than infer a match"
                        )
                    record = dict(
                        profile=profile,
                        assumed_cores=cores,
                        parameters=parameters,
                        installed=captured["key"] in index,
                        **captured,
                    )
                    results.append(record)
                    if len(results) <= 5:
                        print(json.dumps(dict(event="sample", **record)), flush=True)
                    if captured["key"] == args.key:
                        (out / "matched-profile.json").write_text(
                            json.dumps([profile], indent=2) + "\n"
                        )
                        print(
                            json.dumps(
                                dict(
                                    event="EXACT_ERROR_KEY_REPRODUCED",
                                    **record,
                                    output=str(out),
                                )
                            ),
                            flush=True,
                        )
                        return
    finally:
        key_module.compute_binary_key = original
        (out / "keys.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                event="summary",
                exact_key_reproduced=False,
                checked=len(results),
                installed_hits=sum(r["installed"] for r in results),
                output=str(out),
                scope="Candidate core quotas only; no NPU initialization, binary compilation or execution",
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
