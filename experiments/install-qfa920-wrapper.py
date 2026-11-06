#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Explicit isolated QFA v9.2.0 host-wrapper preparation; no kernel install.

The embedded upstream Python/C++ sources and LICENSE retain the CANN Open
Software License Agreement 2.0. The original package is copied, never edited.
This program uses only the standard library and never imports torch or NPU
modules. Run it in the target container after sourcing the original QFM env.
"""

import argparse
import ast
import base64
import gzip
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import re
import shlex
import shutil
import tempfile
from pathlib import Path

SOURCE_REVISION = "dff221330eecd2132233b1e73ed1072cd8925140"
STAGE1_REVISION = "8b6924e68ed27181141cde84fa5a942c0c060aac"
PAYLOAD = {
    "python": {
        "sha256": "3da85c76203bd193481ac33d0516fbe38375b05b0ea8cdd34d712eda449fe681",
        "gzip_base64": (
            "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"
        ),
    },
    "cpp": {
        "sha256": "a1c3a191ecf8b1a2bee857174bfff9a367b0e9cdaa56b64c7be9dddaff1a6320",
        "gzip_base64": (
            "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"
            "Z/qVc+F7jsBCc3YhAAA="
        ),
    },
    "license": {
        "sha256": "89b1c114df62cfd183004f38d18d07f186436c3b37203dd69c3e84becf9d2282",
        "gzip_base64": (
            "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"
            "Ikz8q/8CD8iXp44UAAA="
        ),
    },
    "graph": {
        "sha256": "a72244b5b9ae33e4d321ba10f40824b475e3a3eeaabebbce661973410d417f08",
        "gzip_base64": (
            "H4sIAAAAAAACE+1X32/aMBB+56/weCpSFq19RJo0+nOVqlZr2V6qyjLhAhaO7dgOhf31OyckpITQdOpLtUWChPu+u3PufHfGmfWw"
            "R/DiiVbGEadMNG8IqNRZQ8i4yWWxUUkhCQWfGGbWJQ1vYoeCSiGdAY2UjAw4oDPD9LxUQIBpbgmz+LhHcwa58gudMUirTLC5P2iI"
            "OrkM49VJIVyCcWBKc0e5tr+mEAlmgNpMewSmQQUZmHGLSjReUammQOuWCtqg2yoq47RwxxxXslzLQwHWLK01l7MSHsl1QM555AJy"
            "g8vBEGRaQEB+SrQRkDMmBJt4wZ32Vlm3XGCQKTdUT06aIbnyhHOIt5G40y9+Fkk4BxsZrtFJE4nbw7MvuzNwGJmYZcJVWEAkrBzN"
            "JE8zoJIlXbcKy5wq80Tx0eVr+ut9hsFnjo3XGkqpf+5oLlKJZo4ySydrBzbYFVCBKe1oyxkmrX8hr1N8cYn7YY88Foq5N8SeyyUT"
            "3O9O27GuNu/BTa2olQ7ItU/+Xs9VepwzvZxBx3f3Z99H1/d09Gt0fTM6vbkgX8nYZNCDVQTaketc5cIYZYYHVC6ZsNDr8XgfPiyc"
            "fdu/Lba7/rF68lduMvinRE/Be8fCp/IjSZ7qXQv7EfmRMekuBbPzkXNyu1fSYdkoKtGiKVo2RWleYkzAHgPt0LIdSv0KaYLTaUh8"
            "N6iAiVDRgjo/GobVaHgs1J+wZm6VrL25phv7rzKjjFpIBaI0fRt9sezAR3JmcU6mb+B2M8zlwnbgYXuSNGF20YGbgGNTHA5d3KvY"
            "JWxVhjlv0Eg6Dr/UzKHXbSoRrWHPXFIBsSuhz8cvMcNn871g7jTPgI/pQdzHsUkQbK0y55WtM4j1Tx9uz/sN3CsfJOCnjYFzJTOS"
            "ljHCPjIkE6VE2do3ZTmsKVg8KiCchhKeKSTarY/S0PLfcDQY7NjdsDdj4OChblvhxUlXaRtGTEocbThh/ZyNlUnAhEXZxb4zUL9j"
            "ws3pZad/lONml/5xO0mh1d4/WvCXXeN10mLZyqp1iFcYh4wU3aAFrfWAFsa28tsc/K/396x3xi2QkbVYTNhn8wPhUf/qgkjlSPXn"
            "6lN/0PsDk4pE5eoOAAA="
        ),
    },
}
KNOWN_HELPERS = {
    "__init__.py": {"cd60c29a125cfbf351c3f5a38529ca922aa93db9996721d947f3ffb532c2193d"},
    "op_builder/__init__.py": {"4ad7d26eebb562fc480bd64786c4874de470cd7f696ff1a49ebdef60d510eced"},
    "op_builder/builder.py": {
        "4f87aea39abd9740756b2fa453df0317e423d7f25a4f6b716e194fa64a79aec4",
        "217a97f2213ade7e14bfcb8d05bbd43d67bd290c6a324454870a3dba64fd2714",
    },
    "common/aclnn_common.h": {
        "8a8bb4fd0c21513fd4a6b879700d1e1fff0ed110a5c481fe6babd933089137c5",
        "ade3f8c2ebe020dbae80c8dd841741297591a86a3b0cb4097b5fc87648d83c31",
    },
    "utils/arg_check.py": {"0542b9344532b96708169af89bef5a2253b5033f60441663f7addfdcb89d836b"},
    "ops/__init__.py": {
        "cb81f62de9423b71f269a5d9ce30b2ee25d821326311064e9de322e65e354b79",
        "01b5ea259e692424c768be557b30376ff98e366604549bfaa50dab7b25bb6299",
    },
}
QFA_NAMES = {
    "q",
    "k",
    "v",
    "q_descale",
    "k_descale",
    "v_descale",
    "quant_mode",
    "block_table",
    "p_scale",
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "seqused_q",
    "seqused_kv",
    "sinks",
    "attn_mask",
    "metadata",
    "softmax_scale",
    "mask_mode",
    "win_left",
    "win_right",
    "max_seqlen_q",
    "max_seqlen_kv",
    "layout_q",
    "layout_q_descale",
    "layout_kv",
    "layout_out",
    "return_softmax_lse",
}
METADATA_NAMES = {
    "num_heads_q",
    "num_heads_kv",
    "head_dim",
    "quant_mode",
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "seqused_q",
    "seqused_kv",
    "batch_size",
    "max_seqlen_q",
    "max_seqlen_kv",
    "head_dim_v",
    "mask_mode",
    "win_left",
    "win_right",
    "layout_q",
    "layout_q_descale",
    "layout_kv",
    "layout_out",
    "is_grad_enabled",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def payload_bytes(name):
    record = PAYLOAD[name]
    data = gzip.decompress(base64.b64decode(record["gzip_base64"], validate=True))
    if sha256(data) != record["sha256"]:
        raise RuntimeError(f"Embedded official {name} SHA256 mismatch")
    return data


def locate_package(source_python=None):
    if source_python is not None:
        root = Path(source_python).expanduser().resolve()
        package = root if root.name == "cann_ops_transformer" else root / "cann_ops_transformer"
    else:
        # Finding a top-level package spec does not execute its __init__.py.
        spec = importlib.util.find_spec("cann_ops_transformer")
        if spec is None or spec.origin is None:
            raise RuntimeError("cann_ops_transformer package not found; source the QFM env or use --source-python")
        package = Path(spec.origin).resolve().parent
    if not (package / "__init__.py").is_file():
        raise RuntimeError(f"Expected a regular cann_ops_transformer package: {package}")
    return package


def inspect_abi(source):
    """Inspect Python signatures through AST, without importing the package."""
    definitions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    result = {}
    for name, expected in (("quant_flash_attn", QFA_NAMES), ("quant_flash_attn_metadata", METADATA_NAMES)):
        node = definitions.get(name)
        if node is None or node.args.vararg or node.args.kwarg:
            raise RuntimeError(f"Official {name} has no unique explicit Python signature")
        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if {item.arg for item in args} != expected:
            raise RuntimeError(f"Official {name} signature differs from the fixed CANN 9.2.0 ABI")
        signature = inspect.Signature(
            [inspect.Parameter(item.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for item in args]
        )
        result[name] = str(signature)
    return result


def validate_helpers(package):
    checks = {}
    for relative, accepted in KNOWN_HELPERS.items():
        path = package / relative
        if not path.is_file():
            raise RuntimeError(f"Required shared helper is missing: {path}")
        digest = sha256(path.read_bytes())
        if digest not in accepted:
            raise RuntimeError(f"Unverified shared helper {path} SHA256={digest}; no package was changed")
        checks[relative] = digest
    if (
        checks["common/aclnn_common.h"] == "ade3f8c2ebe020dbae80c8dd841741297591a86a3b0cb4097b5fc87648d83c31"
        and checks["op_builder/builder.py"] != "217a97f2213ade7e14bfcb8d05bbd43d67bd290c6a324454870a3dba64fd2714"
    ):
        raise RuntimeError("Stage1 common header needs its matching builder include paths; mixed helpers refused")
    # Stage1's common header adds this include. Keep its entire original
    # dependency tree rather than replacing a header shared with QFM.
    pending = [package / "common/aclnn_common.h"]
    visited = set()
    while pending:
        header = pending.pop()
        if header in visited:
            continue
        visited.add(header)
        for include in re.findall(r'^\s*#\s*include\s+"([^"]+)"', header.read_text(), re.MULTILINE):
            if not include.startswith("common/include/"):
                continue
            target = package / include
            if not target.is_file():
                raise RuntimeError(f"Shared helper dependency missing: {target}; cannot isolate this package")
            pending.append(target)
    return checks


def find_qfa_sources(package):
    paths = []
    for path in (package / "ops").rglob("*.py"):
        if "quant_flash_attn" not in path.name and path.name != "__init__.py":
            continue
        data = path.read_text()
        if not re.search(r"^def quant_flash_attn\(", data, re.MULTILINE):
            continue
        definitions = {node.name for node in ast.parse(data).body if isinstance(node, ast.FunctionDef)}
        if {"quant_flash_attn", "quant_flash_attn_metadata"} <= definitions:
            paths.append(path.relative_to(package))
    cpp = Path("csrc/attention/quant_flash_attn.cpp")
    if not paths or not (package / cpp).is_file():
        raise RuntimeError("Source package must contain QFA Python and csrc/attention/quant_flash_attn.cpp")
    return sorted(paths), cpp


def validate_entry_points():
    points = importlib.metadata.entry_points()
    points = (
        points.select(group="cann_ops_transformer.ops")
        if hasattr(points, "select")
        else points.get("cann_ops_transformer.ops", [])
    )
    for point in points:
        if point.name in ("quant_flash_attn", "quant_flash_attn_metadata") and not point.value.startswith(
            "cann_ops_transformer.ops."
        ):
            raise RuntimeError(f"External QFA entry point would override the private wrapper: {point.value}")


def install(source_python=None, prefix=None, output_root="/mnt"):
    sources = {name: payload_bytes(name) for name in PAYLOAD}
    abi = inspect_abi(sources["python"].decode())
    source = locate_package(source_python)
    helpers = validate_helpers(source)
    wrappers, cpp = find_qfa_sources(source)
    graphs = sorted({path.parent / "graph_convert_quant_flash_attn.py" for path in wrappers})
    for path in graphs:
        if not (source / path).is_file():
            raise RuntimeError(f"Required QFA graph converter is missing: {source / path}")
    validate_entry_points()
    if prefix is None:
        destination = Path(tempfile.mkdtemp(prefix="k3-qfa920-runtime.", dir=output_root)).resolve()
    else:
        destination = Path(prefix).expanduser().resolve()
        if destination == source or destination.is_relative_to(source):
            raise RuntimeError("The isolated prefix must be outside the original package")
        destination.mkdir(parents=True, exist_ok=False)
    python_root = destination / "python"
    copied = python_root / "cann_ops_transformer"
    try:
        shutil.copytree(source, copied, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        replacements = (
            [(path, sources["python"]) for path in wrappers]
            + [(path, sources["graph"]) for path in graphs]
            + [(cpp, sources["cpp"])]
        )
        for relative, data in replacements:
            target = copied / relative
            target.chmod(target.stat().st_mode | 0o200)
            target.write_bytes(data)
            if sha256(target.read_bytes()) != sha256(data):
                raise RuntimeError(f"Copied official source SHA256 mismatch: {target}")
        (destination / "QFA920-CANN-LICENSE").write_bytes(sources["license"])
        (destination / "NOTICE").write_text(
            "Copyright (c) 2026 Huawei Technologies Co., Ltd.\n"
            f"Official QFA host wrappers copied verbatim from CANN/ops-transformer {SOURCE_REVISION}.\n"
            "The complete CANN Open Software License Agreement 2.0 is in QFA920-CANN-LICENSE.\n"
            "The remainder of cann_ops_transformer was copied from the original local installation.\n"
            "Only QFA Python, C++ and graph-converter source files were replaced.\n"
            "No operator kernels or original files changed.\n"
        )
        extensions = destination / "torch_extensions"
        extensions.mkdir()
        env = destination / "env.sh"
        env.write_text(
            "# Source AFTER the existing QFM and DSL env.sh, then start a NEW process.\n"
            f"export PYTHONPATH={shlex.quote(str(python_root) + ':' + str(source.parent))}"
            "${PYTHONPATH:+:$PYTHONPATH}\n"
            f"export TORCH_EXTENSIONS_DIR={shlex.quote(str(extensions))}\n"
        )
        report = {
            "event": "qfa920_wrapper_prepared",
            "source_package": str(source),
            "prefix": str(destination),
            "source_revision": SOURCE_REVISION,
            "env": str(env),
            "python_inspection": abi,
            "replaced_python_sources": [str(path) for path in wrappers],
            "replaced_cpp_source": str(cpp),
            "replaced_graph_sources": [str(path) for path in graphs],
            "official_sha256": {name: sha256(data) for name, data in sources.items()},
            "verified_original_helpers": helpers,
            "native_kernel_validation": "not run; matching CANN 9.2.0 QFA and metadata kernels are required",
            "original_package_changed": False,
            "packages_installed": False,
            "npu_initialized": False,
            "opp_and_library_paths_changed": False,
        }
        (destination / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    except Exception:
        # Keep diagnostics; never remove an original directory or force a retry.
        print(f"Preparation incomplete; isolated files retained at {destination}", file=__import__("sys").stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-python", help="Original python/ directory or cann_ops_transformer package directory")
    parser.add_argument("--prefix", help="New, nonexisting isolated runtime directory")
    parser.add_argument("--output-root", default="/mnt", help="Parent for the generated runtime directory")
    parser.add_argument("--show-license", action="store_true")
    args = parser.parse_args()
    if args.show_license:
        print(payload_bytes("license").decode())
        return
    report = install(args.source_python, args.prefix, args.output_root)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    print("NEXT: source " + shlex.quote(report["env"]))
    print("Restart workers. First native QFA use compiles a host extension only; no Attention kernel is built.")


if __name__ == "__main__":
    main()
