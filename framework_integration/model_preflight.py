#!/usr/bin/env python3
"""Read checkpoint JSON only; never import Torch, model code, or download files."""
import argparse
import hashlib
import json
from pathlib import Path


COMPATIBLE="COMPATIBLE_R64_PACKED400"


def _dimension(value):
    return value if type(value) is int and value>=0 else None


def inspect_model_format(engine):
    report=dict(version="recent128-model-format-preflight-v1",status="UNKNOWN_FORMAT",
        model=engine.get("model") if isinstance(engine,dict) else None,
        config_path=None,config_section=None,kv_lora_rank=None,qk_rope_head_dim=None,
        packed_token_bytes=None,expected=dict(kv_lora_rank=512,qk_rope_head_dim=64,packed_token_bytes=400),
        scope="checkpoint JSON format only; no weights, model code, Torch or NPU loaded",
        action="Inner agent must resolve the existing local checkpoint config.json; do not guess from a model name or download/execute model code")
    model=report["model"]
    if not isinstance(model,str) or not model.strip():
        report["reason"]="engine.model must identify the actual local checkpoint"
        return report
    directory=Path(model).expanduser()
    if not directory.is_dir():
        report["reason"]="engine.model is remote or not an existing local checkpoint directory"
        return report
    path=(directory/"config.json").resolve()
    report["config_path"]=str(path)
    try:
        raw=path.read_bytes()
        config=json.loads(raw)
    except (OSError,ValueError) as error:
        report["reason"]="Cannot read the local checkpoint config.json: "+str(error)
        return report
    report["config_sha256"]=hashlib.sha256(raw).hexdigest()
    if not isinstance(config,dict):
        report["reason"]="Checkpoint config.json must contain a JSON object"
        return report
    selected=config.get("text_config",config)
    report["config_section"]="text_config" if "text_config" in config else "root"
    if not isinstance(selected,dict):
        report["reason"]="Checkpoint text_config must contain a JSON object"
        return report
    latent=_dimension(selected.get("kv_lora_rank"))
    report["kv_lora_rank"]=latent
    if "qk_rope_head_dim" in selected:
        rope=_dimension(selected["qk_rope_head_dim"])
        report["rope_dimension_source"]="qk_rope_head_dim"
    else:
        total=_dimension(selected.get("qk_head_dim"))
        nope=_dimension(selected.get("qk_nope_head_dim"))
        rope=total-nope if total is not None and nope is not None and total>=nope else None
        report["rope_dimension_source"]="qk_head_dim - qk_nope_head_dim" if rope is not None else None
    report["qk_rope_head_dim"]=rope
    if latent is None or rope is None:
        report["reason"]="Explicit kv_lora_rank and RoPE dimension are required; absent/invalid values cannot establish the KV format"
        return report
    if latent!=512:
        report.update(status="UNSUPPORTED_LATENT_DIM",reason="This single-pass ABI requires latent dimension 512",
            action="Use an actual compatible checkpoint or implement and validate a distinct kernel contract; do not reshape the cache")
    elif rope==0:
        report.update(status="UNSUPPORTED_R0_PACKED272",packed_token_bytes=272,
            reason="Checkpoint explicitly has NoPE/RoPE0; current single-pass kernel requires RoPE64/packed400",
            action="Stop single-pass model loading. NoPE272 needs a separate kernel contract; do not pad, reinterpret or repack it as packed400")
    elif rope==64:
        report.update(status=COMPATIBLE,packed_token_bytes=400,
            reason="Explicit checkpoint dimensions match the single-pass D512/R64 packed400 contract",
            action="Proceed to the isolated real-model accuracy probe; runtime must still validate actual cache/query shapes and the library entry")
    else:
        report.update(status="UNSUPPORTED_ROPE_DIM",reason="This single-pass ABI requires RoPE dimension 64",
            action="Stop single-pass model loading; implement and validate the actual format separately")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-json",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("Use a new preflight output file")
    report=inspect_model_format(json.loads(args.engine_json.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(status=report["status"],preflight=str(args.output))))
    return 0 if report["status"]==COMPATIBLE else 2


if __name__=="__main__":raise SystemExit(main())
