#!/usr/bin/env python3
"""Run ONE isolated real-model accuracy arm; never start a serving process.

The engine JSON is supplied by the deployment owner/inner agent, preserving
checkpoint, TP, quantization and other model options. This is a wiring smoke,
not a substitute for the existing GLM task-accuracy evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def flush_worker_evidence(worker):
    # vLLM collective_rpc passes its worker as the callable's first argument.
    from sfa_recent128.runtime import flush_all
    return flush_all()


def engine_options(options, *, arm, source_root, library_dir, diagnostics, max_query_chunk,
                   native_accuracy_preprocess=False):
    value = json.loads(json.dumps(options))
    if not value.get("model"):
        raise ValueError("engine JSON must name the actual model checkpoint")
    if value.get("enforce_eager") is not True or value.get("max_num_seqs") != 1:
        raise ValueError("Declare enforce_eager=true and max_num_seqs=1 in the experiment engine JSON")
    if value.get("enable_prefix_caching", True):
        raise ValueError("Declare enable_prefix_caching=false explicitly")
    if value.get("speculative_config") or value.get("kv_transfer_config"):
        raise ValueError("The initial probe excludes MTP/speculation and PD")
    for name in ("prefill_context_parallel_size", "decode_context_parallel_size", "data_parallel_size"):
        if value.get(name, 1) != 1:
            raise ValueError(f"Initial probe requires {name}=1")
    attention = value.get("attention_config", {})
    if isinstance(attention, str):
        attention = json.loads(attention)
    if attention.get("mla_kv_quant_mode") != "mxfp4":
        raise ValueError("Declare attention_config.mla_kv_quant_mode=mxfp4; do not alter the quantization recipe")
    if value.get("enable_chunked_prefill") is not True or "max_num_batched_tokens" not in value:
        raise ValueError("Declare enable_chunked_prefill=true and max_num_batched_tokens explicitly")
    if value["max_num_batched_tokens"] > max_query_chunk:
        raise ValueError("max_num_batched_tokens exceeds the declared ring query-chunk bound")
    additional = value.get("additional_config")
    if additional is None:
        additional = {}
        value["additional_config"] = additional
    if not isinstance(additional, dict):
        raise ValueError("additional_config must be a JSON object")
    additional["sfa_recent128"] = dict(window=0 if arm == "off" else 128,
        execution="accuracy_eager", max_query_chunk=max_query_chunk,
        reader="native" if arm in ("off", "native") else "fused_required",
        preprocess="native_accuracy" if native_accuracy_preprocess else "preserve",
        source_root=str(source_root), library_dir=str(library_dir) if library_dir else None,
        diagnostics_path=str(diagnostics))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-json", type=Path, required=True)
    parser.add_argument("--prompts-jsonl", type=Path, required=True, help="id,prompt JSON objects; use actual model inputs")
    parser.add_argument("--mxllm-root", type=Path, required=True)
    parser.add_argument("--operator-source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--library-dir", type=Path)
    parser.add_argument("--arm", choices=("off", "native", "fused"), required=True)
    parser.add_argument("--max-query-chunk", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--native-accuracy-preprocess", action="store_true",
        help="Explicitly select NATIVE before weight conversion in BOTH OFF/ON arms; no serving-performance claim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new isolated output directory; existing evidence is never overwritten")
    args.output.mkdir(parents=True)
    options = json.loads(args.engine_json.read_text())
    options = engine_options(options, arm=args.arm, source_root=args.operator_source.resolve(),
        library_dir=args.library_dir.resolve() if args.library_dir else None,
        diagnostics=args.output.resolve()/"layers", max_query_chunk=args.max_query_chunk,
        native_accuracy_preprocess=args.native_accuracy_preprocess)
    prompts = [json.loads(line) for line in args.prompts_jsonl.read_text().splitlines() if line.strip()]
    if not prompts or any(not isinstance(p.get("prompt"), str) for p in prompts):
        parser.error("Provide nonempty actual text prompts")
    (args.output/"engine.json").write_text(json.dumps(options, ensure_ascii=False, indent=2)+"\n")
    fingerprint = hashlib.sha256(args.prompts_jsonl.read_bytes()).hexdigest()
    sys.path.insert(0, str(args.mxllm_root.resolve()))
    import ascend_vllm
    if not Path(ascend_vllm.__file__).resolve().is_relative_to(args.mxllm_root.resolve()):
        raise RuntimeError("Imported mxllm differs from the isolated overlay checkout")
    from vllm import LLM, SamplingParams
    llm = LLM(**options)
    sampling = SamplingParams(temperature=0, seed=0, max_tokens=args.max_tokens, logprobs=5)
    with (args.output/"outputs.jsonl").open("w") as stream:
        for i, prompt in enumerate(prompts):
            answer = llm.generate([prompt["prompt"]], sampling, use_tqdm=False)[0]
            completion = answer.outputs[0]
            first = completion.logprobs[0] if completion.logprobs else {}
            item = dict(id=prompt.get("id", i), prompt_token_ids=answer.prompt_token_ids,
                token_ids=list(completion.token_ids), text=completion.text,
                first_token_top_logprobs={str(k): float(v.logprob) for k,v in first.items()})
            stream.write(json.dumps(item, ensure_ascii=False)+"\n")
            stream.flush()
    worker_reports = llm.collective_rpc(flush_worker_evidence)
    (args.output/"probe.json").write_text(json.dumps(dict(status="GENERATION_FINISHED", arm=args.arm,
        prompt_sha256=fingerprint, prompts=len(prompts), max_tokens=args.max_tokens,
        scope="real-model wiring smoke; task accuracy and serving performance remain unmeasured",
        installed_mxllm=str(Path(ascend_vllm.__file__).resolve()), worker_reports=worker_reports), indent=2)+"\n")
    # Worker atexit hooks write the final layer counters on orderly process exit.


if __name__ == "__main__":
    main()
