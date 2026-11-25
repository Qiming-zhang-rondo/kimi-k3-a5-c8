#!/usr/bin/env python3
"""Same-round single-pass versus the verified ABI4 cold-store champion.

Prepared-read and write+read are separate comparisons. Both pipelines use the
SAME native quantizer and combined cold/hot/partition writer. The candidate
still pays for that writer's unused partition: a conservative implementation,
not a claim to an optimized single-pass writer or serving performance.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "direct_bridge"))

ARMS = ("old_champion", "single_pass")
SCOPES = ("prepared_read", "write_read")
FAULT_CODES = ("507015", "507035", "507053", "507036", "507011",
               "ACL_ERROR_RT_AICORE_EXCEPTION", "GM invalid")


class ControlFailed(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ControlFailed(message)


def make_scope_calls(champion, single_read):
    def single_pipeline():
        champion._write()
        return single_read()
    return {
        "prepared_read": {"old_champion": champion.call, "single_pass": single_read},
        "write_read": {"old_champion": lambda: champion.run(write=True, dynamic=True),
                       "single_pass": single_pipeline},
    }


def scope_metadata():
    return {
        "prepared_read": dict(scope="prepared native read only; excludes every write and partition",
            old_attention_launches=2, new_attention_launches=1, writes=False),
        "write_read": dict(scope="same native quantize+combined cold/hot/partition writer, then each reader",
            old_attention_launches=2, new_attention_launches=1, writes=True,
            shared_writer=True, candidate_pays_unused_partition=True,
            candidate_writer_claim="conservative shared-writer result; not an optimized single-pass writer"),
    }


def validate_champion_configuration(configuration, expected_rows=16):
    require(configuration.get("abi_version") == 4, "Old champion must be the existing ABI4 library")
    rows = configuration.get("bf16_merge_tile_rows_effective",
        configuration.get("bf16_merge_tile_rows", configuration.get("merge_tile_rows")))
    require(rows == expected_rows,
            f"Old champion BF16 rows={rows}, expected verified rows={expected_rows}; do not substitute a slower default")


def measure_scopes(scopes, controls, args, benchmark, paired_benchmark, persist=lambda: None, results=None):
    require(all(controls.get(scope, {}).get(arm, {}).get("passed", False)
                for scope in SCOPES for arm in ARMS),
            "Both arms and both scopes must pass complete math gates before ANY timing")
    measured = {} if results is None else results
    for scope in SCOPES:
        measured[scope] = paired_benchmark(scopes[scope]["old_champion"],
                                          scopes[scope]["single_pass"], args, benchmark)
        for mode, result in measured[scope].items():
            require(result.get("status") == "MEASURED" and len(result.get("rounds", [])) == args.repeats,
                    f"Incomplete {scope}/{mode} pair timing")
        persist()
    return measured


def writer_byte_control(runner):
    """Actual combined writer versus native cold write and CPU hot scatter.

    All cache bytes and partition outputs are poisoned, then restored. Device
    faults skip cleanup to avoid issuing another device operation after fault.
    """
    import numpy as np
    from cold_store_runner import cache_views, partition_reference, exact_field, _poison_parts
    torch = runner.base.torch
    views = cache_views(runner)
    outputs = runner._hot_prepare.outputs
    saved = {name: value.clone() for name, value in views.items()}
    saved_parts = {name: value.clone() for name, value in outputs.items()}
    expected_parts = partition_reference(runner)
    host = lambda value: value.detach().cpu().numpy().copy()
    device_fault = False
    try:
        for value in views.values():
            value.fill_(165)
        _poison_parts(outputs, expected_parts)
        runner._write()
        actual_cache = {name: host(value) for name, value in views.items()}
        actual_parts = {name: host(value) for name, value in outputs.items()}
        views["cold"].fill_(165)
        runner._cold_write()  # The independent unchanged native quantize/scatter.
        expected = {"cold": host(views["cold"])}
        slots = host(runner.hot_slots)
        for name, source, width in (("hot_latent", runner.base.current_kv, 1024),
                                    ("hot_rope", runner.base.current_rope, 128)):
            expected[name] = np.full(actual_cache[name].shape, 165, np.uint8)
            expected[name].reshape(-1, width)[slots] = host(source.view(torch.uint8)).reshape(-1, width)
        fields = {name: exact_field(actual_cache[name], expected[name]) for name in views}
        fields.update({"partition_"+name: exact_field(actual_parts[name], value)
                       for name, value in expected_parts.items()})
        return dict(passed=all(value["passed"] for value in fields.values()), fields=fields,
            scope="whole-cache poisoned byte equality and actual combined producer integer/bool equality; restored",
            native_quantizer_unchanged=True, original_cold_index_copy_only_in_untimed_oracle=True)
    except Exception as error:
        device_fault = any(code in str(error) for code in FAULT_CODES)
        raise
    finally:
        if not device_fault:
            for name, value in views.items():
                value.copy_(saved[name])
            for name, value in outputs.items():
                value.copy_(saved_parts[name])


def execute(args, report, persist):
    import numpy as np
    from pinned_payload import activate_payload
    report["payload"] = activate_payload(args.payload_dir)
    from kv_native_twopass import NativeTwoPassBackend
    from variable_bf16_runner import VariableBf16HotRunner, validate_variable_case
    from kv_window_common import attention_reference, save_case, synthetic_case
    from test_kv_hot_fused_a5 import compare, acceptable, paired_benchmark, identity
    from kv_window_backend import benchmark
    from run_direct_ab import make_direct_runner_class, load_bridge, graph_replay_control
    from cold_store_runner import make_cold_store_runner_class, writer_graph_control
    from load_single_pass import load_single_pass

    data = validate_variable_case(synthetic_case(batch=1, queries=args.queries, keys=args.keys,
        heads=64, dim=512, rope_dim=64, topk=2048, page=128, seed=args.seed),
        queries=args.queries, keys=args.keys)
    save_case(args.output / "inputs.npz", data)
    report["input_sha256"] = identity(data)
    backend = NativeTwoPassBackend(data, device=args.device)
    base, torch = backend.base, backend.base.torch
    report["environment"] = backend.environment
    old_bridge = load_bridge(args.champion_library, args.sdk_root, args.champion_build_directory)
    old_configuration = dict(old_bridge.configuration())
    validate_champion_configuration(old_configuration, args.champion_merge_rows)
    report["champion_configuration"] = old_configuration
    report["champion_build"] = getattr(old_bridge, "_sfa_build_identity", {})
    runner_class = make_cold_store_runner_class(make_direct_runner_class(VariableBf16HotRunner))
    champion = runner_class(backend, old_bridge, fused=True, writer="hot_store_partition",
                           aic_count=args.aic_count, packed_output=False)
    require(champion.expected_output_dtype == "torch.bfloat16", "Verified champion requires actual BF16 cold/final IO")
    report["champion_implementation"] = champion.metadata

    tags_cpu = np.asarray(champion.plan["tags"][0], np.int64)
    positions_cpu = np.asarray(champion.positions_cpu, np.int64)
    tags, positions = base.tensor(tags_cpu, torch.int64), base.tensor(positions_cpu, torch.int64)
    ids = base.indices.detach().cpu().numpy()
    require(ids.shape == (args.queries, 1, 2048), "Original TopK shape differs")
    require(np.all((ids[:,0] >= 0) & (ids[:,0] <= positions_cpu[:,None])), "Invalid/future original TopK")
    require(all(len(np.unique(row)) == 2048 for row in ids[:,0]), "Original TopK contains duplicates")
    hot = ids[:,0] >= positions_cpu[:,None]-127
    require(np.all(hot.sum(axis=1) == 128), "Controlled case must select all128 hot tokens")
    selected = ids[:,0][hot]
    require(np.array_equal(tags_cpu[selected % champion.capacity], selected), "Original BF16 ring tags missing")
    require(np.array_equal(base.cu.detach().cpu().numpy(), [args.queries]), "Actual query lengths differ")
    require(np.array_equal(base.lengths.detach().cpu().numpy(), [args.keys]), "Actual KV lengths differ")
    table = base.table.detach().cpu().numpy()
    require(np.array_equal(np.sort(table[0]), np.arange(args.keys//128)), "Cold page table is not the expected physical permutation")
    require(tuple(champion.cold_cache.shape) == (args.keys//128,128,1,400)
            and champion.cold_cache.dtype == torch.int8, "Native C4 packed400 storage changed")
    report["preflight"] = dict(passed=True, batch=1, queries=args.queries, keys=args.keys, heads=64,
        topk=2048, hot_selected=128, cold_selected=1920, ring_capacity=int(champion.capacity),
        aic_count=args.aic_count, aiv_count=2*args.aic_count, old_bf16_merge_rows=args.champion_merge_rows,
        shared_cache_tensors=True, shared_original_topk=True, new_kernel_reads_unsplit_topk=True,
        new_tags_and_positions="same old champion ring plan; setup only")

    new_bridge = load_single_pass(args.single_pass_library, sdk_root=args.sdk_root,
        build_directory=args.single_pass_build_directory, verbose=args.verbose)
    require(dict(new_bridge.configuration()).get("abi_version") == 1, "New single-pass library must be its independent ABI1")
    report["single_pass_configuration"] = dict(new_bridge.configuration())
    report["single_pass_build"] = getattr(new_bridge, "_single_pass_build_identity", {})
    plan = new_bridge.prepare(base.query, args.keys, champion.capacity, args.aic_count, base.scale, False)
    inputs = (base.query, champion.cold_cache, base.indices, base.table, base.cu, base.lengths,
              champion.hot_cache, champion.hot_rope, tags, positions)
    def single_read():
        return plan.run(*inputs, 128, False)[0]
    scopes = make_scope_calls(champion, single_read)
    report["scopes"] = scope_metadata()
    report["controls"] = {}
    outputs = {}
    with torch.inference_mode():
        report["writer_control"] = writer_byte_control(champion)
        persist()
        require(report["writer_control"]["passed"], "Shared champion writer byte/partition control failed")
        champion._write()
        decoded = champion.decoded_kv()
        matching = attention_reference(data, kv_c4=decoded, window=128, hot_kv=data["kv"])
        bf16_math = attention_reference(data)
        for scope in SCOPES:
            report["controls"][scope] = {}
            for arm in ARMS:
                tensor = scopes[scope][arm]()
                require(tensor.dtype == torch.bfloat16, f"{scope}/{arm} changed native output dtype")
                actual = backend.host(tensor).copy()
                outputs[scope+"_"+arm] = actual
                metric = compare(actual, matching)
                metric.update(passed=acceptable(metric, .01), tolerance=.01,
                    reference="independent FP64 whole-selected-set softmax; no two-pass intermediate rounding")
                report["controls"][scope][arm] = metric
                report.setdefault("accuracy_vs_bf16", {})[scope+"_"+arm] = compare(actual, bf16_math)
                persist()
                require(metric["passed"], f"{scope}/{arm} whole mathematical control failed")
        report["cross_checks"] = {}
        for arm in ARMS:
            metric = compare(outputs["write_read_"+arm], outputs["prepared_read_"+arm])
            metric.update(passed=acceptable(metric, .001), tolerance=.001)
            report["cross_checks"][arm+"_pipeline_vs_prepared"] = metric
            require(metric["passed"], arm+" pipeline/prepared mismatch")
        for scope in SCOPES:
            report["cross_checks"][scope+"_single_vs_champion_diagnostic"] = dict(
                compare(outputs[scope+"_single_pass"], outputs[scope+"_old_champion"]),
                acceptance_gate=False, reason="single-pass has no intermediate BF16 two-pass state rounding")
        np.savez_compressed(args.output/"outputs.npz", mathematical_reference=matching, **outputs)
        report["all_accuracy_controls_passed"] = True
        persist()
        if args.timing == "graph":
            report["graph_controls"] = {}
            buffers = {"old_champion": champion.direct_state_buffers, "single_pass": plan.outputs}
            for scope in SCOPES:
                report["graph_controls"][scope] = {}
                for arm in ARMS:
                    gate = graph_replay_control(scopes[scope][arm], buffers[arm], torch,
                        backend.host, dict(compare=compare, acceptable=acceptable))
                    report["graph_controls"][scope][arm+"_output_replay"] = gate
                    persist()
                    require(gate["passed"], f"{scope}/{arm} output graph replay failed")
                    if scope == "write_read":
                        writer_gate = writer_graph_control(champion, scopes[scope][arm])
                        report["graph_controls"][scope][arm+"_writer_bytes_and_partition"] = writer_gate
                        persist()
                        require(writer_gate["passed"], arm+" writer graph failed to restore poisoned current slots")
        report["timings"] = {}
        if args.timing != "none":
            measure_scopes(scopes, report["controls"], args, benchmark, paired_benchmark,
                           persist, results=report["timings"])
    require(identity(data) == report["input_sha256"], "Original CPU source inputs mutated")
    require(np.array_equal(base.indices.detach().cpu().numpy(), ids), "Original GPU TopK changed")
    report.update(status="VALIDATED_ONLY" if args.timing == "none" else "MEASURED",
                  comparison="same-round old verified combined-writer champion versus new single-pass",
                  model_accuracy="NOT_MEASURED", serving_performance="NOT_MEASURED")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--payload-dir", type=Path, required=True)
    p.add_argument("--single-pass-library", type=Path, required=True)
    p.add_argument("--champion-library", type=Path, required=True)
    p.add_argument("--sdk-root", type=Path)
    p.add_argument("--champion-build-directory", type=Path)
    p.add_argument("--single-pass-build-directory", type=Path)
    p.add_argument("--device", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--queries", type=int, default=4)
    p.add_argument("--keys", type=int, choices=(8192,32768), default=8192)
    p.add_argument("--aic-count", type=int, choices=(1,2,4,8,16), default=None)
    p.add_argument("--champion-merge-rows", type=int, choices=(8,16,32), default=16)
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--timing", choices=("none","eager","graph"), default="none")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--fault-marker", type=Path)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None):
    from run_single_pass import select_aic_count
    args = parser().parse_args(argv)
    try:
        args.aic_count = select_aic_count(args.queries, args.aic_count)
    except ValueError as error:
        parser().error(str(error))
    if args.warmup < 0 or min(args.iterations, args.repeats) < 1:
        parser().error("Invalid timing counts")
    args.output = args.output.expanduser().resolve()
    marker = (args.fault_marker or args.output.parent/"DEVICE_FAULT.json").expanduser().resolve()
    if marker.exists():
        parser().error("Existing device-fault marker blocks another device run")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(version="single-pass-versus-abi4-champion-v1", status="STARTED",
        args={name:str(value) if isinstance(value,Path) else value for name,value in vars(args).items()},
        baseline="same-round old ABI4 fused-hot-merge plus combined cold/hot/partition writer",
        serving_performance="NOT_MEASURED", historical_latency_constants_used=False)
    def persist():
        (args.output/"results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    persist()
    try:
        execute(args, report, persist)
    except Exception as error:
        detail = traceback.format_exc()
        report.update(status="CONTROL_FAILED" if isinstance(error, ControlFailed) else "ERROR", traceback=detail)
        if any(code in detail for code in FAULT_CODES):
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(dict(status="DEVICE_FAULT", results=str(args.output/"results.json")))+"\n")
            report["device_fault_marker"] = str(marker)
    persist()
    print(json.dumps(dict(status=report["status"], results=str(args.output/"results.json"))))
    return 0 if report["status"] in ("VALIDATED_ONLY", "MEASURED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
