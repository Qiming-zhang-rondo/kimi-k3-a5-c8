#!/usr/bin/env python3
"""Fixed B1/Q4/K8192/H64 recent128: real native C4 plus direct BF16 hot.

The pinned payload supplies all quantization, cache writes, partition, references
and timing. Only the BF16 hot read / merge is replaced. No installed operator,
model, or old payload is modified. This is not a serving implementation.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import traceback

from pinned_payload import activate_payload

VERSION = "sfa-direct-c4-bf16-ab-v4"
ACCEPTANCE_POLICY = ("Native equivalence remains global/worst-row relative L2 <=0.001. "
    "Only fused BF16 zero-query cold1920/hot128 may instead pass a current-run bit-exact "
    "FP64-to-BF16 RNE proof; every difference from native must be an exact midpoint and one BF16 ULP. "
    "The original failed strict comparison is retained. No nonzero-query or FP32 exception.")
ARMS = ("c4_w0", "installed_native_twopass", "direct_native_twopass", "direct_fused")
DIRECT_ARMS = ("direct_native_twopass", "direct_fused")
PAIRS = (("c4_w0", "installed_native_twopass"),
         ("c4_w0", "direct_native_twopass"),
         ("c4_w0", "direct_fused"),
         ("installed_native_twopass", "direct_fused"),
         ("direct_native_twopass", "direct_fused"))


class ControlFailed(RuntimeError):
    pass


def hot_inputs(runner, parts, base=None):
    base = runner.base if base is None else base
    return (base.q_nope, runner.hot_cache, runner.hot_cache,
            parts["recent_ring"], runner.ring_table, base.cu,
            runner.ring_lengths, base.q_rope, runner.hot_rope)


def make_direct_runner_class(bf16_runner_class):
    class DirectRunner(bf16_runner_class):
        def __init__(self, backend, bridge, *, fused, writer="hot_store_partition", aic_count=1):
            if aic_count not in (1, 2, 4):
                raise ValueError("Direct fixed Q4 work requires 1, 2 or 4 AICs")
            super().__init__(backend, recipe="baseline", window=128, writer=writer)
            self.direct_fused = fused
            # The pinned parent allocated this from an actual native cold read.
            # Select the compiled IO specialization; never convert cold states.
            self.expected_output_dtype = str(self.merge_output.dtype)
            if self.expected_output_dtype not in ("torch.bfloat16", "torch.float32"):
                raise ValueError("Native cold/final output must be BF16 or FP32")
            bf16_io = fused and self.expected_output_dtype == "torch.bfloat16"
            self.direct_plan = bridge.prepare(self.base.q_nope, float(self.base.scale),
                                              fused, not fused, bf16_io, aic_count)
            self.direct_last_state = None
            self.metadata.update(
                direct_bridge=True, direct_fused=fused,
                native_c4_read_inside_every_pipeline=True,
                native_c4_writer_unchanged=True,
                cold_state_pointer_policy="runtime tensors passed every invocation",
                direct_output_dtype=self.expected_output_dtype if fused else "torch.bfloat16",
                expected_cold_output_dtype=self.expected_output_dtype,
                expected_pipeline_output_dtype=self.expected_output_dtype,
                compiled_fused_bf16_io=bf16_io, cold_state_cast_kernel_calls=0,
                direct_configuration=dict(self.direct_plan.configuration()),
                direct_aic_count=aic_count, direct_aiv_count=2*aic_count,
                acceptance_policy=ACCEPTANCE_POLICY,
                merge="native_hot_epilogue" if fused else "unchanged payload Triton",
                production_validation="NOT_VALIDATED")

        def _read(self, parts):
            # Never cache this result at setup: the installed C4 reader allocates
            # output tensors on every eager call, and graph pools own replay data.
            left = self._cold_read(parts)
            inputs = hot_inputs(self, parts)
            if self.direct_fused:
                self.direct_last_state = self.direct_plan.fused_hot(
                    *inputs, *left, parts["old_present"], parts["recent_present"])
                return self.direct_last_state[0]
            self.direct_last_state = self.direct_plan.native_hot(*inputs)
            return self._merge(left, self.direct_last_state, parts)

        def direct_state_buffers(self):
            return tuple(self.direct_last_state or ())

    return DirectRunner


def require(check, label):
    if not check:
        raise ControlFailed(label)


def strict_compare(actual, reference, services, limit):
    result = services["compare"](actual, reference)
    result["passed"] = services["acceptable"](result, limit)
    result["relative_l2_limit"] = limit
    result["worst_row_l2_limit"] = limit
    return result


def require_same_dtype(actual, reference, label):
    require(str(actual.dtype) == str(reference.dtype),
            f"{label}: actual dtype {actual.dtype} != reference dtype {reference.dtype}")


def state_diagnostics(state, host, bytes_host=None, arrays=None, prefix="state"):
    """Record original dtypes and values before rejecting a state contract.

    Numeric NPZ views are FP32 host copies; optional byte arrays preserve the
    exact BF16/FP32 device representation. All copies are outside measurement.
    """
    import numpy as np
    fields = {}
    for field, tensor in zip(("out", "max", "sum"), state):
        values = host(tensor)
        finite_values = values[np.isfinite(values)]
        fields[field] = dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
            stride=list(tensor.stride()), contiguous=bool(tensor.is_contiguous()),
            nonfinite_count=int(values.size-finite_values.size),
            min=float(finite_values.min()) if finite_values.size else None,
            max=float(finite_values.max()) if finite_values.size else None)
        if arrays is not None:
            arrays[f"{prefix}_{field}"] = values
            if bytes_host is not None:
                arrays[f"{prefix}_{field}_raw_bytes"] = bytes_host(tensor)
    return fields


def validate_cold_state(state, expected_dtype):
    require(str(state[0].dtype) in ("torch.bfloat16", "torch.float32"),
            "Native C4 output must be BF16 or FP32; no implicit conversion")
    require(str(state[0].dtype) == expected_dtype,
            "Native C4 output dtype changed after bridge specialization selection")
    require(all(str(t.dtype) == "torch.float32" for t in state[1:]),
            "Native C4 maximum and sum must remain FP32")
    require(all(t.is_contiguous() for t in state), "Native C4 states must already be contiguous")


def evaluate_output_contract(prepared_tensor, pipeline, matching, bf16_math,
                             args, backend, arrays, name, evaluate, round_reference, compare):
    """Round only the host mathematical oracle, never the device cold tensor."""
    rounded = round_reference(matching, prepared_tensor.dtype)
    metrics, output = evaluate(lambda: prepared_tensor, pipeline, rounded, bf16_math, None,
                               args, backend, arrays, name)
    metrics.update(output_dtype=str(prepared_tensor.dtype),
        matching_reference_contract="Mathematical output rounded once to this arm's actual destination dtype",
        kernel_vs_unrounded_matching_reference=compare(output, matching),
        pipeline_vs_unrounded_matching_reference=compare(arrays[name+"_write_read"], matching),
        destination_rounding_vs_unrounded_matching_reference=compare(rounded, matching))
    arrays[name+"_unrounded_matching_reference"] = matching
    return metrics, output


def zero_query_exact_rne_gate(cold, hot, actual, native, *, cold_dtype, hot_dtype, output_dtype):
    """Narrow proof using this invocation's states; never read an earlier report."""
    import numpy as np
    from diagnose_merge_rounding import bf16_neighbors, bits, ordered_codes
    co, cm, cs = (np.asarray(value) for value in cold)
    ho, hm, hs = (np.asarray(value) for value in hot)
    actual, native = np.asarray(actual), np.asarray(native)
    out_shape, stat_shape = (4, 64, 512), (4, 64, 1)
    result = dict(passed=False, basis="exact_zero_query_BF16_RNE",
        strict_relative_l2_limit_unchanged=.001, states_source="current invocation; no saved-result trust",
        dtype_contract=all(str(dtype) == "torch.bfloat16" for dtype in (cold_dtype,hot_dtype,output_dtype)),
        shape_contract=all(v.shape == out_shape for v in (co,ho,actual,native)) and
                       all(v.shape == stat_shape for v in (cm,cs,hm,hs)))
    if not result["dtype_contract"] or not result["shape_contract"]:
        return result
    result["all_finite"] = all(bool(np.isfinite(value).all()) for value in (co,cm,cs,ho,hm,hs,actual,native))
    result["exact_zero_query_statistics"] = bool((cm == 0).all() and (hm == 0).all() and
                                                  (cs == 1920).all() and (hs == 128).all())
    result["all_outputs_on_bf16_grid"] = all(bool(
        np.array_equal(value, value.astype(np.float32)) and
        ((value.astype(np.float32).view(np.uint32) & 0xffff) == 0).all()) for value in (co,ho,actual,native))
    if not all(result[key] for key in ("all_finite", "exact_zero_query_statistics", "all_outputs_on_bf16_grid")):
        return result
    precise = (15.0*co.astype(np.float64)+ho.astype(np.float64))/16.0
    oracle, _, tie, _ = bf16_neighbors(precise)
    # Preserve signed zero in the destination's bit-exact output contract.
    oracle = np.where(precise == 0, np.copysign(np.float32(0), precise), oracle).astype(np.float32)
    actual_codes, native_codes, oracle_codes = bits(actual), bits(native), bits(oracle)
    mismatch = actual_codes != native_codes
    ulp = np.abs(ordered_codes(actual)-ordered_codes(native))
    invalid_mismatch = mismatch & (~tie | (ulp != 1))
    oracle_mismatch = actual_codes != oracle_codes
    result.update(actual_oracle_bit_mismatch_count=int(oracle_mismatch.sum()),
        actual_native_bit_mismatch_count=int(mismatch.sum()),
        native_mismatches_at_exact_midpoint=int((mismatch & tie).sum()),
        native_mismatches_not_at_exact_midpoint=int((mismatch & ~tie).sum()),
        native_mismatches_not_one_ulp=int((mismatch & (ulp != 1)).sum()),
        passed=bool(not oracle_mismatch.any() and not invalid_mismatch.any()))
    return result


def graph_replay_control(call, buffers, torch, host, services):
    """Poison both bridge intermediates and final output before graph replay.

    Static-input graph/eager equality alone cannot prove an external ASC launch
    was captured. If the direct hot launch is absent, poisoned hot states survive
    (two-pass) or the poisoned final output survives (fused), failing this gate.
    """
    eager_output = call()
    reference = host(eager_output).copy()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = call()
    require_same_dtype(output, eager_output, "Graph/eager output contract")
    poisoned = []
    for tensor in (*buffers(), output):
        if tensor.numel() and tensor.is_floating_point():
            tensor.fill_(float("nan"))
            poisoned.append(list(tensor.shape))
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    check = strict_compare(host(output), reference, services, .001)
    check.update(poisoned_buffers=poisoned, one_pipeline_per_graph=True,
                 scope="capture must overwrite poisoned bridge states and final output")
    require(check["passed"], "Graph replay did not reproduce output after bridge-state poison")
    return check


def readonly_preflight(runner):
    """Untimed value checks required by the bridge's shape-only hot path."""
    import numpy as np
    base = runner.base
    require((runner.capacity, runner.hot_width) == (256, 128), "Fixed bridge requires ring256/compact128")
    require(tuple(base.q_nope.shape) == (4, 64, 512), "Query shape mismatch")
    require(str(base.q_nope.dtype) == "torch.bfloat16", "BF16 query required")
    require(tuple(runner.hot_cache.shape) == (2, 128, 1, 512), "One KV head required")
    require(tuple(runner.hot_rope.shape) == (2, 128, 1, 64), "BF16 RoPE layout mismatch")
    require(np.array_equal(base.cu.cpu().numpy(), [4]), "Query cumulative length must be [4]")
    require(np.array_equal(runner.ring_lengths.cpu().numpy(), [256]), "Ring physical length must be [256]")
    require(np.array_equal(runner.ring_table.cpu().numpy(), [[0, 1]]), "Identity two-page ring table required")
    parts = runner.partition()
    indices = parts["recent_ring"].cpu().numpy()
    require(indices.shape == (4, 1, 128) and np.all((indices >= 0) & (indices < 256)),
            "Direct BF16 hot indices must have 128 valid ring slots per row")
    require(all(len(np.unique(row)) == 128 for row in indices[:, 0]), "Hot indices must be unique")
    require(np.all(parts["old_present"].cpu().numpy()) and
            np.all(parts["recent_present"].cpu().numpy()), "Only nonempty cold/hot dispatch is supported")
    require(runner.metadata["old_counts"] == [1920]*4 and runner.metadata["hot_counts"] == [128]*4,
            "Fixed experiment requires selected cold1920/hot128")
    return dict(passed=True, hot_count=[128]*4, cold_count=[1920]*4,
                ring_slots=256, compact=True, real_empty_dispatch_validated=False)


def reader_controls(runner, native_plan, services, arrays=None, label="direct", progress=None):
    """Use real native C4 states and independent installed BF16 hot states."""
    import numpy as np
    base, host, torch = runner.base, runner.backend.host, runner.base.torch
    checks = {}
    for zero in (False, True):
        view = base
        if zero:
            view = copy.copy(base)
            view.q_nope, view.q_rope = torch.zeros_like(base.q_nope), torch.zeros_like(base.q_rope)
            if hasattr(base, "query"):
                # Pinned _read_padded('c4') reads this BF16 concatenation,
                # not q_nope/q_rope. Its dequant scale belongs to KV, not Q.
                view.query = torch.zeros_like(base.query)
            if hasattr(base, "raw_result_shapes"):
                view.raw_result_shapes = {}
        parts = runner.prepared
        left = runner._cold_read(parts, base=view)
        prefix = "zero_query" if zero else "nonzero_query"
        # Persist this before any dtype/shape gate so a new installed binding
        # contract is diagnosable without rerunning or casting its output.
        checks[prefix+"_cold_state"] = state_diagnostics(
            left, host, base.bytes_host, arrays, f"{label}_{prefix}_cold")
        if progress:
            progress(checks)
        validate_cold_state(left, runner.expected_output_dtype)
        installed = runner._hot_read(parts, base=view)
        direct = native_plan.native_hot(*hot_inputs(runner, parts, view))
        if arrays is not None:
            for side, state in (("cold", left), ("installed_hot", installed), ("direct_hot", direct)):
                for field, tensor in zip(("out", "max", "sum"), state):
                    arrays[f"{label}_{prefix}_{side}_{field}"] = host(tensor)
        require_same_dtype(direct[0], installed[0], "Direct/installed native hot")
        checks[prefix+"_direct_hot_vs_installed"] = strict_compare(host(direct[0]), host(installed[0]), services, .001)
        reference_tensor = runner._merge(left, installed, parts)
        reference = host(reference_tensor).copy()
        if runner.direct_fused:
            actual = runner.direct_plan.fused_hot(*hot_inputs(runner, parts, view),
                *left, parts["old_present"], parts["recent_present"])[0]
        else:
            actual = runner._merge(left, direct, parts)
        require_same_dtype(actual, reference_tensor, "Fused or direct two-pass / installed merge")
        strict_merge = strict_compare(host(actual), reference, services, .001)
        checks[prefix+"_same_native_states_merge"] = strict_merge
        merge_acceptance = dict(passed=strict_merge["passed"], basis="strict_native_relative_l2")
        if zero and runner.direct_fused and not strict_merge["passed"]:
            proof = zero_query_exact_rne_gate(tuple(host(t) for t in left), tuple(host(t) for t in direct),
                host(actual), reference, cold_dtype=left[0].dtype, hot_dtype=direct[0].dtype, output_dtype=actual.dtype)
            checks["zero_query_exact_rne_gate"] = proof
            if proof["passed"]:
                merge_acceptance = dict(passed=True, basis="exact_zero_query_BF16_RNE",
                                        original_strict_comparison_passed=False)
        checks[prefix+"_merge_acceptance"] = merge_acceptance
        if arrays is not None:
            arrays[f"{label}_{prefix}_actual"] = host(actual)
            arrays[f"{label}_{prefix}_native_reference"] = reference
        if zero:
            for label, state, count in (("cold", left, 1920), ("direct_hot", direct, 128)):
                maximum, total = (host(t).astype(np.float64) for t in state[1:])
                with np.errstate(over="ignore", invalid="ignore"):
                    actual_count = np.exp(maximum)*total
                finite = bool(np.isfinite(actual_count).all() and np.isfinite(maximum).all()
                              and np.isfinite(total).all() and (total > 0).all())
                error = float(np.max(np.abs(actual_count-count))) if finite else None
                checks["zero_query_"+label+"_counts"] = dict(passed=finite and error <= max(.02, count*1e-4),
                    expected_count=count, max_abs_count_error=error, finite=finite)
    # Each merge's explicit acceptance verdict is authoritative; its original
    # strict result remains untouched for review, including a failed zeroQ row.
    checks["passed"] = all(item["passed"] for key,item in checks.items()
        if isinstance(item, dict) and "passed" in item and not key.endswith("_same_native_states_merge"))
    checks["acceptance_policy"] = ACCEPTANCE_POLICY
    return checks


def timed_pairs(controls, calls, args, benchmark, pair_benchmark):
    require(all(controls.get(name) for name in ARMS), "All four arms must pass correctness before any timing")
    result = {}
    for baseline, candidate in PAIRS:
        result[candidate+"_vs_"+baseline] = pair_benchmark(calls[baseline], calls[candidate], args, benchmark)
        for value in result[candidate+"_vs_"+baseline].values():
            require(value.get("status") == "MEASURED", "Paired timing failed; do not interpret partial pairs")
    return result


def prepared_reader_stages(runners):
    """Independent stages only: fixed real partitions/states, never full-pipeline timing."""
    installed, native, fused = (runners[name] for name in
        ("installed_native_twopass", "direct_native_twopass", "direct_fused"))
    iparts, nparts, fparts = (runner.partition() for runner in (installed, native, fused))
    left, right = installed._cold_read(iparts), installed._hot_read(iparts)
    fused_left = fused._cold_read(fparts)
    native_args, fused_args = hot_inputs(native, nparts), hot_inputs(fused, fparts)
    return {
        "native_c4_cold_read": lambda: installed._cold_read(iparts)[0],
        "installed_bf16_hot_read": lambda: installed._hot_read(iparts)[0],
        "direct_bf16_hot_read": lambda: native.direct_plan.native_hot(*native_args)[0],
        "installed_external_merge": lambda: installed._merge(left, right, iparts),
        "direct_fused_hot_read_merge": lambda: fused.direct_plan.fused_hot(
            *fused_args, *fused_left, fparts["old_present"], fparts["recent_present"])[0],
    }


def load_bridge(library, sdk_root=None, build_directory=None):
    library = Path(library).resolve()
    if library.name != "libsfa_torch_direct.so" or not library.is_file():
        raise ValueError("--bridge-library must name the built libsfa_torch_direct.so")
    path = Path(__file__).resolve().parents[1]/"experimental/torch/load_sfa_direct.py"
    spec = importlib.util.spec_from_file_location("sfa_direct_jit_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_sfa_direct(library.parent, sdk_root=sdk_root, build_directory=build_directory)


def execute(args, report, persist):
    import numpy as np
    report["payload"] = activate_payload(args.payload_dir)
    from kv_native_twopass import NativeTwoPassBackend
    from kv_bf16_hot_ab import Bf16HotRunner, merge_reference_for_dtype
    from kv_window_common import attention_reference, load_case, save_case, synthetic_case
    from test_kv_bf16_hot_ab_a5 import validate_fixed_case
    from test_kv_hot_fused_a5 import compare, acceptable, evaluate, paired_benchmark, identity
    from kv_window_backend import benchmark
    services = dict(compare=compare, acceptable=acceptable)
    data = validate_fixed_case(load_case(args.input) if args.input else synthetic_case(
        batch=1, queries=4, keys=8192, heads=64, dim=512, rope_dim=64,
        topk=2048, page=128, seed=args.seed))
    save_case(args.output/"inputs.npz", data)
    report["input_sha256"] = identity(data)
    backend = NativeTwoPassBackend(data, device=args.device)
    report["environment"] = backend.environment
    bridge = load_bridge(args.bridge_library, args.sdk_root, args.build_directory)
    report["bridge_build"] = getattr(bridge, "_sfa_build_identity", {})
    report["bridge_configuration"] = dict(bridge.configuration())
    runner_class = make_direct_runner_class(Bf16HotRunner)
    installed = Bf16HotRunner(backend, writer=args.writer)
    runners = dict(c4_w0=backend.prepare("baseline", 0), installed_native_twopass=installed,
        direct_native_twopass=runner_class(backend, bridge, fused=False, writer=args.writer,
                                         aic_count=args.direct_aic_count),
        direct_fused=runner_class(backend, bridge, fused=True, writer=args.writer,
                                 aic_count=args.direct_aic_count))
    report["implementations"] = {name: runner.metadata for name, runner in runners.items()}
    arrays, outputs, passed = {}, {}, {}
    report["controls"] = {}
    # This unchanged installed two-pass path is both the independent accuracy
    # reference and the mandatory current performance baseline. Direct native
    # two-pass is an additional bridge-overhead diagnostic, never its substitute.
    for label, method in (("partition", installed.partition_diagnostics),
                          ("writer", installed.writer_diagnostics),
                          ("zero_query", installed.zero_query_diagnostics),
                          ("reader", installed.reader_diagnostics)):
        report["controls"]["installed_"+label] = method()
        persist()
        require(report["controls"]["installed_"+label]["passed"], "Installed native control failed: "+label)
    installed_output_tensor = installed.run(write=True, dynamic=True)
    installed_output = backend.host(installed_output_tensor).copy()
    arrays["installed_native_twopass"] = installed_output
    bf16_math = attention_reference(data)
    cold = runners["c4_w0"].decoded_kv()
    matching = {"c4_w0": attention_reference(data, kv_c4=cold)}
    mixed = attention_reference(data, kv_c4=cold, window=128, hot_kv=data["kv"])
    matching.update(installed_native_twopass=mixed, direct_native_twopass=mixed, direct_fused=mixed)
    native_plan = runners["direct_native_twopass"].direct_plan
    calls = {name: (lambda runner=runner: runner.run(write=True, dynamic=True))
             for name, runner in runners.items()}
    for name, runner in runners.items():
        checks = report["controls"][name] = {}
        if name in DIRECT_ARMS:
            checks["fixed_contract"] = readonly_preflight(runner)
            for label, method in (("partition", runner.partition_diagnostics), ("writer", runner.writer_diagnostics)):
                checks[label] = method()
                require(checks[label]["passed"], name+" "+label+" failed")
            def reader_progress(partial):
                checks["reader"] = partial
                np.savez_compressed(args.output/"outputs.npz", **arrays)
                persist()
            checks["reader"] = reader_controls(runner, native_plan, services, arrays, name, reader_progress)
            np.savez_compressed(args.output/"outputs.npz", **arrays)
            persist()
            require(checks["reader"]["passed"], name+" direct hot/native-state comparison failed")
        prepared_tensor = runner.call()
        metrics, output = evaluate_output_contract(prepared_tensor, calls[name], matching[name], bf16_math,
            args, backend, arrays, name, evaluate, merge_reference_for_dtype, compare)
        checks.update(metrics)
        np.savez_compressed(args.output/"outputs.npz", **arrays)
        persist()
        require(metrics["control_passed"], name+" matching mathematical reference failed")
        outputs[name] = output
        if name in DIRECT_ARMS:
            pipeline_tensor = calls[name]()
            require_same_dtype(pipeline_tensor, installed_output_tensor, "Full pipeline / installed native two-pass")
            checks["pipeline_vs_installed_native_twopass"] = strict_compare(
                backend.host(pipeline_tensor), installed_output, services, .001)
            require(checks["pipeline_vs_installed_native_twopass"]["passed"], name+" strict installed two-pass mismatch")
        passed[name] = True
        persist()
    report["controls"]["fused_vs_direct_native_twopass"] = strict_compare(
        outputs["direct_fused"], outputs["direct_native_twopass"], services, .001)
    require(report["controls"]["fused_vs_direct_native_twopass"]["passed"], "Strict direct two-pass comparison failed")
    if args.timing == "graph":
        report["graph_capture_controls"] = {}
        for name, runner in runners.items():
            buffers = runner.direct_state_buffers if name in DIRECT_ARMS else lambda: ()
            report["graph_capture_controls"][name] = graph_replay_control(
                calls[name], buffers, backend.base.torch, backend.host, services)
            persist()
    np.savez_compressed(args.output/"outputs.npz", **arrays)
    report["timings"] = ({} if args.timing == "none" else
                         timed_pairs(passed, calls, args, benchmark, paired_benchmark))
    persist()
    if args.stages and args.timing != "none":
        report["stage_scope"] = ("Independent prepared reader stages after full-pipeline pairs; "
            "real C4 cold states and fixed partitions prepared outside these stage timers; "
            "not full pipeline latency and not additive")
        report["stage_timings"] = {}
        for stage, call in prepared_reader_stages(runners).items():
            report["stage_timings"][stage] = benchmark(call, warmup=args.warmup,
                iterations=args.iterations, repeats=min(args.repeats, 3), graph=args.timing == "graph")
            persist()
    require(identity(data) == report["input_sha256"], "Input data mutated")
    report.update(status="VALIDATED_ONLY" if args.timing == "none" else "MEASURED",
                  c4_pipeline_validated=True, native_twopass_compared=True)
    persist()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--payload-dir", type=Path, required=True, help="Unmodified hotkv directory from pinned 4a5fd67 payload")
    p.add_argument("--bridge-library", type=Path, required=True, help="CMake-built libsfa_torch_direct.so; JIT compiles only the thin Torch wrapper")
    p.add_argument("--sdk-root", type=Path)
    p.add_argument("--build-directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--input", type=Path)
    p.add_argument("--device", required=True, help="Explicit NPU device already checked idle; no implicit device 0")
    p.add_argument("--writer", choices=("native", "hot_store_partition"), default="hot_store_partition")
    p.add_argument("--direct-aic-count", type=int, choices=(1, 2, 4), default=1,
                   help="Direct hot task count; changes native and fused arms together, never installed baseline")
    p.add_argument("--stages", action="store_true", help="Add independent prepared reader-stage diagnostics after complete pipeline timing")
    p.add_argument("--timing", choices=("none", "eager", "graph"), default="graph")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--fault-marker", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.warmup < 0 or min(args.iterations, args.repeats) < 1:
        raise ValueError("Nonnegative warmup and positive iterations/repeats required")
    # Deliberately not user-relaxable. Native equivalence and mathematical
    # reference have different tolerances, both applied to every query row.
    args.control_l2, args.pipeline_l2 = .01, .001
    args.output = args.output.resolve()
    marker = args.fault_marker or args.output.parent/"DEVICE_FAULT.json"
    if marker.exists():
        raise RuntimeError(f"Prior device fault marker exists: {marker}; do not continue launching")
    marker.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(version=VERSION, status="RUNNING", c4_pipeline_validated=False,
                  native_twopass_compared=False, real_hot_empty_dispatch_validated=False,
                  production_validation="NOT_VALIDATED",
                  acceptance_policy=ACCEPTANCE_POLICY,
                  scope="fixed all128-hot-hit C4 cold + BF16 hot full write/partition/read/merge pipeline",
                  performance_baseline="installed_native_twopass is mandatory; direct_native_twopass isolates the direct launch/kernel path, not pure host overhead",
                  timing_scope="one pipeline per graph; repeated replays outside capture; no historical latency constants",
                  args={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    def persist():
        (args.output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    persist()
    try:
        execute(args, report, persist)
    except Exception as exc:
        text = traceback.format_exc()
        fault = any(code in text for code in ("507015", "ACL_ERROR_RT_AICORE_EXCEPTION", "GM invalid"))
        report.update(status="DEVICE_ERROR" if fault else "CONTROL_FAILED" if isinstance(exc, ControlFailed) else "ERROR",
                      traceback=text)
        if fault:
            marker.write_text(json.dumps(dict(status="DEVICE_FAULT", results=str(args.output/"results.json"))), encoding="utf-8")
        print(text, file=sys.stderr)
    finally:
        persist()
    print(f"{report['status']}: {args.output/'results.json'}", flush=True)
    return 0 if report["status"] in ("VALIDATED_ONLY", "MEASURED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
