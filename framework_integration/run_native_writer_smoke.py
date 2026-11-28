#!/usr/bin/env python3
"""Isolated A5 BF16 writer correctness smoke; no CP/model or performance claim.

Run on the target stack with --device npu:0 --output NEWDIR [--graph].
The writer sees only fixed device buffers. CPU inspection occurs after explicit
synchronization, outside forward and graph capture. This does not test CP
collectives, the quantizer, model weights, or a complete serving path.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
POOLS, TOKENS, CAPACITY, MAX_QUERY = 8, 32, 256, 4


def fixture(torch, phase):
    """Original CPU BF16 patterns, including signed zero, without FP32 input."""
    indices = torch.arange(TOKENS * 576, dtype=torch.int32).reshape(TOKENS, 576)
    bits = (indices * 37 + phase * 101) % 0x7E00 + 0x80
    bits = torch.bitwise_or(bits, (indices % 2) * 0x8000)
    bits[0, 0], bits[0, 1] = 0, 0x8000
    source = bits.to(torch.int16).view(torch.bfloat16)
    rows = torch.arange(TOKENS, dtype=torch.int64)
    requests = rows // MAX_QUERY
    data = dict(source=source,
        banks=((requests * 3 + phase) % POOLS).to(torch.int32),
        positions=254 + rows % MAX_QUERY + phase * CAPACITY,
        epochs=100 + requests + phase * 1000,
        valid=(rows % 7 != phase % 7))
    # Validity alone must never authorize an invalid destination. These rows
    # deliberately remain valid=True while each addressing predicate fails.
    data["banks"][1], data["banks"][2] = -1, POOLS
    data["positions"][3], data["epochs"][4] = -1, -1
    data["valid"][1:5] = True
    return data


def expected_ring(torch, data, phase):
    # Independent CPU assignment reference; the device kernel never sees these.
    latent = torch.full((POOLS, CAPACITY, 512), -17-phase, dtype=torch.bfloat16)
    rope = torch.full((POOLS, CAPACITY, 64), -19-phase, dtype=torch.bfloat16)
    positions = torch.full((POOLS, CAPACITY), -911-phase, dtype=torch.int64)
    epochs = torch.full((POOLS, CAPACITY), -922-phase, dtype=torch.int64)
    for row in range(TOKENS):
        if not bool(data["valid"][row]):
            continue
        bank, position = int(data["banks"][row]), int(data["positions"][row])
        if not 0 <= bank < POOLS or position < 0 or int(data["epochs"][row]) < 0:
            continue
        slot = position % CAPACITY
        latent[bank, slot].copy_(data["source"][row, :512])
        rope[bank, slot].copy_(data["source"][row, 512:])
        positions[bank, slot] = position
        epochs[bank, slot] = data["epochs"][row]
    return latent, rope, positions, epochs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("Use a new isolated output directory; existing output is preserved")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="STARTED", scope="native_writer_only_not_cp_or_model",
        shape=dict(P=POOLS, T=TOKENS, C=CAPACITY, max_query=MAX_QUERY),
        graph_requested=args.graph, cp_collectives_tested=False, model_tested=False,
        native_quantizer_tested=False, performance_claim=False, latency="NOT_MEASURED",
        cases={})

    def persist():
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    persist()
    try:
        import torch
        import torch_npu
        from sfa_recent128.native_writer import PreparedNativeWriter
        from sfa_recent128.prolog_provider import RingBinding
        torch.npu.set_device(args.device)
        device = torch.device(args.device)
        with torch.inference_mode():
            first = fixture(torch, 0)
            buffers = {name: value.to(device=device) for name, value in first.items()}
            # Cache the actual views once. Each original BF16 row has a 576-element stride.
            latent_view, rope_view = buffers["source"][:, :512], buffers["source"][:, 512:]
            if latent_view.stride() != (576, 1) or rope_view.stride() != (576, 1):
                raise RuntimeError("Smoke fixture lost its actual CP-style row gaps")
            ring = RingBinding(
                torch.empty((POOLS, CAPACITY, 512), device=device, dtype=torch.bfloat16),
                torch.empty((POOLS, CAPACITY, 64), device=device, dtype=torch.bfloat16),
                torch.empty((POOLS, CAPACITY), device=device, dtype=torch.int64),
                torch.empty((POOLS, CAPACITY), device=device, dtype=torch.int64),
                buffers["banks"], buffers["positions"], buffers["epochs"], buffers["valid"], MAX_QUERY)
            destinations = (ring.recent_latent, ring.recent_rope,
                            ring.recent_position_tags, ring.recent_epoch_tags)
            addresses = {name: value.data_ptr() for name, value in buffers.items()}
            capturing = [False]
            writer = PreparedNativeWriter(latent_view, rope_view, ring,
                                          check_is_capture=lambda: capturing[0])

            def prepare_phase(phase):
                data = fixture(torch, phase)
                for name, value in data.items():
                    buffers[name].copy_(value)
                for target, poison in zip(destinations, (-17-phase, -19-phase, -911-phase, -922-phase)):
                    target.fill_(poison)
                if addresses != {name: value.data_ptr() for name, value in buffers.items()}:
                    raise RuntimeError("A fixed-address smoke buffer was replaced")
                return data

            def check(name, data, phase):
                # Deliberately outside the writer/forward/capture boundary.
                torch.npu.synchronize()
                expected = expected_ring(torch, data, phase)
                actual = tuple(value.detach().cpu() for value in destinations)
                payload_exact = all(torch.equal(value.view(torch.int16), reference.view(torch.int16))
                                    for value, reference in zip(actual[:2], expected[:2]))
                tags_exact = all(torch.equal(value, reference)
                                 for value, reference in zip(actual[2:], expected[2:]))
                source_exact = torch.equal(buffers["source"].detach().cpu().view(torch.int16),
                                           data["source"].view(torch.int16))
                routing_exact = all(torch.equal(buffers[key].detach().cpu(), data[key])
                                    for key in ("banks", "positions", "epochs", "valid"))
                result = dict(passed=payload_exact and tags_exact and source_exact and routing_exact,
                    bf16_bits_exact=payload_exact, full_ring_tags_exact=tags_exact,
                    invalid_rows_unwritten=payload_exact and tags_exact,
                    source_unchanged=source_exact, routing_unchanged=routing_exact,
                    phase=phase, valid_rows=int(data["valid"].sum()))
                report["cases"][name] = result
                persist()
                if not result["passed"]:
                    raise RuntimeError(name + " failed exact BF16/tag/unchanged-input checks")

            data = prepare_phase(0)
            writer.warmup(latent_view, rope_view)
            check("eager", data, 0)
            if args.graph:
                graph = torch.npu.NPUGraph()
                stream = torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    writer.warmup(latent_view, rope_view)
                    stream.synchronize()
                    capturing[0] = True
                    try:
                        with torch.npu.graph(graph, stream=stream):
                            writer.write(latent_view, rope_view)
                    finally:
                        capturing[0] = False
                torch.npu.current_stream().wait_stream(stream)
                for phase in (1, 2):
                    with torch.npu.stream(stream):
                        data = prepare_phase(phase)
                        graph.replay()
                    torch.npu.current_stream().wait_stream(stream)
                    check("graph_updated_poison_replay_phase" + str(phase), data, phase)
            report["writer"] = writer.configuration()
            # Preserve the implementation's deployment-wide PENDING fields;
            # this result certifies only this isolated invocation/fixture.
            report.update(status="NATIVE_WRITER_DEVICE_VALIDATED_NOT_CP_OR_MODEL",
                          fixed_buffer_addresses=True)
            persist()
        print(args.output / "results.json")
        return 0
    except Exception as error:
        report.update(status="ERROR", error=str(error), traceback=traceback.format_exc(),
                      stop_after_first_failure=True)
        persist()
        print(args.output / "results.json")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
