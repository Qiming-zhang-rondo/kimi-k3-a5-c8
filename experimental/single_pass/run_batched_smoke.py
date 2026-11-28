#!/usr/bin/env python3
"""Isolated A5 correctness gate for batched serving; no private Prolog needed.

Uses exact, synthetically encoded E2M1/E8M0 packed400 inputs, an independent
FP64 mathematical mixed-KV reference, and mutation of fixed-address graph
inputs. It does not test the native quantizer or claim serving/TPOT readiness.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/"framework_integration"))
sys.path.insert(0, str(HERE))


def make_data(batch, queries, phase, seed=20261128, keys=4096, compute_reference=True):
    import numpy as np
    from host_contract import bf16_round, decode_c4_rows
    rng = np.random.default_rng(seed + 313*batch + queries)
    pages = keys//128
    packed = rng.integers(0,256,(batch,keys,400),dtype=np.uint8)
    packed[:,:,384:] = 124
    rope = bf16_round(rng.normal(0,.08,(batch,keys,64)).astype(np.float32))
    packed[:,:,256:384] = (rope.view(np.uint32)>>16).astype('<u2').view(np.uint8).reshape(batch,keys,128)
    decoded, decoded_rope = decode_c4_rows(packed.reshape(-1,400))
    decoded = decoded.reshape(batch,keys,512)
    decoded_rope = decoded_rope.reshape(batch,keys,64)
    cold = np.empty((batch*pages,128,1,400),np.uint8)
    tables = []
    for req in range(batch):
        mapping = np.roll(rng.permutation(pages),phase).astype(np.int32) + req*pages
        tables.append(mapping)
        for logical,physical in enumerate(mapping):cold[physical,:,0] = packed[req,logical*128:(logical+1)*128]
    tables = np.stack(tables)
    t, capacity = batch*queries, 256
    banks = np.roll(np.arange(batch,dtype=np.int32)[::-1],phase)
    epochs = np.arange(batch,dtype=np.int64) + 10 + phase*100
    counts = np.full(batch,queries,np.int32)
    if queries>1:counts[-1] -= 1+phase
    elif phase:counts[-1] = 0
    cumulative = np.cumsum(counts,dtype=np.int32)
    valid = np.arange(t) < int(cumulative[-1])
    position = np.full(t,-1,np.int64)
    query = np.zeros((t,64,576),np.float32)
    ids = np.full((t,1,2048),-1,np.int32)
    hot = np.full((batch,capacity,512),-7.,np.float32)
    hot_rope = np.full((batch,capacity,64),-11.,np.float32)
    tags = np.full((batch,capacity),-1,np.int64)
    epoch_tags = np.full((batch,capacity),-1,np.int64)
    reference = np.zeros((t,64,512),np.float64)
    start = 0
    scale = 1/(576**.5)
    for req,count in enumerate(counts.tolist()):
        final = keys - 1 - phase*7
        first_q = final-count+1
        first_hot = max(0,first_q-127)
        tokens = np.arange(first_hot,final+1,dtype=np.int64)
        values = bf16_round(rng.normal(.03*(req+1),.4,(len(tokens),512)).astype(np.float32))
        ropes = bf16_round(rng.normal(0,.1,(len(tokens),64)).astype(np.float32))
        bank = int(banks[req])
        hot[bank,tokens%capacity] = values
        hot_rope[bank,tokens%capacity] = ropes
        tags[bank,tokens%capacity] = tokens
        epoch_tags[bank,tokens%capacity] = epochs[req]
        for local in range(count):
            row = start+local
            pos = first_q+local
            position[row] = pos
            query[row] = bf16_round(rng.normal(0,.3,(64,576)).astype(np.float32))
            cold_ids = rng.choice(pos-127,1920,replace=False)
            hot_ids = np.arange(pos-127,pos+1)
            selected = np.concatenate((cold_ids,hot_ids)).astype(np.int32)
            rng.shuffle(selected)
            ids[row,0] = selected
            if not compute_reference:
                continue
            latent = decoded[req,selected].copy()
            k_rope = decoded_rope[req,selected].copy()
            is_hot = selected>=pos-127
            slot = selected[is_hot]%capacity
            latent[is_hot] = hot[bank,slot]
            k_rope[is_hot] = hot_rope[bank,slot]
            combined = np.concatenate((latent,k_rope),axis=1).astype(np.float64)
            logits = query[row].astype(np.float64)@combined.T*scale
            logits -= logits.max(axis=1,keepdims=True)
            weights = np.exp(logits)
            weights /= weights.sum(axis=1,keepdims=True)
            reference[row] = weights@latent.astype(np.float64)
        start += count
    arrays = dict(query=query,cold=cold,indices=ids,table=tables,qcum=cumulative,
        kvlen=np.full(batch,keys-phase*7,np.int32),hot=hot,hot_rope=hot_rope,tags=tags,
        epoch_tags=epoch_tags,positions=position,banks=banks,epochs=epochs,valid=valid)
    return arrays, reference, scale


def metric(actual, expected):
    import numpy as np
    actual = np.asarray(actual,np.float64)
    expected = np.asarray(expected,np.float64)
    finite = bool(np.isfinite(actual).all())
    if actual.shape != expected.shape or not finite or not np.isfinite(expected).all():
        return dict(finite=finite, passed=False, reason="shape_or_nonfinite",
                    relative_l2=None, worst_row_relative_l2=None, max_abs=None)
    delta = actual-expected
    relative = float(np.linalg.norm(delta)/max(float(np.linalg.norm(expected)),1e-12))
    rows = delta.reshape(-1,512)
    norms = np.linalg.norm(expected.reshape(-1,512),axis=1)
    errors = np.linalg.norm(rows,axis=1)
    worst = float((errors/np.maximum(norms,1e-12)).max(initial=0))
    return dict(finite=finite, relative_l2=relative, worst_row_relative_l2=worst,
                max_abs=float(np.abs(delta).max(initial=0)), passed=finite and relative<=.01 and worst<=.02)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-dir",type=Path,required=True)
    parser.add_argument("--source-root",type=Path,default=ROOT)
    parser.add_argument("--device",default="npu:0")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--graph",action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():parser.error("Use a fresh isolated output directory")
    args.output.mkdir(parents=True)
    report = dict(status="STARTED",scope="isolated_batched_reader_not_prolog_or_service",
        native_quantizer_tested=False,prolog_validated=False,latency="NOT_MEASURED",
        graph_requested=args.graph,cases={})
    def persist():
        (args.output/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    persist()
    try:
        import numpy as np
        import torch
        import torch_npu
        from sfa_recent128.serving_reader import ServingReader
        from sfa_recent128.prolog_provider import RingBinding
        torch.npu.set_device(args.device)
        config = SimpleNamespace(window=128,execution="serving",single_pass_kernel="batched_v4",
            source_root=str(args.source_root),library_dir=str(args.library_dir))
        for batch,queries in ((1,1),(1,4),(8,4)):
            data,reference,scale = make_data(batch,queries,0)
            impl = SimpleNamespace(kv_lora_rank=512,qk_rope_head_dim=64,num_kv_heads=1,
                                   local_num_heads=64,scale=scale)
            reader = ServingReader(config,impl)
            device = torch.device(args.device)
            dtypes = dict(query=torch.bfloat16,cold=torch.int8,indices=torch.int32,
                table=torch.int32,qcum=torch.int32,kvlen=torch.int32,hot=torch.bfloat16,
                hot_rope=torch.bfloat16,tags=torch.int64,epoch_tags=torch.int64,
                positions=torch.int64,banks=torch.int32,epochs=torch.int64,valid=torch.bool)
            buffers = {k:torch.as_tensor(v.view(np.int8) if k=="cold" else v,
                       device=device,dtype=dtypes[k]).contiguous() for k,v in data.items()}
            t = batch*queries
            token_banks = torch.zeros(t,device=device,dtype=torch.int32)
            token_epochs = torch.zeros(t,device=device,dtype=torch.int64)
            ring = RingBinding(buffers["hot"],buffers["hot_rope"],buffers["tags"],buffers["epoch_tags"],
                token_banks,buffers["positions"],token_epochs,buffers["valid"],queries)
            reader.prepare(device=device,cold_pages=data["cold"].shape[0],
                token_capacity=t,request_capacity=batch,ring_binding=ring,table_columns=data["table"].shape[1])
            def call():
                return reader.read(buffers["query"][:,:,:512],buffers["query"][:,:,512:],buffers["cold"],
                    buffers["indices"],buffers["table"],buffers["qcum"],buffers["kvlen"],ring,
                    buffers["banks"],buffers["epochs"])
            def host(value):return value.detach().float().cpu().numpy()
            def check(name,expected):
                torch.npu.synchronize()
                result = metric(host(output),expected)
                result["faults_zero"] = all(not bool(v.cpu().any()) for v in reader.fault_buffers())
                result["passed"] &= result["faults_zero"]
                report["cases"][name] = result
                persist()
                if not result["passed"]:raise RuntimeError(name+" failed independent mathematical control")
            with torch.inference_mode():
                output = call()
                check(f"b{batch}_q{queries}_eager",reference)
                report["bridge"] = reader.configuration()
                if args.graph:
                    graph = torch.npu.NPUGraph()
                    # Passing this stream explicitly is essential: graph() with
                    # stream=None selects its own internal side stream instead.
                    stream = torch.npu.Stream()
                    stream.wait_stream(torch.npu.current_stream())
                    with torch.npu.stream(stream):
                        # Prepare a separate fixed-address plan for this stream lane.
                        reader.prepare(device=device,cold_pages=data["cold"].shape[0],token_capacity=t,
                            request_capacity=batch,ring_binding=ring,table_columns=data["table"].shape[1])
                        call()
                        with torch.npu.graph(graph,stream=stream):output = call()
                    torch.npu.current_stream().wait_stream(stream)
                    for phase in (0,1):
                        updated,expected,_ = make_data(batch,queries,phase)
                        for key,value in updated.items():
                            array = value.view(np.int8) if key=="cold" else value
                            buffers[key].copy_(torch.as_tensor(array,device=device,dtype=dtypes[key]))
                        output.fill_(float("nan"))
                        graph.replay()
                        check(f"b{batch}_q{queries}_graph_phase{phase}",expected)
            # Independent negative control: stale epoch must latch a device fault
            # which stays set after the tags are repaired and another read runs.
            if queries>1:
                slot = int(data["positions"][0])%256
                bank = int(data["banks"][0])
                # Restore phase0 so the deliberately corrupted slot is selected.
                for key,value in data.items():
                    buffers[key].copy_(torch.as_tensor(value.view(np.int8) if key=="cold" else value,
                                                     device=device,dtype=dtypes[key]))
                buffers["epoch_tags"][bank,slot] = -1
                call();torch.npu.synchronize()
                latched = any(bool(v.cpu().any()) for v in reader.fault_buffers())
                buffers["epoch_tags"][bank,slot] = int(data["epochs"][0])
                call();torch.npu.synchronize()
                retained = any(bool(v.cpu().any()) for v in reader.fault_buffers())
                report["cases"][f"b{batch}_q{queries}_stale_epoch_latch"] = dict(
                    passed=latched and retained,detected=latched,retained_after_repair=retained)
                persist()
                if not(latched and retained):raise RuntimeError("Missing-hot epoch did not latch a fault")
                reader.reset_faults()
        report["status"] = "DEVICE_BATCHED_READER_VALIDATED"
        persist()
        print(args.output/"results.json")
        return 0
    except Exception as error:
        report.update(status="ERROR",error=str(error),traceback=traceback.format_exc(),stop_after_first_failure=True)
        persist()
        print(args.output/"results.json")
        return 1


if __name__=="__main__":raise SystemExit(main())
