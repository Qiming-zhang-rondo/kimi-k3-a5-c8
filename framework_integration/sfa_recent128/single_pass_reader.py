"""Standalone one-attention reader for the explicit eager accuracy experiment.

Head padding, metadata copies and Python allocations are instrumentation, not
a serving implementation. The candidate never calls a cold reader or an
external Python softmax merge. KV-split adds a second device reduction launch;
its caller owns the independent accuracy reference.
"""
from functools import lru_cache
import importlib.util
import math
from pathlib import Path


@lru_cache(maxsize=4)
def load_bridge(source_root, library_dir):
    root=Path(source_root).expanduser().resolve()
    path=root/"experimental/single_pass/load_single_pass.py"
    spec=importlib.util.spec_from_file_location("framework_single_pass_loader",path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot locate the standalone single-pass ABI1 loader")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    library=Path(library_dir).expanduser().resolve()/"libqsfa_recent128_direct.so"
    return module.load_single_pass(library)


def _require(condition,message):
    if not condition:
        raise RuntimeError(message)


class SinglePassReader:
    def __init__(self,config,impl):
        _require(config.execution=="accuracy_eager","Single-pass framework reader is accuracy_eager only")
        _require(config.window==128,"Single-pass framework reader requires recent128")
        _require(impl.qk_rope_head_dim==64,
            "Single-pass ABI1 requires RoPE64/400-byte C4; RoPE0/272-byte cache is unsupported")
        _require(impl.kv_lora_rank==512 and impl.num_kv_heads==1,"Single-pass ABI1 requires D512/KV-head1")
        _require(bool(config.source_root) and bool(config.library_dir),"Single-pass source_root/library_dir are required")
        _require(math.isfinite(float(impl.scale)) and float(impl.scale)>0,"Invalid attention scale")
        self.impl,self.config=impl,config
        self.kernel_variant=getattr(config,"single_pass_kernel","overlay")
        _require(self.kernel_variant in ("overlay","fast_pair","load_tile4","kv_split4"),
            "single_pass_kernel must be overlay, fast_pair, load_tile4 or kv_split4")
        self.load_variant=4 if self.kernel_variant=="load_tile4" else 0
        self.actual_c_entry={"overlay":"QsfaSinglePassLaunch","fast_pair":"QsfaSinglePassLaunchFastPair",
                             "load_tile4":"QsfaSinglePassLaunchLoadTile",
                             "kv_split4":"QsfaSinglePassLaunchKvSplit"}[self.kernel_variant]
        self.device_entry=("qsfa_recent128_kv_split_attention" if self.kernel_variant=="kv_split4" else
            "QsfaSinglePassLaunchLoadTile32" if self.load_variant else self.actual_c_entry)
        self.bridge=load_bridge(str(Path(config.source_root).expanduser().resolve()),
                                str(Path(config.library_dir).expanduser().resolve()))
        actual=dict(self.bridge.configuration())
        _require(actual.get("abi_version")==1 and actual.get("buffers_bytes")==120,
            "Framework single-pass reader needs the independent fifteen-pointer ABI1 library")
        if self.kernel_variant=="fast_pair":
            _require(actual.get("fast_pair_reader_available") is True,
                "single_pass_kernel=fast_pair requires QsfaSinglePassLaunchFastPair; refusing overlay fallback")
        if self.kernel_variant=="kv_split4":
            _require(actual.get("kv_split_reader_available") is True and
                actual.get("kv_split_abi_version")==2 and actual.get("kv_split_tiling_bytes")==112 and
                callable(getattr(self.bridge,"prepare_kv_split",None)),
                "single_pass_kernel=kv_split4 requires real KV-split exports, ABI2/112B and prepare_kv_split; refusing fallback")
        if self.kernel_variant in ("load_tile4","kv_split4"):
            _require(actual.get("load_tile_reader_available") is True,
                "load_tile4 and kv_split4 prefill routing require QsfaSinglePassLaunchLoadTile; refusing fallback")
            variants=actual.get("load_tile_variants")
            geometry=variants.get(4,variants.get("4")) if isinstance(variants,dict) else None
            expected=dict(rows=32,input_stride_bytes=416,input_depth=2,output_depth=1)
            _require(isinstance(geometry,dict) and all(type(geometry.get(name)) is int and
                     geometry[name]==value for name,value in expected.items()),
                "load_tile4 requires verified rows32/stride416/input_depth2/output_depth1 configuration")
        self.bridge_identity=getattr(self.bridge,"_single_pass_build_identity",{})
        self._configuration=actual
        self.plans={}
        self.last_attention_calls=0
        self.last_reduction_calls=0
        self.last_call_metadata={}

    def configuration(self):
        return dict(self._configuration,execution="accuracy_eager",performance_claim=False,
            kernel_variant=self.kernel_variant,entry=self.actual_c_entry,actual_c_entry=self.actual_c_entry,return_stats=False,
            load_variant=self.load_variant,device_entry=self.device_entry,
            kv_split_query_limit=4 if self.kernel_variant=="kv_split4" else None,
            prefill_kernel_variant="load_tile4" if self.kernel_variant=="kv_split4" else self.kernel_variant,
            format="C4 E2M1/E8M0 + BF16 RoPE64; original BF16 hot ring")

    def read(self,query_nope,q_rope,cold,topk,table,q_lengths,kv_lengths,
             hot_latent,hot_rope,tags,positions,capacity):
        import torch
        self.last_attention_calls=0
        self.last_reduction_calls=0
        self.last_call_metadata={"status":"VALIDATING","performance_claim":False}
        q,heads,dim=query_nope.shape
        device=query_nope.device
        _require(1<=q<=1024 and heads>=1 and dim==512,"Expected Q1..1024/Hpositive/D512 query")
        _require(query_nope.dtype==torch.bfloat16 and q_rope is not None and
            tuple(q_rope.shape)==(q,heads,64) and q_rope.dtype==torch.bfloat16,"Original BF16 RoPE64 query is required")
        _require(len(cold.shape)==4 and tuple(cold.shape[1:])==(128,1,400) and cold.dtype==torch.int8,
            "Cold cache must retain all physical pages in INT8 [pages,128,1,400]")
        pages=int(cold.shape[0]); physical_keys=pages*128
        _require(pages>0 and physical_keys<=2147483647,"Invalid physical C4 pool size")
        _require(isinstance(capacity,int) and capacity>=128+q-1 and capacity%128==0,
            "Ring capacity must be page-aligned and retain W+Q-1")
        _require(tuple(hot_latent.shape)==(capacity//128,128,1,512) and
            hot_latent.dtype==torch.bfloat16,"Hot latent ring shape/dtype mismatch")
        _require(tuple(hot_rope.shape)==(capacity//128,128,1,64) and
            hot_rope.dtype==torch.bfloat16,"Hot RoPE ring shape/dtype mismatch")
        _require(tuple(tags.shape)==(capacity,) and tags.dtype==torch.int64 and
            tuple(positions.shape)==(q,) and positions.dtype==torch.int64,"Tags and query positions must retain INT64")
        _require(tuple(topk.shape)==(q,1,2048),"ABI1 requires the original unsplit TopK2048")
        _require(topk.dtype==torch.int32,"Original TopK must be INT32; no lossy index conversion")
        _require(tuple(q_lengths.shape)==(1,) and tuple(kv_lengths.shape)==(1,),"Only one active request is supported")
        _require(q_lengths.dtype in (torch.int32,torch.int64) and kv_lengths.dtype in (torch.int32,torch.int64),
            "Logical sequence lengths must be integer tensors")
        _require(tuple(table.shape[:1])==(1,) and len(table.shape)==2,"Block table must describe one request")
        _require(table.dtype in (torch.int32,torch.int64),"Block table must retain integer physical page IDs")
        _require(all(value.is_contiguous() for value in (cold,hot_latent,hot_rope)),
            "Physical cold/hot pools must already be contiguous; refuse an implicit full-pool copy")
        tensors=(q_rope,cold,topk,table,q_lengths,kv_lengths,hot_latent,hot_rope,tags,positions)
        _require(all(value.device==device for value in tensors),"Single-pass inputs must share the query device")
        # Accuracy instrumentation: validate real logical sequence lengths and
        # physical mapping before padding the ABI's physical-pool-width table.
        q_values=q_lengths.detach().cpu().tolist()
        kv_values=kv_lengths.detach().cpu().tolist()
        _require(q_values==[q] and len(kv_values)==1,"Logical query length differs from this chunk")
        logical_keys=int(kv_values[0])
        _require(q<=logical_keys<=physical_keys,"Logical KV length exceeds physical C4 pool")
        logical_pages=(logical_keys+127)//128
        _require(int(table.shape[1])>=logical_pages,"Block table lacks the actual logical prefix")
        prefix=table[:,:logical_pages]
        mapping=prefix.detach().cpu().tolist()[0]
        _require(all(isinstance(page,int) and 0<=page<pages for page in mapping),"Logical block table points outside the physical pool")
        padded_table=torch.zeros((1,pages),device=device,dtype=torch.int32)
        padded_table[:,:logical_pages].copy_(prefix.to(dtype=torch.int32))
        ql=q_lengths.to(dtype=torch.int32).contiguous()
        kl=kv_lengths.to(dtype=torch.int32).contiguous()
        executed=self.kernel_variant
        dispatch_reason="requested_kernel"
        if self.kernel_variant=="kv_split4":
            executed="kv_split4" if q<=4 else "load_tile4"
            dispatch_reason="q_le_4_kv_split4" if q<=4 else "q_gt_4_prefill_load_tile4"
        split_count=4 if executed=="kv_split4" else 1
        load_variant=4 if executed=="load_tile4" else 0
        entry={"overlay":"QsfaSinglePassLaunch","fast_pair":"QsfaSinglePassLaunchFastPair",
               "load_tile4":"QsfaSinglePassLaunchLoadTile","kv_split4":"QsfaSinglePassLaunchKvSplit"}[executed]
        device_entry=("qsfa_recent128_kv_split_attention" if executed=="kv_split4" else
                      "QsfaSinglePassLaunchLoadTile32" if load_variant else entry)
        aic=q*split_count if executed=="kv_split4" else max(
            count for count in (1,2,4,8,16) if count<=q and q%count==0)
        scale=float(self.impl.scale)
        key=(executed,q,physical_keys,capacity,scale,str(device))
        if key not in self.plans:
            template=query_nope.new_empty((q,64,576))
            if executed=="kv_split4":
                _require(physical_keys>=2048,"KV-split requires an original physical C4 pool of at least 2048 tokens")
                self.plans[key]=self.bridge.prepare_kv_split(template,
                    physical_keys,capacity,split_count,scale,False)
            else:
                self.plans[key]=self.bridge.prepare(template,physical_keys,capacity,aic,scale,False)
        plan=self.plans[key]
        common=(cold,topk.contiguous(),padded_table,ql,kl,
                hot_latent,hot_rope,tags.contiguous(),positions.contiguous())
        result=torch.empty_like(query_nope)
        padding=0
        for start in range(0,heads,64):
            width=min(64,heads-start)
            merged=torch.zeros((q,64,576),device=device,dtype=torch.bfloat16)
            merged[:,:width,:512].copy_(query_nope[:,start:start+width])
            merged[:,:width,512:].copy_(q_rope[:,start:start+width])
            if executed=="kv_split4":
                state=plan.run(merged,*common,128)
                self.last_reduction_calls+=1
            elif executed=="load_tile4":
                state=plan.run(merged,*common,128,False,False,False,False,4)
            elif executed=="fast_pair":
                state=plan.run(merged,*common,128,False,False,True)
            else:
                state=plan.run(merged,*common,128,False)
            self.last_attention_calls+=1
            _require(isinstance(state,(tuple,list)) and len(state)==3,"Single-pass bridge returned an invalid state")
            output=state[0]
            _require(tuple(output.shape)==(q,64,512) and output.dtype==torch.bfloat16,
                "Single-pass output must be BF16 [Q,64,512]")
            result[:,start:start+width].copy_(output[:,:width])
            padding+=q*(64-width)
        self.last_call_metadata=dict(status="EXERCISED",execution="accuracy_eager",queries=q,heads=heads,
            physical_keys=physical_keys,logical_keys=logical_keys,ring_capacity=capacity,aic_count=aic,
            head_groups=self.last_attention_calls,candidate_attention_calls=self.last_attention_calls,
            head_padding_rows=padding,table_logical_columns_copied=logical_pages,
            table_physical_columns=pages,topk_partitioned=executed=="kv_split4",external_merge_calls=0,
            original_topk_unchanged=True,split_count=split_count,
            device_reduction_calls=self.last_reduction_calls,
            device_kernel_launches=self.last_attention_calls+self.last_reduction_calls,
            validation_attention_calls=0,performance_claim=False,
            kernel_variant=self.kernel_variant,executed_kernel_variant=executed,dispatch_reason=dispatch_reason,
            entry=entry,actual_c_entry=entry,return_stats=False,
            load_variant=load_variant,device_entry=device_entry,
            logical_lengths_preserved=True,cold_cache_truncated=False)
        return result
