"""Production runtime dispatch with fake device metadata, no Torch/NPU imports."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sfa_recent128.contract import Recent128Config


class Tensor:
    def __init__(self,shape,dtype="bfloat16",device="npu:0",source=None):
        self.shape=tuple(shape);self.dtype="torch."+dtype;self.device=device;self.source=source
    def element_size(self):return 1 if self.dtype in ("torch.int8","torch.float8_e4m3fn") else 2
    def view(self,*shape):
        if len(shape)==1 and isinstance(shape[0],str):return Tensor(self.shape,shape[0].removeprefix("torch."),self.device,self)
        shape=list(shape);total=1
        for value in self.shape:total*=value
        if -1 in shape:
            divisor=1
            for value in shape:
                if value!=-1:divisor*=value
            shape[shape.index(-1)]=total//divisor
        return Tensor(shape,self.dtype.removeprefix("torch."),self.device,self)
    def is_contiguous(self):return True
    def cpu(self):raise AssertionError("Forward must not read device values")
    def item(self):raise AssertionError("Forward must not extract device scalars")
    def tolist(self):raise AssertionError("Forward must not synchronize device tensors")


class ServingRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.events=[];self.allocations=[];self.capture=SimpleNamespace(capturing=False)
        self.bindings=[];self.providers=[]
        events,allocations=self.events,self.allocations
        testcase=self
        class State:
            def __init__(state,**kwargs):
                events.append(("state_created",kwargs))
                state.request_pool_size=kwargs["request_pool_size"];state.max_tokens=kwargs["max_tokens"]
                state.max_query_per_request=kwargs["max_query_per_request"]
                state.capacity=((128+state.max_query_per_request-1+127)//128)*128
                state.layers=[]
                state.step=SimpleNamespace(request_banks=Tensor((8,),"int32"),request_epochs=Tensor((8,),"int64"))
            def register_layer(state,key):state.layers.append(key)
            def binding(state,key,*,token_rows):
                events.append(("binding",key,token_rows))
                if token_rows>state.max_tokens:raise RuntimeError("token capacity exceeded")
                value=SimpleNamespace(layer=key,rows=token_rows)
                testcase.bindings.append(value);return value
        class Reader:
            def __init__(reader,config,impl,*,execution_lane=None):
                reader.prepared=False;reader.preparations=[];reader.reads=[];reader.result=object()
                reader.execution_lane=execution_lane
                events.append("reader_created")
            def is_prepared(reader,query,cold,table):
                # Enforce the actual reader's contract: this is one packed tensor,
                # not the framework's composed main/indexer cache tuple.
                _=cold.shape
                return reader.prepared
            def prepare(reader,**kwargs):
                if testcase.capture.capturing:raise AssertionError("Plan allocation inside Graph")
                reader.prepared=True;reader.preparations.append(kwargs)
            def read(reader,*args):reader.reads.append(args);return reader.result
        self.State=State;self.Reader=Reader
        fake_torch=ModuleType("torch");fake_torch.bool="torch.bool";fake_torch.int8="torch.int8"
        def empty(shape,**kwargs):
            allocations.append((shape,kwargs))
            return Tensor(shape,"bool",kwargs["device"])
        fake_torch.empty=empty
        fake_reader=ModuleType("sfa_recent128.serving_reader");fake_reader.ServingReader=Reader
        fake_reader.ServingExecutionLane=type("ServingExecutionLane",(),{})
        fake_context=ModuleType("vllm_ascend.ascend_forward_context");fake_context._EXTRA_CTX=self.capture
        fake_top=ModuleType("vllm_ascend");fake_top.__path__=[]
        fake_old=ModuleType("sfa_recent128.runtime")
        class ForbiddenAccuracyRuntime:
            def __init__(self,*args):raise AssertionError("Old accuracy runtime imported/constructed")
        fake_old.Recent128Runtime=ForbiddenAccuracyRuntime
        module_patch=patch.dict(sys.modules,{"torch":fake_torch,"sfa_recent128.serving_reader":fake_reader,
            "vllm_ascend":fake_top,"vllm_ascend.ascend_forward_context":fake_context,"sfa_recent128.runtime":fake_old})
        module_patch.start();self.addCleanup(module_patch.stop)
        spec=importlib.util.spec_from_file_location("sfa_recent128.serving_runtime",ROOT/"sfa_recent128/serving_runtime.py")
        self.runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.runtime)
        self.runtime.ServingState=State
        self.runtime.install_prolog_route=lambda:self.providers.append("checked") or {"provider_available":True}
        self.route_count=1
        @contextmanager
        def route(impl,binding):
            events.append(("route_enter",impl,binding))
            yield SimpleNamespace(provider_calls=testcase.route_count)
            events.append("route_exit")
        self.runtime.prolog_scope=route
        self.config=Recent128Config(window=128,execution="serving",reader="single_pass",single_pass_kernel="batched_v4",
            max_query_chunk=32,source_root="/operator",library_dir="/isolated/library",diagnostics_path=self.directory.name)
        self.vc=SimpleNamespace(model_config=SimpleNamespace(model="actual-model",enforce_eager=False),
            scheduler_config=SimpleNamespace(max_num_seqs=8,max_num_batched_tokens=32),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[4,8,32]))

    def impl(self,layer="layer.0"):
        return SimpleNamespace(vllm_config=self.vc,preprocess_type="PROLOG_V3",enable_mxfp4_kv=True,
            num_kv_heads=1,kv_lora_rank=512,qk_rope_head_dim=64,local_num_heads=64,scale=0.125,
            kv_a_layernorm=SimpleNamespace(weight=Tensor((512,))),layer_name=layer)

    def initialized(self):return self.runtime.ServingRuntime(self.impl(),self.config)

    def test_bad_selection_gathers_actual_reasons_and_geometry_before_any_allocation(self):
        from enum import Enum
        class Preprocess(Enum):
            NATIVE=0
            PROLOG_V3=1
        impl=self.impl()
        impl.preprocess_type=Preprocess.NATIVE
        impl.enable_mxfp4_kv=False
        impl.qk_rope_head_dim=0
        impl.local_num_heads=0
        impl.scale=float("nan")
        impl.dcp_group=object()
        probes=[]
        def reasons(target):
            probes.append(target)
            return ["quantized indexer query projection is unavailable"]
        impl._get_fused_type_unsupported_reasons=reasons
        with self.assertRaisesRegex(RuntimeError,"selected=NATIVE.*quantized indexer"):
            self.runtime.ServingRuntime(impl,self.config)
        self.assertIs(impl.preprocess_type,Preprocess.NATIVE)
        self.assertEqual(probes,[Preprocess.PROLOG_V3])
        self.assertFalse(self.allocations)
        self.assertFalse(self.providers)
        files=list(Path(self.directory.name).glob("serving-contract-*.json"))
        self.assertEqual(len(files),1)
        report=json.loads(files[0].read_text())
        self.assertEqual(report["selected_preprocess"],"NATIVE")
        self.assertGreaterEqual(len(report["errors"]),5)
        self.assertTrue(report["dcp_group_present"])
        self.assertEqual(report["tbe_variant_compilation"],"NOT_TESTED_BY_ROUTE_CHECK")

    def test_reason_probe_failure_and_missing_layer_do_not_hide_route_error(self):
        from enum import Enum
        class Preprocess(Enum):
            NATIVE=0
            PROLOG_V3=1
        impl=self.impl(None)
        impl.preprocess_type=Preprocess.NATIVE
        def reasons(target):raise RuntimeError("upstream probe unavailable")
        impl._get_fused_type_unsupported_reasons=reasons
        report=self.runtime.layer_contract_report(impl,self.config)
        self.assertEqual(report["selected_preprocess"],"NATIVE")
        self.assertIn("upstream probe unavailable",report["upstream_reason_probe_error"])
        self.assertTrue(any("layer_name" in error for error in report["errors"]))
        self.assertTrue(any("not a TBE path" in error for error in report["errors"]))

    def inputs(self, runtime, *, tuple_cache=True):
        metadata=SimpleNamespace(recent128_state=runtime.state,max_query_len=4,block_size=128,
            block_table=Tensor((8,16),"int32"))
        cold=Tensor((16,128,1,400),"int8")
        cache=(cold,object()) if tuple_cache else cold
        values=(Tensor((4,64,512)),Tensor((4,64,64)),cache,Tensor((4,1,2048),"int32"),metadata,
            Tensor((8,),"int32"),Tensor((8,),"int32"))
        return values,cold

    def test_provider_checked_before_state_allocation_capacity_and_shared_layers(self):
        first=self.initialized()
        second=self.runtime.ServingRuntime(self.impl("layer.1"),self.config)
        self.assertIs(first.state,second.state)
        self.assertEqual(first.state.capacity,256)
        self.assertEqual(first.state.layers,["layer.0","layer.1"])
        self.assertEqual(len(self.allocations),1)
        self.assertEqual(len(first.state.runtime_readers),2)
        self.assertFalse(first.producer_calls)
        report=Path(first.report).read_text()
        self.assertIn('"old_accuracy_runtime_used": false',report)
        self.runtime._STATES.clear();self.allocations.clear()
        def unavailable():raise RuntimeError("private producer missing")
        self.runtime.install_prolog_route=unavailable
        with self.assertRaisesRegex(RuntimeError,"private producer missing"):
            self.initialized()
        self.assertFalse(self.allocations)
        self.assertEqual(sum(isinstance(x,tuple) and x[0]=="state_created" for x in self.events),1)

    def test_prolog_scope_requires_bound_state_and_exactly_one_actual_provider(self):
        runtime=self.initialized()
        metadata=SimpleNamespace(recent128_state=runtime.state,max_query_len=32)
        with runtime.prolog_scope(metadata,32):pass
        self.assertEqual(runtime.producer_calls,1)
        for count in (0,2):
            self.route_count=count
            with self.assertRaisesRegex(RuntimeError,"exactly one"):
                with runtime.prolog_scope(metadata,32):pass
        self.assertEqual(runtime.producer_calls,1)
        with self.assertRaisesRegex(RuntimeError,"capacity bound"):
            with runtime.prolog_scope(SimpleNamespace(recent128_state=runtime.state,max_query_len=33),32):pass
        with self.assertRaisesRegex(RuntimeError,"corrected device metadata"):
            with runtime.prolog_scope(SimpleNamespace(recent128_state=object()),4):pass

    def test_tuple_cache_normalization_warmup_then_graph_reuses_plan_without_allocation(self):
        runtime=self.initialized();values,cold=self.inputs(runtime)
        self.assertIs(runtime.read(*values),runtime.reader.result)
        self.assertEqual(len(runtime.reader.preparations),1)
        prepared=runtime.reader.preparations[0]
        self.assertEqual((prepared["cold_pages"],prepared["token_capacity"],prepared["request_capacity"]),(16,4,8))
        forwarded=runtime.reader.reads[0]
        self.assertEqual(tuple(forwarded[2].shape),(16,128,1,400))
        self.assertEqual(forwarded[2].dtype,"torch.int8")
        self.assertIs(forwarded[4],values[4].block_table)
        self.assertIs(forwarded[8],runtime.state.step.request_banks)
        before=len(self.allocations)
        self.capture.capturing=True
        self.assertIs(runtime.read(*values),runtime.reader.result)
        self.assertEqual(len(runtime.reader.preparations),1)
        self.assertEqual(len(self.allocations),before)
        self.assertEqual(runtime.read_calls,2)

    def test_unprepared_capture_fails_before_plan_creation_or_reader_dispatch(self):
        runtime=self.initialized();values,_=self.inputs(runtime,tuple_cache=False)
        self.capture.capturing=True
        with self.assertRaisesRegex(RuntimeError,"before Graph capture"):
            runtime.read(*values)
        self.assertFalse(runtime.reader.preparations)
        self.assertFalse(runtime.reader.reads)
        self.assertEqual(runtime.read_calls,0)

    def test_distinct_routing_lanes_have_distinct_reservations_and_reuse_per_lane(self):
        runtime=self.initialized();values,_=self.inputs(runtime)
        runtime.read(*values)
        main_reader=runtime.reader
        self.assertIsNotNone(main_reader.execution_lane)
        lane=SimpleNamespace(pool_owner=runtime.state,step=runtime.state.step,binding=runtime.state.binding)
        alternate=list(values)
        alternate[4]=SimpleNamespace(recent128_state=lane,max_query_len=4,block_size=128,
            block_table=values[4].block_table)
        runtime.read(*alternate)
        draft_reader=runtime.reader
        self.assertIsNot(main_reader,draft_reader)
        self.assertIsNot(main_reader.execution_lane,draft_reader.execution_lane)
        self.capture.capturing=True
        runtime.read(*values)
        self.assertIs(runtime.reader,main_reader)
        self.assertEqual(len(main_reader.preparations),1)
        self.assertEqual(len(draft_reader.preparations),1)


if __name__=="__main__":unittest.main()
