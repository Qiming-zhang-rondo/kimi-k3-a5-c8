"""Candidate control flow and cold-write graph poison tests, without Torch/NPU."""
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("cold_store_runner_test",ROOT/"direct_bridge/cold_store_runner.py")
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)


class Array:
    def __init__(self,value): self.value=np.asarray(value)
    @property
    def shape(self): return self.value.shape
    def view(self,*shape):
        return self if shape==("byte",) else Array(self.value.reshape(*shape))
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def clone(self): return Array(self.value.copy())
    def copy_(self,other): self.value[:]=other.value;return self
    def fill_(self,value): self.value.fill(value);return self
    def index_fill_(self,dimension,slots,value):
        assert dimension==0
        self.value[slots.value]=value
        return self


class ColdStoreRunnerTests(unittest.TestCase):
    def test_every_step_keeps_native_quantizer_and_current_packed_pointer(self):
        events=[]
        class Quantizer:
            n=0
            def _quantize(self,kv,rope):
                self.n+=1;value=object();events.append(("quantize",self.n,value))
                return value,None,None
        class Parent:
            def __init__(self,*args,**kwargs):
                self.base=SimpleNamespace(current_kv="kv",current_rope="rope",current_slots="slots",
                    torch=SimpleNamespace(uint8="byte"))
                self.variant=Quantizer();self.metadata={}
                self.cold_cache=Array(np.zeros((2,400),np.uint8))
                self._hot_prepare=SimpleNamespace(outputs={name:object() for name in candidate.PART_NAMES})
            def _cold_write(self): events.append("independent native reference")
            def _write(self): raise AssertionError("Original scatter path used")
        class Prepare:
            def __init__(self,old,cache,slots,template):
                self.outputs=old.outputs;self.metadata={"kernel_launches":1}
                events.append(("setup",template))
            def call(self,packed): events.append(("combined",packed));return self.outputs
        cls=candidate.make_cold_store_runner_class(Parent,Prepare)
        runner=cls(None,fused=True,writer="hot_store_partition")
        runner._write();runner._write()
        quantized=[item[2] for item in events if isinstance(item,tuple) and item[0]=="quantize"]
        stores=[item[1] for item in events if isinstance(item,tuple) and item[0]=="combined"]
        self.assertEqual(len(quantized),3)  # One setup template, then one per actual step.
        self.assertEqual(stores,quantized[1:]);self.assertIsNot(stores[0],stores[1])
        self.assertFalse(runner.metadata["native_c4_writer_unchanged"])
        self.assertTrue(runner.metadata["native_c4_quantizer_unchanged"])
        self.assertNotIn("independent native reference",events)
        runner._cold_write();self.assertIn("independent native reference",events)
        with self.assertRaises(ValueError): cls(None,fused=True,writer="native")

    def graph_fixture(self,replay_cold=True,corrupt_untouched=False):
        q=2
        expected={"old":np.array([[[1,2]],[[3,4]]],np.int32),
            "recent_global":np.array([[[9]],[[10]]],np.int32),
            "recent_ring":np.array([[[1]],[[2]]],np.int32),
            "old_present":np.ones(q,np.bool_),"recent_present":np.ones(q,np.bool_),
            "invalid_rows":np.zeros(q,np.bool_)}
        slots=Array(np.array([3,9],np.int64));hot_slots=Array(np.array([1,6],np.int64))
        cold=Array(np.full((16,400),33,np.uint8))
        hot=Array(np.full((8,1024),33,np.uint8));rope=Array(np.full((8,128),33,np.uint8))
        parts={name:Array(np.zeros_like(value)) for name,value in expected.items()}
        cold_source=np.arange(q*400,dtype=np.uint16).astype(np.uint8).reshape(q,400)
        hot_source=np.full((q,1024),47,np.uint8);rope_source=np.full((q,128),59,np.uint8)
        def write(include_cold=True):
            if include_cold: cold.value[slots.value]=cold_source
            hot.value[hot_slots.value]=hot_source;rope.value[hot_slots.value]=rope_source
            for name,value in expected.items(): parts[name].value[:]=value
            return Array(np.full((2,64,512),7,np.float32))  # Same output even if cold scatter is missing.
        class Graph:
            def replay(self):
                write(replay_cold)
                if corrupt_untouched: cold.value[12,0]=0
        torch=SimpleNamespace(uint8="byte",npu=SimpleNamespace(NPUGraph=Graph,
            synchronize=lambda:None,graph=lambda graph:contextlib.nullcontext(graph)))
        runner=SimpleNamespace(base=SimpleNamespace(torch=torch,current_slots=slots),
            hot_slots=hot_slots,cold_cache=cold,hot_cache=hot,hot_rope=rope,
            _hot_prepare=SimpleNamespace(outputs=parts))
        return runner,write,expected

    def test_graph_replay_checks_raw_cold_bytes_even_when_attention_output_is_unchanged(self):
        runner,write,expected=self.graph_fixture(replay_cold=False)
        with patch.object(candidate,"partition_reference",return_value=expected):
            result=candidate.writer_graph_control(runner,write)
        self.assertFalse(result["passed"])
        self.assertFalse(result["fields"]["cold"]["passed"])
        self.assertTrue(result["fields"]["hot_latent"]["passed"])
        self.assertTrue(result["fields"]["partition_old"]["passed"])

    def test_partition_gate_uses_new_writer_and_restores_caches_and_shared_outputs(self):
        fixture,write,expected=self.graph_fixture()
        fixture.base.current_kv,fixture.base.current_rope="kv","rope"
        fixture.variant=SimpleNamespace(_quantize=lambda *_:(object(),None,None))
        fixture.metadata={}
        fixture._hot_prepare.call=lambda: self.fail("Old producer must not validate the candidate")
        class Parent:
            def __init__(self,*args,**kwargs): self.__dict__.update(vars(fixture))
        class Prepare:
            def __init__(self,old,*args): self.outputs=old.outputs;self.metadata={}
            def call(self,packed): write();return self.outputs
        runner=candidate.make_cold_store_runner_class(Parent,Prepare)(None,writer="hot_store_partition")
        before={name:value.value.copy() for name,value in candidate.cache_views(runner).items()}
        parts={name:value.value.copy() for name,value in runner._hot_prepare.outputs.items()}
        with patch.object(candidate,"partition_reference",return_value=expected):
            result=runner.partition_diagnostics()
        self.assertTrue(result["passed"])
        self.assertEqual(result["producer"],"cold_scatter_bf16_hot_store_partition_after_output_poison")
        for name,value in candidate.cache_views(runner).items(): np.testing.assert_array_equal(value.value,before[name])
        for name,value in runner._hot_prepare.outputs.items(): np.testing.assert_array_equal(value.value,parts[name])

    def test_graph_replay_restores_touched_slots_and_preserves_all_history(self):
        runner,write,expected=self.graph_fixture()
        with patch.object(candidate,"partition_reference",return_value=expected):
            self.assertTrue(candidate.writer_graph_control(runner,write)["passed"])
        runner,write,expected=self.graph_fixture(corrupt_untouched=True)
        with patch.object(candidate,"partition_reference",return_value=expected):
            result=candidate.writer_graph_control(runner,write)
        self.assertFalse(result["passed"])
        self.assertEqual(result["fields"]["cold"]["first_bad_positions"],[[12,0]])


if __name__=="__main__": unittest.main()
