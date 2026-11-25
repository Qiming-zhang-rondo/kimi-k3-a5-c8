"""Single-pass framework shape wiring with NumPy fakes, never real Torch/NPU."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("framework_single_pass_reader",ROOT/"sfa_recent128/single_pass_reader.py")
reader_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader_module)


class Tensor:
    def __init__(self,data,dtype="bf16",device="npu:0"):
        self.data=np.asarray(data);self.dtype=dtype;self.device=device
    @property
    def shape(self):return self.data.shape
    def __getitem__(self,index):return Tensor(self.data[index],self.dtype,self.device)
    def copy_(self,value):self.data[...]=value.data;return self
    def contiguous(self):return self
    def is_contiguous(self):return True
    def to(self,dtype=None):return Tensor(self.data,dtype or self.dtype,self.device)
    def detach(self):return self
    def cpu(self):return self
    def tolist(self):return self.data.tolist()
    def new_empty(self,shape):return Tensor(np.zeros(shape),self.dtype,self.device)


class Bridge:
    def __init__(self):self.preparations=[];self.calls=[];self._single_pass_build_identity={"library_sha256":"fake"}
    def configuration(self):return {"abi_version":1,"buffers_bytes":120}
    def prepare(self,query,*args):
        self.preparations.append((query,args))
        bridge=self
        class Plan:
            def run(self,*inputs):
                bridge.calls.append(inputs)
                output=Tensor(np.full((query.shape[0],64,512),len(bridge.calls)),"bf16")
                return output,None,None
        return Plan()


class FrameworkSinglePassReaderTests(unittest.TestCase):
    def setUp(self):
        self.bridge=Bridge()
        fake=types.ModuleType("torch")
        fake.bfloat16="bf16";fake.int8="int8";fake.int32="int32";fake.int64="int64"
        fake.zeros=lambda shape,device,dtype:Tensor(np.zeros(shape),dtype,device)
        fake.empty_like=lambda value:Tensor(np.zeros(value.shape),value.dtype,value.device)
        self.fake=patch.dict(sys.modules,{"torch":fake});self.fake.start();self.addCleanup(self.fake.stop)
        self.loader=patch.object(reader_module,"load_bridge",return_value=self.bridge)
        self.loader_mock=self.loader.start();self.addCleanup(self.loader.stop)
        self.config=types.SimpleNamespace(execution="accuracy_eager",window=128,source_root="/fake/source",library_dir="/fake/lib")
        self.impl=types.SimpleNamespace(qk_rope_head_dim=64,kv_lora_rank=512,num_kv_heads=1,scale=.04)

    def inputs(self,heads=8,q=4):
        capacity=256
        return [Tensor(np.zeros((q,heads,512))),Tensor(np.zeros((q,heads,64))),
            Tensor(np.zeros((32,128,1,400),np.int8),"int8"),Tensor(np.zeros((q,1,2048),np.int32),"int32"),
            Tensor(np.array([[7,0,3,5,6,4,1,2]],np.int32),"int32"),
            Tensor(np.array([q],np.int32),"int32"),Tensor(np.array([1024],np.int32),"int32"),
            Tensor(np.zeros((2,128,1,512))),Tensor(np.zeros((2,128,1,64))),
            Tensor(np.full(capacity,-1,np.int64),"int64"),Tensor(np.arange(q,dtype=np.int64),"int64"),capacity]

    def test_short_query_chunks_choose_only_supported_aic_divisors(self):
        reader=reader_module.SinglePassReader(self.config,self.impl)
        reader.read(*self.inputs(q=3))
        self.assertEqual(reader.last_call_metadata["aic_count"],1)

    def test_padding_preserves_original_topk_logical_lengths_and_all_physical_cache(self):
        reader=reader_module.SinglePassReader(self.config,self.impl)
        values=self.inputs();out=reader.read(*values)
        self.assertEqual(out.shape,(4,8,512));self.assertEqual(reader.last_attention_calls,1)
        call=self.bridge.calls[0]
        self.assertEqual(call[0].shape,(4,64,576))
        self.assertIs(call[1],values[2]);self.assertIs(call[2],values[3])
        self.assertEqual(call[3].shape,(1,32))
        np.testing.assert_array_equal(call[3].data[:,:8],values[4].data)
        np.testing.assert_array_equal(call[3].data[:,8:],0)
        self.assertEqual(call[4].tolist(),[4]);self.assertEqual(call[5].tolist(),[1024])
        self.assertIs(call[8],values[9]);self.assertIs(call[9],values[10])
        self.assertEqual(call[-2:],(128,False))
        self.assertEqual(self.bridge.preparations[0][1][:3],(4096,256,4))
        self.assertFalse(reader.last_call_metadata["topk_partitioned"])
        self.assertEqual(reader.last_call_metadata["validation_attention_calls"],0)

    def test_more_than_64_heads_is_explicit_multiple_candidate_launches_and_prepared_reuse(self):
        reader=reader_module.SinglePassReader(self.config,self.impl)
        values=self.inputs(65)
        out=reader.read(*values)
        self.assertEqual(out.shape,(4,65,512));self.assertEqual(reader.last_attention_calls,2)
        np.testing.assert_array_equal(out.data[:,:64],1)
        np.testing.assert_array_equal(out.data[:,64:],2)
        self.assertEqual(reader.last_call_metadata["head_padding_rows"],4*63)
        reader.read(*values)
        self.assertEqual(len(self.bridge.preparations),1)
        self.assertEqual(reader.last_attention_calls,2)

    def test_fast_pair_is_explicit_actual_export_with_stats_disabled(self):
        self.config.single_pass_kernel="fast_pair"
        self.bridge.configuration=lambda:{"abi_version":1,"buffers_bytes":120,"fast_pair_reader_available":True}
        reader=reader_module.SinglePassReader(self.config,self.impl)
        reader.read(*self.inputs())
        self.assertEqual(self.bridge.calls[0][-4:],(128,False,False,True))
        self.assertFalse(self.bridge.preparations[0][1][-1])
        self.assertEqual(reader.last_call_metadata["kernel_variant"],"fast_pair")
        self.assertEqual(reader.last_call_metadata["entry"],"QsfaSinglePassLaunchFastPair")
        self.assertFalse(reader.last_call_metadata["return_stats"])
        self.assertEqual(reader.last_call_metadata["external_merge_calls"],0)
        self.assertEqual(reader.configuration()["kernel_variant"],"fast_pair")

    def test_missing_fast_pair_export_fails_before_any_overlay_launch(self):
        self.config.single_pass_kernel="fast_pair"
        with self.assertRaisesRegex(RuntimeError,"refusing overlay fallback"):
            reader_module.SinglePassReader(self.config,self.impl)
        self.assertEqual(self.bridge.calls,[])
        self.assertEqual(self.bridge.preparations,[])

    def test_load_tile4_verifies_geometry_and_calls_actual_variant4(self):
        self.config.single_pass_kernel="load_tile4"
        for key in (4,"4"):
            self.bridge.configuration=lambda:{"abi_version":1,"buffers_bytes":120,
                "load_tile_reader_available":True,"load_tile_variants":{key:
                    dict(rows=32,input_stride_bytes=416,input_depth=2,output_depth=1)}}
            reader=reader_module.SinglePassReader(self.config,self.impl)
            reader.read(*self.inputs())
            self.assertEqual(self.bridge.calls[-1][-6:],(128,False,False,False,False,4))
            self.assertFalse(self.bridge.preparations[-1][1][-1])
            self.assertEqual(reader.last_call_metadata["kernel_variant"],"load_tile4")
            self.assertEqual(reader.last_call_metadata["entry"],"QsfaSinglePassLaunchLoadTile")
            self.assertEqual(reader.last_call_metadata["device_entry"],"QsfaSinglePassLaunchLoadTile32")
            self.assertEqual(reader.last_call_metadata["load_variant"],4)
            self.assertEqual(reader.configuration()["load_variant"],4)

    def test_load_tile4_missing_export_or_wrong_geometry_never_launches(self):
        self.config.single_pass_kernel="load_tile4"
        for fields in ({},dict(load_tile_reader_available=True),
                       dict(load_tile_reader_available=True,load_tile_variants={4:
                           dict(rows=16,input_stride_bytes=416,input_depth=2,output_depth=1)}),
                       dict(load_tile_reader_available=True,load_tile_variants={4:
                           dict(rows=32,input_stride_bytes=416,input_depth=2,output_depth=True)})):
            with self.subTest(fields=fields):
                self.bridge.configuration=lambda:dict(abi_version=1,buffers_bytes=120,**fields)
                with self.assertRaisesRegex(RuntimeError,"load_tile4"):
                    reader_module.SinglePassReader(self.config,self.impl)
        self.assertEqual(self.bridge.calls,[])
        self.assertEqual(self.bridge.preparations,[])

    def test_rope_zero_and_wrong_abi_are_rejected_before_any_launch(self):
        self.impl.qk_rope_head_dim=0
        with self.assertRaisesRegex(RuntimeError,"272-byte"):
            reader_module.SinglePassReader(self.config,self.impl)
        self.loader_mock.assert_not_called()
        self.impl.qk_rope_head_dim=64
        self.bridge.configuration=lambda:{"abi_version":4,"buffers_bytes":152}
        with self.assertRaisesRegex(RuntimeError,"ABI1"):
            reader_module.SinglePassReader(self.config,self.impl)
        self.assertEqual(self.bridge.calls,[])

    def test_invalid_mapping_or_truncated_cache_contract_never_launches(self):
        reader=reader_module.SinglePassReader(self.config,self.impl)
        for change in ("mapping","packed_width","topk"):
            with self.subTest(change=change):
                values=self.inputs()
                if change=="mapping":values[4].data[0,0]=32
                elif change=="packed_width":values[2]=Tensor(np.zeros((32,128,1,272)),"int8")
                else:values[3]=Tensor(np.zeros((4,1,128),np.int32),"int32")
                with self.assertRaises(RuntimeError):reader.read(*values)
        self.assertEqual(self.bridge.calls,[])


if __name__=="__main__":unittest.main()
