"""Real runtime oracle path with CPU tensors: page remap and no extra attention."""
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"framework_integration"))
from sfa_recent128.runtime import Recent128Runtime


class Tensor:
    def __init__(self,array): self.array=np.asarray(array);self.device="fake_cpu";self.selected=None
    @property
    def shape(self): return self.array.shape
    def detach(self): return self
    def float(self): return Tensor(self.array.astype(np.float32))
    def cpu(self): return self
    def numpy(self): return self.array
    def index_select(self,axis,indices):
        self.selected=indices.array.copy()
        return Tensor(np.take(self.array,indices.array,axis=axis))
    def view(self,dtype):return Tensor(self.array.view(dtype))


class SinglePassMathRuntimeTest(unittest.TestCase):
    def fixture(self):
        r=Recent128Runtime.__new__(Recent128Runtime)
        r.impl=SimpleNamespace(scale=1/24)
        r.config=SimpleNamespace(source_root=str(ROOT),capacity=256)
        r.step=SimpleNamespace(positions=np.array([255],np.int64))
        r.end=256;r.tags=np.arange(256,dtype=np.int64);r.counts=Counter()
        r.control_max=dict(relative_l2=0.,worst_row_relative_l2=0.,max_abs=0.)
        hot=np.zeros((2,128,1,512),np.float32);hot.reshape(256,512)[128]=2;hot.reshape(256,512)[255]=4
        r.hot=Tensor(hot);r.rope=Tensor(np.zeros((2,128,1,64),np.float32))
        def fail(message):raise RuntimeError(message)
        r._error=fail
        packed=np.zeros((4,128,1,400),np.uint8)
        packed[...,:256]=0x66;packed[...,384:]=127
        packed[3,:,:,:256]=0x22 # logical token0 is physical page3, value1.
        cold=Tensor(packed)
        q=Tensor(np.zeros((1,1,512),np.float32));qr=Tensor(np.zeros((1,1,64),np.float32))
        ids=np.full((1,1,2048),-1,np.int32);ids[0,0,:3]=[0,128,255]
        topk=Tensor(ids);table=Tensor(np.array([[3,1]],np.int32))
        reader=SimpleNamespace(calls=0,last_attention_calls=1,last_call_metadata=dict(
            kernel_variant="overlay",entry="QsfaSinglePassLaunch",return_stats=False,
            external_merge_calls=0,topk_partitioned=False,head_groups=1))
        def read(*args):
            reader.calls+=1
            self.assertIs(args[2],cold)
            self.assertIs(args[3],topk)
            self.assertIs(args[4],table)
            # BF16 RNE of (cold1 + hot2 + hot4)/3.
            return Tensor(np.full((1,1,512),2.328125,np.float32))
        reader.read=read;r.single_pass_reader=reader
        fake_torch=SimpleNamespace(uint8=np.uint8,int64=np.int64,
            as_tensor=lambda a,device=None,dtype=None:Tensor(np.asarray(a,dtype=dtype)))
        inputs=(q,qr,cold,topk,table,Tensor(np.array([1],np.int32)),Tensor(np.array([256],np.int32)),ids)
        return r,reader,cold,fake_torch,inputs

    def test_full_oracle_checks_actual_heads_and_remapped_selected_pages(self):
        r,reader,cold,fake,inputs=self.fixture()
        with patch.dict(sys.modules,{"torch":fake}):out=r._read_single_pass(*inputs)
        self.assertEqual(out.shape,(1,1,512))
        self.assertEqual(reader.calls,1)
        np.testing.assert_array_equal(cold.selected,[3,1])
        self.assertEqual(r.counts["full_math_checked_rows"],1)
        self.assertEqual(r.counts["candidate_attention_calls"],1)
        self.assertEqual(r.counts["validation_attention_calls"],0)
        self.assertLess(r.control_max["relative_l2"],.01)

    def test_future_and_missing_raw_bf16_tags_fail_before_attention(self):
        for fault in ("future","tag"):
            with self.subTest(fault=fault):
                r,reader,cold,fake,inputs=self.fixture()
                if fault=="future":inputs[-1][0,0,2]=256
                else:r.tags[128]=-1
                with patch.dict(sys.modules,{"torch":fake}),self.assertRaises(ValueError if fault=="tag" else RuntimeError):
                    r._read_single_pass(*inputs)
                self.assertEqual(reader.calls,0)


if __name__=="__main__":unittest.main()
