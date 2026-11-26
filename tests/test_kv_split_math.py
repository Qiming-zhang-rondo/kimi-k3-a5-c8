"""Independent host checks for column-wise KV split and stable FP32 merge."""
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experimental/single_pass"))
from kv_split_math import (PartialState, attention_state, merge_partial_states,
                           split_attention_reference, split_topk_columns)


class KvSplitIndicesTest(unittest.TestCase):
    def test_exact_original_membership_order_and_suffix_padding_survive_all_splits(self):
        for count in (0,1,511,512,513,1023,1024,2047,2048):
            ids=np.full((2,1,2048),-1,dtype=np.int32)
            ids[:,:,:count]=np.arange(count,dtype=np.int32)[None,None,::-1]
            for splits in (1,2,4):
                parts=split_topk_columns(ids,splits,keys=4096,positions=np.array([2047,4095]))
                self.assertEqual(len(parts),splits)
                self.assertTrue(all(p.shape==(2,1,2048//splits) and p.flags.c_contiguous for p in parts))
                np.testing.assert_array_equal(np.concatenate(parts,axis=-1),ids)
                self.assertEqual(sum(int(np.sum(p>=0)) for p in parts),count*2)

    def test_duplicate_across_split_boundary_is_rejected(self):
        ids=np.arange(2048,dtype=np.int32).reshape(1,1,-1)
        ids[0,0,1024]=ids[0,0,1023]
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            split_topk_columns(ids,2,keys=4096)

    def test_invalid_sentinel_padding_range_and_future_are_rejected_without_repair(self):
        original=np.arange(2048,dtype=np.int32).reshape(1,1,-1)
        for kind in ("sentinel","padding","range","future"):
            ids=original.copy()
            if kind=="sentinel": ids[0,0,-1]=-2
            elif kind=="padding": ids[0,0,100]=-1
            elif kind=="range": ids[0,0,-1]=4096
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                split_topk_columns(ids,4,keys=4096,positions=np.array([1023]) if kind=="future" else None)
        np.testing.assert_array_equal(original,np.arange(2048).reshape(1,1,-1))

    def test_contract_rejects_wrong_width_dtype_split_count_and_position_shape(self):
        ids=np.arange(2048,dtype=np.int32).reshape(1,1,-1)
        for splits in (0,3,8,True,2.0):
            with self.assertRaises(ValueError): split_topk_columns(ids,splits,keys=4096)
        for bad in (ids[:,:,:1024],ids[:,0],ids.astype(float),np.ones(ids.shape,bool)):
            with self.assertRaises(ValueError): split_topk_columns(bad,2,keys=4096)
        with self.assertRaises(ValueError): split_topk_columns(ids,2,keys=4096,positions=np.array([[4095]]))


class KvSplitMathTest(unittest.TestCase):
    def test_random_multi_query_heads_fp32_partials_match_independent_fp64_attention(self):
        rng=np.random.default_rng(2926)
        scores=rng.normal(0,2,(3,4,2048))
        values=rng.normal(0,3,(3,1,2048,9))
        mask=np.arange(2048)[None,None,:] < np.array([2048,257,0])[:,None,None]
        expected=attention_state(scores,values,valid=mask,dtype=np.float64)
        for splits in (1,2,4):
            actual,partials=split_attention_reference(scores,values,splits,valid=mask)
            self.assertTrue(all(p.output.dtype==np.float32 for p in partials))
            np.testing.assert_allclose(actual.output,expected.output,rtol=2e-6,atol=3e-7)
            np.testing.assert_allclose(actual.maximum,expected.maximum,rtol=1e-6)
            np.testing.assert_allclose(actual.total,expected.total,rtol=2e-6,atol=1e-6)
            np.testing.assert_array_equal(actual.output[2],0)
            self.assertTrue(np.isneginf(actual.maximum[2]).all())
            np.testing.assert_array_equal(actual.total[2],0)

    def test_zero_query_counts_and_partly_empty_partitions(self):
        values=np.arange(2048*3,dtype=np.float64).reshape(1,2048,3)/100
        scores=np.zeros((1,2048),dtype=np.float64)
        for count in (0,1,128,257,512,1024,1920,2048):
            valid=np.arange(2048)[None,:]<count
            for splits in (1,2,4):
                state,partials=split_attention_reference(scores,values,splits,valid=valid)
                self.assertEqual(float(state.total[0,0]),count)
                self.assertEqual(sum(float(p.total[0,0]) for p in partials),count)
                if count:
                    np.testing.assert_allclose(state.output[0],values[0,:count].mean(axis=0),rtol=2e-6)
                    self.assertEqual(float(state.maximum[0,0]),0)
                else:
                    np.testing.assert_array_equal(state.output,0)
                    self.assertTrue(np.isneginf(state.maximum).all())

    def test_large_score_offsets_never_exponentiate_absolute_lse(self):
        rng=np.random.default_rng(412)
        base=rng.normal(0,2,(2,2048))
        values=rng.normal(size=(2,2048,7))
        with warnings.catch_warnings():
            warnings.simplefilter("error",RuntimeWarning)
            for offset in (-10000.,10000.):
                expected=attention_state(base+offset,values,dtype=np.float64)
                actual,_=split_attention_reference(base+offset,values,4)
                np.testing.assert_allclose(actual.output,expected.output,rtol=.003,atol=2e-5)
                self.assertTrue(np.isfinite(actual.output).all())
                self.assertTrue(np.isfinite(actual.total).all())

    def test_normalized_partial_output_uses_its_softmax_sum_in_the_merge(self):
        first=PartialState(np.array([[1.,2.]],np.float32),np.array([[10.]],np.float32),np.array([[2.]],np.float32))
        second=PartialState(np.array([[-3.,5.]],np.float32),np.array([[11.]],np.float32),np.array([[7.]],np.float32))
        actual=merge_partial_states((first,second))
        weight=2*np.exp(-1)
        expected=(first.output*weight+second.output*7)/(weight+7)
        np.testing.assert_allclose(actual.output,expected,rtol=2e-7)
        np.testing.assert_allclose(actual.total,weight+7,rtol=2e-7)
        self.assertEqual(float(actual.maximum[0,0]),11)

    def test_empty_partial_has_zero_weight_and_all_empty_has_canonical_state(self):
        empty=PartialState(np.zeros((2,5),np.float32),np.full((2,1),-np.inf,np.float32),np.zeros((2,1),np.float32))
        live=PartialState(np.full((2,5),3.,np.float32),np.full((2,1),-20.,np.float32),np.full((2,1),2.,np.float32))
        with warnings.catch_warnings():
            warnings.simplefilter("error",RuntimeWarning)
            for states in ((empty,live),(live,empty),(empty,empty,live,empty)):
                merged=merge_partial_states(states)
                np.testing.assert_array_equal(merged.output,live.output)
                np.testing.assert_array_equal(merged.maximum,live.maximum)
                np.testing.assert_array_equal(merged.total,live.total)
            merged=merge_partial_states((empty,empty,empty,empty))
            np.testing.assert_array_equal(merged.output,0)
            np.testing.assert_array_equal(merged.total,0)
            self.assertTrue(np.isneginf(merged.maximum).all())

    def test_padding_nan_slots_do_not_contribute_but_active_nonfinite_rejects(self):
        scores=np.array([[0.,np.nan]])
        values=np.array([[[4.,5.],[np.nan,np.inf]]])
        state=attention_state(scores,values,valid=np.array([[True,False]]))
        np.testing.assert_array_equal(state.output,[[4.,5.]])
        with self.assertRaises(ValueError): attention_state(scores,values)

    def test_malformed_partial_states_reject_instead_of_silently_masking(self):
        good=PartialState(np.zeros((2,3),np.float32),np.zeros((2,1),np.float32),np.ones((2,1),np.float32))
        malformed=(PartialState(good.output,good.maximum,np.full((2,1),-1.,np.float32)),
                   PartialState(good.output,np.full((2,1),np.inf,np.float32),good.total),
                   PartialState(np.ones((2,3),np.float32),np.full((2,1),-np.inf,np.float32),np.zeros((2,1),np.float32)),
                   PartialState(good.output,good.maximum,np.zeros((2,1),np.float32)),
                   PartialState(good.output,good.maximum,np.ones((2,),np.float32)),
                   PartialState(good.output.astype(np.float16),good.maximum,good.total))
        for state in malformed:
            with self.assertRaises(ValueError): merge_partial_states((good,state))
        with self.assertRaises(ValueError): merge_partial_states(())
        with self.assertRaises(ValueError): merge_partial_states((good,good,good))


if __name__=="__main__":
    unittest.main()
