"""CPU-only source/format/one-softmax gates for the new single-pass candidate."""
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experimental/single_pass"))
from host_contract import (ContractError, SinglePassContract, bf16_round,
    decode_c4_rows, dense_attention_reference, online_attention_reference)


def packed_row(code, rope=None, scale=127):
    codes = np.broadcast_to(np.asarray(code, np.uint8), (512,))
    result = np.zeros(400, np.uint8)
    result[:256] = codes[0::2] | (codes[1::2] << 4)
    rope = np.zeros(64, np.float32) if rope is None else bf16_round(rope)
    result[256:384] = (rope.view(np.uint32) >> 16).astype("<u2").view(np.uint8)
    result[384:400] = scale
    return result


def case():
    contract = SinglePassContract(queries=2, keys=256, topk=8, aic_count=2)
    cold = np.zeros((2,128,1,400), np.uint8)
    table = np.array([[1,0]], np.int32)  # Test the true logical->physical page map.
    for token in range(256):
        cold[table[0,token//128],token % 128,0] = packed_row(token % 8)
    latent = np.zeros((256,512), np.float32)
    rope = np.zeros((256,64), np.float32)
    tags = np.full(256,-1,np.int64)
    for token in range(130):
        latent[token % 256] = 8+token % 7
        rope[token % 256] = token % 3
        tags[token % 256] = token
    return contract, dict(query=np.zeros((2,64,576),np.float32), cold_packed=cold,
        block_table=table, topk=np.array([[[0,1,127,128,-1,-1,-1,-1]],
                                        [[0,1,128,129,-1,-1,-1,-1]]],np.int32),
        positions=np.array([128,129],np.int64), hot_latent=latent,
        hot_rope=rope, hot_tags=tags)


class SinglePassContractTest(unittest.TestCase):
    def test_real_mxfp4_400_layout_low_nibble_scale32_rope64(self):
        row = packed_row(np.arange(512,dtype=np.uint8) % 16, rope=np.arange(64,dtype=np.float32)/8)
        row[385] = 128
        latent, rope = decode_c4_rows(row[None])
        np.testing.assert_array_equal(latent[0,:8], [0,.5,1,1.5,2,3,4,6])
        self.assertTrue(np.signbit(latent[0,8]))
        self.assertEqual(latent[0,33], 1.)  # Second scale/32 group doubled.
        np.testing.assert_array_equal(rope[0], np.arange(64)/8)
        row[384] = 255
        with self.assertRaisesRegex(ContractError,"E8M0"):
            decode_c4_rows(row[None])
        with self.assertRaisesRegex(ContractError,"400"):
            decode_c4_rows(np.zeros((1,416),np.uint8))

    def test_zero_query_counts_same_topk_and_true_hot_values(self):
        contract, arrays = case()
        original = arrays["topk"].copy()
        result = online_attention_reference(contract, block_keys=3, **arrays)
        np.testing.assert_array_equal(arrays["topk"], original)
        np.testing.assert_array_equal(result["selected_topk"], original)
        np.testing.assert_array_equal(result["total"], np.full((2,64,1),4.))
        np.testing.assert_array_equal(result["valid_counts"], [4,4])
        np.testing.assert_array_equal(result["hot_counts"], [3,2])
        # Q0: cold token0=0; hot token1=9,127=9,128=10. Q1 token1 is cold .5.
        np.testing.assert_array_equal(result["output"][0], np.full((64,512),7.))
        np.testing.assert_array_equal(result["output"][1], np.full((64,512),(0+.5+10+11)/4))
        self.assertEqual(result["attention_streams_per_query"],1)
        self.assertFalse(result["external_merge"])

    def test_one_online_normalizer_matches_direct_softmax_across_mixed_tiles(self):
        contract, arrays = case()
        rng = np.random.default_rng(11)
        arrays["query"] = bf16_round(rng.normal(0,.05,(2,64,576)))
        dense = dense_attention_reference(contract,**arrays)
        for tile in (1,2,3,128):
            result = online_attention_reference(contract,block_keys=tile,**arrays)
            np.testing.assert_allclose(result["output"],dense["output"],atol=2e-14,rtol=2e-14)
            np.testing.assert_allclose(result["lse"],dense["lse"],atol=2e-14,rtol=2e-14)

    def test_missing_hot_tag_future_duplicate_and_holey_padding_are_rejected(self):
        contract, arrays = case()
        arrays["hot_tags"][127] = -1
        with self.assertRaisesRegex(ContractError,"original BF16"):
            online_attention_reference(contract,**arrays)
        for bad, reason in (([0,129,-1,-1,-1,-1,-1,-1],"Future"),
                            ([0,0,-1,-1,-1,-1,-1,-1],"Duplicate"),
                            ([0,-1,1,-1,-1,-1,-1,-1],"suffix")):
            contract, arrays = case()
            arrays["topk"][0,0] = bad
            with self.subTest(reason=reason), self.assertRaisesRegex(ContractError,reason):
                online_attention_reference(contract,**arrays)

    def test_allhot_never_uses_c4_as_high_precision_and_empty_rows_have_zero_mass(self):
        contract, arrays = case()
        arrays["topk"][0,0] = [1,127,128,-1,-1,-1,-1,-1]
        arrays["topk"][1,0] = -1
        arrays["cold_packed"][:,:,:,384:] = 255  # No selected cold row exists.
        result = online_attention_reference(contract,**arrays)
        np.testing.assert_array_equal(result["output"][0],np.full((64,512),28/3))
        np.testing.assert_array_equal(result["output"][1],np.zeros((64,512)))
        np.testing.assert_array_equal(result["total"][1],np.zeros((64,1)))
        self.assertTrue(np.isneginf(result["lse"][1]).all())

    def test_window0_uses_all_c4_and_ring_retention_limits_are_explicit(self):
        contract, arrays = case()
        off = SinglePassContract(queries=2,keys=256,topk=8,aic_count=2,window=0)
        result = online_attention_reference(off,**arrays)
        np.testing.assert_array_equal(result["hot_counts"],[0,0])
        # C4 codes0,1,7,0 -> 0,.5,6,0; independent original hot values ignored.
        np.testing.assert_array_equal(result["output"][0],np.full((64,512),6.5/4))
        for changes in (dict(heads=32),dict(queries=1025),dict(ring_capacity=128),
                         dict(aic_count=5),dict(scale=math.nan)):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                SinglePassContract(**changes)

    def test_new_pod_consumer_coverage_and_actual_host_compile(self):
        source = ROOT/"native_c4_source/source/ascendc/kv_quant_sparse_flash_attention/arch35/kv_quant_sparse_flash_attention_service_vector_mla.h"
        text = source.read_text()
        consumed = set(re.findall(r"sparseAttnSharedkvBaseParams\.([A-Za-z_]\w*)",text))
        header = ROOT/"experimental/single_pass/single_pass_tiling.h"
        pod = header.read_text()
        self.assertEqual(len(consumed),14)
        for name in consumed:
            self.assertRegex(pod,rf"\b{name};")
        self.assertIn("NOT the installed private C4 tiling layout",pod)
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("Host compiler unavailable; no SDK/device claim")
        with tempfile.TemporaryDirectory(prefix="sfa-single-pass-pod-") as tmp:
            driver = Path(tmp)/"check.cpp"
            driver.write_text('''#include "single_pass_tiling.h"
#include <iostream>
int main(){auto t=recent128_single_pass::MakeTiling(4,8192,4,1.0f/24,true);
static_assert(sizeof(t)==72);static_assert(recent128_single_pass::userWorkspaceBytes==0);
if(t.baseParams.dSizeVInput!=400 || t.baseParams.qSeqSize!=4 || t.baseParams.sparseBlockCount!=2048)return 2;
try{recent128_single_pass::MakeTiling(4,8192,4,0,true);return 3;}catch(const std::invalid_argument&){}
std::cout<<sizeof(t)<<' '<<t.baseParams.dSizeVInput;}
''')
            executable = Path(tmp)/"check"
            subprocess.run([compiler,"-std=c++17","-I",str(header.parent),str(driver),"-o",str(executable)],
                           check=True,capture_output=True,text=True)
            result = subprocess.run([str(executable)],check=True,capture_output=True,text=True)
            self.assertEqual(result.stdout,"72 400")


if __name__ == "__main__":
    unittest.main()
