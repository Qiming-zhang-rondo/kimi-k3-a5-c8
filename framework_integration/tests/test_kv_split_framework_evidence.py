"""Collector must distinguish requested KV-split from real split execution."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("kv_split_framework_collector",ROOT/"collect_framework_evidence.py")
collector=importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class KvSplitFrameworkEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="kv-split-framework-evidence-")
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/"layers").mkdir()
        self.write("probe.json",dict(status="GENERATION_FINISHED",arm="single_pass",
                                     single_pass_kernel="kv_split4"))

    def write(self,name,data):
        (self.root/name).write_text(json.dumps(data)+"\n",encoding="utf-8")

    def layer(self):
        return dict(status="EXERCISED",failure=None,reader="single_pass",single_pass_kernel="kv_split4",
            fusion_exercised=True,performance_claim=False,native_quantizer_changed=False,topk_changed=False,
            control_scope="full_FP64_whole_selected_set",control_limit=.01,
            control_max=dict(relative_l2=.0009,worst_row_relative_l2=.0012,max_abs=.01),
            counters=dict(read_calls=1,query_rows=4,fused_rows=4,fused_kernel_calls=1,
                full_math_checked_rows=4,candidate_attention_calls=1,validation_attention_calls=0,
                single_pass_expected_attention_calls=1,single_pass_kv_split4_calls=1,
                kv_split_attention_calls=1,kv_split_reduction_calls=1,kv_split_rows=4),
            single_pass_last_call=dict(status="EXERCISED",kernel_variant="kv_split4",
                executed_kernel_variant="kv_split4",dispatch_reason="q_le_4_kv_split4",
                queries=4,heads=64,head_groups=1,candidate_attention_calls=1,
                entry="QsfaSinglePassLaunchKvSplit",device_entry="qsfa_recent128_kv_split_attention",
                split_count=4,aic_count=16,device_reduction_calls=1,external_merge_calls=0,
                topk_partitioned=True,original_topk_unchanged=True,return_stats=False,validation_attention_calls=0),
            library_identity=dict(library_sha256="a"*64,configuration=dict(abi_version=1,buffers_bytes=120,
                kv_split_reader_available=True,kv_split_abi_version=2,kv_split_tiling_bytes=112,
                kv_split_reader_entry="QsfaSinglePassLaunchKvSplit",load_tile_reader_available=True,
                load_tile_variants={"4":dict(rows=32,input_stride_bytes=416,input_depth=2,output_depth=1)})))

    def collect(self,layer):
        self.write("layers/recent128-100-layer0.json",layer)
        return collector.collect(self.root)

    @staticmethod
    def fallback_call(layer):
        layer["single_pass_last_call"].update(queries=8,executed_kernel_variant="load_tile4",
            dispatch_reason="q_gt_4_prefill_load_tile4",entry="QsfaSinglePassLaunchLoadTile",
            device_entry="QsfaSinglePassLaunchLoadTile32",load_variant=4,split_count=1,aic_count=8,
            device_reduction_calls=0,topk_partitioned=False)

    def test_real_split_and_matching_reduction_with_full_math_validate_wiring_only(self):
        report=self.collect(self.layer())
        self.assertEqual(report["status"],"WIRING_VALIDATED")
        self.assertEqual(report["model_accuracy"],"NOT_MEASURED")
        self.assertEqual(report["serving_performance"],"NOT_MEASURED")
        self.assertEqual(report["counters"]["kv_split_attention_calls"],1)
        self.assertEqual(report["counters"]["kv_split_reduction_calls"],1)

    def test_only_fallback_cannot_establish_split_wiring(self):
        layer=self.layer()
        self.fallback_call(layer)
        layer["counters"].update(query_rows=8,fused_rows=8,full_math_checked_rows=8,
            kv_split_attention_calls=0,kv_split_reduction_calls=0,kv_split_rows=0,
            kv_split_prefill_fallback_calls=1)
        self.assertEqual(self.collect(layer)["status"],"FAILED")

    def test_last_prefill_fallback_after_real_split_is_explicitly_valid(self):
        layer=self.layer()
        self.fallback_call(layer)
        layer["counters"].update(read_calls=2,query_rows=12,fused_rows=12,full_math_checked_rows=12,
            fused_kernel_calls=2,candidate_attention_calls=2,single_pass_expected_attention_calls=2,
            single_pass_kv_split4_calls=2,kv_split_prefill_fallback_calls=1)
        self.assertEqual(self.collect(layer)["status"],"WIRING_VALIDATED")
        for key,value in (("executed_kernel_variant","kv_split4"),("topk_partitioned",True),
                          ("device_reduction_calls",1),("split_count",4)):
            with self.subTest(key=key):
                bad=deepcopy(layer)
                bad["single_pass_last_call"][key]=value
                self.assertEqual(self.collect(bad)["status"],"FAILED")

    def test_abi_entry_and_reduction_or_requested_coverage_cannot_be_faked(self):
        cases=[]
        no_abi=self.layer()
        del no_abi["library_identity"]["configuration"]["kv_split_abi_version"]
        cases.append(no_abi)
        wrong_size=self.layer()
        wrong_size["library_identity"]["configuration"]["kv_split_tiling_bytes"]=72
        cases.append(wrong_size)
        false_entry=self.layer()
        false_entry["single_pass_last_call"]["entry"]="QsfaSinglePassLaunchLoadTile"
        cases.append(false_entry)
        for name in ("kv_split_reduction_calls","single_pass_expected_attention_calls",
                     "single_pass_kv_split4_calls","kv_split_attention_calls","kv_split_rows"):
            bad=self.layer()
            bad["counters"][name]=0
            cases.append(bad)
        missing_coverage=self.layer()
        del missing_coverage["counters"]["single_pass_kv_split4_calls"]
        cases.append(missing_coverage)
        missing_reduction=self.layer()
        missing_reduction["single_pass_last_call"]["device_reduction_calls"]=0
        cases.append(missing_reduction)
        for index,case in enumerate(cases):
            with self.subTest(index=index):
                self.assertEqual(self.collect(case)["status"],"FAILED")

    def test_head_groups_count_calls_but_do_not_multiply_query_rows(self):
        layer=self.layer()
        layer["single_pass_last_call"].update(heads=128,head_groups=2,
            candidate_attention_calls=2,device_reduction_calls=2)
        layer["counters"].update(fused_kernel_calls=2,candidate_attention_calls=2,
            single_pass_expected_attention_calls=2,single_pass_kv_split4_calls=2,
            kv_split_attention_calls=2,kv_split_reduction_calls=2)
        self.assertEqual(self.collect(layer)["status"],"WIRING_VALIDATED")

    def test_math_or_probe_mismatch_still_blocks_valid_split_counters(self):
        layer=self.layer()
        layer["control_max"]["worst_row_relative_l2"]=.02
        self.assertEqual(self.collect(layer)["status"],"FAILED")
        self.write("probe.json",dict(status="GENERATION_FINISHED",arm="single_pass",
                                     single_pass_kernel="load_tile4"))
        self.assertEqual(self.collect(self.layer())["status"],"FAILED")


if __name__=="__main__":
    unittest.main()
