"""Collector gates on actual layer evidence, never generation alone."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("recent128_framework_collector",ROOT/"collect_framework_evidence.py")
collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)


class FrameworkEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="framework-evidence-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/"layers").mkdir()
        # Also satisfy the explicit orderly-probe gate if it is enabled.
        self.write("probe.json",dict(status="GENERATION_FINISHED",arm="fused",prompts=1,
            prompt_sha256="a"*64,max_tokens=2,scope="wiring smoke only"))

    def write(self,name,data):
        (self.root/name).write_text(json.dumps(data)+"\n",encoding="utf-8")

    def layer(self,name="layer0",**changes):
        data=dict(version="recent128-framework-accuracy-v1",execution="accuracy_eager",
            status="EXERCISED",reader="fused_required",window=128,ring_capacity=1152,
            counters=dict(read_calls=2,query_rows=256,fused_rows=128,fused_kernel_calls=1,
                          raw_bf16_write_rows=256),reasons={},fusion_exercised=True,failure=None,
            control_max=dict(relative_l2=.0001,worst_row_relative_l2=.0002,max_abs=.0003),
            performance_claim=False,prolog_changed=False,native_quantizer_changed=False,topk_changed=False)
        data.update(changes)
        self.write("layers/recent128-100-"+name+".json",data)

    def test_generation_without_layer_execution_is_not_exercised(self):
        result=collector.collect(self.root)
        self.assertEqual(result["status"],"NOT_EXERCISED")
        self.assertEqual(result["model_accuracy"],"NOT_MEASURED")

    def test_missing_or_unfinished_probe_blocks_even_valid_layer_evidence(self):
        self.layer()
        (self.root/"probe.json").unlink()
        self.assertEqual(collector.collect(self.root)["status"],"FAILED")
        self.write("probe.json",dict(status="STARTED",arm="fused"))
        self.assertEqual(collector.collect(self.root)["status"],"FAILED")

    def test_unexercised_or_failed_layer_cannot_be_validated(self):
        for changes in (dict(status="NOT_EXERCISED"),dict(status="FAILED",failure="native control mismatch")):
            with self.subTest(changes=changes):
                self.layer(**changes)
                result=collector.collect(self.root)
                self.assertEqual(result["status"],"FAILED")
                self.assertTrue(result["problems"])

    def test_zero_fusion_is_not_accepted_as_a_fused_run(self):
        self.layer(fusion_exercised=False,counters=dict(read_calls=2,query_rows=128,fused_rows=0))
        result=collector.collect(self.root)
        self.assertEqual(result["status"],"FAILED")
        self.assertTrue(any("fusion" in problem["reason"] for problem in result["problems"]))

    def test_minimal_valid_layers_only_establish_wiring_not_model_accuracy_or_speed(self):
        self.layer()
        self.layer("layer1")
        result=collector.collect(self.root)
        self.assertEqual(result["status"],"WIRING_VALIDATED")
        self.assertEqual(result["layers"],2)
        self.assertEqual(result["counters"]["read_calls"],4)
        self.assertEqual(result["counters"]["fused_rows"],256)
        self.assertEqual(result["model_accuracy"],"NOT_MEASURED")
        self.assertEqual(result["serving_performance"],"NOT_MEASURED")

    def test_changed_contract_or_any_failed_layer_blocks_aggregate(self):
        self.layer("good")
        for flag in ("topk_changed","native_quantizer_changed","performance_claim"):
            with self.subTest(flag=flag):
                self.layer("bad",**{flag:True})
                result=collector.collect(self.root)
                self.assertEqual(result["status"],"FAILED")
                self.assertTrue(result["problems"])

    def test_native_only_collection_is_explicit_and_does_not_claim_fusion(self):
        self.layer(reader="native",fusion_exercised=False,counters=dict(read_calls=1,fused_rows=0))
        result=collector.collect(self.root,require_fusion=False)
        self.assertEqual(result["status"],"WIRING_VALIDATED")
        self.assertFalse(result["fusion_required"])
        self.assertEqual(result["counters"]["fused_rows"],0)
        self.assertEqual(result["model_accuracy"],"NOT_MEASURED")

    def test_single_pass_uses_whole_math_gate_and_requires_full_row_coverage(self):
        self.layer(reader="single_pass",control_scope="full_FP64_whole_selected_set",control_limit=.01,
            counters=dict(read_calls=2,query_rows=256,fused_rows=256,fused_kernel_calls=2,
                          full_math_checked_rows=256,candidate_attention_calls=2,validation_attention_calls=0),
            control_max=dict(relative_l2=.002,worst_row_relative_l2=.003,max_abs=.01))
        self.assertEqual(collector.collect(self.root)["status"],"WIRING_VALIDATED")
        # The old compact-native contract stays strict at .001.
        self.layer(control_max=dict(relative_l2=.002,worst_row_relative_l2=.003,max_abs=.01))
        self.assertEqual(collector.collect(self.root)["status"],"FAILED")
        self.layer(reader="single_pass",control_scope="full_FP64_whole_selected_set",control_limit=.01,
            counters=dict(read_calls=2,query_rows=256,fused_rows=256,fused_kernel_calls=2,
                          full_math_checked_rows=1,candidate_attention_calls=2,validation_attention_calls=0))
        self.assertEqual(collector.collect(self.root)["status"],"FAILED")

    def test_fast_pair_request_requires_actual_export_identity_and_full_call_counts(self):
        self.write("probe.json",dict(status="GENERATION_FINISHED",arm="single_pass",single_pass_kernel="fast_pair"))
        counts=dict(read_calls=2,query_rows=256,fused_rows=256,fused_kernel_calls=2,
            full_math_checked_rows=256,candidate_attention_calls=2,validation_attention_calls=0,
            single_pass_expected_attention_calls=2,single_pass_fast_pair_calls=2)
        call=dict(kernel_variant="fast_pair",entry="QsfaSinglePassLaunchFastPair",return_stats=False,
            external_merge_calls=0,topk_partitioned=False,head_groups=1,candidate_attention_calls=1)
        identity=dict(library_sha256="a"*64,configuration=dict(abi_version=1))
        def write_case(**changes):
            values=dict(reader="single_pass",single_pass_kernel="fast_pair",counters=counts,
                control_scope="full_FP64_whole_selected_set",control_limit=.01,
                single_pass_last_call=call,library_identity=identity)
            values.update(changes)
            self.layer(**values)
        write_case()
        self.assertEqual(collector.collect(self.root)["status"],"WIRING_VALIDATED")
        for changes in (dict(single_pass_last_call=dict(call,entry="QsfaSinglePassLaunch")),
                        dict(library_identity=dict(identity,library_sha256=None)),
                        dict(counters=dict(counts,single_pass_fast_pair_calls=1))):
            with self.subTest(changes=changes):
                write_case(**changes)
                self.assertEqual(collector.collect(self.root)["status"],"FAILED")

    def test_tile4_request_requires_actual_geometry_and_variant_calls(self):
        self.write("probe.json",dict(status="GENERATION_FINISHED",arm="single_pass",single_pass_kernel="load_tile4"))
        counts=dict(read_calls=1,query_rows=4,fused_rows=4,fused_kernel_calls=1,
            full_math_checked_rows=4,candidate_attention_calls=1,validation_attention_calls=0,
            single_pass_expected_attention_calls=1,single_pass_load_tile4_calls=1)
        call=dict(kernel_variant="load_tile4",entry="QsfaSinglePassLaunchLoadTile",load_variant=4,
            device_entry="QsfaSinglePassLaunchLoadTile32",return_stats=False,
            external_merge_calls=0,topk_partitioned=False,head_groups=1,candidate_attention_calls=1)
        geometry=dict(rows=32,input_stride_bytes=416,input_depth=2,output_depth=1)
        def write_case(actual_geometry):
            self.layer(reader="single_pass",single_pass_kernel="load_tile4",counters=counts,
                control_scope="full_FP64_whole_selected_set",control_limit=.01,
                single_pass_last_call=call,library_identity=dict(library_sha256="b"*64,
                    configuration=dict(abi_version=1,load_tile_variants={"4":actual_geometry})))
        write_case(geometry)
        self.assertEqual(collector.collect(self.root)["status"],"WIRING_VALIDATED")
        write_case(dict(geometry,rows=16))
        self.assertEqual(collector.collect(self.root)["status"],"FAILED")


if __name__=="__main__": unittest.main()
