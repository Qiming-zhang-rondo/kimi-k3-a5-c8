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


if __name__=="__main__": unittest.main()
