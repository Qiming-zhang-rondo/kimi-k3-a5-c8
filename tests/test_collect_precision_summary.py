"""Read-only extraction contracts; fixtures contain no device code or large arrays."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE=Path(__file__).resolve().parents[1]/"direct_bridge/collect_precision_summary.py"
spec=importlib.util.spec_from_file_location("collect_precision_summary",SOURCE)
collector=importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class PrecisionSummaryTests(unittest.TestCase):
    def extract(self,report):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"results.json"
            raw=json.dumps(report).encode()
            path.write_bytes(raw)
            result=collector.extract(path)
            self.assertEqual(path.read_bytes(),raw)
            return result

    def test_candidate_writer_partition_and_graph_exact_fields_survive_compaction(self):
        field=dict(passed=True,mismatch_count=0,expected_shape=[8192,400],actual_shape=[8192,400],
                   expected_dtype="uint8",actual_dtype="uint8",first_bad_positions=[],full_array=[1,2,3])
        report=dict(status="MEASURED",controls={"direct_fused_cold_store":dict(control_passed=True,
            writer=dict(passed=True,fields={"cold":field}),partition=dict(passed=True),
            kernel_vs_matching_reference=dict(passed=True,relative_l2=.0018),
            reader=dict(passed=True))},graph_capture_controls={
                "direct_fused_cold_store":dict(passed=True,poisoned_buffers=[[4,64,512]]),
                "cold_store_captured_bytes_and_partition":dict(passed=True,fields={
                    "cold":field,"hot_latent":field,"hot_rope":field,
                    "partition_old":dict(passed=True,mismatch_count=0,shape=[4,1,2048],dtype="int32")})})
        result=self.extract(report)
        candidate=result["controls"]["direct_fused_cold_store"]
        self.assertEqual(candidate["check_statuses"]["writer"],{"passed":True})
        self.assertEqual(candidate["check_statuses"]["partition"],{"passed":True})
        self.assertEqual(candidate["kernel_vs_matching_reference"]["relative_l2"],.0018)
        detail=result["graph_capture_detail"]["cold_store_captured_bytes_and_partition"]
        self.assertTrue(detail["passed"])
        self.assertEqual(set(detail["fields"]),{"cold","hot_latent","hot_rope","partition_old"})
        for name in ("cold","hot_latent","hot_rope"):
            self.assertEqual(detail["fields"][name],{k:v for k,v in field.items()
                if k not in ("first_bad_positions","full_array")})
        self.assertEqual(detail["fields"]["partition_old"]["dtype"],"int32")
        self.assertEqual(result["graph_capture_controls"]["direct_fused_cold_store"],True)
        self.assertNotIn("poisoned_buffers",result["graph_capture_detail"]["direct_fused_cold_store"])

    def test_missing_null_and_legacy_false_remain_distinct_from_overall_success(self):
        report=dict(status="MEASURED",controls={"candidate":dict(control_passed=True,
            writer=dict(passed=None),partition={},reader=dict(passed=True,
                zero_query_same_native_states_merge=dict(passed=False,relative_l2=.000768,
                    max_row_relative_l2=.001010071),
                zero_query_merge_acceptance=dict(passed=True,basis="exact_zero_query_BF16_RNE")))},
            graph_capture_controls={"missing":{},"null":dict(passed=None,fields=None),
                "nested":dict(passed=True,fields={"missing":{},"null":dict(passed=None)})})
        result=self.extract(report)
        candidate=result["controls"]["candidate"]
        self.assertTrue(candidate["control_passed"])
        self.assertEqual(candidate["check_statuses"]["writer"],{"passed":None})
        self.assertEqual(candidate["check_statuses"]["partition"],{})
        self.assertNotIn("zero_query",candidate["check_statuses"])
        self.assertFalse(candidate["reader"]["zero_query_same_native_states_merge"]["passed"])
        self.assertTrue(candidate["reader"]["zero_query_merge_acceptance"]["passed"])
        self.assertIn("controls.candidate.reader.zero_query_same_native_states_merge",result["failed_check_paths"])
        details=result["graph_capture_detail"]
        self.assertEqual(details["missing"],{})
        self.assertEqual(details["null"],{"passed":None,"fields":None})
        self.assertEqual(details["nested"]["fields"],{"missing":{},"null":{"passed":None}})


if __name__=="__main__":unittest.main()
