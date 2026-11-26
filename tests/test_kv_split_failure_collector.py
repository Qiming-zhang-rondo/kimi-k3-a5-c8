"""Read-only diagnostic extraction; no device or Torch dependencies."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experimental/single_pass"))
import collect_kv_split_failure as collector


class FailureCollectorTests(unittest.TestCase):
    def fixture(self,directory,*,full_outputs):
        root=Path(directory)
        partial_path=root/"partial_diagnostics"/"main_w0_split1.npz"
        partial_path.parent.mkdir()
        output=np.ones((1,1,64,512),np.float32)
        maximum=np.zeros((1,1,64),np.float32)
        total=np.full((1,1,64),2048,np.float32)
        np.savez_compressed(partial_path,partial_output=output,partial_maximum=maximum,partial_sum=total)
        source=dict(status="CONTROL_FAILED",version="actual-test-run",
            args=dict(queries=1,splits=[1],return_stats=False),
            bridge_build=dict(library_sha256="a"*64),
            bridge_configuration=dict(kv_split_abi_version=1,kv_split_tiling_bytes=152),
            controls=dict(mathematical={"w0_split1":dict(passed=False,relative_l2=1.0123456789012345)},
                partial_reduce={"w0_split1":dict(passed=False,relative_l2=.9876543210987654)},
                zero_query={"zero_full":{"w0_split1":dict(passed=True,cpu_partial_merge=dict(passed=True))}}),
            graph_controls={},partial_diagnostics={"main_w0_split1":dict(path="/unmounted/old/main_w0_split1.npz")})
        path=root/"results.json"
        path.write_text(json.dumps(source,indent=2)+"\n")
        if full_outputs:
            np.savez_compressed(root/"outputs.npz",w0_split1=np.zeros((1,64,512),np.float32),
                w0_matching_math=np.ones((1,64,512),np.float64))
        return path,source

    def test_missing_final_arrays_never_invent_metrics_and_source_fields_remain_verbatim(self):
        with tempfile.TemporaryDirectory() as directory:
            path,source=self.fixture(directory,full_outputs=False)
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(directory).rglob("*") if p.is_file()}
            report=collector.collect(path)
            self.assertEqual(report["source_results_sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(report["bridge_build"]["library_sha256"],"a"*64)
            self.assertEqual(report["cases"]["w0_split1"]["mathematical"],source["controls"]["mathematical"]["w0_split1"])
            self.assertEqual(report["cases"]["w0_split1"]["npu_vs_cpu_partial_merge"],source["controls"]["partial_reduce"]["w0_split1"])
            self.assertEqual(report["full_outputs"]["status"],"NOT_AVAILABLE")
            partial=report["partial_files"]["main_w0_split1"]
            self.assertEqual(partial["maximum_bits"]["most_frequent"],{"0x00000000":64})
            self.assertEqual(partial["recomputed_cpu_merge_vs_saved_npu"]["status"],"NOT_AVAILABLE")
            self.assertNotIn("saved_output_vs_matching_math",report["cases"]["w0_split1"])
            identity=report["cases"]["w0_split1"]["split1_identity_checks"]
            self.assertEqual(identity["saved_final_vs_bf16_partial0"]["status"],"NOT_AVAILABLE")
            self.assertEqual(identity["partial0_vs_matching_math"]["status"],"NOT_AVAILABLE")
            after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(directory).rglob("*") if p.is_file()}
            self.assertEqual(before,after)

    def test_existing_saved_outputs_produce_query_head_errors_without_changing_reported_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path,source=self.fixture(directory,full_outputs=True)
            report=collector.collect(path)
            metric=report["cases"]["w0_split1"]["saved_output_vs_matching_math"]
            self.assertEqual(metric["relative_l2"],1.)
            self.assertEqual(len(metric["per_query"]["actual_l2"]),1)
            self.assertEqual(len(metric["per_head_across_queries"]["relative_l2"]),64)
            self.assertEqual(metric["per_query"]["cosine"],[0.])
            self.assertEqual(metric["per_head_across_queries"]["cosine"],[0.]*64)
            identity=report["cases"]["w0_split1"]["split1_identity_checks"]
            self.assertEqual(identity["saved_final_vs_bf16_partial0"]["relative_l2"],1.)
            self.assertEqual(identity["partial0_vs_matching_math"]["relative_l2"],0.)
            self.assertEqual(identity["partial0_vs_matching_math"]["per_head_across_queries"]["cosine"],[1.]*64)
            recomputed=report["partial_files"]["main_w0_split1"]["recomputed_cpu_merge_vs_saved_npu"]
            self.assertEqual(recomputed["relative_l2"],1.)
            self.assertEqual(report["cases"]["w0_split1"]["npu_vs_cpu_partial_merge"]["relative_l2"],
                source["controls"]["partial_reduce"]["w0_split1"]["relative_l2"])
            encoded=collector.encode_bounded(report)
            self.assertLessEqual(len(encoded.encode()),50000)
            json.loads(encoded)

    def test_split1_identity_requires_each_saved_array_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            path,_=self.fixture(directory,full_outputs=False)
            np.savez_compressed(Path(directory)/"outputs.npz",w0_split1=np.zeros((1,64,512),np.float32))
            report=collector.collect(path)
            identity=report["cases"]["w0_split1"]["split1_identity_checks"]
            self.assertEqual(identity["saved_final_vs_bf16_partial0"]["status"],"COMPUTED_FROM_SAVED_ARRAYS")
            self.assertEqual(identity["partial0_vs_matching_math"]["status"],"NOT_AVAILABLE")

    def test_budget_and_case_filter_are_explicit_and_json_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path,_=self.fixture(directory,full_outputs=False)
            report=collector.collect(path,case_filter="w0_split1")
            report["bridge_build"]["huge_log"]="x"*100000
            encoded=collector.encode_bounded(report,4096)
            self.assertLessEqual(len(encoded.encode()),4096)
            parsed=json.loads(encoded)
            self.assertEqual(parsed["status"],"OUTPUT_BUDGET_EXCEEDED")
            self.assertFalse(parsed["npu_executed"])
            self.assertEqual(collector.portable(float("nan")),"nan")


if __name__=="__main__":
    unittest.main()
