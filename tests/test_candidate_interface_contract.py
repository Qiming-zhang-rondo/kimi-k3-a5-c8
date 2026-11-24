"""Cross-file ABI checks; these do not compile Torch/AscendC or execute an NPU."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experimental"


def normalize(name):
    return name.replace("_", "").lower()


class CandidateInterfaceTests(unittest.TestCase):
    def test_proto_public_api_and_kernel_have_same_fourteen_input_order(self):
        proto = (EXP / "op_proto/sparse_flash_attention_cold_merge_proto.h").read_text()
        api = (EXP / "op_api/include/aclnn_sparse_flash_attention_cold_merge.h").read_text()
        entry = (EXP / "sparse_flash_attention_cold_merge.cpp").read_text()
        proto_names = re.findall(r"\.INPUT\((\w+),", proto)
        api_body = api.split("aclnnSparseFlashAttentionColdMergeGetWorkspaceSize(", 1)[1].split(");", 1)[0]
        api_names = re.findall(r"const aclTensor\s*\*(\w+)", api_body)[:14]
        kernel_body = entry.split("void sparse_flash_attention_cold_merge(", 1)[1].split(")", 1)[0]
        kernel_names = re.findall(r"__gm__ uint8_t\s*\*(\w+)", kernel_body)[:14]
        expected = ["query", "key", "value", "sparseindices", "blocktable",
                    "actualseqlengthsquery", "actualseqlengthskv", "queryrope", "keyrope",
                    "coldoutput", "coldmax", "coldsum", "coldpresent", "hotpresent"]
        for names in (proto_names, api_names, kernel_names):
            self.assertEqual([normalize(name) for name in names], expected)

    def test_both_core_init_paths_propagate_cold_inputs(self):
        entry = (EXP / "sparse_flash_attention_cold_merge.cpp").read_text()
        calls = re.findall(r"op\.Init\((.*?)\);", entry, re.S)
        self.assertEqual(len(calls), 2)
        cold = "coldOutput, coldMax, coldSum, coldPresent, hotPresent"
        for call in calls:
            self.assertIn(cold, call)
            self.assertLess(call.index("softmaxSum"), call.index("coldOutput"))
            self.assertLess(call.index("hotPresent"), call.index("user"))
        kernel = (EXP / "arch35/sparse_flash_attention_kernel_mla.h").read_text()
        self.assertIn("vecBlock.InitColdStateBuffers("+cold+");", kernel)

    def test_merge_is_after_hot_cleanup_and_final_store_stays_fp32(self):
        vector = (EXP / "arch35/sparse_flash_attention_service_vector_mla.h").read_text()
        process = vector.split("::ProcessVec2(", 1)[1].split("::Bmm2DataCopyOut", 1)[0]
        self.assertLess(process.index("ZeroInvalidAttnOut("), process.index("MergeColdStateDevice("))
        self.assertLess(process.index("MergeColdStateDevice("), process.index("CopyOutAttentionOut("))
        store = vector.split("::Bmm2DataCopyOut", 1)[1].split("::CopyOutAttentionOut", 1)[0]
        default_store = store.split("#else", 1)[1].split("#endif", 1)[0]
        self.assertNotRegex(default_store, r"\bCast\s*\(")
        self.assertIn("IsSameType<OUTPUT_T, float>::value", default_store)
        entry = (EXP / "sparse_flash_attention_cold_merge.cpp").read_text()
        self.assertIn("bfloat16_t, bfloat16_t, float, float,", entry)

    def test_bridge_calls_correct_original_and_distinct_fused_api(self):
        bridge = (EXP / "torch/sfa_cold_merge_bridge.cpp").read_text()
        self.assertIn("ACLNN_CMD(aclnnSparseFlashAttention,", bridge)
        self.assertIn("ACLNN_CMD(aclnnSparseFlashAttentionColdMerge,", bridge)
        self.assertNotIn("aclnnSparseFlashMla", bridge)
        fused = bridge.split("State FusedHot(", 1)[1]
        self.assertIn("TORCH_CHECK(!hotStats", fused)
        self.assertIn('Check(coldOutput, query, {4, 64, 512}, at::kFloat', fused)
        self.assertIn('Check(coldMax, query, {4, 64, 1}, at::kFloat', fused)


if __name__ == "__main__":
    unittest.main()
