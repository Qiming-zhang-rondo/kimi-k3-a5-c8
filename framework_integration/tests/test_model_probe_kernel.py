"""Probe must request the measured entry explicitly without altering OFF/native."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("model_probe_kernel",ROOT/"run_model_probe.py")
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ModelProbeKernelTest(unittest.TestCase):
    def test_recent128_selection_preserves_existing_indexer_mxfp4_config(self):
        additional={"indexer_kv_dtype":"mxfp4","vendor_options":{"retain":True}}
        options=dict(model="actual/checkpoint",enforce_eager=True,max_num_seqs=1,
            enable_prefix_caching=False,attention_config={"mla_kv_quant_mode":"mxfp4"},
            enable_chunked_prefill=True,max_num_batched_tokens=128,additional_config=additional)
        for arm in ("off","single_pass"):
            value=probe.engine_options(options,arm=arm,source_root=Path("/operator-source"),
                library_dir=Path("/verified-lib"),diagnostics=Path("/experiment/layers"),
                max_query_chunk=128,single_pass_kernel="kv_split4")
            self.assertEqual(value["additional_config"]["indexer_kv_dtype"],"mxfp4")
            self.assertEqual(value["additional_config"]["vendor_options"],{"retain":True})
            self.assertEqual(value["additional_config"]["sfa_recent128"]["window"],0 if arm=="off" else 128)
        self.assertEqual(additional,{"indexer_kv_dtype":"mxfp4","vendor_options":{"retain":True}})

    def test_single_pass_defaults_load_tile4_and_other_arms_keep_configuration(self):
        options=dict(model="actual/checkpoint",enforce_eager=True,max_num_seqs=1,
            enable_prefix_caching=False,attention_config={"mla_kv_quant_mode":"mxfp4"},
            enable_chunked_prefill=True,max_num_batched_tokens=128)
        common=dict(source_root=Path("/source"),library_dir=Path("/lib"),
                    diagnostics=Path("/results"),max_query_chunk=128)
        for kernel in ("load_tile4","fast_pair","overlay","kv_split4"):
            config=probe.engine_options(options,arm="single_pass",**common,
                single_pass_kernel=kernel)["additional_config"]["sfa_recent128"]
            self.assertEqual(config["single_pass_kernel"],kernel)
            self.assertEqual(config["reader"],"single_pass")
        default=probe.engine_options(options,arm="single_pass",**common)
        self.assertEqual(default["additional_config"]["sfa_recent128"]["single_pass_kernel"],"load_tile4")
        for arm in ("off","native","fused"):
            config=probe.engine_options(options,arm=arm,**common)["additional_config"]["sfa_recent128"]
            self.assertNotIn("single_pass_kernel",config)
        self.assertNotIn("additional_config",options)

    def test_split_kernel_is_explicit_and_stays_accuracy_only(self):
        options=dict(model="actual/checkpoint",enforce_eager=True,max_num_seqs=1,
            enable_prefix_caching=False,attention_config={"mla_kv_quant_mode":"mxfp4"},
            enable_chunked_prefill=True,max_num_batched_tokens=128)
        common=dict(source_root=Path("/source"),library_dir=Path("/lib"),
            diagnostics=Path("/results"),max_query_chunk=128)
        value=probe.engine_options(options,arm="single_pass",single_pass_kernel="kv_split4",**common)
        config=value["additional_config"]["sfa_recent128"]
        self.assertEqual(config["single_pass_kernel"],"kv_split4")
        self.assertEqual(config["execution"],"accuracy_eager")
        self.assertEqual(config["preprocess"],"preserve")
        with self.assertRaises(ValueError):
            probe.engine_options(dict(options,enforce_eager=False),arm="single_pass",
                single_pass_kernel="kv_split4",**common)


if __name__=="__main__":unittest.main()
