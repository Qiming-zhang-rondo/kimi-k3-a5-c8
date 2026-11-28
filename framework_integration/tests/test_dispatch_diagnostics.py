"""Report actual worker state without importing a second framework instance."""
import builtins
from functools import lru_cache, wraps
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfa_recent128.dispatch_diagnostics import collect_dispatch_state


class DispatchDiagnosticTest(unittest.TestCase):
    def test_effective_true_and_raw_false_are_both_reported_without_changing_config(self):
        raw={"enable_dsa_cp":False}
        config=SimpleNamespace(enable_dsa_cp=True)
        @lru_cache(maxsize=1)
        def flag():return config.enable_dsa_cp
        self.assertTrue(flag())
        impl=SimpleNamespace(vllm_config=SimpleNamespace(additional_config=raw,
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1,decode_context_parallel_size=1)),tp_size=8)
        modules={"vllm_ascend.attention.context_parallel.sfa_cp":SimpleNamespace(enable_dsa_cp=flag),
            "vllm_ascend.ascend_config":SimpleNamespace(get_ascend_config=lambda:config)}
        original_import=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.startswith(("torch","vllm","ascend_vllm","cloud_ops_turbo")):
                raise AssertionError("Must use the actual already-loaded worker modules")
            return original_import(name,*args,**kwargs)
        with patch("builtins.__import__",guarded):report=collect_dispatch_state(impl,modules=modules)
        self.assertIs(report["raw_additional_enable_dsa_cp"],False)
        self.assertIs(report["probes"]["effective_enable_dsa_cp"],True)
        self.assertIs(report["probes"]["validated_ascend_enable_dsa_cp"],True)
        self.assertEqual(report["tp_size"],8)
        self.assertEqual((report["pcp_size"],report["dcp_size"]),(1,1))
        self.assertIs(raw["enable_dsa_cp"],False)
        self.assertTrue(config.enable_dsa_cp)
        self.assertEqual(flag.cache_info().currsize,1)

    def test_bound_override_and_source_identity_are_preserved_when_probes_fail(self):
        def base(self):return None
        @wraps(base)
        def wrapped(self):return base(self)
        class Impl:
            exec_kv=wrapped
        def broken():raise RuntimeError("singleton unavailable")
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"sfa_cp.py";source.write_text("# synthetic metadata source\n")
            modules={"vllm_ascend.attention.context_parallel.sfa_cp":SimpleNamespace(__file__=str(source),enable_dsa_cp=broken),
                "vllm_ascend.ascend_config":SimpleNamespace(get_ascend_config=broken)}
            report=collect_dispatch_state(Impl(),modules=modules)
        chain=report["bound_methods"]["exec_kv"]
        self.assertEqual(len(chain),2)
        self.assertTrue(all(item["file"]==__file__ for item in chain))
        self.assertIn("singleton unavailable",report["probes"]["effective_flag_error"])
        self.assertIn("singleton unavailable",report["probes"]["validated_config_error"])
        identity=report["modules"]["vllm_ascend.attention.context_parallel.sfa_cp"]
        self.assertEqual(len(identity["sha256"]),64)
        self.assertTrue(identity["loaded"])
        self.assertFalse(report["modules"]["vllm_ascend.attention.sfa_v1"]["loaded"])


if __name__=="__main__":unittest.main()
