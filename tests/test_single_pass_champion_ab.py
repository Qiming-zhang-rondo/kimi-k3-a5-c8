"""Focused host contracts for the real same-round champion A/B runner."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"experimental/single_pass"))
import run_champion_ab as ab
from run_single_pass import select_aic_count


class ChampionAbContractTest(unittest.TestCase):
    def test_shared_writer_candidate_never_calls_old_attention_and_scopes_stay_separate(self):
        trace = []
        class Champion:
            def _write(self):
                trace.append("shared_writer")
            def call(self):
                trace.append("old_two_attention_read")
                return "old"
            def run(self, *, write, dynamic):
                self.assertion = (write,dynamic)
                self._write()
                return self.call()
        champion = Champion()
        def new_read():
            trace.append("new_one_attention_read")
            return "new"
        scopes = ab.make_scope_calls(champion,new_read)
        self.assertEqual(scopes["prepared_read"]["single_pass"](),"new")
        self.assertEqual(trace,["new_one_attention_read"])
        trace.clear()
        scopes["write_read"]["single_pass"]()
        self.assertEqual(trace,["shared_writer","new_one_attention_read"])
        trace.clear()
        scopes["write_read"]["old_champion"]()
        self.assertEqual(trace,["shared_writer","old_two_attention_read"])
        self.assertEqual(champion.assertion,(True,True))
        self.assertTrue(ab.scope_metadata()["write_read"]["candidate_pays_unused_partition"])

    def test_no_timing_until_every_full_scope_control_passes_and_pairs_keep_original_rounds(self):
        calls = []
        scopes = {scope:{arm:lambda:None for arm in ab.ARMS} for scope in ab.SCOPES}
        args = SimpleNamespace(repeats=3)
        controls = {scope:{arm:{"passed":True} for arm in ab.ARMS} for scope in ab.SCOPES}
        def pair(*unused):
            calls.append(True)
            return {"graph":dict(status="MEASURED",rounds=[dict(round=i,order=["baseline","candidate"])
                                                           for i in range(args.repeats)])}
        controls["write_read"]["single_pass"]["passed"] = False
        with self.assertRaises(ab.ControlFailed):
            ab.measure_scopes(scopes,controls,args,None,pair)
        self.assertFalse(calls)
        controls["write_read"]["single_pass"]["passed"] = True
        result = ab.measure_scopes(scopes,controls,args,None,pair)
        self.assertEqual(len(calls),2)
        self.assertEqual(len(result["write_read"]["graph"]["rounds"]),3)

    def test_verified_old_abi4_rows16_and_q4_auto_aic4_are_explicit(self):
        ab.validate_champion_configuration(dict(abi_version=4,bf16_merge_tile_rows_effective=16))
        for config in (dict(abi_version=3,bf16_merge_tile_rows=16),
                       dict(abi_version=4,bf16_merge_tile_rows=4)):
            with self.assertRaises(ab.ControlFailed):
                ab.validate_champion_configuration(config)
        args = ab.parser().parse_args(["--payload-dir","/tmp/p","--single-pass-library","/tmp/new.so",
            "--champion-library","/tmp/libsfa_torch_direct.so","--device","npu:1","--output","/tmp/out"])
        self.assertEqual(args.timing,"none")
        self.assertEqual(select_aic_count(args.queries,args.aic_count),4)
        self.assertEqual(args.champion_merge_rows,16)


if __name__ == "__main__":
    unittest.main()
