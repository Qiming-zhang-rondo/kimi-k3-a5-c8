"""Host oracle/build contracts never stand in for an executed A5 producer."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"experimental/prolog_ring_smoke/run_smoke.py"
spec=importlib.util.spec_from_file_location("ring_smoke_harness",SOURCE)
smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke)


class PrologRingSmokeHarnessTest(unittest.TestCase):
    def test_b1_b8_banks_wrap_padding_and_round_ties(self):
        for pools in (1,8):
            case=smoke.make_case(pools)
            initial=smoke.blank_rings(pools,256)
            expected=smoke.expected_rings(case,initial)
            self.assertTrue(np.array_equal(initial["positionTags"],np.full((pools,256),-1)))
            self.assertEqual(np.count_nonzero(expected["positionTags"]>=0),int(case["tokenValid"].sum()))
            for row,valid in enumerate(case["tokenValid"]):
                if not valid:continue
                bank=int(case["tokenBanks"][row]);pos=int(case["tokenPositions"][row]);slot=pos%256
                self.assertEqual(expected["positionTags"][bank,slot],pos)
                self.assertEqual(expected["epochTags"][bank,slot],int(case["tokenEpochs"][row]))
                self.assertEqual(expected["latentRing"][bank,slot,0],1.)
                self.assertEqual(expected["latentRing"][bank,slot,1],1.015625)
                self.assertTrue(np.array_equal(expected["ropeRing"][bank,slot],case["inputRope"][row]))
            off=smoke.expected_rings(case,initial,False)
            self.assertTrue(all(np.array_equal(off[k],v) for k,v in initial.items()))

    def test_dynamic_phase_and_poison_reference_leave_only_active_slots_written(self):
        case=smoke.make_case(8,phase=1)
        initial=smoke.blank_rings(8,256)
        initial["latentRing"].fill(np.nan);initial["ropeRing"].fill(np.nan)
        initial["positionTags"].fill(-999);initial["epochTags"].fill(-999)
        expected=smoke.expected_rings(case,initial)
        written=expected["positionTags"]>=0
        self.assertEqual(int(written.sum()),31)
        self.assertTrue(np.isfinite(expected["latentRing"][written]).all())
        self.assertTrue(np.isnan(expected["latentRing"][~written]).all())
        self.assertTrue(np.isfinite(expected["ropeRing"][written]).all())
        self.assertTrue(np.isnan(expected["ropeRing"][~written]).all())
        self.assertTrue((expected["epochTags"][~written]==-999).all())
        case["tokenBanks"][1]=case["tokenBanks"][0]
        case["tokenPositions"][1]=case["tokenPositions"][0]
        with self.assertRaisesRegex(ValueError,"race"):
            smoke.expected_rings(case,initial)

    def test_host_cli_never_imports_torch_and_does_not_claim_producer_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/"host"
            guard="""import builtins,runpy,sys
old=builtins.__import__
def guarded(name,*a,**k):
 if name.split('.')[0] in {'torch','torch_npu'}:raise AssertionError('device import')
 return old(name,*a,**k)
builtins.__import__=guarded
source=sys.argv.pop(1); runpy.run_path(source,run_name='__main__')
"""
            result=subprocess.run([sys.executable,"-I","-c",guard,str(SOURCE),"--host-only","--output",str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((out/"results.json").read_text())
            self.assertEqual(report["status"],"HOST_REFERENCE_ONLY")
            self.assertEqual(report["scope"],"isolated_ring_writer_not_producer")
            self.assertFalse(report["prolog_validated"])
            self.assertFalse(report["c4_validated"])
            self.assertEqual(report["latency"],"NOT_MEASURED")
            again=subprocess.run([sys.executable,str(SOURCE),"--host-only","--output",str(out)],capture_output=True,text=True)
            self.assertNotEqual(again.returncode,0)

    def test_thin_bridge_uses_real_stream_local_dlopen_and_fixed_abi(self):
        text=smoke.BRIDGE_SOURCE
        for required in ("RTLD_NOW|RTLD_LOCAL","getCurrentNPUStream","NPUGuard","recordStream","sizeof(PrologRingSmokeBuffers)==80"):
            self.assertIn(required,text)
        for forbidden in ("ctypes","aclrtSynchronizeDevice",".cpu()",".item()"):
            self.assertNotIn(forbidden,text)
        cmake=(SOURCE.parent/"CMakeLists.txt").read_text()
        self.assertIn("LANGUAGES ASC CXX",cmake)
        self.assertIn("dav-3510",cmake)
        self.assertNotIn("install(",cmake)
        self.assertNotIn("op_impl",cmake)


if __name__=="__main__":unittest.main()
