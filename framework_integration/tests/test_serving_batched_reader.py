"""Host gates for additive reader ABI and fixed-shape device smoke inputs."""
import ast
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SINGLE = ROOT/"experimental/single_pass"
SERVING = ROOT/"framework_integration/sfa_recent128/serving_reader.py"
sys.path.insert(0,str(SINGLE))


class ServingBatchedReaderTests(unittest.TestCase):
    def test_host_abi_and_native_tnd_geometry(self):
        compiler = shutil.which("c++")
        self.assertIsNotNone(compiler)
        source = r'''
#include "batched_serving_api.h"
#include <cassert>
#include <type_traits>
int main(){
 static_assert(sizeof(BatchedServingBuffers)==160);
 static_assert(std::is_standard_layout<BatchedServingBuffers>::value);
 auto t=recent128_single_pass::MakeBatchedServingTiling(32,8,32768,32,16,.04f);
 assert(t.baseParams.batchSize==8 && t.baseParams.qSeqSize==32);
 assert(t.baseParams.maxBlockNumPerBatch==32 && t.baseParams.dSizeVInput==400);
 assert(t.baseParams.sparseBlockCount==2048 && t.baseParams.outputLayout==1);
 auto p=recent128_single_pass::MakeBatchedServingTiling(8192,8,32768,32,16,.04f);
 assert(p.baseParams.qSeqSize==8192);
 auto e=recent128_single_pass::MakeBatchedServingTiling(1,8,32768,32,1,.04f);
 assert(e.baseParams.batchSize==8);
 bool caught=false;
 try{recent128_single_pass::MakeBatchedServingTiling(32,8,32768,257,16,.04f);}
 catch(const std::invalid_argument&){caught=true;}
 assert(caught);
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary)/"abi"
            result = subprocess.run([compiler,"-std=c++17","-I",str(SINGLE),"-x","c++","-","-o",str(binary)],
                                    input=source,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            result = subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_read_contains_no_device_payload_host_reads_or_new_gpu_allocations(self):
        tree = ast.parse(SERVING.read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ServingReader")
        read = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="read")
        names = {n.func.attr for n in ast.walk(read) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(names & {"cpu","numpy","tolist","item","empty","zeros","ones","tensor","as_tensor","nonzero"})
        self.assertNotIn("reset_faults",names)
        self.assertNotIn("step_marker",ast.get_source_segment(SERVING.read_text(),read))

    def test_stream_change_needs_explicit_execution_reservation(self):
        spec = importlib.util.spec_from_file_location("serving_reader_stream_test", SERVING)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        current = SimpleNamespace(npu_stream=7)
        fake_torch = SimpleNamespace(npu=SimpleNamespace(current_stream=lambda device:current))
        query = SimpleNamespace(shape=(4,64,512),device="npu:0")
        cold = SimpleNamespace(shape=(16,128,1,400))
        table = SimpleNamespace(shape=(1,16))
        def reader(lane):
            value = module.ServingReader.__new__(module.ServingReader)
            value._execution_lane = lane
            value.prepared = {}
            return value
        with patch.dict(sys.modules,{"torch":fake_torch}):
            unbound = reader(None)
            original_lane = unbound._plan_lane(query.device)
            unbound.prepared[("npu:0",original_lane,4,1,16,16,1,256)] = object()
            self.assertTrue(unbound.is_prepared(query,cold,table))
            current.npu_stream = 99  # torch.npu.graph's implicit side stream
            self.assertFalse(unbound.is_prepared(query,cold,table))
            reservation = module.ServingExecutionLane()
            bound = reader(reservation)
            prepared_lane = bound._plan_lane(query.device)
            bound.prepared[("npu:0",prepared_lane,4,1,16,16,1,256)] = object()
            current.npu_stream = 123
            self.assertTrue(bound.is_prepared(query,cold,table))
            independent = reader(module.ServingExecutionLane())
            self.assertNotEqual(bound._plan_lane(query.device),independent._plan_lane(query.device))
            self.assertFalse(independent.is_prepared(query,cold,table))

    def test_where_uses_only_tensor_branches_and_smoke_captures_warmed_stream(self):
        tree = ast.parse(SERVING.read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ServingReader")
        read = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="read")
        calls = [n for n in ast.walk(read) if isinstance(n,ast.Call) and
                 isinstance(n.func,ast.Attribute) and n.func.attr=="where"]
        self.assertEqual(len(calls),2)
        for call in calls:
            self.assertTrue(all(not isinstance(arg,ast.Constant) for arg in call.args[1:3]))
        source = (SINGLE/"run_batched_smoke.py").read_text()
        smoke = ast.parse(source)
        capture = [n for n in ast.walk(smoke) if isinstance(n,ast.Call) and
                   isinstance(n.func,ast.Attribute) and n.func.attr=="graph"]
        self.assertEqual(len(capture),1)
        stream = next(k.value for k in capture[0].keywords if k.arg=="stream")
        self.assertIsInstance(stream,ast.Name)
        self.assertEqual(stream.id,"stream")

    def test_b8_graph_data_mutates_all_routing_metadata_and_preserves_causal_ids(self):
        import numpy as np
        path = SINGLE/"run_batched_smoke.py"
        spec = importlib.util.spec_from_file_location("batched_smoke_host",path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        a,_,_ = module.make_data(8,4,0,compute_reference=False)
        b,_,_ = module.make_data(8,4,1,compute_reference=False)
        for key in ("positions","banks","epochs","valid","qcum","kvlen","table"):
            self.assertFalse(np.array_equal(a[key],b[key]),key)
            self.assertEqual(a[key].shape,b[key].shape,key)
        for arrays in (a,b):
            self.assertEqual(int(arrays["valid"].sum()),int(arrays["qcum"][-1]))
            for row in np.flatnonzero(arrays["valid"]):
                ids=arrays["indices"][row,0]
                self.assertEqual(len(set(ids.tolist())),2048)
                self.assertTrue(np.all((ids>=0)&(ids<=arrays["positions"][row])))
            self.assertTrue(np.all(arrays["indices"][~arrays["valid"]]==-1))

    def test_faults_remain_latched_and_legacy_branch_is_compile_time_isolated(self):
        source=(SINGLE/"batched_serving_bridge.cpp").read_text()
        run=source.split("at::Tensor Run(",1)[1].split("at::Tensor Output()",1)[0]
        self.assertNotIn("faults_.zero_",run)
        header=(SINGLE/"arch35/kv_quant_sparse_flash_attention_service_vector_mla.h").read_text()
        self.assertIn("#if QSFA_RECENT_BATCHED",header)
        self.assertIn("runInfo.attentionOutOffset / (constInfo.gSize * constInfo.dSizeV)",header)
        self.assertIn("ReadServingScalar(recentEpochTagsGm, slot) == recentServingEpoch",header)
        self.assertIn("recentServingBank * recentRingCapacity",header)


if __name__=="__main__":unittest.main()
