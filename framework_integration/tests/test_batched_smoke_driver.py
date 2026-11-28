"""Execute the smoke lifecycle helpers with an inference-mode-sensitive fake."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[2]/"experimental/single_pass/run_batched_smoke.py"


def load_driver():
    spec = importlib.util.spec_from_file_location("batched_smoke_driver_test",SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BatchedSmokeDriverTests(unittest.TestCase):
    def test_default_and_targeted_selection_are_explicit(self):
        driver = load_driver()
        self.assertEqual(driver.selected_cases(),((1,1),(1,4),(8,4)))
        self.assertEqual(driver.selected_cases(["b8_q4"]),((8,4),))
        for names in ([],["b8_q4","b8_q4"],["b8_q1"]):
            with self.assertRaises(ValueError):driver.selected_cases(names)

    def test_negative_control_mutations_and_drain_own_inference_mode(self):
        import numpy as np
        driver = load_driver()
        state = SimpleNamespace(inference=False,fault=False,calls=0,resets=0)
        checks = []
        def require_mode(operation):
            checks.append(operation)
            if not state.inference:raise RuntimeError("Inference tensor mutated outside InferenceMode: "+operation)
        class Mode:
            def __enter__(self):
                self.old = state.inference
                state.inference = True
            def __exit__(self,*args):state.inference = self.old
        class Buffer:
            def __init__(self,value):self.value=np.array(value,copy=True)
            def copy_(self,value):
                require_mode("copy")
                self.value[...] = value
            def __setitem__(self,index,value):
                require_mode("epoch_update")
                self.value[index] = value
        data = dict(positions=np.array([256],np.int64),banks=np.array([0],np.int32),
                    epochs=np.array([7],np.int64),epoch_tags=np.full((1,256),7,np.int64),
                    cold=np.zeros((1,128,1,400),np.uint8))
        buffers = {key:Buffer(value) for key,value in data.items()}
        class Reader:
            def fault_buffers(self):
                return (SimpleNamespace(cpu=lambda:SimpleNamespace(any=lambda:state.fault)),)
            def reset_faults(self):
                require_mode("reset_faults")
                state.resets+=1
                state.fault=False
        def call():
            require_mode("reader_call")
            state.calls+=1
            state.fault |= buffers["epoch_tags"].value[0,0] != 7
        def as_tensor(value,**kwargs):
            require_mode("input_restore")
            return value
        torch = SimpleNamespace(inference_mode=Mode,as_tensor=as_tensor,
            npu=SimpleNamespace(synchronize=lambda:require_mode("synchronize")))
        result=driver.stale_epoch_control(torch,Reader(),buffers,data,call=call,
                                         device="npu:0",dtypes={key:key for key in data})
        self.assertEqual(result,dict(passed=True,detected=True,retained_after_repair=True))
        self.assertEqual(state.calls,2)
        self.assertEqual(state.resets,1)
        self.assertFalse(state.inference)
        self.assertFalse(state.fault)
        self.assertIn("epoch_update",checks)
        self.assertIn("reset_faults",checks)

    def test_b1_q1_second_graph_phase_is_all_invalid(self):
        driver=load_driver()
        a,_,_=driver.make_data(1,1,0,compute_reference=False)
        b,_,_=driver.make_data(1,1,1,compute_reference=False)
        self.assertEqual(a["qcum"].tolist(),[1])
        self.assertEqual(b["qcum"].tolist(),[0])
        self.assertEqual(b["valid"].tolist(),[False])
        self.assertTrue((b["indices"]==-1).all())
        self.assertEqual(b["positions"].tolist(),[-1])


if __name__=="__main__":unittest.main()
