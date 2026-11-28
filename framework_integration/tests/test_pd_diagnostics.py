"""PD identity probes use existing objects and redact method default values."""
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfa_recent128 import pd_diagnostics as diagnostics


class Engine:
    def register_memory(self,address,length,secret="do-not-report-this"):pass
    def batch_transfer_sync_read(self,session,source,target,lengths):pass


class Worker:
    def __init__(self):self.engine=Engine()


class Connector:
    def __init__(self):self.connector_worker=Worker()


class Layer:
    layer_key="model.layers.0"
    producer_kind="native"
    producer_calls=3
    read_calls=3
    report=Path("/local/serving-previous-flush.json")
    native_controller=SimpleNamespace(_plans={"plan":SimpleNamespace(writer=SimpleNamespace(
        warmed=True,_kernel=SimpleNamespace(src="private cached kernel source")))})
    def read(self):pass


class PDDiagnosticsTest(unittest.TestCase):
    def fixtures(self):
        transfer=ModuleType("vllm.distributed.kv_transfer")
        transfer.get_kv_transfer_group=lambda:Connector()
        runtime=ModuleType("sfa_recent128.serving_runtime")
        runtime._STATES={"state":SimpleNamespace(device="npu:0",runtime_layers=[Layer()])}
        return transfer,runtime

    def test_reports_current_worker_and_loaded_kernel_hash_without_payload_or_defaults(self):
        transfer,runtime=self.fixtures()
        report=diagnostics.inspect_live_pd(modules={"vllm.distributed.kv_transfer":transfer,
            "sfa_recent128.serving_runtime":runtime})
        self.assertEqual(report["runtime"]["worker_pid"],os.getpid())
        self.assertEqual(report["runtime"]["layers"][0]["producer_calls"],3)
        self.assertEqual(report["runtime"]["layers"][0]["warmed_native_plan_count"],1)
        self.assertEqual(report["connector"]["qualname"],"Connector")
        self.assertEqual(report["children"]["connector_worker"]["qualname"],"Worker")
        self.assertIn("batch_transfer_sync_read",report["children"]["Worker.engine"]["methods"])
        rendered=json.dumps(report)
        self.assertNotIn("do-not-report-this",rendered)
        self.assertNotIn("private cached kernel source",rendered)
        self.assertIn("not GPU kernel execution or Graph replay",rendered)

    def test_empty_process_is_explicit_and_does_not_create_framework(self):
        report=diagnostics.inspect_live_pd(modules={})
        self.assertEqual(report["status"],"NO_LIVE_CONNECTOR")
        self.assertEqual(report["runtime"]["status"],"NO_EXISTING_WORKER_STATE")
        self.assertEqual(report["device_operations"],0)

    def test_explicit_worker_verification_calls_existing_flush_only(self):
        transfer,runtime=self.fixtures();calls=[]
        runtime.validate_serving_faults=lambda:calls.append("post-request-flush") or dict(status="NO_LATCHED_READER_FAULTS")
        with patch.dict(sys.modules,{"vllm.distributed.kv_transfer":transfer,"sfa_recent128.serving_runtime":runtime}):
            report=diagnostics.verify_worker_recent128(object())
        self.assertEqual(calls,["post-request-flush"])
        self.assertEqual(report["validation"]["status"],"NO_LATCHED_READER_FAULTS")
        self.assertEqual(report["runtime"]["worker_pid"],os.getpid())


if __name__=="__main__":unittest.main()
