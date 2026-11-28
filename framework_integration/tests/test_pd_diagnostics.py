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
        self.assertIn("batch_transfer_sync_read",report["children"]["connector_worker.engine"]["methods"])
        rendered=json.dumps(report)
        self.assertNotIn("do-not-report-this",rendered)
        self.assertNotIn("private cached kernel source",rendered)
        self.assertIn("not GPU kernel execution or Graph replay",rendered)

    def test_multi_connector_recurses_indexed_children_and_stops_cycles(self):
        transfer,runtime=self.fixtures()
        leaf=Connector();nested=SimpleNamespace(_connectors=[leaf])
        root=SimpleNamespace(_connectors=[nested,SimpleNamespace()])
        leaf.connector_worker.worker=root
        transfer.get_kv_transfer_group=lambda:root
        report=diagnostics.inspect_live_pd(modules={"vllm.distributed.kv_transfer":transfer})
        children=report["children"]
        self.assertEqual(children["_connectors[0]._connectors[0].connector_worker"]["qualname"],"Worker")
        self.assertIn("batch_transfer_sync_read",children["_connectors[0]._connectors[0].connector_worker.engine"]["methods"])
        self.assertEqual(children["_connectors[0]._connectors[0].connector_worker.worker"],dict(alias_of="connector"))
        self.assertIn("_connectors[1]",children)

    def test_factory_resolves_nested_configs_without_instantiating_connectors(self):
        from probe_pd_connector import resolve_factory_tree
        class Multi:
            def __init__(self,*args,**kwargs):raise AssertionError("must not construct")
            @classmethod
            def _get_connector_classes_and_configs(cls,*args):raise AssertionError("not called")
        class Leaf:
            def __init__(self,*args,**kwargs):raise AssertionError("must not construct")
        configs=[]
        def config_type(**options):
            configs.append(options)
            return SimpleNamespace(engine_id=options.get("engine_id"),kv_connector_extra_config=options.get("kv_connector_extra_config",{}),name=options["kv_connector"])
        factory=SimpleNamespace(get_connector_class=lambda config:Multi if config.name=="Multi" else Leaf)
        result=resolve_factory_tree(dict(kv_connector="Multi",engine_id="private-engine",kv_connector_extra_config=dict(
            connectors=[dict(kv_connector="Multi",kv_connector_extra_config=dict(connectors=[dict(kv_connector="MooncakeConnectorV1")])),dict(kv_connector="AscendStoreConnector")])),config_type,factory)
        self.assertEqual(result["children"][0]["index"],0)
        self.assertEqual(result["children"][1]["requested"]["kv_connector"],"AscendStoreConnector")
        self.assertEqual(result["children"][0]["children"][0]["requested"]["kv_connector"],"MooncakeConnectorV1")
        self.assertTrue(all(config["engine_id"]=="private-engine" for config in configs))
        self.assertNotIn("private-engine",json.dumps(result))

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
