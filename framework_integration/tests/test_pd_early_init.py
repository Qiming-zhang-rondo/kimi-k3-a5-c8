"""Arenas precede Store setup, including installation inside a live ctor."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfa_recent128 import pd_mooncake as pd


class EarlyInitializationTests(unittest.TestCase):
    def fixture(self, install_inside=False):
        calls=[]
        module=SimpleNamespace()
        vc=object()

        class Bridge:
            def __init__(self,worker):self.prepared=False;calls.append("bridge-created")
            def prepare_arena(self):
                if not self.prepared:calls.append("arena-registered");self.prepared=True

        class Worker:
            def __init__(self,config):
                self.vllm_config=config
                if install_inside:
                    pd.install_worker_hooks(module)
                calls.append("engine-ready")
            def register_kv_caches(self,caches):calls.append("native-register")
            def start_load_kv(self,metadata):pass

        class Facade:
            def __init__(self,config):
                calls.append("facade-entered")
                self.connector_worker=Worker(config)
                calls.append("facade-returned")

        class Send:
            def __init__(self,vllm_config):pass
            def run_busy_loop(self,socket):pass
            def get_and_clear_finished_requests(self):return set()

        class Receive:
            def __init__(self,vllm_config):pass
            def _transfer_kv_cache_all_groups(self,meta):pass
            def _reformat_pending_kv_caches(self,request_id):pass

        module.MooncakeConnectorWorker=Worker
        module.MooncakeConnector=Facade
        module.KVCacheSendingThread=Send
        module.KVCacheRecvingThread=Receive
        return module,vc,calls,Bridge

    def test_initial_install_inside_original_worker_constructor_is_early_enough(self):
        module,vc,calls,Bridge=self.fixture(install_inside=True)
        with patch.object(pd,"is_serving_config",return_value=True),patch.object(pd,"LegacyTailBridge",Bridge):
            connector=module.MooncakeConnector(vc)
            calls.append("store-sibling-setup")
            connector.connector_worker.register_kv_caches({})
        self.assertLess(calls.index("engine-ready"),calls.index("arena-registered"))
        self.assertLess(calls.index("arena-registered"),calls.index("facade-returned"))
        self.assertLess(calls.index("arena-registered"),calls.index("store-sibling-setup"))
        self.assertEqual(calls.count("bridge-created"),1)
        self.assertEqual(calls.count("arena-registered"),1)
        pd._BRIDGES.pop(id(vc),None)

    def test_already_installed_worker_ctor_registers_once_and_preserves_bridge(self):
        module,vc,calls,Bridge=self.fixture()
        with patch.object(pd,"is_serving_config",return_value=True),patch.object(pd,"LegacyTailBridge",Bridge):
            pd.install_worker_hooks(module)
            connector=module.MooncakeConnector(vc)
            first=connector.connector_worker._recent128_tail_bridge
            connector.connector_worker.register_kv_caches({})
        self.assertIs(connector.connector_worker._recent128_tail_bridge,first)
        self.assertEqual(calls.count("bridge-created"),1)
        self.assertEqual(calls.count("arena-registered"),1)
        self.assertLess(calls.index("arena-registered"),calls.index("facade-returned"))
        pd._BRIDGES.pop(id(vc),None)

    def test_off_worker_does_not_allocate_an_arena(self):
        module,vc,calls,Bridge=self.fixture()
        with patch.object(pd,"is_serving_config",return_value=False),patch.object(pd,"LegacyTailBridge",Bridge):
            pd.install_worker_hooks(module)
            connector=module.MooncakeConnector(vc)
            connector.connector_worker.register_kv_caches({})
        self.assertNotIn("bridge-created",calls)
        self.assertNotIn(id(vc),pd._BRIDGES)


if __name__=="__main__":unittest.main()
