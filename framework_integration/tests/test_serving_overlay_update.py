"""Managed runtime upgrades preserve all three reviewed framework source files."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE=Path(__file__).resolve().parents[1]/"install_overlay.py"
spec=importlib.util.spec_from_file_location("serving_runtime_installer",SOURCE)
installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)
NORMALIZE=b'''def _mxfp4_custom_kv_rmsnorm_rope(impl, x, cos, sin):
    k_nope = normalize(x)
    k_rope = rope(x, cos, sin)
    prefix_shape = k_nope.shape[:-1]
    return quantize(k_nope, k_rope, prefix_shape)
'''
WORKER=b'from ascend_vllm.patch.worker import patch_sfa_v1_mxfp4\n'
PLATFORM=b'# platform bootstrap fixture\n'


class ServingOverlayUpdateTest(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.base=Path(self.directory.name);self.root=self.base/"mxllm";self.runtime=self.base/"runtime"
        self.runtime.mkdir()
        for name in ("__init__.py","hooks.py","config_bridge.py"):
            (self.runtime/name).write_text("# old runtime - never import\n")
        self.expected={}
        for relative,contents,key in ((installer.PATCH,NORMALIZE,"expected_source_sha256"),
                (installer.WORKER,WORKER,"expected_worker_sha256"),(installer.PLATFORM,PLATFORM,"expected_platform_sha256")):
            path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(contents)
            self.expected[key]=installer.sha(contents)
        result=self.install(apply=True,output=self.base/"first")
        self.assertEqual(result["status"],"INSTALLED")
        self.original_sources={str(name):(self.root/name).read_bytes() for name in (installer.PATCH,installer.WORKER,installer.PLATFORM)}
        self.original_times={str(name):(self.root/name).stat().st_mtime_ns for name in (installer.PATCH,installer.WORKER,installer.PLATFORM)}
        self.old_hooks=(self.root/"sfa_recent128/hooks.py").read_bytes()
        (self.runtime/"hooks.py").write_text("# new serving-aware hooks - never import\n")
        (self.runtime/"serving_runtime.py").write_text("raise AssertionError('installer must not execute runtime')\n")

    def install(self,**kwargs):
        return installer.install_overlay(self.root,runtime_root=self.runtime,**self.expected,**kwargs)

    def test_runtime_update_backs_up_then_changes_only_managed_package(self):
        with self.assertRaisesRegex(ValueError,"newer runtime.*update-runtime"):
            self.install()
        ready=self.install(update_runtime=True)
        self.assertEqual(ready["status"],"RUNTIME_UPDATE_READY")
        self.assertEqual((self.root/"sfa_recent128/hooks.py").read_bytes(),self.old_hooks)
        updated=self.install(apply=True,update_runtime=True,output=self.base/"upgrade")
        self.assertEqual(updated["status"],"RUNTIME_UPDATED")
        self.assertFalse(updated["source_changed"]);self.assertFalse(updated["device_launched"])
        self.assertEqual({name:(self.root/name).read_bytes() for name in self.original_sources},self.original_sources)
        self.assertEqual({name:(self.root/name).stat().st_mtime_ns for name in self.original_sources},self.original_times)
        backup=Path(updated["backup_directory"])
        self.assertEqual((backup/"runtime/hooks.py").read_bytes(),self.old_hooks)
        self.assertTrue((backup/"runtime"/installer.STATE).is_file())
        self.assertEqual((self.root/"sfa_recent128/hooks.py").read_bytes(),(self.runtime/"hooks.py").read_bytes())
        self.assertTrue((self.root/"sfa_recent128/serving_runtime.py").is_file())
        self.assertEqual(json.loads((backup/"manifest.json").read_text())["status"],"RUNTIME_UPDATED")
        self.assertEqual(self.install()["status"],"ALREADY_INSTALLED")

    def test_update_flag_does_not_waive_tampered_runtime_or_framework_source(self):
        target=self.root/"sfa_recent128/hooks.py"
        target.write_text("# unreviewed local edit\n")
        with self.assertRaisesRegex(ValueError,"differs from the reviewed manifest"):
            self.install(apply=True,update_runtime=True,output=self.base/"rejected")
        self.assertFalse((self.base/"rejected").exists())
        self.assertEqual(target.read_text(),"# unreviewed local edit\n")
        target.write_bytes(self.old_hooks)
        worker=self.root/installer.WORKER
        original=worker.read_bytes();worker.write_bytes(original+b"# unreviewed source change\n")
        with self.assertRaisesRegex(ValueError,"differs from the reviewed manifest"):
            self.install(apply=True,update_runtime=True,output=self.base/"rejected")
        self.assertEqual(worker.read_bytes(),original+b"# unreviewed source change\n")
        self.assertFalse((self.base/"rejected").exists())


if __name__=="__main__":unittest.main()
