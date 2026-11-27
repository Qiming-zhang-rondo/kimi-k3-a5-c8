"""Installer tests use isolated source trees; never import Torch/runtime hooks."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

SOURCE=Path(__file__).resolve().parents[1]/"install_overlay.py"
spec=importlib.util.spec_from_file_location("sfa_overlay_installer",SOURCE)
installer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
ARCHIVE=Path("/tmp/sfa-framework-20261125/mxllm-main")
PR199_ARCHIVE=Path("/tmp/mxllm-pr199-reviewed")
MINIMAL=b'''def _mxfp4_custom_kv_rmsnorm_rope(impl, kv_no_split, cos, sin):
    k_nope = normalize(kv_no_split)
    k_rope = rope(kv_no_split, cos, sin)

    prefix_shape = k_nope.shape[:-1]
    return quantize(k_nope, k_rope, prefix_shape)
'''
WORKER=b'from ascend_vllm.patch.worker import patch_sfa_v1_mxfp4\n'
PLATFORM=b'# Isolated platform bootstrap fixture; never execute framework code.\n'


class OverlayInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)
        self.root=self.directory/"mxllm"
        self.output=self.directory/"experiment"
        self.runtime=self.directory/"runtime"
        self.runtime.mkdir()
        (self.runtime/"__init__.py").write_text('raise AssertionError("installer imported runtime")\n')
        (self.runtime/"hooks.py").write_text('raise AssertionError("installer executed hooks")\n')
        (self.runtime/"config_bridge.py").write_text('raise AssertionError("installer executed config bridge")\n')
        (self.root/installer.PATCH).parent.mkdir(parents=True)
        (self.root/installer.PLATFORM).parent.mkdir(parents=True)

    def prepare(self, source=MINIMAL, worker=WORKER, platform=PLATFORM):
        (self.root/installer.PATCH).write_bytes(source)
        (self.root/installer.WORKER).write_bytes(worker)
        (self.root/installer.PLATFORM).write_bytes(platform)
        return dict(expected_source_sha256=installer.sha(source),expected_worker_sha256=installer.sha(worker),
                    expected_platform_sha256=installer.sha(platform))

    def run_install(self, *, apply=False, **kwargs):
        return installer.install_overlay(self.root,apply=apply,output=self.output,runtime_root=self.runtime,**kwargs)

    def verify_apply_and_idempotence(self, expected):
        original={str(name):(self.root/name).read_bytes() for name in (installer.PATCH,installer.WORKER,installer.PLATFORM)}
        checked=self.run_install(**expected)
        self.assertEqual(checked["status"],"READY")
        self.assertFalse((self.root/"sfa_recent128").exists())
        self.assertFalse(self.output.exists())
        self.assertEqual({n:(self.root/n).read_bytes() for n in original},original)
        applied=self.run_install(apply=True,**expected)
        self.assertEqual(applied["status"],"INSTALLED")
        self.assertFalse(applied["device_launched"])
        source=(self.root/installer.PATCH).read_text()
        self.assertEqual(source.count(installer.TAP),1)
        helper=source.split("def _mxfp4_custom_kv_rmsnorm_rope",1)[1]
        self.assertLess(helper.index("k_rope ="),helper.index("capture_native_kv(impl"))
        self.assertLess(helper.index("capture_native_kv(impl"),helper.index("prefix_shape ="))
        worker=(self.root/installer.WORKER).read_text()
        self.assertTrue(worker.endswith(installer.INSTALL))
        self.assertLess(worker.index("import patch_sfa_v1_mxfp4"),worker.index("from sfa_recent128.hooks import install"))
        platform=(self.root/installer.PLATFORM).read_text()
        self.assertTrue(platform.endswith(installer.PLATFORM_INSTALL))
        self.assertEqual(platform.count(installer.PLATFORM_INSTALL),1)
        self.assertIn("from sfa_recent128.config_bridge import install as _install_recent128_config_bridge",platform)
        backup=Path(applied["backup_directory"])
        for name,data in original.items():
            self.assertEqual((backup/"original"/name).read_bytes(),data)
        self.assertEqual(json.loads((backup/"manifest.json").read_text())["status"],"INSTALLED")
        before={n:(self.root/n).stat().st_mtime_ns for n in original}
        for apply in (False,True):
            repeat=self.run_install(apply=apply,**expected)
            self.assertEqual(repeat["status"],"ALREADY_INSTALLED")
            self.assertFalse(repeat["changed"])
        self.assertEqual({n:(self.root/n).stat().st_mtime_ns for n in original},before)
        self.assertEqual(len(list(self.output.iterdir())),1)

    def test_reviewed_minimal_source_check_apply_backup_and_idempotence(self):
        expected=self.prepare()
        self.verify_apply_and_idempotence(expected)

    def test_real_archived_source_uses_pinned_hashes_and_exact_tap_boundary(self):
        if not (ARCHIVE/installer.PATCH).is_file():
            self.skipTest("Pinned local mxllm archive unavailable; no framework/device test implied")
        source=(ARCHIVE/installer.PATCH).read_bytes()
        worker=(ARCHIVE/installer.WORKER).read_bytes()
        platform=(ARCHIVE/installer.PLATFORM).read_bytes()
        self.assertEqual(installer.sha(source),installer.PINNED_SOURCE_SHA256)
        self.assertEqual(installer.sha(worker),installer.PINNED_WORKER_SHA256)
        self.assertEqual(installer.sha(platform),installer.PINNED_PLATFORM_SHA256)
        self.prepare(source,worker,platform)
        self.verify_apply_and_idempotence({})

    def test_real_pr199_three_file_profile_check_apply_and_idempotence(self):
        paths=(installer.PATCH,installer.WORKER,installer.PLATFORM)
        if not all((PR199_ARCHIVE/path).is_file() for path in paths):
            self.skipTest("Reviewed PR199 raw source fixture unavailable; no network/framework/device test implied")
        contents=[(PR199_ARCHIVE/path).read_bytes() for path in paths]
        self.assertEqual(tuple(installer.sha(data) for data in contents),installer.SOURCE_PROFILES["pr199"])
        expected=self.prepare(*contents)
        self.verify_apply_and_idempotence(expected)

    def test_each_of_three_source_hash_mismatches_refuses_before_any_write(self):
        for name,flag in ((installer.PATCH,"--expected-source-sha256"),
                          (installer.WORKER,"--expected-worker-sha256"),
                          (installer.PLATFORM,"--expected-platform-sha256")):
            with self.subTest(source=str(name)):
                expected=self.prepare()
                key={installer.PATCH:"expected_source_sha256",installer.WORKER:"expected_worker_sha256",
                     installer.PLATFORM:"expected_platform_sha256"}[name]
                expected[key]="0"*64
                before={str(p):(self.root/p).read_bytes() for p in (installer.PATCH,installer.WORKER,installer.PLATFORM)}
                with self.assertRaisesRegex(ValueError,flag):
                    self.run_install(apply=True,**expected)
                self.assertEqual(before,{p:(self.root/p).read_bytes() for p in before})
                self.assertFalse(self.output.exists())
                self.assertFalse((self.root/"sfa_recent128").exists())

    def test_platform_syntax_error_refuses_before_backups_or_writes(self):
        expected=self.prepare(platform=b"def invalid(:\n")
        with self.assertRaises(SyntaxError):
            self.run_install(apply=True,**expected)
        self.assertEqual((self.root/installer.PATCH).read_bytes(),MINIMAL)
        self.assertEqual((self.root/installer.WORKER).read_bytes(),WORKER)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root/"sfa_recent128").exists())

    def test_third_file_write_failure_rolls_back_all_three_sources_and_runtime(self):
        expected=self.prepare()
        original={str(p):(self.root/p).read_bytes() for p in (installer.PATCH,installer.WORKER,installer.PLATFORM)}
        write=installer._write
        def fail_platform_once(path,data):
            if Path(path).resolve()==(self.root/installer.PLATFORM).resolve() and installer.PLATFORM_INSTALL.encode() in data:
                raise OSError("injected third source failure")
            write(path,data)
        with mock.patch.object(installer,"_write",side_effect=fail_platform_once):
            with self.assertRaisesRegex(OSError,"third source failure"):
                self.run_install(apply=True,**expected)
        self.assertEqual({p:(self.root/p).read_bytes() for p in original},original)
        self.assertFalse((self.root/"sfa_recent128").exists())
        backups=list(self.output.iterdir())
        self.assertEqual(len(backups),1)
        manifest=json.loads((backups[0]/"manifest.json").read_text())
        self.assertEqual(manifest["status"],"ROLLED_BACK")
        for name,data in original.items():
            self.assertEqual((backups[0]/"original"/name).read_bytes(),data)

    def test_old_two_file_overlay_requires_explicit_restore_then_reinstall(self):
        expected=self.prepare()
        self.run_install(apply=True,**expected)
        platform=self.root/installer.PLATFORM
        platform.write_bytes(PLATFORM)
        state_path=self.root/"sfa_recent128"/installer.STATE
        state=json.loads(state_path.read_text())
        for key in ("original_sha256","installed_source_sha256"):
            state[key].pop(str(installer.PLATFORM))
        state_path.write_text(json.dumps(state))
        before={str(p):(self.root/p).read_bytes() for p in (installer.PATCH,installer.WORKER,installer.PLATFORM)}
        for apply in (False,True):
            with self.assertRaisesRegex(ValueError,"two-file overlay.*restore.*reinstall"):
                self.run_install(apply=apply,**expected)
        self.assertEqual({p:(self.root/p).read_bytes() for p in before},before)
        self.assertEqual(len(list(self.output.iterdir())),1)

    def test_sha_mismatch_or_bad_boundary_refuses_before_any_write(self):
        expected=self.prepare()
        with self.assertRaisesRegex(ValueError,"actual=.*--expected-source-sha256"):
            self.run_install(apply=True)
        self.assertEqual((self.root/installer.PATCH).read_bytes(),MINIMAL)
        self.assertFalse(self.output.exists())
        changed=MINIMAL.replace(b"    prefix_shape",b"    another_op()\n    prefix_shape")
        expected=self.prepare(changed)
        with self.assertRaisesRegex(ValueError,"insertion boundary"):
            self.run_install(apply=True,**expected)
        self.assertFalse((self.root/"sfa_recent128").exists())
        self.assertFalse(self.output.exists())

    def test_altered_installed_source_or_runtime_is_not_silently_overwritten(self):
        expected=self.prepare()
        self.run_install(apply=True,**expected)
        path=self.root/installer.PATCH
        saved=path.read_bytes()
        path.write_bytes(saved.replace(installer.TAP.encode(),installer.TAP.encode()*2))
        with self.assertRaisesRegex(ValueError,"duplicated"):
            self.run_install(apply=True,**expected)
        path.write_bytes(saved)
        runtime=self.root/"sfa_recent128/hooks.py"
        runtime.write_text("# unexpected edited runtime\n")
        with self.assertRaisesRegex(ValueError,"differs from the reviewed manifest"):
            self.run_install(apply=True,**expected)
        self.assertEqual(runtime.read_text(),"# unexpected edited runtime\n")
        self.assertEqual(len(list(self.output.iterdir())),1)

    def test_altered_installed_platform_is_not_silently_overwritten(self):
        expected=self.prepare()
        self.run_install(apply=True,**expected)
        platform=self.root/installer.PLATFORM
        changed=platform.read_bytes()+b"# unreviewed platform change\n"
        platform.write_bytes(changed)
        with self.assertRaisesRegex(ValueError,"differs from the reviewed manifest"):
            self.run_install(apply=True,**expected)
        self.assertEqual(platform.read_bytes(),changed)
        self.assertEqual(len(list(self.output.iterdir())),1)


if __name__=="__main__":unittest.main()
