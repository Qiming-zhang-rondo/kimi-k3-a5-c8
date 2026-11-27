"""VA/CANN inventory stays read-only and never imports device frameworks."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE=Path(__file__).resolve().parents[1]/"collect_va_environment.py"


class VaEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location("va_environment_under_test",MODULE)
        cls.collector=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=cls.collector
        spec.loader.exec_module(cls.collector)

    def restricted_environment(self,root,*,git=None,ascend=None):
        binary_dir=Path(root)/"allowed_bin"
        binary_dir.mkdir(exist_ok=True)
        if git is not None:
            (binary_dir/"git").symlink_to(git)
        value={"PATH":str(binary_dir),"VA_TEST_PRIVATE_TOKEN":"VA_SECRET_SENTINEL_NOT_FOR_OUTPUT",
               "GITHUB_TOKEN":"GITHUB_SENTINEL_NOT_FOR_OUTPUT"}
        if ascend is not None:value["ASCEND_HOME_PATH"]=str(ascend)
        return value

    def test_actual_git_head_and_dirty_are_observed_without_changing_source(self):
        git=shutil.which("git")
        if git is None:self.skipTest("Git executable not available")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);repo=root/"源代码"
            def command(*arguments):
                process=subprocess.run([git,*arguments],capture_output=True,text=True)
                self.assertEqual(process.returncode,0,process.stderr)
                return process.stdout.strip()
            command("-c","init.defaultBranch=main","init","--quiet",str(repo))
            tracked=repo/"tracked.txt";tracked.write_text("original\n")
            command("-C",str(repo),"add","tracked.txt")
            command("-C",str(repo),"-c","user.name=VA fixture","-c","user.email=fixture@example.invalid",
                "-c","commit.gpgsign=false","-c","core.hooksPath=/dev/null","commit","--quiet","-m","fixture")
            head=command("-C",str(repo),"rev-parse","HEAD")
            env=self.restricted_environment(root,git=git)
            with patch.dict(os.environ,env,clear=True):
                clean=self.collector.collect_environment(source_roots=(str(repo),))
                candidates=[item for item in clean["git_repositories"] if Path(item["root"]).resolve()==repo.resolve()]
                self.assertEqual(len(candidates),1)
                self.assertEqual(candidates[0]["head"],head)
                self.assertFalse(candidates[0]["dirty"])
                tracked.write_text("changed\n")
                dirty=self.collector.collect_environment(source_roots=(str(repo),))
            candidates=[item for item in dirty["git_repositories"] if Path(item["root"]).resolve()==repo.resolve()]
            self.assertEqual(candidates[0]["head"],head)
            self.assertTrue(candidates[0]["dirty"])
            self.assertEqual(tracked.read_text(),"changed\n")

    def test_symlink_toolkit_and_custom_operator_versions_remain_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            toolkit=root/"Ascend"/"cann-9.2.1";toolkit.mkdir(parents=True)
            toolkit_version=toolkit/"version.info";toolkit_version.write_text("Version=9.2.1\n")
            latest=root/"Ascend"/"latest";latest.symlink_to(toolkit,target_is_directory=True)
            custom=root/"cloudops"/"cann_pkg"/"vendors"/"custom_cot"/"version.info"
            custom.parent.mkdir(parents=True);custom.write_text("Version=9.1.0\ncommitid=f6e7b882\n")
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (toolkit_version,custom)}
            with patch.dict(os.environ,self.restricted_environment(root,ascend=latest),clear=True):
                report=self.collector.collect_environment(extra_version_files=(str(custom),))
            paths=[item for item in report["ascend_paths"] if Path(item["path"])==latest]
            self.assertTrue(paths)
            self.assertTrue(paths[0]["is_symlink"])
            self.assertEqual(Path(paths[0]["real_path"]),toolkit.resolve())
            records={Path(item["real_path"]):item for item in report["version_files"] if item.get("exists")}
            self.assertIn(toolkit_version.resolve(),records)
            self.assertIn(custom.resolve(),records)
            self.assertEqual(records[toolkit_version.resolve()]["kind"],"toolkit")
            self.assertEqual(records[custom.resolve()]["kind"],"custom_cot")
            self.assertIn("9.2.1",records[toolkit_version.resolve()]["content"])
            self.assertIn("9.1.0",records[custom.resolve()]["content"])
            self.assertEqual(report["compatibility"],"NOT_VERIFIED")
            self.assertEqual(before,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (toolkit_version,custom)})

    def test_unknown_versions_do_not_claim_compatibility_or_dump_private_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.dict(os.environ,self.restricted_environment(root,ascend=root/"missing"),clear=True):
                report=self.collector.collect_environment(source_roots=(str(root/"absent_source"),))
            self.assertEqual(report["status"],"ENVIRONMENT_CAPTURED")
            self.assertEqual(report["compatibility"],"NOT_VERIFIED")
            self.assertLessEqual(len(report["ascend_environment"]),4)
            encoded=json.dumps(report,ensure_ascii=False)
            for secret in ("VA_TEST_PRIVATE_TOKEN","VA_SECRET_SENTINEL_NOT_FOR_OUTPUT",
                           "GITHUB_TOKEN","GITHUB_SENTINEL_NOT_FOR_OUTPUT"):
                self.assertNotIn(secret,encoded)

    def test_stdlib_isolated_collect_never_imports_numpy_torch_or_torch_npu(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            script=r'''
import importlib.util,json,pathlib,sys
spec=importlib.util.spec_from_file_location("stdlib_va_environment",sys.argv[1])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
report=module.collect_environment(source_roots=(sys.argv[2],),extra_version_files=())
blocked=("torch","torch_npu","numpy")
assert not any(name==item or name.startswith(item+".") for name in sys.modules for item in blocked)
assert report["compatibility"]=="NOT_VERIFIED"
print(json.dumps({"status":report["status"],"frameworks_imported":False}))
'''
            result=subprocess.run([sys.executable,"-I","-S","-c",script,str(MODULE),str(root/"absent")],
                env=self.restricted_environment(root),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            value=json.loads(result.stdout)
            self.assertEqual(value["status"],"ENVIRONMENT_CAPTURED")
            self.assertFalse(value["frameworks_imported"])

    def test_cli_runs_without_site_packages_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/"environment.json"
            version=root/"unknown_version.info";version.write_text("Version=UNKNOWN\n")
            environment=self.restricted_environment(root)
            arguments=[sys.executable,"-I","-S",str(MODULE),"--output",str(output),
                "--source-root",str(root/"missing_source"),"--extra-version-file",str(version)]
            first=subprocess.run(arguments,env=environment,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            captured=json.loads(output.read_text())
            self.assertEqual(captured["compatibility"],"NOT_VERIFIED")
            before=output.read_bytes()
            second=subprocess.run(arguments,env=environment,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual(output.read_bytes(),before)
            self.assertEqual(version.read_text(),"Version=UNKNOWN\n")


if __name__=="__main__":
    unittest.main()
