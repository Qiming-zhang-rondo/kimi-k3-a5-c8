"""Checkpoint-format gates run with JSON only, before framework/model imports."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("framework_model_preflight",ROOT/"model_preflight.py")
preflight=importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


class ModelPreflightTest(unittest.TestCase):
    def test_explicit_local_config_formats_nested_and_derived_dimensions(self):
        fixtures=((dict(kv_lora_rank=512,qk_rope_head_dim=64),preflight.COMPATIBLE,64,"root"),
            (dict(kv_lora_rank=512,qk_rope_head_dim=0),"UNSUPPORTED_R0_PACKED272",0,"root"),
            (dict(qk_rope_head_dim=0,text_config=dict(kv_lora_rank=512,qk_rope_head_dim=64)),preflight.COMPATIBLE,64,"text_config"),
            (dict(kv_lora_rank=512,qk_head_dim=192,qk_nope_head_dim=128),preflight.COMPATIBLE,64,"root"),
            (dict(model_type="glm5.3flash",kv_lora_rank=512),"UNKNOWN_FORMAT",None,"root"),
            (dict(kv_lora_rank=256,qk_rope_head_dim=64),"UNSUPPORTED_LATENT_DIM",64,"root"))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"config.json"
            for config,status,rope,section in fixtures:
                with self.subTest(config=config):
                    path.write_text(json.dumps(config))
                    report=preflight.inspect_model_format(dict(model=directory))
                    self.assertEqual(report["status"],status)
                    self.assertEqual(report["qk_rope_head_dim"],rope)
                    self.assertEqual(report["config_section"],section)
                    self.assertEqual(report["config_path"],str(path.resolve()))
                    self.assertEqual(len(report["config_sha256"]),64)

    def test_remote_or_missing_config_is_unknown_without_guessing_or_download(self):
        report=preflight.inspect_model_format(dict(model="definitely-no-local-model/glm5.3-flash"))
        self.assertEqual(report["status"],"UNKNOWN_FORMAT")
        self.assertIsNone(report["qk_rope_head_dim"])
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(preflight.inspect_model_format(dict(model=directory))["status"],"UNKNOWN_FORMAT")

    def test_probe_preflight_only_needs_no_prompts_framework_or_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"config.json").write_text(json.dumps(dict(kv_lora_rank=512,qk_rope_head_dim=64)))
            engine=root/"engine.json"
            engine.write_text(json.dumps(dict(model=str(root))))
            result=subprocess.run([sys.executable,str(ROOT/"run_model_probe.py"),"--engine-json",str(engine),
                "--arm","single_pass","--preflight-only","--output",str(root/"out")],
                text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads((root/"out/preflight.json").read_text())["status"],preflight.COMPATIBLE)
            self.assertFalse((root/"out/probe.json").exists())
            (root/"config.json").write_text(json.dumps(dict(kv_lora_rank=512,qk_rope_head_dim=0)))
            result=subprocess.run([sys.executable,str(ROOT/"run_model_probe.py"),"--engine-json",str(engine),
                "--arm","single_pass","--preflight-only","--output",str(root/"out_r0")],
                text=True,capture_output=True)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertEqual(json.loads((root/"out_r0/preflight.json").read_text())["status"],"UNSUPPORTED_R0_PACKED272")
            self.assertFalse((root/"out_r0/probe.json").exists())

    def test_actual_nope_run_stops_and_records_failure_before_framework_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"config.json").write_text(json.dumps(dict(kv_lora_rank=512,qk_rope_head_dim=0)))
            engine=root/"engine.json"
            engine.write_text(json.dumps(dict(model=str(root))))
            result=subprocess.run([sys.executable,str(ROOT/"run_model_probe.py"),"--engine-json",str(engine),
                "--arm","single_pass","--output",str(root/"out")],text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("UNSUPPORTED_R0_PACKED272",result.stderr)
            self.assertNotIn("ModuleNotFoundError",result.stderr)
            report=json.loads((root/"out/probe.json").read_text())
            self.assertEqual(report["status"],"MODEL_FORMAT_UNSUPPORTED")
            self.assertEqual(report["model_format_status"],"UNSUPPORTED_R0_PACKED272")
            self.assertFalse((root/"out/engine.json").exists())


if __name__=="__main__":unittest.main()
