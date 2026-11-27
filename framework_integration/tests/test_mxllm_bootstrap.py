"""Real-import fixtures exercise global setup without importing torch or NPU."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "mxllm_bootstrap.py"


class MxllmBootstrapTest(unittest.TestCase):
    def fixture(self, root, *, modern=True, field="dataclass", env=True, plain=False):
        files = {
            "ascend_vllm/__init__.py": (
                "from .utils import adapt_patch\n"
                "def _ensure_global_patch():\n    adapt_patch(is_global_patch=True)\n"
                if modern else "# legacy checkout: no global helper\n"
            ),
            "ascend_vllm/utils.py": (
                "def adapt_patch(is_global_patch=False):\n"
                "    assert is_global_patch is True\n"
                "    from .patch import platform\n"
            ),
            "ascend_vllm/patch/__init__.py": "",
            "ascend_vllm/patch/worker/__init__.py": "raise RuntimeError('WORKER IMPORTED TOO EARLY')\n",
            "ascend_vllm/patch/platform/__init__.py": (
                "from vllm.config import AttentionConfig\n"
                "from vllm_ascend import envs\n"
                + ("AttentionConfig.__dataclass_fields__ = {'mla_kv_quant_mode': object()}\n" if field == "dataclass" else
                   "AttentionConfig.model_fields = {'mla_kv_quant_mode': object()}\n" if field == "pydantic" else
                   "AttentionConfig.mla_kv_quant_mode = 'mxfp4'\n" if plain else "")
                + ("envs.env_variables['ASCEND_VLLM_KV_CONTIG'] = lambda: False\n" if env else "")
            ),
            "vllm/__init__.py": "",
            "vllm/config.py": "class AttentionConfig: pass\n",
            "vllm_ascend/__init__.py": "",
            "vllm_ascend/envs.py": "env_variables = {}\n",
        }
        for relative, text in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def run_bootstrap(self, root, prefix=""):
        code = (
            "import importlib.util, json, os, sys\n"
            f"spec=importlib.util.spec_from_file_location('bootstrap', {str(HELPER)!r})\n"
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
            + prefix + "\n"
            f"report=module.bootstrap_mxllm({str(root)!r})\n"
            "assert 'ascend_vllm.patch.worker' not in sys.modules\n"
            "assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules\n"
            "assert 'ASCEND_VLLM_KV_CONTIG' not in os.environ\n"
            "print(json.dumps(report))\n"
        )
        return subprocess.run([sys.executable, "-I", "-S", "-c", code], text=True, capture_output=True)

    def test_modern_formal_global_entry_before_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root)
            result=self.run_bootstrap(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            value=json.loads(result.stdout)
            self.assertEqual(value["status"], "MXLLM_BOOTSTRAPPED")
            self.assertEqual(value["global_patch_entry"], "ascend_vllm._ensure_global_patch")
            self.assertEqual(value["ascend_vllm_origin"], str((root/"ascend_vllm/__init__.py").resolve()))
            self.assertEqual(value["attention_config_declaration"], "dataclass")

    def test_legacy_real_adapt_entry_and_pydantic_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root, modern=False, field="pydantic")
            result=self.run_bootstrap(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            value=json.loads(result.stdout)
            self.assertEqual(value["global_patch_entry"], "ascend_vllm.utils.adapt_patch")
            self.assertEqual(value["attention_config_declaration"], "pydantic")

    def test_config_class_attribute_is_not_a_declared_schema_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root, field="missing", plain=True)
            result=self.run_bootstrap(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("did not declare AttentionConfig.mla_kv_quant_mode", result.stderr)

    def test_missing_env_registration_fails_without_inventing_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root, env=False)
            result=self.run_bootstrap(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ASCEND_VLLM_KV_CONTIG unregistered", result.stderr)
            self.assertIn("cannot repair a missing registry entry", result.stderr)

    def test_preimported_wrong_checkout_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            requested=Path(directory)/"requested"
            other=Path(directory)/"other"
            self.fixture(requested)
            self.fixture(other)
            prefix=f"sys.path.insert(0, {str(other)!r}); import ascend_vllm"
            result=self.run_bootstrap(requested, prefix)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ascend_vllm source mismatch", result.stderr)

    def test_helper_import_only_is_stdlib_and_does_not_bootstrap(self):
        code=("import importlib.util, sys; "
              f"s=importlib.util.spec_from_file_location('bootstrap', {str(HELPER)!r}); "
              "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
              "assert not any(n in sys.modules for n in ['torch','torch_npu','vllm','ascend_vllm'])")
        result=subprocess.run([sys.executable,"-I","-S","-c",code],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == "__main__": unittest.main()
