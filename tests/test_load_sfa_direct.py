"""Host-only direct-library loader guards; the JIT compiler boundary is mocked.

No test claims that Torch, ASC or an NPU compiled/executed the fixture source.
"""
import importlib.util
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experimental/torch/load_sfa_direct.py"


class DirectLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sfa loader with spaces ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        source = self.root / "repo/experimental/torch/load_sfa_direct.py"
        source.parent.mkdir(parents=True)
        shutil.copyfile(SOURCE, source)
        source.with_name("sfa_direct_bridge.cpp").write_text("// JIT-boundary fixture only\n")
        for name in (
            "standalone/sfa_direct_api.h", "standalone/sfa_fixed_contract.h",
            "standalone/sparse_flash_attention_tiling_data.h",
            "upstream/sgl-kernel-npu-597250fc/csrc/sparse_flash_attention/op_kernel/sparse_flash_attention_tiling_data.h",
        ):
            target = self.root / "repo" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("// Pinned host ABI fixture; not compiled in this test\n")
        self.library_dir = self.root / "asc build"
        self.library_dir.mkdir()
        self.library = self.library_dir / "libsfa_torch_direct.so"
        self.library.write_bytes(b"not a real executable; JIT is mocked")
        self.sdk = self.root / "matching sdk"
        (self.sdk / "include/acl").mkdir(parents=True)
        for name in ("acl.h", "acl_op.h"):
            (self.sdk / "include/acl" / name).write_text("// SDK fixture\n")
        (self.library_dir / "CMakeCache.txt").write_text("SFA_SDK_ROOT:PATH=" + str(self.sdk) + "\n")
        self.npu = self.root / "torch npu"
        headers = self.npu / "include/torch_npu/csrc/core/npu"
        headers.mkdir(parents=True)
        for name in ("NPUGuard.h", "NPUStream.h", "NPUCachingAllocator.h"):
            (headers / name).write_text("// NPU header fixture\n")
        (self.npu / "lib").mkdir()
        (self.npu / "lib/libtorch_npu.so").write_bytes(b"mock")
        self.jit = Mock(return_value=types.SimpleNamespace())
        torch = types.ModuleType("torch")
        torch.__version__ = "fixture"
        torch._C = types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False)
        npu = types.ModuleType("torch_npu")
        npu.__file__ = str(self.npu / "__init__.py")
        npu.__version__ = "fixture"
        utils = types.ModuleType("torch.utils")
        extension = types.ModuleType("torch.utils.cpp_extension")
        extension.load = self.jit
        self.modules = {"torch": torch, "torch_npu": npu, "torch.utils": utils,
                        "torch.utils.cpp_extension": extension}
        spec = importlib.util.spec_from_file_location("direct_loader_fixture", source)
        self.loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.loader)

    def load(self, **kwargs):
        with patch.dict(sys.modules, self.modules):
            return self.loader.load_sfa_direct(self.library_dir, sdk_root=self.sdk, **kwargs)

    def test_only_cpp_compiles_and_exact_library_paths_survive_spaces(self):
        module = self.load(build_directory=self.root / "jit build")
        call = self.jit.call_args.kwargs
        self.assertEqual(len(call["sources"]), 1)
        self.assertEqual(Path(call["sources"][0]).name, "sfa_direct_bridge.cpp")
        self.assertFalse(any(path.endswith(".asc") for path in call["sources"]))
        flags = shlex.split(" ".join(call["extra_ldflags"]))
        self.assertEqual(flags[0], str(self.library))
        self.assertIn("-L" + str(self.npu / "lib"), flags)
        self.assertIn("-ltorch_npu", flags)
        self.assertIn(str(self.library_dir), flags)
        self.assertIn(str(self.sdk / "include"), call["extra_include_paths"])
        self.assertFalse(module._sfa_build_identity["npu_launched_by_loader"])
        self.assertFalse(module._sfa_build_identity["asc_compiled_by_loader"])
        self.assertTrue((self.root / "jit build/sfa_direct_bridge_build.json").is_file())

    def test_cache_reuses_same_binary_but_rejects_in_process_overwrite(self):
        first = self.load()
        self.assertIs(self.load(), first)
        self.assertEqual(self.jit.call_count, 1)
        self.library.write_bytes(b"new ASC binary cannot replace an already loaded SONAME")
        with self.assertRaisesRegex(RuntimeError, "fresh Python process"):
            self.load()
        self.assertEqual(self.jit.call_count, 1)

    def test_mismatched_sdk_is_rejected_before_jit(self):
        (self.library_dir / "CMakeCache.txt").write_text("SFA_SDK_ROOT:PATH=/different/sdk\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.load()
        self.jit.assert_not_called()

    def test_missing_library_and_missing_npu_header_fail_without_build(self):
        self.library.unlink()
        with self.assertRaisesRegex(ValueError, "nonempty libsfa_torch_direct"):
            self.load()
        self.library.write_bytes(b"mock")
        (self.npu / "include/torch_npu/csrc/core/npu/NPUStream.h").unlink()
        with self.assertRaisesRegex(ValueError, "required direct-bridge headers"):
            self.load()
        self.jit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
