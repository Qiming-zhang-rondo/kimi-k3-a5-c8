"""Compile the real BF16 entry wrapper against native-service API mocks."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bf16_entry_helpers", ROOT / "tests/test_phase_b_source_contract.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


class Bf16EntryTests(unittest.TestCase):
    def compile(self, core=None, packed=False):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("Host C++ compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="sfa-bf16-entry-") as temp:
            root = Path(temp)
            (root / "standalone").mkdir()
            (root / "experimental/arch35").mkdir(parents=True)
            for name in ("sfa_fusion_entry.asc", "sfa_fusion_bf16_entry.asc", "sfa_fusion_bf16_packed_entry.asc", "sparse_flash_attention_tiling_data.h"):
                (root / "standalone" / name).write_bytes((ROOT / "standalone" / name).read_bytes())
            (root / "kernel_operator.h").write_text(helpers.OPERATOR)
            kernel = helpers.KERNEL.replace("std::is_same<Out, float>::value",
                                           "std::is_same<Out, AscendC::bfloat16_t>::value")
            kernel += "\nnamespace BaseApiBf16ColdIo = BaseApi; // mock native-service interface only\n"
            kernel += "\nnamespace BaseApiBf16PackedColdIo = BaseApi; // distinct candidate interface\n"
            if core is None:
                kernel = '#error "Host scan parsed device-only BF16 header"\n'
            (root / "experimental/arch35/sparse_flash_attention_kernel_mla.h").write_text(kernel)
            driver = helpers.DRIVER if core else helpers.HOST_DRIVER
            driver = driver.replace('"standalone/sfa_fusion_entry.asc"', '"standalone/sfa_fusion_bf16_entry.asc"')
            driver = driver.replace("sfa_cold_merge_probe(", "sfa_cold_merge_bf16_probe(")
            if packed:
                driver = driver.replace("sfa_fusion_bf16_entry.asc", "sfa_fusion_bf16_packed_entry.asc")
                driver = driver.replace("sfa_cold_merge_bf16_probe(", "sfa_cold_merge_bf16_packed_probe(")
            (root / "driver.cpp").write_text(driver)
            flags = ["-D__NPU_ARCH__=3510", "-D" + core + "=1"] if core else ["-D__NPU_ARCH__=0"]
            built = subprocess.run([compiler, "-std=c++17", "-O0", "-DPROBE_EXPECTED_MODE=0", *flags,
                                    "-I", str(root), "-I", str(ROOT / "standalone"), str(root / "driver.cpp"),
                                    "-o", str(root / "entry")], capture_output=True, text=True, timeout=30)
            self.assertEqual(built.returncode, 0, built.stderr)
            ran = subprocess.run([str(root / "entry")], capture_output=True, text=True, timeout=5)
            self.assertEqual(ran.returncode, 0, ran.stderr)

    def test_aic_bf16_types_and_all_cold_arguments(self):
        self.compile("__DAV_C310_CUBE__")

    def test_aiv_bf16_types_and_tiling_lifetime(self):
        self.compile("__DAV_C310_VEC__")

    def test_packed_entry_has_distinct_types_on_aic_and_aiv(self):
        self.compile("__DAV_C310_CUBE__",packed=True)
        self.compile("__DAV_C310_VEC__",packed=True)

    def test_host_isolation_and_distinct_template_symbols(self):
        self.compile()
        variant = (ROOT / "standalone/sfa_fusion_bf16_entry.asc").read_text()
        self.assertIn("SFA_FUSION_ENTRY_SYMBOL sfa_cold_merge_bf16_probe", variant)
        self.assertIn("SFA_FUSION_RUN_SYMBOL RunSfaArch35Bf16FusedBf16Io", variant)
        for name in ("sparse_flash_attention_kernel_mla.h", "sparse_flash_attention_service_vector_mla.h"):
            source = (ROOT / "experimental/arch35" / name).read_text()
            self.assertIn("namespace BaseApiBf16ColdIo", source)
            self.assertIn("using namespace ::BaseApi", source)


if __name__ == "__main__":
    unittest.main()
