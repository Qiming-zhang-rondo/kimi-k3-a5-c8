"""Host-only C ABI, launch routing and build-isolation tests; no SDK/NPU claim.

The two real wrapper sources compile in separate TUs. Exactly one known ASC
launch per wrapper is replaced by a host recording shim; kernel bodies and
the fixed launch constant are mocked. No fatal signals or ACL calls are used.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
STANDALONE = ROOT / "standalone"
FIELDS = ("query", "key", "value", "indices", "blockTable", "actualQuery", "actualKv",
          "queryRope", "keyRope", "output", "maximum", "sum", "workspace", "tiling",
          "coldOutput", "coldMax", "coldSum", "coldPresent", "hotPresent")
ARCHIVE = STANDALONE / "archive/5407a8e"

PROBE = r'''
#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
namespace Probe {
extern std::array<uint8_t*,19> expected;
extern int originalCalls, fusedCalls;
extern void* expectedStream;
template<class Function,class... Args>
void Launch(Function function,void* stream,Args... args) {
 if(stream!=expectedStream) throw std::runtime_error("caller stream was not forwarded");
 function(args...);
}
template<class... Args> void Record(bool fused,Args... args) {
 std::array<uint8_t*,sizeof...(Args)> actual{{args...}};
 if(actual.size()!=(fused?19:14)) throw std::runtime_error("wrong entry argument count");
 for(unsigned i=0;i<actual.size();++i)
  if(actual[i]!=expected[i]) throw std::runtime_error("wrong entry pointer order");
 if(fused) ++fusedCalls; else ++originalCalls;
}
}
'''


class DirectApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        cls.c_compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")

    def test_actual_two_wrapper_tus_forward_every_pointer_and_current_stream(self):
        if not self.compiler:
            self.skipTest("Host C++ compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="sfa-direct-api-host-") as temp:
            directory = Path(temp)
            (directory / "sfa_direct_api.h").write_bytes((STANDALONE / "sfa_direct_api.h").read_bytes())
            (directory / "probe.h").write_text(PROBE)
            (directory / "sfa_fixed_contract.h").write_text(
                "#pragma once\nnamespace sfa_fixed {inline constexpr unsigned launchAic=1;}\n")
            definitions = []
            for wrapper, entry, count, fused in (
                ("sfa_direct_original_wrapper.asc", "sfa_standalone_entry.asc", 14, False),
                ("sfa_direct_fused_wrapper.asc", "sfa_fusion_entry.asc", 19, True),
                ("sfa_direct_fused_bf16_wrapper.asc", "sfa_fusion_bf16_entry.asc", 19, True),
            ):
                source = (STANDALONE / wrapper).read_text()
                transformed, replacements = re.subn(
                    r"(\w+)<<<\s*sfa_fixed::launchAic\s*,\s*nullptr\s*,\s*stream\s*>>>\s*\(",
                    r"Probe::Launch(\1, stream, ", source)
                self.assertEqual(replacements, 1, "must replace exactly the known ASC launch")
                self.assertNotIn("<<<", transformed)
                (directory / (wrapper + ".cpp")).write_text(transformed)
                symbol = ("sfa_cold_merge_bf16_probe" if "bf16" in wrapper else
                          "sfa_cold_merge_probe" if fused else "sfa_arch35_bf16_user_scratch_probe")
                args = ",".join("uint8_t* p" + str(i) for i in range(count))
                forwarding = ",".join("p" + str(i) for i in range(count))
                (directory / entry).write_text(
                    '#include "probe.h"\ninline void ' + symbol + '(' + args + ') {\n'
                    'Probe::Record(' + ("true" if fused else "false") + ',' + forwarding + ');}\n')
                definitions.append(directory / (wrapper + ".cpp"))
            offsets = "\n".join(
                'static_assert(offsetof(SfaDirectBuffers,' + name + ')=='
                + str(i) + '*sizeof(uint8_t*),"C ABI field offset");' for i, name in enumerate(FIELDS))
            setters = "\n".join('buffers.' + name + '=Probe::expected[' + str(i) + '];'
                                  for i, name in enumerate(FIELDS))
            driver = r'''
#include "sfa_direct_api.h"
#include "probe.h"
#include <cstddef>
#include <iostream>
#include <type_traits>
namespace Probe {
std::array<uint8_t*,19> expected{};
int originalCalls=0,fusedCalls=0;
void* expectedStream=reinterpret_cast<void*>(uintptr_t{42});
}
static_assert(std::is_standard_layout<SfaDirectBuffers>::value,"plain C layout");
static_assert(std::is_trivially_copyable<SfaDirectBuffers>::value,"plain pointer descriptor");
static_assert(sizeof(SfaDirectBuffers)==19*sizeof(uint8_t*),"exact pointer count");
OFFSETS
int main() {
 try {
  std::array<uint8_t,19> sentinel{};
  SfaDirectBuffers buffers{};
  for(unsigned i=0;i<19;++i) Probe::expected[i]=sentinel.data()+i;
  SETTERS
  if(SfaDirectAbiVersion()!=2 || SfaDirectBuffersBytes()!=sizeof(buffers)) return 2;
  SfaLaunchOriginal(Probe::expectedStream,&buffers);
  SfaLaunchFused(Probe::expectedStream,&buffers);
  SfaLaunchFusedBf16(Probe::expectedStream,&buffers);
  if(Probe::originalCalls!=1 || Probe::fusedCalls!=2) return 3;
  std::cout<<"HOST_DIRECT_API_ROUTING_PASS\n";
 } catch(const std::exception& e) {std::cerr<<e.what();return 1;}
}
'''.replace("OFFSETS", offsets).replace("SETTERS", setters)
            (directory / "driver.cpp").write_text(driver)
            binary = directory / "api_host"
            compiled = subprocess.run([self.compiler, "-std=c++17", "-O0", "-I", str(directory),
                                       *map(str, definitions), str(directory / "driver.cpp"), "-o", str(binary)],
                                      capture_output=True, text=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            ran = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
            self.assertEqual(ran.stdout.strip(), "HOST_DIRECT_API_ROUTING_PASS")

    def test_api_header_compiles_as_c_without_sdk_or_cpp_types(self):
        if not self.c_compiler:
            self.skipTest("Host C compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="sfa-direct-c-abi-") as temp:
            directory = Path(temp)
            source = '#include "sfa_direct_api.h"\n#include <stddef.h>\n'
            source += '_Static_assert(sizeof(SfaDirectBuffers)==19*sizeof(uint8_t*),"pointer count");\n'
            for i, field in enumerate(FIELDS):
                source += '_Static_assert(offsetof(SfaDirectBuffers,' + field + ')=='
                source += str(i) + '*sizeof(uint8_t*),"field order");\n'
            source += 'int main(void) {return 0;}\n'
            (directory / "check.c").write_text(source)
            result = subprocess.run([self.c_compiler, "-std=c11", "-Werror", "-I", str(STANDALONE),
                                     "-c", str(directory / "check.c"), "-o", str(directory / "check.o")],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_each_wrapper_owns_only_one_real_entry_without_acl_lifecycle(self):
        original = (STANDALONE / "sfa_direct_original_wrapper.asc").read_text()
        fused = (STANDALONE / "sfa_direct_fused_wrapper.asc").read_text()
        api = (STANDALONE / "sfa_direct_api.h").read_text()
        self.assertIn('#include "sfa_standalone_entry.asc"', original)
        self.assertNotIn('#include "sfa_fusion_entry.asc"', original)
        self.assertIn('#include "sfa_fusion_entry.asc"', fused)
        self.assertNotIn('#include "sfa_standalone_entry.asc"', fused)
        for source in (original, fused, api):
            for forbidden in ("aclInit(", "aclrtSetDevice(", "aclrtResetDevice(", "aclrtSynchronize", "aclrtMalloc("):
                self.assertNotIn(forbidden, source)
        for forbidden in ("kernel_operator.h", "__aicore__", "torch/", "entry.asc", "arch35/"):
            self.assertNotIn(forbidden, api)
        for metadata in ("SfaDirectAbiVersion", "SfaDirectBuffersBytes"):
            self.assertEqual(original.count(metadata + "(void)"), 1)
            self.assertNotIn(metadata, fused)

    def test_shared_target_uses_isolated_tus_pic_and_one_sdk_runtime(self):
        cmake = (STANDALONE / "CMakeLists.txt").read_text()
        self.assertNotIn("/usr/local/Ascend/", cmake)
        target = cmake.split("if(SFA_BUILD_TORCH_DIRECT)\n", 1)[1].split("endif()", 1)[0]
        self.assertIn("add_library(sfa_torch_direct SHARED", target)
        for source in ("sfa_direct_original_wrapper.asc", "sfa_direct_fused_wrapper.asc", "sfa_direct_fused_bf16_wrapper.asc"):
            self.assertIn(source, target)
        self.assertIn("configure_sfa(sfa_torch_direct)", target)
        self.assertIn("POSITION_INDEPENDENT_CODE ON", target)
        self.assertIn("UNITY_BUILD OFF", target)
        self.assertIn('m "${SFA_ACL_LIBRARY}" ascendc_runtime', target)
        self.assertNotIn("sfa_standalone)", target)
        self.assertNotIn("sfa_fusion_probe)", target)
        shared_config = cmake.split("function(configure_sfa target)", 1)[1].split("endfunction()", 1)[0]
        self.assertNotIn("experimental/arch35", shared_config)
        self.assertIn("if(SFA_BUILD_DEVICE_SMOKE OR SFA_BUILD_FUSION_SMOKE OR SFA_BUILD_TORCH_DIRECT)", cmake)
        self.assertIn('require_same_sfa_sdk("ASC compiler" "${CMAKE_ASC_COMPILER}")', cmake)
        self.assertIn('require_same_sfa_sdk("ACL runtime" "${SFA_ACL_LIBRARY}")', cmake)

    def test_exact_bad_source_snapshots_and_disabled_old_entry_paths(self):
        manifest = json.loads((ARCHIVE / "manifest.json").read_text())
        self.assertEqual(manifest["source_commit"], "5407a8e8175f750f859e061a5b993815a8756477")
        self.assertFalse(manifest["device_execution_allowed"])
        self.assertEqual(len(manifest["files"]), 6)
        for item in manifest["files"]:
            with self.subTest(source=item["source_path"]):
                data = (ROOT / item["snapshot_path"]).read_bytes()
                self.assertEqual(len(data), item["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
                self.assertEqual(hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(),
                                 item["git_blob_sha1"])
                stub = (ROOT / item["source_path"]).read_text()
                self.assertIn("FATAL_ERROR" if item["source_path"].endswith(".txt") else "#error", stub)
                self.assertNotIn("aclrtResetDevice", stub)
                self.assertNotIn("<<<", stub)

    def test_legacy_phase_c_option_fails_before_toolkit_discovery(self):
        cmake = shutil.which("cmake")
        if not cmake:
            self.skipTest("CMake unavailable")
        with tempfile.TemporaryDirectory(prefix="sfa-reject-old-bench-") as temp:
            result = subprocess.run([cmake, "-S", str(STANDALONE), "-B", temp, "-DSFA_BUILD_PHASE_C=ON"],
                                    capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SFA_BUILD_PHASE_C is quarantined", result.stdout + result.stderr)
            self.assertNotIn("Could not find a package configuration file provided by", result.stderr)

    def test_actual_cmake_sdk_check_rejects_cached_external_or_symlink_paths(self):
        cmake = shutil.which("cmake")
        if not cmake:
            self.skipTest("CMake unavailable")
        source = (STANDALONE / "CMakeLists.txt").read_text()
        helper = "function(require_same_sfa_sdk label path)" + source.split(
            "function(require_same_sfa_sdk label path)", 1)[1].split("endfunction()", 1)[0] + "endfunction()\n"
        with tempfile.TemporaryDirectory(prefix="sfa-sdk-provenance-") as temp:
            directory = Path(temp)
            sdk = directory / "sdk"
            sdk.mkdir()
            valid = sdk / "libascendcl.so"
            valid.write_bytes(b"host-test-placeholder")
            outside = directory / "other-sdk.so"
            outside.write_bytes(b"host-test-placeholder")
            alias = sdk / "stale-sdk-link.so"
            alias.symlink_to(outside)
            for path, expected in ((valid, 0), (outside, 1), (alias, 1)):
                with self.subTest(path=path.name):
                    script = directory / "check.cmake"
                    script.write_text('set(SFA_SDK_ROOT "' + str(sdk) + '")\n' + helper
                                      + 'require_same_sfa_sdk("test cached ACL" "' + str(path) + '")\n')
                    result = subprocess.run([cmake, "-P", str(script)], capture_output=True, text=True, timeout=5)
                    self.assertEqual(result.returncode == 0, expected == 0, result.stdout + result.stderr)
                    if expected:
                        self.assertIn("outside the selected SFA_SDK_ROOT", result.stderr)


if __name__ == "__main__":
    unittest.main()
