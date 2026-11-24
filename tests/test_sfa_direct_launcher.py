"""Execute actual launcher host control flow with ACL mocks, never an NPU.

Exactly one ASC launch delimiter is removed to make the actual source ordinary
C++; its call targets the entry's empty host-pass wrapper. This does NOT test
ASC lowering, device execution, scheduling, or attention math.
"""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
STANDALONE = ROOT / "standalone"

ACL_STUB = r'''
#pragma once
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cstdint>
using aclError = int;
using aclrtStream = void*;
constexpr int ACL_SUCCESS=0, ACL_MEM_MALLOC_HUGE_FIRST=0;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE=1, ACL_MEMCPY_DEVICE_TO_HOST=2;
namespace Probe {
inline int calls=0, afterFault=0, syncs=0, downloads=0, frees=0, destroys=0,
           finalizes=0, resets=0, kernels=0;
inline bool fault=false;
inline void Call() { ++calls; if (fault) ++afterFault; }
}
inline aclError aclInit(const char*) { Probe::Call(); return 0; }
inline aclError aclrtSetDevice(int32_t) { Probe::Call(); return 0; }
inline aclError aclrtCreateStream(aclrtStream* stream) {
    Probe::Call(); *stream=reinterpret_cast<void*>(uintptr_t{1}); return 0;
}
inline aclError aclrtMalloc(void** p, size_t n, int) {
    Probe::Call(); *p=std::malloc(n); return *p ? 0 : 1;
}
inline aclError aclrtMemcpy(void* dst, size_t capacity, const void* src, size_t n, int kind) {
    Probe::Call();
    if (kind==ACL_MEMCPY_DEVICE_TO_HOST) ++Probe::downloads;
    if (!dst || !src || n>capacity) return 2;
    std::memcpy(dst,src,n); return 0;
}
inline aclError aclrtSynchronizeStream(aclrtStream) {
    Probe::Call(); ++Probe::syncs; Probe::fault=true; return 507015;
}
inline aclError aclrtFree(void* p) { Probe::Call(); ++Probe::frees; std::free(p); return 0; }
inline aclError aclrtDestroyStream(aclrtStream) { Probe::Call(); ++Probe::destroys; return 0; }
inline aclError aclFinalize() { Probe::Call(); ++Probe::finalizes; return 0; }
inline aclError aclrtResetDevice(int32_t) { Probe::Call(); ++Probe::resets; return 0; }
'''

KERNEL_STUB = r'''
#pragma once
#include <cstdint>
#define __gm__
#define __global__
#define __aicore__
#define KERNEL_TYPE_MIX_AIC_1_2 42
#define KERNEL_TASK_TYPE_DEFAULT(value) do { ++Probe::kernels; } while (0)
'''

DRIVER = r'''
#define main launcher_main
#include "sfa_direct_launch_host.cpp"
#undef main
int main(int argc, char** argv) {
    const int result=launcher_main(argc,argv);
    std::cout << "HOST_RESULT {\"exit\":" << result
      << ",\"calls\":" << Probe::calls << ",\"after_fault\":" << Probe::afterFault
      << ",\"syncs\":" << Probe::syncs << ",\"downloads\":" << Probe::downloads
      << ",\"frees\":" << Probe::frees << ",\"destroys\":" << Probe::destroys
      << ",\"finalizes\":" << Probe::finalizes << ",\"resets\":" << Probe::resets
      << ",\"kernels\":" << Probe::kernels << "}\n";
    // Host mallocs deliberately remain owned until process exit after a fault.
    // Calling mock ACL cleanup here would obscure the production stop contract.
    return 0;
}
'''


class DirectLauncherHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("Host C++ compiler unavailable; no device test is claimed")
        cls.build = tempfile.TemporaryDirectory(prefix="sfa-launcher-host-")
        cls.addClassCleanup(cls.build.cleanup)
        directory = Path(cls.build.name)
        source = (STANDALONE / "sfa_direct_launch.asc").read_text()
        transformed, count = re.subn(r"<<<\s*sfa_fixed::launchAic\s*,\s*nullptr\s*,\s*stream\s*>>>", "", source)
        if count != 1 or "<<<" in transformed:
            raise AssertionError("Expected exactly one known ASC launch; do not silently rewrite new launches")
        (directory / "sfa_direct_launch_host.cpp").write_text(transformed)
        (directory / "acl").mkdir()
        (directory / "acl/acl.h").write_text(ACL_STUB)
        (directory / "kernel_operator.h").write_text(KERNEL_STUB)
        (directory / "driver.cpp").write_text(DRIVER)
        cls.binary = directory / "launcher_host"
        result = subprocess.run([compiler,"-std=c++17","-O0","-D__NPU_ARCH__=0",
                                 "-DSFA_PROBE_TEMPLATE_MODE=0","-I",str(directory),
                                 "-I",str(STANDALONE),str(directory/"driver.cpp"),
                                 "-o",str(cls.binary)],capture_output=True,text=True,timeout=30)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def run_case(self, directory, args):
        result = subprocess.run([str(self.binary),*args],cwd=directory,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        records=[line.removeprefix("HOST_RESULT ") for line in result.stdout.splitlines()
                 if line.startswith("HOST_RESULT ")]
        self.assertEqual(len(records),1,result.stdout+result.stderr)
        return json.loads(records[0])

    def test_default_preflight_never_calls_acl(self):
        with tempfile.TemporaryDirectory() as work:
            result=self.run_case(work,[])
            self.assertEqual(result["exit"],0)
            self.assertEqual(result["calls"],0)
            self.assertEqual(result["kernels"],0)
            self.assertFalse((Path(work)/"DEVICE_FAULT.json").exists())

    def test_existing_fault_marker_blocks_run_before_acl(self):
        with tempfile.TemporaryDirectory() as work:
            marker=Path(work)/"DEVICE_FAULT.json"
            marker.write_text('{"previous":true}')
            result=self.run_case(work,["--run","--device","0"])
            self.assertEqual(result["exit"],2)
            self.assertEqual(result["calls"],0)
            self.assertEqual(result["kernels"],0)
            self.assertEqual(marker.read_text(),'{"previous":true}')

    def test_sync_507015_stops_all_subsequent_device_operations(self):
        with tempfile.TemporaryDirectory() as work:
            result=self.run_case(work,["--run","--device","0"])
            self.assertEqual(result["exit"],1)
            self.assertGreater(result["calls"],0)
            self.assertEqual(result["syncs"],1)
            self.assertEqual(result["kernels"],1)
            for field in ("after_fault","downloads","frees","destroys","finalizes","resets"):
                self.assertEqual(result[field],0,field)
            marker=json.loads((Path(work)/"DEVICE_FAULT.json").read_text())
            self.assertEqual(marker["acl_error"],507015)
            self.assertEqual(marker["operation"],"aclrtSynchronizeStream")
            self.assertTrue(marker["device_tests_stopped"])
            report=json.loads((Path(work)/"sfa_device_smoke_results.json").read_text())
            self.assertEqual(report["status"],"EXECUTION_ERROR")


if __name__ == "__main__":
    unittest.main()
