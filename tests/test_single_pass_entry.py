"""Compile the actual entry's host passes without parsing device/VF headers."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
ENTRY=ROOT/"experimental/single_pass/single_pass_entry.asc"


class SinglePassEntryTest(unittest.TestCase):
    def test_original_and_candidate_host_passes_and_wrong_arch_rejection(self):
        compiler=shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("No host C++ compiler")
        with tempfile.TemporaryDirectory(prefix="single-pass-entry-host-") as temp:
            directory=Path(temp)
            (directory/"kernel_operator.h").write_text(
                "#include <cstdint>\n#define __global__\n#define __aicore__\n#define __gm__\n"
                "#define KERNEL_TASK_TYPE_DEFAULT(x)\n")
            for candidate in (0,1):
                result=subprocess.run([compiler,"-std=c++17","-x","c++","-c",str(ENTRY),
                    "-I",str(directory),f"-DQSFA_SINGLE_PASS_CANDIDATE={candidate}",
                    f"-DQSFA_SINGLE_PASS_KERNEL=entry_{candidate}",
                    f"-DQSFA_SINGLE_PASS_RUN=run_{candidate}","-o",str(directory/f"entry{candidate}.o")],
                    text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr)
            rejected=subprocess.run([compiler,"-std=c++17","-x","c++","-fsyntax-only",str(ENTRY),
                "-I",str(directory),"-DQSFA_SINGLE_PASS_CANDIDATE=1",
                "-DQSFA_SINGLE_PASS_KERNEL=bad_arch","-D__NPU_ARCH__=2201"],text=True,capture_output=True)
            self.assertNotEqual(rejected.returncode,0)
            self.assertIn("requires compiler-selected dav-3510",rejected.stderr)


if __name__=="__main__": unittest.main()
