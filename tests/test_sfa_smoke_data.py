import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class SmokeDataHostTests(unittest.TestCase):
    def test_actual_cpp_reference_and_failure_gates(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("Host C++ compiler unavailable; no hardware validation")
        code = r'''
#include "sfa_smoke_data.h"
#include <iostream>
#define CHECK(x) do { if (!(x)) { std::cerr << "failed line " << __LINE__ << "\n"; return 1; } } while (0)
int main() {
 using namespace sfa_smoke;
 CHECK(FromBf16(ToBf16(1.0F)) == 1.0F);
 CHECK(FromBf16(ToBf16(-0.5F)) == -0.5F);
 for (bool zero : {true, false}) {
  auto data=MakeData(zero,0); auto ref=Reference(data);
  std::vector<uint16_t> out; for(auto x:ref.output) out.push_back(ToBf16(x));
  CHECK(Check(data,out,ref.maximum,ref.sum).passed);
  CHECK(ref.output[0] != 0.0F);
  if(zero) { for(auto x:ref.maximum) CHECK(x==0.0F); for(auto x:ref.sum) CHECK(x==128.0F);
   double mean=0; for(size_t j=0;j<128;++j) mean+=FromBf16(data.key[size_t(data.indices[j])*512]);
   CHECK(std::abs(ref.output[0]-mean/128.0)<1e-7);
  }
  auto changed=MakeData(zero,1); CHECK(!Check(changed,out,ref.maximum,ref.sum).passed);
  auto badSum=ref.sum; badSum.back()=0; CHECK(!Check(data,out,ref.maximum,badSum).passed);
  auto badMax=ref.maximum; badMax.back()=std::numeric_limits<float>::quiet_NaN(); CHECK(!Check(data,out,badMax,ref.sum).passed);
  auto badOut=out; for(size_t i=badOut.size()-512;i<badOut.size();++i) badOut[i]=ToBf16(1.0F);
  CHECK(!Check(data,badOut,ref.maximum,ref.sum).passed);
  badOut=out; badOut[0]=0x7f80; CHECK(!Check(data,badOut,ref.maximum,ref.sum).passed);
 }
 return 0;
}
'''
        with tempfile.TemporaryDirectory() as temp:
            source = pathlib.Path(temp) / "check.cpp"
            binary = pathlib.Path(temp) / "check"
            source.write_text(code)
            compiled = subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT / "standalone"), str(source), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            ran = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(ran.returncode, 0, ran.stderr)


if __name__ == "__main__":
    unittest.main()
