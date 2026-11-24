"""Compile and execute the real host-only fusion data/reference header.

These tests exercise ordinary C++ math and memory contracts, not ASC or NPU.
"""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

SOURCE = r'''
#include "sfa_fusion_smoke_data.h"
#include <iostream>
#include <set>
#include <cstring>
#define CHECK(x) do { if (!(x)) { std::cerr << "failed line " << __LINE__ << "\n"; return 1; } } while (0)
namespace f = sfa_fusion_smoke;
using sfa_smoke::ToBf16;
using sfa_smoke::FromBf16;
const float quietNan = std::numeric_limits<float>::quiet_NaN();
int main(int argc, char** argv) {
 CHECK(argc==2);
 const std::string mode(argv[1]);
 auto cases = f::MakeCases();
 if (mode=="contract") {
  CHECK(cases.size()>=7 && f::outputElements==131072 && f::outputBytes==524288);
  std::set<std::string> names;
  for(const auto& d : cases) { CHECK(f::ValidateData(d)); CHECK(names.insert(d.name).second); }
  for(const auto* name : {"mixed_zero","mixed_nonzero","changed_hot","cold_only_flag",
                         "hot_only_flag","both_empty","extreme_cold","per_query_presence"})
   CHECK(names.count(name)==1);
  auto t=f::MakeTiling(); CHECK(f::ValidateTiling(t));
  CHECK(t.baseParams.returnSoftmaxLse==0 && sfa_fixed::MakeTiling().baseParams.returnSoftmaxLse==1);
  CHECK(t.singleCoreParams.usedCoreNum==1 && t.baseParams.sparseBlockCount==128);
  CHECK(!sfa_fixed::ValidateTiling(t));
  t.baseParams.returnSoftmaxLse=1; CHECK(!f::ValidateTiling(t));
  t=f::MakeTiling(); t.baseParams.nNumOfQInOneGroup=1; CHECK(!f::ValidateTiling(t));
  CHECK(!f::ValidateTiling(f::MakeTiling(0)));
  auto bad=cases[0]; bad.coldPresent[0]=2; CHECK(!f::ValidateData(bad));
  bad=cases[0]; bad.coldMax.pop_back(); CHECK(!f::ValidateData(bad));
  bad=cases[0]; bad.hot.indices.back()=-1; CHECK(!f::ValidateData(bad));
  bad=cases[0]; bad.hot.indices[1]=bad.hot.indices[0]; CHECK(!f::ValidateData(bad));
  CHECK(std::isnan(cases[4].coldOutput[0]) && std::isnan(cases[5].coldSum[0]));
  return 0;
 }
 std::vector<uint16_t> hot(f::outputElements,ToBf16(0.3333F));
 std::vector<float> hm(f::rows,0.0F), hs(f::rows,128.0F);
 if (mode=="math") {
  // The zero-query mathematical reference has exactly 128 vs 1920 mass.
  const auto hotMath=sfa_smoke::Reference(cases[0].hot);
  auto zero=f::Reference(cases[0]);
  for(size_t row=0;row<f::rows;++row) {
   CHECK(hotMath.maximum[row]==0 && hotMath.sum[row]==128);
   for(size_t d=0;d<512;++d) {
    size_t i=row*512+d;
    float expected=float((128.0*FromBf16(ToBf16(hotMath.output[i]))+1920.0*cases[0].coldOutput[i])/2048.0);
    CHECK(std::abs(zero[i]-expected)<1e-7);
   }
  }
  // Each control is independently constructed, finite, and accepted as its reference.
  std::vector<float> nonzero, changed;
  for(const auto& data: cases) {
   auto ref=f::Reference(data); CHECK(f::Check(data,ref,ref).passed);
   if(data.name=="mixed_nonzero") nonzero=ref;
   if(data.name=="changed_hot") changed=ref;
  }
  CHECK(!f::Check(cases[1],changed,nonzero).passed);
  // NativeReference must use supplied native BF16 values, not recompute hot math.
  auto merged=f::NativeReference(cases[0],hot,hm,hs);
  for(size_t i=0;i<merged.size();++i)
   CHECK(std::abs(merged[i]-float((128.0*FromBf16(hot[i])+1920.0*cases[0].coldOutput[i])/2048.0))<1e-7);
  std::vector<uint16_t> poisonedHot(f::outputElements,0x7fc1);
  std::vector<float> poisonedStats(f::rows,quietNan);
  auto coldOnly=f::NativeReference(cases[3],poisonedHot,poisonedStats,poisonedStats);
  CHECK(coldOnly==cases[3].coldOutput); // Cold FP32 bits must not round to BF16.
  CHECK(coldOnly[0]!=FromBf16(ToBf16(coldOnly[0])));
  auto hotOnly=f::NativeReference(cases[4],hot,hm,hs);
  for(size_t i=0;i<hot.size();++i) CHECK(hotOnly[i]==FromBf16(hot[i]));
  auto empty=f::NativeReference(cases[5],poisonedHot,poisonedStats,poisonedStats);
  for(float x:empty) CHECK(x==0 && std::isfinite(x));
  auto extreme=f::NativeReference(cases[6],hot,hm,hs);
  for(size_t row=0;row<f::rows;++row) for(size_t d=0;d<512;++d) {
   const size_t i=row*512+d;
   CHECK(extreme[i]==(row%2==0 ? cases[6].coldOutput[i] : FromBf16(hot[i])));
  }
  auto flags=f::NativeReference(cases[7],hot,hm,hs);
  for(size_t row=0;row<f::rows;++row) for(size_t d=0;d<512;++d) {
   const size_t i=row*512+d, q=row/64;
   if(q==1) CHECK(flags[i]==FromBf16(hot[i]));
   if(q==2) CHECK(flags[i]==cases[7].coldOutput[i]);
   if(q==3) CHECK(flags[i]==0);
  }
  return 0;
 }
 if(mode=="rejections") {
  auto ref=f::NativeReference(cases[0],hot,hm,hs);
  CHECK(f::Check(cases[0],ref,ref).passed);
  auto bad=ref; bad.pop_back(); CHECK(!f::Check(cases[0],bad,ref).passed);
  bad=ref; bad[512]=quietNan;
  auto result=f::Check(cases[0],bad,ref);
  CHECK(!result.passed && result.nonfinite==1 && result.message.find("first_bad_row=1")!=std::string::npos);
  bad=ref; for(size_t i=0;i<512;++i) bad[i]*=1.1F;
  result=f::Check(cases[0],bad,ref); CHECK(!result.passed && result.worstRowRelativeL2>0.09);
  bad=ref; std::rotate(bad.begin(),bad.begin()+512,bad.end()); CHECK(!f::Check(cases[0],bad,ref).passed);
  auto coldChanged=cases[0]; for(auto& x:coldChanged.coldOutput) x+=0.01F;
  bad=f::NativeReference(coldChanged,hot,hm,hs); CHECK(!f::Check(cases[0],bad,ref).passed);
  bad.assign(f::outputElements,0); CHECK(!f::Check(cases[0],bad,ref).passed);
  auto invalid=cases[0]; invalid.coldSum[0]=0;
  bad=f::NativeReference(invalid,hot,hm,hs); CHECK(!f::Check(invalid,bad,ref).passed);
  invalid=cases[0]; invalid.coldMax[0]=quietNan;
  bad=f::NativeReference(invalid,hot,hm,hs); CHECK(!f::Check(invalid,bad,ref).passed);
  auto shortHot=hot; shortHot.pop_back(); bool thrown=false;
  try { (void)f::NativeReference(cases[0],shortHot,hm,hs); } catch(const std::invalid_argument&) { thrown=true; }
  CHECK(thrown);
  CHECK(!f::Check(cases[0],ref,ref,-1).passed && !f::Check(cases[0],ref,ref,quietNan).passed);
  bad=ref; bad[0]=quietNan; CHECK(!f::Check(cases[0],ref,bad).passed);
  return 0;
 }
 return 2;
}
'''


class FusionSmokeHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("Host C++ compiler unavailable; no ASC/NPU test")
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        source = pathlib.Path(cls.temp.name) / "fusion_smoke.cpp"
        cls.binary = pathlib.Path(cls.temp.name) / "fusion_smoke"
        source.write_text(SOURCE)
        compiled = subprocess.run(
            [compiler, "-std=c++17", "-O2", "-I", str(ROOT / "standalone"), str(source), "-o", str(cls.binary)],
            capture_output=True, text=True, timeout=60,
        )
        if compiled.returncode:
            raise AssertionError(compiled.stderr)

    def run_mode(self, mode):
        result = subprocess.run([str(self.binary), mode], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fixed_candidate_contract_and_case_shapes(self):
        self.run_mode("contract")

    def test_mathematical_native_rounding_presence_and_extreme_states(self):
        self.run_mode("math")

    def test_poison_shapes_changed_states_rows_and_scaling_rejected(self):
        self.run_mode("rejections")


if __name__ == "__main__":
    unittest.main()
