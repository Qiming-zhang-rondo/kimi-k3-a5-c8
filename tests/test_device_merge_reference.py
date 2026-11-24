"""Execute the actual device header under a small C++ MicroAPI CPU model.

This tests indexing, DMA lengths, presence, arithmetic and bounded-state errors.
It does NOT validate AscendC compilation, register layout/lowering or event timing.
The model uses FP32 lanes and a BF16 RNE emulation; the real header deliberately
uses the native reader's CAST_ROUND, whose device behavior remains an A5 gate.
"""
import math
from pathlib import Path
import random
import shutil
import struct
import subprocess
import tempfile
import unittest

HEADER = Path(__file__).resolve().parents[1] / "experimental/arch35/cold_state_merge_device.h"

STUB = r'''
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#define __aicore__
#define __simd_vf__
#define __ubuf__
#define __gm__
namespace AscendC {
enum class HardEvent {MTE2_V, V_MTE2};
enum class CMPMODE {GE, LE, EQ};
enum class RoundMode {CAST_ROUND, UNKNOWN};
enum class MemType {VEC_STORE, VEC_LOAD};
constexpr int PIPE_V=1;
using TEventID=int;
inline int dmaCalls=0, scalarReads=0, allocatedBytes=0, pending[2][32]={};
template<HardEvent E> void SetFlag(TEventID id) {
 if(pending[int(E)][id]++) throw std::runtime_error("double event set");
}
template<HardEvent E> void WaitFlag(TEventID id) {
 if(--pending[int(E)][id]) throw std::runtime_error("unmatched event wait");
}
template<int> void PipeBarrier() {}
template<class T> struct LocalTensor {
 T* p=nullptr; size_t size=0;
 void* GetPhyAddr() const {if(!p) throw std::runtime_error("absent UB access"); return p;}
 LocalTensor operator[](size_t i) const {if(i>size) throw std::runtime_error("UB offset"); return {p+i,size-i};}
};
template<class T> struct GlobalTensor {
 T* p=nullptr;
 void SetGlobalBuffer(T* q) {p=q;}
 GlobalTensor operator[](size_t i) const {if(!p) throw std::runtime_error("absent GM access"); return {p+i};}
 T GetValue(size_t i) const {++scalarReads; if(!p) throw std::runtime_error("absent GM read"); return p[i];}
};
template<int=0> struct TBuf {
 std::vector<uint32_t> data;
 template<class T> LocalTensor<T> Get() {return {reinterpret_cast<T*>(data.data()),data.size()*4/sizeof(T)};}
};
struct TPipe {
 template<int P> void InitBuffer(TBuf<P>& b,size_t size) {allocatedBytes+=size;b.data.resize((size+3)/4);}
 template<HardEvent E> TEventID AllocEventID() {return 0;}
};
struct DataCopyExtParams {uint16_t blockCount=0;uint32_t blockLen=0,srcStride=0,dstStride=0;};
template<class T> struct DataCopyPadExtParams {bool isPad=false;uint8_t leftPadding=0,rightPadding=0;T paddingValue{};};
template<class T> void DataCopyPad(LocalTensor<T> dst,GlobalTensor<T> src,
 const DataCopyExtParams& c,const DataCopyPadExtParams<T>& pad) {
 ++dmaCalls;
 if(c.blockCount!=1 || c.srcStride || c.dstStride || pad.leftPadding) throw std::runtime_error("unexpected DMA");
 size_t n=c.blockLen/sizeof(T),extra=pad.isPad?pad.rightPadding:0;
 if(n+extra>dst.size || !src.p) throw std::runtime_error("DMA bounds");
 std::memcpy(dst.p,src.p,c.blockLen);
 std::fill(dst.p+n,dst.p+n+extra,pad.paddingValue);
}
struct bfloat16_t {uint16_t bits=0;};
inline uint16_t toBf(float x) {
 uint32_t u;std::memcpy(&u,&x,4);
 if((u&0x7f800000u)==0x7f800000u) return uint16_t((u>>16)|((u&0x7fffffu)?0x40u:0u));
 return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);
}
inline float fromBf(uint16_t b) {uint32_t u=uint32_t(b)<<16;float f;std::memcpy(&f,&u,4);return f;}
namespace MicroAPI {
enum class RegLayout {ZERO}; enum class SatMode {SAT,UNKNOWN};
enum class MaskMergeMode {ZEROING}; enum class MaskPattern {ALL};
enum class LoadDist {DIST_NORM,DIST_BRC_B32}; enum class StoreDist {DIST_NORM_B32};
struct CastTrait {RegLayout layout;SatMode sat;MaskMergeMode mask;RoundMode round;};
template<class T> struct RegTensor {std::array<T,128> x{};};
struct MaskReg {std::array<bool,64> x{};};
template<class T,MaskPattern> MaskReg CreateMask() {MaskReg p;p.x.fill(true);return p;}
template<MemType,MemType> void LocalMemBar() {}
template<class T> void Duplicate(RegTensor<T>& v,T a) {v.x.fill(a);}
template<class T,LoadDist L=LoadDist::DIST_NORM> void LoadAlign(RegTensor<T>& v,T* p) {
 if(!p) throw std::runtime_error("absent VF load");
 for(int i=0;i<64;++i) v.x[i]=p[L==LoadDist::DIST_BRC_B32?0:i];
}
template<class T,StoreDist> void StoreAlign(T* p,RegTensor<T>& v,MaskReg& m) {
 for(int i=0;i<64;++i) if(m.x[i]) p[i]=v.x[i];
}
template<class T,CMPMODE C> void Compares(MaskReg& out,RegTensor<T>& a,T b,MaskReg& mask) {
 for(int i=0;i<64;++i) out.x[i]=mask.x[i] && (C==CMPMODE::GE?a.x[i]>=b:C==CMPMODE::LE?a.x[i]<=b:a.x[i]==b);
}
template<class T> void Select(RegTensor<T>& out,RegTensor<T>& a,RegTensor<T>& b,MaskReg& m) {
 for(int i=0;i<64;++i) out.x[i]=m.x[i]?a.x[i]:b.x[i];
}
#define BINARY(NAME,EXPR) inline void NAME(RegTensor<float>& o,RegTensor<float>& a,RegTensor<float>& b,MaskReg& p) { \
 for(int i=0;i<64;++i) if(p.x[i]) {const float x=a.x[i],y=b.x[i];o.x[i]=(EXPR);} }
BINARY(Max,std::fmax(x,y))
BINARY(ExpSub,std::exp(float(x-y)))
BINARY(Mul,x*y)
BINARY(Add,x+y)
BINARY(Div,x/y)
#undef BINARY
template<class TO,class FROM,const CastTrait& TRAIT> void Cast(RegTensor<TO>& out,RegTensor<FROM>& in,MaskReg& m) {
 static_assert(TRAIT.layout==RegLayout::ZERO,"model only supports ZERO layout");
 for(int i=0;i<64;++i) if(m.x[i]) {
  if constexpr(std::is_same<TO,bfloat16_t>::value) out.x[2*i].bits=toBf(in.x[i]);
  else out.x[i]=fromBf(in.x[2*i].bits);
 }
}
}
}
'''

DRIVER = r'''
#include <iostream>
#include <cstdlib>
#include "cold_state_merge_device.h"
using namespace AscendC;
void read(std::vector<float>& v) {std::cin.read(reinterpret_cast<char*>(v.data()),v.size()*4);}
int main(int argc,char** argv) {
 try {
  int rows=std::atoi(argv[1]),offset=std::atoi(argv[2]),stride=std::atoi(argv[3]);
  int cp=std::atoi(argv[4]),hp=std::atoi(argv[5]);
  std::vector<float> hot(rows*stride),hs(rows),hm(rows),cin(rows*512),csin(rows),cmin(rows);
  read(hot);read(hs);read(hm);read(cin);read(csin);read(cmin);
  if(!std::cin) throw std::runtime_error("input");
  std::vector<float> cold((offset+rows)*512,-12345),cs(offset+rows,-1),cm(offset+rows,-1);
  std::copy(cin.begin(),cin.end(),cold.begin()+offset*512);
  std::copy(csin.begin(),csin.end(),cs.begin()+offset);std::copy(cmin.begin(),cmin.end(),cm.begin()+offset);
  std::vector<uint8_t> cps(offset/64+1,0),hps(offset/64+1,0);cps.back()=cp;hps.back()=hp;
  TPipe pipe;ColdMergeDeviceScratch scratch;scratch.Init(&pipe);
  LocalTensor<float> ho{hot.data(),hot.size()},hSum{hp?hs.data():nullptr,hs.size()},hMax{hp?hm.data():nullptr,hm.size()};
  MergeColdStateDevice(ho,hSum,hMax,cp?(uint8_t*)cold.data():nullptr,cp?(uint8_t*)cm.data():nullptr,
   cp?(uint8_t*)cs.data():nullptr,cps.data(),hps.data(),offset,64,rows,512,stride,scratch);
  for(auto& events:pending) for(int e:events) if(e) throw std::runtime_error("pending event");
  std::cout.write(reinterpret_cast<char*>(hot.data()),hot.size()*4);
  int counters[3]={dmaCalls,scalarReads,allocatedBytes};std::cout.write(reinterpret_cast<char*>(counters),sizeof(counters));
 } catch(const std::exception& e) {std::cerr<<e.what();return 2;}
}
'''


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def bf16(x):
    u = struct.unpack("I", struct.pack("f", x))[0]
    if u & 0x7f800000 == 0x7f800000:
        u = (u & 0xffff0000) | (0x400000 if u & 0x7fffff else 0)
    else:
        u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000
    return struct.unpack("f", struct.pack("I", u))[0]


class DeviceMergeReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("Host C++ compiler unavailable; no device claim")
        cls.compiler = compiler
        cls.tmp = tempfile.TemporaryDirectory(prefix="sfa-device-header-")
        root = Path(cls.tmp.name)
        (root / "kernel_operator.h").write_text(STUB)
        (root / "driver.cpp").write_text(DRIVER)
        cls.exe = root / "driver"
        subprocess.run([compiler, "-std=c++17", "-O0", "-ffp-contract=off", "-I", str(root),
                        "-I", str(HEADER.parent), str(root / "driver.cpp"), "-o", str(cls.exe)],
                       check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def case(self, rows=7, offset=69, stride=544, cp=1, hp=1):
        self.assertLessEqual(offset % 64 + rows, 64)
        rng = random.Random(11)
        return dict(rows=rows, offset=offset, stride=stride, cp=cp, hp=hp,
                    hot=[f32(rng.uniform(-3, 3)) for _ in range(rows*stride)],
                    hs=[f32(rng.uniform(1,128)) for _ in range(rows)],
                    hm=[f32(rng.uniform(-5,5)) for _ in range(rows)],
                    cold=[f32(rng.uniform(-4,4)) for _ in range(rows*512)],
                    cs=[f32(rng.uniform(1,2048)) for _ in range(rows)],
                    cm=[f32(rng.uniform(-5,5)) for _ in range(rows)])

    def run_header(self, c):
        values = sum((c[k] for k in ("hot", "hs", "hm", "cold", "cs", "cm")), [])
        proc = subprocess.run([str(self.exe), *(str(c[k]) for k in ("rows","offset","stride","cp","hp"))],
                              input=struct.pack(f"{len(values)}f", *values), capture_output=True, check=True)
        output = list(struct.unpack(f"{c['rows']*c['stride']}f", proc.stdout[:-12]))
        dma, reads, allocated = struct.unpack("3i", proc.stdout[-12:])
        self.assertEqual(dma, 3*((c["rows"]+3)//4) if c["cp"] else 0)
        self.assertEqual(reads, 2)
        self.assertEqual(allocated, 4*512*4+64)
        return output

    def reference(self, c, row, col):
        cp, hp = bool(c["cp"]), bool(c["hp"])
        if not cp and not hp:
            return 0.
        for present, m, s, upper in ((cp,c["cm"][row],c["cs"][row],2048.25),
                                      (hp,c["hm"][row],c["hs"][row],128.02)):
            if present and (not math.isfinite(m) or not math.isfinite(s) or not .9999 <= s <= upper):
                return math.nan
        cold = c["cold"][row*512+col] if cp else 0.
        hot = bf16(c["hot"][row*c["stride"]+col]) if hp else 0.
        if not hp:
            return cold
        if not cp:
            return hot
        # Independent double log-domain reference, including extreme maxima.
        shift = max(c["cm"][row], c["hm"][row])
        lc = (c["cm"][row]-shift) + math.log(c["cs"][row])
        lh = (c["hm"][row]-shift) + math.log(c["hs"][row])
        logshift = max(lc, lh)
        wc, wh = math.exp(lc-logshift), math.exp(lh-logshift)
        return cold*(wc/(wc+wh)) + hot*(wh/(wc+wh))

    def assert_matches(self, c):
        actual = self.run_header(c)
        for row in range(c["rows"]):
            for col in range(512):
                expect = self.reference(c, row, col)
                result = actual[row*c["stride"]+col]
                if math.isnan(expect):
                    self.assertTrue(math.isnan(result), (row,col,result))
                elif math.isinf(expect):
                    self.assertEqual(expect, result)
                else:
                    self.assertAlmostEqual(result, expect, delta=3e-6*max(1,abs(expect)))
            a=row*c["stride"]+512;b=(row+1)*c["stride"]
            self.assertEqual(actual[a:b],c["hot"][a:b], "must not overwrite UB stride padding")
        return actual

    def test_mixed_head_offset_tail_chunk_and_padding(self):
        self.assert_matches(self.case())

    def test_presence_bytes_and_absent_nan_states(self):
        for cp,hp in ((0,0),(0,255),(128,0),(2,254)):
            with self.subTest(cp=cp,hp=hp):
                c=self.case(cp=cp,hp=hp)
                if not cp:
                    for k in ("cold","cs","cm"): c[k]=[math.nan]*len(c[k])
                if not hp:
                    for k in ("hs","hm"): c[k]=[math.nan]*len(c[k])
                    for r in range(c["rows"]): c["hot"][r*c["stride"]:r*c["stride"]+512]=[math.nan]*512
                self.assert_matches(c)

    def test_zero_query_counts(self):
        c=self.case();c["hm"]=[0.]*c["rows"];c["cm"]=[0.]*c["rows"]
        c["hs"]=[128.]*c["rows"];c["cs"]=[1920.]*c["rows"]
        self.assert_matches(c)

    def test_extreme_finite_maxima(self):
        c=self.case(rows=4,offset=32)
        c["hm"]=[f32(3e38),f32(-3e38),f32(3e38),-1000.]
        c["cm"]=[f32(-3e38),f32(3e38),f32(3e38),1000.]
        self.assert_matches(c)

    def test_invalid_present_statistics_including_single_side(self):
        for cp,hp,key in ((1,1,"cs"),(1,1,"hs"),(1,0,"cs"),(0,1,"hs")):
            for bad in (0.,-1.,math.nan,math.inf,65536.):
                with self.subTest(cp=cp,hp=hp,key=key,bad=bad):
                    c=self.case(rows=1,offset=127,cp=cp,hp=hp);c[key][0]=bad
                    self.assert_matches(c)
        for key in ("cm","hm"):
            for bad in (math.inf,-math.inf,math.nan):
                c=self.case(rows=1,offset=127);c[key][0]=bad;self.assert_matches(c)

    def test_hot_only_rounding_and_cold_only_identity(self):
        for cp,hp in ((0,1),(1,0)):
            c=self.case(rows=1,offset=63,stride=512,cp=cp,hp=hp)
            c["hot"][:4]=[1.00390625,1.01171875,-1.00390625,-1.01171875]
            self.assert_matches(c)

    def test_present_nonfinite_outputs_are_not_sanitized_at_zero_weight(self):
        c=self.case(rows=1,offset=0,stride=512)
        c["hm"]=[0.];c["cm"]=[1000.];c["hot"][0]=math.inf;c["hot"][1]=math.nan
        out=self.assert_matches(c)
        self.assertTrue(math.isnan(out[0]) and math.isnan(out[1]))

    def test_header_is_vector_epilogue_not_host_math_fallback(self):
        text=HEADER.read_text()
        for forbidden in ("std::exp", "std::log", "std::isfinite", "GetValue(column)", "column++"):
            self.assertNotIn(forbidden,text)
        self.assertIn("column += 64",text)
        self.assertIn("CAST_ROUND",text)
        self.assertNotIn("CAST_RINT",text)
        self.assertIn("NOT yet compiled or measured with the A5 SDK",text)

    def test_cube_parse_requires_no_vector_api_or_runtime_fallback(self):
        # Model the AIC template parse with NO MicroAPI/TPipe implementation.
        # This would fail if the non-template dispatcher instantiated its VFs.
        with tempfile.TemporaryDirectory(prefix="sfa-cube-parse-") as tmp:
            root=Path(tmp)
            (root/"kernel_operator.h").write_text('''#pragma once
#include <cstdint>
#define __aicore__
#define __gm__
namespace AscendC { struct TPipe; template<class T> struct LocalTensor; }
''')
            (root/"cube.cpp").write_text('''#include "cold_state_merge_device.h"
using Signature=decltype(&AscendC::MergeColdStateDevice);
static_assert(sizeof(AscendC::ColdMergeDeviceScratch)>0,"complete member type");
template<class T> struct UninstantiatedVectorService {
 AscendC::ColdMergeDeviceScratch scratch;
 void Init(AscendC::TPipe* pipe) {scratch.Init(pipe);}
};
''')
            subprocess.run([self.compiler,"-std=c++17","-D__DAV_C310_CUBE__","-fsyntax-only",
                            "-I",str(root),"-I",str(HEADER.parent),str(root/"cube.cpp")],
                           check=True,capture_output=True,text=True)
        cube=HEADER.read_text().split("#if defined(__DAV_C310_CUBE__)",1)[1].split("#else",1)[0]
        self.assertNotIn("MicroAPI::",cube)
        self.assertNotIn("return;",cube)


if __name__ == "__main__":
    unittest.main()
