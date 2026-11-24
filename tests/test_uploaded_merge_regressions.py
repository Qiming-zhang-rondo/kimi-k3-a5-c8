"""Compile and call the actual merge header using a bounds-checked host stub.

These are host C++ regression probes, not an Ascend compiler, kernel pipeline,
native binding, or NPU validation. Tests assert the intended corrected ABI and
behavior: a header with the original query/head mapping or silent-invalid-state
handling fails. No Torch, NumPy, or compiler-generated fake merge is used.
"""

import ctypes
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FLOAT = ctypes.c_float
FPTR = ctypes.POINTER(FLOAT)
BYTE = ctypes.c_uint8
BPTR = ctypes.POINTER(BYTE)
I64 = ctypes.c_int64


DRIVER = r'''
#include <cstdio>
#include <cstring>
#include <exception>
#include "cold_state_merge.h"

// Host-only BF16 round-to-nearest-even value type. This exercises the actual
// HOT_OUTPUT_T cast in the header; it does not verify AscendC Cast semantics.
class HostBFloat16Round {
public:
    explicit HostBFloat16Round(float value)
    {
        uint32_t word;
        std::memcpy(&word, &value, sizeof(word));
        if ((word & 0x7fffffffU) > 0x7f800000U) {
            bits_ = static_cast<uint16_t>((word >> 16) | 0x0040U);
        } else {
            bits_ = static_cast<uint16_t>((word + 0x7fffU + ((word >> 16) & 1U)) >> 16);
        }
    }
    explicit operator float() const
    {
        uint32_t word = static_cast<uint32_t>(bits_) << 16;
        float value;
        std::memcpy(&value, &word, sizeof(value));
        return value;
    }
private:
    uint16_t bits_;
};

template <typename COLD_T, typename HOT_T>
int Invoke(float *hot, int64_t hotElements, float *sum, float *maximum,
           int64_t statElements, const void *cold, const float *coldMaximum,
           const float *coldSum, const uint8_t *coldPresent,
           const uint8_t *hotPresent, int64_t flatRowOffset, int64_t heads,
           int64_t rows, int64_t dim, int64_t ubStride,
           char *error, int64_t errorCapacity)
{
    try {
        AscendC::LocalTensor<float> hotTensor(hot, hotElements);
        AscendC::LocalTensor<float> sumTensor(sum, statElements);
        AscendC::LocalTensor<float> maximumTensor(maximum, statElements);
        AscendC::MergeColdStateTile<float, COLD_T, HOT_T>(
            hotTensor, sumTensor, maximumTensor,
            reinterpret_cast<const uint8_t *>(cold),
            reinterpret_cast<const uint8_t *>(coldMaximum),
            reinterpret_cast<const uint8_t *>(coldSum), coldPresent, hotPresent,
            flatRowOffset, heads, rows, dim, ubStride);
        return 0;
    } catch (const std::exception &failure) {
        std::snprintf(error, errorCapacity, "%s", failure.what());
        return 1;
    }
}

#define WRAPPER(NAME, TYPE, HOT_TYPE) \
extern "C" int NAME(float *hot, int64_t hotElements, float *sum, float *maximum, \
                    int64_t statElements, const void *cold, const float *coldMaximum, \
                    const float *coldSum, const uint8_t *coldPresent, \
                    const uint8_t *hotPresent, int64_t flatRowOffset, int64_t heads, \
                    int64_t rows, int64_t dim, int64_t ubStride, \
                    char *error, int64_t errorCapacity) \
{ \
    return Invoke<TYPE, HOT_TYPE>(hot, hotElements, sum, maximum, statElements, cold, \
                        coldMaximum, coldSum, coldPresent, hotPresent, flatRowOffset, \
                        heads, rows, dim, ubStride, error, errorCapacity); \
}

WRAPPER(merge_float, float, float)
// A different-width cold dtype is a host template/stride probe. It does not
// claim that a native ACLNN binding accepts FP64 cold output.
WRAPPER(merge_double, double, float)
WRAPPER(merge_float_bf16hot, float, HostBFloat16Round)
'''


def f32(value):
    return FLOAT(value).value


def f32_array(values):
    return (FLOAT*len(values))(*values)


class CompiledHeaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            raise unittest.SkipTest("A host C++ compiler is required for actual-header regression probes")
        cls.workspace = tempfile.TemporaryDirectory(prefix="sfa-merge-host-probes-")
        cls.addClassCleanup(cls.workspace.cleanup)
        path = Path(cls.workspace.name)
        driver = path/"probe.cpp"
        driver.write_text(DRIVER, encoding="utf-8")
        library = path/("probe.dylib" if sys.platform == "darwin" else "probe.so")
        result = subprocess.run(
            [compiler, "-std=c++17", "-O0", "-shared", "-fPIC", str(driver),
             "-I", str(ROOT/"arch35"), "-I", str(ROOT/"tests"/"cpp_stubs"),
             "-o", str(library)], capture_output=True, text=True)
        if result.returncode:
            raise AssertionError("Actual cold_state_merge.h failed host compilation:\n"+result.stdout+result.stderr)
        cls.library = ctypes.CDLL(str(library))
        for name in ("merge_float", "merge_double", "merge_float_bf16hot"):
            function = getattr(cls.library, name)
            function.argtypes = [FPTR, I64, FPTR, FPTR, I64, ctypes.c_void_p,
                                 FPTR, FPTR, BPTR, BPTR, I64, I64, I64, I64, I64,
                                 ctypes.c_char_p, I64]
            function.restype = ctypes.c_int

    def invoke(self, hot, cold, *, hot_sum=None, cold_sum=None, hot_max=None, cold_max=None,
               cold_present=None, hot_present=None, offset=0, heads=1, dim=2,
               stride=None, cold_dtype="float", round_hot_bf16=False):
        stride = dim if stride is None else stride
        if len(hot) % stride or len(cold) % dim:
            raise ValueError("Probe arrays must contain whole UB/GM rows")
        rows, cold_rows = len(hot)//stride, len(cold)//dim
        # Pad GM and presence allocations even for the legacy wrong mapping,
        # so a regression gives wrong values rather than host undefined reads.
        allocated_rows = max(cold_rows, (offset+rows+1)*heads)
        tokens = (allocated_rows+heads-1)//heads
        actual_hot = f32_array(hot)
        hs = f32_array([1.]*rows if hot_sum is None else hot_sum)
        hm = f32_array([0.]*rows if hot_max is None else hot_max)
        ctype = FLOAT if cold_dtype == "float" else ctypes.c_double
        cold_values = list(cold)+[779.]*((allocated_rows-cold_rows)*dim)
        cold_storage = (ctype*len(cold_values))(*cold_values)
        cs_values = [1.]*cold_rows if cold_sum is None else list(cold_sum)
        cm_values = [0.]*cold_rows if cold_max is None else list(cold_max)
        cs = f32_array(cs_values+[1.]*(allocated_rows-len(cs_values)))
        cm = f32_array(cm_values+[0.]*(allocated_rows-len(cm_values)))
        logical_tokens = (cold_rows+heads-1)//heads
        hp_values = [1]*logical_tokens if hot_present is None else list(hot_present)
        cp_values = [1]*logical_tokens if cold_present is None else list(cold_present)
        hp = (BYTE*tokens)(*(hp_values+[0]*(tokens-len(hp_values))))
        cp = (BYTE*tokens)(*(cp_values+[0]*(tokens-len(cp_values))))
        error = ctypes.create_string_buffer(512)
        function = "merge_"+cold_dtype+("_bf16hot" if round_hot_bf16 else "")
        status = getattr(self.library, function)(
            actual_hot, len(actual_hot), hs, hm, rows, cold_storage, cm, cs, cp, hp,
            offset, heads, rows, dim, stride, error, len(error))
        if status:
            self.fail("Actual header raised a host bounds/error probe: "+error.value.decode())
        return list(actual_hot)

    def assertClose(self, actual, expected, atol=2e-6, rtol=2e-6):
        self.assertEqual(len(actual), len(expected))
        for index, (value, wanted) in enumerate(zip(actual, expected)):
            self.assertTrue(math.isfinite(value), (index, value, wanted))
            self.assertLessEqual(abs(value-wanted), atol+rtol*abs(wanted), (index, value, wanted))

    def test_flattened_m_advances_heads_then_tokens(self):
        # Q4,H64, half-tile32: all rows belong to query0, heads0..31.
        dim, heads, queries, rows = 2, 64, 4, 32
        cold = [1000.*query+10.*head+feature for query in range(queries)
                for head in range(heads) for feature in range(dim)]
        actual = self.invoke([float("nan")]*rows*dim, cold, heads=heads, dim=dim,
                             hot_present=[0, 0, 0, 0], cold_present=[1, 1, 1, 1])
        self.assertClose(actual, cold[:rows*dim], atol=0, rtol=0)

    def test_flattened_offset_crosses_token_boundary_and_presence_uses_query(self):
        dim, heads, offset, rows = 3, 4, 3, 4
        cold = [float(i+100) for i in range(3*heads*dim)]
        hot = [float(i+1) for i in range(rows*dim)]
        actual = self.invoke(hot, cold, offset=offset, heads=heads, dim=dim,
                             hot_present=[1, 0, 1], cold_present=[0, 1, 0])
        self.assertClose(actual, hot[:dim]+cold[(offset+1)*dim:(offset+rows)*dim], atol=0, rtol=0)

    def test_cold_gm_dim_is_independent_of_ub_stride_and_padding_is_zero(self):
        actual = self.invoke([float("nan")]*10, [1., 2., 3., 4., 5., 6.],
                             dim=3, stride=5, hot_present=[0, 0], cold_present=[1, 1])
        self.assertClose(actual, [1., 2., 3., 0., 0., 4., 5., 6., 0., 0.], atol=0, rtol=0)

    def test_independent_cold_template_dtype_uses_correct_width(self):
        actual = self.invoke([float("nan")]*4, [1.25, -2.5, 7.75, -8.125], dim=2,
                             hot_present=[0, 0], cold_present=[1, 1], cold_dtype="double")
        self.assertClose(actual, [1.25, -2.5, 7.75, -8.125], atol=0, rtol=0)

    def test_hot_output_rounding_specialization_preserves_cold_fp32(self):
        # Exact BF16 midpoint pairs: ties round to even in opposite directions.
        hot = [1.00390625, 1.01171875]
        rounded_hot = [1., 1.015625]
        cold = [f32(1.002), f32(-2.003)]
        actual = self.invoke(hot, cold, round_hot_bf16=True)
        self.assertClose(actual, [f32((h+c)/2) for h, c in zip(rounded_hot, cold)], atol=0, rtol=0)
        hot_only = self.invoke(hot, cold, round_hot_bf16=True, cold_present=[0])
        self.assertClose(hot_only, rounded_hot, atol=0, rtol=0)
        cold_only = self.invoke(hot, cold, round_hot_bf16=True, hot_present=[0])
        self.assertClose(cold_only, cold, atol=0, rtol=0)

    def test_normal_merge_uses_unnormalized_sum_not_softmax_sum_one(self):
        # Q=0 scores -> segment masses128 and1920; expectedhotweight1/16.
        actual = self.invoke([2., 6.], [10., -2.], hot_sum=[128.], cold_sum=[1920.])
        self.assertClose(actual, [9.5, -1.5])

    def test_invalid_present_statistics_propagate_nan_in_all_presence_modes(self):
        for side in ("hot", "cold"):
            for attribute, invalid_values in (("sum", [0., -1., float("inf"), -float("inf"), float("nan")]),
                                               ("max", [float("inf"), -float("inf"), float("nan")])):
                for invalid in invalid_values:
                    for both in (False, True):
                        with self.subTest(side=side, attribute=attribute, invalid=invalid, both=both):
                            kwargs = {side+"_"+attribute: [invalid],
                                      "hot_present": [1 if side == "hot" or both else 0],
                                      "cold_present": [1 if side == "cold" or both else 0]}
                            actual = self.invoke([2., 4.], [6., 8.], **kwargs)
                            self.assertTrue(all(math.isnan(x) for x in actual), actual)

    def test_negative_sum_must_not_cancel_positive_mass_into_finite_output(self):
        actual = self.invoke([2., 4.], [6., 8.], hot_sum=[2.], cold_sum=[-1.])
        self.assertTrue(all(math.isnan(x) for x in actual), actual)

    def test_both_empty_ignores_nan_payload_and_statistics(self):
        invalid = float("nan")
        actual = self.invoke([invalid, invalid], [invalid, invalid], hot_sum=[invalid],
                             cold_sum=[invalid], hot_max=[invalid], cold_max=[invalid],
                             hot_present=[0], cold_present=[0])
        self.assertClose(actual, [0., 0.], atol=0, rtol=0)

    def test_one_empty_ignores_invalid_other_state_but_preserves_valid_output(self):
        invalid = float("nan")
        actual = self.invoke([1.125, -2.25], [invalid, invalid], cold_sum=[invalid],
                             cold_max=[invalid], hot_present=[1], cold_present=[0])
        self.assertClose(actual, [1.125, -2.25], atol=0, rtol=0)
        actual = self.invoke([invalid, invalid], [3.25, -4.75], hot_sum=[invalid],
                             hot_max=[invalid], hot_present=[0], cold_present=[1])
        self.assertClose(actual, [3.25, -4.75], atol=0, rtol=0)

    def test_finite_positive_sums_do_not_overflow_combined_normalizer(self):
        large = f32(3e38)
        actual = self.invoke([1., 5.], [3., -1.], hot_sum=[large], cold_sum=[large])
        self.assertClose(actual, [2., 2.])

    def test_large_equal_maxima_keep_sum_ratio_when_forming_log_weights(self):
        # max+log(sum) before subtracting a common max would lose the log ratio.
        maximum = f32(1e30)
        actual = self.invoke([1., 5.], [3., -1.], hot_sum=[1.], cold_sum=[3.],
                             hot_max=[maximum], cold_max=[maximum])
        self.assertClose(actual, [2.5, .5])

    def test_underflowed_weight_does_not_mask_nonfinite_present_output(self):
        actual = self.invoke([1., 5.], [float("nan"), float("nan")],
                             hot_max=[1000.], cold_max=[-1000.])
        self.assertTrue(all(math.isnan(x) for x in actual), actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
