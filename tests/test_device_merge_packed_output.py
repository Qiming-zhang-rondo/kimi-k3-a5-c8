"""Execute the actual optional packed epilogue under the existing CPU model.

These tests validate address overlap, final RNE and the unchanged row arithmetic.
They do not validate CANN compilation, VF register scheduling or A5 performance.
"""
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

import test_device_merge_reference as merge_reference

HEADER = merge_reference.HEADER
bf16 = merge_reference.bf16


class PackedMergeOutputTest(unittest.TestCase):
    effective_tile_rows = 16
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if not cls.compiler:
            raise unittest.SkipTest("Host C++ compiler unavailable; no SDK/device claim")
        cls.tmp = tempfile.TemporaryDirectory(prefix="sfa-packed-merge-")
        root = Path(cls.tmp.name)
        cls.exes = {}
        for packed in (0, 1):
            build = root / str(packed)
            build.mkdir()
            cls.exes[packed] = merge_reference.DeviceMergeBf16ReferenceTest.compile_driver(
                cls.compiler, build, True, 16, packed_output=packed,
                merge_namespace="Bf16PackedColdMerge" if packed else None)
        cls.cases = merge_reference.DeviceMergeBf16ReferenceTest()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_header(self, c, packed):
        values = sum((c[k] for k in ("hot", "hs", "hm", "cold", "cs", "cm")), [])
        proc = subprocess.run([str(self.exes[packed]), *(str(c[k]) for k in
                              ("rows", "offset", "stride", "cp", "hp"))],
                              input=struct.pack(f"{len(values)}f", *values),
                              capture_output=True, check=True)
        dma, scalar_reads, allocated = struct.unpack("3i", proc.stdout[-12:])
        tile = self.effective_tile_rows
        stats_stride = (tile+7)//8*8
        self.assertEqual((dma, scalar_reads, allocated),
                         (3*((c["rows"]+tile-1)//tile) if c["cp"] else 0,
                          2, tile*512*2 + 2*stats_stride*4))
        return proc.stdout[:-12]

    def check(self, c):
        original = struct.pack(f"{len(c['hot'])}f", *c["hot"])
        baseline = struct.unpack(f"{len(c['hot'])}f", self.run_header(c, 0))
        packed_bytes = self.run_header(c, 1)
        count = c["rows"]*c["stride"]
        packed = [struct.unpack("f", struct.pack("I", value << 16))[0]
                  for value in struct.unpack(f"{count}H", packed_bytes[:count*2])]
        for row in range(c["rows"]):
            for col in range(512):
                idx = row*c["stride"]+col
                expected = bf16(baseline[idx])
                actual = packed[idx]
                if math.isnan(expected):
                    self.assertTrue(math.isnan(actual), (row, col))
                else:
                    # Exact bits also preserve negative zero and halfway RNE.
                    self.assertEqual(struct.pack("f", actual), struct.pack("f", expected), (row, col))
            pad_start = (row*c["stride"]+512)*2
            pad_end = (row+1)*c["stride"]*2
            self.assertEqual(packed_bytes[pad_start:pad_end], original[pad_start:pad_end],
                             "packed destination stride padding must remain untouched")
        self.assertEqual(packed_bytes[count*2:], original[count*2:],
                         "packing may not write beyond its BF16 destination extent")
        return packed

    def test_two_tiles_and_tail_have_distinct_source_and_packed_offsets(self):
        for rows in (1, 7, 15, 16, 17, 31, 32):
            with self.subTest(rows=rows):
                self.check(self.cases.case(rows=rows, offset=64, stride=544))

    def test_bf16_tile_8_and_32_keep_packed_math_and_true_scratch(self):
        old_exes, old_tile = self.exes, self.effective_tile_rows
        try:
            for requested in (8, 32):
                with tempfile.TemporaryDirectory(prefix="sfa-packed-wide-tile-") as tmp:
                    self.exes = {}
                    self.effective_tile_rows = requested
                    for packed in (0, 1):
                        build = Path(tmp) / str(packed)
                        build.mkdir()
                        self.exes[packed] = merge_reference.DeviceMergeBf16ReferenceTest.compile_driver(
                            self.compiler, build, True, 16, packed_output=packed,
                            merge_namespace="Bf16PackedColdMerge" if packed else None,
                            bf16_tile_rows=requested)
                    for rows in (1, 17, 31, 32):
                        with self.subTest(requested=requested, effective=self.effective_tile_rows, rows=rows):
                            self.check(self.cases.case(rows=rows, offset=64, stride=544))
        finally:
            self.exes, self.effective_tile_rows = old_exes, old_tile

    def test_negative_halfway_final_rne_and_zero_query(self):
        c = self.cases.case(rows=32, offset=64, stride=512)
        c["hs"] = c["cs"] = [1.]*32
        c["hm"] = c["cm"] = [0.]*32
        for row in range(32):
            for start in range(0, 512, 4):
                c["hot"][row*512+start:row*512+start+4] = [1., 1.0078125, -1., -1.0078125]
                c["cold"][row*512+start:row*512+start+4] = [1.0078125, 1.015625, -1.0078125, -1.015625]
        result = self.check(c)
        self.assertEqual(result[:4], [1., 1.015625, -1., -1.015625])
        c["hs"], c["cs"] = [128.]*32, [1920.]*32
        self.check(c)

    def test_absent_sides_do_not_load_nil_or_nan_state(self):
        for cp,hp in ((0,0),(0,1),(1,0)):
            c = self.cases.case(rows=32, offset=64, cp=cp, hp=hp)
            if not cp:
                for key in ("cold", "cs", "cm"):
                    c[key] = [math.nan]*len(c[key])
            if not hp:
                c["hot"] = [math.nan]*len(c["hot"])
                c["hs"] = c["hm"] = [math.nan]*32
            self.check(c)

    def test_present_invalid_stats_and_zero_weight_nonfinite_survive_packing(self):
        c = self.cases.case(rows=32, offset=64)
        c["cs"][16] = math.nan
        c["hm"][17] = math.inf
        self.check(c)
        c = self.cases.case(rows=32, offset=64)
        c["hm"], c["cm"] = [0.]*32, [1000.]*32
        c["hot"][16*c["stride"]:16*c["stride"]+2] = [math.inf, math.nan]
        result = self.check(c)
        self.assertTrue(math.isnan(result[16*c["stride"]]))

    def test_exact_existing_dependency_api_and_final_cast_branch(self):
        root = HEADER.parents[2]
        dep = (root / "deps/ascendc_common/op_kernel/load_store_utils.h").read_text()
        self.assertIn("MicroAPI::SatMode::NO_SAT", dep)
        self.assertIn("AscendC::RoundMode::CAST_RINT", dep)
        self.assertIn("DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>", dep)
        helper = HEADER.read_text()
        self.assertIn("DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>", helper)
        service = HEADER.with_name("sparse_flash_attention_service_vector_mla.h").read_text()
        self.assertIn("#if !SFA_MERGE_PACKED_OUTPUT\n    Cast(attenOut, vec2ResUb, RoundMode::CAST_RINT, vec2CalcSize);", service)


if __name__ == "__main__":
    unittest.main()
