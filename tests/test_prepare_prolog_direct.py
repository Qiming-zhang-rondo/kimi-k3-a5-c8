"""Stage a synthetic operator only; never access private code or an SDK."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_prolog_direct", ROOT / "tools/prepare_prolog_direct.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

KEY = '''#pragma once
#include "ascendc/host_api/tiling/template_argument.h"
#define KEEP_MODE 7
enum class RealMode { first = KEEP_MODE, second = 11 };
// ASCENDC_TPL_SEL(unbalanced comment ( must not be parsed
ASCENDC_TPL_ARGS_DECL(Operator, NESTED(1, "paren ) and comma ,", R"raw(()))raw"));
ASCENDC_TPL_SEL(
    SELECT(NESTED(1, 2)), /* parentheses ( ) ignored */
    SELECT(3)
);
'''
ENTRY = '''#include "kernel_operator.h"
#include "key.h"
namespace optiling { struct alignas(8) RealPod { uint32_t first; uint32_t second; }; }
void unchanged_signature(GM_ADDR tiling, GM_ADDR workspace) {
    GET_TILING_DATA_WITH_STRUCT(optiling::RealPod, tilingDataIn, tiling);
    (void)tilingDataIn; (void)workspace;
}
'''


class PreparePrologDirectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "private-test-operator"
        self.source.mkdir()
        (self.source / "key.h").write_text(KEY)
        (self.source / "kernel.cpp").write_text(ENTRY)
        self.output = self.root / "local-only-stage"

    def prepare(self, **kwargs):
        return MODULE.prepare(self.source, "kernel.cpp", "key.h", self.output, **kwargs)

    def test_balanced_registration_and_source_bytes_are_preserved(self):
        dependency = self.root / "shared-dependency.h"
        dependency.write_text("// local dependency\n")
        (self.source / "dependency.h").symlink_to(dependency)
        manifest = self.prepare()
        self.assertEqual((self.source / "key.h").read_text(), KEY)
        self.assertEqual((self.source / "kernel.cpp").read_text(), ENTRY)
        clone = Path(manifest["cloned_source"])
        staged = (clone / "kernel.asc").read_text()
        self.assertIn("void unchanged_signature(GM_ADDR tiling, GM_ADDR workspace)", staged)
        self.assertIn("qsfa_direct::LoadTilingWords(tilingDataIn, tiling);", staged)
        self.assertIn("__DAV_C310_CUBE__", staged)
        self.assertIn("__DAV_C310_VEC__", staged)
        self.assertIn("GET_TILING_DATA_WITH_STRUCT(optiling::RealPod, tilingDataIn, tiling);", staged)
        self.assertIn('enum class RealMode { first = KEEP_MODE, second = 11 };', (clone / "key.h").read_text())
        self.assertIn('ASCENDC_TPL_ARGS_DECL(Operator, NESTED(1, "paren ) and comma ,", R"raw(()))raw"));',
                      (clone / "key.h").read_text())
        self.assertTrue((clone / "dependency.h").is_symlink())
        self.assertEqual(manifest["disclosure"], "DO_NOT_UPLOAD")
        self.assertEqual(manifest["tiling_load_statements"], 1)
        self.assertEqual(manifest["guarded_template_registration_blocks"], 2)
        self.assertFalse(manifest["source_modified"])
        self.assertEqual(json.loads((self.output / "DO_NOT_UPLOAD_manifest.json").read_text())["status"],
                         "LOCAL_STAGE_PREPARED_NOT_COMPILED")
        with self.assertRaisesRegex(ValueError, "NEW output"):
            self.prepare()

    def test_preprocessing_keeps_host_registration_and_loads_only_real_device_passes(self):
        compiler = shutil.which("c++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("Host preprocessor unavailable; no ASC/device claim")
        # A following statement on the same physical line must not become an
        # accidental operand of the generated #endif directive.
        (self.source / "kernel.cpp").write_text(ENTRY.replace(
            "tilingDataIn, tiling);", "tilingDataIn, tiling); (void)workspace;"))
        manifest = self.prepare()
        clone = Path(manifest["cloned_source"])
        (clone / "kernel_operator.h").write_text('''#pragma once
#include <cstdint>
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
#define GET_TILING_DATA_WITH_STRUCT(T,V,P) T V{}
''')
        host = clone / "ascendc/host_api/tiling/template_argument.h"
        host.parent.mkdir(parents=True)
        host.write_text('''#if defined(QSFA_ASC_DIRECT_BUILD) && QSFA_ASC_DIRECT_BUILD == 1
#error host_template_header_leaked_into_direct_build
#endif
#define ASCENDC_TPL_ARGS_DECL(...) int ORIGINAL_HOST_REGISTRATION;
#define ASCENDC_TPL_SEL(...) int ORIGINAL_HOST_SELECTION;
''')
        for phase in ("normal", "direct-host", "cube", "vector"):
            flags = [] if phase == "normal" else ["-DQSFA_ASC_DIRECT_BUILD=1"]
            if phase in ("cube", "vector"):
                flags += ["-D__NPU_ARCH__=3510", "-D" + (
                    "__DAV_C310_CUBE__" if phase == "cube" else "__DAV_C310_VEC__")]
            result = subprocess.run([compiler, "-std=c++17", "-E", "-P", "-x", "c++",
                "-I" + str(clone), *flags, str(clone / "kernel.asc")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("first = 7, second = 11", result.stdout)
            self.assertEqual("int ORIGINAL_HOST_REGISTRATION" in result.stdout, phase == "normal")
            self.assertEqual("qsfa_direct::LoadTilingWords(tilingDataIn, tiling)" in result.stdout,
                             phase in ("cube", "vector"))

    def test_unknown_macro_format_symlinks_and_oversized_source_refuse_before_output(self):
        for replacement in ("reinterpret_cast<GM_ADDR>(tiling)", "args.tiling", "tiling + 4"):
            (self.source / "kernel.cpp").write_text(ENTRY.replace(
                "tilingDataIn, tiling)", "tilingDataIn, " + replacement + ")"))
            with self.assertRaisesRegex(ValueError, "identifier"):
                self.prepare()
            self.assertFalse(self.output.exists())
        (self.source / "kernel.cpp").write_text(ENTRY)
        (self.source / "key.h").unlink()
        real_header = self.root / "linked-key.h"
        real_header.write_text(KEY)
        (self.source / "key.h").symlink_to(real_header)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.prepare()
        self.assertFalse(self.output.exists())
        (self.source / "key.h").unlink()
        (self.source / "key.h").write_text(KEY)
        with (self.source / "too-large.data").open("wb") as stream:
            stream.truncate(MODULE.MAX_SOURCE_BYTES)
        with self.assertRaisesRegex(ValueError, "256 MiB"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_missing_templates_and_device_constant_inventory_are_explicit(self):
        (self.source / "key.h").write_text(KEY.replace("ASCENDC_TPL_SEL(", "OTHER_SEL("))
        with self.assertRaisesRegex(ValueError, "ARGS_DECL and SEL"):
            self.prepare()
        (self.source / "key.h").write_text(KEY)
        (self.source / "kernel.cpp").write_text(ENTRY + "int kind = ASCENDC_TPL_MIX_AIC_1_1;\n")
        constants = self.root / "real-extracted-constants.h"
        constants.write_text("// test value only; never a real SDK constant claim\nenum { ASCENDC_TPL_MIX_AIC_1_1 = 12345 };\n")
        manifest = self.prepare(device_constants_header=constants)
        self.assertEqual(manifest["remaining_template_symbol_tokens_to_verify"], ["ASCENDC_TPL_MIX_AIC_1_1"])
        self.assertTrue(manifest["device_constants_header_supplied"])
        self.assertIn("never guess", manifest["constants_instruction"])
        self.assertEqual((Path(manifest["cloned_source"]) / "qsfa_actual_device_constants.h").read_bytes(),
                         constants.read_bytes())


if __name__ == "__main__":
    unittest.main()
