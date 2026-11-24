"""Temporary-source regression tests for read-only lexical evidence collection."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1]/"tools"/"collect_source_closure.py"
spec = importlib.util.spec_from_file_location("collect_source_closure", TOOL)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sfa-source-closure-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.bundle, self.source = self.root/"bundle", self.root/"project"
        self.bundle.mkdir()
        self.source.mkdir()

    def put(self, relative, content, root=None):
        path = (self.source if root is None else root)/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)
        return path

    def definition(self):
        return self.put("generated/tiling.h", "struct SparseFlashAttentionTilingDataMla { int actual_field; };\n")

    def run_collect(self, name="output", seeds=("seed.h",), extra=()):
        return collector.collect([self.source, *extra], self.root/name, self.bundle, required_seeds=seeds)

    def test_three_level_include_closure_preserves_bytes_paths_and_sha(self):
        self.put("original/entry.cpp", '#include "seed.h"\n', self.bundle)
        self.put("include/seed.h", '#include "next/l2.h"\n')
        self.put("include/next/l2.h", '#include "l3.h"\n')
        leaf = self.put("include/next/l3.h", b"// third level\r\n#define RAW_BYTE 0xff\r\n")
        self.definition()
        result = self.run_collect()
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["lexical_includes_all_resolved"])
        self.assertEqual(len(result["records"]), 5)
        record = next(x for x in result["records"] if x["source_path"] == str(leaf))
        copied = (self.root/"output"/record["collected_path"]).read_bytes()
        self.assertEqual(copied, leaf.read_bytes())
        self.assertEqual(record["sha256"], hashlib.sha256(copied).hexdigest())
        self.assertEqual(record["collected_path"], "collected/root1/include/next/l3.h")
        saved = json.loads((self.root/"output/source_closure.json").read_text())
        self.assertEqual(saved, result)

    def test_duplicate_basename_conflict_collects_both_and_never_selects_silently(self):
        other = self.root/"other_project"
        self.put("seed.h", "#define WHICH 1\n")
        self.put("seed.h", "#define WHICH 2\n", other)
        self.definition()
        result = self.run_collect(extra=[other])
        seed = result["required_seeds"][0]
        self.assertEqual(seed["status"], "ambiguous")
        self.assertEqual(len({x["sha256"] for x in seed["candidates"]}), 2)
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(len([x for x in result["records"] if x["relative_path"] == "seed.h"]), 2)
        self.assertTrue((self.root/"output/source_closure.json").exists())

    def test_identical_duplicate_candidates_are_explicitly_resolved(self):
        other = self.root/"other_project"
        self.put("seed.h", "#define SAME 1\n")
        self.put("seed.h", "#define SAME 1\n", other)
        self.definition()
        result = self.run_collect(extra=[other])
        self.assertEqual(result["required_seeds"][0]["status"], "resolved_identical_candidates")
        self.assertEqual(len(result["required_seeds"][0]["candidates"]), 2)
        self.assertEqual(result["exit_code"], 0)

    def test_typename_uses_forward_declarations_and_comments_are_not_definitions(self):
        self.put("seed.h", '''
struct SparseFlashAttentionTilingDataMla;
const SparseFlashAttentionTilingDataMla *tiling;
// struct SparseFlashAttentionTilingDataMla { int fake; };
const char* text = "class SparseFlashAttentionTilingDataMla { int fake; };";
REGISTER_TILING_DATA_CLASS(SparseFlashAttention, SparseFlashAttentionTilingDataMla);
''')
        result = self.run_collect()
        self.assertEqual(result["definition_matches"], [])
        self.assertEqual(result["definition_resolution"]["status"], "missing")
        self.assertEqual(result["exit_code"], 1)

    def test_complete_tiling_macro_and_typedef_bodies_are_actual_definition_matches(self):
        for index, (body, pattern) in enumerate((
            ("BEGIN_TILING_DATA_DEF(SparseFlashAttentionTilingDataMla)\n TILING_DATA_FIELD_DEF(int, x);", None),
            ("BEGIN_TILING_DATA_DEF(SparseFlashAttentionTilingDataMla)\n TILING_DATA_FIELD_DEF(int, x);\nEND_TILING_DATA_DEF;", "tiling_definition_macro_body"),
            ("BEGIN_TILING_DATA_DEF(SparseFlashAttentionTilingDataMla)\n TILING_DATA_FIELD_DEF(int, x);\nEND_TILING_DATA_DEF()", "tiling_definition_macro_body"),
            ("typedef struct { int actual; } SparseFlashAttentionTilingDataMla;", "typedef_struct_body"),
        )):
            self.put("seed.h", body)
            result = self.run_collect(name=f"output{index}")
            if pattern is None:
                self.assertEqual(result["definition_matches"], [])
            else:
                self.assertEqual(result["definition_matches"][0]["matches"][0]["pattern"], pattern)
                self.assertEqual(result["exit_code"], 0)

    def test_sdk_include_remains_unresolved_and_does_not_become_fake_source(self):
        self.put("seed.h", '#include "kernel_operator.h"\n#include <vector>\n')
        self.definition()
        result = self.run_collect()
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["status"], "REQUIRED_FOUND_WITH_OPEN_INCLUDES")
        self.assertFalse(result["lexical_includes_all_resolved"])
        self.assertEqual({x["include"] for x in result["unresolved"]}, {"kernel_operator.h", "vector"})
        self.assertTrue(all(x["status"] == "missing" and not x["candidates"] for x in result["unresolved"]))
        self.assertFalse(result["build_validated"])

    def test_qualified_inactive_branch_never_resolves_wrong_arch_basename(self):
        self.put("seed.h", '#if TARGET35\n#include "arch35/kernel.h"\n#else\n#include "arch22/kernel.h"\n#endif\n')
        self.put("arch35/kernel.h", "// target35\n")
        self.definition()
        result = self.run_collect()
        unresolved = next(x for x in result["unresolved"] if x["include"] == "arch22/kernel.h")
        self.assertEqual(unresolved["status"], "missing")
        self.assertEqual(unresolved["candidates"], [])

    def test_missing_seed_keeps_partial_bundle_and_existing_output_is_never_overwritten(self):
        self.definition()
        result = self.run_collect()
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["summary"]["missing_required"], ["seed.h"])
        before = (self.root/"output/source_closure.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_collect()
        self.assertEqual((self.root/"output/source_closure.json").read_bytes(), before)

    def test_symlink_outside_explicit_root_is_not_read(self):
        outside = self.put("outside.h", "struct SparseFlashAttentionTilingDataMla { int secret; };", self.root)
        (self.source/"seed.h").symlink_to(outside)
        result = self.run_collect()
        self.assertEqual(result["records"], [])
        self.assertEqual(result["definition_matches"], [])
        self.assertEqual(result["skipped_symlinks"], [str(self.source/"seed.h")])
        self.assertEqual(result["exit_code"], 1)


if __name__ == "__main__":
    unittest.main()
