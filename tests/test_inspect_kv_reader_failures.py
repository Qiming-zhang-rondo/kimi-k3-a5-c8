import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import inspect_kv_reader_failures as inspect_reader


class ReaderFailureExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def entry(self, **changes):
        entry = dict(name="a_precise_field_match", status="CONTROL_FAILED", recipe="baseline",
                     mode="mixed", reader="hot_merge", hot_block_keys=16,
                     hot_block_heads=16, hot_decode="native_cast",
                     control_failure_reason="original control reason",
                     reader_consistency={"passed": False, "checks": {
                         "combined_vs_native_twopass": {"relative_l2": .00123456789,
                             "tolerance": .001, "passed": False, "rows": [{"row": 0, "max_abs": .25}]}}},
                     partition_consistency={"passed": True, "fields": {"old": {"mismatch_count": 0}}},
                     writer_consistency={"passed": True, "skipped": True})
        entry.update(changes)
        return entry

    def write(self, relative, entries=None, **fields):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        document = dict(version="kv-native-twopass-a5-v4", args={"timing": "graph", "seed": 123},
                        cases=[{"label": "B1_Q4", "variants": entries or [self.entry()]}])
        document.update(fields)
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = inspect_reader.main([str(arg) for arg in args])
        return code, out.getvalue(), err.getvalue()

    def test_newest_recursive_result_selected_by_mtime_not_run_name(self):
        old = self.write("z-new-name/results/results.json", [self.entry(name="old")])
        newest = self.write("a-old-name/results/results.json", [self.entry(name="newest")])
        os.utime(old, ns=(1000, 1000))
        os.utime(newest, ns=(2000, 2000))
        (newest.parent.parent / "launcher.json").write_text(json.dumps({"payload_commit": "a"*40}))
        code, out, err = self.run_cli(self.root)
        self.assertEqual(code, 0)
        result = json.loads(out)
        self.assertEqual(result["selection"]["results_found"], 2)
        self.assertEqual(result["selection"]["source_path"], str(newest.resolve()))
        self.assertEqual(result["entries"][0]["entry"]["name"], "newest")
        self.assertEqual(result["launcher"]["payload_commit"], "a"*40)
        self.assertIn("2 result files found", err)

    def test_explicit_file_never_switches_to_newer_sibling_and_preserves_metrics(self):
        entry = self.entry(traceback="Traceback\noriginal exact failure\n", error={"code": "UB"})
        path = self.write("old/results.json", [entry])
        self.write("new/results.json", [self.entry(name="do not select")])
        original_bytes, original_stat = path.read_bytes(), path.stat()
        (path.parent / "launcher.json").write_text(json.dumps({"payload_commit": "b"*40}))
        code, out, _ = self.run_cli(path)
        self.assertEqual(code, 0)
        result = json.loads(out)
        self.assertEqual(result["selection"]["selection_rule"], "explicit_file")
        self.assertEqual(result["selection"]["results_found"], 1)
        self.assertEqual(result["version"], "kv-native-twopass-a5-v4")
        self.assertEqual(result["args"], {"timing": "graph", "seed": 123})
        self.assertEqual(result["entries"][0]["entry"], entry)
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertEqual(path.stat().st_mtime_ns, original_stat.st_mtime_ns)
        self.assertEqual(result["launcher"]["payload_commit"], "b"*40)

    def test_matching_uses_actual_fields_and_not_deceptive_names(self):
        entries = [self.entry(), self.entry(name="baseline_mixed_hot_merge_k16_h16_dnative_cast", reader="native"),
                   self.entry(name="other_recipe", recipe="alg2_max7"),
                   self.entry(name="other_decode", hot_decode="software"),
                   self.entry(name="other_tile", hot_block_keys=32),
                   self.entry(name="control", mode="split_c4", requested_reader="hot_merge", reader="native"),
                   self.entry(name="actual_native_wins", reader="native", requested_reader="hot_merge")]
        path = self.write("results.json", entries)
        code, out, _ = self.run_cli(path)
        self.assertEqual(code, 0)
        self.assertEqual([x["entry"]["name"] for x in json.loads(out)["entries"]], [entries[0]["name"]])

    def test_reader_fallbacks_and_implementation_tile_fields_are_explicit(self):
        requested = self.entry(name="requested", requested_reader="hot_merge")
        del requested["reader"]
        implemented = self.entry(name="implemented")
        for field in ("reader", "hot_decode", "hot_block_keys", "hot_block_heads"):
            del implemented[field]
        implemented["implementation"] = dict(reader="native_c4_plus_fused_c8_hot_merge",
                                               hot_block_keys=16, hot_block_heads=16, hot_decode="native_cast")
        path = self.write("results.json", [requested, implemented])
        code, out, _ = self.run_cli(path)
        self.assertEqual(code, 0)
        selected = json.loads(out)["entries"]
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["selection_evidence"]["reader"]["source"], "requested_reader")
        self.assertEqual(selected[1]["selection_evidence"]["reader"]["source"], "implementation.reader")
        self.assertNotIn("hot_decode", selected[1]["entry"])

    def test_missing_diagnostics_and_metadata_are_not_fabricated(self):
        entry = self.entry()
        for field in inspect_reader.DIAGNOSTIC_FIELDS:
            del entry[field]
        path = self.write("results.json", [entry])
        document = json.loads(path.read_text())
        del document["version"], document["args"]
        path.write_text(json.dumps(document))
        code, out, _ = self.run_cli(path)
        self.assertEqual(code, 0)
        result = json.loads(out)
        self.assertIsNone(result["version"])
        self.assertIsNone(result["args"])
        self.assertIsNone(result["launcher"]["payload_commit"])
        self.assertEqual(result["missing_report_fields"], ["version", "args"])
        self.assertEqual(result["entries"][0]["missing_diagnostic_fields"], list(inspect_reader.DIAGNOSTIC_FIELDS))
        self.assertEqual(result["entries"][0]["entry"], entry)

    def test_unmatched_latest_file_reports_candidates_without_falling_back(self):
        old = self.write("old/results.json")
        new = self.write("new/results.json", [self.entry(hot_decode="software")])
        os.utime(old, ns=(1000, 1000))
        os.utime(new, ns=(2000, 2000))
        code, out, err = self.run_cli(self.root)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("No matching entry", err)
        self.assertIn("No older result file was substituted", err)
        self.assertIn("software", err)
        self.assertNotIn("relative_l2", err)
        code, out, _ = self.run_cli(self.root, "--decode", "software")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["filter"]["hot_decode"], "software")

    def test_case_only_report_and_missing_selector_do_not_infer_from_name(self):
        missing = self.entry(name="baseline_mixed_hot_merge_k16_h16_dnative_cast")
        del missing["hot_decode"]
        path = self.write("case_000/results.json")
        path.write_text(json.dumps({"label": "only_case", "variants": [missing, self.entry(name="correct")]}))
        code, out, _ = self.run_cli(path)
        self.assertEqual(code, 0)
        selected = json.loads(out)["entries"]
        self.assertEqual([x["entry"]["name"] for x in selected], ["correct"])
        self.assertEqual(selected[0]["case_label"], "only_case")

    def test_read_errors_and_original_report_errors_are_visible(self):
        path = self.root / "bad" / "results.json"
        path.parent.mkdir()
        path.write_text("not json")
        code, out, err = self.run_cli(path)
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("Read failed", err)
        path.write_text(json.dumps({"error": "original exception", "traceback": "original\ntraceback"}))
        code, out, err = self.run_cli(path)
        self.assertEqual(code, 2)
        self.assertIn("original exception", err)
        self.assertIn("original\ntraceback", err)
        self.assertIn("No variant entries", err)

    def test_standalone_python_without_site_packages_never_loads_operators(self):
        path = self.write("results.json")
        code = ("import runpy,sys; sys.argv=" + repr([str(ROOT / "inspect_kv_reader_failures.py"), str(path)])
                + "; runpy.run_path(sys.argv[0], run_name='inspector_import'); "
                  "assert not any(x.startswith(('torch','triton','numpy')) for x in sys.modules)")
        process = subprocess.run([sys.executable, "-S", "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        process = subprocess.run([sys.executable, "-S", "-B", str(ROOT / "inspect_kv_reader_failures.py"), str(path)],
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["selected_count"], 1)


if __name__ == "__main__":
    unittest.main()
