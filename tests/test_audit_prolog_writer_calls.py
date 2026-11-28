"""Local static call gate must not mistake comments/member counts for coverage."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("audit_writer_calls_test",ROOT/"tools/audit_prolog_writer_calls.py")
TOOL=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
FULL='''void Run() {
    core.SetRingWriter(&recent128Writer_);
    recent128Writer_.Init(pipe, binding);
    auto route = recent128Writer_.ResolveRow(token);
    recent128Writer_.WriteLatentSlice(route, exactInput, 0, 512);
    recent128Writer_.WriteRopeSlice(route, actualRope, 0, 64);
    recent128Writer_.PublishTags(route, true);
    recent128Writer_.Release();
}
'''


class AuditPrologWriterCallsTests(unittest.TestCase):
    def analyze(self,text,**kwargs):
        calls,names=TOOL.find_calls(text,"arch35/operator.h")
        return TOOL.summarize(calls,inferred_names=names,files_scanned=1,**kwargs)

    def test_comments_rawstrings_definitions_and_grep_count_do_not_fill_missing_calls(self):
        source=r'''
// recent128Writer_.WriteRopeSlice(a,b,0,64);
/* recent128Writer_.PublishTags(a,true); recent128Writer_.Release(); */
const char* diagnostic = "recent128Writer_.WriteRopeSlice(x)";
const char* raw = R"tag(recent128Writer_.PublishTags(x); recent128Writer_.WriteRopeSlice(x))tag";
#define NEVER_CALLED() recent128Writer_.WriteRopeSlice(x)
#define ALSO_NEVER_CALLED() \
 recent128Writer_.PublishTags(x)
struct Kernel {
 Writer* recent128Writer_ = nullptr;
 void SetRingWriter(Writer* value){recent128Writer_=value;}
 void WriteRopeSlice(Route, Tensor, int, int);
 void PublishTags(Route, bool);
 void Process(){auto route=recent128Writer_->ResolveRow(token);
 recent128Writer_->WriteLatentSlice(route, input, 0, 512);}
};
'''
        self.assertGreaterEqual(source.count("recent128Writer_"),5)
        result=self.analyze(source)
        self.assertEqual(result["status"],"BLOCKED_MISSING_CALLS")
        self.assertEqual(result["missing_mandatory_calls"],["WriteRopeSlice","PublishTags"])
        self.assertEqual(result["mandatory_calls"]["WriteRopeSlice"],[])
        self.assertEqual(result["setter_calls"],[])

    def test_real_member_calls_are_locations_not_a_runtime_claim(self):
        result=self.analyze(FULL)
        self.assertEqual(result["status"],"STATIC_CALLS_PRESENT_NOT_RUNTIME_PROOF")
        self.assertEqual(result["missing_mandatory_calls"],[])
        self.assertEqual(result["setter_inferred_writer_names"],["recent128Writer_"])
        rope=result["mandatory_calls"]["WriteRopeSlice"][0]
        self.assertEqual(rope["file"],"arch35/operator.h")
        self.assertEqual(rope["line"],6)
        self.assertEqual(rope["receiver"],"recent128Writer_")
        self.assertNotIn("exactInput",json.dumps(result))
        self.assertNotIn("actualRope",json.dumps(result))
        self.assertFalse(result["source_copied"])
        self.assertFalse(result["device_launched"])

    def test_unrelated_init_release_do_not_satisfy_writer_lifecycle(self):
        source=FULL.replace('    core.SetRingWriter(&recent128Writer_);\n','').replace(
            '    recent128Writer_.Init(pipe, binding);\n','    anotherOp.Init(pipe);\n').replace(
            '    recent128Writer_.Release();\n','    anotherOp.Release();\n')
        result=self.analyze(source)
        self.assertEqual(result["status"],"STATIC_CALLS_PRESENT_NOT_RUNTIME_PROOF")
        for method in ("Init","Release"):
            self.assertEqual(result["lifecycle"][method]["status"],"REQUIRES_MANUAL_TARGET_RECEIVER")
            self.assertEqual(result["lifecycle"][method]["calls"],[])

    def test_explicit_writer_alias_filters_out_unrelated_unique_method_names(self):
        source=FULL.replace("recent128Writer_","other")+'\nwriter.ResolveRow(t); writer.WriteLatentSlice(r,v,0,512);\n'
        result=self.analyze(source,writer_names=["writer"])
        self.assertEqual(result["status"],"BLOCKED_MISSING_CALLS")
        self.assertEqual(result["missing_mandatory_calls"],["WriteRopeSlice","PublishTags"])
        with self.assertRaises(ValueError):self.analyze(FULL,writer_names=["x;secret()"])

    def test_tree_excludes_sdk_build_public_helper_and_can_explicitly_scan_preprocessed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"arch35").mkdir()
            (root/"arch35/kernel.h").write_text('writer.ResolveRow(t); writer.WriteLatentSlice(a,b,0,512);')
            (root/"prolog_ring_write_device.h").write_text(FULL)
            (root/"build").mkdir();(root/"build/entry.ii").write_text(FULL)
            (root/"SDK").mkdir();(root/"SDK/fake.cpp").write_text(FULL)
            result=TOOL.audit(root)
            self.assertEqual(result["files_scanned"],1)
            self.assertEqual(result["status"],"BLOCKED_MISSING_CALLS")
            result=TOOL.audit(root,preprocessed="build/entry.ii")
            self.assertEqual(result["source_mode"],"explicit_preprocessed")
            self.assertEqual(result["status"],"STATIC_CALLS_PRESENT_NOT_RUNTIME_PROOF")

    def test_cli_writes_new_external_report_without_source_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/"private";source.mkdir()
            (source/"kernel.asc").write_text(FULL+'// PRIVATE_SECRET_NEVER_EMIT\n')
            output=root/"audit"
            with redirect_stdout(StringIO()):
                code=TOOL.main(["--source-root",str(source),"--output",str(output)])
            self.assertEqual(code,0)
            text=(output/"writer_calls.json").read_text()
            self.assertNotIn("PRIVATE_SECRET_NEVER_EMIT",text)
            self.assertNotIn("exactInput",text)
            with self.assertRaises(SystemExit) as raised:
                TOOL.main(["--source-root",str(source),"--output",str(output)])
            self.assertEqual(raised.exception.code,2)
            with self.assertRaises(SystemExit):
                TOOL.main(["--source-root",str(source),"--output",str(ROOT/"private-audit-must-not-create")])
            self.assertFalse((ROOT/"private-audit-must-not-create").exists())


if __name__=="__main__":unittest.main()
