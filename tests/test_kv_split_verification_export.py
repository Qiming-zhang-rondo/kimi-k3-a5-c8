"""Lossless evidence handoff uses only stdlib, never a new operator run."""
import base64
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import string
import subprocess
import sys
import tempfile
import unittest


MODULE=Path(__file__).resolve().parents[1]/"experimental/single_pass/export_verification_evidence.py"


class VerificationExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location("kv_split_verification_export_under_test",MODULE)
        cls.module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=cls.module
        spec.loader.exec_module(cls.module)

    def fixture(self,root,*,large=False,status="CONTROL_FAILED"):
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        event=dict(baseline_median=80.12345678901235,candidate_median=57.98765432109877,
            median_paired_ratio=.7234567890123457,min_paired_ratio=.7000000000000001,
            max_paired_ratio=.7999999999999999,paired_ratios=[.7234567890123457,.7111111111111111])
        wall=dict(baseline_median=81.23456789012346,candidate_median=59.01234567890124,
            median_paired_ratio=.7265432109876543,paired_ratios=[.7265432109876543,.7333333333333333])
        pair=dict(baseline="installed_c4",candidate="w128_split4",
            ratio_definition="candidate / baseline per round",
            timing=dict(graph=dict(status="MEASURED",event_us=event,wall_us=wall,
                rounds=[dict(round=0,order=["baseline","candidate"],unrounded=.12345678901234568)])))
        passed=status=="MEASURED"
        source=dict(version="Unicode-原始测量-🙂",status=status,
            args=dict(queries=4,repeats=20,verify_comparison=True),
            verification_timings={"w128_split4_vs_installed_c4":pair},
            all_accuracy_controls_passed=passed,controls=dict(mathematical=dict(passed=passed)),
            graph_controls=dict(poison=dict(passed=passed)),post_timing_controls=dict(passed=passed),
            bridge_build=dict(library_sha256="a"*64),
            state_note="保留原始状态，不能重写成MEASURED")
        if large:
            rng=random.Random(4733)
            source["unabridged_evidence"]="".join(rng.choices(string.ascii_letters+string.digits,k=220000))
        text=json.dumps(source,ensure_ascii=False,indent=2).replace("\n","\r\n")+"\r\n\r\n"
        raw=text.encode("utf-8")
        path=root/"results.json";path.write_bytes(raw)
        return path,source,raw

    def manifest_path(self,directory):
        candidates=[]
        for path in Path(directory).glob("*.json"):
            data=json.loads(path.read_text())
            if data.get("encoding")=="gzip+base64" and "parts" in data:
                candidates.append(path)
        self.assertEqual(len(candidates),1)
        return candidates[0]

    def part_path(self,manifest_path,record):
        path=Path(record["path"])
        return path if path.is_absolute() else manifest_path.parent/path

    def test_unicode_exact_multichunk_roundtrip_and_original_source_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path,source,raw=self.fixture(Path(directory)/"input",large=True)
            output=Path(directory)/"export"
            manifest=self.module.export_evidence(source_path,output)
            manifest_path=self.manifest_path(output)
            written=json.loads(manifest_path.read_text())
            self.assertEqual(manifest,written)
            self.assertEqual(manifest["encoding"],"gzip+base64")
            self.assertEqual(manifest["source_sha256"],hashlib.sha256(raw).hexdigest())
            self.assertEqual(manifest["source_bytes"],len(raw))
            self.assertEqual(manifest["source_status"],source["status"])
            self.assertEqual(manifest["source_version"],source["version"])
            self.assertGreater(len(manifest["parts"]),1)
            chunks=[]
            for index,record in enumerate(manifest["parts"]):
                path=self.part_path(manifest_path,record)
                self.assertEqual(path.name,f"source.{index:04d}.b64.txt")
                encoded=path.read_bytes();encoded.decode("ascii")
                content=encoded.strip()
                self.assertGreater(len(content),0)
                self.assertLessEqual(len(content),32768)
                self.assertLessEqual(len(encoded),32769) # Exporter retains one text-file newline.
                if index+1<len(manifest["parts"]):self.assertEqual(len(content),32768)
                self.assertEqual(record["bytes"],len(encoded))
                self.assertEqual(record["sha256"],hashlib.sha256(encoded).hexdigest())
                chunks.append(content)
            compressed=base64.b64decode(b"".join(chunks),validate=True)
            self.assertEqual(compressed[4:8],b"\0\0\0\0")
            self.assertEqual(gzip.decompress(compressed),raw)
            self.assertEqual(self.module.restore_evidence(manifest_path),raw)
            self.assertEqual(source_path.read_bytes(),raw)

    def test_deterministic_export_and_existing_directory_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source,_,raw=self.fixture(Path(directory)/"input",large=True)
            first=Path(directory)/"first";second=Path(directory)/"second"
            a=self.module.export_evidence(source,first)
            b=self.module.export_evidence(source,second)
            ap=self.manifest_path(first);bp=self.manifest_path(second)
            self.assertEqual([self.part_path(ap,p).read_bytes() for p in a["parts"]],
                [self.part_path(bp,p).read_bytes() for p in b["parts"]])
            before={str(p.relative_to(first)):p.read_bytes() for p in first.rglob("*") if p.is_file()}
            with self.assertRaises(Exception):self.module.export_evidence(source,first)
            self.assertEqual(before,{str(p.relative_to(first)):p.read_bytes() for p in first.rglob("*") if p.is_file()})
            self.assertEqual(source.read_bytes(),raw)

    def test_restore_rejects_tampered_part_and_every_recorded_integrity_field(self):
        with tempfile.TemporaryDirectory() as directory:
            source,_,_=self.fixture(Path(directory)/"input")
            for kind in ("part_content","part_sha","part_bytes","source_sha","source_bytes"):
                output=Path(directory)/kind
                manifest=self.module.export_evidence(source,output)
                path=self.manifest_path(output)
                if kind=="part_content":
                    part=self.part_path(path,manifest["parts"][0])
                    data=part.read_bytes();part.write_bytes((b"A" if data[:1]!=b"A" else b"B")+data[1:])
                else:
                    modified=copy.deepcopy(manifest)
                    if kind=="part_sha":modified["parts"][0]["sha256"]="0"*64
                    elif kind=="part_bytes":modified["parts"][0]["bytes"]+=1
                    elif kind=="source_sha":modified["source_sha256"]="f"*64
                    elif kind=="source_bytes":modified["source_bytes"]+=1
                    path.write_text(json.dumps(modified,ensure_ascii=False))
                with self.subTest(kind=kind),self.assertRaises(Exception):
                    self.module.restore_evidence(path)

    def test_stdlib_isolated_process_needs_no_numpy_torch_or_site_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            source,_,_=self.fixture(Path(directory)/"input")
            script=r'''
import importlib.util,json,pathlib,sys
spec=importlib.util.spec_from_file_location("standalone_evidence_export",sys.argv[1])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
source=pathlib.Path(sys.argv[2]);output=pathlib.Path(sys.argv[3])
module.export_evidence(source,output)
paths=[p for p in output.glob("*.json") if json.loads(p.read_text()).get("encoding")=="gzip+base64"]
assert len(paths)==1
assert module.restore_evidence(paths[0])==source.read_bytes()
assert not any(k=="numpy" or k.startswith("numpy.") or k=="torch" or k.startswith("torch.") for k in sys.modules)
print("PASS stdlib-only isolated export/restore")
'''
            process=subprocess.run([sys.executable,"-I","-S","-c",script,str(MODULE),str(source),
                str(Path(directory)/"isolated")],capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            self.assertIn("PASS stdlib-only",process.stdout)

    def test_summary_preserves_original_event_wall_ratios_and_failed_global_status(self):
        with tempfile.TemporaryDirectory() as directory:
            for status in ("MEASURED","CONTROL_FAILED","ERROR"):
                source_path,source,raw=self.fixture(Path(directory)/("input_"+status),status=status)
                output=Path(directory)/("export_"+status)
                manifest=self.module.export_evidence(source_path,output)
                summary=json.loads((output/"verification_summary.json").read_text())
                self.assertEqual(manifest["source_status"],status)
                self.assertEqual(summary["status"],status)
                self.assertEqual(summary["all_accuracy_controls_passed"],source["all_accuracy_controls_passed"])
                key="w128_split4_vs_installed_c4"
                original=source["verification_timings"][key]["timing"]["graph"]
                exported=summary["verification_pairs"][key]["timing"]["graph"]
                self.assertEqual(exported["status"],original["status"])
                self.assertEqual(exported["round_count"],len(original["rounds"]))
                self.assertEqual(exported["event_us"],original["event_us"])
                self.assertEqual(exported["wall_us"],original["wall_us"])
                self.assertEqual(self.module.restore_evidence(self.manifest_path(output)),raw)

    def test_handwritten_summary_or_incomplete_measured_source_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            summaries=(dict(status="MEASURED",rows=[dict(result=i) for i in range(6)]),
                dict(status="MEASURED",args=dict(verify_comparison=True),verification_timings={}))
            for index,summary in enumerate(summaries):
                path=root/f"handwritten_{index}.json"
                path.write_text(json.dumps(summary,indent=2))
                output=root/f"rejected_{index}"
                with self.subTest(index=index),self.assertRaises(ValueError):
                    self.module.export_evidence(path,output)
                self.assertFalse(output.exists())


if __name__=="__main__":
    unittest.main()
