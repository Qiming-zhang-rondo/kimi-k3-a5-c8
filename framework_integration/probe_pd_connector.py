#!/usr/bin/env python3
"""Resolve the deployed connector factory without constructing LLM/worker.

Use the service's exact environment, source checkout and engine JSON. This is
a factory identity probe, not proof of a running worker or a PD transfer.
"""
import argparse
import importlib
import inspect
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-json",type=Path,required=True)
    parser.add_argument("--mxllm-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True,help="NEW local output directory")
    args=parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():parser.error("Use a NEW output directory")
    report=dict(status="PD_FACTORY_PROBE_FAILED",scope="factory_only_not_live_worker",model_loaded=False,device_launches=0)
    try:
        engine=json.loads(args.engine_json.read_text())
        options=engine.get("engine_options",engine)
        transfer=options.get("kv_transfer_config")
        if isinstance(transfer,str):transfer=json.loads(transfer)
        if not isinstance(transfer,dict):raise ValueError("Actual engine JSON lacks kv_transfer_config")
        report["requested"]={key:transfer.get(key) for key in ("kv_connector","kv_connector_module_path","kv_role")}
        from mxllm_bootstrap import bootstrap_mxllm
        report["bootstrap"]=bootstrap_mxllm(args.mxllm_root)
        mxllm=importlib.import_module("ascend_vllm")
        register=getattr(mxllm,"register_connector",None)
        if callable(register):register()
        from vllm.config.kv_transfer import KVTransferConfig
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        from sfa_recent128.pd_diagnostics import describe
        cls=KVConnectorFactory.get_connector_class(KVTransferConfig(**transfer))
        report["connector"]=describe(cls)
        module=importlib.import_module(cls.__module__)
        report["worker_candidates"]={name:describe(value) for name,value in vars(module).items()
            if inspect.isclass(value) and (name.endswith("Worker") or name.endswith("Thread"))}
        report["status"]="PD_FACTORY_RESOLVED_NOT_LIVE_WORKER"
    except Exception as error:
        report["error"]=type(error).__name__+": "+str(error)
        report["error_type"]=type(error).__name__
        # Keep full diagnostics local; do not request a private source dump.
        report["traceback"]=traceback.format_exc()
    args.output.mkdir(parents=True)
    (args.output/"pd_factory.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    # Config validation tracebacks may include private transport settings;
    # only this redacted identity summary is intended for sharing.
    summary={key:report.get(key) for key in ("status","requested","connector","worker_candidates","error_type")}
    (args.output/"pd_identity_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(status=report["status"],summary=str(args.output/"pd_identity_summary.json"))))
    return 0 if report["status"]=="PD_FACTORY_RESOLVED_NOT_LIVE_WORKER" else 2


if __name__=="__main__":raise SystemExit(main())
