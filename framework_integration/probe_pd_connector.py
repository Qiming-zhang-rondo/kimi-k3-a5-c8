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


def resolve_factory_tree(transfer, config_type, factory):
    """Resolve nested Multi configs without instantiating any connector/worker.

    Only redacted identity metadata leaves this function. Child order and engine
    inheritance follow MultiConnector._get_connector_classes_and_configs.
    """
    from sfa_recent128.pd_diagnostics import describe

    def resolve(options, depth=0):
        if depth > 32:
            raise ValueError("Connector config exceeds supported nesting depth")
        config = config_type(**options)
        cls = factory.get_connector_class(config)
        module = importlib.import_module(cls.__module__)
        node = dict(requested={key:options.get(key) for key in
            ("kv_connector", "kv_connector_module_path", "kv_role")}, connector=describe(cls),
            worker_candidates={name:describe(value) for name,value in vars(module).items()
                if inspect.isclass(value) and (name.endswith("Worker") or name.endswith("Thread"))},
            children=[])
        extra = getattr(config, "kv_connector_extra_config", None) or {}
        children = extra.get("connectors") if callable(getattr(cls, "_get_connector_classes_and_configs", None)) else None
        if children is not None:
            if not isinstance(children, list):
                raise ValueError("MultiConnector connectors must be a list")
            for index, child in enumerate(children):
                if not isinstance(child, dict):
                    raise ValueError("Each MultiConnector child config must be an object")
                inherited = dict(child)
                if "engine_id" not in inherited and getattr(config, "engine_id", None) is not None:
                    inherited["engine_id"] = config.engine_id
                resolved = resolve(inherited, depth + 1)
                resolved["index"] = index
                node["children"].append(resolved)
        return node

    return resolve(transfer)


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
        resolved=resolve_factory_tree(transfer,KVTransferConfig,KVConnectorFactory)
        report.update({key:resolved[key] for key in ("connector","worker_candidates")})
        report["connector_children"]=resolved["children"]
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
    summary={key:report.get(key) for key in ("status","requested","connector","worker_candidates","connector_children","error_type")}
    (args.output/"pd_identity_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(status=report["status"],summary=str(args.output/"pd_identity_summary.json"))))
    return 0 if report["status"]=="PD_FACTORY_RESOLVED_NOT_LIVE_WORKER" else 2


if __name__=="__main__":raise SystemExit(main())
