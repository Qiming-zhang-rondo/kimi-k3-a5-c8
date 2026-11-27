#!/usr/bin/env python3
"""Same-input native C4 / native BF16 / MXFP4+BF16 split4 comparison on A5.

Reuse the verified bridge, mathematical/partial/graph controls and paired
timing machinery. Only the comparison matrix and reporting are new. This
entry point neither installs packages nor recompiles a device kernel.
"""
import run_kv_split


def parser():
    cli=run_kv_split.parser()
    cli.description=__doc__
    cli.set_defaults(compare_bf16=True,verify_comparison=True,
        windows=[0,128],splits=[4],timing="graph",repeats=20)
    return cli


def main(argv=None):
    return run_kv_split.main(argv,argument_parser=parser())


if __name__=="__main__":
    raise SystemExit(main())
