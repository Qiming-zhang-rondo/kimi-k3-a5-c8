# VA 0.30 Kimi K3 C8 review snapshot

Base: c09d3e20b9c45c80b34ae3781440d90904a44023.
Current installer: `../../va030_qfm/install-va030-qfm-expanded-fia.py`.
`source/` is the readable final source, not a sequence of incremental patches.
`va030-qfm-c8.patch` applies to the clean base; `expanded-fia-from-geometry.patch`
is the delta from the geometry delivery. Old immutable installers remain available.

This candidate adds expanded QK192/V128 FIA Prefill with two frontends:
ordinary MXFP8 projections or package-native CANN MLAPO using private identity UK.
The default FIA form remains latent. Decode DSL mode2 plus QFM, one persistent
FP8 cache, target/draft isolation, APC/page geometry and PD wire format are retained.

See [the configuration and compatibility guide](source/docs/source/developer_guide/kimi_k3_qfm_c8.md).
Standalone single-card correctness/performance probe:
`../../experiments/fia-expanded-probe.py`. It does not modify installed VA.
The installer does not run tests, install packages or compile operators.
CPU regressions and ABI source review are not NPU/model/performance acceptance.
