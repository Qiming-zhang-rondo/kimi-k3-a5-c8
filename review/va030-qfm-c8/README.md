# VA 0.30 Kimi K3 C8 review snapshot

Base: c09d3e20b9c45c80b34ae3781440d90904a44023.
Current installer: `../../va030_qfm/install-va030-qfm-expanded-qfa.py`.
`source/` is the readable final source, not a sequence of incremental patches.
`va030-qfm-c8.patch` applies to the clean base; `qfa-from-expanded-fia.patch`
is the delta from the expanded-FIA delivery. Old immutable installers remain available.

This candidate adds expanded QK192/V128 FIA Prefill with two frontends:
ordinary MXFP8 projections or package-native CANN MLAPO using private identity UK.
An additional QFA option temporarily pads Q/K/V to256, quantizes MXFP8 along
the required axes and uses the fixed CANN9.2 ABI. It is not native192 support.
The default FIA form remains latent. Decode DSL mode2 plus QFM, one persistent
FP8 cache, target/draft isolation, APC/page geometry and PD wire format are retained.

See [the configuration and compatibility guide](source/docs/source/developer_guide/kimi_k3_qfm_c8.md).
Standalone single-card correctness/performance probe:
`../../experiments/expanded-attention-probe.py`. It does not modify installed VA.
The installer does not run tests, install packages or compile operators.
CPU regressions and ABI source review are not NPU/model/performance acceptance.

For QFA, `../../experiments/install-qfa920-wrapper.py` explicitly prepares an
isolated Python/C++ wrapper matching the existing CANN9.2 native kernel. Source
its env after the existing QFM/DSL environments. The original packages stay intact.
It does not rebuild the native Attention kernel; the host extension may compile
on first use. The QFA probe embeds the pinned official P-quantization golden,
comparison criteria, copyright notice and complete CANN Open Software License2.0.
