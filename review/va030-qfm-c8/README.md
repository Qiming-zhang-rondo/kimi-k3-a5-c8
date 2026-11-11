# VA 0.30 Kimi K3 C8 review snapshot

Base: c09d3e20b9c45c80b34ae3781440d90904a44023.
Current installer: `../../va030_qfm/install-va030-qfm-fia-paged.py`.
`source/` contains the readable final source.
`va030-qfm-c8.patch` applies to the clean base;
`fia-paged-from-expanded-qfa.patch` upgrades the expanded-QFA delivery.
Previously published installers and experiment probes remain immutable.

The new `fia_paged` P backend keeps independent dense BF16 latent512/raw64
history pools. The shared FP8/KDA hybrid arena and its physical block strides
remain unchanged. The worker reserves the extra 1152 bytes/token/MLA-layer
before choosing cache block capacity. Paged BF16 FIA reads history directly.
Local writers and generic block copies preserve the BF16 history for chunked
Prefill and APC reuse. This reservation can reduce available cache blocks.

P selects `kimi_k3_c8_prefill_backend=fia_paged`,
`kimi_k3_c8_fia_form=latent`, and `enable_mlapo=false`.
D selects the existing `qfm` backend with `enable_mlapo=true`.
PD producers can use the new path; consumers keep QFM. The versioned FP8 PD
payload is unchanged and excludes the BF16 Prefill pools.
The original QFM, temporary BF16 latent/expanded FIA, and QFA routes remain
selectable. Prefill stays eager; Decode-only graph capture keeps QFM.

The installer does not install native packages, compile operators, or run tests.
See [configuration and compatibility details](source/docs/source/developer_guide/kimi_k3_qfm_c8.md).
The new standalone probe is `../../experiments/fia-paged-probe.py`.
CPU address, allocator and installer checks are not NPU/model/performance
acceptance. Real FIA, APC, DSpark, DCP, graph replay, and PD tests are required
for the selected deployment configuration.
