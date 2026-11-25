# Build Provenance

## Status: NOT_BUILT

This delivery contains source code collected from the as-installed custom_cot 9.1.0 vendor package (commit f6e7b8829e2cbfb1409cc2d070131f65186a49cb, timestamp 20261019_082334953).

No rebuild was performed for this handoff. The compiled kernel binary is already installed in the vendor package; its descriptor SHA256 is recorded in PROVENANCE.md.

## How this operator was originally built (from vendor package provenance)

- Built by the custom_cot vendor package build pipeline (TBE/AscendC)
- Target: Ascend950DT (dav-3510, A5)
- CANN toolchain: 9.2.1 (runtime environment where installed)

## To rebuild from this source (for external implementers)

This source uses the TBE (Tensor Boost Engine) build system:

1. The TBE dynamic entry is `dynamic/kv_quant_sparse_flash_attention.py`
2. The kernel entry is `kv_quant_sparse_flash_attention.cpp` (registers ASCENDC_TPL_ARGS templates)
3. Build environment needs: CANN toolkit with TBE support (tested with 9.2.1), ASCENDC_TPL template registration headers
4. The compiled binary descriptor is in `op_impl/ai_core/tbe/kernel/config/ascend950/kv_quant_sparse_flash_attention.json`

Alternatively, for a standalone AscendC build (bypassing TBE), refer to the BF16 SFA standalone build pattern already proven in this repository (`standalone/` directory) — the C4 kernel has the same overall structure but with MXFP4 dequantization paths.
