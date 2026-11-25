# Native MXFP4 C4 KV Quant Sparse Flash Attention — Source Provenance

## Status: SOURCE_READY

## Operator identification

| Item | Value |
|---|---|
| Operator name | `cloud_kv_quant_sparse_flash_attention` |
| ACLNN entry | `aclnnKvQuantSparseFlashAttention` (V1 in this package) |
| Vendor package | custom_cot |
| Package version | 9.1.0 |
| Package commit | f6e7b8829e2cbfb1409cc2d070131f65186a49cb |
| Package timestamp | 20261019_082334953 |
| CANN | 9.2.1 (runtime), custom_cot 9.1.0 (vendor) |
| Target arch | Ascend950DT (dav-3510, A5) |

## Source layout (this delivery)

All paths relative to `source/`:

| Path | Role |
|---|---|
| `ascendc/kv_quant_sparse_flash_attention/kv_quant_sparse_flash_attention.cpp` | TBE kernel entry (template registration, ASCENDC_TPL_ARGS) |
| `ascendc/kv_quant_sparse_flash_attention/kv_quant_sparse_flash_attention_common.h` | Common types/ConstInfo |
| `ascendc/kv_quant_sparse_flash_attention/kv_quant_sparse_flash_attention_template_tiling_key.h` | Tiling key |
| `ascendc/kv_quant_sparse_flash_attention/kv_quant_sparse_flash_attention_kernel_mla.h` | Kernel top-level dispatcher (non-arch-specific) |
| `ascendc/kv_quant_sparse_flash_attention/kv_quant_sparse_flash_attention_service_cube_mla.h` | Cube service (non-arch) |
| `ascendc/kv_quant_sparse_flash_attention/kv_quant_sparse_flash_attention_service_vector_mla.h` | Vector service (non-arch) |
| `ascendc/kv_quant_sparse_flash_attention/arch35/*.h` | arch35 kernel, kvcache, cube/vector services, common |
| `ascendc/kv_quant_sparse_flash_attention/arch35/common/op_kernel/**` | C4 operator's own common headers (buffer, matmul, vf_*, memcopy, offset_calculator) |
| `ascendc/kv_quant_sparse_flash_attention/arch35/util_regbase.h` | ConstInfo register struct |
| `ascendc/kv_quant_sparse_flash_attention/dynamic/kv_quant_sparse_flash_attention.py` | TBE dynamic build entry |
| `op_proto/inc/kv_quant_sparse_flash_attention_proto.h` | Operator proto (IR definition) |
| `op_api/include/aclnn_kv_quant_sparse_flash_attention.h` | ACLNN C API header |
| `op_api/include/aclnnop/aclnn_kv_quant_sparse_flash_attention.h` | ACLNN dispatch header |
| `op_impl/ai_core/tbe/kernel/config/ascend950/kv_quant_sparse_flash_attention.json` | Compiled kernel binary descriptor |

## Kernel binary SHA256

| File | SHA256 |
|---|---|
| `op_impl/ai_core/tbe/kernel/config/ascend950/kv_quant_sparse_flash_attention.json` | b4b810f9cdcb38a72dbcebcdf96db92cbf3103088184319607b3bf6b41401691 |

## Key architectural notes

- **MXFP4 (C4) layout**: 512-dim latent (NoPE) quantized to MXFP4 with E8M0 block scales; 64-dim RoPE kept BF16. Packed to 400 B/token.
- **Quantized KV read path**: kernel reads packed MXFP4 cold KV, dequantizes inline, computes attention.
- **Same overall structure as BF16 SFA** (cube+vector MIX kernel, TND layout, PA_BSND KV), but KV data is MXFP4 packed format.
- **RoPE64 stays BF16** in both C4 and BF16 variants — only latent 512 dims are quantized.
- **Host tiling**: uses ASCENDC_TPL_ARGS template registration (TBE build system); tiling key in `kv_quant_sparse_flash_attention_template_tiling_key.h`.

## Dependencies NOT included (available from public CANN SDK)

- `kernel_operator.h` — CANN SDK public header
- `lib/matmul_intf.h` — CANN SDK matmul interface
- `kernel_tiling/kernel_tiling.h` — CANN SDK tiling framework
- `ascendc/host_api/tiling/template_argument.h` — CANN SDK template argument registration

These are available in the standard CANN toolkit (tested with CANN 9.2.1) and are not redistributed here.

## Matching status

- Source files are the actual installed vendor files from custom_cot 9.1.0 (commit f6e7b882).
- The compiled kernel JSON SHA256 is recorded above for binary matching.
- This source was NOT rebuilt from scratch in this delivery — it is the as-installed source.
- BUILD.md records NOT_BUILT for this delivery.

## License

Source files carry original Huawei CANN/CLOUD-TURBO copyright headers (Apache-2.0 style where present). Original headers retained verbatim. If any file is export-restricted, it must not leave the internal network — the collector reviewed file headers and found standard open-source-style license text; no export-restriction markers were found in these specific files.
