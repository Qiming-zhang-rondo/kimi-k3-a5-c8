# BUILD_CONTEXT.md — SFA Cold-State Merge: Build Chain and Dependencies

> 执行入口已统一为 [START_HERE.md](START_HERE.md) 和
> [完整执行任务](AUTONOMOUS_EXECUTION_TASK.md)。下面保留的是上传时的定位记录，
> 不是已经验证成功的完整构建链。审查确认：所附 `sparse_flash_mla.cpp` 调用
> `aclnnSparseFlashMla`，不是 Cloud 三输出 SFA 的绑定；dynamic Python 文件是编译适配器，
> 没有提供 host tiling 算法；文中 `cold_merge_custom/` 工程未随该提交上传。
> 后续由内网执行代理自主补齐、构建和实测，将命令、日志及源码直接提交此分支。

## 1. Source-to-deployment correspondence

| Layer | Version | Location (in c4 container, 10.83.89.178) |
|---|---|---|
| CANN toolkit | 9.2.1 | `/usr/local/Ascend/ascend-toolkit/latest/` |
| custom_cot ops package | 9.1.0 (f6e7b882, 20261019) | `/usr/local/Ascend/cann-9.2.1/opp/vendors/custom_cot/` |
| Python wheel (cloud_ops_turbo) | 1.0.0 (cp312) | `/tmp/cloudops/pylib/cloud_ops_turbo/` |
| cann_ops_transformer | 9.2.1 toolkit bundled | `/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/cann_ops_transformer/` |
| Target chip | Ascend950DT (DAV_3510 / A5) | arch35 |

Note: the custom_cot package version (9.1.0) differs from the toolkit (9.2.1).
The experiment container (c4) uses the 9.1.0 ops natively installed; the whl
(cloud_ops_turbo 1.0.0) is designed to pair with this version. The mxllm service
container uses 9.2.0 custom_cot. Both versions exist on the same host; the c4
container's 9.1.0 is the one validated for experiments.

## 2. Actual Python → Torch → ACLNN → kernel call chain

```
[Python] torch.ops.cloud_ops_turbo.cloud_sparse_flash_attention(...)
    ↓ (registered by whl's torch library)
[Pylib] /tmp/cloudops/pylib/cloud_ops_turbo/cloud_sparse_flash_attention.py
    ↓ calls torch.ops.cloud_ops_turbo.cloud_sparse_flash_attention
[Torch binding] cann_ops_transformer/csrc/attention/... → aclnn call
    ↓ (via aclnn_common.h wrapper)
[ACLNN] custom_cot/op_api/include/aclnn_sparse_flash_attention.h
    ↓ (binary .so: custom_cot/op_api/lib/libopapi.so)
[Host tiling] custom_cot_impl/dynamic/sparse_flash_attention.py (TBE dynamic)
    ↓
[Kernel entry] custom_cot_impl/ascendc/sparse_flash_attention/sparse_flash_attention.cpp
    ↓ dispatches to arch35/
[arch35 kernel] arch35/sparse_flash_attention_kernel_mla.h (725 lines)
    ├── Cube: arch35/sparse_flash_attention_service_cube_mla.h (398 lines)
    └── Vector: arch35/sparse_flash_attention_service_vector_mla.h (1054 lines)
         └── ProcessVec2 → final normalization → CopyOutAttentionOut
              ↑ THIS IS WHERE cold-state merge inserts ↑
```

## 3. Source tree and dependency map

### 3.1 Already in repo (`original/` and `deps/`)

**Kernel source (in `original/`):**
- `original/sparse_flash_attention.cpp` — kernel entry (102 lines)
- `original/sparse_flash_attention_common.h` — ConstInfo, RunInfo, SoftmaxConfig (202 lines)
- `original/sparse_flash_attention_template_tiling_key.h` — tiling keys (117 lines)
- `original/arch35/sparse_flash_attention_kernel_mla.h` — main kernel (725 lines)
- `original/arch35/sparse_flash_attention_service_vector_mla.h` — vector service, **merge insertion point** (1054 lines)
- `original/arch35/sparse_flash_attention_service_cube_mla.h` — cube service (398 lines)
- `original/arch35/sparse_flash_attention_kvcache.h` — KV cache helpers (300 lines)
- `original/arch35/sparse_flash_attention_common_arch35.h` — arch35 common (124 lines)

**Project-internal dependencies (in `deps/sparse_flash_attention/arch35/`):**
- `util_regbase.h` — RunInfo, ConstInfo, BufferType, SyncType definitions
- `common/op_kernel/buffer.h` — Buffer class (L1/UB/GM abstraction)
- `common/op_kernel/buffer_manager.h` — buffer lifecycle manager
- `common/op_kernel/buffer_mix_core.h` — mixed-core buffer
- `common/op_kernel/buffers_policy.h` — buffer allocation policies
- `common/op_kernel/matmul.h` — matmul interface
- `common/op_kernel/CopyInL1.h` — L1 copy-in helpers
- `common/op_kernel/FixpipeOut.h` — fixpipe output helpers
- `common/op_kernel/offset_calculator.h` — GM offset calculation
- `common/op_kernel/memcopy/offset_calculator_v2.h` — v2 offset calc
- `common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h` — VF softmax kernel
- `common/op_kernel/arch35/vf/vf_flashupdate_new.h` — VF flash update

**Shared common (in `deps/ascendc_common/`):**
- `op_kernel/load_store_utils.h`
- `op_kernel/math_util.h`
- `op_kernel/platform_util.h`

**ACLNN API layer (in `deps/op_api/`):**
- `include/aclnn_sparse_flash_attention.h` — public ACLNN header

**Proto definitions (in `deps/op_proto/`):**
- `sparse_flash_attention_proto.h`
- `kv_quant_sparse_flash_attention_proto.h`

**Dynamic tiling (in `deps/sparse_flash_attention_dynamic/`):**
- `sparse_flash_attention.py` — TBE dynamic tiling entry

**Python wrapper (in `deps/pylib/`):**
- `cloud_sparse_flash_attention.py` — pylib entry
- `cloud_kv_quant_sparse_flash_attention.py` — KV quant variant
- `sparse_functional.py` — functional variant

**Torch binding (in `deps/cann_ops_transformer/`):**
- `attention/sparse_flash_mla.cpp` — C++ binding for SFA
- `attention/flash_attn.cpp` — flash attention binding
- `attention/flash_mla.cpp` — flash MLA binding
- `common/aclnn_common.h` — ACLNN C++ wrapper
- `common/hccl_common.h` — HCCL wrapper
- `ops/attention/sparse_flash_mla/sparse_flash_mla.py` — Python ops layer
- `ops/attention/sparse_flash_mla/graph_convert_sparse_flash_mla.py` — graph mode

### 3.2 SDK dependencies (NOT included, record only)

These are provided by the CANN toolkit at runtime. Record version and path:

| Header | SDK path (ascend-toolkit) |
|---|---|
| `kernel_operator.h` | `$ASCEND_HOME_PATH/aarch64-linux/include/ascendc/kernel_operator.h` |
| `kernel_operator_list_tensor_intf.h` | `$ASCEND_HOME_PATH/aarch64-linux/include/ascendc/` |
| `kernel_tiling/kernel_tiling.h` | `$ASCEND_HOME_PATH/aarch64-linux/include/ascendc/kernel_tiling/` |
| `lib/matmul_intf.h` | `$ASCEND_HOME_PATH/aarch64-linux/include/ascendc/lib/` |
| `lib/matrix/matmul/tiling.h` | `$ASCEND_HOME_PATH/aarch64-linux/include/ascendc/lib/matrix/matmul/` |
| Compiled `.o` kernels | `custom_cot/op_impl/ai_core/tbe/kernel/ascend950/sparse_flash_attention/*.o` |

## 4. Build conditions

### Available tools in c4 container:
- `cmake` (3.x) ✅
- `make` ✅
- `gcc/g++` (aarch64) ✅
- `ccec` (Ascend C Compiler) ✅
- `atc` (Ascend Tensor Compiler) ✅
- `msopgen` ✅
- `bisheng` / `bishengir-compile` ✅
- `ascendc_pack_kernel` ✅

### Build attempt (standalone direct-invoke kernel):
The `cold_merge_custom` project was created and attempted via CMake with
`--npu-arch=dav-3510`. Compilation started but had errors in the first version
(TBuf position, exp function). After fixes (TBuf<>, __builtin_expf), compilation
was progressing when this snapshot was taken. See `cold_merge_custom/` directory.

### For the FULL operator rebuild:
The operator is part of the `custom_cot` vendor package. Rebuilding it requires:
1. The full project structure with op_host/, op_kernel/, op_api/, op_proto/
2. CANN's operator build system (msopgen or the internal build pipeline)
3. The tiling-data generation step
4. Packaging as .so with proper symbol registration

**Status: the full operator rebuild has NOT been attempted yet** because it
requires understanding the complete build pipeline of the custom_cot package
(which is an internal Huawei build system, not just cmake/make).

## 5. What's still missing

1. **Host-side tiling implementation** (the .cpp that computes tiling data for
   the kernel) — the dynamic/sparse_flash_attention.py exists but the actual
   host binary is compiled into libopapi.so
2. **CMakeLists.txt / build.sh for the full operator** — the custom_cot package
   uses Huawei's internal build system; a standalone CMake equivalent needs to
   be created
3. **The compiled kernel binary (.o)** — pre-compiled for specific tiling keys;
   rebuilding requires the full tiling infrastructure
4. **Integration test harness** — connecting the modified kernel to the
   experiment framework (run_kv_native_twopass.py)

## 6. What has been verified

1. ✅ Math correctness (LSE merge formula verified on NPU, 5/5 tests, rel < 2e-06)
2. ✅ Numerical stability (extreme values, no overflow)
3. ✅ Split softmax == merged (ground truth equivalence, 5/5 tests)
4. ✅ Source code structure (kernel entry → arch35 dispatch → vector service)
5. ✅ Integration point identified (ProcessVec2 L738-L749, after normalization)
6. ⬜ AscendC compilation of modified kernel
7. ⬜ On-device A5 verification
8. ⬜ Paired performance measurement

## 7. Reproducible build commands

```bash
# Environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Standalone kernel test (direct invoke):
cd cold_merge_custom
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Full operator rebuild (requires custom_cot build pipeline):
# TODO: document the exact build steps once the pipeline is understood
```
