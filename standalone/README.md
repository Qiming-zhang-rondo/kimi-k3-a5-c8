# Isolated arch35 compile probe

The default library target is a **baseline source compilation probe**, not a
working fused operator or private-package launch ABI. Optional `sfa_launch`
performs fixed original-BF16 device checks only with `--run --device N`; its
default is CPU preflight. After the reported 507015, follow
[`REPAIR_507015.md`](../REPAIR_507015.md) before returning to the full task.

The corrected entry uses the archived private arch35 `BaseApi` cube/vector
services, BF16 inputs/output, FP32 accumulation and the original six template
axes. It excludes the arch22 common header. AIC receives a null local tiling
pointer as in the original entry; AIV copies exactly 144 bytes (36 uint32 words)
from GM into a local POD whose lifetime encloses `Init` and `Process`. There is
no static UB buffer, rounded-up 160-byte read or GM-to-local pointer cast.

ASC has separate host and device passes. Only a compiler-selected
`__NPU_ARCH__ == 3510` pass may parse the archived arch35/VF implementation.
The host scan still sees the exported signature and the mixed-task metadata.
`SFA_BUILD_CONTEXT` messages record each pass's actual architecture/core macros;
do not define those macros manually or infer API availability from package names.

The non-template `sfa_arch35_bf16_user_scratch_probe` symbol instantiates a fixed
TND/PA_BSND template. Mode 0 and mode 1 are compiled separately; neither is
inferred from a private byte at SGL's `dispatchKey` offset. The target is a
static library and needs no host `main`. Do not report an unused template or an
empty archive as a compiled SFA kernel.

Its workspace parameter is now **raw user scratch**, forwarded unchanged into
the selected kernel. It does not call `GetUserWorkspace`, consume private runtime
tiling bytes, or include a system-workspace prefix. This is restricted to the
fixed host contract that keeps both `SyncAll` branches unreachable. See the
repair document for the source audit, exact 442368-byte allocation and checks.

## Build on the internal matching SDK

From the repository root, use the already configured toolkit which provides
`find_package(ASC)`. If discovery needs help, set its real `ASC_DIR` or
`CMAKE_PREFIX_PATH`; do not source an older custom vendor. Fresh example paths:

```bash
python3 tools/run_logged_stage.py --stage baseline_build --work-dir /tmp/sfa-standalone-evidence -- cmake -S standalone -B /tmp/sfa-standalone-mode0 -DSFA_PROBE_TEMPLATE_MODE=0 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
python3 tools/run_logged_stage.py --stage baseline_build --work-dir /tmp/sfa-standalone-evidence -- cmake --build /tmp/sfa-standalone-mode0 --verbose -j8
python3 tools/run_logged_stage.py --stage baseline_build --work-dir /tmp/sfa-standalone-evidence -- cmake -S standalone -B /tmp/sfa-standalone-mode1 -DSFA_PROBE_TEMPLATE_MODE=1 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
python3 tools/run_logged_stage.py --stage baseline_build --work-dir /tmp/sfa-standalone-evidence -- cmake --build /tmp/sfa-standalone-mode1 --verbose -j8
```

Run each next command only after the previous command succeeds. Use different
fresh build paths if these directories exist. SDK-specific include fixes can be
supplied with `-DSFA_SDK_INCLUDE_DIRS='dir1;dir2'`, from that same toolkit only.
Keep compiler errors and repair ordinary source/build issues without requiring
the user to relay individual files. Do not manually define AIC/AIV compiler
macros to make only one half pass.

For a build result, commit the actual verbose logs, commands, source hashes,
device artifact hashes, and symbol/metadata evidence for both AIC and AIV and
the exported probe kernel. A normal exit from CMake configuration is not a
device compilation result. Default desktop tests use host stubs and do not
claim any ASC compilation.

If compilation fails, collect the build configuration and SDK header evidence
in one step. The internal agent must resolve the actual active SDK root from
its build configuration; it does not need to ask the user for individual files:

```bash
python3 tools/collect_sfa_compile_evidence.py --build-dir /tmp/sfa-standalone-mode0 --output /tmp/sfa-compile-evidence-new --sdk-root "$ASCEND_HOME_PATH"
```

Use `ASCEND_HOME_PATH` only if it names the same toolkit selected by ASC. If not,
pass the actual toolkit path from CMake/compiler configuration. Use a fresh
output directory and include the original failed build log alongside this
evidence. The collector does not execute any command it reads. A collected
report is not a successful compilation.

The 9.2.0 [public SDK header](https://github.com/hicann/asc-devkit/blob/02af6bddfd75b48d17cd2ec0d4da5bb3700cde44/impl/basic_api/kernel_macros.h#L119)
places `MicroAPI = Reg` **inside AscendC**, guarded by `__NPU_ARCH__ == 3510`.
Thus `no member MicroAPI` alone does not prove a missing/newer SDK is required.
Host-pass parsing, a wrong target, or a different resolved header must first be
distinguished using actual pass logs and header evidence. A made-up alias shim
cannot establish API signatures and is intentionally removed.

Require real 3510 AIC and AIV context/artifacts; a host-only archive containing
the guarded signature is not an SFA device build pass. If a 3510 device pass
still fails, retain that exact first error and investigate its actual resolved
headers. The host-pass fix is a hypothesis pending SDK execution, not a claim
that all current build errors are solved.

## Before device execution

The POD is included **unchanged** from pinned SGL commit `597250fc`, with its
full source notice. This only establishes the schema used by this compile
probe. `upstream/FIELD_COMPARISON.json` explicitly does not establish private
host/kernel ABI compatibility. Before running, use a serializer matched to the
schema compiled into this kernel, not opaque bytes from the installed private
operator. A new fixed-shape host writer is allowed if every consumed field,
workspace size, blockDim and template choice is established from the selected
source and checked explicitly. An independent matched public host/kernel build
is another route. Merely having the same total struct size is insufficient.

The unqualified `diff=0` and code-generator-origin statements in commit
`5ad9dbd` were report text without the underlying comparison/build evidence.
Record the exact two file paths, versions and hashes for any new equality
claim. The pinned SGL/vLLM **main arch35 headers in this repository differ** from
`original/`; see `upstream/PROVENANCE.md`. A matching packaged file is a narrower
claim and must identify its scope.

Once the isolated baseline builds and passes its mathematical reference on A5,
apply the existing `experimental/` cold-state epilogue and finish its matching
host/API bridge. Run correctness controls before paired performance, preserving
the baseline quantization recipe. No speedup has yet been measured for this
native fusion candidate.

References: [SGL GM-to-local loading example](https://github.com/sgl-project/sgl-kernel-npu/blob/597250fcc1b31f2607bf817d42bd0699ea60b8eb/csrc/sparse_flash_attention/op_kernel/sparse_flash_attention.cpp#L79),
[official tiling macro contract](https://asc.gitcode.com/api/SIMD-API/basic_api/tool_interface/tiling_data_acquisition/GET_TILING_DATA_WITH_STRUCT.html),
[official ASC static-library build](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00039.html).
