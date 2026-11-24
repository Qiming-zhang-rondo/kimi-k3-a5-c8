# 当前状态：构建平台故障正在定位

收到的内网状态提交：`36445a85d3a3b9f8dbe0e8191ed9913628e00146`。
该提交报告简单ASC模板部分阶段可编译，但原版SFA尚未编译成功，候选尚未在A5运行。
原报告保留在上述Git历史；本轮没有收到完整原始编译日志、probe_report或SFA内核产物。

## 本轮已处理

- 修正平台探针：保存SDK导入、setter模块/签名/返回、目标查询与编译前后的检查点。
  即使子进程SIGSEGV，父进程仍可看到最后到达的阶段。
- 新增自动平台诊断：从真实SDK平台配置提取完整950目标，逐候选在独立进程编译，保留退出码、信号、原始日志和产物校验。
  不把芯片显示名或SHORT_SOC_VERSION当成完整编译目标，不修改安装包/设备/父进程环境。
- 已定位固定commit的真实公开tiling定义、配套host源码与kernel，保存在upstream，并校验10个原始源码/许可证blob的Git SHA和SHA256。
  同名结构存在版本差异，未自动接到私有二进制；见upstream/PROVENANCE.md。
- Mac默认测试已取消主动真实SIGSEGV，改用模拟状态；真实致命信号测试仅Linux显式opt-in，避免用户桌面Python崩溃弹窗。

## 验证范围

| 项目 | 当前结论 |
|---|---|
| 本地测试 | 共62项，61通过、1项真实致命信号测试按默认桌面规则跳过 |
| 融合内核源码 | experimental候选已实现；本轮没有虚构新设备结果 |
| 原版SFA AscendC编译 | 内网报告失败；需运行PLATFORM_RECOVERY.md的自动恢复诊断 |
| 候选AscendC编译/API注册 | 尚未通过设备侧验证 |
| A5精度及配对性能 | 尚未验证；126.17us为此前两遍路径历史基线 |
| 物理硬件访问失败根因 | 未证明；get_platform_info/set_soc的崩溃不足以锁定硬件权限问题 |
| 私有tiling ABI | 未通过公开同名类型推定；只允许匹配来源的host/kernel整体构建 |

内网代理继续按照START_HERE执行，直接提交命令、日志与结果。普通平台配置/编译错误自行处理，不要求用户逐项搬运。

## 内网进展报告（commit 5ad9dbd，尚无原始日志）

### Verified working:
1. ASC language pipeline: find_package(ASC) resolves, ccec with --npu-arch=dav-3510 accepted
2. Tiling struct SparseFlashAttentionTilingDataMla: sgl-kernel-npu public POD accepted by compiler
3. Report claims upstream kernel identical to custom_cot (diff=0); exact compared paths/hashes were not committed.
4. Platform diagnostic: 4 candidates all fail with "Npu Arch Version[3002] is not support"
   in TBE tiling layer (sparse_flash_attention_tiling.cpp:1618)

### Current blocker (advanced from missing tiling struct):
ConstInfo field mismatch between sparse_flash_attention_common.h (old, has gSize/batchSize)
and util_regbase.h (arch35, has n2Size/s1Size/dSize). The arch35 kernel code accesses
fields from util_regbase.h's ConstInfo, but the standalone entry includes common.h's version
first. Fix: remove #include "sparse_flash_attention_common.h" from standalone entry;
let arch35 kernel_mla.h pull in util_regbase.h's ConstInfo via its own include chain.

Report attributes GET_TILING_DATA_WITH_STRUCT to libaihac_codegen.so; the binary
inspection evidence was not committed. An explicit GM-to-local load is possible
and is already used in the pinned SGL entry; no generator-origin assumption is
required for that path.

## 外部代码审查后已修正（未在 A5 编译/运行）

- `standalone` 误用了 arch22 模板，现改回真实 arch35 BaseApi cube/vector 服务。
- 移除旧 common.h 和 static UB 假本地指针；AIV 精确复制144B到本地 POD，AIC 保持nullptr。
- 新增非模板导出入口，固定实例化 Init/Process；构建目标改为无需main的静态库。
- 使用仓库归档源码和原样 pinned POD；取消写死的9.2.1/vendor路径。
- 下一步按 `standalone/README.md` 提交真实编译产物和日志，然后完成匹配host/tiling再上设备。
- 当前没有新增 A5 数值或性能结果；原有 BF16 热环126.17µs仍是用户回传的两遍实现成绩。
- 本地66项宿主测试：65通过、1项Linux显式开启的致命信号测试跳过。新增4项用真实入口和pinned POD、mock服务验证AIC/AIV×mode0/1；不冒充AscendC编译或设备验证。


## 内网初始推断记录：MicroAPI 不可用（后来发现 SDK 中存在别名）

### Evidence
- vf_basic_block_utils.h:58 references `AscendC::MicroAPI::CastTrait`
- Full SDK search: `namespace MicroAPI` NOT FOUND in CANN 9.2.1 (aarch64-linux/asc/include/)
- Public SDK has `AscendC::CastTrait` at basic_api/reg_compute/kernel_reg_compute_utils.h:207
  (same type, different namespace — MicroAPI is a private/extended API)
- custom_cot 9.1.0 was compiled with a DIFFERENT SDK that provides MicroAPI

### Implication
The custom_cot arch35 kernel source cannot be compiled with CANN 9.2.1 toolkit.
Need either:
(a) The matching SDK version that provides AscendC::MicroAPI, or
(b) A compatibility shim mapping MicroAPI to public AscendC reg_compute API, or
(c) Rewrite vf_basic_block_utils.h to use public AscendC API only.

### What compiled successfully before this point
- ASC language mode (CMake find_package(ASC) + ccec --npu-arch=dav-3510)
- Tiling struct from sgl-kernel-npu (public POD)
- All include paths resolved
- ConstInfo issue fixed by Codex (ae36ad6)
- Error occurs in vf_basic_block_utils.h (MicroAPI dependency)

## 内网第二次推断记录（namespace 归属尚未由实际预处理确认）

### Codex fix ae36ad6 result: PARTIALLY EFFECTIVE
- ✅ Fixed ConstInfo field mismatch (n2Size/s1Size/dSize now resolve)
- ✅ Fixed tiling data 144-byte GM→local POD copy
- ✅ Fixed template instantiation (TND/PA_BSND, mode 0/1)
- ❌ Did NOT fix MicroAPI issue — same `no member named 'MicroAPI' in namespace 'AscendC'` error

### MicroAPI Root Cause (refined)
1. SDK kernel_macros.h:122 defines `namespace MicroAPI = Reg;` (alias to Reg namespace)
2. This alias exists in the SDK but is NOT accessible as `AscendC::MicroAPI`
3. The vf_basic_block_utils.h references `AscendC::MicroAPI::CastTrait` — fails because
   `MicroAPI` is not a member of namespace `AscendC` (it likely exists at global scope
   or inside a different namespace)
4. Verified by attempting `namespace AscendC { namespace MicroAPI { using namespace ::AscendC; } }`
   compat shim → redefinition error proves `MicroAPI` alias already exists elsewhere
5. Attempted explicit `#include "impl/basic_api/kernel_macros.h"` → same error persists

### Fix Needed
Rewrite `AscendC::MicroAPI::CastTrait` etc. in vf_basic_block_utils.h to the correct
namespace path for CANN 9.2.1, or provide the matching SDK version that defines
MicroAPI inside namespace AscendC (as used by custom_cot 9.1.0 build).

### Build evidence
- Build command: cmake --build /tmp/sfa_m0e -j8 (mode 0, dav-3510)
- Error file: deps/sparse_flash_attention/arch35/common/op_kernel/arch35/vf/vf_basic_block_utils.h:58
- Same error in both Codex entry (ae36ad6) and manual compat shim attempts
- Full compile_commands.json available at /tmp/sfa_m0e/

## 外部审查：不能据上述摘要判定 SDK 缺 API

`1462a84b` 只提交了状态文字、未使用的compat shim和一个内部头include，
没有提交实际编译日志、compile_commands.json或当前SDK头。故失败pass尚未确认。
公开asc-devkit 9.2.0源码（02af6bdd）kernel_macros.h:119–124明确是
`namespace AscendC { namespace Reg {} ... namespace MicroAPI = Reg; }`，
别名受 `__NPU_ARCH__ == 3510` 控制。它不支持“别名在AscendC之外”的推断。

已修正独立ASC入口：仅3510设备pass解析原arch35/VF头和实现；host pass保留
导出签名和MIX_AIC_1_2任务元信息。错误设备架构明确拒绝，绝不手动补架构宏。
加入SFA_BUILD_CONTEXT编译日志，移除内部kernel_macros.h直接include与猜测类型的shim。
这针对有依据的host/device阶段混编假设；实际机器是否因此恢复，仍须重编确认。

`tools/collect_sfa_compile_evidence.py`一次收集实际编译配置及匹配SDK头的原始字节、
SHA和宏上下文，不执行编译器/NPU、不修改源码/SDK。下一轮直接提交完整日志和此证据，
不能仅提交摘要或再次凭namespace错误要求换SDK。命令与判定见standalone/README.md。

本轮本地73项宿主测试：72通过、1项Linux显式fatal-signal测试跳过。入口控制覆盖
4条设备分支、2条host分支及错误架构拒绝；证据收集器覆盖只读、缺失和路径逃逸。
这是宿主控制测试，实际ASC构建/A5正确性仍未通过；不宣称新增性能收益。

