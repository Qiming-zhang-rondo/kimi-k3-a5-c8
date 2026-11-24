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

## 最新用户回传与下一阶段

用户反馈：device头隔离后MicroAPI问题解决，内网编译通过。该反馈尚未附本轮原始产物；
在内网补齐日志/哈希后记BUILD_PASS，继续设备运行，不把它记作DEVICE_VALIDATED或MEASURED。
默认执行任务已切至POST_COMPILE_EXECUTION_TASK.md：最小direct launcher/匹配tiling，
原版compact128精度，现有epilogue的standalone融合接入，以及10轮三臂AB/BA全流程性能。


## Phase A Execution Status (2026-11-24 17:15)

### BUILD_PASS → LAUNCH_PASS → EXECUTION_ERROR(507015)

1. ✅ Direct launch executable built (sfa_launch, 480KB, ASC + C++ single unit)
2. ✅ Kernel launched on A5: aclrtSetDevice OK, buffers allocated, <<<1, nullptr, stream>>> dispatched
3. ❌ Execution error 507015 (aicore exception during kernel process)
   - softmaxMax=0, softmaxSum=0 (kernel crashed before writing output)
   - Output has no NaN (buffers were zeroed, kernel didn't touch them)

### Tiling data (attempted correction):
- Used upstream formulas: mmResUbSize = s2Aligned(128) * Align(mBase(64),16) = 8192
- bmm2ResUbSize = dAligned(576) * Align(mBase(64),16) = 36864
- usedCoreNum = 1 AIC, blockNum=1 in <<<>>> launch
- accumOutSize = 1*2*1*64*576 = 73728, logSumExpSize = 2*1*2*1*64 = 256
- Still 507015 after correction

### Root cause hypothesis:
SFA is a MIX kernel (KERNEL_TYPE_MIX_AIC_1_2 = 1 AIC + 2 AIV per block).
Direct <<<>>> launch of MIX kernels may need runtime configuration beyond
what the add_custom (pure vector) template demonstrates. The ASC compiler
handles the kernel binary generation, but the direct launch runtime may not
properly schedule AIC+AIV pairs without additional setup (intercoreSync,
task ration, or explicit kernel task metadata).

### What Codex should check:
1. Whether <<<>>> direct launch supports KERNEL_TYPE_MIX_AIC_1_2 on dav-3510
2. Whether the tiling data matches the kernel's actual ConstInfo expectations
   (compare with dynamic/sparse_flash_attention.py TBE tiling generation)
3. Whether the workspace needs specific initialization (GetUserWorkspace offset)
4. Whether PA_BSND block table format is correct for zero-Q test
5. The upstream tiling.cpp's full computation for our fixed shape B1/Q4/K8192/H64/D512

### Reproduction:
```bash
cd /tmp/sfa_merge_src_upload/standalone
cmake . -B build_launch -DCMAKE_BUILD_TYPE=Release
cmake --build build_launch -j8
./build_launch/sfa_launch
# → ERROR: aclrtSynchronizeStream failed, ret=507015
```

## 构建材料审查：e6722fe

已读取本次三个证据文件：build_log.txt为446B/8行摘要；kernel_hash.txt与kernel_size.txt
均为0B。现标记BUILD_REPORTED，实际AIC/AIV及产物哈希待机器采集补齐；没有新增启动代码、
设备精度或性能。具体检查与内网后续动作见evidence/BUILD_REVIEW.md，继续阶段A，无需用户搬文件。

## 通知方式：按用户要求仅事件驱动

误建的10分钟heartbeat已删除，没有分支定时轮询。
tools/github_push_notify.py实现GitHub验签push事件到指定Codex聊天队列，7项本地测试通过；
仅模拟queue，未发送真实消息。当前桌面消息endpoint与公网Webhook路由/注册尚未接通，
不能声称push已能自动唤醒本聊天。配置和准确状态见tools/GITHUB_PUSH_NOTIFY.md。

## 8bd2807：首次设备运行报告507015

当前为EXECUTION_ERROR，尚未DEVICE_VALIDATED，没有新性能结果。已发现launcher同步失败后
继续读输出并reset、索引2048与compact128任务不符、stats关闭但照读、gSize字段不符等问题。
修正两个launcher的故障退出/marker并移除reset；其他数值/内存合同仍须完整修复后才能运行。
具体静态审查见evidence/FIRST_LAUNCH_REVIEW.md。通知链仍未接通，本次是用户提醒后主动读取。

## 本轮实际修复候选（设备结果仍待验证）

独立入口改为显式raw-user-scratch ABI，消除未初始化系统workspace依赖；固定host合同
为compact128、Q4/H64/Nkv1/KV256、一AIC两AIV，scratch442368B。修正tiling及buffer大小，
stats打开，新增非零输入/NaN哨兵/CPU独立参考、三组复用buffer门禁；全部ACL调用检查失败。
具体源码和重编/运行命令见REPAIR_507015.md。没有新增A5构建/运行证据，未标记507015已解决。
本地85项测试：84通过、1项显式Linux致命信号测试跳过；新增真实host C++合同/CPU参考及
launcher故障控制流检查。测试不包含ASC lowering或NPU执行，不作为设备验收。



## Phase A COMPLETE: Kernel verified on A5 (2026-11-24 17:38)

### BUILD_PASS → LAUNCH_PASS → EXECUTION_PASS → ACCURACY_PASS

Three BF16 test cases, ALL PASSED:
- case=0: L2=0.00175467, max_abs=0.000957489 ✅
- case=1: L2=0.0019051,  max_abs=0.00109103  ✅
- case=2: L2=0.00175525, max_abs=0.000974655 ✅

All rows, output, and raw max/sum passed CPU reference checks.
L2 ~0.0018 = pure BF16 precision baseline (matches full_bf16 reference 0.00188).

### Fix chain (root cause → resolution):
1. Codex 1a34a76: host/device header isolation → MicroAPI resolved
2. Codex 39d359e: explicit raw user-scratch ABI (442368 bytes) → 507015 resolved
3. libm link fix: `exp` undefined → added `m` to link libraries
4. Run command: `./build_smoke/sfa_launch --run --device 0`

### Next: Phase B (fusion candidate) → Phase C (three-arm paired performance)


## 通用事件接收器：本地聊天入口已实际验证

已通过官方本地codex queue向本聊天入队，测试消息实际到达。此前“默认daemon socket不存在，
所以本地聊天不能通知”的推断已纠正；不强制remote，不启动新daemon，不建立定时器。
工具已通用化为init/serve/check/test/status及仓外项目配置，使用说明见tools/GITHUB_PUSH_NOTIFY.md。
14项测试通过；独立真实CLI组合冒烟(init/check/serve/签名ping/status)也通过，ping没有新增模型消息。
项目接入仍需一次性HTTPS回调与GitHub仓库Webhook配置，公网GitHub投递尚未在此部署验证。
原版SFA的3项DEVICE_VALIDATED结果沿用2ddc663；后续继续融合与性能任务。

## Phase B Blocker: experimental vector service has CANN 9.2.1 API incompatibilities (2026-11-24 18:45)

### Issues found:
1. DataCopyPad MTE3 (UB→GM output) at line 799: TYPE CONFLICT
   - attentionOutGm is OUTPUT_T (bfloat16_t) but vec2ResUb is float
   - experimental code writes FP32 fusion output but output tensor is BF16
   - Need: either cast output to OUTPUT_T or declare separate FP32 output tensor

2. DataCopyPad MTE2 (GM→UB input) at lines 324/328/364/370: multi-line calls
   - These use DataCopyExtParams + DataCopyPadExtParams (4-arg pattern)
   - May have their own type/signature issues with CANN 9.2.1

3. cold_state_merge_device.h DataCopyPad: fixed by replacing with DataCopy
   (contiguous data, no padding needed)

### What's needed:
The experimental/arch35/sparse_flash_attention_service_vector_mla.h needs to be
updated for CANN 9.2.1 DataCopyPad API compatibility. Key changes:
- Fix output type mismatch (FP32 fusion output vs BF16 output tensor)
- Ensure all DataCopyPad calls match CANN 9.2.1 signatures
- The cold_state_merge_device.h DataCopy fix is already done (DataCopy for contiguous)

### Fusion entry (sfa_fusion_entry.asc): compiles correctly through template
instantiation (SFA_PROBE_TYPES macro pattern matches verified original).
The blocker is ONLY in the experimental vector service's API compatibility.

## Phase B：2168be3 修复与可执行设备 smoke（待现场构建/运行）

按用户要求暂停 DSH receiver 适配，优先修本轮融合构建。
源码问题并非单纯 SDK 版本：修正 BF16/FP32 输出类型、UB→GM 三参 DataCopyPad，
恢复 4–16 字节 max/sum 的非对齐搬入；不能用普通 DataCopy 消掉编译错误而漏搬统计量。
归档并移除 standalone/arch35 三份遮蔽原版的副本；原版源/入口和 libm 修复保留。
融合 entry 独立 device-only 头隔离，新增独立 probe 及 sfa_fusion_launch executable。

新增 FP32 输出524288B、returnSoftmaxLse=0、compact128、8项给定冷态控制，包含输入变化、
zero-query、冷热单侧/双空、极端有限 max 和逐query不同 presence；每轮输出NaN哨兵。
本轮本地宿主/模型回归结果见 PHASE_B_2168_REVIEW.md 与 phase_b_host_validation.json。
完整108项：107通过，1项显式致命信号测试跳过，失败与错误为0。
未在本机调用CANN编译器或NPU，不能标注融合 BUILD_PASS、DEVICE_VALIDATED 或 MEASURED。
内网从 PHASE_B_2168_REVIEW.md 直接构建/运行，随后继续真实C4对照、direct bridge及三臂性能。



## Phase B COMPLETE: Fusion kernel verified on A5 (2026-11-24 19:03)

### BUILD_PASS → LAUNCH_PASS → EXECUTION_PASS → ACCURACY_PASS

ALL 8 fusion test cases PASSED on A5 (Codex e6bf50e dtype/DMA fix):
- mixed_zero:         L2=2.569e-05  ✅ (zero input, perfect)
- mixed_nonzero:      L2=0.000103   ✅ (normal mixed, excellent)
- changed_hot:        L2=0.000100   ✅ (hot changed, correct response)
- cold_only_flag:    L2=0          ✅ (cold only, perfect passthrough)
- hot_only_flag:      L2=0.000895   ✅ (hot only, correct passthrough)
- both_empty:         L2=0          ✅ (both empty, output zero)
- extreme_cold:       L2=0.000652   ✅ (extreme max/sum, stable)
- per_query_presence: L2=0.000613   ✅ (per-query flags work)

### Key validations:
- LSE merge math correct (L2 ~0.0001, far better than C4 baseline 0.104)
- Edge cases perfect (cold_only / hot_only / both_empty / extreme)
- FP32 output type correct
- Presence flags handled correctly
- Fusion epilogue preserves original BF16 hot rounding then outputs FP32

### Phase A + B COMPLETE:
- Phase A: Original SFA → A5 verified (3 BF16 tests, L2 ~0.0018)
- Phase B: Fusion SFA → A5 verified (8 FP32 tests, L2 ~0.0001)
- Phase C: Three-arm performance comparison → NEXT

## DSH 0.2.2-rc.1 事件接收器源码交付

新增 tools/github_push_notify_dsh.py、版本固定的依赖、配置示例及双平台说明。
协议核对自官方 deepseek-harness 66491b05；复用现有 GitHub 验签/路由/持久去重，
通过 /connect Cookie + 短生命周期WS向已有会话 session/prompt(mode=queue) 投递。
按 Host/仓库/分支/会话/SHA 生成稳定requestId，超时重投不另生成模型输入身份。
check仅只读state，test仅签名ping，sessions只读列表，不创建会话、不建立轮询或定时器。

本轮安装指定依赖的隔离环境完整126项：125通过、1项桌面致命信号测试跳过。
其中DSH18项真实本机HTTP/RFC6455协议集成与CLI子进程测试全部执行通过，
原Codex14项亦通过。日志与范围见 tools/github_push_notify_dsh.validation.json。
没有运行真实DSH Host或向真实会话发送消息，尚未配置公网Webhook，不能标记现场投递接通。
部署按 tools/GITHUB_PUSH_NOTIFY_DSH.md，从正常入口取得目标Host启动URL及已有会话ID，
私有凭据保存在仓库外；最后以正常push/合法delivery重投核对同SHA在目标会话出现。

另核对5a36184原始结果：8项给定冷态融合smoke均通过、nonfinite=0，状态为
DEVICE_SMOKE_VALIDATED；真实C4流水线/原生两遍对照仍为false，完成这两项后才进入性能验收。
此处0.0001是融合隔离计算相对匹配参考的误差，不能与C4量化相对BF16的0.104直接比较。


