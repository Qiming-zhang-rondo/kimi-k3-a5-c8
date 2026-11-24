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

