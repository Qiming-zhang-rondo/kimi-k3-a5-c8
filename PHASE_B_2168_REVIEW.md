# Phase B：2168be3 集成问题与修复

本次修复基于 `2168be3e421c8fb87775d1248fdb8747b1483334` 的文件审查。
本地没有 CANN/A5 工具链；以下修复最初由宿主回归检查，设备结论来自内网提交。
最新 `5a361849953a35b22695f98abcbe42e17623f777` 的
[原始结果](https://github.com/Qiming-zhang-rondo/kimi-k3-a5-c8/blob/5a361849953a35b22695f98abcbe42e17623f777/standalone/sfa_fusion_device_results.json)
已确认融合设备smoke **8/8 passed、nonfinite全为0**，状态 `DEVICE_SMOKE_VALIDATED`。
scope 为 `given_fp32_cold_state_plus_native_bf16_hot`；`c4_pipeline_validated`、
`native_twopass_compared`、`real_hot_empty_dispatch_validated` 全为 false。
原版 3 个 DEVICE_VALIDATED 用例仍是 `2ddc663` 的证据；尚无融合性能结论。
当前下一步是本文末尾的真实C4/direct bridge/原生两遍严格对照，随后三臂性能，
不是再次停在已通过的8项隔离smoke。

## 已修正

1. **融合入口选错输出类型。** 新入口原来把第四个模板类型设为 `bfloat16_t`，
   而融合 epilogue 要求 FP32。现在 Q/KV 仍为 BF16，accumulator 和最终融合输出均为 FP32。
   冷状态与两遍参考的合同保留；helper 仍显式执行热侧 BF16 rounding 后再参与合并。
   不把整个融合结果回写 BF16 来消除编译错误。
2. **头文件遮蔽原版。** 新增的 `standalone/arch35/` 三份融合头会被原版入口的
   quoted include 优先选中，改变已验证原版的 Init 签名。它们已移至
   `standalone/archive/2168be3/*.snapshot` 原样保存且不再参与编译。
   `configure_sfa()` 不添加 experimental include；融合入口显式包含
   `../experimental/arch35/sparse_flash_attention_kernel_mla.h`。
3. **小统计量不能替换为普通 DataCopy。** `count<=4` 个 FP32 max/sum 只有 4–16 字节。
   普通连续 DataCopy 按 32 字节向下取整，此处会搬运零字节。候选统一使用
   `experimental/arch35/cold_state_merge_device.h` 中字节长度的 GM→UB DataCopyPad，
   并在 32 字节 stats scratch 行内补齐。没有为了对齐而越界读取更多 GM 元素。
4. **区分搬入与搬出接口。** GM→UB 保留四参 DataCopyPad（含 padding），
   UB→GM 使用三参 DataCopyPad；融合输出两端类型均为 FP32。
   输出 DataCopyExtParams 零初始化，包括预留字段。
5. **宿主扫描不解析 device-only 头。** 融合入口和已验证原版一样只在
   `__NPU_ARCH__==3510` 分支包含 MicroAPI/模板实现；导出签名和混合任务元数据仍供宿主扫描。
   错误架构在头文件解析前拒绝。
6. **独立编译目标。** 新增 `SFA_BUILD_FUSION_PROBE=ON` 和 `sfa_fusion_probe`，默认关闭。
   原版 target 不引用融合入口。保留内网验证所需的 `libm` 链接修复。
7. **新增真正调用融合入口的 smoke executable。** `SFA_BUILD_FUSION_SMOKE=ON` 构建
   `sfa_fusion_launch`：一个 ASC 编译单元直接启动 `sfa_cold_merge_probe`，分配 524288 字节
   FP32 输出与独立冷态、presence buffer，复用原版已验证 raw-user-scratch 协议。
   融合 tiling 明确关闭 returnSoftmaxLse；不返回或宣称 combined stats。

API 依据：官方 [DataCopy 连续搬运](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopy_GMAndUB_continuous.html)
说明 count×sizeof(T) 必须 32 字节对齐，否则向下取整；
[UB→GM DataCopyPad](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_UBToGM.html)
为三参；[GM→UB 四参接口](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0265.html)
单独传 padding。正式编译仍以现场匹配 CANN 的真实头文件为准。

## 宿主验证

```bash
python3 -m unittest discover -s tests -p test_phase_b_source_contract.py -v
python3 -m unittest discover -s tests -p test_standalone_entry.py -v
python3 -m unittest discover -s tests -p test_device_merge_reference.py -v
python3 -m unittest discover -s tests -p test_candidate_interface_contract.py -v
python3 -m unittest discover -s tests -p test_sfa_fusion_smoke_data.py -v
python3 -m unittest discover -s tests -p test_sfa_fusion_launcher.py -v
```

新增 7 项检查：实际融合入口的 AIC/AIV mock 实例化、FP32 类型合同、19 个外部参数及
5 个冷状态参数的顺序、144 字节 tiling 拷贝和生命周期、宿主扫描隔离、错误架构拒绝、
原版/候选头文件隔离、小 stats DMA 及三参输出 DMA 源码回归。实际 merge 头的算术、空侧与 DMA
长度由已有 `test_device_merge_reference.py` 执行验证。
这些是普通 C++ 宿主检查，不能替代 AscendC lowering 与 NPU 运行。

本轮完整回归共108项：107通过，1项桌面致命信号测试按既定规则跳过；失败/错误均为0。
原始日志见 phase_b_host_validation.log，机器可读范围与哈希见 phase_b_host_validation.json。
launcher另验证了507015后停止所有后续ACL调用，以及空kernel不写输出时正确报CONTROL_FAILED。

## 已完成 smoke 的构建与运行复现

以下命令保留作复现背景；本轮已有通过结果，继续末尾的接入任务。
仅在相关源码或接口变更后按需回归，不重复以smoke通过作为最终交付。

在原版已通过的同一容器、同一匹配 SDK 环境，从仓库根目录执行。保留之前成功使用的
ASC_DIR、编译器、SDK include 等实际参数；下例不安装或切换 SDK。若已有设备故障标记，
先停止设备链并保留记录，不能通过另建目录绕过故障门禁。

```bash
set -e
SFA_PHASE_B_ROOT=/tmp/sfa-phase-b-2168-repair
mkdir -p "$SFA_PHASE_B_ROOT/run"
python3 tools/run_logged_stage.py --stage candidate_build \
  --work-dir "$SFA_PHASE_B_ROOT/evidence" -- \
  cmake -S standalone -B "$SFA_PHASE_B_ROOT/build" \
  -DSFA_BUILD_FUSION_PROBE=ON \
  -DSFA_BUILD_FUSION_SMOKE=ON \
  -DSFA_PROBE_TEMPLATE_MODE=0 \
  -DSFA_SDK_ROOT="$ASCEND_HOME_PATH"
python3 tools/run_logged_stage.py --stage candidate_build \
  --work-dir "$SFA_PHASE_B_ROOT/evidence" -- \
  cmake --build "$SFA_PHASE_B_ROOT/build" \
  --target sfa_fusion_probe sfa_fusion_launch -j8 --verbose
"$SFA_PHASE_B_ROOT/build/sfa_fusion_launch"

# 在现场自主检查设备占用后设置 SFA_PHASE_B_DEVICE，不停止现有服务。
: "${SFA_PHASE_B_DEVICE:?先设置已确认空闲的设备编号}"
python3 tools/run_logged_stage.py --stage accuracy \
  --work-dir "$SFA_PHASE_B_ROOT/evidence" --cwd "$SFA_PHASE_B_ROOT/run" -- \
  "$SFA_PHASE_B_ROOT/build/sfa_fusion_launch" --run --device "$SFA_PHASE_B_DEVICE"
```

保留完整构建命令、SDK/编译器信息、退出码和 AIC/AIV 产物证据。
`sfa_fusion_probe` 只是静态库；实际设备测试运行 `sfa_fusion_launch`。
结果位于 `$SFA_PHASE_B_ROOT/run/sfa_fusion_device_results.json`，日志/退出码在 evidence 下。
host-only 检查也可无需 SDK 编译实际 launcher：

```bash
c++ -std=c++17 -O2 -x c++ -DSFA_FUSION_HOST_PREFLIGHT_ONLY=1 \
  -DSFA_PROBE_TEMPLATE_MODE=0 -Istandalone standalone/sfa_fusion_launch.asc \
  -o /tmp/sfa-fusion-host-preflight
/tmp/sfa-fusion-host-preflight
```

8 项设备 smoke 覆盖 zero-query（热质量128、给定冷质量1920）、非零query、热KV变化、
cold-only/hot-only/both-empty presence、极端有限 max 和同一调用内逐query不同presence。
不存在的冷侧用 NaN poison，输出每次用 NaN 填充以检查漏写；所有 hot 索引仍为有效 compact128。
因此空侧项目验证的是 epilogue 的 presence 逻辑，**不代表原生调度支持真正 hot=0**。
独立数学参考沿用原 smoke 的整体/逐行 L2≤0.01，不是放宽原生两遍对照的0.001门槛。

本程序接收的是合成、已知 FP32 冷态。此次通过只记录 DEVICE_SMOKE_VALIDATED，明确
`c4_pipeline_validated=false`、`native_twopass_compared=false`，并不覆盖真正hot=0调度。

## 当前继续执行：真实冷态、直接桥与性能

1. 将真实原生 C4 冷 output/max/sum 接入同一候选；由真实原生BF16热输出/stats构造
   `NativeReference`，同时保留旧两遍实际输出作控制，按整体及逐行L2≤0.001比较。
   不能以合成冷态的smoke或它的数学参考0.01门槛代替真实C4/原生两遍验收。
2. 为旧 harness 实现真实 direct bridge（或完整新 ACLNN 注册），接入当前 NPU stream、
   异步 tiling/workspace 生命周期；不能调用只有声明的新符号或回退旧两遍冒充融合。
3. 全部控制通过后按 POST_COMPILE_EXECUTION_TASK.md 做 W0 / 两遍 / 融合三臂同轮性能。

遇普通编译/API问题在现场连续修复并推送源码及原始日志，无需让用户逐个中转。
任一设备故障立即停止后续 ACL 调用，不 reset、不重试、不测速；保留 DEVICE_FAULT.json。
当前已有给定冷态的融合设备smoke结论；真实C4融合精度、原生两遍严格对照和性能仍待完成。
