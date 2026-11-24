# 单次完整执行入口

**当前优先任务：[POST_COMPILE_EXECUTION_TASK.md](POST_COMPILE_EXECUTION_TASK.md)。**
**5407a8e 的305.84µs及历史比值已撤销性能验收资格。当前按
[PERF_REPAIR_5407.md](PERF_REPAIR_5407.md) 构建真实current-stream桥，
运行真实C4三臂的精度门禁，再做同轮完整流水线计时。旧PhaseC入口已封禁。**
最新提交 `5a361849953a35b22695f98abcbe42e17623f777` 的
[设备结果](https://github.com/Qiming-zhang-rondo/kimi-k3-a5-c8/blob/5a361849953a35b22695f98abcbe42e17623f777/standalone/sfa_fusion_device_results.json)
已确认 `sfa_fusion_launch` 的 8 项全部通过、nonfinite 全为0，状态 `DEVICE_SMOKE_VALIDATED`。
范围是 `given_fp32_cold_state_plus_native_bf16_hot`；真实 C4 流水线、原生两遍比较和真正
hot=0 调度均尚未验证，不能标记阶段B整体完成。
**真实 C4 冷态与当前 NPU stream direct bridge 源码已提供，现场编译并完成原生两遍严格对照，
通过后连续进行三臂配对性能。不要停在重跑已通过的8项。**
[PHASE_B_2168_REVIEW.md](PHASE_B_2168_REVIEW.md) 保留已完成修复和smoke复现命令。
`2ddc663`已提交3组原版A5结果，状态DEVICE_VALIDATED，L2约0.00175–0.00191；
当前继续阶段B融合接入和阶段C配对性能，不重复从头修原版启动。
此前507015的修复记录及复现入口保留在 [REPAIR_507015.md](REPAIR_507015.md)；设备故障门禁继续有效。
保留当前成功构建、原版设备精度和融合smoke结果，继续真实C4接入及同轮三臂性能，
随后连续执行，无需阶段间向用户确认。
SDK恢复工具仅在实际再次出现构建故障时使用，不把已通过的编译从头重做。

内网执行代理请阅读 [AUTONOMOUS_EXECUTION_TASK.md](AUTONOMOUS_EXECUTION_TASK.md)，
在已有工作容器中自主完成依赖发现、可复现构建、融合实现、正确性检查和配对性能测试。
将源码、可复现命令、阶段状态和实际日志提交到 `sfa-cold-merge` 分支，供外部 Codex 直接审查。

历史原版编译报告曾遇到SDK平台初始化崩溃；若实际再次发生，按 [PLATFORM_RECOVERY.md](PLATFORM_RECOVERY.md) 执行自动诊断并继续原任务；
不要把芯片显示名当成SDK目标，也不要仅据状态摘要断言硬件无权限。真实tiling定义参考已收在 `upstream/`。

这次交付的终点是可调用并经过真实验证的候选，或有实际执行证据的外部阻断。
一次 include 缺失、一次编译失败、仅有公式验证、仅写接入说明，都不算完成。

辅助工具：

- `tools/collect_source_closure.py`：在明确给出的源码目录中收集项目依赖和真实 tiling 定义，保留路径、哈希及歧义。
- `tools/run_logged_stage.py`：以明确 argv 运行构建/测试，保留退出码、日志和状态；不把输出中的 PASS 字样当成功。
- `tools/probe_original_sfa.py`：隔离原版源码并调用实际上传的编译 adapter；记录选中源码、生成头与新内核产物。

已经有可继续构建的[原生源码候选](experimental/README.md)，包含实际内核接线和设备向量合并代码。
沿用已验证原版的scratch/启动协议，在这些源码上完成融合构建、A5正确性和性能，不需要重新从零写算法。

这些工具只是执行辅助，不能代替 A5 验证或尚未实现的host/API生成注册。
SDK adapter 平台层不通时，可先按 [standalone/README.md](standalone/README.md) 继续隔离的
arch35 原版编译探针；该路径使用公开 POD，只做编译，不默认兼容私有 tiling ABI。
执行代理需要根据真实工程完成余下工作，并用工具留下可审查证据。
收集器退出0只表示指定必需文件和类型定义已找到；仍须检查 unresolved 清单并实际编译。
日志工具的 PASS 只表示命令成功退出；精度测试必须用失败退出码表达数值门禁失败。

评测沿用 `baseline` 量化配方。`scale_alg=2,dst_type_max=7` 在用户模型评测中出现约两点下降，
尚未归因；本轮融合实验不要混入该参数变化。
