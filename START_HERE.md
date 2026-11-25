# 当前先执行：原生 C4 reader 源码定位与推送

用户最新要求：让内网 agent 尝试推送匹配版本的原生 MXFP4 C4 kernel/host tiling 源码，供外部实现冷 C4 + 热 BF16 的单次 attention。

先完成 [framework_integration/C4_SOURCE_EXPORT_TASK.md](framework_integration/C4_SOURCE_EXPORT_TASK.md)。自行定位实际调用链、最小源码依赖和构建来源，按任务要求尝试提交；本轮不重跑微基准、不停止业务、不修改现有算子。目标仓库公开，仅上传允许外发的源码；若明确受限，提交不含受限代码的接口与来源摘要。不要把普通文件查找继续转给用户。

完成短交接后继续下面的框架精度任务，保留已验证基线。

---

# 当前任务：框架接入与真实模型精度验证

用户已收口单算子调优，停止 Q/tiling/写融合扫描。本分支保留 26661f7 已完成证据和算子源码，新增实际 MXLLM worker overlay。

直接执行 [framework_integration/FRAMEWORK_EXECUTION_TASK.md](framework_integration/FRAMEWORK_EXECUTION_TASK.md)，安装说明与范围见 [framework_integration/README.md](framework_integration/README.md)。自行定位实际模型/源码/环境，修复常规适配并提交代码与小型结果，无需用户逐项中转。

顺序完成：隔离安装 → 同权重同量化 OFF / 原生 recent128 / 融合 recent128 → 用户原 GLM 精度任务 → 整理算子团队交接材料。模型生成成功不等于任务精度通过；无融合覆盖不等于融合通过。初版明确为单请求 eager 精度模式，不能称为服务性能完成，默认不关闭 PROLOG_V3。

本地 28 项 CPU 检查通过；实际模型/NPU 验证待内网执行。不重复已完成的单算子扫描，不覆盖原服务和 vendor 包。
