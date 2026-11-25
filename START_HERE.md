# 当前任务：框架接入与真实模型精度验证

用户已收口单算子调优，停止 Q/tiling/写融合扫描。本分支保留 26661f7 已完成证据和算子源码，新增实际 MXLLM worker overlay。

直接执行 [framework_integration/FRAMEWORK_EXECUTION_TASK.md](framework_integration/FRAMEWORK_EXECUTION_TASK.md)，安装说明与范围见 [framework_integration/README.md](framework_integration/README.md)。自行定位实际模型/源码/环境，修复常规适配并提交代码与小型结果，无需用户逐项中转。

顺序完成：隔离安装 → 同权重同量化 OFF / 原生 recent128 / 融合 recent128 → 用户原 GLM 精度任务 → 整理算子团队交接材料。模型生成成功不等于任务精度通过；无融合覆盖不等于融合通过。初版明确为单请求 eager 精度模式，不能称为服务性能完成，默认不关闭 PROLOG_V3。

本地 28 项 CPU 检查通过；实际模型/NPU 验证待内网执行。不重复已完成的单算子扫描，不覆盖原服务和 vendor 包。
