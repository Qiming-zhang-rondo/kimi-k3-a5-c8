# 当前新实验：保留MXFP4格式，尝试V4 KV-split

用户于2026-11-26明确要求“你试一下”。执行
[experimental/single_pass/KV_SPLIT_TASK.md](experimental/single_pass/KV_SPLIT_TASK.md)。
本次已有真实候选代码：V4相同加载几何、TopK按列切1/2/4片，FP32 partial状态，
同stream独立AIV归约；W0/W128和原V4同库对照。宿主检查通过不代表SDK/A5编译或性能。
先隔离构建、Q4数值门禁、再graph同轮配对；不升级CANN/vendor、不覆盖原82µs已验证产物。

原框架接入继续按 [framework_integration/CURRENT_FRAMEWORK_TASK.md](framework_integration/CURRENT_FRAMEWORK_TASK.md)。
先预检模型格式；当前R64/packed400不能重解释为NoPE272。KV-split结果也不能替代真实模型评分或TPOT。
