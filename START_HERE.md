# 当前任务：只打包已有20轮复测原始结果，不重跑

4ffc47c回报新版热窗50.31us/installed C4 80.99us，但只推送42行摘要，不能独立核对原始轮次和门禁。
**现在不要重编、不要再跑NPU。** 从刚才实际--verify-comparison输出目录找到完整results.json，
按 [experimental/single_pass/KV_SPLIT_FAIR_AB_TASK.md](experimental/single_pass/KV_SPLIT_FAIR_AB_TASK.md)
最顶部命令，用新增stdlib工具导出32KiB文本分片+manifest，直接git add整个导出目录并推送。
工具读取既有字节，保留全部controls/graph/post/20轮event+wall，并验证SHA可精确恢复。
不能把手写fair_ab_20rounds.json传给工具；它会拒绝这种摘要。
用户不需要再搬运日志；agent自主查上次命令/输出路径，无需新测量。

下面保留候选背景，不是本轮重新扫描指令。

用户于2026-11-26明确要求“你试一下”。执行
[experimental/single_pass/KV_SPLIT_TASK.md](experimental/single_pass/KV_SPLIT_TASK.md)。
本次已有真实候选代码：V4相同加载几何、TopK按列切1/2/4片，FP32 partial状态，
同stream独立AIV归约；W0/W128和原V4同库对照。宿主检查通过不代表SDK/A5编译或性能。
先隔离构建、Q4数值门禁、再graph同轮配对；不升级CANN/vendor、不覆盖原82µs已验证产物。

原框架接入继续按 [framework_integration/CURRENT_FRAMEWORK_TASK.md](framework_integration/CURRENT_FRAMEWORK_TASK.md)。
先预检模型格式；当前R64/packed400不能重解释为NoPE272。KV-split结果也不能替代真实模型评分或TPOT。
