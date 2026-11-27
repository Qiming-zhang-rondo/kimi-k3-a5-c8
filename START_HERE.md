# 当前任务：验证KV-split归约的UB复用同步修复

8791d2e已交付原始S1诊断：partial0对数学参考L2约0.00094，final对BF16(partial0)约1.14–1.16。
分片计算在这两项S1检查中正常，故优先修归约。已补每行MTE2搬入前的V_MTE2反向依赖，
防止下一行覆盖上一行VF尚未读完的partialUb；**尚未在A5验证修复效果**。
按 [experimental/single_pass/KV_SPLIT_TASK.md](experimental/single_pass/KV_SPLIT_TASK.md)
最顶部任务，保留旧库SHA，增量重编隔离候选，先精度门禁、通过后同轮配对计时。
不改producer/ABI/cache布局，不换SDK/vendor，不覆盖已验证V4。

下面保留候选背景，不是本轮重新扫描指令。

用户于2026-11-26明确要求“你试一下”。执行
[experimental/single_pass/KV_SPLIT_TASK.md](experimental/single_pass/KV_SPLIT_TASK.md)。
本次已有真实候选代码：V4相同加载几何、TopK按列切1/2/4片，FP32 partial状态，
同stream独立AIV归约；W0/W128和原V4同库对照。宿主检查通过不代表SDK/A5编译或性能。
先隔离构建、Q4数值门禁、再graph同轮配对；不升级CANN/vendor、不覆盖原82µs已验证产物。

原框架接入继续按 [framework_integration/CURRENT_FRAMEWORK_TASK.md](framework_integration/CURRENT_FRAMEWORK_TASK.md)。
先预检模型格式；当前R64/packed400不能重解释为NoPE272。KV-split结果也不能替代真实模型评分或TPOT。
