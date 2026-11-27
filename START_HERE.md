# 当前任务：复测新版50us是否真的快于原生纯C4

用户对cfdb6c3摘要的50us vs旧80us提出质疑，明确要求内网AI实测比较。
立即按 [experimental/single_pass/KV_SPLIT_FAIR_AB_TASK.md](experimental/single_pass/KV_SPLIT_FAIR_AB_TASK.md)
执行：复用已有成功库，只更新Python测试，**无需重编kernel**。
五臂同输入：实际installed C4、同库旧C4、旧V4热窗、新S4纯C4、新S4热窗。
新增`--verify-comparison`做六组直接20轮AB/BA；全部精度/graph门禁先通过，计时后直接读取
最后timed buffer再验（不能先调用算子刷新）。交付完整原始JSON和SHA，不再只交手工摘要。
空闲卡、同scope，所有candidate计时包含attention+reduce；不换CANN/vendor、不停业务。

下面保留候选背景，不是本轮重新扫描指令。

用户于2026-11-26明确要求“你试一下”。执行
[experimental/single_pass/KV_SPLIT_TASK.md](experimental/single_pass/KV_SPLIT_TASK.md)。
本次已有真实候选代码：V4相同加载几何、TopK按列切1/2/4片，FP32 partial状态，
同stream独立AIV归约；W0/W128和原V4同库对照。宿主检查通过不代表SDK/A5编译或性能。
先隔离构建、Q4数值门禁、再graph同轮配对；不升级CANN/vendor、不覆盖原82µs已验证产物。

原框架接入继续按 [framework_integration/CURRENT_FRAMEWORK_TASK.md](framework_integration/CURRENT_FRAMEWORK_TASK.md)。
先预检模型格式；当前R64/packed400不能重解释为NoPE272。KV-split结果也不能替代真实模型评分或TPOT。
