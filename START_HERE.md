# 当前任务：先接自研hotwindow V4最新融合版，检查整体精度

用户已明确先接我们自己的MXFP4+BF16路线，DSL cache迁移暂不实施。
按 [framework_integration/KV_SPLIT_VA_TASK.md](framework_integration/KV_SPLIT_VA_TASK.md) 一次执行：
先在实际目标容器收集环境/预检格式，再自主完成隔离安装、真实模型smoke及原GLM评分。
环境检查是同一任务的第一步，不再等待用户逐项转发版本信息。CANN9.3仍为既定目标；不擅自改运行中容器。
PR199 head342680fb的写点/worker原文件和VA c09d3e20 hook签名已审阅，安装可用`--source-profile pr199`。

## 接入内容与范围

用户明确要求框架接入。按 [framework_integration/KV_SPLIT_VA_TASK.md](framework_integration/KV_SPLIT_VA_TASK.md)
执行已有真实hook和新增`--single-pass-kernel kv_split4`：Q<=4走split4+设备归约，较大prefill明确走旧V4。
复用已经成功的库，隔离VA/MXLLM源码安装；先检查实际checkpoint/缓存格式，再真实模型OFF/V4/KV-split三臂精度。
当前仍是accuracy_eager接线验证，不能拿它测TPOT或宣称生产图/MTP/PD已支持。R64packed400与NoPE272不可混用。
如实际模型格式不兼容，agent自主按任务收齐真实原生源合同并提交必要adapter，不让用户逐项搬文件。

上一轮20轮完整原始数据仍需从既有文件打包：按
[experimental/single_pass/KV_SPLIT_FAIR_AB_TASK.md](experimental/single_pass/KV_SPLIT_FAIR_AB_TASK.md)
顶部CPU导出命令完成，禁止重跑来补已经存在的数据；这不是继续单算子摸高的前置。

下面保留候选背景，不是本轮重新扫描指令。

用户于2026-11-26明确要求“你试一下”。执行
[experimental/single_pass/KV_SPLIT_TASK.md](experimental/single_pass/KV_SPLIT_TASK.md)。
本次已有真实候选代码：V4相同加载几何、TopK按列切1/2/4片，FP32 partial状态，
同stream独立AIV归约；W0/W128和原V4同库对照。宿主检查通过不代表SDK/A5编译或性能。
先隔离构建、Q4数值门禁、再graph同轮配对；不升级CANN/vendor、不覆盖原82µs已验证产物。

原框架接入继续按 [framework_integration/CURRENT_FRAMEWORK_TASK.md](framework_integration/CURRENT_FRAMEWORK_TASK.md)。
先预检模型格式；当前R64/packed400不能重解释为NoPE272。KV-split结果也不能替代真实模型评分或TPOT。
