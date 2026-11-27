# 当前任务：先接自研hotwindow V4最新融合版，检查整体精度

用户最新补充要求：先用新脚本同轮比较纯C4、纯BF16与最新融合版的精度/读取性能，按
[experimental/single_pass/ATTENTION_COMPARE_TASK.md](experimental/single_pass/ATTENTION_COMPARE_TASK.md)
直接跑一组Q4/K8192。复用已成功库，不重编内核；结果自动输出L2、cosine、恢复比例与完整配对记录。
完成后继续下面的框架精度接入。

用户已明确先接我们自己的MXFP4+BF16路线，DSL cache迁移暂不实施。
按 [framework_integration/KV_SPLIT_VA_TASK.md](framework_integration/KV_SPLIT_VA_TASK.md) 一次执行：
先在实际目标容器收集环境/预检格式，再自主完成隔离安装、真实模型smoke及原GLM评分。
环境检查是同一任务的第一步，不再等待用户逐项转发版本信息。CANN9.3仍为既定目标；不擅自改运行中容器。
PR199 head342680fb的写点/worker原文件和VA c09d3e20 hook签名已审阅，安装可用`--source-profile pr199`。

最新回报`74b0773`停在LLM构造前，尚未执行attention。本轮已补正式MXLLM全局初始化、
`sfa_recent128`配置桥接，以及spawn进程的platform安装点。内网agent拉取后在**新隔离PR199 checkout**
重装三文件overlay，继续同一份engine/prompt的OFF/V4/KV-split精度验证；不要直接导入worker，
不要用环境变量默认值掩盖未初始化的注册项。失败时提交自动生成的`startup.json`、
`startup_traceback.txt`、`probe.json`与环境身份，保留完整traceback，不再只提交手写摘要。

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
