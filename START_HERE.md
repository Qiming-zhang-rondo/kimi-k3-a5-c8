# 当前并行任务：真实模型精度 + 冷热互斥读取

用户最新要求是直接接框架使用。单遍算子源码已完成；7a2fad回传摘要报告AIC4/W12894.66µs、installed两遍107.45µs、精度PASS。尚未证明赢过所有历史最强，冠军A/B不再是框架接入前置。

直接执行 [framework_integration/SINGLE_PASS_FRAMEWORK_TASK.md](framework_integration/SINGLE_PASS_FRAMEWORK_TASK.md)。新增 --arm single_pass 已接到实际MXLLM reader，传原始TopK、完整物理C4 pool、BF16热环和位置tags；每64heads组一次candidate attention，CPU完整FP64数学对照不再调用第二次NPU attention。原写量化/TopK保持，window0原路。先复用本轮已通过的ABI1库，不必为额外性能修改等重编。

本地37项框架CPU检查通过，尚未跑真实模型。内网依次完成隔离安装→真实权重prefill/decode smoke→用户原精度任务OFF/ON，并上传实际库SHA、逐层调用/数学覆盖和任务分数。普通路径/编译/API问题自主解决，不逐项向用户要文件。R64/packed400是当前单遍范围，R0/272B不能错误接入或默默用旧两遍称通过。

这是accuracy_eager测试版，包含CPU参考和head补齐；不能拿它测TPOT或称图/并发/MTP/PD已完成。保留PrologV3默认选择，若显式共同NATIVE精度实验必须OFF/ON同设置且在权重转换前。单遍数学.01门槛与standalone一致，旧ABI4 .001门槛不变。

可选冠军对比脚本 experimental/single_pass/run_champion_ab.py 分别提供prepared_read和write_read同轮AB/BA，旧ABI4最强cold-store路径保留；只有用户后续需要再跑，不阻塞模型精度。singlepass脚本今后自动产生evidence_summary.json，请上传机器生成小证据，不再只手写15行summary。生产适配和算子团队交接以真实模型精度结果为准。

## 用户重新开启读取优化（不阻塞框架任务）

新增同库独立routed入口，热行跳过C4读取和解码。按 [experimental/single_pass/ROUTED_READ_TASK.md](experimental/single_pass/ROUTED_READ_TASK.md) 隔离构建，`--routed-ab` 先精度后Q4同轮graph配对。保留原overlay入口及已有94µs报告，不承诺新延迟。模型接入继续用已验证overlay库，新候选通过前不默认替换。禁止自动重跑历史扫描或让用户逐项中转。

## 最新读取优化：cold-pair快速路径

e6ae40c的同库原始证据确认routed117.64µs，比overlay97.64µs慢20.5%，输出一致。当前有效实现仍用overlay，不能把旧routed称优化成功。用户授权继续实现冷pair直接原生搬运，最新任务为 [experimental/single_pass/FAST_PAIR_TASK.md](experimental/single_pass/FAST_PAIR_TASK.md)。先Q4同库三臂精度/graph配对，`--fast-pair-ab`；保留所有原始结果。80µs只是争取目标，未测前不宣称达到。框架模型精度继续并行，不等待本轮摸高。
