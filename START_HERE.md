# 当前任务：修复Prolog命名参数转发，继续原服务启动

最新错误提交 `52db594`：Python wrapper 绑定时 rope_sin 重复，尚未进入该算子的计算。已审阅Cloud源只将9项转为位置参数，rope_sin/cos原本保留在kwargs；不能按报告猜测“30多个位置参数重复”并做覆盖合并。

本轮修复将9个值按原Cloud参数名称还原，与原kwargs合并后全部按名称传给新provider，避免新wrapper的形参顺序差异。显式None保留，真实重复键明确拒绝；OFF沿用原调用。15项相关宿主检查通过，包含会复现原冲突的不同参数顺序32参wrapper；不是A5/服务运行结果。

内网agent直接执行 [SERVING_TASK当前入口](framework_integration/SERVING_TASK.md) 的52db594章节：同步两份Python适配文件到现有隔离框架，启动新worker；撤销私有wrapper中临时按位置猜测/丢None/覆盖键的merge，按真实同名参数转发。沿用已安装variant，不升级CANN、不重编内核或reader、不重做baseline构建，也不要求用户搬私有源码。

随后继续真实producer OFF/ON对照与原服务启动，保留并发8、Graph、MTP、APC和P/D配置。latent/KR/tag真实写入及BF16尾窗恢复仍按既有任务核验，不能因Python参数绑定通过就宣布服务精度/性能已完成。
