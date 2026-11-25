# 当前优先：原生 C4 + BF16 recent128 单次 attention 原型上 A5

C4源码交接已完成：2a2beb51，41个source文件MANIFEST已核对；不要重复导出或继续旧两遍参数扫描。

直接执行 [experimental/single_pass/A5_EXECUTION_TASK.md](experimental/single_pass/A5_EXECUTION_TASK.md)。代码、独立构建入口、Torch当前stream桥和精度/配对脚本已交付；候选每次只有一次attention，保留原TopK，无独立cold attention、外部分区或merge。先Q4/K8192/AIC4编译+精度，全部通过再graph同轮性能。常规编译/API差异自行修复并提交，不逐项让用户中转。

这是源码原型：本地9项CPU专项检查通过，未冒称AscendC编译、NPU精度或性能已通过。首版热行仍先过原C4解码再由BF16覆盖，性能需测。原生C4控制入口保留，实际源码与新72B自有tiling host/device共同重编，绝不拿猜的POD喂安装包。

本机推送接收器已恢复：真实GitHub push被202接受并由官方Codex queue返回0，目标为当前聊天；按sfa-framework-recent128分支路由、忽略[sfa-review]自提交，无定时轮询。工具见 tools/github_push_supervisor.py。密钥/部署配置只留本机，不在仓库。

单遍设备验证通过后继续 [framework_integration/FRAMEWORK_EXECUTION_TASK.md](framework_integration/FRAMEWORK_EXECUTION_TASK.md) 的真实模型精度任务；不把单算子结果称为模型或服务通过。
