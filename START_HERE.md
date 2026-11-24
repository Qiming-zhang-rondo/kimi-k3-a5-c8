# 当前任务：直接测融合性能

执行 [PERFORMANCE_NOW.md](PERFORMANCE_NOW.md)。**复用3d2011e6已经编译并运行的ABI2设备库，
不重编内核，不重复CPU诊断；拉取最新Python后，用新进程、新输出目录直接执行带精度门禁的graph计时。**

83a4001的实际诊断确认：零query融合输出131072个值全部等于独立FP64→BF16 RNE参考；
与旧两遍的4096处差异全为精确半值位置、相邻1 ULP。非零query原0.001门禁已通过。
本轮只为零query增加严格的现算RNE证明分支，保留原比较及其失败记录；其他门禁不变。

今天的交付是四臂同轮性能：原生C4 W0、已安装两遍、direct两遍、direct融合。
最终比较融合/已安装两遍，不能拿历史延迟或仅给定冷态的smoke计时当基线。

保留现有容器、SDK、vendor与baseline量化配方；不停止服务、不升级CANN、不reset设备。
普通运行问题由现场代理连续处理并提交证据，不再要求用户逐项中转。
设备故障立即停止，保留原故障marker，不用新目录掩盖故障。

历史构建和修复记录：[PERF_REPAIR_5407.md](PERF_REPAIR_5407.md)、
[ROUNDING_NEXT.md](ROUNDING_NEXT.md)、[experimental/torch/DIRECT_BRIDGE.md](experimental/torch/DIRECT_BRIDGE.md)。
执行规则见 [AGENTS.md](AGENTS.md)。
