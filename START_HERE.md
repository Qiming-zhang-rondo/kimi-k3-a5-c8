# 当前任务：大chunk形状与合并批量扫描

用户最新要求优先覆盖Q32、64、128、256、512、1024，不能以Q4证明通用最优。
执行 [SHAPE_SWEEP_NEXT.md](SHAPE_SWEEP_NEXT.md)。默认包括Q1/Q4对照、K8K/K32K、
4/8/16 AIC、BF16合并8/16/32行，先返回大chunk覆盖，再继续调参。

56ab1fb已实测Q4/core4/rows16融合113.56µs，较同轮installed两遍快9.5%；
保留此已验证库与结果。本轮动态形状/ABI4和可选packed输出尚待A5验证。
原始18文件payload不改，精度门槛不变；完整逐元素和逐行检查继续执行。

先完成形状扫描，不扩展冷写融合。脚本从已验证build继承同SDK参数、隔离构建，
逐组更新summary.csv及best_by_shape.csv，不等全矩阵结束才返回结果。
设备故障停整个扫描，不reset、不换marker重试。规则见 [AGENTS.md](AGENTS.md)。
