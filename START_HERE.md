# 当前任务：先收集小JSON，继续大形状计时

b4d2163已回传Q4再次MEASURED；Q32–1024尚为CONTROL_FAILED，没有有效性能数据。
12MB原results无需上传。执行 [ZERO_ORACLE_NEXT.md](ZERO_ORACLE_NEXT.md)：
先用纯CPU收集器一次提取旧7份JSON，再复用已成功ABI4库运行Q32及后续形状。

本轮仅Python门禁与数据收集改变，不改kernel、不重编、不放宽非零query的0.001门槛。
旧strict与tie-only proof失败记录保留。新增独立零query证明必须用本次真实状态，
证明FP64加法无残差且融合结果逐BF16 bit等于精确RNE参考；不再要求有偏差的native参考
也只能差半值/1ULP。其他所有精度、分区/写入及graph门禁保持。

主分支继续大形状任务；独立写入融合候选在sfa-write-prepare，暂不混入本轮。
