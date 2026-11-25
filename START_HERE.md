# 独立候选分支：冷缓存scatter并入热环prepare

本分支 `sfa-write-prepare` 基于c2873ee，供正在进行的大形状扫描完成后单独测试。
主分支 `sfa-cold-merge` 的形状扫描和当前113µs实现保持原样，不切换正在运行的checkout。

新增候选见 [WRITE_PREPARE_NEXT.md](WRITE_PREPARE_NEXT.md)：
原生C4量化和cat打包不变，将packed冷字节scatter并入既有BF16热写与two_scan分区，
减少一次独立index_copy_调用。开关默认关闭，保留同轮direct_fused当前实现作基线。
新增完整缓存字节/整数分区/写槽图重放门禁，通过后才计时。

这是Python/Triton改动，未改AscendC库及ABI；可复用形状扫描已成功的ABI4设备库。
29项针对宿主检查通过；新Triton设备编译/A5正确性/性能均未验证，不能先算提速。
