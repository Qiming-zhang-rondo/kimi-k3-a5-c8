# 当前任务：直接构建并验证完整的独立热环 writer

ac8bef44 的 119 行 smoke 存在裸指针调用 GetValue、GM→GM DataCopy、固定第0槽回读、缺少 AIV 编译分支隔离和跨流水事件等问题。仓库现已提供完整可编译候选替换，停止逐个猜 DataCopy 参数。

内网 agent 现在直接执行 [experimental/prolog_ring_smoke/README.md](experimental/prolog_ring_smoke/README.md) 的隔离构建及设备验证命令。复用已经配置成功的真实 CANN 9.2.1 / dav-3510 工具链，选择空闲设备，不改现有 vendor，不停服务，不升级 CANN。

本目录含完整 ASC kernel、固定 ABI 头、CMake、真实 Torch stream 薄桥和验证脚本。测试覆盖 B1Q1/B8Q4、OFF、BF16 RNE、KR、bank/position/epoch、padding、回绕、输入未改及 Graph replay 的数据变化；不需要私有 Prolog 源码外传。

外部已通过 4 项宿主测试和 C++ ABI 头语法检查；尚未在 A5 编译或执行。仅 writer 通过不等于 Prolog/C4 字节控制或服务已完成。设备通过后，把搬运/同步模式移到内网已定位的实际 FP32 KV 量化前写点，继续 [framework_integration/PROLOG_PRODUCER_TASK.md](framework_integration/PROLOG_PRODUCER_TASK.md) 的完整 ABI1 producer 任务。保留原 C4 数据/scale、PrologV3、Indexer MXFP4 以及用户要求的生产能力。

源码/构建/原始日志留内网，只回传部门允许披露的结果摘要或首个错误；不要继续上传内部头文件、schema、实现 diff、二进制或其余私有文件。
