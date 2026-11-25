# 内网执行：真正单次 attention 原型

源码已收齐，本轮直接编译和测试 `experimental/single_pass`，不再重复源码导出，不再扫描旧两遍/epilogue 变体。用户要求尽快拿到可实测原型，常规编译、include、API适配由你自主修复并提交，不逐项让用户中转。

1. 保留原服务、原安装包、ABI4 和已验证两遍路径。选择之前实际通过 C4 实验的容器/SDK和空闲 NPU，记录真实来源。`native_c4_source/PROVENANCE.md` 中 runtime/version 与 descriptor hash 不等于二进制重编匹配证据；不要因此跨版本 source 旧包。
2. 先按 README 构建 `libqsfa_recent128_direct.so`。新的 72B tiling 是本原型 host/device 共同编译的自有 ABI，不是安装包 tiling；不要改回猜测的旧结构，也不要给 SDK 手动定义架构宏。host/device 头隔离沿用已验证方法。
3. 如果发现编译器 API/头大小写/缺公共 include，基于本机真实 SDK 和当前 source 修复；原型代码保持隔离。必须实际看到 AIC/AIV device 产物，不能把仅 host link 成功称 BUILD_PASS。先不运行 benchmark。
4. 构建通过后，仅在空闲卡先跑 Q4/K8192/AIC4 `--timing none`，用现有经过 SHA 校验的 hotkv payload。三条 direct 路径是原始 C4、candidate W0、candidate W128，每条仅一次 ASC attention launch。原始 C4 冷 reader及两遍只在控制组使用，不能把它塞回候选来消掉失败。
5. 候选必须保留原 TopK、真实 BF16 ring、随机物理页表和原量化。先全有效2048，冷1920/热128，未支持形状不得静默替换。完整输出对独立数学参考、最差行、finite 门禁均通过才记 VALIDATED_ONLY。保存 W0 对原始 C4、mixed 对两遍的原始差异；两遍中间有BF16舍入，不能要求其与单遍逐位一致后再随意放宽门槛。
6. 若精度通过，独立输出目录跑同轮 `--timing graph --repeats 10`。输出投毒 gate 必须通过；交替配对报告 candidate vs installed C4 与 installed two-pass。标明 prepared-read-only，不是全流水线/TPOT，不直接拿历史111us比。
7. 若出现真实设备fault，立刻停止后续设备调用并保留 fault marker/首个错误；不要重置卡、换CANN或继续测性能。普通构建错误继续修复，不等用户逐条转发。
8. 提交现场代码修改和小型 `experimental/single_pass/evidence/a5_summary.json`，含commit、CANN/SDK、library SHA、build状态、每条门禁原始数值、真实调用数、配对每轮/中位数、失败与未测项。大npz/log留内网，不必上传12MB。当前首次代码交付不预先声称BUILD_PASS/MEASURED。

首版热行仍经过原C4解码后被BF16覆盖。若慢于两遍，这一轮先给真实数，下一步优化热行旁路及DMA批量，不以复用两次attention伪装成单遍。

单算子单遍验证通过后再接框架候选入口，沿用 `framework_integration` 的精度任务；不要把本轮单层结果称作模型精度通过或生产无损。
