# 当前优先：修复单遍 W0 的基础性能差距

e1af1f5回传单遍精度PASS摘要，但W0=301.78µs、W128=339.22µs、installed C4=80.44µs。性能未达标，不能主要归因热行重复解码，因为W0已经3.75×。

直接执行 [experimental/single_pass/PERFORMANCE_RECOVERY.md](experimental/single_pass/PERFORMANCE_RECOVERY.md)：先用只读脚本提取刚才完整结果（真实AIC/flags/原始C4 direct/配对各轮），用现有库只确认Q4 AIC1/4；再隔离重编新Release/O2+window0原生旁路，并测原始C4/W0/W128。不要重扫其他形状或继续盲改VF，不逐项让用户中转。

已修复默认Q4 AIC1→自动4，disabled recent map写入，W0直接原生C4入口，显式优化构建；本地3项针对性检查通过，新修改的A5性能尚未测。摘要保留，不编造实际AIC已确认或性能已恢复。

候选仍为一遍attention，原TopK与数学门槛不变，旧两遍有效实现和旧库保留。设备fault立即停止。返回小型原始证据到experimental/single_pass/evidence，不上传大npz。单遍达到可用性能后再继续真实模型框架精度任务。
