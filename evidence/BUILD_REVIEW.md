# 构建证据审查：e6722fe

外部审查已读取本提交全部三个 evidence 文件。当前结论为 **BUILD_REPORTED**，
可继续 POST_COMPILE_EXECUTION_TASK.md 阶段A；实际AIC/AIV产物证据待补。

| 文件 | 仓库实际大小 | 检查结果 |
| --- | --- | --- |
| build_log.txt | 446 bytes | 8行人工构建摘要；报告archive111444B、3510及CUBE/VEC，但没有原始编译输出/退出码 |
| kernel_hash.txt | 0 bytes | 空文件，没有实际产物SHA256 |
| kernel_size.txt | 0 bytes | 空文件，没有实际成员/设备代码大小 |

这不是新的MicroAPI故障，也不要求用户再次收集文件。内网执行代理在同轮自主补齐：

1. 对 `/tmp/sfa_m0f/libsfa_standalone.a`（先确认真实产物路径）直接计算SHA256和字节数，
   保存非空输出；列出archive成员及SDK设备metadata/sections，确认两个device image和真实kernel名。
2. 保存实际verbose构建stdout/stderr、退出码、compile_commands.json及CMake配置。
   原始日志若未保留，需要复现构建来取得日志，不能把上述摘要重命名为原始日志。
   可复用tools/run_logged_stage.py和tools/collect_sfa_compile_evidence.py，提交前验证文件非空。
3. 继续原版direct launcher、匹配tiling/workspace和compact128 NPU精度，再接融合与配对性能。
   不因等待外部再确认而停在补材料阶段；构建证据与启动代码一起推进。

当前没有提交新的launcher、A5输出或性能数据；不能将BUILD_REPORTED写作设备精度或融合已通过。
