# 当前阶段：冻结已成功版本，测性能/GPQA并收齐适配缺口

用户已确认第二次bool修复后推理成功，并明确不为新推送重装。本次提交仅文档，不改变runtime；不重写kernel，不重启当前服务做文档同步。最新d3bf7b9的类型求助属于此前材料，不覆盖后续成功结论。

内网agent执行 [BENCHMARK_GPQA_TASK](framework_integration/BENCHMARK_GPQA_TASK.md)：自行定位既有bench_tpot.py与原198题GLM GPQA脚本/数据，冻结实际成功代码哈希、容器/模型/配置和评测协议。先确认真实hot128路径与零sticky faults；Graph Python计数不等于replay次数。性能沿原80请求/8000字符/128输出、并发8和32，同构建OFF/ON只切window，不改原Graph/MTP/APC/P-D和量化。完整GPQA保留原输出预算/模板/解析器，不套性能128token上限。

[ADAPTATION_STATUS](framework_integration/ADAPTATION_STATUS.md) 已给一次性清单：APC真实边界BF16配对和PD真实tail传输在公开代码中只有接口；抢占/恢复生命周期待接；MTP已有位置/路由但提交导出未闭环；Graph与chunkedprefill已有机制，补实际设备覆盖，不重复重写。

先做冷请求/实际APC命中/真实P→D的小型确认，内网已有额外接线则核实复用。缺代码时记录明确阻塞并在独立候选修，不能关闭功能或退C4后宣称原生产hot128 GPQA。不要中途换代码再把前后成绩拼一轮。

回传性能和GPQA小表、冻结身份、实际功能命中/完成状态与原产物目录即可；不上传私有源码、题目、激活或大包，不让用户反复搬文件。
