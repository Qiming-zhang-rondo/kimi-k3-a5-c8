# 当前阶段：补齐 P→D BF16 尾窗，核实当前 worker

内网 agent 直接执行 [PD_TAIL_INTEGRATION_TASK](framework_integration/PD_TAIL_INTEGRATION_TASK.md)。本轮交付可执行身份采集、实际 worker 诊断入口、PD传输核心和worker导入助手；实际Mooncake backend hook尚未接线，D的非零起点错误不能宣布已解决。

先在实际P、D环境解析真实connector（不加载模型），有既有worker RPC则一次刷新当前状态。只推脱敏小摘要，不再让用户搬类名或上传私有源码。793a514的producer/read零计数来自旧PID/旧文件，不能判断后续请求；Graph Python计数不是replay计数。

保留已成功attention/kernel、MXFP4主cache、QLI、Graph/MTP/APC/并发及原P/D配置。无需为了身份采集重启服务；实际适配候选明确验证后再沿既有流程部署。此前性能/GPQA任务继续有效，但先解决真实PD/APC链。

---

以下为前一阶段冻结评测要求；以顶部的新任务为当前入口。

# 当前阶段：冻结已成功版本，测性能/GPQA并收齐适配缺口

用户已确认第二次bool修复后推理成功，并明确不为新推送重装。本次提交仅文档，不改变runtime；不重写kernel，不重启当前服务做文档同步。最新d3bf7b9的类型求助属于此前材料，不覆盖后续成功结论。

内网agent执行 [BENCHMARK_GPQA_TASK](framework_integration/BENCHMARK_GPQA_TASK.md)：自行定位既有bench_tpot.py与原198题GLM GPQA脚本/数据，冻结实际成功代码哈希、容器/模型/配置和评测协议。先确认真实hot128路径与零sticky faults；Graph Python计数不等于replay次数。性能沿原80请求/8000字符/128输出、并发8和32，同构建OFF/ON只切window，不改原Graph/MTP/APC/P-D和量化。完整GPQA保留原输出预算/模板/解析器，不套性能128token上限。

[ADAPTATION_STATUS](framework_integration/ADAPTATION_STATUS.md) 已给一次性清单：APC真实边界BF16配对和PD真实tail传输在公开代码中只有接口；抢占/恢复生命周期待接；MTP已有位置/路由但提交导出未闭环；Graph与chunkedprefill已有机制，补实际设备覆盖，不重复重写。

先做冷请求/实际APC命中/真实P→D的小型确认，内网已有额外接线则核实复用。缺代码时记录明确阻塞并在独立候选修，不能关闭功能或退C4后宣称原生产hot128 GPQA。不要中途换代码再把前后成绩拼一轮。

回传性能和GPQA小表、冻结身份、实际功能命中/完成状态与原产物目录即可；不上传私有源码、题目、激活或大包，不让用户反复搬文件。
