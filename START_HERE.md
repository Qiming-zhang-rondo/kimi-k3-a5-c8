# 当前任务：保留PROLOG_V3的生产热环接入

用户已明确要求可支持的并发8、整模型Graph、MTP、prefix cache和P/D传输全部接入，保留原C4缓存布局和indexer C4。
最新54076f89的17:24:36真实错误：所有配置字段已接受，hooks.weights进入runtime构造，
但当前NATIVE raw-BF16 tap不覆盖生产PROLOG_V3。runtime尚未初始化完成，尚无候选kernel执行证据。
不再为此改JSON字段，不删除guard放行缺失数据，也不关闭生产特性后称完成接入。

内网agent现在按 [framework_integration/PROLOG_PRODUCER_TASK.md](framework_integration/PROLOG_PRODUCER_TASK.md)
一次执行：核实实际Cloud Prolog schema/可选原BF16出口；定位当前匹配源/版本/头文件；
用tools/collect_prolog_sources.py原字节导出完整源码与构建材料，自动小文本分片后推送。
如果已有真实可复用出口，验证后复用；否则在隔离版本化variant内部RMSNorm/RoPE后、MXFP4量化前写BF16热环。
保留原量化输入、原packed400写入、原query输出及原图内原地cache更新，不增加重复投影或整cache复制。
工具仅做stdlib只读采集，3项宿主测试通过；没有新的Prolog device实现或性能结果。

生产适配同时需要多请求row→request/ring-bank寻址、预分配Graph缓冲、MTP事务/有效位置、prefix尾窗快照和PD尾窗完成协议。
现有accuracy_eager adapter是受保护的诊断原型，不能直接删除单请求/PD/图门禁冒充生产支持。
原C4布局不妨碍这些能力；新增BF16热环的数据来源与生命周期仍必须接线。

旧配置问题及已确认字段见 [framework_integration/HOT_WINDOW_ACTIVATION_REPAIR.md](framework_integration/HOT_WINDOW_ACTIVATION_REPAIR.md)。
当前additional_config正确，indexer_kv_dtype=mxfp4和mla_kv_quant_mode=mxfp4均保留。
旧框架精度探针合同见 [framework_integration/KV_SPLIT_VA_TASK.md](framework_integration/KV_SPLIT_VA_TASK.md)，它不代表生产能力已交付。

不重启或修改现有服务、不更换CANN/vendor/wheel、不覆盖已成功库。新增真实激活重要性实验继续暂缓；
此前未发布草稿不包含在本轮提交。单算子三臂脚本仍可独立使用，本轮不重跑扫描。
