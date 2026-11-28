# 冻结成功版本，测性能和完整GPQA

用户已明确：第二次布尔修复后推理成功，开始性能与GPQA。保持当前实际成功的服务实现，**不为了同步新commit重装或重启**。
最新d3bf7b9是此前类型求助材料，不覆盖用户后续成功结论；不再重写kernel。
本文件是测试任务，不是要求部署新runtime。

## 一次收齐当前运行身份

内网agent自行定位原服务启动脚本、实际worker runtime、已有bench_tpot.py和原GLM GPQA脚本/198题集合。
本机未找到可替代的原评测脚本，不能用GPT-OSS测试、另一套题或新答案解析器代替，不让用户搬运私有文件。
记录实际执行文件/reader库SHA、容器与SDK/Python/Torch身份、模型/权重/tokenizer身份、P/D和TP配置、原启动参数、attention/additional config。
测试脚本和题目只记录本地路径/摘要，不上传数据。工作文件有本地正确修复也可冻结，不强行改成仓库某个SHA。

原W4A8 megamoe、main KV C4、indexer MXFP4、Graph、MTP、APC、P/D、util和scheduler参数保持。
不试alg2/max7、不加重复Hadamard、不换C8、不强改Prolog选择。之后任何运行代码变更都开启新run id，不混拼成绩。

## 先确认这个成功版本的实际覆盖

保持原生产功能，先复用原curl检查：冷请求、一次有真实命中记录的APC请求，以及真实P→D请求（若当前服务为PD）。
不要把同提示HTTP成功自动当作APC命中，也不要把直连P的回复当作PD成功。
按 [ADAPTATION_STATUS](ADAPTATION_STATUS.md) 核对当前内网是否另有tail接线；公开runtime只有接口，不能默认已完成。

在实际worker通过已有collective_rpc调用`serving_runtime.validate_worker_serving`，计时外检查非空reader fault buffers且全部fault=0。
ON需要确认真实Native writer/V4路径，不能静默退C4。`NOT_EXERCISED`不通过；新独立Python进程的空状态不能验证服务。
Python producer/read计数只是辅助，Graph replay可能不经过Python，不要求每个实际Graph请求使计数递增。
必要时只对一条请求做计时外trace或实际设备状态检查；不要在每层forward插入tolist/同步/日志。
已确认的路径不重新扫单算子。若APC/PD门槛失败，保留这次失败，优先补已确认的适配缺口，不能关闭功能后把分数称为原生产配置结果。

## 性能：复用原bench_tpot.py

- 客户端并发8、32；每档80请求、8000字符输入、max_output=128，使用原相同prompt集合/seed及warmup策略。
- 客户端并发不等于server max_num_seqs；不为conc32悄悄改服务调度上限。记录二者。
- OFF/ON使用同一成功代码和原启动配置，只切sfa_recent128.window=0/128；保持量化器与indexer配置。
- NPU不与其它压测/GPQA抢资源，不在性能测量同时跑GPQA；不停止无关服务。不能隔离时如实标记结果受干扰，不宣称无回退。
- 优先完成当前ON的首组数据与GPQA；OFF使用同构建独立测试实例或同测试部署按现有流程顺序切换，不为了同步文档重启现服务。
- 严格性能结论需同轮OFF/ON并记录实际顺序，尽量做AB/BA配对；若时间只够一轮，标明单轮观测，不把旧69ms历史数当新基线。
- 记录成功/失败数、实际输入/输出tokens、wall duration、输出tok/s、TTFT和TPOT的原脚本定义及mean/p50/p95。
- 保留每请求原始结果；汇总ON/OFF配对比值。包括内存峰值、可用C4 blocks、热环容量与每rank占用。

GPQA和性能不并发跑；切臂/加载/编译/预热不计入稳态时间。相同APC冷热状态策略必须一致，不全局清理其他用户缓存。

## 精度：原198题GPQA

使用原完整集合及相同题目ID/顺序/模板/seed/sampler/最大输出/答案解析和评分器。
**性能测试的128输出上限不适用于GPQA**，GPQA保持原评测预算及推理设置。
先保留当前ON完整结果。若已有OFF完整198题且模型/量化/协议/配置完全一致，可作为明确注明来源的精度对照；否则同冻结构建补跑OFF。
性能OFF仍需同构建实测，旧精度对照不能替代性能基线。

必须完成198/198，报告正确数/198和百分比；HTTP失败、超时、截断、答案解析失败单列，不能从分母删除。
中途结果标为中途，不用78/94等片段归因。OFF/ON逐题对齐后报告：OFF对→ON错、OFF错→ON对和不一致题数。
不把单算子L2改善直接当GPQA收益，结论以这轮完整模型结果为准。

## 回传小表即可

性能：arm、client_concurrency、server_max_num_seqs、success/total、TTFT p50/p95、TPOT p50/p95、output tok/s、实际tokens、HBM/blocks、配对比值。
GPQA：arm、completed/198、correct/198、score%、HTTP/timeout/truncated/unparsed数量、实际评测设置、两类正确性翻转数。
再附冻结代码/配置摘要、实际hot路径/fault结果、APC命中与PD完成状态、原始产物目录。私有源码/题目/激活/大包不上传。
