# 当前任务：Prolog热环仅在内网实施，不再外传源码

用户已明确45个文件的批量外传不符合部门要求；之前源码采集/上传要求撤销。
不要再上传剩余源码、私有头/schema/diff/二进制，也不将任何外传作为继续实现的条件。
此前采集器不再作为本任务执行入口。所有私有实现、构建与原始测试证据均留内网；
只回传部门允许披露的能力结论和脱敏结果，不允许外发的内容不回传。

最新42b0f78已提供5个关键文件，足够发现当前选择混入旧接口：Python为V3 wrapper，
但所选ACLNN头/kernel入口是旧V1；DynamicQuantQnWithMulQr处理Qn/Qr，不是KV量化前写点。
内网agent在本机追踪真实wrapper→binding→ACLNN→MXFP4 mode4→K/RMSNorm/RoPE写点，
复用已有原BF16出口或在正确私有producer增加可选BF16热环写入。不要在Q量化或旧V1入口盲插，也不用继续向用户索要文件。

具体合同与内网实施/验收见 [framework_integration/PROLOG_PRODUCER_TASK.md](framework_integration/PROLOG_PRODUCER_TASK.md)。
保留原Prolog计算、原C4 packed字节、原地Graph cache更新、原query输出与indexer C4；
不重复投影、不整cache复制、不强制NATIVE后宣称生产性能无回退。新变体在隔离目录构建，保留已验证产物与服务。

用户要求可支持的并发8、Graph、MTP、prefix、PD全部接入。外部Codex继续实现本项目自己的reader/请求映射/静态缓冲；
Prolog私有实现由内网agent就地完成，接口按允许披露的合同对接。逐项真实验证后才能声称生产支持；当前仍未完成。

54076f89的17:24:36错误是在配置解析通过后、runtime构造Prolog检查处；不是候选kernel执行错误。
旧配置问题见 [framework_integration/HOT_WINDOW_ACTIVATION_REPAIR.md](framework_integration/HOT_WINDOW_ACTIVATION_REPAIR.md)，
当前字段已修正，不再来回改JSON。主KV和indexer均保持原MXFP4设置。

不重启/修改现有服务、不升级CANN/vendor/wheel、不覆盖原成功库。新增真实激活重要性实验继续暂缓。
