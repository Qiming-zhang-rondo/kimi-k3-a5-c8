# 当前任务：已找到Prolog KV写点，直接在内网实现热环producer

最新fac32bf36已从真实V3/mode4链定位到RmsNormAndQuantizeCkv中QuantPerTileMxFp4前的FP32 inputLocal。
状态仅IDENTIFIED，尚未实现、未编译、未在NPU执行。用户已授权推进生产接入，内网agent不要再停在等待确认。

外部已交付公共接口适配器framework_integration/sfa_recent128/prolog_provider.py及5项宿主测试。
新约定入口cloud_ops_turbo.cloud_mla_prolog_v3_recent128与函数ABI标记1，原Prolog9位置参数/原kwargs/原5输出保留；
ON增加固定地址BF16 ring与device slot/position/epoch/valid元数据，OFF透明直通。接口不存在就明确报错，不回退。
这是外部框架侧接口代码，尚未挂入生产runtime；不宣称部门内核、B8或Graph已经完成。

内网agent现在按 [framework_integration/PROLOG_PRODUCER_TASK.md](framework_integration/PROLOG_PRODUCER_TASK.md)
直接实施真实V3的wrapper/绑定/新variant/tiling/kernel，内部源码保留内网。先原FP32源不变、侧路非alias BF16 RNE写环，
用正确UB/MTE所有权同步；KR64另取post-RoPE。保持原C4 packed/scale字节、query输出、Prolog原地Graph更新与indexer C4不变。
随后隔离构建，做OFF/ON逐字节控制、BF16参考、B8Q4/padding/重排/MTP和动态Graph毒化重放，再同轮测producer成本。
禁止只给旧函数贴ABI标记；禁止删除guard放行不存在的BF16来源；不重复投影、不整cache复制、不强制NATIVE冒充生产接入。

不再收集或外传剩余40份源、私有头/schema/实现diff/二进制，之前的批量上传要求已撤销。
只回传部门允许披露的实现/构建/数值/图重放/性能结果，不允许外发的内容不回传，不要求用户做源码中转。

producer完成后继续多请求reader寻址、静态Graph缓冲、MTP事务、prefix尾窗快照、PD尾窗传输与完成协议。
并发8/Graph/MTP/prefix/PD是生产目标，当前未完成；主C4 cache布局和QLI路径保持不变。
现有accuracy_eager adapter仍是受保护诊断原型，不删除其限制后假称生产全特性支持。
不重启工作服务、不升级CANN/vendor/wheel、不覆盖成功产物。真实激活重要性实验继续暂缓。
