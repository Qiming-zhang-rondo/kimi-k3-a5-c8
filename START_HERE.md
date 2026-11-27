# 当前任务：立即修正producer原型，编译验证后再启热环

用户希望尽快拉起一次真实热环服务。最新5e7067f0只有未编译实现摘要，尚无设备结果。
先按 [framework_integration/PROLOG_REVIEW_5E7067F0.md](framework_integration/PROLOG_REVIEW_5E7067F0.md)
直接修正，不再等待确认，也不再向用户索要内部源文件。

两个首要错误：六参Cast把512当mask而非总元素数，改四参计数形式；token_ring_slots是请求bank，
地址必须bank*C+token_positions%C，不能ringSlot%C。公共CPU参考producer_reference.py及3项测试已交付，
可用于校验本地NPU输出预期，绝不放进Graph执行路径。

当前3输入latent-only/P1原型缺KR、position/epoch tags、token_positions/epochs、valid mask与多pool，
不能标成完整ABI1或接并发8生产。按原prolog_provider接口补齐，未支持的SplitM/arch22分支须host在launch前拒绝，
不要用no-op假装成功。保持原FP32量化输入不变，BF16 scratch非alias且MTE3读结束后才可复用。

执行顺序：修Cast/寻址与scratch依赖→隔离编译单token/回绕smoke→补齐完整ABI1与B8/padding/MTP/Graph动态重放→
原C4字节/scale逐字节一致与BF16 golden通过→再接生产reader/生命周期并验证真实热环调用。
B1 smoke只证明底层局部，不是完整服务；不得以关闭热环或禁用原生产特性冒充接入成功。

后续实现细节和公共接口见 [framework_integration/PROLOG_PRODUCER_TASK.md](framework_integration/PROLOG_PRODUCER_TASK.md)。
保留Prolog原地写cache、主C4布局、量化参数、indexer C4。不要chmod/sudo继续改工作vendor；使用隔离candidate工程/OPP作用域，
现有已写入的candidate源先留档，不擅自删除或覆盖工作产物。

内部源、diff、头/schema和二进制留内网；只回传部门允许披露的构建/数值/图/性能结论。
不要求剩余40个文件，不开展暂缓的真实激活分析，不重测已成功的SFA tile扫描。
