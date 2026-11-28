# 当前任务：完成真实 Prolog recent128 调用，继续启动服务

最新同步到 `fd7a60d6`：报告称新版 Prolog 已安装在178/209，最终导出函数的 ABI=1 已确认，TBE 首次调用编译及真实 producer/服务结果仍待验证。这个提交仅增加汇总报告，不能从“已安装”推导出写环、Graph、APC 或 P/D 已通过。

内网 agent 直接执行 [SERVING_TASK.md 当前入口](framework_integration/SERVING_TASK.md)。沿当前已经成功的私有 source、SDK、variant 和 wrapper，不重做 baseline、不重新 staging 旧源码、不重编已通过 reader，不再要求用户转发内部源码。

本轮只做一条连续链：本地核实实际 latent/KR/tag 写入及原参数+新增12参数转发 → 现有 producer 入口的 OFF/ON 和 B1/B8/Graph 对照 → 继续原服务启动。已有安装入口不用另搭第二套 bridge；首次 TBE 动态编译也必须实际通过。保留并发8、Graph、MTP、APC/P-D原配置，自动 BF16 尾窗回填仍是已知接线项，不能静默关闭或回退C4。

公共 helper 已修正 TEventID，补齐 VEC3510 guard 与函数 ABI 属性示例。12项本地检查通过（4 SDK工具、6调用审计、2 helper类型/行为），均非目标 SDK/A5 执行证明。

- [inline Writer 接线说明](experimental/prolog_ring_smoke/PROLOG_INLINE_INTEGRATION.md)
- [SDK 兼容性和真实符号核验](experimental/prolog_inplace_build/SDK_COMPATIBILITY.md)
- [本地实际调用审计工具](tools/audit_prolog_writer_calls.py)：只读内网源码，排除注释/字符串，输出留内网；不是运行证明。

已通过的11项独立批量 reader 对照保持有效；本轮不重跑它们，不再启动新的单算子性能扫描或真实激活采集。用户当前目标是拉起自研 MXFP4 主缓存+BF16 recent128 的框架服务，DSL 缓存迁移暂不实施。
