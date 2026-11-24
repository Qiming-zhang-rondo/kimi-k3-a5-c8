# 8bd2807/09a06c4：设备执行失败，不能标作LAUNCH_PASS/DEVICE_VALIDATED

本提交增加了实际launcher，报告stream同步返回507015。这是EXECUTION_ERROR，
不是“同步全部工作正常”；此提交未包含完整设备故障原始日志。精确硬件根因尚未确认。
停止后续设备测试，保留fault marker/日志；继续静态合同检查和构建，不自行复位或反复试参数。

源码已确认的问题：

- launcher两个版本都在同步失败后继续D2H读取输出，并在最后aclrtResetDevice；
  外部修正为立刻非零退出并写DEVICE_FAULT.json，同时移除reset调用。
- 当前传入2048个全有效索引，而任务第一步是compact128热读；这不是同一个zero-query对照。
- returnSoftmaxLse设0，却读取max/sum；验证只打印期望值、检查NaN，未逐head断言有效计数。
  全零buffer即使没有有效计算也可能满足现有“非NaN”门禁，必须补独立参考和实际断言。
- 输入宣称64个Q head，但nNumOfQInOneGroup赋1；原vector源码以此字段设置gSize，
  与当前Q形状合同不符。KV_BYTES还按NUM_HEADS乘64分配，而本任务KV-head为1。
- usedCoreNum=1、launch blockNum=8、workspace固定4MiB及其字段填法尚未给出匹配源码推导。
  这些不能靠注释“typical”或猜测UB参数认定正确，需要按实际内核/SDK逐项推导。
  09a06c4已把blockNum改1并改了tiling数组字段公式，但报告仍507015；这些更改已保留。
- 更关键的是direct启动workspace协议：公开9.2.0的3510实现中GetUserWorkspace读取系统workspace
  地址而非直接返回传入参数。当前入口没有证明该地址由direct-launch建立。需要核对当前9.2.1
  的实际头及编译启动协议；正确实现系统workspace标注/初始化，或明确重建用户scratch ABI并
  审查所有系统区依赖。不能仅增加buffer或盲贴一个标注就宣称修复。
- 新CMake覆盖了原compile-only target并再次写死SDK/vendor路径；旧$BUILD目录也被提交。
  后续修复保留成功构建记录，分开可重现baseline/candidate目标，不覆盖正在工作的环境。

以上是已发现的具体合同问题，不代表已证明507015只有某一个原因。
下一步内网代理先补原始fault日志、实际选中的binary/template、源文件哈希；依据本次
编译POD和真正读取字段实现host writer/workspace/blockDim，保存可审查推导及host预检。
修复完整且确认设备由有权限人员恢复可用后，再按POST_COMPILE_EXECUTION_TASK.md运行。
未通过原版数值前不继续融合性能。不要更名“LAUNCH_PASS”或凭手写tiling反复上卡校准。
