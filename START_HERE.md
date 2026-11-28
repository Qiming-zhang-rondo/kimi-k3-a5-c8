# 当前任务：验证Native writer布尔mask修复，继续同一curl

已审阅3c6f55b：真正修复是 `valid = tl.load(Valid + token) != 0`，现已合入主runtime源。报告5f82f9c将报错解释为标量mask形状不对不准确；Triton支持标量布尔mask广播，boolean scalar type说的是mask元素dtype。只加vector广播不能解决byte/int8类型，where与标量tags也必须是bool。

新增严格host语义回归直接执行实际kernel AST，Valid按GM byte读取，检查所有where/load/store布尔mask、广播、有效地址、原BF16字节与未写区域；恢复旧表达式会失败。9项writer宿主检查通过，仍非真实Triton编译/A5证据。

内网先在目标容器重跑同一小型writer eager/Graph：

```bash
python framework_integration/run_native_writer_smoke.py --device npu:0 --graph --output "$NEW_RUN/native-writer-bool-fix"
```

fixture补充valid=True但bank越界、position负数或epoch负数，验证全mask正确。这个阶段不用加载模型或重编V4/CANN库。确认后按 [NATIVE_DSACP_TASK](framework_integration/NATIVE_DSACP_TASK.md) 同步整个managed runtime（保留手工旧文件备份），新worker再重复原curl，避免仅修改证据文件或已加载旧kernel对象。

保留现有Native/DSACP、Graph、MTP、APC/P-D配置，原MXFP4量化和V4 reader不变。当前修复只是writer编译类型问题；真实TP8交换/路由、P/D/APC尾窗和GPQA仍需按原任务验收，不能把smoke通过或服务启动当作整链正确。
