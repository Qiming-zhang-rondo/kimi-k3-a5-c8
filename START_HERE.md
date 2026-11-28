# 当前动作：修复7d0ee407的Graph stream缓存错配，继续同一批reader测试

最新证据：新批量库已完成编译，B1Q1 eager通过（L2=0.0022918427、faults=0）。Graph在进入attention内核前因预热/捕获stream不同而查不到plan。B1Q4/B8Q4尚未执行，完整Prolog/服务仍未验证。

本次修复仅Python：smoke显式使用已预热的capture stream；生产reader按builder/draft/ubatch预留逻辑执行lane，避免stream切换造成漏查，同时保持不同执行链的scratch隔离。两处where独立核对PyTorch2.10原始schema后改成预分配同dtype/同device Tensor分支；不是把内网的临时补丁原样照搬，也不是NPU独有的scalar限制。

直接按[framework_integration/SERVING_TASK.md](framework_integration/SERVING_TASK.md)最前面的7d0ee407修复步骤继续：更新Python，复用 /tmp/hot128_serving/reader-build 的已编译.so，不重编ASC、不加载模型、不改vendor；新输出目录一次跑原B1Q1/B1Q4/B8Q4+Graph+epoch矩阵。相关27项宿主检查通过，新A5 Graph结果待重跑。

任务文件的完整能力表仍有效：APC/prefix和P/D自动尾窗接线尚未完成，MTP draft层历史热窗需核实，不能将配置允许、writer通过或B1 eager通过当成全功能验证。先处理已知实现缺口，再做完整服务验收，保留MXFP4/并发8/Graph/MTP/APC/P-D目标配置。

内部实现和原始材料留内网，只回传部门允许的状态、首错和测试摘要；不批量外传源码/schema/二进制。