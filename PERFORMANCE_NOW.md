# 首轮融合性能已完成

7ca3b03的 `evidence/phase_c_measured_results.json` 为实际A5四臂同轮graph测量，
10轮AB/BA；已安装两遍125.57µs，direct两遍136.90µs，融合134.36µs。
融合/已安装两遍配对比值1.0714 [1.0695,1.0728]，尚未胜过现有实现。
direct原版与已安装原版的差异包括设备调度/kernel，不等于纯主机桥接开销。

本轮精度与graph投毒重放验收通过，零query按现算独立FP64→BF16 RNE证明分支验收，
原strict比较失败记录仍保存；非零query原0.001门槛通过。
这是固定Q4/all128-hot-hit单层实验，不是服务集成或端到端性能验收。

下一步按 [OPTIMIZE_NEXT.md](OPTIMIZE_NEXT.md) 测新增优化候选。
