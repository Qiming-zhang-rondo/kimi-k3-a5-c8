# 舍入诊断已完成，继续性能

83a4001已提交 `evidence/merge_rounding_diagnosis_3d2011e6.json`。
本轮零query融合结果131072个值与独立FP64→BF16 RNE参考逐值完全一致；
旧两遍4096个不同值均为精确半值处的相邻1 ULP差异。非零query原0.001门禁通过。

不要重跑CPU诊断或重编设备库。下一步执行 [PERFORMANCE_NOW.md](PERFORMANCE_NOW.md)。
原始3d2011e6的CONTROL_FAILED、NPZ及严格比较数值保持原样；新运行会从当次实际冷/热状态
重新验证严格的零query RNE条件，不以旧JSON代替验收，也不全局放宽0.001门槛。
