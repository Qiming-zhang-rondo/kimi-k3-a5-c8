# 当前动作：仅用已有NPZ做CPU定位

3d2011e6：direct两遍与已安装两遍完全一致，融合非零query通过；唯一失败是zero-query
最差行0.001010071，门槛0.001。先确认是否为BF16 halfway舍入差异，不能直接放宽阈值。

在已有结果的机器上从仓库根目录执行，不重编、不重跑NPU、不重新测速：

```bash
python3 direct_bridge/diagnose_merge_rounding.py \
  --npz /tmp/sfa-direct-c4-ab-v2/results/outputs.npz \
  --results /tmp/sfa-direct-c4-ab-v2/results/results.json \
  --output evidence/merge_rounding_diagnosis_3d2011e6.json
```

直接提交并推送这份小JSON到sfa-cold-merge，继续任务，不要求用户转贴。
只读取已有数值，不加载Torch/CANN；原CONTROL_FAILED、0.001门禁和原NPZ保留。
无需追加多轮验证或修改内核。后续依据oracle/tie结果只改必要一处。
