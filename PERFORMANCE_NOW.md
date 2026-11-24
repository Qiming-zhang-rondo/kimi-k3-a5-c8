# 今天直接完成融合性能测量

从本分支拉取最新Python即可。**不要重编ABI2设备库，不改内核，不另跑一轮accuracy：
下面一次graph运行已内置全部精度门禁，通过后自动计时。**

复用3d2011e6运行成功的同一容器、环境、hotkv与 `libsfa_torch_direct.so`。
不要source旧vendor、切换SDK或更改baseline量化参数。若现场原路径不同，只改实际已有路径。

```bash
set -e
SFA_DIRECT_ROOT=/tmp/sfa-direct-c4-ab-v2
# 沿用之前确认空闲的设备，不默认选择0，不停止其他服务。
: "${SFA_DIRECT_DEVICE:?设置之前已确认空闲的npu设备，例如npu:0}"
python3 direct_bridge/run_direct_ab.py \
  --payload-dir "$SFA_DIRECT_ROOT/hotkv" \
  --bridge-library "$SFA_DIRECT_ROOT/build/libsfa_torch_direct.so" \
  --device "$SFA_DIRECT_DEVICE" --timing graph \
  --warmup 20 --iterations 100 --repeats 10 \
  --fault-marker "$SFA_DIRECT_ROOT/DEVICE_FAULT.json" \
  --output "$SFA_DIRECT_ROOT/performance-rne"
```

已存在输出目录不覆盖，另取新名字；故障marker仍使用原路径，不能绕过已有设备故障。

零query旧strict比较仍按0.001计算并记录。仅当当次真实冷/热统计均为max=0、sum=1920/128、
输入输出均为有限BF16网格值、融合输出逐值等于独立FP64 RNE参考，且与旧参考不同的每个值
都证明是精确BF16半值位置的相邻1 ULP，才可按 `exact_zero_query_BF16_RNE` 分支验收。
非零query0.001、数学参考0.01、分区/写入、零Q计数和graph投毒重放门禁保留。

通过后让程序完成四臂与五组AB/BA配对。**先交延迟结果，不追加优化扫描或profiler。**
提交本次 `results.json`、终端日志及一张小表到分支：

| 配置 | pipeline event µs | wall µs | 同轮配对W0比值 | 融合/已安装两遍 | 融合/direct两遍 |
| --- | --- | --- | --- | --- | --- |

报告每轮比值的中位数及min/max，不能称min/max为置信区间。
附本次代码commit、复用库路径/哈希与全部门禁状态；不粘贴整份NPZ。
不引用87/126µs历史常量，不仅用direct两遍充当最终基线。

若仍有普通Python/API错误，现场代理按traceback自行修好、继续这一次测量并提交实际修改；
若任何新精度条件失败，保留原数值并停在门禁，不能强行标MEASURED。
若设备故障，立即停止后续设备调用，保留marker，不reset。
