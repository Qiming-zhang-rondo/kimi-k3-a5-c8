# 一次收集所需信息，不上传12MB全文

用户要求内网AI整理。请从本次已有Q4/Q32/Q64/Q128/Q256/Q512/Q1024的results.json提取，
不需要复制大文件、不需要重编来收集旧证据。脚本只用stdlib，不导入Torch/CANN、不发NPU调用。

先拉取最新sfa-cold-merge，在仓库根目录执行。替换为本轮真实输出根目录；
如果结果层级不同，自主定位这7份文件，不要求用户逐个转贴路径：

```bash
python3 direct_bridge/collect_precision_summary.py \
  --results /tmp/sfa-shape-first-v5/q*/results.json \
  --output evidence/shape_precision_before_v6.json
```

只提交这份小JSON。它会包括：
- 每个形状的实际Q/K/核数、库SHA、SDK、ABI/合并批量与input SHA。
- 所有failed_check_paths，区分究竟是非零、零query、数学参考、分区/写入还是图门禁失败。
- direct-native与fused各自非零/零query原strict比较指标，原值不变。
- 完整RNE证明的标量条件、actual/oracle逐位差异数及native tie/non-tie/ULP统计。
- 精度总判定、graph判定；若已有计时，附真实配对轮次与比值。
- traceback；原results和NPZ保留在机器上。

## 收集后继续同一任务，无需等用户再次确认

已补独立精确零query合并验收。旧0.001 strict及tie-only证明原记录保持，
candidate必须同时满足：实际BF16网格/有限值、max=0、coldSum=1920/hotSum=128、
TwoSum残差全零（独立F64分子精确），最终actual逐BF16 bit等于精确RNE oracle。
native差异只作诊断，不作为独立数学参考是否成立的条件。
非零query、数学参考、写入/分区、pipeline和graph门槛全部保持。

复用这轮已成功运行的ABI4设备库和原环境，用新Python进程/新输出目录先运行Q32。
**不重编kernel，不再重复Q4，不另跑单独accuracy后又重复整轮。**
下面是精简首批目录的示例；若本轮用全矩阵目录，只改为实际已有build-r16路径：

```bash
python3 direct_bridge/run_direct_ab.py \
  --payload-dir /tmp/sfa-direct-payload \
  --bridge-library /tmp/sfa-shape-first-v5/build-r16/libsfa_torch_direct.so \
  --device "$SFA_DIRECT_DEVICE" --queries 32 --keys 8192 --direct-aic-count 16 \
  --artifacts summary --timing graph --warmup 20 --iterations 100 --repeats 10 \
  --fault-marker /tmp/sfa-direct-c4-ab-v3/DEVICE_FAULT.json \
  --output /tmp/sfa-zero-exact-v6/q32
```

新门禁会现算全部值，不能引用旧CSV中的oracle=0来放行。
Q32通过后连续跑Q64、128、256、512、1024，保持同配置与各自新输出目录。
如果非零或其他真实门禁失败，保留失败，不能放宽门槛或标MEASURED。
设备故障立即停止，保留同一marker，不reset、不换目录绕过。

新运行也用同一收集器打包成evidence/shape_precision_after_v6.json，直接推送小JSON与summary小表。
无需推送大outputs.npz或12MB全文。此过程无需用户再次逐项搬文件。
