# 两轮已完成测试：一次收齐小证据，再继续写融合大chunk

主分支37bbc60回报Q32–1024已MEASURED，Q1024融合/installed约0.601。
本分支1c0376b回报Q4写融合约111.9µs、相对同轮读融合0.983；
随后8ac297e已回传Q32/128/512写融合，分别约0.978/0.993/0.996。
这些是上传摘要；原始大JSON仍留机器上，不需要推送12MB。

## 先纯CPU提取已有两轮，不重跑实验

拉取本分支最新收集器；collector不导入Torch/CANN，不运行NPU。
在同一机器仓库根目录执行。若实际路径不同，由现场代理定位已完成结果，不要求用户搬路径：

```bash
python3 direct_bridge/collect_precision_summary.py \
  --results /tmp/sfa-zero-exact-v6/q*/results.json \
            /tmp/sfa-write-prepare-q4/results.json \
            /tmp/sfa-write-prefill-v6/q*/results.json \
  --output evidence/completed_shape_and_write_runs.json
```

直接提交这份小JSON。此次收集器保留：
- 真实Q/K/核数、ABI/BF16行数、库SHA、SDK和输入SHA。
- 各arm的writer/partition/reader/pipeline子项状态（缺失字段和null不会伪装成PASS）。
- exact零Q证明的TwoSum残差和actual/oracle逐位差异计数，保留旧strict/legacy失败记录。
- graph捕获与新写槽字节检查的各字段状态及mismatch_count。
- 每轮AB/BA event值与比值及完整流水线摘要。
不再提交仅STATUS文字或手工只抄几个PASS的摘要。原文件保持在本地。

## 只补尚未完成的Q1024，不重复Q4/32/128/512

8ac297e已经报告Q32/128/512完成，不要按旧任务重复这三组。
先在本地已完成结果中检查Q1024是否也已MEASURED；若已完成仅收集，不重跑。
若未完成，等同一NPU空闲后只补Q1024这一组，K8192、16 AIC、rows16。
复用已成功ABI4库，不重编AscendC，不换SDK/vendor，不同时打开packed候选。

```bash
python3 direct_bridge/run_direct_ab.py \
  --payload-dir /tmp/sfa-direct-payload \
  --bridge-library /tmp/sfa-shape-first-v5/build-r16/libsfa_torch_direct.so \
  --device "$SFA_DIRECT_DEVICE" --queries 1024 --keys 8192 --direct-aic-count 16 \
  --cold-store-candidate --timing graph --stages --artifacts summary \
  --warmup 20 --iterations 100 --repeats 10 \
  --fault-marker /tmp/sfa-direct-c4-ab-v3/DEVICE_FAULT.json \
  --output /tmp/sfa-write-prefill-v6/q1024
```

若原实验使用其他目录或Q1024输出已存在，沿用真实库和结果路径，不能覆盖/重复已完成数据。
新case仍须全部精度、缓存字节、partition、写槽图重放及最终图输出门禁通过。
设备故障立即停止、不reset、不绕过marker。

写融合收益以direct_fused_cold_store_vs_direct_fused同轮配对判定，
Q512约0.4%的摘要差异还需要原有轮次范围；不能在缺少原轮次时宣称稳定胜出。
同时保留candidate/installed与candidate/W0；这些都是单算子结果，不等于生产TPOT。
每组已有结果用同一收集器直接推小JSON，不推12MB全文，不让用户逐项转贴。
