# 收齐512/2048原始证据，复核37us来源与128热命中

33e876b只提交了手工摘要，报告TopK512/D512 W0=24.542us、W128=25.611us、配对中位比1.0339；L2 .1150869→.1143425，相对下降0.647%。热命中4/8/9/11，平均只有8/512=1.56%；不要称为“128热token的完整开销”。两中位数之比1.04356与配对比1.0339可正常不同，勿为此修改计时。

最新afb2392又报告TopK2048/Q4/K8192的DSL W0=36.74us、W128=37.55us、配对比1.0224，L2 .0849→.0844（相对下降0.589%）。仍仅有摘要，无actual hot_counts/门禁/原始配对表。不要让用户重复运行已完成组；先提取该组与512组的真实结果。1.9%与2.2%不能跨实现排名；DSL约37us相对于旧V4约82us的差距是有价值的性能线索，不能仅以维度不同忽略，也不能未经同数学合同对齐就宣称2.19倍加速。

## 1. 不重跑，先自动导出原始结果

自主找到33e876b及afb2392实际运行目录中的results/results.json（默认在`/tmp/dsl-recent128/dsl-recent128-*/`）。分别用新采集脚本生成小型、合法JSON：

```bash
python experimental/dsl_recent128/collect_evidence.py /actual/run/results \
  --output experimental/dsl_recent128/evidence/topk512_uniform.json
# afb2392原始运行目录，按实际selection命名，不凭文件名猜uniform。
python experimental/dsl_recent128/collect_evidence.py /actual/k2048/run/results \
  --output experimental/dsl_recent128/evidence/topk2048_original.json
```

该脚本只读已存在文件，不导入Torch、不运行NPU。需要保存实际模块版本/SHA、matching-reference及最差行误差、零侧/零query控制、graph投毒检查、配对10轮event/wall/ratio、native命中报告。结果缺项如实标INCOMPLETE；若已有原始文件，就原样提取，不再人工重写数字或让用户复制JSON。

如果找不到原始结果，明确标当前仅摘要；保留原首组数字，不将它伪造为完整原始证据。用新独立目录重跑首组一次补证即可。

## 2. 同构建补缺失组，不重复已经完整通过的组

沿用上轮已成功的实际Python/net wheel/CANN和空闲卡，不更换SDK/vendor。若旧native不支持2048，按EXECUTION_TASK.md在隔离环境构建固定公开DSL源码，整个新矩阵都使用该同一构建。不要拿旧512与新2048差值推导形状缩放。

```bash
# 以下两组保持B1/Q4/K8192/H64/D512/W128和默认seed/sink。
python /actual/run/test_dsl_recent128.py --device npu:0 --topk 2048 \
  --selection uniform --timing graph --output /path/to/new/k2048-uniform
python /actual/run/test_dsl_recent128.py --device npu:0 --topk 2048 \
  --selection covered_hot --timing graph --output /path/to/new/k2048-covered
```

每组都有W0与W128，原TopK并集不变；covered_hot明确使每query的128个热token全部入选，仍总共2048。先检查afb2392实际是否已完成其中一组，完整通过的组直接用原始结果、不重跑。uniform的热命中以实际报告为准，不能当作128。资源允许再补512 covered_hot，但不要为此延迟提交前两组结果。

上面的`/actual/run/test_dsl_recent128.py`及同目录dsl_reference.py必须是已校验fad3c596 payload；也可直接使用本仓`experimental/dsl_recent128/test_dsl_recent128.py`，保留文件SHA。源代码若有必要修复，独立记录diff和新SHA，不静默修改已发布计时/参考。

每组完整通过数值和捕获控制后才测性能；原脚本已执行这些门禁。记录无同卡业务干扰，不能停止用户服务或重置卡。任何device fault立即停。

## 3. 复核37us与82us差距的实际来源

先从成功运行和现有源码只读收集，暂不改cache或模型：同NPU型号、实际设备/空闲状态、实际DSL与原生kernel导出/文件SHA、AIC/AIV核数、attention调用数、metadata长度和任务分片（是否沿KV分片/FlashDecode）、临时workspace、QK/PV维度、实际cold/hot有效KV数量、计时scope、sink合同。完整FP64 matching-reference必须覆盖全部2048有效token，防止短profile或长度构造误读成完整2048。native require命中也需原始报告。

V4昨日默认4个AIC按query分工；DSL是否利用更多核沿KV分片是待验证假设，不是既定根因。D512/576与cold320/400的差异影响成本，但不直接证明约2.19倍差距完全来自格式。先取得这些小型证据再决定迁移调度/流水，或做同数学合同的格式适配；不能把当前API直接换进生产框架。

## 4. 提交小型证据，避免大log

对每组运行collect_evidence.py，输出到 `experimental/dsl_recent128/evidence/`，提交合法JSON及6行以内汇总。返回每组W0/W128读耗时、配对比[min,max]、两臂原始BF16误差、matching误差、实际热命中数和native命中状态。

这些都是单算子prepared读、D512 DSL自己的同轮对照。不能与昨日D576/TopK2048/不同格式的82us直接排名；也不能以合成L2改善代替GLM评分。
