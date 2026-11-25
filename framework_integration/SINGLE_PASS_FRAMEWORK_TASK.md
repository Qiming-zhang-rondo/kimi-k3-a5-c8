# 当前优先：把已跑通的单遍算子接入真实模型精度测试

用户已明确先用框架，不再等待单算子冠军对比或继续摸高。7a2fad回传摘要：单遍94.66µs、installed两遍107.45µs、纯C4 80.56µs，报告精度PASS。它是prepared-read数据，不是模型/服务验收；先保留已有工作库，不再把额外微基准设成前置。

本轮目标：**真实模型确实运行 single_pass → 模型 smoke → 原精度任务 OFF/ON**。常规代码定位、依赖/版本适配、编译和执行由内网agent自主完成，用户不再搬文件。

## 一次完成

1. 找到实际GLM‑5.3权重、现有启动参数、当前MXLLM源码、已验证的 `libqsfa_recent128_direct.so`。记录模型D/Rope/local-H、CANN、库SHA和实际来源。先确认本原型匹配 **D512/R64/packed400/KV-head1**。若实际为NoPE0/272B，当前reader明确拒绝；不得把272B当400B或静默退回两遍称single-pass成功。提交实际shape与必要适配代码，保持真实模型路径。
2. 在新的隔离MXLLM源码checkout应用 `install_overlay.py`，保留原服务/旧源码。安装器核对的基线是3ba7355d；若当前源码不同，自主核对raw BF16写点和read hook再适配，不盲用SHA覆盖校验。旧overlay需按manifest恢复后更新或新建checkout，不能直接混用旧runtime包。
3. 从现有真实启动参数生成相同 engine.json，明确 `max_num_seqs=1`、eager、关闭prefix/PD/PCP/DCP/spec。保持真实权重、TP、原MXFP4量化和QLI/TopK配置。若原preprocessing是PrologV3，三臂**显式共同**使用 `--native-accuracy-preprocess`，在权重转换之前选NATIVE。默认不静默关闭Prolog，不用反量化C4当原BF16热环。
4. 顺序独立进程运行 `--arm off`、`--arm native`、`--arm single_pass`。`single_pass`的library-dir指向已测ABI1库；旧`fused`是ABI4热读归约路径，不要混淆。源码入口示例见下；所有路径从已有环境自主填写，不再询问用户。
5. 先用少量真实短/长prompt完成prefill、chunk边界和decode smoke，再接用户原有GLM精度任务。至少一个实际输入超过TopK2048，确认有混合冷热行；不能仅短prompt全热端跑通就报完成。
6. 每层/每rank报告 candidate_attention_calls、query_rows、mixed_rows、full_math_checked_rows、library SHA。Hlocal≤64时每次candidate只发起一次attention；H不足64补齐、H>64分组的实际次数明确记录。验证参考是**CPU完整FP64数学参考**，NPU validation_attention_calls=0；它读取真实选中C4页和原BF16热环，不是第二次attention，也不是采样检查。
7. 数学门禁沿用单遍standalone的global/worst-row≤.01且全finite；旧ABI4相对compact native的.001门槛不变。二者参考合同不同，不能把新门槛解释成放宽旧门禁。collector要求所有query行完成数学检查才记wiring通过。
8. smoke通过后运行用户原GLM精度任务，同一数据集/seed/采样配置重新跑OFF/ON，记录分数、样本数、失败项和native热窗对照。不复用历史90.4/91.4当同构建基线，不拿合成L2收益替代模型评分。测试中断不能报精度通过。
9. 提交现场实际适配及小型 `framework_integration/evidence/single_pass_model_summary.json`，包含三臂状态、真实库SHA/配置、逐层覆盖/最差数学误差、任务分数和明确未测项。原prompt、权重、大型日志留内网；不要上传凭据或完整engine配置中的敏感项。

## 入口（示例路径由本机实际路径替换）

```bash
python framework_integration/install_overlay.py --mxllm-root "$MXLLM_EXPERIMENT_ROOT" --check
python framework_integration/install_overlay.py --mxllm-root "$MXLLM_EXPERIMENT_ROOT" \
  --apply --output "$QSFA_MODEL_OUTPUT/install"

python framework_integration/run_model_probe.py \
  --engine-json "$QSFA_MODEL_OUTPUT/engine.json" \
  --prompts-jsonl "$QSFA_MODEL_OUTPUT/prompts.jsonl" \
  --mxllm-root "$MXLLM_EXPERIMENT_ROOT" \
  --library-dir "$QSFA_SINGLE_PASS_LIBRARY_DIR" \
  --arm single_pass --output "$QSFA_MODEL_OUTPUT/single_pass"

python framework_integration/collect_framework_evidence.py "$QSFA_MODEL_OUTPUT/single_pass"
```

原Prolog模型做共同NATIVE精度实验时，OFF/native/single_pass三条命令均加 `--native-accuracy-preprocess`，不能只改ON。精度入口有CPU完整参考、head补齐、metadata复制，**只用于正确性与任务精度，不用于TPOT**；不能因其慢于服务就误判算子性能，也不能把94.66µs写成模型延迟。

生产接入仍需保留PrologV3真实BF16输出/写环、图和静态缓冲、多请求、MTP与PD生命周期。不要默默关闭这些功能后宣称生产无性能代价。出现设备fault立即停止后续设备工作，保留首个错误，不重置业务卡。

精度通过后准备算子团队交接摘要，先提交材料；接收对象未明确前不自行向其他人发消息。冠军A/B为可选任务，见single_pass目录脚本，不再是本轮模型精度前置。
