# 内网 agent 执行：接入框架并完成真实模型精度验证

用户已决定停止单算子摸高，转入框架。请直接完成以下工作并提交小型证据，普通依赖查找、构建和接口差异由你自主解决，不要逐项让用户中转。不要新增优化扫描、改量化参数或重跑历史微基准。

## 1. 固定隔离工作目录与实际来源

- 从本分支建立独立实验目录；保留原服务与已验证算子库，禁止停止其他业务、重置设备、切换 CANN 或覆盖 vendor 包。
- 自动找到当前用户 GLM‑5.3 checkpoint、真实 MXLLM / vLLM / vllm-ascend 源码和实际启动参数；从现有本机配置读取，摘要不得含 token/凭据或完整环境变量。
- 记录各 commit、dirty 状态、SFA 文件 SHA、CANN/runtime/operator 包版本、模型的 D/Rope/local-H、TP、当前 preprocessing 与 operator library SHA。已验证库 SHA 为 `1dca4c2463e398a01bf37e60dcfb0b7573b533b3c456db8475f95ebae5d4237a`；不同库须说明来源，不得混用 9.1/9.2 旧包重现崩卡。
- 本候选对照的是 MXLLM 3ba7355d / vllm-ascend 760d7ec。若实际代码更新，读取真实源码，移植相同 BF16 写点和读入口；有新增适配直接提交代码。不能只运行 find_spec，也不能用另一算子 API 猜当前签名。

## 2. 在隔离源码安装并准备精度路径

```bash
python -m unittest discover -s framework_integration/tests -v
python framework_integration/install_overlay.py --mxllm-root "$MXLLM_EXPERIMENT_ROOT" --check
python framework_integration/install_overlay.py --mxllm-root "$MXLLM_EXPERIMENT_ROOT" \
  --apply --output "$EXPERIMENT_OUTPUT/install"
```

变量使用本机实际路径。安装器只修改明确的实验源码，先备份、再写 manifest。源码 SHA 不匹配时你自行审阅真实文件后修改 adapter / 传精确 SHA；不能自动将 actual SHA 复制到 override 参数后绕过核对。恢复方法保存在 install manifest。

首轮目标是**真实权重下验证 recent128 精度以及融合输出与原生对照的一致性**，不作性能成绩。使用单机单活动请求、无 PD/CP、关闭 MTP/spec 和 prefix caching、显式 eager、分块 prefill≤512；保持真实 TP 和所有权重量化配方。不要重复加 Hadamard；不要启用 scale_alg=2/max7（已有真实模型掉分证据）。

默认保留 preprocessing。如果实际已是 NATIVE，直接用。若为 PROLOG_V3，其当前 ABI 不外露 BF16 KV，**在三臂均显式使用 `--native-accuracy-preprocess`**，选择点位于权重转换之前。记录这是 NATIVE 精度实验，绝不将其耗时作为 Prolog 服务性能。不要把 packed C4 转 BF16 冒充热环。为最终生产 PrologV3 接入保留明确缺口：需要真实 BF16 输出/写环接口，待精度证据一起交算子团队。

## 3. 顺序完成 OFF / native recent128 / fused recent128

根据当前启动参数生成同一份 engine.json 和真实 prompts.jsonl，三臂顺序运行独立进程。代码入口见 README；所有其他参数相同，保持原 TopK 集合。

- 首先使用真实短/长 prompt，覆盖初始 prefill、跨 chunk、decode 和多次 ring wrap。至少一条超过 2048 token，不能只拿全热端短文本代表 mixed 功能。
- ON 每层必须有原始 BF16 写点证据、真实 mixed 行计数，不能因不支持而 window0 回退。
- 真实 count0..127 用精确 compact native，不能 `-1` 填容量2048调用旧 BF16 reader。恰好128的组必须实际调用 ABI4 融合库。统计每层/每rank融合行数、总行数、调用次数和 fallback 原因。
- 每次融合都和同输入的 compact native 两遍对照；global/worst-row relative L2 均≤0.001。保留未过项，不放宽阈值、不隐藏失败，不拿模型生成成功代替算子通过。
- NoPE0 仅 hot-reader 使用全零 RoPE64，cold packed格式仍按实际模型保持；local H不足64是精度适配补齐，不能以其耗时宣称性能达标。
- 检查原始写/读 slots、首次 chunk、新请求 reset；若出现真实设备故障立刻停止后续设备工作，记录首个 traceback/诊断，不继续测性能。
- 若模型 warmup 使用虚拟 metadata，区分 profile 与真实 forward，修复适配时不得把真实请求静默跳过。若没有任何融合行，报 `NOT_EXERCISED` 并自主准备能覆盖 count128 的真实输入，不得发布融合 PASS。

## 4. 运行用户原有精度任务

接线 smoke 通过后，使用用户当前 GLM 精度评估命令、相同数据集/版本/seed/采样与请求参数，跑 OFF 和 ON，并保留 native recent128 对照。用户之前报告的 90.4/91.4 是历史模型路径分数，不能当本轮同构建基线。

- 输出本轮实际任务分数、样本数量、计分器版本、失败条目数量，native 与 fused 差异；测试中断不能算完成。
- 若 OFF/ON 有掉分，先确认同配置和相同输入、原 TopK/原量化没改，再定位首个分歧；不得用合成 attention L2 改善证明任务精度改善。
- 额外报告首 token / teacher-forced logprob 差异及低 margin 的 token 翻转（可复用原评估工具）；不要求逐字生成一致来掩盖合法的精度变化。
- 完成后收集每层 evidence。收集器只标 wiring，模型精度必须另外有真实评估结果，不要编辑其 `NOT_MEASURED` 常量冒充完成。

## 5. 返回及收口

提交 `framework_integration/evidence/framework_accuracy_summary.json`，包含：实际来源/配置摘要、三臂完成状态、各 rank/layer 覆盖、原生一致性最差值、任务分数和命令引用、尚未适配的生产接口。不上传权重、内部原始 prompt、12MB输出或凭据；必要的最小失败数据脱敏保存。提交任何现场修改的真实代码。

停止在**精度验证完成并整理交接材料**。本版为精度模式，有CPU计数、对照、head适配，不能声称无性能回退，不能把 eager 探针与原 graph/PrologV3 TPOT 相比。生产阶段需要无 CPU 读回、静态分区和多请求生命周期，以及 PrologV3 真 BF16 输出、PD/MTP支持的匹配 ABI。

精度通过后准备给算子团队的简短报告：模型/算子版本、真实任务 OFF/ON 分数、融合一致性和覆盖、单算子已验证性能、生产接口缺口。先将材料提交仓库供用户确认接收对象；不要自行向不明确的联系人发消息。
