# recent128 框架精度接入候选

当前优先执行 [SINGLE_PASS_FRAMEWORK_TASK.md](SINGLE_PASS_FRAMEWORK_TASK.md)。新增 `--arm single_pass` 接入ABI1整TopK单遍reader；旧 `--arm fused` 保留为ABI4热读＋冷态归约对照。单遍框架源码已接线，实际模型仍待内网验证。

单遍精度模式使用所有行的独立FP64完整softmax数学参考（.01，与standalone一致），不调用NPU冷读/热读来生成候选或参考；旧ABI4的.001 compact-native门槛保持。物理C4 pool完整保留、逻辑页表按原映射传递；CPU参考只取TopK选中页，检查缺失BF16 tags和future索引。当前仅R64/packed400，NoPE0显式拒绝。该测试入口包含CPU参考和head补齐，不代表生产性能。

从单算子进入真实模型验证。策略保持 **C4 冷 KV + 原始 BF16 recent128**，模型原来的 TopK、Hadamard 权重和 MXFP4 量化参数均不改动。

这次交付是实际 MXLLM worker overlay、原生 BF16 写点、attention 替换、融合入口调用及模型 smoke。**尚未在实际模型上运行，不是生产性能交付。** 内网执行任务见 [FRAMEWORK_EXECUTION_TASK.md](FRAMEWORK_EXECUTION_TASK.md)，无需用户逐项搬运依赖和错误。

## 接线

1. 在 MXLLM `patch_sfa_v1_mxfp4._mxfp4_custom_kv_rmsnorm_rope` 的 RMSNorm/RoPE 后、MXFP4 前保存真实 BF16。原生量化与 C4 写入仍由原框架执行。
2. 每层维护独立、带 logical-position tags 的 BF16 环；容量按 `align_up(128 + max_query_chunk - 1, 128)`，支持分块 prefill 和串行 decode。不会用 C4 反量化值充当 BF16。
3. 沿用原 TopK，只按每个 query 的位置拆为冷/热两组。热命中数可能为 0..128，原生 BF16 对照按真实数量紧凑调用，避开旧包的 `-1` 尾部填充问题。
4. 热命中恰好 128 的组调用已经测过的 ABI4 BF16 hot-read + cold-state merge。小于 128 的组走精确原生对照并记录原因。原生 C4 cold attention 仍是独立一次调用。
5. 每个实际融合输出均与当次 compact 原生两遍结果进行完整比较，沿用 global/worst-row relative L2 ≤ 0.001 的门禁。最后把候选结果交回模型的 `_v_up_proj` / `_finalize_o_proj`。

TP 后 head 数可以小于 64；精度模式以独立 head 的零填充适配原 H64 入口，大于 64 则分组。NoPE 模型的冷端仍是其真实 272B/R0 布局；仅 BF16 hot reader 的 RoPE64 输入为零，保留模型原 softmax scale。这些适配均需现场精度确认，并计入证据；**不能拿这条带检查/补齐的路径宣称原微基准的性能。**

## 初始范围

- `execution="accuracy_eager"`，单机、单活动请求，Q≤1024。
- 原生 preprocessing，PD / PCP / DCP / MTP / prefix caching 关闭。多请求、spec 回滚、PD 热环传输和图捕获尚未接入。
- 默认 `window=0`，不分配热环、不加载候选库、不增设备操作。
- 默认 `preprocess="preserve"`。遇到 PROLOG_V3 不会静默降级：现有接口没有原始 BF16 KV 输出。
- 若现场确实是 PROLOG_V3，可对 **OFF 和 ON 两臂同时** 显式选择 `preprocess="native_accuracy"`，在权重转换之前选择 NATIVE，仅用于先验证精度。该成绩不代表原 PrologV3 服务路径，也不能用于 TPOT 验收。

## 安装与执行

默认检查只读，应用时保存原始源码和恢复清单。目标必须是隔离的 MXLLM 源码目录。

```bash
python framework_integration/install_overlay.py --mxllm-root /path/to/isolated-mxllm --check
python framework_integration/install_overlay.py --mxllm-root /path/to/isolated-mxllm \
  --apply --output /tmp/recent128-framework/install
```

已核对基线：MXLLM `3ba7355d79a02eb0e292fae92cfacf583b117816`，其 vllm-ascend pin 为 `760d7ecbb962135fff738726825cffacc74f5d2f`。若实际源码不同，安装器拒绝盲改；内网 agent 应先核对真实接口，再传其精确 SHA 或修改相应 adapter，不能自动把报错的 SHA 当成认可。

从当前真实模型启动参数整理 `engine.json`，保留 checkpoint/TP/权重量化/attention 配置，明确 `enforce_eager=true`、`max_num_seqs=1`、`enable_prefix_caching=false`、`enable_chunked_prefill=true`、`max_num_batched_tokens≤512`。`attention_config.mla_kv_quant_mode` 应为 `mxfp4`。真实输入放在 JSONL 的 `id,prompt` 字段中。

```bash
python framework_integration/run_model_probe.py --engine-json /tmp/recent128-framework/engine.json \
  --prompts-jsonl /tmp/recent128-framework/prompts.jsonl --mxllm-root /path/to/isolated-mxllm \
  --library-dir /path/to/verified-direct-build --arm fused --output /tmp/recent128-framework/fused
python framework_integration/collect_framework_evidence.py /tmp/recent128-framework/fused
```

OFF / native 两臂分别使用 `--arm off` / `--arm native`，独立进程、不同输出目录。需要 NATIVE 精度实验时三臂都加同一个 `--native-accuracy-preprocess`，不要只改 ON。探针仅证明真实模型接线，需要继续运行用户原有 GLM 精度任务。

证据区分 `EXERCISED`、融合覆盖率和任务精度；生成完成或 fallback 全通过都不能替代融合已执行、模型精度已测。仅发送摘要和必要源码到 GitHub，原始内部 prompt、权重和大型日志留在内网。
