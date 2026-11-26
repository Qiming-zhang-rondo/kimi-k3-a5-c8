# FP4 主缓存 + FP8 recent128：独立单算子实验

这是用户新要求的 DSL 对照实验，不替换已经验证的 `qsfa_recent128`，也不修改模型、CANN 或 vendor。

## 测什么

- 同一份原始 BF16 Q/KV、同一 TopK、同一 scale/sink，比较 `fp4_w0` 与 `fp4_fp8_w128`。
- Q `[4,64,512]`，KV 长度8192，batch1；默认 TopK512，默认最近128窗口。可用 `--topk 2048`。
- 默认 `--selection uniform` 是合成、因果、无重复的随机索引，窗口内实际命中通常少于128，summary记录每行热token数。`--selection covered_hot` 强制原TopK包含全部最近128，属于覆盖测试，不是模型TopK。
- 冷端为 FP4 E2M1/group16/BF16 scale，320B/token；热端为 FP8 E4M3FN/group32/BF16 scale，544B/token；完整512维=448 latent+64 RoPE，两者均被量化。**这不是现有MXFP4/group32/E8M0缓存。**
- 热环真实分配 `ceil((W+Q-1)/page)*page` 个slot，Q4/W128/page128时是256；按绝对token位置取模，host校验tags，不把窗口外token误读成新token。每query只把原TopK内的热命中转到热端，不增加/重复/删除token。
- FP64完整数学参考分别检查原始BF16误差与当前量化匹配误差；匹配误差全局和最差query/head均<=.01、全finite才通过。独立检查零query与冷端有效数0的纯热路径，W0本身检查热端有效数0。
- 默认sink=0，在该接口里意味着分母加1，全部参考统一此合同。不能把它写成“无sink”或直接复用旧无sink结果。

性能只计 **prepared attention单次调用**，不含缓存写入、partition、metadata、量化。图内一调用、图外重复replay，投毒输出检查捕获有效；10轮AB/BA保存每轮event/wall、配对比值，计时后再次校验输出。CPU准备时间不冒充设备性能。

## 来源和依赖

- 9.3.0 ops接口参考：`hicann/ops-transformer@477e8d2cf7673cbe2a6ab1b8f3cd6f635ee0e637` 的 `experimental/attention/mixed_quant_sparse_flash_mla_dsl/`。
- 已找到公开真实DSL源码：`hicann/cannbot-dsl@c36d3a2e8a60710cf49d1d09b0b126241dcb6530`，`samples/attention/mixed_quant_sparse_flash_mla/`。
- 新源码native包名 `cannbot-arena-net`，Python包 `ops`，运行时DSL>=0.8.0。该commit的K1/K2已动态化，配套metadata返回int32[1024]。脚本直接调用配套metadata，不构造或猜测它的ABI。
- 较早9.3 ops README提到旧 `cannbot-arena-net-ops` 和静态容量128/512。默认512兼容该容量，但二者不能混装；2048在旧包可能缺native profile。不要把新版公开实现的动态能力称为旧9.3发行包已经覆盖。
- 脚本实际import两个模块、记录文件/SHA、版本/签名和native报告路径；设置`CANNBOTDSL_NATIVE_BINARY_MODE=require`，未命中不能静默JIT。`--probe-only`只证明导入，不能证明NPU执行。
- `cannbot-dsl`运行时本身不包含本算子。缺少`ops.mixed_quant_sparse_flash_mla`时执行同目录 [EXECUTION_TASK.md](EXECUTION_TASK.md)，一次完成隔离构建/测试，不让用户逐项搬运报错。

## 本地源码用法

先确认空闲卡和匹配环境；不source旧`/tmp/cloudops` vendor。

```bash
python experimental/dsl_recent128/test_dsl_recent128.py \
  --probe-only --output /tmp/dsl-recent128-probe
python experimental/dsl_recent128/test_dsl_recent128.py \
  --device npu:0 --topk 512 --window 128 --timing graph \
  --output /tmp/dsl-recent128-k512
```

如果net wheel隔离装在 `/path/to/run/pylib`，加 `--dsl-root /path/to/run/pylib`；目录必须含`ops/`，实际导入文件须位于该目录。运行时0.8.0也需来自同一隔离环境，不能只换ops而沿用旧0.7运行时。

新公开native包准备好后用独立输出测 `--topk 2048`；再用 `--selection covered_hot` 做128热命中覆盖。不同TopK/selection分开报告。该DSL D512与昨日D576不同，**不能拿其延迟直接与82.08µs排名**。

产物：`results.json`、`summary.txt`、`native_binary_report.json`（运行时支持时）；launcher额外保存`launcher.json`、`run.log`。设备fault立即停，不重试、重置卡或停止业务。未命中native或缺依赖明确标阻断，不标MEASURED。

本地宿主检查只验证编码、参考、ring/partition和运行器合同；实际NPU精度与性能待内网机器执行。
