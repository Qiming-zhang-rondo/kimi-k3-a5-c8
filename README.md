# Kimi K3 A5 C8：DCP / DSpark 测试版

本分支用于模型责任人验收，不是已完成 A5 验证的稳定版本。
稳定分支 [main](https://github.com/Qiming-zhang-rondo/kimi-k3-a5-c8/tree/main) 及其安装入口未改动。

## 本次范围

- 主模型 MLA 使用 FP8 E4M3 Group128 Cache：512B latent + 128B BF16 raw64 + 16B FP32 scales = 656B/token（不含页 padding）。不是 Group32/E8M0 MXFP8 Cache。
- DSpark 草稿模型保持原权重和 Cache 路径，不继承主模型 C8 开关。
- 主模型多 token 因果验证：按 query 计算可见 KV 前缀，使用设备侧已修正的序列长度。
- DCP：本地分片 Cache、Q heads all-gather、QSFA max/sum 转 LSE、复用原生 AttentionUpdate 合并；处理空分片和 padding。
- 普通与 MLAPO writer 共用上述 reader；接入 eager 与 `FULL_DECODE_ONLY` 的均匀多 token 验证批次。
- 不修改候选生成、接受/拒绝采样或 KDA 状态恢复算法；这些组合仍需模型实测。

## 已验证 / 尚未验证

| 项目 | 状态 |
| --- | --- |
| 本地相关 CPU 回归 | 589 passed，22 subtests passed；不是589项新增测试 |
| 改动文件 Ruff / diff 检查 | 通过 |
| 基础版本安装、旧版升级、重复安装 | 本地临时源码夹具验证通过 |
| A5 新增原生用例 | 已提供6项，尚未执行 |
| 真实多卡 HCCL、DSpark 模型、两者组合 | 待模型责任人验收 |
| 新组合精度、吞吐、TTFT、TPOT、图内存 | 待测，未声明性能收益 |

仓库级 `format.sh ci` 因本机缺少 pre-commit 未完成；未用本地 CPU mock 代替 NPU 验证。

## 安装：只打补丁，不自动测试

在 A5 容器内使用原模型 Python 环境执行。安装器自动查找 editable 安装的 VA 0.29 源码。
只接受以下两种受影响文件状态：

1. 基线 `b3d3bb6805ffaf8fcc20086d5e59dbf2352083b4`，使用完整补丁。
2. 本仓库 `k3-c8-va029-complete-20261028.patch` 的完整结果（含公共开关），使用增量补丁。

```bash
curl -fsSL --connect-timeout 15 --max-time 120 https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/test/c8-dcp-dspark-20261029/install.sh -o /mnt/c8-test.sh && bash /mnt/c8-test.sh
```

需要代理时，预先设置 `https_proxy`，或分别为 curl 与脚本传 `--proxy URL`。
如果该环境确需临时跳过证书校验，下载加 `-k`，脚本加 `--insecure`。
也可显式指定 `--source /path/to/vllm-ascend`。

安装器下载地址固定到提交 `67a21fa3ada065039a749aa28cc52cc24c208f7b`，校验 manifest/patch SHA256、受影响文件完整内容、类型和执行位。
部分旧补丁、未知改动、相关 staged 修改或新增文件冲突会被拒绝；不自动 reset、stash、覆盖或合并。
其它不相关文件不改动。若拒绝，请保留报错与源码状态，不要强制应用。
安装器不安装依赖、不下载权重、不运行测试、不启动或重启服务。测试前人工重启服务以加载新源码。

## 配置原则

沿用同一套主模型权重和已能运行的 DSpark 配置，只在主模型开启：

```text
VLLM_USE_V2_MODEL_RUNNER=1
--dtype bfloat16
--kv-cache-dtype auto
--block-size 128
--additional-config '{"enable_sparse_sfa_c8":true,"enable_mlapo":false,"enable_kv_nz":false}'
```

把这些键合入已有 additional-config，不要覆盖其它业务配置。DSpark 路径通过原有 speculative-config 指定，不需要向本仓库提供权重或真实路径。
DCP 使用 `--decode-context-parallel-size 2` 或 `4`，且必须与当前 TP 配置相容。
先用 `--enforce-eager`，随后移除它并使用 `FULL_DECODE_ONLY`；图 bucket 沿用已验证的 TP/FlashComm 配置。
分别测 `enable_mlapo=false/true`。不能以去掉 eager 后服务启动成功替代真实请求 replay 验证。

## 第一步：单卡、无权重原生专项

进入已打补丁的 VA 源码目录执行：

```bash
python -m pytest -sv --noconftest tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_k3_c8_spec_dcp.py
```

6项用例覆盖多 query、负 slot 写入、DCP 1/2/4分片、空分片、原生 QSFA max/sum、AttentionUpdate 和图 replay 后长度/活跃行变化。
此测试在一张卡上模拟各分片，**不验证 HCCL、完整模型加载或 DSpark 调度**。
测试模块会在非 A5 环境跳过；看到 skipped 不算通过。缺少要求的原生扩展应先解决环境问题。

## 第二步：模型验收

固定相同主/草稿权重、输入与输出长度、并发、TP、其它优化开关，比较：

| 顺序 | C8 | DSpark | DCP | 目的 |
| --- | --- | --- | --- | --- |
| 1 | 关闭 | 关闭 | 1 | 同权重基线 |
| 2 | 开启 | 关闭 | 1 | C8 回归 |
| 3 | 开启 | 开启 | 1 | 主模型验证 + 草稿隔离 |
| 4 | 开启 | 关闭 | 2 / 4 | 实际分片与通信 |
| 5 | 开启 | 开启 | 2 / 4 | 组合验收 |

每组覆盖普通/MLAPO、eager/FULL_DECODE_ONLY；必要时补同 DCP/DSpark 配置的 C8关闭对照以定界。
检查跨页、长短上下文、多个请求、请求退出/补入、多轮接受/拒绝；对齐逐步 logits、生成结果和 Cache 用量。
在 profiling 中确认 QSFA 与实际图 replay、DCP 通信及合并；性能测量使用独占或稳定负载的卡。
不要把单算子分片正确当成主/草稿模型精度验收，不要把“没有崩溃”当成完整支持。

## 保留的边界

本版不支持 PCP、通用 FULL/PIECEWISE、KV transfer、KV layer parallelism；只面向 MRV2 / A5 / 当前VA0.29基线。
DSpark 接入不等于宣称所有投机解码算法均已验证。`kv_b_proj` 仍要求非量化。
环境预期为 CANN 9.2.0，需提供 `cann_ops_transformer.kv_quant_sparse_flash_attention` 等原生接口；无 BF16 外部反量化 fallback。

## 返回测试结果

请记录本分支/安装脚本提交、镜像和 VA/vLLM/torch-npu/CANN 版本、已脱敏启动参数、原生测试完整统计、首个根因报错及所测矩阵。
模型权重和数据集无需上传；可提供脱敏逐请求指标与相关算子 profiling。
