# Kimi K3 / A5 / CANN 9.2.0：QFM 第一阶段验证

本目录是独立验证包，**不打 VA patch、不加载模型、不修改权重、不升级 CANN**。
后续 VA 适配基线为 `c09d3e20b9c45c80b34ae3781440d90904a44023`；本阶段不依赖 VA 安装。

## 运行顺序

1. 在目标 A5 容器内执行环境检查：`python run.py --card 0 --suite inspect`。
   它会打印包版本、已有 MLA Prolog/QFM 注册接口和来源。检查结果不是算子执行通过。
2. 编译并在私有目录部署 **QFM + quant_flash_mla_metadata**，参见 [BUILD.md](BUILD.md)。
   必须在目标容器使用现有 CANN 9.2.0 工具链编译。不要安装面向其他 CANN 版本的预编译包。
3. 在同一容器、同一私有依赖环境下执行：`python run.py --card 0 --suite all`。
   `--card` 是一张有使用授权的空闲物理 A5；不需要 8 张卡。不要在他人忙碌卡上测性能。
4. 回传末尾打印的 `summary.json`；失败时同时回传对应日志。无需模型或数据集。

一键下载入口是仓库根目录 `qfm1.sh`，支持显式的 `--insecure`、`--proxy`，内部文件固定提交并校验哈希。
脚本不自动安装依赖、不自动合入 VA；构建需要显式 `--build-ops`。

## 本阶段实际验证什么

| 项目 | 验证内容 | 不代表什么 |
|---|---|---|
| 普通写入 | 当前 KV 量化为 E4M3，原生 Scatter 写 PA_NZ；逐字节对照、页间 padding、非零 storage offset、未写区域 guard | 尚未接通 V4 直接写 QFM Cache |
| 普通读取 | QFM 读取唯一 full576 FP8 Cache；512 latent 和 raw64 都是 FP8；固定 KV FP32 scale，Q per-token/head scale | 不是原 Group128 packed656，也不是 MXFP8 Group32 Cache |
| Prefill / APC | 每请求 Q 切成最多 16 行的虚拟段，共享原物理 KV 页，长度是 prefix+段末位置，统一批量调用 | 不修改调度器，不复制 512 份历史 Cache，不创建 Q×K token 索引 |
| 边界与 TP shape | Q=1/4/9/15/16/17/33，跨 127/128/129 边界，多请求、不整页前缀，heads=6/12/24/48 | 只覆盖本地算子 shape，不是多卡 TP/HCCL 验收 |
| Graph | 固定容量捕获 metadata+QFM，改变长度、有效行数、Q/scale、页表；空→非空重放 | 不是整模图、FlashComm、混合 batch 的验收 |
| DCP 数学 | 同一张卡模拟两个 KV 分片，包括空分片，按 FP32 LSE 合并并对照全 KV 参考 | 不创建通信组，不证明分布式 DCP/AllGather 没有死锁 |
| 性能 | 默认 Q=8192、KV=131072、heads=12；512 虚拟段单次原生 Attention 调用 | 不是整模 TTFT/TPOT，也不承诺快于 BF16 |

## 精度与性能判据

- 小用例全部 Q 行对照独立 FP32 参考；大用例对段边界、起止位置及均匀样本核对，日志标明 `exhaustive=false`，不能称为 8192 行逐元素全覆盖。
- QFM 内部还有低精度计算，反量化 Q/K 后的 FP32 softmax 是近似参考，不要求 bitwise 相等。
  当前输出阈值：relative L2 ≤ 0.02，且所有元素满足 `abs_error <= 0.025 + 0.03 * abs(reference)`；LSE 单独更严检查。
- 原始 BF16 输入与 FP8 路径的误差另外报告；这不替代模型精度、logits、DSpark 接受率验收。
- 性能分别报告 QFM kernel、metadata、Q 量化+页表组织+metadata+Attention、当前 KV 量化+Scatter、BF16 FIA kernel。
  QFM/FIA 耗时均关闭 LSE 输出，包含预热；报告 NPU event 与同步墙钟时间及逐次样本。
- BF16 FIA 使用相同逻辑 Q/KV 长度，单次完整请求调用。当前 BF16 基线将 12 heads 补至 16，报告中明确记录，不能把该比值视作完美同形 kernel 对比。
- 测试为独立比较临时保留 BF16 输入与参考，并不是生产方案中的 BF16 历史影子 Cache。

默认测试使用完整原生 QFM，不在失败时偷偷切回 QSFA/FIA。缺失依赖、超时、缺少完成标记都不算通过；首次失败后停止后续 worker。
每个 worker 默认 600 秒超时，可用 `--timeout-seconds` 调整。杀死进程不保证已经挂住的 NPU 恢复，失败后先检查设备状态。

## 显存与输出

`python run.py --card 0 --plan-only` 不导入 NPU 库，会输出形状和粗略显存预算。
默认预算上限 4096 MiB 是运行前估算门槛，不是硬隔离或预留；编译器/算子 workspace 未必能准确估计。
实际峰值由各 worker 的 `peak_allocated_mib` 报告；它也不包含所有驱动开销。

结果写入独立目录，含 `summary.json` 和各阶段日志。只有实际执行全部所请求的测试，且所有 worker 明确报成功、退出码为零时，才有 `all_passed=true`。
`inspect/preflight` 通过仅表示清点/加载成功，`native_computation_validated=false`。
默认 `all` 仍明确记录 `fused_prolog_validated=false`，不会伪装成 V4 融合写入通过。

## MLA Prolog V4 的边界

现有 V4 可能经 `cann_ops_transformer::mla_prolog` / `aclnnMlaPrologV4WeightNz` 暴露，不能只查 `torch_npu.npu_mla_prolog_v4` 这个名称。
CANN v9.2.0 的对应接口不能直接提供 QFM 的 full576、per-tensor FP8 ABI：

- per-tensor 路径是分离的 FP8 latent512 / BF16 raw64，Q scale 仅对应 latent；
- 合并的 per-group 路径是 packed656，不是 QFM full576。

所以保留现有 V4，不默认重装。第一阶段验证普通写入；以后若复用 V4，需验证其当前 token 输出到 full576 量化写入的适配，不能将两种 Cache 格式直接互换。

`--suite prolog` 是**可选且默认不跑**的历史 CANNBot-DSL 实验：其匹配 recipe 的旧接口固定 96 heads，并不支持 TP 本地 6/12/24/48 heads。
它需要单独固定的 DSL 和参考源码；不是现有 V4 检查，也不是产品融合路径已完成的证明。

## 固定来源

- QFM / metadata：[ops-transformer 8b6924e](https://gitcode.com/cann/ops-transformer/tree/8b6924e68ed27181141cde84fa5a942c0c060aac/attention/quant_flash_mla)。
- K3 参考实现：[cann-recipes-infer 4447f2f](https://gitcode.com/cann/cann-recipes-infer/blob/4447f2f0972cb9287e069e3994b1d46ffe244ebc/models/kimi_k3/models/modeling_kimi_k3.py)。
- V4 契约：[CANN v9.2.0 V4WeightNz 文档](https://gitcode.com/cann/ops-transformer/blob/v9.2.0/attention/mla_prolog_v3/docs/aclnnMlaPrologV4WeightNz.md)。

这里没有 A5 执行结果；CPU 单测只证明测试脚本的元数据、寻址参考和监督逻辑。A5 编译、原生精度、图重放与性能必须在目标容器实测。
