# recent128 serving 的完整检查范围

本表对应当前公开 `execution=serving` 实现。配置被接受、Python可导入、库已编译、设备对照通过和完整服务通过是不同状态。
检查工具聚合能够在当前阶段检查的条件；需要真实层或设备输入的项目明确标记待检查，不用静态PASS替代。

## 1. 启动前能够检查的配置和文件

| 条件 | 当前要求 |
| --- | --- |
| 激活入口 | `VllmConfig.additional_config.sfa_recent128`；旧环境变量不能单独激活 |
| 模式 | `window=128, execution=serving, preprocess=preserve` |
| reader | `reader=single_pass, single_pass_kernel=batched_v4`；不是旧B1的kv_split4入口 |
| chunk界 | `max_query_chunk`为1..65536的整数，覆盖每请求实际chunk/1+draft tokens；不是只填decode单token数 |
| 请求池 | scheduler真实`max_num_seqs`为正整数；serving允许并发8/32，不能套用旧accuracy_eager的B1限制 |
| token桶 | scheduler真实`max_num_batched_tokens`及Graph桶必须能容纳实际输入/填充；按真实调度配置准备，不把并发数等同token数 |
| CP | 当前sidecar没有PCP/DCP的gather/shard接线；非1必须明确报告，不自动修改生产配置 |
| source_root | 仓库根，包含`experimental/single_pass/batched_serving_bridge.cpp`、`batched_serving_api.h`、`single_pass_api.h`、`single_pass_tiling.h` |
| SDK helper | 同一根的`experimental/torch/load_sfa_direct.py`必须存在；只同步framework_integration不够 |
| library_dir | 包含实际`libqsfa_recent128_direct.so`的目录；batched serving导出ABI1、buffers160B，旧B1库不可代替 |
| 诊断 | `diagnostics_path`为实际可写目录；路径名字不影响算法，例如`/tmp/sfa_diagnostics`有效 |
| SDK/bridge | 活跃SDK与reader构建记录匹配；当前Python/Torch/torch_npu及compiler/ninja满足薄host bridge构建，实际bridge加载由原preflight执行 |
| producer | 实际Prolog层要求`cloud_mla_prolog_v3_recent128`函数整数ABI1；实际Native层要求原始tap、真实BF16 writer与对应CP路由，不能互相冒充 |
| 原选路依赖 | 选择Prolog时已审阅PR199要求原`cloud_mla_prolog_v3`和`cloud_mla_prolog_v3_functional`可调用；Native/DSACP不因缺少Prolog被强行切路 |

环容量`C=ceil((128+max_query_chunk-1)/128)*128`，每层最低sidecar数据量约为`P*C*1168`字节，
其中P是请求池大小；这不含reader workspace、allocator和其他模型显存。预留Q容量不等于已有MTP/APC/PD完整接线。

`enable_dsa_cp`若按已审阅VA版本选择`AscendSFADSACPImpl`，该类原本关闭fused preprocessing，
不会执行Prolog写环。本轮新增Native/DSACP候选，预检标待设备验证；不能在权重转换后改enum，或静默关掉DSACP。

## 2. 层对象建立和原权重准备后必须核实

`preprocess=preserve`的意思是保留框架原选择，**并不强制PROLOG_V3**。
`contract.py`的选路检查只比较实际`impl.preprocess_type`，不检查安装目录或provider ABI；两者分别验证。

- 保留实际层的原权重准备和选路。Prolog层走已有fused producer；Native层只有真实tap/forward/store、BF16 writer和对应路由接通后才接受；MLAPO仍未适配。
- 主缓存为实际MXFP4路径：`enable_mxfp4_kv=True`；D512、RoPE64、KV-head1，稳定唯一`layer_name`。
- `local_num_heads`为正，attention scale有限且大于0。query与原TopK/packedcache真实形状在首次输入时继续校验。
- 上游Prolog选择还依赖层的量化方法、KV producer/consumer角色、norm及fused投影是否存在、indexer query是否有scale、q_proj是否分chunk、A5能力及原Cloud算子是否可调用。
  `dcp_group`和DSACP子类也可能在外层阻止选择；仅查看`_get_fused_type_unsupported_reasons`空列表不代表所有外层gate通过。
- mode4仍沿原tiler、workspace、blockDim和权重格式；不能只将`impl.preprocess_type`赋值为PROLOG_V3绕过检查。

其中部分条件由实际权重类型和层实现决定，通用engine JSON无法证明它们。启动前预检明确列为运行阶段待确认，
层对象出现后聚合诊断一次输出，而不是伪装已全部通过。

## 3. producer和reader真实执行合同

- 新provider以原Cloud字段名称接收输入；新增ring参数完整转发。显式None保留，无重复键覆盖。
- 同一次Prolog中旁路写入真正最终量化输入的BF16_RNE latent，以及真实post-RoPE KR64；原FP32量化输入不改。
- Native使用原RMSNorm的实际BF16输出和真实post-RoPE，不重算FP32 RMSNorm；本轮独立writer读取这些GM张量，DSACP TP>1另外交换原始BF16片。
  该正确性候选增加kernel/通信，不承诺零性能成本；原C4量化器、packed400和store保持。
- position/epoch tags唯一owner；token的bank/position/epoch/valid来自当前GPU元数据。无效和Graph padding不写。
- reader等待整个producer完成；跨stream使用完成事件，不能把tag当同kernel内跨核ready flag。
- OFF/ON原query及C4 packed/scale逐字节不变，ON sidecar与真实来源一致；首次TBE动态编译和实际mode4分支都需经过。
- reader输入：BF16 query `[T,H,512]`、BF16 RoPE `[T,H,64]`、连续INT8 cold `[pages,128,1,400]`、INT32原TopK `[T,1,2048]`。
- 所有输入同设备；GPU请求长度、block table、稳定bank/epoch和token位置/valid齐全。reader计划在capture前按真实shape/lane准备。
- 每条可并行forward/draft/ubatch链独立workspace；固定地址保留，重放时设备内容更新。结束后在计时区外收集sticky faults。

## 4. 保留的服务功能与尚需完成的验收

| 功能 | 必须满足的内容 |
| --- | --- |
| 并发 | 请求池bank/epoch稳定，请求重排/释放复用正确，B8等真实服务验证 |
| Graph | 每个实际shape/lane捕获前预热，重放更新输入/位置/valid/epoch；无CPU读tensor或图内新workspace |
| MTP | corrected positions、target/draft各自真实历史、接受/拒绝和暂存提交正确；容量覆盖1+draft并不足以证明这些 |
| APC | 按实际prefix命中边界恢复配对BF16尾窗，处理驱逐/抢占/LRU；仅恢复历史C4不够 |
| P/D | 原C4传输之外发送真实BF16尾窗，D等待接收完成后首读；长prompt也需要P尾部 |
| 精度/性能 | 真实模型任务精度、同构建同配置OFF/ON TPOT/吞吐/TTFT和显存；合成L2和静态检查不能代替 |

当前公开代码提供部分生命周期API，APC和P/D的自动接线不因安装provider而完成。工具必须列出这些待接线/待验收项，
不为了预检变绿关闭功能，不以C4退回结果冒充完整recent128。

## 聚合预检

```bash
python framework_integration/serving_preflight.py \
  --engine-json "$ACTUAL_ENGINE_JSON" \
  --output "$NEW_PREFLIGHT_DIRECTORY"
```

默认只读JSON/文件，不导入Torch、私有算子或加载权重。可在实际工作环境追加`--check-host`，
独立尝试provider导入及reader宿主ABI等检查；第三方模块自身的初始化由该环境负责，工具不主动执行设备kernel。
不编译reader或模型，不修改engine配置；输出逐项结果、全部可见阻塞和运行阶段待确认项。
