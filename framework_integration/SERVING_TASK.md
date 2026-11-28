# 直接修复 a1ad5cd 的生产接入

## 最新 7d0ee407：继续同一批 reader 验证，不重编内核

新批量库已编译，B1Q1 eager 对独立数学参考通过：relative L2=0.0022918427、max_abs=9.98e-5、faults=0。
Graph 在进入 attention 内核之前因 plan key 不匹配退出，B1Q4/B8Q4 尚未执行。

根因是 torch.npu.graph 未指定 stream 时会切换内部捕获 stream，原代码按当前 stream 查找已预热的 plan，
因此外层 stream 已准备也会查不到。现在 smoke 显式传入同一预热 stream；生产入口按 builder/draft/ubatch
显式预留独立 execution lane，既允许预热/捕获切换 stream，也不合并独立并发 workspace。

两处 where 同步独立修复：PyTorch 2.10 的 where.self_out 两个分支要求 Tensor，ScalarOther 重载没有 out。
这不是 NPU 独有约束。-1 INT32 和 0 BF16 的同设备 Tensor 在 prepare 阶段创建，read 中只广播使用；
原 TopK 不改，输出 dtype 不改，无 read 内新增 Tensor 分配。
依据：[PyTorch 2.10 原始重载](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/native_functions.yaml)、
[Ascend graph 的默认捕获 stream](https://github.com/Ascend/pytorch/blob/master/torch_npu/npu/graphs.py)。

本轮只更新 Python，复用已经通过编译的 /tmp/hot128_serving/reader-build/libqsfa_recent128_direct.so
（原结果 SHA256=64d4a94eb843d97b172ee1dcff61abbe7e83fa515d680138c59c3b5f02b83955）。
不要重编 ASC、加载模型或再改 vendor；用新输出目录一次执行原矩阵：

```bash
python experimental/single_pass/run_batched_smoke.py \
  --library-dir /tmp/hot128_serving/reader-build --source-root "$PWD" \
  --device npu:0 --graph --output "$NEW_RUN/batched-smoke"
```

设备号沿用已确认空闲的测试设备。该命令集中覆盖 B1Q1、B1Q4、B8Q4、Graph 两阶段元数据变化与坏 epoch 锁存。
本地相关 27 项宿主检查通过，A5 Graph 结果待此次重跑；不能把 B1 eager 通过写成 Graph 或 B8 已通过。

## 使用 A5 之前一次说清楚的能力边界

6486b3d 是生产接入**候选代码**，不是完整服务已可用的证明。130 项宿主测试不能代替设备或完整模型验证。
以下是已知边界，必须先在内网解决未接线项，不能加载102个权重分片后再让用户逐个发现。

| 特性 | 已有实现/证据 | 当前缺口或限制 |
| --- | --- | --- |
| 原 MXFP4 主缓存 | 保留 packed400 写入和原 TopK 合同；OFF 不走新路径 | 真实新 Prolog ON/OFF 的 query、C4 packed/scale 尚无设备逐字节对照 |
| 真实 PrologV3 | 公共调用适配、完整 ABI1、实际写点要求、已通过的独立 writer | 私有 producer variant 必须实际编译注册；独立 writer 不是 producer |
| 并发8 | 稳定请求 bank/epoch、批量 TND reader、隔离路由；批量库已编译，B1Q1 eager 精度通过 | B8 尚未执行；不能外推 B1 或旧 P8 writer 成为并发 attention 已通过 |
| Graph | 固定缓冲、按 device/stream/shape 准备 plan；新 smoke 覆盖动态元数据 | 所有 layer/lane/bucket 必须捕获前准备；新 serving 动态 replay 尚未设备验证 |
| MTP | 修正后的 GPU 位置、draft lane 隔离、容量保护、事务 API | 完整接受/拒绝和提交链未验证；draft 层自己的历史 BF16 热窗来源仍须核实，不能拿 target 层就绪代替 |
| APC / prefix cache | 实际 BF16 尾窗快照导出/导入 API | **尚未自动接通 prefix 命中、驱逐、抢占恢复。不是仅缺测试。** 非零位置新请求缺真实尾窗会报错 |
| P/D | 实际尾窗导入及接收完成事件 API | **尚未接通 Mooncake 的 BF16 尾窗发送、接收和通知协议。不是仅缺测试。** C4 传输成功不足以证明 BF16 已到 |
| PCP / DCP | 当前新 sidecar 未有对应 gather/shard/跨 rank 完成协议 | 当前不支持；现在在配置期明确指出，不能因 serving 配置被接受就列为已支持 |
| 性能、显存、整模型精度 | 旧独立算子实验有历史数据 | 新 batched_v4 的速度/容量损失/TPOT/GPQA 均未测；不得引用 B1 KV-split4 的旧速度作为本实现结果 |

硬格式合同一次列明：仅 latent512、RoPE64、KV-head1、page128、packed400 主缓存、BF16 query、INT32 TopK2048；
R0/272、其他 TopK、页大小或格式没有实现。每64个 local heads 一组，不足补齐，多于64会多组调用。
新物理 ring 容量 C=align128(128+max_query_chunk-1)，不是固定128槽。
max_query_chunk 必须覆盖实际每请求 prefill/verify chunk，不能只按 MTP 的4个 token 填写；总 token/pool/表宽也受预分配界限制。

已完成的 A5 设备证据为旧独立 ring writer 8/8，加最新独立批量 reader 的 B1Q1 eager；它们不验证真实 Prolog、C4 量化或完整 MTP。
同一次 Graph 中 valid mask 变化的测试覆盖，也不能用旧 writer 的固定 valid mask 重放替代。

## 一次准备、一次成批验收的执行约定

1. 上 A5 前，先完成上述 Prolog、APC、PD 和 draft 层热窗等已知实现缺口；主机检查/全部编译/ABI与配置问题一次收集，
   能在 CPU 上解决的错误不占设备时段，不以关闭用户特性替代实现。
2. 预先固定一份矩阵：B1/B8、Q1/Q4、eager/Graph、环回绕/padding/请求重排、MTP全接受/部分拒绝/全拒绝、
   APC冷启动/命中、P→D首步/后续步。每项写清原配置、预期和输出位置，不能跑到一半再添加已知必要条件。
3. 已有 run_batched_smoke 是**reader 部分**的成批设备门禁；APC/PD/真 Prolog/完整模型不在它的覆盖内。
   完整矩阵须接好相应实现再执行；不能宣称一个 reader smoke 已经把所有特性一次验完。
4. 同一套构建通过后再加载完整模型，保留原生产特性，完成 OFF/ON 对照和完整 GPQA。
   新设备错误仍可能出现，但每次回传要包含所有已执行项和阻塞项，不能只返回下一处试错指令。

目标固定为原 MXFP4 主 KV + BF16 recent128，保留并发8、Graph、MTP、prefix cache、P/D。
不把 C4 改 C8，不把 max_num_seqs 改1，不强制 eager，不丢弃 PrologV3。
旧 accuracy_eager 是单请求数学验证代码，不能仅删除其 guard 后用于服务。

## 已交付的实际代码

- serving_state：稳定请求 bank/epoch、预分配路由、分 lane 元数据、独立 BF16 ring、真实 prefix/PD 尾窗快照。
- serving_runner：挂接真实 NPUModelRunner 和 SFA metadata builder，使用本轮设备 positions/query_start_loc；MTP 不从乐观 CPU 长度重建位置。
- serving_prolog/hooks：包裹实际 MXLLM `_call_mxfp4_cloud_prolog`；ON 只调用一次显式 recent128 producer，原 query 五项返回和 mode4 C4 写入合同保留。
- serving_runtime/serving_reader：生产分流、完整原始物理 C4 池、批量原 TopK、Graph 固定缓冲和真实批量 reader。
- 新的 additive batched_v4 ASC/bridge：沿用 V4 load32/stride416 流水，按请求 bank+position+epoch 读 BF16 热环；旧 B1 KV-split4 不更改。
- install_overlay --update-runtime：验证已有托管 overlay，备份并更新 runtime，不改原 MXLLM 三处补丁与 vendor。

这些是待 A5 联合编译验证的候选实现，不宣称已拉起服务或测过生产性能。

## 内网 agent 直接依次执行

1. 在当前匹配的 SDK 中完成真实私有 Prolog variant 的编译和注册。新入口必须是
   `cloud_ops_turbo.cloud_mla_prolog_v3_recent128`，函数 `recent128_abi_version=1`，完整 ABI 见 PROLOG_PRODUCER_TASK。
   将已 A5 验证的 writer 搬运/事件写法移到实际 mode4 量化输入和真实 KR 写点，不能给旧函数贴 marker 冒充实现。
   OFF/ON query 与 C4 packed/scale 逐字节不变；sidecar 来自真实 FP32 量化输入的 BF16_RNE。

2. 在独立目录重编本仓 `experimental/single_pass` 的 shared library，保留当前匹配 SDK/dav-3510 和已有 KV-split 开关，
   开启新增 serving batch 目标。不要覆盖已工作的 C4/vendor/旧候选库。先跑本仓批量 reader correctness smoke，
   验证 B1Q1/B1Q4/B8Q4、原 TopK、不同行/epoch/padding、Graph 重放。首个设备异常即停该测试，不重置卡。

   ```bash
   cmake -S experimental/single_pass -B "$HOT_RUN/reader-build" \
     -DSFA_SDK_ROOT="$CURRENT_MATCHED_SDK" \
     -DCMAKE_PREFIX_PATH="$CURRENT_MATCHED_SDK;$CURRENT_MATCHED_SDK/lib64/cmake" \
     -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DQSFA_BUILD_BATCHED_SERVING=ON
   cmake --build "$HOT_RUN/reader-build" -j8
   python experimental/single_pass/run_batched_smoke.py \
     --library-dir "$HOT_RUN/reader-build" --source-root "$PWD" \
     --device npu:0 --graph --output "$HOT_RUN/batched-smoke"
   ```

   设备由内网 agent 确认空闲；SDK 沿用已经通过 writer 的匹配安装，目录变量用内网实际值，不升级、不 source 旧包。

3. 对当前 hot128 专用 MXLLM checkout 更新托管 runtime：

   ```bash
   python framework_integration/install_overlay.py --mxllm-root "$HOT_MXLLM" \
     --source-profile pr199 --apply --update-runtime --output "$HOT_RUN/overlay-backup"
   ```

   复用之前实际通过的 source-profile/expected-SHA 参数。若旧 runtime 曾被内网修改，工具会指出与托管 manifest 不符；
   在内网审查并保留这些修改，不伪造 manifest/强制覆盖。无需把内部实现上传给外部。

4. 仅更新原 additional-config 的 sfa_recent128 子项：

   ```json
   {"window":128,"execution":"serving","preprocess":"preserve",
    "reader":"single_pass","single_pass_kernel":"batched_v4",
    "max_query_chunk":1024,
    "source_root":"本仓绝对目录","library_dir":"新批量库独立目录",
    "diagnostics_path":"本轮独立目录/layers"}
   ```

   max_query_chunk 必须覆盖真实每请求的最大 prefill/verify chunk；不是一律设4，也不是总 batch token 数。
   物理 C 按 W+maxQ-1 对齐到128，与读取窗口128不同。原 vendor additional-config 字段和
   attention-config 中 indexer_kv_dtype=mxfp4、mla_kv_quant_mode=mxfp4 均保留。
   可用 serving_config.py 从现有 JSON 只合并本子项；不重新生成整套启动参数。

5. 在内网接好现有 prefix/Mooncake 生命周期：
   - prefix snapshot 或 P→D 真正传输 BF16 latent/KR + tags，再调用 serving_runner.receive_tail。
   - snapshot_tail 必须对应已提交的前缀边界；PD 的 completion 需要真实接收完成事件。
   - 主 C4 已到达不代表新 BF16 sidecar 已到达。非零起点缺尾窗会明确报错；不能假造 ready 或用 C4 反量化顶替。
   - 新 token/草稿使用本轮修正后的 GPU 位置，按位置因果判断；请求完成显式释放，复用 bank 时 epoch 改变。
   公共生命周期接口已提供，实际 connector 的私有接线在内网就地实施，不能把接口存在写成传输已完成。

6. 按原生产并发8、Graph、MTP、prefix、P/D 参数拉起 hot128 专用服务，baseline 链不动。
   缺私有 producer 就先完成第1步，不再回到 Native 或 max_num_seqs=1。
   完成请求后通过 worker RPC 调用 serving_runtime.validate_serving_faults()，检查所有已准备 bucket 的 sticky 错误；
   此检查在计时外同步，不能每层 .tolist()。故障为零也不代替真实模型精度测试。
   再用相同 seed/请求集对比完整198题 GPQA；78/94 的中途结果仅保留为中途记录，不能归因成最终精度。

366ae84 的加载后报错问题同步修复：完整 engine 配置的静态限制在 config factory 前检查；serving 模式
缺实际 producer/批量库的错误在模型层构造时检查，早于102个权重 shard 的加载。
只有依赖真实权重/内核行为的条件保留到相应初始化或设备执行时，不将未运行的内核伪称已验证。

只回传部门允许的编译状态、首错、实际调用/测试摘要。内部源码、schema、二进制和原始材料留内网。
