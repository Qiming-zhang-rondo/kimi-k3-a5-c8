# 直接修复 a1ad5cd 的生产接入

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
