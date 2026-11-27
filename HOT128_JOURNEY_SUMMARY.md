# Hot128 问题与解决方案总结（2026-11-27 全天）

## 问题 1: 环境变量不生效
- 现象: launch_hot 设置 SFA_RECENT128_CONFIG 环境变量后，profiling 仍显示旧算子
- 根因: 代码从不读取环境变量；正确入口是 VllmConfig.additional_config.sfa_recent128
- 解决: sfa_recent128 配置加入 run_d.sh / run_p.sh 的 --additional-config

## 问题 2: Contract 拒绝 kernel 字段
- 根因: 字段名应为 single_pass_kernel 不是 kernel
- 解决: additional_config 里改名

## 问题 3: 缺 diagnostics_path
- 解决: 加 diagnostics_path: /tmp/sfa_diagnostics/layers

## 问题 4: PROLOG_V3 不暴露 BF16 KV（核心阻塞）
- 现象: ContractError: PROLOG_V3 has no original BF16 KV output
- 根因: PROLOG_V3 融合算子一步完成 RMSNorm+RoPE+MXFP4 量化，量化前 BF16 中间值不存在于输出
- 解决: 设计新 producer variant cloud_mla_prolog_v3_recent128，在 RMSNorm 和 Quant 之间加可选 BF16 环写入
- BF16 写点: kernel_cloud_mla_prolog_split_n_arch35.h 的 RmsNormAndQuantizeCkv，RmsNormNormal 和 QuantPerTileMxFp4 之间

## 问题 5: 实现 bug（Codex 审查发现 4 个）
1. Cast 用错重载: 6参数 mask 版 → 修为 4参数 count 版
2. bank/position 混淆: slot % capacity → 修为 bank * capacity + position % capacity
3. sharedBuf 做 BF16 scratch（流水线冲突）→ 修为独立 TBuf
4. 只实现 3 个 ABI 字段 → 补齐全部 ABI1

教训已写入 ENGINEERING_DISCIPLINE.md

## 问题 6: 我的 smoke test DataCopy 编译失败
- 现象: "requires 4 arguments, but 3 were provided"
- 根因: 裸指针不是 GlobalTensor、缺 AIV 任务声明、GM→GM 不在标准通路
- 解决: Codex 交付完整替代 smoke（prolog_ring_smoke），正确 DataCopy + AIV 隔离 + 跨流水线事件

## 最终结果: 设备验证通过
Status: DEVICE_RING_WRITER_VALIDATED
8/8 测试通过（P1/P8 × OFF/Eager/Graph×2）
全部检查项通过: latentRing, ropeRing, tags, input_unchanged, graph, poison_retained

## 基础设施状态（已完全分离）
启动链:
  普通 C4: launch_d.sh → start_d.sh → run_d.sh
  hot128: launch_hot_d.sh → start_d_hot128.sh → run_d_hot128.sh

每日命令:
  普通 C4: bash /home/z00957521/launch_d.sh（或 launch_p.sh）
  hot128: bash /home/z00957521/launch_hot_d.sh（或 launch_hot_p.sh）

## 下一步
1. 验证过的环写入机制集成回 PROLOG_V3 真实 kernel
2. 验证 C4 packed 逐字节不变
3. B8Q4 + MTP + Graph 完整场景测试
4. GPQA 精度对比（普通 C4 baseline vs hot128）
