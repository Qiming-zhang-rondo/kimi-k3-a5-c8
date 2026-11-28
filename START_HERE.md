# 当前任务：验证真实Native/DSACP候选，修复70669eb首请求行数错配

本轮已有实际代码，不采用2fb7562的`.modified`占位文件：Native原BF16 writer、TP BF16交换、全局写环、本rank query路由和原forward/store接线。原Native量化器/packed400不变，V4 attention库复用；Prolog层继续原provider，实际Native层不再要求Prolog。

直接执行 [NATIVE_DSACP_TASK顶部](framework_integration/NATIVE_DSACP_TASK.md)。先用目标容器的小型eager/Graph fixture验证新writer，再受控备份更新完整managed runtime并启动新worker，保持原生产配置重复同一curl。当前错误来自全局token绑定与TP本rank query行数不同，已通过独立local binding解决，不删行数检查。

```bash
python framework_integration/run_native_writer_smoke.py --device npu:0 --graph --output "$NEW_RUN/native-writer"
python framework_integration/install_overlay.py --mxllm-root "$ACTUAL_MXLLM_CHECKOUT" --source-profile pr199 --apply --update-runtime --replace-edited-runtime --output "$NEW_RUN/overlay-backup"
```

设备/路径由内网沿现有真实环境填写。替换参数仅接管已管理runtime的手工修改并做实际旧文件备份，原MXLLM三源profile/markers仍检查，不覆盖vendor。不重编现有CANN/V4；新Native Triton kernel首调用编译和目标设备运行仍待验证。

169项framework宿主检查+9项controller检查通过。独立writer通过不等于TP8、模型或GPQA通过。TP>1候选每层增加一次BF16 collective和一个写环kernel，没有性能结论。

请求后在实际worker用现有collective_rpc调用serving_runtime.validate_worker_serving，核对producer/read层计数与sticky faults；新Python空进程的NOT_EXERCISED不算通过。P/D和APC自动BF16尾窗仍是明确未完成接线，不可假造ready、静默退C4或关功能换通过。先通过真实生产请求端到端验证，再跑原198题GPQA。

所有私有实现和原始日志留内网，只回传允许的首错、运行身份和控制/计时小表。不要让用户继续搬文件，不重新扩展性能摸高或DSL路线。
