# recent128 V4 框架精度接入

当前执行 [CURRENT_FRAMEWORK_TASK.md](CURRENT_FRAMEWORK_TASK.md)。已验证的V4单算子为32行加载、416B输入、输入2份/输出1份buffer，Q4/K8K prepared-read约82.08µs；框架probe现明确选择load_tile4，核对真实导出/geometry与库SHA，不回退旧overlay。

实际接线保留原C4量化和写入，在RMSNorm/RoPE之后、MXFP4之前捕获原BF16 KV，写带logical-position tags的分层热环。原TopK整体传给一次attention流水，后续回到框架的输出投影。CPU完整FP64参考检查每个真实query/head，validation NPU attention=0。V4输出stats关闭，库/实际entry/variant和调用次数进入证据。

**格式边界：已验证kernel仅D512/R64/packed400。** GLM-5.3-Flash相关源码存在NoPE/R0/packed272路径；先用JSON-only预检读取实际checkpoint配置。不兼容会在加载权重前停止，不把272B当400B或用Python补齐冷cache。当前尚不能声称Flash NoPE模型已接通；任务说明要求内网agent自主一次收齐NoPE对应源合同，不让用户反复中转。

```bash
python framework_integration/run_model_probe.py --engine-json /path/to/engine.json \
  --arm single_pass --preflight-only --output /tmp/recent128-model-preflight
```

兼容R64后：隔离安装overlay，OFF/native/V4独立进程真实prompt smoke，再跑原GLM评分任务。完整命令、Prolog共同NATIVE实验要求及NoPE分支交付见当前任务。

配置additional_config.sfa_recent128新增single_pass_kernel=overlay/fast_pair/load_tile4。旧配置默认overlay兼容；probe默认load_tile4。window0不分配热环或加载候选库。真实库缺入口/geometry不符会失败，不静默fallback。

当前仅accuracy_eager、单活动请求、Q<=1024；不支持prefix/PD/PCP/DCP/MTP、图捕获。head补齐、CPU参考和metadata复制属于精度检查，不能用此路径测TPOT。生产Prolog/静态缓冲/并发与PD生命周期需后续适配，真实模型和服务尚未在本机验证。

历史ABI4热读归约任务保留在FRAMEWORK_EXECUTION_TASK.md；旧94µs单遍接线任务保留在SINGLE_PASS_FRAMEWORK_TASK.md，均不取代当前V4任务。
