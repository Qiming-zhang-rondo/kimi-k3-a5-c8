# 当前任务：按真实Native/DSACP执行路径补热环producer

最新报告41bef5c确认本轮Native路径，但其“永远不选Prolog”“已有smoke可直接复用”“零风险兼容DSACP”的结论过头。对应PR199公开源已独立核对：普通分支可选Prolog；DSACP禁用fused但Native tap有真实原BF16来源，方案可以适配。

直接执行 [NATIVE_DSACP_TASK](framework_integration/NATIVE_DSACP_TASK.md)。原始数据在 `_mxfp4_custom_kv_rmsnorm_rope` 的 `k_nope` 和 `k_rope`，现有overlay已有量化前tap。新Native producer必须同时解决BF16输入ABI、DSACP TP本rank分片的真实高精度KV交换，以及reader的local-query/全局request路由；只删除guard会错误，不能用C4反量化冒充。

原Prolog variant保留给真正选择该路径的层，不再为Native错误重复编Prolog，不强制切换生产路径。D节点原JSON=false却实际DSACP，需核对worker生效配置和加载源码；当前失败诊断已新增dispatch_state，记录canonical/effective flags、TP/PCP/DCP、实际method链/模块文件与SHA。诊断留内网，不再要求用户上传私有源码。

Native/CP适配尚未实现或设备验收，不能宣称服务修通。内网agent按任务在本地完成隔离候选与控制后只回传允许的小表，保留Graph/MTP/APC/P-D、原量化器和packed400。新增GM读写/高精度通信有成本，须实测，不承诺零性能代价。

本轮2项新dispatch测试+7项runtime检查通过；没有NPU执行或性能结论。旧的aggregated预检和完整要求仍适用，但Native接受须等真正producer/CP路由实现，不能通过改enum或关开关绕过。
