# 当前任务：修复生产启动，不降低原配置

最新 366ae84 的加载后才报错问题和 a1ad5cd 的旧单请求 guard 已有实际代码修复候选。
内网 agent 直接执行 [framework_integration/SERVING_TASK.md](framework_integration/SERVING_TASK.md)，不再让用户选择切C8、关Graph、改单请求或丢弃MTP。

本次交付包含 serving runtime、真实 Prolog 调用路由、稳定请求bank/epoch与固定热环、runner/metadata/MTP位置接线、批量原生ASC reader、当前Torch薄桥、独立A5批量正确性脚本，以及托管overlay的runtime更新工具。原主cache仍是MXFP4 packed400，原TopK和query返回不改变。

明确使用 execution=serving / reader=single_pass / single_pass_kernel=batched_v4。它复用V4 load32/stride416流水，不冒充已测B1 KV-split4的速度。旧 accuracy_eager 保持独立；不能只删它的 max_num_seqs/Graph guard。

静态配置检查在权重加载前执行，真实producer和批量库/薄桥能力在模型层构造时检查，避免等102个shard后才报已知问题。

外部125项framework宿主测试、5项producer路由测试通过，新增160B ABI/批量tiling的C++宿主编译检查通过。新批量ASC与真实Prolog联合设备验证尚未运行。b25593fc的8/8通过仅属于独立writer，不可外推成新批量reader或服务已成功。

现在按任务文件在匹配SDK中编译真实私有Prolog variant和独立批量reader，跑run_batched_smoke，再更新hot128专用runtime/config并联调服务。Prefix/PD的真实BF16尾窗回调接口已交付，内网必须接好actual snapshot/传输完成；不能把C4到达当BF16到达。原baseline启动链和vendor保持。

内部实现及原始日志留内网，只回传部门允许的测试状态/首错/摘要，不再批量外传源码、schema或二进制。