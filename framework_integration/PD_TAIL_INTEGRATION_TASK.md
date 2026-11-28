# 当前任务：识别真实 Mooncake 通道，接通 P→D BF16 recent128

内网 agent 直接执行本文件，不再让用户手工收集类名或转述问题。当前 P 已有成功推理，D 的非零起点报 `prefix/PD import_tail first`：原 C4 传输成功不等于 BF16 尾窗已传。保持已成功 attention/kernel、MXFP4 主缓存、QLI、Graph、MTP、APC、并发与原 P/D 配置。

## 先执行一次采集

自行找到实际 P、D 启动用的 engine JSON、MXLLM checkout 和 Python 环境。在各自原环境下执行，变量填写实际路径；不要猜 connector 名字、重装模型、重启成功服务或重新构建算子来做身份采集。

```bash
python framework_integration/probe_pd_connector.py \
  --engine-json "$ACTUAL_ENGINE_JSON" \
  --mxllm-root "$ACTUAL_MXLLM_CHECKOUT" \
  --output "$NEW_RUN/pd-identity"
```

输出 `pd_identity_summary.json` 是允许回传的小型身份摘要；`pd_factory.json` 含本地异常详情，只留内网。摘要失败时先看本地原异常修正采集环境，不把失败当作没有 connector。该探针只解析真实 factory，不加载权重/创建 worker，不证明运行中 worker 或传输正确。

若已有服务控制对象可调用实际 `collective_rpc`，把新增的 `sfa_recent128/pd_diagnostics.py` 放到实际 runtime 的同名包目录（只增加此文件，保留工作版本），通过现有 RPC 调用：

```python
from sfa_recent128.pd_diagnostics import probe_worker_pd, verify_worker_recent128
# 使用现有引擎对象的 collective_rpc；不是让你新建 LLM 或 HTTP 接口。
identities = existing_engine.collective_rpc(probe_worker_pd)
# 实际请求结束后、性能测量以外运行；显式同步 fault 并刷新诊断文件。
verification = existing_engine.collective_rpc(verify_worker_recent128)
```

以上是 callable API 示意，沿实际引擎已有控制路径执行；没有此入口就先回传 factory 身份，明确标记 live RPC 未执行，不创建调试监听端口、不向进程强行注入。不要在空 Python 进程调用来冒充真实 worker。

`793a514` 的零计数来自旧 PID38851/15:21 文件，不能代表16:51请求。文件仅在初始化和显式 flush 更新，Python producer/read 计数不跟踪每次 Graph replay。采集实际 TP worker PID、当前内存计数、已 warm plan、加载代码/缓存 kernel 的 SHA 和 flush 后 faults；不要加每层 `.tolist()`/warning 或每 token 同步。旧文件、无 fault、已 warm 各自都不等于当前请求的 GPU 热环执行证明；有争议再在单个受控请求的真实 worker 检查 device fault/tags 或 profiler，不改正式计时。

## 本次已交付与尚未完成

- `sfa_recent128/pd_tail.py`：versioned descriptor、不可变源快照、真实 engine 注册/同步 RDMA pull、ACK/epoch/nonce/lease 防串请求；18项宿主测试通过。
- `sfa_recent128/pd_worker.py`：worker 多 owner 导出/接收、真实 committed_end、完整 layer/tag/epoch 检查及保留已预留空 lease；7项宿主测试通过。
- 诊断工具3项宿主测试通过。所有这些均不是 A5/RDMA 通过证明。
- **实际 Mooncake backend hook 尚未接线，D 当前报错尚不能宣布已修好。** 保留非零起点 tail guard，不伪造事件/epoch、不从 C4 反量化成 BF16。

本轮先一次回传 P/D 两份身份摘要与实际 worker 采集状态。不要根据显示名称盲改协议：

| 实现身份 | 已核对的接收完成边界 | 注意 |
| --- | --- | --- |
| VA legacy `...kv_p2p.mooncake_connector.MooncakeConnector` | `KVCacheRecvingThread._reformat_pending_kv_caches` 后、发布 native done 前 | `kv_send_thread/kv_recv_thread`；每 shard 回调过早 |
| VA V2 `...kv_p2p.mooncake.connector.MooncakePullConnector` | 接收线程 `_handle_requests` 完成、返回完成集之前 | `_recving_thread`；P 元数据服务在 scheduler，需 worker→scheduler descriptor carrier |
| upstream `vllm...v1.mooncake.mooncake_connector.MooncakeConnector` | `process_pulling_result` 发布 finished_recving 前 | HTTP bootstrap/transfer_id/producer write 协议，不能套 VA hook |

上述来自已核对的公开版本，实际 module/class/method 为准。裸 `MooncakeConnector` 不一定是 VA V2；注册了另一模块不代表当前实例用了它。

## 一次列齐后续 adapter 的合同

1. 每真实 engine/session 维护持久 publisher/receiver，并串行化生命周期操作。不要使用已初始化后无效的全局 `register_buffer()`；新侧缓存须真实 `register_memory(ptr,nbytes)==0`。有界注册池/可靠注销必需；没有注销时核心会保留 owners 避免悬空，不能拿此行为长期跑 GPQA。
2. scheduler 提供真实已物化的 committed_end。`execute` 返回、scheduled token 数及 MTP draft 数都不代表接受边界或 NPU 完成。源快照在原 bank 释放/复用前完成；快照含所有实际 layer 和 owner 的 latent/RoPE/position/epoch，严格最后128区间。
3. worker helper 的 completion 必须有真实 `wait()` 与 `synchronize()`。`validate_snapshot` 在快照拷贝后确认 tags/epoch 全匹配、内存对 RDMA 可见并返回 literal True。先前 producer event 不能代替随后 clone 的完成。不能用空事件通过检查。
4. 通过既有控制 metadata 传 descriptor，保留原 metadata 的所有字段/aggregate、原 C4 transfer 和返回值，不另起网络服务。V2 原 ReqMeta 会丢未知字段，需明确 carrier 和 scheduler 顺序；不能只给 Python dict 加字段。
5. D 为目标实际 device 分配/注册，真实 transfer 成功后生成 completion，完整验证并 import 到该请求已预留 lease。必须在原接收完成信号发布、首个 attention 之前完成，失败不发布成功。
6. ACK 精确匹配 request/nonce/rank，确认 import 后释放源注册/快照。覆盖重试、重复/陈旧 ACK、取消、断线、源释放、目的 lease 复用以及跨 TP rank；锁与 owner 生命周期明确。
7. APC 全前缀命中即使跳过 C4 pull 也不能跳过 BF16 tail 获取；真正共享边界 h 不是源最终长度。APC、抢占恢复、MTP accepted commit 不因 PD 核心存在而自动支持，沿适配清单逐项验证。

身份确认后沿匹配 backend 接线，复用已交付核心；不让内网 agent 发明另一套传输协议。先单个真实 P→D 请求（含短 prompt）、目的 tags/epoch 和 faults，再并发/Graph/MTP/APC 命中组合。性能/GPQA沿 `BENCHMARK_GPQA_TASK.md` 冻结完整候选，不拼接中途换代码的成绩。

## 回传范围

只推小型 `pd_identity_summary.json`、真实 worker 的类/签名/SHA/计数/fault 摘要、实际测试结果和本地产物路径。没有 live RPC 就明确写未执行。不上传完整配置、私有源/头文件/schema、激活、tensor、RDMA地址/descriptor或大日志；不再让用户反复搬材料。
