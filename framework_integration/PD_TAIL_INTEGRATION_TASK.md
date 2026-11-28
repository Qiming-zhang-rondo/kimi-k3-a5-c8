# 当前任务：识别真实 Mooncake 通道，接通 P→D BF16 recent128

## 最新 15b5ec7：注册前移到 Store 初始化之前，并绑定同一个真实 peer

本轮明确核对了精确c09公开源码：C4的 `global_te.register_buffer()` 也是循环调用同一个 `register_memory()`，不存在另一套批量注册API，不能单独调用它消耗原C4的一次性注册标记。之前“位于register_kv_caches前”仍可能晚于Multi的Store兄弟对象初始化共享引擎/建立连接。

新修复在Mooncake worker构造完成后立即注册arena，先于Multi构造Store；复用同一bridge，原KV注册点仅幂等确认。针对hook首次安装在正在执行的原构造函数内的情况，也在facade的真实connector_worker赋值点确保提前注册。模型权重/真实layer rings在精确c09的connector构造前已加载，不推测层名或另建引擎。

**内网现在一次执行：**

1. P、D均拉取本分支，沿下方原manifest流程备份更新runtime，原参数启动新worker。保留attention/CANN/Cloud包、MXFP4、QLI、Graph/MTP/APC/并发。必须全新worker；旧HCCL连接不靠热拷贝文件刷新。
2. 每个实际worker日志应看到 `RECENT128_PD_EARLY_ARENA ... before_store_constructor=1`。沿原成功请求JSON/真实PD代理地址执行同一条curl，内网自行填实际变量，不让用户再转述路径：

   ```bash
   curl --fail-with-body -sS --connect-timeout 15 --max-time 180 \
     "$ACTUAL_PD_CHAT_COMPLETIONS_URL" \
     -H 'Content-Type: application/json' \
     --data-binary "@$ACTUAL_SMOKE_REQUEST_JSON"
   ```

3. 成功证据必须同时有：P `RECENT128_PD_SOURCE_READY ... register_ret=0`；D同一所选peer的 `RECENT128_PD_READ_BEGIN`，其native_rpc/source_rpc/remote端口一致；D `RECENT128_PD_READ_END ... ret=0`；最后 `Recent128 BF16 PD tail imported` 和正常curl输出。无KV-load-recompute、无sticky fault。不同TP peer的端口本来可不同；不能再把任意C4日志的15429直接与BF16所选peer20598比较。
4. 如仍失败，本轮新增日志一次定位到具体边界：`RECENT128_PD_ENGINE_MISMATCH` 是arena/worker/globalTE对象不同；`RECENT128_PD_RPC_MISMATCH` 是引擎实时端口与worker缓存不同；`RECENT128_PD_SOURCE_BINDING_MISMATCH` 是控制线程绑定错误；`RECENT128_PD_SELECTED_PEER_RPC_MISMATCH` 是同一peer的native缓存/控制回包/源实时端口不同。若这些全部通过、P注册ret0且描述符在arena内，RDMA仍报未注册，则保留同一请求上述SOURCE_READY/READ_BEGIN及backend首错，不能继续写“Python没调用P注册”——需据这些实际值定位部署后端的MR发布/连接状态。
5. 完整指针/端口/nonce日志仅留内网。推小型脱敏报告：commit、实际PID/rank、early_arena是否出现、同peer端口是否一致、P register结果、D read结果、import计数、首个错误标签/后端错误和本地产物位置。不要上传私有源码/大日志，也不要重算成功当PD通过。

本轮208项框架检查＋23项传输/arena宿主检查通过，包含构造中途安装、提前注册一次、OFF不分配和同peer路由对照。A5成功与性能仍待本次真实curl，不能保证此候选已在设备解决全部问题。下面各节仅保留此前排障背景。

## 最新 e01b953：修复运行中新注册区域与多接收者释放，先测两条 P/D 请求

最新错误已到真实RDMA read；`13312`字节正是13token的BF16 latent。D本地注册地址与P的源地址不同是正常现象，不能据此断言P漏注册。旧代码先对每个源clone注册并检查ret0再发布READY；但公开旧HCCL后端在首次peer连接时交换/缓存内存描述符，并要求注册区2MiB对齐，逐请求注册新clone仍可能不被已有连接识别。部署二进制根因尚未实证钉死，本轮消除此依赖。

**现在直接执行本节，不再使用逐请求clone注册的59e0df9版本：**

1. P、D都拉取本分支，沿下方原manifest备份/更新runtime流程部署。两边必须使用新版本、启动新worker：arena必须在原native KV注册/线程/peer连接之前注册；只把文件拷进已建链旧进程不能生效。保持原CANN/算子库/权重和生产参数，不重编attention、不重置NPU、不关闭APC/Graph/MTP。
2. 启动日志应出现 `Recent128 BF16 PD arena registered before native KV startup`，包含bytes/slots/layers。P和D各有一个持久、有界、起点和长度都按2MiB对齐的arena；原KV缓存dict和Store兄弟分支不变。请求payload地址全部来自该已注册区域，不再逐请求register/unregister。
3. 先用本次失败的短prompt跑原P→D curl，再用另一个不同前缀短prompt跑第二条，让两次传输复用已建立的连接。第二条验证固定注册区和槽位复用；APC命中路径随后单独验收，APC开关保持原值。
4. **直接传输通过才算本轮通过**：D日志应有 `Recent128 BF16 PD tail imported`；实际worker的`recent128_pd`摘要应 `successful_reads>0`、`completed_imports>0`、`failed_reads=0`，两次curl正常完成且无sticky faults。现有RPC探针已能输出这些不含指针的计数；无RPC就看本轮实际worker导入日志和异常。`Recovered from KV load failure` 后重算成功不能记作PD修复成功。
5. 只推小报告：commit、实际P/D PID、arena bytes/slots/layers、两次是否直接导入、RDMA read/import/failure计数和首错。指针/nonce/descriptor/完整配置和大日志留内网，不让用户继续搬材料。

这次也修了多D共享源的问题：D以真实engine/handshake/request身份订阅，精确ACK只完成该接收者；源必须同时等原native传输完成及全部已发descriptor的接收者ACK后才能复用槽。不能第一个ACK到就注销所有读者的内存。arena注册保留至进程结束，缓冲槽位有界；失败的RDMA目的槽保留防止仍在进行的写与复用冲突，不以无限分配掩盖失败。

本轮202项框架＋23项传输/arena宿主检查通过，未在A5验证。新增的是请求传输边界的缓冲管理，不改变attention计算；额外持久arena的实际显存由启动日志报告，P/D延迟仍需实测。

公开源码依据：[HCCL transport](https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-transfer-engine/src/transport/ascend_transport/hccl_transport/hccl_transport.cpp)、[连接与内存描述符交换](https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-transfer-engine/src/transport/ascend_transport/hccl_transport/ascend_transport_c/hccl_transport_mem_c.cpp)、[Ascend transport文档](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/transport/ascend_transport.md)。这支持修复方向，不是部署backend已确认的证明。

下面0d9421f章节保留原部署命令与首轮接线背景；以本节为当前任务。

## 最新 0d9421f：已提供真实 legacy 子通道接线候选，直接部署验证

两节点的 AscendMultiConnector 源码 SHA 已与公开 c09d3e20 完全匹配。外壳下真实 MooncakeConnectorV1 子对象才是传输位置，Store兄弟分支保持原流程。本轮新增 `pd_connector.py`、`pd_mooncake.py` 并接入 config初始化和runner释放前边界，**不再停留在身份采集**。

内网 agent 现在直接执行：

1. 拉取本分支，冻结现有成功P/D启动命令和runtime备份；不要重编/替换attention库、CANN、Cloud包、MXFP4权重。两边沿既有overlay更新流程同步runtime，新进程才能加载新的config和connector hook。实际checkout自行定位，备份输出用新目录：

   ```bash
   python framework_integration/install_overlay.py \
     --mxllm-root "$ACTUAL_MXLLM_CHECKOUT" \
     --apply --update-runtime --replace-edited-runtime \
     --output "$NEW_BACKUP_DIR"
   ```

   这是受manifest约束的runtime更新；不能直接覆盖私有源码或删掉guard。若原三处source已变且manifest不匹配，保留本地改动与差异，不能强行覆盖。本轮共195项框架宿主检查和18项传输核心检查通过，不等于A5已通过。
2. 在原候选P、D环境以原启动命令启动新worker，原并发/Graph/MTP/APC/P-D/QLI参数全部保留。新增接线随 `sfa_recent128` serving + 原 `kv_transfer_config` 生效，无新端口，无额外功能开关；两边都必须更新。它通过真实Multi children递归定位legacy，不替换Store或修改原C4缓存张量。
3. 服务ready后先跑原先成功的同一条单curl，走实际P→D代理。预期D不再报`Nonzero request start ... import_tail first`。查实际P/D worker的异常和reader sticky faults，确认完整输出；再补一个短prompt，不能用只启动成功替代推理。暂不混跑性能/GPQA或扩展扫描。
4. 有现成worker RPC时调用下方`verify_worker_recent128`获取当前PID/真实子通道/fault；没有RPC就以本轮真实worker日志和请求输出为证，不用旧PID文件判断。上传小型结果：本commit、实际P/D worker PID、请求是否完成、首个异常（如有）与本地日志位置。不要再让用户转述身份或传私有源。

接线实际顺序：scheduler的原`request_finished`给出已物化prompt长度→同轮metadata带`recent128_sends`→runner在原finished bank释放前固定真实BF16尾窗→沿既有GET_META/控制socket发布独立注册快照→D在每个C4 shard DONE之前固定源descriptor→收齐C4后导入BF16→精确nonce ACK→原receive-done。C4全命中而native仍有请求时也会先获取tail，不靠非空block列表决定。

**明确边界：**这是首次真实PD设备候选，尚无A5成功结果。真实尾窗层集/position/epoch不全仍会拒绝，不伪造MTP层或反量化C4。当前为保证跨线程/RDMA可见性，在请求传输边界做真实设备同步和tag校验；不在每层/Graph replay加同步，但P/D传输开销尚未实测，不能宣称性能零代价。此接线没有同时实现Store/APC任意共享前缀h的高精度持久化；那些缺口仍沿适配清单处理，不关APC来冒充通过。不同PP层集的自动拼接尚未实现，真实descriptor必须覆盖本地完整层集。

以下身份采集/接口合同是背景；当前不重复执行已完成的factory身份采集。

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
