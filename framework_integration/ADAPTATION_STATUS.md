# 当前可运行版本之后的适配清单

依据当前公开runtime代码的只读检查。用户已报告修正布尔mask后真实推理成功；本轮保持该运行版本不变。
内网若另有未推送接线，以实际worker和请求证据核验，不把公开代码缺口扩大为所有私有部署都没实现。

| 优先级 | 项目 | 已有内容 | 尚需适配或确认 |
| --- | --- | --- | --- |
| P0：命中时会阻塞 | APC | 真实BF16快照导出/导入、epoch重绑 | 实际prefix命中边界配对、发布/恢复调用点、有界sidecar目录和淘汰；只有C4前缀不能直接恢复热历史 |
| P0：真实P/D时会阻塞 | P→D尾窗 | `snapshot_tail`/`receive_tail`及完成事件接口 | 实际BF16发送、接收、全层完成通知与D首读前导入；成功curl需区分P-only/重算和真实P→D |
| P1：发生抢占时 | 抢占/恢复 | request bank/epoch、结束释放 | 暂停时保存/释放策略、恢复到实际位置的重新绑定；活跃加暂停请求可能耗尽只按max_num_seqs分配的池 |
| P1：尾窗提交与导出 | MTP | corrected GPU positions、draft独立路由、因果/精确tag判定 | 真正接受/拒绝提交与尾窗发布接线，target/draft各层历史、拒绝后重试验收；API未调用不等于正常回退必错 |
| 已实现，需设备覆盖 | Graph | 固定计划、lane隔离、capture禁止新分配、设备valid更新 | 实际所有bucket/lane的预热和重放；Python调用次数不是Graph replay次数 |
| 已实现，需设备覆盖 | chunked prefill | 跨chunk持久ring、C按128+maxQ-1对齐 | maxQ覆盖真实每请求chunk；长chunk/跨环/混合prefill-decode验收。maxQ本身不改变scheduler分块 |

## APC需要补什么

入口是 `serving_runner.py:receive_tail/snapshot_tail`，公开仓库未找到生产cache-manager对它们的调用。
新请求初始位置非0时，`ServingState.bind_step`要求先恢复真实BF16尾窗；它不会从C4反量化代替。

1. 使用scheduler实际可复用边界h，不能直接套原请求完整长度L。末token重算、块对齐等都可能使h与L不同。
   h位置首个query需要此前最多127个token的真实BF16；保存128个尾token是足够的保守区间。
2. 按同一prefix身份/模型与精度配置，在h处保存配对的C4前缀和BF16 sidecar，覆盖所有本rank层。
   当前`export_tail`只能在已提交end导出最新128；没有任意历史边界目录。源请求继续推进后，旧边界数据可能被ring覆盖。
3. 新增受保留区间和position/epoch校验约束的中间边界导出，并接到真实执行完成后的prefix发布事件。
   不能仅放宽export_tail的end检查，也不能调用snapshot_tail就假定scheduler已经提交。
4. sidecar容量必须有界，带LRU/引用保护/C4淘汰联动，不能为所有C4缓存块无界保存BF16。
5. C4命中而BF16配对未命中时，本请求的联合命中应缩短到可用配对边界，再重算后缀；没有配对边界则本次从0重算。
   APC保持开启，不全局关闭。单独重算最后128也不自动成立，因为第一个重算query仍需要更早的BF16热历史。
6. 恢复/重新准入必须更新lease、bank/epoch及实际h。现有acquire遇已有请求直接返回，不能拿旧lease冒充已恢复。
   import保留无效源tag；`tail_ready=True`本身不是所有层/位置完整的证明，发布命中前需验证覆盖。

APC验收至少包含：重复提示、较短公共前缀、同前缀分叉、sidecar单独淘汰、C4块复用、抢占后恢复。
记录实际h和BF16覆盖范围；留内网，不上传prompt/激活/私有源。

## P/D需要补什么

P在真实提交边界保存原始BF16 latent/KR与tags，保持producer→snapshot顺序；快照内存保留到D确认或失败处理完成。
描述符需要关联请求、传输generation、源epoch、区间、层/rank和真实payload位置。不能只复用C4 block到达标记。
D收到实际字节后，建立本地stream完成顺序，导入所有本地层并重绑bank/epoch，再允许非零起点首读。
取消、超时、重复/旧描述符、源bank复用都需有明确生命周期。

C4全prefix命中而不传主缓存时，也仍可能需要独立恢复BF16尾窗。没有C4流量不代表可以跳过tail恢复。
先核实当前内网实际connector是否另有这些接线；已有则验证并复用，不重新造协议。

## 抢占、MTP、Graph和chunk的边界

当前runner只处理finished_req_ids释放，未对抢占/恢复事件接线。请求暂时离开batch仍持有bank，不能无条件视作结束。
恢复时若位置回退，需按真实恢复边界重建/导入；同ID允许回退本身不等于错误。

MTP的begin/resolve/commit接口尚未接到完整生产提交链；当前正确的GPU位置和因果判定不应被撤回或重写。
主要补提交/快照的正确时机及target/draft历史完整性，之后验收接受、拒绝和重试。

Graph下Python producer_calls/read_calls通常只覆盖eager/预热/捕获，不会每次replay递增。
不能要求每个Graph请求的Python计数差为正，也不能把预热计数当作真实请求证明。结合固定捕获配置、实际请求设备状态与sticky faults验收，必要时在计时外单请求trace确认writer/V4。

普通连续chunk不需要每次导入tail或清空ring；新APC/PD起点和普通续chunk是不同生命周期。

## 测试阶段的原则

先保持当前成功实现，用现有服务做一次冷请求、一次真实APC命中、一次真实P→D的最小确认。
哪个功能在实际内网已有完整接线，就记录证据继续；哪个确认缺代码，就在独立候选修复，不能中途替换实现后拼接同一份GPQA/性能结果。
不改已成功的attention kernel，不重开DSL/量化调参/性能摸高支线，不关闭原生产功能掩盖缺口。
