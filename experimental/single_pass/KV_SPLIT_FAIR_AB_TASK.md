# 公平核对“新S4约50µs”与“原生C4约80µs”

本轮只用已经通过数值控制的现有库做同轮单算子对照，不修改kernel，不重新编译，不换CANN、vendor或wheel，不接入模型服务。不能把历史80µs与另一次运行的50µs直接相除。目标是确认差距是否在相同输入、相同计时范围下复现，并区分分K收益和热窗代价。

## 五条真实路径

全部使用同一份query、原始TopK、packed400冷cache、物理页表、长度、scale与原始BF16热环。不得重新随机生成某一臂输入，不做排序、增删TopK或逐臂改量化参数。

| 标记 | 路径 | window | 实际读取调用 |
| --- | --- | --- | --- |
| A | 当前环境已安装的原生C4单路算子 | 0 | 已验证的installed native C4入口 |
| B | 同一候选库中的原始C4对照 | 0 | `QsfaSinglePassLaunchCold`，旧ABI1 |
| C | 同库原V4热窗 | 128 | `QsfaSinglePassLaunchLoadTile(...,4)` |
| D | 新KV-split冷读 | 0 | `QsfaSinglePassLaunchKvSplit`，S4，包含设备归约 |
| E | 新KV-split热窗 | 128 | 同一新入口，S4，包含设备归约 |

A必须真正调用installed native C4，记录其Python/C++入口、实现文件身份及配套vendor；不能将B重复命名成A。若当前安装入口不能读取同一份packed400输入，先报告该臂的实际不兼容，不能暗中换成别的cache精度或别的算子。

固定形状：B1、Q4、K8192、H64、latent512＋RoPE64、TopK2048、page128、seed20261120。同库原C4及V4明确使用4个AIC；installed C4保留其原生tiling调度，若无法读取实际分核信息就明确标记未知，不声称强制为4核。新S4使用Q×S=16个attention任务并执行一次纯AIV归约。这是本轮明确比较的调度差别，不把任务数写成profiler实测active核数。所有臂关闭返回最终max/sum的性能选项，保留候选内部必需的partial统计量。

## 先确认运行环境与原成功产物

内网agent自主定位上轮成功的payload与库，读取现有构建记录及库SHA，复用该产物。只在已确认空闲的同一张A5卡运行；若有业务占卡，不能擅自停止服务。记录实验前后设备/进程状态，不能在有干扰的卡上用阶段时间推断公平结果。

- 保留实际source commit、库完整SHA256、bridge身份、ABI2/112B、原编译器和SDK真实路径、CANN/torch/torch_npu及设备型号。
- loader记录的库SHA必须与实际传入文件一致。不要替换或覆盖已成功的库，也不要在同一Python进程里换库。
- 沿用已有PATH和匹配环境；若ninja仅在`/tmp/cloudops/bin`，只补工具PATH，不source另一版vendor。使用隔离的Torch extension缓存，不能清空服务缓存。
- 输入必须由runner生成一次并被五臂共享；保存输入身份/形状与原TopK、热环tag校验结果。合成输入可以提交，权重和真实用户数据不属于本轮。

## 命令：同一进程先过正确性门禁，再做20轮直接配对

拉取含本任务和`--verify-comparison`实现的提交。由agent将下列变量填成现场真实值，不逐项让用户转发环境信息。`npu:0`仅是写法示例，必须选择已确认空闲的实际卡。

```bash
VERIFY_LIBRARY=/actual/verified/libqsfa_recent128_direct.so
VERIFY_PAYLOAD=/actual/verified/hotkv
VERIFY_DEVICE=npu:0
VERIFY_RUN=$(mktemp -d /tmp/qsfa-fair-ab.XXXXXX)

python experimental/single_pass/run_kv_split.py \
  --verify-comparison \
  --payload-dir "$VERIFY_PAYLOAD" --library "$VERIFY_LIBRARY" \
  --device "$VERIFY_DEVICE" --queries 4 --keys 8192 --aic-count 4 \
  --windows 0 128 --splits 4 --seed 20261120 \
  --timing graph --warmup 20 --iterations 100 --repeats 20 \
  --output "$VERIFY_RUN/graph"
```

该命令内部先做五臂正确性及capture门禁，任何一项失败均不得进入性能测量；计时后还须再次核验，不能使用上一进程的PASS代替。`--verify-comparison`只用于graph/eager、至少20轮，必须同时包含W0/W128及S4，且性能关闭`--return-stats`。本轮无需重扫S1/S2，也无需重编内核。设备fault时立即停止，保留首错和marker；不得复位设备或绕过门禁继续计时。

每一对都做20轮AB/BA交替，保留每轮baseline/candidate原始event和wall时间及两种ratio。至少包含下列**直接配对**，不能由各臂独立中位数代替：

| 直接配对 | 要回答的问题 |
| --- | --- |
| A ↔ E | 新热窗完整读取是否真能比当前installed纯C4更快？ |
| A ↔ B | 旧80µs基线与同库原C4是否在本轮对齐？ |
| A ↔ D | 新S4纯冷读相对installed C4的净收益是多少？ |
| B ↔ D | 相同冷cache，分K及设备归约后的净收益是多少？ |
| C ↔ E | 新S4热窗是否超过同轮已验证V4？ |
| D ↔ E | 同一新S4实现启用recent128的实际代价是多少？ |

六组直接配对保存在`verification_timings`，每组明确baseline/candidate并保留完整rounds；runner原有`timings`也保留。显式记录每对的实际入口，避免跨window时标签互换。

## 计时范围与数值门禁

主结果为graph replay下的prepared read；所有输入、输出和workspace预先准备。D/E的计时必须同时包括attention和归约两次launch，不能只计producer。A/B/C同样计完整读取调用，不能一侧混入metadata准备、分配、H2D拷贝或JIT。原生接口不能预分配的开销应记录实际情况，不能隐瞒到不同scope。

本轮只读固定cache，计时不含写KV、热环写入、TopK生成、索引准备、metadata准备、CPU参考或模型服务；不能把单算子结果称为TPOT收益。所有图每次replay只包含该臂一次完整读取；iterations表示重复replay次数，不能一臂捕获100次而另一臂只捕获1次。

必须保留并通过：

- 每臂对同一原始输入的完整独立数学参考，W0使用实际C4反量化值，W128对同一TopK内命中的token使用原始BF16值。全局和最差query-head均沿原门槛，不能放宽阈值。
- 新候选真实FP32 partial O/max/sum的CPU稳定归约与最终设备输出一致；空片、有效前缀计数及native有限哨兵合同保持现有严格门禁。
- graph replay前投毒final/partial，确认候选attention和归约都被捕获并覆写；installed与原V4也做其输出回放控制。
- 所有臂计时后直接读取最后timed graph/eager留下的输出与partial缓冲区，对照完整数学参考，结果保存在`post_timing_controls`及本机`post_timing_outputs.npz`。**不得先调用算子刷新输出再验证**；installed使用closure保存capture返回的实际tensor。排除捕获/复用造成的漂移、输出未刷新和串臂覆盖。

如graph的排名或scope有疑点，可在确认仍空闲的同一卡补一轮eager，使用同一五臂和20轮配对，仅将`--timing graph`改成`--timing eager`、输出改为`"$VERIFY_RUN/eager"`。eager数据单独保存，不与graph拼成更好看的成绩；无需因此增加额外形状或优化变体。

## 交付完整原始证据

提交到`experimental/single_pass/evidence/kv_split_fair_ab/`：

1. graph的完整`results.json`，包括所有正确性、graph、post-timing结果和六对各20轮event/wall原始数据；可附eager完整结果。
2. 实际命令、source commit、输入身份、库及bridge SHA、ABI、环境、设备空闲证据、计时scope和真实入口映射。
3. 原始日志；若出现失败，保留完整失败状态、已生成的小诊断、partial证据及其来源SHA，不能只上传最终一句结论。
4. 简短结果表作为附加摘要：五臂同轮时间、各直接配对median ratio及[min,max]、快于对照的轮数。不要用该表替代完整JSON，也不要再次只交42行手写摘要。

若JSON过大无法单文件提交，可以gzip无损压缩，或按256KiB无损分块；同时提交manifest，包含原文件名、完整字节数、SHA256，以及压缩件或各块的SHA256与分块顺序，确保外部可以逐字节恢复原JSON。不要删rounds、四舍五入原值或把失败臂过滤掉。大npz可以留内网并记录路径/大小/SHA，不能把“本地有文件”当成已经上传。

本轮通过的含义仅是：在上述固定输入和完整读取scope下，约50µs与约80µs的差距得到或未得到同轮复现。若installed、同库或跨window配对不支持原结论，如实报告，不以历史数据补齐胜负。
