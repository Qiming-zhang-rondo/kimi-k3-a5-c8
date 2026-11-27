# V4 KV-split：真实代码已交付，执行构建和A5单算子验证

**当前任务已转到 [KV_SPLIT_FAIR_AB_TASK.md](KV_SPLIT_FAIR_AB_TASK.md)**：用户要求核实50us与80us，
复用cfdb6c3所述已成功库，运行`--verify-comparison`五臂六组同轮20轮配对。
下面保留构建/修复历史与合同，不是本轮要求重编设备库。

用户要在保留MXFP4主缓存+BF16 recent128格式的前提下，尝试沿KV分片提高Q1/Q4核利用率。
本次不是参数建议；已新增候选attention入口、FP32 partial写出、归约kernel、host/bridge与runner。
**尚未在外部Mac上做CANN编译/NPU运行，不预报40–50us。**

## 当前优先：验证归约partialUb反向同步修复（8791d2e之后）

本轮完整原始证据已到位，不再要求用户搬运日志，也不再停留在上一轮只读取证任务。
`evidence/kv_split_v3_w{0,128}_split1_raw.json`两份由工具直接生成：

| window | partial0 vs matching math L2 | final vs BF16(partial0) L2 |
| --- | --- | --- |
| W0 | 0.000938923 | 1.159897 |
| W128 | 0.0009372584 | 1.140965 |

zero-query最后query的final_vs_partial为0，前三query约0.5–0.7；partial统计量和计数正常。
这将当前S1大错缩小到归约/传输，而非producer计算。S2/S4仍须后续门禁确认，不能据S1替它们报通过。

已发现并修正的代码缺陷：`kv_split_reduce.asc::Run`循环复用单份partialUb，原来只有
MTE2_V正向依赖、MTE3_V输出复用依赖，却没有V_MTE2保护下一轮搬入。MTE3_V只阻塞Vector，
不能阻止MTE2提前覆盖上一轮VF仍读取的partialUb。每轮最开头新增一对V_MTE2 SetFlag/WaitFlag，
已有S_MTE2/MTE2_S/MTE2_V/V_MTE3/MTE3_V保留；数学、producer、ABI、形状和阈值全部不变。
该缺口与现象吻合，**修复尚未在NPU上实证，不宣称已解决全部失败**。
官方反向依赖说明：
https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_00019.html

内网agent现在自主执行：

1. 记录旧候选库SHA（已有失败库为`7d6312e4bf556951dc63f89c6b5b7d739b3f18f89a4a76b25ea588486e35d4b1`），
   保留旧产物；用原成功的SDK/隔离build目录增量构建，仅归约ASC发生设备源码变化。不得覆盖原82us V4验证库。
2. 新Python进程、新output，复用上次实际runner命令，Q4/W0,W128/S1,S2,S4先`--timing none`。
   不升级依赖、不改ABI/tiling或算法，不跳过S1；所有math/partial/zero/graph控制必须通过。
3. 若全绿，按下文原命令做graph 10轮同库V4配对，计时包含attention+reduce；保存所有原始轮次。
4. 若仍失败，新runner自动生成`failure_small_w0_split1.json`和`failure_small_w128_split1.json`。
   直接提交这两份，加实际库SHA/构建命令；不要只上传手工摘要。若S1通过但S2/S4失败，
   用collector按实际失败case导出小JSON。设备故障则按既有规则停止，不能重置卡或停服务。
5. 提交小型构建证据和完整正确性/配对计时JSON；大npz只留内网。只有实际门禁通过才能报修复成功。

## 已完成的历史修复与证据

- 1061f674：host补认native空片max精确`-FLT_MAX`（`0xff7fffff`）；仅sum=0且O=0才忽略该max。任意有限负数/NaN/+inf仍不接受。
- 2b96044：归约统计量改为MTE2搬入UB，再等MTE2_S读Scalar，避免动态GM的GetValue DCache风险；这次修复未消除NPU大误差。
- 90be220：失败时从已保存JSON/NPZ自动导出每份<=50KB的只读诊断，不导Torch、不访问NPU，不改变失败状态。
- 8791d2e：完成W0/W128 S1原始取证。该轮不重编/不重跑的任务已经结束；当前按顶部同步修复任务构建验证。

需要额外case时使用现有CPU工具，不能手工改写或四舍五入原始失败指标：

```bash
python experimental/single_pass/collect_kv_split_failure.py /actual/failed-run/results.json \
  --case w0_split1 > experimental/single_pass/evidence/kv_split_failure_details.json
```

工具缺少实际数组时明确NOT_AVAILABLE。source SHA、真实库SHA和失败状态必须保留；大npz留内网。
本任务不实施用户规划的两套attention后端或DSL缓存布局迁移。

## 合同和隔离

- 现有原V4/原C4各入口保留。新导出`QsfaSinglePassLaunchKvSplit`使用独立**ABI2/112B tiling**；旧ABI1仍72B，15指针描述符不变。
- 新入口沿用V4 load rows32/input416/input2/output1；attention tile仍128。TopK2048按**列**连续切S=1/2/4，不改token ID、不排序、不额外冷热分区；冷400B、BF16 latent512+RoPE64热环均不改。
- task=query*S+split，launch Q*S AIC，最多16；先核对实际device cube_core_num。Q4/S4为16个attention任务。
- workspace为FP32 `O[Q,S,64,512]` normalized partial、max/sum各`[Q,S,64]`，偏移在新POD明确定义；Q4/S4约2.01MiB。新bridge实际分配，不能用旧1B占位。
- **一次并行attention＋一次纯AIV归约**，同stream两次launch；不调用第二次attention。计时必须包含归约。首版不引入全核spin/barrier。
- 部分O保持FP32，按max/sum稳定合并，最终只舍入一次BF16（attention内部概率仍沿原native BF16流水）。empty片sum0时归约不读其max/O；native empty max0与数学-inf均须正确处理。
- empty行清零前不能先改max sentinel：新分支在Vec2输出之后才copy/canonicalize统计量。新回归已覆盖该顺序，旧V4路径不变。
- 原生GM页表、R64/packed400、B1/H64、TopK2048、W0/W128限制保留；不是NoPE272或生产框架交付。

## 1. 隔离构建，复用已成功工具链

先拉当前分支，使用原82us V4成功的同一CANN/ASC/编译器/依赖路径，不source另一版vendor。
选择新的构建目录，不覆盖`/tmp/qsfa-load-tiles-build`已有库：

```bash
cmake -S experimental/single_pass -B /tmp/qsfa-kv-split-build \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510 \
  -DSFA_SDK_ROOT="$ASCEND_HOME_PATH" \
  -DQSFA_BUILD_KV_SPLIT=ON
cmake --build /tmp/qsfa-kv-split-build -j8
```

ASC_DIR/CMAKE_PREFIX_PATH如现场需要，从既有成功CMakeCache读取同一SDK的实际值。普通头文件、函数签名、编译报错由内网agent就地修复并提交diff/原始错误，不逐项让用户中转，不伪造架构宏。old control可用`-DQSFA_BUILD_KV_SPLIT=OFF`单独构建诊断，但不能称候选成功。

复用原成功环境中的ninja PATH；若ninja仅在`/tmp/cloudops/bin`，只补这个工具目录，不加载旧vendor。Torch bridge使用单独`TORCH_EXTENSIONS_DIR=/tmp/qsfa-kv-split-torch-ext`，不要复用服务中的JIT产物或清空既有缓存。

记录新库SHA/编译器/SDK真实路径/arch/实际编译命令，以及`QsfaSinglePassKvSplitAbiVersion=2`、tiling112B导出；bridge实际配置须显示候选存在。候选的Cube/Vector/kernel/三个MXFP4 helper符号均独立命名，勿删这些隔离宏。

## 2. 先Q4的完整数值控制，再性能

自主定位原实验已校验的hotkv payload目录（含test_kv_hot_fused_a5.py等）；`pinned_payload.py`会校验，不重新下载/替换成其它版本。用确认空闲的卡；下面npu:0仅示例，不可停止业务。

```bash
python experimental/single_pass/run_kv_split.py \
  --payload-dir /actual/verified/hotkv \
  --library /tmp/qsfa-kv-split-build/libqsfa_recent128_direct.so \
  --device npu:0 --queries 4 --keys 8192 \
  --windows 0 128 --splits 1 2 4 \
  --timing none --output /tmp/qsfa-kv-split-q4-check
```

默认同时保留：W0原生冷入口（旧ABI load0）/W128原V4（load4），以及各自新S1/S2/S4。S1同样走partial+reduce，用来量化新路径/归约税，不能把S1当旧V4。

必须全过：

- 对实际packed C4和原始BF16热环做完整FP64数学参考，全局/最差query-head<=.01、全finite；不要要求不同分片的BF16概率舍入逐位相等。
- 从真实device partial FP32 O/max/sum独立CPU归约，核对device最终输出。
- zero-Q全2048计数、仅257前缀有效+尾部-1、split尾片空态；检查每片sum准确、active max0、empty输出0且无毒化残留。
- graph测试时将final和partial缓冲投毒后replay，确认两次launch都捕获并覆写。

全过才另用新输出目录运行：

```bash
python experimental/single_pass/run_kv_split.py \
  --payload-dir /actual/verified/hotkv \
  --library /tmp/qsfa-kv-split-build/libqsfa_recent128_direct.so \
  --device npu:0 --queries 4 --keys 8192 \
  --windows 0 128 --splits 1 2 4 \
  --timing graph --warmup 20 --iterations 100 --repeats 10 \
  --output /tmp/qsfa-kv-split-q4-perf
```

有`DEVICE_FAULT.json`则立即停止设备运行，保留首错，不复位/重试。数值失败先定位；不删门禁、不放宽阈值、不拿只计attention的时间当整体候选。

Q4结果回报优先，不等待全扫。通过后可补Q1/K8192、Q4/K32768；保持同库、同输入、同stats开关，W0与W128一起比较。`--return-stats`可独立数值诊断，默认性能关闭，不能跨开关排名。

## 3. 提交小型原始证据

在`experimental/single_pass/evidence/`新增JSON；不提交大npz、SDK或wheel。包括：

- source commit、bridge/library SHA与ABI2/112B、CANN版本及编译真实路径、实际可用核数、Q*S任务数；不要把硬件上限当active核的实测。
- 所有fullmath/partial/zero/graph门禁；失败原始误差照常保留。
- 6个候选W0/W128×S1/S2/S4的完整10轮AB/BA，event/wall和median paired ratio[min,max]，对应同库baseline原始时间。
- 总时间明确含attention+reduce，prepared-read不含写cache/模型/TPOT。
- 从中选稳定赢家；若分片无收益如实报告，40–50us不是验收既定事实。

外部已完成宿主数学、真实源码函数体+UBSan、POD/指针偏移检查；这些不替代你这次的SDK/device验证。
