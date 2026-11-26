# V4 KV-split：真实代码已交付，执行构建和A5单算子验证

用户要在保留MXFP4主缓存+BF16 recent128格式的前提下，尝试沿KV分片提高Q1/Q4核利用率。
本次不是参数建议；已新增候选attention入口、FP32 partial写出、归约kernel、host/bridge与runner。
**尚未在外部Mac上做CANN编译/NPU运行，不预报40–50us。**

## 当前优先：复用229d296已编译库，验证空片哨兵兼容修复

229d296反馈A5编译已成功，运行停在zero-query padded257的host空片max检查。
源码的`NEGATIVE_MIN_VALUE_FP32=0xFF7FFFFF`是有限`-FLT_MAX`；`ZeroInvalidLseVF`
空片写sum0/max=-FLT_MAX，旧校验仅认0/-inf，属于host合同遗漏。

本次只修改Python校验/诊断，不修改设备kernel、ABI或桥接。**直接复用刚编译成功的同一.so，先不要重编。**
拉最新源码，沿原实际命令使用全新output目录先`--timing none`。合法空态仅限O严格0/sum0，
max为0、-inf或精确位模式0xff7fffff；任意负值/NaN/+inf仍拒绝。active finite、zero计数和所有误差阈值不变。
新runner在每次partial门禁前保存本机diagnostic npz及空max位模式直方图小JSON，即使失败也保留证据。

全部数值控制通过后再跑原Q4 graph矩阵。若仍失败，自主从新诊断导出原始位模式、counts、
对应case和误差到小JSON，连同实际库SHA提交；不要只发截断traceback，不上传大npz，不删门禁。
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
