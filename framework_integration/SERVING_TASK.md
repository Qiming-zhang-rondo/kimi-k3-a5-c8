# 直接修复 a1ad5cd 的生产接入

## 当前入口 52db594：修复Python参数转发，不重编内核

本次首错是`cloud_mla_prolog_v3_recent128() got multiple values for argument 'rope_sin'`，
发生在Python绑定参数阶段，还没进入这个provider的计算。报告的“原调用30多个位置参数和关键字重复”与已审阅代码不符：
上游Torch调用全按名称，MXLLM Cloud适配器只把以下9项取为位置参数：

```python
("token_x", "weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr",
 "rmsnorm_gamma_cq", "rmsnorm_gamma_ckv", "kv_cache", "kr_cache")
```

`rope_sin/rope_cos`留在kwargs，不在这9项中。若新wrapper第8/9位置采用Torch的rope顺序，就会出现本次冲突；
私有wrapper内部适配或部署源码漂移也需在内网核实，不能仅按报告断言是哪一层。

公共修复已在serving route提供完整的原9项参数名称，`call_prolog_with_ring`据此命名后
**只调用`provider(**完整命名参数)`**。新wrapper的物理形参顺序不再决定这9项的含义。
它保留所有原kwargs及显式None，不根据新wrapper顺序重新猜映射，也不做signature反射/设备操作。
重复原参数明确拒绝，没有“关键字优先”“None当缺省”的规则；OFF仍调用原MXLLM分支。

内网一次执行：

1. 在现有隔离框架安装中同步本轮`prolog_provider.py`和`serving_prolog.py`，确认工作进程实际import该路径，启动新worker。
   使用既有overlay安装流程；不升级SDK、不改成功的TBE/内核、不重编reader。
2. 保留已安装私有provider，但撤销临时“按30多个位置推测/用None过滤/重复键覆盖”的merge补丁。
   新入口接收原参数的同名keyword和ring keyword；若保留`*args, **kwargs`，本轮路由传入args为空。
   内部继续按真实绑定/TBE声明转发，显式None原样保留；不能静默过滤未知字段或给旧op贴ABI标记。
3. 先用实际wrapper做一次本地仅Python参数绑定对照（不launch、不上传私有签名）：
   `kv_cache/kr_cache`与`rope_sin/rope_cos`身份对应原字段，原映射不被修改，ring绑定完整。
   然后继续原producer OFF/ON控制与服务启动，记录实际首错；不重复已经完成的baseline构建。

这次修改仅Python调用约定，无新增kernel、拷贝、设备同步或图内tensor分配；不作为完整服务性能已验收的证明。
下面的安装与producer任务继续有效，旧章节只作背景。

## 当前入口 fd7a60d6：沿用两节点已安装的variant，完成真实调用对照

最新提交只新增了汇总报告：称178/209均已安装variant，Python函数ABI=1已确认，TBE将在首次调用动态编译。
这推进了安装状态，但尚无真实producer执行或服务启动结果；报告中的“全部解决”不能替代这些结果。
**不重做已经成功的SDK配置/standalone baseline，不重编已通过的reader，也不另搭第二套bridge。**
沿当前已安装wrapper和私有variant，一次执行：

1. 在内网本地核实真实调用包含latent、post-RoPE KR64及position/epoch发布，而不是按成员名grep次数判定。
   可先运行下面的只读工具；它屏蔽注释/字符串，仅报告调用位置，报告留内网，不要求上传私有源或报告。

   ```bash
   python tools/audit_prolog_writer_calls.py \
     --source-root "$PROLOG_VARIANT_SOURCE" \
     --output "/tmp/prolog-writer-audit-$(date +%Y%m%d-%H%M%S)"
   ```

   有多组writer时追加`--writer-name`，指明各个真实变量/指针别名；工具不能证明模板被实例化或分支可达。
   `WriteLatentSlice`、`WriteRopeSlice`、`PublishTags`都要真正执行；核实源与目标行/列偏移、唯一tag owner和同步。
   若使用等价自定义函数，由内网在实际分支核实对应副作用，不能为了通过工具添加无效调用。
2. 在实际wrapper核实原9参/5项返回与新增8地址+4scalar逐项转发，原tiler/workspace/blockDim及mixed任务保持匹配。
   函数本身必须有`cloud_mla_prolog_v3_recent128.recent128_abi_version = 1`；模块变量或仅能import不证明真实写环。
   当前首次服务调用仍会走TBE编译，因此standalone编译成功不能当作TBE实际分支已成功。
3. 复用现有producer小测入口，一次完成OFF/ON、B1Q1/B1Q4/B8Q4、回绕/padding/重排、动态Graph输入。
   OFF/ON的原query与C4 packed/scale逐字节不变；ON的真实BF16 latent/KR及position/epoch正确。
   读环在整个producer完成之后；不通过独立writer或旧Cloud入口冒充这次variant测试。
4. producer对照通过后，继续已有服务启动任务，保留并发8、Graph、MTP、APC和P/D原配置。
   APC和P/D真实BF16尾窗回填仍须接通，下方能力矩阵没有因安装报告而自动变绿；不静默关闭特性或退回C4。

后续只修真实首错，并保留已通过产物；不要求用户继续搬运私有文件。下面的旧提交章节是历史依据，不是重新构建指令。

## 最新 d0190621：variant报告编译成功，先核实完整副作用再执行设备矩阵

新的私有报告说明 Writer variant clean rebuild 成功，追加了8个GM地址和4个scalar；但其调用证据只列
declaration/setter/null-check/ResolveRow/WriteLatentSlice，尚未列出实际RoPE写入和tag发布。
**这不是已经证明它们缺失，而是报告不足以确认完整producer。** 内网在本地一次核实并补齐以下已知检查，再运行NPU；不要求上传私有实现：

1. latent使用全局token行、正确列切片和实际最终QuantPerTileMxFp4输入，原FP32不改。
   若isPertile分支内在tap后还有scale/clip/rotation，tap必须移到这些处理之后、实际量化调用之前。
2. 真实post-RoPE KR64有WriteRopeSlice等实际写入调用；position/epoch有唯一owner发布。
   不能只有8个地址在签名中却没有真实KR/tag副作用。reader等整个producer完成后再启动。
3. SetRingWriter之前Init结果有效，Process之后Release；事件用SDK的TEventID。
   核对原Prolog是否有未登记的硬编码event ID，不能假定AllocEventID自动避开它们；也核对原UB预算加1216B后的总量。
4. 使用独立variant产物名称、真实文件SHA和对应mode4实例，避免复用相同libprolog_direct.so名字却在进程中加载旧baseline。
   只看到LaunchAscendKernel等通用基础设施符号或源码调用次数，不足以证明真实新入口已可调用。
5. 设备测试前必须已有真正的host测试入口/薄桥，传入完整原参数+新增12参数、真实tiling、blockDim和workspace，
   并与实际kernel签名逐项一致。不能先调用旧Cloud接口测一遍，再宣称测到了新variant。
   这个入口不必先完成正式wheel发布，但必须真的启动新kernel。

以上静态接线齐全后，一次完成原已约定OFF/ON+B1/B8+Graph producer控制：原query与C4 packed/scale逐字节不变，
真实BF16 latent/KR和位置/epoch正确，padding/回绕/重排有效。不是重新测试已通过的独立reader。
正式注册为cloud_mla_prolog_v3_recent128后再接服务；APC/P-D自动尾窗和完整MTP仍是已知单独接线项。

## 最新 be040494 / 4940380a：baseline已生成设备代码，冻结后接Writer

本轮报告未接热环的 Prolog baseline 编译成功：库约547KB、设备代码段约373KB。
先保留该成功副本、同一SDK和全部已通过编译参数；从它复制一个variant，只增加已经交付的inline Writer。
不要重新staging回旧源码或丢弃刚修好的LoadData API/模板常量，不重编已通过的reader。
baseline构建成功不等于Writer variant或真实producer已运行通过，也不能单凭launch基础设施符号判断mode4实例完全正确。

已知SDK差异和完整定义/路径核验见 [SDK_COMPATIBILITY.md](../experimental/prolog_inplace_build/SDK_COMPATIBILITY.md)。
两处具体差异已列出：版本宏未定义误选旧LoadData分支、实际9.2.1模板1:1常量值4。
不要猜造ASC_DEVKIT_VERSION_NUM来强迫分支；沿当前已验证的新API和真实常量定义继续。

## 最新 562ca1ca：先修 CPU baseline 编译的两个问题

现在已定位为 host 模板注册混入 direct device target，以及框架 GET_TILING_DATA_WITH_STRUCT 未在直调工程生成。
内网直接执行 [BASELINE_COMPILE_FIX.md](../experimental/prolog_inplace_build/BASELINE_COMPILE_FIX.md)：
工具在本地私有副本隔离注册块，并用真实 device POD 的显式加载替换宏调用；保留原SDK、混合任务、模板实例和tiling/launch合同。
不要修改SDK，不再加无关include或空宏，不在这阶段加载模型/执行NPU。

## 最新 f23f328 / a139cb79：reader通过，集中完成真实Prolog和P/D

已归档的11项独立 reader 对照（两轮）通过：B1Q1、B1Q4、B8Q4 的 eager/Graph，以及坏 epoch 锁存。
B8 L2=0.00194345（eager/Graph phase0）、0.00195217（phase1），faults=0；仍未测 latency、原生量化器或完整服务。
测试使用 current_npu_stream reader，不应把这些结果扩大为显式生产 execution lane、真实 MTP/APC/PD 也已验收。
本阶段不再重复 reader smoke。

Prolog直接执行 [inline writer集成任务](../experimental/prolog_ring_smoke/PROLOG_INLINE_INTEGRATION.md)。
新 header 接受原 Prolog 已有的 LocalTensor，在同一次 launch 内旁路 BF16 写环；不是又写一个独立 kernel 去找 UB。
真实私有 Prolog variant 的编译、接通和输出对照仍未完成，不能将公共 Python 适配或 helper 标为整个 producer 完成。

**a139cb79 的长/短 prompt 分析必须纠正：长 prompt 的初始 decode 同样需要 P→D 尾窗。**
令 L 为 prompt 长度，d 为 D 已经本地物化的新 KV token 数；本轮最后 query 的位置为 L+d-1。
最近128窗口与 prompt 的交集 token 数为 min(L,max(0,128-d))。例如：

| prompt L | D本地新KV数 d | 最近窗口中仍来自P的token数 |
| --- | --- | --- |
| 8192 | 1 | 127 |
| 8192 | 4 | 124 |
| 8192 | 127 | 1 |
| 8192 | 128 | 0 |
| 64 | 1 | 64 |

具体 attention 只读取原 TopK 选中的 token；但不能预先假定它没有选中上述 prompt 尾部。
因此无论 GPQA prompt 是否大于128，都必须保留 P→D 的实际 BF16 尾窗传输及完成通知。
P保存并传输最后min(128,L)个prompt的latent/KR和tags，D收到完整尾窗后再允许首步读取；同时保留原C4传输。
不把缺失 BF16 改为冷C4后仍称为完整 recent128，不以“只影响短prompt”延期这个已知缺口。

## 最新 434b09d：B1七项通过，修复清理上下文后只补跑B8

实际 traceback 停在 B1Q4 的 reader.reset_faults，尚未进入 B8；不是多请求计算失败。
Graph 预热创建的 plan 含 inference tensor，清理在退出 inference_mode 后原地 zero_，违反 PyTorch 的生命周期合同。
reader.read 和 reset_faults 现在自己持有 inference_mode；已处于该模式时直接复用，不额外嵌套。
smoke 的坏 epoch 整段控制也持有该模式，不 clone、不替换图绑定的 buffer、不改计算内核。
原未保护方法在新增主机回归中能复现相同模式错误，保护后原缓存对象与地址不变；32项相关宿主检查通过。

保留现有结果：B1Q1 eager/Graph两阶段 + B1Q4 eager/Graph两阶段/坏epoch锁存，共7项通过。
B1Q1 phase1 的参考为全空/全padding，L2=0证明空输入处理，不能称作相同输入完美重放。

本轮仍只更新 Python，复用 SHA256=64d4a94eb843d97b172ee1dcff61abbe7e83fa515d680138c59c3b5f02b83955 的库。
可只补跑尚未开始的 B8，避免重复占用机时：

```bash
python experimental/single_pass/run_batched_smoke.py \
  --library-dir /tmp/hot128_serving/reader-build --source-root "$PWD" \
  --device npu:0 --graph --case b8_q4 --output "$NEW_RUN/b8-smoke"
```

不传 --case 则仍运行完整矩阵。选定B8通过时结果明确标为子集验证，不能冒充这次全部形状重新通过。
不重编ASC、不加载模型、不修改vendor；沿用空闲测试设备和新的输出目录。

## 最新 7d0ee407：继续同一批 reader 验证，不重编内核

新批量库已编译，B1Q1 eager 对独立数学参考通过：relative L2=0.0022918427、max_abs=9.98e-5、faults=0。
Graph 在进入 attention 内核之前因 plan key 不匹配退出，B1Q4/B8Q4 尚未执行。

根因是 torch.npu.graph 未指定 stream 时会切换内部捕获 stream，原代码按当前 stream 查找已预热的 plan，
因此外层 stream 已准备也会查不到。现在 smoke 显式传入同一预热 stream；生产入口按 builder/draft/ubatch
显式预留独立 execution lane，既允许预热/捕获切换 stream，也不合并独立并发 workspace。

两处 where 同步独立修复：PyTorch 2.10 的 where.self_out 两个分支要求 Tensor，ScalarOther 重载没有 out。
这不是 NPU 独有约束。-1 INT32 和 0 BF16 的同设备 Tensor 在 prepare 阶段创建，read 中只广播使用；
原 TopK 不改，输出 dtype 不改，无 read 内新增 Tensor 分配。
依据：[PyTorch 2.10 原始重载](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/native_functions.yaml)、
[Ascend graph 的默认捕获 stream](https://github.com/Ascend/pytorch/blob/master/torch_npu/npu/graphs.py)。

本轮只更新 Python，复用已经通过编译的 /tmp/hot128_serving/reader-build/libqsfa_recent128_direct.so
（原结果 SHA256=64d4a94eb843d97b172ee1dcff61abbe7e83fa515d680138c59c3b5f02b83955）。
不要重编 ASC、加载模型或再改 vendor；用新输出目录一次执行原矩阵：

```bash
python experimental/single_pass/run_batched_smoke.py \
  --library-dir /tmp/hot128_serving/reader-build --source-root "$PWD" \
  --device npu:0 --graph --output "$NEW_RUN/batched-smoke"
```

设备号沿用已确认空闲的测试设备。该命令集中覆盖 B1Q1、B1Q4、B8Q4、Graph 两阶段元数据变化与坏 epoch 锁存。
本地相关 27 项宿主检查通过，A5 Graph 结果待此次重跑；不能把 B1 eager 通过写成 Graph 或 B8 已通过。

## 使用 A5 之前一次说清楚的能力边界

6486b3d 是生产接入**候选代码**，不是完整服务已可用的证明。130 项宿主测试不能代替设备或完整模型验证。
以下是已知边界，必须先在内网解决未接线项，不能加载102个权重分片后再让用户逐个发现。

| 特性 | 已有实现/证据 | 当前缺口或限制 |
| --- | --- | --- |
| 原 MXFP4 主缓存 | 保留 packed400 写入和原 TopK 合同；OFF 不走新路径 | 真实新 Prolog ON/OFF 的 query、C4 packed/scale 尚无设备逐字节对照 |
| 真实 PrologV3 | 公共调用适配、完整 ABI1、实际写点要求、已通过的独立 writer | 私有 producer variant 必须实际编译注册；独立 writer 不是 producer |
| 并发8 | 独立批量 TND reader 的B8Q4 eager/Graph/epoch控制已通过 | 合成reader不是完整vLLM并发服务验收，真实producer及调度还须联合验证 |
| Graph | 独立 reader 的 B1Q1/B1Q4/B8Q4 Graph 已通过 | 所有 layer/lane/bucket 必须捕获前准备；生产execution lane和完整服务的动态 replay尚未验收 |
| MTP | 修正后的 GPU 位置、draft lane 隔离、容量保护、事务 API | 完整接受/拒绝和提交链未验证；draft 层自己的历史 BF16 热窗来源仍须核实，不能拿 target 层就绪代替 |
| APC / prefix cache | 实际 BF16 尾窗快照导出/导入 API | **尚未自动接通 prefix 命中、驱逐、抢占恢复。不是仅缺测试。** 非零位置新请求缺真实尾窗会报错 |
| P/D | 实际尾窗导入及接收完成事件 API | **尚未接通 Mooncake 的 BF16 尾窗发送、接收和通知协议。不是仅缺测试。** C4 传输成功不足以证明 BF16 已到 |
| PCP / DCP | 当前新 sidecar 未有对应 gather/shard/跨 rank 完成协议 | 当前不支持；现在在配置期明确指出，不能因 serving 配置被接受就列为已支持 |
| 性能、显存、整模型精度 | 旧独立算子实验有历史数据 | 新 batched_v4 的速度/容量损失/TPOT/GPQA 均未测；不得引用 B1 KV-split4 的旧速度作为本实现结果 |

硬格式合同一次列明：仅 latent512、RoPE64、KV-head1、page128、packed400 主缓存、BF16 query、INT32 TopK2048；
R0/272、其他 TopK、页大小或格式没有实现。每64个 local heads 一组，不足补齐，多于64会多组调用。
新物理 ring 容量 C=align128(128+max_query_chunk-1)，不是固定128槽。
max_query_chunk 必须覆盖实际每请求 prefill/verify chunk，不能只按 MTP 的4个 token 填写；总 token/pool/表宽也受预分配界限制。

已完成的 A5 设备证据为旧独立 ring writer 8/8，加独立批量 reader 的B1 eager/Graph共7项；它们不验证真实 Prolog、C4 量化或完整 MTP。
同一次 Graph 中 valid mask 变化的测试覆盖，也不能用旧 writer 的固定 valid mask 重放替代。

## 一次准备、一次成批验收的执行约定

1. 上 A5 前，先完成上述 Prolog、APC、PD 和 draft 层热窗等已知实现缺口；主机检查/全部编译/ABI与配置问题一次收集，
   能在 CPU 上解决的错误不占设备时段，不以关闭用户特性替代实现。
2. 预先固定一份矩阵：B1/B8、Q1/Q4、eager/Graph、环回绕/padding/请求重排、MTP全接受/部分拒绝/全拒绝、
   APC冷启动/命中、P→D首步/后续步。每项写清原配置、预期和输出位置，不能跑到一半再添加已知必要条件。
3. 已有 run_batched_smoke 是**reader 部分**的成批设备门禁；APC/PD/真 Prolog/完整模型不在它的覆盖内。
   完整矩阵须接好相应实现再执行；不能宣称一个 reader smoke 已经把所有特性一次验完。
4. 同一套构建通过后再加载完整模型，保留原生产特性，完成 OFF/ON 对照和完整 GPQA。
   新设备错误仍可能出现，但每次回传要包含所有已执行项和阻塞项，不能只返回下一处试错指令。

目标固定为原 MXFP4 主 KV + BF16 recent128，保留并发8、Graph、MTP、prefix cache、P/D。
不把 C4 改 C8，不把 max_num_seqs 改1，不强制 eager，不丢弃 PrologV3。
旧 accuracy_eager 是单请求数学验证代码，不能仅删除其 guard 后用于服务。

## 已交付的实际代码

- serving_state：稳定请求 bank/epoch、预分配路由、分 lane 元数据、独立 BF16 ring、真实 prefix/PD 尾窗快照。
- serving_runner：挂接真实 NPUModelRunner 和 SFA metadata builder，使用本轮设备 positions/query_start_loc；MTP 不从乐观 CPU 长度重建位置。
- serving_prolog/hooks：包裹实际 MXLLM `_call_mxfp4_cloud_prolog`；ON 只调用一次显式 recent128 producer，原 query 五项返回和 mode4 C4 写入合同保留。
- serving_runtime/serving_reader：生产分流、完整原始物理 C4 池、批量原 TopK、Graph 固定缓冲和真实批量 reader。
- 新的 additive batched_v4 ASC/bridge：沿用 V4 load32/stride416 流水，按请求 bank+position+epoch 读 BF16 热环；旧 B1 KV-split4 不更改。
- install_overlay --update-runtime：验证已有托管 overlay，备份并更新 runtime，不改原 MXLLM 三处补丁与 vendor。

这些是待 A5 联合编译验证的候选实现，不宣称已拉起服务或测过生产性能。

## 内网 agent 直接依次执行

1. 在当前匹配的 SDK 中完成真实私有 Prolog variant 的编译和注册。新入口必须是
   `cloud_ops_turbo.cloud_mla_prolog_v3_recent128`，函数 `recent128_abi_version=1`，完整 ABI 见 PROLOG_PRODUCER_TASK。
   将已 A5 验证的 writer 搬运/事件写法移到实际 mode4 量化输入和真实 KR 写点，不能给旧函数贴 marker 冒充实现。
   OFF/ON query 与 C4 packed/scale 逐字节不变；sidecar 来自真实 FP32 量化输入的 BF16_RNE。

2. 在独立目录重编本仓 `experimental/single_pass` 的 shared library，保留当前匹配 SDK/dav-3510 和已有 KV-split 开关，
   开启新增 serving batch 目标。不要覆盖已工作的 C4/vendor/旧候选库。先跑本仓批量 reader correctness smoke，
   验证 B1Q1/B1Q4/B8Q4、原 TopK、不同行/epoch/padding、Graph 重放。首个设备异常即停该测试，不重置卡。

   ```bash
   cmake -S experimental/single_pass -B "$HOT_RUN/reader-build" \
     -DSFA_SDK_ROOT="$CURRENT_MATCHED_SDK" \
     -DCMAKE_PREFIX_PATH="$CURRENT_MATCHED_SDK;$CURRENT_MATCHED_SDK/lib64/cmake" \
     -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DQSFA_BUILD_BATCHED_SERVING=ON
   cmake --build "$HOT_RUN/reader-build" -j8
   python experimental/single_pass/run_batched_smoke.py \
     --library-dir "$HOT_RUN/reader-build" --source-root "$PWD" \
     --device npu:0 --graph --output "$HOT_RUN/batched-smoke"
   ```

   设备由内网 agent 确认空闲；SDK 沿用已经通过 writer 的匹配安装，目录变量用内网实际值，不升级、不 source 旧包。

3. 对当前 hot128 专用 MXLLM checkout 更新托管 runtime：

   ```bash
   python framework_integration/install_overlay.py --mxllm-root "$HOT_MXLLM" \
     --source-profile pr199 --apply --update-runtime --output "$HOT_RUN/overlay-backup"
   ```

   复用之前实际通过的 source-profile/expected-SHA 参数。若旧 runtime 曾被内网修改，工具会指出与托管 manifest 不符；
   在内网审查并保留这些修改，不伪造 manifest/强制覆盖。无需把内部实现上传给外部。

4. 仅更新原 additional-config 的 sfa_recent128 子项：

   ```json
   {"window":128,"execution":"serving","preprocess":"preserve",
    "reader":"single_pass","single_pass_kernel":"batched_v4",
    "max_query_chunk":1024,
    "source_root":"本仓绝对目录","library_dir":"新批量库独立目录",
    "diagnostics_path":"本轮独立目录/layers"}
   ```

   max_query_chunk 必须覆盖真实每请求的最大 prefill/verify chunk；不是一律设4，也不是总 batch token 数。
   物理 C 按 W+maxQ-1 对齐到128，与读取窗口128不同。原 vendor additional-config 字段和
   attention-config 中 indexer_kv_dtype=mxfp4、mla_kv_quant_mode=mxfp4 均保留。
   可用 serving_config.py 从现有 JSON 只合并本子项；不重新生成整套启动参数。

5. 在内网接好现有 prefix/Mooncake 生命周期：
   - prefix snapshot 或 P→D 真正传输 BF16 latent/KR + tags，再调用 serving_runner.receive_tail。
   - snapshot_tail 必须对应已提交的前缀边界；PD 的 completion 需要真实接收完成事件。
   - 主 C4 已到达不代表新 BF16 sidecar 已到达。非零起点缺尾窗会明确报错；不能假造 ready 或用 C4 反量化顶替。
   - 新 token/草稿使用本轮修正后的 GPU 位置，按位置因果判断；请求完成显式释放，复用 bank 时 epoch 改变。
   公共生命周期接口已提供，实际 connector 的私有接线在内网就地实施，不能把接口存在写成传输已完成。

6. 按原生产并发8、Graph、MTP、prefix、P/D 参数拉起 hot128 专用服务，baseline 链不动。
   缺私有 producer 就先完成第1步，不再回到 Native 或 max_num_seqs=1。
   完成请求后通过 worker RPC 调用 serving_runtime.validate_serving_faults()，检查所有已准备 bucket 的 sticky 错误；
   此检查在计时外同步，不能每层 .tolist()。故障为零也不代替真实模型精度测试。
   再用相同 seed/请求集对比完整198题 GPQA；78/94 的中途结果仅保留为中途记录，不能归因成最终精度。

366ae84 的加载后报错问题同步修复：完整 engine 配置的静态限制在 config factory 前检查；serving 模式
缺实际 producer/批量库的错误在模型层构造时检查，早于102个权重 shard 的加载。
只有依赖真实权重/内核行为的条件保留到相应初始化或设备执行时，不将未运行的内核伪称已验证。

只回传部门允许的编译状态、首错、实际调用/测试摘要。内部源码、schema、二进制和原始材料留内网。
