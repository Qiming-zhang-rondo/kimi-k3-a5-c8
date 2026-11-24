# 当前执行任务：真实 C4 融合对照与三臂配对性能

## 接手后直接继续

最新修复见 [PERF_REPAIR_5407.md](PERF_REPAIR_5407.md)：5407a8e的新bench绕过了输入与精度合同，
305.84µs与历史基线比值不能使用，旧入口已封禁。当前构建SFA_BUILD_TORCH_DIRECT设备库，
通过direct_bridge/run_direct_ab.py接回原SHA固定harness的真实C4流水线；精度门禁全过再测。

仓库 `Qiming-zhang-rondo/kimi-k3-a5-c8`，分支 `sfa-cold-merge`。
`2ddc663` 已提交原版 A5 实测：3 项 DEVICE_VALIDATED，L2 0.00175–0.00191。
提交 `5a361849953a35b22695f98abcbe42e17623f777` 的
[融合结果](https://github.com/Qiming-zhang-rondo/kimi-k3-a5-c8/blob/5a361849953a35b22695f98abcbe42e17623f777/standalone/sfa_fusion_device_results.json)
已确认 8 项全部 passed、nonfinite_count 全为0，状态 `DEVICE_SMOKE_VALIDATED`，范围
`given_fp32_cold_state_plus_native_bf16_hot`。三项范围标志
`c4_pipeline_validated`、`native_twopass_compared`、`real_hot_empty_dispatch_validated` 均为 false。
阶段 A 已有固定 smoke 证据，阶段 B 已有给定冷态的融合设备证据；**下一步直接接真实 C4
冷态和原生两遍对照，再做阶段 C**。不要把重跑这8项作为新的交付终点，也不要宣称阶段B已全面验收。
[PHASE_B_2168_REVIEW.md](PHASE_B_2168_REVIEW.md) 的构建命令保留作复现背景。
保留已成功的 SDK、CMake 参数及构建目录，不重新开始 SoC/namespace 排查。

你在内网拥有执行环境，负责连续完成以下三个阶段及普通实现/构建修复。
本文件是当前优先任务；`AUTONOMOUS_EXECUTION_TASK.md` 保留完整合同与验收背景。
阶段间无需让用户转述结果或再次授权。代码、原始日志和结果直接推送同一分支供外部审查。

本次交付：可直接调用的原版和融合候选、A5 精度结果、同轮三臂性能；不是仅编译一个静态库。
不要等待完整 `.run`/wheel 发布才开展独立单算子测试。使用空闲设备与隔离目录；保持当前
工作环境，不升级 CANN、不混旧 vendor、不停止服务或重置卡。

## 固定实验合同

- B1、Q4、K8192、H64、latent D512、RoPE64、TopK2048、page128、逻辑 W128。
- BF16 热环物理256槽；每行热命中128，冷命中1920。沿用当前合成数据与 query 因果位置。
- 热/冷索引必须是同一 TopK 的互斥、无遗漏拆分，热端 compact128 全有效。
- 冷端仍是原生 C4；热环由原始 BF16 latent/RoPE 写入。保留原有 writer/partition。
- 使用 baseline C4 量化配方；不混入用户模型评测下降的 `scale_alg=2,dst_type_max=7`。
- 热端 BF16 原版输出、FP32 max/sum；冷端归一化 output/max/sum 都是 FP32。
  融合最终 output 为 FP32；不得把冷输出降为 BF16。stats 每 head 一个 float，stride1。
- 当前融合首版不返回 combined stats，明确 `return_softmax_lse=false`。

## 阶段 A：已完成的固定 smoke 合同与保留检查

原版启动和3项固定精度已通过。以下保留原合同；仅在相关实现变更后补回归，
不用重新做原版SDK恢复或重建已有启动器。原版与候选继续保持头文件和目标隔离。

1. 保存当前成功构建命令、完整日志、SDK来源、源码/产物哈希和实际模板模式。
   检查真实3510 AIC、AIV两份设备代码及任务类型，不以host符号、空archive或CMake成功判定。
   `SFA_BUILD_CONTEXT` 应能区分 host 和两个 device pass；当前入口在host不执行设备实现。
2. 新增最小 direct-launch 启动器及Python测试入口，实际调用编出的原版BF16 kernel。
   以当前SDK真实直接启动接口实现；需要Torch包装时使用当前NPU device/stream，
   不默认stream0，不在计算路径同步到CPU，保持异步输入、输出、tiling和workspace生命周期。
   校验设备、dtype、shape、stride、参数顺序、混核任务配置和blockDim。
3. 实现与本次编译POD匹配的固定形状host tiling writer：全部字段/填充初始化、sizeof/offset
   校验；逐项依据选中源码解释字段含义。workspace大小、对齐、系统区、GetUserWorkspace偏移、
   跨核同步及临时buffer范围也从本次源码/SDK推导并记录，不猜一个“大buffer”掩盖缺合同。
   不读取/截取现有ACLNN executor中的私有opaque tiling，也不以同为144B推定安装包ABI。
4. 复用相同Q/K/V/RoPE、block_table和compact128索引，对比重编原版、已部署BF16热读、
   独立数学参考。参考只覆盖本次有效热子集，不能拿完整top2048参考替代。
   Q latent及Q RoPE同时置零：每个head的 `exp(max)*sum` 应为128，输出应对齐相应V均值。
   保存max/sum、首个错误行及实际/参考值；数值阈值沿用旧harness，不能为了通过放宽。
5. 改变query、cold无关的热KV、重复写环后重跑，确认输出随输入变化并覆盖旧buffer。
   对同一合法输入复用静态buffer与graph验证，排除空壳、陈旧数据及漏写。
   原版设备正确后继续阶段B，无需停下来询问用户。

## 阶段 B：把现有融合源码接入同一启动框架

当前先实现可供旧harness直接调用的桥，随后用真实冷态验证；已通过的独立smoke用来定位
新增接入回归，不代替下面的真实流水线验收。

1. 沿用已通过smoke的 `standalone/sfa_fusion_entry.asc` 和 `experimental/arch35/`
   cold-state epilogue，保留原生BF16热端QK/PV流水、device-only头隔离、AIC/AIV模板服务、
   GM到本地POD加载及raw-user-scratch协议。不从TBE入口重新拼装，也不改动已验证原版。
2. 14输入的顺序与 `experimental/op_api/include/`、proto及原候选合同一致；新增五个冷态
   指针在AIC/AIV贯通。保留cold/hot presence逻辑、热输出BF16舍入及最终FP32写回。
   合并仅在完整热归约完成后一次执行；检查DMA/event、UB资源、head/token偏移和空分片。
3. 现有 `experimental/torch/sfa_cold_merge_bridge.cpp` 依赖尚未实现的新ACLNN符号，
   头文件只有声明，不能直接调用此次静态库。优先为测试实现真实direct bridge；
   如果选择生成新ACLNN，则必须完成真实host/API注册，禁止回退到旧两遍实现冒充融合。
   direct bridge 必须使用调用方当前NPU device/stream，保持输入、输出、固定tiling和scratch
   在异步执行及graph replay期间有效；不要在热路径做D2H、同步、设备初始化或新建stream。
   接入前校验固定shape/dtype/stride、compact128索引与FP32冷态合同；不支持的形状直接拒绝。
4. 检验融合输出 vs 现有BF16两遍输出及各自匹配存储精度的独立参考；冷热分区、cache写入
   逐字节一致；prepared、dynamic-read和write+dynamic输出一致。
   冷 output/max/sum 必须由本轮同输入的真实原生C4 reader产生，不再用smoke的合成冷态。
   用同输入的原生BF16热输出及两段真实stats组成 `NativeReference`，并保留原生两遍实际输出
   作为另一控制；融合/两遍沿用整体和逐行L2≤0.001严格门槛，不能拿数学smoke的0.01替代。
   zero-query热质量128、冷质量1920；检查presence单侧、双空、非法非空统计及极端有限值。
   helper支持空侧不等于完整原生调度支持hot=0；首版不支持的输入必须在launch前明确拒绝，
   不能运行未初始化输出。需要扩大支持时实现相应责任核/调度并单独验证。
5. 正确性通过后继续阶段C；普通编译/API问题自主修复并保留日志，不再让用户逐个搬文件。

## 阶段 C：同一轮三臂性能，检验真实收益

在同一空闲设备、同一进程和同一批输入比较：

| 臂 | 实现 |
| --- | --- |
| W0 | 原生纯C4全流程 |
| 两遍 | 当前BF16 hot_store_partition + 原生冷读 + 原生热读 + Triton merge |
| 融合 | 相同cold write、BF16 hot-store/partition、原生冷读 + 新原生热读/冷态合并 |

旧BF16/C8对照入口是main的 `run_kv_bf16_hot_ab.py`，launcher commit `8ccd0d4`，
payload `4a5fd67b9a78766bebfa564782ea2193bc22af1a` 的 `kv-native-twopass/hotkv/`。
沿用原SHA校验；新候选通过独立入口接入，不修改旧payload内容来制造比较。

- warmup20、iterations100、10轮AB/BA；先编译、分配及捕图，再测相同scope。
  分别比较两遍/W0、融合/W0、融合/两遍；保存每轮原始event与wall数据。
- 主指标是write+dynamic完整pipeline，包含新token冷写、热写/分区、冷读、热读/融合。
  prepared read作为另一个scope报告；graph必须捕获真实数据依赖和新kernel。
  不能复用旧结果、遗漏写入、改有效TopK或仅在计时外完成候选的工作。
- 保存配对比值中位数和min/max；min/max不叫置信区间。历史87.53/126.17µs只作背景，
  本轮重测，不将阶段耗时相加/相减或承诺“融合后一定达到87µs”。
- profiler确认独立merge消失、新kernel实际执行、没有额外同步/拷贝；报kernel数、
  阶段耗时、UB/寄存器资源和实际瓶颈。若变慢，保留最快通过方案及失败候选，
  根据profile做有边界修正后配对复测，不能只说“理论可优化”。

## 证据、提交和完成

使用 `tools/run_logged_stage.py` 的 baseline_build/candidate_build/accuracy/performance
保存实际命令、退出码、日志与状态。设备故障后停止设备链并保留DEVICE_FAULT标记；
可继续只读分析/构建，不自行重置设备。普通precision FAIL保存详细错误后修复，不计为通过。

维护 `STATUS.md` 和机器可读结果，区分BUILD_PASS、DEVICE_SMOKE_VALIDATED、DEVICE_VALIDATED、MEASURED、
ERROR、CONTROL_FAILED；包含完整输入合同、源码/二进制来源及哈希、复现命令和结果位置。
代码、构建日志、精度数据、配对原始样本、profiler摘要推送 `sfa-cold-merge`；
大二进制留内网，提交哈希/大小/符号与保存路径，不上传凭据或无关模型数据。

阶段A可先推送代码/日志方便并行审查，随后直接继续B/C，不等待用户中转。
完成应有可启动的原版和融合实现、设备精度与同轮配对性能；仅编译通过或宿主公式验证不算完成。
确实需要新权限/源码且现有环境无法解决，才提交完整证据和最小外部动作；普通缺头、
入口/API适配、编译错误由你继续解决。单算子结果与全模型精度/TPOT分别标注，本轮不部署服务。
