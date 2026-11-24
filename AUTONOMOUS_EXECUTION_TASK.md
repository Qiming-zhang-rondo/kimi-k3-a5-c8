# 内网 AI 完整执行任务：原生 BF16 热端融合冷端状态

## 任务与工作方式

你接手的是一项完整开发任务，请在内网容器里直接完成源码定位、依赖补齐、构建、接口接线、A5 正确性和性能验证，并将结果提交 GitHub。不要只给方案，也不要再让用户逐个搬运头文件、执行诊断或转述日志。

仓库：`Qiming-zhang-rondo/kimi-k3-a5-c8`；协作分支：`sfa-cold-merge`。
源码/依赖审计起点：`b29671a5ba83f535624c29f450f2f362aa753be5`。以远端分支最新提交继续工作，保留此版本作为比较点，不把新提交重置回该版本。
本目录已有 `original/`、`deps/`、`arch35/cold_state_merge.h`、测试和构建背景；helper 目前只是 host 验证过的标量合同，不是已经可运行的原生融合实现。

使用你已具备权限的容器、源码仓、编译器和 GitHub 凭据，自行解决普通编译错误、缺失依赖、路径及 API 适配问题。复用随任务提供的收集、日志工具；先读其帮助和代码，不猜参数。没有工具覆盖的步骤自行实现并留日志。
只能使用空闲 NPU；不停止服务、不杀其他任务、不重置设备、不替换正在服务的算子包。当前任务不需要这些操作。隔离目录、独立算子名及子进程环境足以开展实验。
不要升级 CANN，不混用另一容器的旧 `set_env.bash`；若工作容器与记录不同，先记录实际来源并复现基线，再决定匹配的构建环境。

本目录工具用法（把示例路径和 COMMAND 换成自行发现的真实路径/命令）：

```bash
python3 tools/collect_source_closure.py --source-root /actual/package/root --source-root /actual/source/project --output /fresh/isolated/closure
python3 tools/run_logged_stage.py --stage baseline_build --work-dir /tmp/sfa-evidence --cwd /actual/source/project -- COMMAND ARG...
python3 tools/run_logged_stage.py --stage baseline_build --work-dir /tmp/sfa-evidence -- python3 tools/probe_original_sfa.py --work-dir /tmp/sfa-original-probe --compile
```

收集器只读源目录，输出来源/哈希和隔离副本；缺依赖时保留 partial 结果并返回非零，继续查找真实来源，不能把它当成编译成功。尽量给精确源码根目录，不扫描整盘或提交无关源码。
日志工具阶段名为 `baseline_build`、`candidate_build`、`accuracy`、`performance`，每次生成独立 `stage.log/status.json`。出现设备故障标记 `DEVICE_FAULT.json` 后停止设备测试链；继续只读分析/构建，不删标记强行重试或自行复位设备。

## 已有基线与明确目标

- 固定形状：B=1、Q=4、K=8192、H=64、latent D=512、RoPE=64、TopK=2048、W=128、page=128。
- 当前 BF16 热环物理容量256；合成 TopK 含最近256个可见 token，因此每行热命中恰好128。它不是任意真实 QLI 命中分布。
- 历史同轮结果：C4 W0 pipeline **87.53 μs**；BF16 `hot_store_partition` 两遍路径 **126.17 μs**，配对 W0 比值1.438；L2 分别约0.10410、0.099006。这些是历史对照，必须重测，不能复制为新实现结果。
- 对照 launcher：`8ccd0d418ea1afa83c06c34d83f4716577a1f473` 的仓库根 `run_kv_bf16_hot_ab.py`；固定 payload：`4a5fd67b9a78766bebfa564782ea2193bc22af1a`。
- 对照代码位于 payload 的 `kv-native-twopass/hotkv/`，重点读 `kv_bf16_hot_ab.py`、`kv_native_merge.py`、`test_kv_bf16_hot_ab_a5.py`；遵守 launcher 原有 SHA256 校验。
- 目标：保留原生 C4 冷读、原生 BF16 热端 QK/PV 流水，在热端最终输出阶段合并冷端状态，取消独立 Triton merge，测出相对当前最快两遍路径的实际收益。
- 保持原有 cold write、BF16 hot-store/partition、query、TopK 和量化基底。使用 **baseline C4 配方**；不要启用 `scale_alg=2 + dst_type_max=7`，用户报告该组合模型精度下降约2点，合成单算子 L2 改善不等于模型精度改善。

## 阶段 1：建立可追溯的工作环境与真实调用链

1. 检查仓库工作区、远端分支和本地修改，创建隔离构建目录；不覆盖他人文件或 force-push。记录源码 commit、所有借用文件的原路径及 SHA256。
2. 记录容器、芯片、驱动、CANN、custom_cot、wheel、Python/torch/torch_npu、编译器版本，实际加载的 `.so`/kernel 路径以及空闲设备。记录足够复现的信息，不提交 token、代理凭据或完整敏感环境变量。
3. 复现既有 BF16 两遍基线的正确性；环境不工作时自行核对实际包/库来源，不能通过混包或删门禁制造通过。确认 ninja 和独立 JIT 缓存，不清理用户缓存。
4. 从 `torch.ops.cloud_ops_turbo.cloud_sparse_flash_attention` 追到实际注册绑定、`aclnnSparseFlashAttention`、host tiling、实际 tiling key 与 arch35 kernel。保存符号/源文件/运行证据。
5. 注意：已上传 `deps/cann_ops_transformer/attention/sparse_flash_mla.cpp` 调用的是 **aclnnSparseFlashMla**，返回两个 tensor；它不是上述 Cloud 三输出 SFA 的绑定。自行从 wheel、安装源码或已授权内网仓找到正确实现，不能把错接口改名充数。
6. `BUILD_CONTEXT.md` 是定位线索，不是完整验证记录；核对其中源码/二进制对应关系及工具可用性。`util_regbase.h` 含补充字段注释，需核对真实部署/构建版本，不假定结构体 ABI 已匹配。

## 阶段 2：自己补齐依赖，并先构建原版

1. 从实际安装目录或匹配源码仓递归收集 include 闭包，保留目录结构、许可证、来源及哈希。普通缺头文件不是交回用户的理由。
2. `5c10d9629a5c41e6a9fae27a5cb8146a8c886d20` 已补齐六个 `vf_basic_block_*_sfa.h`、VF utils，以及此前五个 memcopy 头。不要再把这些文件作为向用户索取的事项。用 `tools/probe_original_sfa.py --compile` 在隔离副本实际尝试原版编译，继续自行处理递归依赖和真实 SDK 生成头；该探针不运行 NPU 或改安装包，只有实际调用编译且产生非空 `.o` 才算编译通过。
3. 找到实际 host tiling 实现、生成 tiling data 的步骤、注册/schema、ACLNN 封装、构建和打包配置。动态 `sparse_flash_attention.py` 是编译入口，不等于 host tiling C++ 实现；不要臆造结构体或复制不匹配的 tiling 二进制。
4. 先在隔离目录构建**未修改原版**，逐轮保留命令、首个错误、修复和退出码，直到原版可加载并通过同形状 A5 基线。不要把独立 merge 小程序编译成功当成 SFA 原版编译成功。
5. 原版构建通过与候选构建通过是两个独立关卡。若只能获得直调 kernel 构建入口，可先完成隔离验证，但如实标为直调实验，继续完成正式测试适配，不冒充已打通现有 ACLNN。

## 阶段 3：实现原生融合和真正的新接口

新增独立实验入口，保留原版；实现完整 schema/绑定/ACLNN 或已证明等价的直调入口、tiling、kernel、Python 适配。不得覆写已安装服务算子，也不得将另一个独立 merge kernel 包装成“融合”。

已有 `experimental/` 原生源码候选：新 kernel 入口、五冷输入的 Init/Vec 接线、真正 MicroAPI/DMA epilogue、直接 FP32 store，以及独立 proto/API 声明和 Torch bridge。先审查并继续构建这些文件，不必从零重写。`experimental/README.md` 标明精确范围和未完成的 host/API 实现；API 头只是声明，不代表新符号已经存在。原始 `original/` 不要覆盖。

额外输入为设备上的 `cold_output [T,H,512]`（保持原生冷读实际 dtype）、`cold_max/cold_sum [T,H,1]` FP32、`cold_present/hot_present [T]`。最终输出与当前两遍 merge 相同 dtype；本实验冷端为 FP32 时最终必须 FP32，不能沿用热端 BF16 store。
所有 tensor 有明确 shape/dtype/stride/device/非重叠合同；presence 规范化为逻辑值后使用，禁止在原始 uint8 上用按位 `~` 冒充逻辑非。不得在计时路径 `.item/.cpu/.tolist/nonzero` 获取动态信息。

已经可由源码确认的接线依据：

- `deps/.../util_regbase.h` 的真实 `RunInfo` 包含 `attentionOutOffset`、`vec2MRealSize`、`multiCoreIdxMod2`；`ConstInfo` 包含 `dSizeV/gSize/n2Size/softmaxScale`。
- 当前提供的 MLA 路径不沿 S2 跨核切分；仍要用本次实际 tiling/key 验证选中该路径。仅在热端完整归约后合并一次冷端，不能每个 partial 重复合并。
- `vf_flashupdate_new.h` 的 `REDUCE_SIZE=1`，最终 FlashUpdateLast/LastDiv 已把 FP32 PV 除以 sum；stats 是每个 head 一个 float，不是 stride8。
- 原 `ProcessVec2` 最终 `ZeroInvalidAttnOut` 后、`CopyOutAttentionOut` 前是候选插点。不要在合并后再按“热侧无效”清零，否则会抹掉有效冷侧。
- 首版限定 TND、N2=1、D512：`flatHead=attentionOutOffset/dSizeV+m`，presence token=`flatHead/gSize`；M 轴含 head 行，不是 token 行。真实 GM D 与 UB 对齐 stride 必须分开。
- 原输出 `Cast(...CAST_ROUND...)` 将热输出舍入 BF16。首版在合并前保留该舍入，冷输出独立 dtype，最终走正确的 FP32 store/字节 stride；仅修改输出宏可能误选 half 输入分支，必须核对完整 dispatch。

继续阅读补齐的基础 softmax VF，证明 QK scale、max、sum 的指数域；不能仅从函数名假定自然指数。对归一化输出，质量为 `Z_i=exp(m_i)*s_i`，合并用稳定 log 权重；不要直接计算 `exp(m_i)`。若在未归一化 PV 处融合，必须换用对应分子公式，不能把 PV 再乘一次 sum。
只有一侧有效时返回该侧；双空输出0；非空侧非法 max/sum/NaN/Inf 必须暴露，不能悄悄清零。完整空热段可能跳过 `ProcessVec2`，要增加责任核处理或明确拒绝该输入，不能宣称 epilogue 自然覆盖。
注意新增 scalar/VF/DMA 的同步、buffer 生命周期及 UB 占用；原 V→MTE3 标志不能自动证明新增 scalar 访问安全。性能实现优先小片 cold DMA+向量计算，不能把 host 标量 helper 直接贴进内核后宣布优化完成。
沿用当前 stream、静态 buffer 和 graph 回放；不落地无人使用的 hot output/max/sum 中间结果。诊断输出可以独立开启，正式性能路径不要因此额外写回。

## 阶段 4：正确性先通过，再做同轮性能比较

第一轮仅上述已验证固定形状、compact 热索引128。复用原 harness 数学参考、byte gate 和严格阈值；不要顺便修改旧包的 `-1` padding 行为，也不要任意放宽阈值。

必须检查：

1. 原版重构建输出与已部署原版；新融合输出与当前 BF16 两遍路径；各自对匹配存储精度的独立数学参考。
2. 冷热分区完全一致；C4 cache、BF16 latent/RoPE 写入逐字节一致；prepared、dynamic-read、write+dynamic 全流水输出一致。
3. zero-query：热端 `exp(max)*sum=128`、冷端=1920，确认数据域而不只看最终输出；热端输出本体也单独对参考。
4. 冷单侧、热单侧、双空、极端有限 max/sum、非法非空统计；实际不支持的形状必须明确拒绝而非未初始化输出。
5. 改变冷输出/统计、query/KV、重复写环、复用 buffer/graph 后重跑，检查陈旧数据及覆盖问题；保留首个错误位置、实际/参考值及原始统计。
6. 有意改变中间舍入的实现作为独立变体，不偷换“同精度合同”的候选。所有失败候选保留失败状态，不能给正常性能结论。

通过后，同一空闲设备/进程对三臂做 AB/BA 配对：原生 C4 W0、当前 BF16 fused-writer 两遍、新原生融合。warmup20、iterations100、10轮；先完成编译/捕图再计时。
完整 pipeline 包含当前 token 冷写、热写/分区、冷读和新热读融合；prepared read 另报。保存每轮原始样本、配对比值中位数和 min/max；min/max 不是置信区间，不把不同轮中位数相除冒充配对结果。
用 profiler 核实独立 merge 真正消失、没有隐蔽额外核/同步/拷贝，记录 op/kernel 数、最终 dtype、UB/寄存器和实际瓶颈。阶段时间不可相加或直接从126.17 μs扣除，更不能承诺达到87 μs。
若候选无收益，依据 profiler 做有边界的原生优化并复测；同时保留最快已通过方案。不要通过删写入、复用旧输出、减少有效 TopK 或遗漏数据依赖获得“加速”。

## 持续推进、结果提交与完成定义

在本目录维护 `STATUS.md` 和机器可读状态，至少区分：环境、依赖闭包、原版编译/运行、候选编译/运行、正确性、配对性能、接口集成。每项记录命令、时间、退出码、结果路径和待解决事项。
在受控目录保存完整构建/测试日志，逐阶段提交可审查代码与短结果；大二进制/敏感模型激活留在内网，只提交哈希、保留位置和必要摘要。不要上传凭据或无关内部源码。
向同一 `sfa-cold-merge` 分支正常 commit/push；推送前 fetch 检查新提交，非快进时保留双方更改并合并，不 force-push。记录最终 commit 和可复制的构建、加载、测试命令。
外部 Codex 会依据 GitHub 提交继续代码审查；这不代表已设置自动轮询或无人值守监控。你应主动将当前可审查结果推送，不等用户转发一屏日志才继续。
遇到普通缺依赖/编译错误/路径错误时继续自己处理，不向用户逐个索要文件。只有确实缺失且无已授权来源的源码或仓库权限、设备/驱动物理故障、必须中断服务等硬阻塞，才提交准确阻塞报告：已尝试的方法、具体证据、最小需要的外部动作；不要用“需要更多信息”代替定位。
完成标准：可审查的原生实现与实际入口、匹配的可重复构建、原版与候选 A5 正确性证据、同轮配对性能和 profiler 证据、完整状态/日志/commit。若被硬阻塞，交付已完成部分及精确阻塞，不能写“融合完成”。
单算子通过仍不等于真实 GLM 质量、任意 QLI 命中、连续解码、PD 迁移或端到端 TPOT 通过；这些范围保留明确标注，当前不要求修改服务部署。
