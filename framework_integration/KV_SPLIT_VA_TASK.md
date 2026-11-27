# 当前交付：把KV-split接到真实VA attention入口

**用户最新授权：先接自研MXFP4+BF16 hotwindow V4最新融合版，先看真实模型整体精度；DSL路线暂不迁移。** 环境采集与格式预检是同一任务第一步，agent自主完成后继续隔离安装和模型验证，不再等待用户逐项中转。

用户已明确从单算子转向VA框架接入。当前优先执行本任务；上一轮完整20轮原始证据仍须从既有文件打包推送，但不再为了它重跑单算子或阻塞框架源码适配。

## 本轮启动问题修复（回报74b0773）

回报的三个错误发生在LLM初始化阶段，尚无候选attention执行或模型精度结果。
PR199的`import ascend_vllm`只定义入口；其`_ensure_global_patch()`才会按顺序加载VA全局补丁、
MXLLM的`patch_envs`和`patch_attention_config`。其中环境注册必须在worker读取前完成。
`run_model_probe.py`现执行该正式入口，再验证真实配置字段和环境注册，不直接提前导入worker。

VA会严格校验`additional_config`里的自定义字段。新增配置桥接只在调用其工厂期间隐藏
`sfa_recent128`，用`finally`恢复原配置对象；其他参数继续接受原校验。
安装器现在同时修改platform、worker和BF16捕获点三处，确保spawn的EngineCore也在配置校验前安装桥接。

内网agent现在直接拉取后使用新隔离PR199 checkout重装；旧两文件overlay不能叠装，按原manifest恢复或换新checkout。
确认MXLLM正常插件安装/注册仍指向该源码，保留其正式worker生命周期；仅修改`sys.path`不能替代插件注册。
沿用下文同一engine与prompt命令，先完成smoke，再执行原评分。无需改内核或升级CANN来处理这三个启动错误。
若启动仍失败，将输出目录的`startup.json`、`startup_traceback.txt`、`probe.json`及脱敏环境身份一并提交；
其中stage会区分全局初始化、配置桥接和LLM构造，完整原始异常链自动保存。不要再写十行摘要代替traceback，
也不要在初始化失败后重复跑NPU实验。

## 接线代码和边界

`sfa_recent128.hooks`安装到`vllm_ascend.attention.sfa_v1.AscendSFAImpl`真实类；由MXLLM现有MXFP4 worker patch后安装，原BF16捕获点在原RMSNorm/RoPE之后、MXFP4量化之前。不是把单算子脚本当作模型调用。

新增显式选项`single_pass_kernel=kv_split4`：

- 实际单请求query chunk为Q1..4：调用`prepare_kv_split(..., splits=4)`，原TopK2048按列分片，一次attention launch后同stream一次AIV归约。Q4为16个attention任务，不把任务数写成profiler实测active核数。
- Q>4：明确选择已验证的`load_tile4`路径处理整个prefill chunk，不把长prefill拆成大量Q4调用。每次记录requested/executed variant、实际entry、分片数、attention/reduction次数及prefill路由原因。
- 原选项`load_tile4`可直接作为旧V4对照；`--arm off`关闭热环。缺split导出、ABI2/112B或fallback geometry时直接报错，不静默退回旧路径并谎报新算子已接入。
- 不改MXFP4量化参数、权重旋转、QLI/TopK成员或冷热cache格式。当前原型仍为latent512/RoPE64/packed400/KV-head1；实际NoPE0/272B必须走独立源合同适配，不能重解释内存或靠补零query伪装兼容。

这一版接线沿用已有`accuracy_eager`验证运行时：有CPU完整数学参考、head补齐和metadata复制，不能用于服务TPOT验收。生产多请求/图/MTP/PD/PrologV3写环尚未交付，也不能把50us prepared read称为服务收益。

## 内网agent现在一次执行

1. 自主找到实际GLM checkpoint、现有engine参数、当前MXLLM/VA源码及上轮成功的KV-split库。先运行`collect_va_environment.py`并按`ENVIRONMENT_TASK.md`记录实际身份。CANN9.3是目标，未确认二进制匹配时不把旧库直接当成兼容；有匹配成功产物则复用，确需重编时只在隔离目录按实际SDK构建本候选，不覆盖vendor或原工作库，不停止业务。
2. 在隔离checkout准备安装器；保持真实worker先加载MXFP4 patch再加载本overlay。已审阅PR199 head342680fb原helper和worker，源码SHA见`evidence/pr199_source_contract.json`；该head声明的VA c09d3e20读/写hook签名与现有hook兼容。精确原文件可选`--source-profile pr199`，同时保留原SHA/AST门禁。源码不同则agent就地审阅/适配并提交真实来源，不盲改校验，也不让用户逐项搬文件。
3. 从现有启动配置生成内网engine.json（不上传带敏感路径/配置的完整内容），先运行无权重预检。

```bash
python framework_integration/run_model_probe.py \
  --engine-json "$QSFA_MODEL_OUTPUT/engine.json" --arm single_pass \
  --single-pass-kernel kv_split4 --preflight-only \
  --output "$QSFA_MODEL_OUTPUT/preflight_split4"
```

只有`COMPATIBLE_R64_PACKED400`才进入实际权重加载。UNKNOWN由agent自主读真实model类/配置补齐来源；如确认NoPE272，立即记录实际格式并按`CURRENT_FRAMEWORK_TASK.md`第3节导出原生NoPE读入口和匹配kernel/tiling源合同，不把不兼容当成已经接通，不让用户反复中转。

4. 对兼容模型执行read-only安装检查，再安装隔离overlay：

```bash
python framework_integration/install_overlay.py --mxllm-root "$MXLLM_EXPERIMENT_ROOT" \
  --source-profile pr199 --check
python framework_integration/install_overlay.py --mxllm-root "$MXLLM_EXPERIMENT_ROOT" \
  --source-profile pr199 --apply --output "$QSFA_MODEL_OUTPUT/install_split4"
```

已有旧overlay时按manifest恢复后更新，或用新checkout；不能把不同版本runtime文件混装。engine明确单请求/eager、chunked prefill、关闭prefix/PD/PCP/DCP/MTP，沿用原权重、TP和量化。Prolog默认preserve；若已有模型原路径不能提供原始BF16，只能三臂共同显式使用`--native-accuracy-preprocess`作为离线精度实验，不能只改ON、不能宣称生产性能不变。

5. 同一批真实短/长prompt、相同采样seed，分别独立进程运行OFF、旧V4、新KV-split。长prompt实际token数须超过2048；覆盖chunk边界、decode、环回绕和第二个请求重置。示例新臂：

```bash
python framework_integration/run_model_probe.py \
  --engine-json "$QSFA_MODEL_OUTPUT/engine.json" \
  --prompts-jsonl "$QSFA_MODEL_OUTPUT/prompts.jsonl" \
  --mxllm-root "$MXLLM_EXPERIMENT_ROOT" \
  --library-dir "$QSFA_KV_SPLIT_LIBRARY_DIR" \
  --arm single_pass --single-pass-kernel kv_split4 \
  --output "$QSFA_MODEL_OUTPUT/single_pass_kv_split4"
python framework_integration/collect_framework_evidence.py \
  "$QSFA_MODEL_OUTPUT/single_pass_kv_split4"
```

旧V4改为`--single-pass-kernel load_tile4`，OFF改为`--arm off`；各用新output，不复用旧日志冒充同轮。可加`--arm native`对照，但不是当前必需的额外摸高任务。

## 验收和提交

- 按每层/rank累计记录实际split attention和device reduction；二者必须一一对应。只执行大prefill fallback、从未运行split的模型不能记新后端接线通过。
- 所有真实query/head由原CPU完整FP64数学参考验证，global/worst-row<=.01、全finite。validation NPU attention=0；不通过第二次NPU attention制造参考或替代候选。大prefill fallback也纳入数学覆盖。
- 要求generation完成、collector通过、实际库SHA/ABI/entry正确，且包含真实decode的split调用。任何设备fault立即停止，保留首错，不复位卡或停服务。
- 记录每rank真实local heads及head_padding_rows。当前H64 kernel对小local-H的补齐只是精度adapter，不能据H64微基准宣称TP切分后的真实模型也有同样加速；后续服务性能必须使用真实local-H/批量/图配置验证。
- smoke通过后，按用户现有GLM评分入口对同一构建OFF/V4/KV-split运行原数据集，记录分数和样本数；不以合成L2或历史90.4/91.4代替。任务精度未测时明确NOT_MEASURED。
- 评分入口由agent在现有工作目录/历史命令/评测配置中自主定位。保留同一数据集版本、split/样本ID清单hash、tokenizer/chat template、few-shot、生成长度/截断、采样和seed；原权重/QLI/Hadamard/TopK以及原MXFP4量化参数均一致。特别不能顺带启用scale_alg=2/dst_type_max=7。
- `run_model_probe.py`默认32token生成只做接线smoke，不是整体评分。真实评分必须运行原完整评分集，保存逐样本预测/得分以及完成、失败、缺失计数；在实际评分进程中收集逐层/rank候选和归约调用，不能用另一个smoke进程的计数替代。没有找到评分入口时如实保留未测，不编造分数。
- 最终报告同样本OFF/ON分数、百分点差及得失翻转数。旧V4作为第三臂用于区分热窗策略变化和KV-split数值变化。不得把token/logprob变化或collector的WIRING_VALIDATED写成整体精度通过。
- 提交实际adapter diff、脱敏格式预检、安装/导入来源、每层计数与最大误差、完整collector输出和评分摘要到`framework_integration/evidence/kv_split_va/`。不上传权重/真实prompt/凭据/大NPZ。完整JSON可用现有无损分片办法交付，不能只写PASS。

精度验证通过以后再实施生产静态buffer/图、Prolog BF16写点及多请求/MTP/PD生命周期，用原bench_tpot OFF/V4/KV-split同构建配对验收。当前accuracy模式的延迟不作为生产性能结论。
