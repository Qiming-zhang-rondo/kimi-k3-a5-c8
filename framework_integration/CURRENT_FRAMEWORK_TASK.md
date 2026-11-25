# 当前任务：把82µs V4接到真实框架，先检查实际模型格式

用户已将重点转为框架接入与精度，tile扫描不再是前置。9ffe1e0原始证据：V4(rows32/input416/I2/O1)约82.08µs，对同轮original C4 80.47µs，配对1.01994；比fast-pair快10.5%。这是Q4/K8192 prepared-read，不能写成模型TPOT。

本轮已交付实际worker接线：`--arm single_pass --single-pass-kernel load_tile4` 调用 `QsfaSinglePassLaunchLoadTile(...,4)`，核对真实库geometry，不默默回退overlay。旧overlay/fast-pair显式可选，原始C4写量化/QLI/TopK及权重旋转均不改。

## 1. 无权重预检，先避免格式错配

缓存MXLLM源码注明GLM-5.3-Flash的NoPE路径，但实际目标必须以本机checkpoint配置及实现为准，不按模型名猜。当前已验证kernel仍仅D512/R64/packed400，**不支持R0/packed272**。Python补零query不能改变冷cache字节合同；不得重排全cache伪装成当前单遍入口。

找到已有真实模型启动配置，生成engine.json。预检只读取本地config.json，不导入Torch、vLLM或加载权重：

```bash
python framework_integration/run_model_probe.py \
  --engine-json "$QSFA_MODEL_OUTPUT/engine.json" \
  --arm single_pass --preflight-only --output "$QSFA_MODEL_OUTPUT/preflight"
```

返回0且COMPATIBLE_R64_PACKED400才进入第2步。不兼容/缺维度返回2并保留preflight.json。UNKNOWN时内网agent自主核对现有本地checkpoint及真实model类的有效参数，补充来源证据，不自动下载/执行config Python，不逐项向用户要信息。

若确认R0/272，**本轮不能宣称GLM-5.3-Flash已经接通**，自动完成第3步源码交付；不等待用户继续中转。

## 2. R64/400：隔离安装、真实模型三臂、原精度任务

1. 找到真实MXLLM源码/已验证V4库。优先复用 `/tmp/qsfa-load-tiles-build/libqsfa_recent128_direct.so` 的实际成功产物，核对完整SHA及variant4；不换CANN/vendor，不覆盖服务。桥接默认原ABI1，15指针/72B POD。
2. 隔离checkout应用 `install_overlay.py --check` 后 `--apply`。基线3ba7355d/ascend pin760d7ec；当前源不同就自主核对RMSNorm/RoPE后、MXFP4前的原BF16捕获点，以及真实attention读hook，再做必要adapter修改，不能盲改SHA校验。
3. engine保留权重/TP/量化/TopK，初始精度范围为eager、max_num_seqs1、chunked prefill、Q<=512，关闭prefix/PD/PCP/DCP/MTP。默认preserve原preprocessing；若原模型使用PrologV3且不输出原BF16，OFF/native/V4三臂必须共同显式使用 `--native-accuracy-preprocess`，并在权重转换前选NATIVE。不能只改ON，也不能用此测试验收服务性能。
4. 独立进程顺序跑OFF/native/V4。原始BF16热环按层维护tags，容量保留W+chunk-1；不将C4解码值当原BF16。至少短prompt、>2048真实长prompt、chunk边界和多步decode/ring回绕。

```bash
python framework_integration/run_model_probe.py \
  --engine-json "$QSFA_MODEL_OUTPUT/engine.json" \
  --prompts-jsonl "$QSFA_MODEL_OUTPUT/prompts.jsonl" \
  --mxllm-root "$MXLLM_EXPERIMENT_ROOT" \
  --library-dir /tmp/qsfa-load-tiles-build \
  --arm single_pass --single-pass-kernel load_tile4 \
  --output "$QSFA_MODEL_OUTPUT/single_pass_v4"
python framework_integration/collect_framework_evidence.py "$QSFA_MODEL_OUTPUT/single_pass_v4"
```

OFF/native分别用 `--arm off` / `--arm native`，各用新输出目录；native收集加 `--native-only`。数据和权重保留内网。

每64heads一组candidate attention；H不足64在精度模式补齐，H>64记录实际多次调用。每个真实query/head对CPU完整FP64数学参考，全局/最差行<=.01、全finite；validation NPU attention=0。不使用第二次NPU attention制造候选或参考。collector核对V4实际导出/variant/geometry/库SHA与每次调用覆盖，生成完成本身不能算接线通过。

接线通过后再跑用户原GLM评分任务，同构建OFF/ON、相同数据集/seed/采样与样本数，不能用历史90.4/91.4替代新基线。提交 `framework_integration/evidence/v4_model_summary.json`，包含预检、安装来源、逐层rank覆盖/最大误差、实际调用数、真实任务评分及未测项。

## 3. R0/272：一次收齐对应源合同，准备真正NoPE适配

不要启动大量权重后再失败。提交小型格式证据至 `framework_integration/evidence/model_format_preflight.json`，并在现有容器/源码仓中自主定位实际NoPE native C4 reader，提交下列最小完整源文件与SHA manifest至 `native_nope_source/`：

- Python调用/C++绑定/ACLNN真实签名，以及实际query/cache/scale/page/index格式、shape/dtype、rope_head_dim、scale和quantMode参数；不含权重和prompt。
- NoPE272的GM行字节布局与E2M1/E8M0位置、量化group大小、UB对齐/stride、QK/PV维度和数据类型；实际匹配的entry、vector、cube、kvcache/common必要依赖。
- host tiling producer及device POD真实定义（sizeof/offset/消费字段），构建入口及SDK/架构/flag。无法取得的私有定义明确标缺，不能借R64/400或BF16144B结构猜ABI。
- 可复用的最小NoPE原生读调用与独立反量化数学参考、小型输出误差/真实编译记录，源代码分文件提交，不上传大log/npz/私有SDK或整个权重配置。

现有82µs V4只作为R64分支基线，不把272B重解释为400B；后续NoPE内核需独立编译和实测。普通文件定位/依赖问题内网agent自行处理，不逐项让用户搬运。

## 范围与故障

当前交付accuracy_eager，含CPU全数学、metadata复制和head补齐，不是生产图/并发/MTP/PD版本。精度通过后再做静态buffer、Prolog BF16写点、batch/图与缓存生命周期，并用原bench_tpot验收；不能把单算子82µs推广成端到端性能。

设备fault立即停止后续设备工作，保留首错/marker，不擅自重置业务卡或切换软件栈。框架接入不再等待更多tile摸高。
