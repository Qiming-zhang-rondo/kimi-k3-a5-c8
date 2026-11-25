# C4 冷端 + BF16 recent128：原生单次 attention 原型

最新实际tile候选见 [LOAD_TILE_TASK.md](LOAD_TILE_TASK.md)。`--load-tile-ab`在同库测试8/16/32行加载块及buffer对照；128行attention块不变，当前16行91µs实现保留。新候选宿主布局/UBSan检查通过，尚待A5编译和性能；不是此前只改三常量的故障版本。

**452fa69原始配对：fast-pair91.55µs，比overlay97.60µs快6.2%，全部精度/graph门禁通过。**当前有效读取实现为fast-pair；新增第五入口early-hot尝试提前发起热DMA并共用输入同步，按 [EARLY_HOT_TASK.md](EARLY_HOT_TASK.md) 作独立同库配对。新候选尚未上卡，不预报80µs。框架真实模型精度任务可以同时用原已验证库推进。此原型不依赖已算好的 cold output/max/sum，不在W128候选里调用原生 cold attention，也没有第二次 hot attention 或外部 LSE merge。

输入为原始合并 BF16 query、原 packed400 C4 cache、原 TopK、BF16 热环及位置 tags。沿原生 C4 的 gather/反量化流水，把命中的 BF16 热行写进相同 NZ tile；后面的 QK、online softmax、PV 和最终归一共用一次计算。TopK 不改变。

旧overlay为了保住原生流水，**热token也读取和反量化C4，再用真正BF16热值覆盖**。这个入口保留作对照。routed入口只读取每行所需精度，但上轮慢于overlay；新增fast_pair使全冷pair直接原生DMA，跳过slot解析与数组写入。用 `--fast-pair-ab`测试全部三种读取路径，默认仍是已验证overlay。附加导出不改变15指针/72B tiling；Linux默认overlay仍可使用旧库，fast_pair必须重编四入口库。

最新可选 `--early-hot-ab` 同时保留以上对照并测试热DMA提前入队；需重编五入口库。现有fast-pair的类/入口和同步顺序不变。框架默认reader也不自动切换到实验候选。

753ad93回传early-hot与fast-pair同轮91.7627/91.7554µs，换序无收益。当前下一步按 [STATS_AND_TILE_TASK.md](STATS_AND_TILE_TASK.md)：`--stats-ab`复用已有库测original/fast-pair×stats OFF/ON，不重编ASC，统一框架实际无需返回统计量的合同；再决定KV加载tile8/16/32的资源和源码改造。新的tile内核尚未交付，不能把任务说明当成已经完成扫描。

## 范围与来源

- 初始 Q4/K8192，B1、H64、D512/R64、TopK2048、page128、window0/128。
- `native_c4_source/` 是用户交付的实际 C4 source；41 个源文件的 MANIFEST SHA256 已核对。源快照 `2a2beb51b4db2ae02c9c120527cbc04c4e99bbb7`。
- 原始 C4 kernel 另编译为独立对照入口；候选 vector/kernel 位于本目录 `arch35/`，原文件不改动。
- window0直接调独立原始C4入口；Q4默认自动选4 AIC，仍可显式1作并行对照。构建默认Release并显式ASC -O2。
- 原安装包 C4 tiling 类型定义未包含在交付中，因此本目录明确采用 **新 standalone ABI1 / 72B POD**，host 和 kernel 共同重编。不能把它传给安装包 ACLNN，也不能复用旧 BF16 144B POD。
- 原生 packed400 分支的模板为 BF16 query、INT8 storage、TEMPLATE_MODE=1、TND/PA_BSND、IS_SPLIT_G=false；不是 DT_FLOAT4 模板入口。此分支不使用 GM user scratch，保持 1B 非空占位。禁止走未验证的 SyncAll/空 padding 请求路径。
- host 原型在运行前验证物理页表、原 TopK、causal 范围、BF16 tags；非法 recent tag 的设备 C4 fallback 仅防止非法地址，不能算成功启用热窗。
- 首轮设备 smoke 使用全有效 TopK2048、每行冷1920/热128。CPU 参考覆盖更多冷热/空行情况，不等于这些形状已在 NPU 验证。

## 构建与 smoke

在**已经验证过 C4 的容器和匹配工具链**里执行，不 source 另一版 vendor。复用既有 ASC 环境，隔离目录构建：

```bash
cmake -S experimental/single_pass -B /tmp/qsfa-single-pass-build \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510 \
  -DSFA_SDK_ROOT="$ASCEND_HOME_PATH"
cmake --build /tmp/qsfa-single-pass-build -j8
```

若现场 ASC 配置需要原实验的 `ASC_DIR` / `CMAKE_PREFIX_PATH`，从已成功的 CMakeCache 读取同一 SDK 对应项；不要引入另一套 CANN 或伪造架构宏。

```bash
python experimental/single_pass/run_single_pass.py \
  --payload-dir /path/to/verified/hotkv \
  --library /tmp/qsfa-single-pass-build/libqsfa_recent128_direct.so \
  --device npu:0 --queries 4 --keys 8192 --aic-count 4 \
  --timing none --output /tmp/qsfa-single-pass-smoke
```

`--device` 必须显式选择已确认空闲的卡；上面 `npu:0` 仅为示例。旧 hotkv payload 保持 pinned SHA 校验。新 bridge 使用当前 stream、当前输入地址、预分配输出与 tiling，旧 ABI4 hot-merge 库继续保留。

原始 C4 / 单遍 window0 / 单遍 window128 均先做独立完整 softmax 数学对照；原生两遍路径仅作额外诊断。**单遍只在最终归一后舍入，不要求逐位等于两遍中间 BF16 舍入后的结果。**数学门禁沿用 .01（全局及最差行、全部 finite），原始 C4 与 window0 另作直接对照，报告原始误差。

精度全通过后可另用新输出目录运行 `--timing graph --repeats 10`。图执行先投毒输出确认外部 ASC launch 确实被捕获，再做同轮 AB/BA；计时范围仅 prepared read，不含写环、模型和服务。设备错误出现即停止，保留 fault marker，不能继续跑性能。

## 本地验证

- 实际候选 CopyIn/Overlay 函数体经宿主 C++ DMA 模型编译运行，检查物理地址倒序 pair、NZ 特征分块、16行边界、冷热混合和保护行。
- 新 POD 实际 C++ 编译、14 个真实 consumer 字段、MXFP4 400B 布局、同 TopK 统一 softmax 与错误输入合同。
- 原始/候选 entry 的 host pass 编译隔离及错误架构拒绝。
- 帮助与 Python 语法检查。

以上均不冒充 SDK/device 或性能验证。现场任务见 [A5_EXECUTION_TASK.md](A5_EXECUTION_TASK.md)。
