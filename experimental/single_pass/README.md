# C4 冷端 + BF16 recent128：原生单次 attention 原型

**e1af回传摘要报精度通过，但W0=301.78µs、W128=339.22µs，性能尚未达标。**当前优先执行 [PERFORMANCE_RECOVERY.md](PERFORMANCE_RECOVERY.md)，补齐真实参数并修复W0路径；新默认AIC/Release/O2/旁路修改尚待A5测量。此原型不再依赖已算好的 cold output/max/sum，不在W128候选里调用原生 cold attention，也没有第二次 hot attention 或外部 LSE merge。

输入为原始合并 BF16 query、原 packed400 C4 cache、原 TopK、BF16 热环及位置 tags。沿原生 C4 的 gather/反量化流水，把命中的 BF16 热行写进相同 NZ tile；后面的 QK、online softmax、PV 和最终归一共用一次计算。TopK 不改变。

首版为了保住原生流水，**热 token 暂时也会读取和反量化 C4，再用真正的 BF16 热值覆盖**。这有额外搬运，但不会多算一次 attention。取消这些冗余读取是后续优化项，不先承诺性能。

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
