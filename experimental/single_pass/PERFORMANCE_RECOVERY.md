# 当前任务：先消除 W0 的 3.75× 差距

收到 e1af1f5：上传摘要写全部精度 PASS；W0=301.78µs、W128=339.22µs、installed C4=80.44µs、two-pass=107.3µs。完整构建/运行参数、original_c4_direct 和配对项尚未上传。保留原报告，不将其简单解释成热行 C4 重复解码。

**W0 已是 C4 的3.75×，热窗只额外37.44µs。首先排查基础读取的并行/构建路径，不继续盲扫 hot DMA。** 本轮自主执行到底，普通依赖/编译问题自行解决，不让用户来回传字段。不要停止业务、换CANN、覆盖旧库或放宽精度门槛。

## 已交付的修改

- `--aic-count` 默认自动选择 Q 的最大受支持整除数（上限16）；Q4默认4，仍可显式1作对照。实际值写进 args / launch_geometry。原默认1会把Q4分给一个AIC依次处理，与四路并行的差距高度吻合；**当前不能证明现场确实用了1**。
- CMake默认Release，ASC显式 `-O2`，输出 compile_commands。此前未显式设置优化；是否真的O0要看实际命令，不能凭猜测归因。
- W0在C入口直接调用独立原始C4 kernel。此为明确的window0旁路，不伪装成仍在执行hot候选；原始direct对照继续独立保留。
- 候选禁用热窗时不再写recent token行映射，也不进入hot overlay；原C4加载与数学不变。W128仍是一次attention。

## 立即执行，只有两步

**1. 先用已有结果/已有库确认根因，不需要先重编。**

找到刚才e1af的真实results.json（不要只读手写11行摘要），纯CPU运行：

```bash
python experimental/single_pass/collect_run_evidence.py "$QSFA_PREVIOUS_RUN/results.json" \
  --output experimental/single_pass/evidence/e1af_raw_evidence.json
```

保留原始 args.aic_count / timing / queries，bridge library SHA，五项精度原始值，graph gate，以及 **original_c4_direct、W0、W128 各轮配对**。不存在的字段保留null，失败项不得隐藏。无需上传12MB npz。

然后用当前runner和**已有ABI1库**（接口兼容）跑Q4/K8192显式 `--aic-count 4 --timing graph --repeats 5`，与旧记录的AIC1对照；若原记录已经是4，再补一次显式1判断扩展，不做更多形状扫描。保持输入seed、环境、库不变。当前runner默认4；这一步显式写4避免歧义。

必须分别看 original_c4_direct 和 candidate W0：两者都慢优先查launch/实际编译flag；只有candidate慢才归因recent bookkeeping/资源占用。不要再把W0慢说成BF16热环开销。

**2. 用新代码隔离重编并验证。**

新输出目录构建，保留旧build与库：

```bash
cmake -S experimental/single_pass -B /tmp/qsfa-single-pass-perf-build \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DCMAKE_BUILD_TYPE=Release \
  -DSFA_SDK_ROOT="$ASCEND_HOME_PATH"
cmake --build /tmp/qsfa-single-pass-perf-build -j8
```

从compile_commands提取原始和candidate两个ASC TU的实际 `-O*` / arch，提交小摘要；不上传完整环境变量。有效优化flag看最终命令，不仅看CMake参数。

使用新库、显式AIC4，先 `--timing none`；全部精度门禁通过后再graph配对。运行命令沿用README，payload、设备和SDK路径自动从刚才成功的命令/结果读取，用户无需再提供。新runner/core数以真实args与库SHA为证据。保留原始C4 direct和installed C4，不能只报W128/旧历史比值。

## 返回与继续

每轮用收集器提交 `experimental/single_pass/evidence/` 下小型原始证据；另写一张表：library SHA简称、AIC、优化flag、original C4 direct、candidate W0、W128、installed C4、two-pass及其同轮配对比值。标明prepared-read-only。

根因结论区分已证实与待证实。W0若恢复到原始C4同级，再判断37µs热环开销是否仍存在；若W128仍慢于现有two-pass，保留two-pass作当前有效实现，后续才做hot全热tile旁路/批量DMA，不能仅凭单次attention更少就称更快。

本轮无新增精度豁免。设备fault立即停止，保留marker，不继续测性能。
