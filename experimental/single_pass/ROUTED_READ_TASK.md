# 冷热互斥读取：保留94µs版本，同库配对测试

用户继续要求优化读取端。本轮只改数据源路由，不再增加独立 attention、TopK partition 或 LSE merge。已有框架真实模型精度任务可以使用原已验证 overlay 库继续执行，不被这个候选阻塞。

## 改了什么

- 原 `QsfaSinglePassLaunch` / overlay 保留为独立入口，不替换已验证基线。
- 新 `QsfaSinglePassLaunchRouted` 在每个最多16行的搬运块内按行路由：冷行读取400B C4并反量化；热行只读取真正的BF16 latent/RoPE，不读取该行C4。
- 全冷块调用原反量化；全热块完全跳过反量化；混合块VF跳过热行，RoPE也只复制冷行连续段。禁止仅跳过DMA而解码未初始化的packed槽。
- 两种精度装入同一个NZ tile，后续共用原始QK/online softmax/PV。Q原本为BF16，不反量化Q。原TopK成员和主计时顺序不变。
- 查询位置/窗口起点提升到每次ProcessSparseKv外层。热DMA移到冷解码之前；保留VECOUT复用所需V→MTE2所有权同步及MTE2→MTE3完成同步，不能擅自删除等待称提速。
- 新旧kernel、类、run helper和导出均独立命名，同一库内直接配对；不在同一Python进程加载两份导出同名符号的库作假A/B。

## 内网自主执行

1. 沿用上次通过C4的容器/SDK/payload及空闲NPU。从已有成功命令和results.json读取实际路径，无需让用户重报。不要停止业务、切换CANN或覆盖旧build/library。
2. 新目录重编**当前整库的三个ASC TU**。示例路径可按实际可写空间调整，SDK必须和已通过环境一致：

```bash
cmake -S experimental/single_pass -B /tmp/qsfa-single-pass-routed-build \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DCMAKE_BUILD_TYPE=Release \
  -DSFA_SDK_ROOT="$ASCEND_HOME_PATH"
cmake --build /tmp/qsfa-single-pass-routed-build -j8
```

确认AIC/AIV真实编译；遇MicroAPI控制流/编译API问题按本机SDK自主修复，保留行级skip逻辑及原始baseline。Mac宿主测试不等于device编译通过。

3. 首先只跑Q4/K8192/AIC4，准确性模式。下面三个路径变量代表从原成功记录取得的真实值；不是让用户补材料：

```bash
python experimental/single_pass/run_single_pass.py \
  --payload-dir "$QSFA_VERIFIED_PAYLOAD" \
  --library /tmp/qsfa-single-pass-routed-build/libqsfa_recent128_direct.so \
  --device "$QSFA_IDLE_DEVICE" --queries 4 --keys 8192 --aic-count 4 \
  --routed-ab --return-stats --timing none --output /tmp/qsfa-routed-smoke
```

`--routed-ab`同时运行原始C4、W0、旧overlay、新routed和安装包控制组。它额外用同一TopK成员的“热行集中”和“冷热交错”两种排列检查全热/全冷/混合16行块；仅正确性对照改变顺序，主计时仍使用原TopK。三组输出均对完整独立FP64数学参考，routed与overlay另有.001一致性门禁。原.01数学门槛不放宽，finite/最差行不豁免。

4. 全部通过后，用新输出目录、相同参数改 `--timing graph --repeats 10`。每条候选必须通过输出投毒的graph门禁；保留 `single_pass_w128_routed_vs_overlay` 的同轮AB/BA，不能直接用历史94.66µs算收益。比较两者对同轮installed C4和original C4的比值。仍是prepared-read-only，不含写环，不是模型TPOT。
5. 若Q4确实赢，再补Q1/K8192、Q32/K8192、Q128/K32768三组；按Q自动选AIC，不做更大盲扫。如果routed更慢，保留overlay有效实现，报告实际瓶颈，不强行上线。
6. 提交机器生成的 `evidence_summary.json` 至 `experimental/single_pass/evidence/routed_q4_raw.json` 等小文件，包含实际库SHA/参数/原始门禁/每轮配对，不上传大npz。另写小表包括overlay/routed/原始C4/installed C4及prepared-read范围。

普通编译/路径问题继续自主解决；设备fault立即停后续设备调用、保留首错及marker。禁止重置卡或用错误后计时称MEASURED。

## 当前交付状态

已写源码；宿主模型直接编译实际读取和VF行路由，覆盖全冷/全热/混合、物理pair逆序、尾行、tag和ring回绕，并检查热行C4既不搬运也不读取。MicroAPI浮点算术/设备同步/性能仍须A5验证。新入口没有预报延迟。
