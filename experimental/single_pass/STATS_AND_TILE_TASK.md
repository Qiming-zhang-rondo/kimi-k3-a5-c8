# 80µs目标：先统一输出合同，再决定tile改造

当前有效实现仍是fast-pair约91.7µs。early-hot同轮91.7627对fast-pair91.7554，配对1.00001，确认无收益；不替换fast-pair，不再重复换序扫描。

## 先做不重编的2×2实验

最新runner增加 `--stats-ab`。复用已通过的fast-pair四入口或early-hot五入口ABI1库；不需要重新编AscendC。Python host桥可能按源码hash重编，这是host桥，不是新kernel。

固定同库、同输入、Q4/AIC4/TopK2048，创建四个独立Prepared：

| reader | stats OFF | stats ON |
|---|---|---|
| original C4 | 有 | 有 |
| fast-pair | 有 | 有 |

配对分别测OFF的fast/C4、ON的fast/C4，以及两种reader各自的ON/OFF。保留完整数学、最差行/finite、ON/OFF输出一致性、graph投毒及回放后stats有限正sum门禁。固定输入graph不逐次执行Python，因此不能把设备时间差归因于Python。

当前框架single-pass已经prepare(..., returnStats=False)，实际不需要返回max/sum；softmax内部计算依然保留。此前installed C4固定stats=false，而其他direct plans随 `--return-stats`。另一方面，原始direct C4在stats=true时本来也约80µs，所以不能提前断言这个开关能收回11µs。

在已确认空闲NPU执行（从成功results取实际路径/设备，不让用户逐项提供）：

```bash
python experimental/single_pass/run_single_pass.py \
  --payload-dir /tmp/sfa-direct-payload \
  --library /tmp/qsfa-single-pass-early-hot-build/libqsfa_recent128_direct.so \
  --device "$QSFA_IDLE_DEVICE" --queries 4 --keys 8192 --aic-count 4 \
  --stats-ab --timing none --output /tmp/qsfa-stats-contract-smoke
```

全部准确性通过后换新output，使用 `--timing graph --repeats 10`。本任务不要求 `--early-hot-ab`，不重复研究已打平候选。stats矩阵自己的四条plan固定OFF/ON，不依赖主args.return_stats。

上传自动生成小证据至 `experimental/single_pass/evidence/stats_contract_q4_raw.json`，包含 `stats_ab` 完整四组门禁与逐轮配对、库SHA、args。不可只上传手写表，也不需要大npz。继续标明prepared-read，不是TPOT。

## Tile优化方向（尚未交付新tile内核，不要声称完成扫描）

优先固定attention S2=128，只研究KV加载/反量化块8/16/32，16保留为已验证基线。这个tile写在C++内核，Python只能选择预编译variant，不能仅改Python参数生效。

加载块需要配套修改NZ行跨度、MXFP4 Store、RoPE Copy、UB容量、UB→L1 srcGap、hot slot数组和32位mask边界；特征块宽度16、packed行416、scale位置384不能乱改。Attention128→256涉及Cube/softmax/L1与跨核流水，改动显著更大，不作为第一步。

注意UB预算不能只统计vector的InitLocalBuffer；kernel.InitMMResBuf还有98304B cross-core UB。当前源码各InitBuffer显式申请之和（不是编译器峰值/设备容量保证）：

| 加载行数 | Vector申请 | 加上cross-core UB |
|---|---:|---:|
| 8 | 116480B | 214784B |
| 16 | 145664B | 243968B |
| 32 | 204032B | 302336B |

32行不能默认照搬双buffer配置；实际能否编译、是否需要buffer复用/深度调整，须匹配SDK/编译资源报告。不能把旧Triton可用221184B误当此AscendC kernel容量，也不能仅凭公式宣称资源足够。仅packed400范围下，输入UB从672B/行压至实际416B/行可释放空间，但仍不等于32行双buffer必然能放下。

统计实验完成后，若差距仍约11µs，优先在已有NPU/SDK工具中采集一次预热后的original/fast-pair设备流水profile。现场自主用真实工具的help/现有工作流，不猜CLI参数。重点区分标量发令、KV搬运/布局、向量解码或Cube等待；拿不到pipeline计数就保留明确缺项，不以猜测给每阶段分配µs。

本轮先收2×2结果，不让现场AI自行更改tile常量。后续源码tile候选由外部统一实现后同库A/B；保持TopK、W128、精度合同和真实模型精度任务不变。设备fault仍立即停止后续设备操作并保留首错/marker，不重置卡或换CANN。
