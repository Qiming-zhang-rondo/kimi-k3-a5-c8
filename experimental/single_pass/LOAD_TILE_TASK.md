# 直接执行：8/16/32行加载块同库扫描

本轮交付实际源码。保持attention S2=128、原TopK2048、W128、原量化和精度合同，保留91µs fast-pair基线。新增五种独立编译布局：

| ID | 加载行数 | 输入每行分配 | 输入/输出buffer数 | 显式UB总申请 |
|---|---:|---:|---|---:|
| baseline | 16 | 672B | 2/2 | 243968B |
| 1 | 8 | 672B | 2/2 | 214784B |
| 2 | 16 | 416B | 2/2 | 235776B |
| 3 | 16 | 416B | 2/1 | 216192B |
| 4 | 32 | 416B | 2/1 | 247936B |
| 5 | 32 | 416B | 1/1 | 234624B |

表包含98304B cross-core UB，但不是编译器峰值/硬件容量保证。32行没有照搬双输出buffer。完整联动NZ跨度9/17/33、slot数组、热DMA、MXFP4 VF、RoPE、UB→L1间隔和32位mask；特征宽16、packed stride416/scale384不变。所有类/kernel/run及三个布局相关VF helper独立命名，避免弱符号串版。默认及框架入口不变。

## 执行

281881d记录的故障实验只改了循环和缓冲三处，不能证明根因是UB；还遗漏数组/NZ/mask联动。旧源码已恢复，保留旧库。不要在故障状态继续跑：先按现场既有流程确认设备已恢复或选择另一空闲健康设备，不凭npu-smi正常就绕过fault marker，不擅自重置卡或删marker。

复用前轮已通过的容器/SDK/payload。普通编译、include、API问题自主解决，不让用户逐项中转。新build隔离：

```bash
cmake -S experimental/single_pass -B /tmp/qsfa-load-tiles-build \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DCMAKE_BUILD_TYPE=Release \
  -DSFA_SDK_ROOT="$ASCEND_HOME_PATH"
cmake --build /tmp/qsfa-load-tiles-build -j8

python experimental/single_pass/run_single_pass.py \
  --payload-dir /tmp/sfa-direct-payload \
  --library /tmp/qsfa-load-tiles-build/libqsfa_recent128_direct.so \
  --device "$QSFA_IDLE_DEVICE" --queries 4 --keys 8192 --aic-count 4 \
  --load-tile-ab --load-variants 1 2 3 4 5 \
  --timing none --output /tmp/qsfa-load-tiles-smoke
```

确认AIC/AIV真实编译并保留资源报告。新variant仅packed400 C4/BF16；不扩展到FP8格式。主计时默认stats OFF，全部候选、fast-pair及original C4同合同。源码包不依赖猜测安装包tiling，继续ABI1/15指针/72B POD。

完整数学.01、对fast-pair .001、全热/混合排列及finite/最差行全部通过后，换新output运行相同参数 `--timing graph --repeats 10`。主TopK顺序保持；原始C4和fast-pair对照各自配对，图投毒门禁保留。只先跑Q4/K8192，不自动扩大形状扫描。

上传自动产生的evidence_summary.json至 `experimental/single_pass/evidence/load_tiles_q4_raw.json`，保留完整load_tile_ab、实际库SHA/args、各variant门禁/配对逐轮、编译资源/失败。80µs是目标，不提前宣布达到；prepared-read不等于TPOT。候选慢则保留91µs基线。

设备故障立即停止后续设备调用，保存首错/marker，不用其他variant继续撞卡。宿主已运行实际loader/VF行循环/NZ/CopyOut的五配置150场景及UBSan边界检查，仍不代替A5精度和性能。
