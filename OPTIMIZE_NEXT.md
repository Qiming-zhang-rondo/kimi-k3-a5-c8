# 新候选：按query分核 + 批量融合尾部

此Q4阶段已由56ab1fb/86ef978回传实测完成。当前优先执行
[SHAPE_SWEEP_NEXT.md](SHAPE_SWEEP_NEXT.md) 的Q32–1024形状扫描；以下保留历史复现命令。

用户要求继续优化，优先给出新的实际性能结果。现有7ca3b03基线保留。
本轮两处源码变更：

- direct启动支持1/2/4个AIC（对应2/4/8个AIV），固定Q4在原内核已有分核逻辑下分配。
  tiling.usedCoreNum、实际launch count和每AIC独立442368B scratch必须同步；ABI已升3。
- 冷态合并可编译为4/8/16行一批，默认4。首测16，将每32头的8轮DMA/fence/VF减为2轮；
  stats分区按8个float对齐，尾部仍用带实际字节长度的DataCopyPad。
  BF16批量scratch由4160B增至16512B，宿主预算不代替SDK实际UB检查。
- 数学、舍入、presence及非法统计检查不变；不改原生C4量化/读写，不换TopK或热窗策略。

## 首先只跑4AIC + rows16

在上一轮成功的同一容器/SDK中，从本分支根目录执行。新ABI必须新建目录重编，
不要覆盖 `/tmp/sfa-direct-c4-ab-v3/build` 或安装vendor。
沿用上一轮成功的ASC_DIR、编译器和SDK include配置（可从旧build/CMakeCache.txt读取），
不可猜另一个SDK路径。下面是默认环境已包含同一ASC配置时的命令：

```bash
set -e
SFA_OPT_ROOT=/tmp/sfa-direct-c4-opt-v4
: "${ASCEND_HOME_PATH:?沿用上一轮成功的同一SDK}"
: "${SFA_DIRECT_DEVICE:?设置已确认空闲的npu设备}"
python3 tools/run_logged_stage.py --stage candidate_build \
  --work-dir "$SFA_OPT_ROOT/evidence" -- \
  cmake -S standalone -B "$SFA_OPT_ROOT/build-r16" \
    -DSFA_BUILD_PHASE_C=OFF -DSFA_BUILD_TORCH_DIRECT=ON \
    -DSFA_PROBE_TEMPLATE_MODE=0 -DSFA_MERGE_TILE_ROWS=16 \
    -DSFA_SDK_ROOT="$ASCEND_HOME_PATH"
python3 tools/run_logged_stage.py --stage candidate_build \
  --work-dir "$SFA_OPT_ROOT/evidence" -- \
  cmake --build "$SFA_OPT_ROOT/build-r16" --target sfa_torch_direct -j8

python3 direct_bridge/run_direct_ab.py \
  --payload-dir /tmp/sfa-direct-payload \
  --bridge-library "$SFA_OPT_ROOT/build-r16/libsfa_torch_direct.so" \
  --direct-aic-count 4 --device "$SFA_DIRECT_DEVICE" \
  --timing graph --stages --warmup 20 --iterations 100 --repeats 10 \
  --fault-marker /tmp/sfa-direct-c4-ab-v3/DEVICE_FAULT.json \
  --output "$SFA_OPT_ROOT/aic4-r16"
```

payload使用上轮结果中已验SHA的真实路径，不重新下载/改旧payload。
同一设备继续使用上轮故障marker路径；不能以新目录掩盖故障。
新Python进程将JIT编译匹配ABI3的Torch薄桥；不能加载ABI2库或临时改ABI校验。
`bridge_configuration`和每个direct实现的`direct_configuration`应分别记录真实库的
ABI/merge_tile_rows与实际aic_count/aiv_count/workspace_bytes，确认是3/16及4/8/1769472B。

## 一次运行直接返回结果

原分区/写入、非零query0.001、零query现算RNE证明、数学0.01、pipeline及graph投毒门禁仍保留；
全部通过才做四臂完整流水线五组配对。不要另跑一遍accuracy或扩展模型测试。
优先比较 `direct_fused_vs_installed_native_twopass`，不能仅胜过direct两遍就称优化成功。

`--stages`在完整流水线配对后另测C4冷读、已安装BF16热读、direct热读、原外部merge、
direct融合热读+merge。它们用当次真实冷态和prepared索引；独立阶段不含写入/分区，
不能相加代替完整流水线，也不能当额外加速收益。

把本次results.json、构建/执行日志和小表直接推回本分支：
四臂pipeline event/wall中位数，融合/installed与融合/direct的10轮比值中位数[min,max]，
五个stage的event中位数，配置元数据、全部门禁状态与库SHA。无需上传大NPZ或完整环境变量。
无论快慢都先交这组结果；不要自行扩大扫描或推导“去掉桥开销后的估计速度”。

若SDK确认rows16超UB但无设备调用，保留日志并仅将编译参数退到8重新构建；
不要删除缓冲或掩码来消编译错误。若精度门禁失败先停在该门禁，不标MEASURED。
若设备故障立即停止，保留marker，不reset、不换目录重试。
