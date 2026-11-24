# 撤销 5407a8e 的性能结论，接入真实 C4 三臂测试

**d92f2f59 已确认设备库与Torch桥编译成功，原分区/写环/零Q及已安装两遍均通过。
现场C4冷输出和最终merge输出为BF16，旧FP32-only断言属于接线假设错误。
当前新增独立BF16 IO变体：在同一内核内将BF16冷态转FP32运算，最后按RNE写BF16，
不增加Python转换算子。FP32冷入/FP32输出旧变体保留，max/sum始终FP32。**

本轮BF16适配本地169项：168通过、1项桌面致命信号测试跳过，详情见
`direct_bridge/bf16_io_validation.json`。新BF16变体仍需现场SDK编译及A5严格对照；
原d92f的FP32-only库不能直接复用，必须重编ABI2后使用新Python进程/新输出目录。

审查对象：`5407a8e8175f750f859e061a5b993815a8756477`，实际测量提交 `08f2e0d`。
**305.84µs、3.494×、2.424× 不具备性能验收资格，不能据此判断融合更慢或更快。**
这不是单纯的 eager/graph 差异。当前 CMake 实际构建的 `sfa_pc_fused_bench.asc`：

- 将 `gSize` 从64写成1，实际计算行数与宣称的 Q4/H64 不符；不能仅凭分配大小认定测了64头。
- 将热端 compact128 改成2048索引，没有原来的冷热分区与物理256槽热环合同。
- `coldSum=0` 却 `cold_present=1`，有效冷态不合法；零Q的2048项热sum也超过固定helper支持128项的范围。
- 全零输入、不检查输出/NaN就开始计时；没有任何真实 MXFP4 冷端读取。
- 把仅一次热读/给定冷态合并与历史整条C4流水线延迟相除，不是同轮同范围比较。
- 大量ACL返回码未检查，固定device0并在结束时reset设备，违反当前执行规则。

其他草稿还包含空函数体的“融合入口”、同编译单元包含两套同名BaseApi定义等问题。
空入口不是上述实际CMake目标，不能反过来声称305µs一定测的是空kernel。
错误文件已原样归档至 `standalone/archive/5407a8e/`，原入口显式拒绝构建；
`SFA_BUILD_PHASE_C=ON` 同样拒绝，不允许旧缓存继续执行这批程序。

`actualLenDimsQ/KV=4/8192`、KV按64头分配也违背合同；本固定路径未消费前者，
后者超量分配本身不证明越界。真实错误按实际源码消费者判断，不猜测故障原因。

## 保留已通过结果

`5a36184` 的8项 `DEVICE_SMOKE_VALIDATED` 仍有效：它们验证给定合法FP32冷态加原生BF16热读。
本轮没有修改设备数学实现或抹去这些结果。它们没有验证真实C4流水线、完整原生两遍比较或性能。

## 本次实际新增的代码

- `SFA_BUILD_TORCH_DIRECT`：构建 `libsfa_torch_direct.so`，原版和融合版各占独立ASC编译单元。
  普通C++只看19指针C ABI，不能在一个编译单元同时包含两套原生模板。
- `experimental/torch/sfa_direct_bridge.cpp`：真正调用两个ASC host wrapper，不使用只有声明的新ACLNN。
  `Prepared`预分配真实POD、scratch与output；每次传本步新生成的冷态指针。
  `bf16_io=True`选择BF16冷入/最终BF16，默认False仍为FP32；direct ABI已升为2，旧库必须重编。
  取得当前torch_npu stream并保持提交顺序，recordStream保护异步buffer生命期；不重置/同步设备。
- `direct_bridge/run_direct_ab.py`：继承原BF16热窗runner，保留原生C4写/读、BF16写环、
  hot_store_partition和原Triton merge，只替换热读/融合部位。
- 旧18文件payload固定为 `4a5fd67b9a78766bebfa564782ea2193bc22af1a` 并逐文件SHA256校验。
  不修改旧payload、不在计时外缓存冷读、不把BF16热读重命名为C4 W0。

本机没有Torch-NPU/CANN/A5。新增代码尚需现场编译与设备验证；宿主接口测试不等于新性能结果。

完整本地回归150项：149通过，1项桌面致命信号测试跳过。包含真实桥CPP的宿主API边界模拟、
两份实际ASC wrapper的独立编译单元转发、真实CMake旧入口拒绝/SDK缓存路径检查、
四臂计时门禁和SHA下载校验。原18文件也已实际下载并逐项验证。
机器可读范围与日志见 `direct_bridge/host_validation.json`、`host_validation.log`。

## 内网直接连续执行

在已跑通原版与融合smoke的同一容器和SDK下，从仓库根目录执行。继续保留已确认的
ASC_DIR/编译器参数，不能换旧vendor或升级CANN绕过普通构建问题。构建使用新目录：

```bash
set -e
SFA_DIRECT_ROOT=/tmp/sfa-direct-c4-ab-v2
mkdir -p "$SFA_DIRECT_ROOT"
python3 tools/run_logged_stage.py --stage candidate_build \
  --work-dir "$SFA_DIRECT_ROOT/evidence" -- \
  cmake -S standalone -B "$SFA_DIRECT_ROOT/build" \
    -DSFA_BUILD_PHASE_C=OFF -DSFA_BUILD_TORCH_DIRECT=ON \
    -DSFA_PROBE_TEMPLATE_MODE=0 -DSFA_SDK_ROOT="$ASCEND_HOME_PATH"
python3 tools/run_logged_stage.py --stage candidate_build \
  --work-dir "$SFA_DIRECT_ROOT/evidence" -- \
  cmake --build "$SFA_DIRECT_ROOT/build" --target sfa_torch_direct -j8 --verbose

# 优先复用已有且SHA匹配的hotkv目录；以下下载器只取固定18个文件，不执行NPU代码。
# 需要代理时沿用该机器已可用的https_proxy；不关闭TLS校验。
python3 direct_bridge/fetch_pinned_payload.py --output "$SFA_DIRECT_ROOT/hotkv"

# 现场自主确认空闲设备后设置，例如 npu:0；不要停止服务或绕过既有故障标记。
: "${SFA_DIRECT_DEVICE:?设置已确认空闲的npu设备}"
python3 tools/run_logged_stage.py --stage accuracy \
  --work-dir "$SFA_DIRECT_ROOT/evidence" -- \
  python3 direct_bridge/run_direct_ab.py \
    --payload-dir "$SFA_DIRECT_ROOT/hotkv" \
    --bridge-library "$SFA_DIRECT_ROOT/build/libsfa_torch_direct.so" \
    --device "$SFA_DIRECT_DEVICE" --timing none \
    --fault-marker "$SFA_DIRECT_ROOT/DEVICE_FAULT.json" \
    --output "$SFA_DIRECT_ROOT/accuracy"

# 精度命令退出0才继续；此轮仍会重新执行控制门禁。
python3 tools/run_logged_stage.py --stage performance \
  --work-dir "$SFA_DIRECT_ROOT/evidence" -- \
  python3 direct_bridge/run_direct_ab.py \
    --payload-dir "$SFA_DIRECT_ROOT/hotkv" \
    --bridge-library "$SFA_DIRECT_ROOT/build/libsfa_torch_direct.so" \
    --device "$SFA_DIRECT_DEVICE" --timing graph \
    --warmup 20 --iterations 100 --repeats 10 \
    --fault-marker "$SFA_DIRECT_ROOT/DEVICE_FAULT.json" \
    --output "$SFA_DIRECT_ROOT/performance"
```

已有输出目录不覆盖。普通构建/API问题在现场继续修复并提交源码和完整日志，避免用户逐项中转。
设备故障立即停止后续测量，保留marker，不reset、不重试；不要换目录掩盖同一设备故障。

## 必须先通过的门禁

1. B1/Q4/K8192/H64/Nkv1，真实冷1920/热128互斥分区；热环256槽，compact128全有效。
   桥内热tiling固定g64、KV256、长度数组大小1、POD144B；冷输出保持现场BF16或FP32，
   max/sum必须FP32且sum合法，最终输出与已安装merge类型一致。
2. 重编原版对已部署BF16热reader，融合对真实原生冷态加已部署热读的两遍结果：
   整体与最差行 L2 均≤0.001；各自匹配存储精度数学参考均≤0.01。
3. 零Q实际热质量128、冷质量1920；检查动态分区、cache写入和prepared/pipeline一致性。
4. graph捕获后将桥的hot输出/statistics及最终output投毒，再replay。不能用上次残留输出
   冒充ASC kernel已捕获；失败不能退回eager后仍标记graph MEASURED。

所有门禁通过才测本轮四臂：`c4_w0`、`installed_native_twopass`、
`direct_native_twopass`、`direct_fused`。其中已安装两遍是原有真实基线，direct两遍另用于诊断桥接开销。
四臂每次都执行相同范围的write+dynamic完整流水线，必测五组配对：已安装两遍/W0、
direct两遍/W0、融合/W0、融合/已安装两遍、融合/direct两遍。
保存每轮event/wall和比值，报告min/max而不是置信区间。计时结果中不使用历史87.53/126.17常量。
不能只让融合赢过一个可能变慢的direct两遍，就声称胜过已部署两遍；最终以本轮已安装基线比较。

额外记录profiler：独立merge是否消失、实际kernel数、host queue提交、UB/寄存器和各阶段耗时。
若融合确实慢，保留最快通过的实现，再依据同轮profile优化，不预先承诺达到某个µs目标。

更多桥的参数映射与生命周期见 `experimental/torch/DIRECT_BRIDGE.md`。
