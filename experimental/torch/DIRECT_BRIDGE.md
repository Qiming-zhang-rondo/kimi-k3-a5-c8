# 真实 ASC direct bridge：接入既有 BF16 热窗实验

这是已有原生融合 kernel 的 Torch 接线，不是另一个 attention 实现。
原 `sfa_cold_merge_bridge.cpp` 仍保留作新 ACLNN 注册方式的源码草案；其中
`aclnnSparseFlashAttentionColdMerge` 只有声明，不能用于此次独立设备库。
实际使用 `load_sfa_direct.py` 和 `sfa_direct_bridge.cpp`，链接现场构建的
`libsfa_torch_direct.so`。

## 构建边界

`SFA_BUILD_TORCH_DIRECT=ON` 生成 shared library，原版与融合版各有一个独立 ASC
translation unit。C++ bridge 只调用这两个 host launch wrapper；不能在同一个编译
单元同时包含两套同名 `BaseApi::SFAVectorService` 头，也不能复用旧两遍路径冒充融合。

在原版和融合 smoke 已成功的同一 SDK 环境、仓库根目录构建，保留现场已验证的
ASC_DIR、编译器和其他参数：

```bash
cmake -S standalone -B /tmp/sfa-torch-direct-build \
  -DSFA_BUILD_TORCH_DIRECT=ON \
  -DSFA_PROBE_TEMPLATE_MODE=0 \
  -DSFA_SDK_ROOT="$ASCEND_HOME_PATH"
cmake --build /tmp/sfa-torch-direct-build --target sfa_torch_direct -j8 --verbose
```

首次在现场需要保留真实构建日志、库大小/哈希及 AIC/AIV 产物。现有宿主检查不能代替
ASC 编译、NPU 运行或速度证据。不要安装新 vendor 或改换 CANN 来消除普通构建错误。

## 加载和生命周期

```python
import sys
sys.path.insert(0, "/absolute/path/to/this/repository/experimental/torch")
from load_sfa_direct import load_sfa_direct

bridge = load_sfa_direct(
    "/tmp/sfa-torch-direct-build",
    # 可选：sdk_root="与构建设备库完全相同的 toolkit 路径",
    # 可选：build_directory="/tmp/isolated-torch-jit-build",
    verbose=True,
)
original = bridge.prepare(base.q_nope, base.scale, False, True)
# 只在 setup 探测真实 dtype，不把这个冷态结果复用于 timed pipeline。
cold_template = runner._cold_read(runner.prepared)[0]
if cold_template.dtype not in (base.torch.bfloat16, base.torch.float32):
    raise ValueError("Unsupported actual native C4 output dtype")
fused = bridge.prepare(base.q_nope, base.scale, True, False,
                       bf16_io=cold_template.dtype == base.torch.bfloat16)
```

加载只让 `torch.utils.cpp_extension.load` 编译 `sfa_direct_bridge.cpp`；不会重新编译 ASC、
下载源码、安装软件或启动 kernel。编译自动跟随正在使用的 Torch C++ ABI。loader 验证
设备库、当前 torch_npu 的真实 include/lib、SDK ACL 头及 CMakeCache 中的 SDK 根；禁止
加载后覆盖同路径 `.so` 再在同一进程假装重载。换二进制后使用新 Python 进程。
JIT 目录内的 `sfa_direct_bridge_build.json` 保存库、bridge 源码哈希、SDK 及版本信息。

`Prepared` 必须活到 eager 操作和 graph replay 全部结束。它持有预分配的输出、tiling、
442368 字节 raw user scratch 和内部统计 buffer；prepare 完成后才能捕图/计时。
原版输出 BF16；融合按真实冷态选择 BF16输入/输出或 FP32输入/输出，stats始终FP32。
`bf16_io=True` 仅用于融合，直接从 BF16 cold buffer 搬入，不增加 Python `.float()`/`.to()`
或独立转换 kernel；最终 BF16 输出为262144字节。默认 `bf16_io=False` 保留原有FP32
输入/输出实验合同，output为524288字节。融合不返回 combined stats。桥要求设备库ABI2，
旧ABI1库即使尺寸仍为19指针也会拒绝；修改后须重建设备库并启动新的Python进程。

运行方法接收本次 Tensor 对象，读取本次 `data_ptr()`。冷算子每次新分配输出时，不能
把首次冷态指针固定在 prepare 中。桥使用 query 设备的 `NPUGuard` 和
`getCurrentNPUStream(device).stream()`，向当前流直接提交，不另建 stream，不做 ACL
同步。默认 `.stream()` 会清空 torch_npu 的 host 任务提交队列，确保先前冷读已提交；
不能替换成 `getCurrentNPUStreamNoWait()` 或 `.stream(false)`。相关 storage 还需
桥已调用 `NPUCachingAllocator::recordStream` 维护异步使用期。精度门禁的显式同步由外层 harness 执行。

已核验的官方 Torch-NPU 2.10 源码：
[NPUGuard](https://github.com/Ascend/pytorch/blob/de28db394d8ffd0708d40014fee9eb5f43e51efa/torch_npu/csrc/core/npu/NPUGuard.h)、
[NPUStream 声明](https://github.com/Ascend/pytorch/blob/de28db394d8ffd0708d40014fee9eb5f43e51efa/torch_npu/csrc/core/npu/NPUStream.h)、
[stream() 提交队列处理](https://github.com/Ascend/pytorch/blob/de28db394d8ffd0708d40014fee9eb5f43e51efa/torch_npu/csrc/core/npu/NPUStream.cpp)、
[recordStream](https://github.com/Ascend/pytorch/blob/de28db394d8ffd0708d40014fee9eb5f43e51efa/torch_npu/csrc/core/npu/NPUCachingAllocator.h)、
[官方扩展 include/lib 布局](https://github.com/Ascend/pytorch/blob/de28db394d8ffd0708d40014fee9eb5f43e51efa/torch_npu/utils/cpp_extension.py)。
这些证明 API 来源；现场仍须用实际安装版本的头和库编译。

## 与旧 BF16 harness 的映射

旧 payload 固定为 `4a5fd67b9a78766bebfa564782ea2193bc22af1a` 的
`kv-native-twopass/hotkv/`。不要修改旧 payload 内容或省略 SHA 校验。

| bridge 输入 | 旧 `Bf16HotRunner` / backend 对象 | 固定 shape / dtype |
| --- | --- | --- |
| query | `base.q_nope` | `[4,64,512]` BF16 |
| key/value | `runner.hot_cache`，同一 MLA latent 别名 | `[2,128,1,512]` BF16 |
| sparse indices | `parts["recent_ring"]` | `[4,1,128]` int32，全部有效 |
| block table | `runner.ring_table` | `[1,2]` int32，identity |
| query lengths | `base.cu` | `[1]` int32，值4 |
| KV lengths | `runner.ring_lengths` | `[1]` int32，值256 |
| query RoPE | `base.q_rope` | `[4,64,64]` BF16 |
| key RoPE | `runner.hot_rope` | `[2,128,1,64]` BF16 |
| cold output | 本次 `runner._cold_read(parts)[0]` | `[4,64,512]`，保留实际 BF16/FP32，与所选融合入口一致 |
| cold max/sum | 本次 `runner._cold_read(parts)[1:]` | `[4,64,1]`，始终 FP32 |
| presence | `parts["old_present"]` / `parts["recent_present"]` | `[4]` bool |
| scale | `base.scale`，prepare 时传入 | 使用真实值，不改为其他默认值 |

冷端完整 KV 仍为 K8192，由原生 C4 reader 访问。桥内部的256是热环物理槽数，不能把
冷端 K8192 写进固定热环 tiling。POD 为选定源码的144字节布局，launch为1 AIC/2 AIV，
不复用已安装 ACLNN executor 的 opaque tiling 或系统 workspace 前缀。

仅替换 `Bf16HotRunner._read(parts)` 中的“热读＋合并”，保留其余流程：

```python
left = runner._cold_read(parts)  # 本次真实原生 C4 read，不能在计时外固定结果。
out, _, _ = fused.fused_hot(
    base.q_nope, runner.hot_cache, runner.hot_cache,
    parts["recent_ring"], runner.ring_table, base.cu, runner.ring_lengths,
    base.q_rope, runner.hot_rope,
    *left, parts["old_present"], parts["recent_present"],
)
```

原生两遍对照用同九个热输入调用 `original.native_hot(...)`，得到 BF16 hot output 和
`[4,64,1]` FP32 max/sum，再运行原来的 Triton merge。最终保留真实冷态的输出dtype；
设备证据 `d92f2f59` 的 C4 cold 与已安装两遍 merge 均为BF16，此前固定FP32断言已纠正。
两侧先以FP32参与合并，最终一次BF16 RNE舍入；热端匹配原版BF16输出的舍入仍是独立步骤，
不能把两次舍入混成一个，也不能把FP32未舍入输出直接拿来验收BF16目标的0.001门槛。
独立参考必须按相同最终dtype舍入，严格门槛不变。

`run(write=True,dynamic=True)` 仍必须每次执行真实冷写和 `hot_store_partition`，读取它的
动态输出 buffers；prepared read 另行报告。图捕获后 poison 输出再 replay，检查新 ASC
kernel 确实进图并重新写出结果。真正 hot=0 调度尚不在固定 compact128 支持范围内。

先完成输入/冷热分区/写入控制、原版对已部署 reader、融合对真实冷态两遍参考，再同轮计时。
实际四臂包括C4 W0、已安装两遍、direct两遍诊断与direct融合；必须比较融合/已安装两遍，
不能用一个可能变慢的direct两遍代替当前基线。分别保存 event、wall、每轮比值及 profiler kernel 信息；
不能把宿主提交时间当设备延迟，也不能用更换输入或计时范围制造提升。
