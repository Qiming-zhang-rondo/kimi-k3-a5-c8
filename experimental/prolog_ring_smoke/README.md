# 直接替换ac8bef44的DataCopy失败smoke

本目录是完整、独立的热环writer单元：公共ASC源、明确10指针/80B ABI、CMake和Torch薄桥/验证脚本。
不依赖私有Prolog源码，不需要TBE/proto/vendor安装。不要继续逐个猜DataCopy参数。

原119行smoke的具体问题：

- 缺少已验证KV-split归约所用的AIV-only任务声明与3510 Vector编译隔离，UB/V代码暴露给错误编译pass。
- capacityParam/poolCountParam是裸GM指针，不能直接调用.GetValue；本实现用明确host静态几何参数。
- DataCopy(resultGlobal,ringGlobal,count)是GM→GM，不属于普通UB搬运通路；本实现不做GPU readback，host检查实际ring目标行。
- 旧回读固定ring/tag第0行会验证错地址；本实现校验bank*C+position%C的实际行、KR和epoch。
- PIPE_V/PIPE_MTE3不能替代跨流水依赖；这里使用显式MTE2→S/V、V/S→MTE3及相反方向保护UB复用。

## 内网agent现在直接构建并运行

使用上轮已经成功配置ASC的匹配SDK，空闲A5卡，不改系统vendor和服务。示例为记录中的CANN9.2.1：

```bash
RING_SDK=/usr/local/Ascend/cann-9.2.1
RING_RUN=$(mktemp -d /tmp/qsfa-ring-smoke.XXXXXX)
cmake -S experimental/prolog_ring_smoke -B "$RING_RUN/build" \
  -DSFA_SDK_ROOT="$RING_SDK" \
  -DCMAKE_PREFIX_PATH="$RING_SDK;$RING_SDK/lib64/cmake" \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build "$RING_RUN/build" -j8
python experimental/prolog_ring_smoke/run_smoke.py \
  --library "$RING_RUN/build/libqsfa_prolog_ring_smoke.so" \
  --sdk-root "$RING_SDK" --device npu:0 --graph \
  --output "$RING_RUN/results"
```

设备号由agent确认空闲，不抢业务卡。实际ASCConfig目录不同则沿既有成功CMakeCache定位，
不能改架构宏/伪造SDK来消除报错。若配置或构建失败保存首个完整错误并在本地按真实SDK修复，不反复改原私有Prolog。
Torch桥用当前NPU stream、真实Torch ABI和allocator recordStream，不用ctypes猜stream。

脚本默认验证B1Q1与B8Q4、OFF、512元素BF16 RNE、RoPE、位置/epoch、padding、回绕和源输入未改。
--graph增加固定地址捕获、毒化ring以及改变设备位置/bank/epoch/active mask后的实际replay检查。
读回只看本次kernel留下的真实ring，不另调用算子刷新。

只有这个writer的设备测试通过，才能把其类型/编译guard/搬运与事件模式移回内网Prolog正确KV写点。
它不是Prolog生产实现，不含真实C4量化，不证明C4字节或整条服务已通过；不能拿smoke通过替代producer/服务验收。
后续producer仍须验证C4数据/scale逐字节不变，遵守PROLOG_PRODUCER_TASK和完整ABI1。

源码、构建与原始结果留内网，只回传部门允许披露的编译状态、首错或测试摘要，不再上传私有实现。
外部仅做了宿主检查和ABI头编译；A5/ASC设备编译与执行待本轮实测。
