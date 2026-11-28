# 在真实 Prolog 内直接写环：本轮唯一实现路线

f23f328 不是新的 UB 来源问题。fac32bf36 已定位真实路径：
`RmsNormRopeScatterCkvKr -> RmsNormAndQuantizeCkv -> QuantPerTileMxFp4`。
独立 kernel 不能跨 kernel 读取 UB，这是确定的硬件/执行边界；不再为此让用户选 A/B/C/D。

## 已给出的可复用代码

`prolog_ring_write_device.h` 是公开 inline Vector helper，不是另一个 producer kernel。
它放进真实 Prolog 的 Vector 分支，在同一次 Prolog launch 内使用已有 LocalTensor：

`实际 FP32 quant-input UB -> 独立 BF16_RNE scratch -> BF16 ring GM`

没有 FP32 GM 中转、没有额外 writer kernel。原 FP32 数值和数据类型继续供原 QuantPerTileMxFp4 消费。
KR64 单独取真实 post-RoPE LocalTensor。已有折权重旋转/scale/clip 按真实原流程保留，不重复旋转，不试改 alg2/max7。
helper 不提供也不猜测私有 host launcher、tiling、proto 或 scalar 顺序，不能当作完整 Prolog variant。
仅对 helper 本身做了宿主 shim 编译/行为检查，仍需真实 A5/SDK 联合编译验证。

## 内网现在直接实施

1. 从实际 mode4 运行分支确认全局 token 行、输入 LocalTensor 的布局和真实 latent 列切片。
   SplitN 可能分列；不能认为每核都有完整512，也不能把块内行号当全局 token。
   `colOffset` 是目标 latent/KR 行中的列偏移；传入 source 必须已经是本核 slice 的 LocalTensor 起点。
2. 只在真实 Prolog 的 Vector 分支包含并使用该 helper；保留原混合 AIC/AIV task、调度和 Cube 计算。
   **不能把整个 Prolog 改成 standalone writer 的 AIV-only。**
3. 在 variant 的原 TPipe 上 Init helper，一次初始化、正常收尾 Release；OFF 不分配 UB 或事件。
   new binding 的8个地址与 rows/pools/capacity/enabled 是 helper 内部参数，不是新造的对外二进制 ABI。
   这些值必须由真实新 op 的输入/tiling 接通，host launch 对非法 ON 配置提前拒绝。
   ON额外UB为1216B（1024B cast、4×32B metadata、64B tags），每个HardEvent方向各占一个独占ID。
   在原Prolog的实际UB和事件预算里核对，不借用原FP32输入/共享scratch，不硬编码抢占EVENT_ID0。
4. 对每个本核拥有的 token ResolveRow；在真正量化输入已 ready 的位置调用 WriteLatentSlice。
   KR 真正 post-RoPE ready 后调用 WriteRopeSlice。输入数据 readiness 由原 Prolog 管线负责；helper 保护自己的 cast 和 DMA scratch 生命周期。
   无效/padding token 不写；寻址为 bank*C+position%C，再加真实列偏移。
5. 每行明确一个 tag owner 调用 PublishTags。其余 payload 列可由原多核分片各自写。
   reader 只能在**整个 Prolog kernel 完成**之后同 stream 执行；跨 stream 则等待整个 producer 完成的事件。
   tags 不作为同 kernel 内其他核的 ready flag，因此不要添加全核 spin/barrier。
6. 完整接通已有公共 ABI1：原9参/原5项 query 返回、ring latent/KR、position/epoch、逐 token bank/position/epoch/valid、maxQ/window。
   新 wrapper 需要真实 side-write 才能声明 `recent128_abi_version=1`；不能给旧 op 贴标记替代实现。

helper 的 include、成员类型/对象声明，以及 Init、ResolveRow、WriteLatentSlice、WriteRopeSlice、PublishTags、Release
等所有引用都必须处于同一个**编译器实际选择的** Vector guard 内；只保护 include 会让 Cube/host pass 仍解析到不存在的 helper 类型。
保留原 Cube 分支的计算和完整 launch 参数，不能用全局 AIV-only 绕过。

```cpp
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && defined(__DAV_C310_VEC__)
// helper include / member declaration / each call site: all use this guard.
#endif
```

公共 ABI 标记必须设置在 `cloud_ops_turbo` 最终导出的**函数对象**上，例如真实 wrapper 定义、装饰和导出完成后：

```python
cloud_mla_prolog_v3_recent128.recent128_abi_version = 1
```

仅设置模块级 `recent128_abi_version` 不生效；值须为整数 `1`，不能是字符串或 `True`。
双节点安装与函数标记通过只证明接口已就位。如果 TBE 在首次真实调用时才动态编译，
该次编译、真实 mode4 side-write 以及 OFF/ON 原 C4/query 控制仍待实际完成，不能提前记为 producer 验收通过。

## 一次解决构建，不在设备上试 TBE 宏

这些步骤全部先在内网 CPU 构建阶段完成，不加载模型、不调用 NPU：

1. 找当前实际 Prolog 源码/构建入口，并固定当前匹配 SDK。使用原工程的 verbose 构建，先编译**未改动原 entry**作为工具链对照。
   编译命令、include 链和输出留内网，不要求用户上传内部头文件或日志。
2. 对所谓 kernel_log.h 可变参数错误，依据本轮真实首错查编译器、C++标准、host/Vector/Cube pass 和头文件路径。
   只有报告中的一句描述不能判断根因。先修构建入口的路径/阶段隔离，不修改系统 SDK/kernel_log.h，不伪造架构宏。
3. 原工程 TBE/codegen 链能工作则沿用，保持原 tiler/workspace/dispatch；增加版本化 variant 的额外输入。
   **不是必须为了新 ring write 强行只走 TBE。** 若原源码包只有 kernel、原打包链不可用，可按已成功 ASC CMake 的方式编译独立候选，
   但必须保留真实 Prolog entry/task 和**准确的 device-side tiling POD**及原 host tiler/workspace/dispatch 合同。
4. REGISTER_TILING_DEFAULT / GET_TILING_DATA_WITH_STRUCT 不应靠空宏消除。
   从内网实际 codegen 产物取得原 device 定义，核对大小、字段偏移和序列化；缺这些信息就在内网生成/定位，不能凭字段名拼 ABI。
   若需手写 entry，只替换原宏的加载机制，不能改变其数据布局或启动参数顺序。不能套用 SFA 的 tiling 到 Prolog。
5. 原 entry 先通过编译再编修改 variant；记录成功编译的真实目标和 SDK，随后才运行以下完整 producer 小矩阵。
   不再回到独立 producer 从 GM 寻找实际只存在 UB 的值。

## 设备一次验收的范围

一套构建覆盖 OFF/ON、B1Q1/B1Q4/B8Q4、跨环回绕、padding、请求重排和位置回退；Graph 中更换输入/位置/bank/epoch/valid 后重放。
验证原 query、C4 packed 和 scale 逐字节不变，实际 sidecar 对应真实 FP32 quant-input 的 BF16_RNE，KR 对应真实 post-RoPE。
SplitN 按实际列写全，tags 唯一 owner，reader 不能在 payload 未完成时提前启动。
通过后再接新 serving runtime；APC/PD 的真实尾窗自动传输仍是独立已知接线项，不能因为 reader 通过就略去。
长prompt的初始decode也依赖P的尾部：L=8192且D刚写1个新KV时，最近128中仍有127个prompt token。
P到D必须带实际BF16尾窗，不能因为GPQA题目长就跳过；具体公式和例子见SERVING_TASK顶部。

输入 source 不再讨论：workspace 是 pre-RMSNorm 阶段，已有 functional 返回的是量化 cache，都不能替代本次 UB tap。
FP32 GM sidecar 仍需要改、重编 Prolog，却增加一次 launch 和额外读写（完整 latent 约4KiB/token），不作为本轮性能方案。
