# 修复 562ca1ca 的两个 standalone ASC 编译错误

4940380a 已报告真实 baseline 构建成功；后续版本宏、3510 LoadData API 与 kernel-type 常量核验见
[SDK_COMPATIBILITY.md](SDK_COMPATIBILITY.md)。下文保留 562ca1ca 两项错误的修复依据。

当前报错还在未修改 Prolog 的 baseline CPU 编译阶段，没有进入 NPU/模型执行，不需要安排新的 A5 数值试验。
这里交付本地 staging 工具和真实 POD 加载 helper；不修改原私有源码、SDK 或 vendor，不假造私有 tiling。

## 1. template_argument.h 的 host printf 冲突

独立 direct target 不依赖框架生成的模板注册代码。只在内网副本中，用显式的
`QSFA_ASC_DIRECT_BUILD=1` 隔离以下部分：

- `ascendc/host_api/tiling/template_argument.h` 的 include。
- 完整的 `ASCENDC_TPL_ARGS_DECL(...)`、`ASCENDC_TPL_SEL(...)` 注册块。

真实 enums/constants 与 device POD 保留。原 host tiler/模板生成目标不定义该 flag，保持原注册逻辑。
不要给系统 SDK 的 printf 改宏，不要把所有模板宏清空，也不要把整个 tiling-key header 删除。
官方框架模板示例中 kernel 也会包含该 host_api 头，所以不能概括成“该头永远不能用于 device”；
本次处理的是独立 ASC target 的编译职责隔离。

特别检查 `ASCENDC_TPL_MIX_AIC_1_1` 等设备常量：若其定义只来自被隔离的 host API，
将当前 SDK 的真实公共常量定义拆到 device-safe 公共头，或包含实际定义它的 SDK device 头。
不能猜值，更不能 `#define ... 0` 消除错误；staging manifest 会列出仍需核验的相关符号。

## 2. GET_TILING_DATA_WITH_STRUCT 不在 direct 工程生成

该接口属于框架的宏展开机制，官方说明 kernel 直调工程不支持它：
[官方接口约束](https://asc.gitcode.com/api/SIMD-API/basic_api/tool_interface/tiling_data_acquisition/GET_TILING_DATA_WITH_STRUCT.html)。
“在 broadcast_sch.h 中看到调用”不代表那里定义了宏，不继续靠增加不相关 include 猜测。

本地副本直接声明原本的**实际 device POD**，再将同类型的 GM tiling 完整装入局部对象：

```cpp
optiling::CloudMlaPrologTilingData tilingDataIn{};
qsfa_direct::LoadTilingWords(tilingDataIn, tiling);
```

helper 使用官方 direct 示例同类的 uint32 GM→局部 word-copy，检查标准布局、可平凡复制、size/对齐。
[官方直接加载示例](https://www.hiascend.com/doc_center/source/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0041.html)。
不定义新的 CloudMlaPrologTilingData，不把 host serializer 类内存当作 device POD，
不拿以前上传的旧 V1 同名头替代当前 V3，也不从 SFA 猜 Prolog 的字段顺序或大小。

host 必须确认实际 buffer 非空、>=sizeof(T)、4B对齐，并与真实 host tiler 的序列化布局/字段偏移相符。
如果两侧消费 tiling，CUBE/VEC 各自都保留实际局部对象及 Init/Process 生命周期；不能照搬 SFA 的 Cube nullptr。

## 内网一次应用并完成 CPU 编译

`source-root` 指当前真实 Prolog 的 operator 源码子树；entry/key-header 是该树内已定位的相对路径。
`output` 用仓库和原源码之外的新目录，所有私有副本与清单留内网，不上传。

```bash
python tools/prepare_prolog_direct.py \
  --source-root "$PROLOG_KERNEL_ROOT" \
  --entry "$PROLOG_ENTRY_REL" \
  --template-key-header "$PROLOG_TEMPLATE_KEY_REL" \
  --output "$PRIVATE_STAGE"
```

工具复制本地源码，保留原路径结构，仅改 staged entry/模板注册隔离，放入通用 helper；
若遇到未知语法或不明确的类型，直接指出，不强行改写。
如果 manifest 列出的剩余模板常量确实只在 host API 中定义，内网提取当前 SDK 的真实 device-safe 常量声明，
使用 `--device-constants-header "$ACTUAL_DEVICE_CONSTANTS"` 在另一个新输出目录生成副本；工具会拒绝再次夹带 host 注册头或空模板宏。
这份常量文件和 manifest 也留内网，无需上传。

继续使用已经配置成功的 matched SDK/dav-3510 ASC CMake target：

1. 把 baseline target 的 source/include 指向 staging manifest 中的私有副本，源后缀为 `.asc`。
2. 只给这个独立 ASC target 加 `target_compile_definitions(实际target PRIVATE QSFA_ASC_DIRECT_BUILD=1)`。
   不给原 host tiler/原模板生成目标加该定义，不改 __NPU_ARCH__/__CCE_AICORE__。
3. 依据 manifest 核对仍在 device 使用的 `ASCENDC_TPL_*` 常量的真实定义。
4. 保留真实 tiling key→模板参数、原 blockDim/workspace/launch 参数顺序。
   **必须实例化真实 mode4 specialization 并确认产物中有实际 kernel/launch symbol。**
   未实例化的模板 `.o` 能编译，不等于算子已经生成，不能当成 baseline 编译完成。
5. 在内网保存本轮 verbose compile 命令、首错和输出 target。把已知编译问题集中处理完，再将 inline writer 接入同一真实 Prolog。

staging 成功、宿主 helper 测试通过只说明转换/搬运逻辑的局部检查，不是完整 Prolog/CANN 编译通过。
本次6项宿主检查通过：4项staging保护/解析/预处理测试，2项真实loader的C++17与UBSan检查。
真正生成并核对 baseline/variant 后，再一次完成已列出的 OFF/ON + B1/B8 + Graph producer 验证。

本轮不要再改 reader、反复跑已通过的独立 reader，也不要用 FP32 GM 中转绕开 Prolog 内部写点。
