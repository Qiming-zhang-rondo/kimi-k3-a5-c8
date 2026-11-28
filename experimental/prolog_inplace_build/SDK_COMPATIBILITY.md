# 独立 Prolog ASC target 的 SDK 能力核验

4940380a 的内网脱敏报告已经记录：匹配 CANN 9.2.1 的 baseline 构建成功，生成约 547 KB 动态库，
其中 `.aicore_binary` 约 373 KB，并报告存在 launch 基础设施符号。这里的结论是**未接热环 baseline 已生成设备代码**；
基础设施符号不能替代当前 mode4 specialization 和真实可调用入口的核验。
不能据此把 inline BF16 writer variant、真实 Prolog OFF/ON C4 字节控制、Graph 或模型结果标记为通过。
这些产物和原始构建记录留内网。

d0190621 随后报告 Writer variant 也完成 clean rebuild，且在私有副本把事件类型从 `event_t` 改为 `TEventID`。
公共 helper 已同步采用 `TPipe::AllocEventID` 的高层 `TEventID`；宿主测试现严格区分两种类型，能使旧错误写法编译失败。
这依旧是编译进展，报告只列了 latent 调用证据，KR/tag 及新增参数的真实调用入口须按任务文件核实。

**当前直接沿已经成功的私有 baseline 副本继续接 Writer。** 不为补文档重新 staging、换 SDK、删除已验证的 API 修复或重编 reader。
先将该树/编译参数冻结保存，再从它创建 variant，只增加旁路写环。本说明和 SDK 检查用于防止今后环境/源码变化再次踩坑，不是新的上机前置试验。

## 已知适用范围和编译定义

| 条目 | 本轮要求与已知状态 |
| --- | --- |
| SDK / 架构 | 本轮记录为匹配 CANN 9.2.1、dav-3510；9.2.0/9.3.0 没有本 variant 的等价构建/运行证据，不从版本号推断源码兼容 |
| `QSFA_ASC_DIRECT_BUILD=1` | 只给独立 ASC target；原 host tiler/模板生成目标不加 |
| `ASC_DEVKIT_VERSION_NUM` | 不是本项目可以任意指定的数字；源码若用其选择API，必须确认它在实际device预处理阶段有真实定义，或保留当前已验证的新API适配 |
| `ASCENDC_TPL_MIX_AIC_1_1` | 本轮实际SDK头报告值为4；保留来源/版本，只在对应device常量副本使用，不能外推其它枚举值 |
| `__NPU_ARCH__`、AIC/AIV宏 | 由匹配ASC编译器产生，不人工定义 |
| Writer事件类型 | 使用实际SDK的 `TEventID`，不拿底层 `event_t` 代替 `AllocEventID` 返回类型 |
| include / compiler | 以成功的CMakeCache和verbose命令为准；ASC、ACL、host tiling头与compiler都来自同一安装，保留原相对private头布局 |

## 见到这些症状时先处理已知原因

| 症状 | 已知检查点 |
| --- | --- |
| `LoadDataRepeatParam` 要3参但传4参 | 检查未定义版本宏是否误选旧分支；3510按实际支持的WithStride配套API处理，不盲删dstStride |
| `ASCENDC_TPL_MIX_AIC_1_1` 未声明 | host注册隔离也移除了常量来源；从实际SDK保留真实device声明，不猜值 |
| host `printf` 调device函数 | 独立target的host模板注册污染，按staging隔离职责，不改SDK |
| `GET_TILING_DATA_WITH_STRUCT` 解析失败 | 直调工程不生成框架宏，显式加载真实device POD，不造同名布局 |
| 转到arch22或架构不支持 | 核对.asc/ASC语言和实际target；不能伪造架构宏去绕过 |
| `.o/.so` 出来了但没有目标mode4函数 | 核对模板真实实例化与launch symbol；不能只检查空模板文件编译返回0 |

## 可选的本地SDK证据检查

对今后新环境或重新 staging，可在CPU构建阶段执行：

```bash
python tools/check_prolog_sdk.py --sdk-root "$PROLOG_MATCHED_SDK" \
  --source-root "$PROLOG_SOURCE_TO_BUILD" --output "$NEW_SDK_DIAGNOSTICS"
```

工具只读取版本宏/常量声明与源码使用位置、记录源文件SHA，并执行匹配SDK内的 `bisheng --version`。
它生成独立的宏预处理probe和device编译probe，不运行NPU，不自动修改SDK/源码，也不自动执行未经核对的预处理命令。
`FOUND_IN_HEADER_NOT_RESOLVED` 不等于实际编译阶段已生效；缺宏或扫描不完整时返回2，让使用者按原成功CMake参数核对。
不能为了让检查返回0而补猜值。报告/探针留内网，不要求外传。
当前baseline/variant已编译成功，不要求为这项说明或脚本重编、重跑reader或重新占用设备。

## 本次已定位的两个兼容问题

| 项目 | 已知事实 | 独立 target 应如何处理 |
| --- | --- | --- |
| `ASC_DEVKIT_VERSION_NUM` | 原源码用它选择 SDK API；独立 direct 编译中未定义时，预处理条件把它当作 0，进入旧分支。CANN 的显示版本 `9.2.1` 不能证明这个宏的数值编码。 | 从**同一安装、同一实际构建参数**恢复真实宏定义或原构建系统注入来源，再选择该 SDK 实际支持的 API。未核实不得用 `-DASC_DEVKIT_VERSION_NUM=90000000` 强迫选分支。 |
| `ASCENDC_TPL_MIX_AIC_1_1` | 本轮报告从实际 9.2.1 bisheng 头文件取得值 `4`。官方文档确认其含义为 AIC/AIV 同时运行、Cube:Vector=1:1。 | 保留实际 SDK 的声明/值及其来源，只在隔离 direct target 提供所需的 device-safe 常量。不由这个值推断其他 kernel-type 常量，也不把 `4` 视为跨 SDK 通用值。 |

官方 3510 示例使用 `SetLoadDataRepeatWithStride({0, 1, 0, dstStride})` 配合 `LoadDataWithStride`；
2201 分支使用三成员的 `SetLoadDataRepeat({0, 1, 0})`。
这支持本轮 3510 的 API 方向，不能概括成“所有架构都把旧函数改名即可”。
[官方 SetFmatrix 分架构示例](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_load_aux_config/SetFmatrix.md)

`SetLoadDataRepeatWithStride` 接受 `LoadDataRepeatParamWithStride`，其中包含 `repeatStride`、
`repeatTime`、`repeatMode`、`dstStride`，用于配置 `LoadDataWithStride`。
其产品支持范围为 Ascend 950PR/950DT。实际 typedef、参数类型和声明仍以本次 SDK 头为准，不能删除第四个成员绕过编译。
[官方接口定义](https://asc.gitcode.com/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E7%9F%A9%E9%98%B5%E8%AE%A1%E7%AE%97%EF%BC%88ISASI%EF%BC%89/%E7%9F%A9%E9%98%B5%E8%AE%A1%E7%AE%97%E7%9A%84%E6%90%AC%E5%85%A5/%E8%BE%85%E5%8A%A9%E9%85%8D%E7%BD%AE%E6%8E%A5%E5%8F%A3/SetLoadDataRepeatWithStride.html)

## 版本宏与常量的真实提取方法

1. 从已经成功配置的 `CMakeCache.txt`、`compile_commands.json` 或 verbose 命令取得 compiler 的真实路径、
   sysroot、架构/编译阶段、SDK include 路径和 `-D` 参数。compiler、ASC headers、host headers 必须来自同一安装。
   不用 `latest` 软链接或另一个 SDK 的搜索结果替代当前 target 的解析路径。
2. 在这个已匹配的 SDK 及原算子构建脚本中执行本地文本搜索：

   ```bash
   rg -n 'ASC_DEVKIT_VERSION_NUM|ASCENDC_TPL_MIX_AIC_1_1|SetLoadDataRepeatWithStride' \
     "$PROLOG_MATCHED_SDK" "$PROLOG_ORIGINAL_BUILD_ROOT"
   ```

   保存实际定义文件路径、文件 SHA256 和相关声明位置。宏可能来自版本头，也可能是原构建系统的 compile definition；
   两者都必须追溯，不能搜索不到就自行编码 `9.2.1`。本文没有确认一个适用于所有 SDK 的版本头路径或数值。
3. 用实际定义头建立很小的 probe translation unit。**host probe 不包含 `kernel_operator.h`**，
   避免将 host 模板注册/printf 再混入 device 编译。采用实际 C++ compiler/include 参数预处理：
   `-E -dM` 检查最终宏定义；`-E -H` 检查究竟命中了哪个头文件。
   这两个选项应替换原命令的编译/输出选项，不能把额外 `-c/-o` 留在探针命令里。
   如果宏是表达式，继续在小型 host 程序中按原定义求值，保留它依赖的真实宏；不要按字符串猜结果。
4. `ASCENDC_TPL_MIX_AIC_1_1` 若是 enum 而非宏，`-dM` 不会显示它。
   在只包含实际 host 定义头的小型 host 程序中输出该常量的整数值，并检查其声明类型，
   然后仅提取真实需要的 device-safe 声明。不要把整个 `template_argument.h` 注入 direct device TU，
   不要用 `#ifdef` 判定 enum 是否存在，也不要把模板注册宏定义为空。
5. 恢复到 direct target 时，优先显式包含实际 device-safe 版本头；若原构建确由 `-D` 提供版本宏，
   只重用已经验证的原值并记录来源。保留 `__NPU_ARCH__`、CUBE/VEC 标识由编译器生成，不能伪造。
   对每个实际 device pass 保存预处理结果，确认新 API 分支真的入选，再只编译，不先安排一次 A5 launch。

官方模板文档定义 kernel-type 常量的语义，但未在该接口页承诺所有版本的整数编码。
因此 `4` 的数值证据是本轮实际安装头及其探针，不是从网页枚举顺序数出来的结果。
[官方模板参数与 kernel-type 定义](https://asc.gitcode.com/api/Utils-API/Tiling_template_programming/template_params_definition.html)

## 后续边界

兼容层只解决本次 direct 编译职责和 SDK API 选择。真实 host tiler、tiling bytes/offset、key→模板 specialization、
blockDim、workspace、原 AIC/AIV 比例与初始化协议继续沿用当前 baseline。
新增 BF16 scratch 的资源预算、量化输入未改、V→MTE3/复用依赖和 KR/tag 写入须在 variant 单独验收。
先生成并检查 variant 的实际设备代码，再执行已约定的 producer OFF/ON、B1/B8 和动态 Graph 验证；
baseline 构建成功不要求重跑已经通过的独立 reader，也不授权 FP32 GM 中转或额外 producer kernel。

## 232e229b 的接线和缓存反馈如何落实

`event_t` 与 `TEventID` 的错误属于公共 helper 的真实缺陷，现已改正；shim 也不再把二者模拟为相同类型。
不能把宿主 shim 通过写成实际 SDK 编译通过。

Writer 的命名空间仅存在于真实 VEC3510 pass，**include、成员声明、setter类型、实例、Init、SetRingWriter、
所有Resolve/Write/Publish调用及Release均须在相同guard里**。不是只给include加guard：

```cpp
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && defined(__DAV_C310_VEC__)
// 放置所有 QsfaPrologRingWrite::Writer / Route / Binding 的引用与新增调用。
#endif
```

原 `op.Init` / `op.Process` 的 AIC/AIV 工作保留，不能把 Cube 分支改成空处理，也不能让 Vector 在
Init失败时独自提前return留下Cube等待。ON参数/几何先在host拒绝；内核失败处理遵循真实mixed算子协议。

**grep次数>=5和注释字符串不作为完整实现证明。** 它们看不出真实RoPE/tag调用，也不能证明当前模板分支执行。
任务文件逐项核对实际 `WriteLatentSlice`、`WriteRopeSlice`、`PublishTags`，再核对新入口和新增12参数，
最终仍以真实副作用和原C4/query对照为准。

对 variant 使用新的空 build 目录，不删除已经成功的 baseline：

```bash
PROLOG_NEW_BUILD=$(mktemp -d /tmp/prolog-variant-build.XXXXXX)
# 沿用已通过的相同source/SDK/ASC参数配置到 "$PROLOG_NEW_BUILD"
cmake --build "$PROLOG_NEW_BUILD" --verbose -j8
```

核对verbose/compile_commands中真正编译的是本轮staged entry和修改的header，保存source/SO的SHA和实际kernel符号。
`Building`提示、mtime或so尺寸单项都不足以证明加载到新variant。使用新产物名和SHA，测试bridge指向这个实际库。
不照抄 `rm -rf build` 删除工作基线，也不把更换目录当作真实调用入口已经接通。
