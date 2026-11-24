# 原版 SFA 编译的平台初始化修复路径

## 已知证据及其边界

`36445a85d3a3b9f8dbe0e8191ed9913628e00146` 上传了状态报告：简单ASC工程部分阶段可编译，原版SFA缺少tiling类型，
对 `Ascend950DT` 调用SDK目标设置时发生SIGSEGV。该提交没有原始编译日志、probe_report或SFA内核产物。
这不能证明原版SFA已经编译成功，也不能仅凭 `get_platform_info failed` 就断定NPU硬件访问失败。

编译目标必须使用当前SDK真正支持的完整SoC标识。官方文档说明可配置名称来自平台配置文件：

- [官方算子样例：compiler/data/platform_config/*.ini 文件名](https://github.com/ascend/samples/blob/master/cplusplus/level1_single_api/4_op_dev/1_custom_op/README.md)
- [官方TorchAir文档：架构目录下 data/platform_config](https://github.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/api/scope/limit_core_num.md)

`Ascend950DT` 是报告使用的名称；是否为当前SDK支持的准确目标尚未证明。
`SHORT_SOC_VERSION` 是家族信息，不能直接当成完整目标。缺配置、模块/动态库来源不一致和目标名称不匹配都需检查，
不能据现有摘要锁定其中一个根因，也不能先要求用户换容器或升级CANN。

## 内网代理直接执行

沿用当前已配置的工作容器，在仓库根执行：

```bash
python3 tools/diagnose_sfa_platform.py --work-dir /tmp/sfa-platform-recovery --compile
```

默认从已有SDK环境路径和常见安装位置查找实际平台配置。若已定位实际SDK根，可重复传 `--sdk-root`。
该参数只限定配置发现，不暗中更换Python/动态库环境。工具不会运行NPU内核、升级包、修改vendor或停止服务。

工具将：

1. 保存配置文件实际路径、SHA256、候选完整目标和多个SDK配置冲突。
2. 每个目标启动独立Python子进程，打开faulthandler，执行隔离原版编译探针；一个初始化崩溃不会结束父进程。
3. 保存真实返回码/信号、原始日志，以及探针崩溃前的阶段、setter模块路径/签名/返回值、实际目标查询。
4. 最多尝试4个候选；只有子进程成功、实际调用编译且 `.o` 存在并校验通过才标记 `COMPILE_PASS`。
5. 平台失败或编译失败时保留完整证据及尚未尝试的候选。内网代理继续根据这些证据处理，不向用户逐项索取文件。

若所有候选失败，先核对实际导入的SDK模块、平台数据位置及当前环境是否混用了不同SDK；
可以在独立子进程中显式加载同一SDK自己的环境作对照，但不得混入旧vendor环境或修改父进程/服务配置。
不要删除错误记录后宣称成功。把diagnosis.json、各阶段status.json/log以及probe_report作为正常提交的一部分。

## 已找到真实 tiling 定义，保留严格的来源配对

`upstream/` 提供固定commit的vLLM/SGL公开源定义、host实现和匹配kernel，并保留原始许可证。
`SOURCE_MANIFEST.json` 含Git blob SHA和SHA256；`FIELD_COMPARISON.json` 记录字段顺序及host/POD对应关系。
这些是实际源码，不是猜测的二进制结构。

但同名结构存在不同版本：最后base字段有 `ropeHeadDim` / `dispatchKey` / `keyStride0`，另有版本增加split-KV字段。
不能把一个公开头直接插入私有包，再喂给它未验证的私有tiling字节。

继续推进的两条路径：

- 修好当前SDK平台初始化，取得当前包真正生成/使用的定义，按原计划编译原版和候选。
- 若当前包完整构建源码确实不可得，使用 `upstream/PROVENANCE.md` 中固定版本的公开host tiling和kernel作为匹配源码整体重构，
  先建立该源码版本的原生基线，再移植融合改动。依赖继续取同一commit，不混vLLM/SGL/私有库。

只为了编译检查而使用公开POD时，产物必须标记为对应公开schema的compile-only结果，不能冒充私有包原版编译成功或直接设备执行。

## 本地测试与用户桌面

默认测试使用模拟信号状态，不在Mac上实际触发Python崩溃。真实SIGSEGV探针测试仅限Linux显式opt-in。
实际SDK若在内网子进程初始化时崩溃，诊断器仍会保留其真实信号和faulthandler日志；这与主动制造桌面崩溃是两件事。
