# 原生 BF16 热端合并冷端状态：源码候选

这批代码已实现设备端路径，仍未在匹配 AscendC SDK 编译或 A5 上执行。
它不是可直接安装的 `.run`/wheel，不包含已经生成的新 ACLNN 二进制。

## 已实现

- `sparse_flash_attention_cold_merge.cpp`：独立入口、14个输入，明确 BF16 Q/K/V/RoPE 和 FP32 最终输出的 dispatch。
- `arch35/sparse_flash_attention_kernel_mla.h`：五个冷端指针贯通 AIC/AIV 初始化。
- `arch35/sparse_flash_attention_service_vector_mla.h`：热端归一化和无效热输出清理后调用合并，最终直接写 FP32。
- `arch35/cold_state_merge_device.h`：实际 MicroAPI 向量代码、分块 GM→UB 搬运、BF16 热输出舍入以及冷热权重合并。
- `op_proto/`、`op_api/include/`：新算子的 schema 和 ABI 声明；实际 host/API 实现、注册及编译仍须完成。
- `torch/sfa_cold_merge_bridge.cpp`：正确的原生 `aclnnSparseFlashAttention` 对照，以及独立融合 API 的 Torch 调用源码。缺失新符号会失败，不会自动回退到两遍路径。

当前固定 B1/Q4/K8192/H64/D512/R64/TopK2048/W128，热环256槽，TND/PA_BSND，compact128，热段非空。
第一版不返回 combined softmax stats；必须 `return_softmax_lse=false`。
量化仍使用 baseline，不启用用户模型评测下降的2＋7配方。

## 数值和资源范围

设备 helper 使用原生最大值归一化的统计合同：present 热/冷 sum 约在 [1,128] / [1,2048]，允许窄浮点容差。
它用相对 max 的 ExpSub 和先归一化权重的计算，避免直接 exp(max)。超出声明范围或非法 present stats 会输出 NaN，交由门禁拒绝。
这与 host 参考支持任意有限正 sum 的通用 log 合并范围不同。首次设备测试必须确认实际 reader 分解满足该 native 合同；不满足时需要实现稳定 log 版本，不能删检查或伪造统计。

每次搬运4行冷 FP32 输出，cold scratch 为8192字节，另64字节统计量缓存；同时申请对应事件。
这个算术大小不证明整个原生 kernel 的 UB/事件资源一定够用；以实际 SDK 资源报告为准。
没有手工重写热端 QK/PV，也没有加入额外独立 merge 内核。

## 编译推进

内网代理继续执行仓库 `START_HERE.md`，无需用户再搬运单个文件。
先用已上传的真实 adapter 对隔离原版尝试编译：

```bash
python3 tools/run_logged_stage.py --stage baseline_build --work-dir /tmp/sfa-evidence -- \
  python3 tools/probe_original_sfa.py --work-dir /tmp/sfa-original-probe --compile
```

此命令在当前已配置的 SDK 上运行，不执行 NPU、不安装、不改 vendor。若当前 SoC 上下文不正确，
执行代理自行使用真实 SDK 提供的设置入口（探针提供 `--soc-version`），保留失败日志后修复。
探针必须选中隔离源码、实际调用 compile_op 并生成非空 `.o` 才报告 PASS；PREPARED 或公式测试不能算原生编译通过。
它保留生成 tiling 头和编译元数据，不能承诺 SDK 一定会生成当前缺少的类型定义。

原版编译通过后，按 `cold_merge_bindings.md` 将本目录作为独立候选 overlay，接通真实 host tiling/API 生成与注册。
公共 SDK/shared tiling 布局不能猜；不得只向原来的9输入接口塞入14输入，也不能通过更名另一个 MLA 算子绑定冒充实现。
本目录的 API 头只有声明，复制头文件不会使新符号出现。
用于宿主测试的 stub 目录不得加入真实设备编译 include 路径。

## 本地验证范围

`tests/test_device_merge_reference.py` 直接编译同一设备头，用宿主 MicroAPI/DMA 模型执行数值检查。
`tests/test_candidate_interface_contract.py` 检查 schema/API/kernel 输入顺序、AIC/AIV 指针接线、合并位置和FP32存储一致性。
`tests/test_probe_original_sfa.py` 执行真实上传 adapter 的代码并替换 SDK 边界，验证源码选择、参数透传及产物门禁。
这些不能证明 SDK lowering、硬件时序、原生 host tiling、设备正确性或速度。

设备成功后，再与当前126.17us历史 BF16 两遍方案和纯C4 W0同轮重测；不从独立阶段耗时直接推算收益。
