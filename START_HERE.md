# 当前任务：真实Prolog内部inline写环，保留P→D尾窗

最新7ba01aa证明独立reader的B8Q4 eager/Graph/epoch通过；结合434b09d，分两轮完成B1/B8共11项。它不证明真实Prolog、原生量化器、生产execution lane、整服务或性能已验证。reader阶段不再重复跑。

f23f328的UB输入来源早已定位：实际RmsNormAndQuantizeCkv内的最终QuantPerTileMxFp4输入LocalTensor。独立producer不能读另一个kernel的UB。现在直接走原定B路线，不再让用户选方案：同一Prolog内 FP32 UB→独立BF16 scratch→ring GM，没有FP32 GM中转或额外kernel。

已交付可复用inline代码 experimental/prolog_ring_smoke/prolog_ring_write_device.h，支持真实列分片、KR、bank/position/epoch/valid和独占事件。2项host shim/C++17+UBSan检查通过；helper不是完整Prologvariant，未在A5编译，也不提供/伪造私有tiling或hostlauncher。

内网直接执行[PROLOG_INLINE_INTEGRATION.md](experimental/prolog_ring_smoke/PROLOG_INLINE_INTEGRATION.md)，在现有真实Prolog写点包含helper，保留原混合AIC/AIV任务、真实tiling/workspace/dispatch，CPU构建完成后一次验收producer矩阵。内部实现和原始构建资料留内网，不再要求上传。

同时纠正a139cb79：长prompt初始decode也需要P→D尾窗。例如prompt8192、D刚写1个新KV，最近128仍有127个来自P。GPQA长题不能免除此项，不以部分回退C4冒充完整recent128。完整能力边界、公式和任务见[framework_integration/SERVING_TASK.md](framework_integration/SERVING_TASK.md)。

真实Prolog构建接通、APC/P-D自动尾窗传输、完整MTP/模型精度/性能仍未完成。公共Python适配、inline helper和独立reader的完成状态必须分别汇报，不再合称Prolog或服务已完成。