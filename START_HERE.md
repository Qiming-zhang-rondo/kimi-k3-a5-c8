# 当前动作：修复434b09d的InferenceMode清理错误，只补跑B8

最新结果：B1Q1/B1Q4 的 eager、Graph和坏epoch控制共7项通过。traceback停在B1Q4结束后reset_faults，尚未进入B8，不是B8计算失败。B1Q1phase1为全空/全padding对照，不能称为相同输入完美重放。

本次仅Python修复：reader.read/reset_faults自己持有inference_mode，smoke负控制同样持有；不clone、不更换图绑定的缓存地址，不改ASC/C++。新增主机回归可让原未保护清理方法复现相同错误，保护后通过；32项相关宿主检查通过。

内网直接按[framework_integration/SERVING_TASK.md](framework_integration/SERVING_TASK.md)最前面的434b09d步骤执行。复用 /tmp/hot128_serving/reader-build 的既有库；更新Python后用 --graph --case b8_q4 和新输出目录补跑B8。不重编ASC、不加载模型、不修改vendor。省略--case仍可跑全矩阵；B8定向通过会明确标为子集验证。

完整能力边界仍在任务文件内：真实Prolog/APC/P-D自动尾窗接线和完整MTP仍须完成并验收，B1 reader通过不代表服务已完成。目标MXFP4+BF16 recent128及原生产特性保持。

内部实现/原始材料留内网，只回传部门允许的结果、首错和摘要。