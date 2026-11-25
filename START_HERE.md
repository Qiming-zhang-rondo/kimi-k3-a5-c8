# 当前只推进V4框架接入和真实模型精度

9ffe1e0原始证据确认V4约82.08µs，距同轮C4仅2%；用户下一任务为接框架，先不继续tile扫描。

直接执行 [framework_integration/CURRENT_FRAMEWORK_TASK.md](framework_integration/CURRENT_FRAMEWORK_TASK.md)。probe选择load_tile4，runtime/collector核对实际variant、geometry、库SHA和调用覆盖，CPU全数学不调用第二次NPU attention。

先JSON-only预检实际checkpoint格式。现有kernel仅D512/R64/packed400；GLM-5.3-Flash源码存在R0/272路径，不能靠改shape接入。R64兼容则隔离安装→真实prefill/decode→原精度任务OFF/ON；若确认NoPE272，按任务一次收齐实际NoPE源合同并提交，不让用户逐项中转，不宣称已支持。

48项框架宿主检查通过；实际模型、任务分数和生产图/并发/MTP/PD未验证。82µs范围为prepared-read，不是TPOT。普通源码定位/构建问题内网agent自主解决，不再启动历史优化扫描。
