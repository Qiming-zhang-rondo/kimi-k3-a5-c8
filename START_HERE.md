# 当前任务：修复 e01b953 的 BF16 PD 注册区域错误

内网 agent 直接执行 [PD_TAIL_INTEGRATION_TASK 最新节](framework_integration/PD_TAIL_INTEGRATION_TASK.md)。新代码已把P/D都改为启动建链前注册2MiB对齐的持久有界arena，payload使用区内槽位，不再按请求动态注册clone；同时修复多D共享源时第一个ACK过早释放的问题。

202项框架和23项传输/arena宿主检查通过；A5仍待验证。沿原manifest流程备份更新两节点runtime，再以原参数启动新worker，让注册早于native KV/peer连接。不改attention/CANN/Cloud包，不关闭Graph/MTP/APC/并发/QLI。

先本次失败短prompt，再不同前缀短prompt，两次走真实P→D并复用连接。必须出现真实BF16 tail imported，实际worker统计successful_reads/completed_imports增加、failed_reads为0；重算fallback后输出成功不算本轮通过。只回传PID、arena字节数/槽数/层数、实际计数、curl结果和首错，不上传指针/descriptor/私有源或大日志。
