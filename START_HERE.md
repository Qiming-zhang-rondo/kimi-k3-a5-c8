# 当前任务：真实 P→D BF16 recent128 接线候选已完成，先跑一条 curl

内网 agent 执行 [PD_TAIL_INTEGRATION_TASK 顶部](framework_integration/PD_TAIL_INTEGRATION_TASK.md)，不再重复身份采集。

0d9421f 的 AscendMultiConnector 源码SHA已匹配公开c09d3e20。候选递归连接其真实MooncakeConnectorV1子对象，保留Store、原C4传输和attention内核：scheduler提交边界→P释放前固定BF16尾窗→沿既有控制socket/RDMA传输→D导入完成→精确ACK→原接收完成信号。

本轮195项框架及18项传输核心宿主检查通过；A5尚未验证。按任务备份并更新两节点runtime，以原服务参数启动新worker，先原单curl走真实P→D，再短prompt。不重编kernel/CANN/Cloud包，不关闭Graph/MTP/APC/并发/QLI。当前传输边界同步成本尚未测量，Store/APC任意前缀侧缓存仍是单独适配项。

只推本轮实际PID、curl结果、首错与日志位置的小报告，不上传私有源/大日志，不让用户继续转述问题。通过后继续既定性能和GPQA任务。
