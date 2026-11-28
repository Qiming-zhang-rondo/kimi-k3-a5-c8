# 当前任务：15b5ec7 后的提前注册与同 peer 绑定修复

内网 agent 直接执行 [PD_TAIL_INTEGRATION_TASK 顶部](framework_integration/PD_TAIL_INTEGRATION_TASK.md)。本轮已确认C4和BF16最终都是register_memory，没有另一套神秘API。修复把arena注册前移到Mooncake worker构造完成、Multi构造Store之前；同时校验arena/worker/globalTE引擎、源实时RPC/控制metadata及同一个native peer的端口。

208项框架和23项传输/arena宿主检查通过，A5仍待本次curl。两边按原manifest备份更新runtime、原参数启动新worker，不重编attention/CANN/Cloud包、不关闭原功能。执行任务里的同一条真实PD curl，必须同时出现EARLY_ARENA、SOURCE_READY register_ret0、READ_END ret0、tail imported；重算fallback输出不算PD通过。

任务列了完整失败标签和本地日志字段，若仍错只回传脱敏首错/计数/一致性与产物位置，不让用户再转述、不上传私有源/指针/大日志。未声称设备已通过或其他APC/MTP/恢复适配全部完成。
