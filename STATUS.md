# 当前可审查状态

依赖审计基点：`5c10d9629a5c41e6a9fae27a5cb8146a8c886d20`。

| 项目 | 状态 | 证据/后续动作 |
|---|---|---|
| 前批项目头文件 | 已上传 | 六个基础VF、utils、memcopy目录已在deps；不再向用户索取这些文件 |
| 标量合并合同 | 宿主测试通过 | 直接编译头文件的13项回归 |
| 原生融合源码 | 候选已实现 | experimental目录：GM参数接线、MicroAPI epilogue、FP32输出 |
| 新接口 | 源码声明/桥接已写 | proto/API声明/Torch bridge，尚未生成或注册新二进制符号 |
| 隔离原版编译工具 | 本地逻辑测试通过 | probe_original_sfa.py；实际SDK编译需内网执行 |
| 宿主测试 | 51项通过 | `python3 -m unittest discover -s tests -p 'test_*.py'` |
| 原版 AscendC 编译 | 未取得本轮成功证据 | 内网代理运行probe并记录真实产物/生成头/首个错误，自主继续处理 |
| 候选 AscendC 编译 | 尚未验证 | 需要真实host tiling/API生成注册及SDK编译 |
| A5 正确性/性能 | 尚未验证 | 先数值门禁，再同轮AB/BA，历史126.17us不是候选结果 |

本次外部环境只执行宿主 C++/Python 测试和 prepare-only。没有在本机调用 AscendC 或运行 NPU。
内网执行按START_HERE完整推进，将源码及实际日志直接提交GitHub；普通缺依赖或编译错误由代理处理。
