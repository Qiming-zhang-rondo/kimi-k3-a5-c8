# 当前任务：一次核实Python选路和宿主条件，继续recent128服务

最新错误 `79334045` 是实际层没有选择PROLOG_V3；旧报错中的“explicit ring producer”不是TBE文件存在检查。首次动态编译尚未发生，不能把本错归因于TBE尚无生成产物，也不能通过强改enum或关服务功能绕过。

内网agent直接按 [SERVING_TASK最新入口](framework_integration/SERVING_TASK.md) 继续。先自行定位本轮实际engine JSON，用新目录运行聚合检查：

```bash
python framework_integration/serving_preflight.py --engine-json "$ACTUAL_ENGINE_JSON" --output "$NEW_PREFLIGHT_DIRECTORY" --check-host
```

默认不加--check-host时仅读JSON/文件；加上后在实际worker依赖环境分别核实新provider函数ABI、原Cloud两个选路入口、reader ABI和宿主工具，不编译bridge或调用设备kernel。预检一次列出全部可见问题及明确pending项；它不是完整服务验收。

同步本轮contract.py、serving_runtime.py及launch_guard.py到现有overlay。坏层在分配sidecar之前聚合报告实际选路、层名/实现类、量化类型、角色、几何及原框架可取得的拒绝原因，写入diagnostics_path/serving-contract-*.json。即使reason probe本身失败，也保留根本选路错误。原selector和权重不改。

按这些实际值在内网自主修复选路前提或相应路径的producer接线，不要反复让用户转发私有文件。已审阅DSACP路径可能禁用fused，必须检查真实类和配置，不能假定安装新版Prolog会改变原选路。

[当前全部检查范围与功能边界](framework_integration/SERVING_REQUIREMENTS.md) 已列出。保留并发8、Graph、MTP、APC与P/D；实际尾窗接收/恢复仍需完整接线。复用成功的SDK、variant和reader，不重做baseline构建。143项框架宿主检查通过；实际未选Prolog的原因仍须本轮运行诊断，不能宣称服务已经修通。
