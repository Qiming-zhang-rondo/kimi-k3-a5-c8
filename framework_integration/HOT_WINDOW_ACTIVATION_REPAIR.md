# 5b16195：环境变量没有接到算子配置

## 最新c0a62e7：hook已进入，键名写错

实际错误只列`kernel`。正确键是`single_pass_kernel`；`reader/source_root/library_dir`已支持，
不要根据人工摘要去重写dataclass。缺失的`diagnostics_path`也须补齐。
当前已增加精确提示，并按该原始配置复现了“键名错误→缺诊断路径→成功解析”的宿主检查。

不要继续手写启动JSON。直接用下文`run_model_probe.py`，它自动生成正确键和诊断路径；
另传`--operator-source "$QSFA_OPERATOR_SOURCE"`，该变量指**本算子仓根目录**，包含
`experimental/single_pass/load_single_pass.py`与`host_contract.py`。
`--mxllm-root`才指MXLLM overlay checkout；两者不能混为一个目录。
`--library-dir`指已经成功的`libqsfa_recent128_direct.so`所在目录。

完整的sfa_recent128对象应为（路径由agent填现场真实值，并合并到已有additional_config，不覆盖indexer配置）：

```json
{
  "window": 128,
  "execution": "accuracy_eager",
  "preprocess": "preserve",
  "max_query_chunk": 512,
  "reader": "single_pass",
  "single_pass_kernel": "kv_split4",
  "source_root": "/actual/operator-repository",
  "library_dir": "/tmp/qsfa-kv-split-build",
  "diagnostics_path": "/actual/new-run/layers"
}
```

用户已确认现有additional_config的indexer KV dtype为mxfp4；原样保留，OFF/ON均保持indexer C4。
新宿主检查明确验证：probe只新增/替换自身sfa_recent128键，其它indexer/vendor键和值不变。
原报告“只支持四个字段”“reader等也被拒绝”与实际错误不符，不能继续作为修复依据。

本次提交只有`hot_window_not_loaded.json`，没有真实worker配置、hook、库身份与候选调用计数。
启动器只新增PYTHONPATH、SFA_RECENT128_CONFIG、SFA_RECENT128_LIB_DIR。当前代码没有读取后两个变量；
它们不是有效启用方式。配置入口是实际VllmConfig.additional_config.sfa_recent128，缺失时window=0。
所以“插件忽略PYTHONPATH”仍是猜测，不能跳过配置缺失这个已确认问题去重装环境或修改插件选择。

本轮已加fail-loud保护：如果这些环境变量存在却没有显式框架配置，在VA配置桥/attention初始化时直接报错，
不再把环境变量ON当作真实ON。它不自动把JSON转换成框架配置，也不放开PD/graph/多请求门禁。

## 内网agent下一步直接执行

1. **不再重启或修改现有P/D服务。** 当前交付仍是accuracy_eager接线原型，不能塞进PD/graph服务后称完成生产接入。
   在线服务按原已验证纯C4配置保持运行。只读保存当前启动脚本差异与实际命令，不反复重启验证假设。
2. 在新隔离PR199 checkout按`KV_SPLIT_VA_TASK.md`安装三文件overlay。安装器检查须返回ALREADY_INSTALLED，
   READY仅表示可以安装。检查platform CONFIG_BRIDGE_INSTALL、worker WORKER_INSTALL、量化前RAW_KV_TAP三个唯一marker，
   保留manifest和runtime/source SHA，不把只有sfa_recent128包的PYTHONPATH目录当作已经安装。
3. 在独立验证进程移除这两个无效启用环境变量，使用现成`run_model_probe.py`，显式：

```bash
env -u SFA_RECENT128_CONFIG -u SFA_RECENT128_LIB_DIR \
  python framework_integration/run_model_probe.py \
  --engine-json "$QSFA_MODEL_OUTPUT/engine.json" \
  --prompts-jsonl "$QSFA_MODEL_OUTPUT/prompts.jsonl" \
  --mxllm-root "$MXLLM_EXPERIMENT_ROOT" \
  --operator-source "$QSFA_OPERATOR_SOURCE" \
  --library-dir "$QSFA_KV_SPLIT_LIBRARY_DIR" \
  --arm single_pass --single-pass-kernel kv_split4 \
  --output "$QSFA_MODEL_OUTPUT/activation_check_split4"
```

变量由agent填为真实路径，沿原真实GLM权重/QLI/量化参数，不再让用户逐项搬配置。
该入口生成完整additional_config：window128、reader=single_pass、kernel=kv_split4、source_root、library_dir、diagnostics_path。
原始engine仍须显式eager、单请求、关闭PD/MTP/PCP/DCP/prefix；不要把服务参数混进去。Prolog问题按原任务处理，
若需native_accuracy必须OFF/ON相同离线预处理，不能悄悄在在线服务禁用Prolog。

4. 在**实际EngineCore/worker进程内**记录sys.modules里ascend_vllm/platform/worker/MXFP4 patch与本overlay的__file__/SHA、
   真实AscendSFAImpl._recent128_installed、方法code.co_filename；记录每层_config.window/reader/kernel及_runtime是否存在。
   若导入来源仍不符，按正式entry-point元数据/worker启动顺序修复隔离进程；另起父进程print/find_spec不能证明子进程来源。
5. 运行collector：

```bash
python framework_integration/collect_framework_evidence.py \
  "$QSFA_MODEL_OUTPUT/activation_check_split4"
```

必须生成真实层/rank JSON，候选attention>0、kv_split_attention_calls>0、归约调用等于split调用，
完整数学检查覆盖实际query/head，并保留actual entry、库SHA、ABI、fallback。generation和collector都通过才算接线。
缺文件先查真实diagnostics路径/容器挂载；不能用环境export或一个80us的Cloud算子条目冒充已调用新算子。

## profiler名称的修正

QsfaSinglePassLaunchKvSplit是host C ABI导出，不要求它原样出现在device profiler。
源设备入口为qsfa_recent128_kv_split_attention及qsfa_kv_split_reduce_fp32_states；编译后名称也须结合源码/二进制核对。
判断使用实际入口、设备任务时序与runtime计数，不能仅靠host名字搜索。

本轮[4,64,576]/packed400已与R64候选格式一致，无需再泛泛猜NoPE272；当前阻碍是配置/启动接线。
提交完整startup.json、startup_traceback.txt（若失败）、probe.json、安装manifest、实际worker来源和collector输出。
不再只写39行人工推测；不重测单算子，不重编已成功内核，也不开展已暂缓的真实激活重要性实验。
