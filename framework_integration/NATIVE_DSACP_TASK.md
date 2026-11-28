# 按真实Native/DSACP路径接热环：一次实施任务

## 当前70669eb：已经有真实Native/CP候选代码，按此验证

本轮首请求停在`lane.binding`：全局runner行数与DSACP本rank query行数不同。
本次提交补了实际代码，**不要继续安装2fb7562中`.modified`占位文件**：

- `native_writer.py`：真实BF16 Triton写环，GPU bank/绝对position/epoch/valid驱动，同次写payload和tags，无逐token Python循环。
- `native_cp.py`：原始BF16按TP原顺序交换、全局写环binding、本rank query路由；保持原packed gather不变。
- `native_runtime.py/native_hooks.py`：scope贯穿原Native forward，tap只保留原始引用；原store完成后真实汇聚/写环，再交给原V4读取。
- 实际Native路径不再要求Prolog producer；已选Prolog的层继续原provider。只有实际tap/forward/store接线齐全才接受Native，不能直接放开enum。

此版本是正确性候选：TP>1每层增加一次BF16 collective和一个Native写环kernel，还没有A5或性能通过证据。
169项framework宿主检查及9项controller检查通过，不能替代设备/模型验证。

内网直接执行同一任务，不再让用户转发私有源码：

1. 在原目标容器先执行小型真实writer检查，不加载模型、不编CANN/V4库：

   ```bash
   python framework_integration/run_native_writer_smoke.py \
     --device npu:0 --graph --output "$NEW_RUN/native-writer"
   ```

   设备沿用内网已确认的测试设备；不停止或重置其他服务。首次Native Triton编译和Graph重放在这里验证。
   成功仅证明独立writer，输出明确不是CP或模型验收。
2. 同步整个新的公共runtime包，而不是继续拼接占位方法。已有209手工修改可用受控备份替换：

   ```bash
   python framework_integration/install_overlay.py \
     --mxllm-root "$ACTUAL_MXLLM_CHECKOUT" --source-profile pr199 \
     --apply --update-runtime --replace-edited-runtime \
     --output "$NEW_RUN/overlay-backup"
   ```

   该参数只允许已管理runtime的手工修改；原3份MXLLM源的profile/marker/manifest检查仍必须通过。
   实际旧runtime完整备份在外部目录，原量化源和vendor不改。新worker才会加载新代码，旧已加载进程不会自动更新。
3. 用原生产参数重新启动本轮测试worker，重复同一条curl。缺少tap、scope、真实writer或CP路由会明确拒绝，不静默跳过。
   保留并发、Graph、MTP、APC和P/D配置，不把它们关掉掩盖错误。
4. 通过现有worker RPC在真实worker调用`serving_runtime.validate_worker_serving`，检查层producer/read计数与sticky faults。
   不在另一个新Python进程调用空状态后宣称通过；`NOT_EXERCISED`不算通过，warmup计数也不代替真实payload/重放对照。
5. 原量化/主缓存OFF/ON保持，先验证同一请求的真实Native写入、TP8跨rank热KV和本rank读取。
   **P/D与APC自动BF16尾窗接收仍是已知未完成接线**，缺尾窗必须按原任务补传输/import，不能将其标ready或退C4。
   相同生产请求端到端通过且无fault后，才执行同一套198题GPQA；不能把占位版或C4回退分数记为hot128。

下面是本候选的源合同与实现依据，已实现项不再交给内网重新写一遍。

41bef5c 的方向有可用部分，但“永远不选Prolog”“已有smoke可直接复用”“零风险支持DSACP”均未被证明。
本轮保持生产原选路，给实际Native路径补producer，不强制关DSA-CP或改为Prolog。已有Prolog variant保留给真正选中它的分支。

## 已核对的公开源合同

来源是PR199声明的VA `c09d3e20b9c45c80b34ae3781440d90904a44023` 与已审阅MXLLM PR199源。
这不是声称容器中的文件必定相同；实际worker的路径/散列必须核对。

- `patch_sfa_v1_mxfp4.py::_mxfp4_custom_kv_rmsnorm_rope`：原`npu_rms_norm`返回`k_nope`，原`npu_interleave_rope`返回`k_rope`。
  R64时源形状为`[T,1,1,512]`和`[T,1,1,64]`。它们正是原量化前latent和post-RoPE来源；BF16激活路径保持BF16原值，不能重算FP32 RMSNorm再称原值。
- 原`npu_dynamic_mx_quant`消费`k_nope.contiguous().view(-1,1,512)`，量化配方和packed400不动。
- 本仓`install_overlay.py`已在`k_rope -> prefix_shape`之间插入`capture_native_kv`；现有serving分支有意跳过这个tap且拒绝Native，尚没有生产Native writer。
- `AscendSFADSACPImpl.exec_kv`的packed分支调用base，所以能到该tap；R64 attention也委托base reader。
- **tap只有TP本rank的token分片**。随后`_prepare_kv_for_parallel`汇聚packed C4，`_store_parallel_kv`写完整主缓存；原BF16不在packed gather中。
- DSACP按TP分token，PCP/DCP配置均为1不代表未分片。DSA+DCP组合还可能覆盖reader并绕过base hook，不能按类名一概放行。

对应公开源码：[DSACP类与KV路径](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/attention/context_parallel/sfa_cp.py#L620)、
[实现选择](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/attention/context_parallel/sfa_cp.py#L2063)。

## 先把本次运行身份一次核对清楚

在实际worker使用本轮诊断，`serving-contract-*.json`新增`dispatch_state`：原additional开关、validated AscendConfig开关、
`enable_dsa_cp()`实际结果及cache_info、TP/PCP/DCP、MRO、实际绑定writer/gather/store/reader方法位置和已加载模块SHA。
不清cache、不改开关、不导入另一套框架；诊断仅读现有worker模块。报告留内网。

在对应版本中，false不会无条件选DSACP。若D节点三处开关/类不一致，先按实际模块路径、初始化时序和override修复，
不要把原JSON的false当成整个worker已生效。早前52db594报告到达新provider绑定，也不能与本次Native实例混为“从来没有Prolog”。
不需要重新加载权重只为收集已存在日志；下一次真正启动时诊断自动随原失败收齐。

## Native候选按以下完整链实现

1. producer按真实已选路径分派：Prolog走已有provider；Native走新的原始BF16 sidewrite。
   启动能力检查和runtime只能在Native writer及实际CP路由接通后接受该组合，不能先删guard。
2. 在既有原量化前tap捕获原始`k_nope/k_rope`引用，使用每次调用独立的scope，不能用共享实例字段串扰draft/ubatch。
   保持原`exec_kv`返回对象、RMSNorm/RoPE、量化器、C4 packing、store和query计算不变。
3. 新Native writer直接接受真实BF16 GM输入和现有`RingBinding`，用GPU bank/position/epoch/valid寻址，payload与tag同次写入。
   不复用FP32输入的`QsfaPrologRingSmokeLaunch`冒充BF16接口；复用实现时必须建立明确BF16输入ABI及新对照。
4. 对DSACP完成高精度KV分片交换：使用与原packed gather一致的token顺序、padding和trim，保留真实BF16而不是C4反量化。
   写环必须覆盖随后本rank query可能读取的所有热KV。不能只写local shard却用全局TopK和全局ring路由。
   若先用独立BF16交换作正确性候选，明确记录额外collective/传输字节；之后才尝试合入原通信，不能宣称零性能开销。
5. reader同步适配CP的local query行与全局request身份、绝对position、query boundaries、lengths和bank/epoch。
   `dsa_cp_context.local_start/local_end_with_pad`与全局metadata的对应关系来自实际builder；pad行masked，不靠相同shape猜对应关系。
   固定shape buffer在warmup前分配，Graph replay更新内容；禁止CPU读取GPU tensor或动态compact导致出图。
6. 保留同请求跨chunk历史，C按`align128(128+max_query_chunk-1)`；实际chunk必须在界内。
   P/D、APC和MTP继续真实尾窗/生命周期接线，不以新增Native writer自动宣布它们完成。

## 一次验收并保留生产功能

先独立候选，不覆盖工作基线；不再重编现有Prolog来解决Native选路。

- OFF/ON原query、packed400与E8M0 scale逐字节一致；原始BF16 latent/KR、position/epoch逐项正确。
- TP1退化路径与实际TP8 DSACP分别测；构造热KV跨rank边界和Graph padding，防止local-only写入假通过。
- B1/B8、Q4与真实chunk Q512/Q1024跨chunk/跨环、混合prefill/decode及Graph毒化重放；收集sticky faults。
- 实际P/D/APC/MTP用原配置，不关闭特性；按真实命中/传输边界验证BF16尾窗。
- 同轮记录额外kernel、collective字节和端到端TPOT/TTFT；Native路径可能增加GM读写/通信，不能承诺与纯C4相同性能。

内网agent在本地自主完成上述候选和控制；不得要求用户上传私有头、源码、schema、二进制或大包。
只提交允许的运行身份小表、实际错误、控制结果和计时摘要，不用“零风险/完全兼容”替代设备证据。
