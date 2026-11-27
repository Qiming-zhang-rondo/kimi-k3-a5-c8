# 54076f89：保留生产PrologV3，补齐原始BF16热环写点

用户明确要求保留可支持的并发8、整模型Graph、MTP、prefix cache和PD；不能关掉这些功能后称完成生产接入。
当前真实错误17:24:36已通过sfa_recent128配置解析、进入hooks.weights/Recent128Runtime构造，
在Prolog检查处停止。构造未成功，也尚无实际融合kernel调用证据。它不是旧kernel字段错误，不再改JSON键名。

## 已审阅的真实调用合同

PR199 head342680fb的`patch_sfa_v1_mxfp4.py`：

- `_call_mxfp4_cloud_prolog`固定MXFP4 mode4、repo1、tile128；原地`cloud_mla_prolog_v3`返回5项query相关输出。
- functional额外返回的kv_out/kr_out用于提交量化cache，不是原始BF16 latent；MXFP4 kr_cache为空4D dummy，
  RoPE本来就在packed400里。q_norm是query归一化中间值，不能拿它冒充K latent。
- 当前Python raw-KV tap在NATIVE exec_kv中；生产Prolog不经过该tap。hidden_states/token_x与NZ权重、scale虽在外部，
  latent是在算子内部投影/RMSNorm/RoPE产生。额外重算一次这些步骤会改变性能，不作为生产方案。

这只能证明**已审阅调用合同没有原BF16 K输出**，不能替代当前部门binary的真实schema审阅。
本地尚无实际Cloud Prolog源码/ACLNN头；先核实部门当前包是否已有可复用side output。

## 内网agent一次完成，不再让用户搬文件

本轮不重启服务、不改原CANN/vendor/wheel、不重编原SFA，也不把preprocess改NATIVE冒充生产修复。
在**实际D容器的原匹配环境**中自主定位下列材料，保持全部原始文件与来源身份：

1. 当前被调用的cloud_ops_turbo包路径、版本与两个Prolog Python wrapper；解析实际torch.ops调用和注册schema。
   可在单独子进程导入当前已安装包后只读`_schemas`/`default._schema`，不执行任何算子；
   记录实际C++绑定源或二进制SHA，不假定旧`/tmp/cloudops`就是当前包。
2. 当前对应`aclnnMlaPrologV3`及functional接口头、op_proto、op_host/tiling和真正op_kernel/AscendC源，
   包含实际include依赖与CMake/构建脚本。把量化mode4的KV投影→RMSNorm→RoPE→MXFP4量化→写GM路径定位到具体行。
3. 记录当前源commit/dirty、CANN/compiler与实际vendor版本、已加载.so路径/SHA，关联到正在运行的PR199调用，
   不以官方master或另一个版本同名文件替代。不要上传凭据、完整环境或模型权重。

用本仓stdlib采集工具导出真实source根目录（多root可重复传，路径由agent自主定位）：

```bash
python tools/collect_prolog_sources.py \
  --root "$ACTUAL_CLOUD_PROLOG_WRAPPER_ROOT" \
  --root "$ACTUAL_PROLOG_SOURCE_ROOT" \
  --output /tmp/prolog-producer-source-export
```

直接推送工具生成的manifest和全部小文本part到`framework_integration/evidence/prolog_producer_sources/`，
另附实际schema、入口链与短日志。工具不导入Torch/NPU、原样保留源字节并记录SHA；
预算截断/缺根/只有binary必须如实报告。大文件自动小文本分片，不再以12MB无法推送为由只写人工摘要。
按真实include继续补齐依赖，不能为了匹配工具仅改名，也不能把参考代码误标当前部署源码。

## 保留生产路径的最小实现

若实际已有可用BF16出口，先精确验证dtype、shape、生成阶段及旋转基底；确认后复用。
否则在独立版本化variant中增加**可选BF16热环写入**：

- 保持原9个位置参数、mode4量化参数、原五个query输出和原packed400 C4写入合同不变；未启用热环走原路径。
- 新增预分配BF16 latent/rope ring、逻辑位置/request epoch tags，以及device token→稳定request/ring bank、
  logical_positions、token_valid。不能由物理cache_index反推请求或逻辑位置；不能按当前batch行绑定热环。
- 在同一次KV投影/RMSNorm/RoPE之后、MXFP4量化之前，把内部中间值写入指定热环。
  若内部为FP32，侧路单独BF16 RNE，不改变原C4量化输入的舍入/缩放/权重路径。
- padding/slot<0禁止写入；容量覆盖W与在途spec/chunk；payload和tag均完成后才让同stream的attention读取。
  复用已有UB与正确event/fence，不增一次投影、attention、CPU同步或整cache复制。
- 使用新导出或明确协商的可选ABI，不能向旧binary多塞未经支持的参数，也不能靠改execution名称取得不存在的数据。

内网agent若有完整源和SDK，可按此合同在隔离目录实现、构建新variant并推送真实diff；
保留原已验证库和C4服务，不覆盖原产物。外部Codex继续审阅/修改source；不要求用户逐项转交函数或include。

## 构建与设备验收

先独立验证producer，不用全量模型启动掩盖问题：

1. OFF与原Prolog比较全部原输出、原C4 packed字节及量化参数；不得因新增侧输出改变原计算。
2. ON热环对同一原始prequant数学/已有NATIVE证据，验证BF16舍入、RoPE、逻辑位置、epoch、padding、回绕、
   B8多请求重排、spec接受/拒绝；不能以C4反量化作为参考。
3. Graph固定地址输入/输出/workspace，捕获后改变输入位置/有效mask并毒化热环，replay验证真实更新；
   不通过NATIVE Python tap或额外CPU回填来让测试通过。
4. 保留原Prolog的同轮OFF/ON event/wall和profiler原始计时。新的BF16写有真实成本，只能实测，不能承诺零开销。

这解决BF16 **producer**，不自动等于全部生产生命周期已完成。reader随后接入现有VA GPU
positions/seq_lens/query_start_loc/block_table/slot_mapping，增加多请求寻址、静态buffer和padding掩码；
MTP复用runner已修正的device位置并处理draft提交；prefix用有预算的精确边界BF16尾窗快照；
P/D还需独立BF16尾窗payload、完成事件与身份校验。主C4布局与indexer C4保持不变。

当前阶段必须如实标记为SOURCE_EXPORTED/BUILD_PASS/DEVICE_VALIDATED，只有原始证据通过才推进；
不能再把runtime构造入口叫初始化成功，或把配置过检查叫融合算子已经运行。
