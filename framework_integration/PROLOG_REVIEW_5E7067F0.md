# 5e7067f0审查：写点方向正确，当前三输入原型不能接生产

## 最新192946ea：接受已修方向，剩余三项一次补齐后编译

四参Cast、bank*C+position%C寻址、token_valid/bank范围谓词和静态容量的修改方向正确。
这些项不再反复改写。当前仍NOT_COMPILED；成员变量声明不等于KR/payload/tags已实际正确写出。

下一步只收敛以下三项，然后在内网允许的隔离环境直接编译和验证；不要仅因“等待Codex反馈”自行停住，
若内网用户另有明确暂停指令则遵守该指令：

1. **scratch及流水**：当前PIPE_V只约束V内部，不保证Cast结果在MTE3读取前就绪，也不保证sharedBuf在DMA结束前不被Quant覆写。
   给BF16侧路明确独立区域或已证明不alias的scratch生命周期；Cast→DMA有V_MTE3或等价依赖，
   DMA→后续覆写有MTE3_V/对应写管线依赖。独立区域可并行，不无条件加入host/global同步。
2. **参数链一致**：kernel入口已改8个GM+2个scalar，wrapper/TBE仍旧3输入，当前不能launch。
   同一个独立variant同步Python wrapper、注册/C++ binding、op_proto、TBE、tiling和kernel入口。
   容量/池大小选择明确的静态tiling/attr传递合同；若实际构建链确实支持raw scalar launch则完整生成并校验，
   不手工插入两个scalar却沿旧生成ABI调用。workspace/tiling参数位置必须核对，避免错位导致设备fault。
3. **未实现分支拒绝**：SplitM/arch22可用no-op保持OFF编译，但ON必须由host binding/tiling在launch前拒绝。
   所选A5/mode4分支必须真正写latent、post-RoPE KR和position/epoch tags；不能仅给函数ABI标记1。

wrapper须仍兼容公共prolog_provider的完整关键字合同，max_query_per_request可只在host做容量验证。
编译产物和源都隔离，不再写工作vendor，原C4/indexer配置不动。先B1单token+回绕，再B8Q4/padding/reorder，
最后同图修改输入/位置/有效掩码并毒化热环重放。只有数值通过才计时、才推进模型服务接线。
只回传允许披露的真实编译首错或测试结果，不要求内部源、diff或schema外传。

**P/D作用范围**：公共新接口ring=None时原样走原Prolog；有真实RingBinding时才写环。
P和D可以共享新producer：P至少准备prompt尾窗给D，但P的attention不必同时改热窗；D每步写新token/draft。
两节点source/MD5相同只证明文件同步，尚未编译或真实调用。PD尾窗传输与完成协议仍须另行接通，不能只传C4后假定BF16已到。


本轮只有实现摘要，没有编译或设备结果。状态应为IMPLEMENTED_UNCOMPILED_LATENT_SMOKE，
不是LOCAL_COMPLETE/ABI1_COMPATIBLE，也不能以函数ABI标记替代真实实现。
内部源和diff继续留内网，不要求补传文件。

## 先修两个确定问题

### 1. Cast重载

当前摘要六参`Cast(dst,src,CAST_RINT,512,1,{1,1,8,4})`选择mask/repeat重载，第4参不是总元素数。
常规FP32连续mask每repeat最多64；不能用mask512、repeat1表示512元素。
除非有明确验证的counter模式合同（当前摘要没有），先改计数重载，避免手写mask/stride：

```cpp
Cast(bf16Local, inputLocal, RoundMode::CAST_RINT,
     static_cast<uint32_t>(kvElemCount));
```

必须在实际CANN9.2.1/A5 SDK编译，并检查512个元素全部符合BF16 RNE，不只检查前64个。
官方重载参考：https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0073.html
这是API依据，不把9.0文档当作目标9.2.1构建成功证据。现有已验证kv_split_reduce也使用四参计数形式。

### 2. ringSlots含义与寻址

公共接口的token_ring_slots是**请求池bank编号**，token_positions才是逻辑位置。
当前`ringSlot % capacity`丢失这两者的区别。B1全部bank0时会每个token覆盖slot0；B8也不能隔离请求。
正确逻辑行号为：

```text
bank = token_ring_slots[token]
position = token_positions[token]
row = bank * capacity + position % capacity
latent element offset = row * 512
rope element offset = row * 64
position_tags[row] = position
epoch_tags[row] = token_request_epochs[token]
```

仅当token_valid为true、bank在[0,P)、位置非负时写；padding必须完全不改payload/tag。
如果内部三输入smoke把logical position塞进ringSlots，它是另一临时合同，不能标ABI1。
本仓`producer_reference.expected_ring_writes`是CPU离线参考及回绕/多请求/padding测试，不加入图内执行。

## 关于其余三个问题

- **capacity设备读取**：GetValue是设备标量读，不等于CPU同步或必然打断Graph；容量固定时优先来自tiling/静态形状，
  不增加不必要的每tokenGM读取。须校验C为正、128对齐、C>=128+max_query_per_request-1，再做取模，避免除0/越界。
- **逐token还是批量**：在一次Prolog内按现有token/tile循环写是可行的第一版，不需要按请求额外Python launch。
  先确认tokenIndex是实际全局token行，不是每核局部0；确认每个完整行的writer或分列协作。正确后再按连续目标run合并DMA。
- **single pool**：仅可用于隔离B1/单token底层smoke。实际目标B8，不能用P=1版本接D服务，也不能声称完整ABI1。
- **TBE安装**：不通过chmod/sudo继续写工作vendor的dynamic目录。使用独立源码/构建和candidate vendor/OPP作用域，
  新operator名字，不覆盖原custom_cot；或用本地已有独立AscendC launcher做smoke，但tiling必须来自同一候选源。
  摘要称“隔离”却记录已写当前vendor源码目录，不一致。现有新目录先留档，不擅自删除或替换工作产物；后续停止修改该vendor。

## 完整ABI1还缺的实现

按已交付prolog_provider接口补齐recent_rope、position_tags、epoch_tags、token_positions、
token_request_epochs、token_valid及request-pool寻址。max_query_per_request可以只用于host/tiling容量验证，
不要求kernel逐token读取它。原五个query输出、主C4字节/scale及indexer C4保持不变。

SplitM/arch22的空SetRecent128Ring只可服务OFF编译；ON遇到未实现/未验证的dispatch必须由host binding/tiling在launch前明确拒绝，
不能算子成功返回但完全不写热环。不要靠ABI marker=1掩盖缺失副作用。

## UB与流水同步

BF16 scratch不允许覆盖FP32 inputLocal。Cast(V)到GM写(MTE3)要有就绪依赖；
若sharedBuf紧接着被MXFP4量化覆盖，先确认该次MTE3读取已完成再复用（MTE3→V/相应写管线）。
独立buffer可以与量化并行，不为此增加host/global同步。
官方依赖说明：https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00019.html

## 内网agent现在直接完成，不再等待逐项确认

1. 先修Cast、scratch生命周期和临时B1寻址，独立编译单token/回绕smoke；临时接口保持smoke标记，不对生产宣称ABI1。
2. 随即补齐完整ABI1参数/多pool/KR/tags/valid与未支持分支拒绝，再构建生产候选。
3. 必须原C4 packed及scale在OFF/ON逐字节一致；BF16等于真实prequant FP32的RNE，KR等于post-RoPE参考。
4. B8Q4同一逻辑位置不同bank/epoch、padding、回绕、request重排与MTP暂存/提交不得串写。
5. Graph固定地址捕获后改变输入/slot/位置/valid，毒化payload/tag并replay，检查真实更新和无效行未写。
6. 验证通过才测同轮Prolog OFF/ON；不把scratch独立计时简单相减预测全流程。

只回传部门允许的编译状态、覆盖矩阵、最大误差、C4一致性、Graph结果与配对时间；私有源/diff不外传。
