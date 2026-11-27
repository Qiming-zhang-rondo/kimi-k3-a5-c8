# Prolog热环：仅在内网实施，不再上传内部源码

用户已明确：45个文件的批量外传不符合部门要求。之前的源码采集/上传要求撤销。
不要继续上传剩余文件、内部头文件、实现diff、schema或二进制；不以任何外传材料作为继续实现的前提。
原始代码、构建与测试证据均留内网。对外只回传部门允许披露的能力结论和脱敏测试结果；不允许外发的内容不回传。

## 本轮直接继续实现：fac32bf36已定位正确KV写点

本任务不是继续搜集材料或等待确认。用户已授权推进生产能力；内网agent现在就地编码、构建并验证，内部源码不外发。
fac32bf36报告的真实V3/mode4链已进入RmsNormRopeScatterCkvKr→RmsNormAndQuantizeCkv；
量化前FP32 inputLocal可用。该提交仍是IDENTIFIED，明确未实现、未编译，不应写成生产已接通。

写点语义是BF16_RNE(actual input to QuantPerTileMxFp4)，检查其kc_scale/clip/rotation等是否都已生效。
KR64从真实post-RoPE输出另路捕获；不要把K和KR视为同一个inputLocal，也不要在reader重复scale。
**禁止原地把FP32 inputLocal转换为BF16**。原量化继续读原FP32；侧路用不alias的BF16 scratch，
复用前等对应MTE3读完成，按实际管线所有权放同步；独立buffer可与量化并行，不无条件加整流同步。
多核分列时必须保证整行payload和tags的唯一写者/完成关系。

## 外部接口已经写好，私有实现按此对接

本项目的`sfa_recent128.prolog_provider`提供OFF透明直通、ON元数据校验和明确的能力检查；
新公共Python入口约定为`cloud_ops_turbo.cloud_mla_prolog_v3_recent128`，这是**待内网实现的新接口**，不是声称旧包已有。
新函数对象声明`recent128_abi_version=1`，不存在/不匹配时直接报错，不能回退NATIVE或拿反量化结果伪装。
它接收原Prolog9个位置参数和原kwargs，新增以下明确sidecar参数；原参数与五个query输出不改变：

| 参数 | 合同 |
| --- | --- |
| recent_latent | BF16 [P,C,512]，预分配请求池热环 |
| recent_rope | BF16 [P,C,64]，同请求池/容量 |
| recent_position_tags / recent_epoch_tags | INT64 [P,C] |
| token_ring_slots | INT32 [T]，每token稳定请求池槽，不能是当前batch行或C4物理槽 |
| token_positions / token_request_epochs | INT64 [T] |
| token_valid | BOOL [T]，graph padding及无效slot禁止写入 |
| max_query_per_request | 正整数静态容量界；C按128对齐且C>=128+该界-1 |
| recent_window | 固定128 |

T为token_x前导维的展平token数；所有sidecar与token_x同设备、连续，地址在图捕获前固定。
GPU有效mask、slot、位置和epoch由框架step级准备，新producer按值执行掩码和边界检查，不能在图内CPU读取。
当前Python接口适配器只验证shape/dtype/device等元数据，不调用.item/.cpu/.tolist，不分配替代张量。
它还未挂进生产runtime，也不等于Graph或B8已通过；内网先实现并注册这个producer接口，随后对接生产元数据。

按此公共合同在内网修改真实V3 wrapper/绑定/新op variant/tiling/kernel，旧入口保持原样。
不要求上传这些私有改动。新wrapper可以在模块初始化时为函数设置ABI版本标记；
内部注册名和调用方式在本地自洽，绝不能仅给旧函数贴标记而没有真实热环写入。

先做producer独立控制：OFF与原Prolog原query输出和C4 packed/scale字节一致；ON仍保持C4字节一致，
sidecar等于本次实际FP32量化输入的BF16_RNE。覆盖B1Q1/B1Q4/B8Q4、环回绕、padding、request重排和MTP暂存/提交。
Graph捕获后修改token_x、位置、slot和有效mask并毒化sidecar，replay检查payload及tag真正更新。
只测producer的B8通过不能被写成batched attention已支持；后续reader另有多请求寻址与任务调度工作。

## 42b0f78阶段材料说明（已由fac32bf36的正确KV定位推进）

42b0f78提供的5份关键文件已足够发现一次入口选错，无需再上传其余40份：

- Python cloud_mla_prolog_v3 wrapper调用cloud_ops_turbo_lib.cloud_mla_prolog，返回5项query相关输出。
- 所选aclnn_cloud_mla_prolog.h明确为旧接口，提示迁移V3WeightNz；该头仅有query/queryRope输出。
- 所选cloud_mla_prolog.cpp定义MLA_PROLOG_VERSION=1，当前展示分支不足以确认MXFP4 mode4的真实实现。
- service_dynamic_quant_qn_mul_qr_arch35.h处理mmQn/query与Qr，是Q动态量化，不是K latent量化前写点。
- 公共tiling里出现kvQuantMode字段不能证明入口和真实部署路径已经匹配。

本轮上传的源码根限定为旧cloud_mla_prolog目录。先在内网追踪真实Python→绑定→ACLNN→kernel的mode4链，
不在这些Q量化函数或旧V1入口盲插热环，也不为此继续向用户索要文件。

## 内网agent直接实施

1. 在当前匹配CANN9.2.1的部门环境内，只读核验实际注册/调用链及权重layout、query/cache dtype和返回值。
   源码自行在本机搜索，保持内部路径、实现、二进制在内网。
2. 如果当前真实接口已经能写出同基底、同阶段的原始BF16 K latent，验证后复用；
   否则在独立版本化variant中增加可选的原始BF16热环写入。不能从C4反量化补造它。
3. 写点必须是同一次KV投影→RMSNorm→RoPE之后、MXFP4量化之前；保持Hadamard/折权重基底一致。
   原中间为FP32时侧路单独BF16 RNE，不改变原C4量化输入、scale算法、packed400字节或query输出。
4. 从框架传入预分配BF16 latent/rope ring、逻辑位置和request epoch tags，
   以及device token→稳定request/ring bank、positions、token_valid。不要由物理cache槽反推请求或逻辑位置。
   容量覆盖窗口与在途query/draft；padding和无效slot禁止写入。
5. 复用Prolog内部中间结果与正确MTE/UB同步，payload/tag均完成后让同stream attention读取。
   不增加第二次投影、不重复attention、不整cache复制，也不通过强制NATIVE改变生产预处理。
   未启用热环仍走原Prolog；新增能力使用明确的新导出/可协商接口，不能向旧二进制盲加参数。
6. 源码和构建都在隔离目录完成，保留原已验证产物和服务；不升级环境、不覆盖vendor、不重启正在工作的P/D。

## 内网自行验收

- OFF与原Prolog：原query等输出与C4 packed cache逐字节一致，记录真实差异；不放宽原数值门槛。
- ON：热环对本次真实prequant BF16参考，验证逻辑位置、request隔离、回绕、padding、MTP接受/拒绝；
  不用Q norm或C4反量化当参考。
- Graph：固定地址buffers，捕获后修改设备输入/位置/active mask并毒化热环，再replay检查实际覆盖，
  不靠Python回填或CPU同步使结果通过。
- 性能：同轮Prolog OFF/ON、相同batch8/spec3/graph与原地写cache合同，保存内网原始event/wall及profiler。
  新增BF16存储有真实成本，数据出来前不称零开销。

只在允许披露的范围内回传：真实mode4入口是否匹配、可复用出口是否存在、实现/构建状态、
原C4字节是否一致、BF16热环最大误差、Graph动态重放结果、B8/MTP隔离结果、同轮延迟比值。
这些结论按实测状态写；缺项就标未测，不需要对外发原始内部代码来证明。

外部Codex继续负责本项目自己的reader/请求映射/Graph缓冲及接口对接；
Prolog私有实现由内网agent按上述合同完成，不要求用户充当源码中转。

## 生产范围仍须如实推进

当前54076f89错误发生在配置解析通过、runtime构造的Prolog检查处，并不代表融合kernel执行失败。
改execution名字或删guard不能生成缺失的BF16数据。
producer接好后，还须接多请求row→request/ring寻址、静态图buffer、MTP事务、prefix尾窗快照和PD尾窗完成协议。
主C4布局和indexer C4保持不变；这些能力分别在内网验证，不能把单请求诊断原型直接称为生产全特性完成。
