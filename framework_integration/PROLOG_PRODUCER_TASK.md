# Prolog热环：仅在内网实施，不再上传内部源码

用户已明确：45个文件的批量外传不符合部门要求。之前的源码采集/上传要求撤销。
不要继续上传剩余文件、内部头文件、实现diff、schema或二进制；不以任何外传材料作为继续实现的前提。
原始代码、构建与测试证据均留内网。对外只回传部门允许披露的能力结论和脱敏测试结果；不允许外发的内容不回传。

## 本轮已提供的关键材料结论

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
