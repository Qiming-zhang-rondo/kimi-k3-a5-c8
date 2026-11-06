# Kimi K3 A5 C8：当前源码与性能分析入口

本目录提供可直接浏览的源码快照、相对 VA 基线的统一补丁，以及实际使用的 DSL Prolog 源码。
用于代码评审和性能分析。本次增加按阶段控制 Prolog 的开关；默认行为保持不变。

## 版本与范围

- VA 基线：`c09d3e20b9c45c80b34ae3781440d90904a44023`（0.30）。
- 对应已发布安装补丁：[`install-va030-qfm-scatter-fix.py`](../../va030_qfm/install-va030-qfm-scatter-fix.py)。
- `source/` 中的 23 个文件逐个 SHA256 校验，与该安装脚本的 AFTER 状态完全一致。
- QFM / QFM metadata：CANN ops-transformer `8b6924e68ed27181141cde84fa5a942c0c060aac`。
- DSL Prolog：cannbot-dsl `9e01355ab8f809050579271c0eadcc5858ed20bc`，源码 SHA256 见 manifest。
- `source/` 仅列出补丁涉及的文件，不是完整、可独立安装的 vllm-ascend 仓库。
- `va030-qfm-c8.patch` 为普通文本统一补丁，可对基线阅读 diff。
- 保留 Apache-2.0 与 native/ 下独立的 CANN Open Software License；native 源码版权声明未修改。

## 建议阅读顺序

| 入口 | 作用 |
|---|---|
| [MLA forward](source/vllm_ascend/attention/mla_v1.py#L3505) | 主模型进入 QFM 路径；connector hooks |
| [forward_qfm](source/vllm_ascend/attention/kimi_k3_qfm.py#L543) | 前处理、写 Cache、DCP、Attention、输出投影 |
| [QFMProlog.__call__](source/vllm_ascend/attention/kimi_k3_qfm_prolog.py#L66) | 外层 BF16 hidden 动态量化、mode2、直接写或临时 Cache 后 Scatter |
| [QFMProlog.floating](source/vllm_ascend/attention/kimi_k3_qfm_prolog.py#L111) | FIA Prefill 的 mode0 路径 |
| [Prolog 权重适配](source/vllm_ascend/attention/kimi_k3_qfm_prolog_weights.py#L25) | 三个 W8A8 MXFP8 Group32 投影；私有 ND 权重；kv_b 派生矩阵 BF16 |
| [QFMMetadataBuilder](source/vllm_ascend/attention/kimi_k3_qfm.py#L237) | 虚拟序列、页表、长度和 scheduling metadata |
| [run_qfm_reader](source/vllm_ascend/attention/kimi_k3_qfm.py#L644) | QFM 的实际输入和参数 |
| [DSL _execute_async](native/quant_mla_prolog.py#L8499) | 参数检查、可用核数查询、中间张量申请、程序查找及下发 |
| [DSL PyTorch 注册](native/quant_mla_prolog.py#L8810) | profiling 名称 cannbotdsl_quant_mla_prolog::quant_mla_prolog |
| [现有设计说明](source/docs/source/developer_guide/kimi_k3_qfm_c8.md) | Cache/TP/DSpark/DCP/PD 范围和限制 |

## 当前计算与存储契约

```text
BF16 hidden
  -> 外层动态 MXFP8 Group32 量化
  -> DSL Prolog：q_a / q_b / kv_a 使用 MXFP8 权重和 E8M0 scale
  -> mode2：FP8 Q + 动态 Q scale；全576 FP8 KV Cache
  -> QFM Attention
```

- KV Cache 为 512 latent + 64 raw，全部 E4M3；K3 此路径不执行 RoPE。
- KV descale 固定 1.0，每层共享；不是旧 QSFA packed656 / Group128，也不是 MXFP8 Group32 KV。
- PA_NZ 内核页大小 128；manager block 为其正整数倍。
- 页间 padding 为零时 Prolog 直接写最终 Cache；否则写临时 FP8 后 Scatter，避免丢失 stride/offset。
- `enable_mlapo=false` 使用原有量化 Linear 分步执行，QFM 与 FP8 Cache 格式保持不变。
- FIA Prefill 读取历史 FP8 并临时转为 BF16，Decode 保持 QFM；不新增永久 BF16 Cache。

## 当前性能问题与待核查项

用户报告：同一模型的 C8 QFM 相对 KV 不量化路径，ITL 基本持平，TTFT 约 12s -> 16s。
CPU MLA forward 区间约 3.4ms vs 1.5ms；这些不是 NPU 内核执行耗时。
NPU 时间线上观察到 Event Wait，尚未配对 Record，不能据此认定等待来源或归因 CPU。

以下是分析方向，不是已确定的根因：

1. 8192 Q 在非 DCP eager 下组成 512 条 <=16Q 虚拟序列，一次 native QFM 调用。
   metadata 在 attention group 构建复用；页表复制不是每个 MLA 层都重新做。
2. DCP / graph 路径按每个 Q 一条虚拟序列；实际虚拟行数应按部署配置确认。
3. metadata 与 Attention 的 max_seqlen_kv 当前使用 max_model_len，未按当前 batch 收紧。
4. 页间 padding 时的临时 FP8 写入和额外 Scatter。
5. DSL 包装区间内的设备属性/stream quota 查询、多个中间缓冲区和程序查找；新形状可能触发 JIT。
6. 历史单算子 FIA baseline 用旧接口、对 heads 补成 2 的幂，不能直接等同生产 BF16 FIA 路径。
7. 比较内核时长、CPU 下发间隙与跨 stream 等待的关键路径；不要把重叠 Wait 简单求和。

## 验证记录

- 286 项 CPU 回归、5 项安装/升级检查通过（共享批次准备优化）。
- 先前单卡合成 Prolog/QFM/FIA 正确性测试已通过；不代表所有整模组合验收通过。
- PD 网络传输、真实 HCCL、DSpark 接受率及各图模式组合仍需设备端验收。
- 分阶段 Prolog 开关尚未运行新的 NPU 精度/性能测试，也未宣称 TTFT 回退已修复。


## Prefill 关闭 Prolog，Decode 保留

```json
{
  "enable_kimi_k3_qfm_c8": true,
  "enable_mlapo": true,
  "enable_kimi_k3_mlapo_prefill": false,
  "kimi_k3_c8_prefill_backend": "qfm",
  "enable_kv_nz": false
}
```

`enable_kimi_k3_mlapo_prefill` 默认 true。它只控制前处理，和 FIA/QFM
Attention backend 的选择独立。Prefill 用普通 W8A8 MXFP8 投影和显式 FP8
写入；纯 Decode/DSpark 验证保留 FP8-output mode2。混合批次按 scheduler 阶段标记分段，Decode 段保持 DSL mode2，Q 的 FP8
字节和 scale 原样合并；仅 Prefill Q 在外部量化。混合 QFM 不分配完整 BF16
Q/KV。混合 FIA 另外保留 Prefill 的浮点 Q/KV，DCP 时还有一次 BF16 Q gather，
该组合需要独立测性能。主模型/草稿和 Cache 格式不变。
启动日志 `prefill_mlapo=False` 表示该策略已配置。


## 本轮代码审计与修正

- 两条 writer 的 Cache 均为全576 E4M3、PA_NZ、KV descale=1。未发现阶段开关改变
  dtype、offset/stride 或把草稿 Cache 转为 QFM 的证据。
- 纯 Decode 原有 mode2 分支未改。本轮去掉 stage flag 关闭时每步不必要的 CPU
  clamp/tolist/分段循环；Capture→Prefill→Decode 回归检查 metadata 刷新和 mode2。
- 上版混合批次把 Decode 降为 mode0，再重新量化。现在保持 mode2、原样保留
  Decode FP8 Q 和 scale；只量化普通 Prefill Q。NPU ITL 是否恢复仍需设备验证。
- 普通 latent 投影/RMSNorm 与 DSL 有不同舍入点，同为 FP8 不保证值逐位相同；
  这不构成 ITL 翻倍的根因证明。应结合真实 DSpark 平均接受长度及验证轮耗时判断。
- 找到一个条件兼容缺口：普通 RMSNorm 会加已加载的 ModelSlim norm bias，
  pinned DSL 没有这个参数。加载期新增检查，非零/NaN bias 明确拒绝，避免静默
  漏用；全零或未加载 bias 接受。触发时需普通前处理或 bias-aware Prolog。
- 本轮无新增 native 包和算子编译。未进行真实 NPU graph/HCCL/model acceptance。


## PyTorch 批次准备优化

- 普通 writer 的 NZ Scatter 地址按真实 page stride 每批计算一次，各 MLA 层共享；
  保留负 slots、页间 padding 与 storage offset。混合段仅切片对应地址。
- 共享 invalid/empty-row mask 和临时 Cache 的从零 staging indices，避免逐层重复
  构造；FIA 固定布尔 mask 复用。连续 Cache 的 mode2 Decode 不计算无用 Scatter 地址。
- 单 Q 虚拟序列复用 cumulative lengths，直接复制有效长度；图 metadata 在固定
  地址原地刷新。主模型/草稿 Cache、FP8 格式和 native 量化计算均不变。
- `enable_mlapo=true`、`enable_kimi_k3_mlapo_prefill=false` 时，所有标记为 Prefill
  的 chunk（包括短尾块）都走普通前处理；Decode/DSpark verification 保留 mode2。
- 这些改动去除已确认的重复准备，尚无 NPU TPOT 恢复证据，不能认定已定位两倍
  ITL 退化或 Event Wait 的全部原因。本轮不修改 native 实现、不需要重新编译算子。


## Shared Scatter plan 启动回归修复

图预热在框架的共享 Scatter plan 校验处报错，尚未进入 QFM Attention。
旧报错没有区分 stride、slots 引用、offsets 属性，不能据此确定具体失配项。
本版匹配时仍共享预计算；失配时打印 `QFM Scatter plan fallback` 及具体字段，
使用当前 slots 和实际 Cache stride 重建地址。重建操作保留在捕获图内，重放时随
slots 更新；不复用第一次预热的临时地址，不改变量化、Cache 布局或 native 算子。
CPU 289 项回归覆盖失配后重建、slots 原地刷新、负 slots、页间隙、storage offset；
这些检查不能替代 NPU 图捕获/重放和全模型验收。首次启动请保留 fallback 警告，
其内容用于确认具体失配来源。该修复不证明此前的 ITL 退化已经解决。
