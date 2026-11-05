# Kimi K3 A5 C8：当前源码与性能分析入口

本目录提供可直接浏览的源码快照、相对 VA 基线的统一补丁，以及实际使用的 DSL Prolog 源码。
用于代码评审和性能分析；这里没有修改算法，也没有更新部署安装脚本。

## 版本与范围

- VA 基线：`c09d3e20b9c45c80b34ae3781440d90904a44023`（0.30）。
- 对应已发布安装补丁：[`00dcf114/va030_qfm/install-va030-qfm-pd.py`](../../va030_qfm/install-va030-qfm-pd.py)。
- `source/` 中的 23 个文件逐个 SHA256 校验，与该安装脚本的 AFTER 状态完全一致。
- QFM / QFM metadata：CANN ops-transformer `8b6924e68ed27181141cde84fa5a942c0c060aac`。
- DSL Prolog：cannbot-dsl `9e01355ab8f809050579271c0eadcc5858ed20bc`，源码 SHA256 见 manifest。
- `source/` 仅列出补丁涉及的文件，不是完整、可独立安装的 vllm-ascend 仓库。
- `va030-qfm-c8.patch` 为普通文本统一补丁，可对基线阅读 diff。
- 保留 Apache-2.0 与 native/ 下独立的 CANN Open Software License；native 源码版权声明未修改。

## 建议阅读顺序

| 入口 | 作用 |
|---|---|
| [MLA forward](source/vllm_ascend/attention/mla_v1.py#L3504) | 主模型进入 QFM 路径；connector hooks |
| [forward_qfm](source/vllm_ascend/attention/kimi_k3_qfm.py#L443) | 前处理、写 Cache、DCP、Attention、输出投影 |
| [QFMProlog.__call__](source/vllm_ascend/attention/kimi_k3_qfm_prolog.py#L66) | 外层 BF16 hidden 动态量化、mode2、直接写或临时 Cache 后 Scatter |
| [QFMProlog.floating](source/vllm_ascend/attention/kimi_k3_qfm_prolog.py#L111) | FIA Prefill 的 mode0 路径 |
| [Prolog 权重适配](source/vllm_ascend/attention/kimi_k3_qfm_prolog_weights.py#L25) | 三个 W8A8 MXFP8 Group32 投影；私有 ND 权重；kv_b 派生矩阵 BF16 |
| [QFMMetadataBuilder](source/vllm_ascend/attention/kimi_k3_qfm.py#L200) | 虚拟序列、页表、长度和 scheduling metadata |
| [run_qfm_reader](source/vllm_ascend/attention/kimi_k3_qfm.py#L484) | QFM 的实际输入和参数 |
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

- 198 项 CPU 回归、2 项安装/升级检查通过（发布 PD 候选时）。
- 先前单卡合成 Prolog/QFM/FIA 正确性测试已通过；不代表所有整模组合验收通过。
- PD 网络传输、真实 HCCL、DSpark 接受率及各图模式组合仍需设备端验收。
- 本次发布仅把既有代码展开，未运行新的 NPU 性能测试，也未宣称 TTFT 回退已修复。
