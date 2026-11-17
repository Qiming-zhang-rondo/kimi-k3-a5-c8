BF16 Lightning Indexer 与 MXFP4 QLI：A5 单算子精度对照
==================================================

这份脚本不加载模型，不修改主 attention KV，不使用 DSL，不安装或升级任何包。
在运行模型的同一个 Python 环境中执行。需要 torch、torch_npu、numpy，
以及已经可用的 cann_ops_transformer.ops 原生 QLI V2 和 metadata 接口。
已安装的扩展首次导入时可能初始化/构建自己的绑定；脚本不运行算子源码 build.sh。

本地只做 CPU 单测、接口参数检查和模拟编排；尚未在 A5 上执行真实算子。

1. 运行
-------
下载启动器 run_qli_bf16.py 后：
  python run_qli_bf16.py --keys 8192,32768 --queries 1,4 --topk 2048

启动器从固定 commit 下载三份 SHA256 校验的 Python 文件，使用当前解释器，
将完整日志保存到 /mnt/qli-bf16-随机目录/run.log；报告在其中的 results/。
可以用 --work-dir 更改 /mnt，用 --insecure 适配代理证书（仍校验固定 SHA256）。
直接运行已下载的脚本，无须再次联网：
  python /mnt/qli-bf16-实际目录/test_qli_bf16_accuracy.py --keys 8192 --queries 4 --output /mnt/qli-bf16-new-results

默认：B=2，Q=4，最大 K=8192，Nq=32，Nk=1，D=128，TopK=2048，page=128，
右下对齐 causal，cmp_ratio=1。第二个 batch 的 key 长度少17，包含页尾情况。
--keys 和 --queries 可为逗号分隔列表；这些只是随机单算子用例，不代表真实模型分布。
--distribution outlier 增加离群值压力用例；--seed 指定随机种子。
--repeats 3 表示每条路径共调用3次，后两次与第一次比较集合稳定性；不是性能计时。
--save-inputs 额外保存原始/旋转/反量化/打包输入及可回放的 input_bf16.npz。
默认只保存完整索引与诊断分数，避免大输入重复占空间。

2. 实际执行的四条算子路径
-----------------------
bf16_original：原始 BF16 Q/K -> torch_npu.npu_lightning_indexer。
bf16_rotated：同一 Q/K 做 BF16 normalized Hadamard -> BF16 Lightning Indexer。
bf16_dequantized：Hadamard 后，使用 NPU npu_dynamic_mx_quant 量化至 MXFP4，
                 将实际量化输出解码回 BF16 -> BF16 Lightning Indexer。
mxfp4_qli：同一份 NPU 量化的 Q/K 字节与 E8M0 scale -> 原生 QLI V2，quant_mode=5。

四条路径使用相同 BF16 head weights；QLI 仅将这些相同数值扩大为 FP32。
BF16 路径用 TND/PA_BSND；QLI 用 TND/PA_BBND，独立生成自己的 metadata。
共享同一张随机物理页表和相同逻辑候选域、mask、TopK；不涉及主 KV 量化。

模型调用方式核对自固定的公开 vLLM Ascend revision：
https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/attention/indexer.py
https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/device/device_op.py

3. 如何读报告
-------------
total_mxfp4_vs_bf16：最终关注的 QLI 相对原始 BF16 的集合差异。
hadamard_rounding：变换本身及 BF16 舍入造成的变化。
input_quantization：固定旋转后输入，量化/反量化造成的变化。
qli_vs_dequant_bf16：相同量化数值下，两种算子计算路径的额外差异。
这些差异不是可相加的百分比；TopK 是非线性的，不能相加得到总误差。

先检查 index_validation：每行有效数量、padding、重复及越界。
结构错误不会被删除或过滤后伪装成高重合率。
随后比较逻辑索引集合，而非输出位置：
  recall = 交集数量 / min(TopK, 可见key数)
  replaced_count = BF16选中但另一条路径未选中的数量
每个 query 保存完整 dropped/added 列表及分数最高的示例。
summary.recall 包含 mean/min/p05/p50/p95；终端显示最差行。
row_map 将全局行号映射到 batch/query，均从0开始。

重要：paged BF16 接口不支持 return_value=True。
因此四条路径都返回实际 indices，不索取、不伪造 kernel 分数。
报告中的所有 score/threshold/gap/regret 都是完整候选域上的独立 Float64 诊断计算：
  sum_h weights[h] * ReLU(dot(Q[h], K[token]))
它不模拟 BF16 算子的每一步累加舍入，不能当成位精确 golden。
漏选/新增示例同时给出参考域分数与另一输入域的重算分数。
同一量化值上的 QLI 与 BF16 索引不一致，不自动代表 QLI 有 bug。

boundary_gap 是诊断分数第K名与第K+1名的差；候选不足时为 null。
--tie-atol/--tie-rtol 控制“门槛附近”的描述范围，默认0，仅描述精确同分。
例如 --tie-rtol 0.0078125 可以额外观察相对门槛较近的替换，
不会修改原始重合率，也不是算子正确性的宽松放行阈值。

结果状态：
  MEASURED_IDENTICAL_TOPK：两条主要路径的初次选择相同，且未测到重复调用集合变化；不是全模型精度证明。
  MEASURED_WITH_TOPK_DIFFERENCES：主要路径初次选择或重复调用有集合差异；不是编译/运行失败。
  repeat_set_instability：每条路径后续调用相对首次出现集合变化的次数。
  ERROR_OR_INVALID：接口/运行异常、索引结构错误，或显式设置的 recall gate 未通过。
退出码0表示测量完成，不表示“MXFP4与BF16精度相等”。
可用 --min-recall 0.99 自行设置验收门槛；脚本不擅自规定通用合格重合率。
结构/运行异常或 gate 失败时退出码1，完整 traceback/参数保留在报告。

结果文件：
  results.json：全局版本、参数、输入来源、各用例状态与精度测量。
  case_*/details.json：该用例全部逐行诊断、漏选/新增集合、重复调用对照。
  case_*/arrays.npz：实际索引、完整候选诊断分数、row_map、页表。
  case_*/input_bf16.npz：仅在 --save-inputs 时保存的可重放原始输入。
已存在的非空结果目录不会覆盖。文件只保存在机器本地，不自动上传任何模型张量。

4. 回放真实模型输入
-----------------
要判断 GLM-5.3 退化是否与选择有关，必须补充同一次前向的真实 indexer 输入。
不能拿两次不同模型运行中已经分叉的 hidden states 直接对比，也不能把 MXFP4
反量化数据冒充“原始 BF16”。单算子结果仍不能直接证明端到端模型掉分的因果关系。

脚本提供 save_paged_case 辅助函数，供在 BF16 indexer 的 eager 调用前显式调用。
不建议在 ACLGraph capture 内做 CPU 拷贝/导出。示意（变量对应实际 BF16 调用）：

  import sys
  sys.path.insert(0, "/mnt/qli-bf16-实际下载目录")
  from test_qli_bf16_accuracy import save_paged_case
  save_paged_case(
      "/mnt/layer_XX_step_YY.npz",
      q=q_before_hadamard,              # BF16 [T,32,128]，已完成模型投影和RoPE
      key_cache=original_bf16_key_cache,# BF16 [pages,page,1,128]，完整历史Indexer K
      weights=bf16_head_weights,       # BF16 [T,32]，已包含模型需要的缩放
      block_table=block_table,
      cu_seqlens_q=cu_with_leading_zero,# INT32 [B+1]，必须包含开头0
      k_lengths=sequence_key_lengths,  # INT32 [B]
      topk=2048, mask=3, label="layer_XX_step_YY",
  )

若当前运行只有量化后的 Indexer K Cache，需要在量化前另保留 BF16 历史 K，
或从 BF16 baseline 的同一次前向导出。仅当前 token 的 K 不够；不能传主 attention KV。
辅助函数会解分页为逻辑序列，保存原始 BF16 值（以 float32 无损存储）、mask/TopK
等参数。目标文件已存在则拒绝覆盖。它不会自动注册 hook 或修改模型。

  python run_qli_bf16.py --input /mnt/layer_XX_step_YY.npz
多个样本重复传 --input。回放使用文件里的 TopK、mask、page 和长度，不由随机用例参数覆盖。
输入来源会保留：脚本生成的随机样本回放仍标记 synthetic_replay，不能当作模型证据。

5. CPU 自检
-----------
  python test_qli_bf16_accuracy.py --self-test
只依赖 NumPy，不导入 Torch/NPU，不能替代 A5 上真实算子验证。
