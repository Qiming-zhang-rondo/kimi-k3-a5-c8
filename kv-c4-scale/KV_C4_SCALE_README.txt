Pure main-KV C4 scale experiment for Ascend A5
============================================

目标
  只调整主 MLA KV 的 C4 量化尺度，寻找能降低 attention 输出误差的配置。
  同一份原始 BF16 输入、同一份固定 TopK、window=0；所有 C4 配置继续使用
  同一个量化 attention reader，保持 MXFP4 E2M1 + E8M0/block32 缓存格式。
  这不是 QLI TopK 实验，也不修改模型或证明 GLM 端到端分数已经提升。

四个 C4 对照
  baseline       完全复用此前 C4 路径及其默认量化参数。
  alg2_max6      scale_alg=2, dst_type_max=6.0
  alg2_max7      scale_alg=2, dst_type_max=7.0
  alg2_max7_25   scale_alg=2, dst_type_max=7.25
  候选显式设置 axis=-1、block_size=32、round_mode="rint"。
  dst_type_max 改变尺度选择阈值；FP4 存储格式本身没有变。
  同时运行完整 BF16、C8 对照。BF16 复用已验证的 compact 索引调用方式。

运行
  使用已经验证可运行 Cloud 算子的 Python/CANN 环境，先 source 供应商环境。
  程序不安装 wheel、不升级 CANN、不重编 Cloud 算子。
  launcher 自动补充 /tmp/cloudops/bin/ninja（仅在现有 PATH 无 ninja 时），
  默认 JIT 缓存 /tmp/cloudops/torch_ext，保留现有代理和显式缓存设置。

  python run_kv_c4_scale.py --insecure --work-dir /tmp/kv-c4-scale \
    --keys 32768 --queries 4 --timing eager,graph

  默认 B=1,H=64,D=512,RoPE=64,page=128,TopK=2048，K=32768,Q=4。
  合成输入仅用于确认方向与运行能力；它不能解释模型的 90.4/91.4 分数。
  真实输入：--input /path/to/original_bf16_case.npz，可重复指定多个文件。
  kv_window_capture.py 的 save_paged_mla_case 可在 BF16 模型路径导出原始
  Q、latent KV、RoPE、分页表和固定 sparse_indices。必须是真实量化前 BF16，
  不能用 C4 反量化后转 BF16 来代替；捕获过程不计入耗时。

结果怎么读
  summary.csv：先看每条记录的正确性和计时状态，再比较相对 BF16 输出的
  relative L2，以及相对原 C4 的误差改善比例。同一模式下比较 read 与
  write+read 延迟。改善为正表示本次输入上更准，为负表示变差。
  results.json：完整参数、环境、控制检查、错误 traceback 和性能记录。
  outputs.npz：各路径原始输出和数学参考，按用例保存。
  best_on_these_inputs 仅表示这批输入的最佳值；验证集应与选参样本分开。

正确性与性能边界
  原始 BF16 数学参考与每条路径实际量化字节反量化后的数学参考均使用 FP64。
  kernel_vs_matching_reference 用来检查算子计算是否与其实际缓存一致；
  vs_bf16_operator 用来判断量化质量。量化误差较大不等于算子调用错误。
  同时检查整体与逐行误差；write+read 必须与预先准备的缓存输出一致。
  正确性检查失败的路径不计时；不支持的参数单独报 ERROR，不静默退回默认值。
  graph 失败会保留错误，不替换成 eager 数据。每条 C4 路径每次读只调用一次
  attention，不拆分冷热段、不做输出合并。

  read：预先准备好的完整 C4 缓存上的单 attention 调用。
  write+read：当前 query 对应 token 的量化、写缓存和 attention 调用。
  全历史缓存的初始量化在计时外；不是完整 prefill 性能测量。
  所有重复写使用相同 token/位置，graph 使用固定 shape/TopK；这不是服务吞吐。
  保留实际尺度/字节变化证据，避免把没有改变数据的候选当作有效优化。

参数依据（文档不等于当前安装版本已支持，必须实际调用验证）
  https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_mx_quant.md
  https://docs.vllm.ai/projects/vllm-omni/en/v0.32.0/user_guide/quantization/mxfp4/
  alg2 + 7.25 是待验证候选，不保证对 GLM KV 最优。

CPU 检查
  python test_kv_c4_scale_a5.py --self-test
  本地 CPU/mocks 检查只覆盖测试框架与调用构造；A5 数值和性能仍需实测。
