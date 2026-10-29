QLI V2 / QLI DSL：A5 单算子精度及性能测试

目标
同一份打包 MXFP4 Q/K、E8M0 scale、权重、页表及序列长度，分别调用：
  原生：cann_ops_transformer.ops.quant_lightning_indexer，quant_mode=5
  DSL ：cann_ops_transformer.ops.ds41.quant_lightning_indexer，quant_mode=1
或通过 --dsl-source samples 测试 cannbot-dsl 中的样例。
每个实现单独生成 metadata，不共享 metadata；关闭 candidate 分支。
原生 V2 当前源码 metadata 分配 2048 项，与部分文档中的 1024 不同；
脚本采用各自 metadata 生产接口返回值，并记录实际 shape，DSL 检查为 1024 项。
默认 Q=TND、K=PA_BBND、Nq=32、Nk=1、逻辑 D=128，物理 D=64 字节。
本脚本不比较旧版 INT8 QLI，也不测试 QSLI / candidate 分支。

环境
1. A5 / Ascend 950PR 或 950DT。
2. 按机器实际安装路径 source CANN 的 set_env.sh；使用支持 MXFP4 的配套版本。
   cannbot-dsl 样例当前说明要求 CANN 9.2.0、CANNBotDSL 0.7.0。
3. torch、torch_npu、numpy，以及已安装并可导入的 cann_ops_transformer。
   原生 MXFP4 要求 torch.float4_e2m1fn_x2 和 torch.float8_e8m0fnu。
4. ops 模式需要安装 experimental DSL wheel，以及两份实际 DSL 源码。
   --dsl-root 可指定已有 cannbot-source 仓库、samples 目录或 flat ops 目录。
   脚本仅在当前进程内将选定源码映射到扩展所需的两个 ops.* 模块，仍调用
   已安装的 cann_ops_transformer.ops.ds41，不复制/修改源码或 site-packages。
   保留源文件 __file__，兼容 metadata 同目录的 _aicpu 预编译文件查找。
   未指定路径时，优先已配置的 ops 源码，再找脚本/当前目录旁的 cannbot-source。
   多个源码位置或已加载的同名不同模块会明确报错，不静默覆盖。
   已安装 wheel 的加载顺序固定为：cann_ops_transformer.ops →
   cann_ops_transformer_dsl → cann_ops_transformer_dsl.ops.attention.quant_lightning_indexer_dsl →
   cann_ops_transformer.ops.ds41。即使只测 DSL 也先加载 Native ops。
   原因是 wheel 初始化会直接读取 sys.modules["cann_ops_transformer.ops"]；
   find_spec 能定位文件不代表已经完成模块导入。该顺序不替代源码映射或 NPU 执行检查。
5. 推荐空闲独占设备；保持电源、频率、温度与其他任务干扰条件一致。

执行（在脚本所在目录）
先检查不依赖 NPU 的 CPU 参考和校验器：
  python test_qli_a5.py --self-test

先跑小用例，确认安装、精度与计时链路：
  python test_qli_a5.py --suite smoke --output results_smoke

全套：含 B=12、Sq=6、Sk=64K/128K、TopK=512、causal、cmp_ratio=2：
  python test_qli_a5.py --suite full --output results_full

已安装导师命令构建的 wheel，源码位于已有 cannbot-source 时：
  python test_qli_a5.py --backend both --dsl-source ops --dsl-root /mnt/qli-a5-ready.I3M02Q/cannbot-source --suite full --fail-fast --profile
若 cannbot-dsl 只安装在此前创建的 venv，使用该 venv/bin/python；它通过
system-site-packages 访问原 conda 环境中已安装的 Torch 扩展，无须重装。
--fail-fast 遇到用例失败后不再计时或执行后续长序列用例。

run_qli_installed.py 是带日志的启动入口：下载固定 SHA256 的新版测试脚本，
先执行 DSL metadata 编译/运行检查，再进行上述比较；不安装包或修改编译器。
此前的 AICPU 'cstdint' file not found 是独立工具链问题；路径映射不能修复它。
若再次发生，启动脚本会保留完整错误并停止，不宣称 QLI 已运行成功。
启动时打印 runner_version；diagnostics.json 保存启动器和测试脚本版本/哈希，
避免 /mnt 下不同下载时间的同名脚本混用。

只测一个长序列并导出 profiler（分别导出两实现的 trace）：
  python test_qli_a5.py --suite full --case qli_64k --profile --output results_64k_profile

若要直接调用源码而不经过已安装的 Torch DSL dispatcher：
  python test_qli_a5.py --suite full --dsl-source samples --dsl-root /path/to/cannbot-dsl --output results_samples

可指定 --device npu:1、--backend native 或 dsl、--seed、--warmup 20、
--iterations 100、--repeats 5。同名结果目录若已有 results.json 将被拒绝覆盖。
同一 case 的随机种子不受 --case 筛选影响。完整 CPU golden 可能耗时较长，会校验所有 query。

精度规则
- 独立 NumPy CPU 参考解码低 nibble 在前的 FP4 E2M1，每 32 个逻辑 D 元素
  应用一个 E8M0 scale；按随机物理页表访问 K，包含负权重。
- 点积使用 FP64；QK 经 ReLU 后舍入 BF16，权重舍入 BF16，沿 head 逐次
  FMA 后进行 BF16 RNE 舍入，对应当前 MXFP4 golden 的精度模型。
- mask=3 使用压缩比例、每 batch 有效 Q/K 长度和 residual 计算可见范围。
- 检查输出形状、dtype、有效数量、唯一性、索引边界、padding -1/-inf、
  选中索引处的分数、TopK 边界以及是否遗漏明显更高分 token。
- 默认 rtol=1/128、atol=2.5e-5，可显式调整且会写入报告。这是测试判据，
  不是对所有模型输入精度的保证。当前随机输入使用有限、适中范围的分组 scale。
- 同分候选与 TopK 排序不稳定，因此不要求索引逐项一致；canonical set overlap
  和 native/DSL overlap 仅用于诊断，不单独作为失败判据。
- return_value=True 检查索引和分数；False 路径也检查索引正确性，再做性能测量。
- 任一实现 FAIL/ERROR，最终退出码为 1；精度失败的实现不进行性能测试。
  精度失败保存输入、参考分数和输出 NPZ，可按 seed 重现。

性能口径
- 初次编译/初始化、CPU 参考、H2D/D2H 拷贝与 metadata 生成均不计入测量。
- 测 return_value=False、candidate 关闭的调用；先预热，再多轮连续调用。
- event_median_us：每轮 NPU event 总耗时 / iterations，取轮间中位数。
- wall_median_us：同一批次从提交到 synchronize 完成的墙钟时间 / iterations。
- 两者都是 eager 调用口径。NPU event 时间可能包含主机下发不够快造成的
  空隙，不能直接称为纯 kernel 时间，也不能直接与 README 的 msprof 数字比较。
- --profile 在计时完成后额外采集 5 次调用。查看 profiler 下 kernel 的
  Task Duration / op statistics；若一个调用启动多个 kernel，应统计整个调用的
  kernel 集合，并单独观察归并 kernel。不要拿单个子 kernel 代表整个算子。
- 当前不启用 graph replay；若短 kernel 被下发开销主导，优先用 profiler 数据判断。
- 加速比 = native event median / DSL event median，>1 表示该 eager 口径下 DSL 更快。

输出
results.json：环境、接口路径及 SHA256、参数、种子、逐实现精度、耗时样本和错误堆栈。
summary.csv：每个 case/实现的状态及 event/wall 中位耗时。
npu-smi.txt：设备状态快照（命令可用时）。
profiler/：启用 --profile 后导出的 trace 与分析结果。
把整个结果目录带回来即可进一步判断精度差异与性能瓶颈。

接口核对来源（编写时版本）
ops-transformer commit a18ec2631ccf4b8ed08027d7d38bc3c1816f929a
https://gitcode.com/cann/ops-transformer/blob/master/attention/quant_lightning_indexer_v2/docs/torchapi_quant_lightning_indexer.md
https://gitcode.com/cann/ops-transformer/blob/master/experimental/attention/quant_lightning_indexer_dsl/README.md
https://gitcode.com/cann/ops-transformer/blob/master/attention/quant_lightning_indexer_v2/tests/pytest/quant_lightning_indexer_v2_golden.py
cannbot-dsl commit 606a737579356baa31da5674452f60673681cff0
https://gitcode.com/cann/cannbot-dsl/blob/master/samples/quant_lightning_indexer_dsl/README.md

本地验证仅覆盖 CPU 参考、校验逻辑和语法；真实算子可加载性、NPU 精度及性能
需在 A5 上运行后确认。源码版本与机器安装包不一致时，以报告中的实际模块为准。
