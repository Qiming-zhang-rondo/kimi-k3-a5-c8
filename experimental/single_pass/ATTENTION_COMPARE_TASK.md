# 同轮比较纯C4、纯BF16与最新MXFP4+BF16融合算子

用户本轮要求补齐三臂精度、误差恢复比例及公平性能。本任务不改内核，不重新扫描tile，不安装/升级CANN，
复用当地上轮成功的split4库与已校验payload。框架接入任务保留；先完成这组独立对照。

## 内网agent直接执行

自主定位上轮成功的`libqsfa_recent128_direct.so`与`hotkv` payload，记录真实库SHA及构建身份。
只在已确认空闲的A5卡上测试，不停业务、不复位卡、不加载另一版本vendor。
沿用匹配环境；如需ninja仅将其实际目录补入PATH，使用隔离Torch extension缓存。

将变量填为现场真实路径，然后从拉取后的本仓根目录运行：

```bash
COMPARE_LIBRARY=/actual/verified/libqsfa_recent128_direct.so
COMPARE_PAYLOAD=/actual/verified/hotkv
COMPARE_DEVICE=npu:0
COMPARE_RUN=$(mktemp -d /tmp/qsfa-threeway.XXXXXX)

python experimental/single_pass/run_attention_compare.py \
  --payload-dir "$COMPARE_PAYLOAD" --library "$COMPARE_LIBRARY" \
  --device "$COMPARE_DEVICE" --queries 4 --keys 8192 \
  --timing graph --warmup 20 --iterations 100 --repeats 20 \
  --output "$COMPARE_RUN/results"
```

默认已设置W0/W128、split4及20轮graph，无需另传verify或矩阵开关。先只跑这一组；
如果只要精度可显式`--timing none`，该结果只能是VALIDATED_ONLY。
缺失正式入口/ABI、库不兼容、数学/图/partial门禁失败即停止并提交首错，不以替代实现继续计时。

## 三臂合同

| 主臂 | 实际读取 | 精度格式 |
| --- | --- | --- |
| installed_c4 | 当前已安装cloud_kv_quant_sparse_flash_attention | 同一原始BF16 KV量化为MXFP4；原默认量化参数 |
| installed_bf16 | 当前已安装cloud_sparse_flash_attention | 原始完整BF16 latent与BF16 RoPE |
| w128_split4 | 最新库真实KV-split4 attention与设备归约 | 冷MXFP4＋最近128原始BF16 |

同一Q、TopK、物理页表、长度和scale，TopK不排序、不增删、不重新抽样；纯BF16是全有效容量2048的一次原生调用，
不使用历史有问题的`-1`填充BF16子集。本次固定B1/Q4/K8192/H64/latent512/RoPE64/TopK2048，
每query保证128个近期token命中；这是受控合成输入，不代表真实模型的热窗命中分布。原W0/V4额外输出仅供诊断。

性能为prepared read：三臂全部排除量化、KV写入、写环、QLI/TopK、分配准备和CPU参考；
split4必须包含attention与最终设备归约。三组直接配对为C4↔BF16、C4↔split4、BF16↔split4，
每组20轮AB/BA交替，保存每轮event/wall原值及ratio；不以独立中位数的商冒充配对比值。
先过各自完整数学参考、partial与graph投毒回放门禁；计时后直接读取最后输出，禁止再调用算子刷新。

## 精度指标含义

同时保存三个独立参考，不能混作一个“精度”：

1. `matching_reference`：该臂实际cache值的独立FP64数学结果，检查实现计算是否正确。
2. `vs_original_bf16_math`：原始BF16 Q/K/V/RoPE的独立FP64数学结果，比较量化与最终计算的总误差。
3. `vs_native_bf16_output`：同轮原生BF16实际输出；BF16自比为0，不等于其数学计算误差为0。

每组都报告relative L2、最差query-head L2、cosine均值/最小值/p05、1-cosine、MAE/max_abs及非有限数。
L2越低越好，cosine越接近1越好；cosine按每个query/head的D512向量计算，不将所有head混为一行。
例如输出被整体放大两倍，cosine仍可为1，所以两项都要看。两向量皆零约定cosine=1，单边零=0，
任何NaN/Inf使该比较无效，不剔除坏行来计算好看的均值。

自动给出两种不裁剪的恢复比例：

- 相对C4误差降低：`(C4_L2 - mixed_L2) / C4_L2`。
- 向BF16计算误差底恢复的比例：`(C4_L2 - mixed_L2) / (C4_L2 - BF16_L2)`。

分母为0返回null；劣化时保留负值，不强称正收益。上述量化质量只作报告，不能以C4本来约0.1的误差触发实现错误门禁。

## 一次交付

脚本自动生成`summary.txt/csv/json`，其中优先展示计时后真实输出的精度；`results.json`保留全部门禁、
输入指纹、环境/库身份和原始配对轮次；`outputs.npz`及`post_timing_outputs.npz`留内网并记录SHA。
自动`verification_evidence/`以小文本分片无损保存完整results，不再因JSON过大只上传手写摘要。

将三份summary、自动生成的整个verification_evidence目录、实际命令/日志及卡空闲证据推送到
`experimental/single_pass/evidence/attention_threeway/`。大NPZ无需上传，不包含权重或真实prompt。
只要失败就保留原状态/traceback，不能过滤失败臂或用历史数据填表。

本机宿主测试只验证脚本与指标；本次设备精度、性能及任何框架整体分数均等待真实执行。
