# K3 C8：CANN 风格的 BF16 分页 FIA Prefill 对照

固定代码版本：`f07e5aec674d6f8f1ce9b8b3f7ce49159213d30d`。
框架基线：VA 0.30，`c09d3e20b9c45c80b34ae3781440d90904a44023`。
适用于原版 K3 MLA；K3.1 DSA 使用另一条模型和算子路径。

## 统一框架补丁

包含当前发布的 QFM、latent/expanded FIA、QFA 和新增的 `fia_paged` 代码。
支持干净基线及已识别的旧完整补丁，不需要依次安装历史框架补丁。
脚本自动备份，不安装算子、不编译算子、不自动跑测试。

```bash
curl -kfsSL --proxy http://10.155.96.5:8081 --connect-timeout 15 --max-time 120 https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/f07e5aec674d6f8f1ce9b8b3f7ce49159213d30d/va030_qfm/install-va030-qfm-fia-paged.py -o /mnt/fia-paged-patch.py && python /mnt/fia-paged-patch.py
```

成功标记：`QFM_VA030_FIA_PAGED_APPLIED`；已安装打印 `ALREADY_APPLIED`。
遇到未知或部分混合改动时脚本拒绝，不要强制 reset。
P/D 都更新到该统一版本并重启所有 Worker，旧的内存 KV/APC 页面不能继续使用。

## P 节点配置

把以下字段合并到现有 `additional-config`，其他启动参数保持对照一致：

```json
{
  "enable_kimi_k3_qfm_c8": true,
  "enable_kv_nz": false,
  "enable_mlapo": false,
  "kimi_k3_c8_prefill_backend": "fia_paged",
  "kimi_k3_c8_fia_form": "latent"
}
```

这组配置使用普通投影/Norm，保留 BF16 latent512/raw64 历史，FIA 直接分页读取。
在 DCP=1 时，历史和当前 chunk 一次 causal FIA 计算，再做 V 上投影。
不再生成 O(history) 的框架 Gather 地址、逐 chunk 读取/转换 FP8 历史或合并历史/当前输出。
当前 KV 仍会写入 FP8 主 Cache，供 D 节点使用。

保留 `--kv-cache-dtype auto`、`--block-size 128` 或原来已验证的正整数倍。
Prefill 继续 Eager，现有 `FULL_DECODE_ONLY` 配置可以保留。
如要测试融合 P 前处理，可改 `enable_mlapo=true` 并设置
`enable_kimi_k3_mlapo_prefill=true`，使用已有 DSL 的 floating 前处理。
首次性能对照建议使用上面的普通前处理配置。

## D 节点配置

保持 `kimi_k3_c8_prefill_backend="qfm"`、`enable_mlapo=true`。
D 不分配新增 BF16 历史，继续 DSL Prolog + QFM，PD 只传原 full576 FP8。
这条实验不依赖 QFA，不需要 QFA920 wrapper，也没有新增算子编译步骤。
原来安装的 QFM/DSL 运行环境仍按原方式加载。

## 显存、APC 与兼容边界

- 只对显式启用 `fia_paged` 的主模型 MLA 分配 BF16 历史，额外有效载荷为
  1152 B/token/MLA layer。FP8/KDA 共享池维持原布局，BF16 两池独立连续分配。
- Worker 在自动内存测量和手动 KV-memory 两条路径中预留 BF16 开销，再计算
  主 Cache 的 block 数；可用 block 数和并发容量可能下降。
- 不能同时使用 `num_gpu_blocks_override`，它会绕过新增 BF16 预算。
- APC 保留物理页及 BF16 镜像；部分命中的 COW 和回收清零也同步处理 BF16 池。
  主模型本地 Decode/Verify 写入只镜像本次新增槽位，草稿格式不变。
- 此 backend 支持无外部 KV transfer 的进程或 PD `kv_producer`。
  `kv_consumer`/`kv_both` 选择它会明确报错，因为导入的 FP8 页面没有 BF16 镜像。
  D 节点使用 `qfm` 即可。
- DCP>1 保留历史分片和当前 KV 的原归属：各 rank 分页读取 BF16 历史，
  当前 KV 只由 rank0 贡献，继续使用原 LSE/HCCL 合并。它不会变成 DCP1 的一次调用。
- 原 QFM 候选的 PCP、KV layer parallelism 等边界没有在本次扩展。

启动日志应出现 `prefill_backend=fia_paged fia_form=latent`，并有
`Paged FIA reserves BF16 history:` 的预算日志。
切回旧对照时使用 `prefill_backend="fia"`、`fia_form="latent"`，重新启动 Worker；
这条旧路线从唯一的 FP8 历史生成临时 BF16。`expanded` 是另一个计算形式，需单独比较。

## 单卡原生验证及准备/Attention 性能对照

脚本不修改已安装 VA，不需要模型权重，不导入 QFM/DSL provider。

```bash
curl -kfsSL --proxy http://10.155.96.5:8081 --connect-timeout 15 --max-time 120 https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/f07e5aec674d6f8f1ce9b8b3f7ce49159213d30d/experiments/fia-paged-probe.py -o /mnt/fia-paged-probe.py && python /mnt/fia-paged-probe.py --card 0
```

性能对照在同卡、同负载条件下运行：

```bash
python /mnt/fia-paged-probe.py --card 0 --benchmark --queries 8192 --history 32768 --heads 12
```

默认检查 H12/24/48、跨页历史、多个 chunk、三块存储的独立字节与 guard、
负 slot、模拟 APC 改页表及 Decode 镜像，并核对实际 FIA 输出和 LSE。
基准比较旧 Gather/Cast + 两次 FIA/合并与新 BF16 当前写入 + 单次分页 FIA；
历史数值统一，排除 FP8 round-trip 的数值差异。
计时不包含公共 FP8 writer、投影、V 上投影、Gate、调度、HCCL 或 PD，不能直接当作 TTFT。
`--plan` 仅输出粗略显存估算，不初始化 NPU，不是显存预留或硬上限。

## 当前验证状态

595 项相关 CPU 回归和 15 项交付/脚本检查通过；Ruff 与 diff 检查通过。
完整 `format.sh ci` 因本地没有 pre-commit 未执行成功。
本地无 A5，未执行原生算子、模型精度、实际 APC/DSpark/DCP/PD 或性能验收。
固定版本为实验候选，未宣称 TTFT 改善。
