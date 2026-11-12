# Kimi K3 A5 C8：QFM＋DSL 单路线试跑候选

此分支用于 A5 拉起、精度及性能回归，尚未向上游提交 PR。
原有运行分支 `qfm-stage1` 和旧的固定版本下载地址保持不变。

代码固定版本：`ba722b4719b3052dfc73aca06478e538147e1da7`。
VA 基线：`c09d3e20b9c45c80b34ae3781440d90904a44023`。

## 一条命令更新框架

```bash
curl -kfsSL --proxy http://10.155.96.5:8081 --connect-timeout 15 --max-time 120 https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/ba722b4719b3052dfc73aca06478e538147e1da7/va030_qfm/install-va030-qfm-only.py -o /mnt/qfm-only-patch.py && python /mnt/qfm-only-patch.py
```

成功打印 `QFM_VA030_ONLY_APPLIED`；已安装打印 `ALREADY_APPLIED`。
支持干净基线和已识别的旧完整补丁。迁移会备份受影响文件，删除已知实验模块，
恢复仅属于 BF16 历史实验的两个 Worker 文件。未知或混合改动会拒绝，不要强制 reset。
安装器不安装包、不编译算子、不自动跑测试。

## P/D 使用相同配置

在现有 additional-config 中保留以下字段，其余业务设置按原服务配置：

```json
{
  "enable_kimi_k3_qfm_c8": true,
  "enable_mlapo": true,
  "enable_kv_nz": false
}
```

删除旧的 `kimi_k3_c8_prefill_backend`、`kimi_k3_c8_fia_form`、
`enable_kimi_k3_mlapo_prefill` 和 `enable_kimi_k3_qfm_fia_prefill` 实验字段。
保留原 `--kv-cache-dtype auto` 和 `--block-size 128` 或已经验证的正整数倍。
P/D 更新后重启所有 Worker，重新分配 KV/APC Cache。

P/D 均为 DSL Prolog → full576 FP8 KV Cache → QFM → Value 上投影。
没有 FIA/QFA 回退或额外 BF16 历史池。全阶段要求 enable_mlapo=true。
启动日志应出现 `writer=dsl-prolog`、`reader=qfm`。

继续使用此前实测成功的 QFM、metadata、DSL 运行环境，在启动服务的同一 shell 加载。
这个框架补丁不负责恢复缺失的 Torch 算子注册，不替换 CANN，也不证明旧 9.2.0
二进制归档在其他 9.2.1 镜像上兼容。此单路线不依赖 QFA920 wrapper。

## 范围与验证

- 生产：15 个文件，新增 946 行、删除 6 行；其中 4 个新增文件。
- CPU 测试：8 个文件，2196 行；NPU writer 验证脚本：182 行。
- 生产＋测试合计新增 3324 行，删除 6 行；文档另有 102 行。
- 330 项相关 CPU 回归、21 项迁移检查通过；Ruff 和 diff 检查通过。
- 完整 format.sh ci 因本地未安装 pre-commit 未执行成功。
- 新精简候选未在本地运行 NPU、HCCL、模型精度或性能验收。

原运行版在减层模型、32 并发、APC 关闭、PD 分离、P/D QFM＋DSL 下，
用户报告 TTFT 持平、TPOT 4.2ms → 3.95ms。这不是新候选已经通过整模回归的声明。
先验证拉起和短请求，再复跑同配置对照，之后准备上游提交。

[可读源码](review/va030-qfm-only/source) ·
[对干净基线的 patch](review/va030-qfm-only/va030-qfm-only.patch) ·
[详细契约和限制](review/va030-qfm-only/source/docs/source/developer_guide/kimi_k3_qfm_c8.md)
