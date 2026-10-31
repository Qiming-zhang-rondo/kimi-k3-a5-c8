# 在 CANN 9.2.0 容器构建 QFM 与 metadata

不构建旧的固定 96-head Prolog，不升级 CANN，不修改 VA，不替换全局 Python 包。本地没有 NPU 编译或执行结果。

下载仓库根目录 qfm1.sh 后运行：

```bash
bash /mnt/qfm1.sh --build-ops --insecure --proxy http://10.155.96.5:8081
```

脚本自动寻找现有 CANN 环境与版本记录，在 /mnt/k3-qfm-stage1-deps 创建专用源码和安装目录。
特殊路径可传 --cann-env、--cann-version-file；--jobs 默认为 8。
非空安装目录不会覆盖，可用 --deps-prefix 指定另一个专用目录。

## 实际构建

固定源码：8b6924e68ed27181141cde84fa5a942c0c060aac。两类算子在两个干净源码目录分别构建：

```bash
# QFM 源码目录
bash build.sh --pkg --soc=ascend950 --vendor_name=qfm_stage1 --ops=quant_flash_mla -j8
# metadata 独立源码目录
bash build.sh --pkg --soc=ascend950 --vendor_name=qfm_meta_stage1 --ops=quant_flash_mla_metadata --opkernel_aicpu -j8
# 回到 QFM 源码目录，只构建 Python 前端
bash build.sh --torch_extension
bash build.sh --torch_extension --ops=quant_flash_mla --vendor_name=qfm_stage1
```

不要给 QFM 那条命令加 --opkernel_aicpu：该版本 CMake 会跳过必要的 AI Core 生成步骤。
QFM Python 前端本身注册 QFM 与 metadata 两个接口。

私有安装位置：

- opp/vendors/qfm_stage1_transformer
- opp/vendors/qfm_meta_stage1_transformer
- python：pip --target --no-deps --no-index 安装本次前端 wheel。

两个 vendor 都在测试 Python 启动前启用；原生加载器支持多个 vendor。
setup_qfm.sh 检查 QFM kernel/config、metadata AICPU 库和注册 JSON、两边 op_api 库。
完成后生成 env.sh、build-manifest.json 和日志；qfm1.sh 仅在自己的子进程加载 env.sh。

## 依赖与边界

需要容器已有匹配的 Python/torch/torch_npu/CANN9.2.0，以及 cmake、make、gcc、g++、patch、Python build。
缺依赖会明确失败，不自动升级环境。新算子源码在具体9.2.0包上的兼容性必须实编译确认；失败时回传日志，不关闭兼容检查。

代理传给下载与构建子进程；--insecure 显式启用后只影响下载和构建子进程的 Git TLS，不修改全局 Git 或证书。
部分 CMake URL 下载仍可能要求正确证书，应回传具体错误。

现有 V4 仅做接口检查：其分离 FP8 latent/BF16 raw 或 packed656 不能直接作为 QFM full576 FP8 Cache。
本阶段验证普通量化+Scatter 写入，不声称 V4 融合直写已支持。
可选 prolog_probe.py 不在默认流程中，旧 DSL 固定96-head且不代表产品 TP 支持。
