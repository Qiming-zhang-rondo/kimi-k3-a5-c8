# 内网执行任务：一次完成DSL recent128实验

用户希望验证官方实验接口的FP4主缓存+FP8 recent128。先保留现有QSFA/framework工作和服务。此任务不要求升级CANN；在已有匹配、空闲的A5环境中隔离完成，记录CANN/驱动/DSL版本。不要擅自source另一版custom_cot。

## 1. 先用实际环境探测

运行同目录 `test_dsl_recent128.py --probe-only --output <新目录>`，真正import `ops.mixed_quant_sparse_flash_mla` 和配套metadata，记录结果。若已有包满足合同，先跑TopK512数值（`--timing none`），再graph。不能将导入成功当作编译/执行成功。

## 2. 缺包或native未命中时，自主隔离构建，不向用户逐项提问

公开参考仓 `https://github.com/hicann/cannbot-dsl.git` 固定commit **c36d3a2e8a60710cf49d1d09b0b126241dcb6530**；已有内网镜像可用同commit，不使用未核对master。

其 `packaging/native/README.md` 是构建依据：

```bash
# 在独立源码checkout，不覆盖现有工作目录。
git checkout c36d3a2e8a60710cf49d1d09b0b126241dcb6530
bash packaging/native/run-build.sh --help
# 使用与当前Python和aarch64匹配的官方0.8.0 wheel文件；可下载到独立目录。
python -m pip download --no-deps --only-binary=:all: cannbot-dsl==0.8.0 -d /path/to/run/dsl-wheels
# 按实际路径填写，不能抄另一版CANN的环境脚本。
export PYTHON=/actual/path/to/current/python
export CANN_ENV=/actual/path/to/current/cann/set_env.sh
export CANNBOTDSL_WHEEL=/actual/path/to/matching/cannbot_dsl-0.8.0-wheel.whl
bash packaging/native/run-build.sh --operator mqsmla
```

构建产物默认 `packaging/native/output/operators/mqsmla/wheels/`，以实际日志为准。新分发名 `cannbot-arena-net`，Python包仍叫`ops`。只编译mqsmla，不做全量DS41构建。当前实现K1/K2为动态，**不要杜撰MQSMLA profile JSON**；旧README静态512限制不适用于此固定新源码。

使用新的`venv --system-site-packages`继承已有匹配torch/torch_npu，在该venv安装0.8运行时及新net wheel；或依照仓库真实构建说明的隔离target方式。不要改基础环境或混装新`cannbot-arena-net`与旧`cannbot-arena-net-ops`。普通依赖/头文件/路径问题由你自主解决。旧包若已在基础环境，确保新venv实际导入来源正确、运行时满足0.8，不凭pip show猜包已正确加载。

## 3. 正确性先行

用相同源码脚本和独立输出目录：

```bash
python /path/to/test_dsl_recent128.py --device npu:0 --topk 512 --timing none --output /path/to/run/k512-check
python /path/to/test_dsl_recent128.py --device npu:0 --topk 512 --timing graph --output /path/to/run/k512-perf
python /path/to/test_dsl_recent128.py --device npu:0 --topk 2048 --timing none --output /path/to/run/k2048-check
python /path/to/test_dsl_recent128.py --device npu:0 --topk 2048 --timing graph --output /path/to/run/k2048-perf
```

只有选定卡确认空闲才使用npu:0，否则替换为实际空闲卡。先完成首组512即可报告首个可用结果，不等待全扫。四组通过后可补2048 `--selection covered_hot`，单独记录强制热命中场景。

脚本保留同TopK，W0热长度0，W128按每query最近128分区；冷端单独读和热端单独读控制必须通过。sink0分母加1，不是无sink。数学全局及最差query/head匹配误差<=.01；任何设备fault立即停止设备操作并保留首错，不重置/重试业务卡。

核对native报告含installed_hit，require未发生后端编译。报告schema以真实运行时为准；不要把backend_pipeline:0当作发生编译。源码存在、宿主检查、导入、编译、NPU执行、MEASURED分别记录。

## 4. 一次提交小型证据

在 `experimental/dsl_recent128/evidence/` 提交小JSON/表格，不上传大npz/log、wheel、权重或SDK：

- 原始脚本commit、net源码commit、runtime/net版本、各实际文件SHA、CANN/卡、无并行业务证据。
- 各组Q/K/TopK/window/selection和每query实际热命中数。
- W0/W128 kernel-vs-matching与output-vs-original-BF16的全局/最差行L2；控制项结果。
- prepared读W0/W128中位数、10轮AB/BA比值[min,max]、event/wall原始小表；native命中证据。
- 实际环境若为9.2而非9.3，应明确记录，不能宣称完成9.3环境验收。

这不是昨天D576/MXFP4+BF16算子的直接性能对照；不以DSL的320B与原400B或不同维度的82µs推导加速。若需要与当前方案公平排名，另做同维度同输入/同数学合同适配后再比较。
