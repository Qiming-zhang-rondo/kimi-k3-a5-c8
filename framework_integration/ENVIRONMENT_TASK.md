# 当前先确认导师与目标容器的真实环境

用户已确认导师指mxllm PR #199，并明确希望以CANN 9.3作为目标，以使用DSL attention算子。
目标已定为9.3；当前只读取证是为了选择真实配套VA/vLLM/torch_npu/vendor/DSL包，不是要求保留旧CANN或降级。
先读取现有/候选9.3容器信息；在版本和镜像来源确认前不直接升级运行中容器、不加载模型、不做设备测试、不安装wheel、不停止服务。优先用隔离的9.3环境，后续安装按确认的来源和配套执行。

## 已核实PR #199的源码身份

2026-11-27通过用户已登录GitCode浏览器读取：

- PR：https://gitcode.com/InferTurbo/mxllm/pull/199 ，页面当时仍OPEN，目标分支`v0.30.0`。
- PR head：`342680fb25208c85d5d084b4fae9d051f488abfb`。
- 该head的`pyproject.toml`声明vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`、VA `c09d3e20b9c45c80b34ae3781440d90904a44023`。
- 原文件：https://gitcode.com/InferTurbo/mxllm/blob/342680fb25208c85d5d084b4fae9d051f488abfb/pyproject.toml 。这是PR head的源码依赖，不等于导师镜像实际安装身份；仍须收集并比对。
- PR实际修改PrologV3的ACLGraph写缓存路径，优先`cloud_ops_turbo.cloud_mla_prolog_v3`原地更新；缺该callable时才回退functional+整cache commit。它本身没有在两个改动文件中升级CANN或改VA/vLLM依赖，也不是recent128/DSL attention接入PR。

因此CANN9.3的验收分开核对：PR199原地Prolog能力、DSL运行时/net包的实际算子入口、我们自定义KV-split库的匹配SDK与格式。不能因为PR199能启动就认定DSL reader或旧二进制也已验证。

## 不能把旧pin当成目标环境

旧接线审阅基线为MXLLM `3ba7355d`，缓存依赖声明含VA `760d7ecbb962135fff738726825cffacc74f5d2f`及vLLM `4ba787ec9959e566d3b587a796dff07673324da5`。
这些只是历史源码声明，不能声称是导师或用户当前实际安装版本，也不是本轮推荐安装版本；PR199的实际head依赖以上面新核实的值为准。

同样，之前单算子报告出现的CANN toolkit路径和custom_cot/version.info构建版本，只能代表那次报告。vendor的Version不是当前进程实际加载的完整软件组合，不能仅凭两个版本字符串判定可用或要求升级。

## 内网agent先执行只读收集

在**准备运行模型的那个容器和Python环境**，拉本分支，运行：

```bash
python framework_integration/collect_va_environment.py --output /tmp/va_environment.json
```

输出文件已存在时换新路径，不能覆盖。工具只用标准库，不导入torch、torch_npu、vllm或VA，不启动NPU计算。
若实际源码未安装进当前Python路径，可显式补充真实checkout；若自定义vendor不在已选Ascend路径，可补真实version.info：

```bash
python framework_integration/collect_va_environment.py \
  --source-root /actual/mxllm --source-root /actual/vllm-ascend --source-root /actual/vllm \
  --extra-version-file /actual/custom_cot/version.info \
  --output /tmp/va_environment_with_sources.json
```

agent自主定位实际路径，不逐项让用户搬运。只收集实际存在的包/源码，不为了补字段创建假目录或安装依赖。将输出原样提交至`framework_integration/evidence/environment/target.json`，无需人工改写版本。

若能访问导师环境，在导师已知可用容器运行相同命令，提交`mentor.json`。另外记录PR199实际应用commit、镜像tag及image digest；镜像标识只记录标识字段，不能把容器全量inspect/环境变量或凭据上传。没有导师环境访问权限时先完成target，不假称已复现导师镜像。

## 锁定组合后才继续模型接入

需要对照：

1. 实际MXLLM、VA、vLLM源码HEAD与导入位置；pip显示的版本号不足以区分同版本上的PR代码。
2. 导师PR修改的是哪层，是否要求配套VA/vLLM commit或算子包；以PR本身依赖说明和可用环境为准。
3. 实际CANN toolkit/OPP路径及symlink指向，驱动/设备信息，以及当前环境使用的custom vendor包Version/commit/时间戳。
4. 自定义算子Python wheel真实路径与distribution版本，候选.so SHA/导出身份。不能混用另一容器的Python层与旧vendor后宣称复现导师。
5. 模型实际Rope/latent/cache格式。环境对齐仍不代表R64/packed400 kernel支持NoPE272，格式预检继续保留。

报告默认`compatibility=NOT_VERIFIED`，收集成功只表示拿到环境身份。按PR199和CANN9.3目标由agent写`resolved_stack.json`，分清‘已实测/源码要求/尚未知’，记录9.3的完整release/beta/build后缀；然后继续`KV_SPLIT_VA_TASK.md`的隔离接线、真实模型精度验证。不按版本大小猜测兼容性，不要求用户先降回9.2。

用户规划的两条后端仍独立：自定义MXFP4+BF16与DSL FP4+FP8。DSL还涉及不同group/scale和cache布局，当前不顺带实施迁移；读取实际DSL runtime/net wheel版本和入口后再按既有DSL实验合同核验。`ops`是通用Python包名，find_spec找到它也不能单凭名字认定加载了正确DSL wheel。

已有KV-split接入代码可先审阅：Q<=4选split4，较大prefill保留V4；当前仅accuracy_eager，生产Graph/MTP/PD及TPOT还未验证。此代码不是目标环境兼容性证明。
