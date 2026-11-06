# A5：展开式 FIA / QFA 候选验证

固定代码提交：`d97680798b0e9138d4e679ba6f7e62aecdce7596`。
基线 VA：`c09d3e20b9c45c80b34ae3781440d90904a44023`，CANN9.2.0。
本地463项框架CPU回归、31项探针/安装检查通过；尚无本版NPU精度、性能或整模验收结果。
Ruff/格式/差异检查通过；完整pre-commit因本地缺少该工具未执行。

## 1. 统一框架补丁（不安装算子、不自动测试）

在现有容器中执行，自动定位editable VA；支持干净基线和已识别的旧完整补丁，含geometry和FIA两版。

```bash
curl -kfsSL --proxy http://10.155.96.5:8081 --connect-timeout 15 --max-time 120 https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/d97680798b0e9138d4e679ba6f7e62aecdce7596/va030_qfm/install-va030-qfm-expanded-qfa.py -o /mnt/expanded-attention-patch.py && python /mnt/expanded-attention-patch.py
```

成功标记：`QFM_VA030_EXPANDED_QFA_APPLIED`。已安装则打印`ALREADY_APPLIED`。
遇到未知改动会拒绝，保留备份；不要强制reset。重启所有Worker后生效。

## 2. 先测FIA两种前处理

单卡脚本不修改已安装VA。使用当前CANN环境；无需加载QFA隔离包装。

```bash
curl -kfsSL --proxy http://10.155.96.5:8081 --connect-timeout 15 --max-time 120 https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/d97680798b0e9138d4e679ba6f7e62aecdce7596/experiments/expanded-attention-probe.py -o /mnt/expanded-attention-probe.py && python -u /mnt/expanded-attention-probe.py --card 0 --attention fia
```

默认真实K3投影形状7168/1536，heads12/24/48，Q9，历史129。
`ordinary`是现有MXFP8小算子；`native`是包内CANN NoPE/V4 MLAPO，非Decode DSL。
可加`--frontend ordinary`或`--frontend native`独立测试。
MLAPO版用私有identity UK保留原Q128再接raw64，仍有128→512额外计算，不能预设更快。

## 3. QFA：一次性准备9.2.0配套包装

QFA选官方v9.2.0提交`dff221330eecd2132233b1e73ed1072cd8925140`。
QFM Stage1公共Python wheel带新版QFA tail ABI；不能直接调9.2.0旧native。
下面显式复制当前`cann_ops_transformer`到新私有目录，仅将QFA Python/C++/图转换器固定为9.2.0；
原QFM provider、原包、OPP和库路径保留。只准备包装，不编译Attention内核，不初始化NPU。
首次调用相关C++主机包装可能构建扩展，计时前必须预热。

**先按原服务方式source现有QFM、DSL环境，再执行一次：**

```bash
curl -kfsSL --proxy http://10.155.96.5:8081 --connect-timeout 15 --max-time 120 https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/d97680798b0e9138d4e679ba6f7e62aecdce7596/experiments/install-qfa920-wrapper.py -o /mnt/install-qfa920-wrapper.py && python /mnt/install-qfa920-wrapper.py --prefix /mnt/k3-qfa920-d9768079 && source /mnt/k3-qfa920-d9768079/env.sh
```

该prefix必须是新目录；已经准备过就只执行`source /mnt/k3-qfa920-d9768079/env.sh`。
新容器各自准备包装。此env应在原QFM/DSL env之后source；随后启动新的Python进程/所有Worker。
也可用`--source-python /实际QFM私有runtime/python`明确指定源包。
未知共享helper会明确拒绝，不会猜测兼容或覆盖原包。

## 4. QFA单卡正确性

复用第2步下载的脚本，在第3步环境中执行：

```bash
python -u /mnt/expanded-attention-probe.py --card 0 --attention qfa
```

QFA并非原生192：Q/K192与V128临时补到256，输出裁128，保留原模型softmax scale。
采用MXFP8 mode1；Q/K按D轴Group32，V按token轴Group32。
持久Cache仍是一份full576 FP8 latent，没有新增永久展开Cache或BF16历史池。

参考算法包含官方内部P的FP8舍入，使用官方结果判据；另报相对BF16 FIA的额外量化损失。
失败会保留诊断并拒绝进入性能计时，不能把有限输出当精度通过。
脚本打印实际包装来源与ABI；Python签名检查不等于底层native二进制验收。

需要逐个覆盖V尾块时，可在小算子前处理下运行：

```bash
for K3_HISTORY in 31 32 33 63 64 65 129; do python -u /mnt/expanded-attention-probe.py --card 0 --attention qfa --frontend ordinary --heads 12 --history "$K3_HISTORY" || break; done
```

## 5. 正确性通过后，比较完整链路耗时

```bash
python -u /mnt/expanded-attention-probe.py --card 0 --attention fia --heads 12 --queries 8192 --history 32768 --benchmark
python -u /mnt/expanded-attention-probe.py --card 0 --attention qfa --heads 12 --queries 8192 --history 32768 --benchmark
```

两条均测ordinary/native；可指定`--frontend ordinary`缩小范围。
计时含投影、RMSNorm、当前FP8写入、历史反量化/上投影、Attention、LSE合并及输出复制，
不含权重准备、CPU参考、首次加载/JIT、o_proj/gate、模型调度和通信。
同时看`median_npu_ms`、同步wall、峰值显存；`--plan`可CPU查看粗略显存估计，非硬上限。
共享卡争用会影响性能比较。长case的独立CPU参考也需要时间。

## 6. 整模配置

在现有additional-config里合并，其他TP/DSpark/并发/权重/启动参数保持一致：

```json
{
  "enable_kimi_k3_qfm_c8": true,
  "enable_kv_nz": false,
  "enable_mlapo": true,
  "kimi_k3_c8_prefill_backend": "fia",
  "kimi_k3_c8_fia_form": "expanded",
  "enable_kimi_k3_mlapo_prefill": false
}
```

- `prefill_backend=fia`：BF16展开式FIA；改为`qfa`：临时MXFP8 QFA。
- `enable_kimi_k3_mlapo_prefill=false`：小算子；`true`：包内原生MLAPO。
- `enable_mlapo=true`：Decode保持DSL mode2＋QFM；不是把Decode换成QFA。
- 保留`--kv-cache-dtype auto`和`--block-size 128`或正整数倍。
- FIA/QFA Prefill仍为Eager；FULL_DECODE_ONLY继续使用原QFM图路径。
- 启动日志确认`prefill_backend=... fia_form=expanded prefill_frontend=ordinary/native-prolog`。

历史按KV轴分段，每次保留完整Q。APC/页表/stride、主草稿隔离、混合P/D与DCP协议沿用并有CPU回归；
仍需真实NPU、DSpark接受长度、APC部分命中、图重放、DCP/PD及整模精度验收。
QFA补维后Attention维度乘加量从192+128增至256+256，再叠加临时量化和转置，不能预设优于FIA。

可读源码：[review/va030-qfm-c8/source](review/va030-qfm-c8/source)。
