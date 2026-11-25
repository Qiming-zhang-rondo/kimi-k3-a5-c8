# 下一轮：全冷pair直接走原生搬运

e6ae40c原始证据确认：Q4/K8192/AIC4、return_stats=true、graph10轮，新routed117.64µs对旧overlay97.64µs，配对1.20485，精度逐位一致。不能把“少做6.25%行解码”当成必然加速。当前有效实现仍是overlay。

用户继续要求实现冷端快速路径，争取接近纯C4约80µs。本轮只优化发起KV搬运前的标量依赖；不提前承诺80µs，也不阻塞框架真实精度任务。

## 已实现

- 第四个独立入口 `QsfaSinglePassLaunchFastPair`，旧overlay和上轮routed都原样保留在同一库里。kernel/class/run符号独立，ABI1/15指针/72B tiling不变。
- 两token都在热窗外：范围判断后直接走原生coalesced DMA，不调用Resolve、不读tags、不写slot/token映射。
- 命中热窗候选才查tag；只给实际热行写slot与mask。混合pair只搬冷行C4，热行直接读BF16。保留原生物理pair行序、masked VF及RoPE冷run处理。
- fast类不再包含无用途的16元素token映射数组。没有增加排序、partition或attention调用，也没有删除复用/完成同步。

## 内网执行（自主完成，不让用户逐项中转）

沿用上轮容器、匹配SDK、payload与空闲设备；保留业务和原build。现场上一轮payload为 `/tmp/sfa-direct-payload`，SDK记录为 `/usr/local/Ascend/cann-9.2.1`，是否仍有效自行核对，不source其他vendor。

```bash
cmake -S experimental/single_pass -B /tmp/qsfa-single-pass-fast-pair-build \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DCMAKE_BUILD_TYPE=Release \
  -DSFA_SDK_ROOT="$ASCEND_HOME_PATH"
cmake --build /tmp/qsfa-single-pass-fast-pair-build -j8

python experimental/single_pass/run_single_pass.py \
  --payload-dir /tmp/sfa-direct-payload \
  --library /tmp/qsfa-single-pass-fast-pair-build/libqsfa_recent128_direct.so \
  --device "$QSFA_IDLE_DEVICE" --queries 4 --keys 8192 --aic-count 4 \
  --fast-pair-ab --return-stats --timing none --output /tmp/qsfa-fast-pair-smoke
```

`QSFA_IDLE_DEVICE`由现场确认空闲的设备设置，不擅自停服务。构建必须包含四个ASC TU以及AIC/AIV产物；普通编译/API问题自主修复，不更改原生源码或放宽门槛。

全精度及冷热排列对照通过后，相同参数换新output，运行 `--timing graph --repeats 10`。主计时仍用原始TopK；重排只用于正确性检查，不用重排数据冒充性能收益。保持本轮return_stats=true与e6口径一致。

只需先回报Q4一组：

1. fast_pair对overlay的同轮配对中位数/每轮/最小最大；这是是否替换当前实现的依据。
2. fast_pair对旧routed的同轮配对，区分消除这轮20µs回退与真正超越overlay。
3. original C4、W0、overlay、旧routed、fast_pair、installed C4的同轮时间，库SHA/实际args、数学/最差行/finite、graph门禁。

上传自动生成小证据至 `experimental/single_pass/evidence/fast_pair_q4_raw.json`，不只手写摘要。若仍慢于overlay，保留overlay，报告实际结果，先不要自行启动更多形状扫描。若确实胜出，再补Q1/K8192和Q32/K8192。

此实验仍仅prepared-read，不能声称write+read或TPOT已持平。设备fault立即停止后续设备操作，保留首错及marker，不重置卡后接着报计时。

## 本地验证范围

实际scalar loader与VF行循环在宿主内存模型中编译，旧routed和fast_pair各66场景；检查快路径不解析冷slot、冷数组保持poison、热C4零搬运/读取、物理逆序、混合/全热/全冷和环槽回绕。SDK设备编译、同步和A5性能仍未测。
