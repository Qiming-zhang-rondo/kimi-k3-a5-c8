# 热DMA提前入队：对照91.55µs fast-pair

452fa69原始结果确认fast-pair为91.55µs，比同轮overlay97.60µs快6.2%，数学、排列、graph门禁全通过；距离installed C4约80.44µs仍有11.1µs。本轮只改变冷热搬运的发令/同步顺序，保留已有fast-pair有效实现，不承诺提前达到80µs。

## 新候选的依赖关系

原fast-pair：

```
C4 DMA → input MTE2→V queue join → output allocation/reuse
        → hot BF16 DMA → explicit MTE2→MTE3 join → output V→MTE3 join
```

新early-hot（仅有热行的块）：

```
C4 DMA + output allocation/reuse + hot BF16 DMA
        → input MTE2→V queue join（覆盖此前两类DMA）
        → cold decode → output V→MTE3 join → UB→L1
```

保留VECOUT复用的V→MTE2交接；利用输入队列和输出队列的传递依赖省去独立热MTE2→MTE3事件，不增加UB。全冷块保持原allocation/queue顺序。新旧kernel/class/run/导出独立，默认框架入口不变。

该变换也会让cold decode等待hot搬运，可能损失原先的重叠；是否更快必须实测。不能把事件数少直接称收益。

## 现场自主执行

复用前两轮通过的容器/SDK/payload/空闲NPU，保留旧build/库和业务。先确认本机SDK `TQue<VECIN>` 的 EnQue/DeQue 确实发出MTE2→V依赖、`TQue<VECOUT>` 发出V→MTE3依赖，且未被编译选项禁用。检查SDK实现/生成代码即可，不上传私有SDK。队列模型依据可参见官方 [EnQue API](https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0140.html)；本机实现/真实device产物才是此次顺序判断的依据。若不成立，不运行省等待候选，不默默放宽同步要求。

```bash
cmake -S experimental/single_pass -B /tmp/qsfa-single-pass-early-hot-build \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DCMAKE_BUILD_TYPE=Release \
  -DSFA_SDK_ROOT="$ASCEND_HOME_PATH"
cmake --build /tmp/qsfa-single-pass-early-hot-build -j8

python experimental/single_pass/run_single_pass.py \
  --payload-dir /tmp/sfa-direct-payload \
  --library /tmp/qsfa-single-pass-early-hot-build/libqsfa_recent128_direct.so \
  --device "$QSFA_IDLE_DEVICE" --queries 4 --keys 8192 --aic-count 4 \
  --early-hot-ab --return-stats --timing none --output /tmp/qsfa-early-hot-smoke
```

现场自行选择空闲设备、核对已有payload路径，普通依赖/编译问题自主解决，不让用户来回传字段。此库包含第五个ASC TU，确认AIC/AIV真实编译。15指针/72B tiling ABI不变，旧库仍可跑默认路径，新候选需重编。

`--early-hot-ab`包括original C4、W0、overlay、routed、已验证fast-pair以及early-hot。保留完整数学.01/最差行/finite、候选对overlay .001、热集中与冷热交错排列对照。另有8轮不计时热环刷新：同时翻转latent/RoPE符号，逐轮对照fast-pair，检查是否读了陈旧数据；正常8轮恢复原值再计时。设备异常直接退出，不做finally中的NPU恢复。

全部通过后，原参数换新output，跑 `--timing graph --repeats 10`。图投毒门禁不得省略，主要看 `single_pass_w128_early_hot_vs_fast_pair` 同轮AB/BA；secondary看对overlay比值。主计时原TopK完全不变。return_stats=true保持452fa69口径。

本轮先只测Q4/K8192。上传自动产生的 `evidence_summary.json` 至 `experimental/single_pass/evidence/early_hot_q4_raw.json`，保留库SHA、args、原始门禁、刷新检查和逐轮配对。若慢于91µs fast-pair，继续保留fast-pair，不启动更多扫描。若确实胜出，再补Q1/K8192和Q32/K8192。

仍仅prepared-read，不含写环/模型/TPOT；框架真实模型精度任务并行继续。设备fault立即停止后续设备调用，保存首错与marker；不重置卡、不覆盖vendor、不换CANN。

## 本地验证

实际ProcessSparseKv及loader函数在异步宿主队列模型中运行：MTE2写在join前不可见，验证冷热搬运顺序、输出复用交接、全冷路径和NZ完整数据。原loader132场景、流水模型72场景通过。不代替SDK同步语义、设备编译、A5精度或性能。
