# 原生量化 + 冷scatter/热写/分区一次调用

此候选在独立 `sfa-write-prepare` 分支。先让主分支正在进行的形状扫描继续完成；
不要在它的运行目录checkout或覆盖文件，不抢占同一NPU。

已同步主分支31affe8的独立精确零query参考门禁，保留原strict及legacy失败记录。
因此大Q不必继续卡在旧tie-only条件；仍须本次状态逐位等于精确oracle，
TwoSum无残差，且全部非零/数学/写入/分区/graph门禁通过。原ABI4库可直接复用。

## 实际改变

旧：native quantize + cat → cold index_copy_ → BF16 hot store/partition → 冷/热attention。
新：native quantize + cat → **cold byte scatter + BF16 hot store + partition** → 同样的attention。

每token400字节packed C4格式保持，最近128仍为原BF16 latent/RoPE。
量化参数、量化调用数、cat、TopK、两读及融合尾部数学都不变。
这与之前重做packing的fused_store候选不同，本轮只去掉冷字节的独立scatter调用。
每一步传入当步quantize返回的新packed指针，不缓存setup结果。

## 门禁

- 原writer_diagnostics比较候选_write与保留的旧_cold_write + CPU BF16字节scatter：
  完整冷/热/rope cache逐字节检查，包含全部未写槽。
- 新partition_diagnostics直接调用候选_write，不能调用旧hot_prepare来替新producer验收；
  六个输出全部投毒，按CPU整数/bool oracle逐元素检查，并恢复缓存。
- 额外捕获完整pipeline，再只投毒本步冷/热/rope写槽及所有分区输出，graph replay必须恢复。
  当前token会被attention从热端读取，因此最终attention输出正确并不能证明冷scatter已捕获；
  此门禁会独立查全部cache字节与分区。
- 既有数值参考、非零query 0.001原生等价、独立精确零Q RNE证明、0.01数学参考和图输出门禁保持。

## 先跑一组，同轮对比当前最好路径

在独立checkout中拉取本分支。复用主分支形状扫描已编译成功的**ABI4**库和同一SDK环境，
无需重编AscendC。新Triton内核会在setup/warmup首次编译，计时中没有CPU读、设备同步或新view。

下面路径按形状扫描的默认work-dir；若实际用了精简首批目录，改成对应的build-r16。
设备确认空闲后再执行，marker沿用原路径，不停止正在工作的服务。

```bash
python3 direct_bridge/run_direct_ab.py \
  --payload-dir /tmp/sfa-direct-payload \
  --bridge-library /tmp/sfa-shape-sweep-v5/build-r16/libsfa_torch_direct.so \
  --device "$SFA_DIRECT_DEVICE" --queries 4 --keys 8192 --direct-aic-count 4 \
  --cold-store-candidate --stages --timing graph \
  --warmup 20 --iterations 100 --repeats 10 --artifacts summary \
  --fault-marker /tmp/sfa-direct-c4-ab-v3/DEVICE_FAULT.json \
  --output /tmp/sfa-write-prepare-q4
```

本次保留原四臂，额外增加direct_fused_cold_store，必测候选/原direct_fused、
候选/installed两遍、候选/W0三组配对。**判断收益以候选/同轮原direct_fused为主**，
不能只胜过125µs两遍就宣称比113µs当前实现又有改进。
所有精度/字节/图门禁通过才MEASURED；独立write阶段与完整流水线均记录，不能相加推算延迟。

先把这组结果、3个新增配对的10轮比值及writer/partition/graph门禁推回本分支。
之后再按主分支扫描得到的代表形状复测Q32/128/512；不要在尚无结果前扩成全参数笛卡尔积。
若新kernel编译/控制失败保留traceback，普通问题可在现场修；设备故障立即停止所有后续调用，
保留marker，不reset或绕过。

本机无A5：真实kernel-body在CPU TL模型下的字节/分区逻辑、动态pointer与图故障门禁已测，
不代替Ascend Triton编译、A5实际数值或性能结果。
