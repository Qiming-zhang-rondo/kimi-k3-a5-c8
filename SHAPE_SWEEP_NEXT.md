# 立即执行多形状扫描

本轮重点是形状覆盖，不把16行或4核当通用最优。原Q4快9.5%的结果保留。
Q是query token数；merge rows是每个AIV批量合并的head行数，两者独立。

- Q：4、32、64、128、256、512、1024，另有Q1对照。
- K：8192、32768；B1/H64/D512/R64/TopK2048/W128保持。
- AIC：4/8/16；Q1自动1核、Q4自动4核，核数必须整除Q。
- BF16 merge rows：8/16/32，实编后实测。FP32备用变体仍保持16，不能把32行大scratch施加给FP32。
- 请求64行在当前H64/双AIV映射中没有64行工作：每AIV最多32行。
  矩阵明确记录DEGENERATE_ALIAS_OF32，不把它伪装成一次独立64行性能结果。
  增大Q会增加query任务，不会增加单个AIV的head行数。
- 当前仍是所有query都选中128个热token的受控子集，TopK来自同一输入；
  这是chunk大小扫描，不是变长热命中或服务端chunk调度验证。

## 一条命令启动，逐组返回

ABI4必须隔离重编，旧 `/tmp/sfa-direct-c4-opt-v4/build-r16` 只用来读取已成功的SDK/CMake设置。
不要覆盖旧库，不猜SDK，不source旧vendor。脚本自动继承ASC_DIR、compiler、includes、
SDK root和必要编译flags。沿用当前已正常工作的Python、ninja PATH和Torch JIT目录。

```bash
python3 direct_bridge/run_shape_sweep.py \
  --validated-build /tmp/sfa-direct-c4-opt-v4/build-r16 \
  --payload-dir /tmp/sfa-direct-payload \
  --device "$SFA_DIRECT_DEVICE" \
  --work-dir /tmp/sfa-shape-sweep-v5 \
  --fault-marker /tmp/sfa-direct-c4-ab-v3/DEVICE_FAULT.json
```

设备使用之前已确认空闲的同一NPU；不要启动共享卡上的其他负载，也不能擅自停止服务。
新work-dir必须不存在；marker沿用原路径，不能因新目录绕过已有故障。

默认120组实际配置，另40条64行alias明确跳过；先跑rows16、K8K、每Q最大所选核数的
Q4→32→64→128→256→512→1024覆盖，再扩K32K和其他rows/core组合。
每完成一组都会刷新matrix.json、summary.csv、best_by_shape.csv。
**先把第一批Q4至Q1024的小表推送回来，进程可继续后续扫描；不用等待120组全部完成。**

如果先只要这一批，等价的精简命令是：

```bash
python3 direct_bridge/run_shape_sweep.py \
  --validated-build /tmp/sfa-direct-c4-opt-v4/build-r16 \
  --payload-dir /tmp/sfa-direct-payload --device "$SFA_DIRECT_DEVICE" \
  --queries 4,32,64,128,256,512,1024 --keys 8192 \
  --merge-rows 16 --aic-counts 16 \
  --work-dir /tmp/sfa-shape-first-v5 \
  --fault-marker /tmp/sfa-direct-c4-ab-v3/DEVICE_FAULT.json
```

这是7组（Q4自动4核、其余16核），不是Q4重复执行1024次。
需要先看计划时在任一命令末尾加 --dry-run；不会编译或运行NPU。

## 门禁和结果口径

每组都是实际Q形状的真实C4写/读、BF16写环、动态分区和热读/合并；
热环按真实ring_plan扩容，Q1为128槽，Q1024为1152槽，不能沿用固定256槽。
同形状的C4 W0、installed两遍、direct两遍、direct融合各自做全量精度及graph投毒检查，
通过后才进行10轮AB/BA配对。仍比较融合/同轮installed与W0，不能跨形状除延迟。

--artifacts summary保留完整逐元素/逐行精度、失败位置与元数据，只避免反复压缩大outputs.npz；
不是抽样验收。原18文件和SHA校验不变，量化配方仍baseline。
各形状最优项按当次已测的融合/installed配对比值选取，标记为best observed；
不把跨配置选择称为理论全局最优。

普通编译失败记录相应rows的BUILD_ERROR并保留日志，继续其他可独立编译配置；
设备故障则立刻停止整个扫描，不reset、不继续发NPU调用。CONTROL_FAILED不能标MEASURED。
最后提交小表、matrix.json、实际失败日志及各组results.json，不要求用户逐项搬文件。

## 可选packed输出候选

新第四ASC编译单元将合并最后的寄存器结果直接RNE打包写BF16 UB，省去外层独立Cast。
原未packed路径保留，同.so中用独立entry与namespace隔离。
本轮默认不启用第五臂，以优先完成形状覆盖；后续加 --packed-candidate可同轮比较packed/unpacked，
仍须所有精度与图门禁。它目前只有宿主模型验证，没有新的A5收益声明。
