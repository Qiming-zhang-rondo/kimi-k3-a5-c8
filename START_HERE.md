# 当前任务：补齐两轮小证据，继续写融合prefill

主分支37bbc60回报Q32–1024读取融合已MEASURED；本分支1c0376b回报Q4写融合111.9µs，
较同轮原读融合快约1.7%。现有上传摘要缺少完整门禁/配对分项，原12MB文件不用上传。

按 [COMPLETED_RUNS_NEXT.md](COMPLETED_RUNS_NEXT.md) 先用纯CPU收集器一次提取已有两轮，
无需重跑；Q32/128/512已由8ac297e报告完成，勿重复；仅补尚未完成的Q1024，并复用已成功ABI4库。
原113µs读融合仍保留作每个形状同轮基线，所有精度、缓存字节、partition和graph门禁保持。

候选实现及原命令见 [WRITE_PREPARE_NEXT.md](WRITE_PREPARE_NEXT.md)。
不能将当前单算子结果称为生产TPOT收益，不能覆盖正在运行的主分支checkout或实验环境。
