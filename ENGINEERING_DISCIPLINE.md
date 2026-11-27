# 工程纪律 — PROLOG 项目教训

## 2026-11-27 BF16 Ring Producer 实现中犯的错

### 犯了什么错

1. **Cast API 用错重载**
   - 我看到 vf 文件里有 `Cast(dst, src, CAST_RINT, count, 1, {1,1,8,4})` 格式就直接抄
   - 没查文档，不知道 6 参数版是 mask/repeat 模式，第 4 参不是元素总数
   - 正确做法：用 4 参数 count 版 `Cast(dst, src, CAST_RINT, count)`
   - **正确示例就在已验证的 kv_split_reduce kernel 里，我都没看**

2. **没读合同就猜字段含义**
   - `prolog_provider.py`（Codex 写的）明确写了 `token_ring_slots` 是"请求池编号"
   - 我看到 "slot" 就拿去做 `slot % capacity`，混淆了 bank 和 position
   - 正确：`row = bank * capacity + position % capacity`（bank 和 position 分开）

3. **只实现了 3 个字段就说"完成"**
   - ABI1 合同有 10 个字段（rope ring, position tags, epoch tags, valid...）
   - 我只写了 latent ring 就说 IMPLEMENTATION_COMPLETE

4. **没考虑 UB scratch 生命周期**
   - 用 sharedBuf 做 BF16 cast 的 scratch，没想过它会跟量化竞争
   - AscendC 流水线：V（向量计算）→ MTE3（GM 写）有依赖关系

### 以后必须做的

```
写代码前:
  1. 查 API 文档（不猜签名，不确定的必须查证）
  2. 读合同/接口定义（别人写的接口文档要逐字段理解）
  3. 找已验证代码里的正确示例（本项目已有很多正确用法）
  4. 列出所有必须实现的字段（对照合同逐项检查）

写代码时:
  5. 不确定的 API 用法，宁可多花时间查，不要抄格式
  6. 声称"完成"之前，对照合同逐项验证

提交前:
  7. 把实现摘要给外部审查（Codex），不要自己说"完成"
```

### 本项目 API 参考

- Cast 4 参 count 版: `Cast(dst, src, RoundMode::CAST_RINT, count)` — kv_split_reduce 里有
- 官方文档: https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0073.html
- ABI1 合同: framework_integration/sfa_recent128/prolog_provider.py
- Ring 寻址: `row = bank * capacity + position % capacity`

### 记住

> 你有机器、有源码、有环境，还比不过只看摘要的外部审查。
> 原因不是能力，是纪律：人家先读文档再动手，你先动手再猜文档。
