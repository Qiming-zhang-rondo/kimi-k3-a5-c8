# 内网 agent：找到匹配版本的原生 C4 reader，并尝试提交源码

用户已要求尝试推送源码。此次先做源码定位与交接，不再跑微基准、不改算子、不启动新服务。已验证两遍/读归约融合及原服务全部保留。框架精度任务仍在，完成这次短交接后继续原任务。

目标是让外部 Codex 能基于**当前实际使用的原生 MXFP4 KV Sparse Attention**实现 C4 冷端 + BF16 recent128 热端的单次 attention。当前外部仓只有 BF16 kernel、C4 Python wrapper / op_proto，不能把它们当成 C4 实现重复提交。

## 一次完成以下步骤

1. **先锁定当前运行版本。**从当前 c4 实验容器/环境确定 `cloud_kv_quant_sparse_flash_attention` 实际 wheel、C++ 转发、ACLNN entry、vendor kernel 的对应关系；记录包 `version.info`、commit/timestamp、CANN/编译器版本，以及实际 `.so` / kernel 文件 SHA256。当前候选已知有 `aclnnKvQuantSparseFlashAttentionV2` 这条调用链，但以本机实际证据为准，不猜路径或版本。不要 source 不匹配的旧包、改变全局环境或执行 NPU 算子。
2. **在本机现有安装包、已解包目录、源码 checkout 和构建缓存中定位源。**先查看实际 vendor 的 `op_impl/ai_core/tbe`、`custom_cot_impl/ascendc`、当前构建源码位置，再用 `rg --files` / `rg` 搜索算子名、entry 名、MXFP4/E2M1 分派。找不到时继续查已有 `.run` 归档/源码仓与构建记录，不要立刻让用户找文件。不要只找同名 Python，也不要拿 BF16 SFA、C8-only 或 TurboQuant 冒充 MXFP4 C4。
3. **收齐可修改的最小完整单元。**保留真实目录关系及许可证，包括下面清单；依赖不能靠猜造假的头/结构补齐。若当前源码与部署二进制的 commit 无法对应，明确标记 `SOURCE_MATCH_UNPROVEN`，仍可交付已找到的合法源码，不能声称完全匹配。
4. **检查上传范围。**目标 `Qiming-zhang-rondo/kimi-k3-a5-c8` 当前是公开 GitHub 仓库。只提交允许对外提供的源码，保留原版权/许可证；文件若有明确禁止外发的限制，留在内网，交付不含受限实现的接口/依赖/版本摘要及可用的公开来源。不要整包上传内部 vendor 安装、权重、原始激活、token、配置凭据、完整环境变量或大型二进制。此项无需把普通文件查找/依赖问题转给用户。
5. **直接提交并推送。**本分支 `sfa-framework-recent128`；先 fetch 并保留最新提交，用独立 checkout/worktree，不覆盖正在做模型验证的工作目录。不 force push。提交后回报 commit 与收集状态；若网络/权限失败，保留一次实际错误和本地 commit，不把“已 commit”说成“已 push”。

## 最小文件集合

- 原生 MXFP4 C4 算子的 kernel entry，以及实际使用的 arch35 kernel/service/cache 搬运与反量化文件。
- host tiling 实现、tiling data POD、tiling key、dtype/layout/quant mode 分派。仅 `op_proto` 不是 host tiling。
- 这些文件直接或间接依赖的非 SDK 公共头、编译宏和构建目标。能由公开 CANN SDK 提供的文件记录版本/include 路径即可，不复制整个 SDK。
- 对应的 Python/C++/ACLNN 参数合同、build 命令和源码版本来源。
- packed C4 的 nibble 顺序、E8M0 scale 粒度和字节布局；RoPE64/NoPE0、K/V alias、workspace/system 前缀和多核切分的真实定义位置。保留源码原样，另写说明，不在这轮改格式。

## 仓库交付结构

```text
native_c4_source/
  PROVENANCE.md            # 实际调用链、匹配情况、许可证/公开来源
  MANIFEST.json            # 相对路径、字节数、SHA256、原相对位置
  BUILD.md                 # 已有真实命令/SDK需求；没有构建过就写 NOT_BUILT
  source/                  # 允许上传的最小完整源码，保留其相对目录
framework_integration/evidence/c4_source_export.json
```

`c4_source_export.json` 至少包含：

- `status`: `SOURCE_READY` / `SOURCE_MATCH_UNPROVEN` / `NOT_FOUND` / `EXPORT_RESTRICTED` / `PUSH_FAILED`；
- 当前 operator/CANN 版本、源码 commit、关键二进制 SHA、匹配依据；
- 找到的 kernel entry、C4 sparse gather/反量化位置、host tiling/POD、构建目标；
- 实际收集文件数/字节数、清单路径、未收齐的明确依赖；
- 是否执行过构建/设备测试（本轮预计都为 false）。

若源码确实不存在或明确受限，也提交一次完整结论：查过哪些**相关**位置、实际找到的文件、缺少哪一组源、匹配公开上游是否可用。不要反复只让用户发路径，不上传与目标无关的大日志。**本轮不承诺或伪造单次 attention 已实现。**
