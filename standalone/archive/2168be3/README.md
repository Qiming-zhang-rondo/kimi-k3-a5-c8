此目录原样保存提交 2168be3 中新增的三份 standalone/arch35 融合头。
`.snapshot` 文件仅供追溯，不参与构建。

这些副本曾遮蔽原版头文件，且包含小 max/sum 的零字节 DataCopy 与错误方向的
四参 UB→GM DataCopyPad。不能重新放回 standalone/arch35 或加入 include 路径。
当前候选唯一有效源码在 experimental/arch35，原版源码在 original/arch35。
修复与验证边界见 PHASE_B_2168_REVIEW.md。
