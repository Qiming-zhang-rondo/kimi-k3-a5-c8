# 当前立即执行：补齐三项后直接编译producer，别再停在摘要

192946ea已在本地源修正四参Cast、bank*C+position%C、valid谓词和静态容量，这些方向接受，不再反复改。
当前仍未编译；立即按 [framework_integration/PROLOG_REVIEW_5E7067F0.md](framework_integration/PROLOG_REVIEW_5E7067F0.md)
顶部192946ea审查直接完成以下步骤。在没有内网用户明确暂停指令时，不要以等待Codex反馈为由停住。

1. 修scratch依赖：PIPE_V不覆盖V→MTE3或DMA结束前的scratch复用；用独立区域/正确事件完成所有权闭环。
2. 一次对齐新variant全链：Python、注册/C++ binding、proto、TBE、tiling和kernel入口。8GM+2scalar不能沿旧3参数ABI launch，
   容量/池大小走明确静态合同，workspace/tiling位置逐项核对。
3. ON在host binding/tiling拒绝未实现SplitM/arch22分支；所选A5 mode4真实写latent、post-RoPE KR和position/epoch tags。
4. 立刻隔离编译。编译首错由内网agent按当前SDK/已有源自行修，不再只推实现摘要；有真实外部阻断才按允许范围反馈。
5. 最小B1单token/512全元素/回绕数值：C4 packed和scale逐字节不变、BF16等于prequant RNE。
6. 再B8Q4/padding/reorder、动态Graph毒化重放；通过才测producer成本，再推进模型生产接线。

半小时目标优先拿到真实BUILD_PASS及最小DEVICE_VALIDATED，不能承诺完整生产服务已就绪。
producer编好不自动解决现reader B1寻址/accuracy_eager与PD生命周期；不能删guard或关热环当作接入完成。

公共接口和全部本地实施要求见 [framework_integration/PROLOG_PRODUCER_TASK.md](framework_integration/PROLOG_PRODUCER_TASK.md)。
不重启工作服务、不升级环境、不继续写系统vendor；使用独立候选工程/作用域，原成功产物保留。
内部源/diff/头/schema/二进制留内网，只回传允许的实际构建首错或数值/图测试结果，不要求更多源码上传。

P/D共享producer代码，但ring=None原样走原Prolog。P至少准备prompt尾窗给D，P自己的attention不必改热窗；
D每步更新新token/draft。两个节点源码MD5相同不等于编译/运行已生效，PD仍须传BF16尾窗并等待完成。
indexer C4与主C4布局/量化参数都保持原样。已暂缓的激活重要性实验和SFA tile扫描本轮不执行。
