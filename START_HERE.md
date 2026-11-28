# 当前任务：应用562ca1ca的两个CPU baseline编译修复

两个报错定位明确：host模板注册/printf混入独立ASC target；框架GET_TILING_DATA_WITH_STRUCT宏不在kernel直调工程生成。尚未进入NPU或模型执行。

已交付 tools/prepare_prolog_direct.py 与 experimental/prolog_inplace_build/direct_tiling_load.h。内网直接执行[BASELINE_COMPILE_FIX.md](experimental/prolog_inplace_build/BASELINE_COMPILE_FIX.md)：脚本只处理新的本地私有副本，隔离host元数据注册，保留常量/原框架分支，按实际device POD显式加载tiling。不修改原私有源码/SDK/vendor，不发明Prolog字段和launch参数。6项宿主解析、预处理、原文件保护与C++17/UBSan检查通过；尚未真实CANN编译。

只在独立ASC target定义QSFA_ASC_DIRECT_BUILD=1。真实device常量必须保留，manifest列出待核验符号；原host tiler和模板key生成器不改。需要真实mode4模板实例和实际kernel产物，不能把未实例化模板.o编译成功当作算子完成。

完成本轮CPU构建后，继续已确定的[Prolog内inline写环](experimental/prolog_ring_smoke/PROLOG_INLINE_INTEGRATION.md)，不增加FP32 GM中转或第二个writer kernel。真实Prolog构建和输出验证仍未完成；公共Python适配/reader通过不能合称整个Prolog完成。

独立reader的B1/B8 eager/Graph等11项已通过，不重复该阶段，不在编译阶段加载模型或运行NPU。APC/P-D真实尾窗自动接线仍须完成，长prompt初始decode同样需要P的尾窗；完整边界与任务见[SERVING_TASK.md](framework_integration/SERVING_TASK.md)。

私有staging树、常量文件、原始构建日志和DO_NOT_UPLOAD_manifest留内网，不再批量外传源码/schema/二进制。