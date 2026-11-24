cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)

project(sfa_standalone LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)

set(ASC_ROOT "/usr/local/Ascend/cann-9.2.1/aarch64-linux/asc")
set(PKG_INC "/usr/local/Ascend/cann-9.2.1/opp/vendors/custom_cot/op_impl/ai_core/tbe/custom_cot_impl/ascendc/sparse_flash_attention")
set(COMMON_ASCENDC "/usr/local/Ascend/cann-9.2.1/opp/vendors/custom_cot/op_impl/ai_core/tbe/custom_cot_impl/ascendc")

# tiling data header from sgl-kernel-npu (upstream public source)
set(TILING_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(sfa_standalone sfa_standalone_entry.asc)

target_include_directories(sfa_standalone PRIVATE
    ${TILING_DIR}
    ${ASC_ROOT}
    ${ASC_ROOT}/include
    ${ASC_ROOT}/include/basic_api
    ${ASC_ROOT}/include/adv_api
    ${ASC_ROOT}/include/interface
    ${ASC_ROOT}/include/tiling
    ${ASC_ROOT}/include/utils
    ${ASC_ROOT}/include/comm_api
    ${ASC_ROOT}/impl
    ${ASC_ROOT}/impl/basic_api
    ${PKG_INC}
    ${PKG_INC}/arch35
    ${COMMON_ASCENDC}
)

target_compile_options(sfa_standalone PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
)
