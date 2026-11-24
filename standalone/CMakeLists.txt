cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "A5 compile target")
find_package(ASC REQUIRED)
project(sfa_standalone LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SFA_PROBE_TEMPLATE_MODE "0" CACHE STRING "Original template mode, 0 or 1")
if(NOT SFA_PROBE_TEMPLATE_MODE MATCHES "^[01]$")
    message(FATAL_ERROR "SFA_PROBE_TEMPLATE_MODE must be 0 or 1")
endif()
set(SFA_SDK_ROOT "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Active toolkit root matching find_package(ASC)")
set(SFA_ASC_ROOT "${SFA_SDK_ROOT}/aarch64-linux/asc" CACHE PATH "ASC root in the same toolkit")
set(SFA_SDK_INCLUDE_DIRS "" CACHE STRING "Additional includes from this same SDK only")
option(SFA_BUILD_DEVICE_SMOKE "Build fixed original BF16 direct-launch smoke executable" OFF)
option(SFA_BUILD_FUSION_PROBE "Compile isolated FP32-output fusion candidate (no device validation claim)" OFF)
option(SFA_BUILD_FUSION_SMOKE "Build isolated FP32-output fusion device-smoke executable" OFF)
get_filename_component(SFA_BUNDLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

function(configure_sfa target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
    target_compile_definitions(${target} PRIVATE SFA_PROBE_TEMPLATE_MODE=${SFA_PROBE_TEMPLATE_MODE})
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${SFA_BUNDLE_ROOT}/original"
        "${SFA_BUNDLE_ROOT}/original/arch35"
        "${SFA_BUNDLE_ROOT}/deps/sparse_flash_attention/arch35"
        "${SFA_BUNDLE_ROOT}/deps/ascendc_common" ${SFA_SDK_INCLUDE_DIRS})
    if(SFA_SDK_ROOT)
        foreach(suffix "" "/include" "/include/basic_api" "/include/adv_api"
                "/include/interface" "/include/tiling" "/include/utils" "/include/comm_api"
                "/impl" "/impl/basic_api")
            if(EXISTS "${SFA_ASC_ROOT}${suffix}")
                target_include_directories(${target} PRIVATE "${SFA_ASC_ROOT}${suffix}")
            endif()
        endforeach()
    endif()
endfunction()

# The renamed symbol has a raw user-scratch ABI, not an installed ACLNN ABI.
add_library(sfa_standalone STATIC sfa_standalone_entry.asc)
configure_sfa(sfa_standalone)
add_executable(sfa_host_preflight sfa_direct_launch.cpp)

# Candidate entry uses an explicit experimental header path. Never add the
# experimental directory to configure_sfa: the verified original shares names.
if(SFA_BUILD_FUSION_PROBE OR SFA_BUILD_FUSION_SMOKE)
    if(NOT SFA_PROBE_TEMPLATE_MODE STREQUAL "0")
        message(FATAL_ERROR "Fixed fusion compile probe requires template mode 0")
    endif()
endif()
if(SFA_BUILD_FUSION_PROBE)
    add_library(sfa_fusion_probe STATIC sfa_fusion_entry.asc)
    configure_sfa(sfa_fusion_probe)
endif()

if(SFA_BUILD_DEVICE_SMOKE OR SFA_BUILD_FUSION_SMOKE)
    if(NOT SFA_PROBE_TEMPLATE_MODE STREQUAL "0")
        message(FATAL_ERROR "Fixed smoke launcher requires template mode 0")
    endif()
    if(NOT SFA_SDK_ROOT)
        message(FATAL_ERROR "Set SFA_SDK_ROOT to the active matching toolkit; no fallback vendor is used")
    endif()
    find_path(SFA_ACL_INCLUDE_DIR acl/acl.h
        PATHS "${SFA_SDK_ROOT}/aarch64-linux/include" "${SFA_SDK_ROOT}/include" NO_DEFAULT_PATH)
    find_library(SFA_ACL_LIBRARY ascendcl
        PATHS "${SFA_SDK_ROOT}/aarch64-linux/lib64" "${SFA_SDK_ROOT}/lib64" NO_DEFAULT_PATH)
    if(NOT SFA_ACL_INCLUDE_DIR OR NOT SFA_ACL_LIBRARY)
        message(FATAL_ERROR "Matching ACL header/library not found in SFA_SDK_ROOT")
    endif()
endif()

# Each smoke executable owns exactly one ASC TU including its chosen kernel.
# Do not link the probe archive again or compile the deprecated C++ launcher.
if(SFA_BUILD_DEVICE_SMOKE)
    add_executable(sfa_launch sfa_direct_launch.asc)
    configure_sfa(sfa_launch)
    target_include_directories(sfa_launch PRIVATE "${SFA_ACL_INCLUDE_DIR}")
    target_link_libraries(sfa_launch PRIVATE m "${SFA_ACL_LIBRARY}" ascendc_runtime)
endif()
if(SFA_BUILD_FUSION_SMOKE)
    add_executable(sfa_fusion_launch sfa_fusion_launch.asc)
    configure_sfa(sfa_fusion_launch)
    target_include_directories(sfa_fusion_launch PRIVATE "${SFA_ACL_INCLUDE_DIR}")
    target_link_libraries(sfa_fusion_launch PRIVATE m "${SFA_ACL_LIBRARY}" ascendc_runtime)
endif()
message(STATUS "SFA explicit-user-scratch fixed contract; SDK=${SFA_SDK_ROOT}; compiler=${CMAKE_ASC_COMPILER}")


# Phase C paired performance benchmark
option(SFA_BUILD_PHASE_C "Phase C paired benchmark" OFF)
if(SFA_BUILD_PHASE_C)
    add_executable(sfa_phase_c sfa_pc_fused_bench.asc)
    configure_sfa(sfa_phase_c)
    find_path(SFA_ACL_INC_DIR acl/acl.h PATHS /usr/local/Ascend/cann-9.2.1/aarch64-linux/include NO_DEFAULT_PATH)
    target_include_directories(sfa_phase_c PRIVATE ${SFA_ACL_INC_DIR})
    find_library(SFA_ACL_LIB ascendcl PATHS /usr/local/Ascend/cann-9.2.1/lib64 NO_DEFAULT_PATH)
    target_link_libraries(sfa_phase_c PRIVATE m ${SFA_ACL_LIB})
endif()
