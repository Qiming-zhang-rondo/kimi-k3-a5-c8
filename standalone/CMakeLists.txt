cmake_minimum_required(VERSION 3.16)

# Resolve ASC from the active, matched toolkit (ASC_DIR/CMAKE_PREFIX_PATH).
# This target only compiles archived source; it never installs or launches.
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "A5 compile target")
find_package(ASC REQUIRED)
project(sfa_standalone LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SFA_PROBE_TEMPLATE_MODE "0" CACHE STRING "Compile one original C/V template mode: 0 or 1")
if(NOT SFA_PROBE_TEMPLATE_MODE MATCHES "^[01]$")
    message(FATAL_ERROR "SFA_PROBE_TEMPLATE_MODE must be 0 or 1; it is not a runtime tiling key")
endif()
set(SFA_SDK_INCLUDE_DIRS "" CACHE STRING "Additional includes from the SAME toolkit that provides ASC")
get_filename_component(SFA_BUNDLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# No host main() is required. The non-template kernel instantiates Init/Process;
# an unused kernel template could yield an archive with no device implementation.
add_library(sfa_standalone STATIC sfa_standalone_entry.asc)
target_compile_options(sfa_standalone PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
target_compile_definitions(sfa_standalone PRIVATE SFA_PROBE_TEMPLATE_MODE=${SFA_PROBE_TEMPLATE_MODE})
target_include_directories(sfa_standalone PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${SFA_BUNDLE_ROOT}/original"
    "${SFA_BUNDLE_ROOT}/original/arch35"
    "${SFA_BUNDLE_ROOT}/deps/sparse_flash_attention/arch35"
    "${SFA_BUNDLE_ROOT}/deps/ascendc_common"
    ${SFA_SDK_INCLUDE_DIRS}
)
message(STATUS "SFA compile-only target; public SGL POD, private source ABI NOT validated")
message(STATUS "SFA source: ${SFA_BUNDLE_ROOT}; template mode: ${SFA_PROBE_TEMPLATE_MODE}")
