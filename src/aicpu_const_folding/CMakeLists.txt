# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(proto_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/../../aicpu_common/context/cpu_proto/proto/cpu_attr.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/../../aicpu_common/context/cpu_proto/proto/cpu_node_def.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/../../aicpu_common/context/cpu_proto/proto/cpu_tensor.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/../../aicpu_common/context/cpu_proto/proto/cpu_tensor_shape.proto
)

add_subdirectory(constant_folding_stub)

generate_cann_protobuf(aicpu_proto_folding PROTO_SRCS PROTO_HDRS ${proto_src_files} TARGET)
set(aicpu_proto_folding_include "${CMAKE_BINARY_DIR}/proto/aicpu_proto_folding")
set(aicpu_proto_folding_proto_include "${CMAKE_BINARY_DIR}/proto/aicpu_proto_folding/proto")

add_library(opconstant_folding SHARED
    folding.cc
    folding_utils.cpp
)

add_dependencies(opconstant_folding aicpu_proto_folding)

set(OPBASE_DEFER_GE_TARGETS FALSE)
if(ENABLE_UNIFIED_BUILD AND NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR AND
   NOT "ge-executor" IN_LIST CANN_BINARY_PACKAGES)
    set(OPBASE_DEFER_GE_TARGETS TRUE)
endif()

if(NOT TARGET lowering)
    if(OPBASE_DEFER_GE_TARGETS)
        message(STATUS "Using lowering target supplied later by the unified GE source build")
    else()
        find_library(OPBASE_LOWERING_SHARED_LIBRARY
            NAMES liblowering.so
            PATHS ${ASCEND_HOME_PATH}
            PATH_SUFFIXES lib64
            NO_DEFAULT_PATH
        )
        if(NOT OPBASE_LOWERING_SHARED_LIBRARY)
            message(FATAL_ERROR "liblowering.so was not found under ${ASCEND_HOME_PATH}/lib64")
        endif()
        add_library(lowering SHARED IMPORTED)
        set_target_properties(lowering PROPERTIES
            IMPORTED_LOCATION "${OPBASE_LOWERING_SHARED_LIBRARY}"
        )
        message(STATUS "Using installed lowering library: ${OPBASE_LOWERING_SHARED_LIBRARY}")
    endif()
else()
    message(STATUS "Using existing lowering CMake target")
endif()

set_target_properties(opconstant_folding PROPERTIES
    OUTPUT_NAME "opconstant_folding"
    PREFIX "lib"
)

target_include_directories(opconstant_folding PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${aicpu_proto_folding_include}
    ${aicpu_proto_folding_proto_include}
    ${OPS_BASE_INCLUDE}
    ${OPS_BASE_INCLUDE}/op_common
    ${OPS_BASE_INCLUDE}/op_common/aicpu_common/context/common
    ${OPS_BASE_INCLUDE}/op_common/aicpu_common/context/cpu_proto
    ${OPS_BASE_INCLUDE}/op_common/aicpu_common/context/utils
    ${ASCEND_INCLUDE_DIR}
)

target_compile_options(opconstant_folding PRIVATE
    -ftrapv
    -fstack-protector-all
    -fvisibility-inlines-hidden
    -fvisibility=hidden
    -Wno-deprecated-declarations
)

target_compile_definitions(opconstant_folding PRIVATE
    _FORTIFY_SOURCE=2
    google=ascend_private
    LOG_CPP
    HOST_VISIBILITY
    _GLIBCXX_USE_CXX11_ABI=0
)

target_link_options(opconstant_folding PRIVATE
    -Wl,-z,relro,-z,now
    -Wl,-Bsymbolic -Wl,--exclude-libs,ALL
    $<$<NOT:$<STREQUAL:${RESERVED_SYMBOL},true>>:-s>
)

target_link_libraries(opconstant_folding PRIVATE
    aicpu_context_headers
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:c_sec_headers>
    -Wl,--no-as-needed
    opbase_constant_folding_ops_stub
    graph
    lowering
    register
    metadef
    ascend_protobuf_static
    unified_dlog
    runtime
    -Wl,--as-needed
    -lrt
    -ldl
)

install(TARGETS opconstant_folding
    LIBRARY DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
    COMPONENT opbase
)
