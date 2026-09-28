# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

project(aicpu_const_folding_ut)

add_executable(aicpu_const_folding_ut
    folding_ut.cpp
    ${OPS_BASE_PATH}/src/aicpu_const_folding/folding_utils.cpp
)

add_dependencies(aicpu_const_folding_ut aicpu_proto_folding aicpu_context_host aicpu_nodedef_host)

target_include_directories(aicpu_const_folding_ut PRIVATE
    ${OPS_BASE_PATH}/src/aicpu_const_folding
    ${CMAKE_BINARY_DIR}/proto/aicpu_proto_folding
    ${CMAKE_BINARY_DIR}/proto/aicpu_proto_folding/proto
    ${OPS_BASE_PATH}/include
    ${OPS_BASE_PATH}/include/op_common
    ${OPS_BASE_PATH}/include/op_common/aicpu_common/context/common
    ${OPS_BASE_PATH}/include/op_common/aicpu_common/context/cpu_proto
    ${OPS_BASE_PATH}/include/op_common/aicpu_common/context/utils
    ${ASCEND_HOME_PATH}/include
)

target_compile_definitions(aicpu_const_folding_ut PRIVATE
    google=ascend_private
    LOG_CPP
    HOST_VISIBILITY
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(aicpu_const_folding_ut PRIVATE -Wno-deprecated-declarations)
if(ENABLE_COVERAGE)
    target_compile_options(aicpu_const_folding_ut PRIVATE --coverage -fprofile-arcs -ftest-coverage)
endif()

target_link_libraries(aicpu_const_folding_ut PRIVATE
    intf_pub
    c_sec_headers
    $<TARGET_FILE:aicpu_nodedef_host>
    $<TARGET_FILE:aicpu_context_host>
    c_sec
    graph
    metadef
    ascend_protobuf_static
    unified_dlog
    runtime
    gtest
    gtest_main
    $<$<BOOL:${ENABLE_COVERAGE}>:gcov>
    -ldl
    -lrt
)
