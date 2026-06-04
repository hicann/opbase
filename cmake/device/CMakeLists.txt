# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
include(../fetch_cann_cmake.cmake)
project(opbase_device)

init_cann_project(PRODUCT_SIDE "device" PREPEND_MODULE_PATH)

get_filename_component(OPS_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

include(${OPS_BASE_DIR}/cmake/config.cmake)
include(${OPS_BASE_DIR}/cmake/intf_pub_linux.cmake)
include(${OPS_BASE_DIR}/cmake/func.cmake)

include(${OPS_BASE_DIR}/cmake/third_party/protobuf.cmake)
add_cann_third_party(eigen)

find_cann_package(runtime)
find_cann_package(unified_dlog)

add_cann_subdirectories_relative(${OPS_BASE_DIR}
    ${OPS_BASE_DIR}/aicpu_common/context
)

install(FILES ${PROTOBUF_STATIC_PKG_DIR}/lib/libbase_ascend_protobuf.a
    DESTINATION ${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/lib64
    COMPONENT opbase
)

set_cann_subprj_package()
