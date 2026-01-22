# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set_target_properties(ops_base PROPERTIES OUTPUT_NAME "ops_base")
#### CPACK to package run #####
message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    message(STATUS "Detected architecture: x86_64")
    set(ARCH x86_64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
    message(STATUS "Detected architecture: ARM64")
    set(ARCH aarch64)
else ()
    message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif ()
# 打印路径
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/makeself-fetch.cmake)

set(script_prefix ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/ops_base/scripts)
install(DIRECTORY ${script_prefix}/
    DESTINATION share/info/ops_base/script
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 文件权限
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    DIRECTORY_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 目录权限
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    REGEX "(setenv|prereq_check)\\.(bash|fish|csh)" EXCLUDE
)
set(SCRIPTS_FILES
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/check_version_required.awk
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func.inc
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.sh
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.csh
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.fish
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_compatiable.inc
)

install(FILES ${SCRIPTS_FILES}
    DESTINATION share/info/ops_base/script
)
set(COMMON_FILES
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/install_common_parser.sh
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func_v2.inc
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_installer.inc
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/script_operator.inc
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_cfg.inc
)

set(PACKAGE_FILES
    ${COMMON_FILES}
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/multi_version.inc
)
set(LATEST_MANGER_FILES
    ${COMMON_FILES}
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func.inc
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_compatiable.inc
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/check_version_required.awk
)
set(CONF_FILES
    ${CMAKE_SOURCE_DIR}/scripts/package/common/cfg/path.cfg
    ${CMAKE_SOURCE_DIR}/src/nnopbase/common/op_info_record/dump_tool_config.ini
)
install(FILES ${CMAKE_SOURCE_DIR}/version.info
    DESTINATION share/info/ops_base
)
install(FILES ${CONF_FILES}
    DESTINATION ops_base/conf
)
install(FILES ${PACKAGE_FILES}
    DESTINATION share/info/ops_base/script
)
install(FILES ${LATEST_MANGER_FILES}
    DESTINATION latest_manager
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/scripts/package/latest_manager/scripts/
    DESTINATION latest_manager
)

set(opp_source ${CMAKE_SOURCE_DIR}/include)
install(DIRECTORY ${opp_source}/
    DESTINATION ops_base/include
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE
    GROUP_READ GROUP_EXECUTE
)

set(pkg_inc_src ${CMAKE_SOURCE_DIR}/pkg_inc)
install(DIRECTORY ${pkg_inc_src}/
    DESTINATION ops_base/pkg_inc
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE
    GROUP_READ GROUP_EXECUTE
)
install(TARGETS ops_base
        LIBRARY DESTINATION ops_base/lib)
set(AICPU_LIBS
    aicpu_context
    aicpu_nodedef
    aicpu_context_host
    aicpu_nodedef_host
    aicpu_cust_log
)

install(TARGETS ${AICPU_LIBS}
    ARCHIVE DESTINATION ops_base/lib/aicpu_common
)

install(FILES ${PROTOBUF_STATIC_PKG_DIR}/lib/libbase_ascend_protobuf.a
    DESTINATION ops_base/lib/aicpu_common)
install(FILES ${PROTOBUF_HOST_STATIC_PKG_DIR}/lib/libhost_ascend_protobuf.a
    DESTINATION ops_base/lib/aicpu_common)

message(STATUS "ASCEND_HOME_PATH: $ENV{ASCEND_HOME_PATH}")
get_filename_component(COMPILER_PATH $ENV{ASCEND_HOME_PATH}/compiler REALPATH)
get_filename_component(VER_PATH "${COMPILER_PATH}" DIRECTORY)
message(STATUS "VERSION PATH: ${VER_PATH}")

set(aclnn_source ${CMAKE_SOURCE_DIR}/include/nnopbase/aclnn)
install(DIRECTORY ${aclnn_source}/
    DESTINATION ops_base/aclnn
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE
    GROUP_READ GROUP_EXECUTE
)
set(opdev_source ${CMAKE_SOURCE_DIR}/include/nnopbase/opdev)
install(DIRECTORY ${opdev_source}/
    DESTINATION ops_base/aclnn/opdev
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE
    GROUP_READ GROUP_EXECUTE
)

set(aclnnop_source ${CMAKE_SOURCE_DIR}/include/aclnnop)
install(DIRECTORY ${aclnnop_source}/
        DESTINATION ops_base/aclnnop
        FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE
        GROUP_READ GROUP_EXECUTE
)
install(DIRECTORY ${aclnnop_source}/
        DESTINATION ops_base/aclnnop/level2
        FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE
        GROUP_READ GROUP_EXECUTE
)

install(TARGETS nnopbase
   LIBRARY DESTINATION ops_base/lib/aclnn
)

install(TARGETS dummy_tls
   LIBRARY DESTINATION ops_base/lib/aclnn
)

install(TARGETS stub_nnopbase
   LIBRARY DESTINATION ${ARCH}-linux/devlib/linux/${ARCH}
)

# ============= CPack =============
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}")

set(CPACK_INSTALL_PREFIX "/")

set(CPACK_CMAKE_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
set(CPACK_CMAKE_BINARY_DIR "${CMAKE_BINARY_DIR}")
set(CPACK_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CPACK_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CPACK_ARCH "${ARCH}")
set(CPACK_SET_DESTDIR ON)
set(CPACK_GENERATOR External)
set(CPACK_EXTERNAL_PACKAGE_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/makeself.cmake")
set(CPACK_EXTERNAL_ENABLE_STAGING true)
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_BINARY_DIR}")

message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
include(CPack)