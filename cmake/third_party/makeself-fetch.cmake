# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(MAKESELF_NAME "makeself")

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if(NOT CANN_3RD_LIB_PATH)
    set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/third_party)
endif()

if (NOT CANN_3RD_PKG_PATH)
    set(CANN_3RD_PKG_PATH ${PROJECT_SOURCE_DIR}/third_party/pkg)
endif ()

set(MAKESELF_DOWNLOAD_PATH ${CANN_3RD_PKG_PATH}/makeself)
set(MAKESELF_INSTALL_PATH ${CANN_3RD_LIB_PATH}/makeself)

if (IS_DIRECTORY "${CANN_3RD_LIB_PATH}/${MAKESELF_NAME}/")
    set(MAKESELF_PATH ${CANN_3RD_LIB_PATH}/${MAKESELF_NAME})
else()
    set(MAKESELF_PATH ${MAKESELF_INSTALL_PATH})
endif()

# 默认配置的makeself还是不存在则下载
if (NOT EXISTS "${MAKESELF_PATH}/makeself-header.sh" OR NOT EXISTS "${MAKESELF_PATH}/makeself.sh")
    set(MAKESELF_URL "https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz")
    message(STATUS "Downloading ${MAKESELF_NAME} from ${MAKESELF_URL}")

    include(ExternalProject)
    ExternalProject_Add(makeself
      URL                         ${MAKESELF_URL}
      URL_MD5                     a080aaf744fcae5d96477c0a799ca469
      DOWNLOAD_DIR                ${MAKESELF_DOWNLOAD_PATH}
      SOURCE_DIR                  ${MAKESELF_INSTALL_PATH}
      CONFIGURE_COMMAND           ""
      BUILD_COMMAND               ""
      INSTALL_COMMAND             ""
    )
else ()
    message("makeself found in ${MAKESELF_PATH}, no need download")
endif()

message("MAKESELF_PATH:${MAKESELF_PATH}")
message("CMAKE_BINARY_DIR:${CMAKE_BINARY_DIR}")
execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${MAKESELF_PATH}" "${CMAKE_BINARY_DIR}/makeself"
    COMMAND chmod 700 "${CMAKE_BINARY_DIR}/makeself/makeself.sh"
    COMMAND chmod 700 "${CMAKE_BINARY_DIR}/makeself/makeself-header.sh"
    RESULT_VARIABLE CHMOD_RESULT
    ERROR_VARIABLE CHMOD_ERROR
)