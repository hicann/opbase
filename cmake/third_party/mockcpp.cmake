# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

include_guard(GLOBAL)

if(NOT CANN_3RD_LIB_PATH)
    set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/third_party)
endif()

if (NOT CANN_3RD_PKG_PATH)
    set(CANN_3RD_PKG_PATH ${PROJECT_SOURCE_DIR}/third_party/pkg)
endif ()

set(MOCK_DOWNLOAD_PATH ${CANN_3RD_PKG_PATH}/mockcpp)
set(MOCK_INSTALL_PATH ${CANN_3RD_LIB_PATH}/mockcpp)
set(mockcpp_FLAGS "-fPIC")
set(mockcpp_LINKER_FLAGS "")
set(BUILD_TYPE "DEBUG")
set(BOOST_INCLUDE_DIRS ${CANN_3RD_LIB_PATH}/boost)

if (CMAKE_GENERATOR MATCHES "Unix Makefiles")
    set(IS_MAKE True)
    set(MAKE_CMD "$(MAKE)")
else()
    set(IS_MAKE False)
endif()

if (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(mockcpp_CXXFLAGS "-fPIC")
else()
    set(mockcpp_CXXFLAGS "-fPIC -std=c++11")
endif()

if ((NOT DEFINED ABI_ZERO) OR (ABI_ZERO STREQUAL ""))
    set(ABI_ZERO "true")
endif()

if(ABI_ZERO STREQUAL "true")
    set(mockcpp_CXXFLAGS "${mockcpp_CXXFLAGS} -D_GLIBBCXX_USE_CXX11_ABI=0")
    set(mockcpp_FLAGS "${mockcpp_FLAGS} -D_GLIBBCXX_USE_CXX11_ABI=0")
endif()

if(NOT EXISTS "${MOCK_INSTALL_PATH}/lib/libmockcpp.a" OR FORCE_REBUILD_CANN_3RD)
    set(PATCH_FILE ${CANN_3RD_PKG_PATH}/mockcpp-2.7.patch)
    if(NOT EXISTS ${PATCH_FILE})
        message(STATUS "download mockcpp-2.7.patch")
        file(DOWNLOAD
            "https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h2/mockcpp-2.7_py3.patch"
            ${PATCH_FILE}
            TIMEOUT 60)
    endif()
    include(ExternalProject)
    message(STATUS "start to get mockcpp, install path: ${MOCK_INSTALL_PATH}")
    ExternalProject_Add(mockcpp
        URL                         https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h2/mockcpp-2.7.tar.gz
        DOWNLOAD_DIR                ${MOCK_DOWNLOAD_PATH}
        SOURCE_DIR                  ${MOCK_INSTALL_PATH}
        TLS_VERIFY                  OFF
        PATCH_COMMAND               git init && git apply ${PATCH_FILE}
        CONFIGURE_COMMAND           ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR}
                                    -DCMAKE_CXX_FLAGS=${mockcpp_CXXFLAGS}
                                    -DCMAKE_C_FLAGS=${mockcpp_FLAGS}
                                    -DBOOST_INCLUDE_DIRS=${BOOST_INCLUDE_DIRS}
                                    -DCMAKE_SHARED_LINKER_FLAGS=${mockcpp_LINKER_FLAGS}
                                    -DCMAKE_EXE_LINKER_FLAGS=${mockcpp_LINKER_FLAGS}
                                    -DBUILD_32_BIT_TARGET_BY_64_BIT_COMPILER=OFF
                                    -DCMAKE_INSTALL_PREFIX=${CANN_3RD_LIB_PATH}/mockcpp
                                    <SOURCE_DIR>
        BUILD_COMMAND               ${${BUILD_TYPE}} $<$<BOOL:${IS_MAKE}>:$(MAKE)>
    )
    message(STATUS "get mockcpp")
endif()

add_dependencies(gtest mockcpp)
