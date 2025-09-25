# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

option(ENABLE_DEBUG "Enable debug" OFF)
set(BUILD_MODE "" CACHE STRING "build mode -O0/O1/O2/O3")
if("${BUILD_MODE}" STREQUAL "-O0" OR "${BUILD_MODE}" STREQUAL "-O1" OR
   "${BUILD_MODE}" STREQUAL "-O2" OR "${BUILD_MODE}" STREQUAL "-O3"
  )
  set(COMPILE_OP_MODE ${BUILD_MODE})
else()
  if(ENABLE_UT)
    set(COMPILE_OP_MODE "-O0")
  else()
    set(COMPILE_OP_MODE "-O2")
  endif()
endif()
message(STATUS "ADD COMPILE_OPTIONS: ${COMPILE_OP_MODE}")
add_compile_options(${COMPILE_OP_MODE})

if(ENABLE_DEBUG OR ENABLE_UT)
    add_compile_options(-g)
    message(STATUS "ADD COMPILE_OPTIONS: -g")
endif()

add_compile_options(
    -Wall -Wundef -Wcast-qual -Wpointer-arith -Wdate-time -fno-strict-aliasing
    -fno-common -fno-builtin -fno-omit-frame-pointer -fsigned-char -Wfloat-equal -Wformat=2 -Wshadow
    -Wsign-compare -Wunused-macros -Wvla -Wdisabled-optimization -Wempty-body -Wignored-qualifiers
    -Wtype-limits -Wshift-negative-value -Wswitch-default
    -Wframe-larger-than=8192 -Woverloaded-virtual -fno-sized-deallocation
    -Wnon-virtual-dtor
    -Wwrite-strings -Wmissing-format-attribute -Wformat-nonliteral
    -Wdelete-non-virtual-dtor
    -Wmissing-include-dirs -Wmissing-declarations -Wextra
    -Wredundant-decls -Wfloat-conversion
    -Wno-write-strings -Wall -Wno-dangling-else -Wno-comment -Wno-conversion-null -Wno-return-type
    -Wno-unknown-pragmas -Wno-sign-compare
    -Wno-unused-const-variable
)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Using LLVM Clang compiler, add more compile options")
    add_compile_options(-Wno-tautological-unsigned-enum-zero-compare -Wno-unused-private-field)
else()
    message(STATUS "Using GNU compiler default, add more compile options")
    add_compile_options(-Wimplicit-fallthrough=3 -Wshift-overflow=2 -Wduplicated-cond -Wtrampolines
    -Wsized-deallocation -Wlogical-op -Wsuggest-attribute=format
    -Wduplicated-branches -Wformat-signedness -Wreturn-local-addr)
endif()

if (ENABLE_UT)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/gtest.cmake)
endif ()
