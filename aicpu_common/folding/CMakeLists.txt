cmake_minimum_required(VERSION 3.14.1)
project(base_constant_folding_ops)
set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
set(local_folding_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../pkg_inc/op_common/aicpu_common/context/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../pkg_inc/op_common/aicpu_common/folding
)

message(STATUS "folding PRODUCT_SIDE is ${PRODUCT_SIDE}")
  if("x${PRODUCT_SIDE}" STREQUAL "xdevice")
    message(STATUS "folding PRODUCT_SIDE is device")
    if (MINRC)
      set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
      set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
    else()
      set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-g++)
      set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc)
    endif()
  endif()
else()

set(local_folding_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../pkg_inc/op_common/aicpu_common/context/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../pkg_inc/op_common/aicpu_common/folding
)
endif()

set (local_folding_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/folding.cc
)

add_library(base_constant_folding_ops SHARED
    ${local_folding_src_files}
)

target_compile_options(base_constant_folding_ops PRIVATE
    -O2
    -ftrapv
    -std=c++11
    -fstack-protector-all
    -fvisibility-inlines-hidden
    -fvisibility=hidden
    -Werror
    -Wno-deprecated-declarations
    -fPIC
)

target_compile_definitions(base_constant_folding_ops PRIVATE
    _FORTIFY_SOURCE=2
    google=ascend_private
    LOG_CPP
)

target_link_options(base_constant_folding_ops PRIVATE
    -Wl,-z,relro,-z,now
    -Wl,-Bsymbolic
    $<$<NOT:$<STREQUAL:${RESERVED_SYMBOL},true>>:-s>
    -Wl,-z,noexecstack
)

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  target_link_libraries(base_constant_folding_ops PRIVATE
    $<BUILD_INTERFACE:intf_pub_aicpu>
    -Wl,--whole-archive
    aicpu_context
    -Wl,--no-whole-archive
  )
else()
  target_link_libraries(base_constant_folding_ops PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--whole-archive
    aicpu_context
  )
endif()

target_include_directories(base_constant_folding_ops PRIVATE
     ${local_folding_inc_path}
)
