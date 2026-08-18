cmake_minimum_required(VERSION 3.14.1)
project(cpu_kernels_context)
set(proto_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_attr.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_node_def.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_tensor.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_tensor_shape.proto
)

if(CMAKE_GENERATOR STREQUAL "Unix Makefiles")
set(CMAKE_CXX_COMPILE_OBJECT
    "<CMAKE_CXX_COMPILER> <DEFINES> -D__FILE__='\"$(notdir $(abspath <SOURCE>))\"' -Wno-builtin-macro-redefined <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>"
)
set(CMAKE_C_COMPILE_OBJECT
    "<CMAKE_C_COMPILER> <DEFINES> -D__FILE__='\"$(notdir $(abspath <SOURCE>))\"' -Wno-builtin-macro-redefined <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>"
)
endif()

set (local_context_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/node_def.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/node_def_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor_shape.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor_shape_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/attr_value.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/attr_value_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/device.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/context.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/device_cpu_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/cpu_kernel_register.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/cpu_kernel_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/host_sharder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/device_sharder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/eigen_threadpool.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/eigen_threadpool_embedding.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/cpu_kernel_cache.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/async_event_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/async_cpu_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cust_op/cust_op_log.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cust_op/cust_dlog_record.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/bcast.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/eigen_tensor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/kernel_util.cc
)

set(nodedef_src_files
  ${CMAKE_CURRENT_SOURCE_DIR}/common/node_def_builder.cpp
)

# aicpu_common 头文件分两处存放：开源算子仓使用的在 include/op_common/aicpu_common/context 下，
# 仅 opbase 内部使用的与对应 .cc 同级放在本目录下。
set(AICPU_PUBLIC_INC ${CMAKE_CURRENT_SOURCE_DIR}/../../include/op_common/aicpu_common/context)
add_library(aicpu_context_headers INTERFACE)
target_include_directories(aicpu_context_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto
  ${CMAKE_CURRENT_SOURCE_DIR}/utils
  ${CMAKE_CURRENT_SOURCE_DIR}/cust_op
  ${AICPU_PUBLIC_INC}/common
  ${AICPU_PUBLIC_INC}/cpu_proto
  ${AICPU_PUBLIC_INC}/utils
  ${AICPU_PUBLIC_INC}/cust_op
)

if(PRODUCT_SIDE STREQUAL "device")
  add_subdirectory(device)
else()
  add_subdirectory(host)
endif()
