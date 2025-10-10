cmake_minimum_required(VERSION 3.14.1)
project(cpu_kernels_context)
set(proto_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_attr.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_node_def.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_tensor.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_tensor_shape.proto
)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/bcast.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/eigen_tensor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/kernel_util.cc
)

set(local_context_inc_path
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/aicpu_common/context/common
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/aicpu_common/context/cpu_proto
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/aicpu_common/context/utils

  # cce/aicpu_engine_struct.h, cce/fwk_adpt_struct.h
  ${ASCEND_HOME_PATH}/include/experiment
  # aicpu_task_struct.h
  ${ASCEND_HOME_PATH}/include/experiment/datagw/aicpu/common
  ${ASCEND_HOME_PATH}/include/experiment/datagw/aicpu/aicpu_schedule/aicpu_sharder

  ${C_SEC_INCLUDE}
)

set(nodedef_src_files
  ${CMAKE_CURRENT_SOURCE_DIR}/common/node_def_builder.cpp
)

add_subdirectory(host)
add_subdirectory(device)
