cmake_minimum_required(VERSION 3.14.1)
project(cpu_kernels_context_device)

protobuf_generate(aicpu_proto_device PROTO_SRCS PROTO_HDRS ${proto_src_files} TARGET)
set(aicpu_proto_include "${CMAKE_BINARY_DIR}/proto/aicpu_proto_device")

add_library(aicpu_context STATIC
  ${local_context_src_files}
  ${PROTO_SRCS}
)
add_dependencies(aicpu_context aicpu_proto_device)

target_include_directories(aicpu_context PRIVATE
  ${local_context_inc_path}
  ${aicpu_proto_include}
)

target_compile_definitions(aicpu_context PRIVATE
  _FORTIFY_SOURCE=2
  google=ascend_private
)

target_compile_options(aicpu_context PRIVATE
  -O2
  -ftrapv
  -fstack-protector-all
  -fPIC
  -Werror=sign-compare
  -Werror=unused-function
  -Werror=unused-variable
  -Wno-address
  -fvisibility-inlines-hidden
  -fvisibility=hidden
  -DEIGEN_MPL2_ONLY
  -DEigen=ascend_Eigen
)

target_link_libraries(aicpu_context PRIVATE
  $<BUILD_INTERFACE:intf_pub_aicpu>
  Eigen3::Eigen
  ascend_protobuf_static
  -Wl,-z,relro,-z,now
  $<$<NOT:$<STREQUAL:${RESERVED_SYMBOL},true>>:-s>
  -ldl
  -shared
)

add_library(aicpu_nodedef STATIC
  ${nodedef_src_files}
)

add_dependencies(aicpu_nodedef aicpu_context)

target_include_directories(aicpu_nodedef PRIVATE
    ${local_context_inc_path}
    ${aicpu_proto_include}
)

target_compile_definitions(aicpu_nodedef PRIVATE
  _FORTIFY_SOURCE=2
  google=ascend_private
)

target_compile_options(aicpu_nodedef PRIVATE
  -O2
  -std=c++17
  -ftrapv
  -fstack-protector-all
  -fPIC
  -Werror
  -DEIGEN_MPL2_ONLY
  -DEigen=ascend_Eigen
)

message(STATUS "PRODUCT_SIDE is ${PRODUCT_SIDE}")
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-g++)
set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc)