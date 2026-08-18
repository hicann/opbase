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
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")

set(script_prefix ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/opbase/scripts)
install(DIRECTORY ${script_prefix}/
    DESTINATION share/info/opbase/script
    COMPONENT opbase
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    DIRECTORY_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    REGEX "(setenv|prereq_check)\\.(bash|fish|csh)" EXCLUDE
)
set(SCRIPTS_FILES
    ${CANN_CMAKE_DIR}/scripts/install/check_version_required.awk
    ${CANN_CMAKE_DIR}/scripts/install/common_func.inc
    ${CANN_CMAKE_DIR}/scripts/install/common_interface.sh
    ${CANN_CMAKE_DIR}/scripts/install/common_interface.csh
    ${CANN_CMAKE_DIR}/scripts/install/common_interface.fish
    ${CANN_CMAKE_DIR}/scripts/install/version_compatiable.inc
)

install(FILES ${SCRIPTS_FILES}
    DESTINATION share/info/opbase/script
    COMPONENT opbase
)
set(COMMON_FILES
    ${CANN_CMAKE_DIR}/scripts/install/install_common_parser.sh
    ${CANN_CMAKE_DIR}/scripts/install/common_func_v2.inc
    ${CANN_CMAKE_DIR}/scripts/install/common_installer.inc
    ${CANN_CMAKE_DIR}/scripts/install/script_operator.inc
    ${CANN_CMAKE_DIR}/scripts/install/version_cfg.inc
)

set(PACKAGE_FILES
    ${COMMON_FILES}
    ${CANN_CMAKE_DIR}/scripts/install/multi_version.inc
)
set(LATEST_MANGER_FILES
    ${COMMON_FILES}
    ${CANN_CMAKE_DIR}/scripts/install/common_func.inc
    ${CANN_CMAKE_DIR}/scripts/install/version_compatiable.inc
    ${CANN_CMAKE_DIR}/scripts/install/check_version_required.awk
)
set(CONF_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nnopbase/common/op_info_record/dump_tool_config.ini
)
install(FILES ${CMAKE_BINARY_DIR}/version.opbase.info
    DESTINATION share/info/opbase
    RENAME version.info
    COMPONENT opbase
)
install(FILES ${CONF_FILES}
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/conf
    COMPONENT opbase
)
install(FILES ${PACKAGE_FILES}
    DESTINATION share/info/opbase/script
    COMPONENT opbase
)

set(pkg_inc_src ${CMAKE_CURRENT_SOURCE_DIR}/pkg_inc)
install(DIRECTORY ${pkg_inc_src}/
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc
    COMPONENT opbase
    FILE_PERMISSIONS
    OWNER_READ OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
)
# aicpu_common 头文件源码已拆为两处：开源算子仓使用的在 include/op_common/aicpu_common 下，
# 仅 opbase 内部使用的在 aicpu_common/ 下。装包时合并回同一目标路径，对下游保持不变。
set(aicpu_common_public ${CMAKE_CURRENT_SOURCE_DIR}/include/op_common/aicpu_common)
install(DIRECTORY ${aicpu_common_public}/
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/aicpu_common
    COMPONENT opbase
    FILE_PERMISSIONS
    OWNER_READ OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
)
# 私有头逐个列举，不能整目录安装：aicpu_common/ 下还有 .cc/.cpp/.proto/CMakeLists.txt，
# 以及不属于对外交付集、原本就不打包的 cust_op/cust_dlog_record.h。
set(aicpu_private_common_headers
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/common/async_cpu_kernel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/common/async_event_util.h
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/common/cpu_kernel_cache.h
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/common/device.h
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/common/device_sharder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/common/eigen_threadpool.h
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/common/eigen_threadpool_embedding.h
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/common/host_sharder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/common/kernel_cache.h
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/common/notification.h
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/common/session_cache.h
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/common/sharder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/common/thread_ctx.h
)
set(aicpu_private_cpu_proto_headers
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/cpu_proto/attr_value_impl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/cpu_proto/node_def_impl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/cpu_proto/tensor_impl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_common/context/cpu_proto/tensor_shape_impl.h
)
install(FILES ${aicpu_private_common_headers}
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/aicpu_common/context/common
    COMPONENT opbase
    PERMISSIONS
    OWNER_READ OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
)
install(FILES ${aicpu_private_cpu_proto_headers}
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/aicpu_common/context/cpu_proto
    COMPONENT opbase
    PERMISSIONS
    OWNER_READ OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
)
install(TARGETS ops_base
    LIBRARY DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
    COMPONENT opbase
)
set(AICPU_LIBS
    aicpu_context_host
    aicpu_nodedef_host
)

install(TARGETS ${AICPU_LIBS}
    ARCHIVE DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
    COMPONENT opbase
)

install(FILES $<TARGET_FILE:ascend_protobuf_static>
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
    RENAME libhost_ascend_protobuf.a
    COMPONENT opbase
)

set(aicpu_headers_src
    ${CMAKE_CURRENT_SOURCE_DIR}/include/op_common/aicpu_common/context/cpu_proto/cpu_attr_value.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/op_common/aicpu_common/context/cpu_proto/cpu_tensor_shape.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/op_common/aicpu_common/context/cpu_proto/cpu_tensor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/op_common/aicpu_common/context/common/cpu_context.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/op_common/aicpu_common/context/common/cpu_types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/op_common/aicpu_common/context/common/cpu_kernel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/op_common/aicpu_common/context/common/cpu_kernel_register.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/op_common/aicpu_common/context/cust_op/cust_cpu_utils.h
    )
install(FILES ${aicpu_headers_src}
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include/aicpu
    COMPONENT opbase
)

message(STATUS "ASCEND_HOME_PATH: $ENV{ASCEND_HOME_PATH}")
get_filename_component(COMPILER_PATH $ENV{ASCEND_HOME_PATH}/compiler REALPATH)
get_filename_component(VER_PATH "${COMPILER_PATH}" DIRECTORY)
message(STATUS "VERSION PATH: ${VER_PATH}")

set(aclnn_source ${CMAKE_CURRENT_SOURCE_DIR}/include/nnopbase/aclnn)
install(DIRECTORY ${aclnn_source}/
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnn
    COMPONENT opbase
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE
    GROUP_READ GROUP_EXECUTE
)
set(opdev_source ${CMAKE_CURRENT_SOURCE_DIR}/include/nnopbase/opdev)
install(DIRECTORY ${opdev_source}/
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnn/opdev
    COMPONENT opbase
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE
    GROUP_READ GROUP_EXECUTE
)

set(aclnnop_source ${CMAKE_CURRENT_SOURCE_DIR}/include/aclnnop)
install(DIRECTORY ${aclnnop_source}/
        DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnnop
        COMPONENT opbase
        FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE
        GROUP_READ GROUP_EXECUTE
)
install(DIRECTORY ${aclnnop_source}/
        DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnnop/level2
        COMPONENT opbase
        FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE
        GROUP_READ GROUP_EXECUTE
)

set(op_common_source ${CMAKE_CURRENT_SOURCE_DIR}/include/op_common)
# aicpu_common 已由上面的规则装到 pkg_inc/aicpu_common，此处必须排除，
# 否则会在 include/op_common 下再产生一份冗余副本。
install(DIRECTORY ${op_common_source}/
        DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include/op_common
        COMPONENT opbase
        FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE
        GROUP_READ GROUP_EXECUTE
        PATTERN "aicpu_common" EXCLUDE
)

install(CODE "
set(level2_dest \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnnop/level2\")
file(GLOB level2_headers \"\${level2_dest}/*.h\")
foreach(h \${level2_headers})
  execute_process(
    COMMAND python3 \"${PROJECT_SOURCE_DIR}/scripts/package/common/py/utils/add_deprecation_warning.py\" \"\${h}\"
  )
endforeach()
"
  COMPONENT opbase
)

install(TARGETS nnopbase
   LIBRARY DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
   COMPONENT opbase
)

install(TARGETS dummy_tls
   LIBRARY DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
   COMPONENT opbase
)

install(TARGETS stub_nnopbase
    LIBRARY DESTINATION ${ARCH}-linux/devlib/linux/${ARCH}
    COMPONENT opbase
)

install(FILES ${CMAKE_BINARY_DIR}/version.opbase.info
    DESTINATION opp
    RENAME version.info
    COMPONENT opbase
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/opbase/opp_builtin/__init__.py
    DESTINATION opp/built-in/op_impl/ai_core/tbe/impl
    COMPONENT opbase
)

set(fusion_config_src ${CMAKE_CURRENT_SOURCE_DIR}/scripts/fusion_config)
install(FILES
        ${fusion_config_src}/fusion_pass/config/fusion_config.json
        ${fusion_config_src}/fusion_pass/config/support_fusion_pass.json
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64/plugin/opskernel/fusion_pass/config
    COMPONENT opbase
)
install(FILES ${fusion_config_src}/fusion_rules/ai_core/built_in_graph_rules.json
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64/plugin/opskernel/fusion_rules/ai_core
    COMPONENT opbase
)
install(FILES ${fusion_config_src}/fusion_rules/vector_core/built_in_graph_rules.json
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64/plugin/opskernel/fusion_rules/vector_core
    COMPONENT opbase
)

set_cann_cpack_config(opbase SHARE_INFO_NAME opbase ENABLE_DEVICE "${ENABLE_BUILD_DEVICE}" PACKAGE_TYPE "${PACKAGE_TYPE}")
