# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 旧 include 路径兼容头弃用机制（供 cmake/package.cmake 调用）。
#
# 背景：7 个 aicpu 公共头（context/common: cpu_context/cpu_kernel/cpu_kernel_register/cpu_types；
# context/cpu_proto: cpu_attr_value/cpu_tensor/cpu_tensor_shape）已由 aicpu_headers_src 规则
# 扁平安装到 include/aicpu/，旧嵌套路径 pkg_inc/aicpu_common/context/** 进入弃用过渡期。
#
# 机制：旧路径安装"源内容 + 弃用提示横幅"的生成副本——剥离横幅后与源文件逐字节一致，
# include 解析行为零变化，仅多一条编译期提示（#pragma message 为 note 级诊断，不被
# -Werror 升级为错误：ops-cv/ops-math 等 release 构建在非 Clang 场景启用 -Werror，且
# AICPU target 仍经旧路径裸 include 这些头，#warning 会被升级为编译错误打断下游构建）。
# 横幅插在许可头（首个 */）之后，版权头仍在文件首部；每文件独立 WARN_SHOWN 守卫宏，
# 每编译单元每头仅提示一次。新扁平路径安装干净副本，不受影响。
#
# 到期后（remove_on 日期）：删除本文件、调用点一行、及 scripts/package/opbase/opbase.xml
# 中旧路径 file_info 条目，并在发布说明中公告。

# 生成单个兼容头副本。
#   generate_legacy_compat_header(<src> <gen> <name> <since> <remove_on>)
function(generate_legacy_compat_header src gen name since remove_on)
    file(READ "${src}" _body)
    string(REPLACE "." "_" _guard "${name}")
    string(FIND "${_body}" "*/" _lic_end)
    if(_lic_end GREATER -1)
        math(EXPR _lic_end "${_lic_end} + 2")
        string(SUBSTRING "${_body}" 0 ${_lic_end} _head)
        string(SUBSTRING "${_body}" ${_lic_end} -1 _tail)
    else()
        set(_head "")
        set(_tail "${_body}")
    endif()
    get_filename_component(_gen_dir "${gen}" DIRECTORY)
    file(MAKE_DIRECTORY "${_gen_dir}")
    file(WRITE "${gen}" "${_head}
/* Legacy include path compatibility copy, generated at build time, do not edit. */
#ifndef AICPU_COMMON_LEGACY_${_guard}_WARN_SHOWN
#define AICPU_COMMON_LEGACY_${_guard}_WARN_SHOWN
#pragma message(\"${name} via legacy include path pkg_inc/aicpu_common/context is deprecated since ${since} and will be removed on ${remove_on}. Migrate to: add <CANN>/<arch>-linux/include to the include path and use #include \\\"aicpu/${name}\\\".\")
#endif${_tail}")
endfunction()

# 旧嵌套路径整体安装（含弃用处理）：整目录安装排除 7 个公共头，改为安装带告警横幅的生成副本。
#   install_aicpu_common_legacy_compat(<src_dir> <since> <remove_on>)
#     src_dir   aicpu_common 公共头源目录（include/op_common/aicpu_common）
#     since     弃用起始日期（YYYY-MM-DD）
#     remove_on 移除期限日期（YYYY-MM-DD，建议为起始日 + 1 年兼容期）
function(install_aicpu_common_legacy_compat src_dir since remove_on)
    install(DIRECTORY ${src_dir}/
        DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/aicpu_common
        COMPONENT opbase
        FILE_PERMISSIONS
        OWNER_READ OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        PATTERN "cpu_attr_value.h" EXCLUDE
        PATTERN "cpu_tensor.h" EXCLUDE
        PATTERN "cpu_tensor_shape.h" EXCLUDE
        PATTERN "cpu_context.h" EXCLUDE
        PATTERN "cpu_types.h" EXCLUDE
        PATTERN "cpu_kernel.h" EXCLUDE
        PATTERN "cpu_kernel_register.h" EXCLUDE
    )
    set(_compat_common "")
    set(_compat_cpu_proto "")
    foreach(_hdr
        context/common/cpu_context.h
        context/common/cpu_kernel.h
        context/common/cpu_kernel_register.h
        context/common/cpu_types.h
        context/cpu_proto/cpu_attr_value.h
        context/cpu_proto/cpu_tensor.h
        context/cpu_proto/cpu_tensor_shape.h)
        get_filename_component(_name "${_hdr}" NAME)
        get_filename_component(_dir "${_hdr}" DIRECTORY)
        set(_gen "${CMAKE_CURRENT_BINARY_DIR}/aicpu_common_compat/${_hdr}")
        generate_legacy_compat_header("${src_dir}/${_hdr}" "${_gen}" "${_name}" "${since}" "${remove_on}")
        if(_dir STREQUAL "context/common")
            list(APPEND _compat_common "${_gen}")
        else()
            list(APPEND _compat_cpu_proto "${_gen}")
        endif()
    endforeach()
    install(FILES ${_compat_common}
        DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/aicpu_common/context/common
        COMPONENT opbase
        PERMISSIONS
        OWNER_READ OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
    )
    install(FILES ${_compat_cpu_proto}
        DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/aicpu_common/context/cpu_proto
        COMPONENT opbase
        PERMISSIONS
        OWNER_READ OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
    )
endfunction()
