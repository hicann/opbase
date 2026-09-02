/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file platform_util.h
 * \brief func get platform info
 */

#ifndef OP_COMMON_OP_HOST_UTIL_PLATFORM_UTIL_H
#define OP_COMMON_OP_HOST_UTIL_PLATFORM_UTIL_H

#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/log/log.h"

namespace Ops {
namespace Base {
constexpr ::NpuArch DAV_1001 = static_cast<::NpuArch>(1001);
constexpr ::NpuArch DAV_1002 = static_cast<::NpuArch>(1002);
constexpr ::NpuArch DAV_1003 = static_cast<::NpuArch>(1003);
constexpr ::NpuArch DAV_1004 = static_cast<::NpuArch>(1004);
constexpr ::NpuArch DAV_1999 = static_cast<::NpuArch>(1999);
constexpr ::NpuArch DAV_2002 = static_cast<::NpuArch>(2002);
constexpr ::NpuArch DAV_2003 = static_cast<::NpuArch>(2003);
constexpr ::NpuArch DAV_2004 = static_cast<::NpuArch>(2004);
constexpr ::NpuArch DAV_2006 = static_cast<::NpuArch>(2006);
constexpr ::NpuArch DAV_2102 = static_cast<::NpuArch>(2102);
constexpr ::NpuArch DAV_2103 = static_cast<::NpuArch>(2103);
constexpr ::NpuArch DAV_2104 = static_cast<::NpuArch>(2104);
constexpr ::NpuArch DAV_2201 = static_cast<::NpuArch>(2201);
constexpr ::NpuArch DAV_3002 = static_cast<::NpuArch>(3002);
constexpr ::NpuArch DAV_3003 = static_cast<::NpuArch>(3003);
constexpr ::NpuArch DAV_3004 = static_cast<::NpuArch>(3004);
constexpr ::NpuArch DAV_3102 = static_cast<::NpuArch>(3102);
constexpr ::NpuArch DAV_3103 = static_cast<::NpuArch>(3103);
constexpr ::NpuArch DAV_3113 = static_cast<::NpuArch>(3113);
constexpr ::NpuArch DAV_3502 = static_cast<::NpuArch>(3502);
constexpr ::NpuArch DAV_3505 = static_cast<::NpuArch>(3505);
constexpr ::NpuArch DAV_3510 = static_cast<::NpuArch>(3510);
constexpr ::NpuArch DAV_3801 = static_cast<::NpuArch>(3801);
constexpr ::NpuArch DAV_5101 = static_cast<::NpuArch>(5101);
constexpr ::NpuArch DAV_5102 = static_cast<::NpuArch>(5102);
constexpr ::NpuArch DAV_5161 = static_cast<::NpuArch>(5161);
constexpr ::NpuArch DAV_9201 = static_cast<::NpuArch>(9201);
constexpr ::NpuArch DAV_9202 = static_cast<::NpuArch>(9202);
constexpr ::NpuArch DAV_9301 = static_cast<::NpuArch>(9301);
constexpr ::NpuArch DAV_RESV = static_cast<::NpuArch>(0xFFFF);

template <typename T>
uint32_t GetAivCoreNum(const T* context)
{
    static_assert(std::is_same<typename std::remove_const<T>::type, gert::TilingParseContext>::value ||
                      std::is_same<typename std::remove_const<T>::type, gert::TilingContext>::value,
                  "context should be gert::TilingParseContext or gert::TilingContext");

    OP_CHECK_IF(context == nullptr, OP_LOGE("GetAivCoreNum", "context is nullptr"), return 0);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE("GetAivCoreNum", "platformInfoPtr is nullptr"), return 0);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    return ascendcPlatform.GetCoreNumAiv();
}

template <typename T>
uint32_t GetAicCoreNum(const T* context)
{
    static_assert(std::is_same<typename std::remove_const<T>::type, gert::TilingParseContext>::value ||
                      std::is_same<typename std::remove_const<T>::type, gert::TilingContext>::value,
                  "context should be gert::TilingParseContext or gert::TilingContext");

    OP_CHECK_IF(context == nullptr, OP_LOGE("GetAicCoreNum", "context is nullptr"), return 0);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE("GetAicCoreNum", "platformInfoPtr is nullptr"), return 0);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    return ascendcPlatform.GetCoreNumAic();
}

template <typename T>
uint32_t GetUbSize(const T* context)
{
    static_assert(std::is_same<typename std::remove_const<T>::type, gert::TilingParseContext>::value ||
                      std::is_same<typename std::remove_const<T>::type, gert::TilingContext>::value,
                  "context should be gert::TilingParseContext or gert::TilingContext");

    OP_CHECK_IF(context == nullptr, OP_LOGE("GetUbSize", "context is nullptr"), return 0);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE("GetUbSize", "platformInfoPtr is nullptr"), return 0);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    return ubSize;
}

/**
 * Get the block size of unified buffer in bytes
 */
template <typename T>
uint32_t GetUbBlockSize([[maybe_unused]] const T* context)
{
    return 32U; // will using AscendC api later
}

/**
 * Get the size of vector registers in bytes
 */
template <typename T>
uint32_t GetVRegSize([[maybe_unused]] const T* context)
{
    return 256U;
}

template <typename T>
uint32_t GetSimtMaxThreadNum([[maybe_unused]] const T* context)
{
    return 2048U;
}

/**
 * Get the maximum Dcache size used by simt in bytes: 128 * 1024B
 */
template <typename T>
uint32_t GetSimtMaxDCacheSize([[maybe_unused]] const T* context)
{
    return 131072U;
}

/**
 * Get the cache line size in bytes
 */
template <typename T>
uint32_t GetCacheLineSize([[maybe_unused]] const T* context)
{
    return 256U;
}

/**
 * Get the sector cache line size in bytes
 */
template <typename T>
uint32_t GetSectorCacheLineSize([[maybe_unused]] const T* context)
{
    return GetCacheLineSize(context) / 2U;
}

/**
 * Get the dcache size of nddma
 */
template <typename T>
uint32_t GetNddmaDcacheSize([[maybe_unused]] const T* context)
{
    return 8192U;
}

template <typename T>
uint32_t GetWorkspaceSize(const T* context)
{
    static_assert(std::is_same<typename std::remove_const<T>::type, gert::TilingParseContext>::value ||
                      std::is_same<typename std::remove_const<T>::type, gert::TilingContext>::value,
                  "context should be gert::TilingParseContext or gert::TilingContext");

    OP_CHECK_IF(context == nullptr, OP_LOGE("GetWorkspaceSize", "context is nullptr"), return 0);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE("GetWorkspaceSize", "platformInfoPtr is nullptr"), return 0);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    return ascendcPlatform.GetLibApiWorkSpaceSize();
}
} // namespace Base
} // namespace Ops
#endif // OP_COMMON_OP_HOST_UTIL_PLATFORM_UTIL_H
