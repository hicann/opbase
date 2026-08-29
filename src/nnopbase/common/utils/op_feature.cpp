/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_feature_internal.h"

#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "acl/acl_rt.h"
#include "platform/soc_spec.h"
#ifndef PRODUCT_SIDE_IS_DEVICE
#include "version/runtime_version.h"
#endif

#include "aclnn/acl_meta.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"
#include "opdev/op_errno.h"

#define PKG_VERSION_NUM_9_2_0 90200000

namespace op {
namespace internal {

namespace {
struct PcieAddrRange {
    void* startAddr{nullptr};
    void* endAddr{nullptr};
};

bool g_pcieThroughEnabled{false};
std::vector<PcieAddrRange> g_pcieAddrRanges;
std::once_flag g_pcieThroughOnceFlag;

extern "C" {
__attribute__((weak)) aclError aclrtHostGetDevicePointerAddrRange(PcieAddrRange* addrRange, uint32_t* count);
}

// PCIe through 环境变量总开关名
constexpr const char* PCIE_THROUGH_ENV_NAME = "OP_PCIE_THROUGH_ACCESS_HOST_MEM_CHECK_ENABLE";

bool IsPcieThroughEnvEnabled()
{
    const char* envVal = std::getenv(PCIE_THROUGH_ENV_NAME);
    return (envVal != nullptr) && (std::strcmp(envVal, "1") == 0);
}

bool IsHostDeviceConnectWithPcie([[maybe_unused]] uint32_t deviceId)
{
#if !defined(PRODUCT_SIDE_IS_DEVICE) && defined(RUNTIME_VERSION_NUM) && (RUNTIME_VERSION_NUM >= PKG_VERSION_NUM_9_2_0)
    int64_t hdConnectType = ACL_HOST_DEVICE_CONNECT_TYPE_UB;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_HD_CONNECT_TYPE, &hdConnectType);
    if (ret != ACL_SUCCESS) {
        OP_LOGW("Get HD_CONNECT_TYPE by aclrtGetDeviceInfo failed, return %d", static_cast<int32_t>(ret));
        return false;
    }
    if (hdConnectType != ACL_HOST_DEVICE_CONNECT_TYPE_PCIE) {
        OP_LOGI("HD connect type is %" PRId64 ", not PCIe", static_cast<int64_t>(hdConnectType));
        return false;
    }
    return true;
#else
    OP_LOGW("Can't get host device connect type, because the runtime package version is lower than 9.2.0 when "
            "compiling nnopbase");
    return false;
#endif
}

aclnnStatus GetPcieAddrRange([[maybe_unused]] std::vector<PcieAddrRange>& addrRanges)
{
#if !defined(PRODUCT_SIDE_IS_DEVICE) && defined(RUNTIME_VERSION_NUM) && (RUNTIME_VERSION_NUM >= PKG_VERSION_NUM_9_2_0)
    addrRanges.clear();
    if (aclrtHostGetDevicePointerAddrRange == nullptr) {
        OP_LOGW("aclrtHostGetDevicePointerAddrRange symbol is not found");
        return ACLNN_ERR_RUNTIME_ERROR;
    }

    uint32_t rangeCount = 0;
    auto ret = aclrtHostGetDevicePointerAddrRange(nullptr, &rangeCount);
    if (ret != ACL_SUCCESS) {
        OP_LOGW("aclrtHostGetDevicePointerAddrRange failed to get range count, return %d", static_cast<int32_t>(ret));
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    if (rangeCount == 0U) {
        OP_LOGI("PCIe addr range count is 0");
        return ACLNN_SUCCESS;
    }

    addrRanges.resize(rangeCount);
    ret = aclrtHostGetDevicePointerAddrRange(addrRanges.data(), &rangeCount);
    if (ret != ACL_SUCCESS) {
        OP_LOGW("aclrtHostGetDevicePointerAddrRange failed to get addr ranges, return %d", static_cast<int32_t>(ret));
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
#else
    OP_LOGW(
        "Can't get PCIe addr ranges, because the runtime package version is lower than 9.2.0 when compiling nnopbase");
    return ACLNN_ERR_RUNTIME_ERROR;
#endif
}

} // namespace

aclnnStatus InitPcieThroughInfo()
{
    static aclnnStatus initRet = ACLNN_SUCCESS;
    std::call_once(g_pcieThroughOnceFlag, []() {
        g_pcieThroughEnabled = false;
        NpuArch npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
        if (npuArch != NpuArch::DAV_3510 && npuArch != NpuArch::DAV_9201 && npuArch != NpuArch::DAV_9202) {
            OP_LOGI("Current NPU arch [%u] not support PCIe through.", static_cast<uint32_t>(npuArch));
            return;
        }

        int32_t deviceId = 0;
        auto aclRet = aclrtGetDevice(&deviceId);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "aclrtGetDevice failed, return %d", static_cast<int32_t>(aclRet));
            initRet = ACLNN_ERR_RUNTIME_ERROR;
            return;
        }

        if (!IsHostDeviceConnectWithPcie(static_cast<uint32_t>(deviceId))) {
            OP_LOGI("The connect type between host and device cannot be obtained, or the type is not PCIe, so disable "
                    "PCIe through feature.");
            return;
        }

        if (!IsPcieThroughEnvEnabled()) {
            OP_LOGI("PCIe through env switch [%s] is not set to 1, disable PCIe through feature.",
                    PCIE_THROUGH_ENV_NAME);
            return;
        }

        if (GetPcieAddrRange(g_pcieAddrRanges) != ACLNN_SUCCESS || g_pcieAddrRanges.empty()) {
            OP_LOGI("Can't get PCIe addr ranges, or the range count is 0, so disable PCIe through feature.");
            return;
        }

        g_pcieThroughEnabled = true;
        OP_LOGI("PCIe through feature is enabled, addr range count is %zu", g_pcieAddrRanges.size());
        for (size_t i = 0; i < g_pcieAddrRanges.size(); i++) {
            OP_LOGI("PCIe addr range[%zu], startAddr %p, endAddr %p", i, g_pcieAddrRanges[i].startAddr,
                    g_pcieAddrRanges[i].endAddr);
        }
    });
    return initRet;
}

bool IsPcieThroughEnabled() { return g_pcieThroughEnabled; }

bool IsTensorAddrInPcieRange(const void* const addr)
{
    uintptr_t addrVal = reinterpret_cast<uintptr_t>(addr);              // NOLINT
    for (const auto& range : g_pcieAddrRanges) {
        uintptr_t start = reinterpret_cast<uintptr_t>(range.startAddr); // NOLINT
        uintptr_t end = reinterpret_cast<uintptr_t>(range.endAddr);     // NOLINT
        if (addrVal >= start && addrVal <= end) {
            OP_LOGI("Tensor addr %p is in PCIe range, startAddr %p, endAddr %p", addr, range.startAddr, range.endAddr);
            return true;
        }
    }
    return false;
}

} // namespace internal
} // namespace op
