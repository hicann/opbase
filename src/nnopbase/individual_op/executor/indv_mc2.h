/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDV_MC2_H_
#define INDV_MC2_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl_rt.h"
#include "hccl/hccl_types.h"
#include "individual_op_api.h"
#include "runtime/runtime/kernel.h"
#include "utils/indv_types.h"

static constexpr size_t NNOPBASE_AICPU_PARAM_LEN = 32U;
static const std::string NNOPBASE_MC2_AICPU_SUFFIX = "Mc2AicpuKernel";
static constexpr NnopbaseUChar NNOPBASE_MC2_AICPU_SO_NAME[NNOPBASE_AICPU_PARAM_LEN] = {"libccl_kernel.so"};
static constexpr NnopbaseUChar NNOPBASE_MC2_AICPU_KERNEL_NAME[NNOPBASE_AICPU_PARAM_LEN] = {"RunAicpuKfcSrvLaunch"};
static constexpr NnopbaseUChar NNOPBASE_MC2_SERVER_SO_NAME[NNOPBASE_AICPU_PARAM_LEN] = {"libmc2_server.so"};
static constexpr NnopbaseUChar NNOPBASE_MC2_SERVER_KERNEL_NAME[NNOPBASE_AICPU_PARAM_LEN] = {"Mc2ServerKernel"};
static constexpr uint8_t NNOPBASE_MC2_NOTIFY_COUNT = 2U;
static constexpr uint16_t NNOPBASE_HCCL_DEFAULT_TIME = 1836U;

struct NnopbaseMc2Execution {
    bool enabled = false;
    bool fallback = false;
    NnopbaseHcclServerType serverType = NNOPBASE_HCCL_SERVER_TYPE_END;
    std::vector<HcclComm> commHandles;
    std::vector<void*> contextAddrs;
    rtAicpuArgsEx_t aicpuArgs{};
    rtFusionArgsEx_t fusionArgs{};
    std::vector<aclrtStream> aicpuStreams;
    std::vector<uint64_t> aicpuThreads;
    std::vector<std::pair<aclrtStream, aclrtStream>> aicpuNotifies;

    void ClearRuntimeState()
    {
        enabled = false;
        fallback = false;
        serverType = NNOPBASE_HCCL_SERVER_TYPE_END;
        commHandles.clear();
        contextAddrs.clear();
        aicpuStreams.clear();
        aicpuThreads.clear();
        aicpuNotifies.clear();
    }

    void Reset()
    {
        ClearRuntimeState();
        aicpuArgs = {};
        fusionArgs = {};
    }
};

#endif // INDV_MC2_H_
