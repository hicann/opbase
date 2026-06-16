/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "device_sharder.h"

#include <dlfcn.h>
#include "log.h"

namespace {
const char* const kSharderPath = "libaicpu_sharder.so";
const char* const kParallelForFunc = "ParallelFor";
const char* const kGetCPUNumFunc = "GetCPUNum";
} // namespace

namespace aicpu {
DeviceSharder::DeviceSharder(DeviceType device) : Sharder(device)
{
    sharder_ = dlopen(kSharderPath, RTLD_LAZY | RTLD_GLOBAL);
    if (sharder_ == nullptr) {
        KERNEL_LOG_WARN("Device sharder dlopen so[%s] failed, error[%s]", kSharderPath, dlerror());
        parallel_for_ = nullptr;
        get_cpu_num_ = nullptr;
    } else {
        parallel_for_ = reinterpret_cast<ParallelForFunc>(dlsym(sharder_, kParallelForFunc));
        if (parallel_for_ == nullptr) {
            KERNEL_LOG_WARN("Get function[%s] address failed, error[%s]", kParallelForFunc, dlerror());
        }

        get_cpu_num_ = reinterpret_cast<GetCPUNumFunc>(dlsym(sharder_, kGetCPUNumFunc));
        if (get_cpu_num_ == nullptr) {
            KERNEL_LOG_WARN("Get function[%s] address failed, error[%s]", kGetCPUNumFunc, dlerror());
        }
        KERNEL_LOG_INFO("Device sharder dlopen so[%s] success", kSharderPath);
    }
}

DeviceSharder::~DeviceSharder()
{
    if (sharder_ != nullptr) {
        (void)dlclose(sharder_);
        sharder_ = nullptr;
    }
    parallel_for_ = nullptr;
}

/*
 * ParallelFor shards the "total" units of work.
 */
void DeviceSharder::ParallelFor(
    int64_t total, int64_t per_unit_size, const std::function<void(int64_t, int64_t)>& work) const
{
    if (parallel_for_ != nullptr) {
        parallel_for_(total, per_unit_size, work);
        return;
    }

    KERNEL_LOG_WARN("Function[%s] is null", kParallelForFunc);
    work(0, total);
}

/*
 * Get CPU number
 */
uint32_t DeviceSharder::GetCPUNum() const
{
    if (get_cpu_num_ != nullptr) {
        return get_cpu_num_();
    }

    KERNEL_LOG_WARN("Function[%s] is null", kGetCPUNumFunc);
    return 1;
}
} // namespace aicpu
