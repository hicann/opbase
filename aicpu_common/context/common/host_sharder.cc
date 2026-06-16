/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "host_sharder.h"

#include "eigen_threadpool.h"
#include "log.h"

namespace aicpu {
/*
 * ParallelFor shards the "total" units of work.
 */
void HostSharder::ParallelFor(
    int64_t total, int64_t per_unit_size, const std::function<void(int64_t, int64_t)>& work) const
{
    EigenThreadPool* threadpool = EigenThreadPool::GetInstance();
    if (threadpool == nullptr) {
        KERNEL_LOG_ERROR("Get eigen thread pool failed");
        return;
    }

    threadpool->ParallelFor(total, per_unit_size, work);
}

/*
 * Get CPU number
 */
uint32_t HostSharder::GetCPUNum() const
{
    EigenThreadPool* threadpool = EigenThreadPool::GetInstance();
    if (threadpool == nullptr) {
        KERNEL_LOG_ERROR("Get eigen thread pool failed");
        return 0;
    }

    return threadpool->GetCPUNum();
}
} // namespace aicpu
