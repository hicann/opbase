/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "eigen_threadpool_embedding.h"

#include <sys/sysinfo.h>

#include "log.h"
#include "mmpa/mmpa_api.h"

namespace {
const uint32_t kTaskSize = 40000;
const uint32_t kMaxOverShardingFactor = 4;
const int32_t kHalfCoresNum = 2;
const int32_t kPerformanceMulitCoresNum = 48;
const int32_t kPerformanceMulitCoresNumForLargeCores = 72;
const uint32_t kDecimalScaleNum = 10;
const uint32_t kTotalCostFactor = 210000;
constexpr uint32_t kMaxTaskSize = kTaskSize * kMaxOverShardingFactor;
} // namespace

namespace aicpu {
std::mutex EigenThreadPoolEmbedding::mutex_;
bool EigenThreadPoolEmbedding::init_flag_(false);
int32_t EigenThreadPoolEmbedding::core_num_(0);
std::unique_ptr<Eigen::ThreadPool> EigenThreadPoolEmbedding::eigen_threadpool_(nullptr);
std::unique_ptr<Eigen::ThreadPoolDevice> EigenThreadPoolEmbedding::threadpool_device_(nullptr);

EigenThreadPoolEmbedding* EigenThreadPoolEmbedding::GetInstance()
{
    if (!init_flag_) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!init_flag_) {
            core_num_ = get_nprocs() / kHalfCoresNum;
            core_num_ = (core_num_ > kPerformanceMulitCoresNumForLargeCores) ? kPerformanceMulitCoresNumForLargeCores :
                                                                               core_num_;
            const char* core_num_value = nullptr;
            MM_SYS_GET_ENV(MM_ENV_EMBEDDING_MAX_THREAD_CORE_NUMBER, core_num_value);
            if ((core_num_value != nullptr) && (core_num_value[0U] != '\0')) {
                try {
                    core_num_ = std::strtol(&(core_num_value[0U]), nullptr, kDecimalScaleNum);
                } catch (...) {
                    KERNEL_LOG_WARN("embedding thread can not find core num env, use default value.");
                    core_num_ = kPerformanceMulitCoresNum;
                }
            }
            if (core_num_ <= 0 || core_num_ > get_nprocs()) {
                // obtains the number of CPU cores that can be used by embedding users
                core_num_ = kPerformanceMulitCoresNum;
            }

            eigen_threadpool_.reset(new Eigen::ThreadPool(core_num_));
            threadpool_device_.reset(new Eigen::ThreadPoolDevice(eigen_threadpool_.get(), core_num_));
            init_flag_ = true;
            KERNEL_LOG_INFO("Init embedding thread pool success, core num[%d]", core_num_);
        }
    }
    static EigenThreadPoolEmbedding instance;
    return &instance;
}

void EigenThreadPoolEmbedding::ParallelFor(int64_t total, int64_t per_unit_size, const SharderWork& work) const
{
    double per_unit_cost = 1.0;
    if ((total <= 0) || (work == nullptr) || (per_unit_size <= 0)) {
        KERNEL_LOG_WARN(
            "Invalid param: total[%ld] <= 0 or per_unit_size[%ld] <= 0 or work is nullptr", total, per_unit_size);
        return;
    }
    if ((per_unit_size) <= (total / core_num_)) {
        // run tasks with the maximum number of threads, maximum =
        // kMaxOverShardingFactor * core_num_
        per_unit_cost = (1.0 * kMaxTaskSize * core_num_ / total) > (1.0 * kTotalCostFactor / total) ?
                            (1.0 * kMaxTaskSize * core_num_ / total) :
                            (1.0 * kTotalCostFactor / total);
    } else {
        // the task is fragmented based on the number of data slices.
        per_unit_cost = 1.0 * kMaxTaskSize / per_unit_size;
    }

    threadpool_device_->parallelFor(
        total, Eigen::TensorOpCost(0, 0, per_unit_cost),
        [&work](Eigen::Index first, Eigen::Index last) { work(first, last); });
    KERNEL_LOG_INFO("Eigen threadpool parallel for success.");
}
} // namespace aicpu
