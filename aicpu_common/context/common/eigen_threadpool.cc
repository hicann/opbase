/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "eigen_threadpool.h"

#include <sys/sysinfo.h>

#include "mmpa/mmpa_api.h"
#include "log.h"

namespace {
const uint32_t kTaskSize = 40000;
const uint32_t kMaxOverShardingFactor = 4;
const uint32_t kDecimalScaleNum = 10;
const uint32_t kTotalCostFactor = 210000;
constexpr int32_t kMaxCoreNum = 48;
constexpr uint32_t kMaxTaskSize = kTaskSize * kMaxOverShardingFactor;
} // namespace

namespace aicpu {
std::mutex EigenThreadPool::mutex_;
bool EigenThreadPool::init_flag_(false);
int32_t EigenThreadPool::core_num_(0);
std::unique_ptr<Eigen::ThreadPool> EigenThreadPool::eigen_threadpool_(nullptr);
std::unique_ptr<Eigen::ThreadPoolDevice> EigenThreadPool::threadpool_device_(nullptr);

EigenThreadPool* EigenThreadPool::GetInstance()
{
    if (!init_flag_) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!init_flag_) {
            const char* value = nullptr;
            MM_SYS_GET_ENV(MM_ENV_MAX_COMPILE_CORE_NUMBER, value);
            core_num_ = -1;
            if ((value != nullptr) && (value[0U] != '\0')) {
                try {
                    core_num_ = std::strtol(&(value[0U]), nullptr, kDecimalScaleNum); // 10 is for 10进制
                } catch (...) {
                    KERNEL_LOG_ERROR("eigen thread get core num:%d failed.", core_num_);
                    return nullptr;
                }
            }

            if (core_num_ <= 0 || core_num_ > kMaxCoreNum) {
                core_num_ = std::min(std::max(1, get_nprocs()), kMaxCoreNum);
            }

            eigen_threadpool_.reset(new Eigen::ThreadPool(core_num_));
            threadpool_device_.reset(new Eigen::ThreadPoolDevice(eigen_threadpool_.get(), core_num_));
            init_flag_ = true;
            KERNEL_LOG_EVENT("Eigen thread pool init success, core number[%d]", core_num_);
        }
    }

    static EigenThreadPool instance;
    return &instance;
}

void EigenThreadPool::ParallelFor(int64_t total, int64_t per_unit_size, const SharderWork& work) const
{
    KERNEL_LOG_INFO("Eigen threadpool parallel for begin, total[%ld], per_unit_size[%ld]", total, per_unit_size);
    if ((total <= 0) || (work == nullptr) || (per_unit_size <= 0)) {
        KERNEL_LOG_WARN(
            "Invalid param: total[%ld] <= 0 or per_unit_size[%ld] <= 0 or work "
            "is "
            "nullptr",
            total, per_unit_size);
        return;
    }

    int64_t total_check = static_cast<int64_t>(static_cast<Eigen::Index>(total));
    if (total_check != total) {
        KERNEL_LOG_ERROR("Invalid param: total[%ld], value[%ld] after eigen conversion", total, total_check);
        return;
    }

    double per_unit_cost = 1.0;
    if (per_unit_size >= total) {
        // use the current thread to process the task
        per_unit_cost = 1.0 * kTaskSize / total;
    } else if ((per_unit_size) <= (total / core_num_)) {
        // run tasks with the maximum number of threads, maximum =
        // kMaxOverShardingFactor * core_num_
        per_unit_cost = (1.0 * kMaxTaskSize * core_num_ / total) > (1.0 * kTotalCostFactor / total) ?
                            (1.0 * kMaxTaskSize * core_num_ / total) :
                            (1.0 * kTotalCostFactor / total);
    } else {
        // the task is fragmented based on the number of data slices.
        per_unit_cost = 1.0 * kMaxTaskSize / per_unit_size;
    }

    KERNEL_LOG_INFO("Eigen threadpool parallel for, per_unit_cost[%.6f]", per_unit_cost);

    threadpool_device_->parallelFor(
        total, Eigen::TensorOpCost(0, 0, per_unit_cost),
        [&work](Eigen::Index first, Eigen::Index last) { work(first, last); });
    KERNEL_LOG_INFO("Eigen threadpool parallel for success");
}

/*
 * Get CPU number
 */
uint32_t EigenThreadPool::GetCPUNum() const { return static_cast<uint32_t>(core_num_); }
} // namespace aicpu
