/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cust_dlog_record.h"

#include <unistd.h>
#include <sys/syscall.h>
#include <cstdarg>
#include <cstdio>
#include "securec.h"

extern "C" {
__attribute__((visibility("default"))) int32_t CustSetCpuKernelContext(uint64_t workspace_size, uint64_t workspace_addr)
{
    return aicpu::CustCpuKernelDlogUtils::GetInstance().CustSetCpuKernelContext(workspace_size, workspace_addr);
}
}

namespace {
const int32_t kMaxLogLen = 1024;
const int32_t kInvalidInput = 1;
const int32_t kCreateCtxErr = 2;
} // namespace

namespace aicpu {

CustCpuKernelDlogUtils& CustCpuKernelDlogUtils::GetInstance()
{
    static CustCpuKernelDlogUtils instance;
    return instance;
}

CustCpuKernelDlogUtils::~CustCpuKernelDlogUtils() { ctx_map_.clear(); }

int64_t CustCpuKernelDlogUtils::GetTid()
{
    thread_local static int64_t tid = syscall(__NR_gettid);
    return tid;
}

int32_t CustCpuKernelDlogUtils::CustSetCpuKernelContext(uint64_t workspace_size, uint64_t workspace_addr)
{
    if (workspace_size <= 1UL || workspace_addr == 0UL) {
        return kInvalidInput;
    }

    DeviceType device_type = DEVICE;
    CpuKernelContext* tmp = new (std::nothrow) CpuKernelContext(device_type);
    if (tmp == nullptr) {
        return kCreateCtxErr;
    }

    int64_t tid = GetTid();
    auto ctx = std::shared_ptr<aicpu::CpuKernelContext>(tmp);
    ctx->workspace_addr_ = workspace_addr;
    ctx->workspace_size_ = workspace_size;
    std::lock_guard<std::mutex> lock(ctx_mutex_);
    if (ctx_map_.find(tid) != ctx_map_.end()) {
        return 0;
    }
    ctx_map_[tid] = ctx;
    return 0;
}

std::shared_ptr<CpuKernelContext> CustCpuKernelDlogUtils::GetCpuKernelContext()
{
    int64_t tid = GetTid();
    std::lock_guard<std::mutex> lock(ctx_mutex_);
    if (ctx_map_.find(tid) != ctx_map_.end()) {
        return ctx_map_[tid];
    }
    return nullptr;
}

void CustCpuKernelDlogUtils::DumpLog(int32_t module_id, int32_t level, const char* fmt, va_list args)
{
    const auto it = module_to_string_map_.find(module_id);
    const std::string module_name = (it != module_to_string_map_.end()) ? it->second : "UNKNOWN_MODULE";

    const std::shared_ptr<CpuKernelContext> ctx = GetCpuKernelContext();
    if (ctx == nullptr) {
        return;
    }

    char formatted_msg[kMaxLogLen];
    int32_t len = snprintf_s(formatted_msg, kMaxLogLen, kMaxLogLen - 1, "[%s]", module_name.c_str());
    if (len < 0 || len >= kMaxLogLen) {
        return;
    }

    len += vsnprintf_s(formatted_msg + len, kMaxLogLen - len, kMaxLogLen - len - 1, fmt, args);
    if (len < 0 || len >= kMaxLogLen) {
        return;
    }

    switch (level) {
        case DLOG_DEBUG:
            CustCpuKernelUtils::CustLogDebug(*ctx, formatted_msg);
            break;
        case DLOG_INFO:
            CustCpuKernelUtils::CustLogInfo(*ctx, formatted_msg);
            break;
        case DLOG_WARN:
            CustCpuKernelUtils::CustLogWarning(*ctx, formatted_msg);
            break;
        case DLOG_ERROR:
            CustCpuKernelUtils::CustLogError(*ctx, formatted_msg);
            break;
        default:
            break;
    }
}

} // namespace aicpu