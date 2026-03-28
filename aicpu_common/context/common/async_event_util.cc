/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "async_event_util.h"
#include <dlfcn.h>
#include "log.h"

namespace {
const char* const kSharderPath = "libaicpu_sharder.so";
const char* const kNotifyWaitFunc = "AicpuNotifyWait";
const char* const kRegEventCbFunc = "AicpuRegEventCb";
const char* const kRegEventCbWithTimesFunc = "AicpuRegEventCbWithTimes";
const char* const kUnregEventCbFunc = "AicpuUnregEventCb";
const char* const kRegOpEventCbFunc = "AicpuRegOpEventCb";
const char* const kUnregOpEventCbFunc = "AicpuUnregOpEventCb";
}  // namespace

namespace aicpu {
AsyncEventUtil &AsyncEventUtil::GetInstance() {
    static AsyncEventUtil async_event_util;
    return async_event_util;
}

void AsyncEventUtil::InitEventUtil() {
  notify_wait_func_ = reinterpret_cast<NotifyWaitFunc>(
    dlsym(sharder_, kNotifyWaitFunc));
  if (notify_wait_func_ == nullptr) {
    KERNEL_LOG_ERROR("Get Function[%s] address failed, error[%s]", kNotifyWaitFunc, dlerror());
  }
  reg_event_cb_func_ = reinterpret_cast<RegEventCbFunc>(
    dlsym(sharder_, kRegEventCbFunc));
  if (reg_event_cb_func_ == nullptr) {
    KERNEL_LOG_ERROR("Get Function[%s] address failed, error[%s]", kRegEventCbFunc, dlerror());
  }
  reg_event_cb_with_times_func_ = reinterpret_cast<RegEventCbWithTimesFunc>(
    dlsym(sharder_, kRegEventCbWithTimesFunc));
  if (reg_event_cb_with_times_func_ == nullptr) {
    KERNEL_LOG_ERROR("Get Function[%s] address failed, error[%s]", kRegEventCbWithTimesFunc, dlerror());
  }
  unreg_event_cb_func_ = reinterpret_cast<UnregEventCbFunc>(
    dlsym(sharder_, kUnregEventCbFunc));
  if (unreg_event_cb_func_ == nullptr) {
    KERNEL_LOG_ERROR("Get Function[%s] address failed, error[%s]", kUnregEventCbFunc, dlerror());
  }
  reg_op_event_cb_func_ = PtrToFunc<void, RegEventCbFunc>(dlsym(sharder_, kRegOpEventCbFunc));
  if (reg_op_event_cb_func_ == nullptr) {
    KERNEL_LOG_ERROR("Get Function[%s] address failed, error[%s]", kRegOpEventCbFunc, dlerror());
  }
  unreg_op_event_cb_func_ = PtrToFunc<void, UnregEventCbFunc>(dlsym(sharder_, kUnregOpEventCbFunc));
  if (unreg_op_event_cb_func_ == nullptr) {
    KERNEL_LOG_ERROR("Get Function[%s] address failed, error[%s]", kUnregOpEventCbFunc, dlerror());
  }
}

AsyncEventUtil::AsyncEventUtil() {
  sharder_ = dlopen(kSharderPath, RTLD_LAZY | RTLD_GLOBAL);
  if (sharder_ == nullptr) {
    KERNEL_LOG_ERROR("Device sharder dlopen so [%s] failed, error[%s]",
                    kSharderPath, dlerror());
    notify_wait_func_ = nullptr;
    reg_event_cb_func_ = nullptr;
    reg_event_cb_with_times_func_ = nullptr;
    unreg_event_cb_func_ = nullptr;
    reg_op_event_cb_func_ = nullptr;
    unreg_op_event_cb_func_ = nullptr;
  } else {
    InitEventUtil();
    KERNEL_LOG_INFO("Device sharder dlopen so[%s] success.", kSharderPath);
  }
}

AsyncEventUtil::~AsyncEventUtil() {
  if (sharder_ != nullptr) {
    (void)dlclose(sharder_);
  }
}

void AsyncEventUtil::NotifyWait(void *notify_param, const uint32_t param_len) const {
  if (notify_wait_func_ != nullptr) {
    notify_wait_func_(notify_param, param_len);
    return;
  }
  KERNEL_LOG_ERROR("Function[%s] is null", kNotifyWaitFunc);
}

bool AsyncEventUtil::RegEventCb(const uint32_t event_id, const uint32_t sub_event_id,
                                const std::function<void(void *)> &cb) {
  if (reg_event_cb_func_ != nullptr) {
    return reg_event_cb_func_(event_id, sub_event_id, cb);
  }
  KERNEL_LOG_ERROR("Function[%s] is null.", kRegEventCbFunc);
  return false;
}

bool AsyncEventUtil::RegEventCb(const uint32_t event_id, const uint32_t sub_event_id,
                                const std::function<void(void *)> &cb,
                                const int32_t times) {
  if (reg_event_cb_with_times_func_ != nullptr) {
    return reg_event_cb_with_times_func_(event_id, sub_event_id, cb, times);
  }
  KERNEL_LOG_ERROR("Function[%s] is null.", kRegEventCbWithTimesFunc);
  return false;
}

void AsyncEventUtil::UnregEventCb(const uint32_t event_id, const uint32_t sub_event_id) {
  if (unreg_event_cb_func_ != nullptr) {
    return unreg_event_cb_func_(event_id, sub_event_id);
  }
  KERNEL_LOG_ERROR("Function[%s] is null.", kUnregEventCbFunc);
}

bool AsyncEventUtil::RegOpEventCb(const uint32_t event_id, const uint32_t sub_event_id,
                                const std::function<void(void *)> &cb) const {
  if (reg_op_event_cb_func_ != nullptr) {
    return reg_op_event_cb_func_(event_id, sub_event_id, cb);
  }
  KERNEL_LOG_ERROR("Function[%s] is null.", kRegOpEventCbFunc);
  return false;
}

void AsyncEventUtil::UnregOpEventCb(const uint32_t event_id, const uint32_t sub_event_id) const {
  if (unreg_op_event_cb_func_ != nullptr) {
    return unreg_op_event_cb_func_(event_id, sub_event_id);
  }
  KERNEL_LOG_ERROR("Function[%s] is null.", kUnregOpEventCbFunc);
}

}  // namespace aicpu

