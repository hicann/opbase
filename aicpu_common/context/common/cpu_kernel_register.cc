/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cpu_kernel_register.h"
#include <chrono>
#include "aicpu_context.h"
#include "aicpu_async_event.h"
#include "cpu_kernel.h"
#include "log.h"
#include "status.h"
#include "async_event_util.h"
#include "async_cpu_kernel.h"

namespace {
#define TYPE_REGISTAR(type, fun) type##Registerar(type, fun)
#define TYPE_REGISTARV2(type, fun) type##RegisterarV2(type, fun)
}  // namespace

namespace aicpu {
/*
 * regist kernel.
 */
bool RegistCpuKernel(const std::string &type, const KERNEL_CREATOR_FUN &fun) {
  CpuKernelRegister::Registerar TYPE_REGISTAR(type, fun);
  return true;
}

bool RegistCpuKernelV2(const std::string &type, const KERNEL_CREATOR_FUN &fun) {
  CpuKernelRegister::RegisterarV2 TYPE_REGISTARV2(type, fun);
  return true;
}

/*
 * get instance.
 * @return CpuKernelRegister &: CpuKernelRegister instance
 */
CpuKernelRegister &CpuKernelRegister::Instance() {
  static CpuKernelRegister instance;
  return instance;
}

/*
 * get cpu kernel.
 * param opType: the op type of kernel
 * @return shared_ptr<CpuKernel>: cpu kernel ptr
 */
std::shared_ptr<CpuKernel> CpuKernelRegister::GetCpuKernel(
    const std::string &op_type) {
  std::map<std::string, KERNEL_CREATOR_FUN>::const_iterator iter = creatorMap_.find(op_type);
  if (iter != creatorMap_.cend()) {
    return iter->second();
  }
  KERNEL_LOG_WARN("op type [%s] is not registered in v1.", op_type.c_str());
  return std::shared_ptr<CpuKernel>(nullptr);
}

std::shared_ptr<CpuKernel> CpuKernelRegister::GetCpuKernelV2(
    const std::string &op_type) {
  std::map<std::string, KERNEL_CREATOR_FUN>::const_iterator iterV2 = creatorMapV2_.find(op_type);
  if (iterV2 != creatorMapV2_.cend()) {
    return iterV2->second();
  }
  return std::shared_ptr<CpuKernel>(nullptr);
}

/*
 * check whether op_type is registered in V2, without creating a kernel.
 */
bool CpuKernelRegister::IsRegisteredV2(const std::string &op_type) const {
  return creatorMapV2_.find(op_type) != creatorMapV2_.cend();
}

/*
 * get all cpu kernel registered op types.
 * @return std::vector<string>: all cpu kernel registered op type
 */
std::vector<std::string> CpuKernelRegister::GetAllRegisteredOpTypes() const {
  std::vector<std::string> ret;
  for (auto iter = creatorMap_.begin(); iter != creatorMap_.end(); ++iter) {
    ret.push_back(iter->first);
  }

  return ret;
}

std::vector<std::string> CpuKernelRegister::GetAllRegisteredOpTypesV2() const {
  std::vector<std::string> ret;
  for (auto iter = creatorMapV2_.begin(); iter != creatorMapV2_.end(); ++iter) {
    ret.push_back(iter->first);
  }

  return ret;
}

uint32_t CpuKernelRegister::RunCpuKernelCommon(CpuKernelContext &ctx, const std::string type,
                                               const std::shared_ptr<CpuKernel> kernel) {
  if (aicpu::SetThreadLocalCtx != nullptr) {
    if (aicpu::SetThreadLocalCtx(aicpu::CONTEXT_KEY_OP_NAME, type) !=
        aicpu::AICPU_ERROR_NONE) {
      KERNEL_LOG_ERROR("Set kernel name[%s] to context failed.", type.c_str());
      return KERNEL_STATUS_INNER_ERROR;
    }
  }
  if (aicpu::SetOpname != nullptr) {
    (void)aicpu::SetOpname(type);
  }

  auto start = std::chrono::steady_clock::now();
  uint32_t ret = kernel->Compute(ctx);
  auto end = std::chrono::steady_clock::now();
  KERNEL_LOG_INFO("op type [%s] run cpu kernel finished, run time is [%lf] us.", type.c_str(),
      std::chrono::duration<double, std::micro>(end - start).count());
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  KERNEL_LOG_INFO("op type [%s] run cpu kernel success.", type.c_str());
  return KERNEL_STATUS_OK;
}

/*
 * run cpu kernel.
 * param ctx: context of kernel
 * @return uint32_t: 0->success other->failed
 */
uint32_t CpuKernelRegister::RunCpuKernel(CpuKernelContext &ctx) {
  std::string type = ctx.GetOpType();
  KERNEL_LOG_INFO("op type [%s] run cpu kernel.", type.c_str());
  auto kernel = GetCpuKernel(type);
  if (kernel == nullptr) {
    KERNEL_LOG_WARN("op type [%s] run cpu kernel failed, kernel is null.", type.c_str());
    return KERNEL_STATUS_INNER_ERROR;
  }
  return RunCpuKernelCommon(ctx, type, kernel);
}

uint32_t CpuKernelRegister::RunCpuKernelV2(CpuKernelContext &ctx) {
  std::string type = ctx.GetOpType();
  KERNEL_LOG_INFO("op type [%s] run cpu kernel v2.", type.c_str());
  auto kernel = GetCpuKernelV2(type);
  if (kernel == nullptr) {
    KERNEL_LOG_WARN("op type [%s] run cpu kernel v2 failed, kernel is null.", type.c_str());
    return KERNEL_STATUS_INNER_ERROR;
  }
  return RunCpuKernelCommon(ctx, type, kernel);
}

uint32_t CpuKernelRegister::SetAsyncKernelContext(const std::string &type, const uint8_t wait_type,
                                                  const uint32_t wait_id) {
  if (aicpu::SetThreadLocalCtx != nullptr) {
    if (aicpu::SetThreadLocalCtx(aicpu::CONTEXT_KEY_OP_NAME, type) !=
        aicpu::AICPU_ERROR_NONE) {
      KERNEL_LOG_ERROR("Set kernel name[%s] to context failed.", type.c_str());
      return KERNEL_STATUS_INNER_ERROR;
    }
    if (aicpu::SetThreadLocalCtx(aicpu::CONTEXT_KEY_WAIT_TYPE, std::to_string(wait_type)) !=
        aicpu::AICPU_ERROR_NONE) {
      KERNEL_LOG_ERROR("Set wait type to context failed.");
      return KERNEL_STATUS_INNER_ERROR;
    }
    if (aicpu::SetThreadLocalCtx(aicpu::CONTEXT_KEY_WAIT_ID, std::to_string(wait_id)) !=
        aicpu::AICPU_ERROR_NONE) {
      KERNEL_LOG_ERROR("Set wait id to context failed.");
      return KERNEL_STATUS_INNER_ERROR;
    }
  }
  if (aicpu::SetOpname != nullptr) {
    (void)aicpu::SetOpname(type);
  }
  return KERNEL_STATUS_OK;
}

uint32_t CpuKernelRegister::RunCpuKernelAsyncCommon(CpuKernelContext &ctx,
                                                    const uint8_t wait_type,
                                                    const uint32_t wait_id,
                                                    std::function<uint32_t()> cb,
                                                    const std::shared_ptr<CpuKernel> kernel) {
  std::string type = ctx.GetOpType();
  AsyncCpuKernel *async_kernel = dynamic_cast<AsyncCpuKernel *>(kernel.get());
  if (async_kernel == nullptr) {
    KERNEL_LOG_ERROR("kernel name[%s] does not hava async impl", type.c_str());
    return KERNEL_STATUS_INNER_ERROR;
  }
  uint32_t ret = SetAsyncKernelContext(type, wait_type, wait_id);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  std::shared_ptr<AsyncNotifyInfo> notify_info = nullptr;	 
  try {	 
    notify_info = std::make_shared<AsyncNotifyInfo>();	 
  } catch(std::exception &e) {	 
    KERNEL_LOG_ERROR("Create notify_info failed, reason is %s", e.what());	 
    return KERNEL_STATUS_INNER_ERROR;	 
  }	 
  if (aicpu::GetTaskAndStreamId != nullptr) {	 
    (void)aicpu::GetTaskAndStreamId(notify_info->taskId, notify_info->streamId);	 
  }	 
  if (aicpu::aicpuGetContext != nullptr) {	 
    (void)aicpu::aicpuGetContext(&notify_info->ctx);	 
  }	 
  notify_info->waitType = wait_type;	 
  notify_info->waitId = wait_id;
  auto start = std::chrono::steady_clock::now();
  auto done = [notify_info, kernel, type, cb, start](uint32_t status) {
    auto end = std::chrono::steady_clock::now();
    double dr_us = std::chrono::duration<double, std::micro>(end-start).count();
    KERNEL_LOG_INFO("op type [%s] run cpu kernel async finished, run time is [%lf] us.", type.c_str(), dr_us);
    if (status == KERNEL_STATUS_OK) {
      KERNEL_LOG_INFO("op type [%s] run cpu kernel async success.", type.c_str());
      status = cb();
    }
    notify_info->retCode = status;
    void *param = reinterpret_cast<void *>(notify_info.get());
    KERNEL_LOG_INFO("RunCpuKernelAsync notify event wait, wait_type[%u], "
                    "wait_id[%u], task_id[%lu], stream_id[%u], status[%u]",
                    notify_info->waitType, notify_info->waitId, notify_info->taskId,
                    notify_info->streamId, notify_info->retCode);
    AsyncEventUtil::GetInstance().NotifyWait(param, sizeof(AsyncNotifyInfo));
  };
  return async_kernel->ComputeAsync(ctx, done);
}

uint32_t CpuKernelRegister::RunCpuKernelAsync(CpuKernelContext &ctx,
                                              const uint8_t wait_type,
                                              const uint32_t wait_id,
                                              std::function<uint32_t()> cb) {
  std::string type = ctx.GetOpType();
  KERNEL_LOG_INFO("op type [%s] run cpu kernel async.", type.c_str());
  auto kernel = GetCpuKernel(type);
  if (kernel == nullptr) {
    KERNEL_LOG_WARN("op type [%s] run cpu kernel async failed, kernel is null.", type.c_str());
    return KERNEL_STATUS_INNER_ERROR;
  }
  return RunCpuKernelAsyncCommon(ctx, wait_type, wait_id, cb, kernel);
}

uint32_t CpuKernelRegister::RunCpuKernelAsyncV2(CpuKernelContext &ctx,
                                              const uint8_t wait_type,
                                              const uint32_t wait_id,
                                              std::function<uint32_t()> cb) {
  std::string type = ctx.GetOpType();
  KERNEL_LOG_INFO("op type [%s] run cpu kernel async v2.", type.c_str());
  auto kernel = GetCpuKernelV2(type);
  if (kernel == nullptr) {
    KERNEL_LOG_WARN("op type [%s] run cpu kernel async v2 failed, kernel is null.", type.c_str());
    return KERNEL_STATUS_INNER_ERROR;
  }
  return RunCpuKernelAsyncCommon(ctx, wait_type, wait_id, cb, kernel);
}

CpuKernelRegister::Registerar::Registerar(const std::string &type,
                                          const KERNEL_CREATOR_FUN &fun) {
  CpuKernelRegister::Instance().Register(type, fun);
}

CpuKernelRegister::RegisterarV2::RegisterarV2(const std::string &type,
                                              const KERNEL_CREATOR_FUN &fun) {
  CpuKernelRegister::Instance().RegisterV2(type, fun);
}

// register creator, this function will call in the constructor.
void CpuKernelRegister::Register(const std::string &type,
                                 const KERNEL_CREATOR_FUN &fun) {
  std::map<std::string, KERNEL_CREATOR_FUN>::const_iterator iter =
      creatorMap_.find(type);
  if (iter != creatorMap_.cend()) {
    KERNEL_LOG_WARN("op type [%s] register skipped, already exist in v1.",
                    type.c_str());
    return;
  }

  creatorMap_[type] = fun;
  KERNEL_LOG_DEBUG("op type [%s] register success in v1.", type.c_str());
}

// register creator V2, this function will call in the constructor.
void CpuKernelRegister::RegisterV2(const std::string &type,
                                   const KERNEL_CREATOR_FUN &fun) {
  std::map<std::string, KERNEL_CREATOR_FUN>::const_iterator iter =
      creatorMapV2_.find(type);
  if (iter != creatorMapV2_.cend()) {
    KERNEL_LOG_WARN("op type [%s] register skipped, already exist in v2.",
                    type.c_str());
    return;
  }

  creatorMapV2_[type] = fun;
  KERNEL_LOG_INFO("op type [%s] register success in v2.", type.c_str());
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default")))
std::vector<std::string> GetAllRegisteredOpTypesV2() {
  return aicpu::CpuKernelRegister::Instance().GetAllRegisteredOpTypesV2();
}

__attribute__((visibility("default")))
bool IsRegisteredV2(const std::string &op_type) {
  return aicpu::CpuKernelRegister::Instance().IsRegisteredV2(op_type);
}

__attribute__((visibility("default")))
uint32_t RunCpuKernelV2(aicpu::CpuKernelContext &ctx) {
  return aicpu::CpuKernelRegister::Instance().RunCpuKernelV2(ctx);
}
}  // extern "C"
