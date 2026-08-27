/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "exe_graph/runtime/kernel_run_context.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/tensor_data.h"
#include "exe_graph/runtime/tiling_context.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "block_pool.h"
#include "kernel_utils.h"
#include "op_ctx_def.h"
#include "op_run_context.h"
#include "opdev/object.h"
#include "opdev/op_def.h"
#include "opdev/op_errno.h"
#include "opdev/op_log.h"
#include "tilingctx_builder.h"

namespace op {
namespace internal {
using BlockPool = internal::BlockPool;
using OpImplFunctions = gert::OpImplKernelRegistry::OpImplFunctions;

constexpr size_t GROWTH_FACTOR = 2;

enum class TilingOtherInputIdx : size_t {
    COMPILE_INFO = 0,
    PLATFORM_INFO,
    TILING_FUNC,
    DETERMINISTIC,
    DETERMINISTIC_LEVEL,
    PCIE_THROUGH_FLAG,
    // add new input definitions here
    INPUT_NUM
};
constexpr size_t TILING_INPUT_OTHER_NUM = static_cast<size_t>(TilingOtherInputIdx::INPUT_NUM);

namespace {
inline void SetTilingOtherInput(KernelRunContext* ctx, TilingOtherInputIdx field, AsyncAnyValue* value)
{
    ctx->values[ctx->input_size - TILING_INPUT_OTHER_NUM + static_cast<size_t>(field)] = value;
}

template <typename T>
inline T GetTilingOtherInputValue(KernelRunContext* ctx, TilingOtherInputIdx field)
{
    return *PtrCastTo<T>(
        ctx->values[ctx->input_size - TILING_INPUT_OTHER_NUM + static_cast<size_t>(field)]->data.inplace);
}
} // anonymous namespace

void TilingCtxHolder::BuildTilingCtx()
{
    // +1 for compiled info struct
    size_t tilingCtxSize = sizeof(AsyncAnyValue*) * (tilingCtxCapacity_ + TILING_INPUT_OTHER_NUM + tilingOutputNum_) +
                           sizeof(KernelRunContext);
    tilingCtx_ = static_cast<KernelRunContext*>(malloc(tilingCtxSize));
    OP_CHECK(tilingCtx_ != nullptr, OP_LOGE(ACLNN_ERR_INNER, "malloc failed. [%zu]", tilingCtxSize), return);
    (void)memset_s(tilingCtx_, tilingCtxSize, 0, tilingCtxSize);
    tilingCtx_->output_size = tilingOutputNum_;

    size_t tilingValueSize = sizeof(AsyncAnyValue) * tilingOutputNum_;
    tilingCtxValue_ = static_cast<AsyncAnyValue*>(malloc(tilingValueSize));
    OP_CHECK(tilingCtxValue_ != nullptr, OP_LOGE(ACLNN_ERR_INNER, "malloc failed. [%zu]", tilingValueSize), return);
    (void)memset_s(tilingCtxValue_, tilingValueSize, 0, tilingValueSize);

    // 创建 ExpandableRtsArgBuffer
    rtsArgBuffer_ = new (std::nothrow) ExpandableRtsArgBuffer();
    OP_CHECK(rtsArgBuffer_ != nullptr, OP_LOGE(ACLNN_ERR_INNER, "malloc failed. [%zu]", sizeof(ExpandableRtsArgBuffer)),
             return);
    aclnnStatus res = rtsArgBuffer_->Init(LAUNCH_ARG_INIT_SIZE, TILING_HOST_DATA_INIT_SIZE);
    OP_CHECK(res == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER, "failed to init expandable rts arg buffer."), return);

    // 设置 TilingData
    tilingData_ = rtsArgBuffer_->GetTilingDataPtr();
    rtsArgBuffer_->RegisterHolderTilingDataPtr(&tilingData_);
    tilingData_->capacity_ = TILING_HOST_DATA_INIT_SIZE;
    tilingData_->data_size_ = 0;
    tilingData_->data_ = rtsArgBuffer_->GetTilingDataAddr();

    // 注册 tilingOutput_.tilingData_ 指针
    tilingOutput_.tilingData_ = tilingData_;
    rtsArgBuffer_->RegisterOutputTilingDataPtr(&tilingOutput_.tilingData_);

    // 设置 tilingOutput_.rtsArgBuffer_
    tilingOutput_.rtsArgBuffer_ = rtsArgBuffer_;

    OP_LOGI("TilingData: %p, cap: %zu", tilingData_, TILING_HOST_DATA_INIT_SIZE);

    workspaceSizeVec_ = gert::ContinuousVector::Create<size_t>(MAX_WORKSPACE_NUM);
    if (workspaceSizeVec_ == nullptr) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Create ContinuousVector failed. size[%zu]", MAX_WORKSPACE_NUM);
        return;
    }

    tilingCtxValue_[kOutputTilingData].data.pointer = tilingData_;
    tilingCtxValue_[kOutputWorkspace].data.pointer = workspaceSizeVec_.get();

    // tiling_ctx_outputs_[kOutputTilingKey].data.pointer = &tiling_outputs_->tiling_key_;
    tilingOutput_.tilingKey_ = PtrCastTo<uint64_t>(tilingCtxValue_[kOutputTilingKey].data.inplace);
    tilingOutput_.numBlocks_ = PtrCastTo<int64_t>(tilingCtxValue_[kOutputBlockDim].data.inplace);
    tilingOutput_.atomicCleanFlag_ = PtrCastTo<bool>(tilingCtxValue_[kOutputAtomicCleanFlag].data.inplace);
    tilingOutput_.workspaceSize_ = PtrCastTo<gert::TypedContinuousVector<size_t>>(workspaceSizeVec_.get());
    tilingOutput_.tilingCond_ = PtrCastTo<int64_t>(tilingCtxValue_[kOutputTilingCond].data.inplace);
    tilingOutput_.scheduleMode_ = PtrCastTo<uint8_t>(tilingCtxValue_[kOutputScheduleMode].data.inplace);
    tilingOutput_.dynUBufSize_ = PtrCastTo<uint32_t>(tilingCtxValue_[kOutputLocalMemorySize].data.inplace);
}

aclnnStatus TilingCtxHolder::EnsureTilingCtxCapacity(size_t requiredCapacity)
{
    if (requiredCapacity <= tilingCtxCapacity_) {
        return ACLNN_SUCCESS;
    }

    // Calculate new capacity using growth factor 2x
    const size_t oldCapacity = tilingCtxCapacity_;
    size_t newCapacity = oldCapacity;
    while (newCapacity < requiredCapacity) {
        newCapacity *= GROWTH_FACTOR;
    }

    // Allocate new memory (do NOT use realloc)
    size_t newSize = sizeof(AsyncAnyValue*) * (newCapacity + TILING_INPUT_OTHER_NUM + tilingOutputNum_) +
                     sizeof(KernelRunContext);
    KernelRunContext* newCtx = static_cast<KernelRunContext*>(malloc(newSize));
    OP_CHECK(newCtx != nullptr, OP_LOGE(ACLNN_ERR_INNER, "failed to malloc tilingCtx, size %zu.", newSize),
             return ACLNN_ERR_INNER);

    // Calculate old size for copy
    size_t oldSize = sizeof(AsyncAnyValue*) * (tilingCtxCapacity_ + TILING_INPUT_OTHER_NUM + tilingOutputNum_) +
                     sizeof(KernelRunContext);

    // Copy existing data
    OP_CHECK(memcpy_s(newCtx, newSize, tilingCtx_, oldSize) == EOK,
             OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy tilingCtx."), std::free(newCtx);
             return ACLNN_ERR_INNER);

    // Zero out the new portion
    size_t zeroSize = newSize - oldSize;
    (void)memset_s(PtrCastTo<uint8_t>(newCtx) + oldSize, zeroSize, 0, zeroSize);

    // Free old memory and update pointer
    std::free(tilingCtx_);
    tilingCtx_ = newCtx;
    tilingCtxCapacity_ = newCapacity;

    OP_LOGI("Expanded tilingCtx capacity from %zu to %zu.", oldCapacity, tilingCtxCapacity_);
    return ACLNN_SUCCESS;
}

aclnnStatus TilingCtxHolder::ResetTilingCtx(const KernelContextHolder* kernelCtx)
{
    size_t requiredCapacity = kernelCtx->inputNum_ + kernelCtx->outputNum_ + TILING_INPUT_OTHER_NUM + tilingOutputNum_;
    CHECK_COND(EnsureTilingCtxCapacity(requiredCapacity) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "EnsureTilingCtxCapacity failed.");

    tilingData_->data_size_ = 0;
    tilingCtxValue_[kOutputTilingData].data.pointer = tilingData_;
    *tilingOutput_.atomicCleanFlag_ = false;
    *tilingOutput_.numBlocks_ = 0;
    *tilingOutput_.tilingKey_ = 0;
    *tilingOutput_.scheduleMode_ = 0;
    *tilingOutput_.dynUBufSize_ = 0;
    PtrCastTo<gert::ContinuousVector>(workspaceSizeVec_.get())->SetSize(0);

    tilingCtx_->compute_node_info = kernelCtx->computeNodeInfo_;
    tilingCtx_->kernel_extend_info = &kernelCtx->kernelExtendInfo_;

    size_t opInputNum = kernelCtx->inputNum_;
    size_t opOutputNum = kernelCtx->outputNum_;
    tilingOutput_.inputNum_ = opInputNum;
    tilingOutput_.outputNum_ = opOutputNum;

    tilingCtx_->input_size = opInputNum + opOutputNum + TILING_INPUT_OTHER_NUM;
    for (size_t i = 0; i < opInputNum + opOutputNum; i++) {
        tilingCtx_->values[i] = &kernelCtx->opInArg_[i];
    }
    return ACLNN_SUCCESS;
}

void TilingCtxHolder::FinalizeTilingCtx(size_t tilingInputNum)
{
    for (size_t i = 0; i < tilingOutputNum_; i++) {
        tilingCtx_->values[tilingInputNum + i] = &tilingCtxValue_[i];
    }
    tilingCtx_->output_start = tilingCtx_->values + tilingCtx_->input_size;
}

aclnnStatus TilingCtxHolder::UpdateTilingCtx(const KernelContextHolder* kernelCtx,
                                             const TilingParseCtxHolder* tilingParseCtx)
{
    CHECK_COND(kernelCtx != nullptr, ACLNN_ERR_RUNTIME_ERROR, "kernelCtx is NULL");
    CHECK_COND(tilingParseCtx != nullptr, ACLNN_ERR_RUNTIME_ERROR, "tilingParseCtx is NULL");

    if (ResetTilingCtx(kernelCtx) != ACLNN_SUCCESS) {
        return ACLNN_ERR_INNER;
    }

    uint32_t coreNum = tilingParseCtx->GetCoreNum();
    uint32_t cubeCoreNum = GetThreadLocalContext().opConfigInfo_.aicNum_;
    uint32_t vectorCoreNum = GetThreadLocalContext().opConfigInfo_.aivNum_;
    fe::PlatFormInfos* platformInfo = SocContext::GetPlatformInfo();
    UpdateThradLocalPlatformInfo(platformInfo, coreNum, cubeCoreNum, vectorCoreNum);
    platformInfoValue_.data.pointer = platformInfo;

    *PtrCastTo<int32_t>(deterministicValue_.data.inplace) = GetThreadLocalContext().opConfigInfo_.isDeterministicOn_ ?
                                                                1 :
                                                                0;
    *PtrCastTo<int32_t>(
        deterministicLevelValue_.data.inplace) = GetThreadLocalContext().opConfigInfo_.deterministicLevel_;
    *PtrCastTo<bool>(pcieThroughFlagValue_.data.inplace) = GetThreadLocalContext().opConfigInfo_.usePcieAddr;

    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::COMPILE_INFO, tilingParseCtx->GetCompiledInfoStruct());
    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::PLATFORM_INFO, &platformInfoValue_);
    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::TILING_FUNC, nullptr);
    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::DETERMINISTIC, &deterministicValue_);
    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::DETERMINISTIC_LEVEL, &deterministicLevelValue_);
    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::PCIE_THROUGH_FLAG, &pcieThroughFlagValue_);

    FinalizeTilingCtx(tilingCtx_->input_size);

    OP_LOGI("Update op tiling ctx. input[%zu], output[%zu], compiled Info %p, deterministic %d, deterministicLevel %d, "
            "pcie through flag: %d, tilingDataWrap: %p, coreNum: %u",
            kernelCtx->inputNum_, kernelCtx->outputNum_, tilingParseCtx->GetCompiledInfoStruct(),
            GetTilingOtherInputValue<int32_t>(tilingCtx_, TilingOtherInputIdx::DETERMINISTIC),
            GetTilingOtherInputValue<int32_t>(tilingCtx_, TilingOtherInputIdx::DETERMINISTIC_LEVEL),
            GetTilingOtherInputValue<bool>(tilingCtx_, TilingOtherInputIdx::PCIE_THROUGH_FLAG), tilingData_, coreNum);
    return ACLNN_SUCCESS;
}

aclnnStatus TilingCtxHolder::UpdateTilingCtx(const KernelContextHolder* kernelCtx)
{
    CHECK_COND(kernelCtx != nullptr, ACLNN_ERR_RUNTIME_ERROR, "kernelCtx is NULL");

    if (ResetTilingCtx(kernelCtx) != ACLNN_SUCCESS) {
        return ACLNN_ERR_INNER;
    }

    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::COMPILE_INFO, nullptr);
    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::PLATFORM_INFO, nullptr);
    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::TILING_FUNC, nullptr);
    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::DETERMINISTIC, nullptr);
    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::DETERMINISTIC_LEVEL, nullptr);
    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::PCIE_THROUGH_FLAG, nullptr);
    FinalizeTilingCtx(tilingCtx_->input_size);
    return ACLNN_SUCCESS;
}

aclnnStatus TilingCtxHolder::UpdateTilingCtx(const KernelContextHolder* kernelCtx, const nlohmann::json& opJson)
{
    CHECK_COND(kernelCtx != nullptr, ACLNN_ERR_RUNTIME_ERROR, "kernelCtx is NULL");

    if (ResetTilingCtx(kernelCtx) != ACLNN_SUCCESS) {
        return ACLNN_ERR_INNER;
    }

    uint32_t coreNum = 0;
    fe::PlatFormInfos* platformInfo = SocContext::GetPlatformInfo();
    SetCoreNum(opJson, platformInfo, coreNum);
    platformInfoValue_.data.pointer = platformInfo;

    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::COMPILE_INFO, nullptr);
    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::PLATFORM_INFO, &platformInfoValue_);
    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::TILING_FUNC, nullptr);
    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::DETERMINISTIC, nullptr);
    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::DETERMINISTIC_LEVEL, nullptr);
    SetTilingOtherInput(tilingCtx_, TilingOtherInputIdx::PCIE_THROUGH_FLAG, nullptr);
    FinalizeTilingCtx(tilingCtx_->input_size);

    OP_LOGI("Update static kernel tiling ctx. input[%zu], output[%zu], coreNum: %u", kernelCtx->inputNum_,
            kernelCtx->outputNum_, coreNum);
    return ACLNN_SUCCESS;
}

TilingCtxHolder::~TilingCtxHolder()
{
    FREE(tilingCtx_);
    FREE(tilingCtxValue_);
    delete rtsArgBuffer_;
}
} // namespace internal
} // namespace op
