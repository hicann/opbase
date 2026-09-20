/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/direct_invoke_task.h"

#include <limits>
#include <new>

#include "bridge_graph.h"
#include "kernel_launcher.h"
#include "op_dfx_internal.h"
#include "opdev/op_arg_def.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

namespace op {

namespace {

// GetOpArg never returns null for the arg kinds used here; a malformed list
// is one that claims entries without backing storage.
bool IsWellFormedArgList(const OpArgList* list)
{
    return list != nullptr && (list->count == 0U || list->args != nullptr);
}

// launcher must own args. This function consumes launcher and args on every
// return path: failure deletes launcher, success transfers it to executor.
aclnnStatus RegisterKernelLauncher(uint32_t opType, aclOpExecutor* executor, OpArgContext* args,
                                   KernelLauncher* launcher, bool includeOutShape = false)
{
    if (launcher == nullptr) {
        DestroyOpArgContext(args);
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Kernel launcher must not be null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr || args == nullptr) {
        delete launcher;
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Executor and launcher arguments must not be null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    OpArgList* inputs = args->GetOpArg(OP_INPUT_ARG);
    OpArgList* outputs = args->GetOpArg(OP_OUTPUT_ARG);
    OpArgList* workspaces = args->GetOpArg(OP_WORKSPACE_ARG);
    OpArgList* outShapes = args->GetOpArg(OP_OUTSHAPE_ARG);
    if (!IsWellFormedArgList(inputs) || !IsWellFormedArgList(outputs) || !IsWellFormedArgList(workspaces) ||
        (includeOutShape && !IsWellFormedArgList(outShapes))) {
        delete launcher;
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Launcher graph arguments are malformed.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    aclnnStatus status = includeOutShape ?
                             internal::BuildGraph(executor->GetGraph(), opType, *inputs, *outputs, *workspaces,
                                                  *outShapes) :
                             internal::BuildGraph(executor->GetGraph(), opType, *inputs, *outputs, *workspaces);
    if (status != ACLNN_SUCCESS) {
        delete launcher;
        return status;
    }
    executor->AddToKernelLauncherList(launcher);
    return ACLNN_SUCCESS;
}

// Matches the framework graph builder: a null input tensor means an absent
// optional input and is tolerated; outputs must be present.
enum class TensorRequirement : uint32_t {
    OptionalInput = 0U,
    Required = 1U,
};

bool IsValidTensor(const aclTensor* tensor, TensorRequirement requirement)
{
    if (tensor == nullptr) {
        return requirement == TensorRequirement::OptionalInput;
    }
    return true;
}

bool IsValidTensorArg(const OpArg& arg, TensorRequirement requirement)
{
    if (arg.type == OpArgType::OPARG_ACLTENSOR) {
        return IsValidTensor(static_cast<const aclTensor*>(arg.value.data.pointer), requirement);
    }
    if (arg.type != OpArgType::OPARG_ACLTENSOR_LIST) {
        return false;
    }
    const auto* tensors = static_cast<const aclTensorList*>(arg.value.data.pointer);
    if (tensors == nullptr) {
        return false;
    }
    for (uint64_t index = 0U; index < tensors->Size(); ++index) {
        if (!IsValidTensor((*tensors)[index], requirement)) {
            return false;
        }
    }
    return true;
}

bool IsValidTensorList(const OpArgList& args, TensorRequirement requirement)
{
    if (args.count != 0U && args.args == nullptr) {
        return false;
    }
    for (size_t index = 0U; index < args.count; ++index) {
        if (!IsValidTensorArg(args.args[index], requirement)) {
            return false;
        }
    }
    return true;
}

bool IsValidArgs(const OpArgContext& args)
{
    const OpArgList* inputs = args.GetOpArg(OP_INPUT_ARG);
    const OpArgList* outputs = args.GetOpArg(OP_OUTPUT_ARG);
    const OpArgList* attributes = args.GetOpArg(OP_ATTR_ARG);
    const OpArgList* workspaces = args.GetOpArg(OP_WORKSPACE_ARG);
    // Workspace buffers are requested through the prepare callback and
    // allocated by opbase; the entry workspace list must be absent or empty.
    return inputs != nullptr && outputs != nullptr && attributes != nullptr &&
           (attributes->count == 0U || attributes->args != nullptr) &&
           IsValidTensorList(*inputs, TensorRequirement::OptionalInput) &&
           IsValidTensorList(*outputs, TensorRequirement::Required) &&
           (workspaces == nullptr || workspaces->count == 0U);
}

// Mirrors the AI Core GetWorkspace flow: one DT_UINT8 tensor per requested
// size, allocated from the executor so the memory planner applies its
// alignment and lifetime reuse, then appended as the workspace argument.
aclnnStatus AppendRequestedWorkspaces(aclOpExecutor* executor, const DirectInvokeWorkspaceRequest& request,
                                      OpArgContext* args)
{
    if (request.count == 0U) {
        return ACLNN_SUCCESS;
    }
    if (request.sizes == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Workspace request count is non-zero but sizes is null.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    FVector<aclTensor*> workspaces;
    for (uint32_t index = 0U; index < request.count; ++index) {
        const uint64_t size = request.sizes[index];
        if (size == 0U || size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Workspace request contains an invalid size %llu at index %u.",
                    static_cast<unsigned long long>(size), index);
            return ACLNN_ERR_PARAM_INVALID;
        }
        aclTensor* tensor = executor->AllocTensor(Shape({static_cast<int64_t>(size)}), DataType::DT_UINT8);
        if (tensor == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER, "Failed to allocate requested workspace of %llu bytes.",
                    static_cast<unsigned long long>(size));
            return ACLNN_ERR_INNER;
        }
        workspaces.push_back(tensor);
    }
    aclTensorList* workspaceList = executor->AllocTensorList(workspaces.data(), workspaces.size());
    if (workspaceList == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "Failed to allocate requested workspace list.");
        return ACLNN_ERR_INNER;
    }
    args->AppendOpWorkspaceArg(workspaceList);
    return ACLNN_SUCCESS;
}

class DirectInvokeKernelLauncher final : public KernelLauncher {
public:
    DirectInvokeKernelLauncher(uint32_t opType, const aclOpExecutor* executor, DirectInvokeLaunchFn launch,
                               DirectInvokeDestroyFn destroy, void* launchState, OpArgContext* args)
        : KernelLauncher(opType, DIRECT_INVOKE, executor, internal::ProfilingInfoId{}),
          launch_(launch),
          destroy_(destroy),
          launchState_(launchState),
          args_(args)
    {
        // Attribution comes from the enclosing DFX context captured by the
        // base class (TLS names are macro stringification literals) and from
        // opType, same as the AI Core path; no name override is accepted.
    }

    ~DirectInvokeKernelLauncher() override
    {
        if (launchState_ != nullptr) {
            destroy_(launchState_);
            launchState_ = nullptr;
        }
        DestroyOpArgContext(args_);
        args_ = nullptr;
    }

    aclnnStatus Launch() override
    {
        if (launch_ == nullptr || args_ == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER, "DirectInvoke task launcher is not initialized.");
            return ACLNN_ERR_INNER;
        }
        // Same TLS discipline as the AI Core path: phase 2 entry already
        // restored the L2 context (InitL2Phase2Context), so the launcher only
        // fills in the l0Name captured at registration time.
        auto& threadLocalCtx = op::internal::GetThreadLocalContext();
        threadLocalCtx.logInfo_.l0Name = opLogInfo_.l0Name;
        threadLocalCtx.profilingInfoId_ = profilingInfoId_;

        // DirectInvoke callbacks always submit compute tasks.
        if (threadLocalCtx.opConfigInfo_.isOpDumpEnable_) {
            op::internal::DumpL0(*args_->GetOpArg(op::OP_INPUT_ARG), opLogInfo_, OpInputType, executor_->GetStream());
        }

        const aclnnStatus ret = launch_(launchState_, args_, executor_->GetStream());

        if (threadLocalCtx.opConfigInfo_.isOpDumpEnable_) {
            op::internal::DumpL0(*args_->GetOpArg(op::OP_OUTPUT_ARG), opLogInfo_, OpOutputType, executor_->GetStream());
        }
        if (ret == ACLNN_SUCCESS && op::internal::IsOverflowDumpEnable()) {
            CheckOverflowDump();
        }
        return ret;
    }

    internal::OpKernelBin* GetBin() override { return nullptr; }

    // The task contract requires launchState to be address-independent, so
    // repeatability only depends on the argument storages, same as the AI CPU
    // path.
    bool CheckRepeatable(const std::unordered_map<const aclStorage*, const aclStorage*>& relation,
                         const std::vector<const aclStorage*>& oriStorage) override
    {
        if (args_ == nullptr) {
            return false;
        }
        LauncherRepeatableChecker checker(relation, oriStorage);
        return checker.CheckLauncherRepeatable(args_);
    }

private:
    void CheckOverflowDump()
    {
        aclmdlRICaptureStatus status;
        aclmdlRI captureMdl;
        if ((aclmdlRICaptureGetInfo(executor_->GetStream(), &status, &captureMdl) == ACL_SUCCESS) &&
            (status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE)) {
            OP_LOGI("No need to perform overflow check in the capture scenario.");
            return;
        }
        (void)op::internal::OverflowDumpProcess(args_, const_cast<aclOpExecutor*>(executor_), executor_->GetStream(),
                                                opLogInfo_);
    }

    DirectInvokeLaunchFn launch_;
    DirectInvokeDestroyFn destroy_;
    void* launchState_;
    OpArgContext* args_;
};

} // namespace

aclnnStatus AddDirectInvokeTask([[maybe_unused]] const char* l0Name, uint32_t opType, DirectInvokePrepareFn prepare,
                                DirectInvokeLaunchFn launch, DirectInvokeDestroyFn destroy, aclOpExecutor* executor,
                                OpArgContext* args)
{
    if (prepare == nullptr || launch == nullptr || destroy == nullptr || executor == nullptr || args == nullptr) {
        DestroyOpArgContext(args);
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "DirectInvoke callbacks, executor and arguments must not be null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (opType == 0U || opType >= OpTypeDict::GetAllOpTypeSize() || !IsValidArgs(*args)) {
        DestroyOpArgContext(args);
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid DirectInvoke task or arguments.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    DirectInvokeWorkspaceRequest workspaceRequest{nullptr, 0U, 0U};
    void* launchState = nullptr;
    aclnnStatus status = prepare(args, &workspaceRequest, &launchState);
    if (status != ACLNN_SUCCESS) {
        if (launchState != nullptr) {
            destroy(launchState);
        }
        DestroyOpArgContext(args);
        return status;
    }

    status = AppendRequestedWorkspaces(executor, workspaceRequest, args);
    if (status != ACLNN_SUCCESS) {
        if (launchState != nullptr) {
            destroy(launchState);
        }
        DestroyOpArgContext(args);
        return status;
    }

    // The constructor only stores pointers and cannot throw, so nothrow new
    // fully covers launcher creation failures.
    auto* launcher = new (std::nothrow)
        DirectInvokeKernelLauncher(opType, executor, launch, destroy, launchState, args);
    if (launcher == nullptr) {
        if (launchState != nullptr) {
            destroy(launchState);
        }
        DestroyOpArgContext(args);
        return ACLNN_ERR_INNER;
    }

    // Tasks with dynamic output shapes must not be cached, same as the AI
    // Core path (CreatAiCoreKernelLauncher abandons the cache for them).
    const bool includeOutShape = args->ContainsOpArgType(OP_OUTSHAPE_ARG);
    status = RegisterKernelLauncher(opType, executor, args, launcher, includeOutShape);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    // Reuse the prepared state through explicit executor repeat, but execute
    // the producer callback on every run. Direct runtime dispatches do not
    // populate OpExecCache's task queue, so device-task replay must not bypass
    // Launch(). Dynamic output shapes still prohibit repeat altogether.
    executor->AbandonCache(includeOutShape);
    return ACLNN_SUCCESS;
}

} // namespace op
