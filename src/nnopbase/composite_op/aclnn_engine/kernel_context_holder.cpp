/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_context_holder.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "kernel_utils.h"
#include "opdev/op_errno.h"

namespace op::internal {

void KernelContextHolder::BuildComputeNodeInfo()
{
    computeNodeInfoSize_ = sizeof(ComputeNodeInfo)
        + sizeof(AnchorInstanceInfo) * MAX_OP_ARG_NUM
        + sizeof(CompileTimeTensorDesc) * MAX_OP_ARG_NUM
        + sizeof(RuntimeAttrsDef) + ATTR_CAPACITY;
    computeNodeInfo_ = static_cast<ComputeNodeInfo *>(malloc(computeNodeInfoSize_));
    if (computeNodeInfo_ == nullptr) {
        return;
    }
    (void)memset_s(computeNodeInfo_, computeNodeInfoSize_, 0, computeNodeInfoSize_);
    anchorInfo_ = PtrCastTo<AnchorInstanceInfo>(&computeNodeInfo_->place_holder);
}

void KernelContextHolder::BuildOpInOutArg()
{
    size_t sz = sizeof(AsyncAnyValue) * MAX_OP_ARG_NUM;
    opInArg_ = static_cast<AsyncAnyValue *>(malloc(sz));
    if (opInArg_ == nullptr) {
        return;
    }
    (void)memset_s(opInArg_, sz, 0, sz);
}

void KernelContextHolder::UpdateKernelExtendInfo(const char *kernelType, const char *kernelName)
{
    kernelExtendInfo_.kernel_type_ = kernelType;
    kernelExtendInfo_.kernel_name_ = kernelName;
}

void KernelContextHolder::UpdateCompileDescOffset(size_t irInputNum)
{
    compileDesc_ = PtrCastTo<CompileTimeTensorDesc>(anchorInfo_ + irInputNum);
}

void KernelContextHolder::UpdateAttrDefOffset(size_t inoutNum, size_t attrNum)
{
    size_t attrOffset = sizeof(CompileTimeTensorDesc) * inoutNum;
    attrDef_ = PtrCastTo<RuntimeAttrsDef>(PtrShift(compileDesc_, attrOffset));
    attrDef_->attr_num = attrNum;
    attrDataStart_ = PtrCastTo<uint8_t>(attrDef_ + 1) + sizeof(size_t) * attrNum;
#ifdef DEBUG
    OP_LOGD("Adjust attr offset. [%zu]. inout num %zu. attr num: %zu,  %zu",
            attrOffset, inoutNum, attrNum, PtrOffset(computeNodeInfo_, attrDef_));
#endif
}

aclnnStatus KernelContextHolder::UpdateInputArg(size_t idx, const aclTensor *tensor)
{
    if (tensor == nullptr) {
        // Caution: RT2.0 seems not support null op input in the middle or op args
        OP_LOGI("Update Input Arg Tensor is NULL: [%zu]", idx);
        anchorInfo_[idx].instance_start_ = inputNum_;
        anchorInfo_[idx].instantiation_num_ = 0;
        return ACLNN_SUCCESS;
    }

#ifdef DEBUG
    OP_LOGD("Update Input Arg Tensor: [%zu]. %s", idx, tensor->ToString().GetString());
    if (tensor->GetPlacement() == gert::kOnHost) {
        OP_LOGD("Update Input arg HOST aclTensor. Size: %zu", tensor->Size());
        for (auto i = 0; i < tensor->Size(); i++) {
            uint64_t *p = (static_cast<uint64_t *>(tensor->GetData()) + i);
            OP_LOGD("Update Input arg HOST aclTensor Data: %lu", *p);
        }
    }
#endif
    if (inputNum_ >= MAX_OP_ARG_NUM) {
        OP_LOGE(ACLNN_ERR_INNER, "Too many input args, %zu", MAX_OP_ARG_NUM);
        return ACLNN_ERR_INNER;
    }
    anchorInfo_[idx].instance_start_ = inputNum_;
    anchorInfo_[idx].instantiation_num_ = 1;
    compileDesc_[inputNum_].data_type_ = tensor->GetDataType();
    compileDesc_[inputNum_].storage_format_.SetOriginFormat(tensor->GetOriginalFormat());
    compileDesc_[inputNum_].storage_format_.SetStorageFormat(tensor->GetStorageFormat());
    opInArg_[inputNum_].data.pointer = tensor->GetTensor();
    inputNum_++;
    return ACLNN_SUCCESS;
}

aclnnStatus KernelContextHolder::UpdateInputArg(size_t idx, const aclTensorList *tensorList)
{
    if (tensorList == nullptr || tensorList->Size() == 0) {
        OP_LOGD("Update Input Arg Tensor List is Null Or Empty");
        anchorInfo_[idx].instance_start_ = inputNum_;
        anchorInfo_[idx].instantiation_num_ = 0;
        return ACLNN_SUCCESS;
    }
    size_t start = inputNum_;
    for (size_t i = 0; i < tensorList->Size(); i++) {
        if (UpdateInputArg(idx, (*tensorList)[i]) != ACLNN_SUCCESS) {
            return ACLNN_ERR_INNER;
        };
    }
    if (inputNum_ == start) {
        OP_LOGD("Update Input Arg Tensor List contains only NULL tensor");
    }
    anchorInfo_[idx].instance_start_ = start;
    anchorInfo_[idx].instantiation_num_ = inputNum_ - start;
    return ACLNN_SUCCESS;
}

aclnnStatus KernelContextHolder::UpdateInputArg(size_t idx, OpArg &arg)
{
    switch (arg.type) {
        case OpArgType::OPARG_ACLTENSOR:
            return UpdateInputArg(idx, reinterpret_cast<aclTensor *>(arg->pointer));
        case OpArgType::OPARG_ACLTENSOR_LIST:
            return UpdateInputArg(idx, reinterpret_cast<aclTensorList *>(arg->pointer));
        default:
            OP_LOGE(ACLNN_ERR_INNER, "invalid input arg type %d.", static_cast<int>(arg.type));
            return ACLNN_ERR_INNER;
    }
}

aclnnStatus KernelContextHolder::UpdateOutputArg(size_t idx, const aclTensor *tensor)
{
    if (tensor == nullptr) {
        OP_LOGI("Update Output Arg Tensor List is Null [%zu]", idx);
        return ACLNN_SUCCESS;
    }
    if (inputNum_ + outputNum_ >= MAX_OP_ARG_NUM) {
        OP_LOGE(ACLNN_ERR_INNER, "Too many input/output args, allow:%zu. input: %zu, output: %zu",
                MAX_OP_ARG_NUM, inputNum_, outputNum_);
        return ACLNN_ERR_INNER;
    }
#ifdef DEBUG
    OP_LOGD("Update Output Arg Tensor: [%zu]. %s", idx, tensor->ToString().GetString());
#endif
    compileDesc_[inputNum_ + outputNum_].data_type_ = tensor->GetDataType();
    compileDesc_[inputNum_ + outputNum_].storage_format_.SetOriginFormat(tensor->GetOriginalFormat());
    compileDesc_[inputNum_ + outputNum_].storage_format_.SetStorageFormat(tensor->GetStorageFormat());
    opInArg_[inputNum_ + outputNum_].data.pointer = tensor->GetTensor();
    outputNum_++;
    return ACLNN_SUCCESS;
}

aclnnStatus KernelContextHolder::UpdateOutputArg(size_t idx, const aclTensorList *tensorList)
{
    if (tensorList == nullptr || tensorList->Size() == 0) {
        OP_LOGD("Update Output Arg Tensor List is Null Or Empty");
        return ACLNN_SUCCESS;
    }
    for (size_t i = 0; i < tensorList->Size(); i++) {
        if (UpdateOutputArg(idx, (*tensorList)[i]) != ACLNN_SUCCESS) {
            return ACLNN_ERR_INNER;
        };
    }
    return ACLNN_SUCCESS;
}

aclnnStatus KernelContextHolder::UpdateOutputArg(size_t idx, OpArg &arg)
{
    switch (arg.type) {
        case OpArgType::OPARG_ACLTENSOR:
            return UpdateOutputArg(idx, reinterpret_cast<aclTensor *>(arg->pointer));
        case OpArgType::OPARG_ACLTENSOR_LIST:
            return UpdateOutputArg(idx, reinterpret_cast<aclTensorList *>(arg->pointer));
        default:
            OP_LOGE(ACLNN_ERR_INNER, "invalid output arg type %d.", static_cast<int>(arg.type));
            return ACLNN_ERR_INNER;
    }
}

void KernelContextHolder::UpdateOutputArgIr(size_t idx, const aclTensor *tensor, size_t &seq) const
{
    if (tensor == nullptr) {
        // Caution: RT2.0 seems not support null op input in the middle or op args
        OP_LOGD("Update Input Arg Tensor is NULL: [%zu]", idx);
        outputAnchorInfo_[idx].instance_start_ = seq;
        outputAnchorInfo_[idx].instantiation_num_ = 0;
        return;
    }
    outputAnchorInfo_[idx].instance_start_ = seq++;
    outputAnchorInfo_[idx].instantiation_num_ = 1;
}

void KernelContextHolder::UpdateOutputArgIr(size_t idx, const aclTensorList *tensorList, size_t &seq) const
{
    if (tensorList == nullptr || tensorList->Size() == 0) {
        OP_LOGD("Update IR Output argList is Null Or Empty");
        outputAnchorInfo_[idx].instance_start_ = seq;
        outputAnchorInfo_[idx].instantiation_num_ = 0;
        return;
    }

    size_t start = seq;
    for (size_t i = 0; i < tensorList->Size(); i++) {
        UpdateOutputArgIr(idx, (*tensorList)[i], seq);
    }
    outputAnchorInfo_[idx].instance_start_ = start;
    outputAnchorInfo_[idx].instantiation_num_ = seq - start;
}

void KernelContextHolder::UpdateOutputArgIr(size_t idx, OpArg &arg, size_t &seq) const
{
    switch (arg.type) {
        case OpArgType::OPARG_ACLTENSOR:
            UpdateOutputArgIr(idx, reinterpret_cast<aclTensor *>(arg->pointer), seq);
            break;
        case OpArgType::OPARG_ACLTENSOR_LIST:
            UpdateOutputArgIr(idx, reinterpret_cast<aclTensorList *>(arg->pointer), seq);
            break;
        default:
            OP_LOGW("invalid output arg type %d.", static_cast<int>(arg.type));
            break;
    }
}

void KernelContextHolder::ResetComputeNodeInfo(const char *opType, size_t irInputNum, size_t irOutputNum)
{
    computeNodeInfo_->node_type_ = opType;
    computeNodeInfo_->node_name_ = opType;
    computeNodeInfo_->ir_inputs_num_ = irInputNum;
    computeNodeInfo_->attr_size_ = 0;
    computeNodeInfo_->ir_outputs_num_ = irOutputNum;
    inputNum_ = 0;
    outputNum_ = 0;
    attrNum_ = 0;
    irInputNum_ = irInputNum;
    irOutputNum_ = irOutputNum;
}

void KernelContextHolder::FinalizeComputeNodeInfo(size_t attrNum)
{
    computeNodeInfo_->inputs_num_ = inputNum_;
    computeNodeInfo_->outputs_num_ = outputNum_;
    attrNum_ = attrNum;
    UpdateAttrDefOffset(inputNum_ + outputNum_, attrNum_);
}

KernelContextHolder::~KernelContextHolder()
{
    FREE(computeNodeInfo_);
    FREE(opInArg_);
}

} // namespace op::internal
