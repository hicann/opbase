/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cpu_context.h"
#include "aicpu_context.h"
#include "cpu_node_def.h"
#include "cpu_kernel_utils.h"
#include "device.h"
#include "log.h"
#include "proto/cpu_attr.pb.h"
#include "proto/cpu_node_def.pb.h"
#include "proto/cpu_tensor.pb.h"
#include "sharder.h"
#include "status.h"
#include "securec.h"
#include "attr_value_impl.h"
#include "tensor_impl.h"
#include "cpu_attr_value.h"
#include "cpu_tensor.h"

namespace aicpu {
CpuKernelContext::CpuKernelContext(DeviceType type)
{
    Device* device = new (std::nothrow) Device(type);
    if (device != nullptr) {
        device_.reset(device);
    }
}

uint32_t CpuKernelContext::Init(NodeDef* nodeDef)
{
    KERNEL_CHECK_NULLPTR(nodeDef, KERNEL_STATUS_PARAM_INVALID, "Node def is null.")
    op_ = nodeDef->GetOpType();
    KERNEL_LOG_DEBUG("Construct the ctx of the op[%s] begin.", op_.c_str());
    for (int32_t i = 0; i < nodeDef->InputsSize(); i++) {
        auto input = nodeDef->MutableInputs(i);
        KERNEL_CHECK_NULLPTR(
            input, KERNEL_STATUS_PARAM_INVALID, "Get input[%d] tensor failed in op[%s].", i, op_.c_str())
        inputs_.emplace_back(std::move(input));
    }

    for (int32_t i = 0; i < nodeDef->OutputsSize(); i++) {
        auto output = nodeDef->MutableOutputs(i);
        KERNEL_CHECK_NULLPTR(
            output, KERNEL_STATUS_PARAM_INVALID, "Get output[%d] tensor failed in op[%s].", i, op_.c_str())
        outputs_.emplace_back(std::move(output));
    }

    auto attrMap = nodeDef->Attrs();
    for (auto iter = attrMap.begin(); iter != attrMap.end(); ++iter) {
        auto attr_value_ptr = iter->second;
        KERNEL_CHECK_NULLPTR(
            attr_value_ptr, KERNEL_STATUS_PARAM_INVALID, "Get attr[%s] failed in op[%s].", iter->first.c_str(),
            op_.c_str())
        auto ret = attrs_.insert(std::make_pair(iter->first, std::move(attr_value_ptr)));
        if (!ret.second) {
            KERNEL_LOG_ERROR("Insert attr[%s] failed in op[%s].", iter->first.c_str(), op_.c_str());
            return KERNEL_STATUS_INNER_ERROR;
        }
    }
    workspace_size_ = 0UL;
    workspace_addr_ = 0UL;

    KERNEL_LOG_DEBUG("Construct the ctx of the op[%s] succcess.", op_.c_str());
    return KERNEL_STATUS_OK;
}

uint32_t CpuKernelContext::Init(void* nodeDef)
{
    KERNEL_CHECK_NULLPTR(nodeDef, KERNEL_STATUS_PARAM_INVALID, "Node def is null.")
    aicpuops::NodeDef* node_def = reinterpret_cast<aicpuops::NodeDef*>(nodeDef);
    op_ = node_def->op();
    KERNEL_LOG_INFO("Construct the ctx of the op[%s] begin.", op_.c_str());
    for (int32_t i = 0; i < node_def->inputs_size(); i++) {
        aicpuops::Tensor* aicpuops_input = node_def->mutable_inputs(i);
        KERNEL_CHECK_NULLPTR(
            aicpuops_input, KERNEL_STATUS_PARAM_INVALID, "Protobuf node def mutable inputs[%d] tensor is nullptr.", i)
        TensorImpl* impl_input = new (std::nothrow) TensorImpl(aicpuops_input);
        KERNEL_CHECK_NULLPTR(impl_input, KERNEL_STATUS_PARAM_INVALID, "Create TensorImpl failed.")
        auto input = CpuKernelUtils::CreateTensor(impl_input);
        if (input == nullptr) {
            KERNEL_LOG_ERROR("Create CreateTensor failed.");
            delete impl_input;
            return KERNEL_STATUS_PARAM_INVALID;
        }
        inputs_.emplace_back(std::move(input));
    }

    for (int32_t i = 0; i < node_def->outputs_size(); i++) {
        aicpuops::Tensor* aicpuops_output = node_def->mutable_outputs(i);
        KERNEL_CHECK_NULLPTR(
            aicpuops_output, KERNEL_STATUS_PARAM_INVALID, "Protobuf node def mutable outputs[%d] tensor is nullptr.", i)
        TensorImpl* impl_output = new (std::nothrow) TensorImpl(aicpuops_output);
        KERNEL_CHECK_NULLPTR(impl_output, KERNEL_STATUS_PARAM_INVALID, "Create TensorImpl failed.")
        auto output = CpuKernelUtils::CreateTensor(impl_output);
        if (output == nullptr) {
            KERNEL_LOG_ERROR("Create CreateTensor failed.");
            delete impl_output;
            return KERNEL_STATUS_PARAM_INVALID;
        }
        outputs_.emplace_back(std::move(output));
    }

    auto attr_map = node_def->mutable_attrs();
    KERNEL_CHECK_NULLPTR(attr_map, KERNEL_STATUS_PARAM_INVALID, "Protobuf mutable attrs is null")
    for (auto iter = attr_map->begin(); iter != attr_map->end(); ++iter) {
        aicpuops::AttrValue* attr = &(iter->second);
        AttrValueImpl* impl = new (std::nothrow) AttrValueImpl(attr);
        KERNEL_CHECK_NULLPTR(impl, KERNEL_STATUS_PARAM_INVALID, "Create AttrValueImpl failed.")
        auto attr_value = CpuKernelUtils::CreateAttrValue(impl);
        if (attr_value == nullptr) {
            KERNEL_LOG_ERROR("Create CreateAttrValue failed.");
            delete impl;
            return KERNEL_STATUS_PARAM_INVALID;
        }
        auto ret = attrs_.insert(std::make_pair(iter->first, std::move(attr_value)));
        if (!ret.second) {
            KERNEL_LOG_ERROR("Insert attr[%s] failed in op[%s].", iter->first.c_str(), op_.c_str());
            return KERNEL_STATUS_INNER_ERROR;
        }
    }
    workspace_size_ = 0UL;
    workspace_addr_ = 0UL;

    KERNEL_LOG_DEBUG("Construct the ctx of the op[%s] succcess.", op_.c_str());
    return KERNEL_STATUS_OK;
}

/*
 * get op type.
 * @return string: op type
 */
std::string CpuKernelContext::GetOpType() const { return op_; }

/*
 * get input tensor.
 * @return Tensor *: not null->success, null->failed
 */
Tensor* CpuKernelContext::Input(uint32_t index) const
{
    if (index >= inputs_.size()) {
        KERNEL_LOG_WARN(
            "Input index[%u] should be less than input tensors total "
            "size[%zu].",
            index, inputs_.size());
        return nullptr;
    }

    return inputs_[index].get();
}

/*
 * get output tensor.
 * @return Tensor *: not null->success, null->failed
 */
Tensor* CpuKernelContext::Output(uint32_t index) const
{
    if (index >= outputs_.size()) {
        KERNEL_LOG_WARN(
            "Output index[%u] should be less than output tensors total "
            "size[%zu].",
            index, outputs_.size());
        return nullptr;
    }

    return outputs_[index].get();
}

/*
 * get attr.
 * @return AttrValue *: not null->success, null->failed
 */
AttrValue* CpuKernelContext::GetAttr(std::string name) const
{
    auto it = attrs_.find(name);
    if (it == attrs_.end()) {
        KERNEL_LOG_WARN("Attr[%s] is not exist.", name.c_str());
        return nullptr;
    }

    return (it->second).get();
}

/*
 * get input size.
 * @return uint32_t: input size
 */
uint32_t CpuKernelContext::GetInputsSize() const { return inputs_.size(); }

/*
 * get output size.
 * @return uint32_t: output size
 */
uint32_t CpuKernelContext::GetOutputsSize() const { return outputs_.size(); }

} // namespace aicpu
