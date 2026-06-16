/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "node_def_impl.h"

#include "attr_value_impl.h"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "status.h"
#include "tensor_impl.h"

namespace aicpu {
/*
 * parse parameter from string.
 */
bool NodeDefImpl::ParseFromString(const std::string& str)
{
    if (!nodedef_->ParseFromString(str)) {
        KERNEL_LOG_ERROR("ParseFromString failed");
        return false;
    }

    return true;
}

/*
 * serialize string to node def.
 */
bool NodeDefImpl::SerializeToString(std::string& str) const
{
    if (!nodedef_->SerializeToString(&str)) {
        KERNEL_LOG_ERROR("SerializeToString failed");
        return false;
    }

    return true;
}

/*
 * set op type to node def.
 */
void NodeDefImpl::SetOpType(const std::string& op) { nodedef_->set_op(op); }

/*
 * get op type of node def.
 */
std::string NodeDefImpl::GetOpType() const { return nodedef_->op(); }

/*
 * add input tensor to node def.
 */
std::shared_ptr<Tensor> NodeDefImpl::AddInputs()
{
    auto tensor = nodedef_->add_inputs();
    if (tensor == nullptr) {
        KERNEL_LOG_ERROR("Protobuf node def add tensor is nullptr.");
        return std::shared_ptr<Tensor>(nullptr);
    }

    TensorImpl* impl = new (std::nothrow) TensorImpl(tensor);
    if (impl == nullptr) {
        KERNEL_LOG_ERROR("Create TensorImpl failed.");
        return std::shared_ptr<Tensor>(nullptr);
    }

    auto aicpu_tensor = CpuKernelUtils::CreateTensor(impl);
    if (aicpu_tensor == nullptr) {
        delete impl;
    }
    return aicpu_tensor;
}

/*
 * add output tensor to node def.
 */
std::shared_ptr<Tensor> NodeDefImpl::AddOutputs()
{
    auto tensor = nodedef_->add_outputs();
    if (tensor == nullptr) {
        KERNEL_LOG_ERROR("Protobuf node def add tensor is nullptr.");
        return std::shared_ptr<Tensor>(nullptr);
    }

    TensorImpl* impl = new (std::nothrow) TensorImpl(tensor);
    if (impl == nullptr) {
        KERNEL_LOG_ERROR("Create TensorImpl failed.");
        return std::shared_ptr<Tensor>(nullptr);
    }

    auto aicpu_tensor = CpuKernelUtils::CreateTensor(impl);
    if (aicpu_tensor == nullptr) {
        delete impl;
    }
    return aicpu_tensor;
}

/*
 * add attr to node def.
 */
bool NodeDefImpl::AddAttrs(const std::string& name, const AttrValue* attr)
{
    if (attr == nullptr) {
        KERNEL_LOG_ERROR("Attr is null.");
        return false;
    }

    auto attrs = nodedef_->mutable_attrs();
    KERNEL_CHECK_NULLPTR(attrs, false, "Protobuf mutable attrs is null")
    auto impl = CpuKernelUtils::GetImpl(attr);
    auto pair =
        attrs->insert(google::protobuf::Map<std::string, aicpuops::AttrValue>::value_type(name, *(impl->GetProto())));
    if (!pair.second) {
        KERNEL_LOG_ERROR("Nodedef insert attr %s to nodeDef failed.", name.c_str());
        return false;
    }
    return true;
}

/*
 * get input tensor size of node def.
 */
int32_t NodeDefImpl::InputsSize() const { return nodedef_->inputs_size(); }

/*
 * get output tensor size of node def.
 */
int32_t NodeDefImpl::OutputsSize() const { return nodedef_->outputs_size(); }

/*
 * get input tensor of node def.
 */
std::shared_ptr<Tensor> NodeDefImpl::MutableInputs(int32_t index) const
{
    if ((index >= InputsSize()) || (index < 0)) {
        KERNEL_LOG_ERROR(
            "Index[%d] should be less than input tensors size[%d] and noe less than "
            "0.",
            index, InputsSize());
        return std::shared_ptr<Tensor>(nullptr);
    }

    auto tensor = nodedef_->mutable_inputs(index);
    if (tensor == nullptr) {
        KERNEL_LOG_ERROR("Protobuf node def mutable inputs[%d] tensor is nullptr.", index);
        return std::shared_ptr<Tensor>(nullptr);
    }

    TensorImpl* impl = new (std::nothrow) TensorImpl(tensor);
    if (impl == nullptr) {
        KERNEL_LOG_ERROR("Create TensorImpl failed.");
        return std::shared_ptr<Tensor>(nullptr);
    }

    auto aicpu_tensor = CpuKernelUtils::CreateTensor(impl);
    if (aicpu_tensor == nullptr) {
        delete impl;
    }
    return aicpu_tensor;
}

/*
 * get output tensor of node def.
 */
std::shared_ptr<Tensor> NodeDefImpl::MutableOutputs(int32_t index) const
{
    if ((index >= OutputsSize()) || (index < 0)) {
        KERNEL_LOG_ERROR(
            "Index[%d] should be less than output tensors size[%d] and noe less than "
            "0.",
            index, OutputsSize());
        return std::shared_ptr<Tensor>(nullptr);
    }

    auto tensor = nodedef_->mutable_outputs(index);
    if (tensor == nullptr) {
        KERNEL_LOG_ERROR("Protobuf node def mutable outputs[%d] tensor is nullptr.", index);
        return std::shared_ptr<Tensor>(nullptr);
    }

    TensorImpl* impl = new (std::nothrow) TensorImpl(tensor);
    if (impl == nullptr) {
        KERNEL_LOG_ERROR("Create TensorImpl failed.");
        return std::shared_ptr<Tensor>(nullptr);
    }

    auto aicpu_tensor = CpuKernelUtils::CreateTensor(impl);
    if (aicpu_tensor == nullptr) {
        delete impl;
    }
    return aicpu_tensor;
}

/*
 * get attr of node def.
 */
std::unordered_map<std::string, std::shared_ptr<AttrValue>> NodeDefImpl::Attrs() const
{
    std::unordered_map<std::string, std::shared_ptr<AttrValue>> ret;
    auto attrs_map = nodedef_->mutable_attrs();
    KERNEL_CHECK_NULLPTR(attrs_map, ret, "Protobuf mutable attrs is null")

    for (auto it = attrs_map->begin(); it != attrs_map->end(); ++it) {
        aicpuops::AttrValue* attr = &(it->second);
        AttrValueImpl* impl = new (std::nothrow) AttrValueImpl(attr);
        if (impl == nullptr) {
            KERNEL_LOG_WARN("Create AttrValueImpl failed.");
        }

        auto attr_value = CpuKernelUtils::CreateAttrValue(impl);
        if (attr_value == nullptr) {
            KERNEL_LOG_WARN("Create CreateAttrValue failed.");
            delete impl;
        }
        (void)ret.insert(std::make_pair(it->first, attr_value));
    }

    return ret;
}
} // namespace aicpu
