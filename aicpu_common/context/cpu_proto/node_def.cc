/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cpu_node_def.h"
#include "node_def_impl.h"

namespace aicpu {
NodeDef::NodeDef(NodeDefImpl* impl) : impl_(impl) {}

/*
 * parse parameter from string.
 */
bool NodeDef::ParseFromString(const std::string& str) { return impl_->ParseFromString(str); }

/*
 * serialize string to node def.
 */
bool NodeDef::SerializeToString(std::string& str) const { return impl_->SerializeToString(str); }

/*
 * set op type to node def.
 */
void NodeDef::SetOpType(const std::string& op) { impl_->SetOpType(op); }

/*
 * get op type of node def.
 */
std::string NodeDef::GetOpType() const { return impl_->GetOpType(); }

/*
 * add input tensor to node def.
 */
std::shared_ptr<Tensor> NodeDef::AddInputs() { return impl_->AddInputs(); }

/*
 * add output tensor to node def.
 */
std::shared_ptr<Tensor> NodeDef::AddOutputs() { return impl_->AddOutputs(); }

/*
 * add attr to node def.
 */
bool NodeDef::AddAttrs(const std::string& name, const AttrValue* attr) { return impl_->AddAttrs(name, attr); }

/*
 * get input tensor size of node def.
 */
int32_t NodeDef::InputsSize() const { return impl_->InputsSize(); }

/*
 * get output tensor size of node def.
 */
int32_t NodeDef::OutputsSize() const { return impl_->OutputsSize(); }

/*
 * get input tensor of node def.
 */
std::shared_ptr<Tensor> NodeDef::MutableInputs(int32_t index) const { return impl_->MutableInputs(index); }

/*
 * get output tensor of node def.
 */
std::shared_ptr<Tensor> NodeDef::MutableOutputs(int32_t index) const { return impl_->MutableOutputs(index); }

/*
 * get attr of node def.
 */
std::unordered_map<std::string, std::shared_ptr<AttrValue> > NodeDef::Attrs() const { return impl_->Attrs(); }
} // namespace aicpu
