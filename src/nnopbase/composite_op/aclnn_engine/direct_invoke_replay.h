/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Internal cache snapshot for DirectInvoke tasks.
#ifndef OPDEV_DIRECT_INVOKE_REPLAY_H
#define OPDEV_DIRECT_INVOKE_REPLAY_H

#include <memory>
#include "thread_local_context.h"
#include "opdev/direct_invoke_task.h"
#include "opdev/op_arg_def.h"
#include "opdev/op_cache.h"

namespace op {
namespace internal {

// Owns argument values independently of the original executor and caller.
class DirectInvokeReplay {
public:
    explicit DirectInvokeReplay(std::shared_ptr<void> state) : state_(std::move(state)) {}

    bool Init(const OpArgContext& source, OpExecCache& cache)
    {
        for (size_t kind = 0; kind < OP_ARG_TYPE_NUM; ++kind) {
            const auto& src = source.argLists[kind];
            auto& values = values_[kind];
            values.resize(src.count);
            args_.argLists[kind] = OpArgList(values.data(), values.size());
            args_.argLists[kind].argType = static_cast<int>(kind);
            for (size_t i = 0; i < src.count; ++i) {
                if (!CopyArg(src.args[i], values[i], cache)) return false;
            }
        }
        return true;
    }

    // Only a raw object pointer and address slots enter the byte-copied buffer.
    // The owning shared_ptr remains in OpExecCache, independent of buffer relocation.
    static bool SaveToCache(OpExecCache& cache, std::shared_ptr<DirectInvokeReplay> replay,
                            DirectInvokeLaunchFn launch, const OpLogInfo& logInfo = {},
                            const ProfilingInfoId& profilingInfo = {})
    {
        replay->launch_ = launch;
        replay->logInfo_ = logInfo;
        replay->profilingInfo_ = profilingInfo;
        cache.KeepResource(replay);
        size_t offset = 0;
        size_t capacity = 0;
        cache.NewLaunchCache(&offset, &capacity, RunFromCache, op::DIRECT_INVOKE);
        void* record = cache.AddLaunchData(sizeof(DirectInvokeReplay*));
        if (record == nullptr) return false;
        *PtrCastTo<DirectInvokeReplay*>(record) = replay.get();
        for (const auto& binding : replay->bindings_) {
            AddrRule baseRule = binding.second;
            // Offsets remain in the snapshot. Slots carry storage/workspace bases.
            baseRule.l2TensorOffset = 0;
            baseRule.workspaceOffset = 0;
            if (cache.AddLaunchAddress(baseRule) == nullptr) return false;
        }
        return true;
    }

    static aclnnStatus RunFromCache(aclrtStream stream, void* data)
    {
        if (data == nullptr) return ACLNN_ERR_PARAM_NULLPTR;
        auto* replay = *PtrCastTo<DirectInvokeReplay*>(data);
        if (replay == nullptr || replay->launch_ == nullptr) return ACLNN_ERR_INNER;
        auto& tls = GetThreadLocalContext();
        tls.logInfo_.l0Name = replay->logInfo_.l0Name;
        tls.profilingInfoId_ = replay->profilingInfo_;
        auto* slots = static_cast<char*>(data) + sizeof(DirectInvokeReplay*);
        for (auto& binding : replay->bindings_) {
            void* base = *PtrCastTo<void*>(slots);
            slots += sizeof(void*);
            binding.first->SetStorageAddr(base);
            if (binding.second.isWorkspace) {
                auto& tensorData = binding.first->GetTensor()->MutableTensorData();
                tensorData.SetAddr(static_cast<char*>(base) + binding.second.workspaceOffset, nullptr);
                tensorData.SetSize(op::CalcShapeBytes(binding.first->Size(), binding.first->GetDataType()));
            }
        }
        return replay->launch_(replay->state_.get(), &replay->args_, stream);
    }

private:
    template <typename T, typename D>
    T* Own(T* value, D destroy)
    {
        if (value == nullptr) throw std::bad_alloc();
        std::shared_ptr<void> owner(value, destroy);
        resources_.push_back(std::move(owner));
        return value;
    }

    aclTensor* CopyTensor(const aclTensor* tensor, OpExecCache& cache)
    {
        if (tensor == nullptr) return nullptr;
        auto found = tensorCopies_.find(tensor);
        if (found != tensorCopies_.end()) return found->second;
        AddrRule rule;
        if (cache.RecordAddrRule(tensor, rule) != ACLNN_SUCCESS) {
            valid_ = false;
            return nullptr;
        }
        const auto& shape = tensor->GetViewShape();
        const auto& storage = tensor->GetStorageShape();
        auto* copy = Own(aclCreateTensor(shape.GetDimNum() == 0 ? nullptr : &shape[0], shape.GetDimNum(),
            static_cast<aclDataType>(tensor->GetDataType()), tensor->GetViewStrides().data(),
            tensor->GetViewOffset(), static_cast<aclFormat>(tensor->GetViewFormat()),
            storage.GetDimNum() == 0 ? nullptr : &storage[0], storage.GetDimNum(), nullptr), aclDestroyTensor);
        copy->SetOriginalShape(tensor->GetOriginalShape());
        copy->SetOriginalFormat(tensor->GetOriginalFormat());
        copy->SetStorageFormat(tensor->GetStorageFormat());
        copy->SetStorageOffset(tensor->GetStorageOffset());
        copy->SetFromWorkspace(rule.isWorkspace);
        copy->SetWorkspaceOffset(rule.workspaceOffset);
        bindings_.emplace_back(copy, rule);
        tensorCopies_[tensor] = copy;
        return copy;
    }

    bool CopyArg(const OpArg& source, OpArg& target, OpExecCache& cache)
    {
        target = source;
        target.value.deleter = nullptr;
        switch (source.type) {
            case OpArgType::OPARG_ACLTENSOR:
                target.value.data.pointer = CopyTensor(static_cast<const aclTensor*>(source.value.data.pointer), cache);
                break;
            case OpArgType::OPARG_ACLTENSOR_LIST: {
                const auto* list = static_cast<const aclTensorList*>(source.value.data.pointer);
                if (list == nullptr) break;
                std::vector<const aclTensor*> copies;
                for (size_t i = 0; i < list->Size(); ++i) copies.push_back(CopyTensor((*list)[i], cache));
                target.value.data.pointer = Own(aclCreateTensorList(copies.data(), copies.size()),
                    [](aclTensorList* value) {
                        // Elements have separate owners, including repeated aliases.
                        for (size_t i = 0; i < value->Size(); ++i) (*value)[i] = nullptr;
                        aclDestroyTensorList(value);
                    });
                break;
            }
            case OpArgType::OPARG_ACLSCALAR: {
                const auto* value = static_cast<const aclScalar*>(source.value.data.pointer);
                if (value == nullptr) break;
                target.value.data.pointer = Own(aclCreateScalar(const_cast<void*>(value->GetData()),
                    static_cast<aclDataType>(value->GetDataType())), aclDestroyScalar);
                break;
            }
            case OpArgType::OPARG_INT_LIST: {
                const auto* value = static_cast<const aclIntArray*>(source.value.data.pointer);
                if (value == nullptr) break;
                target.value.data.pointer = Own(aclCreateIntArray(value->GetData(), value->Size()), aclDestroyIntArray);
                break;
            }
            case OpArgType::OPARG_FLOAT_LIST: {
                const auto* value = static_cast<const aclFloatArray*>(source.value.data.pointer);
                if (value == nullptr) break;
                target.value.data.pointer = Own(aclCreateFloatArray(value->GetData(), value->Size()), aclDestroyFloatArray);
                break;
            }
            case OpArgType::OPARG_BOOL_LIST: {
                const auto* value = static_cast<const aclBoolArray*>(source.value.data.pointer);
                if (value == nullptr) break;
                target.value.data.pointer = Own(aclCreateBoolArray(value->GetData(), value->Size()), aclDestroyBoolArray);
                break;
            }
            case OpArgType::OPARG_STRING: {
                if (source.value.data.pointer == nullptr) break;
                auto value = std::make_shared<std::string>(static_cast<const char*>(source.value.data.pointer));
                target.value.data.pointer = const_cast<char*>(value->c_str());
                resources_.push_back(std::move(value));
                break;
            }
            // Inline values are already copied by target = source; no deep copy is needed.
            case OpArgType::OPARG_BOOL:
            case OpArgType::OPARG_INT:
            case OpArgType::OPARG_UINT:
            case OpArgType::OPARG_FLOAT:
            case OpArgType::OPARG_DOUBLE:
            case OpArgType::OPARG_DATATYPE:
            case OpArgType::OPARG_IMPLMODE:
                break;
            default:
                return false;
        }
        return valid_;
    }

    DirectInvokeLaunchFn launch_{nullptr};
    OpLogInfo logInfo_;
    ProfilingInfoId profilingInfo_;
    std::shared_ptr<void> state_;
    OpArgContext args_;
    std::array<std::vector<OpArg>, OP_ARG_TYPE_NUM> values_;
    std::vector<std::shared_ptr<void>> resources_;
    std::unordered_map<const aclTensor*, aclTensor*> tensorCopies_;
    std::vector<std::pair<aclTensor*, AddrRule>> bindings_;
    bool valid_{true};
};

} // namespace internal
} // namespace op
#endif
