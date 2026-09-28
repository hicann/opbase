/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "folding.h"

#include <algorithm>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_attr.pb.h"
#include "cpu_context.h"
#include "cpu_kernel_register.h"
#include "cpu_node_def.pb.h"
#include "cpu_tensor.pb.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/tensor.h"
#include "graph/ascend_string.h"
#include "graph/custom_op.h"
#include "graph/operator_reg.h"
#include "graph/types.h"
#include "folding_utils.h"
#include "log.h"
#include "mmpa/mmpa_api.h"

namespace {
constexpr char kSymGetAllRegisteredOpTypesV2[] = "GetAllRegisteredOpTypesV2";
constexpr char kSymRunCpuKernelV2[] = "RunCpuKernelV2";

constexpr char kIrInputRequired[] = "required";
constexpr char kIrInputOptional[] = "optional";
constexpr char kIrInputDynamic[] = "dynamic";

constexpr char kIrOutputRequired[] = "required";
constexpr char kIrOutputDynamic[] = "dynamic";

constexpr char kVtInt[] = "VT_INT";
constexpr char kVtFloat[] = "VT_FLOAT";
constexpr char kVtBool[] = "VT_BOOL";
constexpr char kVtString[] = "VT_STRING";
constexpr char kVtTensor[] = "VT_TENSOR";
constexpr char kVtDataType[] = "VT_DATA_TYPE";
constexpr char kVtListInt[] = "VT_LIST_INT";
constexpr char kVtListFloat[] = "VT_LIST_FLOAT";
constexpr char kVtListBool[] = "VT_LIST_BOOL";
constexpr char kVtListString[] = "VT_LIST_STRING";
constexpr char kVtListDataType[] = "VT_LIST_DATA_TYPE";
constexpr char kVtListListInt[] = "VT_LIST_LIST_INT";

using IrDefPair = std::pair<ge::AscendString, ge::AscendString>;

using AttrValueMap = google::protobuf::Map<std::string, aicpuops::AttrValue>;
using GetAllRegisteredOpTypesV2Fn = std::vector<std::string> (*)();
using RunCpuKernelV2Fn = uint32_t (*)(aicpu::CpuKernelContext&);

struct V2ModuleBinding {
    void* handle = nullptr;
    GetAllRegisteredOpTypesV2Fn get_all_op_types = nullptr;
    RunCpuKernelV2Fn run_cpu_kernel = nullptr;
    std::string so_name;
};

std::vector<V2ModuleBinding> g_v2_bindings;
std::unordered_map<std::string, const V2ModuleBinding*> g_v2_op_index;
bool g_initialized = false;

constexpr int32_t kAttrSuccess = 0;
constexpr int32_t kAttrAbsent = -1;
constexpr int32_t kAttrUnsupported = -2;

void SetAicpuShape(const gert::Shape& src, const ge::Format format, aicpuops::Tensor* dst)
{
    auto* shape = dst->mutable_tensor_shape();
    if (shape == nullptr) {
        return;
    }
    shape->clear_dim();
    if ((src.GetDimNum() == 1U) && (src.GetDim(0) == ge::UNKNOWN_DIM_NUM)) {
        shape->set_unknown_rank(true);
        shape->set_data_format(static_cast<int32_t>(format));
        return;
    }
    shape->set_unknown_rank(false);
    for (size_t i = 0; i < src.GetDimNum(); ++i) {
        auto* dim = shape->add_dim();
        if (dim != nullptr) {
            dim->set_size(src.GetDim(i));
        }
    }
    shape->set_data_format(static_cast<int32_t>(format));
}

void ConvertGertToAicpuTensor(const gert::Tensor& src, const std::string& name, aicpuops::Tensor* dst)
{
    if (dst == nullptr) {
        return;
    }
    dst->set_name(name);
    dst->set_tensor_type(static_cast<int32_t>(src.GetDataType()));
    dst->set_data_ptr(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(const_cast<void*>(src.GetAddr()))));
    dst->set_data_size(static_cast<uint64_t>(src.GetSize()));
    SetAicpuShape(src.GetStorageShape(), src.GetStorageFormat(), dst);
}

template <typename T, typename AddValue>
int32_t AddListAttr(const gert::HostCpuOpExecutionContext* ctx, const size_t index, AddValue add_value,
                    aicpuops::AttrValue* attr_value)
{
    const auto* attrs = ctx->GetAttrs();
    if (attrs == nullptr) {
        KERNEL_LOG_WARN("Runtime attrs are null when reading list attr[%zu].", index);
        return -1;
    }
    const auto* values = attrs->GetAttrPointer<gert::TypedContinuousVector<T>>(index);
    if (values == nullptr) {
        KERNEL_LOG_WARN("List attr[%zu] has no value or has an unexpected runtime type.", index);
        return -1;
    }
    auto* array = attr_value->mutable_array();
    if (array == nullptr) {
        KERNEL_LOG_WARN("Create AICPU array value for attr[%zu] failed.", index);
        return -1;
    }
    const T* data = values->GetData();
    for (size_t i = 0; i < values->GetSize(); ++i) {
        add_value(array, data[i]);
    }
    return 0;
}

int32_t AddStringListAttr(const gert::HostCpuOpExecutionContext* ctx, const size_t index,
                          aicpuops::AttrValue* attr_value)
{
    const auto* attrs = ctx->GetAttrs();
    if (attrs == nullptr) {
        KERNEL_LOG_WARN("Runtime attrs are null when reading string-list attr[%zu].", index);
        return -1;
    }
    const auto* values = attrs->GetAttrPointer<gert::ContinuousVector>(index);
    if (values == nullptr) {
        KERNEL_LOG_WARN("String-list attr[%zu] has no value or has an unexpected runtime type.", index);
        return -1;
    }
    auto* array = attr_value->mutable_array();
    if (array == nullptr) {
        KERNEL_LOG_WARN("Create AICPU string array value for attr[%zu] failed.", index);
        return -1;
    }
    const char* value = static_cast<const char*>(values->GetData());
    for (size_t i = 0; i < values->GetSize(); ++i) {
        array->add_s(value);
        value += std::strlen(value) + 1U;
    }
    return 0;
}

int32_t AddListListIntAttr(const gert::HostCpuOpExecutionContext* ctx, const size_t index,
                           aicpuops::AttrValue* attr_value)
{
    const auto* attrs = ctx->GetAttrs();
    if (attrs == nullptr) {
        KERNEL_LOG_WARN("Runtime attrs are null when reading list-list-int attr[%zu].", index);
        return -1;
    }
    const auto* values = attrs->GetAttrPointer<gert::ContinuousVectorVector>(index);
    if (values == nullptr) {
        KERNEL_LOG_WARN("List-list-int attr[%zu] has no value or has an unexpected runtime type.", index);
        return -1;
    }
    auto* list_list = attr_value->mutable_list_list_int();
    if (list_list == nullptr) {
        KERNEL_LOG_WARN("Create AICPU list-list-int value for attr[%zu] failed.", index);
        return -1;
    }
    for (size_t i = 0; i < values->GetSize(); ++i) {
        const auto* src = values->Get(i);
        auto* dst = list_list->add_list_list_i();
        if ((src == nullptr) || (dst == nullptr)) {
            KERNEL_LOG_WARN("Read list-list-int attr[%zu] item[%zu] failed.", index, i);
            return -1;
        }
        const auto* data = static_cast<const int64_t*>(src->GetData());
        for (size_t j = 0; j < src->GetSize(); ++j) {
            dst->add_list_i(data[j]);
        }
    }
    return 0;
}

int32_t AddTensorAttr(const gert::HostCpuOpExecutionContext* ctx, const size_t index, aicpuops::AttrValue* attr_value)
{
    const auto* attrs = ctx->GetAttrs();
    if (attrs == nullptr) {
        KERNEL_LOG_WARN("Runtime attrs are null when reading tensor attr[%zu].", index);
        return -1;
    }
    const auto* tensor = attrs->GetAttrPointer<gert::Tensor>(index);
    if (tensor == nullptr) {
        KERNEL_LOG_WARN("Tensor attr[%zu] has no value or has an unexpected runtime type.", index);
        return -1;
    }
    ConvertGertToAicpuTensor(*tensor, "", attr_value->mutable_tensor());
    return 0;
}

template <typename T, typename SetValue>
int32_t AddScalarAttr(const gert::RuntimeAttrs* attrs, const size_t index, SetValue set_value)
{
    const auto* value = attrs->GetAttrPointer<T>(index);
    if (value == nullptr) {
        return kAttrAbsent;
    }
    set_value(*value);
    return kAttrSuccess;
}

int32_t AddScalarAttrToNodeDef(const gert::HostCpuOpExecutionContext* ctx, const size_t index,
                               const std::string& attr_type, aicpuops::AttrValue* attr_value)
{
    const auto* attrs = ctx->GetAttrs();
    if (attr_type == kVtInt) {
        return AddScalarAttr<int64_t>(attrs, index, [attr_value](const int64_t value) { attr_value->set_i(value); });
    }
    if (attr_type == kVtFloat) {
        return AddScalarAttr<float>(attrs, index, [attr_value](const float value) { attr_value->set_f(value); });
    }
    if (attr_type == kVtBool) {
        return AddScalarAttr<bool>(attrs, index, [attr_value](const bool value) { attr_value->set_b(value); });
    }
    if (attr_type == kVtString) {
        const auto* value = attrs->GetAttrPointer<char>(index);
        return value == nullptr ? kAttrAbsent : (attr_value->set_s(value), kAttrSuccess);
    }
    if (attr_type == kVtTensor) {
        return AddTensorAttr(ctx, index, attr_value);
    }
    if (attr_type == kVtDataType) {
        return AddScalarAttr<ge::DataType>(attrs, index, [attr_value](const ge::DataType value) {
            attr_value->set_type(static_cast<int32_t>(value));
        });
    }
    return kAttrUnsupported;
}

int32_t AddListAttrToNodeDef(const gert::HostCpuOpExecutionContext* ctx, const size_t index,
                             const std::string& attr_type, aicpuops::AttrValue* attr_value)
{
    if (attr_type == kVtListInt) {
        return AddListAttr<int64_t>(
            ctx, index, [](aicpuops::AttrValue::ArrayValue* a, const int64_t v) { a->add_i(v); }, attr_value);
    }
    if (attr_type == kVtListFloat) {
        return AddListAttr<float>(
            ctx, index, [](aicpuops::AttrValue::ArrayValue* a, const float v) { a->add_f(v); }, attr_value);
    }
    if (attr_type == kVtListBool) {
        return AddListAttr<bool>(
            ctx, index, [](aicpuops::AttrValue::ArrayValue* a, const bool v) { a->add_b(v); }, attr_value);
    }
    if (attr_type == kVtListString) {
        return AddStringListAttr(ctx, index, attr_value);
    }
    if (attr_type == kVtListDataType) {
        return AddListAttr<ge::DataType>(
            ctx, index,
            [](aicpuops::AttrValue::ArrayValue* a, const ge::DataType v) { a->add_type(static_cast<int32_t>(v)); },
            attr_value);
    }
    if (attr_type == kVtListListInt) {
        return AddListListIntAttr(ctx, index, attr_value);
    }
    return kAttrUnsupported;
}

int32_t AddAttrToNodeDef(const gert::HostCpuOpExecutionContext* ctx, const size_t index, const std::string& attr_name,
                         const std::string& attr_type, aicpuops::AttrValue* attr_value)
{
    if (ctx->GetAttrs() == nullptr) {
        return kAttrAbsent;
    }
    const int32_t scalar_ret = AddScalarAttrToNodeDef(ctx, index, attr_type, attr_value);
    if (scalar_ret != kAttrUnsupported) {
        return scalar_ret;
    }
    const int32_t list_ret = AddListAttrToNodeDef(ctx, index, attr_type, attr_value);
    if (list_ret != kAttrUnsupported) {
        return list_ret;
    }
    KERNEL_LOG_WARN("Attr[%s] type[%s] is not supported.", attr_name.c_str(), attr_type.c_str());
    return kAttrUnsupported;
}

int32_t AddInputTensor(const gert::Tensor* tensor, const std::string& name, aicpuops::NodeDef* node_def)
{
    if (tensor == nullptr) {
        KERNEL_LOG_WARN("Input tensor[%s] is null.", name.c_str());
        return -1;
    }
    auto* aicpu_tensor = node_def->add_inputs();
    if (aicpu_tensor == nullptr) {
        KERNEL_LOG_WARN("Create AICPU input tensor[%s] failed.", name.c_str());
        return -1;
    }
    ConvertGertToAicpuTensor(*tensor, name, aicpu_tensor);
    return 0;
}

int32_t BuildInputTensors(const gert::HostCpuOpExecutionContext* ctx, const std::vector<IrDefPair>& input_descs,
                          aicpuops::NodeDef* node_def)
{
    for (size_t ir_index = 0; ir_index < input_descs.size(); ++ir_index) {
        const std::string name(input_descs[ir_index].first.GetString());
        const std::string type(input_descs[ir_index].second.GetString());
        if (type == kIrInputRequired) {
            if (AddInputTensor(ctx->GetRequiredInputTensor(ir_index), name, node_def) != 0) {
                KERNEL_LOG_WARN("Build required input[%s], IR index[%zu] failed.", name.c_str(), ir_index);
                return -1;
            }
        } else if (type == kIrInputOptional) {
            const auto* tensor = ctx->GetOptionalInputTensor(ir_index);
            if ((tensor != nullptr) && (AddInputTensor(tensor, name, node_def) != 0)) {
                KERNEL_LOG_WARN("Build optional input[%s], IR index[%zu] failed.", name.c_str(), ir_index);
                return -1;
            }
        } else if (type == kIrInputDynamic) {
            const auto* instance_info = ctx->GetIrInputInstanceInfo(ir_index);
            if (instance_info == nullptr) {
                KERNEL_LOG_WARN("Dynamic input[%s], IR index[%zu] has no instance info.", name.c_str(), ir_index);
                return -1;
            }
            const size_t instance_num = instance_info->GetInstanceNum();
            for (size_t relative_index = 0; relative_index < instance_num; ++relative_index) {
                if (AddInputTensor(ctx->GetDynamicInputTensor(ir_index, relative_index), name, node_def) != 0) {
                    KERNEL_LOG_WARN("Build dynamic input[%s], IR index[%zu], relative index[%zu] failed.", name.c_str(),
                                    ir_index, relative_index);
                    return -1;
                }
            }
        } else {
            KERNEL_LOG_WARN("Input[%s] has unknown IR type[%s].", name.c_str(), type.c_str());
            return -1;
        }
    }
    return 0;
}

int32_t ValidateOutputInstanceNum(const std::string& name, const std::string& type, const size_t instance_num)
{
    if ((type == kIrOutputRequired) && (instance_num != 1U)) {
        KERNEL_LOG_WARN("Required output[%s] has instance number[%zu].", name.c_str(), instance_num);
        return -1;
    }
    if ((type != kIrOutputRequired) && (type != kIrOutputDynamic)) {
        KERNEL_LOG_WARN("Output[%s] has unknown IR type[%s].", name.c_str(), type.c_str());
        return -1;
    }
    return 0;
}

bool IsKnownShape(const gert::Shape& shape)
{
    for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) < 0) {
            return false;
        }
    }
    return true;
}

int32_t AddOutputTensorInstances(gert::HostCpuOpExecutionContext* ctx, const std::string& name, const std::string& type,
                                 const size_t instance_start, const size_t instance_num, aicpuops::NodeDef* node_def,
                                 size_t& output_num)
{
    for (size_t relative_index = 0; relative_index < instance_num; ++relative_index) {
        const size_t index = instance_start + relative_index;
        const auto* output_tensor = ctx->GetOutputTensor(index);
        if (output_tensor == nullptr) {
            KERNEL_LOG_WARN("Output[%zu] has no tensor metadata for allocation.", index);
            return -1;
        }
        if (!IsKnownShape(output_tensor->GetStorageShape())) {
            KERNEL_LOG_INFO("Output[%zu] has unknown storage shape, skip AICPU constant folding.", index);
            return -1;
        }
        auto* tensor = ctx->MallocOutputTensor(index, output_tensor->GetShape(), output_tensor->GetFormat(),
                                               output_tensor->GetDataType());
        if ((tensor == nullptr) || ((tensor->GetSize() != 0U) && (tensor->GetAddr() == nullptr))) {
            KERNEL_LOG_WARN("Malloc output tensor[%zu] failed.", index);
            return -1;
        }
        const std::string tensor_name = type == kIrOutputDynamic ? name + std::to_string(relative_index) : name;
        auto* aicpu_tensor = node_def->add_outputs();
        if (aicpu_tensor == nullptr) {
            KERNEL_LOG_WARN("Create AICPU output tensor[%s], context index[%zu] failed.", tensor_name.c_str(), index);
            return -1;
        }
        ConvertGertToAicpuTensor(*tensor, tensor_name, aicpu_tensor);
        ++output_num;
    }
    return 0;
}

int32_t BuildOutputTensors(gert::HostCpuOpExecutionContext* ctx, const std::vector<IrDefPair>& output_descs,
                           aicpuops::NodeDef* node_def)
{
    size_t output_num = 0U;
    for (size_t ir_index = 0; ir_index < output_descs.size(); ++ir_index) {
        const std::string name(output_descs[ir_index].first.GetString());
        const std::string type(output_descs[ir_index].second.GetString());
        const auto* instance_info = ctx->GetIrOutputInstanceInfo(ir_index);
        if (instance_info == nullptr) {
            KERNEL_LOG_WARN("Output[%s], IR index[%zu] has no instance info.", name.c_str(), ir_index);
            return -1;
        }

        const size_t instance_num = instance_info->GetInstanceNum();
        if (ValidateOutputInstanceNum(name, type, instance_num) != 0) {
            KERNEL_LOG_WARN("Validate output[%s], IR index[%zu], type[%s], instance number[%zu] failed.", name.c_str(),
                            ir_index, type.c_str(), instance_num);
            return -1;
        }
        if (AddOutputTensorInstances(ctx, name, type, instance_info->GetInstanceStart(), instance_num, node_def,
                                     output_num) != 0) {
            KERNEL_LOG_WARN("Build output[%s], IR index[%zu], instance start[%zu], instance number[%zu] failed.",
                            name.c_str(), ir_index, instance_info->GetInstanceStart(), instance_num);
            return -1;
        }
    }
    if (output_num != ctx->GetComputeNodeOutputNum()) {
        KERNEL_LOG_WARN("Built output number[%zu] does not match context output number[%zu].", output_num,
                        ctx->GetComputeNodeOutputNum());
        return -1;
    }
    return 0;
}

int32_t BuildNodeDefAttrs(const gert::HostCpuOpExecutionContext* ctx, const std::vector<IrDefPair>& attr_descs,
                          aicpuops::NodeDef* node_def)
{
    auto* attrs = node_def->mutable_attrs();
    if (attrs == nullptr) {
        KERNEL_LOG_WARN("Get mutable AICPU attrs failed.");
        return -1;
    }
    for (size_t index = 0; index < attr_descs.size(); ++index) {
        const std::string name(attr_descs[index].first.GetString());
        const std::string type(attr_descs[index].second.GetString());
        aicpuops::AttrValue attr_value;
        const int32_t ret = AddAttrToNodeDef(ctx, index, name, type, &attr_value);
        if (ret != kAttrSuccess) {
            if (ret == kAttrAbsent) {
                KERNEL_LOG_INFO("Attr[%s] is absent, skip it.", name.c_str());
                continue;
            }
            KERNEL_LOG_WARN("Build attr[%s], index[%zu], type[%s] failed, ret[%d].", name.c_str(), index, type.c_str(),
                            ret);
            return ret;
        }
        const auto result = attrs->insert(AttrValueMap::value_type(name, attr_value));
        if (!result.second) {
            KERNEL_LOG_WARN("Insert attr[%s], index[%zu], type[%s] into AICPU NodeDef failed.", name.c_str(), index,
                            type.c_str());
            return -1;
        }
    }
    return 0;
}

int32_t BuildNodeDef(gert::HostCpuOpExecutionContext* ctx, const std::string& op_type, aicpuops::NodeDef* node_def)
{
    std::vector<IrDefPair> input_descs;
    std::vector<IrDefPair> output_descs;
    std::vector<IrDefPair> attr_descs;
    const auto ret = GetRegisteredIrDefV2(op_type.c_str(), input_descs, output_descs, attr_descs);
    if (ret != ge::SUCCESS) {
        KERNEL_LOG_WARN("GetRegisteredIrDefV2 for op[%s] failed, ret[%u].", op_type.c_str(), ret);
        return -1;
    }
    KERNEL_LOG_DEBUG("Build NodeDef for op[%s], IR input number[%zu], output number[%zu], attr number[%zu].",
                     op_type.c_str(), input_descs.size(), output_descs.size(), attr_descs.size());
    node_def->set_op(op_type);
    if (BuildInputTensors(ctx, input_descs, node_def) != 0) {
        KERNEL_LOG_WARN("Build input tensors for op[%s] failed.", op_type.c_str());
        return -1;
    }
    if (BuildOutputTensors(ctx, output_descs, node_def) != 0) {
        KERNEL_LOG_WARN("Build output tensors for op[%s] failed.", op_type.c_str());
        return -1;
    }
    const int32_t attr_ret = BuildNodeDefAttrs(ctx, attr_descs, node_def);
    if (attr_ret != 0) {
        KERNEL_LOG_WARN("Build attrs for op[%s] failed, ret[%d].", op_type.c_str(), attr_ret);
    }
    return attr_ret;
}

void TryBindV2Symbols(void* handle, const std::string& so_name)
{
    V2ModuleBinding binding;
    binding.handle = handle;
    binding.so_name = so_name;
    binding.get_all_op_types = reinterpret_cast<GetAllRegisteredOpTypesV2Fn>(
        dlsym(handle, kSymGetAllRegisteredOpTypesV2));
    binding.run_cpu_kernel = reinterpret_cast<RunCpuKernelV2Fn>(dlsym(handle, kSymRunCpuKernelV2));

    if ((binding.get_all_op_types == nullptr) || (binding.run_cpu_kernel == nullptr)) {
        KERNEL_LOG_WARN("Required V2 symbols not found in so[%s], skip it.", so_name.c_str());
        (void)dlclose(handle);
        return;
    }
    g_v2_bindings.emplace_back(std::move(binding));
}

void LoadConstantFoldingSo(const std::string& dir_path)
{
    DIR* dir = opendir(dir_path.c_str());
    if (dir == nullptr) {
        KERNEL_LOG_WARN("Failed to open host CPU directory[%s].", dir_path.c_str());
        return;
    }
    std::vector<std::string> so_names;
    struct dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        if (aicpu_folding::IsConstantFoldingSo(entry->d_name)) {
            so_names.emplace_back(entry->d_name);
        }
    }
    closedir(dir);
    std::sort(so_names.begin(), so_names.end());
    for (const auto& so_name : so_names) {
        const std::string lib_path = aicpu_folding::GetRealPath(dir_path + so_name);
        if (lib_path.empty()) {
            continue;
        }
        void* handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            KERNEL_LOG_WARN("dlopen failed: %s, reason: %s", lib_path.c_str(), dlerror());
            continue;
        }
        TryBindV2Symbols(handle, so_name);
    }
}

const V2ModuleBinding* LookupV2Binding(const std::string& op_type)
{
    const auto iter = g_v2_op_index.find(op_type);
    if (iter == g_v2_op_index.end()) {
        return nullptr;
    }
    return iter->second;
}

void RegisterHostCpuFoldingOps(const std::set<std::string>& ops)
{
    static const std::set<std::string> kBlackList = {"Assign", "NoOp", "TruncatedNormal"};
    for (const auto& op_type : ops) {
        if (kBlackList.count(op_type) != 0U) {
            continue;
        }
        KERNEL_LOG_INFO("Register bottom-priority host CPU constant folding implementation for op[%s].",
                        op_type.c_str());
        const ge::BaseOpCreator creator = []() -> std::unique_ptr<ge::BaseCustomOp> {
            return std::make_unique<AicpuHostCpuRouter>();
        };
        const ge::CustomOpCreatorRegister registrar(ge::AscendString(op_type.c_str()), ge::OpBackend::kHostCPU,
                                                    ge::OpRegistrationPriority::kBottom, ge::OpEngine::kHostCpu,
                                                    creator);
    }
}
} // namespace

ge::graphStatus AicpuHostCpuRouter::Execute(gert::HostCpuOpExecutionContext* ctx)
{
    if (ctx == nullptr) {
        KERNEL_LOG_WARN("Host CPU execution context is null.");
        return ge::GRAPH_FAILED;
    }
    const char* node_type = ctx->GetNodeType();
    if (node_type == nullptr) {
        KERNEL_LOG_WARN("Node type in Host CPU execution context is null.");
        return ge::GRAPH_FAILED;
    }
    const std::string op_type(node_type);

    aicpuops::NodeDef node_def;
    if (BuildNodeDef(ctx, op_type, &node_def) != 0) {
        KERNEL_LOG_WARN("Build NodeDef for op[%s] failed.", op_type.c_str());
        return ge::GRAPH_FAILED;
    }

    aicpu::CpuKernelContext cpu_ctx(aicpu::HOST);
    const uint32_t init_ret = cpu_ctx.Init(&node_def);
    if (init_ret != 0U) {
        KERNEL_LOG_WARN("Initialize CPU kernel context for op[%s] failed, ret[%u].", op_type.c_str(), init_ret);
        return ge::GRAPH_FAILED;
    }

    const auto* binding = LookupV2Binding(op_type);
    uint32_t ret = 0U;
    if (binding == nullptr) {
        KERNEL_LOG_DEBUG("Run host CPU kernel for op[%s] via V1 registry.", op_type.c_str());
        ret = aicpu::CpuKernelRegister::Instance().RunCpuKernel(cpu_ctx);
    } else {
        KERNEL_LOG_DEBUG("Run host CPU kernel for op[%s] via V2 so[%s].", op_type.c_str(), binding->so_name.c_str());
        ret = binding->run_cpu_kernel(cpu_ctx);
    }
    if (ret != 0U) {
        KERNEL_LOG_WARN("Run host CPU kernel for op[%s] failed, ret[%u].", op_type.c_str(), ret);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

extern "C" {
__attribute__((visibility("default"))) int32_t ConstantFoldingInitialize(const char* host_cpu_dir)
{
    if (g_initialized) {
        return 0;
    }
    KERNEL_LOG_INFO("Initialize op constant folding begin.");
    if ((host_cpu_dir == nullptr) || (host_cpu_dir[0] == '\0')) {
        KERNEL_LOG_ERROR("Host CPU directory is empty.");
        return -1;
    }

    for (const auto& binding : g_v2_bindings) {
        if (binding.handle != nullptr) {
            (void)dlclose(binding.handle);
        }
    }
    g_v2_bindings.clear();
    g_v2_op_index.clear();
    LoadConstantFoldingSo(aicpu_folding::EnsureTrailingSlash(host_cpu_dir));

    const auto v1_ops = aicpu::CpuKernelRegister::Instance().GetAllRegisteredOpTypes();
    std::set<std::string> all_ops(v1_ops.begin(), v1_ops.end());

    for (const auto& binding : g_v2_bindings) {
        const auto v2_ops = binding.get_all_op_types();
        for (const auto& op_type : v2_ops) {
            const auto result = g_v2_op_index.emplace(op_type, &binding);
            if (!result.second) {
                KERNEL_LOG_WARN("Op[%s] already belongs to V2 so[%s], skip so[%s].", op_type.c_str(),
                                result.first->second->so_name.c_str(), binding.so_name.c_str());
                continue;
            }
            all_ops.insert(op_type);
        }
    }

    if (all_ops.empty()) {
        KERNEL_LOG_WARN("No host CPU kernel was discovered.");
        return -1;
    }
    RegisterHostCpuFoldingOps(all_ops);
    g_initialized = true;
    KERNEL_LOG_INFO("Initialize op constant folding success, op count[%llu].",
                    static_cast<unsigned long long>(all_ops.size()));
    return 0;
}
}
