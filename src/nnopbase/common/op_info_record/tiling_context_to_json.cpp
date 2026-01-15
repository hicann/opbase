/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_context_to_json.h"
#include <map>
#include <cctype>
#include "register/op_impl_registry.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "graph/utils/type_utils.h"
#include "opdev/op_log.h"
#include "opdev/op_errno.h"
#include "bin_to_json.h"

namespace aclnnOpInfoRecord {
namespace {
constexpr size_t STR_PRE = 3UL;
const char *const REQUIRED_STR = "required";
const char *const OPTIONAL_STR = "optional";
const char *const DYNAMIC_STR = "dynamic";

nlohmann::json StrAttrsToJson(const gert::RuntimeAttrs *attrs, size_t index)
{
    nlohmann::json j;
    j["value"] = attrs->GetStr(index);
    return j;
}

nlohmann::json BoolAttrsToJson(const gert::RuntimeAttrs *attrs, size_t index)
{
    nlohmann::json j;
    j["value"] = *attrs->GetBool(index);
    return j;
}

nlohmann::json IntAttrsToJson(const gert::RuntimeAttrs *attrs, size_t index)
{
    nlohmann::json j;
    j["value"] = *attrs->GetInt(index);
    return j;
}

nlohmann::json ListIntAttrsToJson(const gert::RuntimeAttrs *attrs, size_t index)
{
    nlohmann::json j;
    const auto *listInt = attrs->GetListInt(index);
    const int64_t *listIntPtr = listInt->GetData();
    std::vector<int64_t> vecInt;
    for (size_t idx = 0UL; idx < listInt->GetSize(); idx++) {
        vecInt.emplace_back(*(listIntPtr + idx));
    }
    j["value"] = vecInt;
    return j;
}

nlohmann::json FloatAttrsToJson(const gert::RuntimeAttrs *attrs, size_t index)
{
    nlohmann::json j;
    j["value"] = *attrs->GetFloat(index);
    return j;
}

nlohmann::json ListFloatAttrsToJson(const gert::RuntimeAttrs *attrs, size_t index)
{
    nlohmann::json j;
    const auto *listFloat = attrs->GetListFloat(index);
    const float *listFloatPtr = listFloat->GetData();
    std::vector<float> vecFloat;
    for (size_t idx = 0UL; idx < listFloat->GetSize(); idx++) {
        vecFloat.emplace_back(*(listFloatPtr + idx));
    }
    j["value"] = vecFloat;
    return j;
}

nlohmann::json DataTypeAttrsToJson(const gert::RuntimeAttrs *attrs, size_t index)
{
    nlohmann::json j;
    j["value"] = *attrs->GetInt(index);
    return j;
}

nlohmann::json ListDataTypeAttrsToJson(const gert::RuntimeAttrs *attrs, size_t index)
{
    return ListIntAttrsToJson(attrs, index);
}

nlohmann::json ListBoolAttrsToJson(const gert::RuntimeAttrs *attrs, size_t index)
{
    nlohmann::json j;
    const auto *listBool = attrs->GetAttrPointer<gert::TypedContinuousVector<bool>>(index);
    const bool *listBoolPtr = listBool->GetData();
    std::vector<bool> vecBool;
    for (size_t idx = 0UL; idx < listBool->GetSize(); idx++) {
        vecBool.emplace_back(*(listBoolPtr + idx));
    }
    j["value"] = vecBool;
    return j;
}

const std::map<std::string, std::function<nlohmann::json(
    const gert::RuntimeAttrs *, size_t)>> ATTRS_TO_JSON = {
    {"str", StrAttrsToJson},
    {"string", StrAttrsToJson},
    {"bool", BoolAttrsToJson},
    {"int", IntAttrsToJson},
    {"int32", IntAttrsToJson},
    {"float", FloatAttrsToJson},
    {"float32", FloatAttrsToJson},
    {"list_int", ListIntAttrsToJson},
    {"listInt", ListIntAttrsToJson},
    {"list_float", ListFloatAttrsToJson},
    {"list_bool", ListBoolAttrsToJson},
    {"listBool", ListBoolAttrsToJson},
};


std::vector<std::pair<std::string, std::string>> GetBuiltinIrFromJson(const nlohmann::json& supportInfoJsonConfig,
    bool isInput)
{
    std::vector<std::pair<std::string, std::string>> irDesc;
    std::string keyword = isInput ? "inputs" : "outputs";
    // keyword definitely exists
    try {
        for (size_t i = 0U; i < supportInfoJsonConfig[keyword].size(); ++i) {
            auto ele = supportInfoJsonConfig[keyword][i];
            if (ele.is_null()) {
                OP_LOGD("The %zuth element %s in supportInfo is null, skip.", i, keyword.c_str());
                irDesc.push_back({});
                continue;
            }
            if (ele.is_array()) {
                std::string paramType = ele[0].contains("paramType") ? ele[0]["paramType"] : "";    
                irDesc.push_back({ele[0]["name"], paramType});
                OP_LOGD("The %zuth element %s in supportInfo is {name : %s, paramType: %s}", i, keyword.c_str(),
                    ele[0]["name"].get<std::string>().c_str(), paramType.c_str());
            } else {
                std::string paramType = ele.contains("paramType") ? ele["paramType"] : "";  
                irDesc.push_back({ele["name"], paramType});
                OP_LOGD("The %zuth element %s in supportInfo is {name : %s, paramType: %s}", i, keyword.c_str(),
                    ele["name"].get<std::string>().c_str(), paramType.c_str());
            }
        }
    } catch (const nlohmann::json::exception &e) {
        OP_LOGE(ACLNN_ERR_INNER_LOAD_JSON_FAILED, "Read jsonfile failed, reason %s", e.what());
        return {};
    }
    return irDesc;
}

std::vector<std::pair<std::string, std::string>> GetBuiltinAttrFromJson(const nlohmann::json& supportInfoJsonConfig)
{
    if (!supportInfoJsonConfig.contains("attrs") || supportInfoJsonConfig["attrs"].size() == 0) {
        OP_LOGD("No Attrs in supportInfo.");
        return {};
    }
        std::vector<std::pair<std::string, std::string>> attrNames;
    try {
        auto& attrs = supportInfoJsonConfig["attrs"];
        for (size_t i = 0U; i < attrs.size(); ++i) {
            auto ele = attrs[i];
            if (ele.is_null()) {
                continue;
            }
            attrNames.push_back({ele["name"], ele["dtype"]});
        }
    } catch (const nlohmann::json::exception &e) {
        OP_LOGE(ACLNN_ERR_INNER_LOAD_JSON_FAILED, "Read jsonfile failed, reason %s", e.what());
        return {};
    }
    return attrNames;
}

nlohmann::json TensorToJson(const std::string &irName, const gert::CompileTimeTensorDesc *cpTdInfo,
    const gert::StorageShape *opShape, size_t irLoc, const std::string &irType)
{
    if (cpTdInfo == nullptr || opShape == nullptr) {
        return nullptr;
    }
    nlohmann::json j;
    j["name"] = irName;
    j["index"] = irLoc;
    j["param_type"] = irType;
    std::string dType = ge::TypeUtils::DataTypeToSerialString(cpTdInfo->GetDataType()).c_str() + STR_PRE;
    (void)std::transform(dType.begin(), dType.end(), dType.begin(), tolower);
    j["dtype"] = dType;

    j["shape"] = nlohmann::json::array();
    const auto &storageShape = opShape->GetStorageShape();
    for (size_t idx = 0UL; idx < storageShape.GetDimNum(); idx++) {
        j["shape"].emplace_back(storageShape.GetDim(idx));
    }

    j["format"] = ge::TypeUtils::FormatToSerialString(
        static_cast<ge::Format>(ge::GetPrimaryFormat(cpTdInfo->GetStorageFormat())));

    if (ge::HasSubFormat(cpTdInfo->GetStorageFormat())) {
        j["sub_format"] = ge::GetSubFormat(cpTdInfo->GetStorageFormat());
    }

    if (ge::HasC0Format(cpTdInfo->GetStorageFormat())) {
        j["c0_format"] = ge::GetC0Format(cpTdInfo->GetStorageFormat());
    }

    j["origin_shape"] = nlohmann::json::array();
    const auto &originShape = opShape->GetOriginShape();
    for (size_t idx = 0UL; idx < originShape.GetDimNum(); idx++) {
        j["origin_shape"].emplace_back(originShape.GetDim(idx));
    }

    j["origin_format"] = ge::TypeUtils::FormatToSerialString(cpTdInfo->GetOriginFormat());
    j["value_depend"] = false;
    return j;
}

nlohmann::json TransInputInstanceToJson(const gert::TilingContext *ctx, size_t &instanceLoc, size_t irLoc,
    const std::string &irName, const std::string &irType)
{
    const auto computeNodeInfo = ctx->GetComputeNodeInfo();
    const auto inputInstanceInfo = computeNodeInfo->GetInputInstanceInfo(irLoc);
    OP_LOGD("ctx:%p, instanceLoc:%zu, irLoc:%zu, irName:%s, irType:%s, computeNodeInfo:%p, inputInstanceInfo:%p",
        ctx, instanceLoc, irLoc, irName.c_str(), irType.c_str(), computeNodeInfo, inputInstanceInfo);

    if (inputInstanceInfo != nullptr) {
        OP_LOGD("instanceNum:%zu, inputTdInfo:%p, inputShape:%p",
            inputInstanceInfo->GetInstanceNum(), computeNodeInfo->GetInputTdInfo(instanceLoc),
            ctx->GetInputShape(instanceLoc));
    }
    if (irName.empty() || irType.empty() || irType.compare(REQUIRED_STR) == 0 || irType.compare(OPTIONAL_STR) == 0) {
        if (inputInstanceInfo != nullptr && inputInstanceInfo->GetInstanceNum() != 0) {
            nlohmann::json tmpJ = TensorToJson(
                irName, computeNodeInfo->GetInputTdInfo(instanceLoc), ctx->GetInputShape(instanceLoc), irLoc, irType);
            instanceLoc += inputInstanceInfo->GetInstanceNum();
            return tmpJ;
        }
        if (computeNodeInfo->GetInputTdInfo(instanceLoc) == nullptr || ctx->GetInputShape(instanceLoc) == nullptr) {
            instanceLoc++;
        }
        return nullptr;
    }

    if (irType.compare(DYNAMIC_STR) == 0) {
        nlohmann::json arrayJson = nlohmann::json::array();
        if (inputInstanceInfo == nullptr) {
            return arrayJson;
        }
        for (size_t idx = 0; idx < inputInstanceInfo->GetInstanceNum(); idx++) {
            arrayJson.emplace_back(TensorToJson(irName,
                computeNodeInfo->GetInputTdInfo(instanceLoc + idx),
                ctx->GetInputShape(instanceLoc + idx),
                irLoc,
                irType));
        }
        instanceLoc += inputInstanceInfo->GetInstanceNum();
        return arrayJson;
    }
    OP_LOGE(ACLNN_ERR_INNER, "TransInputInstanceToJson not support input ir type[%s]!", irType.c_str());
    return nullptr;
}

const gert::OpImplKernelRegistry::OpImplFunctionsV2 *GetOppImplement(const std::string &nodeType)
{
    // Start by searching in the default library kOpp.
    auto &registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    OP_CHECK(registry != nullptr, OP_LOGE(ACLNN_ERR_INNER, "Cannot get registry from gert with opp version."), return nullptr);
    const gert::OpImplKernelRegistry::OpImplFunctionsV2 *opImplFunc = registry->GetOpImpl(nodeType.c_str());
    if (opImplFunc != nullptr) {
        return opImplFunc;
    }
    // Finally, search from Library kOppKernel.
    OP_LOGW("Start searching %s from opp kernel version registry.", nodeType.c_str());
    auto &kernelRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
        gert::OppImplVersionTag::kOppKernel);
    OP_CHECK(kernelRegistry != nullptr, OP_LOGE(ACLNN_ERR_INNER, "Cannot get registry from gert with opp kernel version."), return nullptr);
    return kernelRegistry->GetOpImpl(nodeType.c_str());
}

nlohmann::json TransOutputInstanceToJson(const gert::TilingContext *ctx, size_t &instanceLoc, size_t irLoc,
    const std::string &irName, const std::string &irType)
{
    const auto computeNodeInfo = ctx->GetComputeNodeInfo();
    const auto outputInstanceInfo = computeNodeInfo->GetOutputInstanceInfo(irLoc);
    if (irType.empty() || irType.compare(REQUIRED_STR) == 0) {
        if (outputInstanceInfo == nullptr) {
            instanceLoc++;
            return nullptr;
        }
        nlohmann::json tmpJ = TensorToJson(
            irName, computeNodeInfo->GetOutputTdInfo(instanceLoc), ctx->GetOutputShape(instanceLoc), irLoc, irType);
        instanceLoc += outputInstanceInfo->GetInstanceNum();
        return tmpJ;
    }
    if (irType.compare(DYNAMIC_STR) == 0) {
        nlohmann::json arrayJson = nlohmann::json::array();
        if (outputInstanceInfo == nullptr) {
            return arrayJson;
        }
        for (size_t idx = 0; idx < outputInstanceInfo->GetInstanceNum(); idx++) {
            arrayJson.emplace_back(
                TensorToJson(irName, computeNodeInfo->GetOutputTdInfo(instanceLoc + idx),
                ctx->GetOutputShape(instanceLoc + idx), irLoc, irType));
        }
        instanceLoc += outputInstanceInfo->GetInstanceNum();
        return arrayJson;
    }
    OP_LOGE(ACLNN_ERR_INNER, "TransInputInstanceToJson not support output ir type[%s]!", irType.c_str());
    return nullptr;
}

int32_t ConstructInputOutputJson(const gert::TilingContext *ctx, const nlohmann::json &supportInfoJsonConfig,
    const gert::ComputeNodeInfo *computeNodeInfo, nlohmann::json &j)
{
    auto opImplFunc = GetOppImplement(computeNodeInfo->GetNodeType());
    if (opImplFunc == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "[%s]TilingContextToJson get op impl func failed!", computeNodeInfo->GetNodeType());
        return -1;
    }
    std::vector<std::pair<std::string, std::string>> inputIr = GetBuiltinIrFromJson(supportInfoJsonConfig, true);

    size_t inputInstanceLoc = 0;
    j["inputs"] = nlohmann::json::array();
    for (size_t irLoc = 0UL; irLoc < inputIr.size(); irLoc++) {
        nlohmann::json tmpJ = TransInputInstanceToJson(
            ctx, inputInstanceLoc, irLoc, inputIr[irLoc].first, inputIr[irLoc].second);
        if (!tmpJ.is_null() && opImplFunc->IsInputDataDependency(irLoc) && inputInstanceLoc >= 1) {
            tmpJ["value_depend"] = true;
            const auto *inputTensor = ctx->GetInputTensor(inputInstanceLoc - 1);
            const auto funcIter = BIN_TO_JSON.find(inputTensor->GetDataType());
            if (funcIter == BIN_TO_JSON.cend()) {
                OP_LOGE(ACLNN_ERR_INNER,
                    "[%s]Not support type[%s] in const data translate!",
                    computeNodeInfo->GetNodeType(),
                    ge::TypeUtils::DataTypeToSerialString(inputTensor->GetDataType()).c_str());
                return -1;
            }
            if (inputTensor->GetAddr() == nullptr) {
                OP_LOGW("Current operator [%s] is empty!", computeNodeInfo->GetNodeType());
                tmpJ["const_data"] = "NULL";
            } else {
                tmpJ["const_data"] = funcIter->second(inputTensor->GetAddr(), inputTensor->GetSize());
            }
        }
        OP_LOGD("[%s]inputs[%zu][%s]!", computeNodeInfo->GetNodeType(), irLoc, tmpJ.dump().c_str());
        j["inputs"].emplace_back(std::move(tmpJ));
    }

    std::vector<std::pair<std::string, std::string>> outputIr = GetBuiltinIrFromJson(supportInfoJsonConfig, false);
    size_t outputInstanceLoc = 0;
    j["outputs"] = nlohmann::json::array();
    for (size_t irLoc = 0UL; irLoc < outputIr.size(); irLoc++) {
        nlohmann::json tmpJ = TransOutputInstanceToJson(
            ctx, outputInstanceLoc, irLoc, outputIr[irLoc].first, outputIr[irLoc].second);
        OP_LOGD("[%s]outputs[%zu][%s]!", computeNodeInfo->GetNodeType(), irLoc, tmpJ.dump().c_str());
        j["outputs"].emplace_back(std::move(tmpJ));
    }
    return 0;
}
nlohmann::json AttrsToJson(const gert::ComputeNodeInfo *computeNodeInfo, const nlohmann::json &supportInfoJsonConfig)
{
    nlohmann::json nullJ(nullptr);
    const auto srcAttrs = computeNodeInfo->GetAttrs();
    if (srcAttrs == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "[%s]RuntimeAttrs is nullptr.", computeNodeInfo->GetNodeType());
        return nullJ;
    }
    auto irAttrNameTypes = GetBuiltinAttrFromJson(supportInfoJsonConfig);
    // The inconsistency between the RuntimeAttrs quantity and the prototype library quantity is not verified.
    size_t recordAttrLen = std::min(irAttrNameTypes.size(), srcAttrs->GetAttrNum());
    OP_LOGD("[%s]RuntimeAttrs size[%zu];irAttrNameTypes size[%zu].", computeNodeInfo->GetNodeType(),
        srcAttrs->GetAttrNum(), irAttrNameTypes.size());

    std::vector<nlohmann::json> attrs;
    for (size_t idx = 0UL; idx < recordAttrLen; idx++) {
        const auto funcIter = ATTRS_TO_JSON.find(irAttrNameTypes[idx].second);
        if (funcIter == ATTRS_TO_JSON.cend()) {
            OP_LOGE(ACLNN_ERR_INNER, "[%s]Not support type in attrs[%s][%zu]!", computeNodeInfo->GetNodeType(),
                irAttrNameTypes[idx].second.c_str(), idx);
            return nullJ;
        }
        nlohmann::json tmpJ = funcIter->second(srcAttrs, idx);
        tmpJ["name"] = irAttrNameTypes[idx].first;
        tmpJ["dtype"] = irAttrNameTypes[idx].second;
        tmpJ["index"] = idx;
        OP_LOGD("[%s]attrs[%zu][%s]!", computeNodeInfo->GetNodeType(), idx, tmpJ.dump().c_str());
        attrs.emplace_back(std::move(tmpJ));
    }
    return attrs;
}
} // namespace

nlohmann::json TilingContextToJson(
    const gert::TilingContext *ctx, const std::map<std::string, std::string> &iniConfigMap,
    const nlohmann::json &supportInfoJsonConfig)
{
    const auto computeNodeInfo = ctx->GetComputeNodeInfo();
    nlohmann::json nullJ(nullptr);
    if (computeNodeInfo == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "TilingContextToJson computeNodeInfo is nullptr!");
        return nullJ;
    }
    if (computeNodeInfo->GetNodeType() == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "TilingContextToJson node type is nullptr!");
        return nullJ;
    }
    std::string opType = computeNodeInfo->GetNodeType();
    nlohmann::json opJson;
    opJson["op_type"] = opType;
    OP_LOGI("TilingContextToJson get op[%s] to serialize!", computeNodeInfo->GetNodeName());

    const auto &iter = iniConfigMap.find(opType);
    if (iter != iniConfigMap.cend()) {
        opJson["tune_mode"] = iter->second;
    } else {
        opJson["tune_mode"] = "all";
    }

    if (ConstructInputOutputJson(ctx, supportInfoJsonConfig, computeNodeInfo, opJson) != 0) {
        OP_LOGE(ACLNN_ERR_INNER, "Failed to construct input/output/attr.");
        return nullJ;
    }
    nlohmann::json attrs = AttrsToJson(computeNodeInfo, supportInfoJsonConfig);
    if (attrs.is_null()) {
        return nullJ;
    }
    opJson["attrs"] = attrs;
    return opJson;
}
} // namespace aclnnOpInfoRecord