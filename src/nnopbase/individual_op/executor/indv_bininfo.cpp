/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "indv_bininfo.h"
#include <mutex>
#include <sys/types.h>
#include <sys/stat.h>
#include <functional>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include "runtime/runtime/kernel.h"
#include "mmpa/mmpa_api.h"
#include "utils/indv_base.h"
#include "utils/thread_var_container.h"
#include "register/op_binary_resource_manager.h"
#include "acl/acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t NNOPBASE_MAX_ERROR_STRLEN = 128U;
constexpr const char *NNOPBASE_DEBUG_OPTIONS_ASSERT = "assert";
constexpr const char *NNOPBASE_DEBUG_OPTIONS_PRINTF = "printf";
constexpr const char *NNOPBASE_DEBUG_OPTIONS_TIMESTAMP = "timestamp";
 
NnopbaseChar *NnopbaseGetmmErrorMsg()
{
    NnopbaseChar errBuf[NNOPBASE_MAX_ERROR_STRLEN + 1U] = {};
    return mmGetErrorFormatMessage(mmGetErrorCode(), &errBuf[0], NNOPBASE_MAX_ERROR_STRLEN);
}

static uint32_t NnopbaseGetBinaryMagic(const bool useCoreTypeMagic, NnopbaseBinInfo *binInfo)
{
    OP_LOGI("Get bin core type is %d.", binInfo->coreType);
    if (!useCoreTypeMagic) {
        return RT_DEV_BINARY_MAGIC_ELF;
    }

    switch (binInfo->coreType) {
        case kMix: {
            return RT_DEV_BINARY_MAGIC_ELF;
        }
        case kAicore: {
            return RT_DEV_BINARY_MAGIC_ELF_AICUBE;
        }
        case kVectorcore: {
            return RT_DEV_BINARY_MAGIC_ELF_AIVEC;
        }
        default: {
            return RT_DEV_BINARY_MAGIC_ELF;
        }
    }
}

aclnnStatus NnopbaseMC2DynamicKernelRegister(const bool useCoreTypeMagic, NnopbaseBinInfo *binInfo)
{
    rtDevBinary_t binary;
    binary.magic = NnopbaseGetBinaryMagic(useCoreTypeMagic, binInfo);
    binary.data = binInfo->bin;
    binary.length = binInfo->binLen;
    NNOPBASE_ASSERT_RTOK_RETVAL(rtRegisterAllKernel(&binary, &binInfo->ccuBinHandle));
    OP_LOGD("Finish mc2 kernel register.");
    return OK;
}

aclnnStatus NnopbaseAclrtBinaryLoad(const bool useCoreTypeMagic, NnopbaseBinInfo *binInfo)
{
    aclrtBinaryLoadOption aclrtBinaryLoadOp = aclrtBinaryLoadOption {
        .type = ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC,
        .value = aclrtBinaryLoadOptionValue {.magic = NnopbaseGetBinaryMagic(useCoreTypeMagic, binInfo)}
    };

    aclrtBinaryLoadOptions aclrtBinaryLoadOpts = aclrtBinaryLoadOptions {
        .options = &aclrtBinaryLoadOp,
        .numOpt = 1U,
    };
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtBinaryLoadFromData(binInfo->bin, binInfo->binLen, &aclrtBinaryLoadOpts,
        &binInfo->binHandle));
    OP_LOGD("Finish kernel register.");
    return OK;
}


aclnnStatus NnopbaseRegisterMemsetBin(std::shared_ptr<NnopbaseMemsetBinInfo> &binInfo)
{
    if (binInfo->bin == nullptr) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseBinInfoReadBinFile(binInfo->binPath.c_str(), binInfo->bin, &binInfo->binLen));
    }
    aclrtBinaryLoadOption aclrtBinaryLoadOp =  aclrtBinaryLoadOption {
        .type = ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC,
        .value = aclrtBinaryLoadOptionValue { .magic = binInfo->magic }
    };
    aclrtBinaryLoadOptions aclrtBinaryLoadOpts =  aclrtBinaryLoadOptions {
        .options = &aclrtBinaryLoadOp,
        .numOpt = 1U,
    };
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtBinaryLoadFromData(binInfo->bin, binInfo->binLen, &aclrtBinaryLoadOpts,
        &binInfo->binHandle));
    OP_LOGD("Finish memset kernel register.");
    return OK;
}

aclnnStatus GetFuncHandleByEntry(void* binHandle, uint64_t funcEntry, aclrtFuncHandle *funcHandle)
{
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtBinaryGetFunctionByEntry(binHandle,
                funcEntry, funcHandle));
    return OK;
}

aclnnStatus GetFuncHandleByKernelName(void* binHandle, const char *kernelName, aclrtFuncHandle *funcHandle)
{
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtBinaryGetFunction(binHandle,
            kernelName, funcHandle));
    return OK;
}

static void NnopbaseSetInitValueInfo(
    const size_t index, const nlohmann::json &param, std::vector<NnopbaseInitValueInfo> &initValues)
{
    NnopbaseInitValueInfo info = {index, op::DataType::DT_FLOAT, 0, 0, 0};
    if (param["dtype"] == "float16") {
        info.dtype = op::DataType::DT_FLOAT16;
        info.floatValue = param["init_value"].get<float32_t>();
    } else if (param["dtype"] == "float32") {
        info.dtype = op::DataType::DT_FLOAT;
        info.floatValue = param["init_value"].get<float32_t>();
    } else if (param["dtype"] == "int64") {
        info.dtype = op::DataType::DT_INT64;
        info.intValue = param["init_value"].get<int64_t>();
    } else if (param["dtype"] == "uint64") {
        info.dtype = op::DataType::DT_UINT64;
        info.intValue = static_cast<int64_t>(param["init_value"].get<uint64_t>());
    } else if (param["dtype"] == "int32") {
        info.dtype = op::DataType::DT_INT32;
        info.intValue = static_cast<int64_t>(param["init_value"].get<int32_t>());
    } else if (param["dtype"] == "uint32") {
        info.dtype = op::DataType::DT_UINT32;
        info.intValue = static_cast<int64_t>(param["init_value"].get<uint32_t>());
    } else if (param["dtype"] == "int16") {
        info.dtype = op::DataType::DT_INT16;
        info.intValue = static_cast<int64_t>(param["init_value"].get<int16_t>());
    } else if (param["dtype"] == "uint16") {
        info.dtype = op::DataType::DT_UINT16;
        info.intValue = static_cast<int64_t>(param["init_value"].get<uint16_t>());
    } else if (param["dtype"] == "int8") {
        info.dtype = op::DataType::DT_INT8;
        info.intValue = static_cast<int64_t>(param["init_value"].get<int8_t>());
    } else if (param["dtype"] == "uint8") {
        info.dtype = op::DataType::DT_UINT8;
        info.intValue = static_cast<int64_t>(param["init_value"].get<uint8_t>());
    } else {
        OP_LOGW("Unknown dtype: %s", param["dtype"].get<std::string>().c_str());
        return;
    }
    initValues.emplace_back(info);
    OP_LOGI("Get init_value dtype is %s, floatValue is %f, intValue is %ld",
        param["dtype"].get<std::string>().c_str(), info.floatValue, info.intValue);
}

static void NnopbaseGetInitValue(nlohmann::json &opJson, NnopbaseBinInfo *binInfo)
{
    binInfo->initValues.clear();
    try {
        for (size_t i = 0; i < opJson["parameters"].size(); i++) {
            const auto &param = opJson["parameters"][i];
            if (!param.is_null()) {
                if (!param.contains("dtype")) {
                    continue;
                }
                NnopbaseSetInitValueInfo(i, param, binInfo->initValues);
            }
            OP_LOGI("InitValues size is %zu.", binInfo->initValues.size());
        }
    } catch (const nlohmann::json::exception &e) {
        OP_LOGW("Not get parameters, reason %s", e.what());
    }
}

static void NnopbaseGetOomFlag(NnopbaseBinInfo* binInfo, nlohmann::json &binJsonInfo)
{
    if (binJsonInfo.contains("supportInfo")) {
        try {
            const std::string debugConfigStr = binJsonInfo["supportInfo"]["op_debug_config"].get<std::string>();
            std::stringstream ss(debugConfigStr);
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (token == "oom") {
                    binInfo->oomFlag = true;
                    break;
                }
            }
            OP_LOGI("OomFlag is %d.", binInfo->oomFlag);
        } catch (const nlohmann::json::exception &e) {
            OP_LOGW("Not get op_debug_config, reason %s.", e.what());
        }
    }
}

static aclnnStatus NnopbaseGetDebugOptions(NnopbaseBinInfo* binInfo, nlohmann::json &binJsonInfo)
{
    try {
        const std::string debugOptions = binJsonInfo["debugOptions"].get<std::string>();
        if (debugOptions.find(NNOPBASE_DEBUG_OPTIONS_PRINTF) != std::string::npos) {
            binInfo->dfxInfo.isPrintEnable = true;
        }
        if (debugOptions.find(NNOPBASE_DEBUG_OPTIONS_ASSERT) != std::string::npos) {
            binInfo->dfxInfo.isAssertEnable = true;
        }
        if (debugOptions.find(NNOPBASE_DEBUG_OPTIONS_TIMESTAMP) != std::string::npos) {
            binInfo->dfxInfo.isTimeStampEnable = true;
        }
        binInfo->debugBufSize = binJsonInfo["debugBufSize"].get<size_t>();
        OP_LOGI("Get printType is [%s], debugBufSize is [%lu bytes].", debugOptions.c_str(), binInfo->debugBufSize);
    } catch (const nlohmann::json::exception &e) {
        OP_LOGW("Not get printType or debugBufSize, reason %s.", e.what());
    }
    return OK;
}

static aclnnStatus NnopbaseSetOpJsonConfig(NnopbaseBinInfo* binInfo, nlohmann::json &binJsonInfo)
{
    if ((binInfo->coreType == kMix) || (binInfo->coreType == kMixAiCore) || (binInfo->coreType == kMixAiv)) {
        try {
            const std::string taskRation = binJsonInfo["taskRation"].get<std::string>();
            const auto &iter = TASK_RATION_MAP.find(taskRation);
            if (iter == TASK_RATION_MAP.end()) {
                OP_LOGW("TaskRation%s is not supported!", taskRation.c_str());
            } else {
                binInfo->taskRation = iter->second;
                OP_LOGI("Get taskRation is [%s].", taskRation.c_str());
            }
        } catch (const nlohmann::json::exception &e) {
            OP_LOGW("Not get taskRation, reason %s.", e.what());
        }
    }

    try {
        const uint32_t opParaSize = binJsonInfo["opParaSize"].get<uint32_t>();
        binInfo->opParaSize = ((opParaSize % 8U) != 0) ? (opParaSize / 8U + 1U) * 8U : opParaSize; // 8byte对齐
        OP_LOGI("Get opParaSize is [%u bytes], binInfo->opParaSize is [%u bytes].", opParaSize, binInfo->opParaSize);
    } catch (const nlohmann::json::exception &e) {
        OP_LOGW("Not get opParaSize, reason %s", e.what());
    }

    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseGetDebugOptions(binInfo, binJsonInfo));
    NnopbaseGetOomFlag(binInfo, binJsonInfo);
    NnopbaseGetInitValue(binJsonInfo, binInfo);

    return OK;
}

static aclnnStatus NnopbaseUpdateBinConfigJsonInfo(nlohmann::json &opJsonInfo, NnopbaseBinInfo *binInfo)
{
    uint64_t tilingKey = 0U;
    try {
        tilingKey = opJsonInfo["tilingKey"].get<uint64_t>();
        const std::string kernelType = opJsonInfo["kernelType"].get<std::string>();
        const auto &iterKernelType = g_nnopbaseKernelTypeMap.find(kernelType);
        NNOPBASE_ASSERT_TRUE_RETVAL(iterKernelType != g_nnopbaseKernelTypeMap.end());
        binInfo->tilingKeyInfo[tilingKey].coreType = iterKernelType->second;

        const std::string taskRation = opJsonInfo["taskRation"].get<std::string>();
        const auto &iterTaskRation = TASK_RATION_MAP.find(taskRation);
        NNOPBASE_ASSERT_TRUE_RETVAL(iterTaskRation != TASK_RATION_MAP.end());
        binInfo->tilingKeyInfo[tilingKey].taskRation = iterTaskRation->second;
        OP_LOGI("Get multiKernelType is [%u], tilingKey is [%lu], kernelType is [%s], taskRation is [%s]",
                binInfo->multiKernelType, tilingKey, kernelType.c_str(), taskRation.c_str());
    } catch (const nlohmann::json::exception &e) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Read jsonfile binInfo failed, reason %s.",
                e.what());
        return ACLNN_ERR_PARAM_INVALID;
    }
    
    // for 1971
    if (binInfo->coreType == kMix) {
        try {
            const uint32_t crossCoreSync = opJsonInfo["crossCoreSync"].get<uint32_t>();
            binInfo->tilingKeyInfo[tilingKey].crossCoreSync = (crossCoreSync == 1);
            OP_LOGI("Get crossCoreSync is [%u].", crossCoreSync);
        } catch (const nlohmann::json::exception &e) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Can not get crossCoreSync, reason %s.",
                e.what());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return OK;
}

static aclnnStatus NnopbaseBinInfoUpdateKernelListConfig(const nlohmann::json &binConfigInfo, NnopbaseBinInfo* binInfo)
{
    if (binInfo->loadBinInfoType != kStaticBinInfo) {
        if (binInfo->multiKernelType == 1) {
            const auto &binConfigInfos = binConfigInfo["kernelList"];
            for (auto kernelInfo : binConfigInfos) {
                NNOPBASE_ASSERT_OK_RETVAL(NnopbaseUpdateBinConfigJsonInfo(kernelInfo, binInfo));
            }
        }
    } else {
        const auto &binConfigInfos = binConfigInfo["kernelList"];
        for (auto kernelInfo : binConfigInfos) {
            // tilingKey 存在时，读取tilingKey等信息
            if (kernelInfo.contains("tilingKey")) {
                NNOPBASE_ASSERT_OK_RETVAL(NnopbaseUpdateBinConfigJsonInfo(kernelInfo, binInfo));
            }
        }
    }
    return OK;
}


aclnnStatus NnopbaseLoadMemsetJson(std::unique_ptr<NnopbaseMemsetInfo> &memsetInfo)
{
    nlohmann::json jsonInfo;
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseReadJsonConfig(memsetInfo->memSetJsonPath, jsonInfo));
    try {
        std::string compileInfoStr = jsonInfo["compileInfo"].dump();
        memsetInfo->compileInfo = compileInfoStr.c_str();
        OP_LOGI("Compile info is %s.", memsetInfo->compileInfo);

        memsetInfo->binInfo->kernelName = jsonInfo["binFileName"].get<std::string>();
        memsetInfo->binInfo->stubName = jsonInfo["kernelName"].get<std::string>();
        OP_LOGI("Get MemSet kernelName is %s, stubName is %s.",
            memsetInfo->binInfo->kernelName.c_str(),
            memsetInfo->binInfo->stubName.c_str());

        const uint32_t opParaSize = jsonInfo["opParaSize"].get<uint32_t>();
        // 8byte对齐
        memsetInfo->binInfo->opParaSize = ((opParaSize % 8U) != 0) ? (opParaSize / 8U + 1U) * 8U : opParaSize;
        OP_LOGI("Get MemSet opParaSize is [%u bytes].", memsetInfo->binInfo->opParaSize);

        const auto &magic = jsonInfo["magic"].get<std::string>();
        if (magic == "RT_DEV_BINARY_MAGIC_ELF_AICUBE") {
            memsetInfo->binInfo->magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
        } else if (magic == "RT_DEV_BINARY_MAGIC_ELF_AIVEC") {
            memsetInfo->binInfo->magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
        }
        std::string coreType = jsonInfo["coreType"].get<std::string>();
        const auto &iterKernelType = g_nnopbaseKernelTypeMap.find(coreType);
        NNOPBASE_ASSERT_TRUE_RETVAL(iterKernelType != g_nnopbaseKernelTypeMap.end());
        memsetInfo->binInfo->coreType = iterKernelType->second;
        OP_LOGI("Memset op magic is %u, coreType is %s.", memsetInfo->binInfo->magic, coreType.c_str());
    } catch (const nlohmann::json::exception &e) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Read Memset json %s failed, errmsg:%s.",
            memsetInfo->memSetJsonPath.c_str(),
            e.what());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return OK;
}

aclnnStatus NnopbaseGetBinPath(const std::string &jsonPath, std::string &binPath)
{
    const size_t pos = jsonPath.find(".json");
    if (pos != std::string::npos) {
        binPath = jsonPath.substr(0U, pos + 1U) + "o";
        OP_LOGI("BinPath is %s.", binPath.c_str());
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Get binPath from jsonPath %s failed.", jsonPath.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return OK;
}

aclnnStatus NnopbaseGetMemsetBinInfo(std::unique_ptr<NnopbaseMemsetInfo> &memsetInfo)
{
    std::string binPath;
    CHECK_COND(NnopbaseGetBinPath(memsetInfo->memSetJsonPath, binPath) == OK,
        ACLNN_ERR_PARAM_INVALID,
        "Get memset binary file failed.");
    NnopbaseChar kernelPath[NNOPBASE_FILE_PATH_MAX_LEN] = {};
    if (mmRealPath(binPath.c_str(), kernelPath, NNOPBASE_FILE_PATH_MAX_LEN) != EN_OK) {
        OP_LOGW("Get Memset kernel path %s failed, errmsg:%s.", binPath.c_str(), NnopbaseGetmmErrorMsg());
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto binInfo = std::make_shared<NnopbaseMemsetBinInfo>();
    NNOPBASE_ASSERT_NOTNULL_RETVAL(binInfo);
    memsetInfo->binInfo = std::move(binInfo);
    memsetInfo->binInfo->binPath = std::string(kernelPath);
    OP_LOGI("Memset binpath is %s.", memsetInfo->binInfo->binPath.c_str());

    return OK;
}

aclnnStatus NnopbaseReadMemsetJsonInfo(const std::string &memsetBasePath, nlohmann::json &memsetJsonInfo,
    std::unique_ptr<NnopbaseMemsetInfo> &memsetInfo, const size_t initNum)
{
    for (auto& bin : memsetJsonInfo["binList"]) {
        for (auto& attr : bin["attrs"]) {
            if (attr["name"] == "sizes") {
                if (attr["value"].size() == initNum) {
                    try {
                        std::string path = bin["binInfo"]["jsonFilePath"].get<std::string>();
                        auto pos = path.find('/'); // 一定存在
                        memsetInfo->memSetJsonPath = memsetBasePath + "/" + path.substr(pos + 1);
                        OP_LOGI("Init num is %zu, jsonPath is %s", initNum, memsetInfo->memSetJsonPath.c_str());
                        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseGetMemsetBinInfo(memsetInfo));
                        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseLoadMemsetJson(memsetInfo));
                        return OK;
                    } catch (const nlohmann::json::exception &e) {
                        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Can not get memset jsonFilePath, reason %s.", e.what());
                        return ACLNN_ERR_PARAM_INVALID;
                    }
                }
            }
        }
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Can not find memset json info, init num is %zu.", initNum);
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus NnopbaseGenMemsetInfo(NnopbaseBinInfo *binInfo, const std::string &oppPath, const std::string &socVersion)
{
    std::string memsetBasePath = oppPath + NNOPBASE_MEMSET_DEFAULT_PATH_V2 + socVersion + "/ops_legacy";
    const std::string memsetDirPathV2 = oppPath + NNOPBASE_MEMSET_DEFAULT_PATH + socVersion + "/ops_legacy" + NNOPBASE_MEMSET_JSON_PATH;
    nlohmann::json memsetJsonInfo;
    if (NnopbaseReadJsonConfig(memsetDirPathV2, memsetJsonInfo) != OK) {
        OP_LOGW("Read %s failed, trying another path.", memsetDirPathV2.c_str());
        memsetBasePath = oppPath + NNOPBASE_MEMSET_DEFAULT_PATH_V2 + socVersion;
        std::string memsetDirPath = memsetBasePath + NNOPBASE_MEMSET_JSON_PATH;
        CHECK_COND(NnopbaseReadJsonConfig(memsetDirPath, memsetJsonInfo) == OK,
        ACLNN_ERR_PARAM_INVALID,
        "Read %s failed.",
        memsetDirPath.c_str());
    }
    binInfo->memsetInfo = std::make_unique<NnopbaseMemsetInfo>();
    NNOPBASE_ASSERT_NOTNULL_RETVAL(binInfo->memsetInfo);
    NNOPBASE_ASSERT_OK_RETVAL(
        NnopbaseReadMemsetJsonInfo(memsetBasePath, memsetJsonInfo, binInfo->memsetInfo, binInfo->initValues.size()));
    return NnopbaseRegisterMemsetBin(binInfo->memsetInfo->binInfo);
}

aclnnStatus NnopbaseInitMemsetV2Info(NnopbaseBinInfo *binInfo)
{
    binInfo->memsetInfo = std::make_unique<NnopbaseMemsetInfo>();
    NNOPBASE_ASSERT_NOTNULL_RETVAL(binInfo->memsetInfo);

    binInfo->memsetInfo->binInfo = std::make_shared<NnopbaseMemsetBinInfo>();
    NNOPBASE_ASSERT_NOTNULL_RETVAL(binInfo->memsetInfo->binInfo);

    return OK;
}

aclnnStatus NnopbaseBinInfoReadJsonFile(NnopbaseBinInfo *binInfo, const std::string &oppPath, std::string &socVersion,
    bool isMemsetV2)
{
    std::string dirPath;
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseGetOpJsonPath(binInfo->binPath, dirPath));
    nlohmann::json binJsonInfo;
    if (binInfo->loadBinInfoType != kStaticBinInfo) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseReadJsonConfig(dirPath, binJsonInfo));
    } else {
        std::tuple<nlohmann::json, nnopbase::Binary> binConfigInfo;
        NNOPBASE_ASSERT_OK_RETVAL(
            nnopbase::OpBinaryResourceManager::GetInstance().GetOpBinaryDescByPath(dirPath.c_str(), binConfigInfo));
        binJsonInfo = std::get<0>(binConfigInfo);
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseBinInfoUpdateKernelListConfig(binJsonInfo, binInfo));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSetOpJsonConfig(binInfo, binJsonInfo));
    if (binInfo->initValues.size() != 0U) {
        if (isMemsetV2) {
            return NnopbaseInitMemsetV2Info(binInfo);
        } else {
            return NnopbaseGenMemsetInfo(binInfo, oppPath, socVersion);
        }
    }
    return OK;
}

aclnnStatus NnopbaseKernelUnRegister(void **handle)
{
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtBinaryUnLoad(*handle));
    *handle = nullptr;
    return OK;
}

aclnnStatus NnopbaseBinInfoReadBinFile(const NnopbaseChar *const binPath, NnopbaseUChar *&bin, uint32_t *binLen)
{
    struct stat statbuf;
    NNOPBASE_ASSERT_TRUE_RETVAL(stat(binPath, &statbuf) != -1);
    NNOPBASE_ASSERT_TRUE_RETVAL(statbuf.st_size > 0);
    const size_t size = static_cast<size_t>(statbuf.st_size);
    auto buf = std::make_unique<NnopbaseUChar[]>(size);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(buf);

    const int32_t fd = open(binPath, O_RDONLY);
    NNOPBASE_ASSERT_TRUE_RETVAL(fd != -1);
    const int32_t len = read(fd, buf.get(), size);
    if (close(fd) == -1) { OP_LOGW("Close bin file %s failed, errno = %d.", binPath, errno); }

    NNOPBASE_ASSERT_TRUE_RETVAL(len != -1);
    bin = buf.release();
    *binLen = size;
    return OK;
}

aclnnStatus NnopbaseGetOpJsonPath(const std::string &binPath, std::string &jsonPath)
{
    const size_t pos = binPath.find(".o");
    if (pos != std::string::npos) {
        jsonPath = binPath.substr(0U, pos + 1U) + "json";
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Get binJsonPath from binPath %s failed.", binPath.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return OK;
}

aclnnStatus NnopbaseReadJsonConfig(const std::string &binaryInfoPath, nlohmann::json &binaryInfoConfig)
{
    std::vector<NnopbaseChar> opInfoPath(NNOPBASE_FILE_PATH_MAX_LEN, '\0');
    if (mmRealPath(binaryInfoPath.c_str(), &(opInfoPath[0U]), NNOPBASE_FILE_PATH_MAX_LEN) != EN_OK) {
        OP_LOGW("Get jsonfile path for %s failed, errmsg:%s.", binaryInfoPath.c_str(),
                NnopbaseGetmmErrorMsg());
        return ACLNN_ERR_PARAM_INVALID;
    }

    OP_LOGI("Get jsonfile path: %s", &(opInfoPath[0U]));

    std::ifstream ifs(&(opInfoPath[0U]));
    NNOPBASE_ASSERT_TRUE_RETVAL(ifs.is_open());
    try {
        ifs >> binaryInfoConfig;
        ifs.close();
    } catch (const nlohmann::json::exception &e) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Read jsonfile [%s] failed, reason %s", &(opInfoPath[0U]), e.what());
        ifs.close();
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGI("Read jsonfile [%s] successfully.", &(opInfoPath[0U]));
    return OK;
}

#ifdef __cplusplus
}
#endif

namespace nnopbase {
std::string GetMagicFormBin(const bool useCoreTypeMagic, NnopbaseBinInfo *binInfo)
{
    if (useCoreTypeMagic) {
        switch (binInfo->coreType) {
            case kAicore: {
                return "RT_DEV_BINARY_MAGIC_ELF_AICUBE";
            }
            case kVectorcore: {
                return "RT_DEV_BINARY_MAGIC_ELF_AIVEC";
            }
            default: {
                return "RT_DEV_BINARY_MAGIC_ELF";
            }
        }
    } else {
        return "RT_DEV_BINARY_MAGIC_ELF";
    }
}
} // nnopbase
