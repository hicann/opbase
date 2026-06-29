/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indv_lib_wrapper.h"
#include "utils/indv_debug_assert.h"

namespace nnopbase {
NnopBaseLoadSo::~NnopBaseLoadSo() = default;

aclnnStatus NnopBaseLoadSo::openSo(const char* loadSoPath)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(loadSoPath);

    OP_LOGI("Start loading dynamic library %s.", loadSoPath);
    soHandle_ = mmDlopen(loadSoPath, MMPA_RTLD_LAZY);
    if (soHandle_ == nullptr) {
        OP_LOGW("Failed to open "
                "dynamic library %s. dlopen error:%s.",
                loadSoPath, mmDlerror());
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGI("Load dynamic library %s successfully.", loadSoPath);

    soPath_ = loadSoPath;
    return OK;
}

void NnopBaseLoadSo::closeSo()
{
    if (soHandle_ != nullptr) {
        mmDlclose(soHandle_);
        soHandle_ = nullptr;
        soPath_ = "";
    }
}

ApiWrapper& ApiWrapper::GetInstance()
{
    static ApiWrapper instance;
    return instance;
}

void* ApiWrapper::GetFunc(const char* funcName)
{
    const std::lock_guard<std::mutex> lk(mutex_);
    if (!hasInit_) {
        if (openSo("libopapi_math.so") != OK) {
            std::string errMsg = "Failed to enable the AutoContiguous() function. Check whether the ops_math operator package"
                " is installed and whether libopapi_math.so is under environment variable LD_LIBRARY_PATH";
            OP_LOGE_FOR_EXECUTION_ERROR_WITHOUT_SOLUTION(errMsg.c_str());
            return nullptr;
        }
        hasInit_ = true;
    }
    if (funcName == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Function name is nullptr!");
        return nullptr;
    }
    const auto& it = functions_.find(funcName);
    if (it != functions_.end()) {
        return it->second;
    }
    void* const func = LoadFunction<void*>(funcName);
    if (func != nullptr) {
        functions_[funcName] = func;
        return func;
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Failed to get function %s!", funcName);
    return nullptr;
}

aclnnStatus ApiWrapper::LoadFunctions()
{
    return OK;
}

ApiWrapper::ApiWrapper() = default;

ApiWrapper::~ApiWrapper()
{
    closeSo();
}

IndvMc2ClientWrapper& IndvMc2ClientWrapper::GetInstance(void)
{
    static IndvMc2ClientWrapper instance;
    return instance;
}

aclnnStatus IndvMc2ClientWrapper::IndvMc2ClientWrapperInit(const char* loadSoPath)
{
    hcclAllocComResourceByTilingHandle = nullptr;
    closeSo();
    aclnnStatus ret = openSo(loadSoPath);
    if (ret != OK) {
        OP_LOGW("Failed to load dynamic library %s, retVal = %d.", loadSoPath, ret);
        return ret;
    }
    NNOPBASE_ASSERT_OK_RETVAL(LoadFunctions());
    return OK;
}

aclnnStatus IndvMc2ClientWrapper::HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* TilingData,
                                                               void** commContext)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclAllocComResourceByTilingHandle);

    HcclResult ret = hcclAllocComResourceByTilingHandle(comm, stream, TilingData, commContext);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclAllocComResourceByTiling "
                "function of the mc2_client module, ret = %d, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

IndvMc2ClientWrapper::IndvMc2ClientWrapper(void) {}

IndvMc2ClientWrapper::~IndvMc2ClientWrapper()
{
    hcclAllocComResourceByTilingHandle = nullptr;
    closeSo();
}

aclnnStatus IndvMc2ClientWrapper::LoadFunctions()
{
    hcclAllocComResourceByTilingHandle = LoadFunction<HcclAllocComResourceByTilingFunc>(
        "HcclAllocComResourceByTiling");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclAllocComResourceByTilingHandle);
    return OK;
}

IndvHcclWrapper& IndvHcclWrapper::GetInstance(void)
{
    static IndvHcclWrapper instance;
    return instance;
}

aclnnStatus IndvHcclWrapper::IndvHcclWrapperInit(const char* loadSoPath, const bool needLoadDavidHcclApi)
{
    hcclAllocComResourceByTilingHandle = nullptr;
    hcclGetAicpuOpStreamAndNotifyHandle = nullptr;
    hcomGetCommHandleByGroupHandle = nullptr;
    hcclGetRankIdHandle = nullptr;
    hcclGetCcuTaskInfoHandle = nullptr;

    // 静态Mc2开启场景重构Topo结构落盘Json用
    hcclGetRankSizeHandle = nullptr;
    hcclRankGraphGetLayersHandle = nullptr;
    hcclRankGraphGetRankSizeByLayerHandle = nullptr;
    hcclRankGraphGetTopoTypeByLayerHandle = nullptr;
    hcclGetHcclBufferHandle = nullptr;
    closeSo();

    NNOPBASE_ASSERT_OK_RETVAL(openSo(loadSoPath));
    NNOPBASE_ASSERT_OK_RETVAL(LoadFunctions());
    if (needLoadDavidHcclApi) {
        NNOPBASE_ASSERT_OK_RETVAL(LoadDavidHcclFunctions());
    }
    return OK;
}

aclnnStatus IndvHcclWrapper::HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* TilingData,
                                                          void** commContext)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclAllocComResourceByTilingHandle);

    HcclResult ret = hcclAllocComResourceByTilingHandle(comm, stream, TilingData, commContext);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclAllocComResourceByTiling "
                "function of the hccl module, ret = %d, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclGetAicpuOpStreamAndNotify(HcclComm comm, aclrtStream* opStream, uint8_t notifyCnt,
                                                           void** aicpuNotify)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetAicpuOpStreamAndNotifyHandle);
    HcclResult ret = hcclGetAicpuOpStreamAndNotifyHandle(comm, opStream, notifyCnt, aicpuNotify);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclGetAicpuOpStreamAndNotify "
                "function of the hccl module, ret = %d, comm = %p, notifyCnt = %u.",
                ret, comm, notifyCnt);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcomGetCommHandleByGroup(const char* group, HcclComm* commHandle)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcomGetCommHandleByGroupHandle);

    HcclResult ret = hcomGetCommHandleByGroupHandle(group, commHandle);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcomGetCommHandleByGroup "
                "function of the hccl module, ret = %d, group = %s.",
                ret, group);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclGetRankId(HcclComm comm, uint32_t* rankSize)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetRankIdHandle);

    HcclResult ret = hcclGetRankIdHandle(comm, rankSize);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclGetRankId "
                "function of the hccl module, ret = %u, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclGetCcuTaskInfo(HcclComm comm, void* fusionArgs, void* ccuTaskGroup)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetCcuTaskInfoHandle);

    HcclResult ret = hcclGetCcuTaskInfoHandle(comm, fusionArgs, ccuTaskGroup);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclGetCcuTaskInfo "
                "function of the hccl module, ret = %u, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclRankGraphGetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclRankGraphGetLayersHandle);

    HcclResult ret = hcclRankGraphGetLayersHandle(comm, netLayers, netLayerNum);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclRankGraphGetLayers "
                "function of the hccl module, ret = %u, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclRankGraphGetRankSizeByLayer(HcclComm comm, uint32_t netLayer, uint32_t* rankNum)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclRankGraphGetRankSizeByLayerHandle);

    HcclResult ret = hcclRankGraphGetRankSizeByLayerHandle(comm, netLayer, rankNum);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclRankGraphGetRankSizeByLayer "
                "function of the hccl module, ret = %u, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclRankGraphGetTopoTypeByLayer(HcclComm comm, uint32_t netLayer, uint32_t* topoType)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclRankGraphGetTopoTypeByLayerHandle);

    HcclResult ret = hcclRankGraphGetTopoTypeByLayerHandle(comm, netLayer, topoType);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclRankGraphGetTopoTypeByLayer "
                "function of the hccl module, ret = %u, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetHcclBufferHandle);

    HcclResult ret = hcclGetHcclBufferHandle(comm, buffer, size);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclGetHcclBuffer "
                "function of the hccl module, ret = %u, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclGetRankSize(HcclComm comm, uint32_t* rankSize)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetRankSizeHandle);

    HcclResult ret = hcclGetRankSizeHandle(comm, rankSize);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclGetRankId "
                "function of the hccl module, ret = %u, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

IndvHcclWrapper::IndvHcclWrapper(void) {}

IndvHcclWrapper::~IndvHcclWrapper()
{
    hcclAllocComResourceByTilingHandle = nullptr;
    hcclGetAicpuOpStreamAndNotifyHandle = nullptr;
    hcomGetCommHandleByGroupHandle = nullptr;
    hcclGetCcuTaskInfoHandle = nullptr;
    hcclGetRankIdHandle = nullptr;

    // 静态Mc2开启场景重构Topo结构落盘Json用
    hcclGetRankSizeHandle = nullptr;
    hcclRankGraphGetLayersHandle = nullptr;
    hcclRankGraphGetRankSizeByLayerHandle = nullptr;
    hcclRankGraphGetTopoTypeByLayerHandle = nullptr;
    hcclGetHcclBufferHandle = nullptr;
    closeSo();
}

aclnnStatus IndvHcclWrapper::LoadFunctions()
{
    hcclAllocComResourceByTilingHandle = LoadFunction<HcclAllocComResourceByTilingFunc>(
        "HcclAllocComResourceByTiling");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclAllocComResourceByTilingHandle);
    hcclGetAicpuOpStreamAndNotifyHandle = LoadFunction<HcclGetAicpuOpStreamAndNotifyFunc>(
        "HcclGetAicpuOpStreamAndNotify");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetAicpuOpStreamAndNotifyHandle);
    hcomGetCommHandleByGroupHandle = LoadFunction<HcomGetCommHandleByGroupFunc>("HcomGetCommHandleByGroup");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcomGetCommHandleByGroupHandle);
    hcclGetRankIdHandle = LoadFunction<HcclGetRankIdFunc>("HcclGetRankId");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetRankIdHandle);

    // 静态Mc2开启场景重构Topo结构落盘Json用
    hcclGetRankSizeHandle = LoadFunction<HcclGetRankSizeFunc>("HcclGetRankSize");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetRankSizeHandle);
    hcclRankGraphGetLayersHandle = LoadFunction<HcclRankGraphGetLayersFunc>("HcclRankGraphGetLayers");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclRankGraphGetLayersHandle);
    hcclRankGraphGetRankSizeByLayerHandle = LoadFunction<HcclRankGraphGetRankSizeByLayerFunc>(
        "HcclRankGraphGetRankSizeByLayer");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclRankGraphGetRankSizeByLayerHandle);
    hcclRankGraphGetTopoTypeByLayerHandle = LoadFunction<HcclRankGraphGetTopoTypeByLayerFunc>(
        "HcclRankGraphGetTopoTypeByLayer");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclRankGraphGetTopoTypeByLayerHandle);
    hcclGetHcclBufferHandle = LoadFunction<HcclGetHcclBufferFunc>("HcclGetHcclBuffer");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetHcclBufferHandle);
    return OK;
}

aclnnStatus IndvHcclWrapper::LoadDavidHcclFunctions()
{
    hcclGetCcuTaskInfoHandle = LoadFunction<HcclGetCcuTaskInfoFunc>("HcclGetCcuTaskInfo");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetCcuTaskInfoHandle);
    return OK;
}

NnopbaseSoLoader& NnopbaseSoLoader::GetInstance()
{
    static NnopbaseSoLoader instance;
    return instance;
}

void* NnopbaseSoLoader::FindFunction(const std::vector<std::string>& soPaths, const std::string& funcName)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& soPath : soPaths) {
        if (loadedLibraries.find(soPath) != loadedLibraries.end()) {
            if (functions_[soPath].find(funcName) != functions_[soPath].end()) {
                OP_LOGI("Get func %s from %s successfully.", funcName.c_str(), soPath.c_str());
                return functions_[soPath][funcName];
            } else {
                void* func_ptr = mmDlsym(loadedLibraries[soPath], funcName.c_str());
                if (func_ptr) {
                    functions_[soPath][funcName] = func_ptr;
                    OP_LOGW("Get funcName %s successfully", funcName.c_str());
                    return func_ptr;
                }
                OP_LOGW("Failed to get function %s from %s!", funcName.c_str(), soPath.c_str());
            }
        } else {
            void* handle = mmDlopen(soPath.c_str(), RTLD_LAZY);
            if (!handle) {
                OP_LOGW("Failed to open %s, dlopen error:%s.", soPath.c_str(), mmDlerror());
                continue;
            }
            loadedLibraries[soPath] = handle;
            OP_LOGW("Open %s successfully", soPath.c_str());

            void* func_ptr = mmDlsym(loadedLibraries[soPath], funcName.c_str());
            if (func_ptr) {
                functions_[soPath][funcName] = func_ptr;
                OP_LOGW("Get funcName %s successfully", funcName.c_str());
                return func_ptr;
            }
        }
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Failed to get function %s!", funcName.c_str());
    return nullptr;
}

void NnopbaseSoLoader::closeSo()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& pair : loadedLibraries) {
        mmDlclose(pair.second);
    }
    loadedLibraries.clear();
    functions_.clear();
}

NnopbaseSoLoader::NnopbaseSoLoader() = default;

NnopbaseSoLoader::~NnopbaseSoLoader()
{
    closeSo();
}

IndvRtsWrapper* IndvRtsWrapper::GetInstance(void)
{
    static IndvRtsWrapper instance;
    static std::once_flag initFlag;
    static bool initSuccess = false;
    std::call_once(initFlag, [&]() { initSuccess = (instance.IndvRtsWrapperInit() == OK); });
    return initSuccess ? &instance : nullptr;
}

aclnnStatus IndvRtsWrapper::AclrtBinarySetExceptionCallback(void* binHandle, void* callback, void* userData)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(rtsRegisterExceptionCallbackHandle);
    auto ret = rtsRegisterExceptionCallbackHandle(binHandle, reinterpret_cast<customOpExceptionCallback>(callback),
                                                  userData);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the aclrtBinarySetExceptionCallback "
                "function of the acl_rt module, ret = %d, callback = %p.",
                ret, callback);
        return ACLNN_ERR_INNER;
    }
    return OK;
}

IndvRtsWrapper::IndvRtsWrapper() {};

IndvRtsWrapper::~IndvRtsWrapper()
{
    rtsRegisterExceptionCallbackHandle = nullptr;
    closeSo();
}

aclnnStatus IndvRtsWrapper::IndvRtsWrapperInit()
{
    rtsRegisterExceptionCallbackHandle = nullptr;
    closeSo();
    const char* ascendHomePath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, ascendHomePath);
    OP_CHECK(ascendHomePath != nullptr,
             OP_LOGE_FOR_CONFIG_ERROR_INVALID_ENVIRONMENT_VARIABLE("Loading libacl_rt.so", "ASCEND_HOME_PATH"),
             return ACLNN_ERR_INNER_NULLPTR);
    std::string rtsLibraryPath = std::string(ascendHomePath) + "/lib64/libacl_rt.so";
    NNOPBASE_ASSERT_OK_RETVAL(openSo(rtsLibraryPath.c_str()));
    NNOPBASE_ASSERT_OK_RETVAL(LoadFunctions());
    return OK;
}

aclnnStatus IndvRtsWrapper::LoadFunctions()
{
    rtsRegisterExceptionCallbackHandle = LoadFunction<NnopbaseRegisterExceptionCallbackFunc>(
        "aclrtBinarySetExceptionCallback");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(rtsRegisterExceptionCallbackHandle);
    return OK;
}
} // namespace nnopbase
