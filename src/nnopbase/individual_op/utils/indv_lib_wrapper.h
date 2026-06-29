/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDV_LIB_WRAPPER_H_
#define INDV_LIB_WRAPPER_H_

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "acl/acl_rt.h"
#include "aclnn/acl_meta.h"
#include "hccl/base.h"
#include "mmpa/mmpa_api.h"
#include "nnopbase_error_msg.h"

namespace nnopbase {
class NnopBaseLoadSo {
public:
    virtual ~NnopBaseLoadSo();

    aclnnStatus openSo(const char* loadSoPath);
    void closeSo();

    template <typename T>
    T LoadFunction(const char* const functionName)
    {
        OP_LOGI("Start loading function %s in %s.", functionName, soPath_.c_str());
        T function = reinterpret_cast<T>(mmDlsym(soHandle_, functionName));
        if (function == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Failed to load function %s in %s. "
                    "dlsym error:%s.",
                    functionName, soPath_.c_str(), mmDlerror());
            return nullptr;
        }
        OP_LOGI("Load function %s in %s successfully.", functionName, soPath_.c_str());

        return function;
    }

    // 待子类实现加载所需的符号表
    virtual aclnnStatus LoadFunctions() = 0;

private:
    void* soHandle_ = nullptr;
    std::string soPath_ = "";
};

class ApiWrapper : public NnopBaseLoadSo {
public:
    static ApiWrapper& GetInstance();
    void* GetFunc(const char* funcName);
    aclnnStatus LoadFunctions() override;

private:
    ApiWrapper();
    ~ApiWrapper() override;

    bool hasInit_ = false;
    std::mutex mutex_;
    std::unordered_map<std::string, void*> functions_;
};

class IndvMc2ClientWrapper : public NnopBaseLoadSo {
public:
    static IndvMc2ClientWrapper& GetInstance(void);

    // 调用端必须保证不多线程调用
    aclnnStatus IndvMc2ClientWrapperInit(const char* loadSoPath);
    aclnnStatus HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* TilingData, void** commContext);

private:
    using HcclAllocComResourceByTilingFunc = HcclResult (*)(HcclComm, void*, void*, void**);

    IndvMc2ClientWrapper(void);
    ~IndvMc2ClientWrapper() override;

    aclnnStatus LoadFunctions() override;

    HcclAllocComResourceByTilingFunc hcclAllocComResourceByTilingHandle = nullptr;
};

class IndvHcclWrapper : public NnopBaseLoadSo {
public:
    static IndvHcclWrapper& GetInstance(void);

    // 调用端必须保证不多线程调用
    aclnnStatus IndvHcclWrapperInit(const char* loadSoPath, const bool needLoadDavidHcclApi);
    aclnnStatus HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* TilingData, void** commContext);
    aclnnStatus HcclGetAicpuOpStreamAndNotify(HcclComm comm, aclrtStream* opStream, uint8_t notifyCnt,
                                              void** aicpuNotify);
    aclnnStatus HcomGetCommHandleByGroup(const char* group, HcclComm* commHandle);
    aclnnStatus HcclGetRankId(HcclComm comm, uint32_t* rankSize);
    aclnnStatus HcclGetCcuTaskInfo(HcclComm comm, void* fusionArgs, void* ccuTaskGroup);
    aclnnStatus HcclRankGraphGetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum);
    aclnnStatus HcclRankGraphGetRankSizeByLayer(HcclComm comm, uint32_t netLayer, uint32_t* rankNum);
    aclnnStatus HcclRankGraphGetTopoTypeByLayer(HcclComm comm, uint32_t netLayer, uint32_t* topoType);
    aclnnStatus HcclGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size);
    aclnnStatus HcclGetRankSize(HcclComm comm, uint32_t* rankSize);

private:
    using HcclAllocComResourceByTilingFunc = HcclResult (*)(HcclComm, void*, void*, void**);
    using HcclGetAicpuOpStreamAndNotifyFunc = HcclResult (*)(HcclComm, aclrtStream*, uint8_t, void**);
    using HcomGetCommHandleByGroupFunc = HcclResult (*)(const char*, HcclComm*);
    using HcclGetCcuTaskInfoFunc = HcclResult (*)(HcclComm, void*, void*);
    using HcclGetRankIdFunc = HcclResult (*)(HcclComm, uint32_t*);

    // 静态Mc2开启场景重构Topo结构落盘Json用
    using HcclGetRankSizeFunc = HcclResult (*)(HcclComm, uint32_t*);
    using HcclRankGraphGetLayersFunc = HcclResult (*)(HcclComm, uint32_t**, uint32_t*);
    using HcclRankGraphGetRankSizeByLayerFunc = HcclResult (*)(HcclComm, uint32_t, uint32_t*);
    using HcclRankGraphGetTopoTypeByLayerFunc = HcclResult (*)(HcclComm, uint32_t, uint32_t*);
    using HcclGetHcclBufferFunc = HcclResult (*)(HcclComm, void**, uint64_t*);

    IndvHcclWrapper(void);
    ~IndvHcclWrapper() override;

    aclnnStatus LoadFunctions() override;
    aclnnStatus LoadDavidHcclFunctions();

    HcclAllocComResourceByTilingFunc hcclAllocComResourceByTilingHandle = nullptr;
    HcclGetAicpuOpStreamAndNotifyFunc hcclGetAicpuOpStreamAndNotifyHandle = nullptr;
    HcomGetCommHandleByGroupFunc hcomGetCommHandleByGroupHandle = nullptr;
    HcclGetCcuTaskInfoFunc hcclGetCcuTaskInfoHandle = nullptr;
    HcclGetRankIdFunc hcclGetRankIdHandle = nullptr;

    // 静态Mc2开启场景重构Topo结构落盘Json用
    HcclGetRankSizeFunc hcclGetRankSizeHandle = nullptr;
    HcclRankGraphGetLayersFunc hcclRankGraphGetLayersHandle = nullptr;
    HcclRankGraphGetRankSizeByLayerFunc hcclRankGraphGetRankSizeByLayerHandle = nullptr;
    HcclRankGraphGetTopoTypeByLayerFunc hcclRankGraphGetTopoTypeByLayerHandle = nullptr;
    HcclGetHcclBufferFunc hcclGetHcclBufferHandle = nullptr;
};

class NnopbaseSoLoader {
public:
    static NnopbaseSoLoader& GetInstance();
    void* FindFunction(const std::vector<std::string>& soPaths, const std::string& funcName);
    void closeSo();

private:
    NnopbaseSoLoader();
    ~NnopbaseSoLoader();

    std::unordered_map<std::string, std::unordered_map<std::string, void*>> functions_;
    std::unordered_map<std::string, void*> loadedLibraries;
    std::mutex mutex_;
};

class IndvRtsWrapper : public NnopBaseLoadSo {
public:
    static IndvRtsWrapper* GetInstance(void);
    aclnnStatus AclrtBinarySetExceptionCallback(void* binHandle, void* callback, void* userData = nullptr);

private:
    using customOpExceptionCallback = void (*)(aclrtExceptionInfo*, void*);
    using NnopbaseRegisterExceptionCallbackFunc = aclError (*)(void*, customOpExceptionCallback, void*);

    IndvRtsWrapper();
    ~IndvRtsWrapper() override;

    aclnnStatus IndvRtsWrapperInit();
    aclnnStatus LoadFunctions() override;

    NnopbaseRegisterExceptionCallbackFunc rtsRegisterExceptionCallbackHandle = nullptr;
};
} // namespace nnopbase

#endif // INDV_LIB_WRAPPER_H_
