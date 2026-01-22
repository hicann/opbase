/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __NNOPBASE_LIB_WRAPPER_H__
#define __NNOPBASE_LIB_WRAPPER_H__

#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>
#include "mmpa/mmpa_api.h"
#include "hccl/base.h"
#include "utils/indv_debug_assert.h"

namespace nnopbase {
class NnopBaseLoadSo {
public:
    virtual ~NnopBaseLoadSo() {}

    aclnnStatus openSo(const char *loadSoPath)
    {
        NNOPBASE_ASSERT_NOTNULL_RETVAL(loadSoPath);

        OP_LOGI("Nnopbase starts to load dynamic library %s.", loadSoPath);
        soHandle_ = mmDlopen(loadSoPath, MMPA_RTLD_LAZY);
        if (soHandle_ == nullptr) {
            OP_LOGW("Nnopbase fails to open "
                    "dynamic library %s. dlopen error:%s.", loadSoPath, mmDlerror());
            return ACLNN_ERR_PARAM_INVALID;
        }
        OP_LOGI("Nnopbase successed to load dynamic library %s.", loadSoPath);

        soPath_ = loadSoPath;
        return OK;
    }

    void closeSo()
    {
        if (soHandle_ != nullptr) {
            mmDlclose(soHandle_);
            soHandle_ = nullptr;
            soPath_ = "";
        }
    }

    template<typename T>
    T LoadFunction(const char *const functionName)
    {
        OP_LOGI("Nnopbase starts to load function %s in %s.", functionName, soPath_.c_str());
        T function = reinterpret_cast<T>(mmDlsym(soHandle_, functionName));
        if (function == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Nnopbase fails to load function %s in %s. "
                    "dlsym error:%s.", functionName, soPath_.c_str(), mmDlerror());
            return nullptr;
        }
        OP_LOGI("Nnopbase sucessed to load function %s in %s.", functionName, soPath_.c_str());

        return function;
    }

    // 待子类实现加载所需的符号表
    virtual aclnnStatus LoadFunctions() = 0;

private:
    void *soHandle_ = nullptr;
    std::string soPath_ = "";
};

class ApiWrapper : public NnopBaseLoadSo {
public:
    static ApiWrapper& GetInstance()
    {
        static ApiWrapper instance;
        return instance;
    }

    void *GetFunc(const char *funcName)
    {
        const std::lock_guard<std::mutex> lk(mutex_);
        if (!hasInit_) {
            if (openSo("libopapi_math.so") != OK) {
                OP_LOGW("Load libopapi_math.so failed, trying alternative dynamic library.");
                // libopapi.so在cann包的lib64路径下，正常配置一定在LD_LIBRARY_PATH下
                if (openSo("libopapi.so") != OK) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Open libopapi_math.so and libopapi.so failed! Please check if built-in opp "
                            "package is installed, and libopapi_math.so or libopapi.so path is in LD_LIBRARY_PATH env!");
                    return nullptr;
                }
            }
            hasInit_ = true;
        }
        if (funcName == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Get funcName is nullptr!");
            return nullptr;
        }
        const auto &it = functions_.find(funcName);
        if (it != functions_.end()) {
            return it->second;
        }
        void *const func = LoadFunction<void *>(funcName);
        if (func != nullptr) {
            functions_[funcName] = func;
            return func;
        }
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Get func %s failed!", funcName);
        return nullptr;
    }

    aclnnStatus LoadFunctions() override
    {
        return OK;
    }

private:
    ApiWrapper() = default;
    ~ApiWrapper() override
    {
        closeSo();
    }

    bool hasInit_ = false;
    std::mutex mutex_;
    std::unordered_map<std::string, void *> functions_;
};


class IndvHcclWrapper : public NnopBaseLoadSo {
public:
    static IndvHcclWrapper& GetInstance(void)
    {
        static IndvHcclWrapper instance;
        return instance;
    }

    // 调用端必须保证不多线程调用
    aclnnStatus IndvHcclWrapperInit(const char *loadSoPath, const bool needLoadDavidHcclApi)
    {
        hcclAllocComResourceByTilingHandle = nullptr;
        hcclGetAicpuOpStreamAndNotifyHandle = nullptr;
        hcomGetCommHandleByGroupHandle = nullptr;
        hcclGetRankIdHandle = nullptr;
        hcclGetCcuTaskInfoHandle = nullptr;
        closeSo();

        NNOPBASE_ASSERT_OK_RETVAL(openSo(loadSoPath));
        NNOPBASE_ASSERT_OK_RETVAL(LoadFunctions());
        if(needLoadDavidHcclApi) {
            NNOPBASE_ASSERT_OK_RETVAL(LoadDavidHcclFunctions());
        }
        return OK;
    }

    aclnnStatus HcclAllocComResourceByTiling(
        HcclComm comm, void *stream, void *TilingData, void **commContext)
    {
        NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclAllocComResourceByTilingHandle);

        HcclResult ret = hcclAllocComResourceByTilingHandle(comm, stream, TilingData, commContext);
        if (ret != HCCL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "Nnopbase fails to invoke the HcclAllocComResourceByTiling "
                    "function of the hccl module. ret = %d, comm = %p.", ret, comm);
            return ACLNN_ERR_INNER;
        }

        return OK;
    }

    aclnnStatus HcclGetAicpuOpStreamAndNotify(HcclComm comm, rtStream_t *Opstream, uint8_t notifyCnt, void **aicpuNotify)
    {
        NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetAicpuOpStreamAndNotifyHandle);

        HcclResult ret = hcclGetAicpuOpStreamAndNotifyHandle(comm, Opstream, notifyCnt, aicpuNotify);
        if (ret != HCCL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "Nnopbase fails to invoke the HcclGetAicpuOpStreamAndNotify "
                    "function of the hccl module. ret = %d, comm = %p, notifyCnt = %u.", ret, comm, notifyCnt);
            return ACLNN_ERR_INNER;
        }

        return OK;
    }

    aclnnStatus HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle)
    {
        NNOPBASE_ASSERT_NOTNULL_RETVAL(hcomGetCommHandleByGroupHandle);

        HcclResult ret = hcomGetCommHandleByGroupHandle(group, commHandle);
        if (ret != HCCL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "Nnopbase fails to invoke the HcomGetCommHandleByGroup "
                    "function of the hccl module. ret = %d, group = %s.", ret, group);
            return ACLNN_ERR_INNER;
        }

        return OK;
    }

    aclnnStatus HcclGetRankId(HcclComm comm, uint32_t *rankSize)
    {
        NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetRankIdHandle);

        HcclResult ret = hcclGetRankIdHandle(comm, rankSize);
        if (ret != HCCL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "Nnopbase fails to invoke the HcclGetRankId "
                    "function of the hccl module. ret = %u, comm = %p.", ret, comm);
            return ACLNN_ERR_INNER;
        }

        return OK;
    }

    aclnnStatus HcclGetCcuTaskInfo(HcclComm comm, void* fusionArgs, void* ccuTaskGroup)
    {
        NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetCcuTaskInfoHandle);

        HcclResult ret = hcclGetCcuTaskInfoHandle(comm, fusionArgs, ccuTaskGroup);
        if (ret != HCCL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "Nnopbase fails to invoke the HcclGetCcuTaskInfo "
                    "function of the ccu task info. ret = %u, comm = %p.", ret, comm);
            return ACLNN_ERR_INNER;
        }

        return OK;
    }

private:
    using HcclAllocComResourceByTilingFunc = HcclResult (*)(HcclComm, void *, void *, void **);
    using HcclGetAicpuOpStreamAndNotifyFunc = HcclResult (*)(HcclComm, rtStream_t *, uint8_t, void **);
    using HcomGetCommHandleByGroupFunc = HcclResult (*)(const char *, HcclComm *);
    using HcclGetCcuTaskInfoFunc = HcclResult (*)(HcclComm, void *, void *);
    using HcclGetRankIdFunc = HcclResult (*)(HcclComm, uint32_t *);

    IndvHcclWrapper(void) {}

    ~IndvHcclWrapper() override
    {
        hcclAllocComResourceByTilingHandle = nullptr;
        hcclGetAicpuOpStreamAndNotifyHandle = nullptr;
        hcomGetCommHandleByGroupHandle = nullptr;
        hcclGetCcuTaskInfoHandle = nullptr;
        hcclGetRankIdHandle = nullptr;

        closeSo();
    }

    aclnnStatus LoadFunctions() override
    {
        hcclAllocComResourceByTilingHandle = LoadFunction<HcclAllocComResourceByTilingFunc>("HcclAllocComResourceByTiling");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclAllocComResourceByTilingHandle);
        hcclGetAicpuOpStreamAndNotifyHandle =
            LoadFunction<HcclGetAicpuOpStreamAndNotifyFunc>("HcclGetAicpuOpStreamAndNotify");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetAicpuOpStreamAndNotifyHandle);
        hcomGetCommHandleByGroupHandle = LoadFunction<HcomGetCommHandleByGroupFunc>("HcomGetCommHandleByGroup");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(hcomGetCommHandleByGroupHandle);
        hcclGetRankIdHandle = LoadFunction<HcclGetRankIdFunc>("HcclGetRankId");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetRankIdHandle);
        return OK;
    }

    aclnnStatus LoadDavidHcclFunctions() {
        hcclGetCcuTaskInfoHandle = LoadFunction<HcclGetCcuTaskInfoFunc>("HcclGetCcuTaskInfo");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetCcuTaskInfoHandle);

        return OK;
    }

    HcclAllocComResourceByTilingFunc hcclAllocComResourceByTilingHandle = nullptr;
    HcclGetAicpuOpStreamAndNotifyFunc hcclGetAicpuOpStreamAndNotifyHandle = nullptr;
    HcomGetCommHandleByGroupFunc hcomGetCommHandleByGroupHandle = nullptr;
    HcclGetCcuTaskInfoFunc hcclGetCcuTaskInfoHandle = nullptr;
    HcclGetRankIdFunc hcclGetRankIdHandle = nullptr;
};

class NnopbaseSoLoader {
public:
    static NnopbaseSoLoader& GetInstance() {
        static NnopbaseSoLoader instance;
        return instance;
    }

    void* FindFunction(const std::vector<std::string>& soPaths, const std::string& funcName) {
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
                    OP_LOGW("Get func %s from %s failed!", funcName.c_str(), soPath.c_str());
                }
            } else {
                void* handle = mmDlopen(soPath.c_str(), RTLD_LAZY);
                if (!handle) {
                    OP_LOGW("Open %s failed, dlopen error:%s.", soPath.c_str(), mmDlerror());
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
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Get func %s failed!", funcName.c_str());
        return nullptr;
    }

    void closeSo() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& pair : loadedLibraries) {
            mmDlclose(pair.second);
        }
        loadedLibraries.clear();
        functions_.clear();
    }

private:
    NnopbaseSoLoader() = default;
    ~NnopbaseSoLoader()
    {
        closeSo();
    }
    std::unordered_map<std::string, std::unordered_map<std::string, void*>> functions_;
    std::unordered_map<std::string, void*> loadedLibraries;
    std::mutex mutex_;
};
} // nnopbase

#endif // __NNOPBASE_LIB_WRAPPER_H__
