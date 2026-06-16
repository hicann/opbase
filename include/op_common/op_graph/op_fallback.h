/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_OP_GRAPH_OP_FALLBACK_H_
#define INC_OP_GRAPH_OP_FALLBACK_H_

#include "aclnn/aclnn_base.h"
#include "exe_graph/runtime/op_execute_context.h"
#include "exe_graph/runtime/tensor.h"
#include "op_fallback_internal.h"
#include "register/op_impl_kernel_registry.h"
#include "register/op_impl_registry.h"

namespace fallback {

#define EXEC_OPAPI_CMD(aclnn_api, ...)                                                                                \
    ({                                                                                                                \
        static auto ret = GRAPH_SUCCESS;                                                                              \
        do {                                                                                                          \
            static const auto ResetCacheThreadLocalAddr = GetOpApiFuncAddr("ResetCacheThreadLocal");                  \
            static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");             \
            static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                           \
            if (getWorkspaceSizeFuncAddr == nullptr || opApiFuncAddr == nullptr ||                                    \
                ResetCacheThreadLocalAddr == nullptr) {                                                               \
                OP_LOGE(                                                                                              \
                    "aclnnfallback", "%s or %s not in  %s or %s  or ResetCacheThreadLocal not found.",                \
                    #aclnn_api "GetWorkspaceSize", #aclnn_api, GetOpApiLibName(), GetOpApiLibName());                 \
                ret = GRAPH_FAILED;                                                                                   \
                break;                                                                                                \
            }                                                                                                         \
            auto ResetCacheThreadLocalFunc = reinterpret_cast<ResetCacheThreadLocal>(ResetCacheThreadLocalAddr);      \
            ResetCacheThreadLocalFunc();                                                                              \
            uint64_t workspace_size = 0;                                                                              \
            uint64_t* workspace_size_addr = &workspace_size;                                                          \
            aclOpExecutor* executor = nullptr;                                                                        \
            aclOpExecutor** executor_addr = &executor;                                                                \
            auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                    \
            static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);        \
            auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                     \
            if (workspace_status != 0) {                                                                              \
                OP_LOGE("aclnnfallback", "call %s failed:", #aclnn_api);                                              \
                ret = GRAPH_FAILED;                                                                                   \
                break;                                                                                                \
            }                                                                                                         \
            void* workspace_addr = nullptr;                                                                           \
            if (workspace_size > 0) {                                                                                 \
                workspace_addr = host_api_ctx->MallocWorkspace(workspace_size);                                       \
                if (workspace_addr == nullptr) {                                                                      \
                    OP_LOGE("aclnnfallback", "call %s allocate workspace failed", #aclnn_api);                        \
                    ret = GRAPH_FAILED;                                                                               \
                    break;                                                                                            \
                }                                                                                                     \
            }                                                                                                         \
            auto acl_stream = host_api_ctx->GetStream();                                                              \
            auto acl_call = [converted_params, workspace_addr, workspace_size, host_api_ctx, acl_stream,              \
                             executor]() -> int {                                                                     \
                using OpApiFunc = int (*)(void*, uint64_t, aclOpExecutor*, const aclrtStream);                        \
                OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                     \
                auto op_api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                    \
                ReleaseConvertTypes(converted_params);                                                                \
                host_api_ctx->FreeWorkspace();                                                                        \
                if (op_api_ret != 0) {                                                                                \
                    OP_LOGE(                                                                                          \
                        "aclnnfallback", "call %s allocate workspace failed op_api_ret: %d", #aclnn_api, op_api_ret); \
                    return GRAPH_FAILED;                                                                              \
                }                                                                                                     \
                return op_api_ret;                                                                                    \
            };                                                                                                        \
                                                                                                                      \
            ret = acl_call();                                                                                         \
        } while (false);                                                                                              \
        (ret);                                                                                                        \
    })

} // namespace fallback

#endif // INC_OP_GRAPH_OP_FALLBACK_H_
