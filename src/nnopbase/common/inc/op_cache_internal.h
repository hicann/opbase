/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ACL_OP_CACHE_INTERNAL_H__
#define __ACL_OP_CACHE_INTERNAL_H__

#include <stdint.h>
#include <type_traits>
#include <mutex>
#include <vector>

#include "acl/acl_base_rt.h"
#include "profiling/aprof_pub.h"

#include "aclnn/acl_meta.h"
#include "opdev/op_cache.h"
#include "opdev/op_log.h"
#include "thread_local_context.h"

namespace op {
namespace internal {

struct TaskInfo {
    uint64_t attrId{0};
    MsprofGeTaskType type{MSPROF_GE_TASK_TYPE_INVALID};
    uint32_t ration{0};
    OpExecMode execMode{OpExecMode::OP_EXEC_MODE_DEFAULT};
};

class OpExecCacheDfx {
public:
    void SetExceptionDumpInfo(const ExceptionDumpInfo& info) { exceptionDumpInfo_.push_back(info); }

    ExceptionDumpInfo GetExceptionDumpInfo(int32_t index) const
    {
        return IsValidIndex(index, exceptionDumpInfo_.size(), "exceptionDumpInfo") ? exceptionDumpInfo_[index] :
                                                                                     ExceptionDumpInfo{};
    }

    void SetProfilingInfoId(const op::internal::ProfilingInfoId& id) { profilingInfoId_.push_back(id); }

    op::internal::ProfilingInfoId GetProfilingInfoId(int32_t index) const
    {
        return IsValidIndex(index, profilingInfoId_.size(), "profilingInfoId") ? profilingInfoId_[index] :
                                                                                 op::internal::ProfilingInfoId{};
    }

    void SetTaskInfo(const TaskInfo& taskInfo) { taskInfo_.push_back(taskInfo); }

    TaskInfo GetTaskInfo(int32_t index) const
    {
        return IsValidIndex(index, taskInfo_.size(), "taskInfo") ? taskInfo_[index] : TaskInfo{};
    }

    MsprofGeTaskType GetTaskType(int32_t index) const
    {
        return IsValidIndex(index, taskInfo_.size(), "taskInfo") ? taskInfo_[index].type : MSPROF_GE_TASK_TYPE_INVALID;
    }

    void SetOpType(uint32_t opType) { opTypes_.push_back(opType); }

    uint32_t GetOpType(int32_t index) const
    {
        return IsValidIndex(index, opTypes_.size(), "opTypes") ? opTypes_[index] : 0U;
    }

private:
    static bool IsValidIndex(int32_t index, size_t size, const char* fieldName)
    {
        if (index >= 0 && static_cast<size_t>(index) < size) {
            return true;
        }
        OP_LOGW("OpExecCacheDfx field %s: index %d is out of range, current size is %zu, return the default value.",
                fieldName, static_cast<int32_t>(index), size);
        return false;
    }

    std::vector<TaskInfo> taskInfo_;
    std::vector<op::internal::ProfilingInfoId> profilingInfoId_;
    std::vector<ExceptionDumpInfo> exceptionDumpInfo_;
    std::vector<uint32_t> opTypes_;
};

class OpExecCacheWrap {
public:
    OpExecCacheWrap() {}
    explicit OpExecCacheWrap(OpExecCache* cache);
    ~OpExecCacheWrap();

    aclnnStatus Run(void* workspaceAddr, const aclrtStream stream);
    std::string ReportAddr();

    uint64_t magicNumber_{K_CACHE_WRAP_MAGIC_NUMBER};
    OpExecCache* opExecCache_;
    std::vector<void*> cachedTensorList_;
    op::internal::OpLogInfo opLogInfo_;
    int32_t hugeMemPoolIndex_{op::kInvalidHugeMemIndexId};
    void* opExecCacheManager_{nullptr};
};

aclnnStatus DoReportAdditionInfo(void* infoLists, const TaskInfo& taskInfo,
                                 const op::internal::ProfilingInfoId& profilingInfoId);

void DoExceptionDump(void* infoLists, void* workspaceAddr, const std::vector<void*>& tensors,
                     const ExceptionDumpInfo& dumpInfo, const aclrtStream stream);

class OpCacheContext {
public:
    void SetOpCache(OpExecCache* cache);

    OpExecCache* GetOpCache();

private:
    OpExecCache* cache_{nullptr};
};

class OpCacheGuard {
public:
    OpCacheGuard() = default;
    ~OpCacheGuard();
};

OpCacheContext& GetOpCacheContext();
OpExecCacheWrap* CreateCacheWrap(OpExecCache* opExecCache);
void* GetOpExecCacheManager();
void ReleaseOpExecCacheManager(void* ptr);
void DisableOpCacheCount();
void ReinitOpCacheManager();

#ifdef __cplusplus
extern "C" {
#endif
void ResetCacheThreadLocal();
void UnInitPTACacheThreadLocal();
void InitPTACacheThreadLocal();
void AddTensorAddrToCachedList(void* addr);
void SetPTAHashKey(uint64_t hash);
void SetPTACacheHashKey(uint8_t* key, size_t len);

#ifdef __cplusplus
}
#endif
} // namespace internal
} // namespace op

#endif