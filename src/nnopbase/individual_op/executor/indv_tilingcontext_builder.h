/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INDV_TILING_CONTEXT_BUILDER_H_
#define INDV_TILING_CONTEXT_BUILDER_H_

#include "utils/indv_base.h"
#include "utils/indv_debug_assert.h"
#include "indv_compute_node_info.h"
#include "indv_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

enum TilingParserIndex {
    // inputs
    kCompileInfo,
    kPlatformInfo,
    kOpType,
    // outputs
    kCompileInfoStruct,
    // add new output definitions here
    kParseOutputNum
};

struct NnopbaseMemSetCompileInfo {
  int32_t workspace_num = 0;
  int32_t core_num = 0;
  uint32_t ub_size = 0;
  int32_t max_repeat_time = 0;
  bool is_dynamic = false;
  std::vector<int64_t> mask_nums;
  std::vector<int64_t> byte_list;
  std::vector<int64_t> _workspace_index_list;
};

struct NnopbaseKernelExtendInfo {
    const NnopbaseChar *kernelName;
    const NnopbaseChar *kernelType;
};

aclnnStatus NnopbaseMemsetTilingContextInit(NnopbaseExecutor *executor);
aclnnStatus NnopnbaseBuildMemsetTilingContext(NnopbaseExecutor *executor);
aclnnStatus NnopbaseBuildAndRunMemsetTilingParse(NnopbaseExecutor *executor);
aclnnStatus NnopbaseGenMemsetV2TilingFunc(NnopbaseExecutor *executor);
aclnnStatus NnopbaseExecutorPlatFormInfosInit();
void NnopbaseUpdatePlatformInfo(const NnopbaseExecutor *executor);
static constexpr int32_t NNOPBASE_DYNAMIC_PARAM_DEF_NUM = 256;
void NnopbaseTilingBuildOpInputs(NnopbaseExecutor *executor);
void NnopbaseTilingBuildOpOutputs(NnopbaseExecutor *executor);
aclnnStatus NnopbaseTilingContextBuild(NnopbaseExecutor *executor);
aclnnStatus NnopbaseMemsetV2TilingContextInit(NnopbaseExecutor *executor);
aclnnStatus NnopbaseMemsetV2TilingContextBuild(NnopbaseExecutor *executor);
aclnnStatus NnopbaseTilingContextInit(NnopbaseExecutor *executor);
void NnopbaseTilingContextDeInit(NnopbaseExecutor *executor);

static inline uint32_t NnopbaseGetKernelRunContextValuesInitNum(NnopbaseExecutor *executor)
{
    uint32_t num = executor->ownArgs.inputs.nonDynamicCnt + executor->ownArgs.outputs.nonDynamicCnt +
                   static_cast<uint32_t>(kInputsAppendEnd) + gert::TilingContext::kOutputNum;

    if (executor->ownArgs.inputs.hasDynamic || executor->ownArgs.outputs.hasDynamic) {
        num += (executor->ownArgs.inputs.dynamicNum * NNOPBASE_DYNAMIC_PARAM_DEF_NUM +
                executor->ownArgs.outputs.dynamicNum * NNOPBASE_DYNAMIC_PARAM_DEF_NUM);
    }
    return num;
}

#ifdef __cplusplus
}
#endif
#endif
