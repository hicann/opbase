/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "indv_tilingcontext_builder.h"
#include <mutex>
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_format.h"
#include "kernel_utils.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "utils/indv_soc.h"

#ifdef __cplusplus
extern "C" {
#endif

struct NnopbasePlatformInfos {
    bool isInit;
    uint32_t aicNum;
    uint32_t aivNum;
    std::unique_ptr<fe::PlatFormInfos> infos;
    std::unique_ptr<fe::PlatFormInfos> memsetInfos;
};

static thread_local struct NnopbasePlatformInfos g_nnopbasePlatformMgr = {false, 0, 0, nullptr, nullptr};
static constexpr uint32_t NNOPBASE_PADDING = 32;

static void NnopbaseUpdateCoreNum(uint32_t aicNum, uint32_t aivNum)
{
    OP_LOGD("update core num, aicNum is %u, aivNum is %u", aicNum, aivNum);
    const std::string socInfoKey = "SoCInfo";
    const std::string vecCoreCntKey = "vector_core_cnt";
    const std::string cubeCoreCntKey = "cube_core_cnt";
    std::map<std::string, std::string> res;
    if (g_nnopbasePlatformMgr.infos->GetPlatformResWithLock(socInfoKey, res)) {
        res[vecCoreCntKey] = std::to_string(aivNum);
        res[cubeCoreCntKey] = std::to_string(aicNum);
        g_nnopbasePlatformMgr.infos->SetPlatformResWithLock(socInfoKey, res);
        g_nnopbasePlatformMgr.infos->SetCoreNum(aivNum);
    }
    if (g_nnopbasePlatformMgr.memsetInfos->GetPlatformResWithLock(socInfoKey, res)) {
        res[vecCoreCntKey] = std::to_string(aivNum);
        res[cubeCoreCntKey] = std::to_string(aicNum);
        g_nnopbasePlatformMgr.memsetInfos->SetPlatformResWithLock(socInfoKey, res);
        g_nnopbasePlatformMgr.memsetInfos->SetCoreNum(aivNum);
    }
}

void NnopbaseUpdatePlatformInfo(const NnopbaseExecutor *executor)
{
    if (g_nnopbasePlatformMgr.aicNum == executor->coreNum.aicNum &&
    g_nnopbasePlatformMgr.aivNum == executor->coreNum.aivNum) {
        // core num不变，不需要更新
        return;
    }
    g_nnopbasePlatformMgr.aicNum = executor->coreNum.aicNum;
    g_nnopbasePlatformMgr.aivNum = executor->coreNum.aivNum;
    if (!g_nnopbasePlatformMgr.isInit) {
        // 没有初始化过，指针是空的，不在此处更新
        return;
    }
    NnopbaseUpdateCoreNum(executor->coreNum.aicNum, executor->coreNum.aivNum);
}

aclnnStatus NnopbaseExecutorPlatFormInfosInit(void)
{
    if (g_nnopbasePlatformMgr.isInit) {
        return OK;
    }
    OP_LOGD("NnopbaseExecutorPlatFormInfosInit Start");
    g_nnopbasePlatformMgr.infos = std::make_unique<fe::PlatFormInfos>();
    g_nnopbasePlatformMgr.memsetInfos = std::make_unique<fe::PlatFormInfos>();
    int32_t deviceId = -1;
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtGetDevice(&deviceId));
    NNOPBASE_ASSERT_RTOK_RETVAL(fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(deviceId,
        *g_nnopbasePlatformMgr.infos, true));
    NNOPBASE_ASSERT_RTOK_RETVAL(fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(deviceId,
        *g_nnopbasePlatformMgr.memsetInfos, true));
    if (nnopbase::IndvSoc::GetInstance().IsCouplingArch()) {
        g_nnopbasePlatformMgr.infos->SetCoreNumByCoreType("AiCore");
    } else {
        g_nnopbasePlatformMgr.infos->SetCoreNumByCoreType("VectorCore");
    }
    nnopbase::NnopbaseGetCoreNum(&g_nnopbasePlatformMgr.aicNum, &g_nnopbasePlatformMgr.aivNum);
    NnopbaseUpdateCoreNum(g_nnopbasePlatformMgr.aicNum, g_nnopbasePlatformMgr.aivNum);
    g_nnopbasePlatformMgr.isInit = true;
    return OK;
}

static inline size_t NnopbaseKernelRunContextOffset(const uint32_t num)
{
    return num * sizeof(void*) + sizeof(NnopbaseKernelRunContext) + NNOPBASE_PADDING;
}

static inline size_t NnopbaseKernelRunContextLen(const uint32_t num)
{
    return NnopbaseKernelRunContextOffset(num) + (num) * sizeof(NnopbaseAsyncAnyValue);
}

void NnopbaseTilingBuildOpInputs(NnopbaseExecutor *executor)
{
    NnopbaseAsyncAnyValue **values = executor->contextExt.context->values;
    NnopbaseCompileTimeTensorDesc *inputTd = executor->contextExt.nodeExt.inputTdStart;
    for (uint32_t i = 0; i < executor->args->inputs.num; i++) {
        if (!executor->args->inputs.extTensors[i].isNull) {
            NnopbaseCompileTimeTensorDesc *td = inputTd + i;
            GertTensor *tensor = &executor->args->inputs.extTensors[i].rt2Tensor;
            td->SetDataType(tensor->GetDataType());
            gert::StorageFormat storageFormat = tensor->GetFormat();
            td->SetStorageFormat(storageFormat.GetStorageFormat());
            td->SetOriginFormat(storageFormat.GetOriginFormat());
            td->SetExpandDimsType(storageFormat.GetExpandDimsType());
            values[i]->data.pointer = tensor;
        } else {
            values[i]->data.pointer = nullptr;
        }
        values[i]->deleter = nullptr;
    }
}

void NnopbaseTilingBuildOpOutputs(NnopbaseExecutor *executor)
{
    NnopbaseCompileTimeTensorDesc *outputTd = executor->contextExt.nodeExt.outputTdStart;
    NnopbaseAsyncAnyValue **values = &executor->contextExt.context->values[executor->args->inputs.num];
    const uint32_t count = executor->args->outputs.num;
    for (uint32_t i = 0; i < count; i++) {
        if (!executor->args->outputs.extTensors[i].isNull) {
            NnopbaseCompileTimeTensorDesc *td = outputTd + i;
            GertTensor *tensor = &executor->args->outputs.extTensors[i].rt2Tensor;
            td->SetDataType(tensor->GetDataType());
            gert::StorageFormat storageFormat = tensor->GetFormat();
            td->SetStorageFormat(storageFormat.GetStorageFormat());
            td->SetOriginFormat(storageFormat.GetOriginFormat());
            td->SetExpandDimsType(storageFormat.GetExpandDimsType());
            values[i]->data.pointer = tensor;
        } else {
            values[i]->data.pointer = nullptr;
        }
        values[i]->deleter = nullptr;
    }
}

// 设置tilingkey, blockdim, sechmode, 直接使用value上的内存，在executor上保存下来地址
// 若无动态输入，在不需要重新刷新，有动态输入场景需要重新刷新
static void NnopbaseTilingSetContextOutputStep1(NnopbaseExecutor *const executor)
{
    NnopbaseAsyncAnyValue **values = executor->contextExt.context->values;
    size_t index = executor->contextExt.context->input_size;
    /* kOutputTilingKey. */
    executor->tilingKey = op::internal::PtrCastTo<uint64_t>(values[index]->data.inplace);
    values[index]->deleter = nullptr;
    index++;
    /* kOutputBlockDim. */
    executor->blockDim = op::internal::PtrCastTo<uint32_t>(values[index]->data.inplace);
    values[index]->deleter = nullptr;
    index++;
    /* kOutputAtomicCleanFlag. */
    executor->needAtomic = op::internal::PtrCastTo<bool>(values[index]->data.inplace);
    values[index]->deleter = nullptr;
    /* 4 for OutputTilingData, OutputWorkspace, OutputTilingCond, OutputScheduleMode. */
    index += 4U;
    /* kOutputScheduleMode. */
    executor->scheMode = op::internal::PtrCastTo<uint32_t>(values[index]->data.inplace);
    values[index]->deleter = nullptr;
    index++;
    /* kOutputLocalMemorySize. */
    executor->dynUbufSize = op::internal::PtrCastTo<uint32_t>(values[index]->data.inplace);
    values[index]->deleter = nullptr;
    index++;
    executor->aicpuBlockDim = op::internal::PtrCastTo<uint32_t>(values[index]->data.inplace);
    values[index]->deleter = nullptr;
}

// 设置TilingData和Workspace，实际内存放在tilingInfo里，value里的addr保存
static void NnopbaseTilingSetContextOutputStep2(NnopbaseExecutor *const executor)
{
    NnopbaseAsyncAnyValue **values = executor->contextExt.context->values;
    auto &tilingInfo = executor->args->tilingInfo;
    // 3 for tilingkey, blockdim, atomic
    size_t index = executor->contextExt.context->input_size + 3U;
    /* kOutputTilingData. */
    values[index]->data.pointer = op::internal::PtrCastTo<void>(tilingInfo.tilingData);
    values[index]->deleter = nullptr;
    index++;
    /* kOutputWorkspace. */
    values[index]->data.pointer = op::internal::PtrCastTo<void>(tilingInfo.workspacesSizes);
    values[index]->deleter = nullptr;
}

aclnnStatus NnopbaseTilingContextBuild(NnopbaseExecutor *executor)
{
    if (!g_nnopbasePlatformMgr.isInit) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorPlatFormInfosInit());
    }
    NnopbaseAsyncAnyValue **values = executor->contextExt.context->values;
    // executor->args->inputs.nonDynamicCnt > executor->args->inputs.requiredCnt for option input
    if ((!executor->contextExt.hasPrepared) || executor->args->inputs.hasDynamic || executor->args->outputs.hasDynamic ||
        (executor->args->inputs.nonDynamicCnt > executor->args->inputs.requiredCnt)) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseTilingContextUpdtPrepare(executor));
        uint32_t index = executor->args->inputs.num + executor->args->outputs.num;
        /* kInputsCompileInfo. */
        values[index]->data.pointer = nullptr;
        values[index]->deleter = nullptr;
        index++;
        /* kInputsPlatformInfo. */
        index++;
        /* kInputsTilingFunc. */
        values[index]->data.pointer = nullptr;
        values[index]->deleter = nullptr;
        index++;
        /* kInputsDetermintstic. */
        values[index]->data.pointer = reinterpret_cast<void *>(NnopbaseGetGlobalDeterministic());
        values[index]->deleter = nullptr;
        NnopbaseTilingSetContextOutputStep1(executor);
        executor->contextExt.hasPrepared = true;
    }
    /* kInputsPlatformInfo. */
    uint32_t platformInfIndex = executor->args->inputs.num + executor->args->outputs.num + 1U;
    values[platformInfIndex]->data.pointer = static_cast<void*>(g_nnopbasePlatformMgr.infos.get());
    values[platformInfIndex]->deleter = nullptr;

    *(executor->tilingKey) = 0U;
    *(executor->blockDim) = 0U;
    *(executor->scheMode) = 0U;
    *(executor->needAtomic) = false;
    *(executor->aicpuBlockDim) = 0U;
    *(executor->dynUbufSize) = 0U;
    NnopbaseTilingSetContextOutputStep2(executor);
    NnopbaseTilingBuildOpInputs(executor);
    NnopbaseTilingBuildOpOutputs(executor);

    if (executor->attrs.num > 0LU) {
        return NnopbaseTilingBuildOpAttrs(executor);
    }
    return OK;
}

static void NnopbaseTilingTensorFiling(NnopbaseExecutor *executor, NnopbaseAsyncAnyValue **values,
    NnopbaseCompileTimeTensorDesc *tensorDesc)
{
    OP_LOGD("Show memsetv2 attribute for %s, init values %llu.", executor->opType,
        executor->args->binInfo->initValues.size());
    for (uint32_t i = 0; i < executor->args->binInfo->initValues.size(); i++) {
        size_t irIndex = executor->args->binInfo->initValues[i].irIndex;
        const uint32_t startIndex =
            executor->args->outputs.paramDescs.instances[irIndex].startIndex;

        if (executor->args->outputs.extTensors[startIndex].isNull) {
            continue;
        }

        NnopbaseCompileTimeTensorDesc *td = tensorDesc + i;
        GertTensor *tensor = &executor->args->outputs.extTensors[startIndex].rt2Tensor;
        td->SetDataType(tensor->GetDataType());

        gert::StorageFormat storageFormat = tensor->GetFormat();
        td->SetStorageFormat(storageFormat.GetStorageFormat());
        td->SetOriginFormat(storageFormat.GetOriginFormat());
        td->SetExpandDimsType(storageFormat.GetExpandDimsType());

        values[i]->data.pointer = tensor;
        values[i]->deleter = nullptr;
    }
}

static void NnopbaseBuildMemsetV2Outputs(NnopbaseExecutor *executor)
{
    size_t inputNum = executor->args->binInfo->tensorNeedMemSetV2;
    NnopbaseAsyncAnyValue **values = &executor->args->binInfo->memsetInfo->contextExt.context->values[inputNum];
    NnopbaseCompileTimeTensorDesc *outputTd = executor->args->binInfo->memsetInfo->contextExt.nodeExt.outputTdStart;

    NnopbaseTilingTensorFiling(executor, values, outputTd);
}

static void NnopbaseBuildMemsetV2Inputs(NnopbaseExecutor *executor)
{
    NnopbaseAsyncAnyValue **values = executor->args->binInfo->memsetInfo->contextExt.context->values;
    NnopbaseCompileTimeTensorDesc *inputTd = executor->args->binInfo->memsetInfo->contextExt.nodeExt.inputTdStart;

    NnopbaseTilingTensorFiling(executor, values, inputTd);
}

aclnnStatus NnopbaseInitContext(NnopbaseKernelRunContextExt *contextExt, const uint32_t num)
{
    const size_t len = NnopbaseKernelRunContextLen(num);
    NnopbaseKernelRunContext *context = (NnopbaseKernelRunContext *)malloc(len);

    NNOPBASE_ASSERT_NOTNULL_RETVAL(context);
    contextExt->contextLen = static_cast<uint32_t>(len);
    contextExt->context = context; // free with executor
    context->input_size = 0LU;
    context->output_size = 0LU;
    context->compute_node_info = contextExt->nodeExt.node;
    context->kernel_extend_info = nullptr;
    context->output_start = nullptr;
    auto value = op::internal::PtrCastTo<NnopbaseAsyncAnyValue>(op::internal::PtrCastTo<NnopbaseUChar>(context) + NnopbaseKernelRunContextOffset(num));
    for (uint32_t i = 0; i < num; i++) {
        context->values[i] = value + i;
    }
    contextExt->hasPrepared = false;
    return OK;
}

aclnnStatus NnopbaseMemsetV2TilingContextInit(NnopbaseExecutor *executor)
{
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseComputeNodeInfoInit(&executor->args->binInfo->memsetInfo->contextExt.nodeExt));
    NnopbaseKernelRunContextExt *contextExt = &executor->args->binInfo->memsetInfo->contextExt;
    const uint32_t num = executor->args->binInfo->tensorNeedMemSetV2 * 2U + static_cast<uint32_t>(kInputsAppendEnd) +
                         static_cast<uint32_t>(gert::TilingContext::kOutputNum) + NNOPBASE_PADDING;
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseInitContext(contextExt, num));

    return OK;
}

aclnnStatus NnopbaseMemsetV2TilingContextBuild(NnopbaseExecutor *executor)
{
    if (!g_nnopbasePlatformMgr.isInit) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorPlatFormInfosInit());
    }
    NnopbaseKernelRunContextExt *contextExt = &executor->args->binInfo->memsetInfo->contextExt;
    NnopbaseAsyncAnyValue **values = contextExt->context->values;
    if ((!contextExt->hasPrepared)) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseMemsetV2TilingContextUpdatePrepare(executor));
        uint32_t index = executor->args->binInfo->tensorNeedMemSetV2 * 2U;
        /* kInputsCompileInfo. */
        values[index]->data.pointer = nullptr;
        values[index]->deleter = nullptr;
        index++;
        /* kInputsPlatformInfo. */
        index++;
        /* kInputsTilingFunc. */
        values[index]->data.pointer = nullptr;
        values[index]->deleter = nullptr;
        index++;
        /* kInputsDetermintstic. */
        values[index]->data.pointer = nullptr;
        values[index]->deleter = nullptr;

        size_t input_size = contextExt->context->input_size;
        /* kOutputTilingKey. */
        executor->args->binInfo->memsetInfo->tilingKey =
            op::internal::PtrCastTo<uint64_t>(values[input_size]->data.inplace);
        values[input_size]->deleter = nullptr;
        input_size++;
        /* kOutputBlockDim. */
        executor->args->binInfo->memsetInfo->blockDim =
            op::internal::PtrCastTo<uint32_t>(values[input_size]->data.inplace);
        values[input_size]->deleter = nullptr;
        input_size++;
        /* kOutputAtomicCleanFlag, does not support atomic clean. */
        values[input_size]->deleter = nullptr;

        /* 4 for OutputTilingData, OutputWorkspace, OutputTilingCond, OutputScheduleMode. */
        input_size += 4U;
        /* kOutputScheduleMode. */
        executor->args->binInfo->memsetInfo->scheMode = op::internal::PtrCastTo<uint32_t>(values[input_size]->data.inplace);
        values[input_size]->deleter = nullptr;
        input_size++;
        /* kOutputLocalMemorySize. */
        executor->args->binInfo->memsetInfo->dynUbufSize = op::internal::PtrCastTo<uint32_t>(values[input_size]->data.inplace);
        values[input_size]->deleter = nullptr;
        input_size++;
        values[input_size]->deleter = nullptr;

        contextExt->hasPrepared = true;
    }
    /* kInputsPlatformInfo. */
    uint32_t platformInfIndex = executor->args->binInfo->tensorNeedMemSetV2 * 2U + 1U;
    values[platformInfIndex]->data.pointer = static_cast<void*>(g_nnopbasePlatformMgr.infos.get());
    values[platformInfIndex]->deleter = nullptr;

    *(executor->args->binInfo->memsetInfo->blockDim) = 0U;
    *(executor->args->binInfo->memsetInfo->tilingKey) = 0U;
    *(executor->args->binInfo->memsetInfo->scheMode) = 0U;
    *(executor->args->binInfo->memsetInfo->dynUbufSize) = 0U;

    // 3 for tilingkey, blockdim, atomic
    size_t index = contextExt->context->input_size + 3U;
    /* kOutputTilingData. */
    values[index]->data.pointer = static_cast<void *>(executor->args->binInfo->memsetInfo->tilingData);
    values[index]->deleter = nullptr;
    index++;
    /* kOutputWorkspace. */
    values[index]->data.pointer = static_cast<void *>(executor->args->binInfo->memsetInfo->workspacesSizes);
    values[index]->deleter = nullptr;

    NnopbaseBuildMemsetV2Inputs(executor);
    NnopbaseBuildMemsetV2Outputs(executor);

    return NnopbaseMemsetV2TilingBuildOpAttrs(executor);
}

aclnnStatus NnopbaseTilingContextInit(NnopbaseExecutor *executor)
{
    NnopbaseKernelRunContextExt *contextExt = &executor->contextExt;
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseComputeNodeInfoInit(&contextExt->nodeExt));
    const uint32_t num = NnopbaseGetKernelRunContextValuesInitNum(executor);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseInitContext(contextExt, num));

    return OK;
}

void NnopbaseTilingContextDeInit(NnopbaseExecutor *executor)
{
    FREE(executor->contextExt.context);
    NnopbaseComputeNodeInfoDeInit(&executor->contextExt.nodeExt);
}

aclnnStatus NnopbaseMemsetTilingContextInit(NnopbaseExecutor *executor)
{
    NnopbaseKernelRunContextExt *contextExt = &executor->args->binInfo->memsetInfo->contextExt;
    const uint32_t num = static_cast<uint32_t>(executor->args->binInfo->initValues.size()) +
        static_cast<uint32_t>(kInputsAppendEnd) + static_cast<uint32_t>(gert::TilingContext::kOutputNum) +
        NNOPBASE_PADDING;
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseInitContext(contextExt, num));

    NnopbaseComputeNodeInfoExt *nodeExt = &contextExt->nodeExt;
    NnopbaseComputeNodeInfo *node = nodeExt->node;
    const size_t count = executor->args->binInfo->initValues.size();
    node->nodeType = NNOPBASE_MEMSET_OP_NAME.c_str();
    node->nodeName = NNOPBASE_MEMSET_OP_NAME.c_str();
    node->irInputsNum = count;
    node->inputsNum = count + 1;  // input[0] is not used for workspace checking

    nodeExt->inputTdStart = op::internal::PtrCastTo<NnopbaseCompileTimeTensorDesc>(nodeExt->instStart + node->irInputsNum);
    nodeExt->outputTdStart = nodeExt->inputTdStart + node->inputsNum;
    nodeExt->attrStart = (NnopbaseRuntimeAttrsDef *)(nodeExt->outputTdStart + node->outputsNum);
    nodeExt->outputInstStart = op::internal::PtrCastTo<NnopbaseAnchorInstanceInfo>(
        op::internal::PtrCastTo<NnopbaseUChar>(nodeExt->attrStart) + nodeExt->node->attrSize);

    const NnopbaseParamDesc *const desc = &executor->args->outputs.paramDescs;
    for (size_t i = 0U; i < count; i++) {
        const uint32_t startIndex = desc->instances[executor->args->binInfo->initValues[i].irIndex].startIndex;
        NNOPBASE_ASSERT_OK_RETVAL(
            NnopbaseComputeNodeSetInstInfo(nodeExt->node->irInputsNum, i, nodeExt->instStart,
                startIndex, desc->instances[i].num));
    }

    // 4 for tiling compile info parsed struct,platform,tilingfunc,deterministic
    contextExt->context->input_size = node->inputsNum + 4;
    contextExt->context->output_size = gert::TilingContext::kOutputNum;
    contextExt->context->output_start = contextExt->context->values + contextExt->context->input_size;
    contextExt->context->compute_node_info = node;
    contextExt->context->kernel_extend_info =
        executor->args->binInfo->memsetInfo->tilingParseContext->kernel_extend_info;

    contextExt->hasPrepared = false;
    return OK;
}

void NnopbaseBuildMemsetInputs(NnopbaseExecutor *executor)
{
    NnopbaseAsyncAnyValue **values = executor->args->binInfo->memsetInfo->contextExt.context->values;
    // input[0] is not used for workspace checking
    values[0]->data.pointer = nullptr;
    values[0]->deleter = nullptr;
    NnopbaseCompileTimeTensorDesc *inputTd = executor->args->binInfo->memsetInfo->contextExt.nodeExt.inputTdStart;
    for (uint32_t i = 0; i < executor->args->binInfo->initValues.size(); i++) {
        size_t irIndex = executor->args->binInfo->initValues[i].irIndex;
        const uint32_t startIndex =
            executor->args->outputs.paramDescs.instances[irIndex].startIndex;
        if (!executor->args->outputs.extTensors[startIndex].isNull) {
            NnopbaseCompileTimeTensorDesc *td = inputTd + i;
            GertTensor *tensor = &executor->args->outputs.extTensors[startIndex].rt2Tensor;
            td->SetDataType(tensor->GetDataType());
            gert::StorageFormat storageFormat = tensor->GetFormat();
            td->SetStorageFormat(storageFormat.GetStorageFormat());
            td->SetOriginFormat(storageFormat.GetOriginFormat());
            td->SetExpandDimsType(storageFormat.GetExpandDimsType());
            *op::internal::PtrCastTo<uint64_t>(values[i + 1]->data.inplace) =
                executor->args->binInfo->initValues[i].tensorDataSize;
        }
        values[i + 1]->deleter = nullptr;
    }
    return;
}

void NnopbaseBuildMemsetAttrs(NnopbaseExecutor *executor)
{
    NnopbaseComputeNodeInfoExt *nodeExt = &executor->args->binInfo->memsetInfo->contextExt.nodeExt;
    NnopbaseRuntimeAttrsDef *attrDef = nodeExt->attrStart;
    constexpr size_t attrNum = 4; // memset has 4 attr
    attrDef->attr_num = attrNum;

    size_t attrSize = sizeof(gert::TypedContinuousVector<size_t>) + sizeof(size_t) * nodeExt->node->irInputsNum;
    std::array<size_t, attrNum> attrSizeArr {attrSize, attrSize, attrSize, attrSize};
    std::array<void*, attrNum> attrAddr {nullptr, nullptr, nullptr, nullptr};

    void *attrPtr = op::internal::PtrCastTo<uint8_t>(attrDef + 1) + sizeof(size_t) * attrNum;
    for (size_t i = 0; i < attrNum; i++) {
        attrDef->offset[i] = op::internal::PtrOffset(attrDef, attrPtr);
        attrAddr[i] = attrPtr;
        attrPtr = op::internal::PtrShift(attrPtr, attrSizeArr[i]);
    }

    auto attrTensorSize = static_cast<gert::ContinuousVector*>(attrAddr[0]);
    auto attrDtype = static_cast<gert::ContinuousVector*>(attrAddr[1]);
    auto attrValueInt = static_cast<gert::ContinuousVector*>(attrAddr[2]);
    auto attrValueFloat = static_cast<gert::ContinuousVector*>(attrAddr[3]);

    for (auto elem : {attrTensorSize, attrDtype, attrValueInt, attrValueFloat}) {
        elem->Init(nodeExt->node->irInputsNum);
        elem->SetSize(nodeExt->node->irInputsNum);
    }

    for (size_t i = 0; i < nodeExt->node->irInputsNum; i++) {
        const auto &elem = executor->args->binInfo->initValues[i];
        size_t *pTensorSize = static_cast<size_t*>(attrTensorSize->MutableData()) + i;
        size_t *pDtype = static_cast<size_t*>(attrDtype->MutableData()) + i;
        size_t *pValueInt = static_cast<size_t*>(attrValueInt->MutableData()) + i;
        size_t *pValueFloat = static_cast<size_t*>(attrValueFloat->MutableData()) + i;

        *pTensorSize = elem.tensorDataSize;
        *op::internal::PtrCastTo<op::DataType>(pDtype) = elem.dtype;
        *op::internal::PtrCastTo<int64_t>(pValueInt) = elem.intValue;
        *op::internal::PtrCastTo<float32_t>(pValueFloat) = elem.floatValue;
    }
}

static void NnopbaseSetMemsetTilingKeyAndBlockDim(NnopbaseExecutor *const executor)
{
    NnopbaseAsyncAnyValue **values = executor->args->binInfo->memsetInfo->contextExt.context->values;
    size_t index = executor->args->binInfo->memsetInfo->contextExt.context->input_size;
    /* kOutputTilingKey. */
    values[index]->deleter = nullptr;
    index++;
    /* kOutputBlockDim. */
    executor->args->binInfo->memsetInfo->blockDim = op::internal::PtrCastTo<uint32_t>(values[index]->data.inplace);
    values[index]->deleter = nullptr;
    index++;
    /* kOutputAtomicCleanFlag, does not support atomic clean. */
    values[index]->deleter = nullptr;
}

aclnnStatus NnopnbaseBuildMemsetTilingContext(NnopbaseExecutor *executor)
{
    NnopbaseKernelRunContextExt *contextExt = &executor->args->binInfo->memsetInfo->contextExt;
    if (!contextExt->hasPrepared) {
        NnopbaseAsyncAnyValue **values = contextExt->context->values;
        uint32_t index = executor->args->binInfo->initValues.size() + 1U; // 1 is for workspace
        /* kInputsCompileInfo. */
        values[index]->data.pointer =
            executor->args->binInfo->memsetInfo->tilingParseContext->values[kCompileInfoStruct]->data.pointer;
        values[index]->deleter = nullptr;
        index++;
        /* kInputsPlatformInfo. */
        values[index]->data.pointer = nullptr;  // tiling里从compileinfo中获取platform
        values[index]->deleter = nullptr;
        index++;
        /* kInputsTilingFunc. */
        values[index]->data.pointer = nullptr;
        values[index]->deleter = nullptr;
        index++;
        /* kInputsDetermintstic. */
        values[index]->data.pointer = reinterpret_cast<void *>(NnopbaseGetGlobalDeterministic());
        values[index]->deleter = nullptr;

        NnopbaseSetMemsetTilingKeyAndBlockDim(executor);
        contextExt->hasPrepared = true;
    }
    *(executor->args->binInfo->memsetInfo->blockDim) = 0U;
    NnopbaseAsyncAnyValue **values = executor->args->binInfo->memsetInfo->contextExt.context->values;
    // 3 for tilingkey, blockdim, atomic
    size_t index = executor->args->binInfo->memsetInfo->contextExt.context->input_size + 3U;
    /* kOutputTilingData. */
    values[index]->data.pointer = op::internal::PtrCastTo<void>(executor->args->binInfo->memsetInfo->tilingData);
    values[index]->deleter = nullptr;
    index++;
    /* kOutputWorkspace. */
    values[index]->data.pointer = nullptr;
    values[index]->deleter = nullptr;

    NnopbaseBuildMemsetInputs(executor);
    NnopbaseBuildMemsetAttrs(executor);

    return OK;
}

aclnnStatus NnopbaseSetMemsetKernelExtendInfo(std::unique_ptr<NnopbaseMemsetInfo> &memsetInfo)
{
    auto kernelExtendInfo = std::make_unique<NnopbaseKernelExtendInfo>();
    NNOPBASE_ASSERT_NOTNULL_RETVAL(kernelExtendInfo);
    kernelExtendInfo->kernelName = memsetInfo->binInfo->kernelName.c_str();
    kernelExtendInfo->kernelType = memsetInfo->binInfo->opType;
    memsetInfo->tilingParseContext->kernel_extend_info = kernelExtendInfo.release();
    return OK;
}

static aclnnStatus NnopbaseSetMemsetTilingContext(std::unique_ptr<NnopbaseMemsetInfo> &memsetInfo, void *compileInfoCreater)
{
    std::size_t tilingParseSize = sizeof(NnopbaseAsyncAnyValue) * static_cast<int32_t>(kParseOutputNum);
    auto tilingParseContextValue = std::make_unique<NnopbaseAsyncAnyValue[]>(kParseOutputNum);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(tilingParseContextValue);
    auto ret = memset_s(tilingParseContextValue.get(), tilingParseSize, 0, tilingParseSize);
    CHECK_COND(ret == EOK,
        ACLNN_ERR_INNER,
        "Memset tilingParseContextValue failed, ret %d, tilingParseContextValue addr is %p, tilingParseSize is %zu.",
        ret,
        tilingParseContextValue.get(),
        tilingParseSize);

    tilingParseContextValue[kCompileInfo].data.pointer = const_cast<NnopbaseChar *>(memsetInfo->compileInfo);
    tilingParseContextValue[kPlatformInfo].data.pointer = static_cast<void*>(g_nnopbasePlatformMgr.memsetInfos.get());
    tilingParseContextValue[kOpType].data.pointer = const_cast<NnopbaseChar *>(memsetInfo->binInfo->opType);
    tilingParseContextValue[kCompileInfo].deleter = nullptr;
    tilingParseContextValue[kPlatformInfo].deleter = nullptr;
    tilingParseContextValue[kOpType].deleter = nullptr;

    std::size_t tilingParseCtxSize =
        sizeof(NnopbaseAsyncAnyValue *) * static_cast<int32_t>(kParseOutputNum) + sizeof(NnopbaseKernelRunContext);
    auto tilingParseContext = std::make_unique<uint8_t[]>(tilingParseCtxSize);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(tilingParseContext);
    ret = memset_s(tilingParseContext.get(), tilingParseCtxSize, 0, tilingParseCtxSize);
    CHECK_COND(ret == EOK,
        ACLNN_ERR_INNER,
        "Memset tilingParseContext failed, ret %d, tilingParseContext addr is %p, tilingParseCtxSize is %zu.",
        ret,
        tilingParseContext.get(),
        tilingParseCtxSize);
    memsetInfo->tilingParseContext = op::internal::PtrCastTo<NnopbaseKernelRunContext>(tilingParseContext.release());
    memsetInfo->tilingParseContext->compute_node_info = memsetInfo->contextExt.nodeExt.node;

    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSetMemsetKernelExtendInfo(memsetInfo));
    // 3 is kCompileInfo, kPlatformInfo and kOpType
    memsetInfo->tilingParseContext->input_size = 3;
    memsetInfo->tilingParseContext->output_size = 1;
    memsetInfo->tilingParseContext->output_start =
        memsetInfo->tilingParseContext->values + memsetInfo->tilingParseContext->input_size;
    memsetInfo->tilingParseContext->values[kCompileInfo] = &tilingParseContextValue[kCompileInfo];
    memsetInfo->tilingParseContext->values[kPlatformInfo] = &tilingParseContextValue[kPlatformInfo];
    memsetInfo->tilingParseContext->values[kOpType] = &tilingParseContextValue[kOpType];
    memsetInfo->tilingParseContext->values[kCompileInfoStruct] = &tilingParseContextValue[kCompileInfoStruct];
    tilingParseContextValue[kCompileInfoStruct].data.pointer = compileInfoCreater;
    tilingParseContextValue[kCompileInfoStruct].deleter = nullptr;
    memsetInfo->tilingParseContextValue = tilingParseContextValue.release();
    return OK;
}

aclnnStatus NnopbaseGenMemsetV2TilingFunc(NnopbaseExecutor *executor)
{
    executor->args->binInfo->memsetInfo->binInfo->opType = NNOPBASE_MEMSET_V2_OP_NAME.c_str();

    auto &registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    NNOPBASE_ASSERT_NOTNULL_RETVAL(registry);
    auto opImpl = registry->GetOpImpl(executor->args->binInfo->memsetInfo->binInfo->opType);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(opImpl);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(opImpl->tiling);

    executor->args->binInfo->memsetInfo->tiling = op::internal::PtrCastTo<TilingFunProtopyte>(opImpl->tiling);

    return OK;
}

aclnnStatus NnopbaseBuildAndRunMemsetTilingParse(NnopbaseExecutor *executor)
{
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorPlatFormInfosInit());
    std::string coreType = executor->args->binInfo->memsetInfo->binInfo->coreType == kAicore ? "AiCore" : "VectorCore";
    g_nnopbasePlatformMgr.memsetInfos->SetCoreNumByCoreType(coreType);

    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseComputeNodeInfoInit(&executor->args->binInfo->memsetInfo->contextExt.nodeExt));
    executor->args->binInfo->memsetInfo->binInfo->opType = NNOPBASE_MEMSET_OP_NAME.c_str();
    auto &registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    NNOPBASE_ASSERT_NOTNULL_RETVAL(registry);
    auto opImpl = registry->GetOpImpl(executor->args->binInfo->memsetInfo->binInfo->opType);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(opImpl);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(opImpl->tiling_parse);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(opImpl->tiling);
    CHECK_COND(
        NnopbaseSetMemsetTilingContext(executor->args->binInfo->memsetInfo, opImpl->compile_info_creator()) == OK,
        ACLNN_ERR_INNER,
        "set memset op tilingContext failed.");

    executor->args->binInfo->memsetInfo->tiling = reinterpret_cast<TilingFun>(opImpl->tiling);
    auto ret = opImpl->tiling_parse(
        op::internal::PtrCastTo<gert::KernelContext>(executor->args->binInfo->memsetInfo->tilingParseContext));
    CHECK_COND((ret == ge::GRAPH_SUCCESS), ACLNN_ERR_INNER_TILING_ERROR, "Memset tiling parse failed, ret is %u.", ret);
    return OK;
}

#ifdef __cplusplus
}
#endif
