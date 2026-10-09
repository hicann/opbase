/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "indv_args.h"
#include "indv_executor.h"
#include "bridge_dfx.h"
#include "utils/indv_soc.h"
#include <algorithm>
#include <cstring>

namespace {
constexpr uint8_t NNOPBASE_OOM_STORAGE_SHAPE_MAGIC = 0x4FU;
constexpr uint8_t NNOPBASE_OOM_STORAGE_SHAPE_TENSOR_KIND = 1U;
constexpr uint8_t NNOPBASE_OOM_STORAGE_SHAPE_TENSOR_LIST_KIND = 2U;
constexpr size_t NNOPBASE_OOM_STORAGE_SHAPE_HEADER_SIZE = 2U;
constexpr size_t NNOPBASE_OOM_STORAGE_SHAPE_DESC_SIZE = 9U;
constexpr size_t NNOPBASE_OOM_STORAGE_SHAPE_TENSOR_SIZE = 21U;
constexpr size_t NNOPBASE_OOM_STORAGE_SHAPE_TENSOR_LIST_HEADER_SIZE = 14U;

struct NnopbaseOomStorageShapeRecord {
    size_t startIndex = 0U;
    uint16_t count = 0U;
    bool isTensorList = false;
};

static aclnnStatus NnopbaseCollectOomStorageShapeRecords(
    const NnopbaseExecutor* const executor, std::vector<NnopbaseOomStorageShapeRecord>& records)
{
    if ((executor == nullptr) || (executor->args == nullptr) || (executor->args->binInfo == nullptr) ||
        (!executor->args->binInfo->oomConfig.storageShapeEnabled)) {
        return OK;
    }

    const auto& instances = executor->ownArgs.inputs.paramDescs.instances;
    const auto& inputInstances = executor->args->inputs.paramDescs.instances;
    const uint32_t instanceCount = executor->ownArgs.inputs.paramDescs.count;
    CHECK_COND(instances.size() >= instanceCount, ACLNN_ERR_PARAM_INVALID,
               "Input instance count[%zu] is less than param count[%u].", instances.size(), instanceCount);
    CHECK_COND(inputInstances.size() >= instanceCount, ACLNN_ERR_PARAM_INVALID,
               "Cached input instance count[%zu] is less than param count[%u].", inputInstances.size(),
               instanceCount);
    for (uint32_t i = 0U; i < instanceCount; ++i) {
        const auto& instance = instances[i];
        const size_t startIndex = inputInstances[i].startIndex;
        if (instance.tensor != nullptr) {
            records.push_back({startIndex, 1U, false});
            continue;
        }
        if (instance.tensorList == nullptr) {
            records.push_back({startIndex, 1U, false});
            continue;
        }
        records.push_back({startIndex, static_cast<uint16_t>(instance.tensorList->Size()), true});
    }
    return OK;
}

static aclnnStatus NnopbaseAppendOomStorageShapeRecord(const NnopbaseExecutorArgs* const args,
                                                       const NnopbaseOomStorageShapeRecord& record,
                                                       NnopbaseExecutorArgsAddr* const argsAddr)
{
    NnopbaseUChar* addr = argsAddr->ptr;
    const uint8_t tensorVersion = args->binInfo->oomConfig.tensorVersion & 0x0FU;

    const uint64_t bodySize = record.isTensorList ?
                                  static_cast<uint64_t>(NNOPBASE_OOM_STORAGE_SHAPE_TENSOR_LIST_HEADER_SIZE -
                                                        NNOPBASE_OOM_STORAGE_SHAPE_HEADER_SIZE +
                                                        static_cast<size_t>(record.count) *
                                                            NNOPBASE_OOM_STORAGE_SHAPE_DESC_SIZE) :
                                  static_cast<uint64_t>(NNOPBASE_OOM_STORAGE_SHAPE_TENSOR_SIZE -
                                                        NNOPBASE_OOM_STORAGE_SHAPE_HEADER_SIZE);
    addr = nnopbase::NnopbaseAppendByte<uint64_t>(addr, bodySize);
    const uint8_t kind = record.isTensorList ? NNOPBASE_OOM_STORAGE_SHAPE_TENSOR_LIST_KIND :
                                               NNOPBASE_OOM_STORAGE_SHAPE_TENSOR_KIND;
    *addr++ = kind;
    *addr++ = 0U; // reserved字段
    if (record.isTensorList) {
        addr = nnopbase::NnopbaseAppendByte<uint16_t>(addr, record.count);
    }

    const auto& extTensors = args->inputs.extTensors;
    const uint32_t tensorNum = record.isTensorList ? record.count : 1U;
    for (size_t i = 0U; i < tensorNum; ++i) {
        const size_t tensorIndex = record.startIndex + i;
        CHECK_COND(tensorIndex < extTensors.size(), ACLNN_ERR_PARAM_INVALID,
                   "Oom storage shape tensor index[%zu] is out of range[%zu].", tensorIndex, extTensors.size());
        *addr++ = tensorVersion;
        const uint64_t storageShapeSize = extTensors[tensorIndex].isNull ?
            0U : static_cast<uint64_t>(extTensors[tensorIndex].storageShape.GetShapeSize());
        OP_LOGI("Oom storage shape tensor index[%zu] storageShapeSize is %llu.", tensorIndex, storageShapeSize);
        addr = nnopbase::NnopbaseAppendByte<uint64_t>(addr, storageShapeSize);
    }
    argsAddr->ptr = addr;
    return OK;
}
} // namespace

static inline NnopbaseUChar* NnopbasePrepareDimInfo(NnopbaseUChar* addr, const GertShape& shape)
{
    const int64_t shapeSize = shape.GetShapeSize();
    if (shapeSize > 0) {    // 非空tensor场景
        const size_t dimNum = shape.GetDimNum();
        if (dimNum == 0U) { // Scalar场景相当于shape={1}的tensor
            addr = nnopbase::NnopbaseAppendByte<uint64_t>(addr, 1ULL);
            addr = nnopbase::NnopbaseAppendByte<uint64_t>(addr, 1ULL);
        } else {
            addr = nnopbase::NnopbaseAppendByte<uint64_t>(addr, dimNum);
            for (size_t k = 0U; k < dimNum; k++) {
                addr = nnopbase::NnopbaseAppendByte<int64_t>(addr, shape.GetDim(k));
            }
        }
    }
    return addr;
}

static void NnopbaseExecutorPrepareIOSize(const NnopbaseExecutor* const executor, NnopbaseUChar*& addr,
                                          NnopbaseUChar*& shapeInfoPtr, const bool isInput)
{
    const NnopbaseTensors& tensors = isInput ? executor->args->inputs : executor->args->outputs;
    const auto& extTensors = tensors.extTensors;
    const auto& paramInstance = tensors.paramDescs.instances;
    size_t j = 0U;
    // set input or output tensor bytes
    for (uint32_t i = 0U; i < tensors.paramDescs.count; i++) {
        if (extTensors[j].isNull) {
            addr = nnopbase::NnopbaseAppendByte<uint64_t>(addr, 0U);
            j += 1U;
            continue;
        }
        if (!paramInstance[i].isDynamic) {
            const GertShape& shape = extTensors[j].rt2Tensor.GetStorageShape();
            // 三类算子要反刷shape的tensor只用填oom size，不用记shape
            if (isInput ||
                executor->args->outputs.outPutShapeMap.find(i) == executor->args->outputs.outPutShapeMap.end()) {
                shapeInfoPtr = NnopbasePrepareDimInfo(shapeInfoPtr, shape);
            }
            addr = nnopbase::NnopbaseAppendByte<uint64_t>(addr, extTensors[j].rt2Tensor.GetSize());
            j += 1U;
        } else {
            const size_t startIndex = paramInstance[i].startIndex;
            const size_t size = paramInstance[i].num;
            /* for each input or output dimNum, count, addr */
            static const size_t K_INPUT_INFO_LEN = 2U * sizeof(uint32_t) + sizeof(void*);
            /* ptr offset, dimNum, count, addr */
            size_t dynamicSize = sizeof(uint64_t) + K_INPUT_INFO_LEN * size;
            for (size_t k = 0U; k < size; k++) {
                const GertShape& shape = extTensors[startIndex + k].rt2Tensor.GetStorageShape();
                dynamicSize += shape.GetDimNum() * sizeof(uint64_t);
            }
            addr = nnopbase::NnopbaseAppendByte<uint64_t>(addr, dynamicSize);
            j += size;
        }
    }
    return;
}

static void NnopbaseExecutorSetDfxInfo(const NnopbaseExecutor* const executor)
{
    size_t startIndex = executor->mc2.commHandles.size();
    NnopbaseUChar* addr = op::internal::PtrCastTo<NnopbaseUChar>(&(executor->args->dfxInfo[startIndex]));
    const auto workspacesSizes = NnopbaseGetWorkspacesSizesFromArgs(executor->args);
    const uint32_t workspaceNum = workspacesSizes->GetSize() == 0UL ? 1U :
                                                                      static_cast<uint32_t>(workspacesSizes->GetSize());
    uint32_t oomNum = executor->args->inputs.paramDescs.count + executor->args->outputs.paramDescs.count +
                      workspaceNum + executor->mc2.commHandles.size();
    if (executor->args->outputs.outPutShapeSize != 0U) {
        oomNum += 1U;
    }
    NnopbaseUChar* shapeInfoPtr = op::internal::PtrCastTo<NnopbaseUChar>(executor->args->dfxInfo.data()) +
                                  oomNum * sizeof(void*);
    // input tensor data size
    NnopbaseExecutorPrepareIOSize(executor, addr, shapeInfoPtr, true);
    // output tensor data size
    NnopbaseExecutorPrepareIOSize(executor, addr, shapeInfoPtr, false);

    if (executor->args->outputs.outPutShapeSize != 0U) {
        addr = nnopbase::NnopbaseAppendByte<uint64_t>(addr, executor->args->outputs.outPutShapeSize);
    }

    const uint64_t workSpaceSize = workspacesSizes->GetSize() == 0UL ? 0UL : workspacesSizes->GetData()[0];
    if ((executor->args->binInfo->dfxInfo.isPrintEnable) || (executor->args->binInfo->dfxInfo.isAssertEnable) ||
        (executor->args->binInfo->dfxInfo.isTimeStampEnable)) {
        addr = nnopbase::NnopbaseAppendByte<uint64_t>(addr, workSpaceSize + executor->args->binInfo->debugBufSize);
    } else {
        addr = nnopbase::NnopbaseAppendByte<uint64_t>(addr, workSpaceSize);
    }
    for (size_t i = 1U; i < workspacesSizes->GetSize(); i++) {
        addr = nnopbase::NnopbaseAppendByte<uint64_t>(addr, workspacesSizes->GetData()[i]);
    }
}

static inline void NnopbaseExecutorGetIoShapeInfoSize(const NnopbaseTensor& tensor, uint32_t& space)
{
    const int64_t shapeSize = tensor.rt2Tensor.GetShapeSize();
    if (shapeSize != 0) { // 非空tensor场景，dimNum为0是scalar
        const size_t dimNum = tensor.rt2Tensor.GetStorageShape().GetDimNum();
        space = dimNum == 0U ? (space + 2U) : (space + (dimNum + 1U));
    }
}

static void NnopbaseExecutorGetInputShapeInfoSize(const NnopbaseTensors& tensors, uint32_t& space)
{
    const auto& extTensors = tensors.extTensors;
    const auto& paramInstance = tensors.paramDescs.instances;
    for (uint32_t i = 0U; i < tensors.paramDescs.count; i++) {
        const size_t startIndex = paramInstance[i].startIndex;
        if ((!paramInstance[i].isDynamic) && (!extTensors[startIndex].isNull)) {
            NnopbaseExecutorGetIoShapeInfoSize(extTensors[startIndex], space);
        }
    }
}

static void NnopbaseExecutorGetOutputShapeInfoSize(const NnopbaseTensors& tensors,
                                                   std::map<uint32_t, aclTensor*> outPutShapeMap, uint32_t& space)
{
    const auto& extTensors = tensors.extTensors;
    const auto& paramInstance = tensors.paramDescs.instances;
    for (uint32_t i = 0U; i < tensors.paramDescs.count; i++) {
        const size_t startIndex = paramInstance[i].startIndex;
        if ((!paramInstance[i].isDynamic) && (!extTensors[startIndex].isNull)) {
            if (outPutShapeMap.find(i) == outPutShapeMap.end()) {
                NnopbaseExecutorGetIoShapeInfoSize(extTensors[startIndex], space);
            }
        }
    }
}

void NnopbaseExecutorPrepareDfxInfo(NnopbaseExecutor* const executor)
{
    // workspace num为0时，args中需要占位
    const auto workspacesSizes = NnopbaseGetWorkspacesSizesFromArgs(executor->args);
    const uint32_t workspaceNum = workspacesSizes->GetSize() == 0UL ? 1U :
                                                                      static_cast<uint32_t>(workspacesSizes->GetSize());
    uint32_t space = 0U;
    NnopbaseExecutorGetInputShapeInfoSize(executor->args->inputs, space);
    NnopbaseExecutorGetOutputShapeInfoSize(executor->args->outputs, executor->args->outputs.outPutShapeMap, space);
    space += (executor->args->inputs.paramDescs.count + executor->args->outputs.paramDescs.count + workspaceNum +
              executor->mc2.commHandles.size());
    if (executor->args->outputs.outPutShapeSize != 0U) {
        space += 1U;
    }
    executor->args->dfxInfo.resize(space);
    OP_LOGI("DfxInfo size is %u.", space);
    NnopbaseExecutorSetDfxInfo(executor);
}

// 扩展区排布：2BHeader(magic + version + reserved) + records
static aclnnStatus NnopbaseAppendOomStorageShapeExt(NnopbaseExecutor* const executor,
                                                    NnopbaseExecutorArgsAddr* const argsAddr)
{
    std::vector<NnopbaseOomStorageShapeRecord> records;
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCollectOomStorageShapeRecords(executor, records));
    if (!records.empty()) {
        *argsAddr->ptr++ = NNOPBASE_OOM_STORAGE_SHAPE_MAGIC;
        *argsAddr->ptr++ = static_cast<NnopbaseUChar>(executor->args->binInfo->oomConfig.version & 0x0FU);
    }
    for (const auto& record : records) {
        OP_LOGI("Oom storage shape record startIndex is %zu, isTensorList is %d, count is %u.",
                record.startIndex, record.isTensorList, record.count);
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAppendOomStorageShapeRecord(executor->args, record, argsAddr));
    }
    // 扩展区尾部按8B对齐补padding，清零防止输出侧注册误消费。
    if (!records.empty()) {
        const uintptr_t endAddr = reinterpret_cast<uintptr_t>(argsAddr->ptr);
        const size_t padding = (~endAddr + 1U) & (NNOPBASE_EIGHT_BYTES - 1U); // (8 - end%8) % 8
        if (padding != 0U) {
            const errno_t ret = memset_s(argsAddr->ptr, padding, 0, padding);
            if (ret != EOK) {
                OP_LOGW("Failed to memset_s OOM storage shape padding, ret is %d, padding size is %zu.", ret,
                        padding);
            }
        }
        argsAddr->ptr += padding;
    }
    return OK;
}

aclnnStatus NnopbaseExecutorArgsGetDfxInfo(NnopbaseExecutor* const executor, NnopbaseExecutorArgsAddr* const argsAddr,
                                           const uint32_t workspaceNum, const aclrtStream stream)
{
    if (executor->args->dfxInfo.empty()) {
        NnopbaseExecutorPrepareDfxInfo(executor);
    }
    for (size_t i = 0U; i < executor->mc2.contextAddrs.size(); i++) {
        executor->args->dfxInfo[i] = 32U;
    }
    if (executor->args->binInfo->oomConfig.flag) {
        uint32_t oomSize = (executor->args->inputs.paramDescs.count + executor->args->outputs.paramDescs.count +
                            workspaceNum + executor->mc2.contextAddrs.size()) *
                           sizeof(void*);
        if (executor->args->outputs.outPutShapeSize != 0U) {
            oomSize += sizeof(void*);
        }
        CHECK_COND((memcpy_s(op::internal::PtrCastTo<void>(argsAddr->ptr), oomSize, executor->args->dfxInfo.data(),
                             oomSize) == EOK),
                   ACLNN_ERR_PARAM_INVALID, "Failed to execute memcpy_s oom info, src is %p, dst is %p, size is %u.",
                   argsAddr->ptr, executor->args->dfxInfo.data(), oomSize);
        argsAddr->ptr += oomSize;
        if (executor->args->binInfo->oomConfig.storageShapeEnabled) {
            NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAppendOomStorageShapeExt(executor, argsAddr));
        }
    }
    if (op::internal::IsArgExceptionDumpEnable()) {
        uint64_t atomicIndex = 0U;
        void* exceptionDumpAddr = nullptr;
        if (NnopbaseIsAclGraphCaptureScene(stream)) {
            exceptionDumpAddr = Adx::AdumpGetDFXInfoAddrForStatic(executor->args->dfxInfo.size(), atomicIndex);
        } else {
            exceptionDumpAddr = Adx::AdumpGetDFXInfoAddrForDynamic(executor->args->dfxInfo.size(), atomicIndex);
        }
        NNOPBASE_ASSERT_NOTNULL_RETVAL(exceptionDumpAddr);
        OP_LOGI("Get atomicIndex is %lu.", atomicIndex);
        argsAddr->ptr = nnopbase::NnopbaseAppendByte<uint64_t>(argsAddr->ptr, atomicIndex);
        CHECK_COND(memcpy_s(exceptionDumpAddr, executor->args->dfxInfo.size() * sizeof(uint64_t),
                            executor->args->dfxInfo.data(), executor->args->dfxInfo.size() * sizeof(uint64_t)) == EOK,
                   ACLNN_ERR_PARAM_INVALID,
                   "Failed to execute memcpy_s dfx info, exceptionDumpAddr is %p, dfx Info addr is %p, size is %zu.",
                   exceptionDumpAddr, executor->args->dfxInfo.data(),
                   executor->args->dfxInfo.size() * sizeof(uint64_t));
    }
    return OK;
}

size_t NnopbaseGetOomInfoExtMaxSize(const NnopbaseExecutor* const executor)
{
    if ((executor == nullptr) || (executor->args == nullptr) || (executor->args->binInfo == nullptr) ||
        (!executor->args->binInfo->oomConfig.flag) || (!executor->args->binInfo->oomConfig.storageShapeEnabled)) {
        return 0U;
    }

    size_t size = 0U;
    const auto& instances = executor->ownArgs.inputs.paramDescs.instances;
    const uint32_t instanceCount = executor->ownArgs.inputs.paramDescs.count;
    const uint32_t validInstanceCount =
        static_cast<uint32_t>(std::min(instances.size(), static_cast<size_t>(instanceCount)));
    if (validInstanceCount != instanceCount) {
        OP_LOGW("Input instance count[%zu] is less than param count[%u], skip invalid OOM size entries.",
                instances.size(), instanceCount);
    }
    for (uint32_t i = 0U; i < validInstanceCount; ++i) {
        const auto& instance = instances[i];
        if (instance.tensor != nullptr) {
            size += NNOPBASE_OOM_STORAGE_SHAPE_TENSOR_SIZE;
        } else if (instance.tensorList != nullptr) {
            size += NNOPBASE_OOM_STORAGE_SHAPE_TENSOR_LIST_HEADER_SIZE +
                    static_cast<size_t>(instance.tensorList->Size()) * NNOPBASE_OOM_STORAGE_SHAPE_DESC_SIZE;
        }
    }
    if (size != 0U) {
        size += NNOPBASE_EIGHT_BYTES - 1U;
        size = (size / NNOPBASE_EIGHT_BYTES) * NNOPBASE_EIGHT_BYTES;
    }
    return size;
}

static void NnopbaseExecutorGetDynamicTensorSize(NnopbaseTensors& tensors)
{
    const auto& extTensors = tensors.extTensors;
    auto& paramInstance = tensors.paramDescs.instances;
    // set input or output tensor bytes
    tensors.dynamicSize = 0U;
    for (uint32_t i = 0U; i < tensors.paramDescs.count; i++) {
        if (paramInstance[i].isDynamic) {
            const size_t startIndex = paramInstance[i].startIndex;
            const size_t size = paramInstance[i].num;
            static const size_t K_INPUT_INFO_LEN = 2U * sizeof(uint32_t) + sizeof(void*);
            size_t dynamicSize = sizeof(uint64_t) + K_INPUT_INFO_LEN * size;
            for (size_t k = 0U; k < size; k++) {
                const GertShape& shape = extTensors[startIndex + k].rt2Tensor.GetStorageShape();
                dynamicSize += shape.GetDimNum() * sizeof(uint64_t);
            }
            tensors.dynamicSize += dynamicSize; // 输入/输出总的内存大小
            OP_LOGI("Tensors[%u] dynamicSize is %zu bytes", i, dynamicSize);
        }
    }
}

size_t NnopbaseCalcArgsSize(NnopbaseExecutor* const executor, const size_t tilingDataSize)
{
    // mc2算子多了个NnopbaseHcclCommParamDesc、hcomHandle.size()个context addr
    const size_t mc2Size = executor->mc2.enabled ?
                               executor->mc2.commHandles.size() * sizeof(void*) + sizeof(NnopbaseHcclCommParamDesc) :
                               0U;
    const size_t irNum = static_cast<size_t>(executor->args->inputs.paramDescs.count +
                                             executor->args->outputs.paramDescs.count);
    // tiling前workspace先按最大申请，input, output, workspaces, 3 for tiling, overflow, ctrlAddr
    size_t argsLen = (irNum + NNOPBASE_NORM_MAX_WORKSPACE_NUMS + 3U) * sizeof(void*) + mc2Size;
    executor->args->tilingDataOffset = argsLen;
    if (executor->args->outputs.outPutShapeSize != 0U) {
        executor->args->tilingDataOffset += sizeof(void*);
        argsLen += sizeof(void*) * 2;                                          // 2 is outputshape and oom
    }
    argsLen += (irNum + NNOPBASE_NORM_MAX_WORKSPACE_NUMS + 1) * sizeof(void*); // oom, 1 is for automicIndex
    argsLen += NnopbaseGetOomInfoExtMaxSize(executor);
    if (executor->hasTiling) {
        NnopbaseExecutorGetDynamicTensorSize(executor->args->inputs);
        NnopbaseExecutorGetDynamicTensorSize(executor->args->outputs);
        argsLen += (tilingDataSize + executor->args->inputs.dynamicSize + executor->args->outputs.dynamicSize);
    }
    if ((executor->args->inputs.hostInputNum > 0) || executor->args->inputs.hasDynamic ||
        executor->args->outputs.hasDynamic || (executor->mc2.enabled && executor->hasTiling)) {
        executor->argsExt.hostInputInfoNum = executor->args->inputs.hostInputNum +
                                             static_cast<uint16_t>(executor->args->inputs.dynamicNum +
                                                                   executor->args->outputs.dynamicNum);
        // MC2算子aicore和aicpu各一份hostInfo
        const size_t hostInfoNum = executor->mc2.enabled ? 2U : 1U;
        const size_t alignHostInputSize = ((executor->args->inputs.hostInputSize + NNOPBASE_SEVENS_BYTES) /
                                           NNOPBASE_EIGHT_BYTES) *
                                          NNOPBASE_EIGHT_BYTES;
        argsLen += (executor->argsExt.hostInputInfoNum * sizeof(rtHostInputInfo_t) * hostInfoNum) + alignHostInputSize;
        if (executor->mc2.enabled && executor->hasTiling) {
            argsLen += sizeof(rtHostInputInfo_t); // aicpuArgs需要存tilingdata hostinfo
        }
    }
    if (executor->mc2.enabled) {
        argsLen += NNOPBASE_AICPU_PARAM_LEN * 2; // 2 is soname/kernelname
        const size_t mc2OpNameLen = strlen(executor->opType) + NNOPBASE_MC2_AICPU_SUFFIX.length();
        if (nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2.serverType)) {
            argsLen += ((mc2OpNameLen + NNOPBASE_SEVENS_BYTES) / NNOPBASE_EIGHT_BYTES) * NNOPBASE_EIGHT_BYTES;
            argsLen += sizeof(NnopbaseHcclCommParamDesc); // 82上parsmdesc组在args最后
        } else {
            argsLen += mc2OpNameLen;
        }
    }
    OP_LOGI("Op[%s] argsLen is %zu", executor->opType, argsLen);
    return argsLen;
}

static void NnopbaseExecutorEncodeDynamicTensors(NnopbaseExecutorArgsAddr* const argsAddr,
                                                 NnopbaseExecutor* const executor, void** const dynamicIOAddr,
                                                 const NnopbaseParamInstance* const paramInstance)
{
    NnopbaseUChar** dynamicIOData = &argsAddr->hostInputData;
    aclrtPlaceHolderInfo** dynamicIOInfo = &argsAddr->hostInputInfo;
    auto& extTensors = paramInstance->isInput ? executor->args->inputs.extTensors : executor->args->outputs.extTensors;

    *dynamicIOAddr = *dynamicIOData;
    const uint32_t startIndex = paramInstance->startIndex;
    const uint32_t size = paramInstance->num;

    // set dynamic addr offset
    uint64_t dynamicOffset;
    (*dynamicIOData) += sizeof(uint64_t);

    // set shape info
    for (uint32_t i = 0U; i < size; i++) {
        const GertShape shape = extTensors[startIndex + i].rt2Tensor.GetStorageShape();
        const uint32_t dimNum = static_cast<uint32_t>(shape.GetDimNum());
        *dynamicIOData = nnopbase::NnopbaseAppendByte<uint32_t>(*dynamicIOData, dimNum);
        *dynamicIOData = nnopbase::NnopbaseAppendByte<uint32_t>(*dynamicIOData, 1U);
        for (size_t j = 0U; j < shape.GetDimNum(); j++) {
            const int64_t dim = shape.GetDim(j);
            *dynamicIOData = nnopbase::NnopbaseAppendByte<int64_t>(*dynamicIOData, dim);
        }
    }

    dynamicOffset = static_cast<uint64_t>((*dynamicIOData) - (NnopbaseUChar*)(*dynamicIOAddr));
    (void)nnopbase::NnopbaseAppendByte<uint64_t>((NnopbaseUChar*)(*dynamicIOAddr), dynamicOffset);

    // set dynamic input or output addr
    const NnopbaseUChar* const args = (NnopbaseUChar*)executor->argsExt.args;
    for (uint32_t i = 0U; i < size; i++) {
        NnopbaseUChar* addr = (NnopbaseUChar*)extTensors[startIndex + i].rt2Tensor.GetAddr();
        extTensors[startIndex + i].argsOffset = static_cast<uint32_t>(
            op::internal::PtrCastTo<NnopbaseUChar>(*dynamicIOData) - args);
        *dynamicIOData = nnopbase::NnopbaseAppendByte<void*>(*dynamicIOData, addr);
    }
    (*dynamicIOInfo)->addrOffset = static_cast<uint32_t>(op::internal::PtrCastTo<NnopbaseUChar>(dynamicIOAddr) - args);
    (*dynamicIOInfo)->dataOffset = static_cast<uint32_t>(op::internal::PtrCastTo<NnopbaseUChar>(*dynamicIOAddr) - args);
    (*dynamicIOInfo)++;
    if ((executor->mc2.enabled) &&
        (!nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2.serverType))) {
        aclrtPlaceHolderInfo** aicpuHostInputInfo = &argsAddr->aicpuHostInputInfo;
        const NnopbaseUChar* const aicpuArgs = (NnopbaseUChar*)executor->mc2.aicpuArgs.args;
        (*aicpuHostInputInfo)->addrOffset = static_cast<uint32_t>(
            op::internal::PtrCastTo<NnopbaseUChar>(dynamicIOAddr) - aicpuArgs);
        (*aicpuHostInputInfo)->dataOffset = static_cast<uint32_t>(
            op::internal::PtrCastTo<NnopbaseUChar>(*dynamicIOAddr) - aicpuArgs);
        (*aicpuHostInputInfo)++;
    }
}

void** NnopbaseExecutorPrepareNullTensors(const NnopbaseExecutor* const executor, void** addr, size_t* tensorIndex)
{
    // 处理可选输入为空的情况
    if (executor->args->binInfo->isStaticShape) { // 静态kernel不占位处理
        *tensorIndex += 1U;
    } else {
        *addr = nullptr;
        addr++;
        *tensorIndex += 1U;
    }
    return addr;
}

static inline void NnopbaseExecutorEncodeHostInput(const NnopbaseExecutor* const executor,
                                                   NnopbaseExecutorArgsAddr* argsAddr, void** inputAddr,
                                                   GertTensor* tensor)
{
    const NnopbaseUChar* const args = (NnopbaseUChar*)executor->argsExt.args;
    NnopbaseUChar** hostInputData = &argsAddr->hostInputData;
    aclrtPlaceHolderInfo** hostInputInfo = &argsAddr->hostInputInfo;
    NnopbaseUChar* addr = (NnopbaseUChar*)tensor->GetAddr();
    size_t size = tensor->GetSize();
    for (size_t i = 0U; i < size; i++) {
        (*hostInputData)[i] = addr[i];
    }
    *inputAddr = *hostInputData;
    (*hostInputInfo)->addrOffset = static_cast<uint32_t>(op::internal::PtrCastTo<NnopbaseUChar>(inputAddr) - args);
    (*hostInputInfo)->dataOffset = static_cast<uint32_t>((*hostInputData) - args);
    if ((executor->mc2.enabled) &&
        (!nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2.serverType))) {
        aclrtPlaceHolderInfo** aicpuHostInputInfo = &argsAddr->aicpuHostInputInfo;
        const NnopbaseUChar* const aicpuArgs = (NnopbaseUChar*)executor->mc2.aicpuArgs.args;
        (*aicpuHostInputInfo)->addrOffset = static_cast<uint32_t>(op::internal::PtrCastTo<NnopbaseUChar>(inputAddr) -
                                                                  aicpuArgs);
        (*aicpuHostInputInfo)->dataOffset = static_cast<uint32_t>((*hostInputData) - aicpuArgs);
        (*aicpuHostInputInfo)++;
    }
    (*hostInputInfo)++;
    size = ((size + NNOPBASE_SEVENS_BYTES) / NNOPBASE_EIGHT_BYTES) * NNOPBASE_EIGHT_BYTES;
    (*hostInputData) += size;
}

void** NnopbaseExecutorPrepareInputsParamsExt(NnopbaseExecutor* const executor, void** addr,
                                              NnopbaseExecutorArgsAddr* const argsAddr)
{
    const NnopbaseUChar* const args = (NnopbaseUChar*)executor->argsExt.args;
    NnopbaseUChar** hostInputData = &argsAddr->hostInputData;
    NnopbaseUChar** ptr = &argsAddr->ptr;
    size_t j = 0U;
    for (uint32_t i = 0U; i < executor->args->inputs.paramDescs.count; i++) {
        if (executor->args->inputs.extTensors[j].isNull) {
            // 处理可选输入为空的情况
            addr = NnopbaseExecutorPrepareNullTensors(executor, addr, &j);
            continue;
        }
        if (!executor->args->inputs.paramDescs.instances[i].isDynamic) { // 没有动态输入
            if (executor->args->inputs.extTensors[j].rt2Tensor.GetPlacement() == gert::kOnDeviceHbm) {
                *addr = executor->args->inputs.extTensors[j].rt2Tensor.GetAddr();
                executor->args->inputs.extTensors[j].argsOffset = static_cast<uint32_t>(
                    op::internal::PtrCastTo<NnopbaseUChar>(addr) - args);
            } else {
                NnopbaseExecutorEncodeHostInput(executor, argsAddr, addr,
                                                &executor->args->inputs.extTensors[j].rt2Tensor);
                *ptr = *hostInputData;
            }
            j += 1U;
        } else {
            argsAddr->hcclDesc->isDyn |= (1ULL << i);
            NnopbaseExecutorEncodeDynamicTensors(argsAddr, executor, addr,
                                                 &executor->args->inputs.paramDescs.instances[i]);
            j += executor->args->inputs.paramDescs.instances[i].num;
            *ptr = *hostInputData;
        }
        addr++;
    }
    return addr;
}

void** NnopbaseExecutorPrepareOutputsParamsExt(NnopbaseExecutor* const executor, void** addr,
                                               NnopbaseExecutorArgsAddr* const argsAddr)
{
    const NnopbaseUChar* const args = (NnopbaseUChar*)executor->argsExt.args;
    NnopbaseUChar** hostInputData = &argsAddr->hostInputData;
    NnopbaseUChar** ptr = &argsAddr->ptr;
    size_t j = 0U;
    for (uint32_t i = 0U; i < executor->args->outputs.paramDescs.count; i++) {
        if (executor->args->outputs.extTensors[j].isNull) {
            // 处理可选输出为空的情况
            addr = NnopbaseExecutorPrepareNullTensors(executor, addr, &j);
            continue;
        }
        if (!executor->args->outputs.paramDescs.instances[i].isDynamic) { // 没有动态输出
            *addr = executor->args->outputs.extTensors[j].rt2Tensor.GetAddr();
            executor->args->outputs.extTensors[j].argsOffset = static_cast<uint32_t>(
                op::internal::PtrCastTo<NnopbaseUChar>(addr) - args);
            j += 1U;
        } else {
            argsAddr->hcclDesc->isDyn |= (1ULL << (i + executor->args->inputs.paramDescs.count));
            NnopbaseExecutorEncodeDynamicTensors(argsAddr, executor, addr,
                                                 &executor->args->outputs.paramDescs.instances[i]);
            j += executor->args->outputs.paramDescs.instances[i].num;
            *ptr = *hostInputData;
        }
        addr++;
    }
    return addr;
}

std::vector<aclrtPlaceHolderInfo> NnopbaseGetRTSPlaceHolder(NnopbaseRTArgsExt* const argsExt)
{
    std::vector<aclrtPlaceHolderInfo> hostInputInfoPtr;
    if (argsExt->hasTiling != 0) {
        hostInputInfoPtr.push_back({argsExt->tilingAddrOffset, argsExt->tilingDataOffset});
    }
    for (int i = 0; i < argsExt->hostInputInfoNum; i++) {
        aclrtPlaceHolderInfo* ptr = argsExt->hostInputInfoPtr + i;
        hostInputInfoPtr.push_back({ptr->addrOffset, ptr->dataOffset});
    }

    return hostInputInfoPtr;
}

void NnopbaseGetIrIndex(const NnopbaseParamDesc& paramDesc, const size_t index, size_t& irIndex, size_t& relativeIndex)
{
    // num为0表示无入参，几乎不存在调此接口场景，直接返回不作处理
    if (paramDesc.count == 0) {
        return;
    }

    for (int64_t i = paramDesc.count - 1U; i >= 0; i--) {
        if (paramDesc.instances[i].startIndex <= index) {
            irIndex = static_cast<size_t>(i);
            relativeIndex = static_cast<size_t>(index - paramDesc.instances[i].startIndex);
            return;
        }
    }
}
