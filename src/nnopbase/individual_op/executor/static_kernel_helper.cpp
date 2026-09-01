/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "static_kernel_helper.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>
#include "indv_collector.h"
#include "indv_executor.h"
#include "indv_executor_utils.h"
#include "opdev/data_type_utils.h"

using namespace op::internal;

namespace Indv {
namespace {
constexpr size_t NNOPBASE_OP_VERB_HEAD_LEN = 2U;
constexpr size_t NNOPBASE_STATIC_BIN_VERBOSE_INIT_LEN = 4096U;
constexpr size_t NNOPBASE_BIN_INFO_KEY_MAX_LEN = std::numeric_limits<uint32_t>::max();

size_t Align8ByteSize(const size_t size)
{
    return ((size + sizeof(uint64_t) - 1U) / sizeof(uint64_t)) * sizeof(uint64_t);
}

class StaticKeySegments {
public:
    explicit StaticKeySegments(std::string opType) : opType_(std::move(opType)) {}

    enum class Id : size_t {
        kHead = 0U,
        kPcie,
        kTensorsWithStride,
        kTensorsWithoutStride,
        kAttrs,
        kMax,
    };

    aclnnStatus InitOpHead(int64_t deterministicMode, int64_t implMode);
    aclnnStatus InitPcie(bool enablePcie);
    aclnnStatus InitTensorWithStride(const NnopbaseStaticTensorNumInfo* tensorNumInfo, const aclTensor* tensors[],
                                     const int64_t* valueDepend);
    aclnnStatus InitTensor(const NnopbaseStaticTensorNumInfo* tensorNumInfo, const aclTensor* tensors[],
                           const int64_t* valueDepend);
    aclnnStatus InitAttr(const NnopbaseStaticTensorNumInfo* tensorNumInfo, const NnopbaseAttrAddr* attrs[]);
    aclnnStatus InitTensorWithStride(const NnopbaseTensors* inputs, size_t inputTensorNum,
                                     const NnopbaseTensors* outputs, size_t outputTensorNum);
    aclnnStatus InitTensor(const NnopbaseTensors* inputs, size_t inputTensorNum, const NnopbaseTensors* outputs,
                           size_t outputTensorNum);
    aclnnStatus InitAttr(const NnopbaseAttrs* attrs);
    NnopbaseBinInfo* FindWithLayout(BinInfoKey& binInfoKey, NnopbaseRegInfo* regInfo, std::initializer_list<Id> layout,
                                    const StaticKernelPlatformInfo* platformInfo) const;

private:
    static constexpr size_t ToIndex(Id id) { return static_cast<size_t>(id); }

    std::string opType_;
    std::array<BinInfoKey, static_cast<size_t>(Id::kMax)> segments_;
};

struct StaticBinSearchContext {
    StaticKeySegments& segments;
    BinInfoKey& binInfoKey;
    NnopbaseRegInfo* regInfo;
    const StaticKernelPlatformInfo* platformInfo;
    bool enablePcie;
};

aclnnStatus EnsureBinInfoKeyCapacity(BinInfoKey& binInfoKey, const size_t appendLen,
                                     const size_t maxLen = NNOPBASE_MAX_STATICKEY_LEN)
{
    if (appendLen == 0U) {
        return OK;
    }

    const size_t usedLen = static_cast<size_t>(binInfoKey.len);
    CHECK_COND(appendLen <= maxLen && usedLen <= maxLen - appendLen, ACLNN_ERR_PARAM_INVALID,
               "Static binInfoKey size is too large[%zu].", usedLen + appendLen);
    if (binInfoKey.verbose.empty()) {
        binInfoKey.verbose.resize(std::max(NNOPBASE_STATIC_BIN_VERBOSE_INIT_LEN, appendLen));
        binInfoKey.bufLen = static_cast<uint32_t>(binInfoKey.verbose.size());
        return OK;
    }

    CHECK_COND(usedLen <= static_cast<size_t>(binInfoKey.bufLen), ACLNN_ERR_PARAM_INVALID,
               "Static binInfoKey length[%zu] is larger than buffer length[%u].", usedLen, binInfoKey.bufLen);
    const size_t remainLen = static_cast<size_t>(binInfoKey.bufLen) - usedLen;
    if (remainLen >= appendLen) {
        return OK;
    }

    const size_t newBufLen = std::max(static_cast<size_t>(binInfoKey.bufLen) * 2U, usedLen + appendLen);
    binInfoKey.verbose.resize(newBufLen);
    binInfoKey.bufLen = static_cast<uint32_t>(newBufLen);
    return OK;
}

aclnnStatus AppendBinInfoKeyBinary(BinInfoKey& binInfoKey, const void* const src, const size_t srcLen,
                                   const size_t maxLen = NNOPBASE_MAX_STATICKEY_LEN)
{
    if (srcLen == 0U) {
        return OK;
    }
    CHECK_COND(src != nullptr, ACLNN_ERR_PARAM_INVALID, "Static binInfoKey append binary source is nullptr.");
    NNOPBASE_ASSERT_OK_RETVAL(EnsureBinInfoKeyCapacity(binInfoKey, srcLen, maxLen));

    const size_t usedLen = static_cast<size_t>(binInfoKey.len);
    NnopbaseUChar* const dst = binInfoKey.verbose.data() + usedLen;
    NnopbaseUChar* const end = NnopbaseAppendBinary(dst, static_cast<size_t>(binInfoKey.bufLen) - usedLen, src, srcLen);
    CHECK_COND(end == dst + srcLen, ACLNN_ERR_PARAM_INVALID, "Failed to append static binInfoKey binary.");
    binInfoKey.len = static_cast<uint32_t>(usedLen + srcLen);
    return OK;
}

aclnnStatus AppendBinInfoKey8Byte(BinInfoKey& binInfoKey, const uint64_t value,
                                  const size_t maxLen = NNOPBASE_MAX_STATICKEY_LEN)
{
    NNOPBASE_ASSERT_OK_RETVAL(EnsureBinInfoKeyCapacity(binInfoKey, sizeof(uint64_t), maxLen));
    const size_t usedLen = static_cast<size_t>(binInfoKey.len);
    NnopbaseUChar* const dst = binInfoKey.verbose.data() + usedLen;
    NnopbaseUChar* const end = NnopbaseAppend8Byte(dst, value);
    CHECK_COND(end == dst + sizeof(uint64_t), ACLNN_ERR_PARAM_INVALID, "Failed to append static binInfoKey 8 bytes.");
    binInfoKey.len = static_cast<uint32_t>(usedLen + sizeof(uint64_t));
    return OK;
}

aclnnStatus AppendBinInfoKeyAlignedBinary(BinInfoKey& binInfoKey, const NnopbaseUChar* const src, const size_t srcLen,
                                          const size_t maxLen = NNOPBASE_MAX_STATICKEY_LEN)
{
    if (srcLen == 0U) {
        return OK;
    }
    CHECK_COND(src != nullptr, ACLNN_ERR_PARAM_INVALID, "Static binInfoKey append aligned binary source is nullptr.");
    const size_t appendLen = Align8ByteSize(srcLen);
    NNOPBASE_ASSERT_OK_RETVAL(EnsureBinInfoKeyCapacity(binInfoKey, appendLen, maxLen));

    const size_t usedLen = static_cast<size_t>(binInfoKey.len);
    NnopbaseUChar* const dst = binInfoKey.verbose.data() + usedLen;
    NnopbaseUChar* const end = NnopbaseExecutor8ByteCopy(srcLen, dst, src);
    CHECK_COND(end == dst + appendLen, ACLNN_ERR_PARAM_INVALID, "Failed to append static binInfoKey aligned binary.");
    binInfoKey.len = static_cast<uint32_t>(usedLen + appendLen);
    return OK;
}

aclnnStatus AppendVerboseSegment(BinInfoKey& binInfoKey, const BinInfoKey& segment)
{
    if (segment.len == 0U) {
        return OK;
    }
    return AppendBinInfoKeyBinary(binInfoKey, segment.verbose.data(), static_cast<size_t>(segment.len),
                                  NNOPBASE_BIN_INFO_KEY_MAX_LEN);
}

NnopbaseBinInfo* StaticKeySegments::FindWithLayout(BinInfoKey& binInfoKey, NnopbaseRegInfo* const regInfo,
                                                   const std::initializer_list<Id> layout,
                                                   const StaticKernelPlatformInfo* const platformInfo) const
{
    binInfoKey.Reset();
    // 根据simplfiedKey指定布局来拼接verboseKey
    for (const auto id : layout) {
        if (AppendVerboseSegment(binInfoKey, segments_[ToIndex(id)]) != OK) {
            return nullptr;
        }
    }

    binInfoKey.hashKey = NnopbaseHashBinary(binInfoKey.verbose.data(), binInfoKey.len) % NNOPBASE_NORM_MAX_BIN_BUCKETS;
    auto binInfo = NnopbaseCollectorFindBinInfo(regInfo, binInfoKey.hashKey, binInfoKey.verbose.data(), binInfoKey.len,
                                                platformInfo);
    return binInfo;
}

NnopbaseBinInfo* FindStaticBinWithStride(const StaticBinSearchContext& ctx)
{
    if (ctx.enablePcie) {
        // 静态simplfiedKey拼接规则，使能pcie时需要包含pcie/stride/offset字段
        return ctx.segments.FindWithLayout(ctx.binInfoKey, ctx.regInfo,
                                           {StaticKeySegments::Id::kHead, StaticKeySegments::Id::kPcie,
                                            StaticKeySegments::Id::kTensorsWithStride, StaticKeySegments::Id::kAttrs},
                                           ctx.platformInfo);
    }
    OP_LOGI("Search static kernel without pcie.");
    // 静态simplfiedKey拼接规则，不带PCIE字段，但包含字段stride/offset
    return ctx.segments.FindWithLayout(
        ctx.binInfoKey, ctx.regInfo,
        {StaticKeySegments::Id::kHead, StaticKeySegments::Id::kTensorsWithStride, StaticKeySegments::Id::kAttrs},
        ctx.platformInfo);
}

NnopbaseBinInfo* FindStaticBinWithoutStride(const StaticBinSearchContext& ctx)
{
    // 静态simplfiedKey拼接规则，字段包含原始输入输出属性基本描述，优先级最低
    return ctx.segments.FindWithLayout(
        ctx.binInfoKey, ctx.regInfo,
        {StaticKeySegments::Id::kHead, StaticKeySegments::Id::kTensorsWithoutStride, StaticKeySegments::Id::kAttrs},
        ctx.platformInfo);
}

std::vector<bool> BuildValueDepend(const NnopbaseStaticTensorNumInfo* const tensorNumInfo,
                                   const int64_t* const valueDepend)
{
    std::vector<bool> valueDependFlags(tensorNumInfo->numTensors, false);
    for (int64_t i = 0; i < tensorNumInfo->numValueDepend; i++) {
        valueDependFlags[valueDepend[i]] = true;
    }
    return valueDependFlags;
}

aclnnStatus AppendApiStaticTensorsKey(BinInfoKey& segment, const NnopbaseStaticTensorNumInfo* const tensorNumInfo,
                                      const aclTensor* tensors[], const int64_t* const valueDepend,
                                      const bool usingStride)
{
    const auto valueDependFlags = BuildValueDepend(tensorNumInfo, valueDepend);
    const NnopbaseUChar* addr = nullptr;
    for (int64_t i = 0; i < tensorNumInfo->numTensors; i++) {
        if (tensors[i] == nullptr) {
            NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKey8Byte(segment, '_'));
            continue;
        }

        const auto dtype = tensors[i]->GetDataType();
        NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKey8Byte(segment, dtype));
        const auto format = tensors[i]->GetStorageFormat();
        NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKey8Byte(segment, format));
        OP_LOGI("Get tensor[%ld] datatype is %d, format is %d.", i, dtype, format);
        const gert::Shape& shape = tensors[i]->GetStorageShape();
        for (size_t j = 0U; j < shape.GetDimNum(); j++) {
            NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKey8Byte(segment, static_cast<uint64_t>(shape.GetDim(j))));
        }
        if (usingStride) {
            const auto stride = tensors[i]->GetTensor()->GetStride();
            const int64_t offset = tensors[i]->GetTensor()->GetOffset();
            std::string strideStr = "[";
            for (size_t j = 0U; j < stride.GetDimNum(); j++) {
                NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKey8Byte(segment, static_cast<uint64_t>(stride.GetStride(j))));
                strideStr += std::to_string(stride.GetStride(j));
                if (j != stride.GetDimNum() - 1U) {
                    strideStr += ", ";
                }
            }
            strideStr += "]";
            OP_LOGI("Tensor[%ld] strideDim is %zu, stride is %s, offset is %lld.", i, stride.GetDimNum(),
                    strideStr.c_str(), offset);
            NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKey8Byte(segment, static_cast<uint64_t>(offset)));
        }
        if (valueDependFlags[i]) {
            const int64_t elementSize = tensors[i]->Size();
            addr = PtrCastTo<NnopbaseUChar>(tensors[i]->GetData());
            const size_t typeSize = op::TypeSize(tensors[i]->GetDataType());
            for (int64_t k = 0; k < elementSize; k++) {
                NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKeyAlignedBinary(segment, addr + typeSize * k, typeSize));
            }
        }
    }
    return OK;
}

aclnnStatus AppendApiStaticAttrsKey(BinInfoKey& segment, const NnopbaseStaticTensorNumInfo* const tensorNumInfo,
                                    const NnopbaseAttrAddr* attrs[])
{
    const NnopbaseUChar* addr = nullptr;
    for (int64_t j = 0; j < tensorNumInfo->numAttrs; j++) {
        if (attrs[j] == nullptr) {
            NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKey8Byte(segment, '_'));
            continue;
        }
        if (attrs[j]->addr == nullptr || attrs[j]->size == 0U) {
            OP_LOGW("Attr[%ld] addr is nullptr or size is 0, skip concat simplifiedKey.", j);
            continue;
        }
        if (!attrs[j]->isVector) {
            addr = PtrCastTo<const NnopbaseUChar>(attrs[j]->addr);
            NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKeyAlignedBinary(segment, addr, attrs[j]->size));
        } else {
            const size_t elementSize = attrs[j]->elementSize;
            if (elementSize == 0U) {
                OP_LOGW("Attr[%ld] elementSize is 0, skip concat simplifiedKey.", j);
                continue;
            }
            const size_t elementNum = attrs[j]->size / elementSize;
            for (size_t i = 0U; i < elementNum; i++) {
                addr = PtrCastTo<const NnopbaseUChar>(attrs[j]->addr) + elementSize * i;
                NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKeyAlignedBinary(segment, addr, elementSize));
            }
        }
    }
    return OK;
}

aclnnStatus StaticKeySegments::InitOpHead(const int64_t deterministicMode, const int64_t implMode)
{
    const size_t len = opType_.size() + NNOPBASE_OP_VERB_HEAD_LEN * sizeof(uint64_t);
    auto& segment = segments_[ToIndex(Id::kHead)];
    segment.Clear();
    segment.verbose.resize(len);
    segment.bufLen = static_cast<uint32_t>(segment.verbose.size());

    NnopbaseUChar* verKey = NnopbaseAppendBinary(segment.verbose.data(), segment.verbose.size(), opType_.data(),
                                                 opType_.size());
    const uint64_t determinLevel = (deterministicMode == 0) ? 0U : 1U;
    verKey = NnopbaseAppend8Byte(verKey, determinLevel);
    verKey = NnopbaseAppend8Byte(verKey, static_cast<uint64_t>(implMode));
    OP_LOGI("Op %s deterministic is %llu, impl mode is %lld.", opType_.c_str(), determinLevel, implMode);
    CHECK_COND(verKey == segment.verbose.data() + segment.verbose.size(), ACLNN_ERR_PARAM_INVALID,
               "Static head segment length is unexpected.");
    segment.len = static_cast<uint32_t>(len);
    return OK;
}

aclnnStatus StaticKeySegments::InitPcie(const bool enablePcie)
{
    auto& segment = segments_[ToIndex(Id::kPcie)];
    segment.Clear();
    segment.verbose.resize(sizeof(uint64_t));
    segment.bufLen = static_cast<uint32_t>(segment.verbose.size());

    NnopbaseUChar* verKey = NnopbaseAppend8Byte(segment.verbose.data(), enablePcie ? 1U : 0U);
    CHECK_COND(verKey == segment.verbose.data() + segment.verbose.size(), ACLNN_ERR_PARAM_INVALID,
               "Static pcie segment length is unexpected.");
    segment.len = static_cast<uint32_t>(sizeof(uint64_t));
    return OK;
}

aclnnStatus StaticKeySegments::InitTensorWithStride(const NnopbaseStaticTensorNumInfo* const tensorNumInfo,
                                                    const aclTensor* tensors[], const int64_t* const valueDepend)
{
    auto& segment = segments_[ToIndex(Id::kTensorsWithStride)];
    segment.Reset();
    NNOPBASE_ASSERT_OK_RETVAL(AppendApiStaticTensorsKey(segment, tensorNumInfo, tensors, valueDepend, true));
    CHECK_COND(segment.len <= NNOPBASE_MAX_STATICKEY_LEN, ACLNN_ERR_PARAM_INVALID,
               "Static API with-stride tensor segment size is too large[%u].", segment.len);
    return OK;
}

aclnnStatus StaticKeySegments::InitTensor(const NnopbaseStaticTensorNumInfo* const tensorNumInfo,
                                          const aclTensor* tensors[], const int64_t* const valueDepend)
{
    auto& segment = segments_[ToIndex(Id::kTensorsWithoutStride)];
    segment.Reset();
    NNOPBASE_ASSERT_OK_RETVAL(AppendApiStaticTensorsKey(segment, tensorNumInfo, tensors, valueDepend, false));
    CHECK_COND(segment.len <= NNOPBASE_MAX_STATICKEY_LEN, ACLNN_ERR_PARAM_INVALID,
               "Static API no-stride tensor segment size is too large[%u].", segment.len);
    return OK;
}

aclnnStatus StaticKeySegments::InitAttr(const NnopbaseStaticTensorNumInfo* const tensorNumInfo,
                                        const NnopbaseAttrAddr* attrs[])
{
    auto& segment = segments_[ToIndex(Id::kAttrs)];
    segment.Reset();
    NNOPBASE_ASSERT_OK_RETVAL(AppendApiStaticAttrsKey(segment, tensorNumInfo, attrs));
    CHECK_COND(segment.len <= NNOPBASE_MAX_STATICKEY_LEN, ACLNN_ERR_PARAM_INVALID,
               "Static API attr segment size is too large[%u].", segment.len);
    return OK;
}

aclnnStatus AppendExecutorStaticTensorsKey(BinInfoKey& segment, const NnopbaseTensors* const tensors,
                                           const size_t tensorNum, const bool usingStride)
{
    const NnopbaseUChar* addr = nullptr;
    for (size_t i = 0U; i < tensorNum; i++) {
        if (tensors->extTensors[i].isNull) {
            OP_LOGI("Tensor[%zu] is null.", i);
            NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKey8Byte(segment, '_'));
            continue;
        }
        if (tensors->extTensors[i].isRequired || tensors->extTensors[i].isOptional) {
            NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKey8Byte(segment, tensors->extTensors[i].rt2Tensor.GetDataType()));
            NNOPBASE_ASSERT_OK_RETVAL(
                AppendBinInfoKey8Byte(segment, tensors->extTensors[i].rt2Tensor.GetStorageFormat()));
            OP_LOGI("Tensor[%zu] datatype is %d, format is %d, isRequired is %s, isOptional is %s.", i,
                    static_cast<int32_t>(tensors->extTensors[i].rt2Tensor.GetDataType()),
                    static_cast<int32_t>(tensors->extTensors[i].rt2Tensor.GetStorageFormat()),
                    tensors->extTensors[i].isRequired ? "true" : "false",
                    tensors->extTensors[i].isOptional ? "true" : "false");
        }
        const GertShape& shape = tensors->extTensors[i].rt2Tensor.GetStorageShape();
        for (size_t j = 0U; j < shape.GetDimNum(); j++) {
            NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKey8Byte(segment, static_cast<uint64_t>(shape.GetDim(j))));
            OP_LOGI("Tensor[%zu] storageShape[%zu] is %ld", i, j, shape.GetDim(j));
        }
        if (usingStride) {
            const auto& stride = tensors->extTensors[i].rt2Tensor.GetStride();
            for (size_t k = 0U; k < stride.GetDimNum(); k++) {
                NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKey8Byte(segment, static_cast<uint64_t>(stride.GetStride(k))));
            }
            NNOPBASE_ASSERT_OK_RETVAL(
                AppendBinInfoKey8Byte(segment, static_cast<uint64_t>(tensors->extTensors[i].rt2Tensor.GetOffset())));
        }
        if (tensors->extTensors[i].valueDepend) {
            addr = PtrCastTo<const NnopbaseUChar>(tensors->extTensors[i].rt2Tensor.GetAddr());
            const auto dtype = tensors->extTensors[i].rt2Tensor.GetDataType();
            const size_t length = tensors->extTensors[i].rt2Tensor.GetSize();
            const size_t typeSize = op::TypeSize(dtype);
            const size_t elementNum = length / typeSize;
            for (size_t k = 0; k < elementNum; k++) {
                NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKeyAlignedBinary(segment, addr + typeSize * k, typeSize));
            }
        }
    }
    return OK;
}

aclnnStatus AppendExecutorStaticAttrsKey(BinInfoKey& segment, const NnopbaseAttrs* const attrs)
{
    const NnopbaseUChar* addr = nullptr;
    for (size_t j = 0; j < attrs->num; j++) {
        // 传入时已校验 attrs[j].addr.addr 不为空
        if (!attrs->attrs[j].addr.isVector) {
            if (attrs->attrs[j].dtype == NnopbaseAttrDtype::kNnopbaseString && attrs->attrs[j].addr.size == 1U) {
                OP_LOGW("For Attr %zu, this is a string type and actual value is empty, skip concating verKey.", j);
                continue;
            }
            addr = PtrCastTo<const NnopbaseUChar>(attrs->attrs[j].addr.addr);
            NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKeyAlignedBinary(segment, addr, attrs->attrs[j].addr.size));
        } else {
            const size_t elementSize = attrs->attrs[j].addr.elementSize;
            if (elementSize == 0) {
                continue;
            }
            for (size_t i = 0U; i < attrs->attrs[j].addr.size / elementSize; i++) {
                addr = PtrCastTo<const NnopbaseUChar>(attrs->attrs[j].addr.addr) + elementSize * i;
                NNOPBASE_ASSERT_OK_RETVAL(AppendBinInfoKeyAlignedBinary(segment, addr, elementSize));
            }
        }
    }
    return OK;
}

aclnnStatus StaticKeySegments::InitTensorWithStride(const NnopbaseTensors* const inputs, const size_t inputTensorNum,
                                                    const NnopbaseTensors* const outputs, const size_t outputTensorNum)
{
    auto& segment = segments_[ToIndex(Id::kTensorsWithStride)];
    segment.Reset();
    NNOPBASE_ASSERT_OK_RETVAL(AppendExecutorStaticTensorsKey(segment, inputs, inputTensorNum, true));
    NNOPBASE_ASSERT_OK_RETVAL(AppendExecutorStaticTensorsKey(segment, outputs, outputTensorNum, true));
    CHECK_COND(segment.len <= NNOPBASE_MAX_STATICKEY_LEN, ACLNN_ERR_PARAM_INVALID,
               "Op %s static with-stride tensor segment size is too large[%u].", opType_.c_str(), segment.len);
    OP_LOGD("Op %s static with-stride tensor segment size is %u.", opType_.c_str(), segment.len);
    return OK;
}

aclnnStatus StaticKeySegments::InitTensor(const NnopbaseTensors* const inputs, const size_t inputTensorNum,
                                          const NnopbaseTensors* const outputs, const size_t outputTensorNum)
{
    auto& segment = segments_[ToIndex(Id::kTensorsWithoutStride)];
    segment.Reset();
    NNOPBASE_ASSERT_OK_RETVAL(AppendExecutorStaticTensorsKey(segment, inputs, inputTensorNum, false));
    NNOPBASE_ASSERT_OK_RETVAL(AppendExecutorStaticTensorsKey(segment, outputs, outputTensorNum, false));
    CHECK_COND(segment.len <= NNOPBASE_MAX_STATICKEY_LEN, ACLNN_ERR_PARAM_INVALID,
               "Op %s static no-stride tensor segment size is too large[%u].", opType_.c_str(), segment.len);
    OP_LOGD("Op %s static no-stride tensor segment size is %u.", opType_.c_str(), segment.len);
    return OK;
}

aclnnStatus StaticKeySegments::InitAttr(const NnopbaseAttrs* const attrs)
{
    auto& segment = segments_[ToIndex(Id::kAttrs)];
    segment.Reset();
    NNOPBASE_ASSERT_OK_RETVAL(AppendExecutorStaticAttrsKey(segment, attrs));
    CHECK_COND(segment.len <= NNOPBASE_MAX_STATICKEY_LEN, ACLNN_ERR_PARAM_INVALID,
               "Op %s static attr segment size is too large[%u].", opType_.c_str(), segment.len);
    return OK;
}

} // namespace

const NnopbaseChar* StaticKernelHelper::FindStaticKernelPath(const aclTensor* tensors[],
                                                             const NnopbaseAttrAddr* attrs[],
                                                             const int64_t valueDepend[],
                                                             const NnopbaseStaticTensorNumInfo* const tensorNumInfo,
                                                             const NnopbaseStaticRuntimeInfo* const staticRuntimeInfo)
{
    NnopbaseRegInfoKey regInfoKey;
    regInfoKey.opType = staticRuntimeInfo->opType;
    regInfoKey.hashKey = static_cast<uint64_t>(
        NnopbaseHashBinary(PtrCastTo<NnopbaseUChar>(regInfoKey.opType.c_str()), regInfoKey.opType.size()) %
        NNOPBASE_NORM_MAX_BIN_BUCKETS);
    OP_LOGI("OpType is %s, hashkey is %lu.", regInfoKey.opType.c_str(), regInfoKey.hashKey);

    NnopbaseRegInfo* regInfo = NnopbaseCollectorFindRegInfoInTbl(gBinCollector, regInfoKey.opType.c_str(),
                                                                 regInfoKey.hashKey);
    if (regInfo == nullptr) {
        return nullptr;
    }
    StaticKernelPlatformInfo platformInfo{{staticRuntimeInfo->aicNum, staticRuntimeInfo->aivNum},
                                          static_cast<int8_t>(staticRuntimeInfo->deterMode)};
    StaticKeySegments segments(regInfo->key.opType);
    if (segments.InitOpHead(staticRuntimeInfo->deterMode, staticRuntimeInfo->implMode) != OK) {
        return nullptr;
    }
    if (segments.InitPcie(staticRuntimeInfo->enablePcie) != OK) {
        return nullptr;
    }
    if (segments.InitTensorWithStride(tensorNumInfo, tensors, valueDepend) != OK) {
        return nullptr;
    }
    if (segments.InitAttr(tensorNumInfo, attrs) != OK) {
        return nullptr;
    }

    BinInfoKey binInfoKey;
    StaticBinSearchContext searchCtx{segments, binInfoKey, regInfo, &platformInfo, staticRuntimeInfo->enablePcie};
    NnopbaseBinInfo* binInfo = FindStaticBinWithStride(searchCtx);
    if (binInfo == nullptr) {
        OP_LOGW("Cannot find static kernel bin with stride information, trying to find without stride again.");
        if (segments.InitTensor(tensorNumInfo, tensors, valueDepend) != OK) {
            return nullptr;
        }
        binInfo = FindStaticBinWithoutStride(searchCtx);
    }
    return (binInfo == nullptr) ? nullptr : binInfo->binPath.c_str();
}

NnopbaseBinInfo* StaticKernelHelper::FindStaticBinInfo(NnopbaseExecutor* const executor, BinInfoKey& binInfoKey)
{
    const NnopbaseTensors* const inputs = &executor->ownArgs.inputs;
    const NnopbaseAttrs* const attrs = &executor->attrs;
    const NnopbaseTensors* const outputs = &executor->ownArgs.outputs;
    const size_t inputTensorNum = inputs->nonDynamicCnt + inputs->dynamicCnt;
    const size_t outputTensorNum = outputs->nonDynamicCnt + outputs->dynamicCnt;

    StaticKeySegments segments(executor->opType);
    if (segments.InitOpHead(executor->deterministicLevel, g_nnopbaseSysCfgParams.precision) != OK) {
        return nullptr;
    }
    if (segments.InitPcie(executor->isEnablePcie) != OK) {
        return nullptr;
    }
    if (segments.InitTensorWithStride(inputs, inputTensorNum, outputs, outputTensorNum) != OK) {
        return nullptr;
    }
    if (segments.InitAttr(attrs) != OK) {
        return nullptr;
    }
    StaticKernelPlatformInfo platformInfo{NnopbaseCoreNum{executor->coreNum.aicNum, executor->coreNum.aivNum},
                                          static_cast<int8_t>(executor->deterministicLevel)};
    StaticBinSearchContext searchCtx{segments, binInfoKey, executor->regInfo, &platformInfo, executor->isEnablePcie};
    NnopbaseBinInfo* binInfo = FindStaticBinWithStride(searchCtx);
    if (binInfo == nullptr) {
        OP_LOGW("Cannot find static kernel bin with stride information, trying to find without stride again.");
        if (segments.InitTensor(inputs, inputTensorNum, outputs, outputTensorNum) != OK) {
            return nullptr;
        }
        binInfo = FindStaticBinWithoutStride(searchCtx);
    }
    return binInfo;
}

} // namespace Indv
