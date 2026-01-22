/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "indv_args_pool.h"
#include "mmpa/mmpa_api.h"
#include "utils/thread_var_container.h"

namespace nnopbase {
namespace {
// cache中的总数，默认1万，用户可控，上限1000万
constexpr size_t NNOPBASE_CACHE_ARGS_NUM = 10000U;
constexpr size_t NNOPBASE_MAX_CACHE_NUM = 10000000U;
constexpr size_t HASH_FACTOR = 2U;
constexpr size_t NNOPBASE_BASE_BYTES = 6U; // datatype(1), format(1), dimNum(4)
constexpr size_t NNOPBASE_SHAPE_BYTES = 8U; // shape dim

size_t CalcKeyLen(NnopbaseExecutor *executor)
{
    auto inputs = &executor->ownArgs.inputs;
    auto outputs = &executor->ownArgs.outputs;
    auto attrs = &executor->attrs;
    size_t keyLen = strlen(executor->opType);
    for (uint32_t i = 0U; i < inputs->num; i++) {
        if (inputs->extTensors[i].isNull) {
            ++keyLen;
            continue;
        }
        keyLen += NNOPBASE_BASE_BYTES +
                  NNOPBASE_SHAPE_BYTES * inputs->extTensors[i].rt2Tensor.MutableStorageShape().GetDimNum();
        if (inputs->extTensors[i].valueDepend) {
            keyLen += inputs->extTensors[i].rt2Tensor.GetSize();
        }
    }
    ++keyLen;
    for (uint32_t i = 0U; i < outputs->num; i++) {
        if (outputs->extTensors[i].isNull) {
            ++keyLen;
            continue;
        }
        keyLen += NNOPBASE_BASE_BYTES +
                  NNOPBASE_SHAPE_BYTES * outputs->extTensors[i].rt2Tensor.MutableStorageShape().GetDimNum();
    }
    ++keyLen;
    if (attrs->num > 0U) {
        for (size_t i = 0U; i < attrs->num; i++) {
            keyLen += attrs->attrs[i].addr.size;
        }
    }
    ++keyLen;
    keyLen += sizeof(NnopbaseCoreNum);
    ++keyLen;
    keyLen += sizeof(uint32_t);
    return keyLen;
}

uint8_t *AddShapeInfo(const GertTensor &rt2Tensor, uint8_t *key)
{
    key = NnopbaseAppend1Byte(key, (NnopbaseUChar) rt2Tensor.GetDataType());
    key = NnopbaseAppend1Byte(key, (NnopbaseUChar) rt2Tensor.GetStorageFormat());
    const GertShape &shape = rt2Tensor.GetStorageShape();
    const size_t dimNum = shape.GetDimNum();
    key = (uint8_t *) NnopbaseAppend4Byte(key, static_cast<uint32_t>(dimNum));
    for (size_t j = 0U; j < dimNum; j++) {
        key = (uint8_t *) NnopbaseAppend8Byte(key, static_cast<uint64_t>(shape.GetDim(j)));
    }
    return key;
}

// 溢出检测、确定性计算等是全局配置选项，不需要在后续每次匹配时都考虑
// optype, input dtype/format/shape, output dtype/format/shape, attr
void GenArgsKey(NnopbaseExecutor *executor)
{
    executor->ownArgs.keyLen = CalcKeyLen(executor);
    OP_LOGI("KeyLen is %zu.", executor->ownArgs.keyLen);
    if (executor->ownArgs.keyLen > executor->ownArgs.inputKey.size()) {
        executor->ownArgs.inputKey.resize(executor->ownArgs.keyLen);
    }

    const auto inputs = &executor->ownArgs.inputs;
    const auto outputs = &executor->ownArgs.outputs;
    const auto attrs = &executor->attrs;
    const auto coreNum = &executor->coreNum;
    uint8_t *key = executor->ownArgs.inputKey.data();
    key = (uint8_t *) NnopbaseAppendBinary(key, executor->ownArgs.inputKey.size(), executor->opType, strlen(executor->opType));
    for (uint32_t i = 0U; i < inputs->num; i++) {
        if (inputs->extTensors[i].isNull) {
            key = NnopbaseAppend1Byte(key, '/');
            continue;
        }
        key = AddShapeInfo(inputs->extTensors[i].rt2Tensor, key);
        if (inputs->extTensors[i].valueDepend) {
            const auto addr = inputs->extTensors[i].rt2Tensor.GetAddr();
            const auto length = inputs->extTensors[i].rt2Tensor.GetSize();
            key = (uint8_t *) NnopbaseAppendBinary(key, length, addr, length);
        }
    }

    key = NnopbaseAppend1Byte(key, '/');
    for (uint32_t i = 0U; i < outputs->num; i++) {
        if (outputs->extTensors[i].isNull) {
            key = NnopbaseAppend1Byte(key, '/');
            continue;
        }
        key = AddShapeInfo(outputs->extTensors[i].rt2Tensor, key);
    }
    key = NnopbaseAppend1Byte(key, '/');
    if (attrs->num > 0U) {
        for (size_t i = 0U; i < attrs->num; i++) {
            key = (NnopbaseUChar*) NnopbaseAppendBinary(key, attrs->attrs[i].addr.size, attrs->attrs[i].addr.addr,
                                                        attrs->attrs[i].addr.size);
        }
    }
    key = NnopbaseAppend1Byte(key, '/');
    key = op::internal::PtrCastTo<NnopbaseUChar>(NnopbaseAppendBinary(key, sizeof(NnopbaseCoreNum), coreNum, sizeof(NnopbaseCoreNum)));
    key = NnopbaseAppend1Byte(key, '/');
    uint32_t mc2RankId = nnopbase::utils::ThreadVarContainer::GetCurMc2RankIdInThread();
    key = op::internal::PtrCastTo<NnopbaseUChar>(NnopbaseAppendBinary(key, sizeof(uint32_t), &mc2RankId, sizeof(uint32_t)));
    executor->ownArgs.seed = NnopbaseHashBinary(executor->ownArgs.inputKey.data(), executor->ownArgs.keyLen);
}
} // namespace

size_t ArgsPool::maxCacheNum = ArgsPool::GetCacheSizeLimit();

size_t ArgsPool::GetCacheSizeLimit()
{
    const char_t *cacheLimit = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ACLNN_CACHE_LIMIT, cacheLimit);
    size_t c = NNOPBASE_CACHE_ARGS_NUM;
    if (cacheLimit != nullptr) {
        try {
            c = std::stoull(cacheLimit);
        } catch (const std::exception &e) {
            OP_LOGI("Env variable ACLNN_CACHE_LIMIT[%s] is invalid! must be a number!", cacheLimit);
        }
    }
    c = std::min(c, NNOPBASE_MAX_CACHE_NUM);
    OP_LOGI("Cachelimit is %zu", c);
    GetInstance().argsCache.reserve(c * HASH_FACTOR);
    GetInstance().argsMap.reserve(c * HASH_FACTOR);
    return c;
}

ArgsPool &ArgsPool::GetInstance()
{
    static ArgsPool args;
    return args;
}

ArgsPool::~ArgsPool()
{
    Finalize();
}

void ArgsPool::Finalize()
{
    for (auto &iter : argsMap) {
        for (auto &args : iter.second) {
            if (args != nullptr) {
                delete args;
                args = nullptr;
            }
        }
    }
    fixedCacheMap.clear();
    argsMap.clear();
    cacheList.clear();
    argsCache.clear();
}

bool ArgsPool::IsArgsMatch(NnopbaseExecutorArgs *const args, NnopbaseExecutor *executor)
{
    if (args->isVist) {
        OP_LOGI("Op %s seed %zu args %p is visit.", executor->opType, args->seed, args);
        return false;
    }
    if (args->keyLen != executor->ownArgs.keyLen) {
        OP_LOGI("Op %s seed %zu key len is not equal, cache len %zu, input len %zu.", executor->opType,
                args->seed, args->keyLen, executor->ownArgs.keyLen);
        return false;
    }
    if (memcmp(args->inputKey.data(), executor->ownArgs.inputKey.data(), args->keyLen) == 0) {
        executor->args = args;
        executor->hasTiling = (!args->binInfo->isStaticShape);
        executor->isCachedArgs = true;
        args->isVist = true;
        Get(executor->args);
        OP_LOGI("Op %s match args cache successfully, seed is %zu, key len is %zu.",
                executor->opType, args->seed, args->keyLen);
        return true;
    }
    return false;
}

bool ArgsPool::MatchArgs(NnopbaseExecutor *executor)
{
    RecordNnopbaseTime(executor, NnopbaseTimeIdx::kMatchCacheStart);
    if (maxCacheNum == 0U) {
        executor->ownArgs.enableCache = false;
        RecordNnopbaseTime(executor, NnopbaseTimeIdx::kMatchCacheEnd);
        return false;
    }
    if (!executor->matchArgsV2) {
        GenArgsKey(executor);
    }
    {
        const std::lock_guard<std::mutex> lk(mutex);
        const auto &iter = argsMap.find(executor->ownArgs.seed);
        if (iter != argsMap.end()) {
            OP_LOGI("Op %s seed %zu args num is %zu.", executor->opType, executor->ownArgs.seed, iter->second.size());
            for (auto &args : iter->second) {
                if (IsArgsMatch(args, executor)) {
                    RecordNnopbaseTime(executor, NnopbaseTimeIdx::kMatchCacheEnd);
                    return true;
                }
            }
        }
    }
    OP_LOGI("Op %s not match args cache, seed is %zu, key len is %zu.", executor->opType,
            executor->ownArgs.seed, executor->ownArgs.keyLen);
    RecordNnopbaseTime(executor, NnopbaseTimeIdx::kMatchCacheEnd);
    return false;
}

aclnnStatus ArgsPool::CreateArgs(NnopbaseExecutor *executor)
{
    if (executor->ownArgs.enableCache) {
        auto args = std::make_unique<NnopbaseExecutorArgs>();
        NNOPBASE_ASSERT_NOTNULL_RETVAL(args);
        {
            const std::lock_guard<std::mutex> lk(mutex);
            (void)argsMap[executor->ownArgs.seed].emplace_back(args.release());
            executor->args = argsMap[executor->ownArgs.seed].back();
            executor->args->isVist = true;
            RecordNnopbaseTime(executor, NnopbaseTimeIdx::kCreateCacheEnd);
            Put(executor->args);
        }
        executor->args->keyLen = executor->ownArgs.keyLen;
        executor->args->seed = executor->ownArgs.seed;
        if (executor->args->keyLen > executor->args->inputKey.size()) {
            executor->args->inputKey.resize(executor->args->keyLen);
        }
        NNOPBASE_ASSERT_TRUE_RETVAL(memcpy_s(executor->args->inputKey.data(),
                                        executor->ownArgs.keyLen,
                                        executor->ownArgs.inputKey.data(),
                                        executor->ownArgs.keyLen) == EOK);
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSaveCachedTensor(&executor->args->inputs, &executor->ownArgs.inputs, true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSaveCachedTensor(&executor->args->outputs, &executor->ownArgs.outputs, false));
        NnopbaseSaveUnContiguousTensors(&executor->args->inputs, &executor->ownArgs.inputs);
    } else {
        executor->args = &executor->ownArgs;
        RecordNnopbaseTime(executor, NnopbaseTimeIdx::kCreateCacheEnd);
    }
    RecordNnopbaseTime(executor, NnopbaseTimeIdx::kAgingCacheEnd);
    return OK;
}

void ArgsPool::EraseArgs(NnopbaseExecutorArgs *const tmp)
{
    auto &argsList = argsMap[tmp->seed];
    for (auto it = argsList.begin(); it != argsList.end(); it++) {
        if (*it == tmp) {
            OP_LOGI("The number of args %zu cached has reached %zu, delete the oldest args %p seed %zu.",
                    argsCache.size(), maxCacheNum, tmp, tmp->seed);
            (void)argsList.erase(it);
            if (argsList.empty()) {
                (void)argsMap.erase(tmp->seed);
            }
            break;
        }
    }
    (void)argsCache.erase(tmp);
    delete tmp;
    return;
}

// 在调用端保证一定传入新的args进来，make出来的args才会调用Put
void ArgsPool::Put(NnopbaseExecutorArgs *const args)
{
    while ((argsCache.size() >= maxCacheNum) && (!cacheList.empty())) {
        const auto &tmp = cacheList.back();
        if (tmp->isVist) {
            OP_LOGW("The number of args cached has reached %zu, but the oldest args %p seed %zu is in use! "
                    "Can't delete args!", maxCacheNum, tmp, tmp->seed);
            break;
        }
        EraseArgs(tmp);
        cacheList.pop_back();
    }
    cacheList.emplace_front(args);
    argsCache[args] = cacheList.begin();
}

void ArgsPool::FixCache(NnopbaseExecutorArgs *const args)
{
    const std::lock_guard<std::mutex> lk(mutex);
    auto &argsList = argsMap[args->seed];
    for (auto it = argsList.begin(); it != argsList.end(); it++) {
        if (*it == args) {
            (void)argsList.erase(it);
            const auto &argsPos = argsCache.find(args);
            if (argsPos != argsCache.end()) {
                cacheList.erase(argsPos->second);
            }
            if (argsList.empty()) {
                (void)argsMap.erase(args->seed);
            }
            break;
        }
    }
    (void)argsCache.erase(args);
    fixedCacheMap[args->seed].push_back(args);
    OP_LOGI("Fix args cache successfully, current fixed cache pool size for seed %zu is %zu", args->seed,
        fixedCacheMap[args->seed].size());
    return;
}

void ArgsPool::ReleaseFixedCache(NnopbaseExecutorArgs *const args)
{
    auto cacheListIter = fixedCacheMap.find(args->seed);
    if (cacheListIter == fixedCacheMap.cend()) {
        OP_LOGD("Cannot find seed %zu in fixed cache pool, args: %p", args->seed, args);
        return;
    }
    auto &fixedCacheList = cacheListIter->second;
    for (auto it = fixedCacheList.begin(); it != fixedCacheList.end(); it++) {
        if (*it == args) {
            Put(args);
            fixedCacheList.erase(it);
            break;
        }
    }
    if (fixedCacheList.empty()) {
        (void)fixedCacheMap.erase(args->seed);
    }
    OP_LOGI("Release fixed args cache successfully, current fixed cache pool size is %zu", fixedCacheMap.size());
    return;
}
} // nnopbase

NnopbaseUChar *NnopbaseAddShapeInfo(const aclTensor *tensor, NnopbaseExecutorArgs *args)
{
    const op::Shape &shape = tensor->GetViewShape();
    const size_t dimNum = shape.GetDimNum();
    size_t len = nnopbase::NNOPBASE_BASE_BYTES + dimNum * nnopbase::NNOPBASE_SHAPE_BYTES;
    if (args->remainKeyLen < len) {
        args->remainKeyLen += NNOPBASE_MAX_ARGS_KEY_LEN;
        args->inputKey.resize(args->inputKey.size() + NNOPBASE_MAX_ARGS_KEY_LEN);
    }
    NnopbaseUChar *key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data()) + args->keyLen;
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(tensor->GetDataType()));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(tensor->GetStorageFormat()));
    key = NnopbaseAppend4Byte(key, static_cast<uint32_t>(dimNum));
    for (size_t j = 0U; j < dimNum; j++) {
        key = NnopbaseAppend8Byte(key, static_cast<uint64_t>(shape.GetDim(j)));
    }
    args->remainKeyLen -= len;
    args->keyLen += len;
    return key;
}

NnopbaseUChar* NnopbaseGenPlaceHolderKey(NnopbaseExecutorArgs *args, NnopbaseUChar *key)
{
    if (args->remainKeyLen < 1) {
        args->remainKeyLen += NNOPBASE_MAX_ARGS_KEY_LEN;
        args->inputKey.resize(args->inputKey.size() + NNOPBASE_MAX_ARGS_KEY_LEN);
        key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    }
    key = NnopbaseAppend1Byte(key, '/');
    args->keyLen += 1;
    args->remainKeyLen -= 1;
    return key;
}

NnopbaseUChar* NnopbaseAddCoreNumInfo(const NnopbaseCoreNum *coreNum, NnopbaseExecutorArgs *args)
{
    NnopbaseUChar *key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    key = NnopbaseGenPlaceHolderKey(args, key);
    if (args->remainKeyLen < sizeof(NnopbaseCoreNum)) {
        args->remainKeyLen += NNOPBASE_MAX_ARGS_KEY_LEN;
        args->inputKey.resize(args->inputKey.size() + NNOPBASE_MAX_ARGS_KEY_LEN);
        key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    }
    key = op::internal::PtrCastTo<NnopbaseUChar>(NnopbaseAppendBinary(key, args->remainKeyLen, coreNum, sizeof(NnopbaseCoreNum)));
    args->keyLen += sizeof(NnopbaseCoreNum);
    args->remainKeyLen -= sizeof(NnopbaseCoreNum);
    return key;
}

NnopbaseUChar* AddMc2RankIdInfoToKey(const uint32_t *mc2RankId, NnopbaseExecutorArgs *args)
{
    NnopbaseUChar *key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    key = NnopbaseGenPlaceHolderKey(args, key);
    if (args->remainKeyLen < sizeof(uint32_t)) {
        args->remainKeyLen += NNOPBASE_MAX_ARGS_KEY_LEN;
        args->inputKey.resize(args->inputKey.size() + NNOPBASE_MAX_ARGS_KEY_LEN);
        key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    }
    key = op::internal::PtrCastTo<NnopbaseUChar>(NnopbaseAppendBinary(key, args->remainKeyLen, mc2RankId, sizeof(uint32_t)));
    args->keyLen += sizeof(uint32_t);
    args->remainKeyLen -= sizeof(uint32_t);
    return key;
}

void NnopbaseGenTensorKey(NnopbaseExecutorArgs *args, const aclTensor *tensor)
{
    NnopbaseUChar *key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    if (tensor == nullptr) {
        key = NnopbaseGenPlaceHolderKey(args, key);
    } else {
        key = NnopbaseAddShapeInfo(tensor, args);
    }
}

void NnopbaseExecutorGenValueDependTensorKey(
    NnopbaseExecutorArgs *args, const void *addr, const uint64_t dim, const uint64_t dataLen, ge::DataType dType)
{
    const size_t shapeLen = nnopbase::NNOPBASE_BASE_BYTES + nnopbase::NNOPBASE_SHAPE_BYTES;
    if (args->remainKeyLen < shapeLen + dataLen) {
        size_t multiples = (shapeLen + dataLen) / NNOPBASE_MAX_ARGS_KEY_LEN + 1;
        args->remainKeyLen += multiples * NNOPBASE_MAX_ARGS_KEY_LEN;
        args->inputKey.resize(args->inputKey.size() + multiples * NNOPBASE_MAX_ARGS_KEY_LEN);
    }
    NnopbaseUChar *key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(dType));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(ge::FORMAT_ND));
    key = NnopbaseAppend4Byte(key, 1);
    key = NnopbaseAppend8Byte(key, dim);
    args->remainKeyLen -= shapeLen;
    args->keyLen += shapeLen;
    key = NnopbaseAppendBinary(key, args->remainKeyLen, addr, dataLen);
    args->remainKeyLen -= dataLen;
    args->keyLen += dataLen;
}

inline bool NnopbaseExecutorCompreTensor(const aclTensor *tensor, const aclTensor *prev)
{
    return (tensor->GetDataType() == prev->GetDataType()) && (tensor->GetViewShape() == prev->GetViewShape()) &&
           (tensor->GetStorageFormat() == prev->GetStorageFormat());
}
 
void NnopbaseExecutorGenTensorListKey(NnopbaseExecutorArgs *args, const aclTensorList *tensorList)
{
    NnopbaseUChar *key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    if ((tensorList == nullptr) || (tensorList->Size() == 0U)) {
        NnopbaseGenPlaceHolderKey(args, key);
        return;
    }
    uint32_t cnt = 1U;
    auto prev = (*tensorList)[0];
    for (uint64_t i = 0U; i < tensorList->Size(); i++) {
        if (i > 0U) {
            if (((*tensorList)[i] != nullptr) && prev != nullptr &&
                NnopbaseExecutorCompreTensor((*tensorList)[i], prev)) {
                cnt++;
            } else {
                if (prev != nullptr) {
                    key = NnopbaseAddShapeInfo(prev, args);
                    if (args->remainKeyLen < 1) {
                        args->remainKeyLen += NNOPBASE_MAX_ARGS_KEY_LEN;
                        args->inputKey.resize(args->inputKey.size() + NNOPBASE_MAX_ARGS_KEY_LEN);
                        key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
                    }
                    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(cnt));
                    args->keyLen += 1;
                    args->remainKeyLen -= 1;
                }
                if ((*tensorList)[i] == nullptr) {
                    key = NnopbaseGenPlaceHolderKey(args, key);
                    prev = (*tensorList)[i];
                    continue;
                }
                prev = (*tensorList)[i];
                cnt = 1;
            }
        }
    }
    if (prev != nullptr) {
        key = NnopbaseAddShapeInfo(prev, args);
        if (args->remainKeyLen < 1) {
            args->remainKeyLen += NNOPBASE_MAX_ARGS_KEY_LEN;
            args->inputKey.resize(args->inputKey.size() + NNOPBASE_MAX_ARGS_KEY_LEN);
            key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
        }
        key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(cnt));
        args->keyLen += 1;
        args->remainKeyLen -= 1;
    }
}

static NnopbaseUChar *NnopbaseAddScalarInfo(NnopbaseExecutorArgs *args, const ge::DataType dtype)
{
    // scalar dimNum为0，无需占用8字节填shape信息
    if (args->remainKeyLen < nnopbase::NNOPBASE_BASE_BYTES) {
        args->remainKeyLen += NNOPBASE_MAX_ARGS_KEY_LEN;
        args->inputKey.resize(args->inputKey.size() + NNOPBASE_MAX_ARGS_KEY_LEN);
    }
    NnopbaseUChar *key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data()) + args->keyLen;
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(dtype));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(ge::FORMAT_ND));
    key = NnopbaseAppend4Byte(key, static_cast<uint32_t>(0U));
    args->remainKeyLen -= nnopbase::NNOPBASE_BASE_BYTES;
    args->keyLen += nnopbase::NNOPBASE_BASE_BYTES;
    return key;
}

void NnopbaseExecutorGenScalarKey(NnopbaseExecutor *executor, const aclScalar *scalar, const uint32_t index,
    const int32_t srcIndex, const ge::DataType dtype)
{
    NnopbaseExecutorArgs *args = &executor->ownArgs;
    if (scalar == nullptr) {
        NnopbaseUChar *key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
        key = NnopbaseGenPlaceHolderKey(args, key);
    } else {
        ge::DataType dataType = scalar->GetDataType();
        if (dtype != ge::DT_UNDEFINED) {
            dataType = dtype;
        } else if ((srcIndex != -1) && (static_cast<uint32_t>(srcIndex) < index)) {
            dataType = args->inputs.extTensors[args->inputs.paramDescs.instances[srcIndex].startIndex]
                    .rt2Tensor.GetDataType();
        }
        (void)NnopbaseAddScalarInfo(args, dataType);
    }
}

static NnopbaseUChar *NnopbaseAddScalarListInfo(NnopbaseExecutorArgs *args, const ge::DataType dtype, const uint64_t size)
{
    size_t len = nnopbase::NNOPBASE_BASE_BYTES + nnopbase::NNOPBASE_SHAPE_BYTES;
    if (args->remainKeyLen < len) {
        args->remainKeyLen += NNOPBASE_MAX_ARGS_KEY_LEN;
        args->inputKey.resize(args->inputKey.size() + NNOPBASE_MAX_ARGS_KEY_LEN);
    }
    NnopbaseUChar *key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data()) + args->keyLen;
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(dtype));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(ge::FORMAT_ND));
    key = NnopbaseAppend4Byte(key, static_cast<uint32_t>(1U));
    key = NnopbaseAppend8Byte(key, static_cast<uint64_t>(size));
    args->remainKeyLen -= nnopbase::NNOPBASE_BASE_BYTES + nnopbase::NNOPBASE_SHAPE_BYTES;
    args->keyLen += nnopbase::NNOPBASE_BASE_BYTES + nnopbase::NNOPBASE_SHAPE_BYTES;
    return key;
}

void NnopbaseExecutorGenScalarListKey(NnopbaseExecutor *executor, const aclScalarList *scalarList, const uint32_t index,
    const int32_t srcIndex, const ge::DataType dtype)
{
    NnopbaseExecutorArgs *args = &executor->ownArgs;
    NnopbaseUChar *key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    if ((scalarList == nullptr) || (scalarList->Size() == 0U)) {
        key = NnopbaseGenPlaceHolderKey(args, key);
    } else {
        ge::DataType dataType = (*scalarList)[0]->GetDataType();
        if (dtype != ge::DT_UNDEFINED) {
            dataType = dtype;
        } else if ((srcIndex != -1) && (static_cast<uint32_t>(srcIndex) < index)) {
            dataType = args->inputs.extTensors[args->inputs.paramDescs.instances[srcIndex].startIndex]
                           .rt2Tensor.GetDataType();
        }
        (void)NnopbaseAddScalarListInfo(args, dataType, scalarList->Size());
    }
}