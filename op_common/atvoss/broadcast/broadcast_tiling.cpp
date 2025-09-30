/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_tiling.cpp
 * \brief atvoss broadcast template tiling 
 */

#include <algorithm>
#include "op_common/atvoss/broadcast/broadcast_tiling.h"

namespace Ops {
namespace Base {
static constexpr int64_t BROADCAST_REPEAT_BYTES = 256;
static constexpr uint64_t BROADCAST_COMPUTE_KEY_OFFSET = 1000000;
static constexpr uint64_t BROADCAST_COMPUTE_KEY = 1;
static constexpr uint64_t MAX_NDDMA_DIM = 5;
static constexpr uint64_t SCHEDULE_KEY_1 = 100;
static constexpr uint64_t SCHEDULE_KEY_2 = 1100;
static constexpr int64_t DEFAULT_SCHEMODE = 1;

static uint64_t BroadcastGetComputeKey()
{
    return BROADCAST_COMPUTE_KEY;
}

static uint64_t BroadcastGetScheduleKey(uint32_t axisInsideUB)
{
    if (axisInsideUB <= MAX_NDDMA_DIM) {
        return SCHEDULE_KEY_1;
    }
    return SCHEDULE_KEY_2;
}

static int64_t BroadcastGetMaxElemNum(int64_t ubSize, const BroadcastComputeParams &computeParams)
{
    int64_t minDtypeBits = computeParams.minDtypeBits;
    int64_t extraSize = computeParams.extraSize[0];
    int64_t bufferDivisor = computeParams.bufferDivisor[0];
    int64_t maxElemNums = (ubSize - extraSize) * BROADCAST_BITS_NUM / bufferDivisor;
    int64_t alignFactor = BROADCAST_REPEAT_BYTES * BROADCAST_BITS_NUM / minDtypeBits;
    int64_t maxElemNumAlign = maxElemNums / alignFactor * alignFactor;
    return maxElemNumAlign;
}

/**
 *  合轴逻辑
 * @param inShapes 输入shape
 * @param outShapes 输出shape
 * @param dims 合轴后轴大小
 * @param strides 合轴后stride大小
 * @return
*/
ge::graphStatus DimensionCollapse(const std::vector<gert::Shape> &inShapes, const gert::Shape &outShapes,
    std::vector<std::vector<int64_t>> &dims, std::vector<std::vector<int64_t>> &strides)
{
    // 获取输出shape的轴数量，并封装输出shape
    uint64_t maxDim = outShapes.GetDimNum();
    std::vector<int64_t> outputShapes;
    for (uint64_t i = 0; i < outShapes.GetDimNum(); i++) {
        outputShapes.push_back(outShapes.GetDim(i));
    }
    
    // 获取输入shape的轴数量，并封装输入shape
    // 对维度不足输出shape的输入进行补维操作
    std::vector<std::vector<int64_t>> inputShapes;
    for (uint64_t i = 0; i < inShapes.size(); i++) {
        uint64_t inputDim = inShapes[i].GetDimNum();
        if (maxDim < inputDim) {
            OP_LOGE("BroadcastTiling", "The %lu input's dim num is not same with output's dim num", i);
            return ge::GRAPH_FAILED;
        }
        int64_t diff = maxDim - inputDim;
        std::vector<int64_t> tmp(diff, 1);
        for (uint64_t j = 0; j < inputDim; j++) {
            tmp.push_back(inShapes[i].GetDim(j));
        }
        inputShapes.push_back(tmp);
    }

    // 设置默认flag大小跟第一个输入大小一致，并初始化为0
    std::vector<int64_t> flags(inputShapes[0].size(), 0);
    // 遍历所有输入的所有维度，校验轴的合法性。
    // 将某个输入的brc轴对应的二进制位置设置为1
    for (uint64_t i = 0; i < inputShapes[0].size(); ++i) {
        int64_t flag = 0;
        for (uint64_t j = 0; j < inputShapes.size(); ++j) {
            flag <<= 1;
            if (inputShapes[j][i] != 1 && inputShapes[j][i] != outputShapes[i]) {
                OP_LOGE("BroadcastTiling", "The %lu input's dim index(%lu) is not same with out, and not 1", j, i);
                return ge::GRAPH_FAILED;
            }
            if (inputShapes[j][i] <= 0) {
                OP_LOGE("BroadcastTiling", "The %lu input's dim index(%lu) must be a positive number", j, i);
                return ge::GRAPH_FAILED;
            }
            if (inputShapes[j][i] == 1) {
                flag++;
            }
        }
        flags[i] = flag;
    }

    // 做输入shape合轴逻辑
    // 遍历所有的输入，遍历所有的轴
    // 1.如果所有输入的相邻两根轴大小都一样，则可以合轴。
    // 2.如果所有输入输出都是1，则可以合轴
    int64_t target = (1 << inputShapes.size()) - 1;
    for (uint64_t i = 0; i < inputShapes.size(); i++) {
        int64_t prevValue = inputShapes[i][0];
        int64_t prevFlag = flags[0];
        std::vector<int64_t> tmp{prevValue};
        for (uint64_t j = 1; j < inputShapes[i].size(); j++) {
            int64_t curValue = inputShapes[i][j];
            int64_t curFlag = flags[j];
            bool isValid = (prevFlag == curFlag) || (prevFlag == target && outputShapes[j - 1] == 1);
            if (isValid) {
                int64_t product = curValue * tmp.back();
                tmp.pop_back();
                tmp.push_back(product);
                prevFlag = curFlag;
                continue;
            }
            if (curFlag == target && outputShapes[j] == 1) {
                continue;
            }
            prevFlag = curFlag;
            tmp.push_back(curValue);
        }
        dims.push_back(tmp);
    }

    // 做输出shape合轴逻辑
    int64_t prevValue = outputShapes[0];
    int64_t prevFlag = flags[0];
    std::vector<int64_t> outputDims{prevValue};
    for (uint64_t j = 1; j < outputShapes.size(); j++) {
        int64_t curValue = outputShapes[j];
        int64_t curFlag = flags[j];
        bool isValid = (prevFlag == curFlag) || (prevFlag == target && outputShapes[j - 1] == 1);
        if (isValid) {
            int64_t product = curValue * outputDims.back();
            outputDims.pop_back();
            outputDims.push_back(product);
            prevFlag = curFlag;
            continue;
        }
        if (curFlag == target && outputShapes[j] == 1) {
            continue;
        }
        prevFlag = curFlag;
        outputDims.push_back(curValue);
    }
    dims.push_back(outputDims);

    // 计算stride信息
    for (uint64_t i = 0; i < dims.size(); i++) {
        std::vector<int64_t> tmp;
        int64_t base = 1;
        for (int64_t j = dims[i].size() - 1; j >= 0; j--) {
            if (dims[i][j] == 1 && i != dims.size() - 1 && dims[dims.size() - 1][j] != 1) {
                tmp.push_back(0);
            } else {
                tmp.push_back(base);
                base *= dims[i][j];
            }
        }
        std::reverse(tmp.begin(), tmp.end());
        strides.push_back(tmp);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DoBrodcastTiling(
    const BroadcastTilingParams &broadcastTilingParams, BroadcastTilingData &broadcastTilingData)
{
    uint64_t computeKey = BroadcastGetComputeKey();
    auto iter = broadcastTilingParams.computeMap.find(computeKey);
    BroadcastComputeParams computeParams;
    if (iter != broadcastTilingParams.computeMap.end()) {
        computeParams = iter->second;
    } else {
        OP_LOGE("BroadcastTiling", "can not find computeKey");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(broadcastTilingParams.ubSize < computeParams.extraSize[0],
                OP_LOGE("BroadcastTiling", "ubSize is smaller than extra size."), return ge::GRAPH_FAILED);

     // 获取最大存活空间大小
    uint64_t maxElemNum = BroadcastGetMaxElemNum(broadcastTilingParams.ubSize, computeParams);
    OP_CHECK_IF(maxElemNum == 0, OP_LOGE("BroadcastTiling", "maxElemNum can not be 0"), return ge::GRAPH_FAILED);

    // 做ub切分
    uint64_t curProduct = 1;
    uint64_t ubSplitAxes = 0;
    bool flag = true;
    for (int64_t i = broadcastTilingData.dims.back().size() - 1; i >= 0; i--) {
        curProduct *= broadcastTilingData.dims.back()[i];
        if (curProduct > maxElemNum) {
            curProduct = curProduct / broadcastTilingData.dims.back()[i];
            ubSplitAxes = static_cast<uint64_t>(i);
            flag = false;
            break;
        }
    }

    if (flag) {
        curProduct = curProduct / broadcastTilingData.dims.back()[0];
    }

    uint32_t ubFormer = 0;
    if (broadcastTilingData.dims.back().size() == 1) {
        ubFormer = maxElemNum;
    } else {
        ubFormer = maxElemNum / curProduct;
    }
    uint64_t ubOuter = (broadcastTilingData.dims.back()[ubSplitAxes] + ubFormer - 1) / ubFormer;
    uint64_t ubTail = broadcastTilingData.dims.back()[ubSplitAxes] - (ubOuter - 1) * ubFormer;
    broadcastTilingData.ubSplitAxis = ubSplitAxes;
    broadcastTilingData.ubFormer = ubFormer;
    broadcastTilingData.ubOuter = ubOuter;
    broadcastTilingData.ubTail = ubTail;

    // 计算ub外轴乘积
    uint64_t fusedProduct = ubOuter;
    for (uint64_t i = 0; i < ubSplitAxes; i++) {
        fusedProduct *= broadcastTilingData.dims.back()[i];
    }

    OP_CHECK_IF(broadcastTilingParams.coreNum == 0, OP_LOGE("BroadcastTiling", "coreNum can not be 0"),
                return ge::GRAPH_FAILED);

    // 对剩余外轴做block切分
    uint64_t blockFormer = (fusedProduct + broadcastTilingParams.coreNum - 1) / broadcastTilingParams.coreNum;
    uint64_t blockNum = (fusedProduct + blockFormer - 1) / blockFormer;
    uint64_t blockTail = fusedProduct - (blockNum - 1) * blockFormer;
    uint64_t dimProductBeforeUbInner = fusedProduct;
    broadcastTilingData.blockFormer = blockFormer;
    broadcastTilingData.blockNum = blockNum;
    broadcastTilingData.blockTail = blockTail;
    broadcastTilingData.dimProductBeforeUbInner = dimProductBeforeUbInner;
    broadcastTilingData.elemNum = maxElemNum;

    uint64_t scheduleKey = BroadcastGetScheduleKey(broadcastTilingData.shapeLen - broadcastTilingData.ubSplitAxis);
    broadcastTilingData.innerKey = computeKey * BROADCAST_COMPUTE_KEY_OFFSET + scheduleKey;
    return ge::GRAPH_SUCCESS;
}

/**
 *  合轴逻辑
 * @param broadcastTilingParams tiling参数
 * @param broadcastTilingData 临时tilingData缓存
 * 
 * @return
*/
ge::graphStatus DoDimensionCollapse(
    const BroadcastTilingParams &broadcastTilingParams, BroadcastTilingData &broadcastTilingData)
{
    std::vector<std::vector<int64_t>> dims;
    std::vector<std::vector<int64_t>> strides;
    ge::graphStatus status =
        DimensionCollapse(broadcastTilingParams.inShape, broadcastTilingParams.outShape, dims, strides);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    broadcastTilingData.shapeLen = dims.back().size();
    OP_CHECK_IF(broadcastTilingData.shapeLen > static_cast<int64_t>(BROADCAST_MAX_DIMS),
                OP_LOGE("BroadcastTiling", "broadcast can't support dim size greater than 8."),
                return ge::GRAPH_FAILED);

    broadcastTilingData.dims = dims;
    broadcastTilingData.strides = strides;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BroadcastTiling(
    const BroadcastTilingParams &broadcastTilingParams, BroadcastTilingData &broadcastTilingData)
{
    auto status = DoDimensionCollapse(broadcastTilingParams, broadcastTilingData);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE("BroadcastTiling", "dimension collapse failed");
        return ge::GRAPH_FAILED;
    }

    status = DoBrodcastTiling(broadcastTilingParams, broadcastTilingData);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE("BroadcastTiling", "inner broadcast tiling failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace Base
} // namespace Ops