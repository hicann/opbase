/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __NNOPBASE_ARGS_H_
#define __NNOPBASE_ARGS_H_

#include <vector>
#include "acl/acl_rt.h"
#include "utils/indv_base.h"

using NnopbaseStaticTensorNumInfo = struct {
    int64_t numTensors;
    int64_t numDynamic;
    int64_t numAttrs;
    int64_t numValueDepend;
};

using NnopbaseRTArgsExt = struct {
    void *args;                             // args host mem addr
    aclrtPlaceHolderInfo *hostInputInfoPtr;  // nullptr means no host mem input
    uint32_t argsSize;                      // input + output + tiling addr size + tiling data size + host mem
    uint32_t tilingAddrOffset;              // tiling addr offset
    uint32_t tilingDataOffset;              // tiling data offset
    uint8_t hasTiling;                      // if has tiling: 0 means no tiling
    uint16_t hostInputInfoNum;              // hostInputInfo num
};

static inline uint16_t NnopbaseGetRTSPlaceHolderNum(NnopbaseRTArgsExt *argsExt) {
    if (argsExt == nullptr) {
        return 0;
    }

    if (argsExt->hasTiling != 0) {
        return argsExt->hostInputInfoNum + 1;
    }

    return argsExt->hostInputInfoNum;
}

static inline std::vector<aclrtPlaceHolderInfo> NnopbaseGetRTSPlaceHolder(NnopbaseRTArgsExt* argsExt) {
    std::vector<aclrtPlaceHolderInfo> hostInputInfoPtr;
    if (argsExt->hasTiling != 0) {
        hostInputInfoPtr.push_back({argsExt->tilingAddrOffset, argsExt->tilingDataOffset});
    }
    for (int i = 0; i < argsExt->hostInputInfoNum; i++) {
        aclrtPlaceHolderInfo *ptr = argsExt->hostInputInfoPtr + i;
        hostInputInfoPtr.push_back({ptr->addrOffset, ptr->dataOffset});
    }

    return hostInputInfoPtr;
}

static inline void NnopbaseGetIrIndex(const NnopbaseParamDesc &paramDesc, const size_t index,
                                      size_t &irIndex, size_t &relativeIndex)
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

#endif