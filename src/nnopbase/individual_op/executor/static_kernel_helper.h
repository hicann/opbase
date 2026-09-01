/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STATIC_KERNEL_HELPER_H_
#define STATIC_KERNEL_HELPER_H_

#include "individual_op_internal.h"
#include "indv_bininfo.h"
#include "indv_executor.h"

namespace Indv {

class StaticKernelHelper {
public:
    static const NnopbaseChar* FindStaticKernelPath(const aclTensor* tensors[], const NnopbaseAttrAddr* attrs[],
                                                    const int64_t valueDepend[],
                                                    const NnopbaseStaticTensorNumInfo* const tensorNumInfo,
                                                    const NnopbaseStaticRuntimeInfo* const staticRuntimeInfo);

    static NnopbaseBinInfo* FindStaticBinInfo(NnopbaseExecutor* const executor, BinInfoKey& binInfoKey);
};

} // namespace Indv

#endif
