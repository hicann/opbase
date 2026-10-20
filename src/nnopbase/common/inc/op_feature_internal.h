/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_INC_OP_FEATURE_INTERNAL_H_
#define OP_API_OP_API_COMMON_INC_OP_FEATURE_INTERNAL_H_

#ifndef PRODUCT_SIDE_IS_DEVICE
#include "version/runtime_version.h"
#endif

#include "aclnn/acl_meta.h"

#define PKG_VERSION_NUM_9_2_0 90200000

#if !defined(PRODUCT_SIDE_IS_DEVICE)
#if defined(RUNTIME_VERSION_NUM)
#if (RUNTIME_VERSION_NUM >= PKG_VERSION_NUM_9_2_0)
#define NNOPBASE_RUNTIME_VERSION_SUPPORT_PCIE_THROUGH 1
#define NNOPBASE_RUNTIME_VERSION_SUPPORT_OP_PROFILING 1
#endif
#endif
#endif

namespace op {
namespace internal {

// 全局只调一次（内部 std::call_once）；
aclnnStatus InitPcieThroughInfo();

bool IsPcieThroughEnabled();

bool IsTensorAddrInPcieRange(const void* const addr);

} // namespace internal
} // namespace op

#endif // OP_API_OP_API_COMMON_INC_OP_FEATURE_INTERNAL_H_
