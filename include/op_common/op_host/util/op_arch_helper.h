/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file op_arch_helper.h
 * \brief add aicore config by npu arch
 */

#ifndef OP_COMMON_OP_HOST_UTIL_OP_ARCH_HELPER_H
#define OP_COMMON_OP_HOST_UTIL_OP_ARCH_HELPER_H

#include "register/op_def_registry.h"

#include "opbase_export.h"
#include "platform_util.h"

namespace opbase {
class OPBASE_API ArchConfigHelper {
public:
    static void AddConfigByArch(ops::OpAICoreDef& aicore, NpuArch arch, ops::OpAICoreConfig& aicoreConfig);

    static void AddConfigByArch(ops::OpAICoreDef& aicore, NpuArch arch);
};
} // namespace opbase
#endif // OP_COMMON_OP_HOST_UTIL_OP_ARCH_HELPER_H
