/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bfloat16.cpp
 * \brief BFloat16 type conversion operators to float8/float6/float4 types
 */
#include "opdev/bfloat16.h"
#include "opdev/float8_e4m3fn.h"
#include "opdev/float8_e5m2.h"
#include "opdev/float8_e8m0.h"
#include "opdev/float6_e3m2.h"
#include "opdev/float6_e2m3.h"
#include "opdev/float4_e2m1.h"
#include "opdev/float4_e1m2.h"
#include "opdev/hifloat4.h"

namespace op {

// Conversion operators from bfloat16 to float8/float6/float4 types
bfloat16::operator Float8E4M3FN() const
{
    return Float8E4M3FN(static_cast<float>(*this));
}

bfloat16::operator Float8E5M2() const
{
    return Float8E5M2(static_cast<float>(*this));
}

bfloat16::operator Float8E8M0() const
{
    return Float8E8M0(static_cast<float>(*this));
}

bfloat16::operator Float6E3M2() const
{
    return Float6E3M2(static_cast<float>(*this));
}

bfloat16::operator Float6E2M3() const
{
    return Float6E2M3(static_cast<float>(*this));
}

bfloat16::operator Float4E2M1() const
{
    return Float4E2M1(static_cast<float>(*this));
}

bfloat16::operator Float4E1M2() const
{
    return Float4E1M2(static_cast<float>(*this));
}

bfloat16::operator HiFloat4() const
{
    return HiFloat4(static_cast<float>(*this));
}

} // namespace op
