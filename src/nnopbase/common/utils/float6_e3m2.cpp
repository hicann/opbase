/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/float6_e3m2.h"

#include <cmath>
#include <cstdint>

#include "float_common_utils.h"

namespace {

// ============= E3M2 format constants (OCP MX MXFP6) =============
constexpr int EXP_BITS = 3;
constexpr int MAN_BITS = 2;
constexpr int EXP_BIAS = 3;
constexpr int SIGN_SHIFT = 5;
constexpr uint8_t SIGN_MASK = 0x20;      // 1 000 00 (bit 5)
constexpr uint8_t EXP_MASK = 0x1C;       // 0 111 00 (bits 4-2)
constexpr uint8_t MAN_MASK = 0x03;       // 0 000 11 (bits 1-0)
constexpr uint8_t ABS_VALUE_MASK = 0x1F; // 0 11111 (exclude sign bit)
constexpr op::internal::SmallFloatFormat FLOAT6_E3M2_FORMAT{EXP_BITS, MAN_BITS, EXP_BIAS,
                                                            op::internal::SmallFloatSpecialEncoding::NONE};
} // anonymous namespace

namespace op {

Float6E3M2::Float6E3M2(float v) : value(FloatToFloat6E3M2(v).value) {}

Float6E3M2::operator float() const { return Float6E3M2ToFloat(*this); }

Float6E3M2::operator double() const { return static_cast<double>(Float6E3M2ToFloat(*this)); }

bool Float6E3M2::IsZero() const
{
    return (value & ABS_VALUE_MASK) == 0; // 排除符号位
}

bool Float6E3M2::IsNaN() const
{
    // OCP MX E3M2 has no NaN representation
    // All bit patterns represent valid numbers
    return false;
}

bool Float6E3M2::IsInf() const
{
    // E3M2 has no infinity representation
    return false;
}

float Float6E3M2::Float6E3M2ToFloat(Float6E3M2 fp6)
{
    if (fp6.IsZero()) {
        return (fp6.value & SIGN_MASK) != 0 ? -0.0f : 0.0f;
    }

    // OCP MX E3M2 has no NaN, all bit patterns are valid numbers

    uint32_t sign = (fp6.value & SIGN_MASK) >> SIGN_SHIFT;
    uint32_t exp = (fp6.value & EXP_MASK) >> MAN_BITS;
    uint32_t man = fp6.value & MAN_MASK;

    // FP32: 1 sign, 8 exp (bias 127), 23 man
    // FP6 E3M2: 1 sign, 3 exp (bias 3), 2 man

    int32_t fp32Exp = static_cast<int32_t>(exp) - EXP_BIAS + FP32_EXP_BIAS_VAL;
    uint32_t fp32Man = man << (FP32_MAN_LEN_VAL - MAN_BITS);

    if (exp == 0) {
        // Denormalized number
        if (man != 0) {
            // Normalize: shift mantissa left until leading 1
            int shift = 0;
            while ((man & (1 << MAN_BITS)) == 0) {
                man <<= 1;
                shift++;
            }
            man &= MAN_MASK; // Remove implicit leading 1
            fp32Exp = 1 - EXP_BIAS + FP32_EXP_BIAS_VAL - shift;
            fp32Man = man << (FP32_MAN_LEN_VAL - MAN_BITS);
        }
    } else {
        // Normal number - add implicit leading 1
        fp32Man |= FP32_IMPLICIT_1_VAL;
    }

    FP32 result;
    result.u = (sign << FP32_SIGN_SHIFT_VAL) | ((fp32Exp & FP32_EXP_MASK_8BIT) << FP32_MAN_LEN_VAL) |
               (fp32Man & FP32_MAN_MASK_23BIT);
    return result.f;
}

Float6E3M2 Float6E3M2::FloatToFloat6E3M2(float f)
{
    // OCP MX E3M2 has no NaN, clamp NaN to positive max value (28.0)
    // NaN has no meaningful sign, so always clamp to +28.0
    if (std::isnan(f)) {
        return Float6E3M2(HIGHEST_VALUE, FromBits());
    }

    FP32 fp32;
    fp32.f = f;
    const uint32_t sign = (fp32.u >> FP32_SIGN_SHIFT_VAL) & 1U;

    if (std::isinf(f)) {
        // E3M2 has no infinity, clamp to max value (preserving sign)
        return Float6E3M2(static_cast<uint8_t>((sign << SIGN_SHIFT) | HIGHEST_VALUE), FromBits());
    }

    // 有限值（含 ±0）：按 roundTiesToEven 统一舍入，溢出钳制到最大幅值
    const uint8_t result = internal::ConvertFiniteFp32ToSmallFloat(fp32.u, FLOAT6_E3M2_FORMAT,
                                                                   internal::SmallFloatOverflowAction::SATURATE_MAX);
    return Float6E3M2(result, FromBits());
}

} // namespace op
