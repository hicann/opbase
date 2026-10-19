/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/float8_e4m3fn.h"

#include <cmath>
#include <cstdint>

#include "float_common_utils.h"

namespace {

// ============= E4M3FN format constants =============
constexpr int EXP_BITS = 4;
constexpr int MAN_BITS = 3;
constexpr int EXP_BIAS = 7;
constexpr int SIGN_SHIFT = 7;
constexpr uint8_t SIGN_MASK = 0x80;      // 1 0000 000
constexpr uint8_t EXP_MASK = 0x78;       // 0 1111 000
constexpr uint8_t MAN_MASK = 0x07;       // 0 0000 111
constexpr uint8_t ABS_VALUE_MASK = 0x7F; // 0 1111 111 (exclude sign bit)
constexpr uint8_t NEG_NAN_VALUE = 0xFF;  // 1 1111 111

constexpr op::internal::SmallFloatFormat FLOAT8_E4M3FN_FORMAT{EXP_BITS, MAN_BITS, EXP_BIAS,
                                                              op::internal::SmallFloatSpecialEncoding::NAN_ONLY};
} // anonymous namespace

namespace op {

Float8E4M3FN::Float8E4M3FN(float v) : value(FloatToFloat8E4M3FN(v).value) {}

Float8E4M3FN::operator float() const { return Float8E4M3FNToFloat(*this); }

Float8E4M3FN::operator double() const { return static_cast<double>(Float8E4M3FNToFloat(*this)); }

bool Float8E4M3FN::IsZero() const { return (value & ABS_VALUE_MASK) == 0; }

bool Float8E4M3FN::IsNaN() const { return value == NAN_VALUE || value == NEG_NAN_VALUE; }

bool Float8E4M3FN::IsInf() const
{
    // E4M3FN has no infinity representation
    return false;
}

float Float8E4M3FN::Float8E4M3FNToFloat(Float8E4M3FN fp8)
{
    if (fp8.IsZero()) {
        return (fp8.value & SIGN_MASK) != 0 ? -0.0f : 0.0f;
    }

    if (fp8.IsNaN()) {
        return std::nanf("");
    }

    uint32_t sign = (fp8.value & SIGN_MASK) >> SIGN_SHIFT;
    uint32_t exp = (fp8.value & EXP_MASK) >> MAN_BITS;
    uint32_t man = fp8.value & MAN_MASK;

    // FP32: 1 sign, 8 exp (bias 127), 23 man
    // FP8 E4M3FN: 1 sign, 4 exp (bias 7), 3 man

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

Float8E4M3FN Float8E4M3FN::FloatToFloat8E4M3FN(float f)
{
    FP32 fp32;
    fp32.f = f;
    const uint32_t sign = (fp32.u >> FP32_SIGN_SHIFT_VAL) & 1U;
    const uint32_t exp = (fp32.u >> FP32_MAN_LEN_VAL) & FP32_EXP_MASK_8BIT;

    // E4M3FN 无无穷大编码：NaN 保持 NaN（保号）；±Inf 按饱和模式钳制到 ±448
    if (exp == FP32_EXP_MAX_VAL) {
        const uint32_t signBits = sign << SIGN_SHIFT;
        if ((fp32.u & FP32_MAN_MASK_23BIT) != 0U) {
            return Float8E4M3FN(static_cast<uint8_t>(signBits | NAN_VALUE), FromBits());
        }
        return Float8E4M3FN(static_cast<uint8_t>(signBits | HIGHEST_VALUE), FromBits());
    }

    // 舍入结果落进 NaN 槽时同样按溢出饱和到最大幅值
    const uint8_t result = internal::ConvertFiniteFp32ToSmallFloat(fp32.u, FLOAT8_E4M3FN_FORMAT,
                                                                   internal::SmallFloatOverflowAction::SATURATE_MAX);
    return Float8E4M3FN(result, FromBits());
}

} // namespace op
