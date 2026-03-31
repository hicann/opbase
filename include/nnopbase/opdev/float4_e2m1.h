/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_FLOAT4_E2M1_H
#define OP_API_FLOAT4_E2M1_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>

namespace op {

/**
 * @brief FP4 E2M1 floating-point format (OCP MX MXFP4)
 *
 * Format: 1 sign bit, 2 exponent bits, 1 mantissa bit
 *         +---+----+-+
 *         | S | EE |M|
 *         +---+----+-+
 *
 * Follows OCP Microscaling Formats (MX) v1.0 specification.
 *
 * - Bias: 1
 * - Max normal value: 6.0 (0x7 = 0 11 1, exp=3, man=1 -> 2^2 * 1.5 = 6.0)
 * - Min normal value: -6.0 (0xF = 1 11 1)
 * - Min positive normal: 2^0 = 1.0 (0x2 = 0 01 0)
 * - Min positive denorm: 2^(1-bias) * (man/2) = 2^0 * 0.5 = 0.5 (0x1 = 0 00 1)
 * - No NaN representation (all bit patterns are valid numbers)
 * - No infinity representation
 */
struct Float4E2M1 {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits()
    {
        return FromBitsTag();
    }

    uint8_t value;

    // ============= E2M1 format constants =============
    // Bit layout: bit3=sign, bits2-1=exp, bit0=man
    static constexpr int EXP_BITS = 2;
    static constexpr int MAN_BITS = 1;
    static constexpr int EXP_BIAS = 1;
    static constexpr int SIGN_SHIFT = 3;
    static constexpr uint8_t SIGN_MASK = 0x08;          // 1 00 0 (bit 3)
    static constexpr uint8_t EXP_MASK = 0x06;           // 0 11 0 (bits 2-1)
    static constexpr uint8_t MAN_MASK = 0x01;           // 0 00 1 (bit 0)
    static constexpr uint8_t MAX_EXP = 0x03;            // 11
    static constexpr uint8_t ABS_VALUE_MASK = 0x07;     // 0 111 (exclude sign bit)
    static constexpr uint8_t HIGHEST_VALUE = 0x07;      // 0 11 1 = 6.0 (max value)
    static constexpr uint8_t LOWEST_VALUE = 0x0F;       // 1 11 1 = -6.0 (min value)
    static constexpr uint8_t MIN_POS_NORMAL_VALUE = 0x02; // 0 01 0 = 1.0 (min positive normal)
    static constexpr uint8_t BITS_MASK = 0x0F;          // 4-bit effective mask
    static constexpr uint8_t EPSILON_VALUE = 0x01;      // 0 00 1 = 0.5 (epsilon)

    // ============= FP32 format constants =============
    static constexpr int FP32_EXP_BIAS_VAL = 127;
    static constexpr int FP32_MAN_LEN_VAL = 23;
    static constexpr int FP32_SIGN_SHIFT_VAL = 31;
    static constexpr uint32_t FP32_EXP_MASK_8BIT = 0xFF;
    static constexpr uint32_t FP32_MAN_MASK_23BIT = 0x7FFFFF;
    static constexpr uint32_t FP32_IMPLICIT_1_VAL = 0x800000;

    // Default constructor - initialize to zero
    Float4E2M1() : value(0) {}

    constexpr Float4E2M1(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits & BITS_MASK)
    {
    }

    Float4E2M1(float v)
    {
        value = FloatToFloat4E2M1(v).value;
    }

    operator float() const
    {
        return Float4E2M1ToFloat(*this);
    }

    operator double() const
    {
        return static_cast<double>(Float4E2M1ToFloat(*this));
    }

    static constexpr Float4E2M1 Epsilon()
    {
        // 2^-1 = 0.5 (the difference between 1.0 and next representable value)
        return Float4E2M1(EPSILON_VALUE, FromBits());  // 0 00 1 = denorm 0.5
    }

    static constexpr Float4E2M1 Highest()
    {
        // exp=3, man=1 -> 2^2 * 1.5 = 6.0 (0 11 1 = 0x7)
        // OCP MX E2M1 has no NaN, all bit patterns are valid
        return Float4E2M1(HIGHEST_VALUE, FromBits());
    }

    static constexpr Float4E2M1 Lowest()
    {
        // 1 11 1 = -6.0 (exp=3, man=1)
        return Float4E2M1(LOWEST_VALUE, FromBits());
    }

    static constexpr Float4E2M1 MinPositiveNormal()
    {
        // 2^0 = 1.0 (0 01 0)
        return Float4E2M1(MIN_POS_NORMAL_VALUE, FromBits());
    }

    bool IsZero() const
    {
        return (value & ABS_VALUE_MASK) == 0;
    }

    bool IsNaN() const
    {
        // OCP MX E2M1 has no NaN representation
        // All bit patterns represent valid numbers
        return false;
    }

    bool IsInf() const
    {
        // E2M1 has no infinity representation
        return false;
    }

private:
    union FP32 {
        uint32_t u;
        float f;
    };

    static float Float4E2M1ToFloat(Float4E2M1 fp4)
    {
        if (fp4.IsZero()) {
            return (fp4.value & SIGN_MASK) ? -0.0f : 0.0f;
        }

        // OCP MX E2M1 has no NaN, all bit patterns are valid numbers

        uint32_t sign = (fp4.value & SIGN_MASK) >> SIGN_SHIFT;
        uint32_t exp = (fp4.value & EXP_MASK) >> MAN_BITS;
        uint32_t man = fp4.value & MAN_MASK;

        // FP32: 1 sign, 8 exp (bias 127), 23 man
        // FP4 E2M1: 1 sign, 2 exp (bias 1), 1 man

        int32_t fp32Exp = static_cast<int32_t>(exp) - EXP_BIAS + FP32_EXP_BIAS_VAL;
        uint32_t fp32Man = man << (FP32_MAN_LEN_VAL - MAN_BITS);

        if (exp == 0) {
            // Denormalized number
            if (man != 0) {
                // OCP MX E2M1 denorm: value = 2^(1-bias) × (man/2) = 2^0 × 0.5 = 0.5
                // FP32 表示: exp=126 对应 2^(-1), 配合隐含的 1.0 得到 0.5
                // man=1 -> 0.5
                fp32Exp = 1 - EXP_BIAS + FP32_EXP_BIAS_VAL - 1;  // = 126, 对应 2^(-1)
                fp32Man = 0;  // FP32 隐含的 1.0 正好表示 E2M1 的 man/2 = 0.5
            }
        } else {
            // Normal number - add implicit leading 1
            fp32Man |= FP32_IMPLICIT_1_VAL;
        }

        FP32 result;
        result.u = (sign << FP32_SIGN_SHIFT_VAL) | ((fp32Exp & FP32_EXP_MASK_8BIT) << FP32_MAN_LEN_VAL) | (fp32Man & FP32_MAN_MASK_23BIT);
        return result.f;
    }

    static Float4E2M1 FloatToFloat4E2M1(float f)
    {
        // OCP MX E2M1 has no NaN, clamp NaN to positive max value (6.0)
        // NaN has no meaningful sign, so always clamp to +6.0
        if (std::isnan(f)) {
            // 0x07 = 0 11 1 = 2^2 * 1.5 = 6.0
            return Float4E2M1(HIGHEST_VALUE, FromBits());
        }

        FP32 fp32;
        fp32.f = f;
        uint32_t sign = (fp32.u >> FP32_SIGN_SHIFT_VAL) & 1;
        uint32_t exp = (fp32.u >> FP32_MAN_LEN_VAL) & FP32_EXP_MASK_8BIT;
        uint32_t man = fp32.u & FP32_MAN_MASK_23BIT;

        if (exp == 0 && man == 0) {
            // Zero
            return Float4E2M1(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
        }

        if (std::isinf(f)) {
            // E2M1 has no infinity, clamp to max value
            // 0x07 = 0 11 1 = 2^2 * 1.5 = 6.0, 0x0F = -6.0
            return Float4E2M1(static_cast<uint8_t>((sign << SIGN_SHIFT) | HIGHEST_VALUE), FromBits());
        }

        // Calculate E2M1 exponent
        int32_t fp4Exp;
        uint32_t fp4Man;

        if (exp == 0) {
            // Denormalized FP32 input
            fp4Exp = 1 - FP32_EXP_BIAS_VAL + EXP_BIAS;
            while ((man & FP32_IMPLICIT_1_VAL) == 0) {
                man <<= 1;
                fp4Exp--;
            }
            man &= FP32_MAN_MASK_23BIT;
        } else {
            fp4Exp = static_cast<int32_t>(exp) - FP32_EXP_BIAS_VAL + EXP_BIAS;
        }

        // Round mantissa from 23 bits to 1 bit with round-to-nearest-even
        int mantissaShift = FP32_MAN_LEN_VAL - MAN_BITS;
        uint32_t manRoundBit = (man >> (mantissaShift - 1)) & 1;
        uint32_t manStickyBits = (man & ((1 << (mantissaShift - 1)) - 1)) ? 1 : 0;
        uint32_t manLsb = (man >> mantissaShift) & 1;

        fp4Man = man >> mantissaShift;
        if (manRoundBit && (manStickyBits || manLsb)) {
            fp4Man++;
            if (fp4Man > MAN_MASK) {
                fp4Man = 0;
                fp4Exp++;
            }
        }

        // Handle overflow/underflow
        if (fp4Exp <= 0) {
            if (fp4Exp < -MAN_BITS) {
                return Float4E2M1(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
            }
            fp4Man = (fp4Man | (1 << MAN_BITS)) >> (1 - fp4Exp);
            fp4Exp = 0;
        } else if (fp4Exp >= static_cast<int32_t>(MAX_EXP + 1)) {
            // Overflow - for E2M1, max exp is 3
            // OCP MX E2M1 has no NaN, clamp to max value (preserving sign)
            // 0x07 = 0 11 1 = +6.0, 0x0F = 1 11 1 = -6.0
            return Float4E2M1(static_cast<uint8_t>((sign << SIGN_SHIFT) | HIGHEST_VALUE), FromBits());
        }

        uint8_t result = static_cast<uint8_t>((sign << SIGN_SHIFT) | (fp4Exp << MAN_BITS) | (fp4Man & MAN_MASK));
        return Float4E2M1(result, FromBits());
    }
};

inline std::ostream &operator<<(std::ostream &os, const Float4E2M1 &dt)
{
    os << static_cast<float>(dt);
    return os;
}

inline Float4E2M1 operator+(Float4E2M1 a, Float4E2M1 b)
{
    return Float4E2M1(static_cast<float>(a) + static_cast<float>(b));
}

inline Float4E2M1 operator-(Float4E2M1 a, Float4E2M1 b)
{
    return Float4E2M1(static_cast<float>(a) - static_cast<float>(b));
}

inline Float4E2M1 operator*(Float4E2M1 a, Float4E2M1 b)
{
    return Float4E2M1(static_cast<float>(a) * static_cast<float>(b));
}

inline Float4E2M1 operator/(Float4E2M1 a, Float4E2M1 b)
{
    return Float4E2M1(static_cast<float>(a) / static_cast<float>(b));
}

inline Float4E2M1 operator-(Float4E2M1 a)
{
    a.value ^= Float4E2M1::SIGN_MASK;
    return a;
}

inline bool operator==(Float4E2M1 a, Float4E2M1 b)
{
    return a.value == b.value;
}

inline bool operator!=(Float4E2M1 a, Float4E2M1 b)
{
    return a.value != b.value;
}

inline bool operator<(Float4E2M1 a, Float4E2M1 b)
{
    return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator<=(Float4E2M1 a, Float4E2M1 b)
{
    return static_cast<float>(a) <= static_cast<float>(b);
}

inline bool operator>(Float4E2M1 a, Float4E2M1 b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}

inline bool operator>=(Float4E2M1 a, Float4E2M1 b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}

inline Float4E2M1 &operator+=(Float4E2M1 &a, Float4E2M1 b)
{
    a = a + b;
    return a;
}

inline Float4E2M1 &operator-=(Float4E2M1 &a, Float4E2M1 b)
{
    a = a - b;
    return a;
}

inline Float4E2M1 &operator*=(Float4E2M1 &a, Float4E2M1 b)
{
    a = a * b;
    return a;
}

inline Float4E2M1 &operator/=(Float4E2M1 &a, Float4E2M1 b)
{
    a = a / b;
    return a;
}

inline Float4E2M1 operator++(Float4E2M1 &a)
{
    a += Float4E2M1(1.0f);
    return a;
}

inline Float4E2M1 operator--(Float4E2M1 &a)
{
    a -= Float4E2M1(1.0f);
    return a;
}

inline Float4E2M1 operator++(Float4E2M1 &a, int)
{
    Float4E2M1 originalValue = a;
    ++a;
    return originalValue;
}

inline Float4E2M1 operator--(Float4E2M1 &a, int)
{
    Float4E2M1 originalValue = a;
    --a;
    return originalValue;
}

} // namespace op

namespace std {

template<>
struct hash<op::Float4E2M1> {
    size_t operator()(const op::Float4E2M1 &v) const
    {
        return hash<float>()(static_cast<float>(v));
    }
};

using op::Float4E2M1;

inline bool isinf(const Float4E2M1 &a [[maybe_unused]])
{
    return false;  // E2M1 has no infinity
}

inline bool isnan(const Float4E2M1 &a [[maybe_unused]])
{
    return false;  // OCP MX E2M1 has no NaN
}

inline bool isfinite(const Float4E2M1 &a [[maybe_unused]])
{
    return true;  // All bit patterns are valid finite numbers
}

inline Float4E2M1 abs(const Float4E2M1 &a)
{
    return Float4E2M1(std::abs(static_cast<float>(a)));
}

inline Float4E2M1 exp(const Float4E2M1 &a)
{
    return Float4E2M1(std::exp(static_cast<float>(a)));
}

inline Float4E2M1 log(const Float4E2M1 &a)
{
    return Float4E2M1(std::log(static_cast<float>(a)));
}

inline Float4E2M1 sqrt(const Float4E2M1 &a)
{
    return Float4E2M1(std::sqrt(static_cast<float>(a)));
}

inline Float4E2M1 pow(const Float4E2M1 &a, const Float4E2M1 &b)
{
    return Float4E2M1(std::pow(static_cast<float>(a), static_cast<float>(b)));
}

inline Float4E2M1 sin(const Float4E2M1 &a)
{
    return Float4E2M1(std::sin(static_cast<float>(a)));
}

inline Float4E2M1 cos(const Float4E2M1 &a)
{
    return Float4E2M1(std::cos(static_cast<float>(a)));
}

inline Float4E2M1 tanh(const Float4E2M1 &a)
{
    return Float4E2M1(std::tanh(static_cast<float>(a)));
}

inline Float4E2M1 floor(const Float4E2M1 &a)
{
    return Float4E2M1(std::floor(static_cast<float>(a)));
}

inline Float4E2M1 ceil(const Float4E2M1 &a)
{
    return Float4E2M1(std::ceil(static_cast<float>(a)));
}

template<>
class numeric_limits<op::Float4E2M1> {
public:
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;  // OCP MX E2M1 has no NaN
    static constexpr bool has_signaling_NaN = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_exact = false;
    static constexpr bool is_integer = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_modulo = false;
    static constexpr bool is_signed = true;
    static constexpr bool is_specialized = true;
    static constexpr int digits = 1;
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 1;
    static constexpr int min_exponent = -2;  // 2^(-2) for denorm
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 2;   // 2^2 for normal
    static constexpr int max_exponent10 = 0;
    static constexpr int radix = 2;
    static constexpr float round_error()
    {
        return 0.5f;
    }
    static constexpr float denorm_min()
    {
        return 0.5f;  // 2^0 * (1/2) = 0.5 (man=1, exp=0)
    }

    static constexpr op::Float4E2M1 min()
    {
        return op::Float4E2M1::MinPositiveNormal();
    }
    static constexpr op::Float4E2M1 lowest()
    {
        return op::Float4E2M1::Lowest();
    }
    static constexpr op::Float4E2M1 max()
    {
        return op::Float4E2M1::Highest();
    }
    static constexpr op::Float4E2M1 epsilon()
    {
        return op::Float4E2M1::Epsilon();
    }
    static constexpr op::Float4E2M1 infinity()
    {
        return op::Float4E2M1::Highest();  // No infinity, use max
    }
    static constexpr op::Float4E2M1 quiet_NaN()
    {
        // OCP MX E2M1 has no NaN, return zero as fallback
        return op::Float4E2M1(0x00, op::Float4E2M1::FromBits());
    }
    static constexpr op::Float4E2M1 signaling_NaN()
    {
        return quiet_NaN();
    }
};

} // namespace std

#endif // OP_API_FLOAT4_E2M1_H
