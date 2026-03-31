/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_FLOAT6_E2M3_H
#define OP_API_FLOAT6_E2M3_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>

namespace op {

/**
 * @brief FP6 E2M3 floating-point format (OCP MX MXFP6)
 *
 * Format: 1 sign bit, 2 exponent bits, 3 mantissa bits
 *         +---+----+---+
 *         | S | EE |MMM|
 *         +---+----+---+
 *
 * Follows OCP Microscaling Formats (MX) v1.0 specification.
 *
 * - Bias: 1
 * - Max normal value: 7.5 (0x1F = 0 11 111, exp=3, man=7 -> 2^2 * 1.875 = 7.5)
 * - Min normal value: -7.5 (0x3F = 1 11 111)
 * - Min positive normal: 2^-1 = 0.5 (0x08 = 0 01 000)
 * - Min positive denorm: 2^(1-bias) * (man/8) = 0.125 (0x01 = 0 00 001)
 * - No NaN representation (all bit patterns are valid numbers)
 * - No infinity representation
 */
struct Float6E2M3 {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits()
    {
        return FromBitsTag();
    }

    uint8_t value;

    // ============= FP32 format constants =============
    static constexpr int FP32_EXP_BIAS_VAL = 127;
    static constexpr int FP32_MAN_LEN_VAL = 23;
    static constexpr int FP32_SIGN_SHIFT_VAL = 31;
    static constexpr uint32_t FP32_EXP_MASK_8BIT = 0xFF;
    static constexpr uint32_t FP32_MAN_MASK_23BIT = 0x7FFFFF;
    static constexpr uint32_t FP32_IMPLICIT_1_VAL = 0x800000;

    // ============= E2M3 format constants =============
    // Bit layout: bit5=sign, bits4-3=exp, bits2-0=man
    static constexpr int EXP_BITS = 2;
    static constexpr int MAN_BITS = 3;
    static constexpr int EXP_BIAS = 1;
    static constexpr int SIGN_SHIFT = 5;
    static constexpr uint8_t SIGN_MASK = 0x20;            // 1 00 000 (bit 5)
    static constexpr uint8_t EXP_MASK = 0x18;             // 0 11 000 (bits 4-3)
    static constexpr uint8_t MAN_MASK = 0x07;             // 0 00 111 (bits 2-0)
    static constexpr uint8_t MAX_EXP = 0x03;              // 11
    static constexpr uint8_t ABS_VALUE_MASK = 0x1F;       // 0 11111 (exclude sign bit)
    static constexpr uint8_t HIGHEST_VALUE = 0x1F;        // 0 11 111 = 7.5
    static constexpr uint8_t LOWEST_VALUE = 0x3F;         // 1 11 111 = -7.5
    static constexpr uint8_t MIN_POS_NORMAL_VALUE = 0x08; // 0 01 000 (min positive normal)
    static constexpr uint8_t BITS_MASK = 0x3F;            // 6-bit valid bits mask
    static constexpr uint8_t EPSILON_VALUE = 0x01;        // 0 00 001 = denorm 0.125

    // Default constructor - initialize to zero
    Float6E2M3() : value(0) {}

    constexpr Float6E2M3(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits & BITS_MASK)
    {
    }

    Float6E2M3(float v)
    {
        value = FloatToFloat6E2M3(v).value;
    }

    operator float() const
    {
        return Float6E2M3ToFloat(*this);
    }

    operator double() const
    {
        return static_cast<double>(Float6E2M3ToFloat(*this));
    }

    static constexpr Float6E2M3 Epsilon()
    {
        // 2^-3 = 0.125 (the difference between 1.0 and next representable value)
        return Float6E2M3(EPSILON_VALUE, FromBits());  // 0 00 001 = denorm 0.125
    }

    static constexpr Float6E2M3 Highest()
    {
        // exp=3, man=7 -> 2^2 * (1 + 7/8) = 4 * 1.875 = 7.5
        // OCP MX E2M3 has no NaN, all bit patterns are valid
        return Float6E2M3(HIGHEST_VALUE, FromBits());  // 0 11 111 = 7.5
    }

    static constexpr Float6E2M3 Lowest()
    {
        return Float6E2M3(LOWEST_VALUE, FromBits());  // 1 11 111 = -7.5
    }

    static constexpr Float6E2M3 MinPositiveNormal()
    {
        // 2^-1 = 0.5 (0 01 000)
        return Float6E2M3(MIN_POS_NORMAL_VALUE, FromBits());
    }

    bool IsZero() const
    {
        return (value & ABS_VALUE_MASK) == 0;  // 排除符号位
    }

    bool IsNaN() const
    {
        // OCP MX E2M3 has no NaN representation
        // All bit patterns represent valid numbers
        return false;
    }

    bool IsInf() const
    {
        // E2M3 has no infinity representation
        return false;
    }

private:
    union FP32 {
        uint32_t u;
        float f;
    };

    static float Float6E2M3ToFloat(Float6E2M3 fp6)
    {
        if (fp6.IsZero()) {
            return (fp6.value & SIGN_MASK) ? -0.0f : 0.0f;
        }

        // OCP MX E2M3 has no NaN, all bit patterns are valid numbers

        uint32_t sign = (fp6.value & SIGN_MASK) >> SIGN_SHIFT;
        uint32_t exp = (fp6.value & EXP_MASK) >> MAN_BITS;
        uint32_t man = fp6.value & MAN_MASK;

        // FP32: 1 sign, 8 exp (bias 127), 23 man
        // FP6 E2M3: 1 sign, 2 exp (bias 1), 3 man

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
                man &= MAN_MASK;  // Remove implicit leading 1
                fp32Exp = 1 - EXP_BIAS + FP32_EXP_BIAS_VAL - shift;
                fp32Man = man << (FP32_MAN_LEN_VAL - MAN_BITS);
            }
        } else {
            // Normal number - add implicit leading 1
            fp32Man |= FP32_IMPLICIT_1_VAL;
        }

        FP32 result;
        result.u = (sign << FP32_SIGN_SHIFT_VAL) | ((fp32Exp & FP32_EXP_MASK_8BIT) << FP32_MAN_LEN_VAL) | (fp32Man & FP32_MAN_MASK_23BIT);
        return result.f;
    }

    static Float6E2M3 FloatToFloat6E2M3(float f)
    {
        // OCP MX E2M3 has no NaN, clamp NaN to positive max value (7.5)
        // NaN has no meaningful sign, so always clamp to +7.5
        if (std::isnan(f)) {
            // 0x1F = 0 11 111 = 2^2 * 1.875 = 7.5
            return Float6E2M3(HIGHEST_VALUE, FromBits());
        }

        FP32 fp32;
        fp32.f = f;
        uint32_t sign = (fp32.u >> FP32_SIGN_SHIFT_VAL) & 1;
        uint32_t exp = (fp32.u >> FP32_MAN_LEN_VAL) & FP32_EXP_MASK_8BIT;
        uint32_t man = fp32.u & FP32_MAN_MASK_23BIT;

        if (exp == 0 && man == 0) {
            // Zero
            return Float6E2M3(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
        }

        if (std::isinf(f)) {
            // E2M3 has no infinity, clamp to max value
            // 0x1F = +7.5, 0x3F = -7.5
            return Float6E2M3(static_cast<uint8_t>((sign << SIGN_SHIFT) | HIGHEST_VALUE), FromBits());
        }

        // Calculate E2M3 exponent
        int32_t fp6Exp;
        uint32_t fp6Man;

        if (exp == 0) {
            // Denormalized FP32 input
            fp6Exp = 1 - FP32_EXP_BIAS_VAL + EXP_BIAS;
            while ((man & FP32_IMPLICIT_1_VAL) == 0) {
                man <<= 1;
                fp6Exp--;
            }
            man &= FP32_MAN_MASK_23BIT;
        } else {
            fp6Exp = static_cast<int32_t>(exp) - FP32_EXP_BIAS_VAL + EXP_BIAS;
        }

        // Round mantissa from 23 bits to 3 bits with round-to-nearest-even
        int mantissaShift = FP32_MAN_LEN_VAL - MAN_BITS;
        uint32_t manRoundBit = (man >> (mantissaShift - 1)) & 1;
        uint32_t manStickyBits = (man & ((1 << (mantissaShift - 1)) - 1)) ? 1 : 0;
        uint32_t manLsb = (man >> mantissaShift) & 1;

        fp6Man = man >> mantissaShift;
        if (manRoundBit && (manStickyBits || manLsb)) {
            fp6Man++;
            if (fp6Man > MAN_MASK) {
                fp6Man = 0;
                fp6Exp++;
            }
        }

        // Handle overflow/underflow
        if (fp6Exp <= 0) {
            if (fp6Exp < -MAN_BITS) {
                return Float6E2M3(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
            }
            fp6Man = (fp6Man | (1 << MAN_BITS)) >> (1 - fp6Exp);
            fp6Exp = 0;
        } else if (fp6Exp >= static_cast<int32_t>(MAX_EXP + 1)) {
            // Overflow - for E2M3, max exp is 3
            // OCP MX E2M3 has no NaN, clamp to max value (preserving sign)
            // 0x1F = 0 11 111 = +7.5, 0x3F = 1 11 111 = -7.5
            return Float6E2M3(static_cast<uint8_t>((sign << SIGN_SHIFT) | HIGHEST_VALUE), FromBits());
        }

        uint8_t result = static_cast<uint8_t>((sign << SIGN_SHIFT) | (fp6Exp << MAN_BITS) | (fp6Man & MAN_MASK));
        return Float6E2M3(result, FromBits());
    }
};

inline std::ostream &operator<<(std::ostream &os, const Float6E2M3 &dt)
{
    os << static_cast<float>(dt);
    return os;
}

inline Float6E2M3 operator+(Float6E2M3 a, Float6E2M3 b)
{
    return Float6E2M3(static_cast<float>(a) + static_cast<float>(b));
}

inline Float6E2M3 operator-(Float6E2M3 a, Float6E2M3 b)
{
    return Float6E2M3(static_cast<float>(a) - static_cast<float>(b));
}

inline Float6E2M3 operator*(Float6E2M3 a, Float6E2M3 b)
{
    return Float6E2M3(static_cast<float>(a) * static_cast<float>(b));
}

inline Float6E2M3 operator/(Float6E2M3 a, Float6E2M3 b)
{
    return Float6E2M3(static_cast<float>(a) / static_cast<float>(b));
}

inline Float6E2M3 operator-(Float6E2M3 a)
{
    a.value ^= Float6E2M3::SIGN_MASK;
    return a;
}

inline bool operator==(Float6E2M3 a, Float6E2M3 b)
{
    return a.value == b.value;
}

inline bool operator!=(Float6E2M3 a, Float6E2M3 b)
{
    return a.value != b.value;
}

inline bool operator<(Float6E2M3 a, Float6E2M3 b)
{
    return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator<=(Float6E2M3 a, Float6E2M3 b)
{
    return static_cast<float>(a) <= static_cast<float>(b);
}

inline bool operator>(Float6E2M3 a, Float6E2M3 b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}

inline bool operator>=(Float6E2M3 a, Float6E2M3 b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}

inline Float6E2M3 &operator+=(Float6E2M3 &a, Float6E2M3 b)
{
    a = a + b;
    return a;
}

inline Float6E2M3 &operator-=(Float6E2M3 &a, Float6E2M3 b)
{
    a = a - b;
    return a;
}

inline Float6E2M3 &operator*=(Float6E2M3 &a, Float6E2M3 b)
{
    a = a * b;
    return a;
}

inline Float6E2M3 &operator/=(Float6E2M3 &a, Float6E2M3 b)
{
    a = a / b;
    return a;
}

inline Float6E2M3 operator++(Float6E2M3 &a)
{
    a += Float6E2M3(1.0f);
    return a;
}

inline Float6E2M3 operator--(Float6E2M3 &a)
{
    a -= Float6E2M3(1.0f);
    return a;
}

inline Float6E2M3 operator++(Float6E2M3 &a, int)
{
    Float6E2M3 originalValue = a;
    ++a;
    return originalValue;
}

inline Float6E2M3 operator--(Float6E2M3 &a, int)
{
    Float6E2M3 originalValue = a;
    --a;
    return originalValue;
}

} // namespace op

namespace std {

template<>
struct hash<op::Float6E2M3> {
    size_t operator()(const op::Float6E2M3 &v) const
    {
        return hash<float>()(static_cast<float>(v));
    }
};

using op::Float6E2M3;

inline bool isinf(const Float6E2M3 &a [[maybe_unused]])
{
    return false;  // E2M3 has no infinity
}

inline bool isnan(const Float6E2M3 &a [[maybe_unused]])
{
    return false;  // OCP MX E2M3 has no NaN
}

inline bool isfinite(const Float6E2M3 &a [[maybe_unused]])
{
    return true;  // All bit patterns are valid finite numbers
}

inline Float6E2M3 abs(const Float6E2M3 &a)
{
    return Float6E2M3(std::abs(static_cast<float>(a)));
}

inline Float6E2M3 exp(const Float6E2M3 &a)
{
    return Float6E2M3(std::exp(static_cast<float>(a)));
}

inline Float6E2M3 log(const Float6E2M3 &a)
{
    return Float6E2M3(std::log(static_cast<float>(a)));
}

inline Float6E2M3 sqrt(const Float6E2M3 &a)
{
    return Float6E2M3(std::sqrt(static_cast<float>(a)));
}

inline Float6E2M3 pow(const Float6E2M3 &a, const Float6E2M3 &b)
{
    return Float6E2M3(std::pow(static_cast<float>(a), static_cast<float>(b)));
}

inline Float6E2M3 sin(const Float6E2M3 &a)
{
    return Float6E2M3(std::sin(static_cast<float>(a)));
}

inline Float6E2M3 cos(const Float6E2M3 &a)
{
    return Float6E2M3(std::cos(static_cast<float>(a)));
}

inline Float6E2M3 tanh(const Float6E2M3 &a)
{
    return Float6E2M3(std::tanh(static_cast<float>(a)));
}

inline Float6E2M3 floor(const Float6E2M3 &a)
{
    return Float6E2M3(std::floor(static_cast<float>(a)));
}

inline Float6E2M3 ceil(const Float6E2M3 &a)
{
    return Float6E2M3(std::ceil(static_cast<float>(a)));
}

template<>
class numeric_limits<op::Float6E2M3> {
public:
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;  // OCP MX E2M3 has no NaN
    static constexpr bool has_signaling_NaN = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_exact = false;
    static constexpr bool is_integer = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_modulo = false;
    static constexpr bool is_signed = true;
    static constexpr bool is_specialized = true;
    static constexpr int digits = 3;
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 1;
    static constexpr int min_exponent = -1;
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 2;
    static constexpr int max_exponent10 = 0;
    static constexpr int radix = 2;
    static constexpr float round_error()
    {
        return 0.5f;
    }
    static constexpr float denorm_min()
    {
        // OCP MX E2M3: min subnorm = S 00 001 = 2^0 * 0.125 = 0.125
        // Formula: (-1)^S * 2^(1-bias) * (M/8) = M/8 where bias=1
        // When M=1: 1/8 = 0.125
        return 0.125f;
    }

    static constexpr op::Float6E2M3 min()
    {
        return op::Float6E2M3::MinPositiveNormal();
    }
    static constexpr op::Float6E2M3 lowest()
    {
        return op::Float6E2M3::Lowest();
    }
    static constexpr op::Float6E2M3 max()
    {
        return op::Float6E2M3::Highest();
    }
    static constexpr op::Float6E2M3 epsilon()
    {
        return op::Float6E2M3::Epsilon();
    }
    static constexpr op::Float6E2M3 infinity()
    {
        return op::Float6E2M3::Highest();  // No infinity, use max
    }
    static constexpr op::Float6E2M3 quiet_NaN()
    {
        // OCP MX E2M3 has no NaN, return zero as fallback
        return op::Float6E2M3(0x00, op::Float6E2M3::FromBits());
    }
    static constexpr op::Float6E2M3 signaling_NaN()
    {
        return quiet_NaN();
    }
};

} // namespace std

#endif // OP_API_FLOAT6_E2M3_H
