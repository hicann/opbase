/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_FLOAT4_E1M2_H
#define OP_API_FLOAT4_E1M2_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>

namespace op {

/**
 * @brief FP4 E1M2 floating-point format
 *
 * Format: 1 sign bit, 1 exponent bit, 2 mantissa bits
 *         +---+--+---+
 *         | S |E |MM |
 *         +---+--+---+
 *
 * Properties:
 * - Bias: 1
 * - Max normal value: 1.75 (0x7 = 0 1 11, exp=1, man=3 -> 2^0 * 1.75)
 * - Min normal value: -1.75 (0xF = 1 1 11, exp=1, man=3 -> -2^0 * 1.75)
 * - Min positive normal: 2^0 = 1.0 (0x4 = 0 1 00)
 * - Min positive denorm: 2^(1-bias) * (man/4) = 2^0 * 0.25 = 0.25 (0x1 = 0 0 01)
 * - No NaN representation (all bit patterns are valid numbers)
 * - No infinity representation
 */
struct Float4E1M2 {
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

    // ============= E1M2 format constants =============
    // Bit layout: bit3=sign, bit2=exp, bits1-0=man
    static constexpr int EXP_BITS = 1;
    static constexpr int MAN_BITS = 2;
    static constexpr int EXP_BIAS = 1;
    static constexpr int SIGN_SHIFT = 3;
    static constexpr uint8_t SIGN_MASK = 0x08;               // 1 0 00 (bit 3)
    static constexpr uint8_t EXP_MASK = 0x04;                // 0 1 00 (bit 2)
    static constexpr uint8_t MAN_MASK = 0x03;                // 0 0 11 (bits 1-0)
    static constexpr uint8_t MAX_EXP = 0x01;                 // 1
    static constexpr uint8_t ABS_VALUE_MASK = 0x07;          // 0 111 (exclude sign bit)
    static constexpr uint8_t BITS_MASK = 0x0F;               // 4-bit mask
    static constexpr uint8_t HIGHEST_VALUE = 0x07;           // 0 1 11 (max value: 1.75)
    static constexpr uint8_t LOWEST_VALUE = 0x0F;            // 1 1 11 (min value: -1.75)
    static constexpr uint8_t MIN_POS_NORMAL_VALUE = 0x04;    // 0 1 00 (min positive normal)
    static constexpr uint8_t EPSILON_VALUE = 0x01;           // epsilon value

    // Default constructor - initialize to zero
    Float4E1M2() : value(0) {}

    constexpr Float4E1M2(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits & BITS_MASK)
    {
    }

    Float4E1M2(float v)
    {
        value = FloatToFloat4E1M2(v).value;
    }

    operator float() const
    {
        return Float4E1M2ToFloat(*this);
    }

    operator double() const
    {
        return static_cast<double>(Float4E1M2ToFloat(*this));
    }

    static constexpr Float4E1M2 Epsilon()
    {
        // 2^-2 = 0.25 (the difference between 1.0 and next representable value)
        return Float4E1M2(EPSILON_VALUE, FromBits());  // 0 0 01 = denorm 0.25
    }

    static constexpr Float4E1M2 Highest()
    {
        // exp=1, man=3 -> 2^0 * 1.75 = 1.75 (0 1 11 = 0x7)
        return Float4E1M2(HIGHEST_VALUE, FromBits());
    }

    static constexpr Float4E1M2 Lowest()
    {
        // E1M2 has no NaN, all bit patterns are valid
        // 1 1 11 = sign=1, exp=1, man=3 -> -2^0 * 1.75 = -1.75
        return Float4E1M2(LOWEST_VALUE, FromBits());  // 1 1 11 = -1.75
    }

    static constexpr Float4E1M2 MinPositiveNormal()
    {
        // 2^0 = 1.0 (0 1 00)
        return Float4E1M2(MIN_POS_NORMAL_VALUE, FromBits());
    }

    bool IsZero() const
    {
        return (value & ABS_VALUE_MASK) == 0;  // 排除符号位
    }

    bool IsNaN() const
    {
        // E1M2 has no NaN representation
        // All bit patterns represent valid numbers
        return false;
    }

    bool IsInf() const
    {
        // E1M2 has no infinity representation
        return false;
    }

private:
    union FP32 {
        uint32_t u;
        float f;
    };

    static float Float4E1M2ToFloat(Float4E1M2 fp4)
    {
        if (fp4.IsZero()) {
            return (fp4.value & SIGN_MASK) ? -0.0f : 0.0f;
        }

        // E1M2 has no NaN, all bit patterns are valid numbers

        uint32_t sign = (fp4.value & SIGN_MASK) >> SIGN_SHIFT;
        uint32_t exp = (fp4.value & EXP_MASK) >> MAN_BITS;
        uint32_t man = fp4.value & MAN_MASK;

        // FP32: 1 sign, 8 exp (bias 127), 23 man
        // FP4 E1M2: 1 sign, 1 exp (bias 1), 2 man

        int32_t fp32Exp;
        uint32_t fp32Man = man << (FP32_MAN_LEN_VAL - MAN_BITS);

        if (exp == 0) {
            // Denormalized number
            // E1M2: value = 2^(1-bias) * (man/4) = 2^0 * (man/4) = man/4
            if (man != 0) {
                // Denorm values: 0.25, 0.5, 0.75 for man=1,2,3
                // Find leading 1 position for normalization
                int shift = 0;
                uint32_t tempMan = man;
                while ((tempMan & (1 << MAN_BITS)) == 0 && shift < MAN_BITS) {
                    tempMan <<= 1;
                    shift++;
                }
                // Effective exponent for denorm is 1-bias=0
                // fp32Exp = FP32_BIAS + (1-bias) - shift = 127 - shift
                fp32Exp = 1 - EXP_BIAS + FP32_EXP_BIAS_VAL - shift;
                fp32Man = (tempMan & MAN_MASK) << (FP32_MAN_LEN_VAL - MAN_BITS);
            } else {
                fp32Exp = 0;
            }
        } else {
            // Normal number: value = 2^(1-1) * (1 + man/4) = (1 + man/4)
            fp32Exp = 1 - EXP_BIAS + FP32_EXP_BIAS_VAL;  // 127
            fp32Man |= FP32_IMPLICIT_1_VAL;  // Add implicit 1
        }

        FP32 result;
        result.u = (sign << FP32_SIGN_SHIFT_VAL) | ((fp32Exp & FP32_EXP_MASK_8BIT) << FP32_MAN_LEN_VAL) | (fp32Man & FP32_MAN_MASK_23BIT);
        return result.f;
    }

    static Float4E1M2 FloatToFloat4E1M2(float f)
    {
        // E1M2 has no NaN, clamp NaN to positive max value (1.75)
        // NaN has no meaningful sign, so always clamp to +1.75
        if (std::isnan(f)) {
            // 0x07 = 0 11 1 = 2^1 * 1.75 = 1.75
            return Float4E1M2(HIGHEST_VALUE, FromBits());
        }

        FP32 fp32;
        fp32.f = f;
        uint32_t sign = (fp32.u >> FP32_SIGN_SHIFT_VAL) & 1;
        uint32_t exp = (fp32.u >> FP32_MAN_LEN_VAL) & FP32_EXP_MASK_8BIT;
        uint32_t man = fp32.u & FP32_MAN_MASK_23BIT;

        if (exp == 0 && man == 0) {
            // Zero
            return Float4E1M2(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
        }

        if (std::isinf(f)) {
            // E1M2 has no infinity, clamp to max value
            // 0x07 = +1.75, 0x0F = -1.75
            return Float4E1M2(static_cast<uint8_t>((sign << SIGN_SHIFT) | HIGHEST_VALUE), FromBits());
        }

        // Calculate E1M2 exponent
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

        // Round mantissa from 23 bits to 2 bits with round-to-nearest-even
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
                return Float4E1M2(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
            }
            fp4Man = (fp4Man | (1 << MAN_BITS)) >> (1 - fp4Exp);
            fp4Exp = 0;
        } else if (fp4Exp >= static_cast<int32_t>(MAX_EXP + 1)) {
            // Overflow - for E1M2, max exp is 1
            // E1M2 has no NaN, clamp to max value (preserving sign)
            // 0x07 = +1.75, 0x0F = -1.75
            return Float4E1M2(static_cast<uint8_t>((sign << SIGN_SHIFT) | HIGHEST_VALUE), FromBits());
        }

        uint8_t result = static_cast<uint8_t>((sign << SIGN_SHIFT) | (fp4Exp << MAN_BITS) | (fp4Man & MAN_MASK));
        return Float4E1M2(result, FromBits());
    }
};

inline std::ostream &operator<<(std::ostream &os, const Float4E1M2 &dt)
{
    os << static_cast<float>(dt);
    return os;
}

inline Float4E1M2 operator+(Float4E1M2 a, Float4E1M2 b)
{
    return Float4E1M2(static_cast<float>(a) + static_cast<float>(b));
}

inline Float4E1M2 operator-(Float4E1M2 a, Float4E1M2 b)
{
    return Float4E1M2(static_cast<float>(a) - static_cast<float>(b));
}

inline Float4E1M2 operator*(Float4E1M2 a, Float4E1M2 b)
{
    return Float4E1M2(static_cast<float>(a) * static_cast<float>(b));
}

inline Float4E1M2 operator/(Float4E1M2 a, Float4E1M2 b)
{
    return Float4E1M2(static_cast<float>(a) / static_cast<float>(b));
}

inline Float4E1M2 operator-(Float4E1M2 a)
{
    a.value ^= Float4E1M2::SIGN_MASK;
    return a;
}

inline bool operator==(Float4E1M2 a, Float4E1M2 b)
{
    return a.value == b.value;
}

inline bool operator!=(Float4E1M2 a, Float4E1M2 b)
{
    return a.value != b.value;
}

inline bool operator<(Float4E1M2 a, Float4E1M2 b)
{
    return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator<=(Float4E1M2 a, Float4E1M2 b)
{
    return static_cast<float>(a) <= static_cast<float>(b);
}

inline bool operator>(Float4E1M2 a, Float4E1M2 b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}

inline bool operator>=(Float4E1M2 a, Float4E1M2 b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}

inline Float4E1M2 &operator+=(Float4E1M2 &a, Float4E1M2 b)
{
    a = a + b;
    return a;
}

inline Float4E1M2 &operator-=(Float4E1M2 &a, Float4E1M2 b)
{
    a = a - b;
    return a;
}

inline Float4E1M2 &operator*=(Float4E1M2 &a, Float4E1M2 b)
{
    a = a * b;
    return a;
}

inline Float4E1M2 &operator/=(Float4E1M2 &a, Float4E1M2 b)
{
    a = a / b;
    return a;
}

inline Float4E1M2 operator++(Float4E1M2 &a)
{
    a += Float4E1M2(1.0f);
    return a;
}

inline Float4E1M2 operator--(Float4E1M2 &a)
{
    a -= Float4E1M2(1.0f);
    return a;
}

inline Float4E1M2 operator++(Float4E1M2 &a, int)
{
    Float4E1M2 originalValue = a;
    ++a;
    return originalValue;
}

inline Float4E1M2 operator--(Float4E1M2 &a, int)
{
    Float4E1M2 originalValue = a;
    --a;
    return originalValue;
}

} // namespace op

namespace std {

template<>
struct hash<op::Float4E1M2> {
    size_t operator()(const op::Float4E1M2 &v) const
    {
        return hash<float>()(static_cast<float>(v));
    }
};

using op::Float4E1M2;

inline bool isinf(const Float4E1M2 &a [[maybe_unused]])
{
    return false;  // E1M2 has no infinity
}

inline bool isnan(const Float4E1M2 &a)
{
    return a.IsNaN();
}

inline bool isfinite(const Float4E1M2 &a)
{
    return !a.IsNaN();
}

inline Float4E1M2 abs(const Float4E1M2 &a)
{
    return Float4E1M2(std::abs(static_cast<float>(a)));
}

inline Float4E1M2 exp(const Float4E1M2 &a)
{
    return Float4E1M2(std::exp(static_cast<float>(a)));
}

inline Float4E1M2 log(const Float4E1M2 &a)
{
    return Float4E1M2(std::log(static_cast<float>(a)));
}

inline Float4E1M2 sqrt(const Float4E1M2 &a)
{
    return Float4E1M2(std::sqrt(static_cast<float>(a)));
}

inline Float4E1M2 pow(const Float4E1M2 &a, const Float4E1M2 &b)
{
    return Float4E1M2(std::pow(static_cast<float>(a), static_cast<float>(b)));
}

inline Float4E1M2 sin(const Float4E1M2 &a)
{
    return Float4E1M2(std::sin(static_cast<float>(a)));
}

inline Float4E1M2 cos(const Float4E1M2 &a)
{
    return Float4E1M2(std::cos(static_cast<float>(a)));
}

inline Float4E1M2 tanh(const Float4E1M2 &a)
{
    return Float4E1M2(std::tanh(static_cast<float>(a)));
}

inline Float4E1M2 floor(const Float4E1M2 &a)
{
    return Float4E1M2(std::floor(static_cast<float>(a)));
}

inline Float4E1M2 ceil(const Float4E1M2 &a)
{
    return Float4E1M2(std::ceil(static_cast<float>(a)));
}

template<>
class numeric_limits<op::Float4E1M2> {
public:
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;  // E1M2 has no NaN
    static constexpr bool has_signaling_NaN = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_exact = false;
    static constexpr bool is_integer = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_modulo = false;
    static constexpr bool is_signed = true;
    static constexpr bool is_specialized = true;
    static constexpr int digits = 2;
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 1;
    static constexpr int min_exponent = 0;  // 2^0 for denorm (man/4)
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 0;   // 2^0 for normal
    static constexpr int max_exponent10 = 0;
    static constexpr int radix = 2;
    static constexpr float round_error()
    {
        return 0.5f;
    }
    static constexpr float denorm_min()
    {
        return 0.25f;  // 2^0 * (1/4) = 0.25 (man=1, exp=0)
    }

    static constexpr op::Float4E1M2 min()
    {
        return op::Float4E1M2::MinPositiveNormal();
    }
    static constexpr op::Float4E1M2 lowest()
    {
        return op::Float4E1M2::Lowest();
    }
    static constexpr op::Float4E1M2 max()
    {
        return op::Float4E1M2::Highest();
    }
    static constexpr op::Float4E1M2 epsilon()
    {
        return op::Float4E1M2::Epsilon();
    }
    static constexpr op::Float4E1M2 infinity()
    {
        return op::Float4E1M2::Highest();  // No infinity, use max
    }
    static constexpr op::Float4E1M2 quiet_NaN()
    {
        // E1M2 has no NaN, return zero as fallback
        return op::Float4E1M2(0x00, op::Float4E1M2::FromBits());
    }
    static constexpr op::Float4E1M2 signaling_NaN()
    {
        return quiet_NaN();
    }
};

} // namespace std

#endif // OP_API_FLOAT4_E1M2_H
