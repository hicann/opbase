/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_FLOAT6_E3M2_H
#define OP_API_FLOAT6_E3M2_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>

#include "opdev/fp16_t.h"
#include "opdev/bfloat16.h"

namespace op {

/**
 * @brief FP6 E3M2 floating-point format (OCP MX MXFP6)
 *
 * Format: 1 sign bit, 3 exponent bits, 2 mantissa bits
 *         +---+-----+--+
 *         | S | EEE |MM|
 *         +---+-----+--+
 *
 * Follows OCP Microscaling Formats (MX) v1.0 specification.
 *
 * - Bias: 3
 * - Max normal value: 28.0 (0x1F = 0 111 11, exp=7, man=3 -> 2^4 * 1.75 = 28.0)
 * - Min normal value: -28.0 (0x3F = 1 111 11)
 * - Min positive normal: 2^-2 = 0.25 (0x04 = 0 001 00)
 * - Min positive denorm: 2^(1-bias) * (man/4) = 0.0625 (0x01 = 0 000 01)
 * - No NaN representation (all bit patterns are valid numbers)
 * - No infinity representation
 */
struct Float6E3M2 {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits()
    {
        return FromBitsTag();
    }

    uint8_t value;

    // Constants for E3M2 format (OCP MX MXFP6)
    static constexpr int EXP_BITS = 3;
    static constexpr int MAN_BITS = 2;
    static constexpr int EXP_BIAS = 3;
    static constexpr uint8_t SIGN_MASK = 0x20;      // 1 000 00 (bit 5)
    static constexpr uint8_t EXP_MASK = 0x1C;       // 0 111 00 (bits 4-2)
    static constexpr uint8_t MAN_MASK = 0x03;       // 0 000 11 (bits 1-0)
    static constexpr uint8_t MAX_EXP = 0x07;        // 111

    // Default constructor - initialize to zero
    Float6E3M2() : value(0) {}

    constexpr Float6E3M2(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits & 0x3F)
    {
    }

    Float6E3M2(float v)
    {
        value = FloatToFloat6E3M2(v).value;
    }

    template<typename T>
    Float6E3M2 &operator=(T other)
    {
        value = FloatToFloat6E3M2(static_cast<float>(other)).value;
        return *this;
    }

    operator float() const
    {
        return Float6E3M2ToFloat(*this);
    }

    operator double() const
    {
        return static_cast<double>(Float6E3M2ToFloat(*this));
    }

    Float6E3M2 &operator=(const double &dVal)
    {
        value = FloatToFloat6E3M2(static_cast<float>(dVal)).value;
        return *this;
    }

    // Conversion to fp16_t
    operator fp16_t() const
    {
        return fp16_t(Float6E3M2ToFloat(*this));
    }

    // Conversion to bfloat16
    operator bfloat16() const
    {
        return bfloat16(static_cast<float>(*this));
    }

    static constexpr Float6E3M2 Epsilon()
    {
        // 2^-2 = 0.25 (the difference between 1.0 and next representable value)
        return Float6E3M2(0x04, FromBits());  // 0 001 00 = 2^(1-3) * 1.0 = 0.25
    }

    static constexpr Float6E3M2 Highest()
    {
        // 28.0 = 0x1F (0 111 11) -> exp=7, man=3 -> 2^4 * 1.75 = 28.0
        // OCP MX E3M2 has no NaN, all bit patterns are valid
        return Float6E3M2(0x1F, FromBits());
    }

    static constexpr Float6E3M2 Lowest()
    {
        return Float6E3M2(0x3F, FromBits());  // 1 111 11 = -28.0
    }

    static constexpr Float6E3M2 MinPositiveNormal()
    {
        // 2^-2 = 0.25 (0 001 00)
        return Float6E3M2(0x04, FromBits());
    }

    bool IsZero() const
    {
        return (value & 0x1F) == 0;  // 排除符号位
    }

    bool IsNaN() const
    {
        // OCP MX E3M2 has no NaN representation
        // All bit patterns represent valid numbers
        return false;
    }

    bool IsInf() const
    {
        // E3M2 has no infinity representation
        return false;
    }

private:
    union FP32 {
        uint32_t u;
        float f;
    };

    static float Float6E3M2ToFloat(Float6E3M2 fp6)
    {
        if (fp6.IsZero()) {
            return (fp6.value & SIGN_MASK) ? -0.0f : 0.0f;
        }

        // OCP MX E3M2 has no NaN, all bit patterns are valid numbers

        uint32_t sign = (fp6.value & SIGN_MASK) >> 5;
        uint32_t exp = (fp6.value & EXP_MASK) >> MAN_BITS;
        uint32_t man = fp6.value & MAN_MASK;

        // FP32: 1 sign, 8 exp (bias 127), 23 man
        // FP6 E3M2: 1 sign, 3 exp (bias 3), 2 man
        // Using FP32_EXP_BIAS and FP32_MAN_LEN from fp16_t.h

        int32_t fp32Exp = static_cast<int32_t>(exp) - EXP_BIAS + FP32_EXP_BIAS;
        uint32_t fp32Man = man << (FP32_MAN_LEN - MAN_BITS);

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
                fp32Exp = 1 - EXP_BIAS + FP32_EXP_BIAS - shift;
                fp32Man = man << (FP32_MAN_LEN - MAN_BITS);
            }
        } else {
            // Normal number - add implicit leading 1
            fp32Man |= (1 << FP32_MAN_LEN);
        }

        FP32 result;
        result.u = (sign << 31) | ((fp32Exp & 0xFF) << FP32_MAN_LEN) | (fp32Man & 0x7FFFFF);
        return result.f;
    }

    static Float6E3M2 FloatToFloat6E3M2(float f)
    {
        // OCP MX E3M2 has no NaN, clamp NaN to positive max value (28.0)
        // NaN has no meaningful sign, so always clamp to +28.0
        if (std::isnan(f)) {
            // 0x1F = 0 111 11 = 2^4 * 1.75 = 28.0
            return Float6E3M2(0x1F, FromBits());
        }

        FP32 fp32;
        fp32.f = f;
        uint32_t sign = (fp32.u >> 31) & 1;
        uint32_t exp = (fp32.u >> 23) & 0xFF;
        uint32_t man = fp32.u & 0x7FFFFF;

        // Using FP32_EXP_BIAS and FP32_MAN_LEN from fp16_t.h

        if (exp == 0 && man == 0) {
            // Zero
            return Float6E3M2(static_cast<uint8_t>(sign << 5), FromBits());
        }

        if (std::isinf(f)) {
            // E3M2 has no infinity, clamp to max value
            // 0x1F = 0 111 11 = +28.0, 0x3F = 1 111 11 = -28.0
            return Float6E3M2(static_cast<uint8_t>((sign << 5) | 0x1F), FromBits());
        }

        // Calculate E3M2 exponent
        int32_t fp6Exp;
        uint32_t fp6Man;

        if (exp == 0) {
            // Denormalized FP32 input
            fp6Exp = 1 - FP32_EXP_BIAS + EXP_BIAS;
            while ((man & 0x800000) == 0) {
                man <<= 1;
                fp6Exp--;
            }
            man &= 0x7FFFFF;
        } else {
            fp6Exp = static_cast<int32_t>(exp) - FP32_EXP_BIAS + EXP_BIAS;
        }

        // Round mantissa from 23 bits to 2 bits with round-to-nearest-even
        int mantissaShift = FP32_MAN_LEN - MAN_BITS;
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
                return Float6E3M2(static_cast<uint8_t>(sign << 5), FromBits());
            }
            fp6Man = (fp6Man | (1 << MAN_BITS)) >> (1 - fp6Exp);
            fp6Exp = 0;
        } else if (fp6Exp >= static_cast<int32_t>(MAX_EXP + 1)) {
            // Overflow - for E3M2, max exp is 7
            // OCP MX E3M2 has no NaN, clamp to max value (preserving sign)
            // 0x1F = +28.0, 0x3F = -28.0
            return Float6E3M2(static_cast<uint8_t>((sign << 5) | 0x1F), FromBits());
        }

        uint8_t result = static_cast<uint8_t>((sign << 5) | (fp6Exp << MAN_BITS) | (fp6Man & MAN_MASK));
        return Float6E3M2(result, FromBits());
    }
};

inline std::ostream &operator<<(std::ostream &os, const Float6E3M2 &dt)
{
    os << static_cast<float>(dt);
    return os;
}

inline Float6E3M2 operator+(Float6E3M2 a, Float6E3M2 b)
{
    return Float6E3M2(static_cast<float>(a) + static_cast<float>(b));
}

inline Float6E3M2 operator-(Float6E3M2 a, Float6E3M2 b)
{
    return Float6E3M2(static_cast<float>(a) - static_cast<float>(b));
}

inline Float6E3M2 operator*(Float6E3M2 a, Float6E3M2 b)
{
    return Float6E3M2(static_cast<float>(a) * static_cast<float>(b));
}

inline Float6E3M2 operator/(Float6E3M2 a, Float6E3M2 b)
{
    return Float6E3M2(static_cast<float>(a) / static_cast<float>(b));
}

inline Float6E3M2 operator-(Float6E3M2 a)
{
    a.value ^= Float6E3M2::SIGN_MASK;
    return a;
}

inline bool operator==(Float6E3M2 a, Float6E3M2 b)
{
    return a.value == b.value;
}

inline bool operator!=(Float6E3M2 a, Float6E3M2 b)
{
    return a.value != b.value;
}

inline bool operator<(Float6E3M2 a, Float6E3M2 b)
{
    return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator<=(Float6E3M2 a, Float6E3M2 b)
{
    return static_cast<float>(a) <= static_cast<float>(b);
}

inline bool operator>(Float6E3M2 a, Float6E3M2 b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}

inline bool operator>=(Float6E3M2 a, Float6E3M2 b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}

inline Float6E3M2 &operator+=(Float6E3M2 &a, Float6E3M2 b)
{
    a = a + b;
    return a;
}

inline Float6E3M2 &operator-=(Float6E3M2 &a, Float6E3M2 b)
{
    a = a - b;
    return a;
}

inline Float6E3M2 &operator*=(Float6E3M2 &a, Float6E3M2 b)
{
    a = a * b;
    return a;
}

inline Float6E3M2 &operator/=(Float6E3M2 &a, Float6E3M2 b)
{
    a = a / b;
    return a;
}

inline Float6E3M2 operator++(Float6E3M2 &a)
{
    a += Float6E3M2(1.0f);
    return a;
}

inline Float6E3M2 operator--(Float6E3M2 &a)
{
    a -= Float6E3M2(1.0f);
    return a;
}

inline Float6E3M2 operator++(Float6E3M2 &a, int)
{
    Float6E3M2 originalValue = a;
    ++a;
    return originalValue;
}

inline Float6E3M2 operator--(Float6E3M2 &a, int)
{
    Float6E3M2 originalValue = a;
    --a;
    return originalValue;
}

} // namespace op

namespace std {

template<>
struct hash<op::Float6E3M2> {
    size_t operator()(const op::Float6E3M2 &v) const
    {
        return hash<float>()(static_cast<float>(v));
    }
};

using op::Float6E3M2;

inline bool isinf(const Float6E3M2 &a [[maybe_unused]])
{
    return false;  // E3M2 has no infinity
}

inline bool isnan(const Float6E3M2 &a [[maybe_unused]])
{
    return false;  // OCP MX E3M2 has no NaN
}

inline bool isfinite(const Float6E3M2 &a [[maybe_unused]])
{
    return true;  // All bit patterns are valid finite numbers
}

inline Float6E3M2 abs(const Float6E3M2 &a)
{
    return Float6E3M2(std::abs(static_cast<float>(a)));
}

inline Float6E3M2 exp(const Float6E3M2 &a)
{
    return Float6E3M2(std::exp(static_cast<float>(a)));
}

inline Float6E3M2 log(const Float6E3M2 &a)
{
    return Float6E3M2(std::log(static_cast<float>(a)));
}

inline Float6E3M2 sqrt(const Float6E3M2 &a)
{
    return Float6E3M2(std::sqrt(static_cast<float>(a)));
}

inline Float6E3M2 pow(const Float6E3M2 &a, const Float6E3M2 &b)
{
    return Float6E3M2(std::pow(static_cast<float>(a), static_cast<float>(b)));
}

inline Float6E3M2 sin(const Float6E3M2 &a)
{
    return Float6E3M2(std::sin(static_cast<float>(a)));
}

inline Float6E3M2 cos(const Float6E3M2 &a)
{
    return Float6E3M2(std::cos(static_cast<float>(a)));
}

inline Float6E3M2 tanh(const Float6E3M2 &a)
{
    return Float6E3M2(std::tanh(static_cast<float>(a)));
}

inline Float6E3M2 floor(const Float6E3M2 &a)
{
    return Float6E3M2(std::floor(static_cast<float>(a)));
}

inline Float6E3M2 ceil(const Float6E3M2 &a)
{
    return Float6E3M2(std::ceil(static_cast<float>(a)));
}

template<>
class numeric_limits<op::Float6E3M2> {
public:
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;  // OCP MX E3M2 has no NaN
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
    static constexpr int min_exponent = -2;
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 4;
    static constexpr int max_exponent10 = 1;
    static constexpr int radix = 2;
    static constexpr float round_error()
    {
        return 0.5f;
    }
    static constexpr float denorm_min()
    {
        // OCP MX E3M2: min subnorm = S 000 01 = 2^(1-bias) * (0 + M/4) = 2^(-2) * 0.25 = 0.0625
        // Formula: (-1)^S * 2^(1-3) * (M/4) = M/16 where bias=3
        // When M=1: 1/16 = 0.0625
        return 0.0625f;
    }

    static constexpr op::Float6E3M2 min()
    {
        return op::Float6E3M2::MinPositiveNormal();
    }
    static constexpr op::Float6E3M2 lowest()
    {
        return op::Float6E3M2::Lowest();
    }
    static constexpr op::Float6E3M2 max()
    {
        return op::Float6E3M2::Highest();
    }
    static constexpr op::Float6E3M2 epsilon()
    {
        return op::Float6E3M2::Epsilon();
    }
    static constexpr op::Float6E3M2 infinity()
    {
        return op::Float6E3M2::Highest();  // No infinity, use max
    }
    static constexpr op::Float6E3M2 quiet_NaN()
    {
        // OCP MX E3M2 has no NaN, return zero as fallback
        return op::Float6E3M2(0x00, op::Float6E3M2::FromBits());
    }
    static constexpr op::Float6E3M2 signaling_NaN()
    {
        return quiet_NaN();
    }
};

} // namespace std

#endif // OP_API_FLOAT6_E3M2_H
