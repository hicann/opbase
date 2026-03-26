/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_FLOAT8_E5M2_H
#define OP_API_FLOAT8_E5M2_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>

#include "opdev/fp16_t.h"
#include "opdev/bfloat16.h"

namespace op {

/**
 * @brief FP8 E5M2 floating-point format (OCP standard)
 *
 * Format: 1 sign bit, 5 exponent bits, 2 mantissa bits
 *         +---+-----+--+
 *         | S | EEEEE|MM|
 *         +---+-----+--+
 *
 * - Bias: 15
 * - Max normal value: 57344 (0x7B)
 * - Min positive normal: 2^-14 = 0.00006103515625 (0x04)
 * - Supports Infinity (0x7C, 0xFC)
 * - Supports NaN (0x7D-0x7F, 0xFD-0xFF)
 */
struct Float8E5M2 {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits()
    {
        return FromBitsTag();
    }

    uint8_t value;

    // Constants for E5M2 format
    static constexpr int EXP_BITS = 5;
    static constexpr int MAN_BITS = 2;
    static constexpr int EXP_BIAS = 15;
    static constexpr uint8_t SIGN_MASK = 0x80;      // 1 00000 00
    static constexpr uint8_t EXP_MASK = 0x7C;       // 0 11111 00
    static constexpr uint8_t MAN_MASK = 0x03;       // 0 00000 11
    static constexpr uint8_t MAX_EXP = 0x1F;        // 11111
    static constexpr uint8_t INF_VALUE = 0x7C;      // 0 11111 00
    static constexpr uint8_t NEG_INF_VALUE = 0xFC;  // 1 11111 00
    static constexpr uint8_t NAN_VALUE = 0x7F;      // 0 11111 11
    static constexpr uint8_t NEG_NAN_VALUE = 0xFF;  // 1 11111 11

    // Default constructor - initialize to zero
    Float8E5M2() : value(0) {}

    constexpr Float8E5M2(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits)
    {
    }

    Float8E5M2(float v)
    {
        value = FloatToFloat8E5M2(v).value;
    }

    template<typename T>
    Float8E5M2 &operator=(T other)
    {
        value = FloatToFloat8E5M2(static_cast<float>(other)).value;
        return *this;
    }

    operator float() const
    {
        return Float8E5M2ToFloat(*this);
    }

    operator double() const
    {
        return static_cast<double>(Float8E5M2ToFloat(*this));
    }

    Float8E5M2 &operator=(const double &dVal)
    {
        value = FloatToFloat8E5M2(static_cast<float>(dVal)).value;
        return *this;
    }

    // Conversion to fp16_t
    operator fp16_t() const
    {
        return fp16_t(Float8E5M2ToFloat(*this));
    }

    // Conversion to bfloat16
    operator bfloat16() const
    {
        return bfloat16(static_cast<float>(*this));
    }

    static constexpr Float8E5M2 Epsilon()
    {
        // 2^-2 = 0.25 at exp = 15 (1.0)
        return Float8E5M2(0x34, FromBits());  // 0 01101 00 = 1.0
    }

    static constexpr Float8E5M2 Highest()
    {
        // 57344.0 = 0x7B (0 11110 11)
        return Float8E5M2(0x7B, FromBits());
    }

    static constexpr Float8E5M2 Lowest()
    {
        // -57344.0 = 0xFB (1 11110 11)
        return Float8E5M2(0xFB, FromBits());
    }

    static constexpr Float8E5M2 MinPositiveNormal()
    {
        // 2^-14 = 0.00006103515625 = 0x04 (0 00001 00)
        return Float8E5M2(0x04, FromBits());
    }

    bool IsZero() const
    {
        return (value & 0x7F) == 0;
    }

    bool IsNaN() const
    {
        return (value & EXP_MASK) == EXP_MASK && (value & MAN_MASK) != 0;
    }

    bool IsInf() const
    {
        return value == INF_VALUE || value == NEG_INF_VALUE;
    }

private:
    union FP32 {
        uint32_t u;
        float f;
    };

    static float Float8E5M2ToFloat(Float8E5M2 fp8)
    {
        if (fp8.IsZero()) {
            return (fp8.value & SIGN_MASK) ? -0.0f : 0.0f;
        }

        if (fp8.IsNaN()) {
            return std::nanf("");
        }

        if (fp8.IsInf()) {
            return (fp8.value & SIGN_MASK) ? -std::numeric_limits<float>::infinity()
                                           : std::numeric_limits<float>::infinity();
        }

        uint32_t sign = (fp8.value & SIGN_MASK) >> 7;
        uint32_t exp = (fp8.value & EXP_MASK) >> MAN_BITS;
        uint32_t man = fp8.value & MAN_MASK;

        // FP32: 1 sign, 8 exp (bias 127), 23 man
        // FP8 E5M2: 1 sign, 5 exp (bias 15), 2 man
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

    static Float8E5M2 FloatToFloat8E5M2(float f)
    {
        if (std::isnan(f)) {
            return Float8E5M2(NAN_VALUE, FromBits());
        }

        FP32 fp32;
        fp32.f = f;
        uint32_t sign = (fp32.u >> 31) & 1;
        uint32_t exp = (fp32.u >> 23) & 0xFF;
        uint32_t man = fp32.u & 0x7FFFFF;

        // Using FP32_EXP_BIAS and FP32_MAN_LEN from fp16_t.h

        if (exp == 0 && man == 0) {
            // Zero
            return Float8E5M2(static_cast<uint8_t>(sign << 7), FromBits());
        }

        if (std::isinf(f)) {
            // E5M2 supports infinity
            return Float8E5M2(static_cast<uint8_t>((sign << 7) | INF_VALUE), FromBits());
        }

        // Calculate E5M2 exponent
        int32_t fp8Exp;
        uint32_t fp8Man;

        if (exp == 0) {
            // Denormalized FP32 input
            fp8Exp = 1 - FP32_EXP_BIAS + EXP_BIAS;
            // Normalize the FP32 denormal
            while ((man & 0x800000) == 0) {
                man <<= 1;
                fp8Exp--;
            }
            man &= 0x7FFFFF;
        } else {
            fp8Exp = static_cast<int32_t>(exp) - FP32_EXP_BIAS + EXP_BIAS;
        }

        // Round mantissa from 23 bits to 2 bits with round-to-nearest-even
        int mantissaShift = FP32_MAN_LEN - MAN_BITS;
        uint32_t manRoundBit = (man >> (mantissaShift - 1)) & 1;
        uint32_t manStickyBits = (man & ((1 << (mantissaShift - 1)) - 1)) ? 1 : 0;
        uint32_t manLsb = (man >> mantissaShift) & 1;

        fp8Man = man >> mantissaShift;
        if (manRoundBit && (manStickyBits || manLsb)) {
            fp8Man++;
            if (fp8Man > MAN_MASK) {
                fp8Man = 0;
                fp8Exp++;
            }
        }

        // Handle overflow/underflow
        if (fp8Exp <= 0) {
            // Underflow to zero or denormal
            if (fp8Exp < -MAN_BITS) {
                return Float8E5M2(static_cast<uint8_t>(sign << 7), FromBits());
            }
            // Create denormal
            fp8Man = (fp8Man | (1 << MAN_BITS)) >> (1 - fp8Exp);
            fp8Exp = 0;
        } else if (fp8Exp >= static_cast<int32_t>(MAX_EXP)) {
            // Overflow to infinity
            return Float8E5M2(static_cast<uint8_t>((sign << 7) | INF_VALUE), FromBits());
        }

        uint8_t result = static_cast<uint8_t>((sign << 7) | (fp8Exp << MAN_BITS) | (fp8Man & MAN_MASK));
        return Float8E5M2(result, FromBits());
    }
};

inline std::ostream &operator<<(std::ostream &os, const Float8E5M2 &dt)
{
    os << static_cast<float>(dt);
    return os;
}

inline Float8E5M2 operator+(Float8E5M2 a, Float8E5M2 b)
{
    return Float8E5M2(static_cast<float>(a) + static_cast<float>(b));
}

inline Float8E5M2 operator-(Float8E5M2 a, Float8E5M2 b)
{
    return Float8E5M2(static_cast<float>(a) - static_cast<float>(b));
}

inline Float8E5M2 operator*(Float8E5M2 a, Float8E5M2 b)
{
    return Float8E5M2(static_cast<float>(a) * static_cast<float>(b));
}

inline Float8E5M2 operator/(Float8E5M2 a, Float8E5M2 b)
{
    return Float8E5M2(static_cast<float>(a) / static_cast<float>(b));
}

inline Float8E5M2 operator-(Float8E5M2 a)
{
    a.value ^= Float8E5M2::SIGN_MASK;
    return a;
}

inline bool operator==(Float8E5M2 a, Float8E5M2 b)
{
    return a.value == b.value;
}

inline bool operator!=(Float8E5M2 a, Float8E5M2 b)
{
    return a.value != b.value;
}

inline bool operator<(Float8E5M2 a, Float8E5M2 b)
{
    return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator<=(Float8E5M2 a, Float8E5M2 b)
{
    return static_cast<float>(a) <= static_cast<float>(b);
}

inline bool operator>(Float8E5M2 a, Float8E5M2 b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}

inline bool operator>=(Float8E5M2 a, Float8E5M2 b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}

inline Float8E5M2 &operator+=(Float8E5M2 &a, Float8E5M2 b)
{
    a = a + b;
    return a;
}

inline Float8E5M2 &operator-=(Float8E5M2 &a, Float8E5M2 b)
{
    a = a - b;
    return a;
}

inline Float8E5M2 &operator*=(Float8E5M2 &a, Float8E5M2 b)
{
    a = a * b;
    return a;
}

inline Float8E5M2 &operator/=(Float8E5M2 &a, Float8E5M2 b)
{
    a = a / b;
    return a;
}

inline Float8E5M2 operator++(Float8E5M2 &a)
{
    a += Float8E5M2(1.0f);
    return a;
}

inline Float8E5M2 operator--(Float8E5M2 &a)
{
    a -= Float8E5M2(1.0f);
    return a;
}

inline Float8E5M2 operator++(Float8E5M2 &a, int)
{
    Float8E5M2 originalValue = a;
    ++a;
    return originalValue;
}

inline Float8E5M2 operator--(Float8E5M2 &a, int)
{
    Float8E5M2 originalValue = a;
    --a;
    return originalValue;
}

} // namespace op

namespace std {

template<>
struct hash<op::Float8E5M2> {
    size_t operator()(const op::Float8E5M2 &v) const
    {
        return hash<float>()(static_cast<float>(v));
    }
};

using op::Float8E5M2;

inline bool isinf(const Float8E5M2 &a)
{
    return a.IsInf();
}

inline bool isnan(const Float8E5M2 &a)
{
    return a.IsNaN();
}

inline bool isfinite(const Float8E5M2 &a)
{
    return !a.IsNaN() && !a.IsInf();
}

inline Float8E5M2 abs(const Float8E5M2 &a)
{
    return Float8E5M2(std::abs(static_cast<float>(a)));
}

inline Float8E5M2 exp(const Float8E5M2 &a)
{
    return Float8E5M2(std::exp(static_cast<float>(a)));
}

inline Float8E5M2 log(const Float8E5M2 &a)
{
    return Float8E5M2(std::log(static_cast<float>(a)));
}

inline Float8E5M2 sqrt(const Float8E5M2 &a)
{
    return Float8E5M2(std::sqrt(static_cast<float>(a)));
}

inline Float8E5M2 pow(const Float8E5M2 &a, const Float8E5M2 &b)
{
    return Float8E5M2(std::pow(static_cast<float>(a), static_cast<float>(b)));
}

inline Float8E5M2 sin(const Float8E5M2 &a)
{
    return Float8E5M2(std::sin(static_cast<float>(a)));
}

inline Float8E5M2 cos(const Float8E5M2 &a)
{
    return Float8E5M2(std::cos(static_cast<float>(a)));
}

inline Float8E5M2 tanh(const Float8E5M2 &a)
{
    return Float8E5M2(std::tanh(static_cast<float>(a)));
}

inline Float8E5M2 floor(const Float8E5M2 &a)
{
    return Float8E5M2(std::floor(static_cast<float>(a)));
}

inline Float8E5M2 ceil(const Float8E5M2 &a)
{
    return Float8E5M2(std::ceil(static_cast<float>(a)));
}

template<>
class numeric_limits<op::Float8E5M2> {
public:
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
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
    static constexpr int min_exponent = -14;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr int radix = 2;
    static constexpr float round_error()
    {
        return 0.5f;
    }
    static constexpr float denorm_min()
    {
        return 0.0000152587890625f;  // 2^-16
    }

    static constexpr op::Float8E5M2 min()
    {
        return op::Float8E5M2::MinPositiveNormal();
    }
    static constexpr op::Float8E5M2 lowest()
    {
        return op::Float8E5M2::Lowest();
    }
    static constexpr op::Float8E5M2 max()
    {
        return op::Float8E5M2::Highest();
    }
    static constexpr op::Float8E5M2 epsilon()
    {
        return op::Float8E5M2::Epsilon();
    }
    static constexpr op::Float8E5M2 infinity()
    {
        return op::Float8E5M2(op::Float8E5M2::INF_VALUE, op::Float8E5M2::FromBits());
    }
    static constexpr op::Float8E5M2 quiet_NaN()
    {
        return op::Float8E5M2(op::Float8E5M2::NAN_VALUE, op::Float8E5M2::FromBits());
    }
    static constexpr op::Float8E5M2 signaling_NaN()
    {
        return quiet_NaN();
    }
};

} // namespace std

#endif // OP_API_FLOAT8_E5M2_H
