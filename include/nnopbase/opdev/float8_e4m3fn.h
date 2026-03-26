/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_FLOAT8_E4M3FN_H
#define OP_API_FLOAT8_E4M3FN_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>

#include "opdev/fp16_t.h"
#include "opdev/bfloat16.h"

namespace op {

/**
 * @brief FP8 E4M3FN floating-point format (OCP standard)
 *
 * Format: 1 sign bit, 4 exponent bits, 3 mantissa bits (FN = Finite Number)
 *         +---+----+---+
 *         | S | EEEE|MMM|
 *         +---+----+---+
 *
 * - Bias: 7
 * - Max normal value: 448 (0x7E)
 * - Min positive normal: 2^-6 = 0.015625 (0x08)
 * - Supports NaN (0x7F, 0xFF only)
 * - No infinity representation (max finite value is 448)
 */
struct Float8E4M3FN {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits()
    {
        return FromBitsTag();
    }

    uint8_t value;

    // Constants for E4M3FN format
    static constexpr int EXP_BITS = 4;
    static constexpr int MAN_BITS = 3;
    static constexpr int EXP_BIAS = 7;
    static constexpr uint8_t SIGN_MASK = 0x80;      // 1 0000 000
    static constexpr uint8_t EXP_MASK = 0x78;       // 0 1111 000
    static constexpr uint8_t MAN_MASK = 0x07;       // 0 0000 111
    static constexpr uint8_t MAX_EXP = 0x0F;        // 1111
    static constexpr uint8_t NAN_VALUE = 0x7F;      // 0 1111 111
    static constexpr uint8_t NEG_NAN_VALUE = 0xFF;  // 1 1111 111

    // Default constructor - initialize to zero
    Float8E4M3FN() : value(0) {}

    constexpr Float8E4M3FN(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits)
    {
    }

    Float8E4M3FN(float v)
    {
        value = FloatToFloat8E4M3FN(v).value;
    }

    template<typename T>
    Float8E4M3FN &operator=(T other)
    {
        value = FloatToFloat8E4M3FN(static_cast<float>(other)).value;
        return *this;
    }

    operator float() const
    {
        return Float8E4M3FNToFloat(*this);
    }

    operator double() const
    {
        return static_cast<double>(Float8E4M3FNToFloat(*this));
    }

    Float8E4M3FN &operator=(const double &dVal)
    {
        value = FloatToFloat8E4M3FN(static_cast<float>(dVal)).value;
        return *this;
    }

    // Conversion to fp16_t
    operator fp16_t() const
    {
        return fp16_t(Float8E4M3FNToFloat(*this));
    }

    // Conversion to bfloat16
    operator bfloat16() const
    {
        return bfloat16(static_cast<float>(*this));
    }

    static constexpr Float8E4M3FN Epsilon()
    {
        // 2^-3 = 0.125
        return Float8E4M3FN(0x20, FromBits());  // 0 0100 000 = 1.0, so epsilon is one mantissa step at 1.0
    }

    static constexpr Float8E4M3FN Highest()
    {
        // 448.0 = 0x7E (0 1111 110)
        return Float8E4M3FN(0x7E, FromBits());
    }

    static constexpr Float8E4M3FN Lowest()
    {
        // -448.0 = 0xFE (1 1111 110)
        return Float8E4M3FN(0xFE, FromBits());
    }

    static constexpr Float8E4M3FN MinPositiveNormal()
    {
        // 2^-6 = 0.015625 = 0x08 (0 0001 000)
        return Float8E4M3FN(0x08, FromBits());
    }

    bool IsZero() const
    {
        return (value & 0x7F) == 0;
    }

    bool IsNaN() const
    {
        return value == NAN_VALUE || value == NEG_NAN_VALUE;
    }

    bool IsInf() const
    {
        // E4M3FN has no infinity representation
        return false;
    }

private:
    union FP32 {
        uint32_t u;
        float f;
    };

    static float Float8E4M3FNToFloat(Float8E4M3FN fp8)
    {
        if (fp8.IsZero()) {
            return (fp8.value & SIGN_MASK) ? -0.0f : 0.0f;
        }

        if (fp8.IsNaN()) {
            return std::nanf("");
        }

        uint32_t sign = (fp8.value & SIGN_MASK) >> 7;
        uint32_t exp = (fp8.value & EXP_MASK) >> MAN_BITS;
        uint32_t man = fp8.value & MAN_MASK;

        // FP32: 1 sign, 8 exp (bias 127), 23 man
        // FP8 E4M3FN: 1 sign, 4 exp (bias 7), 3 man
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

    static Float8E4M3FN FloatToFloat8E4M3FN(float f)
    {
        if (std::isnan(f)) {
            return Float8E4M3FN(NAN_VALUE, FromBits());
        }

        FP32 fp32;
        fp32.f = f;
        uint32_t sign = (fp32.u >> 31) & 1;
        uint32_t exp = (fp32.u >> 23) & 0xFF;
        uint32_t man = fp32.u & 0x7FFFFF;

        // Using FP32_EXP_BIAS and FP32_MAN_LEN from fp16_t.h

        if (exp == 0 && man == 0) {
            // Zero
            return Float8E4M3FN(static_cast<uint8_t>(sign << 7), FromBits());
        }

        if (std::isinf(f)) {
            // E4M3FN has no infinity, clamp to max value
            return Float8E4M3FN(static_cast<uint8_t>((sign << 7) | 0x7E), FromBits());
        }

        // Calculate E4M3FN exponent
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

        // Round mantissa from 23 bits to 3 bits with round-to-nearest-even
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
                return Float8E4M3FN(static_cast<uint8_t>(sign << 7), FromBits());
            }
            // Create denormal
            fp8Man = (fp8Man | (1 << MAN_BITS)) >> (1 - fp8Exp);
            fp8Exp = 0;
        } else if (fp8Exp >= MAX_EXP) {
            // Overflow - clamp to max value (E4M3FN has no infinity)
            return Float8E4M3FN(static_cast<uint8_t>((sign << 7) | 0x7E), FromBits());
        }

        uint8_t result = static_cast<uint8_t>((sign << 7) | (fp8Exp << MAN_BITS) | (fp8Man & MAN_MASK));
        return Float8E4M3FN(result, FromBits());
    }
};

inline std::ostream &operator<<(std::ostream &os, const Float8E4M3FN &dt)
{
    os << static_cast<float>(dt);
    return os;
}

inline Float8E4M3FN operator+(Float8E4M3FN a, Float8E4M3FN b)
{
    return Float8E4M3FN(static_cast<float>(a) + static_cast<float>(b));
}

inline Float8E4M3FN operator-(Float8E4M3FN a, Float8E4M3FN b)
{
    return Float8E4M3FN(static_cast<float>(a) - static_cast<float>(b));
}

inline Float8E4M3FN operator*(Float8E4M3FN a, Float8E4M3FN b)
{
    return Float8E4M3FN(static_cast<float>(a) * static_cast<float>(b));
}

inline Float8E4M3FN operator/(Float8E4M3FN a, Float8E4M3FN b)
{
    // Division by zero: E4M3FN has no infinity encoding, return NaN
    if (b.IsZero()) {
        return Float8E4M3FN(Float8E4M3FN::NAN_VALUE, Float8E4M3FN::FromBits());
    }
    return Float8E4M3FN(static_cast<float>(a) / static_cast<float>(b));
}

inline Float8E4M3FN operator-(Float8E4M3FN a)
{
    a.value ^= Float8E4M3FN::SIGN_MASK;
    return a;
}

inline bool operator==(Float8E4M3FN a, Float8E4M3FN b)
{
    return a.value == b.value;
}

inline bool operator!=(Float8E4M3FN a, Float8E4M3FN b)
{
    return a.value != b.value;
}

inline bool operator<(Float8E4M3FN a, Float8E4M3FN b)
{
    return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator<=(Float8E4M3FN a, Float8E4M3FN b)
{
    return static_cast<float>(a) <= static_cast<float>(b);
}

inline bool operator>(Float8E4M3FN a, Float8E4M3FN b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}

inline bool operator>=(Float8E4M3FN a, Float8E4M3FN b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}

inline Float8E4M3FN &operator+=(Float8E4M3FN &a, Float8E4M3FN b)
{
    a = a + b;
    return a;
}

inline Float8E4M3FN &operator-=(Float8E4M3FN &a, Float8E4M3FN b)
{
    a = a - b;
    return a;
}

inline Float8E4M3FN &operator*=(Float8E4M3FN &a, Float8E4M3FN b)
{
    a = a * b;
    return a;
}

inline Float8E4M3FN &operator/=(Float8E4M3FN &a, Float8E4M3FN b)
{
    a = a / b;
    return a;
}

inline Float8E4M3FN operator++(Float8E4M3FN &a)
{
    a += Float8E4M3FN(1.0f);
    return a;
}

inline Float8E4M3FN operator--(Float8E4M3FN &a)
{
    a -= Float8E4M3FN(1.0f);
    return a;
}

inline Float8E4M3FN operator++(Float8E4M3FN &a, int)
{
    Float8E4M3FN originalValue = a;
    ++a;
    return originalValue;
}

inline Float8E4M3FN operator--(Float8E4M3FN &a, int)
{
    Float8E4M3FN originalValue = a;
    --a;
    return originalValue;
}

} // namespace op

namespace std {

template<>
struct hash<op::Float8E4M3FN> {
    size_t operator()(const op::Float8E4M3FN &v) const
    {
        return hash<float>()(static_cast<float>(v));
    }
};

using op::Float8E4M3FN;

inline bool isinf(const Float8E4M3FN &a [[maybe_unused]])
{
    return false;  // E4M3FN has no infinity
}

inline bool isnan(const Float8E4M3FN &a)
{
    return a.IsNaN();
}

inline bool isfinite(const Float8E4M3FN &a)
{
    return !a.IsNaN();
}

inline Float8E4M3FN abs(const Float8E4M3FN &a)
{
    return Float8E4M3FN(std::abs(static_cast<float>(a)));
}

inline Float8E4M3FN exp(const Float8E4M3FN &a)
{
    return Float8E4M3FN(std::exp(static_cast<float>(a)));
}

inline Float8E4M3FN log(const Float8E4M3FN &a)
{
    return Float8E4M3FN(std::log(static_cast<float>(a)));
}

inline Float8E4M3FN sqrt(const Float8E4M3FN &a)
{
    return Float8E4M3FN(std::sqrt(static_cast<float>(a)));
}

inline Float8E4M3FN pow(const Float8E4M3FN &a, const Float8E4M3FN &b)
{
    return Float8E4M3FN(std::pow(static_cast<float>(a), static_cast<float>(b)));
}

inline Float8E4M3FN sin(const Float8E4M3FN &a)
{
    return Float8E4M3FN(std::sin(static_cast<float>(a)));
}

inline Float8E4M3FN cos(const Float8E4M3FN &a)
{
    return Float8E4M3FN(std::cos(static_cast<float>(a)));
}

inline Float8E4M3FN tanh(const Float8E4M3FN &a)
{
    return Float8E4M3FN(std::tanh(static_cast<float>(a)));
}

inline Float8E4M3FN floor(const Float8E4M3FN &a)
{
    return Float8E4M3FN(std::floor(static_cast<float>(a)));
}

inline Float8E4M3FN ceil(const Float8E4M3FN &a)
{
    return Float8E4M3FN(std::ceil(static_cast<float>(a)));
}

template<>
class numeric_limits<op::Float8E4M3FN> {
public:
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = true;
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
    static constexpr int max_digits10 = 2;
    static constexpr int min_exponent = -6;
    static constexpr int min_exponent10 = -1;
    static constexpr int max_exponent = 9;
    static constexpr int max_exponent10 = 2;
    static constexpr int radix = 2;
    static constexpr float round_error()
    {
        return 0.5f;
    }
    static constexpr float denorm_min()
    {
        return 0.001953125f;  // 2^-9
    }

    static constexpr op::Float8E4M3FN min()
    {
        return op::Float8E4M3FN::MinPositiveNormal();
    }
    static constexpr op::Float8E4M3FN lowest()
    {
        return op::Float8E4M3FN::Lowest();
    }
    static constexpr op::Float8E4M3FN max()
    {
        return op::Float8E4M3FN::Highest();
    }
    static constexpr op::Float8E4M3FN epsilon()
    {
        return op::Float8E4M3FN::Epsilon();
    }
    static constexpr op::Float8E4M3FN infinity()
    {
        return op::Float8E4M3FN::Highest();  // No infinity, return max
    }
    static constexpr op::Float8E4M3FN quiet_NaN()
    {
        return op::Float8E4M3FN(op::Float8E4M3FN::NAN_VALUE, op::Float8E4M3FN::FromBits());
    }
    static constexpr op::Float8E4M3FN signaling_NaN()
    {
        return quiet_NaN();
    }
};

} // namespace std

#endif // OP_API_FLOAT8_E4M3FN_H
