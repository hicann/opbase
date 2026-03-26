/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_FLOAT8_E8M0_H
#define OP_API_FLOAT8_E8M0_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>

#include "opdev/fp16_t.h"
#include "opdev/bfloat16.h"

namespace op {

/**
 * @brief FP8 E8M0 floating-point format (scale format)
 *
 * Format: 8 exponent bits, 0 mantissa bits (unsigned)
 *         +--------+
 *         | EEEEEEEE|
 *         +--------+
 *
 * This is a special format used for scaling factors in MX formats.
 * It represents only positive powers of 2: value = 2^(exponent - bias)
 *
 * Follows OCP Microscaling Formats (MX) v1.0 specification.
 *
 * - Bias: 127
 * - Max value: 2^127 (0xFE, exponent=254)
 * - Min positive value: 2^-126 (0x01, exponent=1)
 * - NaN: 0xFF (exponent=255) - reserved for NaN
 * - Zero is represented by 0x00 (exponent=0)
 * - No infinity encoding (only NaN at 0xFF)
 */
struct Float8E8M0 {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits()
    {
        return FromBitsTag();
    }

    uint8_t value;

    // Constants for E8M0 format (unsigned, 8 exponent bits)
    static constexpr int EXP_BITS = 8;
    static constexpr int MAN_BITS = 0;
    static constexpr int EXP_BIAS = 127;
    static constexpr uint8_t EXP_MASK = 0xFF;       // 11111111 (all 8 bits for exponent)
    static constexpr uint8_t MAX_EXP = 0xFE;        // 254 (max normal exponent)
    static constexpr uint8_t NAN_EXP = 0xFF;        // 255 (NaN)
    static constexpr uint8_t ZERO_EXP = 0x00;       // 0 (zero)
    static constexpr uint8_t NAN_VALUE = 0xFF;      // 255 (NaN only, no infinity encoding)

    // Default constructor - initialize to zero
    Float8E8M0() : value(0) {}

    constexpr Float8E8M0(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits)
    {
    }

    Float8E8M0(float v)
    {
        value = float_to_float8_e8m0(v).value;
    }

    template<typename T>
    Float8E8M0 &operator=(T other)
    {
        value = float_to_float8_e8m0(static_cast<float>(other)).value;
        return *this;
    }

    operator float() const
    {
        return float8_e8m0_to_float(*this);
    }

    operator double() const
    {
        return static_cast<double>(float8_e8m0_to_float(*this));
    }

    Float8E8M0 &operator=(const double &dVal)
    {
        value = float_to_float8_e8m0(static_cast<float>(dVal)).value;
        return *this;
    }

    // Conversion to fp16_t
    operator fp16_t() const
    {
        return fp16_t(float8_e8m0_to_float(*this));
    }

    // Conversion to bfloat16
    operator bfloat16() const
    {
        return bfloat16(static_cast<float>(*this));
    }

    static constexpr Float8E8M0 Epsilon()
    {
        // epsilon = nextafter(1.0) - 1.0 = 2.0 - 1.0 = 1.0
        return Float8E8M0(0x7F, FromBits());  // 2^0 = 1.0
    }

    static constexpr Float8E8M0 Highest()
    {
        // 2^127 (exponent=254)
        return Float8E8M0(0xFE, FromBits());
    }

    static constexpr Float8E8M0 Lowest()
    {
        // E8M0 is unsigned, lowest positive is MinPositive
        return Float8E8M0::MinPositive();
    }

    static constexpr Float8E8M0 MinPositive()
    {
        // 2^-126 (exponent=1)
        return Float8E8M0(0x01, FromBits());
    }

    static constexpr Float8E8M0 One()
    {
        // 2^0 = 1.0 (exponent=127)
        return Float8E8M0(0x7F, FromBits());
    }

    bool IsZero() const
    {
        return value == 0;
    }

    bool IsNaN() const
    {
        // NaN: exponent = 255
        return value == NAN_VALUE;
    }

    bool IsInf() const
    {
        // E8M0 has no infinity encoding, only NaN at 0xFF
        return false;
    }

    // Get the exponent value (unbiased)
    int32_t GetExponent() const
    {
        if (IsZero() || IsNaN() || IsInf()) {
            return 0;
        }
        return static_cast<int32_t>(value) - EXP_BIAS;
    }

private:
    static float float8_e8m0_to_float(Float8E8M0 fp8)
    {
        if (fp8.IsZero()) {
            return 0.0f;
        }

        if (fp8.IsNaN()) {
            return std::nanf("");
        }

        // E8M0 has no infinity encoding, only NaN at 0xFF

        uint32_t exp = fp8.value;

        // E8M0: value = 2^(exp - 127)
        // FP32: value = 2^(exp - 127) with mantissa = 0
        // For E8M0, the FP32 exponent equals the E8M0 exponent (same bias)

        union {
            uint32_t u;
            float f;
        } result;
        result.u = (exp << 23);  // sign=0, exponent=exp, mantissa=0
        return result.f;
    }

    static Float8E8M0 float_to_float8_e8m0(float f)
    {
        if (std::isnan(f)) {
            return Float8E8M0(NAN_VALUE, FromBits());
        }

        if (f == 0.0f) {
            return Float8E8M0(0x00, FromBits());
        }

        // E8M0 is unsigned, negative values are not representable
        if (f < 0.0f) {
            return Float8E8M0(NAN_VALUE, FromBits());
        }

        // E8M0 has no infinity encoding, clamp to max value (0xFE)
        if (std::isinf(f)) {
            return Float8E8M0(0xFE, FromBits());
        }

        // E8M0 is unsigned, negative values are not representable
        if (f < 0.0f) {
            return Float8E8M0(NAN_VALUE, FromBits());
        }

        // E8M0 can only represent powers of 2
        // Find the exponent such that 2^exp is closest to f
        union {
            uint32_t u;
            float f;
        } fp32;
        fp32.f = f;

        uint32_t fp32Exp = (fp32.u >> 23) & 0xFF;
        uint32_t fp32Man = fp32.u & 0x7FFFFF;

        // Round to nearest power of 2
        // If mantissa >= 0x400000 (0.5), round up the exponent
        if (fp32Man >= 0x400000) {
            fp32Exp++;
        }

        // Handle overflow
        if (fp32Exp > 254) {
            // E8M0 has no infinity encoding, clamp to max value (0xFE)
            return Float8E8M0(0xFE, FromBits());
        }

        // Handle underflow (fp32Exp == 0 means denormal or zero)
        if (fp32Exp == 0) {
            return Float8E8M0(0x00, FromBits());
        }

        // E8M0 stores only the exponent (same bias as FP32)
        return Float8E8M0(static_cast<uint8_t>(fp32Exp), FromBits());
    }
};

inline std::ostream &operator<<(std::ostream &os, const Float8E8M0 &dt)
{
    os << static_cast<float>(dt);
    return os;
}

inline Float8E8M0 operator*(Float8E8M0 a, Float8E8M0 b)
{
    // Multiplying two powers of 2: 2^a * 2^b = 2^(a+b)
    if (a.IsZero() || b.IsZero()) {
        return Float8E8M0(0x00, Float8E8M0::FromBits());
    }
    if (a.IsNaN() || b.IsNaN()) {
        return Float8E8M0(Float8E8M0::NAN_VALUE, Float8E8M0::FromBits());
    }

    int32_t exp_a = a.GetExponent();
    int32_t exp_b = b.GetExponent();
    int32_t exp_result = exp_a + exp_b + Float8E8M0::EXP_BIAS;

    if (exp_result <= 0) {
        return Float8E8M0(0x00, Float8E8M0::FromBits());
    }
    if (exp_result >= 255) {
        // Overflow to max value (E8M0 has no infinity encoding)
        return Float8E8M0(0xFE, Float8E8M0::FromBits());
    }

    return Float8E8M0(static_cast<uint8_t>(exp_result), Float8E8M0::FromBits());
}

inline Float8E8M0 operator/(Float8E8M0 a, Float8E8M0 b)
{
    // Dividing two powers of 2: 2^a / 2^b = 2^(a-b)
    // Check NaN first (highest priority)
    if (a.IsNaN() || b.IsNaN()) {
        return Float8E8M0(Float8E8M0::NAN_VALUE, Float8E8M0::FromBits());
    }
    // Division by zero (including 0/0): E8M0 has no infinity, return NaN
    if (b.IsZero()) {
        return Float8E8M0(Float8E8M0::NAN_VALUE, Float8E8M0::FromBits());
    }
    // Zero divided by non-zero: result is zero
    if (a.IsZero()) {
        return Float8E8M0(0x00, Float8E8M0::FromBits());
    }
    // E8M0 has no infinity encoding, IsInf() always returns false

    int32_t exp_a = a.GetExponent();
    int32_t exp_b = b.GetExponent();
    int32_t exp_result = exp_a - exp_b + Float8E8M0::EXP_BIAS;

    if (exp_result <= 0) {
        return Float8E8M0(0x00, Float8E8M0::FromBits());
    }
    if (exp_result >= 255) {
        // Overflow: E8M0 has no infinity encoding, clamp to max value
        return Float8E8M0(0xFE, Float8E8M0::FromBits());
    }

    return Float8E8M0(static_cast<uint8_t>(exp_result), Float8E8M0::FromBits());
}

inline Float8E8M0 operator-([[maybe_unused]] Float8E8M0 a)
{
    // E8M0 is unsigned, negation is not defined
    return Float8E8M0(Float8E8M0::NAN_VALUE, Float8E8M0::FromBits());
}

inline bool operator==(Float8E8M0 a, Float8E8M0 b)
{
    return a.value == b.value;
}

inline bool operator!=(Float8E8M0 a, Float8E8M0 b)
{
    return a.value != b.value;
}

inline bool operator<(Float8E8M0 a, Float8E8M0 b)
{
    return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator<=(Float8E8M0 a, Float8E8M0 b)
{
    return static_cast<float>(a) <= static_cast<float>(b);
}

inline bool operator>(Float8E8M0 a, Float8E8M0 b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}

inline bool operator>=(Float8E8M0 a, Float8E8M0 b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}

inline Float8E8M0 &operator*=(Float8E8M0 &a, Float8E8M0 b)
{
    a = a * b;
    return a;
}

inline Float8E8M0 &operator/=(Float8E8M0 &a, Float8E8M0 b)
{
    a = a / b;
    return a;
}

} // namespace op

namespace std {

template<>
struct hash<op::Float8E8M0> {
    size_t operator()(const op::Float8E8M0 &v) const
    {
        return hash<float>()(static_cast<float>(v));
    }
};

using op::Float8E8M0;

inline bool isinf(const Float8E8M0 &a)
{
    return a.IsInf();
}

inline bool isnan(const Float8E8M0 &a)
{
    return a.IsNaN();
}

inline bool isfinite(const Float8E8M0 &a)
{
    // E8M0 has no infinity encoding, only NaN at 0xFF
    // Zero (0x00) is a valid finite value in this context
    return !a.IsNaN();
}

inline Float8E8M0 abs(const Float8E8M0 &a)
{
    // E8M0 is unsigned, so abs is identity
    return a;
}

inline Float8E8M0 sqrt(const Float8E8M0 &a)
{
    if (a.IsNaN()) {
        return Float8E8M0(Float8E8M0::NAN_VALUE, Float8E8M0::FromBits());
    }
    if (a.IsZero()) {
        return Float8E8M0(0x00, Float8E8M0::FromBits());
    }
    if (a.IsInf()) {
        return a;
    }
    // sqrt(2^n) = 2^(n/2)
    int32_t exp = a.GetExponent();
    if (exp % 2 != 0) {
        exp++;  // Round up for odd exponents
    }
    int32_t new_exp = exp / 2 + Float8E8M0::EXP_BIAS;
    return Float8E8M0(static_cast<uint8_t>(new_exp), Float8E8M0::FromBits());
}

inline Float8E8M0 pow(const Float8E8M0 &a, int n)
{
    if (a.IsNaN()) {
        return a;
    }
    if (a.IsZero()) {
        return Float8E8M0(0x00, Float8E8M0::FromBits());
    }
    // (2^e)^n = 2^(e*n)
    int32_t exp = a.GetExponent();
    int32_t new_exp = exp * n + Float8E8M0::EXP_BIAS;

    if (new_exp <= 0) {
        return Float8E8M0(0x00, Float8E8M0::FromBits());
    }
    if (new_exp >= 255) {
        // Overflow: E8M0 has no infinity encoding, clamp to max value
        return Float8E8M0(0xFE, Float8E8M0::FromBits());
    }

    return Float8E8M0(static_cast<uint8_t>(new_exp), Float8E8M0::FromBits());
}

template<>
class numeric_limits<op::Float8E8M0> {
public:
    static constexpr bool has_infinity = false;  // E8M0 has no infinity encoding
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_exact = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_modulo = false;
    static constexpr bool is_signed = false;  // E8M0 is unsigned
    static constexpr bool is_specialized = true;
    static constexpr int digits = 0;  // No mantissa bits
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 0;
    static constexpr int min_exponent = -126;
    static constexpr int min_exponent10 = -38;
    static constexpr int max_exponent = 127;
    static constexpr int max_exponent10 = 38;
    static constexpr int radix = 2;
    static constexpr float round_error()
    {
        return 1.0f;  // No fractional precision
    }
    static constexpr float denorm_min()
    {
        return 0.0f;  // No denormals
    }

    static constexpr op::Float8E8M0 min()
    {
        return op::Float8E8M0::MinPositive();
    }
    static constexpr op::Float8E8M0 lowest()
    {
        return op::Float8E8M0::MinPositive();  // Unsigned, lowest = min
    }
    static constexpr op::Float8E8M0 max()
    {
        return op::Float8E8M0::Highest();
    }
    static constexpr op::Float8E8M0 epsilon()
    {
        return op::Float8E8M0::Epsilon();
    }
    static constexpr op::Float8E8M0 infinity()
    {
        // E8M0 has no infinity encoding, return max value
        return op::Float8E8M0::Highest();
    }
    static constexpr op::Float8E8M0 quiet_NaN()
    {
        return op::Float8E8M0(op::Float8E8M0::NAN_VALUE, op::Float8E8M0::FromBits());
    }
    static constexpr op::Float8E8M0 signaling_NaN()
    {
        return quiet_NaN();
    }
};

} // namespace std

#endif // OP_API_FLOAT8_E8M0_H
