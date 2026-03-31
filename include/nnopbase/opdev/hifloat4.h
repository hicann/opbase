/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_HIFLOAT4_H
#define OP_API_HIFLOAT4_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>

namespace op {

/**
 * @brief HiFloat4 floating-point format
 *
 * HiFloat4 is a 4-bit floating-point format designed for high-efficiency
 * neural network computations. It provides an optimal balance between
 * dynamic range and precision for quantized inference and training scenarios.
 *
 * Format: 1 sign bit, 1 exponent bit, 2 mantissa bits
 *         +---+--+---+
 *         | S |E |MM |
 *         +---+--+---+
 *
 * Key characteristics:
 * - Bias: 1
 * - Max normal value: 1.75 (0x7 = 0 1 11, exp=1, man=3 -> 2^0 * 1.75)
 * - Min normal value: -1.75 (0xF = 1 1 11, exp=1, man=3 -> -2^0 * 1.75)
 * - Min positive normal: 2^0 = 1.0 (0x4 = 0 1 00)
 * - Min positive denorm: 2^(1-bias) * (man/4) = 2^0 * 0.25 = 0.25 (0x1 = 0 0 01)
 * - No NaN representation (all bit patterns are valid numbers)
 * - No infinity representation
 *
 * This format is particularly suitable for:
 * - Quantized neural network inference
 * - Low-precision training with gradient scaling
 * - Memory-bandwidth constrained applications
 */
struct HiFloat4 {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits()
    {
        return FromBitsTag();
    }

    uint8_t value;

    // Constants for HiFloat4 format
    // Bit layout: bit3=sign, bit2=exp, bits1-0=man
    static constexpr int EXP_BITS = 1;
    static constexpr int MAN_BITS = 2;
    static constexpr int EXP_BIAS = 1;
    static constexpr uint8_t SIGN_MASK = 0x08;      // 1 0 00 (bit 3)
    static constexpr uint8_t EXP_MASK = 0x04;       // 0 1 00 (bit 2)
    static constexpr uint8_t MAN_MASK = 0x03;       // 0 0 11 (bits 1-0)
    static constexpr uint8_t MAX_EXP = 0x01;        // 1

    // HiFloat4 format semantic constants
    static constexpr int SIGN_SHIFT = 3;                    // Sign bit position
    static constexpr uint8_t ABS_VALUE_MASK = 0x07;         // Mask excluding sign bit
    static constexpr uint8_t HIGHEST_VALUE = 0x07;          // Max value 1.75
    static constexpr uint8_t LOWEST_VALUE = 0x0F;           // Min value -1.75
    static constexpr uint8_t MIN_POS_NORMAL_VALUE = 0x04;   // Min positive normal 1.0
    static constexpr uint8_t BITS_MASK = 0x0F;              // 4-bit effective mask
    static constexpr uint8_t EPSILON_VALUE = 0x01;          // Epsilon value

    // ============= FP32 format constants =============
    static constexpr int FP32_EXP_BIAS_VAL = 127;
    static constexpr int FP32_MAN_LEN_VAL = 23;
    static constexpr int FP32_SIGN_SHIFT_VAL = 31;
    static constexpr uint32_t FP32_EXP_MASK_8BIT = 0xFF;
    static constexpr uint32_t FP32_MAN_MASK_23BIT = 0x7FFFFF;
    static constexpr uint32_t FP32_IMPLICIT_1_VAL = 0x800000;

    // Default constructor - initialize to zero
    HiFloat4() : value(0) {}

    constexpr HiFloat4(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits & BITS_MASK)
    {
    }

    HiFloat4(float v)
    {
        value = FloatToHiFloat4(v).value;
    }

    operator float() const
    {
        return HiFloat4ToFloat(*this);
    }

    operator double() const
    {
        return static_cast<double>(HiFloat4ToFloat(*this));
    }

    static constexpr HiFloat4 Epsilon()
    {
        // 2^-2 = 0.25 (the difference between 1.0 and next representable value)
        return HiFloat4(EPSILON_VALUE, FromBits());  // 0 0 01 = denorm 0.25
    }

    static constexpr HiFloat4 Highest()
    {
        // exp=1, man=3 -> 2^0 * 1.75 = 1.75 (0 1 11 = 0x7)
        return HiFloat4(HIGHEST_VALUE, FromBits());
    }

    static constexpr HiFloat4 Lowest()
    {
        // 1 1 11 = sign=1, exp=1, man=3 -> -2^0 * 1.75 = -1.75
        return HiFloat4(LOWEST_VALUE, FromBits());  // 1 1 11 = -1.75
    }

    static constexpr HiFloat4 MinPositiveNormal()
    {
        // 2^0 = 1.0 (0 1 00)
        return HiFloat4(MIN_POS_NORMAL_VALUE, FromBits());
    }

    bool IsZero() const
    {
        return (value & ABS_VALUE_MASK) == 0;  // 排除符号位
    }

    bool IsNaN() const
    {
        // HiFloat4 has no NaN representation
        // All bit patterns represent valid numbers
        return false;
    }

    bool IsInf() const
    {
        // HiFloat4 has no infinity representation
        return false;
    }

private:
    union FP32 {
        uint32_t u;
        float f;
    };

    static float HiFloat4ToFloat(HiFloat4 fp4)
    {
        if (fp4.IsZero()) {
            return (fp4.value & SIGN_MASK) ? -0.0f : 0.0f;
        }

        uint32_t sign = (fp4.value & SIGN_MASK) >> SIGN_SHIFT;
        uint32_t exp = (fp4.value & EXP_MASK) >> MAN_BITS;
        uint32_t man = fp4.value & MAN_MASK;

        // FP32: 1 sign, 8 exp (bias 127), 23 man
        // HiFloat4: 1 sign, 1 exp (bias 1), 2 man

        int32_t fp32_exp;
        uint32_t fp32_man = man << (FP32_MAN_LEN_VAL - MAN_BITS);

        if (exp == 0) {
            // Denormalized number
            // HiFloat4: value = 2^(1-bias) * (man/4) = 2^0 * (man/4) = man/4
            if (man != 0) {
                // Denorm values: 0.25, 0.5, 0.75 for man=1,2,3
                // Find leading 1 position for normalization
                int shift = 0;
                uint32_t temp_man = man;
                while ((temp_man & (1 << MAN_BITS)) == 0 && shift < MAN_BITS) {
                    temp_man <<= 1;
                    shift++;
                }
                // Effective exponent for denorm is 1-bias=0
                // fp32_exp = FP32_BIAS + (1-bias) - shift = 127 - shift
                fp32_exp = 1 - EXP_BIAS + FP32_EXP_BIAS_VAL - shift;
                fp32_man = (temp_man & MAN_MASK) << (FP32_MAN_LEN_VAL - MAN_BITS);
            } else {
                fp32_exp = 0;
            }
        } else {
            // Normal number: value = 2^(1-1) * (1 + man/4) = (1 + man/4)
            fp32_exp = 1 - EXP_BIAS + FP32_EXP_BIAS_VAL;  // 127
            fp32_man |= FP32_IMPLICIT_1_VAL;  // Add implicit 1
        }

        FP32 result;
        result.u = (sign << FP32_SIGN_SHIFT_VAL) | ((fp32_exp & FP32_EXP_MASK_8BIT) << FP32_MAN_LEN_VAL) | (fp32_man & FP32_MAN_MASK_23BIT);
        return result.f;
    }

    static HiFloat4 FloatToHiFloat4(float f)
    {
        // HiFloat4 has no NaN, clamp NaN to positive max value (1.75)
        // NaN has no meaningful sign, so always clamp to +1.75
        if (std::isnan(f)) {
            // 0x07 = 0 11 1 = 2^1 * 1.75 = 1.75
            return HiFloat4(HIGHEST_VALUE, FromBits());
        }

        FP32 fp32;
        fp32.f = f;
        uint32_t sign = (fp32.u >> FP32_SIGN_SHIFT_VAL) & 1;
        uint32_t exp = (fp32.u >> FP32_MAN_LEN_VAL) & FP32_EXP_MASK_8BIT;
        uint32_t man = fp32.u & FP32_MAN_MASK_23BIT;

        if (exp == 0 && man == 0) {
            // Zero
            return HiFloat4(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
        }

        if (std::isinf(f)) {
            // HiFloat4 has no infinity, clamp to max value
            // 0x07 = +1.75, 0x0F = -1.75
            return HiFloat4(static_cast<uint8_t>((sign << SIGN_SHIFT) | HIGHEST_VALUE), FromBits());
        }

        // Calculate HiFloat4 exponent
        int32_t fp4_exp;
        uint32_t fp4_man;

        if (exp == 0) {
            // Denormalized FP32 input
            fp4_exp = 1 - FP32_EXP_BIAS_VAL + EXP_BIAS;
            while ((man & FP32_IMPLICIT_1_VAL) == 0) {
                man <<= 1;
                fp4_exp--;
            }
            man &= FP32_MAN_MASK_23BIT;
        } else {
            fp4_exp = static_cast<int32_t>(exp) - FP32_EXP_BIAS_VAL + EXP_BIAS;
        }

        // Round mantissa from 23 bits to 2 bits with round-to-nearest-even
        int mantissa_shift = FP32_MAN_LEN_VAL - MAN_BITS;
        uint32_t man_round_bit = (man >> (mantissa_shift - 1)) & 1;
        uint32_t man_sticky_bits = (man & ((1 << (mantissa_shift - 1)) - 1)) ? 1 : 0;
        uint32_t man_lsb = (man >> mantissa_shift) & 1;

        fp4_man = man >> mantissa_shift;
        if (man_round_bit && (man_sticky_bits || man_lsb)) {
            fp4_man++;
            if (fp4_man > MAN_MASK) {
                fp4_man = 0;
                fp4_exp++;
            }
        }

        // Handle overflow/underflow
        if (fp4_exp <= 0) {
            if (fp4_exp < -MAN_BITS) {
                return HiFloat4(static_cast<uint8_t>(sign << SIGN_SHIFT), FromBits());
            }
            fp4_man = (fp4_man | (1 << MAN_BITS)) >> (1 - fp4_exp);
            fp4_exp = 0;
        } else if (fp4_exp >= static_cast<int32_t>(MAX_EXP + 1)) {
            // Overflow - for HiFloat4, max exp is 1
            // HiFloat4 has no NaN, clamp to max value (preserving sign)
            // 0x07 = +1.75, 0x0F = -1.75
            return HiFloat4(static_cast<uint8_t>((sign << SIGN_SHIFT) | HIGHEST_VALUE), FromBits());
        }

        uint8_t result = static_cast<uint8_t>((sign << SIGN_SHIFT) | (fp4_exp << MAN_BITS) | (fp4_man & MAN_MASK));
        return HiFloat4(result, FromBits());
    }
};

inline std::ostream &operator<<(std::ostream &os, const HiFloat4 &dt)
{
    os << static_cast<float>(dt);
    return os;
}

inline HiFloat4 operator+(HiFloat4 a, HiFloat4 b)
{
    return HiFloat4(static_cast<float>(a) + static_cast<float>(b));
}

inline HiFloat4 operator-(HiFloat4 a, HiFloat4 b)
{
    return HiFloat4(static_cast<float>(a) - static_cast<float>(b));
}

inline HiFloat4 operator*(HiFloat4 a, HiFloat4 b)
{
    return HiFloat4(static_cast<float>(a) * static_cast<float>(b));
}

inline HiFloat4 operator/(HiFloat4 a, HiFloat4 b)
{
    return HiFloat4(static_cast<float>(a) / static_cast<float>(b));
}

inline HiFloat4 operator-(HiFloat4 a)
{
    a.value ^= HiFloat4::SIGN_MASK;
    return a;
}

inline bool operator==(HiFloat4 a, HiFloat4 b)
{
    return a.value == b.value;
}

inline bool operator!=(HiFloat4 a, HiFloat4 b)
{
    return a.value != b.value;
}

inline bool operator<(HiFloat4 a, HiFloat4 b)
{
    return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator<=(HiFloat4 a, HiFloat4 b)
{
    return static_cast<float>(a) <= static_cast<float>(b);
}

inline bool operator>(HiFloat4 a, HiFloat4 b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}

inline bool operator>=(HiFloat4 a, HiFloat4 b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}

inline HiFloat4 &operator+=(HiFloat4 &a, HiFloat4 b)
{
    a = a + b;
    return a;
}

inline HiFloat4 &operator-=(HiFloat4 &a, HiFloat4 b)
{
    a = a - b;
    return a;
}

inline HiFloat4 &operator*=(HiFloat4 &a, HiFloat4 b)
{
    a = a * b;
    return a;
}

inline HiFloat4 &operator/=(HiFloat4 &a, HiFloat4 b)
{
    a = a / b;
    return a;
}

inline HiFloat4 operator++(HiFloat4 &a)
{
    a += HiFloat4(1.0f);
    return a;
}

inline HiFloat4 operator--(HiFloat4 &a)
{
    a -= HiFloat4(1.0f);
    return a;
}

inline HiFloat4 operator++(HiFloat4 &a, int)
{
    HiFloat4 original_value = a;
    ++a;
    return original_value;
}

inline HiFloat4 operator--(HiFloat4 &a, int)
{
    HiFloat4 original_value = a;
    --a;
    return original_value;
}

} // namespace op

namespace std {

template<>
struct hash<op::HiFloat4> {
    size_t operator()(const op::HiFloat4 &v) const
    {
        return hash<float>()(static_cast<float>(v));
    }
};

using op::HiFloat4;

inline bool isinf(const HiFloat4 &a [[maybe_unused]])
{
    return false;  // HiFloat4 has no infinity
}

inline bool isnan(const HiFloat4 &a)
{
    return a.IsNaN();
}

inline bool isfinite(const HiFloat4 &a)
{
    return !a.IsNaN();
}

inline HiFloat4 abs(const HiFloat4 &a)
{
    return HiFloat4(std::abs(static_cast<float>(a)));
}

inline HiFloat4 exp(const HiFloat4 &a)
{
    return HiFloat4(std::exp(static_cast<float>(a)));
}

inline HiFloat4 log(const HiFloat4 &a)
{
    return HiFloat4(std::log(static_cast<float>(a)));
}

inline HiFloat4 sqrt(const HiFloat4 &a)
{
    return HiFloat4(std::sqrt(static_cast<float>(a)));
}

inline HiFloat4 pow(const HiFloat4 &a, const HiFloat4 &b)
{
    return HiFloat4(std::pow(static_cast<float>(a), static_cast<float>(b)));
}

inline HiFloat4 sin(const HiFloat4 &a)
{
    return HiFloat4(std::sin(static_cast<float>(a)));
}

inline HiFloat4 cos(const HiFloat4 &a)
{
    return HiFloat4(std::cos(static_cast<float>(a)));
}

inline HiFloat4 tanh(const HiFloat4 &a)
{
    return HiFloat4(std::tanh(static_cast<float>(a)));
}

inline HiFloat4 floor(const HiFloat4 &a)
{
    return HiFloat4(std::floor(static_cast<float>(a)));
}

inline HiFloat4 ceil(const HiFloat4 &a)
{
    return HiFloat4(std::ceil(static_cast<float>(a)));
}

template<>
class numeric_limits<op::HiFloat4> {
public:
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;
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
    static constexpr int min_exponent = 0;
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 0;
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

    static constexpr op::HiFloat4 min()
    {
        return op::HiFloat4::MinPositiveNormal();
    }
    static constexpr op::HiFloat4 lowest()
    {
        return op::HiFloat4::Lowest();
    }
    static constexpr op::HiFloat4 max()
    {
        return op::HiFloat4::Highest();
    }
    static constexpr op::HiFloat4 epsilon()
    {
        return op::HiFloat4::Epsilon();
    }
    static constexpr op::HiFloat4 infinity()
    {
        return op::HiFloat4::Highest();  // No infinity, use max
    }
    static constexpr op::HiFloat4 quiet_NaN()
    {
        // HiFloat4 has no NaN, return zero as fallback
        return op::HiFloat4(0x00, op::HiFloat4::FromBits());
    }
    static constexpr op::HiFloat4 signaling_NaN()
    {
        return quiet_NaN();
    }
};

} // namespace std

#endif // OP_API_HIFLOAT4_H
