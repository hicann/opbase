/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_HIFP8_H
#define OP_API_HIFP8_H

#include <cmath>
#include <cstring>
#include <limits>
#include <array>

#include "securec.h"
#include "opdev/op_log.h"

namespace op {

// HiFloat8 (HiF8) is an 8-bit floating point format developed by Huawei.
// It uses a tapered precision encoding with variable-length prefix codes.
//
// Format overview:
// - Normal mode: standard binary scientific notation
// - Denormal mode: extended exponent range for small values
// - Exponent range: [-22, 15] (38 exponent values total)
// - Mantissa width varies: 0-3 bits depending on exponent magnitude

struct HiFloat8 {
    struct FromBitsTag {};
    static constexpr FromBitsTag FromBits()
    {
        return FromBitsTag();
    }

    constexpr HiFloat8(uint8_t bits, [[maybe_unused]] FromBitsTag fromBits) : value(bits)
    {
    }

    // Default constructor yields zero value
    HiFloat8() : value(HIF8_ZERO_VALUE)
    {}

    explicit HiFloat8(float f32)
    {
        value = BitsFromFp32(BitCast<uint32_t>(f32));
    }

    // Bit cast utility
    template <typename To, typename From>
    static auto BitCast(const From &src) -> To {
        static_assert(sizeof(To) == sizeof(From), "");
        To dst = 0;
        OP_CHECK(memcpy_s(&dst, sizeof(To), &src, sizeof(From)) == EOK,
            OP_LOGE(ACLNN_ERR_INNER, "call memcpy_s failed."),;);
        return dst;
    }

    // Convert from raw bits
    static constexpr HiFloat8 FromRawBits(uint8_t bits)
    {
        HiFloat8 hif8(bits, FromBits());
        return hif8;
    }

    // Check if the value is NaN
    bool IsNaN() const
    {
        return value == HIF8_NAN_VALUE;
    }

    // Check if the value is Infinity
    bool IsInf() const
    {
        return (value & static_cast<uint8_t>(~HIF8_SIGN_MASK)) == HIF8_INF_VALUE;
    }

    // Check if the value is zero
    bool IsZero() const
    {
        return (value & static_cast<uint8_t>(~HIF8_SIGN_MASK)) == HIF8_ZERO_VALUE;
    }

    static float Hifp8ToFloat(HiFloat8 hif8)
    {
        if (hif8.IsNaN()) {
            return NAN;
        } else if (hif8.IsInf()) {
            return (hif8.value & HIF8_SIGN_MASK) != 0 ? -std::numeric_limits<float>::infinity()
                                                : std::numeric_limits<float>::infinity();
        }

        const float sign = (hif8.value & HIF8_SIGN_MASK) != 0 ? -1.0F : 1.0F;
        const auto pair = DOT_TABLE[(hif8.value & HIF8_DML_MASK) >> HIF8_DOT_OFFSET];
        const uint8_t dot = pair.first;
        const uint8_t width = pair.second;

        if (dot == HIF8_DML_FLAG) {
            const int8_t dml_exp = static_cast<int8_t>(
                hif8.value & static_cast<uint8_t>(~(HIF8_SIGN_MASK | HIF8_DML_MASK)));
            if (dml_exp == 0) {
                return 0.0F;
            }
            return sign * powf(EXP_BASE, static_cast<float>(dml_exp - HIF8_DML_EXP_BIAS));
        }

        const uint8_t mantissa_width = HIF8_MSB_INDEX - (dot + width);
        float mantissa_value = 0.0F;
        for (uint8_t n = 0; n < mantissa_width; n++) {
            if ((hif8.value & static_cast<uint8_t>(1U << n)) == 0) {
                continue;
            }
            mantissa_value += powf(EXP_BASE, -static_cast<float>(mantissa_width - n));
        }

        const int8_t exp = ExtractExponent(dot, mantissa_width, hif8.value);
        return sign * powf(EXP_BASE, static_cast<float>(exp)) * (1.0F + mantissa_value);
    }

    // Convert to float
    operator float() const
    {
        return Hifp8ToFloat(*this);
    }

    // Comparison operators
    friend bool operator==(const HiFloat8 lhs, const HiFloat8 rhs) noexcept
    {
        return lhs.value == rhs.value;
    }

    friend bool operator!=(const HiFloat8 lhs, const HiFloat8 rhs) noexcept
    {
        return lhs.value != rhs.value;
    }

    friend bool operator<(const HiFloat8 lhs, const HiFloat8 rhs) noexcept
    {
        return static_cast<float>(lhs) < static_cast<float>(rhs);
    }

    friend bool operator<=(const HiFloat8 lhs, const HiFloat8 rhs) noexcept
    {
        return static_cast<float>(lhs) <= static_cast<float>(rhs);
    }

    friend bool operator>(const HiFloat8 lhs, const HiFloat8 rhs) noexcept
    {
        return static_cast<float>(lhs) > static_cast<float>(rhs);
    }

    friend bool operator>=(const HiFloat8 lhs, const HiFloat8 rhs) noexcept
    {
        return static_cast<float>(lhs) >= static_cast<float>(rhs);
    }

    uint8_t value;

    // Special values
    static constexpr uint8_t HIF8_NAN_VALUE = 0b10000000;  // NaN representation
    static constexpr uint8_t HIF8_ZERO_VALUE = 0;          // Zero representation
    static constexpr uint8_t HIF8_INF_VALUE = 0b01101111;  // INF representation

    // Internal constants
    static constexpr uint8_t HIF8_SIGN_MASK = 0b10000000;  // Sign mask for external operators
    static constexpr uint8_t HIF8_DML_MASK = 0b01111000;
    static constexpr uint8_t HIF8_DOT_OFFSET = 3;
    static constexpr int8_t HIF8_DML_EXP_BIAS = 23;
    static constexpr uint8_t HIF8_DML_FLAG = static_cast<uint8_t>(-1);
    static constexpr float EXP_BASE = 2.0F;

    // Bit width constants
    static constexpr uint8_t HIF8_BITS = 8;                                  // HiF8 total bits
    static constexpr uint8_t HIF8_MSB_INDEX = HIF8_BITS - 1;                 // HiF8 MSB index (7)
    static constexpr uint8_t UINT32_BITS = 32;                               // uint32_t bit width
    static constexpr uint8_t CLZ_OFFSET_FOR_UINT8 = UINT32_BITS - HIF8_BITS; // __builtin_clz offset for uint8_t (24)

    // Exponent to mantissa width mapping
    static constexpr std::array<uint8_t, 24> HIF8_EXP_TO_MANTISSA_WIDTH = {{
        3, 3, 3, 3,                   // [0, 3]
        2, 2, 2, 2,                   // [4, 7]
        1, 1, 1, 1, 1, 1, 1, 1,       // [8, 15]
        0, 0, 0, 0, 0, 0, 0,          // [16, 22]
        static_cast<uint8_t>(-1),     // [23]
    }};

    // Exponent to Dot prefix mapping
    static constexpr std::array<uint8_t, 16> HIF8_EXP_TO_DOT = {{
        0b0001,                                                               // [0]
        0b0010,                                                               // [1]
        0b0100, 0b0100,                                                       // [2, 3]
        0b1000, 0b1000, 0b1000, 0b1000,                                       // [4, 7]
        0b1100, 0b1100, 0b1100, 0b1100, 0b1100, 0b1100, 0b1100, 0b1100,       // [8, 15]
    }};

    // Dot decoding table: {dot_type, width}
    static const std::array<std::pair<uint8_t, uint8_t>, 16> DOT_TABLE;

    static constexpr int8_t HIF8_DML_EXP_MAX = -16;
    static constexpr int8_t HIF8_NML_EXP_MAX = 15;
    static constexpr int8_t HIF8_DML_EXP_MIN = -22;

    static constexpr uint8_t IEEE_FP32_EXP_BITS = 8;
    static constexpr uint8_t IEEE_FP32_MAN_BITS = 32 - 1 - IEEE_FP32_EXP_BITS;

    // Bit extraction utility
    static uint32_t Extract32(uint32_t u32, uint8_t width, uint8_t offset)
    {
        return (u32 >> offset) & ((1U << width) - 1U);
    }

    // Bit deposition utility
    static uint32_t Deposit32(uint32_t u32, uint8_t width, uint8_t offset, uint32_t val)
    {
        const uint32_t mask = (1U << width) - 1U;
        return (u32 & ~(mask << offset)) | ((val & mask) << offset);
    }

    // Encode exponent and mantissa to HiF8 format
    static uint8_t EncodeHiF8(int8_t exp, uint8_t mantissa)
    {
        if (exp <= HIF8_DML_EXP_MAX) {
            return static_cast<uint8_t>(exp + HIF8_DML_EXP_BIAS);
        }
        const uint8_t mag = static_cast<uint8_t>(exp < 0 ? -exp : exp);
        auto encode_em = [](uint8_t m, uint8_t se_bit) -> uint8_t {
            if (m == 0) {
                return 0;
            }
            // Replace the most-significant bit in `m` to `se_bit`.
            return Deposit32(m, 1, HIF8_MSB_INDEX - (__builtin_clz(m) - CLZ_OFFSET_FOR_UINT8), se_bit);
        };
        return static_cast<uint8_t>(HIF8_EXP_TO_DOT[mag] << HIF8_DOT_OFFSET) |
               static_cast<uint8_t>(encode_em(mag, exp < 0) << HIF8_EXP_TO_MANTISSA_WIDTH[mag]) |
               mantissa;
    }

    // Extract exponent from encoded value
    static int8_t ExtractExponent(uint8_t dot, uint8_t mantissa_width, uint8_t raw)
    {
        if (dot == 0) {
            return 0;
        }
        const uint8_t em = Extract32(raw, dot, mantissa_width);
        const uint8_t se = em & static_cast<uint8_t>(1U << (dot - 1U));
        const uint8_t mag = Deposit32(em, 1, dot - 1U, 1);
        return (se != 0 ? -1 : 1) * static_cast<int8_t>(mag);
    }

    // IEEE floating point type classification
    enum class IeeeType : int8_t {
        ZERO,
        NOT_A_NUMBER,
        POSITIVE_INFINITY,
        ORDINARY,
    };

    // Unpack IEEE floating point value
    template <uint8_t exp_bits, uint8_t mantissa_bits>
    static IeeeType UnpackIeeeFp(uint32_t bits, uint8_t *sign_bit, int8_t *exp, uint32_t *mantissa)
    {
        constexpr uint32_t ieee_sign_bit = 1U << (exp_bits + mantissa_bits);
        constexpr int8_t bias = static_cast<int8_t>((1U << (exp_bits - 1U)) - 1U);

        if ((bits & ~ieee_sign_bit) == 0) {
            return IeeeType::ZERO;
        }

        *sign_bit = (bits & ieee_sign_bit) != 0 ? HIF8_SIGN_MASK : 0U;
        *exp = static_cast<int8_t>(Extract32(bits, exp_bits, mantissa_bits)) - bias;
        *mantissa = Extract32(bits, mantissa_bits, 0);

        // Handle NaN or Inf.
        if (static_cast<uint8_t>(*exp) == (1U << (exp_bits - 1U))) {
            return *mantissa != 0 ? IeeeType::NOT_A_NUMBER : IeeeType::POSITIVE_INFINITY;
        }
        // Handle IEEE subnormal values.
        if (*exp == -bias && *mantissa != 0) {
            const uint8_t shift = mantissa_bits - (UINT32_BITS - __builtin_clz(*mantissa)) + 1;
            *exp = -bias + 1 - shift;
            *mantissa = Extract32(*mantissa << shift, mantissa_bits, 0);
        }
        return IeeeType::ORDINARY;
    }

    // Round mantissa to specified width
    template <uint8_t mantissa_bits>
    static bool RoundMantissa(uint32_t *mantissa, uint32_t probe_bit)
    {
        const uint32_t tmp = *mantissa + probe_bit;
        *mantissa = Extract32(tmp, mantissa_bits, 0);
        return (tmp & (1U << mantissa_bits)) != 0;
    }

    // Convert IEEE floating point bits to HiF8
    template <uint8_t exp_bits, uint8_t mantissa_bits>
    static uint8_t HiF8FromIeeeBits(uint32_t bits)
    {
        uint8_t sign_bit = 0;
        int8_t exp = 0;
        uint32_t mantissa = 0;
        const auto type = UnpackIeeeFp<exp_bits, mantissa_bits>(bits, &sign_bit, &exp, &mantissa);

        if (type == IeeeType::ZERO || (type == IeeeType::ORDINARY && exp < HIF8_DML_EXP_MIN - 1)) {
            return 0;
        } else if (type == IeeeType::NOT_A_NUMBER) {
            return HIF8_NAN_VALUE;
        } else if (type == IeeeType::POSITIVE_INFINITY || (type == IeeeType::ORDINARY && exp > HIF8_NML_EXP_MAX)) {
            return sign_bit | HIF8_INF_VALUE;
        }

        // TA rounding.
        const uint8_t bits_to_discard = mantissa_bits - HIF8_EXP_TO_MANTISSA_WIDTH[static_cast<ssize_t>(std::abs(exp))];
        if (RoundMantissa<mantissa_bits>(&mantissa, 1U << (bits_to_discard - 1U))) {
            exp += 1;
            // Handle overflow after TA rounding.
            if (exp > HIF8_NML_EXP_MAX) {
                return sign_bit | HIF8_INF_VALUE;
            }
        }
        return sign_bit | EncodeHiF8(exp, mantissa >> bits_to_discard);
    }

    // Convert FP32 bits to HiF8 bits
    static uint8_t BitsFromFp32(uint32_t f32)
    {
        return HiF8FromIeeeBits<IEEE_FP32_EXP_BITS, IEEE_FP32_MAN_BITS>(f32);
    }
};

// Static member initialization
inline const std::array<std::pair<uint8_t, uint8_t>, 16> HiFloat8::DOT_TABLE = {{
    { HIF8_DML_FLAG, 4 },                        // 0b0000
    { 0, 4 },                                   // 0b0001
    { 1, 3 }, { 1, 3 },                         // 0b001.
    { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 },     // 0b01..
    { 3, 2 }, { 3, 2 }, { 3, 2 }, { 3, 2 },     // 0b10..
    { 4, 2 }, { 4, 2 }, { 4, 2 }, { 4, 2 },     // 0b11..
}};

// Static constexpr member definitions
constexpr uint8_t HiFloat8::HIF8_ZERO_VALUE;
constexpr uint8_t HiFloat8::HIF8_SIGN_MASK;
constexpr uint8_t HiFloat8::HIF8_INF_VALUE;
constexpr uint8_t HiFloat8::HIF8_DML_MASK;
constexpr uint8_t HiFloat8::HIF8_DOT_OFFSET;
constexpr int8_t HiFloat8::HIF8_DML_EXP_BIAS;
constexpr uint8_t HiFloat8::HIF8_DML_FLAG;
constexpr float HiFloat8::EXP_BASE;
constexpr uint8_t HiFloat8::HIF8_BITS;
constexpr uint8_t HiFloat8::HIF8_MSB_INDEX;
constexpr uint8_t HiFloat8::UINT32_BITS;
constexpr uint8_t HiFloat8::CLZ_OFFSET_FOR_UINT8;
constexpr std::array<uint8_t, 24> HiFloat8::HIF8_EXP_TO_MANTISSA_WIDTH;
constexpr std::array<uint8_t, 16> HiFloat8::HIF8_EXP_TO_DOT;

// Type alias for compatibility
using hif8_t = HiFloat8;

// Arithmetic operators
inline HiFloat8 operator+(HiFloat8 a, HiFloat8 b)
{
    return HiFloat8(static_cast<float>(a) + static_cast<float>(b));
}

inline HiFloat8 operator-(HiFloat8 a, HiFloat8 b)
{
    return HiFloat8(static_cast<float>(a) - static_cast<float>(b));
}

inline HiFloat8 operator*(HiFloat8 a, HiFloat8 b)
{
    return HiFloat8(static_cast<float>(a) * static_cast<float>(b));
}

inline HiFloat8 operator/(HiFloat8 a, HiFloat8 b)
{
    return HiFloat8(static_cast<float>(a) / static_cast<float>(b));
}

inline HiFloat8 operator-(HiFloat8 a)
{
    a.value ^= HiFloat8::HIF8_SIGN_MASK;
    return a;
}

// Compound assignment operators
inline HiFloat8 &operator+=(HiFloat8 &a, HiFloat8 b)
{
    a = a + b;
    return a;
}

inline HiFloat8 &operator-=(HiFloat8 &a, HiFloat8 b)
{
    a = a - b;
    return a;
}

inline HiFloat8 &operator*=(HiFloat8 &a, HiFloat8 b)
{
    a = a * b;
    return a;
}

inline HiFloat8 &operator/=(HiFloat8 &a, HiFloat8 b)
{
    a = HiFloat8(static_cast<float>(a) / static_cast<float>(b));
    return a;
}

// Stream output operator
inline std::ostream &operator<<(std::ostream &os, const HiFloat8 &dt)
{
    os << static_cast<float>(dt);
    return os;
}

static_assert(sizeof(HiFloat8) == sizeof(uint8_t), "sizeof HiFloat8 must be 1");

} // namespace op

namespace std {

template<>
struct hash<op::HiFloat8> {
    size_t operator()(const op::HiFloat8 &v) const
    {
        return hash<float>()(static_cast<float>(v));
    }
};

inline bool isinf(const op::HiFloat8 &a)
{
    return a.IsInf();
}

inline bool isnan(const op::HiFloat8 &a)
{
    return a.IsNaN();
}

inline bool isfinite(const op::HiFloat8 &a)
{
    return !a.IsInf() && !a.IsNaN();
}

inline op::HiFloat8 abs(const op::HiFloat8 &a)
{
    return op::HiFloat8(std::abs(static_cast<float>(a)));
}

inline op::HiFloat8 exp(const op::HiFloat8 &a)
{
    return op::HiFloat8(std::exp(static_cast<float>(a)));
}

inline op::HiFloat8 log(const op::HiFloat8 &a)
{
    return op::HiFloat8(std::log(static_cast<float>(a)));
}

inline op::HiFloat8 sqrt(const op::HiFloat8 &a)
{
    return op::HiFloat8(std::sqrt(static_cast<float>(a)));
}

inline op::HiFloat8 pow(const op::HiFloat8 &a, const op::HiFloat8 &b)
{
    return op::HiFloat8(std::pow(static_cast<float>(a), static_cast<float>(b)));
}

inline op::HiFloat8 sin(const op::HiFloat8 &a)
{
    return op::HiFloat8(std::sin(static_cast<float>(a)));
}

inline op::HiFloat8 cos(const op::HiFloat8 &a)
{
    return op::HiFloat8(std::cos(static_cast<float>(a)));
}

inline op::HiFloat8 tan(const op::HiFloat8 &a)
{
    return op::HiFloat8(std::tan(static_cast<float>(a)));
}

inline op::HiFloat8 tanh(const op::HiFloat8 &a)
{
    return op::HiFloat8(std::tanh(static_cast<float>(a)));
}

inline op::HiFloat8 floor(const op::HiFloat8 &a)
{
    return op::HiFloat8(std::floor(static_cast<float>(a)));
}

inline op::HiFloat8 ceil(const op::HiFloat8 &a)
{
    return op::HiFloat8(std::ceil(static_cast<float>(a)));
}

template<>
class numeric_limits<op::HiFloat8> {
public:
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_exact = false;
    static constexpr bool is_integer = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_modulo = false;
    static constexpr bool is_signed = true;
    static constexpr bool is_specialized = true;
    static constexpr int digits = 3;           // Maximum mantissa bits
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 2;
    static constexpr int min_exponent = -22;   // Minimum exponent
    static constexpr int min_exponent10 = -6;
    static constexpr int max_exponent = 15;    // Maximum exponent
    static constexpr int max_exponent10 = 4;
    static constexpr int radix = 2;

    static constexpr op::HiFloat8 min()
    {
        return op::HiFloat8::FromRawBits(0b00001000);  // Smallest positive normal
    }

    static constexpr op::HiFloat8 lowest()
    {
        return op::HiFloat8::FromRawBits(0b11111111);  // Most negative value
    }

    static constexpr op::HiFloat8 max()
    {
        return op::HiFloat8::FromRawBits(0b01101111);  // Largest positive value (infinity)
    }

    static constexpr op::HiFloat8 epsilon()
    {
        return op::HiFloat8::FromRawBits(0b00000001);  // Machine epsilon
    }

    static constexpr op::HiFloat8 round_error()
    {
        return op::HiFloat8::FromRawBits(0b00011000);  // 0.5f in HiF8 format
    }

    static constexpr op::HiFloat8 infinity()
    {
        return op::HiFloat8::FromRawBits(op::HiFloat8::HIF8_INF_VALUE);
    }

    static constexpr op::HiFloat8 quiet_NaN()
    {
        return op::HiFloat8::FromRawBits(op::HiFloat8::HIF8_NAN_VALUE);
    }

    static constexpr op::HiFloat8 signaling_NaN()
    {
        return op::HiFloat8::FromRawBits(op::HiFloat8::HIF8_NAN_VALUE);
    }

    static constexpr op::HiFloat8 denorm_min()
    {
        return op::HiFloat8::FromRawBits(0b00000001);  // Smallest positive denormal
    }
};

} // namespace std

#endif // OP_API_HIFP8_H
