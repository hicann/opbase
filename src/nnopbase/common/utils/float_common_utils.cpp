/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "float_common_utils.h"

namespace op {
namespace {

constexpr int UINT32_BITS = 32;

struct Fp32Parts {
    uint32_t sign;
    uint32_t exp;
    uint32_t man;
};

struct SmallFloatFields {
    int32_t expField;
    uint32_t manField;
};

Fp32Parts SplitFp32Bits(uint32_t bits)
{
    return Fp32Parts{(bits >> FP32_SIGN_SHIFT_VAL) & 1U, (bits >> FP32_MAN_LEN_VAL) & FP32_EXP_MASK_8BIT,
                     bits & FP32_MAN_MASK_23BIT};
}

// fp32 次正规数没有隐含的 1，需左移补齐并下调指数。
int32_t NormalizeSignificand(const Fp32Parts& parts, uint32_t* significand)
{
    if (parts.exp != 0) {
        *significand = parts.man | FP32_IMPLICIT_1_VAL;
        return static_cast<int32_t>(parts.exp) - FP32_EXP_BIAS_VAL;
    }

    int32_t expUnbiased = 1 - FP32_EXP_BIAS_VAL;
    uint32_t sig = parts.man;
    while ((sig & FP32_IMPLICIT_1_VAL) == 0U) {
        sig <<= 1;
        expUnbiased--;
    }
    *significand = sig;
    return expUnbiased;
}

// 需要丢弃的低位数。次正规结果须额外多丢 (1 - biasedExp) 位，使 1 LSB 与最小次正规
// 2^(1 - bias - manBits) 对齐，从而与正规数共用同一轮 RNE。
int32_t ComputeDiscardShift(int32_t biasedExp, int manBits)
{
    int32_t shift = FP32_MAN_LEN_VAL - manBits;
    if (biasedExp <= 0) {
        shift += (1 - biasedExp);
    }
    return shift;
}

uint32_t RoundToNearestEven(uint32_t significand, int32_t shift)
{
    uint32_t kept = significand >> shift;
    const uint32_t roundBit = (significand >> (shift - 1)) & 1U;
    const uint32_t sticky = (significand & ((1U << (shift - 1)) - 1U)) != 0U ? 1U : 0U;
    if (roundBit != 0U && (sticky != 0U || (kept & 1U) != 0U)) {
        kept++;
    }
    return kept;
}

SmallFloatFields ComposeTargetFields(uint32_t kept, int32_t biasedExp, int manBits)
{
    if (biasedExp < 1) {
        return kept >= (1U << manBits) ? SmallFloatFields{1, 0} : SmallFloatFields{0, kept};
    }

    const uint32_t maxMan = (1U << manBits) - 1U;
    if (kept == (1U << (manBits + 1))) {
        return SmallFloatFields{biasedExp + 1, 0};
    }
    return SmallFloatFields{biasedExp, kept & maxMan};
}

// 由目标格式的特殊值约定推出最大有限值所在的指数域 / 尾数域：
//   NONE     -> (maxExp,     maxMan)      全 1 指数与全 1 尾数都是合法值
//   NAN_ONLY -> (maxExp,     maxMan - 1)  全 1 尾数是 NaN 槽
//   INF_NAN  -> (maxExp - 1, maxMan)      全 1 指数整体保留给 Inf/NaN
void GetMaxFiniteFields(const op::internal::SmallFloatFormat& targetFormat, int32_t* maxFiniteExp,
                        uint32_t* maxFiniteMan)
{
    int32_t exp = (1 << targetFormat.expBits) - 1;
    uint32_t man = (1U << targetFormat.manBits) - 1U;
    switch (targetFormat.encoding) {
        case op::internal::SmallFloatSpecialEncoding::NAN_ONLY:
            man -= 1U;
            break;
        case op::internal::SmallFloatSpecialEncoding::INF_NAN:
            exp -= 1;
            break;
        case op::internal::SmallFloatSpecialEncoding::NONE:
        default:
            break;
    }
    *maxFiniteExp = exp;
    *maxFiniteMan = man;
}

// 溢出统一判定为"舍入结果超过该格式的最大有限值"，最大值的位置由 encoding 决定。
uint8_t EncodeWithOverflowAction(uint32_t signField, const SmallFloatFields& fields,
                                 const op::internal::SmallFloatFormat& targetFormat,
                                 op::internal::SmallFloatOverflowAction overflowAction)
{
    const int32_t maxExp = (1 << targetFormat.expBits) - 1;
    const uint32_t maxMan = (1U << targetFormat.manBits) - 1U;
    const uint32_t expShift = static_cast<uint32_t>(targetFormat.manBits);

    int32_t maxFiniteExp = 0;
    uint32_t maxFiniteMan = 0;
    GetMaxFiniteFields(targetFormat, &maxFiniteExp, &maxFiniteMan);

    const bool overflow = (fields.expField > maxFiniteExp) ||
                          (fields.expField == maxFiniteExp && fields.manField > maxFiniteMan);
    if (!overflow) {
        return static_cast<uint8_t>(signField | (static_cast<uint32_t>(fields.expField) << expShift) |
                                    fields.manField);
    }

    switch (overflowAction) {
        case op::internal::SmallFloatOverflowAction::SATURATE_MAX:
            return static_cast<uint8_t>(signField | (static_cast<uint32_t>(maxFiniteExp) << expShift) |
                                        maxFiniteMan);
        case op::internal::SmallFloatOverflowAction::OVF_NAN:
            // 全 1 指数 + 全 1 尾数（NAN_ONLY / INF_NAN 的 NaN 码）
            return static_cast<uint8_t>(signField | (static_cast<uint32_t>(maxExp) << expShift) | maxMan);
        case op::internal::SmallFloatOverflowAction::OVF_INF:
        default:
            // 全 1 指数 + 尾数 0（INF_NAN 的 Inf 码）
            return static_cast<uint8_t>(signField | (static_cast<uint32_t>(maxExp) << expShift));
    }
}
} // namespace

namespace internal {

uint8_t ConvertFiniteFp32ToSmallFloat(uint32_t fp32Bits, const SmallFloatFormat& targetFormat,
                                      SmallFloatOverflowAction overflowAction)
{
    const Fp32Parts parts = SplitFp32Bits(fp32Bits);
    const uint32_t signField = parts.sign << (targetFormat.expBits + targetFormat.manBits);

    if (parts.exp == 0 && parts.man == 0) {
        return static_cast<uint8_t>(signField);
    }

    uint32_t significand = 0;
    const int32_t expUnbiased = NormalizeSignificand(parts, &significand);
    const int32_t biasedExp = expUnbiased + targetFormat.bias;

    const int32_t shift = ComputeDiscardShift(biasedExp, targetFormat.manBits);
    if (shift >= UINT32_BITS) {
        return static_cast<uint8_t>(signField); // 远低于最小次正规
    }

    const uint32_t kept = RoundToNearestEven(significand, shift);
    return EncodeWithOverflowAction(signField, ComposeTargetFields(kept, biasedExp, targetFormat.manBits),
                                    targetFormat, overflowAction);
}

} // namespace internal
} // namespace op
