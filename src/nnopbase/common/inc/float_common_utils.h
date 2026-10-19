/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPBASE_FLOAT_COMMON_UTILS_H
#define OPBASE_FLOAT_COMMON_UTILS_H

#include <cstdint>

namespace op {

// ============= FP32 format constants =============
constexpr int FP32_EXP_BIAS_VAL = 127;
constexpr int FP32_MAN_LEN_VAL = 23;
constexpr int FP32_SIGN_SHIFT_VAL = 31;
constexpr uint32_t FP32_EXP_MASK_8BIT = 0xFF;
constexpr uint32_t FP32_MAN_MASK_23BIT = 0x7FFFFF;
constexpr uint32_t FP32_IMPLICIT_1_VAL = 0x800000;
constexpr uint32_t FP32_EXP_MAX_VAL = 0xFF;

namespace internal {

// 目标格式对"指数域全 1"的编码约定。它决定最大有限值的位置，以及哪些舍入结果算溢出。
enum class SmallFloatSpecialEncoding {
    NONE,     // 无 NaN/Inf 编码，所有位模式都是有效数（fp4/fp6 等）
    NAN_ONLY, // 仅 NaN：占"指数域全 1 + 尾数全 1"（E4M3FN）
    INF_NAN,  // "指数域全 1"整体保留给 Inf/NaN（E5M2）
};

struct SmallFloatFormat {
    int expBits;
    int manBits;
    int bias;
    SmallFloatSpecialEncoding encoding;
};

// 溢出策略。与格式编码正交：同一格式可在调用点切换不同的溢出动作。
enum class SmallFloatOverflowAction {
    SATURATE_MAX, // 钳制到最大有限幅值并保留符号
    OVF_NAN,      // 溢出 -> NaN（要求 encoding != NONE）
    OVF_INF,      // 溢出 -> Inf（要求 encoding == INF_NAN）
};

// 将有限的 fp32（±0 由本函数处理；±Inf/NaN 必须由调用方先行处理）按 roundTiesToEven 转换为
// 目标格式编码。次正规以 1 LSB = 2^(1 - bias - manBits) 为目标粒度参与同一轮 RNE，
// 不能在正规舍入后再右移截断。
uint8_t ConvertFiniteFp32ToSmallFloat(uint32_t fp32Bits, const SmallFloatFormat& targetFormat,
                                      SmallFloatOverflowAction overflowAction);

} // namespace internal
} // namespace op

#endif // OPBASE_FLOAT_COMMON_UTILS_H
