/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "op_common/op_host/util/math_util.h"

using namespace Ops::Base;
TEST(TestMathUtil, testFloorDiv)
{
    EXPECT_EQ(FloorDiv<int32_t>(10, 3), 3);
    EXPECT_EQ(FloorDiv<int32_t>(87, 20), 4);
    EXPECT_EQ(FloorDiv<int32_t>(1500, 512), 2);
    EXPECT_EQ(FloorDiv<int32_t>(130, 64), 2);
    EXPECT_EQ(FloorDiv<int64_t>(999, 100), 9);
    EXPECT_EQ(FloorDiv<int64_t>(10, 3), 3);
    EXPECT_EQ(FloorDiv<int32_t>(std::numeric_limits<int32_t>::max(), 8), 268435455);
    EXPECT_EQ(FloorDiv<int64_t>(std::numeric_limits<int64_t>::max(), 64), 144115188075855871);
}

TEST(TestMathUtil, testFloorAlign)
{
    EXPECT_EQ(FloorAlign<int32_t>(12345, 4096), 12288);
    EXPECT_EQ(FloorAlign<int32_t>(130, 32), 128);
    EXPECT_EQ(FloorAlign<int32_t>(1023, 64), 960);
    EXPECT_EQ(FloorAlign<int32_t>(87, 20), 80);
    EXPECT_EQ(FloorAlign<int32_t>(185, 60), 180);
    EXPECT_EQ(FloorAlign<int32_t>(17, 4), 16);
    int64_t numX = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) * 2;
    EXPECT_EQ(FloorAlign<int64_t>(numX, 4), 4294967292);
}

TEST(TestMathUtil, testCeilDiv)
{
    EXPECT_EQ(CeilDiv<int32_t>(5000, 4096), 2);
    EXPECT_EQ(CeilDiv<int32_t>(130, 32), 5);
    EXPECT_EQ(CeilDiv<int32_t>(2500, 1024), 3);
    EXPECT_EQ(CeilDiv<int32_t>(999, 100), 10);
    EXPECT_EQ(CeilDiv<int32_t>(87, 20), 5);
    EXPECT_EQ(CeilDiv<int32_t>(std::numeric_limits<int32_t>::max(), 8), 268435456);
    EXPECT_EQ(CeilDiv<int64_t>(std::numeric_limits<int64_t>::max(), 8), 1152921504606846976);
}

TEST(TestMathUtil, testCeilAlign)
{
    EXPECT_EQ(CeilAlign<int32_t>(262144, 64), 262144);
    EXPECT_EQ(CeilAlign<int32_t>(1000, 64), 1024);
    EXPECT_EQ(CeilAlign<int32_t>(257, 32), 288);
    EXPECT_EQ(CeilAlign<int32_t>(1023, 8), 1024);
    EXPECT_EQ(CeilAlign<int32_t>(12345, 4096), 16384);
    EXPECT_EQ(CeilAlign<int32_t>(100000, 512), 100352);
    EXPECT_EQ(CeilAlign<int32_t>(std::numeric_limits<int32_t>::max(), 8), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(CeilAlign<int64_t>(std::numeric_limits<int64_t>::max(), 8), std::numeric_limits<int64_t>::max());
    EXPECT_EQ(CeilAlign<uint64_t>(std::numeric_limits<uint64_t>::max(), 8), std::numeric_limits<uint64_t>::max());
}

namespace {
constexpr int64_t kMaxSplitCount = std::numeric_limits<int32_t>::max();

struct SplitFactorInput {
    int64_t length;
    int32_t factor;
};

class SplitFactorInvalidTest : public testing::TestWithParam<SplitFactorInput> {};

TEST_P(SplitFactorInvalidTest, RejectsRequestWithoutChangingResult)
{
    const auto input = GetParam();
    SplitResult result{19, 23, 29};
    ASSERT_FALSE(SplitIntoEqualByFactor(input.length, input.factor, result));
    EXPECT_EQ(result.splitCount, 19);
    EXPECT_EQ(result.splitFactor, 23);
    EXPECT_EQ(result.splitTailFactor, 29);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidCountsAndInputs, SplitFactorInvalidTest,
    testing::Values(SplitFactorInput{kMaxSplitCount + 1, 1}, SplitFactorInput{2 * kMaxSplitCount + 1, 2},
                    SplitFactorInput{kMaxSplitCount * kMaxSplitCount + 1, std::numeric_limits<int32_t>::max()},
                    SplitFactorInput{std::numeric_limits<int64_t>::max(), std::numeric_limits<int32_t>::max()},
                    SplitFactorInput{0, 1}, SplitFactorInput{-1, 1}, SplitFactorInput{1, 0}, SplitFactorInput{1, -1}));

struct SplitFactorCase {
    int64_t length;
    int32_t factor;
    int32_t count;
    int64_t bodyFactor;
    int64_t tailFactor;
};

class SplitFactorValidTest : public testing::TestWithParam<SplitFactorCase> {};

TEST_P(SplitFactorValidTest, ReturnsCompleteRepresentablePartition)
{
    const auto input = GetParam();
    SplitResult result;
    ASSERT_TRUE(SplitIntoEqualByFactor(input.length, input.factor, result));
    EXPECT_EQ(result.splitCount, input.count);
    EXPECT_EQ(result.splitFactor, input.bodyFactor);
    EXPECT_EQ(result.splitTailFactor, input.tailFactor);
    const int64_t coveredLength = (int64_t{result.splitCount} - 1) * result.splitFactor + result.splitTailFactor;
    EXPECT_EQ(coveredLength, input.length);
}

INSTANTIATE_TEST_SUITE_P(ValidBoundariesAndControls, SplitFactorValidTest,
                         testing::Values(SplitFactorCase{kMaxSplitCount, 1, std::numeric_limits<int32_t>::max(), 1, 1},
                                         SplitFactorCase{
                                             kMaxSplitCount * kMaxSplitCount, std::numeric_limits<int32_t>::max(),
                                             std::numeric_limits<int32_t>::max(), kMaxSplitCount, kMaxSplitCount},
                                         SplitFactorCase{12, 3, 4, 3, 3}, SplitFactorCase{10, 3, 4, 3, 1},
                                         SplitFactorCase{5, 10, 1, 0, 5}, SplitFactorCase{1, 1, 1, 0, 1}));
} // namespace
