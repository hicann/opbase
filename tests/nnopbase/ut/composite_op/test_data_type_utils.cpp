/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/data_type_utils.h"
#include "gtest/gtest.h"

class TestDataTypeUtils : public testing::Test {};

TEST_F(TestDataTypeUtils, TestCanCast)
{
    EXPECT_TRUE(op::CanCast(op::DataType::DT_INT32, op::DataType::DT_FLOAT));
    EXPECT_TRUE(op::CanCast(op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX64));
    EXPECT_TRUE(op::CanCast(op::DataType::DT_BOOL, op::DataType::DT_BOOL));
    EXPECT_FALSE(op::CanCast(op::DataType::DT_FLOAT, op::DataType::DT_INT32));
    EXPECT_FALSE(op::CanCast(op::DataType::DT_COMPLEX64, op::DataType::DT_INT32));
    EXPECT_FALSE(op::CanCast(op::DataType::DT_FLOAT, op::DataType::DT_BOOL));
}

#define TEST_PROMOTE(typeA, typeB, TypeExpected) EXPECT_EQ(TypeExpected, op::PromoteType(typeA, typeB))

TEST_F(TestDataTypeUtils, TestPromoteType)
{
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_UINT8, op::DataType::DT_UINT8);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_INT8, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_INT16, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_INT32, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_INT64, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_BOOL, op::DataType::DT_UINT8);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_BF16, op::DataType::DT_BF16);

    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_INT8, op::DataType::DT_INT8);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_INT16, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_INT32, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_INT64, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_BOOL, op::DataType::DT_INT8);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_BF16, op::DataType::DT_BF16);

    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_UINT8, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_INT16, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_INT64, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_BOOL, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_BF16, op::DataType::DT_BF16);

    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_UINT8, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_INT8, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_INT16, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_INT32, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_BOOL, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_BF16, op::DataType::DT_BF16);

    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_UINT8, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_INT8, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_INT16, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_INT32, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_INT64, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_BOOL, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_BF16, op::DataType::DT_BF16);

    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_UINT8, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_INT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_BOOL, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT);

    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_UINT8, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_INT8, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_INT16, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_INT64, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_BOOL, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_FLOAT);

    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_UINT8, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_INT8, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_INT16, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_INT32, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_INT64, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_BOOL, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_BF16, op::DataType::DT_DOUBLE);

    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_UINT8, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_INT8, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_INT16, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_INT32, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_INT64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_FLOAT16, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_FLOAT, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_BOOL, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_BF16, op::DataType::DT_COMPLEX64);

    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_UINT8, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_INT8, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_INT16, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_INT32, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_INT64, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_FLOAT16, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_FLOAT, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_BOOL, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_BF16, op::DataType::DT_COMPLEX128);

    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_UINT8, op::DataType::DT_UINT8);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_INT8, op::DataType::DT_INT8);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_INT16, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_INT32, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_INT64, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_BOOL, op::DataType::DT_BOOL);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_BF16, op::DataType::DT_BF16);

    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_UINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_INT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_INT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_INT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_INT64, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_FLOAT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_FLOAT, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_DOUBLE, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_COMPLEX64, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_COMPLEX128, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_BOOL, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_QINT8, op::DataType::DT_QINT8);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_BF16, op::DataType::DT_UNDEFINED);

    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_UINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_INT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_INT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_INT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_INT64, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_FLOAT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_FLOAT, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_DOUBLE, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_COMPLEX64, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_COMPLEX128, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_BOOL, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_QINT16, op::DataType::DT_QINT16);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_BF16, op::DataType::DT_UNDEFINED);

    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_UINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_INT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_INT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_INT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_INT64, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_FLOAT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_FLOAT, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_DOUBLE, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_COMPLEX64, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_COMPLEX128, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_BOOL, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_QINT32, op::DataType::DT_QINT32);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_BF16, op::DataType::DT_UNDEFINED);

    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_UINT8, op::DataType::DT_BF16);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_INT8, op::DataType::DT_BF16);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_INT16, op::DataType::DT_BF16);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_INT32, op::DataType::DT_BF16);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_INT64, op::DataType::DT_BF16);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_BOOL, op::DataType::DT_BF16);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_BF16, op::DataType::DT_BF16);
}

TEST_F(TestDataTypeUtils, ToDataType) { EXPECT_EQ(op::ToDataType("DT_FLOAT16"), op::DataType::DT_FLOAT16); }

TEST_F(TestDataTypeUtils, ToString1)
{
    EXPECT_EQ(op::ToString({op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT}), "[DT_FLOAT16,DT_FLOAT,]");
    EXPECT_EQ(op::ToString({}), "[]");
}

TEST_F(TestDataTypeUtils, CheckTypeContainerOverloads)
{
    constexpr op::DataType t1 = op::DataType::DT_INT4;
    constexpr op::DataType t2 = op::DataType::DT_INT8;
    constexpr op::DataType miss = op::DataType::DT_FLOAT;

    // 正常列表 - 命中
    EXPECT_TRUE(op::CheckType(t2, std::array<op::DataType, 2>{t1, t2}));
    EXPECT_TRUE(op::CheckType(t2, std::vector<op::DataType>{t1, t2}));

    // 正常列表 - 未命中
    EXPECT_FALSE(op::CheckType(miss, std::array<op::DataType, 2>{t1, t2}));
    EXPECT_FALSE(op::CheckType(miss, std::vector<op::DataType>{t1, t2}));

    // 空列表
    EXPECT_FALSE(op::CheckType(miss, std::vector<op::DataType>{}));

    // 单元素
    EXPECT_TRUE(op::CheckType(t1, std::vector<op::DataType>{t1}));
}

TEST_F(TestDataTypeUtils, ToStringContainerOverloads)
{
    // initializer_list 版维持现状（含尾随逗号）
    EXPECT_EQ(op::ToString(std::initializer_list<op::DataType>{op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT}),
              "[DT_FLOAT16,DT_FLOAT,]");
    // array / vector 版无尾随逗号
    EXPECT_EQ(op::ToString(std::array<op::DataType, 2>{op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT}),
              "[DT_FLOAT16,DT_FLOAT]");
    EXPECT_EQ(op::ToString(std::vector<op::DataType>{op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT}),
              "[DT_FLOAT16,DT_FLOAT]");
    // 空列表
    EXPECT_EQ(op::ToString(std::vector<op::DataType>{}), "[]");
}

TEST_F(TestDataTypeUtils, TypeSize)
{
    EXPECT_EQ(op::TypeSize(op::internal::DT_MAX), sizeof(op::complex128));
    EXPECT_EQ(op::TypeSize(op::DataType::DT_UNDEFINED), sizeof(op::complex128));
}

TEST_F(TestDataTypeUtils, ShapeBtyes)
{
    int64_t shape = 17;
    int64_t fp16Btyes = op::CalcShapeBytes(shape, op::DataType::DT_FLOAT16);
    EXPECT_EQ(fp16Btyes, 34);

    int64_t int4Btyes = op::CalcShapeBytes(shape, op::DataType::DT_INT4);
    EXPECT_EQ(int4Btyes, 9);

    int64_t int4BtyesCeil = op::CalcShapeBytes(shape, op::DataType::DT_INT4, true);
    EXPECT_EQ(int4BtyesCeil, 8);

    shape = std::numeric_limits<int64_t>::max();
    fp16Btyes = op::CalcShapeBytes(shape, op::DataType::DT_FLOAT16);
    EXPECT_EQ(fp16Btyes, -2);

    int4Btyes = op::CalcShapeBytes(shape, op::DataType::DT_INT4);
    EXPECT_EQ(int4Btyes, 0);
}

TEST_F(TestDataTypeUtils, AdditionalDataTypeConstants)
{
    // 补充常量的取值与 ge 的枚举一致，高低版本头文件下都由本仓常量提供
    EXPECT_EQ(static_cast<int32_t>(op::internal::DT_HIFLOAT4), 42);
    EXPECT_EQ(static_cast<int32_t>(op::internal::DT_HIFLOAT4_SCALE), 43);
    EXPECT_EQ(static_cast<int32_t>(op::internal::DT_MAX), 44);
    EXPECT_EQ(static_cast<int32_t>(op::internal::ACL_HIFLOAT4), 42);
    EXPECT_EQ(static_cast<int32_t>(op::internal::ACL_HIFLOAT4_SCALE), 43);
    EXPECT_EQ(op::ToString(op::internal::DT_HIFLOAT4), "DT_HIFLOAT4");
    EXPECT_EQ(op::ToString(op::internal::DT_HIFLOAT4_SCALE), "DT_HIFLOAT4_SCALE");
    // internal::DT_MAX 不是真实类型，取不到名称
    EXPECT_EQ(op::ToString(op::internal::DT_MAX), "UNDEFINED");
    EXPECT_GT(static_cast<int32_t>(op::internal::DT_MAX), static_cast<int32_t>(op::internal::DT_HIFLOAT4_SCALE));
    EXPECT_FALSE(op::IsNumberType(op::internal::DT_MAX));
    EXPECT_FALSE(op::IsFloatingType(op::internal::DT_MAX));
}

// internal::DT_MAX 是我们类型域的上界（不属于任何真实类型）：涉及下标访问的接口必须拦住它，避免越界
TEST_F(TestDataTypeUtils, DataTypeTableBoundIsOutOfRange)
{
    EXPECT_EQ(op::TypeSize(op::internal::DT_MAX), sizeof(op::complex128));
    EXPECT_EQ(op::PromoteType(op::internal::DT_MAX, op::internal::DT_HIFLOAT4_SCALE), op::DataType::DT_UNDEFINED);
    EXPECT_EQ(op::PromoteType(op::internal::DT_HIFLOAT4, op::internal::DT_MAX), op::DataType::DT_UNDEFINED);
}

TEST_F(TestDataTypeUtils, Hifloat4TypeSize)
{
    // hifloat4 为 4bit，TypeSize 以 kDataTypeSizeBitOffset 编码位宽；hifloat4_scale 每元素 4 字节
    EXPECT_EQ(op::TypeSize(op::internal::DT_HIFLOAT4), op::kDataTypeSizeBitOffset + 4);
    EXPECT_EQ(op::TypeSize(op::internal::DT_HIFLOAT4_SCALE), 4);
}

TEST_F(TestDataTypeUtils, Hifloat4ShapeBtyes)
{
    // 17 个 hifloat4 = 68bit，按字节向上取整为 9 字节
    EXPECT_EQ(op::CalcShapeBytes(17, op::internal::DT_HIFLOAT4), 9);
    EXPECT_EQ(op::CalcShapeBytes(17, op::internal::DT_HIFLOAT4_SCALE), 68);
}

TEST_F(TestDataTypeUtils, Hifloat4ToAclDataType)
{
    EXPECT_EQ(op::ToAclDataType(op::internal::DT_HIFLOAT4), op::internal::ACL_HIFLOAT4);
    EXPECT_EQ(op::ToAclDataType(op::internal::DT_HIFLOAT4_SCALE), op::internal::ACL_HIFLOAT4_SCALE);
    // 未纳入转换列表的类型仍返回 UNDEFINED
    EXPECT_EQ(op::ToAclDataType(op::DataType::DT_QINT8), ACL_DT_UNDEFINED);
}

TEST_F(TestDataTypeUtils, Hifloat4TypePredicates)
{
    EXPECT_TRUE(op::IsNumberType(op::internal::DT_HIFLOAT4));
    EXPECT_TRUE(op::IsNumberType(op::internal::DT_HIFLOAT4_SCALE));
    EXPECT_TRUE(op::IsRealNumberType(op::internal::DT_HIFLOAT4));
    EXPECT_TRUE(op::IsRealNumberType(op::internal::DT_HIFLOAT4_SCALE));
    EXPECT_TRUE(op::IsFloatingType(op::internal::DT_HIFLOAT4));
    EXPECT_TRUE(op::IsFloatingType(op::internal::DT_HIFLOAT4_SCALE));
    // hifloat4_scale 按浮点/数值类型对待，故允许 float 到 hifloat4_scale 的转换判定
    EXPECT_TRUE(op::CanCast(op::DataType::DT_FLOAT, op::internal::DT_HIFLOAT4_SCALE));
    // 浮点转整型仍不允许
    EXPECT_FALSE(op::CanCast(op::internal::DT_HIFLOAT4, op::DataType::DT_INT32));
}
