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
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>
#include "op_common/log/log.h"

namespace {

struct CapturedLog {
    int32_t moduleId;
    int32_t level;
    std::string message;
};

std::vector<int32_t> g_checkedLevels;
std::vector<CapturedLog> g_logs;

void ClearCapturedLogs()
{
    g_checkedLevels.clear();
    g_logs.clear();
}

} // namespace

extern "C" int32_t CheckLogLevel(int32_t moduleId, int32_t level)
{
    EXPECT_EQ(moduleId, OP_MODULE_ID);
    g_checkedLevels.push_back(level);
    return 1;
}

extern "C" __attribute__((format(printf, 3, 4))) void DlogRecord(int32_t moduleId, int32_t level,
                                                                  const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    va_list argsCopy;
    va_copy(argsCopy, args);
    const int messageSize = std::vsnprintf(nullptr, 0, fmt, argsCopy);
    va_end(argsCopy);
    ASSERT_GE(messageSize, 0);

    std::vector<char> buffer(static_cast<size_t>(messageSize) + 1U);
    std::vsnprintf(buffer.data(), buffer.size(), fmt, args);
    va_end(args);
    g_logs.push_back({moduleId, level, std::string(buffer.data(), static_cast<size_t>(messageSize))});
}

class TestOpsBaseLog : public ::testing::Test {
protected:
    void SetUp() override { ClearCapturedLogs(); }
    void TearDown() override { ClearCapturedLogs(); }
};

TEST_F(TestOpsBaseLog, TestLog1)
{
    OP_LOGD("TestContent", "TestContent of value is %d", 1);
    OP_LOGI("TestContent", "TestContent of value is %d", 2);
    OP_LOGW("TestContent", "TestContent of value is %d", 3);

    ASSERT_EQ(g_checkedLevels, (std::vector<int32_t>{DLOG_DEBUG, DLOG_INFO, DLOG_WARN}));
    ASSERT_EQ(g_logs.size(), 3U);
    EXPECT_EQ(g_logs[0].moduleId, OP_MODULE_ID);
    EXPECT_EQ(g_logs[1].moduleId, OP_MODULE_ID);
    EXPECT_EQ(g_logs[2].moduleId, OP_MODULE_ID);
    EXPECT_EQ(g_logs[0].level, DLOG_DEBUG);
    EXPECT_EQ(g_logs[1].level, DLOG_INFO);
    EXPECT_EQ(g_logs[2].level, DLOG_WARN);
    EXPECT_NE(g_logs[0].message.find("OpName:[TestContent]"), std::string::npos);
    EXPECT_NE(g_logs[1].message.find("OpName:[TestContent]"), std::string::npos);
    EXPECT_NE(g_logs[2].message.find("OpName:[TestContent]"), std::string::npos);
    EXPECT_NE(g_logs[0].message.find("TestContent of value is 1"), std::string::npos);
    EXPECT_NE(g_logs[1].message.find("TestContent of value is 2"), std::string::npos);
    EXPECT_NE(g_logs[2].message.find("TestContent of value is 3"), std::string::npos);
}
