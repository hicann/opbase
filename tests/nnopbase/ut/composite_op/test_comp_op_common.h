/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_COMPOSITE_OP_TEST_COMP_OP_COMMON_H
#define TESTS_COMPOSITE_OP_TEST_COMP_OP_COMMON_H

#include <cstddef>
#include <cstdlib>
#include <string>

namespace op::internal::test {
// ExpandableRtsArgBuffer 测试用初始容量（远小于生产值，用于加速测试）
constexpr size_t TEST_LAUNCH_ARG_INIT_CAP = 1000;
constexpr size_t TEST_TILING_HOST_DATA_INIT_CAP = 2000;

// ExpandableRtsArgBuffer 扩容测试用极小容量（用于验证扩容触发逻辑）
constexpr size_t TEST_LAUNCH_ARG_EXPAND_CAP = 4;
constexpr size_t TEST_TILING_HOST_DATA_EXPAND_CAP = 4;

class ScopedEnvironmentVariable {
public:
    explicit ScopedEnvironmentVariable(const char* name) : name_(name)
    {
        const char* value = std::getenv(name_.c_str());
        if (value != nullptr) {
            existed_ = true;
            value_ = value;
        }
    }

    ~ScopedEnvironmentVariable()
    {
        if (existed_) {
            setenv(name_.c_str(), value_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
    ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) = delete;

private:
    std::string name_;
    std::string value_;
    bool existed_ {false};
};
} // namespace op::internal::test

#endif
