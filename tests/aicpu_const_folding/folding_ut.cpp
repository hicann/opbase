/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../../src/aicpu_const_folding/folding.cc"

namespace {
using RegisteredOp = std::tuple<std::string, ge::OpBackend, ge::OpRegistrationPriority, ge::OpEngine>;

std::vector<RegisteredOp>& RegisteredOps()
{
    static std::vector<RegisteredOp> registered_ops;
    return registered_ops;
}

class FoldingTest : public testing::Test {
protected:
    void SetUp() override
    {
        g_initialized = false;
        g_v2_op_index.clear();
        g_v2_bindings.clear();
        RegisteredOps().clear();
    }
};
} // namespace

namespace ge {
CustomOpCreatorRegister::CustomOpCreatorRegister(const AscendString& op_type, const OpBackend backend,
                                                 const OpRegistrationPriority priority, const OpEngine engine,
                                                 const BaseOpCreator& creator)
{
    if (creator != nullptr) {
        RegisteredOps().emplace_back(op_type.GetString(), backend, priority, engine);
    }
}
} // namespace ge

namespace gert {
Tensor* HostCpuOpExecutionContext::MallocOutputTensor(size_t, const StorageShape&, const StorageFormat&, ge::DataType)
{
    return nullptr;
}
} // namespace gert

extern "C" ge::Status GetRegisteredIrDefV2(const char*, std::vector<std::pair<ge::AscendString, ge::AscendString>>&,
                                           std::vector<std::pair<ge::AscendString, ge::AscendString>>&,
                                           std::vector<std::pair<ge::AscendString, ge::AscendString>>&)
{
    return ge::GRAPH_FAILED;
}

TEST(FoldingUtilsTest, PathAndSoFiltering)
{
    EXPECT_EQ(aicpu_folding::EnsureTrailingSlash(""), "");
    EXPECT_EQ(aicpu_folding::EnsureTrailingSlash("/tmp/"), "/tmp/");
    EXPECT_EQ(aicpu_folding::EnsureTrailingSlash("/tmp"), "/tmp/");
    EXPECT_FALSE(aicpu_folding::IsConstantFoldingSo("x.so"));
    EXPECT_FALSE(aicpu_folding::IsConstantFoldingSo("libopconstant_folding_x.a"));
    EXPECT_FALSE(aicpu_folding::IsConstantFoldingSo("libconstant_folding_ops.so"));
    EXPECT_TRUE(aicpu_folding::IsConstantFoldingSo("libopconstant_folding_math.so"));
    EXPECT_EQ(aicpu_folding::GetRealPath("/path/that/does/not/exist"), "");
    EXPECT_FALSE(aicpu_folding::GetRealPath("/tmp").empty());
}

TEST_F(FoldingTest, InitializeValidatesEnvironmentAndIsIdempotent)
{
    EXPECT_EQ(ConstantFoldingInitialize(nullptr), -1);
    EXPECT_EQ(ConstantFoldingInitialize(""), -1);
    EXPECT_FALSE(g_initialized);

    g_initialized = true;
    EXPECT_EQ(ConstantFoldingInitialize("/path/that/does/not/exist"), 0);
}

TEST_F(FoldingTest, RegisterHostCpuFoldingOpsFiltersBlacklist)
{
    RegisterHostCpuFoldingOps({"Good", "Assign", "NoOp", "TruncatedNormal"});
    ASSERT_EQ(RegisteredOps().size(), 1U);
    EXPECT_EQ(std::get<0>(RegisteredOps()[0]), "Good");
    EXPECT_EQ(std::get<1>(RegisteredOps()[0]), ge::OpBackend::kHostCPU);
    EXPECT_EQ(std::get<2>(RegisteredOps()[0]), ge::OpRegistrationPriority::kBottom);
    EXPECT_EQ(std::get<3>(RegisteredOps()[0]), ge::OpEngine::kHostCpu);
}

TEST(FoldingPureTest, OutputInstanceValidation)
{
    EXPECT_EQ(ValidateOutputInstanceNum("y", kIrOutputRequired, 1U), 0);
    EXPECT_NE(ValidateOutputInstanceNum("y", kIrOutputRequired, 0U), 0);
    EXPECT_EQ(ValidateOutputInstanceNum("y", kIrOutputDynamic, 2U), 0);
    EXPECT_EQ(ValidateOutputInstanceNum("y", kIrOutputDynamic, 0U), 0);
    EXPECT_NE(ValidateOutputInstanceNum("y", "bad", 1U), 0);
}

TEST(FoldingPureTest, KnownShapeValidation)
{
    EXPECT_TRUE(IsKnownShape(gert::Shape({2, 3})));
    EXPECT_TRUE(IsKnownShape(gert::Shape({})));
    EXPECT_FALSE(IsKnownShape(gert::Shape({-1, 3})));
    EXPECT_FALSE(IsKnownShape(gert::Shape({ge::UNKNOWN_DIM_NUM})));
}

TEST(FoldingPureTest, RouterRejectsNullContext)
{
    AicpuHostCpuRouter router;
    EXPECT_EQ(router.Execute(nullptr), ge::GRAPH_FAILED);
}
