/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "acl/acl_rt.h"
#include "op_dfx_internal.h"
#include "thread_local_context.h"

#include "depends/acl/aclrt_stub.h"

using namespace op::internal;

namespace {

// 记录 aclrtGetStreamAttribute 的调用情况，并允许用例设定其返回码与写入 value 的值
class StreamAttrRecordStub : public AclrtStub {
public:
    aclError aclrtGetStreamAttribute([[maybe_unused]] aclrtStream stream, aclrtStreamAttr stmAttrType,
                                     aclrtStreamAttrValue* value) override
    {
        ++callCount_;
        lastAttrType_ = stmAttrType;
        value->cacheOpInfoSwitch = switchValue_;
        return ret_;
    }

    void SetResult(aclError ret, uint32_t switchValue)
    {
        ret_ = ret;
        switchValue_ = switchValue;
    }

    int32_t callCount_{0};
    // 默认取一个非目标枚举值，避免接口未被调用时与期望的属性类型巧合相等
    aclrtStreamAttr lastAttrType_{ACL_STREAM_ATTR_FAILURE_MODE};

private:
    aclError ret_{ACL_SUCCESS};
    uint32_t switchValue_{1U};
};

} // namespace

class OpDfxTest : public testing::Test {
protected:
    void SetUp() override { AclrtStub::GetInstance()->Install(&stub_); }

    void TearDown() override
    {
        AclrtStub::GetInstance()->UnInstall();
        // 开关存放在 thread_local 上下文，跨用例保留，复位以避免污染后续用例
        op::internal::GetThreadLocalContext().cacheOpInfoSwitch_ = false;
    }

    // 预置为期望值的相反值，使断言能够证明开关确实由 GetCacheOpInfoSwitch 写入
    void PresetCacheOpInfoSwitch(bool value) { op::internal::GetThreadLocalContext().cacheOpInfoSwitch_ = value; }

    StreamAttrRecordStub stub_;
};

// stream 为空：不调用 aclrtGetStreamAttribute，开关取 value 初值 0，即 false
TEST_F(OpDfxTest, GetCacheOpInfoSwitchNullStreamTest)
{
    PresetCacheOpInfoSwitch(true);

    GetCacheOpInfoSwitch(nullptr);

    EXPECT_EQ(stub_.callCount_, 0);
    EXPECT_FALSE(op::internal::GetThreadLocalContext().cacheOpInfoSwitch_);
}

// stream 非空但取流属性失败：仍调用一次接口，且失败分支把 value 重置为 0，开关应为 false
// 桩故意写入 1，用于验证失败分支的重置确实生效
TEST_F(OpDfxTest, GetCacheOpInfoSwitchStreamAttrFailedTest)
{
    PresetCacheOpInfoSwitch(true);
    stub_.SetResult(static_cast<aclError>(1), 1U);

    aclrtStream stream = (aclrtStream)0x1;
    GetCacheOpInfoSwitch(stream);

    EXPECT_EQ(stub_.callCount_, 1);
    EXPECT_EQ(stub_.lastAttrType_, ACL_STREAM_ATTR_CACHE_OP_INFO);
    EXPECT_FALSE(op::internal::GetThreadLocalContext().cacheOpInfoSwitch_);
}

// stream 非空、取流属性成功且值为非 0：开关应为 true
TEST_F(OpDfxTest, GetCacheOpInfoSwitchStreamAttrSuccessEnabledTest)
{
    PresetCacheOpInfoSwitch(false);
    stub_.SetResult(ACL_SUCCESS, 1U);

    aclrtStream stream = (aclrtStream)0x1;
    GetCacheOpInfoSwitch(stream);

    EXPECT_EQ(stub_.callCount_, 1);
    EXPECT_EQ(stub_.lastAttrType_, ACL_STREAM_ATTR_CACHE_OP_INFO);
    EXPECT_TRUE(op::internal::GetThreadLocalContext().cacheOpInfoSwitch_);
}

// stream 非空、取流属性成功但值为 0：开关应为 false，说明开关按 value 映射而非按调用成败
TEST_F(OpDfxTest, GetCacheOpInfoSwitchStreamAttrSuccessDisabledTest)
{
    PresetCacheOpInfoSwitch(true);
    stub_.SetResult(ACL_SUCCESS, 0U);

    aclrtStream stream = (aclrtStream)0x1;
    GetCacheOpInfoSwitch(stream);

    EXPECT_EQ(stub_.callCount_, 1);
    EXPECT_EQ(stub_.lastAttrType_, ACL_STREAM_ATTR_CACHE_OP_INFO);
    EXPECT_FALSE(op::internal::GetThreadLocalContext().cacheOpInfoSwitch_);
}
