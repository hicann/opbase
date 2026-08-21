/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 本用例看护 Issue#318 的对齐修复，验证自定义内存池（BlockCache / HugeMemPool 两条路径）
// 经 Object::operator new 返回的用户地址满足 STD_MAX_ALIGN 对齐，且申请/释放无异常。
//
// 看护边界说明：
// - Issue#318 的崩溃现场为 x86_64 + GCC 13 生成 movaps 写 8 mod 16 地址触发 SIGSEGV。
//   本仓 UT 环境（aarch64 + GCC 9.4，且编译选项 -fno-sanitize=alignment 关闭对齐检查）
//   无法复现 movaps 崩溃现场。
// - 本用例的价值是“锁定修复、防回归”：通过 static_assert（编译期）+ 地址对齐断言（运行期）
//   看护对齐契约成立。若 BlockHeader 被改回 24 字节或 GetAddr 入口对齐被移除，
//   static_assert 立即编译失败或运行期断言失败。
// - 期望值 addr % STD_MAX_ALIGN == 0 基于地址公式逻辑推导，非“actual 当期望”：
//   路径 A：addr = malloc 基址(≥16 对齐) + sizeof(BlockHeader)(32, 16 倍数) → 16 对齐
//   路径 B：addr = huge 基址(16 对齐) + offset(初值 64 + ΣAlignUp(size,16)) → 16 对齐

#include "gtest/gtest.h"
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "opdev/object.h"
#include "opdev/op_executor.h"
#include "block_pool.h"
#include "block_store.h"
#include "bridge_pool.h"
#include "thread_local_context.h"

using namespace op;
using namespace op::internal;

// 大页内存初始化接口（huge_mem.cpp 导出，extern "C"），供 HugeMemPool 路径用例使用
extern "C" int InitHugeMemThreadLocal(void* arg, bool sync);
extern "C" void UnInitHugeMemThreadLocal(void* arg, bool sync);
extern "C" void ReleaseHugeMem(void* arg, bool sync);

// ===== 编译期看护：BlockHeader 布局对齐（Issue #318 根因）=====
// 若 BlockHeader 被改回自然布局（sizeof=24），以下 static_assert 立即编译失败。
// 期望值由 alignas(STD_MAX_ALIGN) 的标准语义推导（[basic.align]/3：sizeof 向上取整为对齐倍数）。
static_assert(sizeof(op::internal::BlockStore::BlockHeader) % op::internal::STD_MAX_ALIGN == 0,
              "BlockHeader size must be a multiple of STD_MAX_ALIGN so that head+1 is aligned");
static_assert(alignof(op::internal::BlockStore::BlockHeader) >= op::internal::STD_MAX_ALIGN,
              "BlockHeader alignment must cover STD_MAX_ALIGN");

class AlignmentTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// 校验对象地址满足 STD_MAX_ALIGN 对齐的辅助宏，附带对象类型与指针便于失败时定位
#define EXPECT_ALIGNED(ptr) EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % op::internal::STD_MAX_ALIGN, 0U)

// ===== 用例 1：BlockHeader 布局对齐（运行期呼应编译期 static_assert）=====
TEST_F(AlignmentTest, BlockHeaderSizeIsAligned)
{
    EXPECT_EQ(sizeof(op::internal::BlockStore::BlockHeader) % op::internal::STD_MAX_ALIGN, 0U);
    EXPECT_GE(alignof(op::internal::BlockStore::BlockHeader), op::internal::STD_MAX_ALIGN);
}

// ===== 用例 2：BlockCache 路径混合交织场景（Issue #318 崩溃路径）=====
// 顺序：aclCreateTensor → new aclOpExecutor → new aclTensor，循环 20 轮，
// 验证每个对象地址 16 对齐，且 delete/aclDestroyTensor 释放无异常。
// 无 InitHugeMemThreadLocal，poolIndex_ 无效 → Allocate 走 BlockCache::CacheAlloc。
// 路径区分：日志 pool index 为 -1；地址不在 huge block 区间（InHugeMemRange=false）。
TEST_F(AlignmentTest, MixedSequence_BlockCachePath_AlignedAndNoLeak)
{
    constexpr int32_t kRounds = 20;
    int64_t dims[] = {2, 3};
    int64_t strides[] = {3, 1};

    for (int32_t round = 0; round < kRounds; round++) {
        // 1. aclCreateTensor（API 路径，走 Object::operator new → Allocate → BlockCache）
        aclTensor* apiTensor = aclCreateTensor(dims, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, dims, 2, nullptr);
        ASSERT_NE(apiTensor, nullptr) << "round " << round;
        EXPECT_ALIGNED(apiTensor) << "round " << round << " apiTensor";

        // 2. new aclOpExecutor（直接 new，走同一 operator new）
        aclOpExecutor* exec = new aclOpExecutor;
        ASSERT_NE(exec, nullptr) << "round " << round;
        EXPECT_ALIGNED(exec) << "round " << round << " exec";

        // 3. new aclTensor（直接 new，public 构造重载）
        aclTensor* directTensor = new aclTensor(op::DataType::DT_FLOAT, op::Format::FORMAT_NHWC, op::Format::FORMAT_ND);
        ASSERT_NE(directTensor, nullptr) << "round " << round;
        EXPECT_ALIGNED(directTensor) << "round " << round << " directTensor";

        // 释放（顺序与创建相反），验证 delete/aclDestroyTensor 无异常
        delete directTensor;
        delete exec; // exec 析构会 delete allocatedObjList_（本用例未 AllocTensor，列表为空）
        aclDestroyTensor(apiTensor);
    }
}

// ===== 用例 3：HugeMemPool 路径混合交织场景（防同源崩溃）=====
// InitHugeMemThreadLocal 设有效 poolIndex_ → Allocate 走 GetAddr → HugeMemPool。
// 同样 aclCreateTensor → new aclOpExecutor → new aclTensor 循环 20 轮，
// 额外断言 InHugeMemRange 确认确实走大页路径，并看护 offset 累加后地址仍对齐。
// 路径区分：日志 pool index 为有效值（如 15）；地址落在 huge block 区间（InHugeMemRange=true）。
TEST_F(AlignmentTest, MixedSequence_HugeMemPath_AlignedAndNoLeak)
{
    InitHugeMemThreadLocal(nullptr, false); // 触发 HugeMemPool 路径
    constexpr int32_t kRounds = 20;
    int64_t dims[] = {2, 3};
    int64_t strides[] = {3, 1};

    for (int32_t round = 0; round < kRounds; round++) {
        aclTensor* apiTensor = aclCreateTensor(dims, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, dims, 2, nullptr);
        ASSERT_NE(apiTensor, nullptr) << "round " << round;
        ASSERT_TRUE(op::internal::BlockPool::InHugeMemRange(apiTensor))
            << "round " << round << " apiTensor 未走大页路径";
        EXPECT_ALIGNED(apiTensor) << "round " << round << " apiTensor";

        aclOpExecutor* exec = new aclOpExecutor;
        ASSERT_NE(exec, nullptr) << "round " << round;
        ASSERT_TRUE(op::internal::BlockPool::InHugeMemRange(exec)) << "round " << round << " exec 未走大页路径";
        EXPECT_ALIGNED(exec) << "round " << round << " exec";

        aclTensor* directTensor = new aclTensor(op::DataType::DT_FLOAT, op::Format::FORMAT_NHWC, op::Format::FORMAT_ND);
        ASSERT_NE(directTensor, nullptr) << "round " << round;
        ASSERT_TRUE(op::internal::BlockPool::InHugeMemRange(directTensor))
            << "round " << round << " directTensor 未走大页路径";
        EXPECT_ALIGNED(directTensor) << "round " << round << " directTensor";

        delete directTensor;
        delete exec;
        aclDestroyTensor(apiTensor);
    }
    // 必须用 ReleaseHugeMem 清理：其内部 FreeHugeMem 读 poolIndex_ 归还 baseArray_ 后再置 -1。
    // 若先 UnInitHugeMemThreadLocal 置 -1，FreeHugeMem 早退，污染全局 hugeMemArray_ 配额。
    ReleaseHugeMem(nullptr, false);
}

// ===== 用例 4：路径切换场景（BlockCache → HugeMemPool）=====
// 真实 executor 场景中，线程可能先无 poolIndex（BlockCache）跑，后切大页路径。
// 看护切换后对齐仍成立，且两阶段释放均无异常。
TEST_F(AlignmentTest, PathSwitch_BlockCacheToHugeMem_Aligned)
{
    // 阶段 1：BlockCache 路径
    aclOpExecutor* e1 = new aclOpExecutor;
    ASSERT_NE(e1, nullptr);
    EXPECT_FALSE(op::internal::BlockPool::InHugeMemRange(e1)) << "阶段1应走 BlockCache";
    EXPECT_ALIGNED(e1);
    delete e1;

    // 阶段 2：切换到 HugeMemPool 路径
    InitHugeMemThreadLocal(nullptr, false);
    aclOpExecutor* e2 = new aclOpExecutor;
    ASSERT_NE(e2, nullptr);
    ASSERT_TRUE(op::internal::BlockPool::InHugeMemRange(e2)) << "阶段2应走大页";
    EXPECT_ALIGNED(e2);
    delete e2;
    ReleaseHugeMem(nullptr, false);
}
