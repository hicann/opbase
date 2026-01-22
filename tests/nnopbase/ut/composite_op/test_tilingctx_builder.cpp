/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <array>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <random>
#include <thread>
#include <pthread.h>
#include <nlohmann/json.hpp>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"
#include "kernel_mgr.h"
#include "memset_ctx_holder.h"
#include "kernel_context_holder.h"
#include "tilingctx_builder.h"
#include "op_kernel.h"
#include "op_ctx_def.h"
#include "opdev/op_def.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_errno.h"
#include "thread_local_context.h"
#include "kernel_workspace.h"
#include "op_run_context.h"
#include "depends/platform/platform_stub.h"

typedef std::function<void(void)> Functional;
namespace op::internal {
extern void SetCoreNum(const nlohmann::json &opJson, fe::PlatFormInfos *platformInfo, uint32_t &coreNum);
}
class TilingCtxBuildUT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // aclInit(nullptr)
        setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1); // does overwrite
        op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    }

    static void TearDownTestCase() {}

    op::internal::OpKernelBin *CreateFakeOpKernelBin()
    {
        uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
        const char *p = std::getenv("ASCEND_OPP_PATH");
        EXPECT_NE(p, nullptr);
        op::internal::KeyAndDetail key;
        key.key = "hahaha";
        size_t hashKey = 125;
        char jsonPath[1024];
        char binPath[1024];
        snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
            "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/axpy/Axpy_233851a3505389e43928a8bba133a74d_high_performance.json",
            p);
        snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
            "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/axpy/Axpy_233851a3505389e43928a8bba133a74d_high_performance.o",
            p);
        op::internal::OpKernelBin *fakeBin = new op::internal::OpKernelBin(
            opType, jsonPath, jsonPath, binPath, key, hashKey, op::internal::BinType::DYNAMIC_BIN, false, false);
        OP_LOGI("CreateFakeOpKernelBin fakeBin: %p", fakeBin);
        return fakeBin;
    }
};

static void TilingCtxBuildUTCase1()
{
    op::Shape selfShape{33, 15, 1, 48};
    op::Shape otherShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(otherShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    float alpha = 13.37;
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);

    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    auto input = OP_INPUT(self.get(), other.get());
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto ws = OP_WORKSPACE(out.get());

    const char *p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);
    op::internal::KeyAndDetail key;
    key.key = "hahaha";
    size_t hashKey = 125;
    char jsonPath[1024];
    char binPath[1024];
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
        "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/axpy/Axpy_233851a3505389e43928a8bba133a74d_high_performance.json",
        p);
    snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
        "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/axpy/Axpy_233851a3505389e43928a8bba133a74d_high_performance.o",
        p);

    op::internal::OpKernelBin kernelBin(
        opType, jsonPath, jsonPath, binPath, key, hashKey, op::internal::BinType::DYNAMIC_BIN, false, false);
    EXPECT_EQ(kernelBin.InitTilingParseCtx(), ACLNN_SUCCESS);

    auto ctx = op::MakeOpArgContext(input, output, attr, ws);
    ge::AscendString opTypeAscendStr = op::OpTypeDict::ToString(opType);
    const char *opTypeStr = opTypeAscendStr.GetString();
    op::internal::OpRunContextMgr::opRunCtx_.kernelCtx_.UpdateComputeNodeInfo(opTypeStr,
        *ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
        *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG),
        *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    op::internal::OpRunContextMgr::opRunCtx_.kernelCtx_.UpdateKernelExtendInfo(opTypeStr, opTypeStr);

    uint32_t aic = op::internal::GetThreadLocalContext().opConfigInfo_.aicNum_;
    uint32_t aiv = op::internal::GetThreadLocalContext().opConfigInfo_.aivNum_;
    auto res = op::internal::OpRunContextMgr::opRunCtx_.tilingCtx_.UpdateTilingCtx(
        &(op::internal::OpRunContextMgr::opRunCtx_.kernelCtx_),
        kernelBin.tilingParseCtxHolder_[op::internal::ThreadCoreNum(aic, aiv)].get());
    EXPECT_EQ(res, ACLNN_SUCCESS);
    for (auto &[key, value] : kernelBin.tilingParseCtxHolder_) {
        if (value.get()) {
            value.get()->ReleaseTilingParse();
        }
    }
    op::DestroyOpArgContext(ctx);
}

static void TilingCtxBuildUTCase2()
{
    op::Shape selfShape{33, 15, 1, 48};
    op::Shape otherShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(otherShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    float alpha = 13.37;
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);

    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    auto input = OP_INPUT(self.get(), other.get());
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto ws = OP_WORKSPACE(out.get());

    auto ctx = op::MakeOpArgContext(input, output, attr, ws);
    ge::AscendString opTypeAscendStr = op::OpTypeDict::ToString(opType);
    const char *opTypeStr = opTypeAscendStr.GetString();
    op::internal::OpRunContextMgr::opRunCtx_.kernelCtx_.UpdateComputeNodeInfo(opTypeStr,
        *ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
        *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG),
        *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    op::internal::OpRunContextMgr::opRunCtx_.kernelCtx_.UpdateKernelExtendInfo(opTypeStr, opTypeStr);

    auto res = op::internal::OpRunContextMgr::opRunCtx_.tilingCtx_.UpdateTilingCtx(
        &(op::internal::OpRunContextMgr::opRunCtx_.kernelCtx_));
    EXPECT_EQ(res, ACLNN_SUCCESS);
    op::DestroyOpArgContext(ctx);
}

static void TilingCtxBuildUTCase3()
{
    op::Shape tShape{1, 2, 3};
    int64_t shape[] = {4, 5, 6};
    aclTensor t1(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t2(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor *nullTensor = nullptr;
    aclTensorList *nullTensorList = nullptr;
    aclTensor *list2[] = {aclCreateTensor(shape, 3, aclDataType::ACL_FLOAT,
                                          nullptr, 0, aclFormat::ACL_FORMAT_ND, shape, 3, nullptr),
                          aclCreateTensor(shape, 3, aclDataType::ACL_FLOAT,
                                          nullptr, 0, aclFormat::ACL_FORMAT_ND, shape, 3, nullptr),
                          nullptr};
    aclTensorList *tensorList = aclCreateTensorList(list2, 3);

    bool boolAttr = false;
    op::DataType dtypeAttr = op::DataType::DT_FLOAT16;
    float scalar_value = 3.14;
    aclScalar *scalarAttr = aclCreateScalar(&scalar_value, aclDataType::ACL_FLOAT);
    aclScalar *nullScalarAttr = nullptr;
    int64_t array[] = {3, 4, 5};
    aclIntArray *intArrayAttr = aclCreateIntArray(array, sizeof(array) / sizeof(array[0]));
    aclIntArray *nullIntArrayAttr = nullptr;
    float farray[] = {1.1, 2.2, 3,3};
    aclFloatArray *floatArrayAttr = aclCreateFloatArray(farray, sizeof(farray) / sizeof(farray[0]));
    aclFloatArray *nullFloatArrayAttr = nullptr;
    std::string stringAttr = "123";
    std::string s = "456";
    std::string *stringPAttr = &s;
    char *charPAttr = "789";
    double doubleAttr = 3.14159;
    float floatAttr = 3.14;

    auto input_arg = OP_INPUT(&t1, nullTensor, nullTensorList, tensorList);
    auto output_arg = OP_OUTPUT(&t2);
    auto attr_arg = OP_ATTR(boolAttr, dtypeAttr, scalarAttr, nullScalarAttr,
        intArrayAttr, nullIntArrayAttr, floatArrayAttr, nullFloatArrayAttr,
        stringAttr, stringPAttr, charPAttr, doubleAttr, floatAttr);

    auto ctx = op::MakeOpArgContext(input_arg, output_arg, attr_arg);

    EXPECT_EQ(ctx->ContainsOpArgType(op::OpArgDef::OP_INPUT_ARG), true);
    EXPECT_EQ(ctx->ContainsOpArgType(op::OpArgDef::OP_OUTPUT_ARG), true);
    EXPECT_EQ(ctx->ContainsOpArgType(op::OpArgDef::OP_ATTR_ARG), true);

    op::OpArgList &input = *ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG);
    op::OpArgList &output = *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG);
    op::OpArgList &attr = *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG);

    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    gert::TilingContext *tilingCtx =
        op::internal::OpRunContextMgr::opRunCtx_.UpdateTilingCtx(opType, input, output, attr);
    EXPECT_NE(tilingCtx, nullptr);

    auto resShape = tilingCtx->GetInputShape(0);
    EXPECT_EQ(resShape->GetStorageShape().GetDimNum(), 3);
    EXPECT_EQ(resShape->GetStorageShape().GetDim(0), 1);
    EXPECT_EQ(resShape->GetStorageShape().GetDim(1), 2);
    EXPECT_EQ(resShape->GetStorageShape().GetDim(2), 3);

    resShape = tilingCtx->GetInputShape(1);
    EXPECT_EQ(resShape->GetStorageShape().GetDimNum(), 3);
    EXPECT_EQ(resShape->GetStorageShape().GetDim(0), 4);
    EXPECT_EQ(resShape->GetStorageShape().GetDim(1), 5);
    EXPECT_EQ(resShape->GetStorageShape().GetDim(2), 6);

    resShape = tilingCtx->GetInputShape(2);
    EXPECT_EQ(resShape->GetStorageShape().GetDimNum(), 3);
    EXPECT_EQ(resShape->GetStorageShape().GetDim(0), 4);
    EXPECT_EQ(resShape->GetStorageShape().GetDim(1), 5);
    EXPECT_EQ(resShape->GetStorageShape().GetDim(2), 6);

    resShape = tilingCtx->GetOutputShape(0);
    EXPECT_EQ(resShape->GetStorageShape().GetDimNum(), 3);
    EXPECT_EQ(resShape->GetStorageShape().GetDim(0), 1);
    EXPECT_EQ(resShape->GetStorageShape().GetDim(1), 2);
    EXPECT_EQ(resShape->GetStorageShape().GetDim(2), 3);

    auto attrs = tilingCtx->GetAttrs();
    EXPECT_NE(attrs, nullptr);
    EXPECT_EQ(attrs->GetAttrNum(), 13);

    auto bp = attrs->GetBool(0);
    EXPECT_NE(bp, nullptr);
    EXPECT_EQ(*bp, false);

    auto dp = attrs->GetInt(1);
    EXPECT_NE(dp, nullptr);
    EXPECT_EQ(*dp, static_cast<int64_t>(op::DataType::DT_FLOAT16));

    auto fp = attrs->GetFloat(2);
    EXPECT_NE(fp, nullptr);
    EXPECT_LE(*fp - scalar_value, std::numeric_limits<float>::epsilon());

    dp = attrs->GetInt(3);
    EXPECT_NE(dp, nullptr);
    EXPECT_EQ(*dp, 0);

    auto iap = attrs->GetListInt(4);
    EXPECT_NE(iap, nullptr);
    EXPECT_EQ(iap->GetData()[0], 3);
    EXPECT_EQ(iap->GetData()[1], 4);
    EXPECT_EQ(iap->GetData()[2], 5);

    iap = attrs->GetListInt(5);
    EXPECT_NE(iap, nullptr);
    EXPECT_EQ(iap->GetSize(), 0);

    auto fap = attrs->GetListFloat(6);
    EXPECT_NE(fap, nullptr);
    EXPECT_LE(fap->GetData()[0] - 1.1, std::numeric_limits<float>::epsilon());
    EXPECT_LE(fap->GetData()[1] - 2.2, std::numeric_limits<float>::epsilon());
    EXPECT_LE(fap->GetData()[2] - 3.3, std::numeric_limits<float>::epsilon());

    fap = attrs->GetListFloat(7);
    EXPECT_NE(fap, nullptr);
    EXPECT_EQ(fap->GetSize(), 0);

    auto sp = attrs->GetStr(8);
    EXPECT_NE(sp, nullptr);
    EXPECT_EQ(0, strcmp(sp, "123"));

    sp = attrs->GetStr(9);
    EXPECT_NE(sp, nullptr);
    EXPECT_EQ(0, strcmp(sp, "456"));

    sp = attrs->GetStr(10);
    EXPECT_NE(sp, nullptr);
    EXPECT_EQ(0, strcmp(sp, "789"));

    fp = attrs->GetFloat(11);
    EXPECT_NE(fp, nullptr);
    EXPECT_LE(*fp - static_cast<float>(doubleAttr), std::numeric_limits<float>::epsilon());

    fp = attrs->GetFloat(12);
    EXPECT_NE(fp, nullptr);
    EXPECT_LE(*fp - floatAttr, std::numeric_limits<float>::epsilon());

    op::DestroyOpArgContext(ctx);
    aclDestroyTensorList(tensorList);
    aclDestroyScalar(scalarAttr);
    aclDestroyIntArray(intArrayAttr);
}

static void GetTilingResFromCacheCase()
{
    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};
    op::Shape wsShape{32};

    int64_t dim = 0;
    bool descending = true;

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto idx = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);

    uint32_t opType = op::OpTypeDict::ToOpType("Sort");
    auto input = OP_INPUT(self.get());
    auto output =
        OP_OUTPUT(out.get(), idx.get(), static_cast<aclTensor *>(nullptr), static_cast<aclTensorList *>(nullptr));
    auto attr = OP_ATTR(dim, descending);
    auto ctx = op::MakeOpArgContext(input, output, attr);

    op::internal::GetLauncherCtx().Reset();

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor *executor = uniqueExecutor.get();
    aclTensorList *workspace2 = nullptr;
    op::internal::GetWorkspace(opType, &workspace2, executor, 
        *ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
        *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG),
        *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));

    const op::internal::TilingResCache *tilingCache = nullptr;
    tilingCache = op::internal::GetLauncherCtx().GetTilingResCache();
    EXPECT_NE(tilingCache, nullptr);

    const op::internal::TilingCtxOutput *res =
        op::internal::OpRunContextMgr::opRunCtx_.tilingCtx_.GetTilingResFromCache(*tilingCache);
    EXPECT_NE(res, nullptr);
    aclDestroyTensorList((const aclTensorList *)workspace2);
    uniqueExecutor.ReleaseTo(&executor);
    op::DestroyOpArgContext(ctx);
}

static void TilingParseCtxHolderFreeTest()
{
    auto MyTilingParseCtx = std::make_unique<op::internal::TilingParseCtxHolder>();
    MyTilingParseCtx->tilingParseInfo_.compileInfoStruct_ = (void *)malloc(sizeof(void *));
    EXPECT_NE(MyTilingParseCtx->tilingParseInfo_.compileInfoStruct_, nullptr);
    EXPECT_EQ(MyTilingParseCtx->tilingParseInfoDeleter, nullptr);
}

static void SetCoreNumTestAiCore(){
    nlohmann::json mixAiCore = { {"coreType", "AiCore"}};
    fe::PlatFormInfos *plat = &op::internal::SocContext::platformInfo_;
    uint32_t coreNum = 0;
    op::internal::SetCoreNum(mixAiCore, plat, coreNum);
    sleep(1);
    uint32_t currCoreNum = plat->GetCoreNum();
    EXPECT_EQ(currCoreNum, 0);
}
 
static void SetCoreNumTestMIX(){
    nlohmann::json mixMIX = { {"coreType", "MIX"}};
    fe::PlatFormInfos *plat = &op::internal::SocContext::platformInfo_;
    uint32_t coreNum = 0;
    op::internal::SetCoreNum(mixMIX, plat, coreNum);
    sleep(1);
    uint32_t currCoreNum = plat->GetCoreNum();
    EXPECT_EQ(currCoreNum, 0);
}
 
static void SetCoreNumTestVectorCore(){
    nlohmann::json mixVector = { {"coreType", "VectorCore"}};
    fe::PlatFormInfos *plat = &op::internal::SocContext::platformInfo_;
    uint32_t coreNum = 0;
    op::internal::SetCoreNum(mixVector, plat, coreNum);
    sleep(1);
    uint32_t currCoreNum = plat->GetCoreNum();
    EXPECT_EQ(currCoreNum, 0);
}
 
static void SetCoreNumTestAIV(){
    nlohmann::json mixAIV = { {"coreType", "MIX_AIV"}};
    fe::PlatFormInfos *plat = &op::internal::SocContext::platformInfo_;
    uint32_t coreNum = 0;
    op::internal::SetCoreNum(mixAIV, plat, coreNum);
    sleep(1);
    uint32_t currCoreNum = plat->GetCoreNum();
    EXPECT_EQ(currCoreNum, 0);
}

static void SetCoreNumTest310PMIXAIV(){
    nlohmann::json mixAIV = { {"coreType", "MIX_AIV"}};
    fe::PlatFormInfos *plat = &op::internal::SocContext::platformInfo_;
    PlatformInfoStub::GetInstance()->SetSoCVersion("Ascend310P", "Ascend310P3");
    uint32_t coreNum = 0;
    op::internal::SetCoreNum(mixAIV, plat, coreNum);
    sleep(1);
    uint32_t currCoreNum = plat->GetCoreNum();
    EXPECT_EQ(currCoreNum, 0);
    PlatformInfoStub::GetInstance()->Reset();
}

TEST_F(TilingCtxBuildUT, TilingCtxBuildUT_StringATTR)
{
    op::Shape selfShape{33, 15, 1, 48};
    op::Shape otherShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(otherShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    float alpha = 13.37;
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);

    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    auto input = OP_INPUT(self.get(), other.get());
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto ws = OP_WORKSPACE(out.get());

    ge::AscendString opTypeAscendStr = op::OpTypeDict::ToString(opType);
    const char *opTypeStr = opTypeAscendStr.GetString();
    string attrs = "1234";
    int64_t v = 7;
     op::internal::OpRunContextMgr::opRunCtx_.kernelCtx_.UpdateAttrDefOffset(3, 1);
    auto res = op::internal::OpRunContextMgr::opRunCtx_.kernelCtx_.UpdateAttrArg(
        0, v, op::internal::OpRunContextMgr::opRunCtx_.kernelCtx_.attrDataStart_);
    EXPECT_EQ(res, ACLNN_SUCCESS);
}

TEST_F(TilingCtxBuildUT, TilingCtxMultiThreadTest){
    vector<Functional> funVec = {TilingCtxBuildUTCase1, TilingCtxBuildUTCase2, TilingCtxBuildUTCase3,
        GetTilingResFromCacheCase, TilingParseCtxHolderFreeTest, SetCoreNumTestAiCore, SetCoreNumTestMIX,
        SetCoreNumTestVectorCore, SetCoreNumTestAIV, SetCoreNumTest310PMIXAIV};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, funVec.size()-1);
    const uint64_t threadCount = funVec.size() * 100;
    vector<std::thread> threadVec;
    for (int i = 0; i<threadCount; i++) {
        threadVec.emplace_back(std::thread(funVec[distrib(gen)]));
    }
    for (int i = 0; i<threadCount; i++) {
        threadVec[i].join();
    }
}

static void TilingCtxBuildLaunchOneThread(op::internal::OpKernelBin *opKernelBin, pthread_barrier_t *barrier)
{
    if (barrier) {
        OP_LOGI("pthread_barrier_wait=========================");
        pthread_barrier_wait(barrier);
    }
    EXPECT_EQ(opKernelBin->InitTilingParseCtx(), ACLNN_SUCCESS);
    op::Shape selfShape{33, 15, 1, 48};
    op::Shape otherShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(otherShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    float alpha = 13.37;
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);

    auto input = OP_INPUT(self.get(), other.get());
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto ws = OP_WORKSPACE(out.get());

    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    auto ctx = op::MakeOpArgContext(input, output, attr, ws);
    ge::AscendString opTypeAscendStr = op::OpTypeDict::ToString(opType);
    const char *opTypeStr = opTypeAscendStr.GetString();
    op::internal::OpRunContextMgr::opRunCtx_.kernelCtx_.UpdateComputeNodeInfo(opTypeStr,
        *ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
        *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG),
        *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    op::internal::OpRunContextMgr::opRunCtx_.kernelCtx_.UpdateKernelExtendInfo(opTypeStr, opTypeStr);

    uint32_t aic = op::internal::GetThreadLocalContext().opConfigInfo_.aicNum_;
    uint32_t aiv = op::internal::GetThreadLocalContext().opConfigInfo_.aivNum_;
    auto res = op::internal::OpRunContextMgr::opRunCtx_.tilingCtx_.UpdateTilingCtx(
        &(op::internal::OpRunContextMgr::opRunCtx_.kernelCtx_),
        opKernelBin->tilingParseCtxHolder_[op::internal::ThreadCoreNum(aic, aiv)].get());
    EXPECT_EQ(res, ACLNN_SUCCESS);
    op::DestroyOpArgContext(ctx);
}

TEST_F(TilingCtxBuildUT, TilingCtxMultiThreadTest2)
{
    op::internal::OpKernelBin *fakeBin = CreateFakeOpKernelBin();
    constexpr uint32_t BARRIER_THREAD_COUNT = 100;
    constexpr uint32_t LAUNCH_THREADS_TIMES = 3;

    for (uint32_t i = 0; i < LAUNCH_THREADS_TIMES; i++) {
        pthread_barrier_t barrier;
        auto ret = pthread_barrier_init(&barrier, nullptr, BARRIER_THREAD_COUNT);  // 初始化屏障，指定需要等待的线程数
        if (ret != 0) {
            std::cout << "pthread_barrier_init failed, error code: " << ret << std::endl;
        }
        vector<std::thread> threadVec;
        for (int i = 0; i < BARRIER_THREAD_COUNT; i++) {
            threadVec.emplace_back(std::thread(TilingCtxBuildLaunchOneThread, fakeBin, &barrier));
        }
        for (int i = 0; i < BARRIER_THREAD_COUNT; i++) {
            threadVec[i].join();
        }
        ret = pthread_barrier_destroy(&barrier);  // 销毁屏障
        if (ret != 0) {
            std::cout << "pthread_barrier_destroy failed, error code: " << ret << std::endl;
        }
    }

    for (auto &[key, value] : fakeBin->tilingParseCtxHolder_) {
        if (value.get()) {
            value.get()->ReleaseTilingParse();
        }
    }
    delete fakeBin;
}

TEST_F(TilingCtxBuildUT, TilingCtxMultiThreadTest3)
{
    op::internal::OpKernelBin *fakeBin = CreateFakeOpKernelBin();

    const uint64_t threadCount = 200;
    vector<std::thread> threadVec;
    for (uint64_t i = 0; i < threadCount; i++) {
        threadVec.emplace_back(std::thread(TilingCtxBuildLaunchOneThread, fakeBin, nullptr));
    }
    for (uint64_t i = 0; i < threadCount; i++) {
        threadVec[i].join();
    }
    for (auto &[key, value] : fakeBin->tilingParseCtxHolder_) {
        if (value.get()) {
            value.get()->ReleaseTilingParse();
        }
    }
    delete fakeBin;
}
