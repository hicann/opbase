/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>

#include <gtest/gtest.h>

#include "bridge_dfx.h"
#include "executor/indv_args.h"
#include "executor/indv_bininfo.h"
#include "executor/indv_executor.h"

namespace {
void SetParamInstance(NnopbaseParamInstance& instance, uint32_t startIndex, uint32_t num, bool isDynamic, bool isInput)
{
    instance.startIndex = startIndex;
    instance.num = num;
    instance.cfgNum = isDynamic ? 2U : 1U;
    instance.isDynamic = isDynamic;
    instance.isInput = isInput;
}

void SetTensor(NnopbaseTensor& tensor, void* addr, gert::TensorPlacement placement,
               std::initializer_list<int64_t> shape, ge::DataType dtype = ge::DT_FLOAT)
{
    tensor.isNull = false;
    GertShape gertShape(shape);
    tensor.storageShape = gertShape;
    tensor.rt2Tensor.MutableOriginShape() = gertShape;
    tensor.rt2Tensor.MutableStorageShape() = gertShape;
    tensor.rt2Tensor.SetDataType(dtype);
    tensor.rt2Tensor.SetOriginFormat(ge::FORMAT_ND);
    tensor.rt2Tensor.SetStorageFormat(ge::FORMAT_ND);
    tensor.rt2Tensor.MutableTensorData().SetPlacement(placement);
    tensor.rt2Tensor.SetSize(sizeof(uint64_t));
    ASSERT_EQ(tensor.rt2Tensor.MutableTensorData().SetAddr(addr, nullptr), ge::GRAPH_SUCCESS);
}
} // namespace

class NnopbaseIndvArgsTest : public testing::Test {
protected:
    void SetUp() override
    {
        executor_.args = &args_;
        executor_.argsExt.args = launchBuf_.data();
        args_.binInfo = &binInfo_;
    }

    NnopbaseExecutor executor_{};
    NnopbaseExecutorArgs args_{};
    NnopbaseBinInfo binInfo_{};
    std::array<NnopbaseUChar, 2048U> launchBuf_{};
};

TEST_F(NnopbaseIndvArgsTest, PrepareInputsParamsExtEncodesDeviceHostDynamicAndOptionalInputs)
{
    args_.inputs.paramDescs.count = 4U;
    args_.inputs.paramDescs.instances.resize(4U);
    SetParamInstance(args_.inputs.paramDescs.instances[0], 0U, 1U, false, true);
    SetParamInstance(args_.inputs.paramDescs.instances[1], 1U, 1U, false, true);
    SetParamInstance(args_.inputs.paramDescs.instances[2], 2U, 1U, false, true);
    SetParamInstance(args_.inputs.paramDescs.instances[3], 3U, 2U, true, true);
    args_.inputs.extTensors.resize(5U);
    args_.inputs.num = 5U;

    uint64_t deviceData = 0x1111U;
    uint64_t hostData = 0x2222U;
    uint64_t dynData0 = 0x3333U;
    uint64_t dynData1 = 0x4444U;
    SetTensor(args_.inputs.extTensors[0], &deviceData, gert::kOnDeviceHbm, {4});
    args_.inputs.extTensors[1].isNull = true;
    SetTensor(args_.inputs.extTensors[2], &hostData, gert::kOnHost, {1}, ge::DT_UINT64);
    SetTensor(args_.inputs.extTensors[3], &dynData0, gert::kOnDeviceHbm, {2, 3});
    SetTensor(args_.inputs.extTensors[4], &dynData1, gert::kOnDeviceHbm, {5});

    auto* params = reinterpret_cast<void**>(launchBuf_.data());
    auto* hostInfo = reinterpret_cast<aclrtPlaceHolderInfo*>(launchBuf_.data() + 512U);
    auto* hostDataStart = launchBuf_.data() + 768U;
    NnopbaseHcclCommParamDesc hcclDesc{};
    NnopbaseExecutorArgsAddr argsAddr{hostDataStart, hostInfo, launchBuf_.data() + 1600U, nullptr, &hcclDesc};

    void** ret = NnopbaseExecutorPrepareInputsParamsExt(&executor_, params, &argsAddr);

    ASSERT_EQ(ret, params + 4U);
    EXPECT_EQ(params[0], &deviceData);
    EXPECT_EQ(params[1], nullptr);
    EXPECT_EQ(params[2], hostDataStart);
    EXPECT_EQ(*reinterpret_cast<uint64_t*>(params[2]), hostData);
    EXPECT_EQ(hcclDesc.isDyn, 1ULL << 3U);

    auto* dynamicBase = reinterpret_cast<NnopbaseUChar*>(params[3]);
    ASSERT_NE(dynamicBase, nullptr);
    const auto dynamicDataOffset = *reinterpret_cast<uint64_t*>(dynamicBase);
    EXPECT_EQ(dynamicDataOffset, 48U);
    EXPECT_EQ(*reinterpret_cast<void**>(dynamicBase + dynamicDataOffset), &dynData0);
    EXPECT_EQ(*reinterpret_cast<void**>(dynamicBase + dynamicDataOffset + sizeof(void*)), &dynData1);

    const auto base = reinterpret_cast<uintptr_t>(launchBuf_.data());
    EXPECT_EQ(hostInfo[0].addrOffset, reinterpret_cast<uintptr_t>(&params[2]) - base);
    EXPECT_EQ(hostInfo[0].dataOffset, reinterpret_cast<uintptr_t>(hostDataStart) - base);
    EXPECT_EQ(hostInfo[1].addrOffset, reinterpret_cast<uintptr_t>(&params[3]) - base);
    EXPECT_EQ(hostInfo[1].dataOffset, reinterpret_cast<uintptr_t>(dynamicBase) - base);
    EXPECT_GT(args_.inputs.extTensors[3].argsOffset, 0U);
    EXPECT_GT(args_.inputs.extTensors[4].argsOffset, args_.inputs.extTensors[3].argsOffset);
}

TEST_F(NnopbaseIndvArgsTest, PrepareNullTensorsReservesLaunchSlotOnlyForDynamicShapeBin)
{
    std::array<void*, 2U> params = {reinterpret_cast<void*>(0x1234), reinterpret_cast<void*>(0x5678)};
    size_t tensorIndex = 0U;

    binInfo_.isStaticShape = false;
    void** ret = NnopbaseExecutorPrepareNullTensors(&executor_, params.data(), &tensorIndex);
    EXPECT_EQ(ret, params.data() + 1U);
    EXPECT_EQ(tensorIndex, 1U);
    EXPECT_EQ(params[0], nullptr);

    params = {reinterpret_cast<void*>(0x1234), reinterpret_cast<void*>(0x5678)};
    tensorIndex = 0U;
    binInfo_.isStaticShape = true;
    ret = NnopbaseExecutorPrepareNullTensors(&executor_, params.data(), &tensorIndex);
    EXPECT_EQ(ret, params.data());
    EXPECT_EQ(tensorIndex, 1U);
    EXPECT_EQ(params[0], reinterpret_cast<void*>(0x1234));
}

TEST_F(NnopbaseIndvArgsTest, GetIrIndexMapsFlattenedDynamicTensorIndexBackToIrInput)
{
    NnopbaseParamDesc desc{};
    desc.count = 3U;
    desc.instances.resize(3U);
    SetParamInstance(desc.instances[0], 0U, 1U, false, true);
    SetParamInstance(desc.instances[1], 1U, 3U, true, true);
    SetParamInstance(desc.instances[2], 4U, 1U, false, true);

    size_t irIndex = 0U;
    size_t relativeIndex = 0U;
    NnopbaseGetIrIndex(desc, 3U, irIndex, relativeIndex);

    EXPECT_EQ(irIndex, 1U);
    EXPECT_EQ(relativeIndex, 2U);
}

TEST_F(NnopbaseIndvArgsTest, AppendOomStorageShapeForIgnoreContiguousTensor)
{
    const int64_t viewShape[] = {2, 2};
    const int64_t viewStrides[] = {3, 1};
    const int64_t storageShape[] = {2, 3};
    uint8_t data[32] = {};
    aclTensor* tensor = aclCreateTensor(viewShape, 2U, aclDataType::ACL_FLOAT, viewStrides, 0,
                                        aclFormat::ACL_FORMAT_ND, storageShape, 2U, data);
    ASSERT_NE(tensor, nullptr);

    args_.inputs.paramDescs.count = 1U;
    args_.inputs.paramDescs.instances.resize(1U);
    SetParamInstance(args_.inputs.paramDescs.instances[0U], 0U, 1U, false, true);
    args_.inputs.paramDescs.instances[0U].ignoreCont = true;
    args_.inputs.paramDescs.instances[0U].tensor = tensor;
    args_.inputs.extTensors.resize(1U);
    args_.inputs.extTensors[0U].storageShape = tensor->GetStorageShape();
    args_.inputs.num = 1U;
    executor_.ownArgs.inputs.paramDescs.count = 1U;
    executor_.ownArgs.inputs.paramDescs.instances.resize(1U);
    SetParamInstance(executor_.ownArgs.inputs.paramDescs.instances[0U], 0U, 1U, false, true);
    executor_.ownArgs.inputs.paramDescs.instances[0U].ignoreCont = true;
    executor_.ownArgs.inputs.paramDescs.instances[0U].tensor = tensor;
    args_.dfxInfo.resize(2U);
    binInfo_.oomConfig.flag = true;
    binInfo_.oomConfig.storageShapeEnabled = true;
    binInfo_.oomConfig.version = 3U;
    binInfo_.oomConfig.tensorVersion = 5U;
    NnopbaseExecutorArgsAddr argsAddr{nullptr, nullptr, launchBuf_.data(), nullptr, nullptr};

    ASSERT_EQ(NnopbaseExecutorArgsGetDfxInfo(&executor_, &argsAddr, 1U, nullptr), OK);
    const auto* extension = launchBuf_.data() + 16U;
    EXPECT_EQ(extension[0U], 0x4FU);
    EXPECT_EQ(extension[1U], 0x03U);

    uint64_t size = 0U;
    std::memcpy(&size, extension + 2U, sizeof(size));
    EXPECT_EQ(size, 19U);
    EXPECT_EQ(extension[10U], 1U);
    EXPECT_EQ(extension[11U], 0U);
    EXPECT_EQ(extension[12U], 5U);
    uint64_t storageSize = 0U;
    std::memcpy(&storageSize, extension + 13U, sizeof(storageSize));
    EXPECT_EQ(storageSize, 6U);
    const size_t exceptionDumpSize = op::internal::IsArgExceptionDumpEnable() ? sizeof(uint64_t) : 0U;
    EXPECT_EQ(argsAddr.ptr, launchBuf_.data() + 40U + exceptionDumpSize);

    aclDestroyTensor(tensor);
}

TEST_F(NnopbaseIndvArgsTest, AppendOomStorageShapeForTensorList)
{
    const int64_t viewShape[] = {2, 2};
    const int64_t contiguousStrides[] = {2, 1};
    const int64_t nonContiguousStrides[] = {3, 1};
    const int64_t storageShape0[] = {2, 3};
    const int64_t storageShape1[] = {2, 4};
    uint8_t data0[32] = {};
    uint8_t data1[32] = {};
    aclTensor* tensor0 = aclCreateTensor(viewShape, 2U, aclDataType::ACL_FLOAT, nonContiguousStrides, 0,
                                         aclFormat::ACL_FORMAT_ND, storageShape0, 2U, data0);
    aclTensor* tensor1 = aclCreateTensor(viewShape, 2U, aclDataType::ACL_FLOAT, contiguousStrides, 0,
                                         aclFormat::ACL_FORMAT_ND, storageShape1, 2U, data1);
    ASSERT_NE(tensor0, nullptr);
    ASSERT_NE(tensor1, nullptr);
    const aclTensor* tensorListData[] = {tensor0, tensor1};
    aclTensorList* tensorList = aclCreateTensorList(tensorListData, 2U);
    ASSERT_NE(tensorList, nullptr);

    args_.inputs.paramDescs.count = 1U;
    args_.inputs.paramDescs.instances.resize(1U);
    SetParamInstance(args_.inputs.paramDescs.instances[0U], 0U, 2U, true, true);
    args_.inputs.paramDescs.instances[0U].ignoreCont = true;
    args_.inputs.paramDescs.instances[0U].tensorList = tensorList;
    args_.inputs.extTensors.resize(2U);
    args_.inputs.extTensors[0U].storageShape = tensor0->GetStorageShape();
    args_.inputs.extTensors[1U].storageShape = tensor1->GetStorageShape();
    args_.inputs.num = 2U;
    executor_.ownArgs.inputs.paramDescs.count = 1U;
    executor_.ownArgs.inputs.paramDescs.instances.resize(1U);
    SetParamInstance(executor_.ownArgs.inputs.paramDescs.instances[0U], 0U, 2U, true, true);
    executor_.ownArgs.inputs.paramDescs.instances[0U].ignoreCont = true;
    executor_.ownArgs.inputs.paramDescs.instances[0U].tensorList = tensorList;
    args_.dfxInfo.resize(2U);
    binInfo_.oomConfig.flag = true;
    binInfo_.oomConfig.storageShapeEnabled = true;
    binInfo_.oomConfig.version = 2U;
    binInfo_.oomConfig.tensorVersion = 7U;
    NnopbaseExecutorArgsAddr argsAddr{nullptr, nullptr, launchBuf_.data(), nullptr, nullptr};

    ASSERT_EQ(NnopbaseExecutorArgsGetDfxInfo(&executor_, &argsAddr, 1U, nullptr), OK);
    const auto* extension = launchBuf_.data() + 16U;
    EXPECT_EQ(extension[0U], 0x4FU);
    EXPECT_EQ(extension[1U], 0x02U);

    uint64_t size = 0U;
    std::memcpy(&size, extension + 2U, sizeof(size));
    EXPECT_EQ(size, 30U);
    EXPECT_EQ(extension[10U], 2U);
    EXPECT_EQ(extension[11U], 0U);
    EXPECT_EQ(extension[12U], 2U);
    EXPECT_EQ(extension[13U], 0U);
    EXPECT_EQ(extension[14U], 7U);
    uint64_t storageSize0 = 0U;
    std::memcpy(&storageSize0, extension + 15U, sizeof(storageSize0));
    EXPECT_EQ(storageSize0, 6U);
    EXPECT_EQ(extension[23U], 7U);
    uint64_t storageSize1 = 0U;
    std::memcpy(&storageSize1, extension + 24U, sizeof(storageSize1));
    EXPECT_EQ(storageSize1, 8U);
    const size_t exceptionDumpSize = op::internal::IsArgExceptionDumpEnable() ? sizeof(uint64_t) : 0U;
    EXPECT_EQ(argsAddr.ptr, launchBuf_.data() + 48U + exceptionDumpSize);

    aclDestroyTensorList(tensorList);
}

TEST_F(NnopbaseIndvArgsTest, AppendOomStorageShapeForMixedTensorAndTensorList)
{
    const int64_t viewShape[] = {2, 2};
    const int64_t tensorStrides[] = {3, 1};
    const int64_t listStrides0[] = {4, 1};
    const int64_t listStrides1[] = {2, 1};
    const int64_t tensorStorageShape[] = {2, 3};
    const int64_t listStorageShape0[] = {2, 4};
    const int64_t listStorageShape1[] = {2, 5};
    uint8_t tensorData[32] = {};
    uint8_t listData0[32] = {};
    uint8_t listData1[32] = {};
    aclTensor* tensor = aclCreateTensor(viewShape, 2U, aclDataType::ACL_FLOAT, tensorStrides, 0,
                                         aclFormat::ACL_FORMAT_ND, tensorStorageShape, 2U, tensorData);
    aclTensor* listTensor0 = aclCreateTensor(viewShape, 2U, aclDataType::ACL_FLOAT, listStrides0, 0,
                                              aclFormat::ACL_FORMAT_ND, listStorageShape0, 2U, listData0);
    aclTensor* listTensor1 = aclCreateTensor(viewShape, 2U, aclDataType::ACL_FLOAT, listStrides1, 0,
                                              aclFormat::ACL_FORMAT_ND, listStorageShape1, 2U, listData1);
    ASSERT_NE(tensor, nullptr);
    ASSERT_NE(listTensor0, nullptr);
    ASSERT_NE(listTensor1, nullptr);
    const aclTensor* tensorListData[] = {listTensor0, listTensor1};
    aclTensorList* tensorList = aclCreateTensorList(tensorListData, 2U);
    ASSERT_NE(tensorList, nullptr);

    args_.inputs.paramDescs.count = 2U;
    args_.inputs.paramDescs.instances.resize(2U);
    SetParamInstance(args_.inputs.paramDescs.instances[0U], 0U, 1U, false, true);
    SetParamInstance(args_.inputs.paramDescs.instances[1U], 1U, 2U, true, true);
    args_.inputs.paramDescs.instances[0U].tensor = tensor;
    args_.inputs.paramDescs.instances[1U].tensorList = tensorList;
    args_.inputs.extTensors.resize(3U);
    args_.inputs.extTensors[0U].storageShape = tensor->GetStorageShape();
    args_.inputs.extTensors[1U].storageShape = listTensor0->GetStorageShape();
    args_.inputs.extTensors[2U].storageShape = listTensor1->GetStorageShape();
    args_.inputs.num = 3U;

    executor_.ownArgs.inputs.paramDescs.count = 2U;
    executor_.ownArgs.inputs.paramDescs.instances.resize(2U);
    SetParamInstance(executor_.ownArgs.inputs.paramDescs.instances[0U], 0U, 1U, false, true);
    SetParamInstance(executor_.ownArgs.inputs.paramDescs.instances[1U], 1U, 2U, true, true);
    executor_.ownArgs.inputs.paramDescs.instances[0U].tensor = tensor;
    executor_.ownArgs.inputs.paramDescs.instances[1U].tensorList = tensorList;

    args_.dfxInfo.resize(3U);
    binInfo_.oomConfig.flag = true;
    binInfo_.oomConfig.storageShapeEnabled = true;
    binInfo_.oomConfig.version = 1U;
    binInfo_.oomConfig.tensorVersion = 2U;
    NnopbaseExecutorArgsAddr argsAddr{nullptr, nullptr, launchBuf_.data(), nullptr, nullptr};

    ASSERT_EQ(NnopbaseExecutorArgsGetDfxInfo(&executor_, &argsAddr, 1U, nullptr), OK);
    const auto* tensorRecord = launchBuf_.data() + 24U;
    EXPECT_EQ(tensorRecord[0U], 0x4FU);
    EXPECT_EQ(tensorRecord[1U], 1U);
    uint64_t tensorRecordSize = 0U;
    std::memcpy(&tensorRecordSize, tensorRecord + 2U, sizeof(tensorRecordSize));
    EXPECT_EQ(tensorRecordSize, 19U);
    EXPECT_EQ(tensorRecord[10U], 1U);
    EXPECT_EQ(tensorRecord[12U], 2U);
    uint64_t tensorStorageSize = 0U;
    std::memcpy(&tensorStorageSize, tensorRecord + 13U, sizeof(tensorStorageSize));
    EXPECT_EQ(tensorStorageSize, 6U);

    // 新协议：扩展区仅一个2B总Header，RecordBody连续拼接，无每条Record的独立Header
    const auto* tensorListRecord = tensorRecord + 21U;
    uint64_t tensorListRecordSize = 0U;
    std::memcpy(&tensorListRecordSize, tensorListRecord, sizeof(tensorListRecordSize));
    EXPECT_EQ(tensorListRecordSize, 30U);
    EXPECT_EQ(tensorListRecord[8U], 2U);
    EXPECT_EQ(tensorListRecord[9U], 0U);
    EXPECT_EQ(tensorListRecord[10U], 2U);
    EXPECT_EQ(tensorListRecord[12U], 2U);
    EXPECT_EQ(tensorListRecord[21U], 2U);
    uint64_t listStorageSize0 = 0U;
    uint64_t listStorageSize1 = 0U;
    std::memcpy(&listStorageSize0, tensorListRecord + 13U, sizeof(listStorageSize0));
    std::memcpy(&listStorageSize1, tensorListRecord + 22U, sizeof(listStorageSize1));
    EXPECT_EQ(listStorageSize0, 8U);
    EXPECT_EQ(listStorageSize1, 10U);

    const size_t exceptionDumpSize = op::internal::IsArgExceptionDumpEnable() ? sizeof(uint64_t) : 0U;
    EXPECT_EQ(argsAddr.ptr, launchBuf_.data() + 80U + exceptionDumpSize);
    aclDestroyTensor(tensor);
    aclDestroyTensorList(tensorList);
}

TEST_F(NnopbaseIndvArgsTest, RefreshCachedStorageShapeBeforeOomSerialization)
{
    const int64_t viewShape[] = {2, 2};
    const int64_t viewStrides[] = {3, 1};
    const int64_t storageShape[] = {2, 4};
    uint8_t data[32] = {};
    aclTensor* tensor = aclCreateTensor(viewShape, 2U, aclDataType::ACL_FLOAT, viewStrides, 0,
                                        aclFormat::ACL_FORMAT_ND, storageShape, 2U, data);
    ASSERT_NE(tensor, nullptr);

    GertShape oldStorageShape({2, 3});
    args_.inputs.paramDescs.count = 1U;
    args_.inputs.paramDescs.instances.resize(1U);
    SetParamInstance(args_.inputs.paramDescs.instances[0U], 0U, 1U, false, true);
    args_.inputs.extTensors.resize(1U);
    args_.inputs.extTensors[0U].storageShape = oldStorageShape;
    args_.inputs.num = 1U;
    executor_.ownArgs.inputs.paramDescs.count = 1U;
    executor_.ownArgs.inputs.paramDescs.instances.resize(1U);
    SetParamInstance(executor_.ownArgs.inputs.paramDescs.instances[0U], 0U, 1U, false, true);
    executor_.ownArgs.inputs.paramDescs.instances[0U].tensor = tensor;

    args_.dfxInfo.resize(2U);
    binInfo_.oomConfig.flag = true;
    binInfo_.oomConfig.storageShapeEnabled = true;
    binInfo_.oomConfig.version = 1U;
    binInfo_.oomConfig.tensorVersion = 2U;

    ASSERT_EQ(NnopbaseRefreshInputStorageShape(&executor_), OK);
    EXPECT_EQ(args_.inputs.extTensors[0U].storageShape, tensor->GetStorageShape());
    EXPECT_EQ(args_.inputs.extTensors[0U].storageShape.GetShapeSize(), 8);

    NnopbaseExecutorArgsAddr argsAddr{nullptr, nullptr, launchBuf_.data(), nullptr, nullptr};
    ASSERT_EQ(NnopbaseExecutorArgsGetDfxInfo(&executor_, &argsAddr, 1U, nullptr), OK);
    uint64_t storageShapeSize = 0U;
    std::memcpy(&storageShapeSize, launchBuf_.data() + 29U, sizeof(storageShapeSize));
    EXPECT_EQ(storageShapeSize, 8U);

    aclDestroyTensor(tensor);
}

TEST_F(NnopbaseIndvArgsTest, OomStorageShapeMaxSizeReservesAlignmentPadding)
{
    const int64_t shape[] = {2, 2};
    const int64_t storageShape[] = {2, 3};
    uint8_t data[32] = {};
    aclTensor* tensor = aclCreateTensor(shape, 2U, aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, storageShape, 2U, data);
    ASSERT_NE(tensor, nullptr);

    executor_.ownArgs.inputs.paramDescs.count = 1U;
    executor_.ownArgs.inputs.paramDescs.instances.resize(1U);
    SetParamInstance(executor_.ownArgs.inputs.paramDescs.instances[0U], 0U, 1U, false, true);
    executor_.ownArgs.inputs.paramDescs.instances[0U].tensor = tensor;
    binInfo_.oomConfig.flag = true;
    binInfo_.oomConfig.storageShapeEnabled = true;

    EXPECT_EQ(NnopbaseGetOomInfoExtMaxSize(&executor_), 24U);

    aclDestroyTensor(tensor);
}
