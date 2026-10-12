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
#include "mockcpp/mockcpp.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define private public
#define protected public
#include "opdev/direct_invoke_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_arg_def.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "kernel_launcher.h"
#undef protected
#undef private

#include "aclnn/acl_meta.h"
#include "direct_invoke_replay.h"
#include "op_cache_internal.h"
#include "depends/profiler/profiler_stub.h"

namespace {

class DirectInvokeProfilerStub : public ProfilerStub {
public:
    int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi* api) override
    {
        (void)agingFlag;
        if (api != nullptr && api->type == MSPROF_REPORT_NODE_HOST_OP_EXEC_TYPE) {
            ++hostExecReportCount;
            lastHostSummaryId = api->itemId;
        } else if (api != nullptr && api->type == MSPROF_REPORT_NODE_LAUNCH_TYPE) {
            ++kernelLaunchReportCount;
        }
        return 0;
    }

    int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length) override
    {
        (void)agingFlag;
        (void)length;
        if (data != nullptr) {
            const auto* info = static_cast<const MsprofCompactInfo*>(data);
            if (info->type == MSPROF_REPORT_NODE_BASIC_INFO_TYPE) {
                ++basicInfoReportCount;
            } else if (info->type == MSPROF_REPORT_NODE_ATTR_INFO_TYPE) {
                ++attrInfoReportCount;
            }
        }
        return 0;
    }

    static void Reset()
    {
        basicInfoReportCount = 0U;
        attrInfoReportCount = 0U;
        hostExecReportCount = 0U;
        kernelLaunchReportCount = 0U;
        lastHostSummaryId = 0U;
    }

    inline static uint32_t basicInfoReportCount{0U};
    inline static uint32_t attrInfoReportCount{0U};
    inline static uint32_t hostExecReportCount{0U};
    inline static uint32_t kernelLaunchReportCount{0U};
    inline static uint64_t lastHostSummaryId{0U};
};

DirectInvokeProfilerStub g_directInvokeProfilerStub;

struct FixtureConfig {
    aclnnStatus prepareStatus{ACLNN_SUCCESS};
    aclnnStatus launchStatus{ACLNN_SUCCESS};
    uint32_t prepareCount{0U};
    uint32_t launchCount{0U};
    uint32_t destroyCount{0U};
    aclrtStream observedStream{nullptr};
    void* observedWorkspaceAddress{nullptr};
    void* observedState{nullptr};
    void* observedInputAddress{nullptr};
    void* observedOutputAddress{nullptr};
    uint32_t observedWorkspaceCount{0U};
    bool observedArguments{false};
    bool observedEmptyWorkspaceAtPrepare{false};
    // Workspace request reported by the fixture prepare.
    std::vector<uint64_t> workspaceSizes{};
    bool requestNullSizes{false};
};

// Observation hooks only: callbacks never derive computation from these globals.
thread_local FixtureConfig* g_fixtureObserver = nullptr;
struct ObserveFixture {
    FixtureConfig* previous;
    explicit ObserveFixture(FixtureConfig& config) : previous(g_fixtureObserver) { g_fixtureObserver = &config; }
    ~ObserveFixture() { g_fixtureObserver = previous; }
};

#define FIXTURE_ATTR(config)                                                                             \
    OP_ATTR(static_cast<uint64_t>((config).prepareStatus), static_cast<uint64_t>((config).launchStatus), \
            static_cast<uint64_t>((config).workspaceSizes.size()),                                       \
            (config).workspaceSizes.empty() ? uint64_t{0} : (config).workspaceSizes[0],                  \
            (config).workspaceSizes.size() < 2 ? uint64_t{0} : (config).workspaceSizes[1], (config).requestNullSizes)

uint64_t TotalWorkspaceBytes(const FixtureConfig& config)
{
    uint64_t total = 0U;
    for (const uint64_t size : config.workspaceSizes) {
        total += size;
    }
    return total;
}

struct FixtureState {
    FixtureConfig* config; // Test observations only.
    aclnnStatus launchStatus;
    std::vector<uint64_t> workspaceSizes;
};

struct MultiDispatchConfig {
    uint32_t prepareCount{0U};
    uint32_t launchCount{0U};
    uint32_t destroyCount{0U};
    uint32_t dispatchCount{0U};
    std::array<uint32_t, 2U> dispatchOrder{};
    std::array<aclrtStream, 2U> observedStreams{};
};

thread_local MultiDispatchConfig* g_multiDispatchObserver = nullptr;

struct MultiDispatchState {
    MultiDispatchConfig* config;
};

aclnnStatus PrepareMultiDispatch(const op::OpArgContext* args, op::DirectInvokeWorkspaceRequest* workspaceRequest,
                                 void** launchState) noexcept
{
    auto* config = g_multiDispatchObserver;
    if (config == nullptr || args == nullptr || workspaceRequest == nullptr || launchState == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    ++config->prepareCount;
    workspaceRequest->count = 0U;
    auto* state = new (std::nothrow) MultiDispatchState{config};
    if (state == nullptr) {
        return ACLNN_ERR_INNER;
    }
    *launchState = state;
    return ACLNN_SUCCESS;
}

aclnnStatus LaunchMultipleOperations(void* launchState, const op::OpArgContext* args, aclrtStream stream) noexcept
{
    auto* state = static_cast<MultiDispatchState*>(launchState);
    if (state == nullptr || state->config == nullptr || args == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    MultiDispatchConfig& config = *state->config;
    ++config.launchCount;
    // Model two producer-owned asynchronous dispatches from one launch callback.
    // opbase passes the caller's stream through unchanged and keeps one launcher
    // lifecycle around the whole callback.
    config.dispatchOrder[config.dispatchCount] = 1U;
    config.observedStreams[config.dispatchCount++] = stream;
    config.dispatchOrder[config.dispatchCount] = 2U;
    config.observedStreams[config.dispatchCount++] = stream;
    return ACLNN_SUCCESS;
}

void DestroyMultiDispatch(void* launchState) noexcept
{
    auto* state = static_cast<MultiDispatchState*>(launchState);
    if (state == nullptr) {
        return;
    }
    ++state->config->destroyCount;
    delete state;
}

aclnnStatus PrepareFixture(const op::OpArgContext* args, op::DirectInvokeWorkspaceRequest* workspaceRequest,
                           void** launchState) noexcept
{
    auto* config = g_fixtureObserver;
    if (config == nullptr || args == nullptr || workspaceRequest == nullptr || launchState == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    ++config->prepareCount;
    const op::OpArgList* inputs = args->GetOpArg(op::OP_INPUT_ARG);
    const op::OpArgList* outputs = args->GetOpArg(op::OP_OUTPUT_ARG);
    const op::OpArgList* attributes = args->GetOpArg(op::OP_ATTR_ARG);
    const op::OpArgList* workspaces = args->GetOpArg(op::OP_WORKSPACE_ARG);
    config->observedArguments = inputs != nullptr && inputs->count == 1U && outputs != nullptr &&
                                outputs->count == 1U && attributes != nullptr && attributes->count == 6U;
    // The entry workspace list must still be empty while prepare runs; opbase
    // appends the requested workspaces only after prepare succeeds.
    config->observedEmptyWorkspaceAtPrepare = workspaces == nullptr || workspaces->count == 0U;

    if (attributes == nullptr || attributes->count != 6U) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto* state = new (std::nothrow) FixtureState{config, ACLNN_SUCCESS, {}};
    if (state == nullptr) {
        return ACLNN_ERR_INNER;
    }
    *launchState = state;
    state->launchStatus = static_cast<aclnnStatus>(attributes->args[1].value.data.value);
    const auto count = attributes->args[2].value.data.value;
    if (count > 2U) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (uint64_t i = 0U; i < count; ++i) {
        state->workspaceSizes.push_back(attributes->args[3U + i].value.data.value);
    }
    workspaceRequest->sizes = attributes->args[5].value.data.value ? nullptr : state->workspaceSizes.data();
    workspaceRequest->count = static_cast<uint32_t>(count);
    return static_cast<aclnnStatus>(attributes->args[0].value.data.value);
}

aclnnStatus LaunchFixture(void* launchState, const op::OpArgContext* args, aclrtStream stream) noexcept
{
    auto* state = static_cast<FixtureState*>(launchState);
    if (state == nullptr || state->config == nullptr || args == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    FixtureConfig& config = *state->config;
    ++config.launchCount;
    config.observedStream = stream;
    config.observedState = launchState;
    const auto* inputs = args->GetOpArg(op::OP_INPUT_ARG);
    const auto* outputs = args->GetOpArg(op::OP_OUTPUT_ARG);
    if (inputs->count == 1U && outputs->count == 1U) {
        config.observedInputAddress = static_cast<const aclTensor*>(inputs->args[0].value.data.pointer)->GetData();
        config.observedOutputAddress = static_cast<const aclTensor*>(outputs->args[0].value.data.pointer)->GetData();
    }

    const op::OpArgList* workspaces = args->GetOpArg(op::OP_WORKSPACE_ARG);
    if (workspaces == nullptr) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (state->workspaceSizes.empty()) {
        return workspaces->count == 0U ? state->launchStatus : ACLNN_ERR_PARAM_INVALID;
    }
    // opbase appends the requested workspaces as one tensor list argument.
    if (workspaces->count != 1U || workspaces->args[0].type != op::OpArgType::OPARG_ACLTENSOR_LIST) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto* workspaceList = static_cast<const aclTensorList*>(workspaces->args[0].value.data.pointer);
    if (workspaceList == nullptr || workspaceList->Size() != state->workspaceSizes.size()) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    config.observedWorkspaceCount = static_cast<uint32_t>(workspaceList->Size());
    for (uint64_t index = 0U; index < workspaceList->Size(); ++index) {
        const aclTensor* workspace = (*workspaceList)[index];
        if (workspace == nullptr || !workspace->IsFromWorkspace() || workspace->GetStorageAddr() == nullptr) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    config.observedWorkspaceAddress = (*workspaceList)[0]->GetStorageAddr();
    return state->launchStatus;
}

void DestroyFixture(void* launchState) noexcept
{
    auto* state = static_cast<FixtureState*>(launchState);
    if (state == nullptr) {
        return;
    }
    ++state->config->destroyCount;
    delete state;
}

uint32_t DirectInvokeFixtureOpTypeId()
{
    static uint32_t opType = 0U;
    static bool initialized = false;
    if (!initialized) {
        EXPECT_EQ(op::OpTypeDict::Add(opType, "DirectInvokeFixture"), ACLNN_SUCCESS);
        initialized = true;
    }
    return opType;
}

struct TensorFixture {
    float inputData[4]{1.0F, 2.0F, 3.0F, 4.0F};
    float outputData[4]{};
    op::Shape shape{4};
    std::unique_ptr<aclTensor> input{
        std::make_unique<aclTensor>(shape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, inputData)};
    std::unique_ptr<aclTensor> output{
        std::make_unique<aclTensor>(shape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, outputData)};

    TensorFixture() { output->SetFromWorkspace(false); }
};

void ExpectTaskRejectedBeforePrepare(op::DirectInvokePrepareFn prepare, op::DirectInvokeLaunchFn launch,
                                     op::DirectInvokeDestroyFn destroy, FixtureConfig& config)
{
    TensorFixture tensors;
    auto executor = CREATE_EXECUTOR();
    auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                     FIXTURE_ATTR(config), OP_WORKSPACE());

    EXPECT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), prepare, launch, destroy,
                                      executor.get(), args),
              ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(config.prepareCount, 0U);
    EXPECT_EQ(config.launchCount, 0U);
    EXPECT_EQ(config.destroyCount, 0U);
}

} // namespace

TEST(DirectInvokeTaskTest, RunsNeutralTaskWithFinalWorkspaceAddressAndStream)
{
    TensorFixture tensors;
    FixtureConfig config;
    config.workspaceSizes = {4096U};
    ObserveFixture observer(config);
    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();

    op::OpArgContext* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                                 FIXTURE_ATTR(config), OP_WORKSPACE());
    ASSERT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                      LaunchFixture, DestroyFixture, executor, args),
              ACLNN_SUCCESS);
    EXPECT_EQ(config.prepareCount, 1U);
    EXPECT_TRUE(config.observedArguments);
    EXPECT_TRUE(config.observedEmptyWorkspaceAtPrepare);

    const uint64_t workspaceBytes = uniqueExecutor->GetWorkspaceSize();
    ASSERT_GE(workspaceBytes, TotalWorkspaceBytes(config));
    std::vector<uint8_t> runtimeWorkspace(workspaceBytes);
    const auto stream = reinterpret_cast<aclrtStream>(0x2468);
    uniqueExecutor.ReleaseTo(&executor);
    ASSERT_EQ(CommonOpExecutorRun(runtimeWorkspace.data(), workspaceBytes, executor, stream), ACLNN_SUCCESS);

    EXPECT_EQ(config.launchCount, 1U);
    EXPECT_EQ(config.destroyCount, 1U);
    EXPECT_EQ(config.observedStream, stream);
    EXPECT_EQ(config.observedWorkspaceCount, 1U);
    const uintptr_t observed = reinterpret_cast<uintptr_t>(config.observedWorkspaceAddress);
    const uintptr_t begin = reinterpret_cast<uintptr_t>(runtimeWorkspace.data());
    EXPECT_GE(observed, begin);
    EXPECT_LT(observed, begin + runtimeWorkspace.size());
}

TEST(DirectInvokeTaskTest, AllowsMultipleProducerDispatchesOnCallerStream)
{
    TensorFixture tensors;
    MultiDispatchConfig config;
    g_multiDispatchObserver = &config;
    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                     OP_ATTR(int32_t{1}), OP_WORKSPACE());
    ASSERT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareMultiDispatch,
                                      LaunchMultipleOperations, DestroyMultiDispatch, executor, args),
              ACLNN_SUCCESS);

    const auto stream = reinterpret_cast<aclrtStream>(0x369C);
    uniqueExecutor.ReleaseTo(&executor);
    ASSERT_EQ(CommonOpExecutorRun(nullptr, 0U, executor, stream), ACLNN_SUCCESS);

    EXPECT_EQ(config.prepareCount, 1U);
    EXPECT_EQ(config.launchCount, 1U);
    EXPECT_EQ(config.dispatchCount, 2U);
    EXPECT_EQ(config.dispatchOrder, (std::array<uint32_t, 2U>{1U, 2U}));
    EXPECT_EQ(config.observedStreams, (std::array<aclrtStream, 2U>{stream, stream}));
    EXPECT_EQ(config.destroyCount, 1U);
}

TEST(DirectInvokeTaskTest, RejectsMissingCallbacksBeforePrepare)
{
    FixtureConfig config;
    ObserveFixture observer(config);
    ExpectTaskRejectedBeforePrepare(nullptr, LaunchFixture, DestroyFixture, config);
    ExpectTaskRejectedBeforePrepare(PrepareFixture, nullptr, DestroyFixture, config);
    ExpectTaskRejectedBeforePrepare(PrepareFixture, LaunchFixture, nullptr, config);
}

TEST(DirectInvokeTaskTest, RejectsNonEmptyWorkspaceArgsAtEntry)
{
    // Workspace buffers are requested through prepare; supplying any workspace
    // argument at entry is rejected, even a valid executor-allocated one.
    TensorFixture tensors;
    FixtureConfig config;
    ObserveFixture observer(config);
    auto executor = CREATE_EXECUTOR();
    const aclTensor* workspace = executor->AllocTensor(tensors.shape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    ASSERT_NE(workspace, nullptr);
    auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                     FIXTURE_ATTR(config), OP_WORKSPACE(workspace));

    EXPECT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                      LaunchFixture, DestroyFixture, executor.get(), args),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(config.prepareCount, 0U);
    EXPECT_EQ(config.destroyCount, 0U);
}

TEST(DirectInvokeTaskTest, DestroysStateWhenPrepareFails)
{
    TensorFixture tensors;
    FixtureConfig config;
    config.prepareStatus = ACLNN_ERR_INNER;
    ObserveFixture observer(config);
    auto executor = CREATE_EXECUTOR();
    auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                     FIXTURE_ATTR(config), OP_WORKSPACE());

    EXPECT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                      LaunchFixture, DestroyFixture, executor.get(), args),
              ACLNN_ERR_INNER);
    EXPECT_EQ(config.prepareCount, 1U);
    EXPECT_EQ(config.launchCount, 0U);
    EXPECT_EQ(config.destroyCount, 1U);
}

TEST(DirectInvokeTaskTest, RejectsOversizedWorkspaceAndDestroysState)
{
    TensorFixture tensors;
    FixtureConfig config;
    config.workspaceSizes = {UINT64_MAX};
    ObserveFixture observer(config);
    auto executor = CREATE_EXECUTOR();
    auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                     FIXTURE_ATTR(config), OP_WORKSPACE());

    EXPECT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                      LaunchFixture, DestroyFixture, executor.get(), args),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(config.prepareCount, 1U);
    EXPECT_EQ(config.launchCount, 0U);
    EXPECT_EQ(config.destroyCount, 1U);
}

TEST(DirectInvokeTaskTest, RejectsZeroSizedWorkspaceEntryAndDestroysState)
{
    TensorFixture tensors;
    FixtureConfig config;
    config.workspaceSizes = {1024U, 0U};
    ObserveFixture observer(config);
    auto executor = CREATE_EXECUTOR();
    auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                     FIXTURE_ATTR(config), OP_WORKSPACE());

    EXPECT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                      LaunchFixture, DestroyFixture, executor.get(), args),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(config.prepareCount, 1U);
    EXPECT_EQ(config.launchCount, 0U);
    EXPECT_EQ(config.destroyCount, 1U);
}

TEST(DirectInvokeTaskTest, RejectsNullSizesWithNonZeroCountAndDestroysState)
{
    TensorFixture tensors;
    FixtureConfig config;
    config.workspaceSizes = {1024U};
    config.requestNullSizes = true;
    ObserveFixture observer(config);
    auto executor = CREATE_EXECUTOR();
    auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                     FIXTURE_ATTR(config), OP_WORKSPACE());

    EXPECT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                      LaunchFixture, DestroyFixture, executor.get(), args),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(config.prepareCount, 1U);
    EXPECT_EQ(config.launchCount, 0U);
    EXPECT_EQ(config.destroyCount, 1U);
}

TEST(DirectInvokeTaskTest, PropagatesLaunchFailureAndDestroysState)
{
    TensorFixture tensors;
    FixtureConfig config;
    config.launchStatus = ACLNN_ERR_INNER;
    config.workspaceSizes = {1024U};
    ObserveFixture observer(config);
    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                     FIXTURE_ATTR(config), OP_WORKSPACE());
    ASSERT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                      LaunchFixture, DestroyFixture, executor, args),
              ACLNN_SUCCESS);

    const uint64_t workspaceBytes = uniqueExecutor->GetWorkspaceSize();
    ASSERT_GT(workspaceBytes, 0U);
    std::vector<uint8_t> runtimeWorkspace(workspaceBytes);
    uniqueExecutor.ReleaseTo(&executor);
    EXPECT_EQ(
        CommonOpExecutorRun(runtimeWorkspace.data(), workspaceBytes, executor, reinterpret_cast<aclrtStream>(0x1357)),
        ACLNN_ERR_INNER);
    EXPECT_EQ(config.prepareCount, 1U);
    EXPECT_EQ(config.launchCount, 1U);
    EXPECT_EQ(config.destroyCount, 1U);
}

TEST(DirectInvokeTaskTest, AcceptsNullOptionalInputTensor)
{
    TensorFixture tensors;
    FixtureConfig config;
    ObserveFixture observer(config);
    {
        auto executor = CREATE_EXECUTOR();
        auto* args = op::GetOpArgContext(OP_INPUT(static_cast<const aclTensor*>(nullptr)),
                                         OP_OUTPUT(tensors.output.get()), FIXTURE_ATTR(config), OP_WORKSPACE());

        EXPECT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                          LaunchFixture, DestroyFixture, executor.get(), args),
                  ACLNN_SUCCESS);
        EXPECT_EQ(config.prepareCount, 1U);
        EXPECT_EQ(config.launchCount, 0U);
    }
    EXPECT_EQ(config.destroyCount, 1U);
}

TEST(DirectInvokeTaskTest, RejectsNullOutputTensor)
{
    TensorFixture tensors;
    FixtureConfig config;
    ObserveFixture observer(config);
    auto executor = CREATE_EXECUTOR();
    auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(static_cast<const aclTensor*>(nullptr)),
                                     FIXTURE_ATTR(config), OP_WORKSPACE());

    EXPECT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                      LaunchFixture, DestroyFixture, executor.get(), args),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(config.prepareCount, 0U);
}

TEST(DirectInvokeTaskTest, UsesEngineNeutralComputeCategory)
{
    TensorFixture tensors;
    FixtureConfig config;
    ObserveFixture observer(config);
    auto executor = CREATE_EXECUTOR();
    auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                     FIXTURE_ATTR(config), OP_WORKSPACE());
    ASSERT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                      LaunchFixture, DestroyFixture, executor.get(), args),
              ACLNN_SUCCESS);
    ASSERT_FALSE(executor->kernelLaunchObjList_.empty());
    EXPECT_EQ(executor->kernelLaunchObjList_.back()->coreType_, op::DIRECT_INVOKE);
}

TEST(DirectInvokeTaskTest, AcceptsTensorListArguments)
{
    TensorFixture tensors;
    FixtureConfig config;
    ObserveFixture observer(config);
    const aclTensor* inputArray[] = {tensors.input.get()};
    // Lists passed as op args are owned by the executor afterwards (same
    // convention as test_dump.cpp), so they are intentionally not destroyed.
    aclTensorList* inputList = aclCreateTensorList(inputArray, 1U);
    ASSERT_NE(inputList, nullptr);
    {
        auto executor = CREATE_EXECUTOR();
        auto* args = op::GetOpArgContext(OP_INPUT(inputList), OP_OUTPUT(tensors.output.get()), FIXTURE_ATTR(config),
                                         OP_WORKSPACE());

        EXPECT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                          LaunchFixture, DestroyFixture, executor.get(), args),
                  ACLNN_SUCCESS);
        EXPECT_EQ(config.prepareCount, 1U);
    }
}

TEST(DirectInvokeTaskTest, RejectsMalformedTensorListArguments)
{
    TensorFixture tensors;
    FixtureConfig config;
    ObserveFixture observer(config);
    auto executor = CREATE_EXECUTOR();

    // A null tensor list pointer is malformed, unlike a null optional input tensor.
    auto* nullListArgs = op::GetOpArgContext(OP_INPUT(static_cast<const aclTensorList*>(nullptr)),
                                             OP_OUTPUT(tensors.output.get()), FIXTURE_ATTR(config), OP_WORKSPACE());
    EXPECT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                      LaunchFixture, DestroyFixture, executor.get(), nullListArgs),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(config.prepareCount, 0U);

    // A non-empty workspace list at entry is rejected, including list form.
    const aclTensor* workspace = executor->AllocTensor(tensors.shape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    ASSERT_NE(workspace, nullptr);
    const aclTensor* workspaceArray[] = {workspace};
    aclTensorList* workspaceList = aclCreateTensorList(workspaceArray, 1U);
    ASSERT_NE(workspaceList, nullptr);
    auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                     FIXTURE_ATTR(config), OP_WORKSPACE(workspaceList));
    EXPECT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                      LaunchFixture, DestroyFixture, executor.get(), args),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(config.prepareCount, 0U);
}

TEST(DirectInvokeTaskTest, AllocatesRequestedWorkspaceBuffers)
{
    TensorFixture tensors;
    FixtureConfig config;
    config.workspaceSizes = {128U, 4096U};
    ObserveFixture observer(config);
    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                     FIXTURE_ATTR(config), OP_WORKSPACE());
    ASSERT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                      LaunchFixture, DestroyFixture, executor, args),
              ACLNN_SUCCESS);
    EXPECT_TRUE(config.observedEmptyWorkspaceAtPrepare);

    const uint64_t workspaceBytes = uniqueExecutor->GetWorkspaceSize();
    ASSERT_GE(workspaceBytes, TotalWorkspaceBytes(config));
    std::vector<uint8_t> runtimeWorkspace(workspaceBytes);
    uniqueExecutor.ReleaseTo(&executor);
    ASSERT_EQ(
        CommonOpExecutorRun(runtimeWorkspace.data(), workspaceBytes, executor, reinterpret_cast<aclrtStream>(0x2468)),
        ACLNN_SUCCESS);
    EXPECT_EQ(config.launchCount, 1U);
    EXPECT_EQ(config.observedWorkspaceCount, 2U);
    const uintptr_t observed = reinterpret_cast<uintptr_t>(config.observedWorkspaceAddress);
    const uintptr_t begin = reinterpret_cast<uintptr_t>(runtimeWorkspace.data());
    EXPECT_GE(observed, begin);
    EXPECT_LT(observed, begin + runtimeWorkspace.size());
}

TEST(DirectInvokeTaskTest, AcceptsZeroWorkspaceRequest)
{
    TensorFixture tensors;
    FixtureConfig config;
    ObserveFixture observer(config);
    {
        auto executor = CREATE_EXECUTOR();
        auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                         FIXTURE_ATTR(config), OP_WORKSPACE());

        EXPECT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                          LaunchFixture, DestroyFixture, executor.get(), args),
                  ACLNN_SUCCESS);
        EXPECT_EQ(config.prepareCount, 1U);
        EXPECT_TRUE(config.observedEmptyWorkspaceAtPrepare);
    }
    EXPECT_EQ(config.destroyCount, 1U);
}

TEST(DirectInvokeTaskTest, ReportsNoCallbackBoundaryProfiling)
{
    auto& threadLocalCtx = op::internal::GetThreadLocalContext();
    const op::internal::OpLogInfo oldLogInfo = threadLocalCtx.logInfo_;
    threadLocalCtx.logInfo_.l2ApiName = "aclnnDirectInvokeFixture";
    threadLocalCtx.logInfo_.l2SequenceCounter = 42U;
    const bool oldReportFlag = op::internal::opProfilingSwitch.reportFlag;
    const bool oldKernelLaunchFlag = op::internal::opProfilingSwitch.kernelLaunchFlag;
    const bool oldAdditionInfoFlag = op::internal::opProfilingSwitch.additionInfoFlag;
    op::internal::opProfilingSwitch.reportFlag = true;
    op::internal::opProfilingSwitch.kernelLaunchFlag = true;
    op::internal::opProfilingSwitch.additionInfoFlag = true;
    ProfilerStub::GetInstance()->Install(&g_directInvokeProfilerStub);

    {
        // Each task captures the current L2 context when it is added. Executor
        // execution may temporarily replace TLS, so model a fresh L2 API entry
        // for every independently constructed task.
        threadLocalCtx.logInfo_.l2ApiName = "aclnnDirectInvokeFixture";
        threadLocalCtx.logInfo_.l2SequenceCounter = 42U;
        DirectInvokeProfilerStub::Reset();
        TensorFixture tensors;
        FixtureConfig config;
        config.workspaceSizes = {1024U};
        ObserveFixture observer(config);
        auto uniqueExecutor = CREATE_EXECUTOR();
        aclOpExecutor* executor = uniqueExecutor.get();
        auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                         FIXTURE_ATTR(config), OP_WORKSPACE());
        ASSERT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                          LaunchFixture, DestroyFixture, executor, args),
                  ACLNN_SUCCESS);
        const uint64_t workspaceBytes = uniqueExecutor->GetWorkspaceSize();
        ASSERT_GT(workspaceBytes, 0U);
        std::vector<uint8_t> runtimeWorkspace(workspaceBytes);
        uniqueExecutor.ReleaseTo(&executor);
        ASSERT_EQ(CommonOpExecutorRun(runtimeWorkspace.data(), workspaceBytes, executor,
                                      reinterpret_cast<aclrtStream>(0x2468)),
                  ACLNN_SUCCESS);
        // The callback boundary is not a profiling surface: no host-exec,
        // kernel-launch, or basic-info records may be emitted by the launcher.
        EXPECT_EQ(DirectInvokeProfilerStub::basicInfoReportCount, 0U);
        EXPECT_EQ(DirectInvokeProfilerStub::hostExecReportCount, 0U);
        EXPECT_EQ(DirectInvokeProfilerStub::kernelLaunchReportCount, 0U);
    }

    ProfilerStub::GetInstance()->UnInstall();
    op::internal::opProfilingSwitch.reportFlag = oldReportFlag;
    op::internal::opProfilingSwitch.kernelLaunchFlag = oldKernelLaunchFlag;
    op::internal::opProfilingSwitch.additionInfoFlag = oldAdditionInfoFlag;
    threadLocalCtx.logInfo_ = oldLogInfo;
}

TEST(DirectInvokeTaskTest, SkipsAttrInfoUnderLevel2Profiling)
{
    const bool oldReportFlag = op::internal::opProfilingSwitch.reportFlag;
    const bool oldKernelLaunchFlag = op::internal::opProfilingSwitch.kernelLaunchFlag;
    const bool oldAdditionInfoFlag = op::internal::opProfilingSwitch.additionInfoFlag;
    const bool oldLevel2Flag = op::internal::opProfilingSwitch.level2ProfilingFlag;
    op::internal::opProfilingSwitch.reportFlag = true;
    op::internal::opProfilingSwitch.kernelLaunchFlag = true;
    op::internal::opProfilingSwitch.additionInfoFlag = false;
    op::internal::opProfilingSwitch.level2ProfilingFlag = true;
    ProfilerStub::GetInstance()->Install(&g_directInvokeProfilerStub);

    // Compute tasks do not report device-task metadata at the callback boundary.
    {
        DirectInvokeProfilerStub::Reset();
        TensorFixture tensors;
        FixtureConfig config;
        ObserveFixture observer(config);
        auto uniqueExecutor = CREATE_EXECUTOR();
        aclOpExecutor* executor = uniqueExecutor.get();
        auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                         FIXTURE_ATTR(config), OP_WORKSPACE());
        ASSERT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                          LaunchFixture, DestroyFixture, executor, args),
                  ACLNN_SUCCESS);
        uniqueExecutor.ReleaseTo(&executor);
        ASSERT_EQ(CommonOpExecutorRun(nullptr, 0U, executor, reinterpret_cast<aclrtStream>(0x2468)), ACLNN_SUCCESS);
        EXPECT_EQ(DirectInvokeProfilerStub::attrInfoReportCount, 0U);
    }

    ProfilerStub::GetInstance()->UnInstall();
    op::internal::opProfilingSwitch.reportFlag = oldReportFlag;
    op::internal::opProfilingSwitch.kernelLaunchFlag = oldKernelLaunchFlag;
    op::internal::opProfilingSwitch.additionInfoFlag = oldAdditionInfoFlag;
    op::internal::opProfilingSwitch.level2ProfilingFlag = oldLevel2Flag;
}

TEST(DirectInvokeTaskTest, DelegatesRepeatabilityToArgChecker)
{
    // With every tensor arg skipped by the repeatable checker (workspace-backed
    // storages need no address refresh), the DirectInvoke task launcher must report
    // repeatable instead of the unconditional false it used to return.
    TensorFixture tensors;
    tensors.input->SetFromWorkspace(true);
    tensors.output->SetFromWorkspace(true);
    FixtureConfig config;
    ObserveFixture observer(config);
    auto executor = CREATE_EXECUTOR();
    auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                     FIXTURE_ATTR(config), OP_WORKSPACE());
    ASSERT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                      LaunchFixture, DestroyFixture, executor.get(), args),
              ACLNN_SUCCESS);
    EXPECT_TRUE(executor->CheckLauncherRepeatable());
}

TEST(DirectInvokeTaskTest, PreservesFrameworkRepeatRestrictions)
{
    for (const bool alreadyDisabled : {false, true}) {
        TensorFixture tensors;
        tensors.input->SetFromWorkspace(true);
        tensors.output->SetFromWorkspace(true);
        FixtureConfig config;
        ObserveFixture observer(config);
        auto executor = CREATE_EXECUTOR();
        if (alreadyDisabled) {
            executor->AbandonCache(true);
        }
        auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                         FIXTURE_ATTR(config), OP_WORKSPACE());
        ASSERT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                          LaunchFixture, DestroyFixture, executor.get(), args),
                  ACLNN_SUCCESS);
        EXPECT_EQ(executor->SetRepeatable(), alreadyDisabled ? ACLNN_ERR_INNER : ACLNN_SUCCESS);
        EXPECT_EQ(executor->IsRepeatable(), !alreadyDisabled);
    }
}

TEST(DirectInvokeTaskTest, DynamicOutputShapesDisableRepeat)
{
    TensorFixture tensors;
    FixtureConfig config;
    ObserveFixture observer(config);
    auto executor = CREATE_EXECUTOR();
    auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                     FIXTURE_ATTR(config), OP_OUTSHAPE(tensors.output.get(), uint64_t{0}));
    ASSERT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                      LaunchFixture, DestroyFixture, executor.get(), args),
              ACLNN_SUCCESS);
    EXPECT_EQ(executor->SetRepeatable(), ACLNN_ERR_INNER);
    EXPECT_FALSE(executor->IsRepeatable());
}

TEST(DirectInvokeTaskTest, PreparationUsesValueArgumentsAndOwnsWorkspaceRequest)
{
    TensorFixture tensors;
    FixtureConfig config;
    config.workspaceSizes = {128U, 4096U};
    ObserveFixture observer(config);
    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                     FIXTURE_ATTR(config), OP_WORKSPACE());
    // All functional inputs were copied into args. Changing the caller's data
    // before prepare must not change its result or the requested workspace.
    config.workspaceSizes.clear();
    config.prepareStatus = ACLNN_ERR_INNER;
    config.launchStatus = ACLNN_ERR_INNER;
    config.requestNullSizes = true;
    ASSERT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), PrepareFixture,
                                      LaunchFixture, DestroyFixture, executor, args),
              ACLNN_SUCCESS);
    const uint64_t workspaceBytes = uniqueExecutor->GetWorkspaceSize();
    ASSERT_GE(workspaceBytes, 4224U);
    std::vector<uint8_t> workspace(workspaceBytes);
    uniqueExecutor.ReleaseTo(&executor);
    ASSERT_EQ(CommonOpExecutorRun(workspace.data(), workspaceBytes, executor, reinterpret_cast<aclrtStream>(0x2468)),
              ACLNN_SUCCESS);
    EXPECT_EQ(config.observedWorkspaceCount, 2U);
    EXPECT_EQ(config.prepareCount, 1U);
    EXPECT_EQ(config.destroyCount, 1U);
}

TEST(DirectInvokeTaskTest, StatelessCallbacksRequireNoDestroy)
{
    uint32_t launches = 0U;
    uint32_t destroys = 0U;
    static thread_local uint32_t* observedLaunches;
    static thread_local uint32_t* observedDestroys;
    observedLaunches = &launches;
    observedDestroys = &destroys;
    auto prepare = +[](const op::OpArgContext*, op::DirectInvokeWorkspaceRequest* request,
                       void** state) noexcept -> aclnnStatus {
        if (*state != nullptr || request->sizes != nullptr || request->count != 0U || request->reserved != 0U) {
            return ACLNN_ERR_INNER;
        }
        return ACLNN_SUCCESS;
    };
    auto launch = +[](void* state, const op::OpArgContext*, aclrtStream) noexcept -> aclnnStatus {
        ++*observedLaunches;
        return state == nullptr ? ACLNN_SUCCESS : ACLNN_ERR_INNER;
    };
    auto destroy = +[](void*) noexcept { ++*observedDestroys; };
    auto uniqueExecutor = CREATE_EXECUTOR();
    auto* executor = uniqueExecutor.get();
    auto* args = op::GetOpArgContext(OP_INPUT(), OP_OUTPUT(), OP_ATTR());
    ASSERT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(), prepare, launch, destroy,
                                      executor, args),
              ACLNN_SUCCESS);
    uniqueExecutor.ReleaseTo(&executor);
    EXPECT_EQ(CommonOpExecutorRun(nullptr, 0U, executor, reinterpret_cast<aclrtStream>(0x2468)), ACLNN_SUCCESS);
    EXPECT_EQ(launches, 1U);
    EXPECT_EQ(destroys, 0U);
}

TEST(DirectInvokeTaskTest, RepeatReusesPreparationAndAlwaysLaunchesWithCurrentAddresses)
{
    struct ResetMocks { ~ResetMocks() { GlobalMockObject::verify(); } } resetMocks;
    MOCKER(op::internal::CheckCacheable).stubs().will(returnValue(true));
    auto& tls = op::internal::GetThreadLocalContext();
    struct Restore {
        op::internal::OpThreadLocalContext& tls;
        bool cacheHasFull;
        uint64_t hash;
        std::vector<const aclStorage*> storages;
        size_t storageCount;
        ~Restore() {
            tls.cacheHasFull_ = cacheHasFull;
            tls.hashKey_ = hash;
            tls.cachedStorageList_ = std::move(storages);
            tls.cachedStorageListSize_ = storageCount;
        }
    } restore{tls, tls.cacheHasFull_, tls.hashKey_, tls.cachedStorageList_, tls.cachedStorageListSize_};
    tls.cacheHasFull_ = false;
    tls.hashKey_ = 0xD1U;
    TensorFixture tensors;
    tensors.input->SetFromWorkspace(false);
    tensors.output->SetFromWorkspace(false);
    tls.cachedStorageList_ = {tensors.input->GetStorage(), tensors.output->GetStorage()};
    tls.cachedStorageListSize_ = 2U;
    FixtureConfig config;
    config.workspaceSizes = {1537U};
    ObserveFixture observer(config);
    {
        auto executor = CREATE_EXECUTOR();
        ASSERT_NE(executor->GetOpExecCache(), nullptr);
        ASSERT_TRUE(executor->GetOpExecCache()->IsOpCacheValid());
        auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                         FIXTURE_ATTR(config), OP_WORKSPACE());
        ASSERT_EQ(op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(),
                                         PrepareFixture, LaunchFixture, DestroyFixture, executor.get(), args),
                  ACLNN_SUCCESS);
        // This must hold independently of DFX switches that can also reject cache.
        EXPECT_TRUE(executor->GetOpExecCache()->IsOpCacheValid());
        // OpArgContext snapshots output tensors; model the L2-to-L0 relation
        // used by the public address setters when repeating an executor.
        auto* outputArg = static_cast<aclTensor*>(args->GetOpArg(op::OP_OUTPUT_ARG)->args[0].value.data.pointer);
        executor->AddTensorRelation(tensors.output.get(), outputArg);
        ASSERT_EQ(executor->SetRepeatable(), ACLNN_SUCCESS);
        const auto bytes = executor->GetWorkspaceSize();
        std::vector<uint8_t> firstWorkspace(bytes), secondWorkspace(bytes);
        auto firstStream = reinterpret_cast<aclrtStream>(0x2468);
        auto secondStream = reinterpret_cast<aclrtStream>(0x3579);
        ASSERT_EQ(CommonOpExecutorRun(firstWorkspace.data(), bytes, executor.get(), firstStream), ACLNN_SUCCESS);
        EXPECT_EQ(executor->GetOpExecCache()->taskQueue_.size(), 1U);
        void* state = config.observedState;
        void* firstWorkspaceAddress = config.observedWorkspaceAddress;
        float newInput[4]{}, newOutput[4]{};
        tensors.input->SetStorageAddr(newInput);
        tensors.output->SetStorageAddr(newOutput);
        ASSERT_EQ(CommonOpExecutorRun(secondWorkspace.data(), bytes, executor.get(), secondStream), ACLNN_SUCCESS);
        EXPECT_EQ(config.prepareCount, 1U);
        EXPECT_EQ(config.launchCount, 2U);
        EXPECT_EQ(config.destroyCount, 0U);
        EXPECT_EQ(config.observedState, state);
        EXPECT_EQ(config.observedStream, secondStream);
        EXPECT_EQ(config.observedInputAddress, newInput);
        EXPECT_EQ(config.observedOutputAddress, newOutput);
        EXPECT_NE(config.observedWorkspaceAddress, firstWorkspaceAddress);
    }
    // The repeat executor's destructor deleted its opExecCache while
    // CommonOpExecutorRun left it in the thread-local op cache context
    // (SetOpCache is never reset). Clear the dangling pointer so later tests
    // that launch directly cannot dereference it. See ~OpExecutorImpl.
    op::internal::GetOpCacheContext().SetOpCache(nullptr);
    EXPECT_EQ(config.destroyCount, 1U);
}

TEST(DirectInvokeTaskTest, CacheSnapshotOwnsArgumentsAndRebindsOffsetsAndWorkspace)
{
    using namespace op;
    using namespace op::internal;
    int destroyed = 0;
    struct State { int* destroyed; int calls{0}; };
    auto state = std::shared_ptr<void>(new State{&destroyed}, [](void* p) {
        auto* s = static_cast<State*>(p);
        ++*s->destroyed;
        delete s;
    });
    float first[32]{}, second[32]{};
    uint8_t workspace[128]{};
    {
        OpExecCache cache;
        alignas(64) char buffer[512]{};
        cache.SetCacheBuf(buffer);
        cache.hashKey_ = 1;
        const int64_t dims[] = {4}, strides[] = {2}, storage[] = {32};
        auto* input = aclCreateTensor(dims, 1, ACL_FLOAT, strides, 3, ACL_FORMAT_ND, storage, 1, first);
        input->SetStorageOffset(2);
        auto* output = aclCreateTensor(dims, 1, ACL_FLOAT, strides, 1, ACL_FORMAT_ND, storage, 1, first);
        auto* scratch = aclCreateTensor(dims, 1, ACL_UINT8, strides, 2, ACL_FORMAT_ND, storage, 1, workspace);
        scratch->SetFromWorkspace(true);
        scratch->SetWorkspaceOffset(16);
        cache.cachedStorageList_ = {input->GetStorage(), output->GetStorage()};
        const aclTensor* items[] = {nullptr, input, input};
        auto* list = aclCreateTensorList(items, 3);
        const aclTensor* workspaces[] = {scratch};
        auto* workspaceList = aclCreateTensorList(workspaces, 1);
        int64_t attrValues[] = {7, 9};
        auto* array = aclCreateIntArray(attrValues, 2);
        float scalarValue = 0.5F;
        auto* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
        auto* args = GetOpArgContext(OP_INPUT(list), OP_OUTPUT(output), OP_ATTR(array, scalar, "snapshot"),
                                     OP_WORKSPACE(workspaceList));
        auto replay = std::make_shared<DirectInvokeReplay>(state);
        ASSERT_TRUE(replay->Init(*args, cache));
        DestroyOpArgContext(args);
        aclDestroyIntArray(array);
        aclDestroyScalar(scalar);
        for (size_t i = 0; i < list->Size(); ++i) (*list)[i] = nullptr;
        aclDestroyTensorList(list);
        aclDestroyTensorList(workspaceList);
        aclDestroyTensor(input);
        aclDestroyTensor(output);
        DirectInvokeLaunchFn launch = [](void* p, const OpArgContext* current, aclrtStream stream) noexcept {
            auto* s = static_cast<State*>(p);
            ++s->calls;
            const auto* list = static_cast<const aclTensorList*>(current->GetOpArg(OP_INPUT_ARG)->args[0].value.data.pointer);
            EXPECT_EQ((*list)[0], nullptr);
            EXPECT_EQ((*list)[1], (*list)[2]);
            EXPECT_EQ((*list)[1]->GetData(), static_cast<char*>(reinterpret_cast<void*>(stream)) + 5 * sizeof(float));
            const auto* attrs = current->GetOpArg(OP_ATTR_ARG);
            EXPECT_EQ((*static_cast<const aclIntArray*>(attrs->args[0].value.data.pointer))[1], 9);
            EXPECT_FLOAT_EQ(*static_cast<const float*>(static_cast<const aclScalar*>(attrs->args[1].value.data.pointer)->GetData()), 0.5F);
            EXPECT_STREQ(static_cast<const char*>(attrs->args[2].value.data.pointer), "snapshot");
            const auto* work = static_cast<const aclTensorList*>(current->GetOpArg(OP_WORKSPACE_ARG)->args[0].value.data.pointer);
            EXPECT_EQ(static_cast<char*>((*work)[0]->GetData()) - static_cast<char*>((*work)[0]->GetStorageAddr()), 18);
            return ACLNN_SUCCESS;
        };
        ASSERT_TRUE(DirectInvokeReplay::SaveToCache(cache, replay, launch));
        cache.Finalize();
        ASSERT_NE(cache.cacheBuf_, buffer + 1);
        std::memset(buffer, 0xa5, sizeof(buffer));
        replay.reset();
        auto* observedState = static_cast<State*>(state.get());
        state.reset();
        ASSERT_EQ(destroyed, 0);
        EXPECT_EQ(cache.Run(workspace, reinterpret_cast<aclrtStream>(first), {first, second}), ACLNN_SUCCESS);
        EXPECT_EQ(cache.Run(workspace, reinterpret_cast<aclrtStream>(second), {second, first}), ACLNN_SUCCESS);
        EXPECT_EQ(observedState->calls, 2);
        EXPECT_EQ(destroyed, 0);
    }
    EXPECT_EQ(destroyed, 1);
}

TEST(DirectInvokeTaskTest, CacheAssociatesDfxWithMixedTasksAndStopsOnError)
{
    struct ResetMocks { ~ResetMocks() { GlobalMockObject::verify(); } } resetMocks;
    using namespace op::internal;
    MOCKER(IsExceptionDumpEnable).stubs().will(returnValue(true));
    MOCKER(DoExceptionDump).expects(exactly(2));
    MOCKER(DoReportAdditionInfo).expects(exactly(2)).will(returnValue(ACLNN_SUCCESS));
    MOCKER(ReportCacheOpInfoFromCache).expects(exactly(2));
    const auto oldProfiling = opProfilingSwitch;
    const auto oldTls = GetThreadLocalContext();
    struct RestoreDfx {
        decltype(opProfilingSwitch) profiling;
        decltype(oldTls) tls;
        ~RestoreDfx() { opProfilingSwitch = profiling; GetThreadLocalContext() = tls; }
    } restore{oldProfiling, oldTls};
    opProfilingSwitch = {};
    opProfilingSwitch.kernelLaunchFlag = true;
    opProfilingSwitch.additionInfoFlag = true;
    GetThreadLocalContext().cacheOpInfoSwitch_ = true;
    OpExecCache cache;
    alignas(64) char buffer[512]{};
    cache.SetCacheBuf(buffer);
    std::vector<int> order;
    struct Record { std::vector<int>* order; int id; uint32_t blocks; aclnnStatus result; };
    auto runner = [](aclrtStream, void* data) -> aclnnStatus {
        Record record;
        std::memcpy(&record, data, sizeof(record));
        record.order->push_back(record.id);
        if (record.blocks != 0) {
            EXPECT_EQ(GetThreadLocalContext().numBlocks_, record.blocks);
            EXPECT_EQ(GetThreadLocalContext().profilingInfoId_.summaryItemId_, record.id);
        }
        return record.result;
    };
    auto add = [&](int id, bool hasDfx, aclnnStatus result) {
        size_t offset = 0, capacity = 0;
        cache.NewLaunchCache(&offset, &capacity, runner,
            hasDfx ? op::AI_CORE : op::DIRECT_INVOKE);
        const Record record{&order, id, hasDfx ? static_cast<uint32_t>(id * 10) : 0U, result};
        std::memcpy(cache.AddLaunchData(sizeof(record)), &record, sizeof(record));
        if (hasDfx) {
            cache.SetBlockDim(record.blocks);
            cache.opExecCacheDfx_->SetProfilingInfoId(ProfilingInfoId(0, 0, id));
            TaskInfo info{};
            info.type = MSPROF_GE_TASK_TYPE_AI_CORE;
            cache.opExecCacheDfx_->SetTaskInfo(info);
            cache.opExecCacheDfx_->SetExceptionDumpInfo({});
            cache.SetCacheTensorInfo(nullptr);
        }
    };
    add(1, false, ACLNN_SUCCESS);
    add(2, true, ACLNN_SUCCESS);
    add(3, false, ACLNN_SUCCESS);
    add(4, true, ACLNN_SUCCESS);
    add(5, false, ACLNN_ERR_INNER);
    add(6, true, ACLNN_SUCCESS);
    ASSERT_EQ(cache.taskQueue_.size(), 6U);
    ASSERT_EQ(cache.numBlocks_.size(), 3U);
    ASSERT_EQ(cache.cacheTensorInfoLists_.size(), 3U);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(cache.taskQueue_[i].coreType == op::AI_CORE, i % 2 == 1);
        EXPECT_NE(cache.taskQueue_[i].runner, nullptr);
    }
    EXPECT_EQ(cache.Run(nullptr, nullptr, {}), ACLNN_ERR_INNER);
    EXPECT_EQ(order, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(DirectInvokeTaskTest, FrameworkDfxClosesProfilingOnRunnerFailure)
{
    using namespace op::internal;
    const auto oldProfiling = opProfilingSwitch;
    const auto oldTls = GetThreadLocalContext();
    struct RestoreDfx {
        decltype(opProfilingSwitch) profiling;
        decltype(oldTls) tls;
        ~RestoreDfx()
        {
            ProfilerStub::GetInstance()->UnInstall();
            opProfilingSwitch = profiling;
            GetThreadLocalContext() = tls;
        }
    } restore{oldProfiling, oldTls};
    opProfilingSwitch = {};
    opProfilingSwitch.reportFlag = true;
    opProfilingSwitch.kernelLaunchFlag = true;
    DirectInvokeProfilerStub::Reset();
    ProfilerStub::GetInstance()->Install(&g_directInvokeProfilerStub);
    OpExecCache cache;
    char buffer[32]{};
    cache.SetCacheBuf(buffer);
    size_t offset = 0, capacity = 0;
    cache.NewLaunchCache(&offset, &capacity,
        [](aclrtStream, void*) -> aclnnStatus { return ACLNN_ERR_INNER; });
    cache.SetBlockDim(32);
    cache.opExecCacheDfx_->SetProfilingInfoId(ProfilingInfoId(0, 0, 42));
    bool submitted = false;
    cache.NewLaunchCache(&offset, &capacity, [](aclrtStream, void* data) -> aclnnStatus {
        bool* flag = nullptr;
        std::memcpy(&flag, data, sizeof(flag));
        *flag = true;
        return ACLNN_SUCCESS;
    }, op::DIRECT_INVOKE);
    bool* flag = &submitted;
    std::memcpy(cache.AddLaunchData(sizeof(flag)), &flag, sizeof(flag));
    EXPECT_EQ(cache.Run(nullptr, nullptr, {}), ACLNN_ERR_INNER);
    EXPECT_FALSE(submitted);
    EXPECT_EQ(DirectInvokeProfilerStub::kernelLaunchReportCount, 1U);
    EXPECT_EQ(DirectInvokeProfilerStub::basicInfoReportCount, 0U);
}

TEST(DirectInvokeTaskTest, CacheRecordCapacityFailureReleasesPreparedState)
{
    using namespace op;
    using namespace op::internal;
    int destroyed = 0;
    // Fail both the header allocation and the first address slot allocation.
    for (size_t capacity : {size_t{0}, sizeof(DirectInvokeReplay*)}) {
        OpExecCache cache;
        char buffer[64]{};
        cache.SetCacheBuf(buffer);
        cache.cacheCap_ = capacity;
        cache.hashKey_ = 1;
        auto state = std::shared_ptr<void>(new int(1), [&](void* p) { ++destroyed; delete static_cast<int*>(p); });
        auto replay = std::make_shared<DirectInvokeReplay>(state);
        const int64_t dims[] = {4}, strides[] = {1};
        auto* scratch = aclCreateTensor(dims, 1, ACL_UINT8, strides, 0, ACL_FORMAT_ND, dims, 1, nullptr);
        scratch->SetFromWorkspace(true);
        const aclTensor* tensors[] = {scratch};
        auto* workspace = aclCreateTensorList(tensors, 1);
        auto* args = GetOpArgContext(OP_INPUT(), OP_OUTPUT(), OP_WORKSPACE(workspace));
        ASSERT_TRUE(replay->Init(*args, cache));
        DestroyOpArgContext(args);
        aclDestroyTensorList(workspace);
        EXPECT_FALSE(DirectInvokeReplay::SaveToCache(cache, replay,
            [](void*, const OpArgContext*, aclrtStream) noexcept { return ACLNN_SUCCESS; }));
        EXPECT_FALSE(cache.IsOpCacheValid());
    }
    EXPECT_EQ(destroyed, 2);
}

// Exercise the public L2 lookup/executor lifecycle, rather than calling Run on
// a hand-built OpExecCache. Keep observation state alive through cache cleanup.
class DirectInvokeL2CacheTest : public testing::Test {
protected:
    void SetUp() override
    {
        savedContext_ = op::internal::GetThreadLocalContext();
        op::internal::GetOpCacheContext().SetOpCache(nullptr);
        op::internal::ReinitOpCacheManager();
        op::internal::UnInitPTACacheThreadLocal();
        op::internal::GetThreadLocalContext().opConfigInfo_ = op::OpConfigInfo{};
        MOCKER(op::internal::CheckCacheable).stubs().will(returnValue(true));
    }

    void TearDown() override
    {
        op::internal::GetOpCacheContext().SetOpCache(nullptr);
        op::internal::ReinitOpCacheManager();
        op::internal::GetThreadLocalContext() = std::move(savedContext_);
        GlobalMockObject::verify();
    }

    bool Lookup(const TensorFixture& tensors, aclOpExecutor** executor, uint64_t* bytes)
    {
        auto& tls = op::internal::GetThreadLocalContext();
        tls.logInfo_.l2ApiName = api_;
        tls.cacheHasFull_ = false;
        tensors.input->SetFromWorkspace(false);
        tensors.output->SetFromWorkspace(false);
        // Include every prepare input, just as an L2 API does for AI Core.
        return op::internal::GetFromCache(executor, bytes, api_,
            std::make_tuple(tensors.input.get(), static_cast<uint64_t>(config_.prepareStatus),
                static_cast<uint64_t>(config_.launchStatus), static_cast<uint64_t>(config_.workspaceSizes.size()),
                config_.workspaceSizes.empty() ? uint64_t{0} : config_.workspaceSizes[0],
                config_.workspaceSizes.size() < 2 ? uint64_t{0} : config_.workspaceSizes[1], config_.requestNullSizes),
            std::make_tuple(tensors.output.get()));
    }

    aclnnStatus Register(const TensorFixture& tensors, aclOpExecutor** executor, uint64_t* bytes)
    {
        auto owner = CREATE_EXECUTOR();
        auto* args = op::GetOpArgContext(OP_INPUT(tensors.input.get()), OP_OUTPUT(tensors.output.get()),
                                         FIXTURE_ATTR(config_), OP_WORKSPACE());
        auto status = op::AddDirectInvokeTask("DirectInvokeFixture", DirectInvokeFixtureOpTypeId(),
            PrepareFixture, LaunchFixture, DestroyFixture, owner.get(), args);
        if (status != ACLNN_SUCCESS) return status;
        *bytes = owner->GetWorkspaceSize();
        owner.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    static constexpr const char* api_ = "aclnnDirectInvokeCacheFixture";
    FixtureConfig config_;
    op::internal::OpThreadLocalContext savedContext_;
};

TEST_F(DirectInvokeL2CacheTest, HitAfterExecutorDestructionReusesStateAndCurrentAddresses)
{
    ObserveFixture observer(config_);
    config_.workspaceSizes = {1537U};
    aclOpExecutor* executor = nullptr;
    uint64_t bytes = 0;
    std::vector<uint8_t> firstWorkspace;
    void* preparedState = nullptr;
    {
        TensorFixture first;
        ASSERT_FALSE(Lookup(first, &executor, &bytes));
        ASSERT_EQ(Register(first, &executor, &bytes), ACLNN_SUCCESS);
        firstWorkspace.resize(bytes);
        ASSERT_EQ(CommonOpExecutorRun(firstWorkspace.data(), bytes, executor,
                                     reinterpret_cast<aclrtStream>(0x2468)), ACLNN_SUCCESS);
        preparedState = config_.observedState;
        EXPECT_EQ(config_.prepareCount, 1U);
        EXPECT_EQ(config_.destroyCount, 0U);
    } // Both the original executor and caller tensor descriptors are gone.

    TensorFixture second;
    const uint64_t expectedBytes = bytes;
    ASSERT_TRUE(Lookup(second, &executor, &bytes));
    ASSERT_NE(GetOpExecCacheFromExecutor(executor), nullptr);
    EXPECT_EQ(bytes, expectedBytes);
    std::vector<uint8_t> secondWorkspace(bytes);
    const auto stream = reinterpret_cast<aclrtStream>(0x3579);
    ASSERT_EQ(CommonOpExecutorRun(secondWorkspace.data(), bytes, executor, stream), ACLNN_SUCCESS);
    EXPECT_EQ(config_.prepareCount, 1U);
    EXPECT_EQ(config_.launchCount, 2U);
    EXPECT_EQ(config_.observedState, preparedState);
    EXPECT_EQ(config_.observedInputAddress, second.input->GetData());
    EXPECT_EQ(config_.observedOutputAddress, second.output->GetData());
    EXPECT_EQ(config_.observedStream, stream);
    const auto workspaceAddress = reinterpret_cast<uintptr_t>(config_.observedWorkspaceAddress);
    EXPECT_GE(workspaceAddress, reinterpret_cast<uintptr_t>(secondWorkspace.data()));
    EXPECT_LT(workspaceAddress, reinterpret_cast<uintptr_t>(secondWorkspace.data()) + bytes);
    EXPECT_EQ(config_.destroyCount, 0U);

    // A cached launch error is returned once, without prepare or a second launch.
    static_cast<FixtureState*>(preparedState)->launchStatus = ACLNN_ERR_INNER;
    ASSERT_TRUE(Lookup(second, &executor, &bytes));
    EXPECT_EQ(CommonOpExecutorRun(secondWorkspace.data(), bytes, executor, stream), ACLNN_ERR_INNER);
    EXPECT_EQ(config_.prepareCount, 1U);
    EXPECT_EQ(config_.launchCount, 3U);

    // A changed preparation input must miss the same L2 cache.
    config_.workspaceSizes[0]++;
    EXPECT_FALSE(Lookup(second, &executor, &bytes));
    op::internal::GetOpCacheContext().SetOpCache(nullptr);
    op::internal::ReinitOpCacheManager();
    EXPECT_EQ(config_.destroyCount, 1U);
}

TEST_F(DirectInvokeL2CacheTest, FailedFirstLaunchDoesNotPublishReplayOrRetainState)
{
    ObserveFixture observer(config_);
    config_.launchStatus = ACLNN_ERR_INNER;
    TensorFixture tensors;
    aclOpExecutor* executor = nullptr;
    uint64_t bytes = 0;
    ASSERT_FALSE(Lookup(tensors, &executor, &bytes));
    ASSERT_EQ(Register(tensors, &executor, &bytes), ACLNN_SUCCESS);
    std::vector<uint8_t> workspace(bytes);
    EXPECT_EQ(CommonOpExecutorRun(workspace.data(), bytes, executor, nullptr), ACLNN_ERR_INNER);
    EXPECT_EQ(config_.prepareCount, 1U);
    EXPECT_EQ(config_.launchCount, 1U);
    EXPECT_EQ(config_.destroyCount, 1U);
    EXPECT_FALSE(Lookup(tensors, &executor, &bytes));
}

TEST_F(DirectInvokeL2CacheTest, UnmappedStorageDisablesCacheWithoutFailingLaunch)
{
    ObserveFixture observer(config_);
    TensorFixture l2Tensors;
    aclOpExecutor* executor = nullptr;
    uint64_t bytes = 0;
    ASSERT_FALSE(Lookup(l2Tensors, &executor, &bytes));

    // Model L0 tensors whose storage has no relation to the L2 arguments.
    // Their addresses are valid for this invocation, but cannot be rebound.
    TensorFixture l0Tensors;
    l0Tensors.input->SetFromWorkspace(false);
    l0Tensors.output->SetFromWorkspace(false);
    ASSERT_EQ(Register(l0Tensors, &executor, &bytes), ACLNN_SUCCESS);
    std::vector<uint8_t> workspace(bytes);
    EXPECT_EQ(CommonOpExecutorRun(workspace.data(), bytes, executor, nullptr), ACLNN_SUCCESS);
    EXPECT_EQ(config_.prepareCount, 1U);
    EXPECT_EQ(config_.launchCount, 1U);
    EXPECT_EQ(config_.observedInputAddress, l0Tensors.input->GetData());
    EXPECT_EQ(config_.observedOutputAddress, l0Tensors.output->GetData());
    EXPECT_EQ(config_.destroyCount, 1U);
    EXPECT_FALSE(Lookup(l2Tensors, &executor, &bytes));
}
