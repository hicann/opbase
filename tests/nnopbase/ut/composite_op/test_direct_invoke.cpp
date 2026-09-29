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
        EXPECT_FALSE(executor->GetOpExecCache()->IsOpCacheValid());
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
        // Callback failures on subsequent executions must not be hidden by replay.
        static_cast<FixtureState*>(state)->launchStatus = ACLNN_ERR_INNER;
        EXPECT_EQ(CommonOpExecutorRun(secondWorkspace.data(), bytes, executor.get(), secondStream), ACLNN_ERR_INNER);
        EXPECT_EQ(config.prepareCount, 1U);
        EXPECT_EQ(config.launchCount, 3U);
        EXPECT_EQ(config.destroyCount, 0U);
    }
    // The repeat executor's destructor deleted its opExecCache while
    // CommonOpExecutorRun left it in the thread-local op cache context
    // (SetOpCache is never reset). Clear the dangling pointer so later tests
    // that launch directly cannot dereference it. See ~OpExecutorImpl.
    op::internal::GetOpCacheContext().SetOpCache(nullptr);
    EXPECT_EQ(config.destroyCount, 1U);
}
