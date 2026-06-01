/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You should have received a copy of the License along with this program.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "aclnn/acl_meta.h"
#include "bridge_pool.h"
#include "thread_local_context.h"

using namespace op;

namespace op {
namespace internal {
void* Allocate(size_t size);
void DeAllocate(void* addr);
} // namespace internal
} // namespace op

extern "C" int InitHugeMemThreadLocal(void* arg, bool sync);
extern "C" void UnInitHugeMemThreadLocal(void* arg, bool sync);
extern "C" void ReleaseHugeMem(void* arg, bool sync);

static aclTensor* CreateSimpleTensor()
{
    std::vector<int64_t> shape = {2, 3, 4};
    std::vector<int64_t> strides = {12, 4, 1};
    return aclCreateTensor(
        shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
}

class BlockCacheMultiThreadUt : public testing::Test
{
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

TEST_F(BlockCacheMultiThreadUt, SingleThreadBatchCreateDestroy)
{
    const int count = 1024;
    std::vector<aclTensor*> tensors;
    tensors.reserve(count);
    for (int i = 0; i < count; i++) {
        aclTensor* t = CreateSimpleTensor();
        ASSERT_NE(t, nullptr);
        tensors.push_back(t);
    }
    for (int i = 0; i < count; i++) {
        EXPECT_EQ(aclDestroyTensor(tensors[i]), OK);
    }
}

TEST_F(BlockCacheMultiThreadUt, SingleThreadRepeatedAllocFree)
{
    for (int round = 0; round < 100; round++) {
        std::vector<aclTensor*> tensors;
        for (int i = 0; i < 64; i++) {
            aclTensor* t = CreateSimpleTensor();
            ASSERT_NE(t, nullptr);
            tensors.push_back(t);
        }
        for (auto* t : tensors) {
            EXPECT_EQ(aclDestroyTensor(t), OK);
        }
    }
}

TEST_F(BlockCacheMultiThreadUt, SingleThreadRandomAllocFree)
{
    std::vector<aclTensor*> pool;
    std::mt19937 gen(42);
    for (int round = 0; round < 2000; round++) {
        if (pool.empty() || (gen() % 2 == 0)) {
            aclTensor* t = CreateSimpleTensor();
            ASSERT_NE(t, nullptr);
            pool.push_back(t);
        } else {
            size_t idx = gen() % pool.size();
            EXPECT_EQ(aclDestroyTensor(pool[idx]), OK);
            pool[idx] = pool.back();
            pool.pop_back();
        }
    }
    for (auto* t : pool) {
        EXPECT_EQ(aclDestroyTensor(t), OK);
    }
}

TEST_F(BlockCacheMultiThreadUt, MultiThreadIndependentAllocFree)
{
    auto worker = []() {
        for (int round = 0; round < 50; round++) {
            std::vector<aclTensor*> tensors;
            for (int i = 0; i < 32; i++) {
                aclTensor* t = CreateSimpleTensor();
                if (t == nullptr) {
                    break;
                }
                tensors.push_back(t);
            }
            for (auto* t : tensors) {
                aclDestroyTensor(t);
            }
        }
    };

    const int threadCount = 16;
    std::vector<std::thread> threads;
    for (int i = 0; i < threadCount; i++) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }
}

TEST_F(BlockCacheMultiThreadUt, MultiThreadFrequentCreateDestroy)
{
    const int threadCount = 20;
    const int roundsPerThread = 100;
    auto worker = []() {
        for (int round = 0; round < roundsPerThread; round++) {
            aclTensor* t = CreateSimpleTensor();
            if (t != nullptr) {
                aclDestroyTensor(t);
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < threadCount; i++) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }
}

TEST_F(BlockCacheMultiThreadUt, MultiThreadCrossThreadAllocFree)
{
    const int tensorCount = 256;
    std::vector<aclTensor*> tensors(tensorCount, nullptr);
    std::atomic<int> allocIdx{0};
    std::atomic<int> freeIdx{0};

    auto allocator = [&]() {
        while (true) {
            int idx = allocIdx.fetch_add(1);
            if (idx >= tensorCount) {
                break;
            }
            tensors[idx] = CreateSimpleTensor();
        }
    };

    auto freer = [&]() {
        while (true) {
            int idx = freeIdx.load(std::memory_order_relaxed);
            if (idx >= tensorCount) {
                break;
            }
            if (tensors[idx] == nullptr) {
                std::this_thread::yield();
                continue;
            }
            aclTensor* t = tensors[idx];
            tensors[idx] = nullptr;
            aclDestroyTensor(t);
            freeIdx.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; i++) {
        threads.emplace_back(allocator);
    }
    for (int i = 0; i < 4; i++) {
        threads.emplace_back(freer);
    }
    for (auto& t : threads) {
        t.join();
    }
}

TEST_F(BlockCacheMultiThreadUt, MultiThreadCrossThreadShuffle)
{
    const int poolSize = 128;
    std::vector<aclTensor*> pool(poolSize, nullptr);
    std::mutex poolMutex;

    for (int i = 0; i < poolSize; i++) {
        pool[i] = CreateSimpleTensor();
        ASSERT_NE(pool[i], nullptr);
    }

    auto worker = [&]() {
        std::mt19937 gen(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        for (int round = 0; round < 200; round++) {
            int idx = gen() % poolSize;
            aclTensor* old = nullptr;
            {
                std::lock_guard<std::mutex> lock(poolMutex);
                old = pool[idx];
                pool[idx] = nullptr;
            }
            if (old != nullptr) {
                aclDestroyTensor(old);
            }
            aclTensor* fresh = CreateSimpleTensor();
            {
                std::lock_guard<std::mutex> lock(poolMutex);
                pool[idx] = fresh;
            }
        }
    };

    const int threadCount = 10;
    std::vector<std::thread> threads;
    for (int i = 0; i < threadCount; i++) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }

    for (auto* t : pool) {
        if (t != nullptr) {
            EXPECT_EQ(aclDestroyTensor(t), OK);
        }
    }
}

TEST_F(BlockCacheMultiThreadUt, ThreadLifecycleBlockCacheDrain)
{
    const int cycleCount = 50;
    const int tensorsPerCycle = 64;
    for (int cycle = 0; cycle < cycleCount; cycle++) {
        std::vector<aclTensor*> tensors;
        tensors.reserve(tensorsPerCycle);
        std::thread([&tensors]() {
            for (int i = 0; i < tensorsPerCycle; i++) {
                aclTensor* t = CreateSimpleTensor();
                if (t != nullptr) {
                    tensors.push_back(t);
                }
            }
        }).join();
        for (auto* t : tensors) {
            EXPECT_EQ(aclDestroyTensor(t), OK);
        }
    }
}

TEST_F(BlockCacheMultiThreadUt, ThreadLifecycleWithHugeMem)
{
    const int cycleCount = 30;
    const int tensorsPerCycle = 48;
    for (int cycle = 0; cycle < cycleCount; cycle++) {
        std::vector<aclTensor*> tensors;
        tensors.reserve(tensorsPerCycle);
        std::thread([&tensors]() {
            InitHugeMemThreadLocal(nullptr, false);
            for (int i = 0; i < tensorsPerCycle; i++) {
                aclTensor* t = CreateSimpleTensor();
                if (t != nullptr) {
                    tensors.push_back(t);
                }
            }
            for (auto* t : tensors) {
                aclDestroyTensor(t);
            }
            ReleaseHugeMem(nullptr, false);
            UnInitHugeMemThreadLocal(nullptr, false);
        }).join();
    }
}

TEST_F(BlockCacheMultiThreadUt, MultiThreadBarrierAllocFree)
{
    const int threadCount = 10;
    const int batchSize = 32;
    const int barrierRounds = 20;

    std::mutex mtx;
    std::condition_variable cv;
    int count = 0;
    int generation = 0;

    auto barrier = [&]() {
        std::unique_lock<std::mutex> lock(mtx);
        int gen = generation;
        count++;
        if (count == threadCount) {
            count = 0;
            generation++;
            cv.notify_all();
        } else {
            cv.wait(lock, [&]() { return gen != generation; });
        }
    };

    auto worker = [&](int threadId) {
        for (int r = 0; r < barrierRounds; r++) {
            std::vector<aclTensor*> localTensors;
            for (int i = 0; i < batchSize; i++) {
                aclTensor* t = CreateSimpleTensor();
                if (t != nullptr) {
                    localTensors.push_back(t);
                }
            }

            barrier();

            for (auto* t : localTensors) {
                aclDestroyTensor(t);
            }

            barrier();
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < threadCount; i++) {
        threads.emplace_back(worker, i);
    }
    for (auto& t : threads) {
        t.join();
    }
}

TEST_F(BlockCacheMultiThreadUt, MixedSizeAllocFree)
{
    auto createTensorBySize = [](int sizeClass) -> aclTensor* {
        switch (sizeClass) {
            case 0: {
                std::vector<int64_t> shape = {1};
                return aclCreateTensor(
                    shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                    nullptr);
            }
            case 1: {
                std::vector<int64_t> shape = {16, 16};
                return aclCreateTensor(
                    shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                    nullptr);
            }
            case 2: {
                std::vector<int64_t> shape = {64, 64};
                return aclCreateTensor(
                    shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                    nullptr);
            }
            case 3: {
                std::vector<int64_t> shape = {256, 256};
                return aclCreateTensor(
                    shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                    nullptr);
            }
            default:
                return nullptr;
        }
    };

    const int threadCount = 8;
    const int opsPerThread = 500;
    auto worker = [&]() {
        std::mt19937 gen(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::vector<aclTensor*> local;
        for (int i = 0; i < opsPerThread; i++) {
            int action = gen() % 3;
            if (action == 0 || local.empty()) {
                int sizeClass = gen() % 4;
                aclTensor* t = createTensorBySize(sizeClass);
                if (t != nullptr) {
                    local.push_back(t);
                }
            } else if (action == 1 && !local.empty()) {
                size_t idx = gen() % local.size();
                aclDestroyTensor(local[idx]);
                local[idx] = local.back();
                local.pop_back();
            } else {
                if (!local.empty()) {
                    size_t idx = gen() % local.size();
                    aclDestroyTensor(local[idx]);
                    local[idx] = CreateSimpleTensor();
                }
            }
        }
        for (auto* t : local) {
            aclDestroyTensor(t);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < threadCount; i++) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }
}

TEST_F(BlockCacheMultiThreadUt, CrossThreadProducerConsumer)
{
    const int producerCount = 4;
    const int consumerCount = 4;
    const int totalTensors = 400;

    std::vector<aclTensor*> queue;
    std::mutex queueMutex;
    std::condition_variable cvProduced;
    std::condition_variable cvConsumed;
    std::atomic<int> producedCount{0};
    std::atomic<int> consumedCount{0};
    bool done = false;

    auto producer = [&]() {
        for (int i = 0; i < totalTensors / producerCount; i++) {
            aclTensor* t = CreateSimpleTensor();
            if (t == nullptr) {
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                queue.push_back(t);
            }
            producedCount.fetch_add(1);
            cvProduced.notify_one();
        }
    };

    auto consumer = [&]() {
        while (true) {
            aclTensor* t = nullptr;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                cvProduced.wait(lock, [&]() { return !queue.empty() || done; });
                if (!queue.empty()) {
                    t = queue.back();
                    queue.pop_back();
                } else if (done) {
                    break;
                }
            }
            if (t != nullptr) {
                aclDestroyTensor(t);
                consumedCount.fetch_add(1);
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < producerCount; i++) {
        threads.emplace_back(producer);
    }
    for (int i = 0; i < consumerCount; i++) {
        threads.emplace_back(consumer);
    }

    for (int i = 0; i < producerCount; i++) {
        threads[i].join();
    }
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        done = true;
    }
    cvProduced.notify_all();
    for (int i = producerCount; i < producerCount + consumerCount; i++) {
        threads[i].join();
    }

    for (auto* t : queue) {
        aclDestroyTensor(t);
    }
}

TEST_F(BlockCacheMultiThreadUt, RapidThreadCreateDestroyStress)
{
    const int totalThreads = 100;
    const int tensorsPerThread = 20;
    std::atomic<int> successCount{0};

    auto worker = [&]() {
        std::vector<aclTensor*> tensors;
        for (int i = 0; i < tensorsPerThread; i++) {
            aclTensor* t = CreateSimpleTensor();
            if (t != nullptr) {
                tensors.push_back(t);
                successCount.fetch_add(1);
            }
        }
        for (auto* t : tensors) {
            aclDestroyTensor(t);
        }
    };

    for (int i = 0; i < totalThreads; i++) {
        std::thread(worker).join();
    }

    EXPECT_EQ(successCount.load(), totalThreads * tensorsPerThread);
}

TEST_F(BlockCacheMultiThreadUt, MultiThreadHugeMemFallbackToCache)
{
    const int threadCount = 4;
    const int tensorsPerThread = 200;

    auto worker = [&]() {
        InitHugeMemThreadLocal(nullptr, false);
        std::vector<aclTensor*> tensors;
        for (int i = 0; i < tensorsPerThread; i++) {
            aclTensor* t = CreateSimpleTensor();
            if (t != nullptr) {
                tensors.push_back(t);
            }
        }
        for (auto* t : tensors) {
            aclDestroyTensor(t);
        }
        ReleaseHugeMem(nullptr, false);
        UnInitHugeMemThreadLocal(nullptr, false);
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < threadCount; i++) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }
}
