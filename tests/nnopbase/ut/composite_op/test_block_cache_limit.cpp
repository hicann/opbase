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
#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

#include "block_pool.h"
#include "block_store.h"

using namespace op::internal;

namespace {
constexpr size_t kMaxCacheInstances = 100;
constexpr size_t kBlockSize = 256;
// BlockStore[1] (256B) has 16384 blocks. Allocate more to exhaust it.
constexpr size_t kBlockStore256Capacity = 16384;
constexpr size_t kAllocCount = 500;
constexpr size_t kTotalThreadsPhase1 = 50;
constexpr size_t kTotalThreadsPhase2 = 50;
constexpr size_t kStressThreadCount = 200;
constexpr size_t kStressIterCount = 1000;

size_t GetProcessRssKB()
{
    FILE *f = fopen("/proc/self/status", "r");
    if (f == nullptr) {
        return 0;
    }
    char line[256];
    size_t rssKB = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%zu", &rssKB);
            break;
        }
    }
    fclose(f);
    return rssKB;
}
}

class BlockCacheLimitTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// ─── Test 1: verify cacheDisabled_ flag is set correctly ───

TEST_F(BlockCacheLimitTest, CacheDisabledAfter100Threads)
{
    // Each thread allocates once to trigger BlockCache construction.
    // Threads 1~100: cache enabled (cacheDisabled_ = false)
    // Thread 101+: cache disabled (cacheDisabled_ = true)
    // Verify by checking cacheExt_ of the allocated block:
    //   cacheDisabled=false -> cacheExt_ = BlockCache* (not NOT_IN_CACHE)
    //   cacheDisabled=true  -> cacheExt_ = NOT_IN_CACHE (BlockPool::Malloc sets this)

    std::atomic<size_t> disabledCount{0};
    std::atomic<size_t> enabledCount{0};

    auto worker = [&]() {
        void *p = BlockCache::CacheAlloc(kBlockSize);
        if (p != nullptr) {
            BlockStore::BlockHeader *head = BlockStore::GetBlockHeader(p);
            if (head->cacheExt_ == BlockStore::NOT_IN_CACHE) {
                disabledCount++;
            } else {
                enabledCount++;
            }
            BlockCache::CacheFree(p);
        }
    };

    const size_t totalThreads = kMaxCacheInstances + 20;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < totalThreads; i++) {
        threads.emplace_back(worker);
    }
    for (auto &t : threads) {
        t.join();
    }

    printf("[Test1] enabled=%zu, disabled=%zu, total=%zu\n",
        enabledCount.load(), disabledCount.load(), totalThreads);
    EXPECT_GT(disabledCount.load(), 0U);
    EXPECT_EQ(enabledCount.load() + disabledCount.load(), totalThreads);
}

// ─── Test 2: memory stability after exceeding cache limit ───
//
// Key insight: VmRSS only reflects std::malloc'd memory (SYS_TAG blocks).
// BlockStore pre-allocated memory is already counted in RSS at startup,
// so orphaned BlockStore blocks don't change RSS.
// To make leak visible: exhaust BlockStore first → force SYS_TAG path.

TEST_F(BlockCacheLimitTest, MemoryStableAfterCacheLimit)
{
    // Step 1: Exhaust BlockStore[1] (256B, 16384 blocks)
    // Hold these blocks so BlockStore free list is empty.
    // After this, all 256B allocations through CacheAlloc go through std::malloc (SYS_TAG).
    std::vector<void *> exhaustBlocks;
    for (size_t i = 0; i < kBlockStore256Capacity + 512; i++) {
        void *p = BlockPool::Malloc(kBlockSize);
        if (p == nullptr) {
            break;
        }
        exhaustBlocks.push_back(p);
    }
    printf("[Test2] Exhausted BlockStore: %zu blocks held (%zu KB)\n",
        exhaustBlocks.size(), exhaustBlocks.size() * (kBlockSize + sizeof(BlockStore::BlockHeader)) / 1024);

    size_t rssBefore = GetProcessRssKB();
    printf("[Test2] RSS before phases: %zu KB\n", rssBefore);

    // Step 2: Phase 1 - threads with cache enabled (first 50 threads)
    // Each thread: alloc many blocks → free all to cache → thread exits → blocks orphaned
    // Since BlockStore is exhausted, blocks are SYS_TAG (std::malloc).
    // Leak = real memory → RSS grows.
    for (size_t i = 0; i < kTotalThreadsPhase1; i++) {
        std::thread t([&]() {
            // Hold all blocks first, then free to cache
            std::vector<void *> blocks;
            for (size_t j = 0; j < kAllocCount; j++) {
                void *p = BlockCache::CacheAlloc(kBlockSize);
                if (p != nullptr) {
                    blocks.push_back(p);
                }
            }
            // Free all → goes to cacheHead_ (not BlockPool)
            for (void *p : blocks) {
                BlockCache::CacheFree(p);
            }
            // Thread exits → ~BlockCache()=default → cacheHead_ blocks orphaned → LEAK
        });
        t.join();
    }

    size_t rssAfterPhase1 = GetProcessRssKB();
    size_t phase1Growth = (rssAfterPhase1 > rssBefore) ? (rssAfterPhase1 - rssBefore) : 0;
    printf("[Test2] Phase1 (%zu threads, cache enabled): RSS %zu -> %zu KB, growth=%zu KB\n",
        kTotalThreadsPhase1, rssBefore, rssAfterPhase1, phase1Growth);

    // Step 3: Phase 2 - threads with cache disabled (thread 101+)
    // cacheDisabled_=true → CacheAlloc → BlockPool::Malloc (SYS_TAG from std::malloc)
    // CacheFree → BlockPool::Free → std::free immediately. No caching. No leak.
    std::vector<size_t> rssSnapshots;
    for (size_t i = 0; i < kTotalThreadsPhase2; i++) {
        std::thread t([&]() {
            std::vector<void *> blocks;
            for (size_t j = 0; j < kAllocCount; j++) {
                void *p = BlockCache::CacheAlloc(kBlockSize);
                if (p != nullptr) {
                    blocks.push_back(p);
                }
            }
            for (void *p : blocks) {
                BlockCache::CacheFree(p);
            }
        });
        t.join();

        if ((i + 1) % 10 == 0) {
            size_t rss = GetProcessRssKB();
            rssSnapshots.push_back(rss);
            printf("[Test2] Phase2 thread %zu, RSS: %zu KB\n", i + 1, rss);
        }
    }

    size_t rssAfterPhase2 = GetProcessRssKB();
    size_t phase2Growth = (rssAfterPhase2 > rssAfterPhase1) ? (rssAfterPhase2 - rssAfterPhase1) : 0;
    printf("[Test2] Phase2 (%zu threads, cache disabled): RSS %zu -> %zu KB, growth=%zu KB\n",
        kTotalThreadsPhase2, rssAfterPhase1, rssAfterPhase2, phase2Growth);

    // Step 4: Clean up exhaust blocks
    for (void *p : exhaustBlocks) {
        BlockPool::Free(p);
    }

    // Verify: Phase 1 should show RSS growth (SYS_TAG blocks leaked)
    // Phase 2 should show minimal growth (blocks freed immediately, no caching)
    EXPECT_GT(phase1Growth, 0U);
    EXPECT_LT(phase2Growth, phase1Growth);
}

// ─── Test 3: cross-thread free with mixed enabled/disabled threads ───

TEST_F(BlockCacheLimitTest, CrossThreadFreeWithMixedCacheState)
{
    // Fill up 100 thread slots to ensure subsequent threads are disabled
    for (size_t i = 0; i < kMaxCacheInstances; i++) {
        std::thread t([&]() {
            void *p = BlockCache::CacheAlloc(kBlockSize);
            if (p != nullptr) {
                BlockCache::CacheFree(p);
            }
        });
        t.join();
    }

    // Producer allocates blocks and stores them
    std::vector<void *> blocks;
    std::mutex mtx;
    std::thread producer([&]() {
        for (size_t i = 0; i < 200; i++) {
            void *p = BlockCache::CacheAlloc(kBlockSize);
            if (p != nullptr) {
                std::lock_guard<std::mutex> lock(mtx);
                blocks.push_back(p);
            }
        }
    });
    producer.join();

    // Consumer (cacheDisabled thread) frees all blocks
    std::thread consumer([&]() {
        std::vector<void *> localBlocks;
        {
            std::lock_guard<std::mutex> lock(mtx);
            localBlocks = std::move(blocks);
        }
        for (void *p : localBlocks) {
            BlockCache::CacheFree(p);
        }
    });
    consumer.join();

    SUCCEED();
}

// ─── Test 4: high concurrency stress test ───

TEST_F(BlockCacheLimitTest, StressHighConcurrency)
{
    std::atomic<size_t> successCount{0};
    std::atomic<size_t> failCount{0};

    auto worker = [&]() {
        for (size_t i = 0; i < kStressIterCount; i++) {
            void *p = BlockCache::CacheAlloc(kBlockSize);
            if (p != nullptr) {
                BlockCache::CacheFree(p);
                successCount++;
            } else {
                failCount++;
            }
        }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < kStressThreadCount; i++) {
        threads.emplace_back(worker);
    }
    for (auto &t : threads) {
        t.join();
    }

    printf("[Test4] stress test: success=%zu, fail=%zu\n",
        successCount.load(), failCount.load());
    EXPECT_EQ(successCount.load(), kStressThreadCount * kStressIterCount);
    EXPECT_EQ(failCount.load(), 0U);
}
