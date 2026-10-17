/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sync.h
 * \brief
 */
#ifndef UTIL_SYNC_H_
#define UTIL_SYNC_H_

#include "atvoss/util/kernel_vec_intf_apt.h"
#include "atvoss/util/node.h"
namespace Ops {
namespace Base {
using namespace AscendC;
/**
 * 调用get_buf函数
 * @tparam p TPosition的类型
 * @param bufID 当前处理的buf的真是ID
 */
template <TPosition p>
__aicore__ inline void GetBuf(uint8_t bufID)
{
    if constexpr (p == TPosition::VECIN) {
        Mutex::Lock<PIPE_MTE2>(bufID);
    } else if constexpr (p == TPosition::VECCALC) {
        Mutex::Lock<PIPE_V>(bufID);
    } else if constexpr (p == TPosition::VECOUT) {
        Mutex::Lock<PIPE_MTE3>(bufID);
    }
}

/**
 * 调用rls_buf函数
 * @tparam p TPosition的类型
 * @param bufID 当前处理的buf的真实ID
 */
template <TPosition p>
__aicore__ inline void RlsBuf(uint8_t bufID)
{
    if constexpr (p == TPosition::VECIN) {
        Mutex::Unlock<PIPE_MTE2>(bufID);
    } else if constexpr (p == TPosition::VECCALC) {
        Mutex::Unlock<PIPE_V>(bufID);
    } else if constexpr (p == TPosition::VECOUT) {
        Mutex::Unlock<PIPE_MTE3>(bufID);
    }
}

/**
 * 插入GetBuf同步函数
 * @tparam p TPosition的类型
 * @param bufID 当前处理的buf的真实ID
 */
template <TPosition p>
__aicore__ void inline GetTensor(uint8_t bufID)
{
    RUN_LOG("GetTensor: ID: %d, Position = %d", bufID, p);
    GetBuf<p>(bufID);
}

/**
 * 插入RlsBuf同步函数
 * @tparam p TPosition的类型
 * @param bufID 当前处理的buf的真实ID
 */
template <TPosition p>
__aicore__ void inline ReleaseTensor(uint8_t bufID)
{
    RUN_LOG("ReleaseTensor: ID: %d, Position = %d", bufID, p);
    RlsBuf<p>(bufID);
}

/* *
 * Per-buffer flag 同步的 eventId 编码（方案：通道内槽位索引，非原始 buffer ID）：
 *   MTE2 通道 （V_MTE2 / MTE2_V / MTE3_MTE2）：eventID = CopyIn 槽位 k，ping 为 k，pong 为 k + MteCount。
 *   MTE3 通道 （V_MTE3 / MTE3_V）：eventID = MTE3 生产者槽位 j，ping 为 j，pong 为 j + MteCount。
 * 保证任意 eventID < 8（要求 Mte2Count <= 4、Mte3Count <= 4），避免原始 buffer ID 越界/冲突。
*/
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
template <int Mte2Count, int Mte3Count>
__aicore__ inline void SetLoopSyncFlag()
{
    static_assert(Mte2Count <= 4, "Mte2Count exceeds 8-eventID limit (ping+pong)");
    static_assert(Mte3Count <= 4, "Mte3Count exceeds 8-eventID limit (ping+pong)");
    for (int evt = 0; evt < Mte2Count * 2; evt++) {
        SetFlag<HardEvent::V_MTE2>(evt);
        SetFlag<HardEvent::MTE3_MTE2>(evt);
    }
    for (int evt = 0; evt < Mte3Count * 2; evt++) {
        SetFlag<HardEvent::MTE3_V>(evt);
    }
}

template <int Mte2Count, int Mte3Count>
__aicore__ inline void WaitLoopSyncFlag()
{
    for (int evt = 0; evt < Mte2Count * 2; evt++) {
        WaitFlag<HardEvent::V_MTE2>(evt);
        WaitFlag<HardEvent::MTE3_MTE2>(evt);
    }
    for (int evt = 0; evt < Mte3Count * 2; evt++) {
        WaitFlag<HardEvent::MTE3_V>(evt);
    }
}

template <int Mte2Count>
__aicore__ inline void SetMte3Mte2Flags(int32_t pingPong)
{
    const int base = (pingPong == 0) ? 0 : Mte2Count;
    for (int evt = 0; evt < Mte2Count; evt++) {
        SetFlag<HardEvent::MTE3_MTE2>(base + evt);
    }
}

/**
 * Per-buffer flag 同步公共基类
 * 封装MTE2/MTE3事件槽位解析与V计算前后的批量flag同步，供broadcast/elewise各调度模板继承复用。
 * @tparam DAG DAG图类型
 * @tparam Policy MTE2源槽位解析策略
 */
template <typename DAG, Mte2SrcPolicy Policy>
class PerBufferFlagSync {
protected:
    constexpr static int Mte2Count = GetMte2ProducerCount<DAG, Policy>();
    constexpr static int Mte3Count = GetMte3ProducerCount<DAG, Policy>();

    /**
     * 获取pos节点的MTE2事件ID
     * @tparam pos CopyIn节点在FunList内的索引
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     * @return 事件ID
     */
    template <int pos>
    __aicore__ constexpr static inline int Mte2Evt(int32_t pingPong)
    {
        static_assert(IsMte2Producer<DAG, Policy, pos>(), "pos is not an MTE2 producer");
        return GetMte2Slot<DAG, Policy, pos>() + (pingPong == 0 ? 0 : Mte2Count);
    }

    /**
     * 获取pos节点的MTE3事件ID
     * @tparam pos CopyOut节点在FunList内的索引
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     * @return 事件ID
     */
    template <int pos>
    __aicore__ constexpr static inline int Mte3Evt(int32_t pingPong)
    {
        static_assert(IsMte3Producer<DAG, Policy, pos>(), "pos is not an MTE3 producer");
        return GetMte3Slot<DAG, Policy, pos>() + (pingPong == 0 ? 0 : Mte3Count);
    }

    /**
     * 等待Op所有输入的MTE2_V事件
     * @tparam Op 当前处理的Op类型
     * @tparam pos Op在FunList内的索引
     * @tparam argIdx 当前处理的输入参数索引
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     */
    template <typename Op, int pos, int argIdx>
    __aicore__ inline void WaitMte2VFlags(int32_t pingPong)
    {
        if constexpr (argIdx < Op::InputSize) {
            using InputOp = typename Op::InArgs::template At<argIdx>;
            constexpr int mte2SrcPos = ResolveMte2SrcPos<DAG, InputOp, Policy, pos>();
            if constexpr (mte2SrcPos >= 0) {
                WaitFlag<HardEvent::MTE2_V>(Mte2Evt<mte2SrcPos>(pingPong));
            }
            WaitMte2VFlags<Op, pos, argIdx + 1>(pingPong);
        }
    }

    /**
     * 为Op所有输入设置V_MTE2事件
     * @tparam Op 当前处理的Op类型
     * @tparam pos Op在FunList内的索引
     * @tparam argIdx 当前处理的输入参数索引
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     */
    template <typename Op, int pos, int argIdx>
    __aicore__ inline void SetVMte2Flags(int32_t pingPong)
    {
        if constexpr (argIdx < Op::InputSize) {
            using InputOp = typename Op::InArgs::template At<argIdx>;
            constexpr int mte2SrcPos = ResolveMte2SrcPos<DAG, InputOp, Policy, pos>();
            if constexpr (mte2SrcPos >= 0) {
                SetFlag<HardEvent::V_MTE2>(Mte2Evt<mte2SrcPos>(pingPong));
            }
            SetVMte2Flags<Op, pos, argIdx + 1>(pingPong);
        }
    }
};
#endif

} // namespace Base
} // namespace Ops
#endif // UTIL_SYNC_H_