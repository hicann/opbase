/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_UBBRC_SCH_H_
#define BROADCAST_UBBRC_SCH_H_

#include "broadcast_sch_base.h"

#pragma "lib"
namespace Ops {
namespace Base {
using Ops::Base::BroadcastTiling;

template <class BrcDag, int64_t R>
class BroadcastUbSch : public BroadcastBaseSch<BrcDag, false, R> {
public:
    __aicore__ inline explicit BroadcastUbSch(const BroadcastBaseTilingData<BrcDag>* baseTilingData)
        : BroadcastBaseSch<BrcDag, false, R>(baseTilingData), tilingData(baseTilingData)
    {}
    /**
     * 初始化BroadcastUbSch对象
     * @tparam Args 输入Args类型
     * @param pipe 流水线指针
     * @param args 输入args参数列表，需要匹配DAG图中PlaceHolder的顺序[In0, In1..., Out0, Out1...]
     */
    template <class... Args>
    __aicore__ inline void Init(TPipe* pipe, Args... args)
    {
        static_assert(BrcDag::InputSize + BrcDag::OutputSize == sizeof...(Args),
                      "BroadcastUbSch.Init args num should match DAG holders.");
        InitInputArgs<0>(args...); // 调用入参分析,input, output
        this->template SetScalar<0>(0);
        TBuf<TPosition::VECCALC> buf;
        this->blockEleNum_ = tilingData->elemNum;
        this->blockLen_ = this->blockEleNum_ * BrcDag::MaxDtypeBytes;
        pipe->InitBuffer(buf, this->blockLen_ * bufferNum);
        this->tensorPool_ = buf.Get<uint8_t>();
        // 根据原始shape获取
        if constexpr (BrcDag::VecBrcSize > 0 || BrcDag::CopyBrcSize > 0) {
            int64_t dstUblength[2] = {1};
            this->template GetUbBroadcastShapeInfo(tilingData->outputDims, dstUblength, this->dstUbFormerShape_);
            this->runningRank_ = tilingData->shapeLen - tilingData->ubSplitAxis;
            if (tilingData->outputDims[tilingData->ubSplitAxis] != 1) {
                this->ubFormer_ = tilingData->ubFormer;
                this->ubTail_ = tilingData->ubTail;
            } else {
                this->ubFormer_ = 1;
                this->ubTail_ = 1;
            }
        }
        if constexpr (BrcDag::VecBrcSize > 0) {
            this->template GetVecBrcInfo<0>();
        }
        if constexpr (BrcDag::CopyBrcSize > 0) {
            GetCopyInBrcInfo<0>();
        }
    }

    /**
     * 执行BroadcastUbSch
     */
    __aicore__ inline void Process()
    {
        ubLoopNum_ = AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1 ? tilingData->blockTail :
                                                                            tilingData->blockFormer;
        int64_t axesIndices[BROADCAST_MAX_DIMS] = {0};
        BroadcastGetAxesIndices(axesIndices, tilingData->blockFormer * AscendC::GetBlockIdx(), tilingData->outputDims,
                                tilingData->ubSplitAxis, tilingData->dimProductBeforeUbInner);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
        SetLoopSyncFlag<BroadcastBaseSch<BrcDag, false, R>::Mte2Count, BroadcastBaseSch<BrcDag, false, R>::Mte3Count>();
#endif
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum_; ubLoopIdx++) {
            this->vBrcIndex_ = 0;
            cBrcIndex_ = 0;
            if (ubLoopIdx != 0) {
                BroadcastUpdateAxesIndices(axesIndices, tilingData->outputDims, tilingData->ubSplitAxis,
                                           tilingData->ubOuter);
            }

            int64_t ubSplitSize = axesIndices[tilingData->ubSplitAxis] == tilingData->ubOuter - 1 ?
                                      tilingData->ubTail :
                                      tilingData->ubFormer;
            // ub整循环处理的元素个数
            Run<0, false>(ubSplitSize, axesIndices, ubLoopIdx, ubLoopIdx & 1);
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
        WaitLoopSyncFlag<BroadcastBaseSch<BrcDag, false, R>::Mte2Count, BroadcastBaseSch<BrcDag, false, R>::Mte3Count>();
#endif
    }

protected:
    // 初始化输出
    template <int start, class... Args>
    __aicore__ inline void InitOutputArgs(GM_ADDR y, Args... args)
    {
        this->outGm_[start] = y;
        if constexpr (start + 1 < BrcDag::OutputSize) {
            InitOutputArgs<start + 1>(args...);
        }
    }

    // 初始化输入
    template <int start, class... Args>
    __aicore__ inline void InitInputArgs(GM_ADDR x, Args... args)
    {
        this->inGm_[start] = x;
        if constexpr (start + 1 < BrcDag::InputSize) {
            InitInputArgs<start + 1>(args...);
        } else {
            InitOutputArgs<0>(args...);
        }
    }

    // 获取CopyInBrc信息
    template <int index = 0>
    __aicore__ inline void GetCopyInBrcInfo()
    {
        using copyInBrcOp = typename BrcDag::CopyBrcNodes::template At<index>;
        using inputType = typename copyInBrcOp::template FunInArgType<0>;
        this->template GetUbBroadcastShapeInfo(tilingData->inputBrcDims[index], copyInBrcInputLength[index],
                                               copyInBrcFormerShape[index]);
        if (tilingData->inputBrcDims[index][tilingData->ubSplitAxis] != 1) {
            cBrcFormer[index] = tilingData->ubFormer;
            cBrcTail[index] = tilingData->ubTail;
        } else {
            cBrcFormer[index] = 1;
            cBrcTail[index] = 1;
        }
        if constexpr (index + 1 < BrcDag::CopyBrcSize) {
            GetCopyInBrcInfo<index + 1>();
        }
    }

    /**
     * CopyInBrc调用的Broadcast函数
     * @tparam T 数据类型
     * @param dst 目的tensor
     * @param src 源tensor
     * @param realIdx 当前处理的索引
     */
    template <typename T>
    __aicore__ inline void CvecBrc(LocalTensor<T>& dst, LocalTensor<T>& src, int64_t realIdx)
    {
        BroadcastTiling runningTiling;
        if (realIdx == tilingData->ubOuter - 1) {
            this->dstUbFormerShape_[0] = this->ubTail_;
            copyInBrcFormerShape[cBrcIndex_][0] = cBrcTail[cBrcIndex_];
        } else {
            this->dstUbFormerShape_[0] = this->ubFormer_;
            copyInBrcFormerShape[cBrcIndex_][0] = cBrcFormer[cBrcIndex_];
        }
        if constexpr (R == -1) {
            GetBroadcastTilingInfo<T>(this->runningRank_, this->dstUbFormerShape_, copyInBrcFormerShape[cBrcIndex_],
                                      false, runningTiling);
            Broadcast(dst, src, this->dstUbFormerShape_, copyInBrcFormerShape[cBrcIndex_], &runningTiling);
        } else {
            GetBroadcastTilingInfo<T, R>(R, this->dstUbFormerShape_, copyInBrcFormerShape[cBrcIndex_], false,
                                         runningTiling);
            Broadcast<T, R>(dst, src, this->dstUbFormerShape_, copyInBrcFormerShape[cBrcIndex_], &runningTiling);
        }
    }

    /**
     * 跳过搬入广播输入时设置同步标志
     * @tparam pos CopyInBrc在FunList内的索引
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     */
    template <int pos>
    __aicore__ inline void SetSkipBrcFlag(int32_t pingPong)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
        WaitFlag<HardEvent::V_MTE2>(this->template Mte2Evt<pos>(pingPong));
        WaitFlag<HardEvent::MTE3_MTE2>(this->template Mte2Evt<pos>(pingPong));
        SetFlag<HardEvent::V_MTE2>(this->template Mte2Evt<pos>(pingPong));
        if constexpr (IsMte3Producer<BrcDag, Mte2SrcPolicy::Strict, pos>()) {
            WaitFlag<HardEvent::MTE3_V>(this->template Mte3Evt<pos>(pingPong));
            SetFlag<HardEvent::V_MTE3>(this->template Mte3Evt<pos>(pingPong));
        }
#endif
    }

    /**
     * 搬入广播输入数据
     * @tparam inputType 输入数据类型
     * @tparam pos CopyInBrc在FunList内的索引
     * @tparam BufList 缓冲区ID列表类型
     * @param inTensor 输入的LocalTensor对象
     * @param globalTensor 输入的GlobalTensor对象
     * @param inputLength 搬入的数据长度
     * @param intList 输入/输出tensor所在的buffer id列表
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     */
    template <typename inputType, int pos, typename BufList>
    __aicore__ inline void CopyInBrcInput(LocalTensor<inputType>& inTensor, GlobalTensor<inputType>& globalTensor,
                                          int64_t inputLength, const BufList& intList, int32_t pingPong)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
        WaitFlag<HardEvent::V_MTE2>(this->template Mte2Evt<pos>(pingPong));
        WaitFlag<HardEvent::MTE3_MTE2>(this->template Mte2Evt<pos>(pingPong));
        Vec::CopyIn<inputType>(inTensor, globalTensor, inputLength);
        SetFlag<HardEvent::MTE2_V>(this->template Mte2Evt<pos>(pingPong));
#else
        // Set getBuf
        GetTensor<TPosition::VECIN>(intList[1]);
        // Run copyIn
        CopyInWithPreAlloc<inputType, true>(inTensor, globalTensor, inputLength);
        // Set rlsBuf
        ReleaseTensor<TPosition::VECIN>(intList[1]);
#endif
    }

    /**
     * 执行广播计算
     * @tparam inputType 输入数据类型
     * @tparam BufList 缓冲区ID列表类型
     * @param outTensor 输出的LocalTensor对象
     * @param inTensor 输入的LocalTensor对象
     * @param realIdx 当前处理的索引
     * @param intList 输入/输出tensor所在的buffer id列表
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     */
    template <typename inputType, int pos, typename BufList>
    __aicore__ inline void RunCvecBrc(LocalTensor<inputType>& outTensor, LocalTensor<inputType>& inTensor,
                                      int64_t realIdx, const BufList& intList, int32_t pingPong)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
        WaitFlag<HardEvent::MTE2_V>(this->template Mte2Evt<pos>(pingPong));
        if constexpr (IsMte3Producer<BrcDag, Mte2SrcPolicy::Strict, pos>()) {
            WaitFlag<HardEvent::MTE3_V>(this->template Mte3Evt<pos>(pingPong));
        }
        CvecBrc<inputType>(outTensor, inTensor, realIdx);
        SetFlag<HardEvent::V_MTE2>(this->template Mte2Evt<pos>(pingPong));
        if constexpr (IsMte3Producer<BrcDag, Mte2SrcPolicy::Strict, pos>()) {
            SetFlag<HardEvent::V_MTE3>(this->template Mte3Evt<pos>(pingPong));
        }
#else
        GetTensor<TPosition::VECCALC>(intList[0]);
        GetTensor<TPosition::VECCALC>(intList[1]);
        CvecBrc<inputType>(outTensor, inTensor, realIdx);
        ReleaseTensor<TPosition::VECCALC>(intList[1]);
        ReleaseTensor<TPosition::VECCALC>(intList[0]);
#endif
    }

    /**
     * CopyInAndVecBrc函数
     * @tparam Op CopyInBrc的Bind节点
     * @tparam pos CopyInBrc在FunList内的索引
     * @param ubSplitSize UB切分大小
     * @param axesIndices 当前处理的索引
     * @param ubLoopIdx 当前处理的循环变量
     * @param pingPong DoubleBuffer中所处的ping/pong阶段
     */
    template <typename Op, int pos>
    __aicore__ inline void CopyInAndVecBrc(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS],
                                           int64_t ubLoopIdx, int32_t pingPong)
    {
        static_assert(Op::InHolders::Size == 1, "CopyIn input inHolders num should be 1.");
        using input = typename Op::InHolders::template At<0>;
        using inputType = typename Op::template FunInArgType<0>;
        int64_t realIdx = (AscendC::GetBlockIdx() * tilingData->blockFormer + ubLoopIdx) % tilingData->ubOuter;

        if ((tilingData->inputBrcStrides[cBrcIndex_][tilingData->ubSplitAxis] == 0) && ubLoopIdx > 0 && realIdx > 0) {
            SetSkipBrcFlag<pos>(pingPong);
            cBrcIndex_++;
            return;
        }
        // Prepare input args
        auto intList = this->template GetBufId<pos, true>(pingPong);
        LocalTensor<inputType> inTensor = this->tensorPool_[intList[1] * this->blockLen_]
                                              .template ReinterpretCast<inputType>();
#ifndef __CCE_KT_TEST__
        inTensor.SetBufferLen(this->blockEleNum_);
#endif
        int64_t gmOffset = BroadcastGetGmOffset(axesIndices, tilingData->inputBrcStrides[cBrcIndex_],
                                                tilingData->ubSplitAxis, tilingData->ubFormer);
        int64_t inputLength = copyInBrcInputLength[cBrcIndex_][1];
        if (tilingData->inputBrcStrides[cBrcIndex_][tilingData->ubSplitAxis] != 0) {
            inputLength = ubSplitSize * tilingData->inputBrcStrides[cBrcIndex_][tilingData->ubSplitAxis];
        } else {
            inputLength = realIdx == tilingData->ubOuter - 1 ? copyInBrcInputLength[cBrcIndex_][1] :
                                                               copyInBrcInputLength[cBrcIndex_][0];
        }

        GlobalTensor<inputType> globalTensor;
        globalTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ inputType*>(this->inGm_[input::Pos] + gmOffset * sizeof(inputType)));
        CopyInBrcInput<inputType, pos, decltype(intList)>(inTensor, globalTensor, inputLength, intList, pingPong);

        LocalTensor<inputType> outTensor = this->tensorPool_[intList[0] * this->blockLen_]
                                               .template ReinterpretCast<inputType>();
#ifndef __CCE_KT_TEST__
        outTensor.SetBufferLen(this->blockEleNum_);
#endif
        RunCvecBrc<inputType, pos, decltype(intList)>(outTensor, inTensor, realIdx, intList, pingPong);
        cBrcIndex_++;
    }

    // 遍历执行图
    template <int pos = 0, bool insideIf = false>
    __aicore__ inline void Run(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS], int64_t ubLoopIdx,
                               int32_t pingPong)
    {
        if constexpr (pos >= BrcDag::FunList::Size) {
            return;
        }

        // Run current func
        using Op = typename BrcDag::FunList::template At<pos>;
        using Func = typename Op::Fun;
        if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyIn>::Value && !insideIf) {
            constexpr int vecIndex = BrcDag::template VecBrcIdxDepend<Op>;
            if constexpr (vecIndex >= 0) {
                if ((tilingData->inputVecBrcStrides[vecIndex] != 0) ||
                    (ubLoopIdx == 0 ||
                     (AscendC::GetBlockIdx() * tilingData->blockFormer + ubLoopIdx) % tilingData->ubOuter == 0)) {
                    Run<pos, true>(ubSplitSize, axesIndices, ubLoopIdx, pingPong);
                }
                using vecBrcOp = typename BrcDag::VecBrcNodes::template At<vecIndex>;
                constexpr int vecBrcOpPos = BrcDag::FunList::template GetIndex<vecBrcOp>();
                Run<vecBrcOpPos + 1, false>(ubSplitSize, axesIndices, ubLoopIdx, pingPong);
                return;
            }
        }

        if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyIn>::Value) {
            this->template CopyIn<Op, pos>(axesIndices, ubLoopIdx, pingPong);
        } else if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyInBrc>::Value) {
            CopyInAndVecBrc<Op, pos>(ubSplitSize, axesIndices, ubLoopIdx, pingPong);
        } else if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyOut>::Value) {
            int64_t gmOffset = BroadcastGetGmOffset(axesIndices, tilingData->outputStrides, tilingData->ubSplitAxis,
                                                    tilingData->ubFormer);
            int64_t tileLength = ubSplitSize * tilingData->outputStrides[tilingData->ubSplitAxis];
            this->template CopyOut<Op, pos>(gmOffset, tileLength, pingPong);
        } else if constexpr (__aux::IsSameTemplateType<Func, Vec::Brc>::Value) {
            this->template VecBroadcast<Op, pos>(ubLoopIdx, pingPong);
        } else {
            uint64_t tileLength = ubSplitSize * tilingData->outputStrides[tilingData->ubSplitAxis];
            this->template RunNormalOp<Op, pos>(tileLength, pingPong);
        }

        if constexpr (insideIf && __aux::IsSameTemplateType<Func, Vec::Brc>::Value) {
            return;
        }

        // Run next func
        if constexpr (pos + 1 < BrcDag::FunList::Size) {
            Run<pos + 1, insideIf>(ubSplitSize, axesIndices, ubLoopIdx, pingPong);
        }
    }

private:
    constexpr static int bufferNum = BrcDag::template GetBufferNum<false, true>();
    const BroadcastBaseTilingData<BrcDag>* tilingData;

    constexpr static uint32_t COPY_BRC_SIZE = BrcDag::CopyBrcSize > 0 ? BrcDag::CopyBrcSize : 1;
    int64_t copyInBrcInputLength[COPY_BRC_SIZE][2] = {{1}};
    uint32_t copyInBrcFormerShape[COPY_BRC_SIZE][8] = {{1}};

    uint32_t cBrcFormer[COPY_BRC_SIZE] = {1};
    uint32_t cBrcTail[COPY_BRC_SIZE] = {1};

    uint32_t cBrcIndex_ = 0;
    int64_t ubLoopNum_ = 0;
};
} // namespace Base
} // namespace Ops
#endif // BROADCAST_UBBRC_SCH_H_