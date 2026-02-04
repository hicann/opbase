/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __OP_CTX_DEF_H__
#define __OP_CTX_DEF_H__
#include <cstddef>
#include <iostream>
#include <sstream>
#include <vector>

#include "exe_graph/runtime/continuous_vector.h"
#include "opdev/common_types.h"
#include "opdev/op_def.h"

// Structs blow are from GE RT2.0. Must keep sync with GE.
namespace op::internal {
using AttrPtr = void *;
struct ComputeNodeInfo {
    const ge::char_t *node_type_;
    const ge::char_t *node_name_;
    size_t ir_inputs_num_;
    size_t inputs_num_;
    size_t outputs_num_;
    size_t ir_outputs_num_;
    size_t attr_size_;
    uint8_t reserved_[24]; // Reserved field, 32+8, do not directly use when only 8-byte left
    // following by input-AnchorInstanceInfo, inputs-outputs-CompileTimeTensorDesc, RuntimeAttrs,
    // output-AnchorInstanceInfo
    uint64_t place_holder;

    /*
    AnchorInstanceInfo[ir_inputs_num_]
    CompileTimeTensorDesc[inputs_num_ + outputs_num_]
    RuntimeAttrsDef
    */
};

struct AnchorInstanceInfo {
    uint32_t instance_start_;
    uint32_t instantiation_num_;
    uint8_t reserved_[40];
};

struct CompileTimeTensorDesc {
    DataType data_type_;
    gert::StorageFormat storage_format_;
    uint8_t reserved_[40];
};

struct RuntimeAttrsDef {
    size_t attr_num;
    uint8_t reserved_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left
    size_t offset[0];
};

struct KernelExtendInfo {
    const ge::char_t *kernel_name_;
    const ge::char_t *kernel_type_;
};

struct ExtendedTilingBuffer {
public:
    ExtendedTilingBuffer() = default;
    ~ExtendedTilingBuffer()
    {
        if (addr_) {
            std::free(addr_);
            addr_ = nullptr;
        }
    }

    aclnnStatus Init(size_t size)
    {
        addr_ = std::malloc(size);
        OP_CHECK(addr_ != nullptr, OP_LOGE(ACLNN_ERR_INNER, "failed to malloc size %zu.", size),
            return ACLNN_ERR_INNER);
        offset_ = 0;
        size_ = size;
        return ACLNN_SUCCESS;
    }

    void *Data() const
    {
        return static_cast<uint8_t *>(addr_) + offset_;
    }

    aclnnStatus Append(const void *data, size_t len)
    {
        if (offset_ + len > size_) {
            OP_CHECK(Enlarge(offset_ + len) == ACLNN_SUCCESS,
                OP_LOGE(ACLNN_ERR_INNER, "failed to enlarge, offset %ld, len %zu.", offset_, len),
                return ACLNN_ERR_INNER);
        }
        OP_CHECK(memcpy_s(static_cast<uint8_t *>(addr_) + offset_, size_ - offset_, data, len) == EOK,
            OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy."), return ACLNN_ERR_INNER);
        offset_ += len;
        return ACLNN_SUCCESS;
    }

    aclnnStatus Seek(off_t len, std::ios_base::seekdir dir = std::ios_base::cur)
    {
        off_t newOffset = 0;
        switch (dir) {
            case std::ios_base::beg:
                newOffset = len;
                break;
            case std::ios_base::cur:
                newOffset = offset_ + len;
                break;
            case std::ios_base::end:
                newOffset = size_ + len;
                break;
            default:
                OP_LOGE(ACLNN_ERR_INNER, "invalid seek direction: %d.", dir);
                return ACLNN_ERR_INNER;
        }
        if (static_cast<size_t>(newOffset) > size_) {
            OP_CHECK(Enlarge(newOffset) == ACLNN_SUCCESS,
                OP_LOGE(ACLNN_ERR_INNER, "failed to enlarge, size %ld.", newOffset), return ACLNN_ERR_INNER);
        }
        offset_ = newOffset;
        OP_LOGD("seek len %ld offset %ld.", len, offset_);
        return ACLNN_SUCCESS;
    }

private:
    aclnnStatus Enlarge(size_t size)
    {
        OP_CHECK(size_ != 0, OP_LOGE(ACLNN_ERR_INNER, "Current size is 0."), return ACLNN_ERR_INNER);
        constexpr size_t DOUBLE = 2;
        size_t newSize = size_;
        while (newSize < size) {
            newSize *= DOUBLE;
        }
        void *newAddr = std::malloc(newSize);
        OP_CHECK(newAddr != nullptr, OP_LOGE(ACLNN_ERR_INNER, "failed to malloc size %zu.", newSize),
            return ACLNN_ERR_INNER);
        OP_CHECK(memcpy_s(newAddr, size_, addr_, size_) == EOK, OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy."),
            std::free(newAddr);
            return ACLNN_ERR_INNER);
        std::free(addr_);
        addr_ = newAddr;
        size_ = newSize;
        return ACLNN_SUCCESS;
    }

    void *addr_{ nullptr };
    off_t offset_{ 0 };
    size_t size_{ 0 };
};

struct TilingData {
    size_t capacity_;
    size_t data_size_;
    void *data_;
    ExtendedTilingBuffer *buffer_;
    uint8_t reserved_[32];
};

// Tiling output
struct TilingCtxOutput {
    uint64_t *tilingKey_;
    int64_t *numBlocks_;
    bool *atomicCleanFlag_;
    TilingData *tilingData_;
    gert::TypedContinuousVector<size_t> *workspaceSize_;
    int64_t *tilingCond_;
    uint8_t *scheduleMode_;
    uint32_t *localMemorySize_;

    size_t inputNum_;
    size_t outputNum_;
};

constexpr size_t MAX_WORKSPACE_NUM = 64;            // max number of op workspace
constexpr size_t LAUNCH_ARG_SIZE = 128 * 1024;      // kernel launch arg size before tiling data
constexpr size_t MAX_TILING_DATA_SIZE = 800 * 1024; // max raw tiling data size

constexpr size_t MAX_ATTR_STRING_SIZE = 1024; // max op attr string length
constexpr size_t ATTR_CAPACITY = 32 * 1024;   // max total op attr data size in Bytes
constexpr size_t MAX_OP_ARG_NUM = 1024;       // max op input/output arg count
constexpr size_t MAX_OP_TYPE_COUNT = 512;     // max registed op type
} // namespace op::internal

#endif
