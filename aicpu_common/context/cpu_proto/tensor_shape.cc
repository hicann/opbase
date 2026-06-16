/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cpu_tensor_shape.h"
#include "tensor_shape_impl.h"

namespace aicpu {
TensorShape::TensorShape(TensorShapeImpl* tensorShape) : impl_(tensorShape) {}

/*
 * get dims value of tensor shape.
 */
std::vector<int64_t> TensorShape::GetDimSizes() const { return impl_->GetDimSizes(); }

/*
 * set dims value to tensor shape.
 */
void TensorShape::SetDimSizes(const std::vector<int64_t>& dims) { impl_->SetDimSizes(dims); }

/*
 * get format value of tensor shape.
 */
Format TensorShape::GetFormat() const { return impl_->GetFormat(); }

/*
 * set format value to tensor shape.
 */
void TensorShape::SetFormat(Format format) { impl_->SetFormat(format); }

/*
 * get unknown rank value of tensor shape.
 */
bool TensorShape::GetUnknownRank() const { return impl_->GetUnknownRank(); }

/*
 * set unknown rank value to tensor shape.
 */
void TensorShape::SetUnknownRank(bool unknown_rank) { impl_->SetUnknownRank(unknown_rank); }

/*
 * get dims size of tensor shape.
 */
int32_t TensorShape::GetDims() const { return impl_->GetDims(); }

/*
 * get dim value of tensor shape index dim.
 */
int64_t TensorShape::GetDimSize(int32_t index) const { return impl_->GetDimSize(index); }

/*
 * get data elements number.
 */
int64_t TensorShape::NumElements() const { return impl_->NumElements(); }
} // namespace aicpu