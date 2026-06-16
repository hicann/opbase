/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cpu_tensor.h"
#include "tensor_impl.h"

namespace aicpu {
Tensor::Tensor(TensorImpl* impl) : impl_(impl) {}

/*
 * get tensor shape value of tensor.
 */
std::shared_ptr<TensorShape> Tensor::GetTensorShape() const { return impl_->GetTensorShape(); }

/*
 * set tensor shape value to tensor.
 */
bool Tensor::SetTensorShape(const TensorShape* shape) { return impl_->SetTensorShape(shape); }

/*
 * get data type value of tensor.
 */
DataType Tensor::GetDataType() const { return impl_->GetDataType(); }

/*
 * set data type value to tensor.
 */
void Tensor::SetDataType(DataType type) { impl_->SetDataType(type); }

/*
 * get data ptr of tensor.
 */
void* Tensor::GetData() const { return impl_->GetData(); }

/*
 * set data ptr to tensor.
 */
void Tensor::SetData(void* addr) { impl_->SetData(addr); }

/*
 * get data size of tensor.
 */
uint64_t Tensor::GetDataSize() const { return impl_->GetDataSize(); }

/*
 * set data size to tensor.
 */
void Tensor::SetDataSize(uint64_t size) { impl_->SetDataSize(size); }

/*
 * calculate data size by tensor shape.
 */
int64_t Tensor::CalcDataSizeByShape() const { return impl_->CalcDataSizeByShape(); }

/*
 * get data elements number.
 */
int64_t Tensor::NumElements() const { return impl_->NumElements(); }
} // namespace aicpu