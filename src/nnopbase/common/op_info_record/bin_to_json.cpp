/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bin_to_json.h"
#include "opdev/op_dfx.h"

namespace aclnnOpInfoRecord {
namespace {
nlohmann::json Int8BinToJson(const void *ptr, size_t len)
{
    std::vector<int8_t> data;
    for (size_t idx = 0UL; idx < len;) {
        data.emplace_back(*(int8_t *)((uint8_t *)ptr + idx));
        idx += sizeof(int8_t);
    }
    return data;
}

nlohmann::json Uint8BinToJson(const void *ptr, size_t len)
{
    std::vector<uint8_t> data;
    for (size_t idx = 0UL; idx < len;) {
        data.emplace_back(*(uint8_t *)((uint8_t *)ptr + idx));
        idx += sizeof(uint8_t);
    }
    return data;
}

nlohmann::json Int16BinToJson(const void *ptr, size_t len)
{
    std::vector<int16_t> data;
    for (size_t idx = 0UL; idx < len;) {
        data.emplace_back(*(int16_t *)((uint8_t *)ptr + idx));
        idx += sizeof(int16_t);
    }
    return data;
}

nlohmann::json Uint16BinToJson(const void *ptr, size_t len)
{
    std::vector<uint16_t> data;
    for (size_t idx = 0UL; idx < len;) {
        data.emplace_back(*(uint16_t *)((uint8_t *)ptr + idx));
        idx += sizeof(uint16_t);
    }
    return data;
}

nlohmann::json Int32BinToJson(const void *ptr, size_t len)
{
    std::vector<int32_t> data;
    for (size_t idx = 0UL; idx < len;) {
        data.emplace_back(*(int32_t *)((uint8_t *)ptr + idx));
        idx += sizeof(int32_t);
    }
    return data;
}

nlohmann::json Uint32BinToJson(const void *ptr, size_t len)
{
    std::vector<uint32_t> data;
    for (size_t idx = 0UL; idx < len;) {
        data.emplace_back(*(uint32_t *)((uint8_t *)ptr + idx));
        idx += sizeof(uint32_t);
    }
    return data;
}

nlohmann::json Int64BinToJson(const void *ptr, size_t len)
{
    std::vector<int64_t> data;
    data.reserve(len / sizeof(int64_t));
    for (size_t idx = 0UL; idx < len;) {
        data.emplace_back(*(int64_t *)((uint8_t *)ptr + idx));
        idx += sizeof(int64_t);
    }
    return data;
}

nlohmann::json Uint64BinToJson(const void *ptr, size_t len)
{
    std::vector<uint64_t> data;
    for (size_t idx = 0UL; idx < len;) {
        data.emplace_back(*(uint64_t *)((uint8_t *)ptr + idx));
        idx += sizeof(uint64_t);
    }
    return data;
}

nlohmann::json FloatBinToJson(const void *ptr, size_t len)
{
    std::vector<float> data;
    for (size_t idx = 0UL; idx < len;) {
        data.emplace_back(*(float *)((uint8_t *)ptr + idx));
        idx += sizeof(float);
    }
    return data;
}

nlohmann::json DoubleBinToJson(const void *ptr, size_t len)
{
    std::vector<double> data;
    for (size_t idx = 0UL; idx < len;) {
        data.emplace_back(*op::internal::PtrCastTo<const double>(static_cast<const uint8_t *>(ptr) + idx));
        idx += sizeof(double);
    }
    return data;
}
} // namespace

const std::map<ge::DataType, std::function<nlohmann::json(const void *, size_t)>> BIN_TO_JSON = {
    {ge::DT_INT8, Int8BinToJson},
    {ge::DT_UINT8, Uint8BinToJson},
    {ge::DT_INT16, Int16BinToJson},
    {ge::DT_UINT16, Uint16BinToJson},
    {ge::DT_INT32, Int32BinToJson},
    {ge::DT_UINT32, Uint32BinToJson},
    {ge::DT_INT64, Int64BinToJson},
    {ge::DT_UINT64, Uint64BinToJson},
    {ge::DT_FLOAT, FloatBinToJson},
    {ge::DT_DOUBLE, DoubleBinToJson},
};
} // namespace aclnnOpInfoRecord