/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_OP_GRAPH_OP_FALLBACK_INTERNAL_H_
#define INC_OP_GRAPH_OP_FALLBACK_INTERNAL_H_

#include <dlfcn.h>

#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "aclnn/aclnn_base.h"
#include "exe_graph/runtime/tensor.h"
#include "op_common/log/log.h"

namespace fallback {

using namespace ge;

using aclOpExecutor = struct aclOpExecutor;
using aclTensor = struct aclTensor;
using aclScalar = struct aclScalar;
using aclTensorList = struct aclTensorList;

using _aclCreateTensor = aclTensor* (*)(const int64_t* view_dims, uint64_t view_dims_num, aclDataType data_type,
                                        const int64_t* stride, int64_t offset, aclFormat format,
                                        const int64_t* storage_dims, uint64_t storage_dims_num, void* tensor_data);

using _aclCreateScalar = aclScalar* (*)(void* value, aclDataType data_type);
using _aclCreateTensorList = aclTensorList* (*)(const aclTensor* const* value, uint64_t size);

using _aclDestroyTensor = int (*)(const aclTensor* tensor);
using _aclDestroyScalar = int (*)(const aclScalar* scalar);
using _aclDestroyTensorList = int (*)(const aclTensorList* array);

using ResetCacheThreadLocal = void (*)();

#define GET_OP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

namespace std_utils {
template <std::size_t... Is>
struct index_sequence {};

template <std::size_t N, std::size_t... Is>
struct make_index_sequence_helper : make_index_sequence_helper<N - 1, N - 1, Is...> {};

template <std::size_t... Is>
struct make_index_sequence_helper<0, Is...> {
    using type = index_sequence<Is...>;
};

template <std::size_t N>
using make_index_sequence = typename make_index_sequence_helper<N>::type;
} // namespace std_utils

inline const char* GetOpApiLibName(void) { return "libopapi.so"; }

inline const char* GetCustOpApiLibName(void) { return "libcust_opapi.so"; }

inline void* GetOpApiFuncAddrInLib(void* handler, const char* libName, const char* apiName)
{
    auto funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        OP_LOGW("aclnnfallback", "dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void* GetOpApiLibHandler(const char* libName)
{
    auto handler = dlopen(libName, RTLD_NOW | RTLD_NOLOAD);
    if (handler == nullptr) {
        handler = dlopen(libName, RTLD_LAZY);
    }
    if (handler == nullptr) {
        OP_LOGW("aclnnfallback", "dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}

inline void* GetAclnnArrdByApiName(const char* apiName)
{
    std::vector<std::string> libs = {"libaclnn_ops_infer.so", "libaclnn_ops_train.so", "libaclnn_math.so",
                                     "libaclnn_rand.so",      "libaclnn_sparse.so",    "libaclnn_fft.so"};
    for (const auto& libName : libs) {
        static auto libHandler = GetOpApiLibHandler(libName.c_str());
        if (libHandler != nullptr) {
            auto funcAddr = GetOpApiFuncAddrInLib(libHandler, libName.c_str(), apiName);
            if (funcAddr != nullptr) {
                return funcAddr;
            }
        }
    }
    OP_LOGE("aclnnfallback", "api %s can't find in any aclnn lib.", apiName);
    return nullptr;
}

inline void* GetOpApiFuncAddr(const char* apiName)
{
    static auto custOpApiHandler = GetOpApiLibHandler(GetCustOpApiLibName());
    if (custOpApiHandler != nullptr) {
        auto funcAddr = GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }

    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    if (opApiHandler != nullptr) {
        auto funcAddr = GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }
    OP_LOGD("aclnnfallback", "opapi lib is not exist,will use aclnn lib.");
    return GetAclnnArrdByApiName(apiName);
}

inline aclDataType GetConvertType(const gert::Tensor* ge_tensor)
{
    auto dataType_ge = ge_tensor->GetDataType();
    auto dataType = aclDataType::ACL_FLOAT16;
    if (dataType_ge == DT_FLOAT) {
        dataType = aclDataType::ACL_FLOAT;
    } else if (dataType_ge == DT_BF16) {
        dataType = aclDataType::ACL_BF16;
    } else if (dataType_ge == DT_BOOL) {
        dataType = aclDataType::ACL_BOOL;
    } else if (dataType_ge == DT_INT64) {
        dataType = aclDataType::ACL_INT64;
    } else if (dataType_ge == DT_INT32) {
        dataType = aclDataType::ACL_INT32;
    } else if (dataType_ge == DT_UINT64) {
        dataType = aclDataType::ACL_UINT64;
    } else if (dataType_ge == DT_UINT32) {
        dataType = aclDataType::ACL_UINT32;
    } else if (dataType_ge == DT_INT8) {
        dataType = aclDataType::ACL_INT8;
    } else if (dataType_ge == DT_UINT8) {
        dataType = aclDataType::ACL_UINT8;
    } else if (dataType_ge == DT_INT4) {
        dataType = aclDataType::ACL_INT4;
    } else if (dataType_ge == DT_FLOAT8_E4M3FN) {
        dataType = aclDataType::ACL_FLOAT8_E4M3FN;
    } else if (dataType_ge == DT_FLOAT8_E5M2) {
        dataType = aclDataType::ACL_FLOAT8_E5M2;
    } else if (dataType_ge == DT_FLOAT4_E2M1) {
        dataType = aclDataType::ACL_FLOAT4_E2M1;
    } else if (dataType_ge == DT_FLOAT4_E1M2) {
        dataType = aclDataType::ACL_FLOAT4_E1M2;
    } else if (dataType_ge == DT_FLOAT8_E8M0) {
        dataType = aclDataType::ACL_FLOAT8_E8M0;
    } else if (dataType_ge == DT_HIFLOAT8) {
        dataType = aclDataType::ACL_HIFLOAT8;
    } else if (dataType_ge == DT_COMPLEX64) {
        dataType = aclDataType::ACL_COMPLEX64;
    } else if (dataType_ge == DT_DOUBLE) {
        dataType = aclDataType::ACL_DOUBLE;
    } else {
        dataType = aclDataType::ACL_FLOAT16;
    }
    return dataType;
}

inline aclTensor* ConvertType(const gert::Tensor* ge_tensor)
{
    if (ge_tensor == nullptr) {
        return nullptr;
    }

    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    OP_CHECK_IF(aclCreateTensor == nullptr, OP_LOGE("aclnnfallback", "aclCreateTensor nullptr"), return nullptr);

    void* device_addr = nullptr;
    auto tensor_place = ge_tensor->GetPlacement();
    OP_LOGD("aclnnfallback", "GetPlacement: tensor place is %d", tensor_place);
    device_addr = (void*)ge_tensor->GetAddr();

    auto dataType = GetConvertType(ge_tensor);
    OP_LOGD("aclnnfallback", "aclCreateTensor: tensor type is %d", dataType);

    auto gert_shape = ge_tensor->GetStorageShape();
    std::vector<int64_t> shape;
    for (size_t i = 0; i < gert_shape.GetDimNum(); ++i) {
        shape.push_back(gert_shape.GetDim(i));
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    aclTensor* out = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        device_addr);

    OP_CHECK_IF(out == nullptr, OP_LOGE("aclnnfallback", "out nullptr"), return nullptr);

    return out;
}

inline aclTensorList* ConvertType(std::vector<const gert::Tensor*>& ge_tenserList)
{
    OP_CHECK_IF(ge_tenserList.size() == 0, OP_LOGE("aclnnfallback", "ge_tenserList size 0"), return nullptr);

    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    OP_CHECK_IF(aclCreateTensorList == nullptr, OP_LOGE("aclnnfallback", "ge_tenserList size 0"), return nullptr);

    std::vector<aclTensor*> tmp;
    for (size_t i = 0; i < ge_tenserList.size(); i++) {
        auto t_acl = ConvertType(ge_tenserList[i]);
        tmp.push_back(t_acl);
    }

    aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());
    return tensorList;
}

template <typename T>
T ConvertType(T value)
{
    return value;
}

template <typename T>
inline aclScalar* ConvertScalarType(T value)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    OP_CHECK_IF(aclCreateScalar == nullptr, OP_LOGE("aclnnfallback", "aclCreateScalar nullptr"), return nullptr);
    if (typeid(value) == typeid(float)) {
        return aclCreateScalar(&value, aclDataType::ACL_FLOAT);
    }
    return nullptr;
}

inline void Release(aclTensor* p)
{
    static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
    OP_CHECK_IF(aclDestroyTensor == nullptr, OP_LOGE("aclnnfallback", "aclDestroyTensor is null"), return);
    aclDestroyTensor(p);
}

inline void Release(aclScalar* p)
{
    static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
    OP_CHECK_IF(aclDestroyScalar == nullptr, OP_LOGE("aclnnfallback", "aclDestroyScalar is null"), return);
    aclDestroyScalar(p);
}

inline void Release(aclTensorList* p)
{
    static const auto aclDestroyTensorList = GET_OP_API_FUNC(aclDestroyTensorList);
    OP_CHECK_IF(aclDestroyTensorList == nullptr, OP_LOGE("aclnnfallback", "aclDestroyTensorList is null"), return);
    aclDestroyTensorList(p);
}

template <typename T>
void Release(T value)
{
    (void)value;
}

template <typename Tuple, size_t... I>
void CallRelease(Tuple t, std_utils::index_sequence<I...>)
{
    (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(Tuple& t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    CallRelease(t, std_utils::make_index_sequence<size>{});
}

template <typename... Ts>
auto ConvertTypes(Ts&... args) -> decltype(std::make_tuple(ConvertType(args)...))
{
    auto tp = std::make_tuple(ConvertType(args)...);
    return tp;
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std_utils::index_sequence<I...>) -> int
{
    return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t) -> int
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std_utils::make_index_sequence<size>{});
}

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr, std_utils::index_sequence<I...>)
    -> int (*)(typename std::decay<decltype(std::get<I>(params))>::type...)
{
    using OpApiFunc = int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
    auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
    return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr) -> typename std::enable_if<
    std::tuple_size<Tuple>::value != 0,
    decltype(ConvertToOpApiFunc(
        params, opApiAddr, std_utils::make_index_sequence<std::tuple_size<Tuple>::value>{}))>::type
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return ConvertToOpApiFunc(params, opApiAddr, std_utils::make_index_sequence<size>{});
}

} // namespace fallback

#endif // INC_OP_GRAPH_OP_FALLBACK_INTERNAL_H_
