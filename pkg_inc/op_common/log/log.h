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
 * \file log.h
 * \brief
 */
#ifndef OP_COMMON_LOG_LOG_H
#define OP_COMMON_LOG_LOG_H

#include <string>
#include <sstream>
#include <cinttypes>
#include <unistd.h>
#include <sys/syscall.h>
#include "exe_graph/runtime/shape.h"
#include "base/err_msg.h"
#include "dlog_pub.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "op_common/op_host/util/opbase_export.h"
namespace Ops {
namespace Base {

#ifndef OP_SUBMOD_NAME
#define OP_SUBMOD_NAME "OPS_BASE"
#endif

static inline uint64_t GetTid()
{
    const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
    return tid;
}

template <class T>
typename std::enable_if<std::is_same<const char*, typename std::decay<T>::type>::value, const char*>::type GetSafeStr(
    T name)
{
    return name;
}

template <class T>
typename std::enable_if<std::is_same<char*, typename std::decay<T>::type>::value, const char*>::type GetSafeStr(T name)
{
    return name;
}

template <class T>
typename std::enable_if<std::is_same<std::string, typename std::decay<T>::type>::value, const char*>::type GetSafeStr(
    const T& name)
{
    return name.c_str();
}

inline const std::string& GetOpInfo(const std::string& str)
{
    return str;
}

inline const char* GetOpInfo(const char* str)
{
    return (str == nullptr) ? "nil" : str;
}

template <class T>
constexpr bool IsContextType()
{
    return !std::is_same<const char*, typename std::decay<T>::type>::value &&
           !std::is_same<char*, typename std::decay<T>::type>::value &&
           !std::is_same<std::string, typename std::decay<T>::type>::value;
}

template <class T>
typename std::enable_if<IsContextType<T>(), std::string>::type GetOpInfo(T context)
{
    if (context == nullptr) {
        return "nil:nil";
    }
    std::string opInfo = context->GetNodeType() != nullptr ? context->GetNodeType() : "nil";
    opInfo += ":";
    opInfo += context->GetNodeName() != nullptr ? context->GetNodeName() : "nil";
    return opInfo;
}

#define OpLogSub(moduleId, level, opInfo, fmt, ...)                                                                   \
    do {                                                                                                              \
        if (CheckLogLevel(static_cast<int>(moduleId), level) == 1) {                                                  \
            DlogRecord(                                                                                               \
                static_cast<int>(moduleId), level,                                                                    \
                "[%s:%d][%s]"                                                                                         \
                "[%s][%" PRIu64 "] OpName:[%s] " fmt,                                                                 \
                __FILE__, __LINE__, OP_SUBMOD_NAME, __FUNCTION__, Ops::Base::GetTid(), Ops::Base::GetSafeStr(opInfo), \
                ##__VA_ARGS__);                                                                                       \
        }                                                                                                             \
    } while (0)

#define OpLogErrSub(moduleId, level, opInfo, fmt, ...)                                                                \
    do {                                                                                                              \
        if (CheckLogLevel(static_cast<int>(moduleId), level) == 1) {                                                  \
            DlogRecord(                                                                                               \
                static_cast<int>(moduleId), level,                                                                    \
                "[%s:%d][%s]"                                                                                         \
                "[%s][%" PRIu64 "] OpName:[%s] " fmt,                                                                 \
                __FILE__, __LINE__, OP_SUBMOD_NAME, __FUNCTION__, Ops::Base::GetTid(), Ops::Base::GetSafeStr(opInfo), \
                ##__VA_ARGS__);                                                                                       \
        }                                                                                                             \
    } while (0)

#define D_OP_LOGI(opName, fmt, ...) OpLogSub(OP, DLOG_INFO, opName, fmt, ##__VA_ARGS__)
#define D_OP_LOGW(opName, fmt, ...) OpLogSub(OP, DLOG_WARN, opName, fmt, ##__VA_ARGS__)
#define D_OP_LOGE(opName, fmt, ...) OpLogErrSub(OP, DLOG_ERROR, opName, fmt, ##__VA_ARGS__)
#define D_OP_LOGD(opName, fmt, ...) OpLogSub(OP, DLOG_DEBUG, opName, fmt, ##__VA_ARGS__)
#define unlikely(x) __builtin_expect((x), 0)
#define likely(x) __builtin_expect((x), 1)

#define OP_LOGI(opName, ...) D_OP_LOGI(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_LOGW(opName, ...) D_OP_LOGW(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_LOGE_WITHOUT_REPORT(opName, ...) D_OP_LOGE(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_LOGE(opName, ...)                           \
    do {                                               \
        OP_LOGE_WITHOUT_REPORT(opName, ##__VA_ARGS__); \
        REPORT_INNER_ERR_MSG("EZ9999", ##__VA_ARGS__); \
    } while (0)

/**
 * opName:string
 * index:int
 * incorrectShape:string
 * correctShape:string
 * errMessage:OP[opName] indexth input has incorrect shape[incorrectShape], it should be[correctShape].
 */
#define OP_LOGE_WITH_INVALID_INPUT_SHAPE(opName, index, incorrectShape, correctShape)                              \
    do {                                                                                                           \
        std::string index_str = std::to_string(index);                                                             \
        OP_LOGE_WITHOUT_REPORT(                                                                                    \
            opName, "OP[%s] %sth input has incorrect shape[: %s], it should be[: %s].", opName, index_str.c_str(), \
            incorrectShape, correctShape);                                                                         \
        const std::vector<const char*> msgKey = {"op_name", "index", "incorrect_shape", "correct_shape"};          \
        const std::vector<const char*> msgvalue = {opName, index_str.c_str(), incorrectShape, correctShape};       \
        REPORT_PREDEFINED_ERR_MSG("EZ0001", msgKey, msgvalue);                                                     \
    } while (0)

/**
 * opName:string
 * attrName:string
 * incorrectVal:string
 * correctVal:string
 * errMessage:OP[opName] attr[: attrName], has incorrect value[: incorrectVal], it should be[: correctVal].
 */
#define OP_LOGE_WITH_INVALID_ATTR(opName, attrName, incorrectVal, correctVal)                              \
    do {                                                                                                   \
        OP_LOGE_WITHOUT_REPORT(                                                                            \
            opName, "OP[%s] attr[: %s], has incorrect value[: %s], it should be[: %s].", opName, attrName, \
            incorrectVal, correctVal);                                                                     \
        const std::vector<const char*> msgKey = {"op_name", "attr_name", "incorrect_val", "correct_val"};  \
        const std::vector<const char*> msgvalue = {opName, attrName, incorrectVal, correctVal};            \
        REPORT_PREDEFINED_ERR_MSG("EZ0002", msgKey, msgvalue);                                             \
    } while (0)

/**
 * opName:string
 * attrName:string
 * incorrectSize:string
 * correctSize:string
 * errMessage:OP[opName] attr[: attrName], has incorrect size[: incorrectSize], it should be[: correctSize].
 */
#define OP_LOGE_WITH_INVALID_ATTR_SIZE(opName, attrName, incorrectSize, correctSize)                        \
    do {                                                                                                    \
        OP_LOGE_WITHOUT_REPORT(                                                                             \
            opName, "OP[%s] attr[: %s], has incorrect size[: %s], it should be[: %s].", opName, attrName,   \
            incorrectSize, correctSize);                                                                    \
        const std::vector<const char*> msgKey = {"op_name", "attr_name", "incorrect_size", "correct_size"}; \
        const std::vector<const char*> msgvalue = {opName, attrName, incorrectSize, correctSize};           \
        REPORT_PREDEFINED_ERR_MSG("EZ0003", msgKey, msgvalue);                                              \
    } while (0)

/**
 * opName:string
 * paramName:string
 * errMessage:OP[opName] get [paramName] failed.
 */
#define OP_LOGE_WITH_INVALID_INPUT(opName, paramName)                                 \
    do {                                                                              \
        OP_LOGE_WITHOUT_REPORT(opName, "OP[%s] get [%s] failed.", opName, paramName); \
        const std::vector<const char*> msgKey = {"op_name", "param_name"};            \
        const std::vector<const char*> msgvalue = {opName, param_name};               \
        REPORT_PREDEFINED_ERR_MSG("EZ0004", msgKey, msgvalue);                        \
    } while (0)

/**
 * opName:string
 * index:int
 * incorrectSize:string
 * correctSize:string
 * errMessage:OP[opName] indexth input has incorrect shape size[: incorrectSize], it should be[: correctSize].
 */
#define OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE(opName, index, incorrectSize, correctSize)                    \
    do {                                                                                                   \
        std::string index_str = std::to_string(index);                                                     \
        OP_LOGE_WITHOUT_REPORT(                                                                            \
            opName, "OP[%s] %sth input has incorrect shape size[: %s], it should be[: %s].", opName,       \
            index_str.c_str(), incorrectSize, correctSize);                                                \
        const std::vector<const char*> msgKey = {"op_name", "index", "incorrect_size", "correct_size"};    \
        const std::vector<const char*> msgvalue = {opName, index_str.c_str(), incorrectSize, correctSize}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0005", msgKey, msgvalue);                                             \
    } while (0)

/**
 * opName:string
 * paramName:string
 * dataFormat:string
 * expectedFormatList:string
 * errMessage:OP[opName] paramName has incorrect format[: dataFormat], it should be expectedFormatList.
 */
#define OP_LOGE_WITH_INVALID_INPUT_FORMAT(opName, paramName, dataFormat, expectedFormatList)                      \
    do {                                                                                                          \
        OP_LOGE_WITHOUT_REPORT(                                                                                   \
            opName, "OP[%s] %s has incorrect format[: %s], it should be %s.", opName, paramName, dataFormat,      \
            expectedFormatList);                                                                                  \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "data_format", "expected_format_list"}; \
        const std::vector<const char*> msgvalue = {opName, paramName, dataFormat, expectedFormatList};            \
        REPORT_PREDEFINED_ERR_MSG("EZ0006", msgKey, msgvalue);                                                    \
    } while (0)

/**
 * opName:string
 * paramName:string
 * dataDtype:string
 * expectedDtypeList:string
 * errMessage:OP[opName] paramName has incorrect dtype[: dataDtype], it should be expectedDtypeList.
 */
#define OP_LOGE_WITH_INVALID_INPUT_DTYPE(opName, paramName, dataDtype, expectedDtypeList)                       \
    do {                                                                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                                 \
            opName, "OP[%s] %s has incorrect dtype[: %s], it should be %s.", opName, paramName, dataDtype,      \
            expectedDtypeList);                                                                                 \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "data_dtype", "expected_dtype_list"}; \
        const std::vector<const char*> msgvalue = {opName, paramName, dataDtype, expectedDtypeList};            \
        REPORT_PREDEFINED_ERR_MSG("EZ0007", msgKey, msgvalue);                                                  \
    } while (0)

// ============================================================================
// Input Validation Macros (EZ0008-EZ0011)
// ============================================================================

/**
 * EZ0008: 输入shape异常
 * opName:string - 算子名称
 * inputName:string - 输入tensor名称
 * incorrectShape:string - 实际的形状，格式如 "[1,128,128]"
 * correctShape:string - 期望的形状，格式如 "[1,256,256]"
 * errMessage: OP opName input inputName has incorrect shape [incorrectShape], it should be [correctShape].
 */
#define OP_LOGE_FOR_INVALID_INPUT_SHAPE(opName, inputName, incorrectShape, correctShape)                              \
    do {                                                                                                              \
        OP_LOGE_WITHOUT_REPORT(                                                                                       \
            opName, "OP %s input %s has incorrect shape [%s], it should be [%s].", opName, inputName, incorrectShape, \
            correctShape);                                                                                            \
        const std::vector<const char*> msgKey = {"op_name", "input_name", "incorrect_shape", "correct_shape"};        \
        const std::vector<const char*> msgvalue = {opName, inputName, incorrectShape, correctShape};                  \
        REPORT_PREDEFINED_ERR_MSG("EZ0008", msgKey, msgvalue);                                                        \
    } while (0)

/**
 * EZ0009: 输入shape dim异常
 * opName:string - 算子名称
 * inputName:string - 输入tensor名称
 * incorrectDim:string - 实际的维度，格式如 "2D" 或 "3D"
 * correctDim:string - 期望的维度，格式如 "4D"
 * errMessage: OP opName input inputName has incorrect shape dim [incorrectDim], it should be [correctDim].
 */
#define OP_LOGE_FOR_INVALID_INPUT_SHAPE_DIM(opName, inputName, incorrectDim, correctDim)                   \
    do {                                                                                                   \
        OP_LOGE_WITHOUT_REPORT(                                                                            \
            opName, "OP %s input %s has incorrect shape dim [%s], it should be [%s].", opName, inputName,  \
            incorrectDim, correctDim);                                                                     \
        const std::vector<const char*> msgKey = {"op_name", "input_name", "incorrect_dim", "correct_dim"}; \
        const std::vector<const char*> msgvalue = {opName, inputName, incorrectDim, correctDim};           \
        REPORT_PREDEFINED_ERR_MSG("EZ0009", msgKey, msgvalue);                                             \
    } while (0)

/**
 * EZ0010: 输入shape异常（含原因）
 * opName:string - 算子名称
 * inputName:string - 输入tensor名称
 * incorrectShape:string - 实际的形状（错误），格式如 "[1,128]" 或 "3D"
 * reason:string - 异常原因描述
 * errMessage: OP opName input inputName has incorrect shape [incorrectShape], reason: reason.
 */
#define OP_LOGE_FOR_INVALID_INPUT_SHAPE_WITH_REASON(opName, inputName, incorrectShape, reason)                 \
    do {                                                                                                       \
        OP_LOGE_WITHOUT_REPORT(                                                                                \
            opName, "OP %s input %s has incorrect shape [%s], reason: %s.", opName, inputName, incorrectShape, \
            reason);                                                                                           \
        const std::vector<const char*> msgKey = {"op_name", "input_name", "incorrect_shape", "reason"};        \
        const std::vector<const char*> msgvalue = {opName, inputName, incorrectShape, reason};                 \
        REPORT_PREDEFINED_ERR_MSG("EZ0010", msgKey, msgvalue);                                                 \
    } while (0)

/**
 * EZ0011: 输入shape size异常
 * opName:string - 算子名称
 * inputName:string - 输入tensor名称
 * incorrectSize:string - 实际的大小，元素总数
 * correctSize:string - 期望的大小，元素总数
 * errMessage: OP opName input inputName has incorrect shape size incorrectSize, it should be correctSize.
 */
#define OP_LOGE_FOR_INVALID_INPUT_SHAPESIZE(opName, inputName, incorrectSize, correctSize)                            \
    do {                                                                                                              \
        OP_LOGE_WITHOUT_REPORT(                                                                                       \
            opName, "OP %s input %s has incorrect shape size %s, it should be %s.", opName, inputName, incorrectSize, \
            correctSize);                                                                                             \
        const std::vector<const char*> msgKey = {"op_name", "input_name", "incorrect_size", "correct_size"};          \
        const std::vector<const char*> msgvalue = {opName, inputName, incorrectSize, correctSize};                    \
        REPORT_PREDEFINED_ERR_MSG("EZ0011", msgKey, msgvalue);                                                        \
    } while (0)

// ============================================================================
// Output Validation Macros (EZ0012-EZ0017)
// ============================================================================

/**
 * EZ0012: 输出shape异常
 * opName:string - 算子名称
 * outputName:string - 输出tensor名称
 * incorrectShape:string - 实际的形状，格式如 "[1,128,128]"
 * correctShape:string - 期望的形状，格式如 "[1,256,256]"
 * errMessage: OP opName output outputName has incorrect shape [incorrectShape], it should be [correctShape].
 */
#define OP_LOGE_FOR_INVALID_OUTPUT_SHAPE(opName, outputName, incorrectShape, correctShape)                      \
    do {                                                                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                                 \
            opName, "OP %s output %s has incorrect shape [%s], it should be [%s].", opName, outputName,         \
            incorrectShape, correctShape);                                                                      \
        const std::vector<const char*> msgKey = {"op_name", "output_name", "incorrect_shape", "correct_shape"}; \
        const std::vector<const char*> msgvalue = {opName, outputName, incorrectShape, correctShape};           \
        REPORT_PREDEFINED_ERR_MSG("EZ0012", msgKey, msgvalue);                                                  \
    } while (0)

/**
 * EZ0013: 输出shape dim异常
 * opName:string - 算子名称
 * outputName:string - 输出tensor名称
 * incorrectDim:string - 实际的维度，格式如 "2D" 或 "3D"
 * correctDim:string - 期望的维度，格式如 "4D"
 * errMessage: OP opName output outputName has incorrect shape dim [incorrectDim], it should be [correctDim].
 */
#define OP_LOGE_FOR_INVALID_OUTPUT_SHAPE_DIM(opName, outputName, incorrectDim, correctDim)                  \
    do {                                                                                                    \
        OP_LOGE_WITHOUT_REPORT(                                                                             \
            opName, "OP %s output %s has incorrect shape dim [%s], it should be [%s].", opName, outputName, \
            incorrectDim, correctDim);                                                                      \
        const std::vector<const char*> msgKey = {"op_name", "output_name", "incorrect_dim", "correct_dim"}; \
        const std::vector<const char*> msgvalue = {opName, outputName, incorrectDim, correctDim};           \
        REPORT_PREDEFINED_ERR_MSG("EZ0013", msgKey, msgvalue);                                              \
    } while (0)

/**
 * EZ0014: 输出shape异常（含原因）
 * opName:string - 算子名称
 * outputName:string - 输出tensor名称
 * incorrectShape:string - 实际的形状（错误），格式如 "[1,128]" 或 "3D"
 * reason:string - 异常原因描述
 * errMessage: OP opName output outputName has incorrect shape [incorrectShape], reason: reason.
 */
#define OP_LOGE_FOR_INVALID_OUTPUT_SHAPE_WITH_REASON(opName, outputName, incorrectShape, reason)                 \
    do {                                                                                                         \
        OP_LOGE_WITHOUT_REPORT(                                                                                  \
            opName, "OP %s output %s has incorrect shape [%s], reason: %s.", opName, outputName, incorrectShape, \
            reason);                                                                                             \
        const std::vector<const char*> msgKey = {"op_name", "output_name", "incorrect_shape", "reason"};         \
        const std::vector<const char*> msgvalue = {opName, outputName, incorrectShape, reason};                  \
        REPORT_PREDEFINED_ERR_MSG("EZ0014", msgKey, msgvalue);                                                   \
    } while (0)

/**
 * EZ0015: 输出shape size异常
 * opName:string - 算子名称
 * outputName:string - 输出tensor名称
 * incorrectSize:string - 实际的大小，元素总数
 * correctSize:string - 期望的大小，元素总数
 * errMessage: OP opName output outputName has incorrect shape size incorrectSize, it should be correctSize.
 */
#define OP_LOGE_FOR_INVALID_OUTPUT_SHAPESIZE(opName, outputName, incorrectSize, correctSize)                  \
    do {                                                                                                      \
        OP_LOGE_WITHOUT_REPORT(                                                                               \
            opName, "OP %s output %s has incorrect shape size %s, it should be %s.", opName, outputName,      \
            incorrectSize, correctSize);                                                                      \
        const std::vector<const char*> msgKey = {"op_name", "output_name", "incorrect_size", "correct_size"}; \
        const std::vector<const char*> msgvalue = {opName, outputName, incorrectSize, correctSize};           \
        REPORT_PREDEFINED_ERR_MSG("EZ0015", msgKey, msgvalue);                                                \
    } while (0)

/**
 * EZ0016: 输出format异常
 * opName:string - 算子名称
 * outputName:string - 输出tensor名称
 * incorrectFormat:string - 错误的格式
 * correctFormat:string - 正确的格式
 * errMessage: OP opName output outputName has incorrect format incorrectFormat, it should be correctFormat.
 */
#define OP_LOGE_FOR_INVALID_OUTPUT_FORMAT(opName, outputName, incorrectFormat, correctFormat)                         \
    do {                                                                                                              \
        OP_LOGE_WITHOUT_REPORT(                                                                                       \
            opName, "OP %s output %s has incorrect format %s, it should be %s.", opName, outputName, incorrectFormat, \
            correctFormat);                                                                                           \
        const std::vector<const char*> msgKey = {"op_name", "output_name", "incorrect_format", "correct_format"};     \
        const std::vector<const char*> msgvalue = {opName, outputName, incorrectFormat, correctFormat};               \
        REPORT_PREDEFINED_ERR_MSG("EZ0016", msgKey, msgvalue);                                                        \
    } while (0)

/**
 * EZ0017: 输出dtype异常
 * opName:string - 算子名称
 * outputName:string - 输出tensor名称
 * incorrectDtype:string - 错误的数据类型
 * correctDtype:string - 正确的数据类型
 * errMessage: OP opName output outputName has incorrect dtype incorrectDtype, it should be correctDtype.
 */
#define OP_LOGE_FOR_INVALID_OUTPUT_DTYPE(opName, outputName, incorrectDtype, correctDtype)                          \
    do {                                                                                                            \
        OP_LOGE_WITHOUT_REPORT(                                                                                     \
            opName, "OP %s output %s has incorrect dtype %s, it should be %s.", opName, outputName, incorrectDtype, \
            correctDtype);                                                                                          \
        const std::vector<const char*> msgKey = {"op_name", "output_name", "incorrect_dtype", "correct_dtype"};     \
        const std::vector<const char*> msgvalue = {opName, outputName, incorrectDtype, correctDtype};               \
        REPORT_PREDEFINED_ERR_MSG("EZ0017", msgKey, msgvalue);                                                      \
    } while (0)

#define OP_LOGD(opName, ...) D_OP_LOGD(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_CHECK_IF(condition, log, return_expr) \
    do {                                         \
        if (unlikely(condition)) {               \
            log;                                 \
            return_expr;                         \
        }                                        \
    } while (0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                           \
    do {                                                                                                   \
        if (unlikely((ptr) == nullptr)) {                                                                  \
            const char* name = (unlikely(((context) == nullptr) || (context)->GetNodeName() == nullptr)) ? \
                                   "nil" :                                                                 \
                                   (context)->GetNodeName();                                               \
            OP_LOGE(name, "%s is nullptr!", #ptr);                                                         \
            return ge::GRAPH_FAILED;                                                                       \
        }                                                                                                  \
    } while (0)
OPBASE_API std::string ToString(ge::DataType type);
OPBASE_API std::string ToString(ge::Format format);
OPBASE_API std::string ToString(const gert::Shape& shape);
OPBASE_API std::string ToString(const std::vector<const gert::Shape*>& v);
} // namespace Base
} // namespace Ops
#endif // OP_COMMON_LOG_LOG_H
