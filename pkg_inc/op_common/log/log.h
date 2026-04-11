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
// General Parameter Validation Macros (EZ0008-EZ0024)
// These macros do not distinguish between input/output and use generic "parameter" naming
// ============================================================================

/**
 * EZ0008: Parameter shape error (single parameter, with correct shape)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectShape:string - Actual shape, format like "[1,128,128]"
 * correctShape:string - Expected shape, format like "[1,256,256]"
 * errMessage: OP opName's parameter paramName has incorrect shape [incorrectShape], it should be [correctShape].
 */
#define OP_LOGE_FOR_INVALID_SHAPE(opName, paramName, incorrectShape, correctShape)                            \
    do {                                                                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                                 \
            opName, "OP %s's parameter %s has incorrect shape [%s], it should be [%s].", opName, paramName,    \
            incorrectShape, correctShape);                                                                     \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_shape", "correct_shape"}; \
        const std::vector<const char*> msgvalue = {opName, paramName, incorrectShape, correctShape};           \
        REPORT_PREDEFINED_ERR_MSG("EZ0008", msgKey, msgvalue);                                                 \
    } while (0)

/**
 * EZ0009: Parameter shape error (single parameter, with reason)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectShape:string - Actual incorrect shape, format like "[1,128]"
 * reason:string - Reason for the error
 * errMessage: OP opName's parameter paramName has incorrect shape [incorrectShape]. Reason: reason.
 */
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, paramName, incorrectShape, reason)                    \
    do {                                                                                                      \
        OP_LOGE_WITHOUT_REPORT(                                                                               \
            opName, "OP %s's parameter %s has incorrect shape [%s]. Reason: %s.", opName, paramName,          \
            incorrectShape, reason);                                                                          \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_shape", "reason"};      \
        const std::vector<const char*> msgvalue = {opName, paramName, incorrectShape, reason};                \
        REPORT_PREDEFINED_ERR_MSG("EZ0009", msgKey, msgvalue);                                                \
    } while (0)

/**
 * EZ0010: Parameters shape error (multiple parameters, with reason)
 * opName:string - Operator name
 * paramNames:string - Parameter names
 * incorrectShapes:string - Actual incorrect shapes
 * reason:string - Reason for the error
 * errMessage: OP opName's parameters paramNames have incorrect shapes incorrectShapes. Reason: reason.
 */
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, paramNames, incorrectShapes, reason)                      \
    do {                                                                                                          \
        OP_LOGE_WITHOUT_REPORT(                                                                                   \
            opName, "OP %s's parameters %s have incorrect shapes %s. Reason: %s.", opName, paramNames,            \
            incorrectShapes, reason);                                                                             \
        const std::vector<const char*> msgKey = {"op_name", "param_names", "incorrect_shapes", "reason"};         \
        const std::vector<const char*> msgvalue = {opName, paramNames, incorrectShapes, reason};                  \
        REPORT_PREDEFINED_ERR_MSG("EZ0010", msgKey, msgvalue);                                                    \
    } while (0)

/**
 * EZ0011: Parameter shape dim error (single parameter, with correct dim)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectDim:string - Actual dimension, format like "2D" or "3D"
 * correctDim:string - Expected dimension, format like "4D"
 * errMessage: OP opName's parameter paramName has incorrect shape dim incorrectDim, it should be correctDim.
 */
#define OP_LOGE_FOR_INVALID_SHAPEDIM(opName, paramName, incorrectDim, correctDim)                              \
    do {                                                                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                                 \
            opName, "OP %s's parameter %s has incorrect shape dim %s, it should be %s.", opName, paramName,     \
            incorrectDim, correctDim);                                                                         \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_dim", "correct_dim"};     \
        const std::vector<const char*> msgvalue = {opName, paramName, incorrectDim, correctDim};               \
        REPORT_PREDEFINED_ERR_MSG("EZ0011", msgKey, msgvalue);                                                 \
    } while (0)

/**
 * EZ0012: Parameter shape dim error (single parameter, with reason)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectDim:string - Actual incorrect dimension
 * reason:string - Reason for the error
 * errMessage: OP opName's parameter paramName has incorrect shape dim incorrectDim. Reason: reason.
 */
#define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, paramName, incorrectDim, reason)                     \
    do {                                                                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                                 \
            opName, "OP %s's parameter %s has incorrect shape dim %s. Reason: %s.", opName, paramName,         \
            incorrectDim, reason);                                                                             \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_dim", "reason"};         \
        const std::vector<const char*> msgvalue = {opName, paramName, incorrectDim, reason};                  \
        REPORT_PREDEFINED_ERR_MSG("EZ0012", msgKey, msgvalue);                                                \
    } while (0)

/**
 * EZ0013: Parameters shape dim error (multiple parameters, with reason)
 * opName:string - Operator name
 * paramNames:string - Parameter names
 * incorrectDims:string - Actual incorrect dimensions
 * reason:string - Reason for the error
 * errMessage: OP opName's parameters paramNames have incorrect shape dims incorrectDims. Reason: reason.
 */
#define OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opName, paramNames, incorrectDims, reason)                    \
    do {                                                                                                          \
        OP_LOGE_WITHOUT_REPORT(                                                                                   \
            opName, "OP %s's parameters %s have incorrect shape dims %s. Reason: %s.", opName, paramNames,         \
            incorrectDims, reason);                                                                               \
        const std::vector<const char*> msgKey = {"op_name", "param_names", "incorrect_dims", "reason"};           \
        const std::vector<const char*> msgvalue = {opName, paramNames, incorrectDims, reason};                    \
        REPORT_PREDEFINED_ERR_MSG("EZ0013", msgKey, msgvalue);                                                    \
    } while (0)

/**
 * EZ0014: Parameter shape size error (single parameter, with correct size)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectSize:string - Actual size (total element count)
 * correctSize:string - Expected size (total element count)
 * errMessage: OP opName's parameter paramName has incorrect shape size incorrectSize, it should be correctSize.
 */
#define OP_LOGE_FOR_INVALID_SHAPESIZE(opName, paramName, incorrectSize, correctSize)                          \
    do {                                                                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                                 \
            opName, "OP %s's parameter %s has incorrect shape size %s, it should be %s.", opName, paramName,    \
            incorrectSize, correctSize);                                                                       \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_size", "correct_size"};  \
        const std::vector<const char*> msgvalue = {opName, paramName, incorrectSize, correctSize};             \
        REPORT_PREDEFINED_ERR_MSG("EZ0014", msgKey, msgvalue);                                                 \
    } while (0)

/**
 * EZ0015: Parameters shape size error (multiple parameters, with reason)
 * opName:string - Operator name
 * paramNames:string - Parameter names
 * incorrectSizes:string - Actual incorrect sizes
 * reason:string - Reason for the error
 * errMessage: OP opName's parameters paramNames have incorrect shape sizes incorrectSizes. Reason: reason.
 */
#define OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opName, paramNames, incorrectSizes, reason)                     \
    do {                                                                                                            \
        OP_LOGE_WITHOUT_REPORT(                                                                                     \
            opName, "OP %s's parameters %s have incorrect shape sizes %s. Reason: %s.", opName, paramNames,         \
            incorrectSizes, reason);                                                                                 \
        const std::vector<const char*> msgKey = {"op_name", "param_names", "incorrect_sizes", "reason"};            \
        const std::vector<const char*> msgvalue = {opName, paramNames, incorrectSizes, reason};                     \
        REPORT_PREDEFINED_ERR_MSG("EZ0015", msgKey, msgvalue);                                                      \
    } while (0)

/**
 * EZ0016: Parameter format error (single parameter, with correct format)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectFormat:string - Actual format (e.g., "NHWC")
 * correctFormat:string - Expected format (e.g., "NCHW")
 * errMessage: OP opName's parameter paramName has incorrect format incorrectFormat, it should be correctFormat.
 */
#define OP_LOGE_FOR_INVALID_FORMAT(opName, paramName, incorrectFormat, correctFormat)                          \
    do {                                                                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                                 \
            opName, "OP %s's parameter %s has incorrect format %s, it should be %s.", opName, paramName,       \
            incorrectFormat, correctFormat);                                                                   \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_format", "correct_format"}; \
        const std::vector<const char*> msgvalue = {opName, paramName, incorrectFormat, correctFormat};         \
        REPORT_PREDEFINED_ERR_MSG("EZ0016", msgKey, msgvalue);                                                 \
    } while (0)

/**
 * EZ0017: Parameters format error (multiple parameters, with reason)
 * opName:string - Operator name
 * paramNames:string - Parameter names
 * incorrectFormats:string - Actual incorrect formats
 * reason:string - Reason for the error
 * errMessage: OP opName's parameters paramNames have incorrect formats incorrectFormats. Reason: reason.
 */
#define OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName, paramNames, incorrectFormats, reason)                      \
    do {                                                                                                            \
        OP_LOGE_WITHOUT_REPORT(                                                                                     \
            opName, "OP %s's parameters %s have incorrect formats %s. Reason: %s.", opName, paramNames,              \
            incorrectFormats, reason);                                                                               \
        const std::vector<const char*> msgKey = {"op_name", "param_names", "incorrect_formats", "reason"};           \
        const std::vector<const char*> msgvalue = {opName, paramNames, incorrectFormats, reason};                    \
        REPORT_PREDEFINED_ERR_MSG("EZ0017", msgKey, msgvalue);                                                      \
    } while (0)

/**
 * EZ0018: Parameter dtype error (single parameter, with correct dtype)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectDtype:string - Actual data type (e.g., "INT32")
 * correctDtype:string - Expected data type (e.g., "FLOAT16")
 * errMessage: OP opName's parameter paramName has incorrect dtype incorrectDtype, it should be correctDtype.
 */
#define OP_LOGE_FOR_INVALID_DTYPE(opName, paramName, incorrectDtype, correctDtype)                            \
    do {                                                                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                                 \
            opName, "OP %s's parameter %s has incorrect dtype %s, it should be %s.", opName, paramName,        \
            incorrectDtype, correctDtype);                                                                     \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_dtype", "correct_dtype"}; \
        const std::vector<const char*> msgvalue = {opName, paramName, incorrectDtype, correctDtype};           \
        REPORT_PREDEFINED_ERR_MSG("EZ0018", msgKey, msgvalue);                                                 \
    } while (0)

/**
 * EZ0019: Parameter dtype error (single parameter, with reason)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectDtype:string - Actual incorrect data type
 * reason:string - Reason for the error
 * errMessage: OP opName's parameter paramName has incorrect dtype incorrectDtype. Reason: reason.
 */
#define OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, paramName, incorrectDtype, reason)                       \
    do {                                                                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                                 \
            opName, "OP %s's parameter %s has incorrect dtype %s. Reason: %s.", opName, paramName,             \
            incorrectDtype, reason);                                                                           \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_dtype", "reason"};       \
        const std::vector<const char*> msgvalue = {opName, paramName, incorrectDtype, reason};                 \
        REPORT_PREDEFINED_ERR_MSG("EZ0019", msgKey, msgvalue);                                                 \
    } while (0)

/**
 * EZ0020: Parameters dtype error (multiple parameters, with reason)
 * opName:string - Operator name
 * paramNames:string - Parameter names
 * incorrectDtypes:string - Actual incorrect data types
 * reason:string - Reason for the error
 * errMessage: OP opName's parameters paramNames have incorrect dtypes incorrectDtypes. Reason: reason.
 */
#define OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, paramNames, incorrectDtypes, reason)                        \
    do {                                                                                                            \
        OP_LOGE_WITHOUT_REPORT(                                                                                     \
            opName, "OP %s's parameters %s have incorrect dtypes %s. Reason: %s.", opName, paramNames,              \
            incorrectDtypes, reason);                                                                               \
        const std::vector<const char*> msgKey = {"op_name", "param_names", "incorrect_dtypes", "reason"};           \
        const std::vector<const char*> msgvalue = {opName, paramNames, incorrectDtypes, reason};                    \
        REPORT_PREDEFINED_ERR_MSG("EZ0020", msgKey, msgvalue);                                                      \
    } while (0)

/**
 * EZ0021: Parameter tensor num error (single parameter, with correct num)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectNum:int64_t - Actual tensor count
 * correctNum:string - Expected tensor count
 * errMessage: OP opName's parameter paramName has invalid tensor num incorrectNum, it should be correctNum.
 */
#define OP_LOGE_FOR_INVALID_TENSORNUM(opName, paramName, incorrectNum, correctNum)                             \
    do {                                                                                                         \
        OP_LOGE_WITHOUT_REPORT(                                                                                  \
            opName, "OP %s's parameter %s has invalid tensor num %" PRId64 ", it should be %s.", opName,         \
            paramName, static_cast<int64_t>(incorrectNum), correctNum);                                         \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_num", "correct_num"};      \
        std::string incorrectNumStr = std::to_string(static_cast<int64_t>(incorrectNum));                        \
        const std::vector<const char*> msgvalue = {opName, paramName, incorrectNumStr.c_str(), correctNum};      \
        REPORT_PREDEFINED_ERR_MSG("EZ0021", msgKey, msgvalue);                                                   \
    } while (0)

/**
 * EZ0022: Parameters tensor num error (multiple parameters, with reason)
 * opName:string - Operator name
 * paramNames:string - Parameter names
 * incorrectNums:string - Actual incorrect tensor counts
 * reason:string - Reason for the error
 * errMessage: OP opName's parameters paramNames have invalid tensor nums incorrectNums. Reason: reason.
 */
#define OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(opName, paramNames, incorrectNums, reason)                      \
    do {                                                                                                            \
        OP_LOGE_WITHOUT_REPORT(                                                                                     \
            opName, "OP %s's parameters %s have invalid tensor nums %s. Reason: %s.", opName, paramNames,           \
            incorrectNums, reason);                                                                                 \
        const std::vector<const char*> msgKey = {"op_name", "param_names", "incorrect_nums", "reason"};             \
        const std::vector<const char*> msgvalue = {opName, paramNames, incorrectNums, reason};                      \
        REPORT_PREDEFINED_ERR_MSG("EZ0022", msgKey, msgvalue);                                                      \
    } while (0)

/**
 * EZ0023: Parameter value error (single parameter, with reason)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectValue:string - Actual incorrect value
 * reason:string - Reason for the error
 * errMessage: OP opName's parameter paramName has incorrect value incorrectValue. Reason: reason.
 */
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, paramName, incorrectValue, reason)                      \
    do {                                                                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                                 \
            opName, "OP %s's parameter %s has incorrect value %s. Reason: %s.", opName, paramName,              \
            incorrectValue, reason);                                                                           \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_value", "reason"};       \
        const std::vector<const char*> msgvalue = {opName, paramName, incorrectValue, reason};                 \
        REPORT_PREDEFINED_ERR_MSG("EZ0023", msgKey, msgvalue);                                                 \
    } while (0)

/**
 * EZ0024: Parameters value error (multiple parameters, with reason)
 * opName:string - Operator name
 * paramNames:string - Parameter names
 * incorrectValues:string - Actual incorrect values
 * reason:string - Reason for the error
 * errMessage: OP opName's parameters paramNames have incorrect values incorrectValues. Reason: reason.
 */
#define OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, paramNames, incorrectValues, reason)                        \
    do {                                                                                                            \
        OP_LOGE_WITHOUT_REPORT(                                                                                     \
            opName, "OP %s's parameters %s have incorrect values %s. Reason: %s.", opName, paramNames,              \
            incorrectValues, reason);                                                                               \
        const std::vector<const char*> msgKey = {"op_name", "param_names", "incorrect_values", "reason"};           \
        const std::vector<const char*> msgvalue = {opName, paramNames, incorrectValues, reason};                    \
        REPORT_PREDEFINED_ERR_MSG("EZ0024", msgKey, msgvalue);                                                      \
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
