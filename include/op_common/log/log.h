/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
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
} // namespace Base
} // namespace Ops
#endif // OP_COMMON_LOG_LOG_H
