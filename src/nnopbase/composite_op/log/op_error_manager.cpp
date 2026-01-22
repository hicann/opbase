/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_log.h"

#include <string>
#include <vector>
#include "securec.h"
#include "base/err_msg.h"

static const std::string g_errorInfoJson = R"(
{
    "error_info_list": [
    {
        "errClass": "AclNN Errors",
        "errTitle": "AclNN_Inner_Error",
        "ErrCode": "EZ9999",
        "ErrMessage": "%s",
        "Arglist": "message",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "1. Check whether the CANN package is correct.\n2. Check whether the environment variable is correct."
        }
    },
    {
        "errClass": "AclNN Errors",
        "errTitle": "AclNN_Parameter_Error",
        "ErrCode": "EZ1001",
        "ErrMessage": "%s",
        "Arglist": "message"
    },
    {
        "errClass": "AclNN Errors",
        "errTitle": "AclNN_Runtime_Error",
        "ErrCode": "EZ9903",
        "ErrMessage": "%s",
        "Arglist": "message",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "In this scenario, collect the plog when the fault occurs and locate the fault based on the plog."
        }
    }
    ]
})";

REG_FORMAT_ERROR_MSG(g_errorInfoJson.c_str(), g_errorInfoJson.length());

constexpr size_t LIMIT_PREDEFINED_MESSAGE = 1024U;

void ReportErrorMessageInner(const std::string &code, const char *fmt, ...)
{
    std::vector<char> buf(LIMIT_PREDEFINED_MESSAGE, '\0');

    va_list argList;
    va_start(argList, fmt);
    auto ret = vsnprintf_s(buf.data(), LIMIT_PREDEFINED_MESSAGE, LIMIT_PREDEFINED_MESSAGE - 1U, fmt, argList);
    if (ret == -1) {
        OP_LOGW("Construct report error message fail, maybe the length of error message exceed limits: %zu",
                LIMIT_PREDEFINED_MESSAGE);
    }
    va_end(argList);

    const std::vector<const char *> msgKey = {"message"};
    const std::vector<const char *> msgvalue = {buf.data()};
    REPORT_PREDEFINED_ERR_MSG(code.c_str(), msgKey, msgvalue);
}
