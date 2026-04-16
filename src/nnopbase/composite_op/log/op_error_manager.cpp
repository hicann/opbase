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
#include "base/dlog_pub.h"

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
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "Config_Error_Invalid_Environment_Variable",
        "ErrCode": "EZ1002",
        "ErrMessage": "%s failed. Environment variable %s is not configured.",
        "Arglist": "funcDesc, envName",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "Reset the environment variable by referring to the setup guide."
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "Config_Error_Invalid_Environment_Variable",
        "ErrCode": "EZ1003",
        "ErrMessage": "%s failed. Value %s for environment variable %s is invalid. Reason: %s.",
        "Arglist": "funcDesc, value, envName, reason",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "Reset the environment variable by referring to the setup guide."
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "File_Operation_Error_Open",
        "ErrCode": "EZ1004",
        "ErrMessage": "Failed to open file %s. Reason: %s.",
        "Arglist": "file, reason",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "N/A"
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "File_Operation_Error_Parse",
        "ErrCode": "EZ1005",
        "ErrMessage": "Failed to parse file %s. Reason: %s.",
        "Arglist": "file, reason",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "N/A"
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "File_Operation_Error_Parse",
        "ErrCode": "EZ1006",
        "ErrMessage": "Failed to parse file %s. Reason: %s.",
        "Arglist": "file, reason",
        "suggestion": {
            "Possible Cause": "1.The custom operator JSON file is damaged. 2.The built-in operator JSON file is damaged.",
            "Solution": "1.Reinstall the custom operator package. 2.Reinstall the built-in operator package."
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "Not_Supported_Data_Type",
        "ErrCode": "EZ1007",
        "ErrMessage": "Operator %s does not support data type %s. The supported data type range is %s.",
        "Arglist": "opName, dataType, dtypeRange",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "N/A"
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "Invalid_Argument_Tensor_Input_Shape",
        "ErrCode": "EZ1008",
        "ErrMessage": "Input parameter %s of operator %s has incorrect shape dim %s. Reason: %s.",
        "Arglist": "paraName, opName, dim, reason",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "N/A"
        }
    },
    {
        "errClass": "Nnopbase Errors",
        "errTitle": "Execution_Error",
        "ErrCode": "EZ1009",
        "ErrMessage": "Failed to execute operator %s. Reason: %s.",
        "Arglist": "opName, reason",
        "suggestion": {
            "Possible Cause": "N/A",
            "Solution": "1.If the tiling or infershape function does not exsit, check whether the tiling or infershape function is registered successfully. 2.If the tiling or infershape function fails to be executed, check the implementation logic of the tiling or infershape function."
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

void DlogRecordInner(int32_t moduleId, int32_t level, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    DlogVaList(moduleId, level, fmt, args);
    va_end(args);
}

int32_t CheckLogLevelInner(int32_t moduleId, int32_t level)
{
    return CheckLogLevel(moduleId, level);
}