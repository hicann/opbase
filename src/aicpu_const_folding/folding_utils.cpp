/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "folding_utils.h"

#include <climits>
#include <cstring>

namespace aicpu_folding {

static const char kPathSeparator = '/';
static const char* const kConstantFoldingSoPrefix = "libopconstant_folding_";
static const char* const kConstantFoldingSoSuffix = ".so";

std::string GetRealPath(const std::string& path)
{
    char resoved_path[PATH_MAX] = {0};
    if (realpath(path.c_str(), resoved_path) != nullptr) {
        return std::string(resoved_path);
    }
    return "";
}

std::string EnsureTrailingSlash(const std::string& path)
{
    if (path.empty() || path.back() == kPathSeparator) {
        return path;
    }
    return path + kPathSeparator;
}

bool IsConstantFoldingSo(const std::string& file_name)
{
    if (file_name.find(kConstantFoldingSoPrefix) != 0U) {
        return false;
    }
    size_t suffix_len = strlen(kConstantFoldingSoSuffix);
    if (file_name.length() < suffix_len) {
        return false;
    }
    if (file_name.compare(file_name.length() - suffix_len, suffix_len, kConstantFoldingSoSuffix) != 0) {
        return false;
    }
    return true;
}

} // namespace aicpu_folding
