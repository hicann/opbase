/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indv_soc.h"
#include <algorithm>
#include <cctype>
#include <string>
#include <map>
#include <vector>
#include "platform.h"
#include "indv_base.h"
#include "utils/indv_debug_assert.h"

namespace nnopbase {
IndvSoc& IndvSoc::GetInstance(void)
{
    static IndvSoc instance;
    return instance;
}

IndvSoc::IndvSoc(void)
{
}

void IndvSoc::Init(void)
{
    socVersion = GetCurrentSocVersionInternal();
    isInit = true;
}

const std::map<std::string, uint32_t>& IndvSoc::GetSocTypeMap(void) const
{
    return supportSocMap;
}

const std::map<std::string, uint32_t>& IndvSoc::GetSupportHcclSocMap(void) const
{
    return supportHcclSocMap;
}

std::string IndvSoc::GetCurrentSocVersionInternal(void) const
{
    ge::AscendString curSocVersion = op::ToString(op::GetCurrentPlatformInfo().GetSocVersion());
    string socStr = curSocVersion.GetString();
    std::transform(socStr.begin(), socStr.end(), socStr.begin(), [](unsigned char c) { return std::tolower(c); });
    return socStr;
}

const std::string &IndvSoc::GetCurSocVersion(void)
{
    if (!isInit) {
        Init();
    }
    return socVersion;
}

bool IndvSoc::SupportCurrentSoc(void) const
{
    auto iter = std::find(socOpsSubpathList.begin(), socOpsSubpathList.end(), socVersion);
    return iter != socOpsSubpathList.end();
}

bool IndvSoc::UseCoreTypeMagic(void) const
{
    return (socVersion == OPS_SUBPATH_ASCEND910B) || (socVersion == OPS_SUBPATH_ASCEND910_93);
}

bool IndvSoc::SupportMc2FusionLaunch(void) const
{
    return (socVersion == OPS_SUBPATH_ASCEND950) || (socVersion == OPS_SUBPATH_ASCEND910_96);
}

bool IndvSoc::NeedAlignInitValues(void) const
{
    // 910B以及后继芯片场景 需要在Memset的输入tensor上做shape对齐
    return (socVersion != OPS_SUBPATH_ASCEND310B) && (socVersion != OPS_SUBPATH_ASCEND310P) &&
        (socVersion != OPS_SUBPATH_ASCEND610LITE) && (socVersion != OPS_SUBPATH_ASCEND910);
}

bool IndvSoc::NeedsExtraMemoryForOverflowDump(void) const
{
    return socVersion == OPS_SUBPATH_ASCEND910B;
}

bool IndvSoc::SupportL0ExceptionDump(void) const
{
    return (socVersion == OPS_SUBPATH_ASCEND910 || socVersion == OPS_SUBPATH_ASCEND310B);
}

bool IndvSoc::IsCouplingArch(void) const
{
    return (socVersion == OPS_SUBPATH_ASCEND910 || socVersion == OPS_SUBPATH_ASCEND310P);
}

uint32_t *IndvSoc::GetNonFiniteCheckSocSupportList(uint32_t &socSupportListLen) const
{
    static uint32_t socSupportList[] = {SOC_VERSION_ASCEND910B, SOC_VERSION_ASCEND910_93};
    socSupportListLen = sizeof(socSupportList) / sizeof(uint32_t);
    return socSupportList;
}

void IndvSoc::Reset(void)
{
    isInit = false;
    socVersion = "";
}
} // namespace