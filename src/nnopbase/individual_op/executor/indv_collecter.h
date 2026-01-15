/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDV_COLLECTER_H_
#define INDV_COLLECTER_H_

#include <string>
#include "nlohmann/json.hpp"
#include "utils/indv_base.h"
#include "utils/indv_dlist.h"
#include "indv_bininfo.h"
#include "runtime/runtime/dev.h"
#include "platform/platform_info.h"

#ifdef __cplusplus
extern "C" {
#endif

const char *const SCENE = "scene.info";
const size_t SCENE_VALUE_COUNT = 2U;
const size_t SCENE_KEY_INDEX = 0U;
const size_t SCENE_VALUE_INDEX = 1U;
const char *const SCENE_OS = "os";
const char *const SCENE_ARCH = "arch";
constexpr const char* SOC_NAME_ASCEND310 = "Ascend310";
constexpr const char* SOC_NAME_ASCEND610 = "Ascend610";
constexpr const char* SOC_NAME_BS9SX1AA = "BS9SX1AA";
constexpr const char* SOC_NAME_BS9SX2AA = "BS9SX2AA";
constexpr const char* SOC_NAME_BS9SX2AB = "BS9SX2AB";
constexpr const char* SOC_NAME_ASCEND310P7 = "Ascend310P7";
constexpr const char* SOC_NAME_ASCEND310P5 = "Ascend310P5";
constexpr const char* SOC_NAME_ASCEND310P3 = "Ascend310P3";
constexpr const char* SOC_NAME_ASCEND310P1 = "Ascend310P1";
constexpr const char* SOC_NAME_ASCEND310P3VIR01 = "Ascend310P3Vir01";
constexpr const char* SOC_NAME_ASCEND310P3VIR02 = "Ascend310P3Vir02";
constexpr const char* SOC_NAME_ASCEND310P3VIR04 = "Ascend310P3Vir04";
constexpr const char* SOC_NAME_ASCEND310P3VIR08 = "Ascend310P3Vir08";
constexpr const char* SOC_NAME_ASCEND910 = "Ascend910";
constexpr const char* SOC_NAME_ASCEND910A = "Ascend910A";
constexpr const char* SOC_NAME_ASCEND910B = "Ascend910B";
constexpr const char* SOC_NAME_ASCEND910B1 = "Ascend910B1";
constexpr const char* SOC_NAME_ASCEND910B2 = "Ascend910B2";
constexpr const char* SOC_NAME_ASCEND910B2C = "Ascend910B2C";
constexpr const char* SOC_NAME_ASCEND910B3 = "Ascend910B3";
constexpr const char* SOC_NAME_ASCEND910B4 = "Ascend910B4";
constexpr const char* SOC_NAME_ASCEND910B4_1 = "Ascend910B4-1";
constexpr const char* SOC_NAME_ASCEND910_9391 = "Ascend910_9391";
constexpr const char* SOC_NAME_ASCEND910_9381 = "Ascend910_9381";
constexpr const char* SOC_NAME_ASCEND910_9372 = "Ascend910_9372";
constexpr const char* SOC_NAME_ASCEND910_9382 = "Ascend910_9382";
constexpr const char* SOC_NAME_ASCEND910_9392 = "Ascend910_9392";
constexpr const char* SOC_NAME_ASCEND910_9362 = "Ascend910_9362";
constexpr const char* SOC_NAME_ASCEND910PROA = "Ascend910ProA";
constexpr const char* SOC_NAME_ASCEND910PROB = "Ascend910ProB";
constexpr const char* SOC_NAME_ASCEND310B1 = "Ascend310B1";
constexpr const char* SOC_NAME_ASCEND310B2 = "Ascend310B2";
constexpr const char* SOC_NAME_ASCEND310B3 = "Ascend310B3";
constexpr const char* SOC_NAME_ASCEND310B4 = "Ascend310B4";
constexpr const char* SOC_NAME_ASCEND031 = "Ascend031";
constexpr const char* SOC_NAME_ASCEND910PREMIUMA = "Ascend910PremiumA";
constexpr const char* SOC_NAME_ASCEND610LITE = "Ascend610Lite";
constexpr const char* SOC_NAME_ASCEND910_5591 = "Ascend910_5591";
constexpr const char* SOC_NAME_MC61AM21A = "MC61AM21A";
constexpr const char* SOC_NAME_ASCEND910_950Y = "Ascend910_950y";
constexpr const char* SOC_NAME_ASCEND910_950Z = "Ascend910_950z";
constexpr const char* SOC_NAME_ASCEND910_957B = "Ascend910_957b";
constexpr const char* SOC_NAME_ASCEND910_957D = "Ascend910_957d";
constexpr const char* SOC_NAME_ASCEND910_9589 = "Ascend910_9589";
constexpr const char* SOC_NAME_ASCEND910_958A = "Ascend910_958a";
constexpr const char* SOC_NAME_ASCEND910_958B = "Ascend910_958b";
constexpr const char* SOC_NAME_ASCEND910_9599 = "Ascend910_9599";
constexpr const char* SOC_NAME_ASCEND910_9581 = "Ascend910_9581";
constexpr const char* SOC_NAME_ASCEND910_9579 = "Ascend910_9579";
constexpr const char* SOC_NAME_ASCEND910_9591 = "Ascend910_9591";
constexpr const char* SOC_NAME_ASCEND910_9592 = "Ascend910_9592";
constexpr const char* SOC_NAME_ASCEND910_9582 = "Ascend910_9582";
constexpr const char* SOC_NAME_ASCEND910_9584 = "Ascend910_9584";
constexpr const char* SOC_NAME_ASCEND910_9587 = "Ascend910_9587";
constexpr const char* SOC_NAME_ASCEND910_9588 = "Ascend910_9588";
constexpr const char* SOC_NAME_ASCEND910_9572 = "Ascend910_9572";
constexpr const char* SOC_NAME_ASCEND910_9574 = "Ascend910_9574";
constexpr const char* SOC_NAME_ASCEND910_9575 = "Ascend910_9575";
constexpr const char* SOC_NAME_ASCEND910_9576 = "Ascend910_9576";
constexpr const char* SOC_NAME_ASCEND910_9577 = "Ascend910_9577";
constexpr const char* SOC_NAME_ASCEND910_9578 = "Ascend910_9578";
constexpr const char* SOC_NAME_ASCEND910_9691 = "Ascend910_9691";
constexpr const char* SOC_NAME_ASCEND910_9699 = "Ascend910_9699";

constexpr const char* OPS_SUBPATH_ASCEND910 = "ascend910";
constexpr const char* OPS_SUBPATH_ASCEND910B = "ascend910b";
constexpr const char* OPS_SUBPATH_ASCEND910_93 = "ascend910_93";
constexpr const char* OPS_SUBPATH_ASCEND910_95 = "ascend910_95";
constexpr const char* OPS_SUBPATH_ASCEND310P = "ascend310p";
constexpr const char* OPS_SUBPATH_ASCEND310B = "ascend310b";
constexpr const char* OPS_SUBPATH_BS9SX1A = "bs9sx1a";
constexpr const char* OPS_SUBPATH_BS9SX2A = "bs9sx2a";
constexpr const char* OPS_SUBPATH_ASCEND610LITE = "ascend610lite";
constexpr const char* OPS_SUBPATH_ASCEND910_55 = "ascend910_55";
constexpr const char* OPS_SUBPATH_MC61AM21A = "mc61am21a";
constexpr const char* OPS_SUBPATH_ASCEND910_96 = "ascend910_96";

static constexpr uint32_t SOC_VERSION_910A = 1U;
static constexpr uint32_t SOC_VERSION_910B = 2U;
static constexpr uint32_t SOC_VERSION_910_93 = 3U;
static constexpr uint32_t SOC_VERSION_910_95 = 4U;
static constexpr uint32_t SOC_VERSION_310P = 5U;
static constexpr uint32_t SOC_VERSION_310B = 6U;
static constexpr uint32_t SOC_VERSION_BS9SX1A = 7U;
static constexpr uint32_t SOC_VERSION_ASCEND610Lite = 8U;
static constexpr uint32_t SOC_VERSION_910_55 = 9U;
static constexpr uint32_t SOC_VERSION_MC61AM21A = 10U;
static constexpr uint32_t SOC_VERSION_BS9SX2A = 11U;
static constexpr uint32_t SOC_VERSION_910_96 = 12U;

constexpr const char* NNOPBASE_SIMPLIFIED_KEY_MODE_JSON_KEY = "simplifiedKeyMode";
constexpr int32_t NNOPBASE_SIMPLIFIED_KEY_MODE_CUSTOMIZED = 2;

typedef struct {
    DList head;
} RegInfoBucket;

typedef struct {
    RegInfoBucket buckets[NNOPBASE_NORM_MAX_BIN_BUCKETS];
} RegInfoTbl;

typedef struct {
    RegInfoTbl regInfoTbl;
    std::string socVersion;
    bool isAscend19x1 = false; // for 910_93 & 910b
    bool isMc2FusionLaunch = false; // 对于910_95后的芯片，mc2算子使用fusion launch
    std::string oppPath;
    bool isMemsetV2 = false;  // for memsetV2 flag
} NnopbaseBinCollecter;

extern NnopbaseBinCollecter *gBinCollecter;
aclnnStatus NnopbaseCollecterWork(NnopbaseBinCollecter *const collecter);
NnopbaseBinInfo* NnopbaseCollecterFindBinInfo(NnopbaseRegInfo *const regInfo, const size_t hashKey,
    const NnopbaseUChar *const verbose, const uint32_t verbLen);
void NnopbaseCollecterInsertBinInfo(NnopbaseRegInfo *const regInfo, NnopbaseBinInfo* binInfo);
aclnnStatus NnopbaseCollecterAddBinInfo(const string &key, NnopbaseRegInfo *const regInfo, const NnopbaseJsonInfo &jsonInfo,
    const NnopbaseUChar *const verbose, const uint32_t len);
aclnnStatus NnopbaseCollecterAddRegInfoToTbl(NnopbaseBinCollecter *const collecter, const NnopbaseJsonInfo &jsonInfo,
                                             const uint64_t hashKey, NnopbaseRegInfo *&reg, gert::OppImplVersionTag oppImplVersion);
aclnnStatus NnopbaseCollecterAddRepoInfo(NnopbaseBinCollecter *const collecter, const NnopbaseJsonInfo &jsonInfo,
                                         const std::string &key, gert::OppImplVersionTag oppImplVersion);
aclnnStatus NnopbaseCollecterConvertCustomizedVerbKey(const NnopbaseChar *const strKey,  NnopbaseUChar *const binKey,
                                                      uint32_t *const size);
aclnnStatus NnopbaseCollecterConvertDynamicVerbKey(const NnopbaseChar *const strKey, NnopbaseUChar *const binKey,
                                                   uint32_t *const size);
aclnnStatus NnopbaseCollecterConvertStaticVerbKey(const NnopbaseChar *const strKey,
                                                  NnopbaseUChar *const binKey, uint32_t *const size);
aclnnStatus NnopbaseSetCollecterSocVersion(NnopbaseBinCollecter *collecter, const std::string &socVersion);

void NnopbaseCollecterOpRegInfoDestroy(NnopbaseRegInfo **regInfo);
aclnnStatus NnopbaseCollecterGcRegInfo(const void *const data);
aclnnStatus NnopbaseCollecterReadDynamicKernelOpInfoConfig(NnopbaseBinCollecter *const collecter,
    const nlohmann::json &binaryInfoConfig, const std::string &basePath, gert::OppImplVersionTag oppImplVersion,
    const std::string pkgName);

void NnopbaseGetCustomOpApiPath(std::vector<std::string> &basePath);
void NnopbaseGetOppApiPath(std::vector<std::string> &basePath);
void NnopbaseGetCustomOppPath(std::vector<std::pair<std::string, gert::OppImplVersionTag>> &basePath);
void NnopbaseGetOppPath(NnopbaseBinCollecter *const collecter, std::vector<std::pair<std::string, gert::OppImplVersionTag>> &basePath);
aclnnStatus NnopbaseGetCurEnvPackageOsAndCpuType(std::string &hostEnvOs, std::string &hostEnvCpu);
aclnnStatus NnopbaseLoadTilingSo(std::vector<std::pair<std::string, gert::OppImplVersionTag>> &basePath);
aclnnStatus NnopbaseCollecterSetTiling(const NnopbaseJsonInfo &jsonInfo, TilingFun *const tiling, gert::OppImplVersionTag oppImplVersion);
bool NnopbaseReadConfigFile(const std::string &configPath, std::vector<std::string> &subPath);
NnopbaseUChar *NnopbaseCollecterGenStaticKey(NnopbaseUChar *verKey, const NnopbaseRegInfoKey *const regInfoKey,
    const NnopbaseStaticTensorNumInfo *const tensorNumInfo, const aclTensor* tensors[], const NnopbaseAttrAddr *attrs[],
    const int64_t implMode, const int64_t deterMin, const int64_t *const vDepend);
const char *NnopbaseCollecterGetSimplifiedKey(
    const NnopbaseChar *const opType, const uint64_t key, const NnopbaseUChar *verbose, const uint32_t verbLen);
aclnnStatus NnopbaseCollecterGetStaticKernelPathAndReadConfig(NnopbaseBinCollecter *const collecter);
aclnnStatus NnopbaseCollecterDeleteStaticBins(NnopbaseRegInfo *regInfo);
void NnopbaseSplitStr(const std::string &configPath, const std::string &pattern, std::vector<std::string> &subPaths);
aclnnStatus NnopbaseCollecterReadDebugKernelOpInfoConfig(NnopbaseBinCollecter *const collecter,
    nlohmann::json &binaryInfoConfig, const std::string &basePath, gert::OppImplVersionTag oppImplVersion);
aclnnStatus NnopbaseCollecterGetDynamicKernelPathAndReadConfig(NnopbaseBinCollecter *const collecter,
    const std::vector<std::pair<std::string, gert::OppImplVersionTag>> &basePath);
aclnnStatus NnopbaseCollecterReadStaticKernelOpInfoConfig(NnopbaseBinCollecter *const collecter,
    nlohmann::json &binaryInfoConfig, const std::string &basePath, gert::OppImplVersionTag oppImplVersion);
NnopbaseUChar *NnopbaseBeyond8BtyeCopy(
    const int32_t start, const int32_t end, const NnopbaseChar *const strKey, NnopbaseUChar *verKey);
aclnnStatus NnopbaseCollecterOpRegInfoInit(NnopbaseRegInfo *regInfo, const NnopbaseJsonInfo &jsonInfo,
                                           const uint64_t hashKey, gert::OppImplVersionTag oppImplVersion);
aclnnStatus NnopbaseUpdateStaticJsonInfo(nlohmann::json &binInfo, NnopbaseJsonInfo &jsonInfo);

aclnnStatus NnopbaseUpdateStaticBinJsonInfos(NnopbaseBinCollecter *const collecter, const NnopbaseChar *const opType);
aclnnStatus NnopbaseRefreshStaticKernelInfos(NnopbaseBinCollecter *const collecter);
static inline aclnnStatus NnopbaseCollecterInit(NnopbaseBinCollecter *collecter)
{
    for (size_t i = 0U; i < NNOPBASE_NORM_MAX_BIN_BUCKETS; i++) {
        DoubleListInit(&collecter->regInfoTbl.buckets[i].head);
    }
    NnopbaseChar socType[NNOPBASE_SOC_VERSION_MAX];
    NNOPBASE_ASSERT_RTOK_RETVAL(rtGetSocVersion(socType, NNOPBASE_SOC_VERSION_MAX));
    std::string socVersion = std::string(socType);
    NNOPBASE_ASSERT_RTOK_RETVAL(fe::PlatformInfoManager::GeInstance().InitRuntimePlatformInfos(socVersion));
    return NnopbaseSetCollecterSocVersion(collecter, socVersion);
}

static inline void NnopbaseCollecterInitBinTbl(BinTbl *binTbl)
{
    for (size_t i = 0U; i < NNOPBASE_NORM_MAX_BIN_BUCKETS; i++) {
        DoubleListInit(&binTbl->buckets[i].head);
        binTbl->buckets[i].isVist = false;
    }
}

NnopbaseRegInfo* NnopbaseCollecterFindRegInfoInTbl(const NnopbaseBinCollecter *const collecter,
                                                   const NnopbaseChar *const opType, const uint64_t hashKey);
#ifdef __cplusplus
}
#endif
#endif
