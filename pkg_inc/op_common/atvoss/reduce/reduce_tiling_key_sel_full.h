/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_tiling_key_sel_full.h
 * \brief reduce tiling key select batch invariant
 */

#ifndef _REDUCE_TILING_KEY_SEL_FULL_H_
#define _REDUCE_TILING_KEY_SEL_FULL_H_

#include "reduce_tiling_key_decl_full.h"

ASCENDC_TPL_SEL(
    // Empty
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_EMPTY_FULL()),
    // A
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_A_FULL()),
    // AR
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_AR_NORMAL_FULL()), ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_AR_GROUP_FULL()),
    // ARA
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_NORMAL_FULL()),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_GROUP_FULL()),
    // ARAR
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARAR_NORMAL_FULL()),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARAR_GROUP_FULL()),
    // ARARARAR
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARAR_NORMAL_FULL()),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARAR_GROUP_FULL()),
    // ARARARARA
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARARA_NORMAL_FULL()),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARARA_GROUP_FULL()));

#endif