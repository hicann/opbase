/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "device.h"

#include <new>

#include "device_sharder.h"
#include "host_sharder.h"

namespace aicpu {
Device::Device(DeviceType device) : device_(device), sharder_(InitSharder(device)) {};

Device::~Device()
{
    if (sharder_ != nullptr) {
        delete sharder_;
    }
}

/*
 * get device type.
 * @return DeviceType: HOST/DEVICE
 */
DeviceType Device::GetDeviceType() const { return device_; }

/*
 * get sharder.
 * @return Sharder *: host or device sharder
 */
const Sharder* Device::GetSharder() const
{
    if (sharder_ != nullptr) {
        return sharder_;
    }
    return nullptr;
}

/*
 * init sharder.
 * param device: type of device
 * @return Sharder *: not null->success, null->success
 */
Sharder* Device::InitSharder(DeviceType device_type) const
{
    if (device_type == DEVICE) {
        return new (std::nothrow) DeviceSharder(device_type);
    } else {
        return new (std::nothrow) HostSharder(device_type);
    }
}
} // namespace aicpu
