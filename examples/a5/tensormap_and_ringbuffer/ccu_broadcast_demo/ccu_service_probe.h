/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#pragma once

#include <stdint.h>

#include "platform_comm/ccu/service_mailbox.h"

inline constexpr uint32_t kCcuServiceFloats = 64;
inline constexpr uint32_t kCcuServiceOffset = 32;
inline constexpr uint32_t kCcuServiceShortCount = 16;
inline constexpr uint32_t kCcuServiceLongCount = 32;
inline constexpr uint32_t kCcuServiceMaxRounds = 10000;
inline constexpr float kCcuServiceSentinel = -987654.0f;

struct alignas(64) CcuServiceProbe {
    simpler::ccu_service::Mailbox slot;
    alignas(64) uint64_t verified;
    uint64_t failure;
};

static_assert(sizeof(CcuServiceProbe) == 256);
