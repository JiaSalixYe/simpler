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

#include <stddef.h>
#include <stdint.h>

namespace simpler::ccu_service {

inline constexpr uint64_t kIdle = 0;
inline constexpr uint64_t kExecute = 1;
inline constexpr uint64_t kStop = 2;
inline constexpr uint32_t kPayloadWords = 4;

struct alignas(64) Mailbox {
    uint64_t generation;
    uint64_t payload[kPayloadWords];
    // Only one device producer owns the slot. Publish after flushing the payload.
    alignas(64) uint64_t command;
    // Execute clears command before completing. Stop leaves a terminal command.
    alignas(64) uint64_t completed;
};

static_assert(sizeof(Mailbox) == 192);
static_assert(offsetof(Mailbox, command) == 64);
static_assert(offsetof(Mailbox, completed) == 128);

}  // namespace simpler::ccu_service
